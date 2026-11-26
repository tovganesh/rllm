//! Integer dot products between block-quantized weights and Q8-quantized activations.
//!
//! As in llama.cpp, activations are quantized once per matrix-vector product into
//! Q8_0-style blocks (32 int8 values with an f16-rounded scale), and each weight block
//! is multiplied with them in integer arithmetic:
//!
//! `dot(w, a) = Σ_blocks (Σ_i w_q[i]·a_q[i]) · d_w · d_a`
//!
//! On aarch64 CPUs with the `dotprod` extension the inner sum uses `SDOT` (16 int8
//! multiply-adds per instruction); elsewhere a portable scalar loop is used.

// Block-size slicing with `chunks_exact(QK)` mirrors the ggml layouts directly.
#![allow(clippy::chunks_exact_to_as_chunks)]

use half::f16;

/// Values per quantization block.
pub const QK: usize = 32;
const Q8_0_BYTES: usize = 34;
const Q4_0_BYTES: usize = 18;
const Q5_0_BYTES: usize = 22;

/// A row of activations quantized to Q8 blocks.
#[derive(Debug, Clone, Default)]
pub struct Q8Activations {
    /// Per-block scale (already rounded through f16, as stored by llama.cpp).
    pub d: Vec<f32>,
    /// Quantized values, `32 * d.len()` of them.
    pub qs: Vec<i8>,
    /// Per-block sum of the quantized values (used to fold Q4_0's `-8` offset).
    pub sums: Vec<i32>,
}

/// Quantizes `x` (length a multiple of 32) like llama.cpp's `quantize_row_q8_0`.
pub fn quantize_activations(x: &[f32]) -> Q8Activations {
    quantize_activations_impl(x, true)
}

/// Like [`quantize_activations`], computing per-block sums only when `with_sums`
/// (needed by the Q4_0 four-row kernel).
pub fn quantize_activations_impl(x: &[f32], with_sums: bool) -> Q8Activations {
    debug_assert_eq!(x.len() % QK, 0);
    let nb = x.len() / QK;
    let mut d = Vec::with_capacity(nb);
    let mut sums = Vec::with_capacity(nb);
    let mut qs = vec![0i8; x.len()];
    for (block, out) in x.chunks_exact(QK).zip(qs.chunks_exact_mut(QK)) {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        for (o, v) in out.iter_mut().zip(block) {
            *o = (v * inv).round() as i8;
        }
        d.push(f16::from_f32(scale).to_f32());
        if with_sums {
            sums.push(out.iter().map(|&v| v as i32).sum());
        }
    }
    Q8Activations { d, qs, sums }
}

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

#[cfg(target_arch = "aarch64")]
fn has_dotprod() -> bool {
    static DOTPROD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DOTPROD.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

/// Dot product of one `Q8_0` weight row with quantized activations.
#[inline]
pub fn vec_dot_q8_0_q8(w: &[u8], a: &Q8Activations, a_block_offset: usize, nb: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: the dotprod feature was detected at runtime; slices are bounds-checked
        // by the debug assertions in the callers and sized by `nb`.
        return unsafe { neon::dot_q8_0_q8(w, a, a_block_offset, nb) };
    }
    #[cfg(target_arch = "x86_64")]
    if super::x86::has_avx2() {
        // SAFETY: AVX2/FMA/F16C detected at run time; the kernel asserts its bounds.
        return unsafe { super::x86::dot_q8_0(w, a, a_block_offset, nb) };
    }
    scalar_dot_q8_0_q8(w, a, a_block_offset, nb)
}

/// Dot product of one `Q4_0` weight row with quantized activations.
#[inline]
pub fn vec_dot_q4_0_q8(w: &[u8], a: &Q8Activations, a_block_offset: usize, nb: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: as above.
        return unsafe { neon::dot_q4_0_q8(w, a, a_block_offset, nb) };
    }
    #[cfg(target_arch = "x86_64")]
    if super::x86::has_avx2() && !a.sums.is_empty() {
        // SAFETY: AVX2/FMA/F16C detected at run time; the kernel asserts its bounds
        // (including the per-block activation sums it uses for the `-8` offset).
        return unsafe { super::x86::dot_q4_0(w, a, a_block_offset, nb) };
    }
    scalar_dot_q4_0_q8(w, a, a_block_offset, nb)
}

/// Dot product of one `Q5_0` weight row with quantized activations.
#[inline]
pub fn vec_dot_q5_0_q8(w: &[u8], a: &Q8Activations, a_block_offset: usize, nb: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: as above.
        return unsafe { neon::dot_q5_0_q8(w, a, a_block_offset, nb) };
    }
    scalar_dot_q5_0_q8(w, a, a_block_offset, nb)
}

/// Portable reference implementation of [`vec_dot_q5_0_q8`].
pub fn scalar_dot_q5_0_q8(w: &[u8], a: &Q8Activations, off: usize, nb: usize) -> f32 {
    let mut sum = 0f32;
    for b in 0..nb {
        let wb = &w[b * Q5_0_BYTES..(b + 1) * Q5_0_BYTES];
        let qh = u32::from_le_bytes([wb[2], wb[3], wb[4], wb[5]]);
        let aq = &a.qs[(off + b) * QK..(off + b + 1) * QK];
        let mut isum = 0i32;
        for j in 0..16 {
            let x0 = ((wb[6 + j] & 0x0F) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
            let x1 = ((wb[6 + j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
            isum += x0 * aq[j] as i32 + x1 * aq[j + 16] as i32;
        }
        sum += isum as f32 * (f16_at(wb, 0) * a.d[off + b]);
    }
    sum
}

/// Dot products of four weight rows with the same quantized activations.
#[inline]
pub fn vec_dot_x4(is_q8: bool, w: [&[u8]; 4], a: &Q8Activations, off: usize, nb: usize) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: dotprod detected at runtime; each row holds `nb` blocks (caller).
        return unsafe {
            if is_q8 {
                neon::dot_q8_0_q8_x4(w, a, off, nb)
            } else {
                neon::dot_q4_0_q8_x4(w, a, off, nb)
            }
        };
    }
    let dot = if is_q8 { vec_dot_q8_0_q8 } else { vec_dot_q4_0_q8 };
    [
        dot(w[0], a, off, nb),
        dot(w[1], a, off, nb),
        dot(w[2], a, off, nb),
        dot(w[3], a, off, nb),
    ]
}

/// Dot products of one `Q8_0` weight row with `R` activation rows whose blocks start
/// at `offs[r]`; each weight block is loaded once for all rows.
#[inline]
pub fn dot_q8_0_rows<const R: usize>(w: &[u8], a: &Q8Activations, offs: [usize; R], nb: usize) -> [f32; R] {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: dotprod detected at runtime; bounds are asserted inside.
        return unsafe { neon::dot_q8_0_rows(w, a, offs, nb) };
    }
    #[cfg(target_arch = "x86_64")]
    if super::x86::has_avx2() {
        // SAFETY: AVX2/FMA/F16C detected at run time; the kernel asserts its bounds.
        return unsafe { super::x86::dot_q8_0_rows(w, a, offs, nb) };
    }
    std::array::from_fn(|r| scalar_dot_q8_0_q8(w, a, offs[r], nb))
}

/// Dot products of one `Q4_0` weight row with `R` activation rows (see
/// [`dot_q8_0_rows`]); a multi-row kernel exists on x86_64 (AVX2).
#[inline]
pub fn dot_q4_0_rows<const R: usize>(w: &[u8], a: &Q8Activations, offs: [usize; R], nb: usize) -> [f32; R] {
    #[cfg(target_arch = "x86_64")]
    if super::x86::has_avx2() && !a.sums.is_empty() {
        // SAFETY: AVX2/FMA/F16C detected at run time; the kernel asserts its bounds
        // (including the per-block activation sums it uses for the `-8` offset).
        return unsafe { super::x86::dot_q4_0_rows(w, a, offs, nb) };
    }
    std::array::from_fn(|r| vec_dot_q4_0_q8(w, a, offs[r], nb))
}

/// Portable reference implementation of [`vec_dot_q8_0_q8`].
pub fn scalar_dot_q8_0_q8(w: &[u8], a: &Q8Activations, off: usize, nb: usize) -> f32 {
    let mut sum = 0f32;
    for b in 0..nb {
        let wb = &w[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES];
        let aq = &a.qs[(off + b) * QK..(off + b + 1) * QK];
        let isum: i32 = wb[2..].iter().zip(aq).map(|(&x, &y)| x as i8 as i32 * y as i32).sum();
        sum += isum as f32 * (f16_at(wb, 0) * a.d[off + b]);
    }
    sum
}

/// Portable reference implementation of [`vec_dot_q4_0_q8`].
pub fn scalar_dot_q4_0_q8(w: &[u8], a: &Q8Activations, off: usize, nb: usize) -> f32 {
    let mut sum = 0f32;
    for b in 0..nb {
        let wb = &w[b * Q4_0_BYTES..(b + 1) * Q4_0_BYTES];
        let aq = &a.qs[(off + b) * QK..(off + b + 1) * QK];
        let mut isum = 0i32;
        for j in 0..16 {
            let byte = wb[2 + j];
            isum += ((byte & 0x0F) as i32 - 8) * aq[j] as i32 + ((byte >> 4) as i32 - 8) * aq[j + 16] as i32;
        }
        sum += isum as f32 * (f16_at(wb, 0) * a.d[off + b]);
    }
    sum
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    use std::arch::asm;

    /// `acc += Σ a[i]·b[i]` per 32-bit lane (SDOT). `vdotq_s32` is only stable since
    /// Rust 1.98, so the instruction is emitted directly.
    #[inline(always)]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
        acc
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q8_0_q8(w: &[u8], a: &Q8Activations, off: usize, nb: usize) -> f32 {
        debug_assert!(w.len() >= nb * Q8_0_BYTES && a.qs.len() >= (off + nb) * QK);
        let wp = w.as_ptr();
        let ap = a.qs.as_ptr().add(off * QK);
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut b = 0;
        // Two blocks per iteration for instruction-level parallelism.
        while b + 1 < nb {
            let w0 = wp.add(b * Q8_0_BYTES);
            let w1 = wp.add((b + 1) * Q8_0_BYTES);
            let a0 = ap.add(b * QK);
            let a1 = ap.add((b + 1) * QK);
            let mut i0 = vdupq_n_s32(0);
            let mut i1 = vdupq_n_s32(0);
            i0 = sdot(i0, vld1q_s8(w0.add(2) as *const i8), vld1q_s8(a0));
            i0 = sdot(i0, vld1q_s8(w0.add(18) as *const i8), vld1q_s8(a0.add(16)));
            i1 = sdot(i1, vld1q_s8(w1.add(2) as *const i8), vld1q_s8(a1));
            i1 = sdot(i1, vld1q_s8(w1.add(18) as *const i8), vld1q_s8(a1.add(16)));
            let d0 = f16_at(w, b * Q8_0_BYTES) * a.d[off + b];
            let d1 = f16_at(w, (b + 1) * Q8_0_BYTES) * a.d[off + b + 1];
            sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(i0), d0);
            sum1 = vmlaq_n_f32(sum1, vcvtq_f32_s32(i1), d1);
            b += 2;
        }
        if b < nb {
            let w0 = wp.add(b * Q8_0_BYTES);
            let a0 = ap.add(b * QK);
            let mut i0 = vdupq_n_s32(0);
            i0 = sdot(i0, vld1q_s8(w0.add(2) as *const i8), vld1q_s8(a0));
            i0 = sdot(i0, vld1q_s8(w0.add(18) as *const i8), vld1q_s8(a0.add(16)));
            sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(i0), f16_at(w, b * Q8_0_BYTES) * a.d[off + b]);
        }
        vaddvq_f32(vaddq_f32(sum0, sum1))
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q8_0_rows<const R: usize>(w: &[u8], a: &Q8Activations, offs: [usize; R], nb: usize) -> [f32; R] {
        assert!(w.len() >= nb * Q8_0_BYTES);
        for &o in &offs {
            assert!(a.qs.len() >= (o + nb) * QK && a.d.len() >= o + nb);
        }
        let wp = w.as_ptr();
        let mut acc = [vdupq_n_f32(0.0); R];
        for b in 0..nb {
            let blk = wp.add(b * Q8_0_BYTES);
            let w0 = vld1q_s8(blk.add(2) as *const i8);
            let w1 = vld1q_s8(blk.add(18) as *const i8);
            let dw = f16_at(w, b * Q8_0_BYTES);
            for r in 0..R {
                let ap = a.qs.as_ptr().add((offs[r] + b) * QK);
                let i = sdot(sdot(vdupq_n_s32(0), w0, vld1q_s8(ap)), w1, vld1q_s8(ap.add(16)));
                acc[r] = vmlaq_n_f32(acc[r], vcvtq_f32_s32(i), dw * a.d[offs[r] + b]);
            }
        }
        acc.map(|v| vaddvq_f32(v))
    }

    /// Four Q8_0 weight rows against the same activations (activation blocks loaded once).
    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q8_0_q8_x4(w: [&[u8]; 4], a: &Q8Activations, off: usize, nb: usize) -> [f32; 4] {
        let ap = a.qs.as_ptr().add(off * QK);
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..nb {
            let a0 = vld1q_s8(ap.add(b * QK));
            let a1 = vld1q_s8(ap.add(b * QK + 16));
            let da = a.d[off + b];
            for r in 0..4 {
                let wp = w[r].as_ptr().add(b * Q8_0_BYTES);
                let mut i = sdot(vdupq_n_s32(0), vld1q_s8(wp.add(2) as *const i8), a0);
                i = sdot(i, vld1q_s8(wp.add(18) as *const i8), a1);
                acc[r] = vmlaq_n_f32(acc[r], vcvtq_f32_s32(i), f16_at(w[r], b * Q8_0_BYTES) * da);
            }
        }
        [
            vaddvq_f32(acc[0]),
            vaddvq_f32(acc[1]),
            vaddvq_f32(acc[2]),
            vaddvq_f32(acc[3]),
        ]
    }

    /// Four Q4_0 weight rows against the same activations. Nibbles are used unsigned and
    /// the `-8` offset is applied once per block as `8 * sum(a)`.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q4_0_q8_x4(w: [&[u8]; 4], a: &Q8Activations, off: usize, nb: usize) -> [f32; 4] {
        let ap = a.qs.as_ptr().add(off * QK);
        let mask = vdupq_n_u8(0x0F);
        let mut acc = [vdupq_n_f32(0.0); 4];
        for b in 0..nb {
            let a0 = vld1q_s8(ap.add(b * QK));
            let a1 = vld1q_s8(ap.add(b * QK + 16));
            let da = a.d[off + b];
            let corr = vsetq_lane_s32::<0>(8 * a.sums[off + b], vdupq_n_s32(0));
            for r in 0..4 {
                let bytes = vld1q_u8(w[r].as_ptr().add(b * Q4_0_BYTES + 2));
                let lo = vreinterpretq_s8_u8(vandq_u8(bytes, mask));
                let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes));
                let mut i = sdot(vdupq_n_s32(0), lo, a0);
                i = sdot(i, hi, a1);
                let i = vsubq_s32(i, corr);
                acc[r] = vmlaq_n_f32(acc[r], vcvtq_f32_s32(i), f16_at(w[r], b * Q4_0_BYTES) * da);
            }
        }
        [
            vaddvq_f32(acc[0]),
            vaddvq_f32(acc[1]),
            vaddvq_f32(acc[2]),
            vaddvq_f32(acc[3]),
        ]
    }

    /// Q5_0: the fifth bit of value `j` is bit `j` of the 32-bit `qh`; it is expanded to
    /// `0x10` per byte with a bit-test against `[1, 2, 4, ..., 128]`.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q5_0_q8(w: &[u8], a: &Q8Activations, off: usize, nb: usize) -> f32 {
        assert!(w.len() >= nb * Q5_0_BYTES && a.qs.len() >= (off + nb) * QK);
        let wp = w.as_ptr();
        let ap = a.qs.as_ptr().add(off * QK);
        let mask = vdupq_n_u8(0x0F);
        let bit16 = vdupq_n_u8(0x10);
        let sixteen = vdupq_n_s8(16);
        let bits = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let block = |b: usize| -> int32x4_t {
            let p = wp.add(b * Q5_0_BYTES);
            let h = [*p.add(2), *p.add(3), *p.add(4), *p.add(5)];
            let h_lo = vcombine_u8(vdup_n_u8(h[0]), vdup_n_u8(h[1]));
            let h_hi = vcombine_u8(vdup_n_u8(h[2]), vdup_n_u8(h[3]));
            let bytes = vld1q_u8(p.add(6));
            let lo = vorrq_u8(vandq_u8(bytes, mask), vandq_u8(vtstq_u8(h_lo, bits), bit16));
            let hi = vorrq_u8(vshrq_n_u8::<4>(bytes), vandq_u8(vtstq_u8(h_hi, bits), bit16));
            let lo = vsubq_s8(vreinterpretq_s8_u8(lo), sixteen);
            let hi = vsubq_s8(vreinterpretq_s8_u8(hi), sixteen);
            let a0 = ap.add(b * QK);
            sdot(sdot(vdupq_n_s32(0), lo, vld1q_s8(a0)), hi, vld1q_s8(a0.add(16)))
        };
        let mut b = 0;
        while b + 1 < nb {
            let (i0, i1) = (block(b), block(b + 1));
            sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(i0), f16_at(w, b * Q5_0_BYTES) * a.d[off + b]);
            sum1 = vmlaq_n_f32(
                sum1,
                vcvtq_f32_s32(i1),
                f16_at(w, (b + 1) * Q5_0_BYTES) * a.d[off + b + 1],
            );
            b += 2;
        }
        if b < nb {
            sum0 = vmlaq_n_f32(sum0, vcvtq_f32_s32(block(b)), f16_at(w, b * Q5_0_BYTES) * a.d[off + b]);
        }
        vaddvq_f32(vaddq_f32(sum0, sum1))
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q4_0_q8(w: &[u8], a: &Q8Activations, off: usize, nb: usize) -> f32 {
        debug_assert!(w.len() >= nb * Q4_0_BYTES && a.qs.len() >= (off + nb) * QK);
        let wp = w.as_ptr();
        let ap = a.qs.as_ptr().add(off * QK);
        let mask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let mut sum = vdupq_n_f32(0.0);
        for b in 0..nb {
            let bytes = vld1q_u8(wp.add(b * Q4_0_BYTES + 2));
            let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, mask)), eight);
            let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), eight);
            let a0 = ap.add(b * QK);
            let mut i = vdupq_n_s32(0);
            i = sdot(i, lo, vld1q_s8(a0));
            i = sdot(i, hi, vld1q_s8(a0.add(16)));
            sum = vmlaq_n_f32(sum, vcvtq_f32_s32(i), f16_at(w, b * Q4_0_BYTES) * a.d[off + b]);
        }
        vaddvq_f32(sum)
    }
}

/// Weight formats with integer dot-product kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QFormat {
    /// `Q8_0` weights, `Q8_0` activations.
    Q8_0,
    /// `Q4_0` weights, `Q8_0` activations.
    Q4_0,
    /// `Q5_0` weights, `Q8_0` activations.
    Q5_0,
    /// `Q4_K` weights, `Q8_K` activations.
    Q4K,
    /// `Q6_K` weights, `Q8_K` activations.
    Q6K,
}

impl QFormat {
    /// The format for a row-major quantized dtype, if it has an integer kernel.
    pub fn from_dtype(dtype: rllm_core::dtype::DType) -> Option<Self> {
        use rllm_core::dtype::DType;
        Some(match dtype {
            DType::Q8_0 => Self::Q8_0,
            DType::Q4_0 => Self::Q4_0,
            DType::Q5_0 => Self::Q5_0,
            DType::Q4_K => Self::Q4K,
            DType::Q6_K => Self::Q6K,
            _ => return None,
        })
    }

    /// `(values, bytes)` per weight block (and activation block).
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::Q8_0 => (QK, Q8_0_BYTES),
            Self::Q4_0 => (QK, Q4_0_BYTES),
            Self::Q5_0 => (QK, Q5_0_BYTES),
            Self::Q4K => (kdot::QK_K, kdot::Q4_K_BYTES),
            Self::Q6K => (kdot::QK_K, kdot::Q6_K_BYTES),
        }
    }
}

use super::kdot::{self, Q8KActivations};

/// Activations quantized for a given weight format.
enum Acts {
    Q8(Q8Activations),
    K(Q8KActivations),
}

impl Acts {
    fn new(fmt: QFormat, x: &[f32]) -> Self {
        match fmt {
            QFormat::Q4K | QFormat::Q6K => Acts::K(kdot::quantize_q8_k(x)),
            // Per-block sums are only used by the Q4_0 four-row kernel.
            _ => Acts::Q8(quantize_activations_impl(x, fmt == QFormat::Q4_0)),
        }
    }

    /// Dot product of weight row `w` with activation blocks `off..off + nb`.
    #[inline]
    fn dot(&self, fmt: QFormat, w: &[u8], off: usize, nb: usize) -> f32 {
        match (self, fmt) {
            (Acts::Q8(a), QFormat::Q8_0) => vec_dot_q8_0_q8(w, a, off, nb),
            (Acts::Q8(a), QFormat::Q4_0) => vec_dot_q4_0_q8(w, a, off, nb),
            (Acts::Q8(a), QFormat::Q5_0) => vec_dot_q5_0_q8(w, a, off, nb),
            (Acts::K(a), QFormat::Q4K) => kdot::vec_dot_q4_k_q8_k(w, a, off, nb),
            (Acts::K(a), QFormat::Q6K) => kdot::vec_dot_q6_k_q8_k(w, a, off, nb),
            _ => unreachable!("activation quantization does not match the weight format"),
        }
    }

    /// Dot products of weight row `w` with activation rows `i0..i0 + R` (each `nb`
    /// blocks long), unpacking each weight block once where a multi-row kernel exists.
    #[inline]
    fn dot_rows<const R: usize>(&self, fmt: QFormat, w: &[u8], i0: usize, nb: usize) -> [f32; R] {
        let offs: [usize; R] = std::array::from_fn(|r| (i0 + r) * nb);
        match (self, fmt) {
            (Acts::Q8(a), QFormat::Q8_0) => dot_q8_0_rows(w, a, offs, nb),
            (Acts::Q8(a), QFormat::Q4_0) => dot_q4_0_rows(w, a, offs, nb),
            (Acts::K(a), QFormat::Q4K) => kdot::dot_q4_k_rows(w, a, offs, nb),
            (Acts::K(a), QFormat::Q6K) => kdot::dot_q6_k_rows(w, a, offs, nb),
            _ => offs.map(|o| self.dot(fmt, w, o, nb)),
        }
    }

    /// `out[i] = dot(w, activation row first + i)` for `out.len()` activation rows.
    fn dot_all_rows(&self, fmt: QFormat, w: &[u8], nb: usize, first: usize, out: &mut [f32]) {
        let mut i = 0;
        while out.len() - i >= MULTI_ROWS {
            out[i..i + MULTI_ROWS].copy_from_slice(&self.dot_rows::<MULTI_ROWS>(fmt, w, first + i, nb));
            i += MULTI_ROWS;
        }
        let r = first + i;
        match out.len() - i {
            0 => {}
            1 => out[i] = self.dot(fmt, w, r * nb, nb),
            2 => out[i..].copy_from_slice(&self.dot_rows::<2>(fmt, w, r, nb)),
            3 => out[i..].copy_from_slice(&self.dot_rows::<3>(fmt, w, r, nb)),
            4 => out[i..].copy_from_slice(&self.dot_rows::<4>(fmt, w, r, nb)),
            5 => out[i..].copy_from_slice(&self.dot_rows::<5>(fmt, w, r, nb)),
            6 => out[i..].copy_from_slice(&self.dot_rows::<6>(fmt, w, r, nb)),
            7 => out[i..].copy_from_slice(&self.dot_rows::<7>(fmt, w, r, nb)),
            _ => unreachable!(),
        }
    }
}

/// Activation rows handled per call of a multi-row kernel.
const MULTI_ROWS: usize = 8;

/// Batches of at least this many rows block [`linear_quantized`] over activation rows.
const LARGE_BATCH_ROWS: usize = 32;

/// `out[N] = W[N, K] · x[K]` for a quantized weight in format `fmt`.
pub fn gemv_quantized(fmt: QFormat, weight: &[u8], x: &[f32], out: &mut [f32], n: usize, k: usize) {
    let (bv, bb) = fmt.block();
    let nb = k / bv;
    let row_bytes = nb * bb;
    debug_assert_eq!(weight.len(), n * row_bytes);
    let a = Acts::new(fmt, x);
    let rows_per_task = super::gemv_rows_per_task(row_bytes).next_multiple_of(4);
    let row = |j: usize| &weight[j * row_bytes..(j + 1) * row_bytes];
    // Decode path: run on the low-latency spinning pool (see `crate::pool`).
    crate::pool::for_each_chunk_mut(out, rows_per_task, |t, chunk| {
        let j0 = t * rows_per_task;
        let mut r = 0;
        // Four rows at a time for (unrepacked) Q4_0 only: +3-5% decode on an M4, while
        // for Q8_0 the single-row kernel measured 2-4% faster.
        if let (QFormat::Q4_0, Acts::Q8(a)) = (fmt, &a) {
            while r + 4 <= chunk.len() {
                let j = j0 + r;
                chunk[r..r + 4].copy_from_slice(&vec_dot_x4(
                    false,
                    [row(j), row(j + 1), row(j + 2), row(j + 3)],
                    a,
                    0,
                    nb,
                ));
                r += 4;
            }
        }
        for (rr, o) in chunk.iter_mut().enumerate().skip(r) {
            *o = a.dot(fmt, row(j0 + rr), 0, nb);
        }
    });
}

/// `out[M, N] = X[M, K] · W[N, K]ᵀ` for a quantized weight. Every activation row
/// is quantized once; each task owns a slab of weight rows and reuses it across all
/// `M` activation rows while it is hot in cache.
pub fn linear_quantized(fmt: QFormat, weight: &[u8], x: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    let (bv, bb) = fmt.block();
    let nb = k / bv;
    let row_bytes = nb * bb;
    debug_assert_eq!(weight.len(), n * row_bytes);
    debug_assert_eq!(x.len(), m * k);
    // Q8_0 with i8mm: SMMLA (2x2 tiles of 8-deep int8 products) on interleaved row pairs.
    if fmt == QFormat::Q8_0 && super::repack::gemm_q8_0_i8mm(weight, x, out, m, n, k).is_some() {
        return;
    }
    let a = Acts::new(fmt, x); // m rows of nb blocks each, contiguous

    // Compute [N, M] so each task writes a contiguous slab, then transpose.
    // Small batches: each task's weight rows stay hot while it walks all activation
    // rows. Large batches (prompts without a vendor BLAS) would re-read every activation
    // row (1 MB at 512 rows of 2048) from L2/L3 once per weight row, so each task takes
    // a slab of ~64 KB of weight rows and walks the activations in ~128 KB blocks.
    let (rows_per_task, act_block) = if m < LARGE_BATCH_ROWS {
        ((super::gemv_rows_per_task(row_bytes) / m).max(1), m)
    } else {
        let tasks_min = 4 * crate::threads::decode_threads();
        let rows = (64 * 1024 / row_bytes).clamp(1, n.div_ceil(tasks_min).max(1));
        let act_row_bytes = k + 4 * nb;
        (
            rows,
            (128 * 1024 / act_row_bytes).max(MULTI_ROWS) / MULTI_ROWS * MULTI_ROWS,
        )
    };
    let mut out_t = vec![0f32; n * m];
    crate::pool::for_each_chunk_mut(&mut out_t, rows_per_task * m, |t, slab| {
        for i0 in (0..m).step_by(act_block) {
            let i1 = (i0 + act_block).min(m);
            for (r, row) in slab.chunks_mut(m).enumerate() {
                let j = t * rows_per_task + r;
                a.dot_all_rows(
                    fmt,
                    &weight[j * row_bytes..(j + 1) * row_bytes],
                    nb,
                    i0,
                    &mut row[i0..i1],
                );
            }
        }
    });
    for j in 0..n {
        for i in 0..m {
            out[i * n + j] = out_t[j * m + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::quant::{dequantize_q4_0, dequantize_q8_0, quantize_f32_to_q4_0};

    fn q8_0_weights(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for block in x.chunks(QK) {
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
            for v in block {
                out.push((if d != 0.0 { v / d } else { 0.0 }).round() as i8 as u8);
            }
        }
        out
    }

    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7919 + seed * 104729) % 2001) as f32 - 1000.0) / 400.0)
            .collect()
    }

    #[test]
    fn simd_and_scalar_dots_agree_and_track_the_dense_result() {
        for nb in [1usize, 2, 3, 18, 48] {
            let k = nb * QK;
            let wf = data(k, 1);
            let x = data(k, 2);
            let a = quantize_activations(&x);

            let w8 = q8_0_weights(&wf);
            let (fast, slow) = (vec_dot_q8_0_q8(&w8, &a, 0, nb), scalar_dot_q8_0_q8(&w8, &a, 0, nb));
            assert!(
                (fast - slow).abs() <= 1e-3 * slow.abs().max(1.0),
                "q8 nb={nb}: {fast} vs {slow}"
            );
            let mut wd = vec![0f32; k];
            dequantize_q8_0(&w8, &mut wd);
            let dense: f32 = wd.iter().zip(&x).map(|(p, q)| p * q).sum();
            assert!(
                (fast - dense).abs() <= 0.02 * dense.abs().max(1.0),
                "q8 vs dense nb={nb}: {fast} vs {dense}"
            );

            let mut w4 = vec![0u8; nb * Q4_0_BYTES];
            quantize_f32_to_q4_0(&wf, &mut w4);
            let (fast, slow) = (vec_dot_q4_0_q8(&w4, &a, 0, nb), scalar_dot_q4_0_q8(&w4, &a, 0, nb));
            assert!(
                (fast - slow).abs() <= 1e-3 * slow.abs().max(1.0),
                "q4 nb={nb}: {fast} vs {slow}"
            );
            dequantize_q4_0(&w4, &mut wd);
            let dense: f32 = wd.iter().zip(&x).map(|(p, q)| p * q).sum();
            assert!(
                (fast - dense).abs() <= 0.02 * dense.abs().max(1.0),
                "q4 vs dense nb={nb}: {fast} vs {dense}"
            );
        }
    }

    #[test]
    fn q5_0_dot_matches_dequantized_reference() {
        use crate::ops::quant::dequantize_q5_0;
        for nb in [1usize, 2, 7] {
            let k = nb * QK;
            let mut s = 99u64 + nb as u64;
            let mut w: Vec<u8> = (0..nb * Q5_0_BYTES)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (s >> 33) as u8
                })
                .collect();
            for b in 0..nb {
                w[b * Q5_0_BYTES..b * Q5_0_BYTES + 2]
                    .copy_from_slice(&f16::from_f32(0.02 + 0.01 * b as f32).to_le_bytes());
            }
            let a = quantize_activations(&data(k, 4));
            let xq: Vec<f32> = (0..k).map(|j| a.qs[j] as f32 * a.d[j / QK]).collect();
            let mut wd = vec![0f32; k];
            dequantize_q5_0(&w, &mut wd);
            let dense: f32 = wd.iter().zip(&xq).map(|(p, q)| p * q).sum();
            let (fast, slow) = (vec_dot_q5_0_q8(&w, &a, 0, nb), scalar_dot_q5_0_q8(&w, &a, 0, nb));
            assert!(
                (fast - slow).abs() <= 1e-4 * slow.abs().max(1.0),
                "nb={nb}: {fast} vs {slow}"
            );
            assert!(
                (slow - dense).abs() <= 1e-4 * dense.abs().max(1.0),
                "nb={nb}: {slow} vs {dense}"
            );
        }
    }

    #[test]
    fn four_row_kernels_match_single_row() {
        for nb in [1usize, 3, 18] {
            let k = nb * QK;
            let x = data(k, 9);
            let a = quantize_activations(&x);
            let rows: Vec<Vec<f32>> = (0..4).map(|r| data(k, 20 + r)).collect();
            let w8: Vec<Vec<u8>> = rows.iter().map(|r| q8_0_weights(r)).collect();
            let got = vec_dot_x4(true, [&w8[0], &w8[1], &w8[2], &w8[3]], &a, 0, nb);
            for r in 0..4 {
                let want = scalar_dot_q8_0_q8(&w8[r], &a, 0, nb);
                assert!((got[r] - want).abs() <= 1e-3 * want.abs().max(1.0), "q8 nb={nb} r={r}");
            }
            let w4: Vec<Vec<u8>> = rows
                .iter()
                .map(|r| {
                    let mut b = vec![0u8; nb * Q4_0_BYTES];
                    quantize_f32_to_q4_0(r, &mut b);
                    b
                })
                .collect();
            let got = vec_dot_x4(false, [&w4[0], &w4[1], &w4[2], &w4[3]], &a, 0, nb);
            for r in 0..4 {
                let want = scalar_dot_q4_0_q8(&w4[r], &a, 0, nb);
                assert!((got[r] - want).abs() <= 1e-3 * want.abs().max(1.0), "q4 nb={nb} r={r}");
            }
        }
    }

    /// Batches of 1..=11 rows (multi-row SDOT kernels in groups of 8 plus every tail
    /// size, and the Q8_0 SMMLA kernel with odd row counts and a non-multiple-of-4
    /// weight row count) against per-row matrix-vector products.
    #[test]
    fn multi_row_kernels_match_per_row_gemv() {
        use crate::ops::quant::{quantize_f32_to_q4_0, quantize_f32_to_q4_k, quantize_f32_to_q6_k};
        let (n, k) = (38, 512);
        let wf = data(n * k, 5);
        let mut q4_0 = vec![0u8; n * k / QK * Q4_0_BYTES];
        quantize_f32_to_q4_0(&wf, &mut q4_0);
        for (fmt, w) in [
            (QFormat::Q8_0, q8_0_weights(&wf)),
            (QFormat::Q4_0, q4_0),
            (QFormat::Q4K, quantize_f32_to_q4_k(&wf)),
            (QFormat::Q6K, quantize_f32_to_q6_k(&wf)),
        ] {
            // 40 and 300 rows take the blocked large-batch path (300 spans two blocks of
            // activation rows at k = 512).
            for m in (1..=11).chain([40, 300]) {
                let x = data(m * k, 6 + m);
                let mut batched = vec![0f32; m * n];
                linear_quantized(fmt, &w, &x, &mut batched, m, n, k);
                for i in 0..m {
                    let mut row = vec![0f32; n];
                    gemv_quantized(fmt, &w, &x[i * k..(i + 1) * k], &mut row, n, k);
                    for (j, (b, g)) in batched[i * n..(i + 1) * n].iter().zip(&row).enumerate() {
                        assert!(
                            (b - g).abs() <= 1e-4 * g.abs().max(1.0),
                            "{fmt:?} m={m} [{i},{j}]: {b} vs {g}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn batched_linear_matches_per_row_gemv() {
        let (m, n, k) = (5, 70, 96);
        let w = q8_0_weights(&data(n * k, 3));
        let x = data(m * k, 4);
        let mut batched = vec![0f32; m * n];
        linear_quantized(QFormat::Q8_0, &w, &x, &mut batched, m, n, k);
        for i in 0..m {
            let mut row = vec![0f32; n];
            gemv_quantized(QFormat::Q8_0, &w, &x[i * k..(i + 1) * k], &mut row, n, k);
            for (b, g) in batched[i * n..(i + 1) * n].iter().zip(&row) {
                assert!((b - g).abs() <= 1e-4 * g.abs().max(1.0), "row {i}: {b} vs {g}");
            }
        }
    }
}

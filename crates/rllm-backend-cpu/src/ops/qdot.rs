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
use rayon::prelude::*;

/// Values per quantization block.
pub const QK: usize = 32;
const Q8_0_BYTES: usize = 34;
const Q4_0_BYTES: usize = 18;

/// A row of activations quantized to Q8 blocks.
#[derive(Debug, Clone, Default)]
pub struct Q8Activations {
    /// Per-block scale (already rounded through f16, as stored by llama.cpp).
    pub d: Vec<f32>,
    /// Quantized values, `32 * d.len()` of them.
    pub qs: Vec<i8>,
}

/// Quantizes `x` (length a multiple of 32) like llama.cpp's `quantize_row_q8_0`.
pub fn quantize_activations(x: &[f32]) -> Q8Activations {
    debug_assert_eq!(x.len() % QK, 0);
    let nb = x.len() / QK;
    let mut d = Vec::with_capacity(nb);
    let mut qs = vec![0i8; x.len()];
    for (block, out) in x.chunks_exact(QK).zip(qs.chunks_exact_mut(QK)) {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        for (o, v) in out.iter_mut().zip(block) {
            *o = (v * inv).round() as i8;
        }
        d.push(f16::from_f32(scale).to_f32());
    }
    Q8Activations { d, qs }
}

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

fn has_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static DOTPROD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *DOTPROD.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
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
    scalar_dot_q4_0_q8(w, a, a_block_offset, nb)
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


/// `out[N] = W[N, K] · x[K]` for a `Q8_0` (`is_q8 = true`) or `Q4_0` weight.
pub fn gemv_quantized(is_q8: bool, weight: &[u8], x: &[f32], out: &mut [f32], n: usize, k: usize) {
    let nb = k / QK;
    let row_bytes = nb * if is_q8 { Q8_0_BYTES } else { Q4_0_BYTES };
    debug_assert_eq!(weight.len(), n * row_bytes);
    let a = quantize_activations(x);
    let dot = if is_q8 { vec_dot_q8_0_q8 } else { vec_dot_q4_0_q8 };
    let rows_per_task = super::gemv_rows_per_task(row_bytes);
    out.par_chunks_mut(rows_per_task).enumerate().for_each(|(t, chunk)| {
        for (r, o) in chunk.iter_mut().enumerate() {
            let j = t * rows_per_task + r;
            *o = dot(&weight[j * row_bytes..(j + 1) * row_bytes], &a, 0, nb);
        }
    });
}

/// `out[M, N] = X[M, K] · W[N, K]ᵀ` for a `Q8_0` / `Q4_0` weight. Every activation row
/// is quantized once; each task owns a slab of weight rows and reuses it across all
/// `M` activation rows while it is hot in cache.
pub fn linear_quantized(is_q8: bool, weight: &[u8], x: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    let nb = k / QK;
    let row_bytes = nb * if is_q8 { Q8_0_BYTES } else { Q4_0_BYTES };
    debug_assert_eq!(weight.len(), n * row_bytes);
    debug_assert_eq!(x.len(), m * k);
    let a = quantize_activations(x); // m rows of nb blocks each, contiguous
    let dot = if is_q8 { vec_dot_q8_0_q8 } else { vec_dot_q4_0_q8 };

    // Compute [N, M] so each task writes a contiguous slab, then transpose.
    let rows_per_task = (super::gemv_rows_per_task(row_bytes) / m).max(1);
    let mut out_t = vec![0f32; n * m];
    out_t.par_chunks_mut(rows_per_task * m).enumerate().for_each(|(t, slab)| {
        for (r, row) in slab.chunks_mut(m).enumerate() {
            let j = t * rows_per_task + r;
            let w = &weight[j * row_bytes..(j + 1) * row_bytes];
            for (i, o) in row.iter_mut().enumerate() {
                *o = dot(w, &a, i * nb, nb);
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
        (0..n).map(|i| (((i * 7919 + seed * 104729) % 2001) as f32 - 1000.0) / 400.0).collect()
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
            assert!((fast - slow).abs() <= 1e-3 * slow.abs().max(1.0), "q8 nb={nb}: {fast} vs {slow}");
            let mut wd = vec![0f32; k];
            dequantize_q8_0(&w8, &mut wd);
            let dense: f32 = wd.iter().zip(&x).map(|(p, q)| p * q).sum();
            assert!((fast - dense).abs() <= 0.02 * dense.abs().max(1.0), "q8 vs dense nb={nb}: {fast} vs {dense}");

            let mut w4 = vec![0u8; nb * Q4_0_BYTES];
            quantize_f32_to_q4_0(&wf, &mut w4);
            let (fast, slow) = (vec_dot_q4_0_q8(&w4, &a, 0, nb), scalar_dot_q4_0_q8(&w4, &a, 0, nb));
            assert!((fast - slow).abs() <= 1e-3 * slow.abs().max(1.0), "q4 nb={nb}: {fast} vs {slow}");
            dequantize_q4_0(&w4, &mut wd);
            let dense: f32 = wd.iter().zip(&x).map(|(p, q)| p * q).sum();
            assert!((fast - dense).abs() <= 0.02 * dense.abs().max(1.0), "q4 vs dense nb={nb}: {fast} vs {dense}");
        }
    }

    #[test]
    fn batched_linear_matches_per_row_gemv() {
        let (m, n, k) = (5, 70, 96);
        let w = q8_0_weights(&data(n * k, 3));
        let x = data(m * k, 4);
        let mut batched = vec![0f32; m * n];
        linear_quantized(true, &w, &x, &mut batched, m, n, k);
        for i in 0..m {
            let mut row = vec![0f32; n];
            gemv_quantized(true, &w, &x[i * k..(i + 1) * k], &mut row, n, k);
            assert_eq!(&batched[i * n..(i + 1) * n], &row[..]);
        }
    }
}

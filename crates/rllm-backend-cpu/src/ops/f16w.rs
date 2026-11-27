//! Linear layers with F16 weights (F16 GGUF files), kept in F16 in memory.
//!
//! Decode reads every weight once per token, so storing F16 instead of widening to
//! F32 at load halves the bytes per token. Values are widened in registers (`FCVTL`)
//! and multiplied in F32, so results equal an F32 matrix-vector product over the same
//! (exactly representable) weights up to summation order. Small batches convert each
//! 8-value chunk once for up to [`ROWS`] activation rows. Large batches dequantize
//! weight chunks to F32 for the GEMM (in the backend), like the quantized formats.

use half::f16;

/// Activation rows per call of the multi-row kernel.
const ROWS: usize = 8;

/// Bytes per task of the parallel loops, as for the quantized matrix-vector products.
fn rows_per_task(k: usize) -> usize {
    super::gemv_rows_per_task(2 * k)
}

fn as_f16(w: &[u8]) -> &[f16] {
    debug_assert_eq!(w.as_ptr() as usize % 2, 0);
    // SAFETY: `f16` is 2 bytes with alignment 2; weight storage is aligned (checked above
    // in debug builds; tensor storage is 64-byte aligned).
    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f16, w.len() / 2) }
}

/// `out[N] = W[N, K] · x[K]` for an F16 weight given as raw little-endian bytes.
pub fn gemv_f16(weight: &[u8], x: &[f32], out: &mut [f32], n: usize, k: usize) {
    let w = as_f16(weight);
    debug_assert_eq!(w.len(), n * k);
    let per_task = rows_per_task(k);
    crate::pool::for_each_chunk_mut(out, per_task, |t, chunk| {
        for (r, o) in chunk.iter_mut().enumerate() {
            let j = t * per_task + r;
            *o = dot_f16(&w[j * k..(j + 1) * k], x);
        }
    });
}

/// `out[M, N] = X[M, K] · W[N, K]ᵀ` for an F16 weight and a small `M`: each weight row
/// is converted once per group of up to [`ROWS`] activation rows.
pub fn linear_f16(weight: &[u8], x: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    let w = as_f16(weight);
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(x.len(), m * k);
    let per_task = (rows_per_task(k) / m.div_ceil(ROWS)).max(1);
    let mut out_t = vec![0f32; n * m];
    crate::pool::for_each_chunk_mut(&mut out_t, per_task * m, |t, slab| {
        for (r, dst) in slab.chunks_mut(m).enumerate() {
            let j = t * per_task + r;
            let wr = &w[j * k..(j + 1) * k];
            let mut i = 0;
            while i < m {
                let g = (m - i).min(ROWS);
                dot_f16_rows(wr, &x[i * k..(i + g) * k], k, &mut dst[i..i + g]);
                i += g;
            }
        }
    });
    for j in 0..n {
        for i in 0..m {
            out[i * n + j] = out_t[j * m + i];
        }
    }
}

/// Dot product of an F16 row with an F32 vector.
pub fn dot_f16(w: &[f16], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    if w.len().is_multiple_of(16) {
        // SAFETY: lengths checked; NEON is baseline on aarch64.
        return unsafe { neon::dot(w, x) };
    }
    #[cfg(target_arch = "x86_64")]
    if super::x86::has_avx2() {
        // SAFETY: AVX2/FMA/F16C detected at run time; the kernel asserts its bounds.
        return unsafe { super::x86::dot_f16(w, x) };
    }
    w.iter().zip(x).map(|(a, b)| a.to_f32() * b).sum()
}

/// `out[r] = w · x[r]` for `out.len()` (<= [`ROWS`]) contiguous activation rows of `k`.
pub fn dot_f16_rows(w: &[f16], x: &[f32], k: usize, out: &mut [f32]) {
    debug_assert!(out.len() <= ROWS && x.len() == out.len() * k && w.len() == k);
    #[cfg(target_arch = "aarch64")]
    if k.is_multiple_of(8) {
        // SAFETY: lengths checked; NEON is baseline on aarch64.
        unsafe {
            match out.len() {
                8 => neon::dot_rows::<8>(w, x, k, out),
                7 => neon::dot_rows::<7>(w, x, k, out),
                6 => neon::dot_rows::<6>(w, x, k, out),
                5 => neon::dot_rows::<5>(w, x, k, out),
                4 => neon::dot_rows::<4>(w, x, k, out),
                3 => neon::dot_rows::<3>(w, x, k, out),
                2 => neon::dot_rows::<2>(w, x, k, out),
                _ => neon::dot_rows::<1>(w, x, k, out),
            }
        }
        return;
    }
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot_f16(w, &x[r * k..(r + 1) * k]);
    }
}

/// Widens `n` (a multiple of 8) little-endian F16 values at `src` to F32 at `dst`.
///
/// # Safety
/// `src` must be readable for `2 * n` bytes and `dst` writable for `n` floats.
#[cfg(target_arch = "aarch64")]
pub unsafe fn widen(src: *const u8, dst: *mut f32, n: usize) {
    use std::arch::aarch64::vst1q_f32;
    for i in (0..n).step_by(8) {
        let (lo, hi) = neon::cvt8_unaligned(src.add(2 * i));
        vst1q_f32(dst.add(i), lo);
        vst1q_f32(dst.add(i + 4), hi);
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use half::f16;
    use std::arch::aarch64::*;
    use std::arch::asm;

    /// Eight F16 values widened to two F32 vectors (`FCVTL`/`FCVTL2`; Rust's f16
    /// vector types are not stable).
    #[inline(always)]
    unsafe fn cvt8(p: *const f16) -> (float32x4_t, float32x4_t) {
        let h = vld1q_u16(p as *const u16);
        let (lo, hi): (float32x4_t, float32x4_t);
        asm!("fcvtl {lo:v}.4s, {h:v}.4h", "fcvtl2 {hi:v}.4s, {h:v}.8h", h = in(vreg) h, lo = out(vreg) lo, hi = out(vreg) hi,
             options(pure, nomem, nostack));
        (lo, hi)
    }

    /// Like [`cvt8`] for a byte pointer with no alignment requirement.
    #[inline(always)]
    pub unsafe fn cvt8_unaligned(p: *const u8) -> (float32x4_t, float32x4_t) {
        let h = vreinterpretq_u16_u8(vld1q_u8(p));
        let (lo, hi): (float32x4_t, float32x4_t);
        asm!("fcvtl {lo:v}.4s, {h:v}.4h", "fcvtl2 {hi:v}.4s, {h:v}.8h", h = in(vreg) h, lo = out(vreg) lo, hi = out(vreg) hi,
             options(pure, nomem, nostack));
        (lo, hi)
    }

    /// Single row, 16 values per step with four accumulators.
    pub unsafe fn dot(w: &[f16], x: &[f32]) -> f32 {
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut a = [vdupq_n_f32(0.0); 4];
        for i in (0..w.len()).step_by(16) {
            let (l0, h0) = cvt8(wp.add(i));
            let (l1, h1) = cvt8(wp.add(i + 8));
            a[0] = vfmaq_f32(a[0], vld1q_f32(xp.add(i)), l0);
            a[1] = vfmaq_f32(a[1], vld1q_f32(xp.add(i + 4)), h0);
            a[2] = vfmaq_f32(a[2], vld1q_f32(xp.add(i + 8)), l1);
            a[3] = vfmaq_f32(a[3], vld1q_f32(xp.add(i + 12)), h1);
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])))
    }

    /// `R` activation rows against one weight row; each 8-value chunk is converted once.
    pub unsafe fn dot_rows<const R: usize>(w: &[f16], x: &[f32], k: usize, out: &mut [f32]) {
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut a0 = [vdupq_n_f32(0.0); R];
        let mut a1 = [vdupq_n_f32(0.0); R];
        for i in (0..k).step_by(8) {
            let (lo, hi) = cvt8(wp.add(i));
            for r in 0..R {
                let xr = xp.add(r * k + i);
                a0[r] = vfmaq_f32(a0[r], vld1q_f32(xr), lo);
                a1[r] = vfmaq_f32(a1[r], vld1q_f32(xr.add(4)), hi);
            }
        }
        for r in 0..R {
            out[r] = vaddvq_f32(vaddq_f32(a0[r], a1[r]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, k: usize, m: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let mut s = 12345u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let w16: Vec<f16> = (0..n * k).map(|_| f16::from_f32(rnd())).collect();
        let x: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let wf: Vec<f32> = w16.iter().map(|v| v.to_f32()).collect();
        let bytes = w16.iter().flat_map(|v| v.to_le_bytes()).collect();
        (bytes, x, wf)
    }

    fn reference(wf: &[f32], x: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = (0..k).map(|t| wf[j * k + t] as f64 * x[i * k + t] as f64).sum::<f64>() as f32;
            }
        }
        out
    }

    #[test]
    fn convert_f16_matches_scalar() {
        let vals: Vec<f16> = (0..37).map(|i| f16::from_f32(i as f32 * 0.37 - 5.0)).collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = vec![0f32; 37];
        crate::ops::quant::convert_f16(&bytes, &mut out);
        for (a, b) in out.iter().zip(&vals) {
            assert_eq!(*a, b.to_f32());
        }
    }

    #[test]
    fn f16_kernels_match_f32_reference() {
        // k = 72 is not a multiple of 16: the single-row kernel falls back to scalar.
        for (n, k) in [(37, 256), (9, 72)] {
            let (w, x, wf) = data(n, k, 11);
            let mut out = vec![0f32; n];
            gemv_f16(&w, &x[..k], &mut out, n, k);
            let r = reference(&wf, &x[..k], 1, n, k);
            for (a, b) in out.iter().zip(&r) {
                assert!((a - b).abs() < 1e-4, "gemv {a} vs {b}");
            }
            for m in 2..=11 {
                let mut out = vec![0f32; m * n];
                linear_f16(&w, &x[..m * k], &mut out, m, n, k);
                let r = reference(&wf, &x[..m * k], m, n, k);
                for (a, b) in out.iter().zip(&r) {
                    assert!((a - b).abs() < 1e-4, "m {m}: {a} vs {b}");
                }
            }
        }
    }
}

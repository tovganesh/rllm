//! Matrix-matrix (GEMM) and matrix-vector (GEMV) multiplication with SIMD vectorization.

use rayon::prelude::*;

/// Computes the dot product of two F32 slices using SIMD intrinsics.
/// Uses ARM NEON (`vfmaq_f32`) on `aarch64` and loop unrolling fallback.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let mut sum_v = unsafe { vdupq_n_f32(0.0) };
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                sum_v = vfmaq_f32(sum_v, va, vb);
            }
            i += 4;
        }
        let mut sum = unsafe { vaddvq_f32(sum_v) };
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        let mut i = 0;
        while i + 4 <= len {
            sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            i += 4;
        }
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// GEMV: Multiplies a matrix `[N, K]` by a vector `[K]`, writing output to `[N]`.
/// Heavily utilized during autoregressive decoding steps where batch sequence length is 1.
pub fn gemv_f32(mat: &[f32], vec: &[f32], out: &mut [f32], n: usize, k: usize) {
    debug_assert_eq!(mat.len(), n * k);
    debug_assert_eq!(vec.len(), k);
    debug_assert_eq!(out.len(), n);

    let rows = super::gemv_rows_per_task(k * 4);
    if n > rows {
        out.par_chunks_mut(rows)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * rows;
                for (i, val) in out_chunk.iter_mut().enumerate() {
                    let row = row_start + i;
                    let mat_row = &mat[row * k..(row + 1) * k];
                    *val = dot_f32(mat_row, vec);
                }
            });
    } else {
        for row in 0..n {
            let mat_row = &mat[row * k..(row + 1) * k];
            out[row] = dot_f32(mat_row, vec);
        }
    }
}

/// GEMM: `C[M, N] = A[M, K] · B[K, N]` with both operands row-major.
///
/// Parallelised over rows of `C`. The inner loop is an AXPY over a contiguous row of
/// `B`, so memory is streamed sequentially (no column-strided access).
pub fn gemm_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    let row = |i: usize, c_row: &mut [f32]| {
        c_row.fill(0.0);
        let a_row = &a[i * k..(i + 1) * k];
        for (p, &a_ip) in a_row.iter().enumerate() {
            if a_ip == 0.0 {
                continue;
            }
            let b_row = &b[p * n..(p + 1) * n];
            for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_ip * b_pj;
            }
        }
    };

    if m == 1 {
        row(0, c);
    } else {
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| row(i, c_row));
    }
}

/// GEMM with transposed right operand: `C[M, N] = A[M, K] · W[N, K]ᵀ`.
///
/// This is the natural layout for linear layers whose weights are stored
/// `[out_features, in_features]`: every output element is a dot product of two
/// contiguous rows. Parallelised over output columns for small `M` (decode) and over
/// rows of `C` otherwise.
pub fn gemm_nt_f32(
    a: &[f32],
    w: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(c.len(), m * n);

    if m < 4 {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            gemv_f32(w, a_row, &mut c[i * n..(i + 1) * n], n, k);
        }
    } else {
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            let a_row = &a[i * k..(i + 1) * k];
            for (j, c_ij) in c_row.iter_mut().enumerate() {
                *c_ij = dot_f32(a_row, &w[j * k..(j + 1) * k]);
            }
        });
    }
}

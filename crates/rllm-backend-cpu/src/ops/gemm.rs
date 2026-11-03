use rayon::prelude::*;

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

/// GEMV: Matrix [N, K] multiplied by Vector [K] -> Output [N]
/// Used heavily in LLM token generation phase where batch_size * seq_len = 1.
pub fn gemv_f32(mat: &[f32], vec: &[f32], out: &mut [f32], n: usize, k: usize) {
    debug_assert_eq!(mat.len(), n * k);
    debug_assert_eq!(vec.len(), k);
    debug_assert_eq!(out.len(), n);

    if n >= 64 {
        out.par_chunks_mut(16)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * 16;
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

/// GEMM: [M, K] * [K, N] -> [M, N]
/// Parallelized over M rows.
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

    if m == 1 {
        // Transpose view or direct GEMV if B is row-major
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[p] * b[p * n + j];
            }
            c[j] = sum;
        }
        return;
    }

    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_row[p] * b[p * n + j];
            }
            c_row[j] = sum;
        }
    });
}


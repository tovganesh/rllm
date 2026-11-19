//! Large single-precision GEMM for batched (prefill) matrix products.
//!
//! On macOS this calls Accelerate's `cblas_sgemm`, which runs on Apple's matrix units;
//! elsewhere it falls back to the portable row-parallel kernel in [`crate::ops::gemm`].

/// `C[M, N] = A[M, K] · W[N, K]ᵀ` (all row-major, contiguous).
pub fn sgemm_nt(a: &[f32], w: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    #[cfg(target_os = "macos")]
    {
        accelerate::sgemm_nt(a, w, c, m, k, n);
    }
    #[cfg(not(target_os = "macos"))]
    {
        crate::ops::gemm::gemm_nt_f32(a, w, c, m, k, n);
    }
}

/// `C[:, j0..j0 + nc] = A[M, K] · W[nc, K]ᵀ`, where `C` is row-major with leading
/// dimension `ldc` (used to fill a column block of a wider output).
#[allow(clippy::too_many_arguments)]
pub fn sgemm_nt_cols(a: &[f32], w: &[f32], c: &mut [f32], m: usize, k: usize, nc: usize, j0: usize, ldc: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(w.len(), nc * k);
    debug_assert!(c.len() >= (m - 1) * ldc + j0 + nc);
    #[cfg(target_os = "macos")]
    {
        accelerate::sgemm_nt_ld(a, w, &mut c[j0..], m, k, nc, ldc);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut tmp = vec![0f32; m * nc];
        crate::ops::gemm::gemm_nt_f32(a, w, &mut tmp, m, k, nc);
        for (i, row) in tmp.chunks_exact(nc).enumerate() {
            c[i * ldc + j0..i * ldc + j0 + nc].copy_from_slice(row);
        }
    }
}

/// Whether [`sgemm_nt`] uses a vendor BLAS.
pub const HAS_VENDOR_BLAS: bool = cfg!(target_os = "macos");

#[cfg(target_os = "macos")]
mod accelerate {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    pub fn sgemm_nt_ld(a: &[f32], w: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, ldc: usize) {
        assert!(m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize && ldc >= n);
        // SAFETY: a is m*k, w is n*k, c holds (m-1)*ldc + n elements (checked by caller).
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, NO_TRANS, TRANS, m as i32, n as i32, k as i32, 1.0, a.as_ptr(), k as i32,
                w.as_ptr(), k as i32, 0.0, c.as_mut_ptr(), ldc as i32,
            );
        }
    }

    pub fn sgemm_nt(a: &[f32], w: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        assert!(m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize);
        // SAFETY: slice lengths were checked by the caller (m*k, n*k, m*n); leading
        // dimensions describe contiguous row-major storage.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, NO_TRANS, TRANS, m as i32, n as i32, k as i32, 1.0, a.as_ptr(), k as i32,
                w.as_ptr(), k as i32, 0.0, c.as_mut_ptr(), n as i32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_nt_matches_naive() {
        let (m, k, n) = (37, 70, 23);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 31 % 97) as f32 - 48.0) / 17.0).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 53 % 89) as f32 - 44.0) / 13.0).collect();
        let mut c = vec![0f32; m * n];
        sgemm_nt(&a, &w, &mut c, m, k, n);
        for i in 0..m {
            for j in 0..n {
                let r: f32 = (0..k).map(|p| a[i * k + p] * w[j * k + p]).sum();
                assert!((c[i * n + j] - r).abs() <= 1e-3 * r.abs().max(1.0));
            }
        }
    }
}

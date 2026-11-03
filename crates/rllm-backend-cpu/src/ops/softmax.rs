//! Numerically stable softmax implementation.

use rayon::prelude::*;

/// Computes in-place softmax across rows of length `row_len`.
/// Subtracts the maximum element before exponentiation to prevent numerical overflow:
/// $$\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$
pub fn softmax_inplace(x: &mut [f32], row_len: usize) {
    debug_assert_eq!(x.len() % row_len, 0);

    x.par_chunks_mut(row_len).for_each(|row| {
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum_exp += *val;
        }
        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
        for val in row.iter_mut() {
            *val *= inv_sum;
        }
    });
}

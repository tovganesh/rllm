use rayon::prelude::*;

pub fn rms_norm_f32(
    x: &[f32],
    weight: &[f32],
    out: &mut [f32],
    batch_tokens: usize,
    hidden_dim: usize,
    eps: f32,
) {
    debug_assert_eq!(x.len(), batch_tokens * hidden_dim);
    debug_assert_eq!(weight.len(), hidden_dim);
    debug_assert_eq!(out.len(), batch_tokens * hidden_dim);

    out.par_chunks_mut(hidden_dim)
        .zip(x.par_chunks(hidden_dim))
        .for_each(|(out_row, x_row)| {
            let mut sum_sq = 0.0f32;
            for &val in x_row {
                sum_sq += val * val;
            }
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
            let inv_rms = 1.0f32 / rms;

            for i in 0..hidden_dim {
                out_row[i] = x_row[i] * inv_rms * weight[i];
            }
        });
}

pub fn layer_norm_f32(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    batch_tokens: usize,
    hidden_dim: usize,
    eps: f32,
) {
    debug_assert_eq!(x.len(), batch_tokens * hidden_dim);
    debug_assert_eq!(weight.len(), hidden_dim);
    debug_assert_eq!(out.len(), batch_tokens * hidden_dim);

    out.par_chunks_mut(hidden_dim)
        .zip(x.par_chunks(hidden_dim))
        .for_each(|(out_row, x_row)| {
            let mean: f32 = x_row.iter().sum::<f32>() / hidden_dim as f32;
            let var: f32 = x_row
                .iter()
                .map(|&val| {
                    let diff = val - mean;
                    diff * diff
                })
                .sum::<f32>()
                / hidden_dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..hidden_dim {
                let normalized = (x_row[i] - mean) * inv_std * weight[i];
                out_row[i] = if let Some(b) = bias {
                    normalized + b[i]
                } else {
                    normalized
                };
            }
        });
}


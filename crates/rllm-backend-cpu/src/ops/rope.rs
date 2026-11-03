//! Rotary Positional Embeddings (RoPE) for queries and keys.

/// Applies Rotary Positional Embeddings to queries `q` and keys `k` in-place.
pub fn rope_f32(
    q: &mut [f32],
    k: &mut [f32],
    num_heads_q: usize,
    num_heads_k: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half_dim = head_dim / 2;
    debug_assert_eq!(cos.len(), half_dim);
    debug_assert_eq!(sin.len(), half_dim);

    // Apply to Q
    for h in 0..num_heads_q {
        let head_start = h * head_dim;
        let head = &mut q[head_start..head_start + head_dim];
        for i in 0..half_dim {
            let v0 = head[2 * i];
            let v1 = head[2 * i + 1];
            let c = cos[i];
            let s = sin[i];
            head[2 * i] = v0 * c - v1 * s;
            head[2 * i + 1] = v0 * s + v1 * c;
        }
    }

    // Apply to K
    for h in 0..num_heads_k {
        let head_start = h * head_dim;
        let head = &mut k[head_start..head_start + head_dim];
        for i in 0..half_dim {
            let v0 = head[2 * i];
            let v1 = head[2 * i + 1];
            let c = cos[i];
            let s = sin[i];
            head[2 * i] = v0 * c - v1 * s;
            head[2 * i + 1] = v0 * s + v1 * c;
        }
    }
}

/// Precomputes RoPE cosine and sine frequency tables up to `max_seq_len`.
/// $$\text{freq}_i = \frac{1}{\theta^{2i/d}}$$
pub fn precompute_rope_freqs(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_table = Vec::with_capacity(max_seq_len * half_dim);
    let mut sin_table = Vec::with_capacity(max_seq_len * half_dim);

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let freq = 1.0f32 / theta.powf((2 * i) as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            cos_table.push(val.cos());
            sin_table.push(val.sin());
        }
    }

    (cos_table, sin_table)
}

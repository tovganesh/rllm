//! Rotary Positional Embeddings (RoPE) for queries and keys.

/// Applies Rotary Positional Embeddings in place to `x` laid out as
/// `[num_tokens, num_heads, head_dim]`, using row `positions[t]` of the
/// `[max_positions, head_dim / 2]` cosine / sine tables for token `t`.
///
/// Adjacent element pairs `(x[2i], x[2i+1])` are rotated by angle `pos * freq_i`.
pub fn rope_f32(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    positions: &[usize],
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let half_dim = head_dim / 2;
    let token_stride = num_heads * head_dim;
    debug_assert_eq!(x.len(), positions.len() * token_stride);

    for (token, &pos) in x.chunks_exact_mut(token_stride).zip(positions) {
        let cos = &cos_table[pos * half_dim..(pos + 1) * half_dim];
        let sin = &sin_table[pos * half_dim..(pos + 1) * half_dim];
        for head in token.chunks_exact_mut(head_dim) {
            for i in 0..half_dim {
                let v0 = head[2 * i];
                let v1 = head[2 * i + 1];
                head[2 * i] = v0 * cos[i] - v1 * sin[i];
                head[2 * i + 1] = v0 * sin[i] + v1 * cos[i];
            }
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

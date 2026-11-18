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

/// Precomputes RoPE cosine and sine tables `[max_seq_len, head_dim / 2]`.
/// $$\text{freq}_i = \frac{1}{\theta^{2i/d}}$$
pub fn precompute_rope_freqs(head_dim: usize, max_seq_len: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    precompute_rope_freqs_scaled(head_dim, max_seq_len, theta, None)
}

/// Like [`precompute_rope_freqs`], with optional per-frequency divisors
/// (`rope_freqs.weight` in LLaMA 3.1+ GGUF files): angle = `pos * freq_i / factor_i`.
///
/// Angles are computed in f64 so that large positions keep full precision.
pub fn precompute_rope_freqs_scaled(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    freq_factors: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_table = Vec::with_capacity(max_seq_len * half_dim);
    let mut sin_table = Vec::with_capacity(max_seq_len * half_dim);
    let freqs: Vec<f64> = (0..half_dim)
        .map(|i| {
            let f = 1.0f64 / (theta as f64).powf((2 * i) as f64 / head_dim as f64);
            f / freq_factors.map_or(1.0, |ff| ff[i] as f64)
        })
        .collect();

    for pos in 0..max_seq_len {
        for &freq in &freqs {
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos_table.push(c as f32);
            sin_table.push(s as f32);
        }
    }

    (cos_table, sin_table)
}

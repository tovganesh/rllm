//! LLaMA 3.x / Mistral / Qwen architecture with Grouped-Query Attention (GQA) and PagedAttention decoding.

use crate::config::LlamaConfig;
use rllm_backend_cpu::ops::rope::precompute_rope_freqs;
use rllm_core::{
    backend::{Backend, BlockTable},
    device::Device,
    error::Result,
    shape::Shape,
    tensor::Tensor,
};
use std::sync::Arc;

/// Self-attention projection tensors ($Q, K, V, O$).
pub struct LlamaAttention {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
}

/// SwiGLU feed-forward multilayer perceptron tensors (Gate, Up, Down).
pub struct LlamaMlp {
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

/// Single transformer decoder layer.
pub struct LlamaDecoderLayer {
    pub input_layernorm: Tensor,
    pub self_attn: LlamaAttention,
    pub post_attention_layernorm: Tensor,
    pub mlp: LlamaMlp,
}

/// Autoregressive causal language model with PagedAttention support.
pub struct LlamaModel<B: Backend> {
    pub config: LlamaConfig,
    pub backend: Arc<B>,
    pub embed_tokens: Tensor,
    pub layers: Vec<LlamaDecoderLayer>,
    pub norm: Tensor,
    pub lm_head: Tensor,
    pub cos_table: Tensor,
    pub sin_table: Tensor,
}

impl<B: Backend> LlamaModel<B> {
    /// Creates a synthetic model with pseudo-random weights for benchmarking and tests.
    pub fn new_random(config: LlamaConfig, backend: Arc<B>) -> Result<Self> {
        let dev = backend.device();
        let h = config.hidden_size;
        let head_dim = config.head_dim();
        let kv_dim = config.num_key_value_heads * head_dim;
        let inter = config.intermediate_size;

        let embed_tokens = Tensor::from_f32(
            &vec![0.02f32; config.vocab_size * h],
            Shape::new(vec![config.vocab_size, h]),
            dev,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let input_layernorm =
                Tensor::from_f32(&vec![1.0f32; h], Shape::new(vec![h]), dev)?;
            let post_attention_layernorm =
                Tensor::from_f32(&vec![1.0f32; h], Shape::new(vec![h]), dev)?;

            let self_attn = LlamaAttention {
                q_proj: Tensor::from_f32(&vec![0.01f32; h * h], Shape::new(vec![h, h]), dev)?,
                k_proj: Tensor::from_f32(
                    &vec![0.01f32; h * kv_dim],
                    Shape::new(vec![kv_dim, h]),
                    dev,
                )?,
                v_proj: Tensor::from_f32(
                    &vec![0.01f32; h * kv_dim],
                    Shape::new(vec![kv_dim, h]),
                    dev,
                )?,
                o_proj: Tensor::from_f32(&vec![0.01f32; h * h], Shape::new(vec![h, h]), dev)?,
            };

            let mlp = LlamaMlp {
                gate_proj: Tensor::from_f32(
                    &vec![0.01f32; h * inter],
                    Shape::new(vec![inter, h]),
                    dev,
                )?,
                up_proj: Tensor::from_f32(
                    &vec![0.01f32; h * inter],
                    Shape::new(vec![inter, h]),
                    dev,
                )?,
                down_proj: Tensor::from_f32(
                    &vec![0.01f32; inter * h],
                    Shape::new(vec![h, inter]),
                    dev,
                )?,
            };

            layers.push(LlamaDecoderLayer {
                input_layernorm,
                self_attn,
                post_attention_layernorm,
                mlp,
            });
        }

        let norm = Tensor::from_f32(&vec![1.0f32; h], Shape::new(vec![h]), dev)?;
        let lm_head = Tensor::from_f32(
            &vec![0.01f32; config.vocab_size * h],
            Shape::new(vec![config.vocab_size, h]),
            dev,
        )?;

        let (cos_raw, sin_raw) = precompute_rope_freqs(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        );

        let cos_table = Tensor::from_f32(
            &cos_raw,
            Shape::new(vec![config.max_position_embeddings, head_dim / 2]),
            dev,
        )?;
        let sin_table = Tensor::from_f32(
            &sin_raw,
            Shape::new(vec![config.max_position_embeddings, head_dim / 2]),
            dev,
        )?;

        Ok(Self {
            config,
            backend,
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_table,
            sin_table,
        })
    }

    /// Single autoregressive decode iteration across batched sequences using PagedAttention:
    /// - `tokens`: 1 token per sequence in batch -> shape `[batch_size]`.
    /// - `k_caches`: Vector of physical key caches (one per layer).
    /// - `v_caches`: Vector of physical value caches (one per layer).
    /// - `block_tables`: Logical-to-physical block table mappings for each sequence.
    /// - `context_lens`: Total active tokens in context for each sequence.
    pub fn forward_decode(
        &self,
        tokens: &[u32],
        k_caches: &mut [Tensor],
        v_caches: &mut [Tensor],
        block_tables: &[BlockTable],
        context_lens: &[usize],
    ) -> Result<Tensor> {
        let batch_size = tokens.len();
        let h = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        // 1. Token embeddings lookup -> [batch_size, hidden_size]
        let mut hidden = self.backend.embedding(&self.embed_tokens, tokens)?;

        // 2. Transformer decoder layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed =
                self.backend
                    .rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let mut q = self.backend.matmul(&normed, &layer.self_attn.q_proj.transpose(0, 1)?)?;
            let mut k = self.backend.matmul(&normed, &layer.self_attn.k_proj.transpose(0, 1)?)?;
            let v = self.backend.matmul(&normed, &layer.self_attn.v_proj.transpose(0, 1)?)?;

            q = q.reshape(Shape::new(vec![batch_size, num_q_heads, head_dim]))?;
            k = k.reshape(Shape::new(vec![batch_size, num_kv_heads, head_dim]))?;

            // Apply RoPE rotation for each sequence position
            for b in 0..batch_size {
                let pos = context_lens[b] - 1;
                let half_d = head_dim / 2;
                let cos_slice = &self.cos_table.as_f32_slice()?[pos * half_d..(pos + 1) * half_d];
                let sin_slice = &self.sin_table.as_f32_slice()?[pos * half_d..(pos + 1) * half_d];

                let cos_t = Tensor::from_f32(cos_slice, Shape::new(vec![half_d]), Device::Cpu)?;
                let sin_t = Tensor::from_f32(sin_slice, Shape::new(vec![half_d]), Device::Cpu)?;

                self.backend.rope(&mut q, &mut k, &cos_t, &sin_t)?;
            }

            // Write new K and V into physical cache blocks
            let k_cache = &mut k_caches[layer_idx];
            let v_cache = &mut v_caches[layer_idx];
            let block_size = block_tables[0].block_size;
            let block_stride = num_kv_heads * block_size * head_dim;

            let k_slice = k.as_f32_slice()?;
            let v_slice = v.as_f32_slice()?;
            let k_cache_mut = k_cache.as_mut_f32_slice()?;
            let v_cache_mut = v_cache.as_mut_f32_slice()?;

            for b in 0..batch_size {
                let pos = context_lens[b] - 1;
                let block_idx = pos / block_size;
                let block_offset = pos % block_size;
                let phys_block_id = block_tables[b].block_ids[block_idx];

                for kv_h in 0..num_kv_heads {
                    let src_offset = b * (num_kv_heads * head_dim) + kv_h * head_dim;
                    let dst_offset = phys_block_id * block_stride
                        + kv_h * (block_size * head_dim)
                        + block_offset * head_dim;

                    k_cache_mut[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&k_slice[src_offset..src_offset + head_dim]);
                    v_cache_mut[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&v_slice[src_offset..src_offset + head_dim]);
                }
            }

            // PagedAttention decode step
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let attn_out = self.backend.paged_attention(
                &q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                scale,
            )?;

            let attn_flat = attn_out.reshape(Shape::new(vec![batch_size, h]))?;
            let attn_proj = self
                .backend
                .matmul(&attn_flat, &layer.self_attn.o_proj.transpose(0, 1)?)?;

            // Residual connection: hidden = hidden + attn_proj
            let hidden_slice = hidden.as_mut_f32_slice()?;
            let proj_slice = attn_proj.as_f32_slice()?;
            for i in 0..hidden_slice.len() {
                hidden_slice[i] += proj_slice[i];
            }

            // Post-attention RMSNorm + SwiGLU MLP
            let ffn_normed = self.backend.rms_norm(
                &hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            let gate = self
                .backend
                .matmul(&ffn_normed, &layer.mlp.gate_proj.transpose(0, 1)?)?;
            let up = self
                .backend
                .matmul(&ffn_normed, &layer.mlp.up_proj.transpose(0, 1)?)?;
            let act = self.backend.swiglu(&gate, &up)?;
            let ffn_out = self
                .backend
                .matmul(&act, &layer.mlp.down_proj.transpose(0, 1)?)?;

            let ffn_slice = ffn_out.as_f32_slice()?;
            let hidden_slice = hidden.as_mut_f32_slice()?;
            for i in 0..hidden_slice.len() {
                hidden_slice[i] += ffn_slice[i];
            }
        }

        // Final RMSNorm and LM Head projection
        let final_normed = self
            .backend
            .rms_norm(&hidden, &self.norm, self.config.rms_norm_eps)?;

        let logits = self
            .backend
            .matmul(&final_normed, &self.lm_head.transpose(0, 1)?)?;

        Ok(logits)
    }
}

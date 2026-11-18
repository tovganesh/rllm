//! LLaMA 3.x / Mistral / Qwen architecture with Grouped-Query Attention (GQA) and PagedAttention decoding.

use crate::config::LlamaConfig;
use rllm_backend_cpu::ops::rope::precompute_rope_freqs;
use crate::init::SyntheticInit;
use rllm_core::{
    backend::{Backend, BlockTable},
    error::{Result, RllmError},
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
    /// Creates a synthetic model with deterministic pseudo-random weights
    /// (fixed seed) for benchmarking and tests.
    pub fn new_random(config: LlamaConfig, backend: Arc<B>) -> Result<Self> {
        Self::new_seeded(config, backend, 0x5EED)
    }

    /// Creates a synthetic model whose weights are drawn uniformly from
    /// `±1/sqrt(fan_in)` using the given seed. Norm weights are initialised around 1.
    pub fn new_seeded(config: LlamaConfig, backend: Arc<B>, seed: u64) -> Result<Self> {
        let dev = backend.device();
        let h = config.hidden_size;
        let head_dim = config.head_dim();
        let kv_dim = config.num_key_value_heads * head_dim;
        let inter = config.intermediate_size;
        let mut init = SyntheticInit::new(seed);

        let embed_tokens = init.uniform(&[config.vocab_size, h], 1.0, dev)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer {
                input_layernorm: init.norm_weight(h, dev)?,
                self_attn: LlamaAttention {
                    q_proj: init.linear(h, h, dev)?,
                    k_proj: init.linear(kv_dim, h, dev)?,
                    v_proj: init.linear(kv_dim, h, dev)?,
                    o_proj: init.linear(h, h, dev)?,
                },
                post_attention_layernorm: init.norm_weight(h, dev)?,
                mlp: LlamaMlp {
                    gate_proj: init.linear(inter, h, dev)?,
                    up_proj: init.linear(inter, h, dev)?,
                    down_proj: init.linear(h, inter, dev)?,
                },
            });
        }

        let norm = init.norm_weight(h, dev)?;
        let lm_head = init.linear(config.vocab_size, h, dev)?;

        let (cos_raw, sin_raw) = precompute_rope_freqs(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        );
        let table_shape = Shape::new(vec![config.max_position_embeddings, head_dim / 2]);
        let cos_table = Tensor::from_f32(&cos_raw, table_shape.clone(), dev)?;
        let sin_table = Tensor::from_f32(&sin_raw, table_shape, dev)?;

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

    /// Single autoregressive decode step: one token per sequence.
    ///
    /// - `tokens`: the newest token of each sequence, `[batch_size]`.
    /// - `context_lens`: total tokens of each sequence **including** the new token, so the
    ///   new token sits at position `context_lens[b] - 1`.
    ///
    /// Returns logits `[batch_size, vocab_size]`.
    pub fn forward_decode(
        &self,
        tokens: &[u32],
        k_caches: &mut [Tensor],
        v_caches: &mut [Tensor],
        block_tables: &[BlockTable],
        context_lens: &[usize],
    ) -> Result<Tensor> {
        if context_lens.contains(&0) {
            return Err(RllmError::BackendError(
                "forward_decode: context length must be >= 1".to_string(),
            ));
        }
        let positions: Vec<usize> = context_lens.iter().map(|&l| l - 1).collect();
        let logits_rows: Vec<usize> = (0..tokens.len()).collect();
        self.forward(
            &ForwardBatch {
                tokens,
                positions: &positions,
                block_tables,
                logits_rows: &logits_rows,
            },
            k_caches,
            v_caches,
        )
    }

    /// General forward pass over a flattened batch of tokens from one or more sequences.
    ///
    /// Each row `i` is token `batch.tokens[i]` at absolute position `batch.positions[i]`
    /// of the sequence whose KV blocks are `batch.block_tables[i]`. The key/value of every
    /// row is written to the paged cache before attention, and row `i` attends to
    /// positions `0..=positions[i]` of its sequence — so passing all prompt tokens of a
    /// sequence performs causal **prefill**, and passing one token per sequence performs
    /// batched **decode**. Both can be mixed in one call.
    ///
    /// Returns logits `[batch.logits_rows.len(), vocab_size]` for the selected rows only.
    pub fn forward(
        &self,
        batch: &ForwardBatch<'_>,
        k_caches: &mut [Tensor],
        v_caches: &mut [Tensor],
    ) -> Result<Tensor> {
        let num_rows = batch.tokens.len();
        let h = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let max_pos = self.config.max_position_embeddings;

        if num_rows == 0 {
            return Err(RllmError::BackendError("forward: empty batch".to_string()));
        }
        if batch.positions.len() != num_rows || batch.block_tables.len() != num_rows {
            return Err(RllmError::BackendError(format!(
                "forward: {} tokens but {} positions and {} block tables",
                num_rows,
                batch.positions.len(),
                batch.block_tables.len()
            )));
        }
        if k_caches.len() != self.layers.len() || v_caches.len() != self.layers.len() {
            return Err(RllmError::BackendError(format!(
                "forward: expected {} KV cache layers, got {} / {}",
                self.layers.len(),
                k_caches.len(),
                v_caches.len()
            )));
        }
        for (&pos, bt) in batch.positions.iter().zip(batch.block_tables) {
            if pos >= max_pos {
                return Err(RllmError::BackendError(format!(
                    "forward: position {} exceeds max_position_embeddings {}",
                    pos, max_pos
                )));
            }
            if pos / bt.block_size >= bt.block_ids.len() {
                return Err(RllmError::BackendError(format!(
                    "forward: position {} not covered by {} KV blocks",
                    pos,
                    bt.block_ids.len()
                )));
            }
        }
        if let Some(&bad) = batch.logits_rows.iter().find(|&&r| r >= num_rows) {
            return Err(RllmError::BackendError(format!(
                "forward: logits row {} out of range for {} rows",
                bad, num_rows
            )));
        }

        let context_lens: Vec<usize> = batch.positions.iter().map(|&p| p + 1).collect();
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // 1. Token embeddings -> [num_rows, hidden_size]
        let mut hidden = self.backend.embedding(&self.embed_tokens, batch.tokens)?;

        // 2. Decoder layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed =
                self.backend
                    .rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let mut q = self.backend.linear(&normed, &layer.self_attn.q_proj)?;
            let mut k = self.backend.linear(&normed, &layer.self_attn.k_proj)?;
            let v = self.backend.linear(&normed, &layer.self_attn.v_proj)?;

            q = q.reshape(Shape::new(vec![num_rows, num_q_heads, head_dim]))?;
            k = k.reshape(Shape::new(vec![num_rows, num_kv_heads, head_dim]))?;

            // Rotary embeddings, each row at its own absolute position.
            self.backend
                .rope(&mut q, &mut k, &self.cos_table, &self.sin_table, batch.positions)?;

            // Write every row's K and V into its sequence's physical cache slot.
            let k_cache = &mut k_caches[layer_idx];
            let v_cache = &mut v_caches[layer_idx];
            {
                let k_slice = k.as_f32_slice()?;
                let v_slice = v.as_f32_slice()?;
                let k_cache_mut = k_cache.as_mut_f32_slice()?;
                for (row, (&pos, bt)) in batch.positions.iter().zip(batch.block_tables).enumerate() {
                    let block_stride = num_kv_heads * bt.block_size * head_dim;
                    let phys_block_id = bt.block_ids[pos / bt.block_size];
                    for kv_h in 0..num_kv_heads {
                        let src = row * (num_kv_heads * head_dim) + kv_h * head_dim;
                        let dst = phys_block_id * block_stride
                            + kv_h * (bt.block_size * head_dim)
                            + (pos % bt.block_size) * head_dim;
                        k_cache_mut[dst..dst + head_dim].copy_from_slice(&k_slice[src..src + head_dim]);
                    }
                }
                let v_cache_mut = v_cache.as_mut_f32_slice()?;
                for (row, (&pos, bt)) in batch.positions.iter().zip(batch.block_tables).enumerate() {
                    let block_stride = num_kv_heads * bt.block_size * head_dim;
                    let phys_block_id = bt.block_ids[pos / bt.block_size];
                    for kv_h in 0..num_kv_heads {
                        let src = row * (num_kv_heads * head_dim) + kv_h * head_dim;
                        let dst = phys_block_id * block_stride
                            + kv_h * (bt.block_size * head_dim)
                            + (pos % bt.block_size) * head_dim;
                        v_cache_mut[dst..dst + head_dim].copy_from_slice(&v_slice[src..src + head_dim]);
                    }
                }
            }

            // Causal paged attention: row i sees positions 0..=positions[i].
            let attn_out = self.backend.paged_attention(
                &q,
                k_cache,
                v_cache,
                batch.block_tables,
                &context_lens,
                scale,
            )?;

            let attn_flat = attn_out.reshape(Shape::new(vec![num_rows, h]))?;
            let attn_proj = self
                .backend
                .linear(&attn_flat, &layer.self_attn.o_proj)?;
            add_inplace(&mut hidden, &attn_proj)?;

            // Post-attention RMSNorm + SwiGLU MLP
            let ffn_normed = self.backend.rms_norm(
                &hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            let gate = self
                .backend
                .linear(&ffn_normed, &layer.mlp.gate_proj)?;
            let up = self
                .backend
                .linear(&ffn_normed, &layer.mlp.up_proj)?;
            let act = self.backend.swiglu(&gate, &up)?;
            let ffn_out = self
                .backend
                .linear(&act, &layer.mlp.down_proj)?;
            add_inplace(&mut hidden, &ffn_out)?;
        }

        // 3. Final norm + LM head, only for the requested rows.
        let selected = if batch.logits_rows.len() == num_rows
            && batch.logits_rows.iter().enumerate().all(|(i, &r)| i == r)
        {
            hidden
        } else {
            let src = hidden.as_f32_slice()?;
            let mut rows = Vec::with_capacity(batch.logits_rows.len() * h);
            for &r in batch.logits_rows {
                rows.extend_from_slice(&src[r * h..(r + 1) * h]);
            }
            Tensor::from_f32(
                &rows,
                Shape::new(vec![batch.logits_rows.len(), h]),
                self.backend.device(),
            )?
        };
        let final_normed = self
            .backend
            .rms_norm(&selected, &self.norm, self.config.rms_norm_eps)?;
        self.backend
            .linear(&final_normed, &self.lm_head)
    }
}

/// Flattened batch of tokens for [`LlamaModel::forward`].
#[derive(Debug, Clone, Copy)]
pub struct ForwardBatch<'a> {
    /// Token ID of each row.
    pub tokens: &'a [u32],
    /// Absolute position of each row within its sequence.
    pub positions: &'a [usize],
    /// KV block table of the sequence each row belongs to.
    pub block_tables: &'a [BlockTable],
    /// Rows for which logits are returned (typically the last row of each sequence).
    pub logits_rows: &'a [usize],
}

fn add_inplace(dst: &mut Tensor, src: &Tensor) -> Result<()> {
    let src = src.as_f32_slice()?;
    let dst = dst.as_mut_f32_slice()?;
    if src.len() != dst.len() {
        return Err(RllmError::BackendError(format!(
            "residual add: length mismatch {} vs {}",
            dst.len(),
            src.len()
        )));
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
    Ok(())
}

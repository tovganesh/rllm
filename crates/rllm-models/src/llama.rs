//! LLaMA 3.x / Mistral / Qwen architecture with Grouped-Query Attention (GQA) and PagedAttention decoding.

use crate::config::LlamaConfig;
use crate::init::SyntheticInit;
use rllm_backend_cpu::ops::rope::precompute_rope_freqs;
use rllm_core::{
    backend::{Backend, BlockTable},
    dtype::DType,
    error::{Result, RllmError},
    shape::Shape,
    tensor::Tensor,
};
use std::sync::Arc;

/// Self-attention projection tensors ($Q, K, V, O$).
pub struct LlamaAttention {
    /// Fused `[q | k | v]` projection (see [`LlamaModel::fuse_projections`]); when set,
    /// `q_proj`, `k_proj` and `v_proj` are row views of it.
    pub qkv_proj: Option<Tensor>,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
}

/// SwiGLU feed-forward multilayer perceptron tensors (Gate, Up, Down).
pub struct LlamaMlp {
    /// Fused `[gate | up]` projection; when set, `gate_proj` and `up_proj` are views of it.
    pub gate_up_proj: Option<Tensor>,
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
    /// Storage type of the KV cache allocated by [`crate::LanguageModel::init_kv_cache`]:
    /// `F16` (default for loaded models, as in llama.cpp: half the memory and bandwidth)
    /// or `F32`. Executors that wrap this model (e.g. Metal) follow it.
    pub kv_dtype: DType,
    /// KV cache used by [`crate::LanguageModel::forward_cached`].
    pub(crate) kv_cache: std::sync::Mutex<crate::lm::KvCache>,
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
                    qkv_proj: None,
                    q_proj: init.linear(h, h, dev)?,
                    k_proj: init.linear(kv_dim, h, dev)?,
                    v_proj: init.linear(kv_dim, h, dev)?,
                    o_proj: init.linear(h, h, dev)?,
                },
                post_attention_layernorm: init.norm_weight(h, dev)?,
                mlp: LlamaMlp {
                    gate_up_proj: None,
                    gate_proj: init.linear(inter, h, dev)?,
                    up_proj: init.linear(inter, h, dev)?,
                    down_proj: init.linear(h, inter, dev)?,
                },
            });
        }

        let norm = init.norm_weight(h, dev)?;
        let lm_head = init.linear(config.vocab_size, h, dev)?;

        let (cos_raw, sin_raw) = precompute_rope_freqs(head_dim, config.max_position_embeddings, config.rope_theta);
        let table_shape = Shape::new(vec![config.max_position_embeddings, head_dim / 2]);
        let cos_table = Tensor::from_f32(&cos_raw, table_shape.clone(), dev)?;
        let sin_table = Tensor::from_f32(&sin_raw, table_shape, dev)?;

        let mut model = Self {
            config,
            backend,
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_table,
            sin_table,
            // Synthetic models keep an F32 cache so tests can compare against exact
            // references; loaded models default to F16 (`LoadOptions::kv_dtype`).
            kv_dtype: DType::F32,
            kv_cache: Default::default(),
        };
        model.fuse_projections()?;
        model.pack_for_backend()?;
        Ok(model)
    }

    /// Lets the backend convert every linear weight into its preferred layout
    /// ([`Backend::pack_weight`]), then re-fuses the Q/K/V and gate/up projections so the
    /// fused matrices use the packed layout too (packing can also make the dtypes of a
    /// layer's parts agree, enabling fusion). A tied embedding table follows the output
    /// head. Idempotent.
    pub fn pack_for_backend(&mut self) -> Result<()> {
        let backend = self.backend.clone();
        let mut changed = false;
        let mut pack = |t: &mut Tensor| -> Result<()> {
            if let Some(p) = backend.pack_weight(t)? {
                *t = p;
                changed = true;
            }
            Ok(())
        };
        for layer in &mut self.layers {
            let (a, m) = (&mut layer.self_attn, &mut layer.mlp);
            for w in [
                &mut a.q_proj,
                &mut a.k_proj,
                &mut a.v_proj,
                &mut a.o_proj,
                &mut m.gate_proj,
                &mut m.up_proj,
                &mut m.down_proj,
            ] {
                pack(w)?;
            }
        }
        let tied = self.lm_head.storage().as_cpu()?.as_bytes().as_ptr()
            == self.embed_tokens.storage().as_cpu()?.as_bytes().as_ptr();
        pack(&mut self.lm_head)?;
        if tied {
            self.embed_tokens = self.lm_head.clone();
        }
        if changed {
            for layer in &mut self.layers {
                layer.self_attn.qkv_proj = None;
                layer.mlp.gate_up_proj = None;
            }
            self.fuse_projections()?;
        }
        Ok(())
    }

    /// Concatenates each layer's Q/K/V and gate/up weights into single matrices and
    /// re-points the individual projections at row views of them (no extra memory).
    /// The forward pass then runs one linear layer for Q/K/V and one for gate/up,
    /// quantizing activations once per fused matrix. Idempotent; call it again after
    /// replacing individual projection tensors. Layers whose parts differ in dtype or
    /// width are left unfused.
    pub fn fuse_projections(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            let a = &mut layer.self_attn;
            if let Some(fused) = concat_rows(&[&a.q_proj, &a.k_proj, &a.v_proj])? {
                let (nq, nk) = (a.q_proj.dims()[0], a.k_proj.dims()[0]);
                let nv = a.v_proj.dims()[0];
                a.q_proj = fused.slice_rows(0, nq)?;
                a.k_proj = fused.slice_rows(nq, nk)?;
                a.v_proj = fused.slice_rows(nq + nk, nv)?;
                a.qkv_proj = Some(fused);
            }
            let m = &mut layer.mlp;
            if let Some(fused) = concat_rows(&[&m.gate_proj, &m.up_proj])? {
                let ng = m.gate_proj.dims()[0];
                let nu = m.up_proj.dims()[0];
                m.gate_proj = fused.slice_rows(0, ng)?;
                m.up_proj = fused.slice_rows(ng, nu)?;
                m.gate_up_proj = Some(fused);
            }
        }
        Ok(())
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
        // Run the whole pass on a Rayon worker: the many small parallel kernels inside
        // then use work-stealing instead of each injecting into the pool from outside
        // and blocking the caller.
        rayon::scope(|_| self.forward_inner(batch, k_caches, v_caches))
    }

    fn forward_inner(
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
            let normed = self
                .backend
                .rms_norm(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            // Fused projections pay off for decode-sized batches (one activation
            // quantization, fewer parallel regions); for prefill-sized batches the separate
            // row views are faster (no column split, smaller GEMMs). Measured on an M4.
            let use_fused = num_rows < FUSED_MAX_ROWS;
            let (mut q, mut k, v) = match layer.self_attn.qkv_proj.as_ref().filter(|_| use_fused) {
                Some(qkv) => {
                    let fused = self.backend.linear(&normed, qkv)?;
                    let kv_dim = num_kv_heads * head_dim;
                    split_columns(&fused, &[h, kv_dim, kv_dim], self.backend.device())?
                }
                None => (
                    self.backend.linear(&normed, &layer.self_attn.q_proj)?,
                    self.backend.linear(&normed, &layer.self_attn.k_proj)?,
                    self.backend.linear(&normed, &layer.self_attn.v_proj)?,
                ),
            };

            q = q.reshape(Shape::new(vec![num_rows, num_q_heads, head_dim]))?;
            k = k.reshape(Shape::new(vec![num_rows, num_kv_heads, head_dim]))?;

            // Rotary embeddings, each row at its own absolute position.
            self.backend
                .rope(&mut q, &mut k, &self.cos_table, &self.sin_table, batch.positions)?;

            // Write every row's K and V into its sequence's physical cache slot (the
            // cache may be F32 or F16).
            let k_cache = &mut k_caches[layer_idx];
            let v_cache = &mut v_caches[layer_idx];
            write_kv(k_cache, k.as_f32_slice()?, batch, num_kv_heads, head_dim)?;
            write_kv(v_cache, v.as_f32_slice()?, batch, num_kv_heads, head_dim)?;

            // Causal paged attention: row i sees positions 0..=positions[i].
            let attn_out =
                self.backend
                    .paged_attention(&q, k_cache, v_cache, batch.block_tables, &context_lens, scale)?;

            let attn_flat = attn_out.reshape(Shape::new(vec![num_rows, h]))?;
            let attn_proj = self.backend.linear(&attn_flat, &layer.self_attn.o_proj)?;
            add_inplace(&mut hidden, &attn_proj)?;

            // Post-attention RMSNorm + SwiGLU MLP
            let ffn_normed =
                self.backend
                    .rms_norm(&hidden, &layer.post_attention_layernorm, self.config.rms_norm_eps)?;
            let act = match layer.mlp.gate_up_proj.as_ref().filter(|_| use_fused) {
                Some(gu) => {
                    let fused = self.backend.linear(&ffn_normed, gu)?;
                    swiglu_fused(&fused, self.config.intermediate_size, self.backend.device())?
                }
                None => {
                    let gate = self.backend.linear(&ffn_normed, &layer.mlp.gate_proj)?;
                    let up = self.backend.linear(&ffn_normed, &layer.mlp.up_proj)?;
                    self.backend.swiglu(&gate, &up)?
                }
            };
            let ffn_out = self.backend.linear(&act, &layer.mlp.down_proj)?;
            add_inplace(&mut hidden, &ffn_out)?;
        }

        // 3. Final norm + LM head, only for the requested rows.
        let selected =
            if batch.logits_rows.len() == num_rows && batch.logits_rows.iter().enumerate().all(|(i, &r)| i == r) {
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
        let final_normed = self.backend.rms_norm(&selected, &self.norm, self.config.rms_norm_eps)?;
        self.backend.linear(&final_normed, &self.lm_head)
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

/// Batches with fewer rows than this use the fused Q/K/V and gate/up projections.
const FUSED_MAX_ROWS: usize = 32;

/// Row-concatenates matrices with the same width and dtype into one contiguous tensor.
/// Returns `None` when they differ (the layer then stays unfused).
/// Copies each row's `[num_kv_heads, head_dim]` keys or values into its paged cache
/// slot, converting to the cache's storage type (F32 or F16).
fn write_kv(
    cache: &mut Tensor,
    src: &[f32],
    batch: &ForwardBatch<'_>,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    use half::slice::HalfFloatSliceExt;
    let dtype = cache.dtype();
    let storage = cache.storage_mut().as_cpu_mut()?;
    for (row, (&pos, bt)) in batch.positions.iter().zip(batch.block_tables).enumerate() {
        let block_stride = num_kv_heads * bt.block_size * head_dim;
        let phys_block_id = bt.block_ids[pos / bt.block_size];
        for kv_h in 0..num_kv_heads {
            let s = row * (num_kv_heads * head_dim) + kv_h * head_dim;
            let d = phys_block_id * block_stride + kv_h * (bt.block_size * head_dim) + (pos % bt.block_size) * head_dim;
            let src = &src[s..s + head_dim];
            match dtype {
                DType::F32 => storage.as_mut_slice::<f32>()[d..d + head_dim].copy_from_slice(src),
                DType::F16 => storage.as_mut_slice::<half::f16>()[d..d + head_dim].convert_from_f32_slice(src),
                other => {
                    return Err(RllmError::BackendError(format!(
                        "KV cache dtype {other} is not supported (use F32 or F16)"
                    )))
                }
            }
        }
    }
    Ok(())
}

fn concat_rows(parts: &[&Tensor]) -> Result<Option<Tensor>> {
    let cols = parts[0].dims()[1];
    let dtype = parts[0].dtype();
    if parts
        .iter()
        .any(|t| t.dims().len() != 2 || t.dims()[1] != cols || t.dtype() != dtype)
    {
        return Ok(None);
    }
    let rows: usize = parts.iter().map(|t| t.dims()[0]).sum();
    let mut bytes = Vec::with_capacity(dtype.storage_bytes(rows * cols));
    for t in parts {
        let len = dtype.storage_bytes(t.numel());
        bytes.extend_from_slice(&t.storage().as_cpu()?.as_bytes()[t.offset()..t.offset() + len]);
    }
    Ok(Some(Tensor::new(
        Shape::new(vec![rows, cols]),
        dtype,
        parts[0].device(),
        rllm_core::storage::Storage::Cpu(rllm_core::storage::CpuStorage::from_bytes(bytes)),
    )))
}

/// Splits the columns of `[rows, sum(widths)]` into three contiguous tensors.
fn split_columns(x: &Tensor, widths: &[usize; 3], dev: rllm_core::device::Device) -> Result<(Tensor, Tensor, Tensor)> {
    let total: usize = widths.iter().sum();
    let src = x.as_f32_slice()?;
    let rows = src.len() / total;
    let mut parts: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (p, &w) in parts.iter_mut().zip(widths) {
        p.reserve(rows * w);
    }
    for row in src.chunks_exact(total) {
        let mut off = 0;
        for (p, &w) in parts.iter_mut().zip(widths) {
            p.extend_from_slice(&row[off..off + w]);
            off += w;
        }
    }
    let t = |v: &Vec<f32>, w: usize| Tensor::from_f32(v, Shape::new(vec![rows, w]), dev);
    Ok((
        t(&parts[0], widths[0])?,
        t(&parts[1], widths[1])?,
        t(&parts[2], widths[2])?,
    ))
}

/// SwiGLU over a fused `[rows, 2 * inter]` gate/up output: `silu(gate) * up`.
fn swiglu_fused(gu: &Tensor, inter: usize, dev: rllm_core::device::Device) -> Result<Tensor> {
    use rayon::prelude::*;
    let src = gu.as_f32_slice()?;
    let rows = src.len() / (2 * inter);
    let mut out = vec![0f32; rows * inter];
    out.par_chunks_mut(inter)
        .zip(src.par_chunks(2 * inter))
        .with_min_len((16 * 1024 / inter.max(1)).max(1))
        .for_each(|(o, r)| {
            let (g, u) = r.split_at(inter);
            for ((o, &g), &u) in o.iter_mut().zip(g).zip(u) {
                *o = g / (1.0 + (-g).exp()) * u;
            }
        });
    Tensor::from_f32(&out, Shape::new(vec![rows, inter]), dev)
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

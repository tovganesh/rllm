//! BERT-family encoder for dense text embeddings (bge, MiniLM, ...).
//!
//! Standard post-norm BERT: token + position + token-type embeddings → LayerNorm →
//! N × [multi-head bidirectional self-attention → residual → LayerNorm → GELU FFN →
//! residual → LayerNorm] → pooling (mean or `[CLS]`) → L2 normalisation.
//!
//! Real weights are loaded from GGUF files with `general.architecture = "bert"`;
//! [`BertEmbeddingModel::new_random`] builds a seeded synthetic model for tests.

use crate::config::{BertConfig, BertPooling};
use crate::gguf::{GgufValue, MappedGguf};
use crate::init::SyntheticInit;
use crate::loader::load_tensor;
use crate::tokenizer::Tokenizer;
use crate::wordpiece::WordPieceTokenizer;
use rayon::prelude::*;
use rllm_backend_cpu::ops::activations::gelu_f32;
use rllm_backend_cpu::ops::gemm::dot_f32;
use rllm_backend_cpu::ops::norm::layer_norm_f32;
use rllm_core::{
    backend::Backend,
    device::Device,
    error::{Result, RllmError},
    shape::Shape,
    tensor::Tensor,
};
use std::path::Path;
use std::sync::Arc;
use tracing::info;

/// Weights of one encoder layer. Linear weights are `[out, in]`; biases and norm
/// parameters are plain vectors.
pub struct BertLayer {
    pub q_proj: Tensor,
    pub q_bias: Vec<f32>,
    pub k_proj: Tensor,
    pub k_bias: Vec<f32>,
    pub v_proj: Tensor,
    pub v_bias: Vec<f32>,
    pub o_proj: Tensor,
    pub o_bias: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub ffn_up: Tensor,
    pub ffn_up_bias: Vec<f32>,
    pub ffn_down: Tensor,
    pub ffn_down_bias: Vec<f32>,
    pub out_norm_weight: Vec<f32>,
    pub out_norm_bias: Vec<f32>,
}

/// Transformer encoder that maps token IDs to an L2-normalised embedding vector.
pub struct BertEmbeddingModel<B: Backend> {
    pub config: BertConfig,
    pub backend: Arc<B>,
    /// `[vocab, hidden]`, F32 or quantized.
    pub word_embeddings: Tensor,
    /// `[max_positions, hidden]`.
    pub position_embeddings: Vec<f32>,
    /// Row 0 of the token-type table (single-segment inputs).
    pub token_type_embedding: Vec<f32>,
    pub embeddings_norm_weight: Vec<f32>,
    pub embeddings_norm_bias: Vec<f32>,
    pub layers: Vec<BertLayer>,
}

impl<B: Backend> BertEmbeddingModel<B> {
    /// Creates a synthetic model with deterministic pseudo-random weights.
    pub fn new_random(config: BertConfig, backend: Arc<B>) -> Result<Self> {
        let dev = backend.device();
        let (h, inter) = (config.hidden_size, config.intermediate_size);
        let mut init = SyntheticInit::new(0xBE27);
        let mut vec = |n: usize, scale: f32| -> Vec<f32> { (0..n).map(|_| init.next_f32() * scale).collect() };
        let ones = vec![1.0f32; h];
        let zeros = vec![0.0f32; h];

        let position_embeddings = vec(config.max_position_embeddings * h, 0.1);
        let token_type_embedding = vec(h, 0.1);
        let mut init = SyntheticInit::new(0xBE28);
        let word_embeddings = init.uniform(&[config.vocab_size, h], 1.0, dev)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer {
                q_proj: init.linear(h, h, dev)?,
                q_bias: zeros.clone(),
                k_proj: init.linear(h, h, dev)?,
                k_bias: zeros.clone(),
                v_proj: init.linear(h, h, dev)?,
                v_bias: zeros.clone(),
                o_proj: init.linear(h, h, dev)?,
                o_bias: zeros.clone(),
                attn_norm_weight: ones.clone(),
                attn_norm_bias: zeros.clone(),
                ffn_up: init.linear(inter, h, dev)?,
                ffn_up_bias: vec![0.0; inter],
                ffn_down: init.linear(h, inter, dev)?,
                ffn_down_bias: zeros.clone(),
                out_norm_weight: ones.clone(),
                out_norm_bias: zeros.clone(),
            });
        }
        Ok(Self {
            config,
            backend,
            word_embeddings,
            position_embeddings,
            token_type_embedding,
            embeddings_norm_weight: ones,
            embeddings_norm_bias: zeros,
            layers,
        })
    }

    /// Loads a BERT-architecture embedding model from an opened GGUF file.
    pub fn from_mapped(g: &MappedGguf, backend: Arc<B>) -> Result<Self> {
        let config = BertConfig::from_gguf(&g.header)?;
        if backend.device() != Device::Cpu {
            return Err(RllmError::ModelError("GGUF loading currently targets the CPU backend only".to_string()));
        }
        let (h, inter) = (config.hidden_size, config.intermediate_size);
        let vecf = |name: &str, n: usize| -> Result<Vec<f32>> { load_tensor(g, name, &[n], false)?.to_vec::<f32>() };
        let mat = |name: &str, shape: &[usize]| load_tensor(g, name, shape, true);

        let word_embeddings = mat("token_embd.weight", &[config.vocab_size, h])?;
        let position_embeddings =
            load_tensor(g, "position_embd.weight", &[config.max_position_embeddings, h], false)?.to_vec::<f32>()?;
        let token_type_embedding = match g.tensor_info("token_types.weight") {
            Some(info) => {
                let rows = info.shape.first().copied().unwrap_or(1);
                let all = load_tensor(g, "token_types.weight", &[rows, h], false)?.to_vec::<f32>()?;
                all[..h].to_vec()
            }
            None => vec![0.0; h],
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = |n: &str| format!("blk.{i}.{n}");
            layers.push(BertLayer {
                q_proj: mat(&p("attn_q.weight"), &[h, h])?,
                q_bias: vecf(&p("attn_q.bias"), h)?,
                k_proj: mat(&p("attn_k.weight"), &[h, h])?,
                k_bias: vecf(&p("attn_k.bias"), h)?,
                v_proj: mat(&p("attn_v.weight"), &[h, h])?,
                v_bias: vecf(&p("attn_v.bias"), h)?,
                o_proj: mat(&p("attn_output.weight"), &[h, h])?,
                o_bias: vecf(&p("attn_output.bias"), h)?,
                attn_norm_weight: vecf(&p("attn_output_norm.weight"), h)?,
                attn_norm_bias: vecf(&p("attn_output_norm.bias"), h)?,
                ffn_up: mat(&p("ffn_up.weight"), &[inter, h])?,
                ffn_up_bias: vecf(&p("ffn_up.bias"), inter)?,
                ffn_down: mat(&p("ffn_down.weight"), &[h, inter])?,
                ffn_down_bias: vecf(&p("ffn_down.bias"), h)?,
                out_norm_weight: vecf(&p("layer_output_norm.weight"), h)?,
                out_norm_bias: vecf(&p("layer_output_norm.bias"), h)?,
            });
        }
        info!(
            "Embedding model: {} layers, hidden {}, {} heads, {:?} pooling, max {} tokens",
            config.num_hidden_layers, h, config.num_attention_heads, config.pooling, config.max_position_embeddings
        );
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embedding,
            embeddings_norm_weight: vecf("token_embd_norm.weight", h)?,
            embeddings_norm_bias: vecf("token_embd_norm.bias", h)?,
            layers,
            config,
            backend,
        })
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Encodes token IDs into an L2-normalised embedding.
    ///
    /// Inputs longer than `max_position_embeddings` are truncated (keeping the final
    /// token, typically `[SEP]`, when truncation happens).
    ///
    /// # Errors
    ///
    /// Returns an error if `token_ids` is empty or contains an ID outside the vocabulary.
    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        // See `LlamaModel::forward`: run inside the Rayon pool.
        rayon::scope(|_| self.embed_inner(token_ids))
    }

    fn embed_inner(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RllmError::BackendError("embed: input must contain at least one token".to_string()));
        }
        let max = self.config.max_position_embeddings;
        let ids: Vec<u32> = if token_ids.len() > max {
            let mut v = token_ids[..max - 1].to_vec();
            v.push(*token_ids.last().unwrap());
            v
        } else {
            token_ids.to_vec()
        };
        let n = ids.len();
        let h = self.config.hidden_size;
        let eps = self.config.layer_norm_eps;
        let dev = self.backend.device();

        // 1. Embeddings
        let tok = self.backend.embedding(&self.word_embeddings, &ids)?;
        let mut x = tok.to_vec::<f32>()?;
        for (pos, row) in x.chunks_mut(h).enumerate() {
            let p = &self.position_embeddings[pos * h..(pos + 1) * h];
            for ((v, pe), te) in row.iter_mut().zip(p).zip(&self.token_type_embedding) {
                *v += pe + te;
            }
        }
        let mut x = layer_norm(&x, &self.embeddings_norm_weight, &self.embeddings_norm_bias, n, h, eps);

        // 2. Encoder layers
        for layer in &self.layers {
            let xt = Tensor::from_f32(&x, Shape::new(vec![n, h]), dev)?;
            let q = self.linear_bias(&xt, &layer.q_proj, &layer.q_bias)?;
            let k = self.linear_bias(&xt, &layer.k_proj, &layer.k_bias)?;
            let v = self.linear_bias(&xt, &layer.v_proj, &layer.v_bias)?;
            let ctx = self.attention(&q, &k, &v, n);
            let attn = self.linear_bias(&Tensor::from_f32(&ctx, Shape::new(vec![n, h]), dev)?, &layer.o_proj, &layer.o_bias)?;
            for (a, b) in x.iter_mut().zip(&attn) {
                *a += b;
            }
            x = layer_norm(&x, &layer.attn_norm_weight, &layer.attn_norm_bias, n, h, eps);

            let xt = Tensor::from_f32(&x, Shape::new(vec![n, h]), dev)?;
            let up = self.linear_bias(&xt, &layer.ffn_up, &layer.ffn_up_bias)?;
            let mut act = vec![0.0f32; up.len()];
            gelu_f32(&up, &mut act);
            let inter = self.config.intermediate_size;
            let down = self.linear_bias(&Tensor::from_f32(&act, Shape::new(vec![n, inter]), dev)?, &layer.ffn_down, &layer.ffn_down_bias)?;
            for (a, b) in x.iter_mut().zip(&down) {
                *a += b;
            }
            x = layer_norm(&x, &layer.out_norm_weight, &layer.out_norm_bias, n, h, eps);
        }

        // 3. Pooling
        let mut pooled = match self.config.pooling {
            BertPooling::Cls => x[..h].to_vec(),
            BertPooling::Mean => {
                let mut p = vec![0.0f32; h];
                for row in x.chunks(h) {
                    for (acc, v) in p.iter_mut().zip(row) {
                        *acc += v;
                    }
                }
                p.iter_mut().for_each(|v| *v /= n as f32);
                p
            }
        };

        // 4. L2 normalisation
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        pooled.iter_mut().for_each(|v| *v /= norm);
        Ok(pooled)
    }

    fn linear_bias(&self, x: &Tensor, w: &Tensor, bias: &[f32]) -> Result<Vec<f32>> {
        let mut y = self.backend.linear(x, w)?.to_vec::<f32>()?;
        for row in y.chunks_mut(bias.len()) {
            for (v, b) in row.iter_mut().zip(bias) {
                *v += b;
            }
        }
        Ok(y)
    }

    /// Bidirectional multi-head attention over `n` tokens (no mask).
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], n: usize) -> Vec<f32> {
        let h = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let d = h / heads;
        let scale = 1.0 / (d as f32).sqrt();
        // Compute per (head, query) in parallel, then scatter into [n, h].
        let per_head: Vec<Vec<f32>> = (0..heads * n)
            .into_par_iter()
            .map(|idx| {
                let (head, i) = (idx / n, idx % n);
                let qi = &q[i * h + head * d..i * h + (head + 1) * d];
                let mut scores: Vec<f32> =
                    (0..n).map(|j| dot_f32(qi, &k[j * h + head * d..j * h + (head + 1) * d]) * scale).collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let mut out = vec![0.0f32; d];
                for (j, s) in scores.iter().enumerate() {
                    let w = s / sum;
                    for (o, vv) in out.iter_mut().zip(&v[j * h + head * d..j * h + (head + 1) * d]) {
                        *o += w * vv;
                    }
                }
                out
            })
            .collect();
        let mut ctx = vec![0.0f32; n * h];
        for (idx, out) in per_head.into_iter().enumerate() {
            let (head, i) = (idx / n, idx % n);
            ctx[i * h + head * d..i * h + (head + 1) * d].copy_from_slice(&out);
        }
        ctx
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], n: usize, h: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * h];
    layer_norm_f32(x, w, Some(b), &mut out, n, h, eps);
    out
}

/// Tokenizer used by a [`TextEmbedder`].
#[derive(Debug, Clone)]
pub enum EmbeddingTokenizer {
    /// WordPiece tokenizer loaded with a real BERT model.
    WordPiece(WordPieceTokenizer),
    /// Legacy word-list tokenizer for synthetic models in tests.
    Legacy(Tokenizer),
}

/// A text-in, vector-out embedding model: tokenizer plus encoder.
pub struct TextEmbedder<B: Backend> {
    /// Encoder weights.
    pub model: BertEmbeddingModel<B>,
    /// Tokenizer matching the encoder's vocabulary.
    pub tokenizer: EmbeddingTokenizer,
    /// Model name (`general.name`) or a synthetic label.
    pub name: String,
}

impl<B: Backend> TextEmbedder<B> {
    /// Loads a BERT-architecture embedding model and its WordPiece tokenizer from GGUF.
    pub fn from_gguf(path: impl AsRef<Path>, backend: Arc<B>) -> Result<Self> {
        let path = path.as_ref();
        let g = MappedGguf::open(path)?;
        let tokenizer = WordPieceTokenizer::from_gguf(&g.header)?;
        let model = BertEmbeddingModel::from_mapped(&g, backend)?;
        if tokenizer.vocab_size() != model.config.vocab_size {
            return Err(RllmError::ModelError(format!(
                "Tokenizer has {} tokens but the embedding model expects {}",
                tokenizer.vocab_size(),
                model.config.vocab_size
            )));
        }
        let name = g
            .header
            .get("general.name")
            .and_then(GgufValue::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
        info!("Loaded embedding model {} from {}", name, path.display());
        Ok(Self {
            model,
            tokenizer: EmbeddingTokenizer::WordPiece(tokenizer),
            name,
        })
    }

    /// Synthetic embedder (random weights, word-list tokenizer) for tests and demos.
    pub fn synthetic(config: BertConfig, backend: Arc<B>) -> Result<Self> {
        Ok(Self {
            model: BertEmbeddingModel::new_random(config, backend)?,
            tokenizer: EmbeddingTokenizer::Legacy(Tokenizer::new()),
            name: "synthetic-bert".to_string(),
        })
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.model.dim()
    }

    /// Cosine-similarity cutoff separating related from unrelated query/passage pairs,
    /// for embedding models that have been calibrated (see `docs/ARCHITECTURE.md`).
    /// Returns `None` for unknown models.
    ///
    /// Calibration (paraphrased vs. off-topic questions over a small factual corpus):
    /// bge-small-en-v1.5 related ≥ 0.67, unrelated ≤ 0.55 → 0.6;
    /// all-MiniLM-L6-v2 related ≥ 0.42, unrelated ≤ 0.14 → 0.3.
    pub fn calibrated_dense_threshold(&self) -> Option<f32> {
        let name = self.name.to_lowercase();
        if name.contains("bge") {
            Some(0.6)
        } else if name.contains("minilm") {
            Some(0.3)
        } else {
            None
        }
    }

    /// Whether this embedder has real (trained) weights.
    pub fn is_synthetic(&self) -> bool {
        matches!(self.tokenizer, EmbeddingTokenizer::Legacy(_))
    }

    /// Token IDs the embedder would see for `text`.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        match &self.tokenizer {
            EmbeddingTokenizer::WordPiece(t) => t.encode(text, true, false),
            EmbeddingTokenizer::Legacy(t) => {
                let max = self.model.config.vocab_size as u32;
                t.encode(text).into_iter().map(|id| id % max).collect()
            }
        }
    }

    /// Embeds a text into an L2-normalised vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenize(text);
        if ids.is_empty() {
            return Err(RllmError::BackendError("embed: text produced no tokens".to_string()));
        }
        self.model.embed(&ids)
    }
}

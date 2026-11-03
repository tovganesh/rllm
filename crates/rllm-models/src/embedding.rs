//! Bidirectional transformer encoder model for high-throughput dense vector embeddings.

use crate::config::BertConfig;
use rllm_backend_cpu::ops::activations::gelu_f32;
use rllm_backend_cpu::ops::norm::layer_norm_f32;
use rllm_core::{
    backend::Backend,
    error::Result,
    shape::Shape,
    tensor::Tensor,
};
use std::sync::Arc;

/// Single bidirectional transformer encoder layer.
pub struct BertLayer {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub attn_norm: Tensor,
    pub intermediate_proj: Tensor,
    pub output_proj: Tensor,
    pub output_norm: Tensor,
}

/// Transformer encoder model that converts token ID sequences to L2-normalized unit embeddings.
pub struct BertEmbeddingModel<B: Backend> {
    pub config: BertConfig,
    pub backend: Arc<B>,
    pub word_embeddings: Tensor,
    pub position_embeddings: Tensor,
    pub token_type_embeddings: Tensor,
    pub embeddings_norm: Tensor,
    pub layers: Vec<BertLayer>,
}

impl<B: Backend> BertEmbeddingModel<B> {
    /// Creates a synthetic model with pseudo-random weights.
    pub fn new_random(config: BertConfig, backend: Arc<B>) -> Result<Self> {
        let dev = backend.device();
        let h = config.hidden_size;
        let inter = config.intermediate_size;

        let mut word_data = Vec::with_capacity(config.vocab_size * h);
        for i in 0..(config.vocab_size * h) {
            word_data.push(((i % 17) as f32 - 8.0) * 0.02);
        }
        let word_embeddings = Tensor::from_f32(
            &word_data,
            Shape::new(vec![config.vocab_size, h]),
            dev,
        )?;

        let mut pos_data = Vec::with_capacity(config.max_position_embeddings * h);
        for i in 0..(config.max_position_embeddings * h) {
            pos_data.push(((i % 13) as f32 - 6.0) * 0.01);
        }
        let position_embeddings = Tensor::from_f32(
            &pos_data,
            Shape::new(vec![config.max_position_embeddings, h]),
            dev,
        )?;
        let token_type_embeddings =
            Tensor::from_f32(&vec![0.01f32; 2 * h], Shape::new(vec![2, h]), dev)?;
        let embeddings_norm = Tensor::from_f32(&vec![1.0f32; h], Shape::new(vec![h]), dev)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer {
                q_proj: Tensor::from_f32(&vec![0.01f32; h * h], Shape::new(vec![h, h]), dev)?,
                k_proj: Tensor::from_f32(&vec![0.01f32; h * h], Shape::new(vec![h, h]), dev)?,
                v_proj: Tensor::from_f32(&vec![0.01f32; h * h], Shape::new(vec![h, h]), dev)?,
                o_proj: Tensor::from_f32(&vec![0.01f32; h * h], Shape::new(vec![h, h]), dev)?,
                attn_norm: Tensor::from_f32(&vec![1.0f32; h], Shape::new(vec![h]), dev)?,
                intermediate_proj: Tensor::from_f32(
                    &vec![0.01f32; h * inter],
                    Shape::new(vec![inter, h]),
                    dev,
                )?,
                output_proj: Tensor::from_f32(
                    &vec![0.01f32; inter * h],
                    Shape::new(vec![h, inter]),
                    dev,
                )?,
                output_norm: Tensor::from_f32(&vec![1.0f32; h], Shape::new(vec![h]), dev)?,
            });
        }

        Ok(Self {
            config,
            backend,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_norm,
            layers,
        })
    }

    /// Forward pass generating a normalized embedding vector for input token IDs.
    /// Performs mean-pooling over sequence tokens followed by L2 normalization.
    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        let h = self.config.hidden_size;

        // 1. Embedding lookup
        let mut hidden = self.backend.embedding(&self.word_embeddings, token_ids)?;
        let hidden_slice = hidden.as_mut_f32_slice()?;

        // Add position embeddings
        let pos_slice = self.position_embeddings.as_f32_slice()?;
        for pos in 0..seq_len {
            for d in 0..h {
                hidden_slice[pos * h + d] += pos_slice[pos * h + d];
            }
        }

        // LayerNorm on embeddings
        let norm_weight = self.embeddings_norm.as_f32_slice()?;
        let mut normed_buf = vec![0.0f32; seq_len * h];
        layer_norm_f32(
            hidden_slice,
            norm_weight,
            None,
            &mut normed_buf,
            seq_len,
            h,
            self.config.layer_norm_eps,
        );
        hidden_slice.copy_from_slice(&normed_buf);

        // 2. Transformer layers
        for layer in &self.layers {
            let q = self
                .backend
                .matmul(&hidden, &layer.q_proj.transpose(0, 1)?)?;
            let k = self
                .backend
                .matmul(&hidden, &layer.k_proj.transpose(0, 1)?)?;
            let v = self
                .backend
                .matmul(&hidden, &layer.v_proj.transpose(0, 1)?)?;

            let attn_scores = self.backend.matmul(&q, &k.transpose(0, 1)?)?;
            let attn_probs = self.backend.softmax(&attn_scores)?;
            let attn_out = self.backend.matmul(&attn_probs, &v)?;
            let proj_out = self
                .backend
                .matmul(&attn_out, &layer.o_proj.transpose(0, 1)?)?;

            let hidden_slice = hidden.as_mut_f32_slice()?;
            let proj_slice = proj_out.as_f32_slice()?;
            for i in 0..hidden_slice.len() {
                hidden_slice[i] += proj_slice[i];
            }

            let mut post_attn = vec![0.0f32; seq_len * h];
            layer_norm_f32(
                hidden_slice,
                layer.attn_norm.as_f32_slice()?,
                None,
                &mut post_attn,
                seq_len,
                h,
                self.config.layer_norm_eps,
            );

            // Feed-forward: intermediate_proj -> gelu -> output_proj
            let inter_t = Tensor::from_f32(
                &post_attn,
                Shape::new(vec![seq_len, h]),
                self.backend.device(),
            )?;
            let inter_out = self
                .backend
                .matmul(&inter_t, &layer.intermediate_proj.transpose(0, 1)?)?;
            let mut gelu_buf = vec![0.0f32; inter_out.numel()];
            gelu_f32(inter_out.as_f32_slice()?, &mut gelu_buf);
            let gelu_t = Tensor::from_f32(
                &gelu_buf,
                Shape::new(vec![seq_len, self.config.intermediate_size]),
                self.backend.device(),
            )?;
            let ffn_out = self
                .backend
                .matmul(&gelu_t, &layer.output_proj.transpose(0, 1)?)?;

            let ffn_slice = ffn_out.as_f32_slice()?;
            for i in 0..hidden_slice.len() {
                hidden_slice[i] = post_attn[i] + ffn_slice[i];
            }

            let mut post_ffn = vec![0.0f32; seq_len * h];
            layer_norm_f32(
                hidden_slice,
                layer.output_norm.as_f32_slice()?,
                None,
                &mut post_ffn,
                seq_len,
                h,
                self.config.layer_norm_eps,
            );
            hidden_slice.copy_from_slice(&post_ffn);
        }

        // 3. Mean Pooling over all sequence tokens
        let mut pooled = vec![0.0f32; h];
        let hidden_slice = hidden.as_f32_slice()?;
        for pos in 0..seq_len {
            for d in 0..h {
                pooled[d] += hidden_slice[pos * h + d];
            }
        }
        for d in 0..h {
            pooled[d] /= seq_len as f32;
        }

        // 4. L2 Normalization -> Unit Vector on hypersphere
        let norm_sq: f32 = pooled.iter().map(|&v| v * v).sum();
        let inv_norm = 1.0f32 / (norm_sq.sqrt().max(1e-12));
        for d in 0..h {
            pooled[d] *= inv_norm;
        }

        Ok(pooled)
    }
}

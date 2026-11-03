//! # `rllm-models`
//!
//! Model architecture implementations and weight loaders for `rllm`.
//!
//! Includes:
//! - Grouped-Query Attention (GQA) LLaMA 3.x / Mistral / Qwen 2.5 with RoPE and PagedAttention decoding.
//! - BERT / ModernBERT embedding model with mean pooling and L2 normalization for dense RAG retrieval.
//! - Binary GGUF format parser for edge block-quantized weights.
//! - Named weight storage and mapping dictionary.

pub mod config;
pub mod embedding;
pub mod gguf;
pub mod llama;
pub mod weights;

pub use config::{BertConfig, LlamaConfig};
pub use embedding::BertEmbeddingModel;
pub use gguf::{GgufFile, GgufTensorInfo};
pub use llama::LlamaModel;
pub use weights::Weights;

#[cfg(test)]
mod tests {
    use super::*;
    use rllm_backend_cpu::CpuBackend;
    use rllm_core::{
        backend::BlockTable,
        device::Device,
        dtype::DType,
        shape::Shape,
        tensor::Tensor,
    };
    use std::sync::Arc;

    #[test]
    fn test_llama_forward_decode() {
        let backend = Arc::new(CpuBackend);
        let config = LlamaConfig::tiny_test_config();
        let model = LlamaModel::new_random(config.clone(), backend).unwrap();

        let block_size = 16;
        let num_blocks = 4;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();

        let mut k_caches = vec![
            Tensor::zeros(
                Shape::new(vec![num_blocks, num_kv_heads, block_size, head_dim]),
                DType::F32,
                Device::Cpu,
            )
            .unwrap(),
            Tensor::zeros(
                Shape::new(vec![num_blocks, num_kv_heads, block_size, head_dim]),
                DType::F32,
                Device::Cpu,
            )
            .unwrap(),
        ];
        let mut v_caches = vec![
            Tensor::zeros(
                Shape::new(vec![num_blocks, num_kv_heads, block_size, head_dim]),
                DType::F32,
                Device::Cpu,
            )
            .unwrap(),
            Tensor::zeros(
                Shape::new(vec![num_blocks, num_kv_heads, block_size, head_dim]),
                DType::F32,
                Device::Cpu,
            )
            .unwrap(),
        ];

        let tokens = vec![42u32];
        let block_tables = vec![BlockTable {
            block_size,
            block_ids: vec![0],
        }];
        let context_lens = vec![1];

        let logits = model
            .forward_decode(
                &tokens,
                &mut k_caches,
                &mut v_caches,
                &block_tables,
                &context_lens,
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, config.vocab_size]);
    }

    #[test]
    fn test_bert_embedding() {
        let backend = Arc::new(CpuBackend);
        let config = BertConfig::tiny_test_config();
        let model = BertEmbeddingModel::new_random(config.clone(), backend).unwrap();

        let token_ids = vec![101u32, 205, 200, 102];
        let emb = model.embed(&token_ids).unwrap();

        assert_eq!(emb.len(), config.hidden_size);
        let norm_sq: f32 = emb.iter().map(|&v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-4);
    }
}

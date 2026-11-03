//! Model architecture configurations for autoregressive transformers and embedding models.
//!
//! Provides configuration structs, dimension calculations, memory footprint estimation,
//! and standard configurations (such as standard Llama, edge-4gb, Llama 3.2 1B, and BERT).

use serde::{Deserialize, Serialize};

/// Configuration parameters for LLaMA-style decoder architectures.
///
/// Encapsulates model dimensions, attention head counts, RMSNorm epsilon,
/// RoPE rotary base theta, and maximum context lengths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    /// Hidden dimension ($d_{model}$) of token representations.
    pub hidden_size: usize,
    /// Intermediate dimension of the SwiGLU MLP feed-forward network.
    pub intermediate_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value attention heads for Grouped Query Attention (GQA).
    pub num_key_value_heads: usize,
    /// Total number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Size of the token vocabulary.
    pub vocab_size: usize,
    /// Epsilon value added to variance for numerical stability in RMSNorm.
    pub rms_norm_eps: f32,
    /// Base frequency ($\theta$) for Rotary Position Embeddings (RoPE).
    pub rope_theta: f32,
    /// Maximum context position index supported by the model.
    pub max_position_embeddings: usize,
}

impl Default for LlamaConfig {
    /// Default to Llama-3-8B configuration.
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8, // Grouped Query Attention (GQA) 4:1
            num_hidden_layers: 32,
            vocab_size: 128256,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 8192,
        }
    }
}

impl LlamaConfig {
    /// Tiny configuration useful for fast unit tests, smartphone edge testing, and CI.
    pub fn tiny_test_config() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            vocab_size: 1000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }

    /// Dimension of each individual attention head ($d_{head} = \frac{d_{model}}{n_{heads}}$).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Model architecture designed specifically to fit comfortably in a 4GB machine (iOS, Android, Raspberry Pi, Laptop).
    ///
    /// Total parameters: ~400M.
    /// Memory footprint in FP32: ~1.6 GB. In FP16: ~800 MB. In Q4_0: ~225 MB.
    /// Leaves ample headroom (>2GB) for the operating system, KV-cache, and user apps.
    pub fn edge_4gb() -> Self {
        Self {
            hidden_size: 1536,
            intermediate_size: 4096,
            num_attention_heads: 12,
            num_key_value_heads: 4, // Grouped Query Attention 3:1
            num_hidden_layers: 12,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 4096,
        }
    }

    /// LLaMA-3.2-1B class architecture (~1.2B parameters).
    /// Fits in ~700 MB in Q4_0 or ~2.4 GB in FP16.
    pub fn llama_3_2_1b() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 8192,
            num_attention_heads: 32,
            num_key_value_heads: 8, // Grouped Query Attention 4:1
            num_hidden_layers: 16,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 8192,
        }
    }

    /// Total count of trainable / active parameters in the model.
    pub fn num_parameters(&self) -> usize {
        let h = self.hidden_size;
        let head_dim = self.head_dim();
        let kv_dim = self.num_key_value_heads * head_dim;
        let inter = self.intermediate_size;

        let embedding_params = self.vocab_size * h;
        let attn_per_layer = (h * h) + 2 * (kv_dim * h) + (h * h);
        let mlp_per_layer = 3 * (inter * h);
        let norms_per_layer = 2 * h;
        let layer_params = (attn_per_layer + mlp_per_layer + norms_per_layer) * self.num_hidden_layers;
        let final_norm = h;
        let lm_head = self.vocab_size * h;

        embedding_params + layer_params + final_norm + lm_head
    }

    /// Estimated memory footprint in Megabytes: `(weights_mb, kv_cache_mb, total_mb)`.
    pub fn estimated_memory_mb(
        &self,
        dtype: rllm_core::dtype::DType,
        context_len: usize,
        batch_size: usize,
    ) -> (f64, f64, f64) {
        let params = self.num_parameters();
        let weights_bytes = dtype.storage_bytes(params);
        let weights_mb = weights_bytes as f64 / (1024.0 * 1024.0);

        // KV cache: 2 (K and V) * layers * kv_heads * head_dim * context_len * batch_size * element_size
        let head_dim = self.head_dim();
        let kv_elements = 2 * self.num_hidden_layers * self.num_key_value_heads * head_dim * context_len * batch_size;
        let kv_bytes = kv_elements * 4; // F32 KV cache
        let kv_mb = kv_bytes as f64 / (1024.0 * 1024.0);

        let total_mb = weights_mb + kv_mb;
        (weights_mb, kv_mb, total_mb)
    }
}

/// Configuration parameters for BERT-style encoder architectures used for embeddings and reranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    /// Hidden dimension of token representations.
    pub hidden_size: usize,
    /// Intermediate dimension of the position-wise feed-forward network.
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of encoder layers.
    pub num_hidden_layers: usize,
    /// Size of the vocabulary.
    pub vocab_size: usize,
    /// Epsilon value for LayerNorm.
    pub layer_norm_eps: f32,
    /// Maximum position embeddings supported.
    pub max_position_embeddings: usize,
}

impl Default for BertConfig {
    /// Standard BGE / MiniLM embedding model configuration.
    fn default() -> Self {
        Self {
            hidden_size: 384,
            intermediate_size: 1536,
            num_attention_heads: 12,
            num_hidden_layers: 6,
            vocab_size: 30522,
            layer_norm_eps: 1e-12,
            max_position_embeddings: 512,
        }
    }
}

impl BertConfig {
    /// Tiny configuration useful for fast unit tests.
    pub fn tiny_test_config() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            vocab_size: 1000,
            layer_norm_eps: 1e-5,
            max_position_embeddings: 128,
        }
    }

    /// Attention head dimension ($d_{head} = \frac{d_{model}}{n_{heads}}$).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

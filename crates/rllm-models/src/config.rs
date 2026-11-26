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
    /// How token states are pooled into a single vector.
    pub pooling: BertPooling,
}

/// Pooling strategy of a BERT-style embedding model (`bert.pooling_type` in GGUF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BertPooling {
    /// Mean of all token states (e.g. all-MiniLM-L6-v2).
    Mean,
    /// State of the first (`[CLS]`) token (e.g. bge).
    Cls,
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
            pooling: BertPooling::Mean,
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
            pooling: BertPooling::Mean,
        }
    }

    /// Attention head dimension ($d_{head} = \frac{d_{model}}{n_{heads}}$).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Reads a BERT configuration from GGUF metadata (`general.architecture = "bert"`).
    pub fn from_gguf(g: &crate::gguf::GgufFile) -> rllm_core::error::Result<Self> {
        use rllm_core::error::RllmError;
        let arch = g.get_str("general.architecture")?;
        if arch != "bert" {
            return Err(RllmError::ModelError(format!(
                "Unsupported embedding architecture '{arch}' (only 'bert' is supported)"
            )));
        }
        if g.get("bert.attention.causal").and_then(crate::gguf::GgufValue::as_bool) == Some(true) {
            return Err(RllmError::ModelError(
                "Causal BERT models are not supported".to_string(),
            ));
        }
        let pooling = match g.get("bert.pooling_type").and_then(crate::gguf::GgufValue::as_u64) {
            None | Some(1) => BertPooling::Mean,
            Some(2) => BertPooling::Cls,
            Some(other) => return Err(RllmError::ModelError(format!("Unsupported bert.pooling_type {other}"))),
        };
        let hidden_size = g.get_u64("bert.embedding_length")? as usize;
        let num_attention_heads = g.get_u64("bert.attention.head_count")? as usize;
        if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(RllmError::ModelError(
                "hidden size not divisible by head count".to_string(),
            ));
        }
        let vocab_size = g
            .get("tokenizer.ggml.tokens")
            .and_then(crate::gguf::GgufValue::as_array)
            .map(|a| a.len())
            .ok_or_else(|| RllmError::ModelError("tokenizer.ggml.tokens missing".to_string()))?;
        Ok(Self {
            hidden_size,
            intermediate_size: g.get_u64("bert.feed_forward_length")? as usize,
            num_attention_heads,
            num_hidden_layers: g.get_u64("bert.block_count")? as usize,
            vocab_size,
            layer_norm_eps: g.get_f64("bert.attention.layer_norm_epsilon")? as f32,
            max_position_embeddings: g.get_u64("bert.context_length")? as usize,
            pooling,
        })
    }
}

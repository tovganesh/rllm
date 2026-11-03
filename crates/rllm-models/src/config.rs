use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // Default to Llama-3-8B configuration
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
    /// Tiny configuration useful for fast unit tests, smartphone edge testing, and CI
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
            max_position_embeddings: 512,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Model architecture designed specifically to fit comfortably in a 4GB machine (iOS, Android, Raspberry Pi, Laptop).
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

    /// Estimated memory footprint in Megabytes: (weights_mb, kv_cache_mb, total_mb)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
    pub max_position_embeddings: usize,
}

impl Default for BertConfig {
    fn default() -> Self {
        // Standard BGE / MiniLM embedding model config
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

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}


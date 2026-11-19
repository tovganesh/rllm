//! Executor-agnostic language-model interface used by the scheduler.
//!
//! An implementation owns its paged KV cache (CPU tensors, GPU buffers, ...). The
//! scheduler only manages block tables and calls [`LanguageModel::forward_cached`]
//! with flattened batches.

use crate::config::LlamaConfig;
use crate::llama::{ForwardBatch, LlamaModel};
use rllm_core::{
    backend::Backend,
    error::{Result, RllmError},
    shape::Shape,
    tensor::Tensor,
};

/// A causal language model with an internal paged KV cache.
pub trait LanguageModel: Send + Sync {
    /// Model hyper-parameters.
    fn config(&self) -> &LlamaConfig;

    /// Human-readable executor name (e.g. `"cpu"`, `"metal"`).
    fn device_name(&self) -> String;

    /// Allocates (or re-allocates, discarding contents) a KV cache of `num_blocks`
    /// blocks of `block_size` tokens per layer.
    fn init_kv_cache(&self, num_blocks: usize, block_size: usize) -> Result<()>;

    /// Runs one forward pass over `batch` (see [`LlamaModel::forward`] for semantics)
    /// using the internal KV cache. Returns logits for `batch.logits_rows`, flattened
    /// to `logits_rows.len() * vocab_size` values.
    fn forward_cached(&self, batch: &ForwardBatch<'_>) -> Result<Vec<f32>>;
}

/// Per-layer K and V cache tensors `[num_blocks, kv_heads, block_size, head_dim]`.
#[derive(Default)]
pub(crate) struct KvCache {
    pub(crate) k: Vec<Tensor>,
    pub(crate) v: Vec<Tensor>,
}

impl<B: Backend> LanguageModel for LlamaModel<B> {
    fn config(&self) -> &LlamaConfig {
        &self.config
    }

    fn device_name(&self) -> String {
        self.backend.device().to_string()
    }

    fn init_kv_cache(&self, num_blocks: usize, block_size: usize) -> Result<()> {
        let c = &self.config;
        let dev = self.backend.device();
        let shape = Shape::new(vec![num_blocks, c.num_key_value_heads, block_size, c.head_dim()]);
        let mut cache = KvCache::default();
        for _ in 0..c.num_hidden_layers {
            cache.k.push(Tensor::zeros(shape.clone(), self.kv_dtype, dev)?);
            cache.v.push(Tensor::zeros(shape.clone(), self.kv_dtype, dev)?);
        }
        *self.kv_cache.lock().map_err(|_| RllmError::BackendError("KV cache lock poisoned".into()))? = cache;
        Ok(())
    }

    fn forward_cached(&self, batch: &ForwardBatch<'_>) -> Result<Vec<f32>> {
        let mut cache = self
            .kv_cache
            .lock()
            .map_err(|_| RllmError::BackendError("KV cache lock poisoned".into()))?;
        if cache.k.is_empty() {
            return Err(RllmError::BackendError("KV cache not initialised; call init_kv_cache".into()));
        }
        let KvCache { k, v } = &mut *cache;
        self.forward(batch, k, v)?.to_vec::<f32>()
    }
}

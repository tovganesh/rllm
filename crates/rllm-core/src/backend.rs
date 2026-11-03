use crate::device::Device;
use crate::error::Result;
use crate::tensor::Tensor;

/// Block table mapping logical blocks to physical block indices in PagedAttention KV cache.
#[derive(Debug, Clone)]
pub struct BlockTable {
    pub block_size: usize,
    pub block_ids: Vec<usize>,
}

pub trait Backend: Send + Sync + 'static {
    fn device(&self) -> Device;

    /// Matrix multiplication: out = A @ B
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Root Mean Square Normalization: out = (x / sqrt(mean(x^2) + eps)) * weight
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// In-place or functional Rotary Position Embedding (RoPE)
    fn rope(&self, q: &mut Tensor, k: &mut Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<()>;

    /// SwiGLU activation: out = (x_gate * sigmoid(x_gate)) * x_up
    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor>;

    /// Softmax along the last dimension
    fn softmax(&self, x: &Tensor) -> Result<Tensor>;

    /// Embedding table lookup: retrieves embedding rows for given token IDs
    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor>;

    /// PagedAttention decoding kernel:
    /// Computes multi-head attention for query tokens against paged KV cache blocks.
    fn paged_attention(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        block_tables: &[BlockTable],
        context_lens: &[usize],
        scale: f32,
    ) -> Result<Tensor>;

    /// Quantized GEMV: multiplies quantized weights (Q4_0 or Q8_0) with float activation vector
    fn gemv_quantized(&self, weight: &Tensor, act: &Tensor) -> Result<Tensor>;
}


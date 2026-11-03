//! Backend execution traits for hardware-agnostic tensor compute.

use crate::device::Device;
use crate::error::Result;
use crate::tensor::Tensor;

/// Block table mapping logical sequence blocks to physical block indices
/// in the PagedAttention virtual memory KV cache pool.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Fixed number of tokens per physical memory block (typically 16).
    pub block_size: usize,
    /// Physical block IDs currently allocated to this sequence in ascending logical order.
    pub block_ids: Vec<usize>,
}

/// Hardware backend abstraction interface.
///
/// Implemented by `CpuBackend`, `MetalBackend`, and `CudaBackend`.
/// Provides high-level dispatch for fundamental neural network primitives
/// optimized for autoregressive decoding and continuous batching.
pub trait Backend: Send + Sync + 'static {
    /// Returns the target hardware device for this backend instance.
    fn device(&self) -> Device;

    /// Matrix multiplication: $C = A \times B$.
    /// Supports 2D GEMM and 1D vector GEMV.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Root Mean Square Normalization (RMSNorm):
    /// $$y_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}} \cdot w_i$$
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// Rotary Positional Embedding (RoPE):
    /// Rotates pairs of elements $(x_{2i}, x_{2i+1})$ using precomputed $\cos$ and $\sin$ frequencies.
    fn rope(&self, q: &mut Tensor, k: &mut Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<()>;

    /// SwiGLU activation function:
    /// $$\text{SwiGLU}(x_{\text{gate}}, x_{\text{up}}) = (x_{\text{gate}} \cdot \sigma(x_{\text{gate}})) \cdot x_{\text{up}}$$
    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor>;

    /// Numerically stable softmax along the last dimension.
    fn softmax(&self, x: &Tensor) -> Result<Tensor>;

    /// Embedding table lookup: gathers embedding rows for a slice of input token IDs.
    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor>;

    /// PagedAttention decoding kernel:
    /// Computes multi-head attention for query vectors against non-contiguous physical KV blocks.
    fn paged_attention(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        block_tables: &[BlockTable],
        context_lens: &[usize],
        scale: f32,
    ) -> Result<Tensor>;

    /// Quantized GEMV: Multiplies a block-quantized weight matrix (Q4_0 or Q8_0)
    /// directly with a floating-point activation vector with zero full-matrix dequantization overhead.
    fn gemv_quantized(&self, weight: &Tensor, act: &Tensor) -> Result<Tensor>;
}

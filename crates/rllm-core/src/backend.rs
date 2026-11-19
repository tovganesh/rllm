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
/// Implemented by `CpuBackend`. (GPU execution uses whole-model executors such as
/// `rllm_backend_metal::MetalLlama` instead of per-operation dispatch.)
/// Provides high-level dispatch for fundamental neural network primitives
/// optimized for autoregressive decoding and continuous batching.
pub trait Backend: Send + Sync + 'static {
    /// Returns the target hardware device for this backend instance.
    fn device(&self) -> Device;

    /// Matrix multiplication: $C = A \times B$.
    /// Supports 2D GEMM and 1D vector GEMV.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Linear layer without bias: `y[..., out] = x[..., in] · Wᵀ`.
    ///
    /// `w` is stored `[out_features, in_features]` and may be F32 or a block-quantized
    /// type supported by the backend (on CPU: Q8_0, Q4_0, Q5_0, Q4_K, Q6_K and the
    /// repacked Q4_0x4). Quantized weights are used directly, without materialising an
    /// F32 copy.
    fn linear(&self, x: &Tensor, w: &Tensor) -> Result<Tensor>;

    /// Optionally converts a `[out, in]` linear weight into a backend-preferred layout
    /// with identical values (e.g. interleaved rows for SIMD kernels). Returns `None`
    /// when the weight should stay as it is. The result must still be accepted by
    /// [`Backend::linear`] and [`Backend::embedding`]. Row views of a packed weight
    /// are only valid at the backend's row granularity (4 rows for CPU `Q4_0x4`).
    fn pack_weight(&self, _w: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Root Mean Square Normalization (RMSNorm):
    /// $$y_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}} \cdot w_i$$
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// Rotary Positional Embedding (RoPE), applied in place.
    ///
    /// Rotates adjacent pairs $(x_{2i}, x_{2i+1})$ of every head of every token using the
    /// row of the precomputed tables selected by that token's position.
    ///
    /// * `q` — `[num_tokens, num_heads_q, head_dim]`
    /// * `k` — `[num_tokens, num_heads_kv, head_dim]`
    /// * `cos_table`, `sin_table` — `[max_positions, head_dim / 2]`
    /// * `positions` — absolute position of each of the `num_tokens` rows
    fn rope(
        &self,
        q: &mut Tensor,
        k: &mut Tensor,
        cos_table: &Tensor,
        sin_table: &Tensor,
        positions: &[usize],
    ) -> Result<()>;

    /// SwiGLU activation function:
    /// $$\text{SwiGLU}(x_{\text{gate}}, x_{\text{up}}) = (x_{\text{gate}} \cdot \sigma(x_{\text{gate}})) \cdot x_{\text{up}}$$
    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor>;

    /// Numerically stable softmax along the last dimension.
    fn softmax(&self, x: &Tensor) -> Result<Tensor>;

    /// Embedding table lookup: gathers rows of `table` (`[vocab, hidden]`, F32 or a
    /// supported quantized type) for the given token IDs into an F32 `[ids.len(), hidden]`.
    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor>;

    /// PagedAttention kernel.
    ///
    /// Each row `i` of `q` (`[num_rows, num_heads_q, head_dim]`) attends to the first
    /// `context_lens[i]` positions of the sequence described by `block_tables[i]`.
    /// Passing one row per prompt token with `context_lens[i] = position + 1` yields
    /// causal prefill attention; one row per sequence yields batched decode.
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

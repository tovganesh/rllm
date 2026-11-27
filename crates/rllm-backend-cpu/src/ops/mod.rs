pub mod activations;
pub mod blas;
pub mod f16w;
pub mod gemm;
pub mod kdot;
pub mod norm;
pub mod paged_attn;
pub mod qdot;
pub mod quant;
pub mod repack;
pub mod rope;
pub mod softmax;
pub mod tiles;
#[cfg(target_arch = "x86_64")]
pub mod x86;

/// Minimum number of elements per parallel task for cheap element-wise work. Below
/// this, Rayon runs the loop inline on the calling thread (no fork/join overhead).
pub const ELEMENTWISE_GRAIN: usize = 16 * 1024;

/// Target bytes of weight data streamed per parallel task in matrix-vector products.
pub fn gemv_task_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("RLLM_TASK_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128 * 1024)
    })
}

/// Rows per task for a matrix-vector product whose rows are `row_bytes` long.
pub fn gemv_rows_per_task(row_bytes: usize) -> usize {
    (gemv_task_bytes() / row_bytes.max(1)).max(8)
}

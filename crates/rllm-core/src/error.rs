use thiserror::Error;

#[derive(Error, Debug)]
pub enum RllmError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Dimension out of range: dim {dim} for shape {shape:?}")]
    DimOutOfRange {
        dim: usize,
        shape: Vec<usize>,
    },

    #[error("Incompatible shapes for broadcast: {lhs:?} and {rhs:?}")]
    BroadcastIncompatible {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    #[error("Device mismatch: tensor on {actual:?}, expected {expected:?}")]
    DeviceMismatch {
        expected: String,
        actual: String,
    },

    #[error("DType mismatch: tensor is {actual:?}, expected {expected:?}")]
    DTypeMismatch {
        expected: String,
        actual: String,
    },

    #[error("Unsupported operation '{op}' for dtype {dtype:?} on device {device:?}")]
    UnsupportedOp {
        op: &'static str,
        dtype: String,
        device: String,
    },

    #[error("Out of memory: requested {requested_bytes} bytes on {device:?}")]
    OutOfMemory {
        requested_bytes: usize,
        device: String,
    },

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Scheduler error: {0}")]
    SchedulerError(String),

    #[error("RAG error: {0}")]
    RagError(String),
}

pub type Result<T> = std::result::Result<T, RllmError>;


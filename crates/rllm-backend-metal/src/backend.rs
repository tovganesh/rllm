//! Metal compute backend driver for Apple Silicon and iOS devices.

use crate::shaders::SHADERS_MSL;
use rllm_core::{
    backend::{Backend, BlockTable},
    device::Device,
    error::{Result, RllmError},
    tensor::Tensor,
};
use tracing::info;

/// Apple Silicon Metal execution backend.
#[derive(Debug, Clone)]
pub struct MetalBackend {
    /// Device identifier for multi-GPU Apple systems.
    pub device_id: usize,
    /// Availability flag on host OS (true on macOS and iOS).
    pub is_available: bool,
}

impl MetalBackend {
    /// Initializes the Metal backend for the specified GPU device index.
    pub fn new(device_id: usize) -> Self {
        #[cfg(target_os = "macos")]
        let available = true;
        #[cfg(not(target_os = "macos"))]
        let available = false;

        if available {
            info!("MetalBackend initialized for Apple Silicon / Metal device {}", device_id);
        }

        Self {
            device_id,
            is_available: available,
        }
    }

    /// Returns embedded MSL shader source strings.
    pub fn msl_source(&self) -> &'static str {
        SHADERS_MSL
    }
}

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        Device::Metal(self.device_id)
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        Err(RllmError::BackendError(
            "Direct Metal GPU pipeline dispatch requires compiled metal pipeline state".to_string(),
        ))
    }

    fn rms_norm(&self, _x: &Tensor, _weight: &Tensor, _eps: f32) -> Result<Tensor> {
        Err(RllmError::BackendError(
            "Metal rms_norm pipeline stub".to_string(),
        ))
    }

    fn rope(
        &self,
        _q: &mut Tensor,
        _k: &mut Tensor,
        _freqs_cos: &Tensor,
        _freqs_sin: &Tensor,
    ) -> Result<()> {
        Err(RllmError::BackendError(
            "Metal rope pipeline stub".to_string(),
        ))
    }

    fn swiglu(&self, _gate: &Tensor, _up: &Tensor) -> Result<Tensor> {
        Err(RllmError::BackendError(
            "Metal swiglu pipeline stub".to_string(),
        ))
    }

    fn softmax(&self, _x: &Tensor) -> Result<Tensor> {
        Err(RllmError::BackendError(
            "Metal softmax pipeline stub".to_string(),
        ))
    }

    fn embedding(&self, _table: &Tensor, _ids: &[u32]) -> Result<Tensor> {
        Err(RllmError::BackendError(
            "Metal embedding pipeline stub".to_string(),
        ))
    }

    fn paged_attention(
        &self,
        _q: &Tensor,
        _k_cache: &Tensor,
        _v_cache: &Tensor,
        _block_tables: &[BlockTable],
        _context_lens: &[usize],
        _scale: f32,
    ) -> Result<Tensor> {
        Err(RllmError::BackendError(
            "Metal paged_attention pipeline stub".to_string(),
        ))
    }

    fn gemv_quantized(&self, _weight: &Tensor, _act: &Tensor) -> Result<Tensor> {
        Err(RllmError::BackendError(
            "Metal gemv_quantized pipeline stub".to_string(),
        ))
    }
}

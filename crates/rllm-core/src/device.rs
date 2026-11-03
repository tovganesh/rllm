//! Target hardware compute devices.

use serde::{Deserialize, Serialize};

/// Target hardware device for tensor allocation and compute dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    /// Host CPU using SIMD vector instructions (ARM NEON, x86 AVX2/AVX-512).
    Cpu,
    /// Apple Silicon GPU (macOS / iOS) using Metal compute shaders and unified memory.
    Metal(usize),
    /// NVIDIA GPU using CUDA and cuBLAS runtime.
    Cuda(usize),
}

impl Device {
    /// Returns true if the device is a host CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns true if the device is an Apple Silicon Metal GPU.
    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal(_))
    }

    /// Returns true if the device is an NVIDIA CUDA GPU.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Selects the best available default device for the host platform.
    pub fn default_device() -> Self {
        #[cfg(target_os = "macos")]
        {
            Device::Metal(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Device::Cpu
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Metal(id) => write!(f, "metal:{}", id),
            Device::Cuda(id) => write!(f, "cuda:{}", id),
        }
    }
}

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Metal(usize),
    Cuda(usize),
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal(_))
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

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


//! Physical storage buffers and device memory pointers.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, RllmError};
use std::sync::Arc;

/// Physical storage buffer backend for tensors.
#[derive(Debug, Clone)]
pub enum Storage {
    /// Host CPU memory buffer wrapped in a reference-counted byte array.
    Cpu(CpuStorage),
    /// Apple Silicon Metal GPU memory buffer descriptor.
    Metal(MetalStorage),
    /// NVIDIA CUDA GPU device memory pointer.
    Cuda(CudaStorage),
}

/// Contiguous, aligned host CPU memory buffer.
#[derive(Debug, Clone)]
pub struct CpuStorage {
    data: Arc<Vec<u8>>,
}

impl CpuStorage {
    /// Allocates an empty zeroed byte buffer of `size_in_bytes`.
    pub fn new(size_in_bytes: usize) -> Self {
        Self {
            data: Arc::new(vec![0u8; size_in_bytes]),
        }
    }

    /// Creates storage by copying data from a generic slice.
    pub fn from_vec<T: Copy>(data: &[T]) -> Self {
        let byte_len = std::mem::size_of_val(data);
        let mut bytes = vec![0u8; byte_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                byte_len,
            );
        }
        Self {
            data: Arc::new(bytes),
        }
    }

    /// Creates storage from an existing vector of bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            data: Arc::new(bytes),
        }
    }

    /// Read-only slice access to underlying raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Mutable slice access to underlying raw bytes (triggers copy-on-write if shared).
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Typed read-only slice cast.
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        let ptr = self.data.as_ptr() as *const T;
        let len = self.data.len() / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Typed mutable slice cast.
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        let len = self.data.len() / std::mem::size_of::<T>();
        let ptr = Arc::make_mut(&mut self.data).as_mut_ptr() as *mut T;
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

/// Metal memory buffer descriptor for Apple Silicon unified memory.
#[derive(Debug, Clone)]
pub struct MetalStorage {
    pub buffer_address: usize,
    pub length: usize,
    pub device_id: usize,
}

/// CUDA device memory pointer descriptor.
#[derive(Debug, Clone)]
pub struct CudaStorage {
    pub device_ptr: u64,
    pub length: usize,
    pub device_id: usize,
}

impl Storage {
    /// Allocates storage memory for a given device, dtype, and element count.
    pub fn allocate(device: Device, dtype: DType, numel: usize) -> Result<Self> {
        let bytes = dtype.storage_bytes(numel);
        match device {
            Device::Cpu => Ok(Storage::Cpu(CpuStorage::new(bytes))),
            Device::Metal(id) => Ok(Storage::Metal(MetalStorage {
                buffer_address: 0,
                length: bytes,
                device_id: id,
            })),
            Device::Cuda(id) => Ok(Storage::Cuda(CudaStorage {
                device_ptr: 0,
                length: bytes,
                device_id: id,
            })),
        }
    }

    /// Device where this storage buffer resides.
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Metal(m) => Device::Metal(m.device_id),
            Storage::Cuda(c) => Device::Cuda(c.device_id),
        }
    }

    /// Casts storage to a CPU storage reference, returning an error if located on another device.
    pub fn as_cpu(&self) -> Result<&CpuStorage> {
        match self {
            Storage::Cpu(s) => Ok(s),
            other => Err(RllmError::DeviceMismatch {
                expected: "cpu".to_string(),
                actual: other.device().to_string(),
            }),
        }
    }

    /// Casts storage to a mutable CPU storage reference.
    pub fn as_cpu_mut(&mut self) -> Result<&mut CpuStorage> {
        let dev = self.device().to_string();
        match self {
            Storage::Cpu(s) => Ok(s),
            _ => Err(RllmError::DeviceMismatch {
                expected: "cpu".to_string(),
                actual: dev,
            }),
        }
    }
}

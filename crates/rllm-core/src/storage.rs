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

/// Alignment (in bytes) guaranteed for the start of every [`CpuStorage`] buffer.
///
/// 64 bytes covers every scalar type used by kernels as well as cache-line and
/// AVX-512 / NEON vector alignment.
pub const CPU_STORAGE_ALIGN: usize = 64;

/// One cache-line sized, 64-byte aligned unit of backing memory.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct AlignedChunk([u8; CPU_STORAGE_ALIGN]);

/// Contiguous, 64-byte aligned host CPU memory buffer.
///
/// Backed by a vector of aligned chunks so that reinterpreting the bytes as
/// `&[f32]` (or any type with alignment <= 64) is always sound.
#[derive(Clone)]
pub struct CpuStorage {
    data: Arc<Vec<AlignedChunk>>,
    len: usize,
}

impl std::fmt::Debug for CpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuStorage").field("len", &self.len).finish()
    }
}

impl CpuStorage {
    /// Allocates an empty zeroed byte buffer of `size_in_bytes`.
    pub fn new(size_in_bytes: usize) -> Self {
        let chunks = size_in_bytes.div_ceil(CPU_STORAGE_ALIGN);
        Self {
            data: Arc::new(vec![AlignedChunk([0u8; CPU_STORAGE_ALIGN]); chunks]),
            len: size_in_bytes,
        }
    }

    /// Creates storage by copying data from a generic slice.
    pub fn from_vec<T: Copy>(data: &[T]) -> Self {
        let byte_len = std::mem::size_of_val(data);
        let mut storage = Self::new(byte_len);
        // SAFETY: `data` is a valid slice of `byte_len` bytes, and the destination
        // buffer was just allocated with at least `byte_len` bytes.
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        storage.as_mut_bytes().copy_from_slice(src);
        storage
    }

    /// Creates storage by copying an existing vector of bytes into aligned memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let mut storage = Self::new(bytes.len());
        storage.as_mut_bytes().copy_from_slice(&bytes);
        storage
    }

    /// Length of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer holds zero bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read-only slice access to underlying raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the chunk vector holds at least `len` initialised bytes.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.len) }
    }

    /// Mutable slice access to underlying raw bytes (triggers copy-on-write if shared).
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        let len = self.len;
        let ptr = Arc::make_mut(&mut self.data).as_mut_ptr() as *mut u8;
        // SAFETY: the chunk vector holds at least `len` initialised bytes and we hold
        // a unique reference after `make_mut`.
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Typed read-only slice cast.
    ///
    /// # Panics
    ///
    /// Panics if `T` requires more than [`CPU_STORAGE_ALIGN`] bytes of alignment.
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        assert!(std::mem::align_of::<T>() <= CPU_STORAGE_ALIGN);
        let len = self.len / std::mem::size_of::<T>();
        // SAFETY: the buffer start is 64-byte aligned (checked above to satisfy `T`),
        // and `len * size_of::<T>() <= self.len` initialised bytes are available.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, len) }
    }

    /// Typed mutable slice cast.
    ///
    /// # Panics
    ///
    /// Panics if `T` requires more than [`CPU_STORAGE_ALIGN`] bytes of alignment.
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        assert!(std::mem::align_of::<T>() <= CPU_STORAGE_ALIGN);
        let len = self.len / std::mem::size_of::<T>();
        let ptr = Arc::make_mut(&mut self.data).as_mut_ptr() as *mut T;
        // SAFETY: as in `as_slice`, plus uniqueness from `make_mut`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_storage_is_aligned_and_roundtrips() {
        for n in [0usize, 1, 3, 17, 1000] {
            let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
            let s = CpuStorage::from_vec(&data);
            assert_eq!(s.len(), n * 4);
            assert_eq!(s.as_bytes().as_ptr() as usize % CPU_STORAGE_ALIGN, 0);
            assert_eq!(s.as_slice::<f32>(), data.as_slice());
        }
        let s = CpuStorage::from_bytes(vec![1, 2, 3]);
        assert_eq!(s.as_bytes(), &[1, 2, 3]);
        assert_eq!(s.as_bytes().as_ptr() as usize % CPU_STORAGE_ALIGN, 0);
    }
}

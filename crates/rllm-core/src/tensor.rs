use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, RllmError};
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    strides: Vec<usize>,
    dtype: DType,
    device: Device,
    storage: Storage,
    offset: usize,
}

impl Tensor {
    pub fn new(shape: Shape, dtype: DType, device: Device, storage: Storage) -> Self {
        let strides = shape.default_strides();
        Self {
            shape,
            strides,
            dtype,
            device,
            storage,
            offset: 0,
        }
    }

    pub fn from_f32(data: &[f32], shape: Shape, device: Device) -> Result<Self> {
        if shape.numel() != data.len() {
            return Err(RllmError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        let storage = match device {
            Device::Cpu => Storage::Cpu(CpuStorage::from_vec(data)),
            _ => Storage::allocate(device, DType::F32, shape.numel())?,
        };
        Ok(Self::new(shape, DType::F32, device, storage))
    }

    pub fn zeros(shape: Shape, dtype: DType, device: Device) -> Result<Self> {
        let storage = Storage::allocate(device, dtype, shape.numel())?;
        Ok(Self::new(shape, dtype, device, storage))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.default_strides()
    }

    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        if self.numel() != new_shape.numel() {
            return Err(RllmError::ShapeMismatch {
                expected: vec![self.numel()],
                got: vec![new_shape.numel()],
            });
        }
        let strides = new_shape.default_strides();
        Ok(Self {
            shape: new_shape,
            strides,
            dtype: self.dtype,
            device: self.device,
            storage: self.storage.clone(),
            offset: self.offset,
        })
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.shape.rank() || dim1 >= self.shape.rank() {
            return Err(RllmError::DimOutOfRange {
                dim: dim0.max(dim1),
                shape: self.shape.dims().to_vec(),
            });
        }
        let mut new_dims = self.shape.dims().to_vec();
        let mut new_strides = self.strides.clone();
        new_dims.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(Self {
            shape: Shape::new(new_dims),
            strides: new_strides,
            dtype: self.dtype,
            device: self.device,
            storage: self.storage.clone(),
            offset: self.offset,
        })
    }

    pub fn to_vec<T: Copy>(&self) -> Result<Vec<T>> {
        let cpu_storage = self.storage.as_cpu()?;
        let slice: &[T] = cpu_storage.as_slice();
        let numel = self.numel();
        let start = self.offset / std::mem::size_of::<T>();
        if start + numel > slice.len() {
            return Err(RllmError::BackendError(
                "Tensor slice bounds out of range for storage".to_string(),
            ));
        }
        Ok(slice[start..start + numel].to_vec())
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(RllmError::DTypeMismatch {
                expected: "f32".to_string(),
                actual: self.dtype.to_string(),
            });
        }
        let cpu = self.storage.as_cpu()?;
        let all: &[f32] = cpu.as_slice();
        let offset = self.offset / 4;
        Ok(&all[offset..offset + self.numel()])
    }

    pub fn as_mut_f32_slice(&mut self) -> Result<&mut [f32]> {
        if self.dtype != DType::F32 {
            return Err(RllmError::DTypeMismatch {
                expected: "f32".to_string(),
                actual: self.dtype.to_string(),
            });
        }
        let offset = self.offset / 4;
        let numel = self.numel();
        let cpu = self.storage.as_cpu_mut()?;
        let all: &mut [f32] = cpu.as_mut_slice();
        Ok(&mut all[offset..offset + numel])
    }
}

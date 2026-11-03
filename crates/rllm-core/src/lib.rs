pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use backend::{Backend, BlockTable};
pub use device::Device;
pub use dtype::DType;
pub use error::{Result, RllmError};
pub use shape::Shape;
pub use storage::{CpuStorage, Storage};
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let shape = Shape::new(vec![2, 3]);
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_f32(&data, shape, Device::Cpu).unwrap();
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.dims(), &[2, 3]);
        let readback: Vec<f32> = tensor.to_vec().unwrap();
        assert_eq!(readback, data);
    }

    #[test]
    fn test_shape_broadcast() {
        let s1 = Shape::new(vec![2, 1, 4]);
        let s2 = Shape::new(vec![3, 4]);
        let b = Shape::broadcast(&s1, &s2).unwrap();
        assert_eq!(b.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_quantized_dtype_bytes() {
        assert_eq!(DType::Q4_0.storage_bytes(32), 18);
        assert_eq!(DType::Q4_0.storage_bytes(64), 36);
        assert_eq!(DType::Q8_0.storage_bytes(32), 34);
    }
}


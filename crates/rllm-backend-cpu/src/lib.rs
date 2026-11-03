//! # `rllm-backend-cpu`
//!
//! High-performance CPU compute backend for ARM NEON and x86 AVX2/AVX-512 SIMD vector architectures.
//!
//! Provides cache-tiled matrix multiplication, PagedAttention multi-head decoding,
//! vector RMSNorm, Rotary Position Embeddings (RoPE), and direct 4-bit/8-bit quantized GEMV.

pub mod backend;
pub mod ops;

pub use backend::CpuBackend;

#[cfg(test)]
mod tests {
    use super::*;
    use ops::quant::quantize_f32_to_q4_0;
    use rllm_core::{
        backend::{Backend, BlockTable},
        device::Device,
        dtype::DType,
        shape::Shape,
        storage::{CpuStorage, Storage},
        tensor::Tensor,
    };

    #[test]
    fn test_cpu_matmul() {
        let backend = CpuBackend;
        let a = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
            Device::Cpu,
        )
        .unwrap();
        let b = Tensor::from_f32(
            &[7.0, 8.0, 9.0, 1.0, 2.0, 3.0],
            Shape::new(vec![3, 2]),
            Device::Cpu,
        )
        .unwrap();

        let c = backend.matmul(&a, &b).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        let data: Vec<f32> = c.to_vec().unwrap();
        assert_eq!(data, vec![31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_cpu_rmsnorm() {
        let backend = CpuBackend;
        let x = Tensor::from_f32(
            &[2.0, -2.0, 2.0, -2.0],
            Shape::new(vec![1, 4]),
            Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::from_f32(
            &[1.0, 1.0, 1.0, 1.0],
            Shape::new(vec![4]),
            Device::Cpu,
        )
        .unwrap();

        let y = backend.rms_norm(&x, &weight, 1e-6).unwrap();
        let data: Vec<f32> = y.to_vec().unwrap();
        for (a, b) in data.iter().zip(&[1.0f32, -1.0, 1.0, -1.0]) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_quantized_q4_gemv() {
        let backend = CpuBackend;
        let k = 32;
        let n = 2;
        let original_weights = vec![1.0f32; n * k];
        let mut q4_bytes = vec![0u8; n * 18];

        quantize_f32_to_q4_0(&original_weights[0..k], &mut q4_bytes[0..18]);
        quantize_f32_to_q4_0(&original_weights[k..2 * k], &mut q4_bytes[18..36]);

        let weight_tensor = Tensor::new(
            Shape::new(vec![n, k]),
            DType::Q4_0,
            Device::Cpu,
            Storage::Cpu(CpuStorage::from_bytes(q4_bytes)),
        );

        let act = Tensor::from_f32(&vec![1.0f32; k], Shape::new(vec![k]), Device::Cpu).unwrap();
        let res = backend.gemv_quantized(&weight_tensor, &act).unwrap();
        let data: Vec<f32> = res.to_vec().unwrap();

        for &val in &data {
            assert!((val - 32.0).abs() < 1.0);
        }
    }

    #[test]
    fn test_paged_attention_basic() {
        let backend = CpuBackend;
        let num_heads_q = 2;
        let num_heads_kv = 1;
        let head_dim = 4;
        let block_size = 2;
        let context_len = 2;

        let q = Tensor::from_f32(
            &vec![1.0f32; num_heads_q * head_dim],
            Shape::new(vec![1, num_heads_q, head_dim]),
            Device::Cpu,
        )
        .unwrap();

        let k_cache = Tensor::from_f32(
            &vec![1.0f32; 1 * num_heads_kv * block_size * head_dim],
            Shape::new(vec![1, num_heads_kv, block_size, head_dim]),
            Device::Cpu,
        )
        .unwrap();

        let v_cache = Tensor::from_f32(
            &vec![2.0f32; 1 * num_heads_kv * block_size * head_dim],
            Shape::new(vec![1, num_heads_kv, block_size, head_dim]),
            Device::Cpu,
        )
        .unwrap();

        let block_tables = vec![BlockTable {
            block_size,
            block_ids: vec![0],
        }];
        let context_lens = vec![context_len];

        let out = backend
            .paged_attention(
                &q,
                &k_cache,
                &v_cache,
                &block_tables,
                &context_lens,
                1.0 / (head_dim as f32).sqrt(),
            )
            .unwrap();

        assert_eq!(out.dims(), &[1, num_heads_q, head_dim]);
        let data: Vec<f32> = out.to_vec().unwrap();
        for &val in &data {
            assert!((val - 2.0).abs() < 1e-4);
        }
    }
}

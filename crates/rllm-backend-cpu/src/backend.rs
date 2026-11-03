use crate::ops::{
    activations::swiglu_f32,
    gemm::gemm_f32,
    norm::rms_norm_f32,
    paged_attn::paged_attention_decode_single,
    quant::{gemv_q4_0_f32, gemv_q8_0_f32},
    rope::rope_f32,
    softmax::softmax_inplace,
};
use rllm_core::{
    backend::{Backend, BlockTable},
    device::Device,
    dtype::DType,
    error::{Result, RllmError},
    shape::Shape,
    tensor::Tensor,
};

#[derive(Debug, Clone, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_dims = a.dims();
        let b_dims = b.dims();

        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(RllmError::BackendError(
                "Matmul requires tensors of rank >= 2".to_string(),
            ));
        }

        let m = a_dims[a_dims.len() - 2];
        let k_a = a_dims[a_dims.len() - 1];
        let k_b = b_dims[b_dims.len() - 2];
        let n = b_dims[b_dims.len() - 1];

        if k_a != k_b {
            return Err(RllmError::ShapeMismatch {
                expected: vec![m, k_a],
                got: vec![k_b, n],
            });
        }

        let a_slice = a.as_f32_slice()?;
        let b_slice = b.as_f32_slice()?;

        let mut out_dims = a_dims[..a_dims.len() - 2].to_vec();
        out_dims.push(m);
        out_dims.push(n);
        let out_shape = Shape::new(out_dims);

        let mut out = Tensor::zeros(out_shape, DType::F32, Device::Cpu)?;
        let out_slice = out.as_mut_f32_slice()?;

        gemm_f32(a_slice, b_slice, out_slice, m, k_a, n);
        Ok(out)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let hidden_dim = *x.dims().last().ok_or_else(|| {
            RllmError::BackendError("RMSNorm input tensor cannot be scalar".to_string())
        })?;
        let batch_tokens = x.numel() / hidden_dim;

        let mut out = Tensor::zeros(x.shape().clone(), DType::F32, Device::Cpu)?;
        let x_slice = x.as_f32_slice()?;
        let w_slice = weight.as_f32_slice()?;
        let out_slice = out.as_mut_f32_slice()?;

        rms_norm_f32(x_slice, w_slice, out_slice, batch_tokens, hidden_dim, eps);
        Ok(out)
    }

    fn rope(
        &self,
        q: &mut Tensor,
        k: &mut Tensor,
        freqs_cos: &Tensor,
        freqs_sin: &Tensor,
    ) -> Result<()> {
        let q_dims = q.dims();
        let _k_dims = k.dims();

        let head_dim = *q_dims.last().unwrap();
        let num_heads_q = q.numel() / head_dim;
        let num_heads_k = k.numel() / head_dim;

        let cos_slice = freqs_cos.as_f32_slice()?;
        let sin_slice = freqs_sin.as_f32_slice()?;
        let q_slice = q.as_mut_f32_slice()?;
        let k_slice = k.as_mut_f32_slice()?;

        rope_f32(
            q_slice,
            k_slice,
            num_heads_q,
            num_heads_k,
            head_dim,
            cos_slice,
            sin_slice,
        );
        Ok(())
    }

    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        if gate.dims() != up.dims() {
            return Err(RllmError::ShapeMismatch {
                expected: gate.dims().to_vec(),
                got: up.dims().to_vec(),
            });
        }
        let mut out = Tensor::zeros(gate.shape().clone(), DType::F32, Device::Cpu)?;
        let g_slice = gate.as_f32_slice()?;
        let u_slice = up.as_f32_slice()?;
        let out_slice = out.as_mut_f32_slice()?;

        swiglu_f32(g_slice, u_slice, out_slice);
        Ok(out)
    }

    fn softmax(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        let row_len = *x.dims().last().unwrap();
        let out_slice = out.as_mut_f32_slice()?;
        softmax_inplace(out_slice, row_len);
        Ok(out)
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let t_dims = table.dims();
        let vocab_size = t_dims[0];
        let hidden_dim = t_dims[1];

        let table_slice = table.as_f32_slice()?;
        let out_shape = Shape::new(vec![ids.len(), hidden_dim]);
        let mut out = Tensor::zeros(out_shape, DType::F32, Device::Cpu)?;
        let out_slice = out.as_mut_f32_slice()?;

        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= vocab_size {
                return Err(RllmError::BackendError(format!(
                    "Token ID {} exceeds vocabulary size {}",
                    id, vocab_size
                )));
            }
            let src = &table_slice[id * hidden_dim..(id + 1) * hidden_dim];
            let dst = &mut out_slice[i * hidden_dim..(i + 1) * hidden_dim];
            dst.copy_from_slice(src);
        }

        Ok(out)
    }

    fn paged_attention(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        block_tables: &[BlockTable],
        context_lens: &[usize],
        scale: f32,
    ) -> Result<Tensor> {
        let batch_size = block_tables.len();
        let q_dims = q.dims(); // [batch_size, num_heads_q, head_dim]
        let num_heads_q = q_dims[1];
        let head_dim = q_dims[2];

        let k_dims = k_cache.dims(); // [num_blocks, num_heads_kv, block_size, head_dim]
        let num_heads_kv = k_dims[1];

        let mut out = Tensor::zeros(q.shape().clone(), DType::F32, Device::Cpu)?;

        let q_slice = q.as_f32_slice()?;
        let k_slice = k_cache.as_f32_slice()?;
        let v_slice = v_cache.as_f32_slice()?;
        let out_slice = out.as_mut_f32_slice()?;

        let q_stride = num_heads_q * head_dim;

        for b in 0..batch_size {
            let q_b = &q_slice[b * q_stride..(b + 1) * q_stride];
            let out_b = &mut out_slice[b * q_stride..(b + 1) * q_stride];

            paged_attention_decode_single(
                q_b,
                k_slice,
                v_slice,
                &block_tables[b],
                context_lens[b],
                num_heads_q,
                num_heads_kv,
                head_dim,
                scale,
                out_b,
            );
        }

        Ok(out)
    }

    fn gemv_quantized(&self, weight: &Tensor, act: &Tensor) -> Result<Tensor> {
        let w_dims = weight.dims();
        let n = w_dims[0];
        let k = act.numel();

        let mut out = Tensor::zeros(Shape::new(vec![n]), DType::F32, Device::Cpu)?;
        let act_slice = act.as_f32_slice()?;
        let out_slice = out.as_mut_f32_slice()?;

        let cpu_storage = weight.storage().as_cpu()?;
        let bytes = cpu_storage.as_bytes();

        match weight.dtype() {
            DType::Q4_0 => {
                gemv_q4_0_f32(bytes, act_slice, out_slice, n, k);
            }
            DType::Q8_0 => {
                gemv_q8_0_f32(bytes, act_slice, out_slice, n, k);
            }
            other => {
                return Err(RllmError::UnsupportedOp {
                    op: "gemv_quantized",
                    dtype: other.to_string(),
                    device: "cpu".to_string(),
                });
            }
        }

        Ok(out)
    }
}

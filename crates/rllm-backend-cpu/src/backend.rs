use crate::ops::{
    activations::swiglu_f32,
    gemm::{gemm_f32, gemm_nt_f32},
    norm::rms_norm_f32,
    paged_attn::paged_attention_head,
    quant::{gemv_q4_0_f32, gemv_q8_0_f32},
    rope::rope_f32,
    softmax::softmax_inplace,
};
use rayon::prelude::*;
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

    /// Matrix multiplication `A[..., M, K] · B[K, N]`.
    ///
    /// Leading dimensions of `A` are flattened into rows. `B` must be rank 2 and may be
    /// either contiguous or a transposed view of a contiguous `[N, K]` matrix (the
    /// usual `weight.transpose(0, 1)` pattern), which dispatches to a dot-product kernel
    /// without copying. Any other strided operand is materialised first.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_dims = a.dims();
        let b_dims = b.dims();

        if a_dims.len() < 2 || b_dims.len() != 2 {
            return Err(RllmError::BackendError(format!(
                "Matmul requires A of rank >= 2 and B of rank 2, got {:?} x {:?}",
                a_dims, b_dims
            )));
        }

        let k = a_dims[a_dims.len() - 1];
        let (k_b, n) = (b_dims[0], b_dims[1]);
        if k != k_b {
            return Err(RllmError::ShapeMismatch {
                expected: vec![k, n],
                got: b_dims.to_vec(),
            });
        }
        let m = a.numel() / k.max(1);

        let a = a.contiguous()?;
        let a_slice = a.as_f32_slice()?;

        let mut out_dims = a_dims[..a_dims.len() - 1].to_vec();
        out_dims.push(n);
        let mut out = Tensor::zeros(Shape::new(out_dims), DType::F32, Device::Cpu)?;
        let out_slice = out.as_mut_f32_slice()?;

        // B is a transposed view of a contiguous [N, K] matrix: strides == [1, K].
        let is_transposed_contiguous = b.strides() == [1, k] && (n > 1 || k > 1);
        if b.is_contiguous() {
            gemm_f32(a_slice, b.as_f32_slice()?, out_slice, m, k, n);
        } else if is_transposed_contiguous {
            gemm_nt_f32(a_slice, b.storage_f32_slice()?, out_slice, m, k, n);
        } else {
            let b = b.contiguous()?;
            gemm_f32(a_slice, b.as_f32_slice()?, out_slice, m, k, n);
        }
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
        cos_table: &Tensor,
        sin_table: &Tensor,
        positions: &[usize],
    ) -> Result<()> {
        let head_dim = *q.dims().last().ok_or_else(|| {
            RllmError::BackendError("RoPE query tensor cannot be scalar".to_string())
        })?;
        let num_tokens = positions.len();
        let per_token = num_tokens * head_dim;
        if num_tokens == 0 || !q.numel().is_multiple_of(per_token) || !k.numel().is_multiple_of(per_token) {
            return Err(RllmError::BackendError(format!(
                "RoPE: q {:?} / k {:?} not divisible into {} tokens of head_dim {}",
                q.dims(), k.dims(), num_tokens, head_dim
            )));
        }
        let num_heads_q = q.numel() / (num_tokens * head_dim);
        let num_heads_k = k.numel() / (num_tokens * head_dim);

        let half_dim = head_dim / 2;
        let cos_slice = cos_table.as_f32_slice()?;
        let sin_slice = sin_table.as_f32_slice()?;
        let max_positions = cos_slice.len() / half_dim.max(1);
        if let Some(&bad) = positions.iter().find(|&&p| p >= max_positions) {
            return Err(RllmError::BackendError(format!(
                "RoPE position {} exceeds table size {}",
                bad, max_positions
            )));
        }

        rope_f32(q.as_mut_f32_slice()?, num_heads_q, head_dim, positions, cos_slice, sin_slice);
        rope_f32(k.as_mut_f32_slice()?, num_heads_k, head_dim, positions, cos_slice, sin_slice);
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
        let q_dims = q.dims(); // [num_rows, num_heads_q, head_dim]
        if q_dims.len() != 3 {
            return Err(RllmError::BackendError(format!(
                "paged_attention expects q of rank 3, got {:?}",
                q_dims
            )));
        }
        let (num_rows, num_heads_q, head_dim) = (q_dims[0], q_dims[1], q_dims[2]);
        let k_dims = k_cache.dims(); // [num_blocks, num_heads_kv, block_size, head_dim]
        let num_heads_kv = k_dims[1];
        if block_tables.len() != num_rows || context_lens.len() != num_rows {
            return Err(RllmError::BackendError(format!(
                "paged_attention: {} query rows but {} block tables / {} context lengths",
                num_rows,
                block_tables.len(),
                context_lens.len()
            )));
        }
        if num_heads_kv == 0 || num_heads_q % num_heads_kv != 0 {
            return Err(RllmError::BackendError(format!(
                "paged_attention: {} query heads not divisible by {} KV heads",
                num_heads_q, num_heads_kv
            )));
        }
        for (bt, &len) in block_tables.iter().zip(context_lens) {
            if len == 0 || bt.block_ids.len() * bt.block_size < len {
                return Err(RllmError::BackendError(format!(
                    "paged_attention: context length {} not covered by {} blocks of {}",
                    len,
                    bt.block_ids.len(),
                    bt.block_size
                )));
            }
        }
        let heads_per_group = num_heads_q / num_heads_kv;

        let mut out = Tensor::zeros(q.shape().clone(), DType::F32, Device::Cpu)?;
        let q_slice = q.as_f32_slice()?;
        let k_slice = k_cache.as_f32_slice()?;
        let v_slice = v_cache.as_f32_slice()?;
        let out_slice = out.as_mut_f32_slice()?;

        // Parallelise over every (row, head) pair so both decode (many sequences)
        // and prefill (many tokens of one sequence) use all cores.
        out_slice
            .par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(idx, out_head)| {
                let row = idx / num_heads_q;
                let head = idx % num_heads_q;
                let q_head = &q_slice[idx * head_dim..(idx + 1) * head_dim];
                paged_attention_head(
                    q_head,
                    k_slice,
                    v_slice,
                    &block_tables[row],
                    context_lens[row],
                    head / heads_per_group,
                    num_heads_kv,
                    head_dim,
                    scale,
                    out_head,
                );
            });

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

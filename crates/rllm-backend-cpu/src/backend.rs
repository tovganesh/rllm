use crate::ops::{
    activations::swiglu_f32,
    blas::{sgemm_nt, sgemm_nt_cols},
    gemm::{gemm_f32, gemm_nt_f32},
    norm::rms_norm_f32,
    paged_attn::{paged_attention_kv_group, KvData},
    qdot::{gemv_quantized, linear_quantized, QFormat},
    quant::{dequantize_into, gemv_q4_0_f32, gemv_q8_0_f32, q5_0_to_q8_0},
    repack::{dequantize_q4_0x4_group, dequantize_q4_0x4_row, gemm_q4_0x4, gemv_q4_0x4, has_i8mm, repack_q4_0},
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

/// Weight rows dequantized per GEMM call in the batched path (swept 128..n on an M4:
/// 2048 was best; smaller chunks pay per-call GEMM overhead).
const DEQUANT_CHUNK_ROWS: usize = 2048;

/// Batches with at least this many rows use dequantized-weight F32 GEMM instead of
/// integer dot products (llama.cpp uses the same cut-over for its BLAS path).
pub const BATCHED_F32_MIN_ROWS: usize = 32;

/// Whether batches of at least [`BATCHED_F32_MIN_ROWS`] rows dequantize the weight and
/// use F32 GEMM. Only with a vendor BLAS (Accelerate): the portable GEMM is far slower
/// than the integer multi-row kernels (on an x86 runner, prompts ran at 1-2% of
/// llama.cpp's speed), and llama.cpp's own CPU backend keeps quantized weights in
/// integer arithmetic at every batch size without one.
const DEQUANT_BATCHES: bool = crate::ops::blas::HAS_VENDOR_BLAS;

/// Whether a batch of `m` rows against a repacked `Q4_0` or a `Q8_0` `[n, k]` weight
/// should use the i8mm integer GEMM rather than dequantization + F32 GEMM. Measured on
/// an M4: i8mm is 3-6x faster than per-row SDOT below 32 rows; from 32 to ~192 rows it
/// beats dequantize + Accelerate only for large matrices (Llama-3.2-1B Q4_0: +12% at
/// 128 prompt tokens, +65% at 64; Q8_0: 2x at 32, +29% at 64, even at 128), whose F32
/// copies spill out of cache, and loses on small ones (SmolLM2-135M: -25% at 128).
fn use_i8mm(m: usize, n: usize, k: usize) -> bool {
    has_i8mm() && (m < BATCHED_F32_MIN_ROWS || (m < 192 && n * k >= 1 << 22))
}

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
            if m >= BATCHED_F32_MIN_ROWS {
                sgemm_nt(a_slice, b.storage_f32_slice()?, out_slice, m, k, n);
            } else {
                gemm_nt_f32(a_slice, b.storage_f32_slice()?, out_slice, m, k, n);
            }
        } else {
            let b = b.contiguous()?;
            gemm_f32(a_slice, b.as_f32_slice()?, out_slice, m, k, n);
        }
        Ok(out)
    }

    fn linear(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let w_dims = w.dims();
        let x_dims = x.dims();
        if w_dims.len() != 2 || x_dims.is_empty() {
            return Err(RllmError::BackendError(format!(
                "linear expects W of rank 2 and x of rank >= 1, got W {:?}, x {:?}",
                w_dims, x_dims
            )));
        }
        let (n, k) = (w_dims[0], w_dims[1]);
        if *x_dims.last().unwrap() != k {
            return Err(RllmError::ShapeMismatch {
                expected: vec![k],
                got: vec![*x_dims.last().unwrap()],
            });
        }

        if w.dtype() == DType::F32 {
            // Flatten leading dims of x into rows, then A·Wᵀ via the NT kernel.
            let m = x.numel() / k.max(1);
            let x2 = x.contiguous()?.reshape(Shape::new(vec![m, k]))?;
            let y = self.matmul(&x2, &w.transpose(0, 1)?)?;
            let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
            out_dims.push(n);
            return y.reshape(Shape::new(out_dims));
        }

        let dtype = w.dtype();
        let fmt = QFormat::from_dtype(dtype);
        if fmt.is_none() && !matches!(dtype, DType::Q4_0x4 | DType::F16) {
            return Err(RllmError::UnsupportedOp {
                op: "linear",
                dtype: dtype.to_string(),
                device: "cpu".to_string(),
            });
        }
        let block = dtype.block_info().map_or(1, |(values, _)| values);
        if k % block != 0 {
            return Err(RllmError::QuantizationError(format!(
                "linear: in_features {k} is not a multiple of the {block}-element {dtype} block"
            )));
        }
        let w_bytes = dtype.storage_bytes(n * k);
        let all = w.storage().as_cpu()?.as_bytes();
        let w_bytes = all
            .get(w.offset()..w.offset() + w_bytes)
            .ok_or_else(|| RllmError::BackendError("quantized weight storage too small".to_string()))?;

        let x = x.contiguous()?;
        let x_slice = x.as_f32_slice()?;
        let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(n);
        let mut out = Tensor::zeros(Shape::new(out_dims), DType::F32, Device::Cpu)?;
        let out_slice = out.as_mut_f32_slice()?;
        let m = x_slice.len() / k;
        let repacked = dtype == DType::Q4_0x4;
        let small = m < BATCHED_F32_MIN_ROWS || !DEQUANT_BATCHES;
        let integer_path = small || ((repacked || fmt == Some(QFormat::Q8_0)) && use_i8mm(m, n, k));
        if dtype == DType::F16 && small {
            // F16 weights (F16 GGUF files) stay F16: half the bytes of F32 per token.
            if m == 1 {
                crate::ops::f16w::gemv_f16(w_bytes, x_slice, out_slice, n, k);
            } else {
                crate::ops::f16w::linear_f16(w_bytes, x_slice, out_slice, m, n, k);
            }
        } else if m == 1 {
            // Decode: activations quantized to Q8, integer dot products.
            match fmt {
                Some(fmt) => gemv_quantized(fmt, w_bytes, x_slice, out_slice, n, k),
                None => gemv_q4_0x4(w_bytes, x_slice, out_slice, n, k),
            }
        } else if integer_path {
            match fmt {
                Some(fmt) => linear_quantized(fmt, w_bytes, x_slice, out_slice, m, n, k),
                None => gemm_q4_0x4(w_bytes, x_slice, out_slice, m, n, k),
            }
        } else {
            // Large batches (prefill): dequantize the weight (or widen F16) and use an
            // F32 GEMM, as llama.cpp's CPU backend does for batches of 32+ rows.
            // The weight is processed in row chunks small enough to stay in cache between
            // dequantization and the GEMM that consumes them.
            let row_bytes = dtype.storage_bytes(k);
            let chunk = DEQUANT_CHUNK_ROWS.min(n);
            let mut wf = vec![0f32; chunk * k];
            let mut j0 = 0;
            while j0 < n {
                let nc = chunk.min(n - j0);
                let src = &w_bytes[j0 * row_bytes..(j0 + nc) * row_bytes];
                if repacked {
                    wf[..nc * k]
                        .par_chunks_mut(4 * k)
                        .zip(src.par_chunks(4 * row_bytes))
                        .for_each(|(dst, src)| {
                            let (r0, rest) = dst.split_at_mut(k);
                            let (r1, rest) = rest.split_at_mut(k);
                            let (r2, r3) = rest.split_at_mut(k);
                            dequantize_q4_0x4_group(src, [r0, r1, r2, r3]);
                        });
                } else {
                    wf[..nc * k]
                        .par_chunks_mut(k)
                        .zip(src.par_chunks(row_bytes))
                        .try_for_each(|(dst, src)| dequantize_into(dtype, src, dst))?;
                }
                sgemm_nt_cols(x_slice, &wf[..nc * k], out_slice, m, k, nc, j0, n);
                j0 += nc;
            }
        }
        Ok(out)
    }

    /// Repacks `Q4_0` matrices with a multiple of 4 rows into interleaved `Q4_0x4` (aarch64), and
    /// expands `Q5_0` losslessly to `Q8_0` (the SDOT Q8_0 kernel is ~1.75x faster than
    /// unpacking 5-bit values on an M4, at 34 instead of 22 bytes per block).
    fn pack_weight(&self, w: &Tensor) -> Result<Option<Tensor>> {
        let dims = w.dims();
        if dims.len() != 2 || !dims[1].is_multiple_of(32) {
            return Ok(None);
        }
        let len = w.dtype().storage_bytes(w.numel());
        let bytes = || -> Result<&[u8]> { Ok(&w.storage().as_cpu()?.as_bytes()[w.offset()..w.offset() + len]) };
        let (dtype, packed) = match w.dtype() {
            // The interleaved layout has SDOT/SMMLA kernels only; x86 keeps plain Q4_0 rows
            // (AVX2 dot product).
            DType::Q4_0 if cfg!(target_arch = "aarch64") && dims[0].is_multiple_of(4) => {
                (DType::Q4_0x4, repack_q4_0(bytes()?, dims[0], dims[1]))
            }
            DType::Q5_0 => (DType::Q8_0, q5_0_to_q8_0(bytes()?)),
            _ => return Ok(None),
        };
        Ok(Some(Tensor::new(
            w.shape().clone(),
            dtype,
            Device::Cpu,
            rllm_core::storage::Storage::Cpu(rllm_core::storage::CpuStorage::from_bytes(packed)),
        )))
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let hidden_dim = *x
            .dims()
            .last()
            .ok_or_else(|| RllmError::BackendError("RMSNorm input tensor cannot be scalar".to_string()))?;
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
        let head_dim = *q
            .dims()
            .last()
            .ok_or_else(|| RllmError::BackendError("RoPE query tensor cannot be scalar".to_string()))?;
        let num_tokens = positions.len();
        let per_token = num_tokens * head_dim;
        if num_tokens == 0 || !q.numel().is_multiple_of(per_token) || !k.numel().is_multiple_of(per_token) {
            return Err(RllmError::BackendError(format!(
                "RoPE: q {:?} / k {:?} not divisible into {} tokens of head_dim {}",
                q.dims(),
                k.dims(),
                num_tokens,
                head_dim
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

        rope_f32(
            q.as_mut_f32_slice()?,
            num_heads_q,
            head_dim,
            positions,
            cos_slice,
            sin_slice,
        );
        rope_f32(
            k.as_mut_f32_slice()?,
            num_heads_k,
            head_dim,
            positions,
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
        if t_dims.len() != 2 {
            return Err(RllmError::BackendError(format!(
                "embedding table must be rank 2, got {:?}",
                t_dims
            )));
        }
        let (vocab_size, hidden_dim) = (t_dims[0], t_dims[1]);
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= vocab_size) {
            return Err(RllmError::BackendError(format!(
                "Token ID {} exceeds vocabulary size {}",
                bad, vocab_size
            )));
        }

        let out_shape = Shape::new(vec![ids.len(), hidden_dim]);
        let mut out = Tensor::zeros(out_shape, DType::F32, Device::Cpu)?;
        let out_slice = out.as_mut_f32_slice()?;

        match table.dtype() {
            DType::F32 => {
                let table_slice = table.as_f32_slice()?;
                for (dst, &id) in out_slice.chunks_exact_mut(hidden_dim).zip(ids) {
                    let id = id as usize;
                    dst.copy_from_slice(&table_slice[id * hidden_dim..(id + 1) * hidden_dim]);
                }
            }
            DType::Q4_0x4 => {
                let bytes = &table.storage().as_cpu()?.as_bytes()[table.offset()..];
                for (dst, &id) in out_slice.chunks_exact_mut(hidden_dim).zip(ids) {
                    dequantize_q4_0x4_row(bytes, hidden_dim, id as usize, dst);
                }
            }
            dtype if dtype.is_quantized() || matches!(dtype, DType::F16 | DType::BF16) => {
                let row_bytes = dtype.storage_bytes(hidden_dim);
                let bytes = &table.storage().as_cpu()?.as_bytes()[table.offset()..];
                for (dst, &id) in out_slice.chunks_exact_mut(hidden_dim).zip(ids) {
                    dequantize_into(
                        dtype,
                        &bytes[id as usize * row_bytes..(id as usize + 1) * row_bytes],
                        dst,
                    )?;
                }
            }
            other => {
                return Err(RllmError::UnsupportedOp {
                    op: "embedding",
                    dtype: other.to_string(),
                    device: "cpu".to_string(),
                })
            }
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
        let k_slice = kv_data(k_cache)?;
        let v_slice = kv_data(v_cache)?;
        let out_slice = out.as_mut_f32_slice()?;

        // One work item per (row, KV head): it computes all `heads_per_group` query
        // heads that share the KV head, reading each cached key/value once. Items are
        // batched so short-context decode does not pay fork/join overhead.
        let group_len = heads_per_group * head_dim;
        let max_ctx = context_lens.iter().copied().max().unwrap_or(1);
        let min_items = (crate::ops::ELEMENTWISE_GRAIN / (max_ctx * group_len).max(1)).max(1);
        let run_items = |t: usize, items: &mut [f32]| {
            for (j, out_group) in items.chunks_mut(group_len).enumerate() {
                let idx = t * min_items + j;
                let row = idx / num_heads_kv;
                let kv_head = idx % num_heads_kv;
                let q_group = &q_slice[idx * group_len..(idx + 1) * group_len];
                paged_attention_kv_group(
                    q_group,
                    k_slice,
                    v_slice,
                    &block_tables[row],
                    context_lens[row],
                    kv_head,
                    num_heads_kv,
                    head_dim,
                    scale,
                    out_group,
                );
            }
        };
        // Decode-sized batches use the spinning pool; prefill stays on Rayon (the two
        // pools share the performance cores, and spinning workers would slow it down).
        if num_rows < BATCHED_F32_MIN_ROWS {
            crate::pool::for_each_chunk_mut(out_slice, min_items * group_len, run_items);
        } else {
            out_slice
                .par_chunks_mut(min_items * group_len)
                .enumerate()
                .for_each(|(t, items)| run_items(t, items));
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

/// A paged KV cache tensor's data in its storage type.
fn kv_data(t: &Tensor) -> Result<KvData<'_>> {
    Ok(match t.dtype() {
        DType::F32 => KvData::F32(t.as_f32_slice()?),
        DType::F16 => KvData::F16(t.storage().as_cpu()?.as_slice::<half::f16>()),
        other => {
            return Err(RllmError::UnsupportedOp {
                op: "paged_attention",
                dtype: other.to_string(),
                device: "cpu".to_string(),
            })
        }
    })
}

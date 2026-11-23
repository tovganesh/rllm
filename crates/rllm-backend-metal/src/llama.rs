//! LLaMA executor on Apple GPUs via Metal.
//!
//! The whole forward pass (all layers) is encoded into one command buffer, so a decode
//! step costs one GPU submission. Weights are copied once into shared-memory buffers
//! (unified memory: no PCIe transfer); the paged KV cache lives in GPU buffers.

use crate::shaders::SHADERS_MSL;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
use rllm_core::{
    backend::Backend,
    dtype::DType,
    error::{Result, RllmError},
    tensor::Tensor,
};
use rllm_models::{ForwardBatch, LanguageModel, LlamaConfig, LlamaModel};
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::info;

/// Weight storage formats with Metal kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WKind {
    F32,
    F16,
    Q8,
    Q4,
    Q4K,
    Q6K,
}

impl WKind {
    fn suffix(self) -> &'static str {
        match self {
            WKind::F32 => "f32",
            WKind::F16 => "f16",
            WKind::Q8 => "q8_0",
            WKind::Q4 => "q4_0",
            WKind::Q4K => "q4_k",
            WKind::Q6K => "q6_k",
        }
    }
}

/// A `[rows, cols]` weight matrix in a GPU buffer.
struct Weight {
    buf: Buffer,
    kind: WKind,
    rows: usize,
    cols: usize,
}

/// One decoder layer. Q/K/V and gate/up projections are fused into single matrices
/// (`[q | k | v]` and `[gate | up]` row blocks) so each is one dispatch per storage
/// format: parts in different formats (e.g. Q4_K q/k with a Q6_K v) stay separate
/// segments that write adjacent column ranges of the same output.
struct Layer {
    attn_norm: Buffer,
    qkv: Vec<Weight>,
    o: Weight,
    ffn_norm: Buffer,
    gate_up: Vec<Weight>,
    /// `gate_up` is one segment with gate and up rows interleaved in 32-row chunks
    /// (see [`interleave_gate_up`]); otherwise `[gate | up]` row blocks.
    gu_interleaved: bool,
    down: Weight,
}

/// Rows per chunk of an interleaved gate/up weight (one half of a 64-row GEMM tile).
const GLU_CHUNK: usize = 32;

/// Raw rows of `gate` and `up` (same format and shape) interleaved in [`GLU_CHUNK`]-row
/// chunks, so each 64-row GEMM tile holds matching gate and up outputs and can apply
/// SwiGLU in its epilogue.
fn interleave_gate_up(gate: &[u8], up: &[u8], rows: usize) -> Vec<u8> {
    let row_bytes = gate.len() / rows;
    let chunk = GLU_CHUNK * row_bytes;
    let mut out = Vec::with_capacity(2 * gate.len());
    for (g, u) in gate.chunks(chunk).zip(up.chunks(chunk)) {
        out.extend_from_slice(g);
        out.extend_from_slice(u);
    }
    out
}

/// Scratch buffers reused across forward passes, grown on demand.
#[derive(Default)]
struct Scratch {
    rows: usize,
    logit_rows: usize,
    table_len: usize,
    bufs: HashMap<&'static str, Buffer>,
}

struct State {
    kv: Vec<(Buffer, Buffer)>,
    block_size: usize,
    num_blocks: usize,
    scratch: Scratch,
}

/// A LLaMA-architecture model executing on a Metal GPU.
pub struct MetalLlama {
    config: LlamaConfig,
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<String, ComputePipelineState>,
    embed: Weight,
    lm_head: Weight,
    final_norm: Buffer,
    layers: Vec<Layer>,
    cos: Buffer,
    sin: Buffer,
    /// KV cache element type, `"f16"` or `"f32"` (kernel-name suffix).
    kv_type: &'static str,
    state: Mutex<State>,
}

/// Query rows per tile of the prefill attention kernel (4 simdgroups x 8 rows).
const ATTN_TILE_ROWS: usize = 32;

/// Keys per step of the prefill attention kernel (must match `ATTN_KB` in the shaders).
const ATTN_KB: usize = 32;

/// Keys per threadgroup of the decode attention kernel; longer contexts are split over
/// several threadgroups and combined. Swept on an M4 (Llama-3.2-1B, 2k/8k context):
/// 128 and 512 were up to 7% slower, no splitting 14–28% slower.
const DECODE_CHUNK: usize = 256;

/// With fewer decode-attention threadgroups than this at [`DECODE_CHUNK`], chunks of
/// [`DECODE_CHUNK_SMALL`] keys are used instead, so a model with few KV heads still
/// spreads a short context over the GPU (SmolLM2-135M, 3 KV heads, 1k context: +10%;
/// at 4k, 256-key chunks are 4% faster).
const DECODE_MIN_GROUPS: usize = 32;
const DECODE_CHUNK_SMALL: usize = 128;

/// Rows at or below which a linear layer uses matrix-vector kernels; larger batches use
/// the small-batch GEMM (`mms`, F16/quantized weights) or the tiled GEMM. Measured
/// end to end on Llama-3.2-1B (M4) at 3-4 rows: `mms` is ~25% faster for K-quants,
/// `mv4` ~10% faster for Q8_0, and they tie for Q4_0. F32 weights have no `mms`.
fn mv_max_rows(kind: WKind) -> usize {
    match kind {
        WKind::F32 => 8,
        WKind::Q4K | WKind::Q6K => 2,
        WKind::F16 | WKind::Q8 | WKind::Q4 => 4,
    }
}

/// Rows at or below which the small-batch GEMM (`mms`) is used instead of the 64x64
/// tiled GEMM. On Llama-3.2-1B shapes (M4) it is 2.3-3.4x faster at 8-16 rows and
/// 1.4x at 32; the tiled GEMM wins from 64 rows.
const MMS_MAX_ROWS: usize = 32;

/// The 64x32-tile GEMM (`mml_*`) is used for all formats but Q6_K, and for Q6_K when the
/// 64x64 one would launch fewer threadgroups than this (M4, Llama-3.2-1B shapes: 64x32
/// is 1-10% faster for the former at 128-512 tokens; for Q6_K the 64x64 tile is 3-6%
/// faster on large grids).
const MML_MAX_TILES: usize = 256;

/// Query heads per KV head with a dedicated GQA decode attention kernel
/// (`attention_gqa{G}_*`); other models use the per-query-head kernel.
const GQA_GROUPS: [usize; 4] = [2, 3, 4, 8];

impl MetalLlama {
    /// Whether a Metal device is available.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Uploads a CPU-loaded model to the default Metal device. F32, Q8_0, Q4_0, Q4_K and
    /// Q6_K weights are uploaded as they are (F32 as F16 when exact); the CPU-only
    /// formats are converted: repacked `Q4_0x4` back to Q4_0, Q5_0 losslessly to Q8_0.
    pub fn from_cpu_model<B: Backend>(model: &LlamaModel<B>) -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| RllmError::BackendError("No Metal device available".to_string()))?;
        let c = model.config.clone();
        let hd = c.head_dim();
        if !hd.is_multiple_of(32) || hd > 128 || !c.hidden_size.is_multiple_of(32) || !c.intermediate_size.is_multiple_of(32) {
            return Err(RllmError::BackendError(format!(
                "Metal executor needs head_dim % 32 == 0 (<= 128) and hidden/intermediate sizes divisible by 32; got head_dim {hd}, hidden {}, intermediate {}",
                c.hidden_size, c.intermediate_size
            )));
        }

        // The KV cache follows the CPU model's setting (F16 by default for loaded models).
        let kv_type = match model.kv_dtype {
            DType::F16 => "f16",
            DType::F32 => "f32",
            other => return Err(RllmError::BackendError(format!("Metal KV cache cannot be {other} (use F16 or F32)"))),
        };
        let library = device
            .new_library_with_source(SHADERS_MSL, &CompileOptions::new())
            .map_err(|e| RllmError::BackendError(format!("Metal shader compilation failed: {e}")))?;
        let mut pipelines = HashMap::new();
        let names = [
            "mv_f32", "mv_q8_0", "mv_q4_0", "mv_f16", "mv_q4_k", "mv_q6_k", "mm_f32", "mm_q8_0", "mm_q4_0", "mm_f16",
            "mm_q4_k", "mm_q6_k", "mml_q8_0", "mml_q4_0", "mml_f16", "mml_q4_k", "mml_q6_k", "mml1_q8_0", "mml1_q4_0", "mml1_f16", "mml1_q4_k", "mml1_q6_k", "mmlg_q8_0", "mmlg_q4_0", "mmlg_f16", "mmlg_q4_k", "mmlg_q6_k", "mmlg1_q8_0", "mmlg1_q4_0", "mmlg1_f16", "mmlg1_q4_k", "mmlg1_q6_k", "to_half", "embed_f32", "embed_q8_0", "embed_q4_0", "embed_f16", "embed_q4_k", "embed_q6_k",
            "rms_norm", "rms_norm_h", "rope", "swiglu", "swiglu_h", "swiglu_i", "swiglu_ih", "attn_reduce",
        ];
        let mv_names: Vec<String> = ["f32", "f16", "q8_0", "q4_0", "q4_k", "q6_k"]
            .iter()
            .flat_map(|f| [2, 4, 8].map(|r| format!("mv{r}_{f}")))
            .chain(["f16", "q8_0", "q4_0", "q4_k", "q6_k"].iter().flat_map(|f| {
                ["mms1", "mms2", "mms2r2", "mms3r2", "mms4r2"].map(|t| format!("{t}_{f}"))
            }))
            .collect();
        let kv_names: Vec<String> = ["f32", "f16"]
            .iter()
            .flat_map(|t| {
                std::iter::once(format!("kv_write_{t}")).chain([32, 64, 96, 128].iter().flat_map(move |hd| {
                    [format!("attention_hd{hd}_{t}"), format!("attention_tile32_hd{hd}_{t}")]
                }))
            })
            .collect();
        // GQA decode attention for this model's group size, when instantiated.
        let group = c.num_attention_heads / c.num_key_value_heads;
        let gqa_names: Vec<String> = if GQA_GROUPS.contains(&group) {
            ["f32", "f16"].iter().map(|t| format!("attention_gqa{group}_hd{hd}_{t}")).collect()
        } else {
            Vec::new()
        };
        for name in names.iter().copied().chain(kv_names.iter().chain(&mv_names).chain(&gqa_names).map(String::as_str)) {
            let f = library
                .get_function(name, None)
                .map_err(|e| RllmError::BackendError(format!("Metal function {name}: {e}")))?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| RllmError::BackendError(format!("Metal pipeline {name}: {e}")))?;
            pipelines.insert(name.to_string(), p);
        }

        let upload = |t: &Tensor| -> Result<Weight> {
            let dims = t.dims();
            if dims.len() != 2 {
                return Err(RllmError::BackendError(format!("expected a matrix, got {dims:?}")));
            }
            // Formats without Metal kernels: the CPU-repacked Q4_0 is unpacked; Q5_0 and
            // the K-quants become Q8_0 (what the GGUF loader produced before the CPU
            // backend gained native kernels for them).
            let converted = to_metal_format(t)?;
            let t = converted.as_ref().unwrap_or(t);
            let kind = match t.dtype() {
                DType::F32 => WKind::F32,
                DType::F16 => WKind::F16,
                DType::Q8_0 => WKind::Q8,
                DType::Q4_0 => WKind::Q4,
                DType::Q4_K => WKind::Q4K,
                DType::Q6_K => WKind::Q6K,
                other => {
                    return Err(RllmError::BackendError(format!("Metal executor does not support {other} weights")))
                }
            };
            // F32 weights that round-trip exactly through F16 (e.g. loaded from an F16
            // GGUF) are stored as F16: half the memory traffic, bit-identical values.
            if kind == WKind::F32 {
                let vals = t.as_f32_slice()?;
                let halves: Vec<half::f16> = vals.iter().map(|&v| half::f16::from_f32(v)).collect();
                if halves.iter().zip(vals).all(|(h, &v)| h.to_f32() == v) {
                    return Ok(Weight { buf: new_buffer_with(&device, as_bytes(&halves)), kind: WKind::F16, rows: dims[0], cols: dims[1] });
                }
            }
            let bytes = &t.storage().as_cpu()?.as_bytes()[t.offset()..];
            let len = t.dtype().storage_bytes(t.numel());
            Ok(Weight {
                buf: new_buffer_with(&device, &bytes[..len]),
                kind,
                rows: dims[0],
                cols: dims[1],
            })
        };
        let vec_buf = |t: &Tensor| -> Result<Buffer> { Ok(new_buffer_with(&device, as_bytes(t.as_f32_slice()?))) };
        // Row-concatenation of matrices with equal column counts: consecutive parts with
        // the same storage format are concatenated as raw blocks into one segment.
        let upload_concat = |parts: &[&Tensor]| -> Result<Vec<Weight>> {
            let cols = parts[0].dims()[1];
            if parts.iter().any(|t| t.dims().len() != 2 || t.dims()[1] != cols) {
                return Err(RllmError::BackendError("cannot fuse matrices with different widths".into()));
            }
            // Convert CPU-only formats first so that parts that end up in the same
            // Metal format are still concatenated.
            let converted: Vec<Option<Tensor>> = parts.iter().map(|t| to_metal_format(t)).collect::<Result<_>>()?;
            let parts: Vec<&Tensor> = parts.iter().zip(&converted).map(|(t, c)| c.as_ref().unwrap_or(t)).collect();
            let mut segments = Vec::new();
            let mut i = 0;
            while i < parts.len() {
                let dtype = parts[i].dtype();
                let mut j = i;
                let mut bytes = Vec::new();
                while j < parts.len() && parts[j].dtype() == dtype {
                    let t = parts[j];
                    let len = dtype.storage_bytes(t.numel());
                    bytes.extend_from_slice(&t.storage().as_cpu()?.as_bytes()[t.offset()..t.offset() + len]);
                    j += 1;
                }
                let rows: usize = parts[i..j].iter().map(|t| t.dims()[0]).sum();
                let shape = rllm_core::shape::Shape::new(vec![rows, cols]);
                let storage = rllm_core::storage::Storage::Cpu(rllm_core::storage::CpuStorage::from_bytes(bytes));
                segments.push(upload(&Tensor::new(shape, dtype, rllm_core::device::Device::Cpu, storage))?);
                i = j;
            }
            Ok(segments)
        };
        // Gate/up: interleaved into one segment when both parts share a format.
        let upload_gate_up = |gate: &Tensor, up: &Tensor| -> Result<(Vec<Weight>, bool)> {
            let (gc, uc) = (to_metal_format(gate)?, to_metal_format(up)?);
            let (g, u) = (gc.as_ref().unwrap_or(gate), uc.as_ref().unwrap_or(up));
            let rows = g.dims()[0];
            if g.dtype() != u.dtype() || g.dims() != u.dims() || g.dims().len() != 2 || !rows.is_multiple_of(GLU_CHUNK) {
                return Ok((upload_concat(&[gate, up])?, false));
            }
            let len = g.dtype().storage_bytes(g.numel());
            let gb = &g.storage().as_cpu()?.as_bytes()[g.offset()..g.offset() + len];
            let ub = &u.storage().as_cpu()?.as_bytes()[u.offset()..u.offset() + len];
            let shape = rllm_core::shape::Shape::new(vec![2 * rows, g.dims()[1]]);
            let storage = rllm_core::storage::Storage::Cpu(rllm_core::storage::CpuStorage::from_bytes(interleave_gate_up(gb, ub, rows)));
            Ok((vec![upload(&Tensor::new(shape, g.dtype(), rllm_core::device::Device::Cpu, storage))?], true))
        };

        let embed = upload(&model.embed_tokens)?;
        let tied = model.lm_head.storage().as_cpu()?.as_bytes().as_ptr()
            == model.embed_tokens.storage().as_cpu()?.as_bytes().as_ptr();
        let lm_head = if tied {
            Weight { buf: embed.buf.clone(), kind: embed.kind, rows: embed.rows, cols: embed.cols }
        } else {
            upload(&model.lm_head)?
        };
        let mut layers = Vec::with_capacity(model.layers.len());
        for l in &model.layers {
            let (gate_up, gu_interleaved) = upload_gate_up(&l.mlp.gate_proj, &l.mlp.up_proj)?;
            layers.push(Layer {
                attn_norm: vec_buf(&l.input_layernorm)?,
                qkv: upload_concat(&[&l.self_attn.q_proj, &l.self_attn.k_proj, &l.self_attn.v_proj])?,
                o: upload(&l.self_attn.o_proj)?,
                ffn_norm: vec_buf(&l.post_attention_layernorm)?,
                gate_up,
                gu_interleaved,
                down: upload(&l.mlp.down_proj)?,
            });
        }
        let final_norm = vec_buf(&model.norm)?;
        let cos = vec_buf(&model.cos_table)?;
        let sin = vec_buf(&model.sin_table)?;
        info!("Metal executor ready on {} ({} layers)", device.name(), layers.len());

        Ok(Self {
            config: c,
            queue: device.new_command_queue(),
            device,
            pipelines,
            embed,
            lm_head,
            final_norm,
            layers,
            cos,
            sin,
            kv_type,
            state: Mutex::new(State { kv: Vec::new(), block_size: 0, num_blocks: 0, scratch: Scratch::default() }),
        })
    }

    /// Name of the GPU.
    pub fn gpu_name(&self) -> String {
        self.device.name().to_string()
    }

    fn pipeline(&self, name: &str) -> &ComputePipelineState {
        &self.pipelines[name]
    }

    fn validate(&self, batch: &ForwardBatch<'_>, block_size: usize, num_blocks: usize) -> Result<()> {
        let n = batch.tokens.len();
        let c = &self.config;
        if n == 0 || batch.positions.len() != n || batch.block_tables.len() != n {
            return Err(RllmError::BackendError("forward: malformed batch".to_string()));
        }
        for (i, (&pos, bt)) in batch.positions.iter().zip(batch.block_tables).enumerate() {
            if batch.tokens[i] as usize >= c.vocab_size {
                return Err(RllmError::BackendError(format!("Token ID {} exceeds vocabulary size {}", batch.tokens[i], c.vocab_size)));
            }
            if pos >= c.max_position_embeddings {
                return Err(RllmError::BackendError(format!("position {pos} exceeds max_position_embeddings")));
            }
            if bt.block_size != block_size || pos / block_size >= bt.block_ids.len() {
                return Err(RllmError::BackendError(format!("position {pos} not covered by the block table")));
            }
            if bt.block_ids.iter().any(|&b| b >= num_blocks) {
                return Err(RllmError::BackendError("block id outside the KV cache".to_string()));
            }
        }
        if batch.logits_rows.iter().any(|&r| r >= n) {
            return Err(RllmError::BackendError("logits row out of range".to_string()));
        }
        Ok(())
    }
}

/// Dense F32 copy of a matrix in any CPU weight format.
fn dequantize_to_f32(t: &Tensor) -> Result<Vec<f32>> {
    use rllm_backend_cpu::ops::{quant::dequantize_into, repack::dequantize_q4_0x4_row};
    let mut out = vec![0f32; t.numel()];
    let bytes = &t.storage().as_cpu()?.as_bytes()[t.offset()..t.offset() + t.dtype().storage_bytes(t.numel())];
    match t.dtype() {
        DType::F32 => return t.to_vec::<f32>(),
        DType::Q4_0x4 => {
            let k = t.dims()[1];
            for (r, row) in out.chunks_exact_mut(k).enumerate() {
                dequantize_q4_0x4_row(bytes, k, r, row);
            }
        }
        other => dequantize_into(other, bytes, &mut out)?,
    }
    Ok(out)
}

/// Converts a CPU weight whose format has no Metal kernel: `Q4_0x4` back to `Q4_0`,
/// `Q5_0` losslessly to `Q8_0`, and K-quants whose rows are not whole super-blocks to
/// `Q8_0`. Returns `None` for formats Metal handles.
fn to_metal_format(t: &Tensor) -> Result<Option<Tensor>> {
    use rllm_backend_cpu::ops::{
        quant::{q5_0_to_q8_0, quantize_f32_to_q8_0},
        repack::unpack_q4_0x4,
    };
    use rllm_core::{shape::Shape, storage::{CpuStorage, Storage}};
    let (dtype, bytes) = match t.dtype() {
        DType::Q4_0x4 => {
            let (n, k) = (t.dims()[0], t.dims()[1]);
            let src = &t.storage().as_cpu()?.as_bytes()[t.offset()..t.offset() + t.dtype().storage_bytes(t.numel())];
            (DType::Q4_0, unpack_q4_0x4(src, n, k))
        }
        DType::Q5_0 => {
            let src = &t.storage().as_cpu()?.as_bytes()[t.offset()..t.offset() + t.dtype().storage_bytes(t.numel())];
            (DType::Q8_0, q5_0_to_q8_0(src))
        }
        DType::Q4_K | DType::Q6_K if !t.dims()[1].is_multiple_of(256) => {
            let dense = dequantize_to_f32(t)?;
            let mut bytes = vec![0u8; dense.len() / 32 * 34];
            quantize_f32_to_q8_0(&dense, &mut bytes);
            (DType::Q8_0, bytes)
        }
        _ => return Ok(None),
    };
    Ok(Some(Tensor::new(Shape::new(t.dims().to_vec()), dtype, rllm_core::device::Device::Cpu, Storage::Cpu(CpuStorage::from_bytes(bytes)))))
}

fn as_bytes<T: Copy>(s: &[T]) -> &[u8] {
    // SAFETY: plain-old-data reinterpretation of an initialised slice.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn new_buffer_with(device: &Device, bytes: &[u8]) -> Buffer {
    device.new_buffer_with_data(
        bytes.as_ptr() as *const std::ffi::c_void,
        bytes.len().max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn write_buffer<T: Copy>(buf: &Buffer, data: &[T]) {
    let bytes = as_bytes(data);
    assert!(bytes.len() as u64 <= buf.length());
    // SAFETY: shared-storage buffer large enough for `bytes`; no GPU work is in flight
    // (the previous command buffer was waited on).
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.contents() as *mut u8, bytes.len()) }
}

/// Encoder helper: binds buffers and scalar arguments, then dispatches.
struct Enc<'a> {
    enc: &'a ComputeCommandEncoderRef,
}

#[derive(Clone, Copy)]
enum Arg<'a> {
    B(&'a Buffer),
    /// Buffer bound at a byte offset.
    BO(&'a Buffer, u64),
    U(u32),
    F(f32),
}

impl Enc<'_> {
    fn bind(&self, p: &ComputePipelineState, args: &[Arg<'_>]) {
        self.enc.set_compute_pipeline_state(p);
        for (i, a) in args.iter().enumerate() {
            let i = i as u64;
            match a {
                Arg::B(b) => self.enc.set_buffer(i, Some(b), 0),
                Arg::BO(b, off) => self.enc.set_buffer(i, Some(b), *off),
                Arg::U(v) => self.enc.set_bytes(i, 4, v as *const u32 as *const std::ffi::c_void),
                Arg::F(v) => self.enc.set_bytes(i, 4, v as *const f32 as *const std::ffi::c_void),
            }
        }
    }

    fn groups(&self, p: &ComputePipelineState, args: &[Arg<'_>], grid: (u64, u64, u64), tg: u64) {
        self.bind(p, args);
        self.enc.dispatch_thread_groups(MTLSize::new(grid.0, grid.1, grid.2), MTLSize::new(tg, 1, 1));
    }

    fn threads(&self, p: &ComputePipelineState, args: &[Arg<'_>], grid: (u64, u64, u64)) {
        self.bind(p, args);
        let w = p.thread_execution_width().min(grid.0.max(1));
        let tg_x = w.max(1);
        let tg_y = (p.max_total_threads_per_threadgroup() / tg_x).clamp(1, grid.1.max(1)).min(8);
        self.enc.dispatch_threads(MTLSize::new(grid.0, grid.1, grid.2), MTLSize::new(tg_x, tg_y, 1));
    }
}

impl MetalLlama {
    /// `y[m, rows] = x[m, cols] · Wᵀ`, or `y += …` when `accumulate`.
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, e: &Enc<'_>, w: &Weight, x: &Buffer, xh: &Buffer, y: &Buffer, m: usize, accumulate: bool) {
        self.linear_segments(e, std::slice::from_ref(w), x, xh, y, m, accumulate, false);
    }

    /// Linear layer whose weight rows are split into segments (possibly of different
    /// formats); segment `i` writes output columns starting after segments `0..i`.
    /// For prefill-sized batches the input is converted to half precision once into
    /// `xh`, which the tiled GEMM kernels read.
    #[allow(clippy::too_many_arguments)]
    /// With `half_ready`, `xh` already holds the input in half precision (written by the
    /// previous kernel; only valid when [`Self::reads_only_half`]) and `x` is not read.
    #[allow(clippy::too_many_arguments)]
    fn linear_segments(&self, e: &Enc<'_>, segs: &[Weight], x: &Buffer, xh: &Buffer, y: &Buffer, m: usize, accumulate: bool, half_ready: bool) {
        let ldy = segs.iter().map(|w| w.rows).sum::<usize>() as u32;
        if !half_ready && segs.iter().any(|w| w.kind != WKind::F32 && m > mv_max_rows(w.kind)) {
            let quads = (m * segs[0].cols / 4) as u64;
            e.threads(self.pipeline("to_half"), &[Arg::B(x), Arg::B(xh)], (quads, 1, 1));
        }
        let mut col = 0u64;
        for w in segs {
            let (n, k) = (w.rows as u32, w.cols as u32);
            let args = [
                Arg::B(&w.buf),
                Arg::B(x),
                Arg::BO(y, col * 4),
                Arg::U(n),
                Arg::U(k),
                Arg::U(m as u32),
                Arg::U(accumulate as u32),
                Arg::U(ldy),
            ];
            if m <= mv_max_rows(w.kind) {
                // One activation row: the tuned decode kernels; 2..8 rows: kernels that
                // decode each weight chunk once for all rows.
                let name = match m {
                    1 => format!("mv_{}", w.kind.suffix()),
                    2 => format!("mv2_{}", w.kind.suffix()),
                    3 | 4 => format!("mv4_{}", w.kind.suffix()),
                    _ => format!("mv8_{}", w.kind.suffix()),
                };
                let p = self.pipeline(&name);
                e.groups(p, &args, ((w.rows as u64).div_ceil(8), 1, 1), 64);
            } else if w.kind != WKind::F32 && m <= MMS_MAX_ROWS {
                let nt = m.div_ceil(8);
                // Row groups per simdgroup (see the `mms` kernel): 2 from 16 tokens,
                // except for F16 weights at 16.
                let rg = if nt == 1 || (nt == 2 && w.kind == WKind::F16) { 1 } else { 2 };
                let name = if rg == 1 { format!("mms{nt}_{}", w.kind.suffix()) } else { format!("mms{nt}r{rg}_{}", w.kind.suffix()) };
                let mut args = args;
                args[1] = Arg::B(xh);
                e.groups(self.pipeline(&name), &args, ((w.rows as u64).div_ceil(16 * rg as u64), 1, 1), 64);
            } else if w.kind == WKind::F32 {
                // Full-precision weights keep the F32 tile kernel (the large-tile kernel stores
                // weights as half precision, which is lossless only for F16/quantized weights).
                e.groups(self.pipeline("mm_f32"), &args, ((w.rows as u64).div_ceil(32), (m as u64).div_ceil(32), 1), 128);
            } else {
                let mut args = args;
                args[1] = Arg::B(xh);
                let tiles = w.rows.div_ceil(64) * m.div_ceil(64);
                if w.kind != WKind::Q6K || tiles < MML_MAX_TILES {
                    // Two blocks per stage, or one for rows with an odd number of blocks.
                    let stage = if (w.cols / 32).is_multiple_of(2) { "mml" } else { "mml1" };
                    let p = self.pipeline(&format!("{stage}_{}", w.kind.suffix()));
                    e.groups(p, &args, ((w.rows as u64).div_ceil(64), (m as u64).div_ceil(32), 1), 128);
                } else {
                    let p = self.pipeline(&format!("mm_{}", w.kind.suffix()));
                    e.groups(p, &args, ((w.rows as u64).div_ceil(64), (m as u64).div_ceil(64), 1), 128);
                }
            }
            col += w.rows as u64;
        }
    }

    /// Whether the gate/up GEMM for `m` rows runs as the 64x32 tile kernel with SwiGLU
    /// in its epilogue: the tiled-GEMM range, except Q6_K on large grids (where the 64x64
    /// tile is faster).
    fn glu_fused(w: &Weight, m: usize) -> bool {
        let tiles = w.rows.div_ceil(64) * m.div_ceil(64);
        w.kind != WKind::F32 && m > MMS_MAX_ROWS && (w.kind != WKind::Q6K || tiles < MML_MAX_TILES)
    }

    /// Interleaved gate/up projection of `m` rows with SwiGLU fused: writes
    /// `silu(gate) * up` as half precision to `out` (`[m, rows / 2]`). With `half_ready`,
    /// `xh` already holds the input in half precision.
    #[allow(clippy::too_many_arguments)]
    fn gate_up_glu(&self, e: &Enc<'_>, w: &Weight, x: &Buffer, xh: &Buffer, out: &Buffer, m: usize, half_ready: bool) {
        if !half_ready {
            e.threads(self.pipeline("to_half"), &[Arg::B(x), Arg::B(xh)], ((m * w.cols / 4) as u64, 1, 1));
        }
        let stage = if (w.cols / 32).is_multiple_of(2) { "mmlg" } else { "mmlg1" };
        let args = [
            Arg::B(&w.buf),
            Arg::B(xh),
            Arg::B(out),
            Arg::U(w.rows as u32),
            Arg::U(w.cols as u32),
            Arg::U(m as u32),
            Arg::U(0),
            Arg::U((w.rows / 2) as u32),
        ];
        e.groups(self.pipeline(&format!("{stage}_{}", w.kind.suffix())), &args, ((w.rows as u64).div_ceil(64), (m as u64).div_ceil(32), 1), 128);
    }

    /// Whether every segment of a linear layer reads only the half-precision input (`xh`)
    /// for `m` rows, so the producer may write half precision directly.
    fn reads_only_half(segs: &[Weight], m: usize) -> bool {
        segs.iter().all(|w| w.kind != WKind::F32 && m > mv_max_rows(w.kind))
    }

    /// RMSNorm into `y` (F32), or into `xh` as half precision when the next linear
    /// layer reads only that; returns whether it wrote `xh`.
    #[allow(clippy::too_many_arguments)]
    fn rms_for(&self, e: &Enc<'_>, x: &Buffer, w: &Buffer, y: &Buffer, xh: &Buffer, rows: usize, next: &[Weight]) -> bool {
        let half = Self::reads_only_half(next, rows);
        let c = &self.config;
        let args = [
            Arg::B(x),
            Arg::B(w),
            Arg::B(if half { xh } else { y }),
            Arg::B(x),
            Arg::U(c.hidden_size as u32),
            Arg::F(c.rms_norm_eps),
            Arg::U(0),
        ];
        e.groups(self.pipeline(if half { "rms_norm_h" } else { "rms_norm" }), &args, (rows as u64, 1, 1), 256);
        half
    }

    fn rms(&self, e: &Enc<'_>, x: &Buffer, w: &Buffer, y: &Buffer, rows: usize, gather: Option<&Buffer>) {
        let c = &self.config;
        let args = [
            Arg::B(x),
            Arg::B(w),
            Arg::B(y),
            Arg::B(gather.unwrap_or(x)),
            Arg::U(c.hidden_size as u32),
            Arg::F(c.rms_norm_eps),
            Arg::U(gather.is_some() as u32),
        ];
        e.groups(self.pipeline("rms_norm"), &args, (rows as u64, 1, 1), 256);
    }
}

impl LanguageModel for MetalLlama {
    fn config(&self) -> &LlamaConfig {
        &self.config
    }

    fn device_name(&self) -> String {
        format!("metal ({})", self.device.name())
    }

    fn init_kv_cache(&self, num_blocks: usize, block_size: usize) -> Result<()> {
        if !block_size.is_multiple_of(8) {
            return Err(RllmError::BackendError("Metal executor needs a KV block size divisible by 8".into()));
        }
        let c = &self.config;
        let elem = if self.kv_type == "f16" { 2 } else { 4 };
        let bytes = (num_blocks * c.num_key_value_heads * block_size * c.head_dim() * elem) as u64;
        let mut st = self.state.lock().map_err(|_| RllmError::BackendError("Metal state poisoned".into()))?;
        st.kv = (0..c.num_hidden_layers)
            .map(|_| {
                (
                    self.device.new_buffer(bytes, MTLResourceOptions::StorageModePrivate),
                    self.device.new_buffer(bytes, MTLResourceOptions::StorageModePrivate),
                )
            })
            .collect();
        // Zero the cache: the tiled attention kernel reads whole 8-key groups (masking
        // unused keys), so unwritten slots must hold finite values.
        objc::rc::autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            for (k, v) in &st.kv {
                blit.fill_buffer(k, metal::NSRange::new(0, bytes), 0);
                blit.fill_buffer(v, metal::NSRange::new(0, bytes), 0);
            }
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });
        st.block_size = block_size;
        st.num_blocks = num_blocks;
        Ok(())
    }

    fn forward_cached(&self, batch: &ForwardBatch<'_>) -> Result<Vec<f32>> {
        let mut guard = self.state.lock().map_err(|_| RllmError::BackendError("Metal state poisoned".into()))?;
        let st = &mut *guard;
        if st.kv.is_empty() {
            return Err(RllmError::BackendError("KV cache not initialised; call init_kv_cache".into()));
        }
        self.validate(batch, st.block_size, st.num_blocks)?;

        let c = &self.config;
        let n = batch.tokens.len();
        let r = batch.logits_rows.len();
        let (h, inter, hd) = (c.hidden_size, c.intermediate_size, c.head_dim());
        let (nq, nkv) = (c.num_attention_heads, c.num_key_value_heads);
        let kv_dim = nkv * hd;
        let qkv_dim = h + 2 * kv_dim;

        // Flatten block tables, sharing one copy between consecutive rows of a sequence.
        let mut tables: Vec<u32> = Vec::new();
        let mut table_off: Vec<u32> = Vec::with_capacity(n);
        for (i, bt) in batch.block_tables.iter().enumerate() {
            if i > 0 && batch.block_tables[i - 1].block_ids == bt.block_ids {
                table_off.push(table_off[i - 1]);
            } else {
                table_off.push(tables.len() as u32);
                tables.extend(bt.block_ids.iter().map(|&b| b as u32));
            }
        }
        let positions: Vec<u32> = batch.positions.iter().map(|&p| p as u32).collect();
        // Group consecutive rows of one sequence (same table, consecutive positions) into
        // tiles of up to 32 for the tiled attention kernel; lone rows (decode) use the
        // per-row kernel, which parallelises over keys instead.
        let mut tiles: Vec<[u32; 2]> = Vec::new();
        let mut singles: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n
                && j - i < ATTN_TILE_ROWS
                && table_off[j] == table_off[i]
                && batch.positions[j] == batch.positions[j - 1] + 1
            {
                j += 1;
            }
            if j - i >= 2 {
                tiles.push([i as u32, (j - i) as u32]);
            } else {
                singles.push(i as u32);
            }
            i = j;
        }
        let logit_rows: Vec<u32> = batch.logits_rows.iter().map(|&x| x as u32).collect();

        // (Re)allocate scratch when the batch outgrows it.
        let sc = &mut st.scratch;
        if n > sc.rows || r > sc.logit_rows || tables.len() > sc.table_len {
            // (q gets 32 rows of slack: the tiled attention kernel loads 8-row fragments for
            // each of its four simdgroups.)
            let rows = n.max(sc.rows);
            let lrows = r.max(sc.logit_rows);
            let tlen = tables.len().max(sc.table_len).max(1);
            let f = |len: usize| self.device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared);
            sc.bufs.clear();
            for (name, len) in [
                ("tokens", rows),
                ("positions", rows),
                ("table_off", rows),
                ("logit_rows", lrows),
                ("tables", tlen),
                ("hidden", rows * h),
                ("normed", rows * h),
                ("qkv", (rows + ATTN_TILE_ROWS) * qkv_dim),
                ("tiles", rows * 2),
                ("singles", rows),
                ("attn", rows * h),
                ("gu", rows * 2 * inter),
                ("act", rows * inter),
                ("final", lrows * h),
                // Half-precision copy of a linear layer's input (prefill GEMM).
                // At least MMS_MAX_ROWS rows: the small-batch GEMM reads whole 8-row
                // fragments (rows past the batch only feed discarded outputs).
                ("xh", rows.max(lrows).max(MMS_MAX_ROWS) * h.max(inter) / 2 + 4),
                ("logits", lrows * c.vocab_size),
            ] {
                sc.bufs.insert(name, f(len));
            }
            sc.rows = rows;
            sc.logit_rows = lrows;
            sc.table_len = tlen;
        }
        // Decode attention splits each row's context into chunks of DECODE_CHUNK keys
        // (one threadgroup each); partials need a buffer sized by the longest context.
        let max_single_ctx = singles.iter().map(|&r| batch.positions[r as usize] + 1).max().unwrap_or(0);
        let attn_heads = if GQA_GROUPS.contains(&(nq / nkv)) && std::env::var_os("RLLM_NO_GQA_ATTN").is_none() { nkv } else { nq };
        let decode_chunk = if attn_heads * singles.len() * max_single_ctx.div_ceil(DECODE_CHUNK) < DECODE_MIN_GROUPS {
            DECODE_CHUNK_SMALL
        } else {
            DECODE_CHUNK
        };
        let nsplit = max_single_ctx.div_ceil(decode_chunk).max(1);
        let part_len = (singles.len() * nq * nsplit * (hd + 2)).max(1) * 4;
        if sc.bufs.get("attn_part").is_none_or(|buf| (buf.length() as usize) < part_len) {
            sc.bufs.insert("attn_part", self.device.new_buffer(part_len as u64, MTLResourceOptions::StorageModePrivate));
        }
        let b = |name: &str| &sc.bufs[name];
        write_buffer(b("tokens"), batch.tokens);
        write_buffer(b("positions"), &positions);
        write_buffer(b("table_off"), &table_off);
        write_buffer(b("logit_rows"), &logit_rows);
        write_buffer(b("tables"), &tables);
        write_buffer(b("tiles"), &tiles);
        write_buffer(b("singles"), &singles);

        let scale = 1.0 / (hd as f32).sqrt();
        let result = objc::rc::autoreleasepool(|| -> Result<Vec<f32>> {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let e = Enc { enc };

            // Embedding lookup.
            e.threads(
                self.pipeline(&format!("embed_{}", self.embed.kind.suffix())),
                &[Arg::B(&self.embed.buf), Arg::B(b("tokens")), Arg::B(b("hidden")), Arg::U(h as u32)],
                ((h / 8) as u64, n as u64, 1),
            );

            for (li, layer) in self.layers.iter().enumerate() {
                let (kc, vc) = &st.kv[li];
                // Attention block: norm -> fused QKV -> RoPE on q and k heads -> KV write ->
                // attention -> output projection accumulated into the residual stream.
                let ready = self.rms_for(&e, b("hidden"), &layer.attn_norm, b("normed"), b("xh"), n, &layer.qkv);
                self.linear_segments(&e, &layer.qkv, b("normed"), b("xh"), b("qkv"), n, false, ready);
                e.threads(
                    self.pipeline("rope"),
                    &[
                        Arg::B(b("qkv")),
                        Arg::B(&self.cos),
                        Arg::B(&self.sin),
                        Arg::B(b("positions")),
                        Arg::U((nq + nkv) as u32),
                        Arg::U(hd as u32),
                        Arg::U(qkv_dim as u32),
                    ],
                    ((hd / 2) as u64, (nq + nkv) as u64, n as u64),
                );
                e.threads(
                    self.pipeline(&format!("kv_write_{}", self.kv_type)),
                    &[
                        Arg::B(b("qkv")),
                        Arg::U(h as u32),
                        Arg::U((h + kv_dim) as u32),
                        Arg::U(qkv_dim as u32),
                        Arg::B(kc),
                        Arg::B(vc),
                        Arg::B(b("tables")),
                        Arg::B(b("table_off")),
                        Arg::B(b("positions")),
                        Arg::U(nkv as u32),
                        Arg::U(hd as u32),
                        Arg::U(st.block_size as u32),
                    ],
                    (hd as u64, nkv as u64, n as u64),
                );
                let attn_base = [
                    Arg::B(b("qkv")),
                    Arg::B(kc),
                    Arg::B(vc),
                    Arg::B(b("attn")),
                    Arg::B(b("tables")),
                    Arg::B(b("table_off")),
                    Arg::B(b("positions")),
                    Arg::U(nq as u32),
                    Arg::U(nkv as u32),
                    Arg::U(hd as u32),
                    Arg::U(st.block_size as u32),
                    Arg::F(scale),
                ];
                let attn_args = |extra| {
                    let mut v = attn_base.to_vec();
                    v.push(Arg::B(extra));
                    v.push(Arg::U(qkv_dim as u32));
                    v
                };
                if !tiles.is_empty() {
                    // Dynamic threadgroup memory: K/V staging (2 x KB x hd halves), reused
                    // for the 4 x 8 x hd float output at the end.
                    let tg_bytes = (2 * ATTN_KB * hd * 2).max(4 * 8 * hd * 4) as u64;
                    e.enc.set_threadgroup_memory_length(0, tg_bytes);
                    e.groups(self.pipeline(&format!("attention_tile32_hd{hd}_{}", self.kv_type)), &attn_args(b("tiles")), (nq as u64, tiles.len() as u64, 1), 128);
                }
                if !singles.is_empty() {
                    let mut args = attn_args(b("singles"));
                    args.extend([Arg::B(b("attn_part")), Arg::U(decode_chunk as u32), Arg::U(nsplit as u32)]);
                    let name = if attn_heads == nkv && nkv != nq {
                        format!("attention_gqa{}_hd{hd}_{}", nq / nkv, self.kv_type)
                    } else {
                        format!("attention_hd{hd}_{}", self.kv_type)
                    };
                    e.groups(self.pipeline(&name), &args, (attn_heads as u64, singles.len() as u64, nsplit as u64), 128);
                    if nsplit > 1 {
                        e.groups(
                            self.pipeline("attn_reduce"),
                            &[
                                Arg::B(b("attn_part")),
                                Arg::B(b("attn")),
                                Arg::B(b("singles")),
                                Arg::U(nq as u32),
                                Arg::U(hd as u32),
                                Arg::U(nsplit as u32),
                            ],
                            (nq as u64, singles.len() as u64, 1),
                            hd as u64,
                        );
                    }
                }
                self.linear(&e, &layer.o, b("attn"), b("xh"), b("hidden"), n, true);

                // MLP block: norm -> fused gate/up -> SwiGLU -> down projection accumulated
                // into the residual stream.
                let ready = self.rms_for(&e, b("hidden"), &layer.ffn_norm, b("normed"), b("xh"), n, &layer.gate_up);
                let down = std::slice::from_ref(&layer.down);
                let down_half = Self::reads_only_half(down, n);
                if layer.gu_interleaved && down_half && Self::glu_fused(&layer.gate_up[0], n) {
                    // Gate/up GEMM with SwiGLU in its epilogue, writing the down
                    // projection's half-precision input to `act`.
                    self.gate_up_glu(&e, &layer.gate_up[0], b("normed"), b("xh"), b("act"), n, ready);
                    self.linear_segments(&e, down, b("act"), b("act"), b("hidden"), n, true, true);
                } else {
                    self.linear_segments(&e, &layer.gate_up, b("normed"), b("xh"), b("gu"), n, false, ready);
                    let name = match (layer.gu_interleaved, down_half) {
                        (false, false) => "swiglu",
                        (false, true) => "swiglu_h",
                        (true, false) => "swiglu_i",
                        (true, true) => "swiglu_ih",
                    };
                    e.threads(
                        self.pipeline(name),
                        &[Arg::B(b("gu")), Arg::B(if down_half { b("xh") } else { b("act") }), Arg::U(inter as u32)],
                        ((n * inter) as u64, 1, 1),
                    );
                    self.linear_segments(&e, down, b("act"), b("xh"), b("hidden"), n, true, down_half);
                }
            }

            // Final norm on the requested rows only, then the LM head.
            self.rms(&e, b("hidden"), &self.final_norm, b("final"), r, Some(b("logit_rows")));
            self.linear(&e, &self.lm_head, b("final"), b("xh"), b("logits"), r, false);

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            if cb.status() == metal::MTLCommandBufferStatus::Error {
                return Err(RllmError::BackendError("Metal command buffer failed".to_string()));
            }
            let len = r * c.vocab_size;
            // SAFETY: the GPU work has completed; the shared buffer holds `len` floats.
            let logits = unsafe { std::slice::from_raw_parts(b("logits").contents() as *const f32, len) };
            Ok(logits.to_vec())
        });
        result
    }
}

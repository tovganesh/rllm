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
}

impl WKind {
    fn suffix(self) -> &'static str {
        match self {
            WKind::F32 => "f32",
            WKind::F16 => "f16",
            WKind::Q8 => "q8_0",
            WKind::Q4 => "q4_0",
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
/// (`[q | k | v]` and `[gate | up]` row blocks) so each is one dispatch.
struct Layer {
    attn_norm: Buffer,
    qkv: Weight,
    o: Weight,
    ffn_norm: Buffer,
    gate_up: Weight,
    down: Weight,
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
    state: Mutex<State>,
}

/// Rows at or below which linear layers use the matrix-vector kernel.
const MV_MAX_ROWS: usize = 8;

impl MetalLlama {
    /// Whether a Metal device is available.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Uploads a CPU-loaded model to the default Metal device. F32, Q8_0 and Q4_0 weights
    /// are uploaded as they are (F32 as F16 when exact); the CPU-only formats are
    /// converted: repacked `Q4_0x4` back to Q4_0, and Q5_0 / Q4_K / Q6_K to Q8_0.
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

        let library = device
            .new_library_with_source(SHADERS_MSL, &CompileOptions::new())
            .map_err(|e| RllmError::BackendError(format!("Metal shader compilation failed: {e}")))?;
        let mut pipelines = HashMap::new();
        let names = [
            "mv_f32", "mv_q8_0", "mv_q4_0", "mv_f16", "mm_f32", "mm_q8_0", "mm_q4_0", "mm_f16", "embed_f32", "embed_q8_0",
            "embed_q4_0", "embed_f16",
            "rms_norm", "rope", "kv_write", "attention", "attention_tile", "swiglu",
        ];
        for name in names {
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
                DType::Q8_0 => WKind::Q8,
                DType::Q4_0 => WKind::Q4,
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
        // Row-concatenation of matrices with equal column counts. Raw blocks are
        // concatenated when every part has the same storage type; otherwise all parts
        // are dequantized to F32.
        let upload_concat = |parts: &[&Tensor]| -> Result<Weight> {
            let cols = parts[0].dims()[1];
            let rows: usize = parts.iter().map(|t| t.dims()[0]).sum();
            if parts.iter().any(|t| t.dims().len() != 2 || t.dims()[1] != cols) {
                return Err(RllmError::BackendError("cannot fuse matrices with different widths".into()));
            }
            // Convert CPU-only formats first so that e.g. Q4_K + Q6_K parts (both Q8_0
            // on Metal) are still concatenated as raw blocks.
            let converted: Vec<Option<Tensor>> = parts.iter().map(|t| to_metal_format(t)).collect::<Result<_>>()?;
            let parts: Vec<&Tensor> = parts.iter().zip(&converted).map(|(t, c)| c.as_ref().unwrap_or(t)).collect();
            let dtype = parts[0].dtype();
            if parts.iter().all(|t| t.dtype() == dtype) {
                let mut bytes = Vec::new();
                for t in parts {
                    let len = dtype.storage_bytes(t.numel());
                    bytes.extend_from_slice(&t.storage().as_cpu()?.as_bytes()[t.offset()..t.offset() + len]);
                }
                let shape = rllm_core::shape::Shape::new(vec![rows, cols]);
                let storage = rllm_core::storage::Storage::Cpu(rllm_core::storage::CpuStorage::from_bytes(bytes));
                return upload(&Tensor::new(shape, dtype, rllm_core::device::Device::Cpu, storage));
            }
            let mut dense = Vec::with_capacity(rows * cols);
            for t in parts {
                dense.extend(dequantize_to_f32(t)?);
            }
            upload(&Tensor::from_f32(&dense, rllm_core::shape::Shape::new(vec![rows, cols]), rllm_core::device::Device::Cpu)?)
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
            layers.push(Layer {
                attn_norm: vec_buf(&l.input_layernorm)?,
                qkv: upload_concat(&[&l.self_attn.q_proj, &l.self_attn.k_proj, &l.self_attn.v_proj])?,
                o: upload(&l.self_attn.o_proj)?,
                ffn_norm: vec_buf(&l.post_attention_layernorm)?,
                gate_up: upload_concat(&[&l.mlp.gate_proj, &l.mlp.up_proj])?,
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
/// `Q5_0` / `Q4_K` / `Q6_K` to `Q8_0`. Returns `None` for formats Metal handles.
fn to_metal_format(t: &Tensor) -> Result<Option<Tensor>> {
    use rllm_backend_cpu::ops::{quant::quantize_f32_to_q8_0, repack::unpack_q4_0x4};
    use rllm_core::{shape::Shape, storage::{CpuStorage, Storage}};
    let (dtype, bytes) = match t.dtype() {
        DType::Q4_0x4 => {
            let (n, k) = (t.dims()[0], t.dims()[1]);
            let src = &t.storage().as_cpu()?.as_bytes()[t.offset()..t.offset() + t.dtype().storage_bytes(t.numel())];
            (DType::Q4_0, unpack_q4_0x4(src, n, k))
        }
        DType::Q5_0 | DType::Q4_K | DType::Q6_K => {
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
    fn linear(&self, e: &Enc<'_>, w: &Weight, x: &Buffer, y: &Buffer, m: usize, accumulate: bool) {
        let (n, k) = (w.rows as u32, w.cols as u32);
        let args = [Arg::B(&w.buf), Arg::B(x), Arg::B(y), Arg::U(n), Arg::U(k), Arg::U(m as u32), Arg::U(accumulate as u32)];
        if m <= MV_MAX_ROWS {
            let p = self.pipeline(&format!("mv_{}", w.kind.suffix()));
            e.groups(p, &args, ((w.rows as u64).div_ceil(8), 1, 1), 64);
        } else if w.kind == WKind::F32 {
            // Full-precision weights keep the F32 tile kernel (the large-tile kernel stores
            // weights as half precision, which is lossless only for F16/quantized weights).
            e.groups(self.pipeline("mm_f32"), &args, ((w.rows as u64).div_ceil(32), (m as u64).div_ceil(32), 1), 128);
        } else {
            let p = self.pipeline(&format!("mm_{}", w.kind.suffix()));
            e.groups(p, &args, ((w.rows as u64).div_ceil(64), (m as u64).div_ceil(64), 1), 128);
        }
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
        let bytes = (num_blocks * c.num_key_value_heads * block_size * c.head_dim() * 4) as u64;
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
        // tiles of up to 8 for the tiled attention kernel; lone rows (decode) use the
        // per-row kernel, which parallelises over keys instead.
        let mut tiles: Vec<[u32; 2]> = Vec::new();
        let mut singles: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n
                && j - i < 8
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
            // (q gets 8 rows of slack: the tiled attention kernel loads 8-row fragments.)
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
                ("qkv", (rows + 8) * qkv_dim),
                ("tiles", rows * 2),
                ("singles", rows),
                ("attn", rows * h),
                ("gu", rows * 2 * inter),
                ("act", rows * inter),
                ("final", lrows * h),
                ("logits", lrows * c.vocab_size),
            ] {
                sc.bufs.insert(name, f(len));
            }
            sc.rows = rows;
            sc.logit_rows = lrows;
            sc.table_len = tlen;
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
                self.rms(&e, b("hidden"), &layer.attn_norm, b("normed"), n, None);
                self.linear(&e, &layer.qkv, b("normed"), b("qkv"), n, false);
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
                    self.pipeline("kv_write"),
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
                    e.groups(self.pipeline("attention_tile"), &attn_args(b("tiles")), (nq as u64, tiles.len() as u64, 1), 32);
                }
                if !singles.is_empty() {
                    e.groups(self.pipeline("attention"), &attn_args(b("singles")), (nq as u64, singles.len() as u64, 1), 128);
                }
                self.linear(&e, &layer.o, b("attn"), b("hidden"), n, true);

                // MLP block: norm -> fused gate/up -> SwiGLU -> down projection accumulated
                // into the residual stream.
                self.rms(&e, b("hidden"), &layer.ffn_norm, b("normed"), n, None);
                self.linear(&e, &layer.gate_up, b("normed"), b("gu"), n, false);
                e.threads(
                    self.pipeline("swiglu"),
                    &[Arg::B(b("gu")), Arg::B(b("act")), Arg::U(inter as u32)],
                    ((n * inter) as u64, 1, 1),
                );
                self.linear(&e, &layer.down, b("act"), b("hidden"), n, true);
            }

            // Final norm on the requested rows only, then the LM head.
            self.rms(&e, b("hidden"), &self.final_norm, b("final"), r, Some(b("logit_rows")));
            self.linear(&e, &self.lm_head, b("final"), b("logits"), r, false);

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

//! Direct quantized matrix-vector kernels (Q4_0, Q8_0) for edge and mobile execution,
//! and dequantization of ggml block formats.

// Block-size slicing with `chunks_exact(BLOCK_BYTES)` mirrors the ggml layouts directly.
#![allow(clippy::chunks_exact_to_as_chunks)]

use half::f16;
use rayon::prelude::*;

/// Number of values grouped into a single Q4_0 block.
pub const QK4_0: usize = 32;

/// Number of values grouped into a single Q8_0 block.
pub const QK8_0: usize = 32;

/// Memory layout of a 32-element Q4_0 block (18 bytes).
#[repr(C)]
pub struct BlockQ4_0 {
    /// 16-bit half-precision delta scaling factor.
    pub d: f16,
    /// 16 bytes storing 32 packed 4-bit nibbles.
    pub qs: [u8; 16],
}

/// Memory layout of a 32-element Q8_0 block (34 bytes).
#[repr(C)]
pub struct BlockQ8_0 {
    /// 16-bit half-precision delta scaling factor.
    pub d: f16,
    /// 32 bytes storing 32 8-bit signed integers.
    pub qs: [i8; 32],
}

/// Computes the dot product of a Q4_0 quantized weight row with an F32 activation vector.
#[inline]
pub fn dot_q4_0_f32(q4_bytes: &[u8], f32_vec: &[f32]) -> f32 {
    let k = f32_vec.len();
    let num_blocks = k / QK4_0;
    debug_assert_eq!(q4_bytes.len(), num_blocks * 18);

    let mut sum = 0.0f32;
    for b in 0..num_blocks {
        let block_offset = b * 18;
        let d_bytes = [q4_bytes[block_offset], q4_bytes[block_offset + 1]];
        let d = f16::from_le_bytes(d_bytes).to_f32();
        let qs = &q4_bytes[block_offset + 2..block_offset + 18];
        let x = &f32_vec[b * QK4_0..(b + 1) * QK4_0];

        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let byte = qs[i];
            let v0 = (byte & 0x0F) as i8 - 8;
            let v1 = (byte >> 4) as i8 - 8;
            block_sum += v0 as f32 * x[i] + v1 as f32 * x[i + 16];
        }
        sum += block_sum * d;
    }
    sum
}

/// Computes the dot product of a Q8_0 quantized weight row with an F32 activation vector.
#[inline]
pub fn dot_q8_0_f32(q8_bytes: &[u8], f32_vec: &[f32]) -> f32 {
    let k = f32_vec.len();
    let num_blocks = k / QK8_0;
    debug_assert_eq!(q8_bytes.len(), num_blocks * 34);

    let mut sum = 0.0f32;
    for b in 0..num_blocks {
        let block_offset = b * 34;
        let d_bytes = [q8_bytes[block_offset], q8_bytes[block_offset + 1]];
        let d = f16::from_le_bytes(d_bytes).to_f32();
        let qs = &q8_bytes[block_offset + 2..block_offset + 34];
        let x = &f32_vec[b * QK8_0..(b + 1) * QK8_0];

        let mut block_sum = 0.0f32;
        for i in 0..32 {
            let val = qs[i] as i8;
            block_sum += val as f32 * x[i];
        }
        sum += block_sum * d;
    }
    sum
}

/// Multiplies a matrix `[N, K]` stored in `Q4_0` by vector `[K]` into `out [N]` in F32.
/// Eliminates intermediate full-matrix dequantization memory allocations.
pub fn gemv_q4_0_f32(
    weight_bytes: &[u8],
    act: &[f32],
    out: &mut [f32],
    n: usize,
    k: usize,
) {
    let bytes_per_row = (k / QK4_0) * 18;
    debug_assert_eq!(weight_bytes.len(), n * bytes_per_row);
    debug_assert_eq!(act.len(), k);
    debug_assert_eq!(out.len(), n);

    out.par_chunks_mut(16)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * 16;
            for (i, val) in out_chunk.iter_mut().enumerate() {
                let row = row_start + i;
                let row_bytes = &weight_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                *val = dot_q4_0_f32(row_bytes, act);
            }
        });
}

/// Multiplies a matrix `[N, K]` stored in `Q8_0` by vector `[K]` into `out [N]` in F32.
pub fn gemv_q8_0_f32(
    weight_bytes: &[u8],
    act: &[f32],
    out: &mut [f32],
    n: usize,
    k: usize,
) {
    let bytes_per_row = (k / QK8_0) * 34;
    debug_assert_eq!(weight_bytes.len(), n * bytes_per_row);
    debug_assert_eq!(act.len(), k);
    debug_assert_eq!(out.len(), n);

    out.par_chunks_mut(16)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * 16;
            for (i, val) in out_chunk.iter_mut().enumerate() {
                let row = row_start + i;
                let row_bytes = &weight_bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
                *val = dot_q8_0_f32(row_bytes, act);
            }
        });
}

/// Quantizes an F32 slice (length a multiple of 32) into `Q8_0` blocks, like ggml's
/// reference `quantize_row_q8_0`.
pub fn quantize_f32_to_q8_0(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % QK8_0, 0);
    assert_eq!(dst.len(), src.len() / QK8_0 * 34);
    for (block, out) in src.chunks_exact(QK8_0).zip(dst.chunks_exact_mut(34)) {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        for (o, v) in out[2..].iter_mut().zip(block) {
            *o = (v * id).round() as i8 as u8;
        }
    }
}

/// Quantizes an uncompressed F32 slice into `Q4_0` format blocks.
pub fn quantize_f32_to_q4_0(src: &[f32], dst: &mut [u8]) {
    let k = src.len();
    assert_eq!(k % QK4_0, 0);
    let num_blocks = k / QK4_0;
    assert_eq!(dst.len(), num_blocks * 18);

    for b in 0..num_blocks {
        let block_f32 = &src[b * QK4_0..(b + 1) * QK4_0];
        let mut amax = 0.0f32;
        for &val in block_f32 {
            amax = amax.max(val.abs());
        }
        let d = amax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = f16::from_f32(d);

        let out_block = &mut dst[b * 18..(b + 1) * 18];
        let le_bytes = d_f16.to_le_bytes();
        out_block[0] = le_bytes[0];
        out_block[1] = le_bytes[1];

        for i in 0..16 {
            let x0 = block_f32[i] * id;
            let x1 = block_f32[i + 16] * id;
            let q0 = ((x0 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
            let q1 = ((x1 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
            out_block[2 + i] = q0 | (q1 << 4);
        }
    }
}

// ───────────────────────── dequantization (ggml block formats) ─────────────────────────

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

/// Dequantizes `Q4_0` blocks (32 values / 18 bytes) into `out`.
pub fn dequantize_q4_0(src: &[u8], out: &mut [f32]) {
    for (block, y) in src.chunks_exact(18).zip(out.chunks_exact_mut(QK4_0)) {
        let d = f16_at(block, 0);
        for i in 0..16 {
            let byte = block[2 + i];
            y[i] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            y[i + 16] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
    }
}

/// Dequantizes `Q8_0` blocks (32 values / 34 bytes) into `out`.
pub fn dequantize_q8_0(src: &[u8], out: &mut [f32]) {
    for (block, y) in src.chunks_exact(34).zip(out.chunks_exact_mut(QK8_0)) {
        let d = f16_at(block, 0);
        for i in 0..32 {
            y[i] = block[2 + i] as i8 as f32 * d;
        }
    }
}

/// Dequantizes `Q4_1` blocks (32 values / 20 bytes: scale, min, 16 nibble bytes).
pub fn dequantize_q4_1(src: &[u8], out: &mut [f32]) {
    for (block, y) in src.chunks_exact(20).zip(out.chunks_exact_mut(32)) {
        let (d, m) = (f16_at(block, 0), f16_at(block, 2));
        for i in 0..16 {
            let byte = block[4 + i];
            y[i] = (byte & 0x0F) as f32 * d + m;
            y[i + 16] = (byte >> 4) as f32 * d + m;
        }
    }
}

/// Dequantizes `Q5_0` blocks (32 values / 22 bytes: scale, 32 high bits, 16 nibble bytes).
pub fn dequantize_q5_0(src: &[u8], out: &mut [f32]) {
    for (block, y) in src.chunks_exact(22).zip(out.chunks_exact_mut(32)) {
        let d = f16_at(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
            let q = block[6 + j];
            y[j] = (((q & 0x0F) | xh0) as i32 - 16) as f32 * d;
            y[j + 16] = (((q >> 4) | xh1) as i32 - 16) as f32 * d;
        }
    }
}

/// Converts `Q5_0` blocks to `Q8_0` without loss: each 5-bit value `q - 16` is stored
/// as an int8 with the block's own scale, so dot products are bit-identical.
pub fn q5_0_to_q8_0(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / 22 * 34);
    for block in src.chunks_exact(22) {
        out.extend_from_slice(&block[..2]);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let q = &block[6..22];
        for j in 0..32 {
            let nib = if j < 16 { q[j] & 0x0F } else { q[j - 16] >> 4 };
            out.push((((nib as u32) | (((qh >> j) & 1) << 4)) as i32 - 16) as i8 as u8);
        }
    }
    out
}

/// Dequantizes `Q5_1` blocks (32 values / 24 bytes: scale, min, 32 high bits, 16 nibble bytes).
pub fn dequantize_q5_1(src: &[u8], out: &mut [f32]) {
    for (block, y) in src.chunks_exact(24).zip(out.chunks_exact_mut(32)) {
        let (d, m) = (f16_at(block, 0), f16_at(block, 2));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
            let q = block[8 + j];
            y[j] = ((q & 0x0F) | xh0) as f32 * d + m;
            y[j + 16] = ((q >> 4) | xh1) as f32 * d + m;
        }
    }
}

/// Number of values in one K-quant super-block.
pub const QK_K: usize = 256;

/// Unpacks the 6-bit scale and min for sub-block `j` of a Q4_K / Q5_K super-block.
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (
            ((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)) as f32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32,
        )
    }
}

/// Dequantizes `Q4_K` super-blocks (256 values / 144 bytes) into `out`.
pub fn dequantize_q4_k(src: &[u8], out: &mut [f32]) {
    for (block, y) in src.chunks_exact(144).zip(out.chunks_exact_mut(QK_K)) {
        let d = f16_at(block, 0);
        let dmin = f16_at(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..144];
        for (chunk, j) in (0..QK_K).step_by(64).enumerate() {
            let (sc1, m1) = scale_min_k4(2 * chunk, scales);
            let (sc2, m2) = scale_min_k4(2 * chunk + 1, scales);
            let q = &qs[chunk * 32..chunk * 32 + 32];
            for l in 0..32 {
                y[j + l] = d * sc1 * (q[l] & 0x0F) as f32 - dmin * m1;
                y[j + 32 + l] = d * sc2 * (q[l] >> 4) as f32 - dmin * m2;
            }
        }
    }
}

/// Quantizes `src` (length a multiple of 256) to `Q4_K` with a simple min/max fit per
/// 32-value sub-block (not llama.cpp's error-minimising search). For tests and tools.
pub fn quantize_f32_to_q4_k(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / QK_K * 144);
    for x in src.chunks_exact(QK_K) {
        let mut scales = [0f32; 8];
        let mut mins = [0f32; 8];
        for (j, sub) in x.chunks_exact(32).enumerate() {
            let lo = sub.iter().cloned().fold(f32::INFINITY, f32::min).min(0.0);
            let hi = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            scales[j] = (hi - lo) / 15.0;
            mins[j] = -lo;
        }
        let max_sc = scales.iter().cloned().fold(0f32, f32::max);
        let max_m = mins.iter().cloned().fold(0f32, f32::max);
        let d = f16::from_f32(max_sc / 63.0);
        let dmin = f16::from_f32(max_m / 63.0);
        let (df, dminf) = (d.to_f32(), dmin.to_f32());
        let q6 = |v: f32, base: f32| if base > 0.0 { (v / base).round().clamp(0.0, 63.0) as u8 } else { 0 };
        let ls: Vec<u8> = scales.iter().map(|&v| q6(v, df)).collect();
        let lm: Vec<u8> = mins.iter().map(|&v| q6(v, dminf)).collect();
        let mut packed = [0u8; 12];
        for j in 0..8 {
            if j < 4 {
                packed[j] = ls[j];
                packed[j + 4] = lm[j];
            } else {
                packed[j + 4] = (ls[j] & 0x0F) | ((lm[j] & 0x0F) << 4);
                packed[j - 4] |= (ls[j] >> 4) << 6;
                packed[j] |= (lm[j] >> 4) << 6;
            }
        }
        let mut qs = [0u8; 128];
        for j in 0..8 {
            let (sc, m) = scale_min_k4(j, &packed);
            let (step, off) = (df * sc, dminf * m);
            for l in 0..32 {
                let q = if step > 0.0 { ((x[j * 32 + l] + off) / step).round().clamp(0.0, 15.0) as u8 } else { 0 };
                let byte = &mut qs[(j / 2) * 32 + l];
                *byte |= if j % 2 == 0 { q } else { q << 4 };
            }
        }
        out.extend_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&dmin.to_le_bytes());
        out.extend_from_slice(&packed);
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantizes `src` (length a multiple of 256) to `Q6_K` with an absmax fit per
/// 16-value group. For tests and tools.
pub fn quantize_f32_to_q6_k(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / QK_K * 210);
    for x in src.chunks_exact(QK_K) {
        let group_scale: Vec<f32> = x.chunks_exact(16).map(|g| g.iter().fold(0f32, |m, v| m.max(v.abs())) / 31.0).collect();
        let max_sc = group_scale.iter().cloned().fold(0f32, f32::max);
        let d = f16::from_f32(max_sc / 127.0);
        let df = d.to_f32();
        let sc: Vec<i8> = group_scale.iter().map(|&v| if df > 0.0 { (v / df).round().clamp(-128.0, 127.0) as i8 } else { 0 }).collect();
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for (p, &v) in x.iter().enumerate() {
            let step = df * sc[p / 16] as f32;
            let q = if step != 0.0 { (v / step).round().clamp(-32.0, 31.0) as i32 + 32 } else { 32 } as u8;
            let (half, lo) = (p / 128, p % 128);
            let (g, l) = (lo / 32, lo % 32);
            let qli = half * 64 + (g & 1) * 32 + l;
            ql[qli] |= if g >= 2 { (q & 0x0F) << 4 } else { q & 0x0F };
            qh[half * 32 + l] |= (q >> 4) << (2 * g);
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(sc.iter().map(|&v| v as u8));
        out.extend_from_slice(&d.to_le_bytes());
    }
    out
}

/// Dequantizes `Q5_K` super-blocks (256 values / 176 bytes) into `out`.
pub fn dequantize_q5_k(src: &[u8], out: &mut [f32]) {
    for (block, y) in src.chunks_exact(176).zip(out.chunks_exact_mut(QK_K)) {
        let d = f16_at(block, 0);
        let dmin = f16_at(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let (mut u1, mut u2) = (1u8, 2u8);
        for (chunk, j) in (0..QK_K).step_by(64).enumerate() {
            let (sc1, m1) = scale_min_k4(2 * chunk, scales);
            let (sc2, m2) = scale_min_k4(2 * chunk + 1, scales);
            let ql = &qs[chunk * 32..chunk * 32 + 32];
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                y[j + l] = d * sc1 * ((ql[l] & 0x0F) + hi1) as f32 - dmin * m1;
                y[j + 32 + l] = d * sc2 * ((ql[l] >> 4) + hi2) as f32 - dmin * m2;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Dequantizes `Q6_K` super-blocks (256 values / 210 bytes) into `out`.
pub fn dequantize_q6_k(src: &[u8], out: &mut [f32]) {
    for (block, y) in src.chunks_exact(210).zip(out.chunks_exact_mut(QK_K)) {
        let ql_all = &block[0..128];
        let qh_all = &block[128..192];
        let sc_all = &block[192..208];
        let d = f16_at(block, 208);
        for half in 0..2 {
            let ql = &ql_all[half * 64..];
            let qh = &qh_all[half * 32..];
            let sc = &sc_all[half * 8..];
            let y = &mut y[half * 128..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

/// Dequantizes row-major data of any supported block format (or F16/BF16/F32 bytes)
/// into `out`. `Q4_0x4` is not row-major and is handled by [`super::repack`].
pub fn dequantize_into(dtype: rllm_core::dtype::DType, src: &[u8], out: &mut [f32]) -> rllm_core::error::Result<()> {
    use rllm_core::dtype::DType;
    match dtype {
        DType::F32 => {
            for (b, y) in src.chunks_exact(4).zip(out.iter_mut()) {
                *y = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        DType::F16 => convert_f16(src, out),
        DType::BF16 => convert_bf16(src, out),
        DType::Q8_0 => dequantize_q8_0(src, out),
        DType::Q4_0 => dequantize_q4_0(src, out),
        DType::Q5_0 => dequantize_q5_0(src, out),
        DType::Q4_K => dequantize_q4_k(src, out),
        DType::Q6_K => dequantize_q6_k(src, out),
        other => {
            return Err(rllm_core::error::RllmError::UnsupportedOp {
                op: "dequantize",
                dtype: other.to_string(),
                device: "cpu".to_string(),
            })
        }
    }
    Ok(())
}

/// Converts little-endian IEEE half-precision values to F32.
pub fn convert_f16(src: &[u8], out: &mut [f32]) {
    let n = (src.len() / 2).min(out.len());
    #[cfg(target_arch = "aarch64")]
    {
        // Eight values per FCVTL/FCVTL2 pair (the scalar loop below is ~8x slower and
        // made prefill with F16 weights 13% slower than with F32 weights).
        let main = n - n % 8;
        // SAFETY: `main` values are in bounds of both slices; unaligned loads/stores.
        unsafe { super::f16w::widen(src.as_ptr(), out.as_mut_ptr(), main) };
        for (b, y) in src[2 * main..2 * n].chunks_exact(2).zip(out[main..n].iter_mut()) {
            *y = f16::from_le_bytes([b[0], b[1]]).to_f32();
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (b, y) in src.chunks_exact(2).zip(out.iter_mut()) {
        *y = f16::from_le_bytes([b[0], b[1]]).to_f32();
    }
}

/// Converts little-endian bfloat16 values to F32.
pub fn convert_bf16(src: &[u8], out: &mut [f32]) {
    for (b, y) in src.chunks_exact(2).zip(out.iter_mut()) {
        *y = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
    }
}

/// Batched quantized linear: `out[M, N] = act[M, K] · W[N, K]ᵀ` for a `Q8_0` or `Q4_0`
/// weight. Parallelised over weight rows; each weight row is dequantized once into a
/// small buffer and reused for all `M` activation rows, so the weight matrix is streamed
/// from memory once regardless of `M`.
pub fn linear_quantized_f32(
    dtype_q8: bool,
    weight_bytes: &[u8],
    act: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let (block_elems, block_bytes) = if dtype_q8 { (QK8_0, 34) } else { (QK4_0, 18) };
    let row_bytes = (k / block_elems) * block_bytes;
    debug_assert_eq!(weight_bytes.len(), n * row_bytes);
    debug_assert_eq!(act.len(), m * k);
    debug_assert_eq!(out.len(), m * n);

    const ROWS_PER_TASK: usize = 8;
    // Compute the transposed result [N, M] so each task owns a contiguous output slab.
    let mut out_t = vec![0f32; n * m];
    out_t
        .par_chunks_mut(ROWS_PER_TASK * m)
        .enumerate()
        .for_each(|(task, slab)| {
            let mut w_row = vec![0f32; k];
            for (r, out_row) in slab.chunks_mut(m).enumerate() {
                let j = task * ROWS_PER_TASK + r;
                let bytes = &weight_bytes[j * row_bytes..(j + 1) * row_bytes];
                if dtype_q8 {
                    dequantize_q8_0(bytes, &mut w_row);
                } else {
                    dequantize_q4_0(bytes, &mut w_row);
                }
                for (i, o) in out_row.iter_mut().enumerate() {
                    *o = crate::ops::gemm::dot_f32(&w_row, &act[i * k..(i + 1) * k]);
                }
            }
        });
    for j in 0..n {
        for i in 0..m {
            out[i * n + j] = out_t[j * m + i];
        }
    }
}

#[cfg(test)]
mod dequant_tests {
    use super::*;

    #[test]
    fn q4_0_and_q8_0_dequant_match_dot_kernels() {
        let x: Vec<f32> = (0..64).map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.0).collect();
        let act: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let mut q4 = vec![0u8; 2 * 18];
        quantize_f32_to_q4_0(&x, &mut q4);
        let mut deq = vec![0f32; 64];
        dequantize_q4_0(&q4, &mut deq);
        let dense: f32 = deq.iter().zip(&act).map(|(a, b)| a * b).sum();
        assert!((dense - dot_q4_0_f32(&q4, &act)).abs() < 1e-4);
        // Quantisation error is bounded by half a step of the block scale.
        for (a, b) in x.iter().zip(&deq) {
            assert!((a - b).abs() <= 2.0 / 8.0 + 1e-3);
        }

        let mut q8 = vec![0u8; 2 * 34];
        for (b, chunk) in x.chunks(32).enumerate() {
            let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            q8[b * 34..b * 34 + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
            for (i, v) in chunk.iter().enumerate() {
                q8[b * 34 + 2 + i] = (v / d).round() as i8 as u8;
            }
        }
        dequantize_q8_0(&q8, &mut deq);
        let dense: f32 = deq.iter().zip(&act).map(|(a, b)| a * b).sum();
        assert!((dense - dot_q8_0_f32(&q8, &act)).abs() < 1e-4);
    }

    #[test]
    fn q5_0_to_q8_0_is_lossless() {
        let mut s = 7u64;
        let src: Vec<u8> = (0..22 * 5)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect();
        let mut src = src;
        for b in 0..5 {
            src[b * 22..b * 22 + 2].copy_from_slice(&f16::from_f32(0.01 * (b + 1) as f32).to_le_bytes());
        }
        let (mut a, mut b) = (vec![0f32; 160], vec![0f32; 160]);
        dequantize_q5_0(&src, &mut a);
        dequantize_q8_0(&q5_0_to_q8_0(&src), &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn k_quant_quantizers_round_trip() {
        let x: Vec<f32> = (0..512).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let mut y = vec![0f32; 512];
        dequantize_q4_k(&quantize_f32_to_q4_k(&x), &mut y);
        let e4 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(e4 < 0.2, "q4_k max error {e4}");
        dequantize_q6_k(&quantize_f32_to_q6_k(&x), &mut y);
        let e6 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(e6 < 0.05, "q6_k max error {e6}");
    }

    #[test]
    fn half_precision_conversions() {
        let mut out = [0f32; 2];
        convert_f16(&[0x00, 0x3C, 0x00, 0xC0], &mut out); // 1.0, -2.0
        assert_eq!(out, [1.0, -2.0]);
        convert_bf16(&[0x80, 0x3F, 0x40, 0xC0], &mut out); // 1.0, -3.0
        assert_eq!(out, [1.0, -3.0]);
    }
}

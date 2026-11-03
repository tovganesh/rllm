//! Direct quantized matrix-vector kernels (Q4_0, Q8_0) for edge and mobile execution.

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

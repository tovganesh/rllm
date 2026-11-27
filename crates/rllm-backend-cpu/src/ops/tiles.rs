//! `Q8_0` / `Q4_0` weights repacked into interleaved groups of eight rows
//! (`DType::Q8_0x8`, `DType::Q4_0x8`) for the x86 tile kernels in [`super::x86`].
//!
//! For every group of eight rows and every 32-value block the layout is
//!
//! ```text
//! [d_r0 .. d_r7]                                     8 × f16 scales          (16 bytes)
//! chunk 0: [r0 q[0..4]][r1 q[0..4]] .. [r7 q[0..4]]   4 bytes of each row     (32 bytes)
//! chunk 1: [r0 q[4..8]] ..                                                    (32 bytes)
//! ..
//! ```
//!
//! with 8 chunks for `Q8_0` (the 32 int8 values) and 4 for `Q4_0` (its 16 nibble bytes;
//! byte `i` holds values `i` (low nibble) and `16 + i` (high)). Blocks are the same size
//! as the source format's. One 256-bit load is a 4-byte chunk of all eight rows, so a
//! broadcast 4-byte chunk of activations yields eight row partial sums at once (as in
//! llama.cpp's x86 `q4_0_8x8` repacking).

#![allow(clippy::chunks_exact_to_as_chunks)]

#[cfg(target_arch = "x86_64")]
use super::qdot::quantize_activations_impl;
use super::qdot::QK;
use super::quant::{dequantize_q4_0, dequantize_q8_0};
use rayon::prelude::*;
use rllm_core::dtype::DType;

/// Rows per interleaved group.
pub const GROUP: usize = 8;

/// `(bytes per block, quant bytes per row and block)`.
fn geometry(dtype: DType) -> (usize, usize) {
    match dtype {
        DType::Q8_0 | DType::Q8_0x8 => (34, 32),
        DType::Q4_0 | DType::Q4_0x8 => (18, 16),
        other => panic!("no 8-row tile layout for {other}"),
    }
}

/// Repacks a row-major `Q8_0` or `Q4_0` matrix (`n % 8 == 0`, `k % 32 == 0`) into the
/// 8-row layout. Returns the packed bytes and dtype.
pub fn pack(dtype: DType, src: &[u8], n: usize, k: usize) -> (DType, Vec<u8>) {
    assert!(n.is_multiple_of(GROUP) && k.is_multiple_of(QK));
    let (bb, qb) = geometry(dtype);
    let nb = k / QK;
    let row_bytes = nb * bb;
    assert_eq!(src.len(), n * row_bytes);
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(GROUP * row_bytes).enumerate().for_each(|(g, dst)| {
        for b in 0..nb {
            let o = &mut dst[b * GROUP * bb..(b + 1) * GROUP * bb];
            for r in 0..GROUP {
                let blk = &src[(GROUP * g + r) * row_bytes + b * bb..][..bb];
                o[2 * r..2 * r + 2].copy_from_slice(&blk[..2]);
                for c in 0..qb / 4 {
                    o[16 + 32 * c + 4 * r..16 + 32 * c + 4 * r + 4].copy_from_slice(&blk[2 + 4 * c..2 + 4 * c + 4]);
                }
            }
        }
    });
    let packed = match dtype {
        DType::Q8_0 => DType::Q8_0x8,
        _ => DType::Q4_0x8,
    };
    (packed, out)
}

/// One row (`row`) of a packed `[n, k]` matrix in its source format (`Q8_0` / `Q4_0`).
fn unpack_row(dtype: DType, src: &[u8], k: usize, row: usize, out: &mut [u8]) {
    let (bb, qb) = geometry(dtype);
    let nb = k / QK;
    let grp = &src[(row / GROUP) * GROUP * nb * bb..];
    let r = row % GROUP;
    for b in 0..nb {
        let i = &grp[b * GROUP * bb..(b + 1) * GROUP * bb];
        let blk = &mut out[b * bb..(b + 1) * bb];
        blk[..2].copy_from_slice(&i[2 * r..2 * r + 2]);
        for c in 0..qb / 4 {
            blk[2 + 4 * c..2 + 4 * c + 4].copy_from_slice(&i[16 + 32 * c + 4 * r..16 + 32 * c + 4 * r + 4]);
        }
    }
}

/// Inverse of [`pack`]: the row-major source bytes and dtype.
pub fn unpack(dtype: DType, src: &[u8], n: usize, k: usize) -> (DType, Vec<u8>) {
    let (bb, _) = geometry(dtype);
    let row_bytes = k / QK * bb;
    let mut out = vec![0u8; n * row_bytes];
    out.par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(r, row)| unpack_row(dtype, src, k, r, row));
    let plain = match dtype {
        DType::Q8_0x8 => DType::Q8_0,
        _ => DType::Q4_0,
    };
    (plain, out)
}

/// Dequantizes row `row` of a packed `[n, k]` matrix into `out` (`k` values).
pub fn dequantize_row(dtype: DType, src: &[u8], k: usize, row: usize, out: &mut [f32]) {
    let (bb, _) = geometry(dtype);
    let mut tmp = vec![0u8; k / QK * bb];
    unpack_row(dtype, src, k, row, &mut tmp);
    match dtype {
        DType::Q8_0x8 => dequantize_q8_0(&tmp, out),
        _ => dequantize_q4_0(&tmp, out),
    }
}

/// Raw output pointer shared by tasks writing disjoint elements.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: tasks write disjoint output columns (see `linear`).
#[cfg(target_arch = "x86_64")]
unsafe impl Send for OutPtr {}
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for OutPtr {}
#[cfg(target_arch = "x86_64")]
impl OutPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Activation rows per tile (the kernels keep one accumulator per row).
#[cfg(target_arch = "x86_64")]
const TILE_ROWS: usize = 4;

/// `out[M, N] = X[M, K] · W[N, K]ᵀ` for a packed `Q8_0x8` / `Q4_0x8` weight. Requires
/// [`super::x86::has_avx2`] (the only kernels for these layouts).
#[cfg(target_arch = "x86_64")]
pub fn linear(dtype: DType, weight: &[u8], x: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(super::x86::has_avx2());
    let (bb, _) = geometry(dtype);
    let nb = k / QK;
    let group_bytes = GROUP * nb * bb;
    assert!(n.is_multiple_of(GROUP) && weight.len() == n / GROUP * group_bytes);
    assert!(x.len() == m * k && out.len() == m * n);
    let is_q8 = dtype == DType::Q8_0x8;
    // Q4_0 kernels take the `-8` offset from per-block activation sums.
    let a = quantize_activations_impl(x, !is_q8);
    let groups = n / GROUP;
    // ~64 KB of weights per task, at least two tasks per thread; large batches walk the
    // activations in ~128 KB blocks so both stay in cache.
    let min_tasks = 2 * crate::threads::decode_threads();
    let per_task = (64 * 1024 / group_bytes).clamp(1, groups.div_ceil(min_tasks).max(1));
    let act_block = (128 * 1024 / (k + 8 * nb)).max(TILE_ROWS) / TILE_ROWS * TILE_ROWS;
    let tasks = groups.div_ceil(per_task);
    let ptr = OutPtr(out.as_mut_ptr());
    crate::pool::for_each_index(tasks, |t| {
        let (g0, g1) = (t * per_task, ((t + 1) * per_task).min(groups));
        for i0 in (0..m).step_by(act_block) {
            let i1 = (i0 + act_block).min(m);
            for g in g0..g1 {
                let grp = &weight[g * group_bytes..(g + 1) * group_bytes];
                let mut i = i0;
                while i < i1 {
                    let rows = (i1 - i).min(TILE_ROWS);
                    let at = OutPtr(ptr.get().wrapping_add(i * n + g * GROUP));
                    match rows {
                        4 => tile::<4>(is_q8, grp, &a, i, nb, at, n),
                        3 => tile::<3>(is_q8, grp, &a, i, nb, at, n),
                        2 => tile::<2>(is_q8, grp, &a, i, nb, at, n),
                        _ => tile::<1>(is_q8, grp, &a, i, nb, at, n),
                    }
                    i += rows;
                }
            }
        }
    });
}

/// One tile: the group's eight weight rows against activation rows `i..i + R`, written
/// to `at + r * n` (eight floats per row).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn tile<const R: usize>(
    is_q8: bool,
    grp: &[u8],
    a: &super::qdot::Q8Activations,
    i: usize,
    nb: usize,
    at: OutPtr,
    n: usize,
) {
    let rows: [usize; R] = std::array::from_fn(|r| i + r);
    let mut res = [[0f32; GROUP]; R];
    // SAFETY: AVX2 checked by the caller; the kernels assert their slice bounds.
    unsafe {
        if is_q8 {
            super::x86::tile_q8_0x8::<R>(grp, a, rows, nb, &mut res);
        } else {
            super::x86::tile_q4_0x8::<R>(grp, a, rows, nb, &mut res);
        }
    }
    for (r, v) in res.iter().enumerate() {
        // SAFETY: `at + r * n .. + 8` lies in the output (row i + r, columns of this
        // group); each such span is written by exactly one task.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), at.get().add(r * n), GROUP) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::quant::{quantize_f32_to_q4_0, quantize_f32_to_q8_0};

    fn data(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn weights(dtype: DType, wf: &[f32]) -> Vec<u8> {
        let mut w = vec![0u8; dtype.storage_bytes(wf.len())];
        match dtype {
            DType::Q8_0 => quantize_f32_to_q8_0(wf, &mut w),
            _ => quantize_f32_to_q4_0(wf, &mut w),
        }
        w
    }

    #[test]
    fn pack_round_trips_and_rows_dequantize() {
        let (n, k) = (24, 96);
        let wf = data(n * k, 3);
        for dtype in [DType::Q8_0, DType::Q4_0] {
            let w = weights(dtype, &wf);
            let (packed, p) = pack(dtype, &w, n, k);
            assert_eq!(p.len(), w.len());
            assert_eq!(unpack(packed, &p, n, k), (dtype, w.clone()));
            let mut full = vec![0f32; n * k];
            crate::ops::quant::dequantize_into(dtype, &w, &mut full).unwrap();
            let mut row = vec![0f32; k];
            for r in [0, 7, 8, 13, 23] {
                dequantize_row(packed, &p, k, r, &mut row);
                assert_eq!(row, full[r * k..(r + 1) * k], "{dtype} row {r}");
            }
        }
    }

    /// The tile kernels against the row-major integer path, for decode (1 row), tile
    /// remainders and a batch spanning two activation blocks (300 rows at k = 512).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn tile_kernels_match_row_major_kernels() {
        if !crate::ops::x86::has_avx2() {
            eprintln!("skipping: no AVX2");
            return;
        }
        use crate::ops::qdot::{linear_quantized, QFormat};
        let (n, k) = (40, 512);
        let wf = data(n * k, 5);
        for (dtype, fmt) in [(DType::Q8_0, QFormat::Q8_0), (DType::Q4_0, QFormat::Q4_0)] {
            let w = weights(dtype, &wf);
            let (packed, p) = pack(dtype, &w, n, k);
            for m in [1, 2, 3, 5, 9, 300] {
                let x = data(m * k, 7 + m as u32);
                let mut want = vec![0f32; m * n];
                linear_quantized(fmt, &w, &x, &mut want, m, n, k);
                let mut got = vec![0f32; m * n];
                linear(packed, &p, &x, &mut got, m, n, k);
                for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                        "{dtype} m={m} [{}, {}]: {g} vs {w}",
                        idx / n,
                        idx % n
                    );
                }
            }
        }
    }
}

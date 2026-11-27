//! Weights repacked into interleaved groups of eight rows for the x86 tile kernels in
//! [`super::x86`] (`DType::Q8_0x8`, `Q4_0x8`, `Q4_Kx8`, `Q6_Kx8`).
//!
//! The quantized values of a group are stored in 32-byte chunks holding four
//! consecutive bytes of each of the eight rows, so one 256-bit load is the same four
//! values of all eight rows, and a broadcast 4-byte chunk of activations gives eight row
//! partial sums per `maddubs` (as in llama.cpp's x86 `q4_0_8x8` / `q4_K_8x8` repacking).
//! Per 32-value block (`Q8_0x8`, `Q4_0x8`):
//!
//! ```text
//! [d_r0 .. d_r7]                                     8 × f16 scales          (16 bytes)
//! chunk 0: [r0 q[0..4]][r1 q[0..4]] .. [r7 q[0..4]]   4 bytes of each row     (32 bytes)
//! chunk 1: [r0 q[4..8]] ..                                                    (32 bytes)
//! ..
//! ```
//!
//! with 8 chunks for `Q8_0` (the 32 int8 values) and 4 for `Q4_0` (its 16 nibble bytes;
//! byte `i` holds values `i` (low nibble) and `16 + i` (high)). Per 256-value
//! super-block (`Q4_Kx8`, 1,184 bytes): `d[8]`, `dmin[8]` (f16), the 6-bit sub-block
//! scales and mins unpacked to bytes as `[sub-block][row]` (64 + 64), then 32 chunks of
//! `qs`. `Q6_Kx8` (1,680 bytes): `d[8]`, the int8 sub-block scales as `[sub-block][row]`
//! (128), 32 chunks of `ql`, 16 chunks of `qh`. All but `Q4_Kx8` keep the source size.

#![allow(clippy::chunks_exact_to_as_chunks)]

use super::kdot::q4_k_scales_mins;
#[cfg(target_arch = "x86_64")]
use super::kdot::quantize_q8_k;
#[cfg(target_arch = "x86_64")]
use super::qdot::{quantize_activations_impl, QK};
use super::quant::{dequantize_q4_0, dequantize_q4_k, dequantize_q6_k, dequantize_q8_0};
use rayon::prelude::*;
use rllm_core::dtype::DType;

/// Rows per interleaved group.
pub const GROUP: usize = 8;

/// The packed dtype for a source format, or `None` if it has no tile layout.
pub fn packed_dtype(dtype: DType) -> Option<DType> {
    Some(match dtype {
        DType::Q8_0 => DType::Q8_0x8,
        DType::Q4_0 => DType::Q4_0x8,
        DType::Q4_K => DType::Q4_Kx8,
        DType::Q6_K => DType::Q6_Kx8,
        _ => return None,
    })
}

/// The source format of a packed dtype.
fn source_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Q8_0x8 => DType::Q8_0,
        DType::Q4_0x8 => DType::Q4_0,
        DType::Q4_Kx8 => DType::Q4_K,
        DType::Q6_Kx8 => DType::Q6_K,
        other => panic!("{other} is not an 8-row tile layout"),
    }
}

/// Copies 4-byte chunks: row `r`'s bytes `src[4c..4c + 4]` to `dst[32c + 4r..]`
/// (`pack == true`), or back.
fn chunks(dst: &mut [u8], src: &mut [u8], r: usize, n_chunks: usize, pack: bool) {
    for c in 0..n_chunks {
        let (t, s) = (&mut dst[32 * c + 4 * r..32 * c + 4 * r + 4], &mut src[4 * c..4 * c + 4]);
        if pack {
            t.copy_from_slice(s);
        } else {
            s.copy_from_slice(t);
        }
    }
}

/// Inverse of [`q4_k_scales_mins`]: the 12 packed bytes of eight 6-bit scales and mins.
fn encode_q4_k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for j in 0..4 {
        q[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6);
        q[j + 4] = (mn[j] & 63) | ((mn[j + 4] >> 4) << 6);
        q[j + 8] = (sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4);
    }
    q
}

/// Converts row `r`'s source block `blk` into / out of a packed group block `grp`
/// (`pack` selects the direction). `blk` is one block of the source format.
fn convert_block(src_dtype: DType, grp: &mut [u8], blk: &mut [u8], r: usize, pack: bool) {
    // grp[t] <- blk[s] (pack) or blk[s] <- grp[t].
    let copy = |grp: &mut [u8], blk: &mut [u8], t: std::ops::Range<usize>, s: std::ops::Range<usize>| {
        if pack {
            grp[t].copy_from_slice(&blk[s]);
        } else {
            blk[s].copy_from_slice(&grp[t]);
        }
    };
    match src_dtype {
        DType::Q8_0 | DType::Q4_0 => {
            copy(grp, blk, 2 * r..2 * r + 2, 0..2);
            let n_chunks = if src_dtype == DType::Q8_0 { 8 } else { 4 };
            chunks(&mut grp[16..], &mut blk[2..], r, n_chunks, pack);
        }
        DType::Q4_K => {
            copy(grp, blk, 2 * r..2 * r + 2, 0..2);
            copy(grp, blk, 16 + 2 * r..16 + 2 * r + 2, 2..4);
            if pack {
                let (sc, mn) = q4_k_scales_mins(&blk[4..16]);
                for j in 0..8 {
                    grp[32 + 8 * j + r] = sc[j];
                    grp[96 + 8 * j + r] = mn[j];
                }
            } else {
                let sc: [u8; 8] = std::array::from_fn(|j| grp[32 + 8 * j + r]);
                let mn: [u8; 8] = std::array::from_fn(|j| grp[96 + 8 * j + r]);
                blk[4..16].copy_from_slice(&encode_q4_k_scales(&sc, &mn));
            }
            chunks(&mut grp[160..], &mut blk[16..], r, 32, pack);
        }
        DType::Q6_K => {
            copy(grp, blk, 2 * r..2 * r + 2, 208..210);
            for j in 0..16 {
                copy(grp, blk, 16 + 8 * j + r..16 + 8 * j + r + 1, 192 + j..193 + j);
            }
            chunks(&mut grp[144..], &mut blk[..128], r, 32, pack);
            chunks(&mut grp[1168..], &mut blk[128..192], r, 16, pack);
        }
        other => panic!("no 8-row tile layout for {other}"),
    }
}

/// `(values per block, source bytes per block, packed bytes per row and block)`.
fn geometry(src_dtype: DType) -> (usize, usize, usize) {
    let (bv, bb) = src_dtype.block_info().expect("block format");
    let packed = packed_dtype(src_dtype).expect("tile layout").block_info().unwrap().1;
    (bv, bb, packed)
}

/// Repacks a row-major `Q8_0`, `Q4_0`, `Q4_K` or `Q6_K` matrix (`n % 8 == 0`, `k` a
/// multiple of the block) into the 8-row layout. Returns the packed dtype and bytes.
pub fn pack(dtype: DType, src: &[u8], n: usize, k: usize) -> (DType, Vec<u8>) {
    let (bv, bb, pb) = geometry(dtype);
    assert!(n.is_multiple_of(GROUP) && k.is_multiple_of(bv));
    let nb = k / bv;
    let row_bytes = nb * bb;
    assert_eq!(src.len(), n * row_bytes);
    let mut out = vec![0u8; n * nb * pb];
    out.par_chunks_mut(GROUP * nb * pb).enumerate().for_each(|(g, dst)| {
        let mut blk = vec![0u8; bb];
        for b in 0..nb {
            let o = &mut dst[b * GROUP * pb..(b + 1) * GROUP * pb];
            for r in 0..GROUP {
                blk.copy_from_slice(&src[(GROUP * g + r) * row_bytes + b * bb..][..bb]);
                convert_block(dtype, o, &mut blk, r, true);
            }
        }
    });
    (packed_dtype(dtype).unwrap(), out)
}

/// One row (`row`) of a packed `[n, k]` matrix in its source format.
fn unpack_row(dtype: DType, src: &[u8], k: usize, row: usize, out: &mut [u8]) {
    let src_dtype = source_dtype(dtype);
    let (bv, bb, pb) = geometry(src_dtype);
    let nb = k / bv;
    let start = (row / GROUP) * GROUP * nb * pb;
    let mut grp = vec![0u8; GROUP * pb];
    for b in 0..nb {
        grp.copy_from_slice(&src[start + b * GROUP * pb..start + (b + 1) * GROUP * pb]);
        convert_block(src_dtype, &mut grp, &mut out[b * bb..(b + 1) * bb], row % GROUP, false);
    }
}

/// Inverse of [`pack`]: the source dtype and row-major bytes.
pub fn unpack(dtype: DType, src: &[u8], n: usize, k: usize) -> (DType, Vec<u8>) {
    let src_dtype = source_dtype(dtype);
    let row_bytes = src_dtype.storage_bytes(k);
    let mut out = vec![0u8; n * row_bytes];
    out.par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(r, row)| unpack_row(dtype, src, k, r, row));
    (src_dtype, out)
}

/// Dequantizes row `row` of a packed `[n, k]` matrix into `out` (`k` values).
pub fn dequantize_row(dtype: DType, src: &[u8], k: usize, row: usize, out: &mut [f32]) {
    let src_dtype = source_dtype(dtype);
    let mut tmp = vec![0u8; src_dtype.storage_bytes(k)];
    unpack_row(dtype, src, k, row, &mut tmp);
    match src_dtype {
        DType::Q8_0 => dequantize_q8_0(&tmp, out),
        DType::Q4_0 => dequantize_q4_0(&tmp, out),
        DType::Q4_K => dequantize_q4_k(&tmp, out),
        _ => dequantize_q6_k(&tmp, out),
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

/// Quantized activations for a packed format.
#[cfg(target_arch = "x86_64")]
enum Acts {
    Q8(super::qdot::Q8Activations),
    K(super::kdot::Q8KActivations),
}

/// `out[M, N] = X[M, K] · W[N, K]ᵀ` for a weight in an 8-row tile layout. Requires
/// [`super::x86::has_avx2`] (the only kernels for these layouts).
#[cfg(target_arch = "x86_64")]
pub fn linear(dtype: DType, weight: &[u8], x: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(super::x86::has_avx2());
    let (bv, _, pb) = geometry(source_dtype(dtype));
    let nb = k / bv;
    let group_bytes = GROUP * nb * pb;
    assert!(n.is_multiple_of(GROUP) && weight.len() == n / GROUP * group_bytes);
    assert!(x.len() == m * k && out.len() == m * n);
    let a = match dtype {
        // Q4_0 kernels take the `-8` offset from per-block activation sums.
        DType::Q8_0x8 | DType::Q4_0x8 => Acts::Q8(quantize_activations_impl(x, dtype == DType::Q4_0x8)),
        _ => Acts::K(quantize_q8_k(x)),
    };
    let groups = n / GROUP;
    // ~64 KB of weights per task, at least two tasks per thread; large batches walk the
    // activations in ~128 KB blocks so both stay in cache.
    let min_tasks = 2 * crate::threads::decode_threads();
    let per_task = (64 * 1024 / group_bytes).clamp(1, groups.div_ceil(min_tasks).max(1));
    let act_block = (128 * 1024 / (k + 8 * k / QK)).max(TILE_ROWS) / TILE_ROWS * TILE_ROWS;
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
                        4 => tile::<4>(dtype, grp, &a, i, nb, at, n),
                        3 => tile::<3>(dtype, grp, &a, i, nb, at, n),
                        2 => tile::<2>(dtype, grp, &a, i, nb, at, n),
                        _ => tile::<1>(dtype, grp, &a, i, nb, at, n),
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
fn tile<const R: usize>(dtype: DType, grp: &[u8], a: &Acts, i: usize, nb: usize, at: OutPtr, n: usize) {
    use super::x86;
    let rows: [usize; R] = std::array::from_fn(|r| i + r);
    let mut res = [[0f32; GROUP]; R];
    // SAFETY: AVX2 checked by `linear`; the kernels assert their slice bounds.
    unsafe {
        match (dtype, a) {
            (DType::Q8_0x8, Acts::Q8(a)) => x86::tile_q8_0x8::<R>(grp, a, rows, nb, &mut res),
            (DType::Q4_0x8, Acts::Q8(a)) => x86::tile_q4_0x8::<R>(grp, a, rows, nb, &mut res),
            (DType::Q4_Kx8, Acts::K(a)) => x86::tile_q4_kx8::<R>(grp, a, rows, nb, &mut res),
            (DType::Q6_Kx8, Acts::K(a)) => x86::tile_q6_kx8::<R>(grp, a, rows, nb, &mut res),
            _ => unreachable!("activations do not match {dtype}"),
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
            DType::Q4_0 => quantize_f32_to_q4_0(wf, &mut w),
            DType::Q4_K => w = crate::ops::quant::quantize_f32_to_q4_k(wf),
            _ => w = crate::ops::quant::quantize_f32_to_q6_k(wf),
        }
        w
    }

    #[test]
    fn pack_round_trips_and_rows_dequantize() {
        let (n, k) = (24, 512);
        let wf = data(n * k, 3);
        for dtype in [DType::Q8_0, DType::Q4_0, DType::Q4_K, DType::Q6_K] {
            let w = weights(dtype, &wf);
            let (packed, p) = pack(dtype, &w, n, k);
            assert_eq!(p.len(), packed.storage_bytes(n * k));
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
        for (dtype, fmt) in [
            (DType::Q8_0, QFormat::Q8_0),
            (DType::Q4_0, QFormat::Q4_0),
            (DType::Q4_K, QFormat::Q4K),
            (DType::Q6_K, QFormat::Q6K),
        ] {
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

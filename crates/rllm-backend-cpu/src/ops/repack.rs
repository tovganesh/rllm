//! `Q4_0` weights repacked into interleaved groups of four rows (`DType::Q4_0x4`).
//!
//! For every group of four rows and every 32-value block the layout is
//!
//! ```text
//! [d_r0 d_r1 d_r2 d_r3]                    4 × f16 scales             (8 bytes)
//! [r0 q[0..8]][r1 q[0..8]][r2 ..][r3 ..]    bytes 0..8 of each row     (32 bytes)
//! [r0 q[8..16]][r1 q[8..16]][r2 ..][r3 ..]  bytes 8..16 of each row    (32 bytes)
//! ```
//!
//! — 72 bytes, exactly four `Q4_0` blocks. Each nibble is stored XOR 8, i.e. as a signed
//! 4-bit value `q - 8`, so `x << 4` (low nibble) and `x & 0xF0` (high nibble) give
//! `16·(q - 8)` as int8 without a subtraction; the factor 16 is folded into the scale.
//!
//! A 16-byte load of one chunk holds two rows × eight values, which is both two SDOT
//! lanes pairs (decode) and exactly the 2×8 operand of the i8mm `SMMLA` instruction
//! (batched rows), as in llama.cpp's `q4_0_4x8` repacking.

#![allow(clippy::chunks_exact_to_as_chunks)]

use super::qdot::{quantize_activations, quantize_activations_impl, Q8Activations, QK};
use half::f16;
use rayon::prelude::*;

const Q4_0_BYTES: usize = 18;
const Q8_0_BYTES: usize = 34;
/// Bytes of one interleaved 4-row block.
pub const GROUP_BLOCK_BYTES: usize = 4 * Q4_0_BYTES;

/// Repacks a row-major `Q4_0` matrix of `n` rows (`n % 4 == 0`) and `k` columns.
pub fn repack_q4_0(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(n.is_multiple_of(4) && k.is_multiple_of(QK));
    let nb = k / QK;
    let row_bytes = nb * Q4_0_BYTES;
    assert_eq!(src.len(), n * row_bytes);
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(4 * row_bytes).enumerate().for_each(|(g, dst)| {
        for b in 0..nb {
            let o = &mut dst[b * GROUP_BLOCK_BYTES..(b + 1) * GROUP_BLOCK_BYTES];
            for r in 0..4 {
                let blk = &src[(4 * g + r) * row_bytes + b * Q4_0_BYTES..][..Q4_0_BYTES];
                o[2 * r..2 * r + 2].copy_from_slice(&blk[..2]);
                for c in 0..2 {
                    for t in 0..8 {
                        o[8 + c * 32 + r * 8 + t] = blk[2 + c * 8 + t] ^ 0x88;
                    }
                }
            }
        }
    });
    out
}

/// Inverse of [`repack_q4_0`].
pub fn unpack_q4_0x4(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    let nb = k / QK;
    let row_bytes = nb * Q4_0_BYTES;
    assert_eq!(src.len(), n * row_bytes);
    let mut out = vec![0u8; src.len()];
    for (g, grp) in src.chunks_exact(4 * row_bytes).enumerate() {
        for b in 0..nb {
            let i = &grp[b * GROUP_BLOCK_BYTES..(b + 1) * GROUP_BLOCK_BYTES];
            for r in 0..4 {
                let blk = &mut out[(4 * g + r) * row_bytes + b * Q4_0_BYTES..][..Q4_0_BYTES];
                blk[..2].copy_from_slice(&i[2 * r..2 * r + 2]);
                for c in 0..2 {
                    for t in 0..8 {
                        blk[2 + c * 8 + t] = i[8 + c * 32 + r * 8 + t] ^ 0x88;
                    }
                }
            }
        }
    }
    out
}

/// Dequantizes one 4-row group (`nb` interleaved blocks) into four rows of `32 * nb`.
pub fn dequantize_q4_0x4_group(src: &[u8], out: [&mut [f32]; 4]) {
    let nb = src.len() / GROUP_BLOCK_BYTES;
    let [o0, o1, o2, o3] = out;
    let rows = [o0, o1, o2, o3];
    for (r, y) in rows.into_iter().enumerate() {
        for b in 0..nb {
            let blk = &src[b * GROUP_BLOCK_BYTES..(b + 1) * GROUP_BLOCK_BYTES];
            let d = f16::from_le_bytes([blk[2 * r], blk[2 * r + 1]]).to_f32();
            let y = &mut y[b * QK..(b + 1) * QK];
            for c in 0..2 {
                for t in 0..8 {
                    let q = blk[8 + c * 32 + r * 8 + t] as i8;
                    // Signed nibbles: low = (q << 4) >> 4, high = q >> 4.
                    y[c * 8 + t] = ((q << 4) >> 4) as f32 * d;
                    y[16 + c * 8 + t] = (q >> 4) as f32 * d;
                }
            }
        }
    }
}

/// Dequantizes row `row` of a repacked `[n, k]` matrix into `out` (`k` values).
pub fn dequantize_q4_0x4_row(src: &[u8], k: usize, row: usize, out: &mut [f32]) {
    let nb = k / QK;
    let grp = &src[(row / 4) * nb * GROUP_BLOCK_BYTES..(row / 4 + 1) * nb * GROUP_BLOCK_BYTES];
    let r = row % 4;
    for b in 0..nb {
        let blk = &grp[b * GROUP_BLOCK_BYTES..(b + 1) * GROUP_BLOCK_BYTES];
        let d = f16::from_le_bytes([blk[2 * r], blk[2 * r + 1]]).to_f32();
        for c in 0..2 {
            for t in 0..8 {
                let q = blk[8 + c * 32 + r * 8 + t] as i8;
                out[b * QK + c * 8 + t] = ((q << 4) >> 4) as f32 * d;
                out[b * QK + 16 + c * 8 + t] = (q >> 4) as f32 * d;
            }
        }
    }
}

/// Portable reference: the four dot products of one group with activation blocks
/// `off..off + nb`.
pub fn scalar_dot_q4_0x4(grp: &[u8], a: &Q8Activations, off: usize, nb: usize) -> [f32; 4] {
    let mut out = [0f32; 4];
    for b in 0..nb {
        let blk = &grp[b * GROUP_BLOCK_BYTES..(b + 1) * GROUP_BLOCK_BYTES];
        let aq = &a.qs[(off + b) * QK..(off + b + 1) * QK];
        for (r, o) in out.iter_mut().enumerate() {
            let mut isum = 0i32;
            for c in 0..2 {
                for t in 0..8 {
                    let q = blk[8 + c * 32 + r * 8 + t] as i8;
                    isum += ((q << 4) >> 4) as i32 * aq[c * 8 + t] as i32 + (q >> 4) as i32 * aq[16 + c * 8 + t] as i32;
                }
            }
            *o += isum as f32 * f16::from_le_bytes([blk[2 * r], blk[2 * r + 1]]).to_f32() * a.d[off + b];
        }
    }
    out
}

fn has_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Whether the int8 matrix-multiply (`SMMLA`) path is available.
pub fn has_i8mm() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::arch::is_aarch64_feature_detected!("i8mm")
                && std::arch::is_aarch64_feature_detected!("dotprod")
                && std::env::var("RLLM_NO_I8MM").is_err()
        })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Four dot products of one group with activation blocks `off..off + nb`.
#[inline]
pub fn vec_dot_q4_0x4(grp: &[u8], a: &Q8Activations, off: usize, nb: usize) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: dotprod detected at runtime; lengths asserted inside.
        return unsafe { neon::dot_q4_0x4(grp, a, off, nb) };
    }
    scalar_dot_q4_0x4(grp, a, off, nb)
}

/// Batches below this many rows run on the low-latency decode pool.
const SMALL_BATCH_ROWS: usize = 32;

/// Raw pointer that tasks writing disjoint output elements may share.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: every task writes a disjoint set of elements (distinct column groups).
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    /// Going through a method makes closures capture the whole (Send) wrapper rather
    /// than the raw pointer field.
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// `out[N] = W[N, K] · x[K]` for a repacked weight.
pub fn gemv_q4_0x4(weight: &[u8], x: &[f32], out: &mut [f32], n: usize, k: usize) {
    let nb = k / QK;
    let group_bytes = nb * GROUP_BLOCK_BYTES;
    debug_assert_eq!(weight.len(), n / 4 * group_bytes);
    let a = quantize_activations(x);
    let groups_per_task = (super::gemv_rows_per_task(nb * Q4_0_BYTES) / 4).max(2);
    // Decode path: run on the low-latency spinning pool (see `crate::pool`).
    crate::pool::for_each_chunk_mut(out, 4 * groups_per_task, |t, chunk| {
        for (g, o) in chunk.chunks_exact_mut(4).enumerate() {
            let gi = t * groups_per_task + g;
            o.copy_from_slice(&vec_dot_q4_0x4(&weight[gi * group_bytes..(gi + 1) * group_bytes], &a, 0, nb));
        }
    });
}

/// `out[M, N] = X[M, K] · W[N, K]ᵀ` for a repacked weight, using `SMMLA` when the CPU
/// has i8mm and per-row SDOT otherwise.
pub fn gemm_q4_0x4(weight: &[u8], x: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    let nb = k / QK;
    let group_bytes = nb * GROUP_BLOCK_BYTES;
    debug_assert_eq!(weight.len(), n / 4 * group_bytes);
    debug_assert_eq!(x.len(), m * k);
    let a = quantize_activations(x);
    let groups = n / 4;
    let ptr = OutPtr(out.as_mut_ptr());

    #[cfg(target_arch = "aarch64")]
    if has_i8mm() {
        let packed = pack_pairs(&a, m, nb);
        // Enough tasks for load balance; each task's weights stay in cache while it
        // walks all activation rows.
        let threads = rayon::current_num_threads().max(1);
        let groups_per_task = groups.div_ceil(threads * 4).max(1);
        let task = |t: usize| {
            let g0 = t * groups_per_task;
            let g1 = (g0 + groups_per_task).min(groups);
            // SAFETY: i8mm detected; the task writes only columns 4*g0..4*g1.
            unsafe { neon::gemm_i8mm_task(weight, group_bytes, &packed, m, n, nb, g0, g1, ptr.get()) };
        };
        // Small batches (decode) on the spinning pool; prefill-sized ones on Rayon.
        let tasks = groups.div_ceil(groups_per_task);
        if m < SMALL_BATCH_ROWS {
            crate::pool::for_each_index(tasks, task);
        } else {
            (0..tasks).into_par_iter().for_each(task);
        }
        return;
    }

    let groups_per_task = (super::gemv_rows_per_task(nb * Q4_0_BYTES) / (4 * m)).max(1);
    crate::pool::for_each_index(groups.div_ceil(groups_per_task), |t| {
        for g in t * groups_per_task..((t + 1) * groups_per_task).min(groups) {
            let grp = &weight[g * group_bytes..(g + 1) * group_bytes];
            for i in 0..m {
                let v = vec_dot_q4_0x4(grp, &a, i * nb, nb);
                // SAFETY: column group g belongs to this task only; i < m, 4g+3 < n.
                unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), ptr.get().add(i * n + 4 * g), 4) };
            }
        }
    });
}

/// `out[M, N] = X[M, K] · W[N, K]ᵀ` for a row-major `Q8_0` weight using `SMMLA`, or
/// `None` without i8mm. Pairs of weight rows are interleaved on the fly (64-bit zips),
/// so no repacked copy of the weight is needed. Rows past the last multiple of four
/// are computed with the SDOT multi-row kernel.
pub fn gemm_q8_0_i8mm(weight: &[u8], x: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) -> Option<()> {
    if !has_i8mm() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let nb = k / QK;
        let row_bytes = nb * Q8_0_BYTES;
        debug_assert_eq!(weight.len(), n * row_bytes);
        debug_assert_eq!(x.len(), m * k);
        let a = quantize_activations_impl(x, false);
        let packed = pack_pairs(&a, m, nb);
        let groups = n / 4;
        let ptr = OutPtr(out.as_mut_ptr());
        let groups_per_task = (super::gemv_rows_per_task(row_bytes) / (4 * m.div_ceil(4))).max(1);
        let task = |t: usize| {
            let g0 = t * groups_per_task;
            let g1 = (g0 + groups_per_task).min(groups);
            // SAFETY: i8mm detected; the task writes only columns 4*g0..4*g1.
            unsafe { neon::gemm_q8_0_i8mm_task(weight, row_bytes, &packed, m, n, nb, g0, g1, ptr.get()) };
        };
        let tasks = groups.div_ceil(groups_per_task);
        if m < SMALL_BATCH_ROWS {
            crate::pool::for_each_index(tasks, task);
        } else {
            (0..tasks).into_par_iter().for_each(task);
        }
        for j in 4 * groups..n {
            let w = &weight[j * row_bytes..(j + 1) * row_bytes];
            for i in 0..m {
                out[i * n + j] = super::qdot::vec_dot_q8_0_q8(w, &a, i * nb, nb);
            }
        }
        Some(())
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (weight, x, out, m, n, k);
        None
    }
}

/// Activations of rows `2p, 2p+1` interleaved in 8-byte chunks per block (the `SMMLA`
/// left operand), with per-lane scales `[d0, d0, d1, d1] / 16`. An odd last row is
/// paired with zeros.
struct PackedPairs {
    qs: Vec<i8>,
    scales: Vec<f32>,
}

fn pack_pairs(a: &Q8Activations, m: usize, nb: usize) -> PackedPairs {
    let pairs = m.div_ceil(2);
    let mut qs = vec![0i8; pairs * nb * 64];
    let mut scales = vec![0f32; pairs * nb * 4];
    for p in 0..pairs {
        for b in 0..nb {
            let o = &mut qs[(p * nb + b) * 64..][..64];
            let s = &mut scales[(p * nb + b) * 4..][..4];
            for h in 0..2 {
                let row = 2 * p + h;
                if row >= m {
                    continue;
                }
                let src = &a.qs[(row * nb + b) * QK..][..QK];
                for c in 0..4 {
                    o[c * 16 + h * 8..c * 16 + h * 8 + 8].copy_from_slice(&src[c * 8..c * 8 + 8]);
                }
                let d = a.d[row * nb + b] / 16.0;
                s[2 * h] = d;
                s[2 * h + 1] = d;
            }
        }
    }
    PackedPairs { qs, scales }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    use std::arch::asm;

    #[inline(always)]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
        acc
    }

    /// `acc(2×2) += a(2×8) · b(2×8)ᵀ` on int8 rows; lanes are `[a0·b0, a0·b1, a1·b0, a1·b1]`.
    #[inline(always)]
    unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
        acc
    }

    /// Four little-endian f16 values at `p` as f32 (`FCVTL`; f16 vector types are not
    /// stable in Rust).
    #[inline(always)]
    unsafe fn f16x4(p: *const u8) -> float32x4_t {
        let h = vld1_u8(p);
        let out: float32x4_t;
        asm!("fcvtl {o:v}.4s, {i:v}.4h", o = out(vreg) out, i = in(vreg) h, options(pure, nomem, nostack));
        out
    }

    /// Four f16 bit patterns as f32 (`FCVTL`).
    #[inline(always)]
    unsafe fn f16x4_from(h: uint16x4_t) -> float32x4_t {
        let out: float32x4_t;
        asm!("fcvtl {o:v}.4s, {i:v}.4h", o = out(vreg) out, i = in(vreg) h, options(pure, nomem, nostack));
        out
    }

    #[inline(always)]
    unsafe fn dup8(p: *const i8) -> int8x16_t {
        let h = vld1_s8(p);
        vcombine_s8(h, h)
    }

    /// One block of one group against one activation block: `[r0, r1, r2, r3]` int sums
    /// (scaled by 16).
    #[inline(always)]
    unsafe fn group_block(blk: *const u8, a: *const i8) -> int32x4_t {
        let hi_mask = vdupq_n_s8(0xF0u8 as i8);
        let c0_01 = vld1q_s8(blk.add(8) as *const i8);
        let c0_23 = vld1q_s8(blk.add(24) as *const i8);
        let c1_01 = vld1q_s8(blk.add(40) as *const i8);
        let c1_23 = vld1q_s8(blk.add(56) as *const i8);
        let (a0, a1, a2, a3) = (dup8(a), dup8(a.add(8)), dup8(a.add(16)), dup8(a.add(24)));
        let mut s01 = sdot(vdupq_n_s32(0), vshlq_n_s8::<4>(c0_01), a0);
        let mut s23 = sdot(vdupq_n_s32(0), vshlq_n_s8::<4>(c0_23), a0);
        s01 = sdot(s01, vshlq_n_s8::<4>(c1_01), a1);
        s23 = sdot(s23, vshlq_n_s8::<4>(c1_23), a1);
        s01 = sdot(s01, vandq_s8(c0_01, hi_mask), a2);
        s23 = sdot(s23, vandq_s8(c0_23, hi_mask), a2);
        s01 = sdot(s01, vandq_s8(c1_01, hi_mask), a3);
        s23 = sdot(s23, vandq_s8(c1_23, hi_mask), a3);
        vpaddq_s32(s01, s23)
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q4_0x4(grp: &[u8], a: &Q8Activations, off: usize, nb: usize) -> [f32; 4] {
        assert!(grp.len() >= nb * GROUP_BLOCK_BYTES && a.qs.len() >= (off + nb) * QK);
        let wp = grp.as_ptr();
        let ap = a.qs.as_ptr().add(off * QK);
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut b = 0;
        while b + 1 < nb {
            let (p0, p1) = (wp.add(b * GROUP_BLOCK_BYTES), wp.add((b + 1) * GROUP_BLOCK_BYTES));
            let i0 = group_block(p0, ap.add(b * QK));
            let i1 = group_block(p1, ap.add((b + 1) * QK));
            acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(i0), vmulq_n_f32(f16x4(p0), a.d[off + b] * (1.0 / 16.0)));
            acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(i1), vmulq_n_f32(f16x4(p1), a.d[off + b + 1] * (1.0 / 16.0)));
            b += 2;
        }
        if b < nb {
            let p0 = wp.add(b * GROUP_BLOCK_BYTES);
            let i0 = group_block(p0, ap.add(b * QK));
            acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(i0), vmulq_n_f32(f16x4(p0), a.d[off + b] * (1.0 / 16.0)));
        }
        let mut out = [0f32; 4];
        vst1q_f32(out.as_mut_ptr(), vaddq_f32(acc0, acc1));
        out
    }

    /// Groups `g0..g1` against all `m` activation rows, `TP` row pairs at a time.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "dotprod,i8mm")]
    pub unsafe fn gemm_i8mm_task(
        weight: &[u8],
        group_bytes: usize,
        packed: &PackedPairs,
        m: usize,
        n: usize,
        nb: usize,
        g0: usize,
        g1: usize,
        out: *mut f32,
    ) {
        assert!(weight.len() >= g1 * group_bytes);
        let pairs = m.div_ceil(2);
        assert!(packed.qs.len() >= pairs * nb * 64);
        let mut p = 0;
        while p < pairs {
            let tp = (pairs - p).min(4);
            for g in g0..g1 {
                let w = weight.as_ptr().add(g * group_bytes);
                match tp {
                    4 => tile::<4>(w, packed, nb, p, g, m, n, out),
                    3 => tile::<3>(w, packed, nb, p, g, m, n, out),
                    2 => tile::<2>(w, packed, nb, p, g, m, n, out),
                    _ => tile::<1>(w, packed, nb, p, g, m, n, out),
                }
            }
            p += tp;
        }
    }

    /// One 4-column group × `TP` row pairs over all `nb` blocks.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn tile<const TP: usize>(
        w: *const u8,
        packed: &PackedPairs,
        nb: usize,
        p0: usize,
        g: usize,
        m: usize,
        n: usize,
        out: *mut f32,
    ) {
        let hi_mask = vdupq_n_s8(0xF0u8 as i8);
        let mut acc01 = [vdupq_n_f32(0.0); TP];
        let mut acc23 = [vdupq_n_f32(0.0); TP];
        for b in 0..nb {
            let blk = w.add(b * GROUP_BLOCK_BYTES);
            let d = f16x4(blk);
            let d01 = vcombine_f32(vget_low_f32(d), vget_low_f32(d));
            let d23 = vcombine_f32(vget_high_f32(d), vget_high_f32(d));
            let c0_01 = vld1q_s8(blk.add(8) as *const i8);
            let c0_23 = vld1q_s8(blk.add(24) as *const i8);
            let c1_01 = vld1q_s8(blk.add(40) as *const i8);
            let c1_23 = vld1q_s8(blk.add(56) as *const i8);
            let (l0_01, l0_23) = (vshlq_n_s8::<4>(c0_01), vshlq_n_s8::<4>(c0_23));
            let (l1_01, l1_23) = (vshlq_n_s8::<4>(c1_01), vshlq_n_s8::<4>(c1_23));
            let (h0_01, h0_23) = (vandq_s8(c0_01, hi_mask), vandq_s8(c0_23, hi_mask));
            let (h1_01, h1_23) = (vandq_s8(c1_01, hi_mask), vandq_s8(c1_23, hi_mask));
            for t in 0..TP {
                let pb = (p0 + t) * nb + b;
                let ap = packed.qs.as_ptr().add(pb * 64);
                let (x0, x1, x2, x3) = (vld1q_s8(ap), vld1q_s8(ap.add(16)), vld1q_s8(ap.add(32)), vld1q_s8(ap.add(48)));
                let mut i01 = smmla(vdupq_n_s32(0), x0, l0_01);
                let mut i23 = smmla(vdupq_n_s32(0), x0, l0_23);
                i01 = smmla(i01, x1, l1_01);
                i23 = smmla(i23, x1, l1_23);
                i01 = smmla(i01, x2, h0_01);
                i23 = smmla(i23, x2, h0_23);
                i01 = smmla(i01, x3, h1_01);
                i23 = smmla(i23, x3, h1_23);
                let xs = vld1q_f32(packed.scales.as_ptr().add(pb * 4));
                acc01[t] = vfmaq_f32(acc01[t], vcvtq_f32_s32(i01), vmulq_f32(d01, xs));
                acc23[t] = vfmaq_f32(acc23[t], vcvtq_f32_s32(i23), vmulq_f32(d23, xs));
            }
        }
        for t in 0..TP {
            // acc01 = [x0·r0, x0·r1, x1·r0, x1·r1], acc23 likewise for rows r2, r3.
            let row0 = 2 * (p0 + t);
            vst1q_f32(out.add(row0 * n + 4 * g), vcombine_f32(vget_low_f32(acc01[t]), vget_low_f32(acc23[t])));
            if row0 + 1 < m {
                vst1q_f32(out.add((row0 + 1) * n + 4 * g), vcombine_f32(vget_high_f32(acc01[t]), vget_high_f32(acc23[t])));
            }
        }
    }
    /// Weight-row groups `g0..g1` (4 rows each) of a row-major `Q8_0` matrix against all
    /// `m` activation rows, up to 4 row pairs at a time.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "dotprod,i8mm")]
    pub unsafe fn gemm_q8_0_i8mm_task(
        weight: &[u8],
        row_bytes: usize,
        packed: &PackedPairs,
        m: usize,
        n: usize,
        nb: usize,
        g0: usize,
        g1: usize,
        out: *mut f32,
    ) {
        assert!(weight.len() >= 4 * g1 * row_bytes);
        let pairs = m.div_ceil(2);
        assert!(packed.qs.len() >= pairs * nb * 64 && packed.scales.len() >= pairs * nb * 4);
        let mut p = 0;
        while p < pairs {
            let tp = (pairs - p).min(4);
            for g in g0..g1 {
                let w = weight.as_ptr().add(4 * g * row_bytes);
                match tp {
                    4 => tile_q8_0::<4>(w, row_bytes, packed, nb, p, g, m, n, out),
                    3 => tile_q8_0::<3>(w, row_bytes, packed, nb, p, g, m, n, out),
                    2 => tile_q8_0::<2>(w, row_bytes, packed, nb, p, g, m, n, out),
                    _ => tile_q8_0::<1>(w, row_bytes, packed, nb, p, g, m, n, out),
                }
            }
            p += tp;
        }
    }

    /// Weight rows `4g..4g+4` × `TP` activation row pairs over all `nb` blocks.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn tile_q8_0<const TP: usize>(
        w: *const u8,
        row_bytes: usize,
        packed: &PackedPairs,
        nb: usize,
        p0: usize,
        g: usize,
        m: usize,
        n: usize,
        out: *mut f32,
    ) {
        let rows = [w, w.add(row_bytes), w.add(2 * row_bytes), w.add(3 * row_bytes)];
        let mut acc01 = [vdupq_n_f32(0.0); TP];
        let mut acc23 = [vdupq_n_f32(0.0); TP];
        // Packed activation scales are d/16 (shared with the Q4_0x4 kernel).
        let sixteen = vdupq_n_f32(16.0);
        for b in 0..nb {
            let o = b * Q8_0_BYTES;
            let ld = |r: usize, i: usize| vreinterpretq_s64_s8(vld1q_s8(rows[r].add(o + 2 + 16 * i) as *const i8));
            let zip = |lo: int64x2_t, hi: int64x2_t| (vreinterpretq_s8_s64(vzip1q_s64(lo, hi)), vreinterpretq_s8_s64(vzip2q_s64(lo, hi)));
            // c{chunk}_{rows}: 8 values of each of two rows, the SMMLA right operand.
            let (c0_01, c1_01) = zip(ld(0, 0), ld(1, 0));
            let (c2_01, c3_01) = zip(ld(0, 1), ld(1, 1));
            let (c0_23, c1_23) = zip(ld(2, 0), ld(3, 0));
            let (c2_23, c3_23) = zip(ld(2, 1), ld(3, 1));
            // The four rows' f16 scales gathered into one vector and widened (FCVTL).
            let mut h = vdup_n_u16(0);
            h = vset_lane_u16::<0>(std::ptr::read_unaligned(rows[0].add(o) as *const u16), h);
            h = vset_lane_u16::<1>(std::ptr::read_unaligned(rows[1].add(o) as *const u16), h);
            h = vset_lane_u16::<2>(std::ptr::read_unaligned(rows[2].add(o) as *const u16), h);
            h = vset_lane_u16::<3>(std::ptr::read_unaligned(rows[3].add(o) as *const u16), h);
            let d = vmulq_f32(f16x4_from(h), sixteen);
            let d01 = vcombine_f32(vget_low_f32(d), vget_low_f32(d));
            let d23 = vcombine_f32(vget_high_f32(d), vget_high_f32(d));
            for t in 0..TP {
                let pb = (p0 + t) * nb + b;
                let ap = packed.qs.as_ptr().add(pb * 64);
                let (x0, x1, x2, x3) = (vld1q_s8(ap), vld1q_s8(ap.add(16)), vld1q_s8(ap.add(32)), vld1q_s8(ap.add(48)));
                let mut i01 = smmla(vdupq_n_s32(0), x0, c0_01);
                let mut i23 = smmla(vdupq_n_s32(0), x0, c0_23);
                i01 = smmla(i01, x1, c1_01);
                i23 = smmla(i23, x1, c1_23);
                i01 = smmla(i01, x2, c2_01);
                i23 = smmla(i23, x2, c2_23);
                i01 = smmla(i01, x3, c3_01);
                i23 = smmla(i23, x3, c3_23);
                let xs = vld1q_f32(packed.scales.as_ptr().add(pb * 4));
                acc01[t] = vfmaq_f32(acc01[t], vcvtq_f32_s32(i01), vmulq_f32(d01, xs));
                acc23[t] = vfmaq_f32(acc23[t], vcvtq_f32_s32(i23), vmulq_f32(d23, xs));
            }
        }
        for t in 0..TP {
            let row0 = 2 * (p0 + t);
            vst1q_f32(out.add(row0 * n + 4 * g), vcombine_f32(vget_low_f32(acc01[t]), vget_low_f32(acc23[t])));
            if row0 + 1 < m {
                vst1q_f32(out.add((row0 + 1) * n + 4 * g), vcombine_f32(vget_high_f32(acc01[t]), vget_high_f32(acc23[t])));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::qdot::scalar_dot_q4_0_q8;
    use crate::ops::quant::{dequantize_q4_0, quantize_f32_to_q4_0};

    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|i| (((i * 7919 + seed * 104729) % 2001) as f32 - 1000.0) / 400.0).collect()
    }

    fn q4(n: usize, k: usize, seed: usize) -> Vec<u8> {
        let w = data(n * k, seed);
        let mut b = vec![0u8; n * k / 32 * 18];
        quantize_f32_to_q4_0(&w, &mut b);
        b
    }

    #[test]
    fn repack_round_trips_and_dequantizes() {
        let (n, k) = (8, 96);
        let w = q4(n, k, 1);
        let r = repack_q4_0(&w, n, k);
        assert_eq!(unpack_q4_0x4(&r, n, k), w);
        let mut want = vec![0f32; n * k];
        dequantize_q4_0(&w, &mut want);
        let mut got = vec![0f32; k];
        for row in 0..n {
            dequantize_q4_0x4_row(&r, k, row, &mut got);
            assert_eq!(got, want[row * k..(row + 1) * k]);
        }
        let mut rows = vec![vec![0f32; k]; 4];
        let [a, b, c, d] = &mut rows[..] else { unreachable!() };
        dequantize_q4_0x4_group(&r[(k / 32) * 72..], [a, b, c, d]);
        assert_eq!(rows.concat(), want[4 * k..]);
    }

    #[test]
    fn dots_match_unpacked_q4_0() {
        for nb in [1usize, 2, 5] {
            let (n, k) = (4, nb * QK);
            let w = q4(n, k, 2 + nb);
            let r = repack_q4_0(&w, n, k);
            let a = quantize_activations(&data(k, 3));
            let row_bytes = nb * 18;
            let got = vec_dot_q4_0x4(&r, &a, 0, nb);
            let reference = scalar_dot_q4_0x4(&r, &a, 0, nb);
            for i in 0..4 {
                let want = scalar_dot_q4_0_q8(&w[i * row_bytes..(i + 1) * row_bytes], &a, 0, nb);
                assert!((got[i] - want).abs() <= 1e-4 * want.abs().max(1.0), "nb={nb} r={i}: {} vs {want}", got[i]);
                assert!((reference[i] - want).abs() <= 1e-4 * want.abs().max(1.0));
            }
        }
    }

    #[test]
    fn gemm_and_gemv_match_per_row_reference() {
        let (n, k) = (40, 128);
        let nb = k / QK;
        let w = q4(n, k, 7);
        let r = repack_q4_0(&w, n, k);
        for m in [1usize, 2, 3, 7, 9, 33] {
            let x = data(m * k, 8 + m);
            let mut out = vec![0f32; m * n];
            if m == 1 {
                gemv_q4_0x4(&r, &x, &mut out, n, k);
            } else {
                gemm_q4_0x4(&r, &x, &mut out, m, n, k);
            }
            let a = quantize_activations(&x);
            for i in 0..m {
                for j in 0..n {
                    let want = scalar_dot_q4_0_q8(&w[j * nb * 18..(j + 1) * nb * 18], &a, i * nb, nb);
                    let got = out[i * n + j];
                    assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "m={m} ({i},{j}): {got} vs {want}");
                }
            }
        }
    }
}

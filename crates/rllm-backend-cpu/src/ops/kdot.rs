//! Native dot products for the K-quant formats `Q4_K` and `Q6_K`.
//!
//! As in llama.cpp, activations are quantized to `Q8_K`: 256-value super-blocks with
//! an f32 scale and per-16-value sums (`bsums`). The sums let the per-sub-block offsets
//! of the weights (`Q4_K` mins, `Q6_K`'s `-32`) be applied once per sub-block instead
//! of per value:
//!
//! `dot = d_a · (d_w · Σ_j sc_j · Σ_l q_l·a_l − dmin_w · Σ_j m_j · Σ_l a_l)`
//!
//! On aarch64 with `dotprod` the inner sums use `SDOT`; elsewhere a scalar loop is used.

#![allow(clippy::chunks_exact_to_as_chunks)]

use half::f16;

/// Values per K-quant super-block.
pub const QK_K: usize = 256;
/// Bytes of one `Q4_K` super-block.
pub const Q4_K_BYTES: usize = 144;
/// Bytes of one `Q6_K` super-block.
pub const Q6_K_BYTES: usize = 210;

/// Rows of activations quantized to `Q8_K` super-blocks.
#[derive(Debug, Clone, Default)]
pub struct Q8KActivations {
    /// Per-super-block scale.
    pub d: Vec<f32>,
    /// Quantized values, `256 * d.len()` of them.
    pub qs: Vec<i8>,
    /// Sums of each run of 16 quantized values, `16 * d.len()` of them.
    pub bsums: Vec<i16>,
}

/// Quantizes `x` (length a multiple of 256) like llama.cpp's `quantize_row_q8_K_ref`.
pub fn quantize_q8_k(x: &[f32]) -> Q8KActivations {
    debug_assert_eq!(x.len() % QK_K, 0);
    let nb = x.len() / QK_K;
    let mut d = Vec::with_capacity(nb);
    let mut qs = vec![0i8; x.len()];
    let mut bsums = vec![0i16; nb * 16];
    for ((block, out), sums) in x
        .chunks_exact(QK_K)
        .zip(qs.chunks_exact_mut(QK_K))
        .zip(bsums.chunks_exact_mut(16))
    {
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            d.push(0.0);
            continue;
        }
        let iscale = -127.0 / max;
        for (o, &v) in out.iter_mut().zip(block) {
            *o = ((iscale * v).round_ties_even() as i32).min(127) as i8;
        }
        for (s, q) in sums.iter_mut().zip(out.chunks_exact(16)) {
            *s = q.iter().map(|&v| v as i16).sum();
        }
        d.push(1.0 / iscale);
    }
    Q8KActivations { d, qs, bsums }
}

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

/// Unpacks the eight 6-bit scales and mins of a `Q4_K` super-block.
#[inline]
pub(crate) fn q4_k_scales_mins(s: &[u8]) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let u = |i: usize| u32::from_le_bytes([s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]]);
    let (u0, u1, u2) = (u(0), u(1), u(2));
    let mins = [u1 & KMASK1, ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4)];
    let scales = [u0 & KMASK1, (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4)];
    let bytes = |w: [u32; 2]| {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&w[0].to_le_bytes());
        b[4..].copy_from_slice(&w[1].to_le_bytes());
        b
    };
    (bytes(scales), bytes(mins))
}

#[cfg(target_arch = "aarch64")]
fn has_dotprod() -> bool {
    static DOTPROD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DOTPROD.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

/// Dot product of one `Q4_K` row (`nb` super-blocks) with activation super-blocks
/// `off..off + nb`.
#[inline]
pub fn vec_dot_q4_k_q8_k(w: &[u8], a: &Q8KActivations, off: usize, nb: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: dotprod detected at runtime; slice lengths are asserted inside.
        return unsafe { neon::dot_q4_k(w, a, off, nb) };
    }
    #[cfg(target_arch = "x86_64")]
    if super::x86::has_avx2() {
        // SAFETY: AVX2/FMA/F16C detected at run time; the kernel asserts its bounds.
        return unsafe { super::x86::dot_q4_k(w, a, off, nb) };
    }
    scalar_dot_q4_k_q8_k(w, a, off, nb)
}

/// Dot product of one `Q6_K` row with activation super-blocks `off..off + nb`.
#[inline]
pub fn vec_dot_q6_k_q8_k(w: &[u8], a: &Q8KActivations, off: usize, nb: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: as above.
        return unsafe { neon::dot_q6_k(w, a, off, nb) };
    }
    #[cfg(target_arch = "x86_64")]
    if super::x86::has_avx2() {
        // SAFETY: AVX2/FMA/F16C detected at run time; the kernel asserts its bounds.
        return unsafe { super::x86::dot_q6_k(w, a, off, nb) };
    }
    scalar_dot_q6_k_q8_k(w, a, off, nb)
}

/// Dot products of one `Q4_K` row with `R` activation rows whose super-blocks start at
/// `offs[r]`. Each weight super-block is unpacked once for all `R` rows.
#[inline]
pub fn dot_q4_k_rows<const R: usize>(w: &[u8], a: &Q8KActivations, offs: [usize; R], nb: usize) -> [f32; R] {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: dotprod detected at runtime; slice lengths are asserted inside.
        return unsafe { neon::dot_q4_k_rows(w, a, offs, nb) };
    }
    std::array::from_fn(|r| vec_dot_q4_k_q8_k(w, a, offs[r], nb))
}

/// Dot products of one `Q6_K` row with `R` activation rows (see [`dot_q4_k_rows`]).
#[inline]
pub fn dot_q6_k_rows<const R: usize>(w: &[u8], a: &Q8KActivations, offs: [usize; R], nb: usize) -> [f32; R] {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: as above.
        return unsafe { neon::dot_q6_k_rows(w, a, offs, nb) };
    }
    std::array::from_fn(|r| vec_dot_q6_k_q8_k(w, a, offs[r], nb))
}

/// Portable reference implementation of [`vec_dot_q4_k_q8_k`].
pub fn scalar_dot_q4_k_q8_k(w: &[u8], a: &Q8KActivations, off: usize, nb: usize) -> f32 {
    let mut sum = 0f32;
    for i in 0..nb {
        let blk = &w[i * Q4_K_BYTES..(i + 1) * Q4_K_BYTES];
        let (scales, mins) = q4_k_scales_mins(&blk[4..16]);
        let qs = &blk[16..];
        let aq = &a.qs[(off + i) * QK_K..(off + i + 1) * QK_K];
        let bs = &a.bsums[(off + i) * 16..(off + i + 1) * 16];
        let mut isum = 0i32;
        for j in 0..4 {
            let q = &qs[j * 32..j * 32 + 32];
            let (lo, hi): (i32, i32) = (0..32).fold((0, 0), |(l, h), t| {
                (
                    l + (q[t] & 0x0F) as i32 * aq[j * 64 + t] as i32,
                    h + (q[t] >> 4) as i32 * aq[j * 64 + 32 + t] as i32,
                )
            });
            isum += lo * scales[2 * j] as i32 + hi * scales[2 * j + 1] as i32;
        }
        let msum: i32 = (0..8)
            .map(|j| mins[j] as i32 * (bs[2 * j] as i32 + bs[2 * j + 1] as i32))
            .sum();
        let da = a.d[off + i];
        sum += da * f16_at(blk, 0) * isum as f32 - da * f16_at(blk, 2) * msum as f32;
    }
    sum
}

/// Portable reference implementation of [`vec_dot_q6_k_q8_k`].
pub fn scalar_dot_q6_k_q8_k(w: &[u8], a: &Q8KActivations, off: usize, nb: usize) -> f32 {
    let mut sum = 0f32;
    for i in 0..nb {
        let blk = &w[i * Q6_K_BYTES..(i + 1) * Q6_K_BYTES];
        let aq = &a.qs[(off + i) * QK_K..(off + i + 1) * QK_K];
        let mut isum = 0i32;
        for half in 0..2 {
            let ql = &blk[half * 64..];
            let qh = &blk[128 + half * 32..];
            let sc = &blk[192 + half * 8..];
            let aq = &aq[half * 128..];
            // Sub-block sums of 16 values each, in output order (8 per half).
            let mut sub = [0i32; 8];
            for l in 0..32 {
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                let is = l / 16;
                sub[is] += q1 * aq[l] as i32;
                sub[is + 2] += q2 * aq[l + 32] as i32;
                sub[is + 4] += q3 * aq[l + 64] as i32;
                sub[is + 6] += q4 * aq[l + 96] as i32;
            }
            isum += (0..8).map(|s| sc[s] as i8 as i32 * sub[s]).sum::<i32>();
        }
        sum += a.d[off + i] * f16_at(blk, 208) * isum as f32;
    }
    sum
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

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q4_k(w: &[u8], a: &Q8KActivations, off: usize, nb: usize) -> f32 {
        assert!(w.len() >= nb * Q4_K_BYTES && a.qs.len() >= (off + nb) * QK_K);
        let m4 = vdupq_n_u8(0x0F);
        let mut sumf = 0f32;
        for i in 0..nb {
            let blk = w.as_ptr().add(i * Q4_K_BYTES);
            let bs = a.bsums.as_ptr().add((off + i) * 16);
            let da = a.d[off + i];
            let d = da * f16_at(w, i * Q4_K_BYTES);
            let dmin = da * f16_at(w, i * Q4_K_BYTES + 2);
            let (scales, mins) = q4_k_scales_mins(std::slice::from_raw_parts(blk.add(4), 12));

            // Σ_j m_j · (sum of the 32 activations of sub-block j).
            let q8sums = vpaddq_s16(vld1q_s16(bs), vld1q_s16(bs.add(8)));
            let mins = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins.as_ptr())));
            let prod = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
            );
            sumf -= dmin * vaddvq_s32(prod) as f32;

            let mut q4 = blk.add(16);
            let mut q8 = a.qs.as_ptr().add((off + i) * QK_K);
            let mut sumi = vdupq_n_s32(0);
            for j in 0..4 {
                let b0 = vld1q_u8(q4);
                let b1 = vld1q_u8(q4.add(16));
                q4 = q4.add(32);
                let lo = sdot(
                    sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(b0, m4)), vld1q_s8(q8)),
                    vreinterpretq_s8_u8(vandq_u8(b1, m4)),
                    vld1q_s8(q8.add(16)),
                );
                let hi = sdot(
                    sdot(
                        vdupq_n_s32(0),
                        vreinterpretq_s8_u8(vshrq_n_u8::<4>(b0)),
                        vld1q_s8(q8.add(32)),
                    ),
                    vreinterpretq_s8_u8(vshrq_n_u8::<4>(b1)),
                    vld1q_s8(q8.add(48)),
                );
                q8 = q8.add(64);
                sumi = vmlaq_n_s32(sumi, lo, scales[2 * j] as i32);
                sumi = vmlaq_n_s32(sumi, hi, scales[2 * j + 1] as i32);
            }
            sumf += d * vaddvq_s32(sumi) as f32;
        }
        sumf
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q6_k(w: &[u8], a: &Q8KActivations, off: usize, nb: usize) -> f32 {
        assert!(w.len() >= nb * Q6_K_BYTES && a.qs.len() >= (off + nb) * QK_K);
        let m4 = vdupq_n_u8(0x0F);
        let m3 = vdupq_n_u8(0x03);
        let mut sum = 0f32;
        for i in 0..nb {
            let blk = w.as_ptr().add(i * Q6_K_BYTES);
            let mut ql = blk;
            let mut qh = blk.add(128);
            let sc = blk.add(192) as *const i8;
            let bs = a.bsums.as_ptr().add((off + i) * 16);
            let mut q8 = a.qs.as_ptr().add((off + i) * QK_K);

            // The stored 6-bit values are unsigned (q + 32); fold the -32 via bsums.
            let scales = vld1q_s8(sc);
            let (s_lo, s_hi) = (vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales)));
            let (b_lo, b_hi) = (vld1q_s16(bs), vld1q_s16(bs.add(8)));
            let prod = vaddq_s32(
                vaddq_s32(
                    vmull_s16(vget_low_s16(b_lo), vget_low_s16(s_lo)),
                    vmull_s16(vget_high_s16(b_lo), vget_high_s16(s_lo)),
                ),
                vaddq_s32(
                    vmull_s16(vget_low_s16(b_hi), vget_low_s16(s_hi)),
                    vmull_s16(vget_high_s16(b_hi), vget_high_s16(s_hi)),
                ),
            );
            let isum_mins = vaddvq_s32(prod);

            let mut isum = vdupq_n_s32(0);
            let mut s = 0;
            for _ in 0..2 {
                let h0 = vld1q_u8(qh);
                let h1 = vld1q_u8(qh.add(16));
                qh = qh.add(32);
                let l = [
                    vld1q_u8(ql),
                    vld1q_u8(ql.add(16)),
                    vld1q_u8(ql.add(32)),
                    vld1q_u8(ql.add(48)),
                ];
                ql = ql.add(64);

                let hb = [
                    vshlq_n_u8::<4>(vandq_u8(h0, m3)),
                    vshlq_n_u8::<4>(vandq_u8(h1, m3)),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(h0), m3)),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(h1), m3)),
                ];
                for k in 0..4 {
                    let q = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l[k], m4), hb[k]));
                    isum = vmlaq_n_s32(
                        isum,
                        sdot(vdupq_n_s32(0), q, vld1q_s8(q8.add(16 * k))),
                        *sc.add(s + k) as i32,
                    );
                }
                q8 = q8.add(64);
                s += 4;

                let hb = [
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(h0), m3)),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(h1), m3)),
                    vshlq_n_u8::<4>(vshrq_n_u8::<6>(h0)),
                    vshlq_n_u8::<4>(vshrq_n_u8::<6>(h1)),
                ];
                for k in 0..4 {
                    let q = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(l[k]), hb[k]));
                    isum = vmlaq_n_s32(
                        isum,
                        sdot(vdupq_n_s32(0), q, vld1q_s8(q8.add(16 * k))),
                        *sc.add(s + k) as i32,
                    );
                }
                q8 = q8.add(64);
                s += 4;
            }
            sum += a.d[off + i] * f16_at(w, i * Q6_K_BYTES + 208) * (vaddvq_s32(isum) - 32 * isum_mins) as f32;
        }
        sum
    }

    /// Asserts that activation super-blocks `offs[r]..offs[r] + nb` exist for every row.
    fn check_rows<const R: usize>(w: &[u8], bytes: usize, a: &Q8KActivations, offs: &[usize; R], nb: usize) {
        assert!(w.len() >= nb * bytes);
        for &o in offs {
            assert!(a.qs.len() >= (o + nb) * QK_K && a.bsums.len() >= (o + nb) * 16 && a.d.len() >= o + nb);
        }
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q4_k_rows<const R: usize>(w: &[u8], a: &Q8KActivations, offs: [usize; R], nb: usize) -> [f32; R] {
        check_rows(w, Q4_K_BYTES, a, &offs, nb);
        let m4 = vdupq_n_u8(0x0F);
        let mut sumf = [0f32; R];
        for i in 0..nb {
            let blk = w.as_ptr().add(i * Q4_K_BYTES);
            let dw = f16_at(w, i * Q4_K_BYTES);
            let dmw = f16_at(w, i * Q4_K_BYTES + 2);
            let (scales, mins) = q4_k_scales_mins(std::slice::from_raw_parts(blk.add(4), 12));
            let mins = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins.as_ptr())));
            let mut sumi = [vdupq_n_s32(0); R];
            for r in 0..R {
                let bs = a.bsums.as_ptr().add((offs[r] + i) * 16);
                let q8sums = vpaddq_s16(vld1q_s16(bs), vld1q_s16(bs.add(8)));
                let prod = vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                    vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
                );
                sumf[r] -= a.d[offs[r] + i] * dmw * vaddvq_s32(prod) as f32;
            }
            for j in 0..4 {
                let b0 = vld1q_u8(blk.add(16 + 32 * j));
                let b1 = vld1q_u8(blk.add(32 + 32 * j));
                let l0 = vreinterpretq_s8_u8(vandq_u8(b0, m4));
                let l1 = vreinterpretq_s8_u8(vandq_u8(b1, m4));
                let h0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b0));
                let h1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b1));
                let (s0, s1) = (scales[2 * j] as i32, scales[2 * j + 1] as i32);
                for r in 0..R {
                    let q8 = a.qs.as_ptr().add((offs[r] + i) * QK_K + 64 * j);
                    let lo = sdot(sdot(vdupq_n_s32(0), l0, vld1q_s8(q8)), l1, vld1q_s8(q8.add(16)));
                    let hi = sdot(sdot(vdupq_n_s32(0), h0, vld1q_s8(q8.add(32))), h1, vld1q_s8(q8.add(48)));
                    sumi[r] = vmlaq_n_s32(vmlaq_n_s32(sumi[r], lo, s0), hi, s1);
                }
            }
            for r in 0..R {
                sumf[r] += a.d[offs[r] + i] * dw * vaddvq_s32(sumi[r]) as f32;
            }
        }
        sumf
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot_q6_k_rows<const R: usize>(w: &[u8], a: &Q8KActivations, offs: [usize; R], nb: usize) -> [f32; R] {
        check_rows(w, Q6_K_BYTES, a, &offs, nb);
        let m4 = vdupq_n_u8(0x0F);
        let m3 = vdupq_n_u8(0x03);
        let mut sum = [0f32; R];
        for i in 0..nb {
            let blk = w.as_ptr().add(i * Q6_K_BYTES);
            let sc = blk.add(192) as *const i8;
            let scales = vld1q_s8(sc);
            let (s_lo, s_hi) = (vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales)));
            let mut isum = [vdupq_n_s32(0); R];
            let mut mins = [0i32; R];
            for r in 0..R {
                let bs = a.bsums.as_ptr().add((offs[r] + i) * 16);
                let (b_lo, b_hi) = (vld1q_s16(bs), vld1q_s16(bs.add(8)));
                let prod = vaddq_s32(
                    vaddq_s32(
                        vmull_s16(vget_low_s16(b_lo), vget_low_s16(s_lo)),
                        vmull_s16(vget_high_s16(b_lo), vget_high_s16(s_lo)),
                    ),
                    vaddq_s32(
                        vmull_s16(vget_low_s16(b_hi), vget_low_s16(s_hi)),
                        vmull_s16(vget_high_s16(b_hi), vget_high_s16(s_hi)),
                    ),
                );
                mins[r] = vaddvq_s32(prod);
            }
            for half in 0..2 {
                let qh = blk.add(128 + 32 * half);
                let ql = blk.add(64 * half);
                let h0 = vld1q_u8(qh);
                let h1 = vld1q_u8(qh.add(16));
                let l = [
                    vld1q_u8(ql),
                    vld1q_u8(ql.add(16)),
                    vld1q_u8(ql.add(32)),
                    vld1q_u8(ql.add(48)),
                ];
                for part in 0..2 {
                    let q: [int8x16_t; 4] = if part == 0 {
                        let hb = [
                            vshlq_n_u8::<4>(vandq_u8(h0, m3)),
                            vshlq_n_u8::<4>(vandq_u8(h1, m3)),
                            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(h0), m3)),
                            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(h1), m3)),
                        ];
                        std::array::from_fn(|k| vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l[k], m4), hb[k])))
                    } else {
                        let hb = [
                            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(h0), m3)),
                            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(h1), m3)),
                            vshlq_n_u8::<4>(vshrq_n_u8::<6>(h0)),
                            vshlq_n_u8::<4>(vshrq_n_u8::<6>(h1)),
                        ];
                        std::array::from_fn(|k| vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(l[k]), hb[k])))
                    };
                    let s = 8 * half + 4 * part;
                    let scs: [i32; 4] = std::array::from_fn(|k| *sc.add(s + k) as i32);
                    for r in 0..R {
                        let q8 = a.qs.as_ptr().add((offs[r] + i) * QK_K + 128 * half + 64 * part);
                        let mut acc = isum[r];
                        for k in 0..4 {
                            acc = vmlaq_n_s32(acc, sdot(vdupq_n_s32(0), q[k], vld1q_s8(q8.add(16 * k))), scs[k]);
                        }
                        isum[r] = acc;
                    }
                }
            }
            let dw = f16_at(w, i * Q6_K_BYTES + 208);
            for r in 0..R {
                sum[r] += a.d[offs[r] + i] * dw * (vaddvq_s32(isum[r]) - 32 * mins[r]) as f32;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::quant::{dequantize_q4_k, dequantize_q6_k};

    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 7919 + seed * 104729) % 2001) as f32 - 1000.0) / 400.0)
            .collect()
    }

    /// Random-looking but deterministic super-blocks with small f16 scales.
    fn random_blocks(nb: usize, bytes: usize, d_offsets: &[usize], seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut out: Vec<u8> = (0..nb * bytes)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect();
        for b in 0..nb {
            for (n, &o) in d_offsets.iter().enumerate() {
                let d = f16::from_f32(0.01 + 0.003 * (b + n) as f32);
                out[b * bytes + o..b * bytes + o + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        out
    }

    /// `dense` uses the dequantized weights and dequantized activations, so the integer
    /// kernels must reproduce it up to float rounding.
    fn check(name: &str, fast: f32, slow: f32, dense: f32) {
        assert!(
            (fast - slow).abs() <= 1e-4 * slow.abs().max(1.0),
            "{name}: simd {fast} vs scalar {slow}"
        );
        assert!(
            (slow - dense).abs() <= 1e-4 * dense.abs().max(1.0),
            "{name}: int {slow} vs dense {dense}"
        );
    }

    #[test]
    fn q8_k_quantization_round_trips() {
        let x = data(512, 5);
        let a = quantize_q8_k(&x);
        for (b, block) in x.chunks(QK_K).enumerate() {
            for (j, &v) in block.iter().enumerate() {
                let r = a.qs[b * QK_K + j] as f32 * a.d[b];
                assert!((r - v).abs() <= a.d[b].abs() * 0.51, "{v} vs {r}");
            }
            for s in 0..16 {
                let want: i16 = a.qs[b * QK_K + 16 * s..b * QK_K + 16 * s + 16]
                    .iter()
                    .map(|&q| q as i16)
                    .sum();
                assert_eq!(a.bsums[b * 16 + s], want);
            }
        }
    }

    #[test]
    fn k_quant_dots_match_dequantized_reference() {
        for nb in [1usize, 2, 3] {
            let k = nb * QK_K;
            let x = data(k, 7);
            let a = quantize_q8_k(&x);
            let x: Vec<f32> = (0..k).map(|j| a.qs[j] as f32 * a.d[j / QK_K]).collect();
            let mut wd = vec![0f32; k];

            let w4 = random_blocks(nb, Q4_K_BYTES, &[0, 2], 11 + nb as u64);
            dequantize_q4_k(&w4, &mut wd);
            let dense: f32 = wd.iter().zip(&x).map(|(p, q)| p * q).sum();
            check(
                "q4_k",
                vec_dot_q4_k_q8_k(&w4, &a, 0, nb),
                scalar_dot_q4_k_q8_k(&w4, &a, 0, nb),
                dense,
            );

            let w6 = random_blocks(nb, Q6_K_BYTES, &[208], 23 + nb as u64);
            dequantize_q6_k(&w6, &mut wd);
            let dense: f32 = wd.iter().zip(&x).map(|(p, q)| p * q).sum();
            check(
                "q6_k",
                vec_dot_q6_k_q8_k(&w6, &a, 0, nb),
                scalar_dot_q6_k_q8_k(&w6, &a, 0, nb),
                dense,
            );
        }
    }
}

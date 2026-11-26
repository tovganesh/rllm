//! AVX2 kernels for x86_64 (with FMA and F16C), chosen at run time by [`has_avx2`].
//!
//! The int8 dot products follow llama.cpp's AVX2 kernels (MIT): a signed×signed byte
//! product is formed as `maddubs(|x|, sign(y, x))` (unsigned × signed, 16-bit pair sums)
//! and widened with `madd(·, 1)`; K-quant sub-block scales are applied with a second
//! `madd` against the 16-bit pair sums. F16 data is widened with F16C.
//!
//! # Safety
//!
//! Every function here is `unsafe`: callers must have checked [`has_avx2`] and pass
//! slices holding the blocks they name (asserted at entry).

#![allow(clippy::missing_safety_doc)]

use super::kdot::{Q8KActivations, Q4_K_BYTES, Q6_K_BYTES, QK_K};
use super::qdot::{Q8Activations, QK};
use half::f16;
use std::arch::x86_64::*;

const Q8_0_BYTES: usize = 34;
const Q4_0_BYTES: usize = 18;

/// Whether the CPU has AVX2, FMA and F16C (every x86_64 CPU since Haswell / Zen 1).
pub fn has_avx2() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    })
}

#[inline(always)]
unsafe fn f16_at(p: *const u8) -> f32 {
    let bits = u16::from_le_bytes([*p, *p.add(1)]);
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
}

#[inline(always)]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}

/// Eight 32-bit sums of the products of the signed bytes of `x` and `y` (pairs of four).
#[inline(always)]
unsafe fn mul_sum_i8(x: __m256i, y: __m256i) -> __m256i {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), _mm256_set1_epi16(1))
}

/// 32 signed values `q - 8` from the 16 nibble bytes of a `Q4_0` block: low nibbles
/// (values 0..16) in the low lane, high nibbles (16..32) in the high lane.
#[inline(always)]
unsafe fn q4_0_values(p: *const u8) -> __m256i {
    let t = _mm_loadu_si128(p as *const __m128i);
    let m4 = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(t, m4);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(t), m4);
    _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8))
}

/// One `Q8_0` or `Q4_0` weight row against activation blocks `off..off + nb`.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q8_0(w: &[u8], a: &Q8Activations, off: usize, nb: usize) -> f32 {
    assert!(w.len() >= nb * Q8_0_BYTES && a.qs.len() >= (off + nb) * QK && a.d.len() >= off + nb);
    let (wp, ap) = (w.as_ptr(), a.qs.as_ptr().add(off * QK));
    let mut acc = _mm256_setzero_ps();
    for b in 0..nb {
        let blk = wp.add(b * Q8_0_BYTES);
        let x = _mm256_loadu_si256(blk.add(2) as *const __m256i);
        let y = _mm256_loadu_si256(ap.add(b * QK) as *const __m256i);
        let d = _mm256_set1_ps(f16_at(blk) * *a.d.get_unchecked(off + b));
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(mul_sum_i8(x, y)), acc);
    }
    hsum_ps(acc)
}

/// As [`dot_q8_0`] for `Q4_0` weights.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q4_0(w: &[u8], a: &Q8Activations, off: usize, nb: usize) -> f32 {
    assert!(w.len() >= nb * Q4_0_BYTES && a.qs.len() >= (off + nb) * QK && a.d.len() >= off + nb);
    let (wp, ap) = (w.as_ptr(), a.qs.as_ptr().add(off * QK));
    let mut acc = _mm256_setzero_ps();
    for b in 0..nb {
        let blk = wp.add(b * Q4_0_BYTES);
        let x = q4_0_values(blk.add(2));
        let y = _mm256_loadu_si256(ap.add(b * QK) as *const __m256i);
        let d = _mm256_set1_ps(f16_at(blk) * *a.d.get_unchecked(off + b));
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(mul_sum_i8(x, y)), acc);
    }
    hsum_ps(acc)
}

/// One `Q8_0` weight row against `R` activation rows (blocks from `offs[r]`); each
/// weight block is loaded once for all rows.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q8_0_rows<const R: usize>(w: &[u8], a: &Q8Activations, offs: [usize; R], nb: usize) -> [f32; R] {
    assert!(w.len() >= nb * Q8_0_BYTES);
    for &o in &offs {
        assert!(a.qs.len() >= (o + nb) * QK && a.d.len() >= o + nb);
    }
    let wp = w.as_ptr();
    let mut acc = [_mm256_setzero_ps(); R];
    for b in 0..nb {
        let blk = wp.add(b * Q8_0_BYTES);
        let x = _mm256_loadu_si256(blk.add(2) as *const __m256i);
        let ax = _mm256_sign_epi8(x, x);
        let dw = f16_at(blk);
        for r in 0..R {
            let y = _mm256_loadu_si256(a.qs.as_ptr().add((offs[r] + b) * QK) as *const __m256i);
            let s = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(y, x)), _mm256_set1_epi16(1));
            let d = _mm256_set1_ps(dw * *a.d.get_unchecked(offs[r] + b));
            acc[r] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(s), acc[r]);
        }
    }
    let mut out = [0f32; R];
    for r in 0..R {
        out[r] = hsum_ps(acc[r]);
    }
    out
}

/// One `Q4_K` weight row against `Q8_K` activation super-blocks `off..off + nb`.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q4_k(w: &[u8], a: &Q8KActivations, off: usize, nb: usize) -> f32 {
    assert!(w.len() >= nb * Q4_K_BYTES && a.qs.len() >= (off + nb) * QK_K && a.bsums.len() >= (off + nb) * 16);
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();
    for i in 0..nb {
        let blk = w.as_ptr().add(i * Q4_K_BYTES);
        let da = *a.d.get_unchecked(off + i);
        let d = da * f16_at(blk);
        let dmin = -da * f16_at(blk.add(2));
        let (scales, mins) = super::kdot::q4_k_scales_mins(std::slice::from_raw_parts(blk.add(4), 12));
        let mut sm = [0u8; 16];
        sm[..8].copy_from_slice(&scales);
        sm[8..].copy_from_slice(&mins);
        // 16-bit scales (low lane) and mins (high lane).
        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_loadu_si128(sm.as_ptr() as *const __m128i));
        // Sums of each 32-value sub-block of the activations, times the mins.
        let q8sums = _mm256_loadu_si256(a.bsums.as_ptr().add((off + i) * 16) as *const __m256i);
        let q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256::<1>(q8sums));
        let prod = _mm_madd_epi16(_mm256_extracti128_si256::<1>(mins_and_scales), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        let sc128 = _mm256_castsi256_si128(mins_and_scales);
        let scales = _mm256_set_m128i(sc128, sc128);
        let mut q4 = blk.add(16);
        let mut q8 = a.qs.as_ptr().add((off + i) * QK_K);
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            // Broadcast 16-bit scale 2j (low nibbles) and 2j + 1 (high nibbles): bytes
            // 2s, 2s + 1 of each lane.
            let (sl, sh) = (4 * j, 4 * j + 2);
            let scale_l = _mm256_shuffle_epi8(scales, _mm256_set1_epi16(((sl + 1) << 8) | sl));
            let scale_h = _mm256_shuffle_epi8(scales, _mm256_set1_epi16(((sh + 1) << 8) | sh));
            let bits = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4l = _mm256_and_si256(bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16::<4>(bits), m4);
            let q8l = _mm256_loadu_si256(q8 as *const __m256i);
            let q8h = _mm256_loadu_si256(q8.add(32) as *const __m256i);
            q8 = q8.add(64);
            let p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
            let p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    let m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    let m = _mm_add_ss(m, _mm_movehdup_ps(m));
    hsum_ps(acc) + _mm_cvtss_f32(m)
}

/// One `Q6_K` weight row against `Q8_K` activation super-blocks `off..off + nb`.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q6_k(w: &[u8], a: &Q8KActivations, off: usize, nb: usize) -> f32 {
    assert!(w.len() >= nb * Q6_K_BYTES && a.qs.len() >= (off + nb) * QK_K && a.d.len() >= off + nb);
    let m4 = _mm256_set1_epi8(0x0F);
    let m2 = _mm256_set1_epi8(3);
    let m32s = _mm256_set1_epi8(32);
    let mut acc = _mm256_setzero_ps();
    for i in 0..nb {
        let blk = w.as_ptr().add(i * Q6_K_BYTES);
        let d = *a.d.get_unchecked(off + i) * f16_at(blk.add(208));
        let scales = _mm_loadu_si128(blk.add(192) as *const __m128i);
        let (mut ql, mut qh) = (blk, blk.add(128));
        let mut q8 = a.qs.as_ptr().add((off + i) * QK_K);
        let mut sumi = _mm256_setzero_si256();
        for j in 0..2 {
            let bits1 = _mm256_loadu_si256(ql as *const __m256i);
            let bits2 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
            let bits_h = _mm256_loadu_si256(qh as *const __m256i);
            ql = ql.add(64);
            qh = qh.add(32);
            // 6-bit values of the four 32-value groups of this half.
            let h0 = _mm256_slli_epi16::<4>(_mm256_and_si256(bits_h, m2));
            let h1 = _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(bits_h), m2));
            let h2 = _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(bits_h), m2));
            let h3 = _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(bits_h), m2));
            let q = [
                _mm256_or_si256(_mm256_and_si256(bits1, m4), h0),
                _mm256_or_si256(_mm256_and_si256(bits2, m4), h1),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(bits1), m4), h2),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(bits2), m4), h3),
            ];
            for (k, &qk) in q.iter().enumerate() {
                let y = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                // (q - 32)·y = q·y - 32·y, both as 16-bit pair sums.
                let p16 = _mm256_sub_epi16(_mm256_maddubs_epi16(qk, y), _mm256_maddubs_epi16(m32s, y));
                // Scales 2s and 2s + 1 (s = 4j + k) for the two 16-value halves.
                let s = (4 * j + k) as i8;
                let sel = _mm_unpacklo_epi64(_mm_set1_epi8(2 * s), _mm_set1_epi8(2 * s + 1));
                let sc = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, sel));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sc, p16));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    hsum_ps(acc)
}

/// Dot product of F16 values (`w`) with F32 values (`x`), `w.len() == x.len()`.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_f16(w: &[f16], x: &[f32]) -> f32 {
    let n = w.len().min(x.len());
    let (wp, xp) = (w.as_ptr() as *const __m128i, x.as_ptr());
    let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i + 16 <= n {
        a0 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(wp.add(i / 8))),
            _mm256_loadu_ps(xp.add(i)),
            a0,
        );
        a1 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(wp.add(i / 8 + 1))),
            _mm256_loadu_ps(xp.add(i + 8)),
            a1,
        );
        i += 16;
    }
    if i + 8 <= n {
        a0 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(wp.add(i / 8))),
            _mm256_loadu_ps(xp.add(i)),
            a0,
        );
        i += 8;
    }
    let mut s = hsum_ps(_mm256_add_ps(a0, a1));
    while i < n {
        s += w[i].to_f32() * x[i];
        i += 1;
    }
    s
}

/// `out += s · v` for F16 `v`.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn axpy_f16(out: &mut [f32], s: f32, v: &[f16]) {
    let n = out.len().min(v.len());
    let (op, vp) = (out.as_mut_ptr(), v.as_ptr() as *const __m128i);
    let sv = _mm256_set1_ps(s);
    let mut i = 0;
    while i + 8 <= n {
        let y = _mm256_fmadd_ps(
            sv,
            _mm256_cvtph_ps(_mm_loadu_si128(vp.add(i / 8))),
            _mm256_loadu_ps(op.add(i)),
        );
        _mm256_storeu_ps(op.add(i), y);
        i += 8;
    }
    while i < n {
        out[i] += s * v[i].to_f32();
        i += 1;
    }
}

/// Widens `n` little-endian F16 values at `src` to F32 at `dst`.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn widen_f16(src: &[u8], dst: &mut [f32]) {
    let n = (src.len() / 2).min(dst.len());
    let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
    let mut i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(
            dp.add(i),
            _mm256_cvtph_ps(_mm_loadu_si128(sp.add(2 * i) as *const __m128i)),
        );
        i += 8;
    }
    while i < n {
        dst[i] = f16::from_le_bytes([src[2 * i], src[2 * i + 1]]).to_f32();
        i += 1;
    }
}

/// Dot product of two F32 slices of equal length.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (u, s) in acc.iter_mut().enumerate() {
            *s = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(i + 8 * u)),
                _mm256_loadu_ps(bp.add(i + 8 * u)),
                *s,
            );
        }
        i += 32;
    }
    while i + 8 <= n {
        acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc[0]);
        i += 8;
    }
    let mut s = hsum_ps(_mm256_add_ps(
        _mm256_add_ps(acc[0], acc[1]),
        _mm256_add_ps(acc[2], acc[3]),
    ));
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

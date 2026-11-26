//! Metal Shading Language kernels for the LLaMA executor.
//!
//! Activations are F32. Weights are F32, F16, Q8_0, Q4_0, Q4_K or Q6_K in their GGUF
//! block layout. Kernels assume `in_features % 32 == 0` (`% 256` for K-quants) and
//! `head_dim % 32 == 0` (checked on the host). Linear kernels write output rows with
//! stride `ldy`, so a fused matrix stored as segments of different formats can write
//! adjacent column ranges of one output.
//!
//! Linear kernels take an `accumulate` flag (`y += W·x`), used to fold residual
//! connections into the attention-output and down projections. Q/K/V and gate/up
//! projections are fused on the host, so attention-side kernels take row strides.

/// Kernel source, compiled at runtime with `newLibraryWithSource`.
pub const SHADERS_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct block_q8_0 { half d; char  qs[32]; };
struct block_q4_0 { half d; uchar qs[16]; };
struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };   // 256 values
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };  // 256 values

// ── weight access: 8 consecutive values (quarter q of 32-value block ib of a row) ──
struct WF32 {};
struct WQ8 {};
struct WQ4 {};
struct WF16 {};
struct WQ4K {};
struct WQ6K {};

// 6-bit scale and min of sub-block j (0..7) of a Q4_K super-block.
inline float2 scale_min_k4(uint j, device const uchar* q) {
    if (j < 4) return float2(q[j] & 63, q[j + 4] & 63);
    return float2((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WF32) {
    device const float* r = (device const float*)w + (ulong)row * k + ib * 32 + q * 8;
    for (uint j = 0; j < 8; ++j) out[j] = r[j];
}
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WF16) {
    device const half4* r = (device const half4*)((device const half*)w + (ulong)row * k + ib * 32 + q * 8);
    const float4 a = float4(r[0]), b = float4(r[1]);
    for (uint j = 0; j < 4; ++j) { out[j] = a[j]; out[4 + j] = b[j]; }
}
// Q8_0 / Q4_0: 2-byte loads (blocks are 2-byte aligned) instead of byte loads; the GEMM
// is 1.5-3% faster. Products stay F32 (exact: an 8-bit integer times an F16 scale).
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WQ8) {
    device const block_q8_0* b = (device const block_q8_0*)w + (ulong)row * (k / 32) + ib;
    device const char2* p = (device const char2*)(b->qs + q * 8);
    const float d = float(b->d);
    for (uint j = 0; j < 4; ++j) {
        const char2 c = p[j];
        out[2 * j] = d * float(c.x);
        out[2 * j + 1] = d * float(c.y);
    }
}
// Q4_0 nibbles are masked in place and scaled by d, d/16, d/256 or d/4096 (powers of
// two, so every product and the -8d offset are exact: bit-identical to shifting).
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WQ4) {
    device const block_q4_0* b = (device const block_q4_0*)w + (ulong)row * (k / 32) + ib;
    device const ushort* p = (device const ushort*)(b->qs + (q % 2) * 8);
    const float d = float(b->d);
    const float d1 = q < 2 ? d : d / 16.0f, d2 = d1 / 256.0f, md = -8.0f * d;
    const ushort mask0 = q < 2 ? 0x000F : 0x00F0, mask1 = mask0 << 8;
    for (uint j = 0; j < 4; ++j) {
        const ushort c = p[j];
        out[2 * j] = d1 * float(c & mask0) + md;
        out[2 * j + 1] = d2 * float(c & mask1) + md;
    }
}

// K-quants: `ib` is the 32-value block within the row; 8 per 256-value super-block.
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WQ4K) {
    device const block_q4_K* b = (device const block_q4_K*)w + (ulong)row * (k / 256) + ib / 8;
    const uint j = ib % 8;
    const float2 sm = scale_min_k4(j, b->scales);
    const float d = float(b->d) * sm.x, mn = float(b->dmin) * sm.y;
    // 8-byte aligned: blocks are 144 bytes, qs starts at 16.
    const uint2 v = *(device const uint2*)(b->qs + (j / 2) * 32 + q * 8);
    const uint sh = (j % 2) * 4;
    for (uint t = 0; t < 4; ++t) {
        out[t] = d * float((v.x >> (8 * t + sh)) & 0x0F) - mn;
        out[4 + t] = d * float((v.y >> (8 * t + sh)) & 0x0F) - mn;
    }
}
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WQ6K) {
    device const block_q6_K* b = (device const block_q6_K*)w + (ulong)row * (k / 256) + ib / 8;
    const uint p = (ib % 8) * 32 + q * 8;
    const uint hf = p / 128, g = (p % 128) / 32, l = p % 32;
    device const uchar* ql = b->ql + hf * 64 + (g & 1) * 32 + l;
    device const uchar* qh = b->qh + hf * 32 + l;
    const float d = float(b->d) * float(b->scales[hf * 8 + l / 16 + 2 * g]);
    // Blocks are 210 bytes, so only 2-byte alignment is guaranteed.
    const uint sh = 2 * g, lsh = (g >= 2) ? 4 : 0;
    device const ushort* l2 = (device const ushort*)ql;
    device const ushort* h2 = (device const ushort*)qh;
    for (uint t = 0; t < 4; ++t) {
        const uint lv = l2[t], hv = h2[t];
        out[2 * t]     = d * (float(((lv >> lsh) & 0x0F) | (((hv >> sh) & 3) << 4)) - 32.0f);
        out[2 * t + 1] = d * (float(((lv >> (8 + lsh)) & 0x0F) | (((hv >> (8 + sh)) & 3) << 4)) - 32.0f);
    }
}

// 16 consecutive values: half `h` (0 or 1) of 32-value block `ib` of a row. K-quants
// decode the sub-block scale once and Q4_K reads its 16 bytes in one load (the GEMM is
// 5% faster than with two `load8` calls).
template <typename T>
inline void load16(device const uchar* w, uint row, uint k, uint ib, uint h, thread float* out, T t) {
    load8(w, row, k, ib, 2 * h, out, t);
    load8(w, row, k, ib, 2 * h + 1, out + 8, t);
}
// Q4_0: nibbles are masked in place and scaled by d, d/16 or d/256, d/4096 (exact powers
// of two) instead of shifted (as llama.cpp does; the GEMM is 3-6% faster).
inline void load16(device const uchar* w, uint row, uint k, uint ib, uint h, thread float* out, WQ4) {
    device const block_q4_0* b = (device const block_q4_0*)w + (ulong)row * (k / 32) + ib;
    device const ushort* qs = (device const ushort*)b->qs;
    const float d = float(b->d);
    const float d1 = h ? d / 16.0f : d, d2 = d1 / 256.0f, md = -8.0f * d;
    const ushort mask0 = h ? 0x00F0 : 0x000F, mask1 = mask0 << 8;
    for (uint i = 0; i < 8; ++i) {
        const ushort c = qs[i];
        out[2 * i] = d1 * float(c & mask0) + md;
        out[2 * i + 1] = d2 * float(c & mask1) + md;
    }
}
inline void load16(device const uchar* w, uint row, uint k, uint ib, uint h, thread float* out, WQ4K) {
    device const block_q4_K* b = (device const block_q4_K*)w + (ulong)row * (k / 256) + ib / 8;
    const uint j = ib % 8;
    const float2 sm = scale_min_k4(j, b->scales);
    const float d = float(b->d) * sm.x, mn = float(b->dmin) * sm.y;
    const uint4 v = *(device const uint4*)(b->qs + (j / 2) * 32 + h * 16);  // 16-byte aligned
    const uint sh = (j % 2) * 4;
    for (uint t = 0; t < 4; ++t)
        for (uint u = 0; u < 4; ++u) out[4 * t + u] = d * float((v[t] >> (8 * u + sh)) & 0x0F) - mn;
}
inline void load16(device const uchar* w, uint row, uint k, uint ib, uint h, thread float* out, WQ6K) {
    device const block_q6_K* b = (device const block_q6_K*)w + (ulong)row * (k / 256) + ib / 8;
    const uint p = (ib % 8) * 32 + h * 16;
    const uint hf = p / 128, g = (p % 128) / 32, l = p % 32;
    device const ushort* l2 = (device const ushort*)(b->ql + hf * 64 + (g & 1) * 32 + l);
    device const ushort* h2 = (device const ushort*)(b->qh + hf * 32 + l);
    const float d = float(b->d) * float(b->scales[hf * 8 + l / 16 + 2 * g]);
    const uint sh = 2 * g, lsh = (g >= 2) ? 4 : 0;
    for (uint t = 0; t < 8; ++t) {
        const uint lv = l2[t], hv = h2[t];
        out[2 * t]     = d * (float(((lv >> lsh) & 0x0F) | (((hv >> sh) & 3) << 4)) - 32.0f);
        out[2 * t + 1] = d * (float(((lv >> (8 + lsh)) & 0x0F) | (((hv >> (8 + sh)) & 3) << 4)) - 32.0f);
    }
}

#define LINEAR_ARGS \
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]], constant uint& n [[buffer(3)]], \
    constant uint& k [[buffer(4)]], constant uint& m [[buffer(5)]], \
    constant uint& accumulate [[buffer(6)]], constant uint& ldy [[buffer(7)]]

// ── matrix-vector: y[m, n] (+)= x[m, k] · W[n, k]^T for small m (decode) ──
// Threadgroup = 2 simdgroups; each simdgroup computes 4 output rows.
template <typename T>
inline void mv_impl(device const uchar* w, device const float* x, device float* y,
                    uint n, uint k, uint m, uint acc, uint ldy, uint tg, uint sg, uint lane) {
    const uint nb = k / 32;
    const uint row0 = (tg * 2 + sg) * 4;
    const uint q = lane % 4;
    for (uint mi = 0; mi < m; ++mi) {
        device const float* xr = x + (ulong)mi * k;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint ib = lane / 4; ib < nb; ib += 8) {
            float xv[8];
            for (uint j = 0; j < 8; ++j) xv[j] = xr[ib * 32 + q * 8 + j];
            for (uint r = 0; r < 4; ++r) {
                const uint row = row0 + r;
                if (row < n) {
                    float wv[8];
                    load8(w, row, k, ib, q, wv, T());
                    float s = 0.0f;
                    for (uint j = 0; j < 8; ++j) s += wv[j] * xv[j];
                    sums[r] += s;
                }
            }
        }
        for (uint r = 0; r < 4; ++r) {
            const float t = simd_sum(sums[r]);
            const uint row = row0 + r;
            if (lane == 0 && row < n) {
                device float* o = y + (ulong)mi * ldy + row;
                *o = acc ? *o + t : t;
            }
        }
    }
}

#define MV_KERNEL(NAME, TAG) \
kernel void NAME(LINEAR_ARGS, uint tg [[threadgroup_position_in_grid]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    mv_impl<TAG>(w, x, y, n, k, m, accumulate, ldy, tg, sg, lane); }

MV_KERNEL(mv_f32, WF32)
MV_KERNEL(mv_f16, WF16)

MV_KERNEL(mv_q8_0, WQ8)

// ── Q8_0 matrix-vector with split K (after llama.cpp's kernel_mul_mv_q8_0_f32, MIT) ──
// One activation row (m = 1), one weight row per threadgroup; its 8 simdgroups split K
// (4 lanes per block, 8 values each, 64 blocks per step) and add their sums through
// threadgroup memory. Used for rows of at least 64 blocks (`MV_SPLIT_MIN_COLS` in
// llama.rs): per Llama-3.2-1B layer on an M4 9% faster than `mv_q8_0` (2 simdgroups x 4
// rows, every lane over all K); 2 or 4 rows per threadgroup and 2, 4 or 16 simdgroups
// were 1-15% slower. On SmolLM2-135M (576-value rows) most simdgroups would idle and
// decode was 20% slower.
constant constexpr uint MV8_SG = 8;
kernel void mvs_q8_0(LINEAR_ARGS, uint row [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[MV8_SG];
    const uint nb = k / 32;
    device const block_q8_0* ax = (device const block_q8_0*)w + (ulong)row * nb;
    const short il = lane % 4;
    const uint ib0 = sg * 8 + lane / 4;
    float sum = 0.0f;
    device const float* yb = x + ib0 * 32 + il * 8;
    for (uint ib = ib0; ib < nb; ib += MV8_SG * 8, yb += MV8_SG * 8 * 32) {
        const float4 y0 = ((device const float4*)yb)[0], y1 = ((device const float4*)yb)[1];
        device const char* qs = ax[ib].qs + il * 8;
        const float sq = dot(float4(qs[0], qs[1], qs[2], qs[3]), y0) + dot(float4(qs[4], qs[5], qs[6], qs[7]), y1);
        sum += sq * float(ax[ib].d);
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float s = 0.0f;
        for (uint g = 0; g < MV8_SG; ++g) s += red[g];
        y[row] = accumulate ? y[row] + s : s;
    }
}

// ── Q4_0 matrix-vector: two lanes per block, 16 blocks per simdgroup step ──
// Lane half `il` reads qs[il..il+8] (two-byte loads): low nibbles pair with
// x[il..il+8], high nibbles with x[16+il..16+il+8]. The -8 offset is applied once per
// block as 8 * sum(x). Activations are loaded once for the simdgroup's 4 rows.
kernel void mv_q4_0(LINEAR_ARGS, uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint nb = k / 32;
    const uint row0 = (tg * 2 + sg) * 4;
    const uint ix = lane / 2, il = (lane % 2) * 8;
    device const block_q4_0* wb = (device const block_q4_0*)w;
    for (uint mi = 0; mi < m; ++mi) {
        device const float* xr = x + (ulong)mi * k;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint ib = ix; ib < nb; ib += 16) {
            float xl[8], xh[8];
            device const float4* pl = (device const float4*)(xr + ib * 32 + il);
            device const float4* ph = (device const float4*)(xr + ib * 32 + 16 + il);
            const float4 l0 = pl[0], l1 = pl[1], h0 = ph[0], h1 = ph[1];
            for (uint j = 0; j < 4; ++j) { xl[j] = l0[j]; xl[4 + j] = l1[j]; xh[j] = h0[j]; xh[4 + j] = h1[j]; }
            const float sumy = (l0[0] + l0[1] + l0[2] + l0[3]) + (l1[0] + l1[1] + l1[2] + l1[3])
                             + (h0[0] + h0[1] + h0[2] + h0[3]) + (h1[0] + h1[1] + h1[2] + h1[3]);
            for (uint r = 0; r < 4; ++r) {
                const uint row = row0 + r;
                if (row >= n) break;
                device const block_q4_0* b = wb + (ulong)row * nb + ib;
                device const ushort* q = (device const ushort*)(b->qs + il);
                float acc = 0.0f;
                for (uint j = 0; j < 4; ++j) {
                    const ushort v = q[j];
                    acc += float(v & 0x000F) * xl[2 * j] + float((v >> 8) & 0x000F) * xl[2 * j + 1]
                         + float((v >> 4) & 0x000F) * xh[2 * j] + float(v >> 12) * xh[2 * j + 1];
                }
                sums[r] += float(b->d) * (acc - 8.0f * sumy);
            }
        }
        for (uint r = 0; r < 4; ++r) {
            const float t = simd_sum(sums[r]);
            const uint row = row0 + r;
            if (lane == 0 && row < n) {
                device float* o = y + (ulong)mi * ldy + row;
                *o = accumulate ? *o + t : t;
            }
        }
    }
}

// ── K-quant matrix-vector (ported from llama.cpp's kernel_mul_mv_q4_K/q6_K_f32, MIT) ──
// 2 simdgroups x 2 weight rows per threadgroup (`MVK_ROWS`; keep in sync with
// `mv_rows_per_group` in llama.rs). Per Llama-3.2-1B layer on an M4 these are 13% (Q4_K)
// and 4% (Q6_K) faster than the previous 4-rows-per-simdgroup kernels, which shifted
// each nibble out; 1 or 4 rows per simdgroup and 1 or 4 simdgroups were slower or equal.
// Rows past n re-read the last row; their outputs are not written.
constant constexpr uint MVK_ROWS = 2;

// Q4_K: 8 lanes per super-block, 4 super-blocks per step. Lane (iq, ir) covers values
// 8*ir..+8 of sub-blocks 2iq, 2iq+1, 2iq+4, 2iq+5. Nibbles are masked in place (q, 16q,
// 256q, 4096q) and rescaled after the sums; mins are applied through sums of x.
kernel void mv_q4_k(LINEAR_ARGS, uint tg [[threadgroup_position_in_grid]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const short ix = lane / 8, iq = (lane % 8) / 4, ir = lane % 4;
    const uint nb = k / 256;
    const uint first = (tg * 2 + sg) * MVK_ROWS;
    const uint rowu = nb * sizeof(block_q4_K) / 2;  // row stride in ushorts
    device const block_q4_K* xb = (device const block_q4_K*)w + (ulong)min(first, n - 1) * nb;
    for (uint mi = 0; mi < m; ++mi) {
        device const float* y4 = x + (ulong)mi * k + ix * 256 + 64 * iq + 8 * ir;
        float sumf[MVK_ROWS] = {0.0f, 0.0f};
        float yl[16], yh[16];
        for (uint ib = ix; ib < nb; ib += 4, y4 += 4 * 256) {
            float4 sumy = 0.0f;
            for (short i = 0; i < 8; ++i) {
                yl[i] = y4[i];
                yl[i + 8] = y4[i + 32];
                yh[i] = y4[i + 128];
                yh[i + 8] = y4[i + 160];
                sumy += float4(yl[i], yl[i + 8], yh[i], yh[i + 8]);
            }
            device const ushort* sc = (device const ushort*)xb[ib].scales + iq;
            device const ushort* q1 = (device const ushort*)xb[ib].qs + 16 * iq + 4 * ir;
            device const half* dh = &xb[ib].d;
            for (uint r = 0; r < MVK_ROWS; ++r) {
                // 6-bit scales and mins of sub-blocks 2iq, 2iq+1 (bytes 0-3) and 2iq+4, 2iq+5 (4-7).
                ushort sc16[4];
                sc16[0] = sc[0] & 0x3f3f;
                sc16[1] = sc[2] & 0x3f3f;
                sc16[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
                sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
                thread const uchar* sc8 = (thread const uchar*)sc16;
                device const ushort* q2 = q1 + 32;
                float4 acc1 = 0.0f, acc2 = 0.0f;
                for (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }
                sumf[r] += float(dh[0]) * ((acc1[0] + acc1[1] / 256.0f) * sc8[0] + (acc1[2] + acc1[3] / 256.0f) * sc8[1] / 16.0f +
                                           (acc2[0] + acc2[1] / 256.0f) * sc8[4] + (acc2[2] + acc2[3] / 256.0f) * sc8[5] / 16.0f) -
                           float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                const uint step = first + r + 1 < n ? rowu : 0;
                q1 += step;
                sc += step;
                dh += step;
            }
        }
        for (uint r = 0; r < MVK_ROWS; ++r) {
            const float s = simd_sum(sumf[r]);
            if (lane == 0 && first + r < n) {
                device float* o = y + (ulong)mi * ldy + first + r;
                *o = accumulate ? *o + s : s;
            }
        }
    }
}

// Q6_K: 16 lanes per super-block, 2 super-blocks per step. Lane covers values l0..l0+4
// (l0 = 4*(tid%8)) of the four 32-value groups of half ip = tid/8 (tid = lane/2).
kernel void mv_q6_k(LINEAR_ARGS, uint tg [[threadgroup_position_in_grid]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint nb = k / 256;
    const uint first = (tg * 2 + sg) * MVK_ROWS;
    const uint rowb = nb * sizeof(block_q6_K);
    device const block_q6_K* xb = (device const block_q6_K*)w + (ulong)min(first, n - 1) * nb;
    const short tid = lane / 2, ix = lane % 2, ip = tid / 8, l0 = 4 * (tid % 8), is = 8 * ip + l0 / 16;
    for (uint mi = 0; mi < m; ++mi) {
        device const float* yy = x + (ulong)mi * k + 128 * ip + l0;
        float sumf[MVK_ROWS] = {0.0f, 0.0f};
        float yl[16];
        for (uint i = ix; i < nb; i += 2) {
            device const uchar* q1 = xb[i].ql + 64 * ip + l0;
            device const uchar* qh = xb[i].qh + 32 * ip + l0;
            device const char* sc = xb[i].scales + is;
            device const half* dh = &xb[i].d;
            device const float* yp = yy + i * 256;
            for (short l = 0; l < 4; ++l) {
                yl[4 * l + 0] = yp[l + 0];
                yl[4 * l + 1] = yp[l + 32];
                yl[4 * l + 2] = yp[l + 64];
                yl[4 * l + 3] = yp[l + 96];
            }
            for (uint r = 0; r < MVK_ROWS; ++r) {
                float4 sums = 0.0f;
                for (short l = 0; l < 4; ++l) {
                    const uchar a = q1[l], b = q1[l + 32], h = qh[l];
                    sums[0] += yl[4 * l + 0] * (float((a & 0xF) | ((h & 0x03) << 4)) - 32.0f);
                    sums[1] += yl[4 * l + 1] * (float((b & 0xF) | ((h & 0x0C) << 2)) - 32.0f);
                    sums[2] += yl[4 * l + 2] * (float((a >> 4) | (h & 0x30)) - 32.0f);
                    sums[3] += yl[4 * l + 3] * (float((b >> 4) | ((h & 0xC0) >> 2)) - 32.0f);
                }
                sumf[r] += float(dh[0]) * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
                const uint step = first + r + 1 < n ? rowb : 0;
                q1 += step;
                qh += step;
                sc += step;
                dh += step / 2;
            }
        }
        for (uint r = 0; r < MVK_ROWS; ++r) {
            const float s = simd_sum(sumf[r]);
            if (lane == 0 && first + r < n) {
                device float* o = y + (ulong)mi * ldy + first + r;
                *o = accumulate ? *o + s : s;
            }
        }
    }
}

// ── small-batch matrix-vector: 2..8 activation rows share each decoded weight chunk ──
// Same thread layout as the single-row kernels (2 simdgroups x 4 weight rows), but each
// lane decodes a chunk of 8 weight values once and applies it to all `m <= NR`
// activation rows, so weights are read and unpacked once per call rather than once per
// row. A format describes its lane layout: `LPB` lanes per block of `BV` values, each
// lane owning `C` chunks of 8 values; `dec` returns chunk `c` dequantized and its
// offset within the activation row.
struct MvF32  { enum : uint { BV = 32, LPB = 4, C = 1 }; };
struct MvF16  { enum : uint { BV = 32, LPB = 4, C = 1 }; };
struct MvQ8   { enum : uint { BV = 32, LPB = 4, C = 1 }; };
struct MvQ4   { enum : uint { BV = 32, LPB = 2, C = 2 }; };
struct MvQ4K  { enum : uint { BV = 256, LPB = 8, C = 4 }; };
struct MvQ6K  { enum : uint { BV = 256, LPB = 8, C = 4 }; };

inline uint mv_xoff(uint ib, uint il, uint c, MvF32) { return ib * 32 + il * 8; }
inline uint mv_xoff(uint ib, uint il, uint c, MvF16) { return ib * 32 + il * 8; }
inline uint mv_xoff(uint ib, uint il, uint c, MvQ8)  { return ib * 32 + il * 8; }
inline uint mv_xoff(uint ib, uint il, uint c, MvQ4)  { return ib * 32 + c * 16 + il * 8; }
inline uint mv_xoff(uint ib, uint il, uint c, MvQ4K) { return ib * 256 + (il / 2) * 64 + (c / 2) * 32 + (il % 2) * 16 + (c % 2) * 8; }
inline uint mv_xoff(uint ib, uint il, uint c, MvQ6K) { return ib * 256 + (il / 4) * 128 + c * 32 + (il % 4) * 8; }

inline uint mv_dec(device const uchar* w, uint row, uint nb, uint ib, uint il, uint c, thread float4& w0, thread float4& w1, MvF32) {
    device const float4* r = (device const float4*)((device const float*)w + ((ulong)row * nb + ib) * 32 + il * 8);
    w0 = r[0]; w1 = r[1];
    return ib * 32 + il * 8;
}
inline uint mv_dec(device const uchar* w, uint row, uint nb, uint ib, uint il, uint c, thread float4& w0, thread float4& w1, MvF16) {
    device const half4* r = (device const half4*)((device const half*)w + ((ulong)row * nb + ib) * 32 + il * 8);
    w0 = float4(r[0]); w1 = float4(r[1]);
    return ib * 32 + il * 8;
}
inline uint mv_dec(device const uchar* w, uint row, uint nb, uint ib, uint il, uint c, thread float4& w0, thread float4& w1, MvQ8) {
    device const block_q8_0* b = (device const block_q8_0*)w + (ulong)row * nb + ib;
    const float d = float(b->d);
    device const char* q = b->qs + il * 8;
    w0 = d * float4(q[0], q[1], q[2], q[3]);
    w1 = d * float4(q[4], q[5], q[6], q[7]);
    return ib * 32 + il * 8;
}
// Lane half `il` owns bytes il*8..il*8+8: low nibbles (c = 0) are values il*8.., high
// nibbles (c = 1) values 16+il*8...
inline uint mv_dec(device const uchar* w, uint row, uint nb, uint ib, uint il, uint c, thread float4& w0, thread float4& w1, MvQ4) {
    device const block_q4_0* b = (device const block_q4_0*)w + (ulong)row * nb + ib;
    const float d = float(b->d);
    device const ushort* q = (device const ushort*)(b->qs + il * 8);  // 2-byte aligned
    const uint2 v = uint2(uint(q[0]) | (uint(q[1]) << 16), uint(q[2]) | (uint(q[3]) << 16));
    const uint sh = c * 4;
    for (uint t = 0; t < 4; ++t) {
        w0[t] = d * (float((v.x >> (8 * t + sh)) & 0x0F) - 8.0f);
        w1[t] = d * (float((v.y >> (8 * t + sh)) & 0x0F) - 8.0f);
    }
    return ib * 32 + c * 16 + il * 8;
}
// Lane t = 2*cc + h covers bytes cc*32 + h*16 .. +16 of qs: chunk c = 2*nib + part takes
// bytes part*8..part*8+8 of those, low (nib 0, sub-block 2cc) or high nibbles (2cc+1).
inline uint mv_dec(device const uchar* w, uint row, uint nb, uint ib, uint il, uint c, thread float4& w0, thread float4& w1, MvQ4K) {
    device const block_q4_K* b = (device const block_q4_K*)w + (ulong)row * nb + ib;
    const uint cc = il / 2, h = il % 2, nib = c / 2, part = c % 2;
    const float2 sm = scale_min_k4(2 * cc + nib, b->scales);
    const float d = float(b->d) * sm.x, mn = float(b->dmin) * sm.y;
    const uint2 v = *(device const uint2*)(b->qs + cc * 32 + h * 16 + part * 8);
    const uint sh = nib * 4;
    for (uint t = 0; t < 4; ++t) {
        w0[t] = d * float((v.x >> (8 * t + sh)) & 0x0F) - mn;
        w1[t] = d * float((v.y >> (8 * t + sh)) & 0x0F) - mn;
    }
    return ib * 256 + cc * 64 + nib * 32 + h * 16 + part * 8;
}
// Lane t covers half hf = t/4, positions l0..l0+8 (l0 = 8*(t%4)) of 32-value group g = c.
inline uint mv_dec(device const uchar* w, uint row, uint nb, uint ib, uint il, uint c, thread float4& w0, thread float4& w1, MvQ6K) {
    device const block_q6_K* b = (device const block_q6_K*)w + (ulong)row * nb + ib;
    const uint hf = il / 4, l0 = (il % 4) * 8;
    device const uchar* ql = b->ql + hf * 64 + (c & 1) * 32 + l0;
    device const uchar* qh = b->qh + hf * 32 + l0;
    const float d = float(b->d) * float(b->scales[hf * 8 + l0 / 16 + 2 * c]);
    const uint lsh = (c >= 2) ? 4 : 0, hsh = 2 * c;
    for (uint t = 0; t < 4; ++t) {
        w0[t] = d * (float(((ql[t] >> lsh) & 0x0F) | (((qh[t] >> hsh) & 3) << 4)) - 32.0f);
        w1[t] = d * (float(((ql[4 + t] >> lsh) & 0x0F) | (((qh[4 + t] >> hsh) & 3) << 4)) - 32.0f);
    }
    return ib * 256 + hf * 128 + c * 32 + l0;
}

template <typename F, uint NR>
kernel void mvn(LINEAR_ARGS, uint tg [[threadgroup_position_in_grid]],
                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint nb = k / F::BV;
    const uint row0 = (tg * 2 + sg) * 4;
    const uint il = lane % F::LPB;
    float sums[NR][4];
    for (uint mi = 0; mi < NR; ++mi)
        for (uint r = 0; r < 4; ++r) sums[mi][r] = 0.0f;
    for (uint ib = lane / F::LPB; ib < nb; ib += 32 / F::LPB) {
        for (uint c = 0; c < F::C; ++c) {
            const uint xo = mv_xoff(ib, il, c, F());
            // Activations for this chunk are loaded once per group of (up to) 4 rows and
            // reused for all 4 weight rows; more than 4 rows in registers spills.
            for (uint g = 0; g < NR; g += 4) {
                constexpr uint G = NR < 4 ? NR : 4;
                float4 xa[G], xb[G];
                for (uint i = 0; i < G; ++i) {
                    const uint mr = min(g + i, m - 1);  // rows past m repeat the last (discarded)
                    device const float4* xp = (device const float4*)(x + (ulong)mr * k + xo);
                    xa[i] = xp[0];
                    xb[i] = xp[1];
                }
                for (uint r = 0; r < 4; ++r) {
                    const uint row = row0 + r;
                    if (row < n) {
                        float4 w0, w1;
                        mv_dec(w, row, nb, ib, il, c, w0, w1, F());
                        for (uint i = 0; i < G; ++i) sums[g + i][r] += dot(w0, xa[i]) + dot(w1, xb[i]);
                    }
                }
            }
        }
    }
    for (uint mi = 0; mi < NR; ++mi) {
        if (mi >= m) break;
        for (uint r = 0; r < 4; ++r) {
            const float t = simd_sum(sums[mi][r]);
            const uint row = row0 + r;
            if (lane == 0 && row < n) {
                device float* o = y + (ulong)mi * ldy + row;
                *o = accumulate ? *o + t : t;
            }
        }
    }
}

typedef decltype(mvn<MvF32, 2>) mvn_t;
#define MVN(F, SUF, NR) template [[host_name("mv" #NR "_" SUF)]] kernel mvn_t mvn<F, NR>;
#define MVN_ALL(F, SUF) MVN(F, SUF, 2) MVN(F, SUF, 4) MVN(F, SUF, 8)
MVN_ALL(MvF32, "f32")
MVN_ALL(MvF16, "f16")
MVN_ALL(MvQ8, "q8_0")
MVN_ALL(MvQ4, "q4_0")
MVN_ALL(MvQ4K, "q4_k")
MVN_ALL(MvQ6K, "q6_k")

// ── few-row GEMM for 2-8 activation rows on simdgroup matrices (Q8_0 / Q4_0) ──
// Ported from llama.cpp's `kernel_mul_mv_mma_blk` (MIT). A threadgroup computes 8*NT
// weight rows x 8*RT activation rows; its NSG simdgroups split K (simdgroup s takes
// blocks s, s + NSG, ...) and add their partial tiles through threadgroup memory at the
// end. Each lane fills its two elements of every 8x8 fragment straight from registers,
// with no threadgroup staging: the weight fragment holds raw quants as exact halves
// (`1024 + q` built by OR-ing q into the bits of 1024.0h, minus the bias), the block
// scale is applied once per block to the product, and the activation fragment reads F32
// activations in the k order that the quant layout gives (so `x` needs no conversion to
// half). The next block's quants and activations load before the current block's
// products. In an 8x8 fragment lane l holds row fm, columns fn and fn + 1.
// Per Llama-3.2-1B layer on an M4: Q4_0 is 8% faster than `mv2` at 2 rows, 38% faster
// than `mv4` at 3-4 and 27% faster than `mms` at 8; Q8_0 is equal at 2 and 6-13% faster
// at 3-8. One row keeps `mv` (as fast for Q8_0, 4% faster for Q4_0). The kernel takes
// RT 8-row activation tiles; `RT = 2` (9-16 rows) was faster than `mmh` alone, but not
// than `mmh` with split-K, so only `RT = 1` is instantiated.
inline half2 mvm_1024_plus(ushort2 q) { return as_type<half2>(q | ushort2(0x6400)); }

// Q4_0: lane (row fm, column fn) holds qs ushorts fn, fn + 1. A high nibble stays in
// place as 16*q, so the activations it multiplies (b1) are divided by 16. Activation lane
// k = fm holds values 2k, 2k + 1 (b0, low nibbles) and 2k + 16, 2k + 17 (b1) of a block;
// step s takes b1 for odd s and the .y value for s >= 2.
struct MvmQ4 {
    typedef block_q4_0 block;
    typedef ushort2 quants;
    enum { b1 = 8 };   // float2 offset of b1 within a block of activations
    static short a_off(short fn) { return 1 + fn; }
    static short b_off(short fm) { return 2 * fm; }
    static quants load(device const ushort* qs) { return ushort2(qs[0], qs[1]); }
    static quants prep(quants q) { return q; }
    static float2 prep_b1(float2 v) { return v * (1.0f / 16); }
    static bool b1_step(short s) { return s % 2 != 0; }
    static bool y_step(short s) { return s >= 2; }
    static half2 frag(quants q, short s) {
        const ushort2 qq = s < 2 ? q : q >> 8;
        return s % 2 == 0 ? mvm_1024_plus(qq & ushort2(0x000F)) - half2(1024.0h + 8.0h)
                          : mvm_1024_plus(qq & ushort2(0x00F0)) - half2(1024.0h + 128.0h);
    }
};
// Q8_0: lane (row fm, column fn) holds qs bytes 2fn .. 2fn + 7; flipping the sign bit of
// a byte gives the unsigned q + 128. Activation lane k = fm holds values b, b + 1 (b0) and
// b + 4, b + 5 (b1) of a block, b = 8 (k / 2) + 2 (k % 2).
struct MvmQ8 {
    typedef block_q8_0 block;
    typedef ushort4 quants;
    enum { b1 = 2 };
    static short a_off(short fn) { return 1 + 2 * fn; }
    static short b_off(short fm) { return 8 * (fm / 2) + 2 * (fm % 2); }
    static quants load(device const ushort* qs) { return ushort4(qs[0], qs[1], qs[2], qs[3]); }
    static quants prep(quants q) { return q ^ ushort4(0x8080); }
    static float2 prep_b1(float2 v) { return v; }
    static bool b1_step(short s) { return s >= 2; }
    static bool y_step(short s) { return s % 2 != 0; }
    static half2 frag(quants q, short s) {
        const ushort2 w = s < 2 ? q.xy : q.zw;
        const ushort2 qq = s % 2 == 0 ? w : w >> 8;
        return mvm_1024_plus(qq & ushort2(0x00FF)) - half2(1024.0h + 128.0h);
    }
};

// Sums the K slices of the NSG simdgroups of an `mvm`-style kernel and writes the
// 8*NT x 8*RT output tile; element idx of a tile is lane idx/2's element idx%2.
template <short NT, short RT, short NSG>
inline void mvm_store(thread float (&acc)[RT][NT][2], threadgroup float* red, device float* y, uint n, uint m,
                      uint accumulate, uint ldy, uint i01, uint i11, ushort lane, ushort sg) {
    for (short rt = 0; rt < RT; ++rt)
        for (short t = 0; t < NT; ++t) {
            red[((sg * RT + rt) * NT + t) * 64 + 2 * lane] = acc[rt][t][0];
            red[((sg * RT + rt) * NT + t) * 64 + 2 * lane + 1] = acc[rt][t][1];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short idx = sg * 32 + lane; idx < RT * NT * 64; idx += NSG * 32) {
        float sum = 0.0f;
        for (short s = 0; s < NSG; ++s) sum += red[s * (RT * NT * 64) + idx];
        const short rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2;
        const uint r0 = i01 + 8 * t + (((l / 4) & 4) + ((l / 2) % 4));
        const uint r1 = i11 + 8 * rt + (((l / 4) & 2) * 2 + (l % 2) * 2) + idx % 2;
        if (r0 < n && r1 < m) {
            device float* o = y + (ulong)r1 * ldy + r0;
            *o = accumulate ? *o + sum : sum;
        }
    }
}

template <short NT, short RT, short NSG, typename Q>
kernel void mvm(LINEAR_ARGS, uint2 tgp [[threadgroup_position_in_grid]],
                ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[NSG * RT * NT * 64];
    const short fm = ((lane / 4) & 4) + ((lane / 2) % 4), fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint i01 = tgp.x * (8 * NT), i11 = tgp.y * (8 * RT);
    const uint nb = k / 32;
    constexpr short us_blk = sizeof(typename Q::block) / 2;
    // Rows past n (weights) and m (activations) repeat the last row; their outputs are
    // not written.
    device const ushort* xq[NT];
    for (short t = 0; t < NT; ++t) {
        const uint r = min(i01 + 8 * t + fm, n - 1);
        xq[t] = (device const ushort*)(w + (ulong)r * nb * sizeof(typename Q::block)) + Q::a_off(fn);
    }
    device const float2* yp[RT][2];
    for (short rt = 0; rt < RT; ++rt)
        for (short e = 0; e < 2; ++e)
            yp[rt][e] = (device const float2*)(x + (ulong)min(i11 + 8 * rt + fn + e, m - 1) * k + Q::b_off(fm));
    float acc[RT][NT][2];
    for (short rt = 0; rt < RT; ++rt)
        for (short t = 0; t < NT; ++t) acc[rt][t][0] = acc[rt][t][1] = 0.0f;
    typename Q::quants q[NT];
    float d[NT];
    float2 b0[RT][2], b1[RT][2];
    const uint ib0 = min((uint)sg, nb - 1);
    for (short t = 0; t < NT; ++t) {
        device const ushort* qs = xq[t] + ib0 * us_blk;
        q[t] = Q::load(qs);
        d[t] = float(as_type<half>(*(qs - Q::a_off(fn))));
    }
    for (short rt = 0; rt < RT; ++rt)
        for (short e = 0; e < 2; ++e) {
            b0[rt][e] = yp[rt][e][ib0 * 16];
            b1[rt][e] = yp[rt][e][ib0 * 16 + Q::b1];
        }
    for (uint ib = sg; ib < nb; ib += NSG) {
        typename Q::quants qc[NT];
        float dc[NT];
        float2 b0c[RT][2], b1c[RT][2];
        for (short t = 0; t < NT; ++t) {
            qc[t] = Q::prep(q[t]);
            dc[t] = d[t];
        }
        for (short rt = 0; rt < RT; ++rt)
            for (short e = 0; e < 2; ++e) {
                b0c[rt][e] = b0[rt][e];
                b1c[rt][e] = Q::prep_b1(b1[rt][e]);
            }
        const uint ibn = min(ib + NSG, nb - 1);
        for (short t = 0; t < NT; ++t) {
            device const ushort* qs = xq[t] + ibn * us_blk;
            q[t] = Q::load(qs);
            d[t] = float(as_type<half>(*(qs - Q::a_off(fn))));
        }
        for (short rt = 0; rt < RT; ++rt)
            for (short e = 0; e < 2; ++e) {
                b0[rt][e] = yp[rt][e][ibn * 16];
                b1[rt][e] = yp[rt][e][ibn * 16 + Q::b1];
            }
        simdgroup_float8x8 mp[RT][NT];
        for (short rt = 0; rt < RT; ++rt)
            for (short t = 0; t < NT; ++t) mp[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (short s = 0; s < 4; ++s) {
            simdgroup_float8x8 mb[RT];
            for (short rt = 0; rt < RT; ++rt) {
                const float2 v0 = Q::b1_step(s) ? b1c[rt][0] : b0c[rt][0];
                const float2 v1 = Q::b1_step(s) ? b1c[rt][1] : b0c[rt][1];
                mb[rt].thread_elements()[0] = Q::y_step(s) ? v0.y : v0.x;
                mb[rt].thread_elements()[1] = Q::y_step(s) ? v1.y : v1.x;
            }
            for (short t = 0; t < NT; ++t) {
                const half2 h = Q::frag(qc[t], s);
                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = h.x;
                ma.thread_elements()[1] = h.y;
                for (short rt = 0; rt < RT; ++rt) simdgroup_multiply_accumulate(mp[rt][t], ma, mb[rt], mp[rt][t]);
            }
        }
        for (short t = 0; t < NT; ++t)
            for (short rt = 0; rt < RT; ++rt) {
                acc[rt][t][0] = fma(dc[t], mp[rt][t].thread_elements()[0], acc[rt][t][0]);
                acc[rt][t][1] = fma(dc[t], mp[rt][t].thread_elements()[1], acc[rt][t][1]);
            }
    }
    mvm_store<NT, RT, NSG>(acc, red, y, n, m, accumulate, ldy, i01, i11, lane, sg);
}
// Per threadgroup: Q4_0 two 8-row weight tiles and 8 simdgroups, Q8_0 one tile and 16
// simdgroups (swept on an M4 over 1, 2 or 4 tiles and 8 or 16 simdgroups, then end to end
// for Q8_0). Keep in sync with `mvm_shape` in llama.rs.
typedef decltype(mvm<2, 1, 8, MvmQ4>) mvm_t;
template [[host_name("mvm_q4_0")]] kernel mvm_t mvm<2, 1, 8, MvmQ4>;
template [[host_name("mvm_q8_0")]] kernel mvm_t mvm<1, 1, 16, MvmQ8>;

// ── few-row MMA GEMM for K-quants (2-8 activation rows) ──
// Same scheme as `mvm`: one 64-weight unit per simdgroup step, quants as exact halves,
// scales applied to each unit's MMA result, K split over NSG simdgroups, the next unit
// prefetched. Q4_K is adapted from llama.cpp's `kernel_mul_mv_mma_q5_K_f32` (MIT; Q4_K
// is Q5_K without the high bits): a unit is a pair of 32-value sub-blocks sharing qs
// bytes (low / high nibble), lane k-index j of MMA step s holds byte
// 8*(j/2) + 2*(j%2) + 4*(s/2) + s%2 of the pair; mins are removed with per-sub-block
// activation sums.
inline float2 mvk_scale_min(thread const uint* w, short j) {
    if (j < 4) return float2((w[0] >> 8 * j) & 63, (w[1] >> 8 * j) & 63);
    const short s = 8 * (j - 4);
    return float2(((w[2] >> s) & 0xF) | (((w[0] >> (s + 6)) & 3) << 4),
                  ((w[2] >> (s + 4)) & 0xF) | (((w[1] >> (s + 6)) & 3) << 4));
}
struct MvkQ4K { uint2 q; uint sc[3]; uint dm; };
template <short NT>
inline void mvk_load(device const block_q4_K* const* xr, uint ip, short fn, thread MvkQ4K* a) {
    for (short t = 0; t < NT; ++t) {
        device const block_q4_K* xb = xr[t] + ip / 4;
        device const uint* sp = (device const uint*)xb->scales;
        a[t].q = *((device const uint2*)(xb->qs + 32 * (ip % 4)) + fn / 2);
        a[t].sc[0] = sp[0];
        a[t].sc[1] = sp[1];
        a[t].sc[2] = sp[2];
        a[t].dm = *(device const uint*)xb;
    }
}
// xs[rt][e][h]: values b, b+1 and b+4, b+5 (b = 8*(fm/2) + 2*(fm%2)) of sub-block h.
template <short RT>
inline void mvk_load_x(device const float2* const (*yp)[2], uint ip, thread float2 (*xs)[2][2][2]) {
    for (short rt = 0; rt < RT; ++rt)
        for (short e = 0; e < 2; ++e)
            for (short h = 0; h < 2; ++h) {
                xs[rt][e][h][0] = yp[rt][e][ip * 32 + h * 16];
                xs[rt][e][h][1] = yp[rt][e][ip * 32 + h * 16 + 2];
            }
}
template <short NT, short RT, short NSG>
kernel void mvm_q4k(LINEAR_ARGS, uint2 tgp [[threadgroup_position_in_grid]],
                    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[NSG * RT * NT * 64];
    const short fm = ((lane / 4) & 4) + ((lane / 2) % 4), fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint i01 = tgp.x * (8 * NT), i11 = tgp.y * (8 * RT);
    device const block_q4_K* xr[NT];
    for (short t = 0; t < NT; ++t) xr[t] = (device const block_q4_K*)w + (ulong)min(i01 + 8 * t + fm, n - 1) * (k / 256);
    device const float2* yp[RT][2];
    for (short rt = 0; rt < RT; ++rt)
        for (short e = 0; e < 2; ++e)
            yp[rt][e] = (device const float2*)(x + (ulong)min(i11 + 8 * rt + fn + e, m - 1) * k + 8 * (fm / 2) + 2 * (fm % 2));
    float acc[RT][NT][2];
    for (short rt = 0; rt < RT; ++rt)
        for (short t = 0; t < NT; ++t) acc[rt][t][0] = acc[rt][t][1] = 0.0f;
    const uint np = k / 64;
    MvkQ4K an[NT];
    float2 xn[RT][2][2][2];
    const uint ip0 = min((uint)sg, np - 1);
    mvk_load<NT>(xr, ip0, fn, an);
    mvk_load_x<RT>(yp, ip0, xn);
    for (uint ip = sg; ip < np; ip += NSG) {
        MvkQ4K ac[NT];
        float2 xs[RT][2][2][2];
        for (short t = 0; t < NT; ++t) ac[t] = an[t];
        for (short rt = 0; rt < RT; ++rt)
            for (short e = 0; e < 2; ++e)
                for (short h = 0; h < 2; ++h) {
                    xs[rt][e][h][0] = xn[rt][e][h][0];
                    xs[rt][e][h][1] = xn[rt][e][h][1];
                }
        const uint ipn = min(ip + NSG, np - 1);
        mvk_load<NT>(xr, ipn, fn, an);
        mvk_load_x<RT>(yp, ipn, xn);
        // Activation sums per sub-block (over the 8 lanes sharing fn).
        float c[RT][2][2];
        for (short rt = 0; rt < RT; ++rt)
            for (short e = 0; e < 2; ++e)
                for (short h = 0; h < 2; ++h) {
                    float u = xs[rt][e][h][0].x + xs[rt][e][h][0].y + xs[rt][e][h][1].x + xs[rt][e][h][1].y;
                    u += simd_shuffle_xor(u, 2);
                    u += simd_shuffle_xor(u, 4);
                    u += simd_shuffle_xor(u, 16);
                    c[rt][e][h] = u;
                }
        for (short t = 0; t < NT; ++t) {
            const float2 dm = float2(as_type<half2>(ac[t].dm));
            // a[h][word][odd bytes]: low nibbles q, high nibbles 16q.
            half2 a[2][2][2];
            for (short wd = 0; wd < 2; ++wd) {
                const ushort2 qw = as_type<ushort2>(wd ? ac[t].q.y : ac[t].q.x);
                a[0][wd][0] = mvm_1024_plus(qw & ushort2(0x000F)) - half2(1024.0h);
                a[0][wd][1] = mvm_1024_plus((qw >> 8) & ushort2(0x000F)) - half2(1024.0h);
                a[1][wd][0] = mvm_1024_plus(qw & ushort2(0x00F0)) - half2(1024.0h);
                a[1][wd][1] = mvm_1024_plus((qw >> 8) & ushort2(0x00F0)) - half2(1024.0h);
            }
            for (short h = 0; h < 2; ++h) {
                simdgroup_float8x8 mp[RT];
                for (short rt = 0; rt < RT; ++rt) mp[rt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                for (short s = 0; s < 4; ++s) {
                    simdgroup_half8x8 ma;
                    ma.thread_elements()[0] = a[h][s / 2][s % 2].x;
                    ma.thread_elements()[1] = a[h][s / 2][s % 2].y;
                    for (short rt = 0; rt < RT; ++rt) {
                        simdgroup_float8x8 mb;
                        mb.thread_elements()[0] = s % 2 == 0 ? xs[rt][0][h][s / 2].x : xs[rt][0][h][s / 2].y;
                        mb.thread_elements()[1] = s % 2 == 0 ? xs[rt][1][h][s / 2].x : xs[rt][1][h][s / 2].y;
                        simdgroup_multiply_accumulate(mp[rt], ma, mb, mp[rt]);
                    }
                }
                const float2 sm = mvk_scale_min(ac[t].sc, 2 * (ip % 4) + h);
                const float ds = dm.x * sm.x * (h == 0 ? 1.0f : 1.0f / 16), dmn = dm.y * sm.y;
                for (short rt = 0; rt < RT; ++rt) {
                    acc[rt][t][0] = fma(ds, mp[rt].thread_elements()[0], fma(-dmn, c[rt][0][h], acc[rt][t][0]));
                    acc[rt][t][1] = fma(ds, mp[rt].thread_elements()[1], fma(-dmn, c[rt][1][h], acc[rt][t][1]));
                }
            }
        }
    }
    mvm_store<NT, RT, NSG>(acc, red, y, n, m, accumulate, ldy, i01, i11, lane, sg);
}

// Q6_K: unit ip is groups g = ip%2 and g + 2 of half (ip/2)%2 of super-block ip/4, which
// share ql bytes (low / high nibble) and qh bytes (bits 2g, 2g + 4). Lane k-index j of MMA
// step s holds byte 16*(s/2) + s%2 + 2*j of the group, so steps 0-1 and 2-3 each stay in
// one 16-value scale group. Quants minus 32 (times 16 for high nibbles) are exact halves.
struct MvkQ6K { ushort2 l[2], h[2]; char2 s0, s1; half d; };
template <short NT>
inline void mvk_load(device const block_q6_K* const* xr, uint ip, short fn, thread MvkQ6K* a) {
    const uint hf = (ip / 2) % 2, g = ip % 2;
    for (short t = 0; t < NT; ++t) {
        device const block_q6_K* xb = xr[t] + ip / 4;
        // Blocks are 210 bytes: 2-byte loads.
        device const ushort* ql = (device const ushort*)(xb->ql + hf * 64 + g * 32) + fn;
        device const ushort* qh = (device const ushort*)(xb->qh + hf * 32) + fn;
        a[t].l[0] = ushort2(ql[0], ql[1]);
        a[t].l[1] = ushort2(ql[8], ql[9]);
        a[t].h[0] = ushort2(qh[0], qh[1]);
        a[t].h[1] = ushort2(qh[8], qh[9]);
        device const char2* sc = (device const char2*)(xb->scales + hf * 8 + 2 * g);
        a[t].s0 = sc[0];
        a[t].s1 = sc[2];
        a[t].d = xb->d;
    }
}
// xs[rt][e][h][q]: values 2*fm, 2*fm + 1 of 16-value quarter q of group g + 2h.
template <short RT>
inline void mvk6_load_x(device const float* const (*yp)[2], uint ip, thread float2 (*xs)[2][2][2]) {
    const uint base = (ip / 4) * 256 + ((ip / 2) % 2) * 128 + (ip % 2) * 32;
    for (short rt = 0; rt < RT; ++rt)
        for (short e = 0; e < 2; ++e)
            for (short h = 0; h < 2; ++h) {
                xs[rt][e][h][0] = *(device const float2*)(yp[rt][e] + base + 64 * h);
                xs[rt][e][h][1] = *(device const float2*)(yp[rt][e] + base + 64 * h + 16);
            }
}
template <short NT, short RT, short NSG>
kernel void mvm_q6k(LINEAR_ARGS, uint2 tgp [[threadgroup_position_in_grid]],
                    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[NSG * RT * NT * 64];
    const short fm = ((lane / 4) & 4) + ((lane / 2) % 4), fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint i01 = tgp.x * (8 * NT), i11 = tgp.y * (8 * RT);
    device const block_q6_K* xr[NT];
    for (short t = 0; t < NT; ++t) xr[t] = (device const block_q6_K*)w + (ulong)min(i01 + 8 * t + fm, n - 1) * (k / 256);
    device const float* yp[RT][2];
    for (short rt = 0; rt < RT; ++rt)
        for (short e = 0; e < 2; ++e) yp[rt][e] = x + (ulong)min(i11 + 8 * rt + fn + e, m - 1) * k + 2 * fm;
    float acc[RT][NT][2];
    for (short rt = 0; rt < RT; ++rt)
        for (short t = 0; t < NT; ++t) acc[rt][t][0] = acc[rt][t][1] = 0.0f;
    const uint np = k / 64;
    MvkQ6K an[NT];
    float2 xn[RT][2][2][2];
    const uint ip0 = min((uint)sg, np - 1);
    mvk_load<NT>(xr, ip0, fn, an);
    mvk6_load_x<RT>(yp, ip0, xn);
    for (uint ip = sg; ip < np; ip += NSG) {
        const short g2 = 2 * (ip % 2);
        MvkQ6K ac[NT];
        float2 xs[RT][2][2][2];
        for (short t = 0; t < NT; ++t) ac[t] = an[t];
        for (short rt = 0; rt < RT; ++rt)
            for (short e = 0; e < 2; ++e)
                for (short h = 0; h < 2; ++h) {
                    xs[rt][e][h][0] = xn[rt][e][h][0];
                    xs[rt][e][h][1] = xn[rt][e][h][1];
                }
        const uint ipn = min(ip + NSG, np - 1);
        mvk_load<NT>(xr, ipn, fn, an);
        mvk6_load_x<RT>(yp, ipn, xn);
        for (short t = 0; t < NT; ++t) {
            const float d = float(ac[t].d);
            for (short h = 0; h < 2; ++h) {
                const short hs = g2 + 4 * h;
                const char2 sc = h ? ac[t].s1 : ac[t].s0;
                for (short q = 0; q < 2; ++q) {
                    const ushort2 ql = ac[t].l[q], qh = ac[t].h[q];
                    // Even bytes (step 2q), odd bytes (step 2q + 1).
                    half2 a[2];
                    if (h == 0) {
                        a[0] = mvm_1024_plus((ql & ushort2(0x000F)) | (((qh >> hs) & ushort2(3)) << 4)) - half2(1024.0h + 32.0h);
                        a[1] = mvm_1024_plus(((ql >> 8) & ushort2(0x000F)) | (((qh >> (8 + hs)) & ushort2(3)) << 4)) - half2(1024.0h + 32.0h);
                    } else {
                        a[0] = mvm_1024_plus((ql & ushort2(0x00F0)) | (((qh >> hs) & ushort2(3)) << 8)) - half2(1024.0h + 512.0h);
                        a[1] = mvm_1024_plus(((ql >> 8) & ushort2(0x00F0)) | (((qh >> (8 + hs)) & ushort2(3)) << 8)) - half2(1024.0h + 512.0h);
                    }
                    simdgroup_float8x8 mp[RT];
                    for (short rt = 0; rt < RT; ++rt) mp[rt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    for (short s = 0; s < 2; ++s) {
                        simdgroup_half8x8 ma;
                        ma.thread_elements()[0] = a[s].x;
                        ma.thread_elements()[1] = a[s].y;
                        for (short rt = 0; rt < RT; ++rt) {
                            simdgroup_float8x8 mb;
                            mb.thread_elements()[0] = s == 0 ? xs[rt][0][h][q].x : xs[rt][0][h][q].y;
                            mb.thread_elements()[1] = s == 0 ? xs[rt][1][h][q].x : xs[rt][1][h][q].y;
                            simdgroup_multiply_accumulate(mp[rt], ma, mb, mp[rt]);
                        }
                    }
                    const float ds = d * float(q ? sc.y : sc.x) * (h == 0 ? 1.0f : 1.0f / 16);
                    for (short rt = 0; rt < RT; ++rt) {
                        acc[rt][t][0] = fma(ds, mp[rt].thread_elements()[0], acc[rt][t][0]);
                        acc[rt][t][1] = fma(ds, mp[rt].thread_elements()[1], acc[rt][t][1]);
                    }
                }
            }
        }
    }
    mvm_store<NT, RT, NSG>(acc, red, y, n, m, accumulate, ldy, i01, i11, lane, sg);
}
// Two 8-row weight tiles and 8 simdgroups per threadgroup for both (swept on an M4 over
// 1, 2 or 4 tiles and 4, 8 or 16 simdgroups: 2x8 and 2x4 tie, others 4-45% slower).
// llama.cpp dequantizes Q4_K and Q6_K to F32 per lane here (`kernel_mul_mv_mma_gen`); a
// port of that was 3-16% slower than these per Llama-3.2-1B layer. Rows need k % 256 == 0.
typedef decltype(mvm_q4k<2, 1, 8>) mvm_q4k_t;
typedef decltype(mvm_q6k<2, 1, 8>) mvm_q6k_t;
template [[host_name("mvm_q4_k")]] kernel mvm_q4k_t mvm_q4k<2, 1, 8>;
template [[host_name("mvm_q6_k")]] kernel mvm_q6k_t mvm_q6k<2, 1, 8>;

// ── small-batch GEMM: up to 8*NT activation rows on simdgroup matrices ──
// Each simdgroup owns 8 weight rows. Per 32-value block, its 32 lanes dequantize the
// 8x32 weight tile once (8 values each, as the tiled GEMM does) into threadgroup memory
// as half, then multiply NT 8-token fragments of the half-precision activations, loaded
// straight from device memory, with it. Unlike the tiled GEMM this needs no 64-token
// tile and launches one threadgroup per 16 weight rows, so small batches fill the GPU.
// Rows of `x` past `m` are read but only feed output rows that are never written, so
// `x` needs 8*NT readable rows.
// One step of `mms`: B blocks (32*B values) starting at block ib0.
template <typename T, uint NT, uint B, uint RG>
inline void mms_step(device const uchar* w, device const half* x, thread const uint* rows, uint k, uint ib0, uint lr, uint lq,
                     threadgroup half* sw, thread simdgroup_float8x8* acc) {
    constexpr uint KW = 32 * B;
    // All weight loads of the tile are issued before any is used.
    float wv[RG][B][8];
    for (uint g = 0; g < RG; ++g)
        for (uint b = 0; b < B; ++b) load8(w, rows[g], k, ib0 + b, lq, wv[g][b], T());
    for (uint g = 0; g < RG; ++g)
        for (uint b = 0; b < B; ++b)
            for (uint j = 0; j < 8; ++j) sw[(g * 8 + lr) * KW + b * 32 + lq * 8 + j] = half(wv[g][b][j]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // acc[t][g] (tokens x rows) += X[t*8.., kk..] (row-major, from device) * W[rows, kk..]^T
    // (transposed load from threadgroup memory). Each X fragment serves RG row groups.
    for (uint kk = 0; kk < KW; kk += 8) {
        simdgroup_half8x8 wt[RG];
        for (uint g = 0; g < RG; ++g) simdgroup_load(wt[g], sw + g * 8 * KW + kk, KW, ulong2(0, 0), true);
        for (uint t = 0; t < NT; ++t) {
            simdgroup_half8x8 xm;
            simdgroup_load(xm, x + (ulong)(t * 8) * k + ib0 * 32 + kk, k);
            for (uint g = 0; g < RG; ++g) simdgroup_multiply_accumulate(acc[t * RG + g], xm, wt[g], acc[t * RG + g]);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename T, uint NT, uint SG, uint BPI, uint RG>
kernel void mms(device const uchar* w [[buffer(0)]], device const half* x [[buffer(1)]],
                device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                constant uint& k [[buffer(4)]], constant uint& m [[buffer(5)]],
                constant uint& accumulate [[buffer(6)]], constant uint& ldy [[buffer(7)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half sw_all[SG][RG * 8 * 32 * BPI];
    threadgroup float so_all[SG][64];
    threadgroup half* sw = sw_all[sg];
    threadgroup float* so = so_all[sg];
    const uint row0 = (tg * SG + sg) * 8 * RG;
    if (row0 >= n) return;
    const uint lr = lane / 4, lq = lane % 4;
    uint rows[RG];
    for (uint g = 0; g < RG; ++g) rows[g] = min(row0 + g * 8 + lr, n - 1);  // rows past n are computed and discarded
    const uint nb = k / 32;
    simdgroup_float8x8 acc[NT * RG];
    for (uint t = 0; t < NT * RG; ++t) acc[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint ib = 0;
    for (; ib + BPI <= nb; ib += BPI) mms_step<T, NT, BPI, RG>(w, x, rows, k, ib, lr, lq, sw, acc);
    for (; ib < nb; ++ib) mms_step<T, NT, 1, RG>(w, x, rows, k, ib, lr, lq, sw, acc);
    for (uint t = 0; t < NT; ++t) {
        for (uint g = 0; g < RG; ++g) {
            simdgroup_store(acc[t * RG + g], so, 8);  // so[token * 8 + r]
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint r = g * 8 + e % 8, tok = t * 8 + e / 8;
                if (tok < m && row0 + r < n) {
                    device float* o = y + (ulong)tok * ldy + row0 + r;
                    *o = accumulate ? *o + so[e] : so[e];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// 2 simdgroups x 8·RG weight rows per threadgroup, 2 blocks per step (swept on an M4:
// 1 or 4 simdgroups and 1, 4 or 8 blocks per step were up to 20% slower). Used for up
// to 8 tokens (NT = 1, RG = 1) and for 17-24 (NT = 3 token fragments, RG = 2 row groups
// per simdgroup so each activation fragment serves two products: -12 to -23% per
// Llama-3.2-1B layer; padding 24 tokens to 32 was 19-31% slower). 9-16 tokens use
// `mmh` on matrices of at least 2048 rows and NT = 2 here otherwise; 25 and more the
// tiled GEMM.
typedef decltype(mms<WQ8, 1, 2, 2, 1>) mms_t;
#define MMS(TAG, SUF, NT) template [[host_name("mms" #NT "_" SUF)]] kernel mms_t mms<TAG, NT, 2, 2, 1>;
#define MMSR(TAG, SUF, NT, RG) template [[host_name("mms" #NT "r" #RG "_" SUF)]] kernel mms_t mms<TAG, NT, 2, 2, RG>;
#define MMS_ALL(TAG, SUF) MMS(TAG, SUF, 1) MMS(TAG, SUF, 2) MMSR(TAG, SUF, 2, 2) MMSR(TAG, SUF, 3, 2)
MMS_ALL(WQ8, "q8_0")
MMS_ALL(WQ4, "q4_0")
MMS_ALL(WF16, "f16")
MMS_ALL(WQ4K, "q4_k")
MMS_ALL(WQ6K, "q6_k")

constant constexpr uint BK = 32;

// ── F32-weight tiled matrix-matrix (prefill), 32x32 output tile, full F32 ──
template <typename T>
inline void mm32_impl(device const uchar* w, device const float* x, device float* y,
                      uint n, uint k, uint m, uint acc_in, uint ldy, uint2 tg, uint tid, uint sg,
                      threadgroup float* sa, threadgroup float* sb, threadgroup float* sc) {
    const uint row0 = tg.y * 32, col0 = tg.x * 32;
    const uint lr = tid / 4, lq = tid % 4;
    const uint si = (sg / 2) * 16, sj = (sg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint a = 0; a < 2; ++a)
        for (uint b = 0; b < 2; ++b) acc[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < k; k0 += BK) {
        const uint xr = row0 + lr;
        for (uint j = 0; j < 8; ++j) sa[lr * BK + lq * 8 + j] = (xr < m) ? x[(ulong)xr * k + k0 + lq * 8 + j] : 0.0f;
        float wv[8];
        if (col0 + lr < n) load8(w, col0 + lr, k, k0 / 32, lq, wv, T());
        else for (uint j = 0; j < 8; ++j) wv[j] = 0.0f;
        for (uint j = 0; j < 8; ++j) sb[lr * BK + lq * 8 + j] = wv[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 fa[2], fb[2];
            for (uint a = 0; a < 2; ++a) simdgroup_load(fa[a], sa + (si + a * 8) * BK + kk, BK);
            for (uint b = 0; b < 2; ++b) simdgroup_load(fb[b], sb + (sj + b * 8) * BK + kk, BK, ulong2(0, 0), true);
            for (uint a = 0; a < 2; ++a)
                for (uint b = 0; b < 2; ++b) simdgroup_multiply_accumulate(acc[a][b], fa[a], fb[b], acc[a][b]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Accumulation (y += ...) and partial tiles go through threadgroup staging so the
    // k-loop above stays identical to the plain kernel (keeps accumulators in registers).
    if (!acc_in && row0 + 32 <= m && col0 + 32 <= n) {
        for (uint a = 0; a < 2; ++a)
            for (uint b = 0; b < 2; ++b)
                simdgroup_store(acc[a][b], y + (ulong)(row0 + si + a * 8) * ldy + col0 + sj + b * 8, ldy);
    } else {
        for (uint a = 0; a < 2; ++a)
            for (uint b = 0; b < 2; ++b) simdgroup_store(acc[a][b], sc + (si + a * 8) * 32 + sj + b * 8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 32 * 32; i += 128) {
            const uint r = row0 + i / 32, c = col0 + i % 32;
            if (r < m && c < n) {
                device float* o = y + (ulong)r * ldy + c;
                *o = acc_in ? *o + sc[i] : sc[i];
            }
        }
    }
}

kernel void mm_f32(LINEAR_ARGS, uint2 tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[32 * BK];
    threadgroup float sb[32 * BK];
    threadgroup float sc[32 * 32];
    mm32_impl<WF32>(w, x, y, n, k, m, accumulate, ldy, tg, tid, sg, sa, sb, sc);
}

// ── quantized / F16-weight tiled matrix-matrix (prefill), 64x64 output tile ──
// 64 activation rows x 64 weight rows per threadgroup (128 threads, 4 simdgroups, each
// 32x32 = 4x4 accumulators), k-step 64. Activations arrive as half precision (converted
// once per linear layer, not once per weight tile) and both tiles are half in
// threadgroup memory, stored so fragments load untransposed; products are
// half x half -> float. Each k-step a thread dequantizes one 32-value weight block
// (row-per-thread mapping, conflict-free tile stores) or, for formats whose blocks are
// not word-aligned (Q6_K), four threads share a block (coalesced reads).
// Chosen with tests/gemm_bench.rs on an M4 against BK=32, float activations, 64x128
// tiles (4 or 8 simdgroups) and 32 accumulators per simdgroup (spills: 5.5x slower).
constant constexpr uint TQ = 64;
constant constexpr uint TK = 64;

template <typename T, bool BLOCK_MAJOR>
inline void mmq_impl(device const uchar* w, device const half* x, device float* y,
                     uint n, uint k, uint m, uint acc_in, uint ldy, uint2 tg, uint tid, uint sg,
                     threadgroup char* shmem) {
    threadgroup half* sw = (threadgroup half*)shmem;             // [TK][TQ] 8 KB
    threadgroup half* sx = (threadgroup half*)(shmem + 8192);    // [TQ][TK] 8 KB
    const uint row0 = tg.y * TQ, col0 = tg.x * TQ;
    const uint xi = (sg / 2) * 32, wj = (sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint a = 0; a < 4; ++a)
        for (uint b = 0; b < 4; ++b) acc[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < k; k0 += TK) {
        if (BLOCK_MAJOR) {
            for (uint idx = tid; idx < TQ * 8; idx += 128) {
                const uint wr = idx / 8, q8 = idx % 8;
                float v[8];
                if (col0 + wr < n) load8(w, col0 + wr, k, k0 / 32 + q8 / 4, q8 % 4, v, T());
                else for (uint j = 0; j < 8; ++j) v[j] = 0.0f;
                for (uint j = 0; j < 8; ++j) sw[(q8 * 8 + j) * TQ + wr] = half(v[j]);
            }
        } else {
            const uint wr = tid % TQ, blk = tid / TQ;   // 64 rows x 2 blocks = 128 threads
            for (uint q = 0; q < 4; ++q) {
                float v[8];
                if (col0 + wr < n) load8(w, col0 + wr, k, k0 / 32 + blk, q, v, T());
                else for (uint j = 0; j < 8; ++j) v[j] = 0.0f;
                for (uint j = 0; j < 8; ++j) sw[(blk * 32 + q * 8 + j) * TQ + wr] = half(v[j]);
            }
        }
        for (uint idx = tid; idx < TQ * 8; idx += 128) {
            const uint xr = idx / 8, xq = idx % 8;
            threadgroup half4* dst = (threadgroup half4*)(sx + xr * TK + xq * 8);
            if (row0 + xr < m) {
                device const half4* src = (device const half4*)(x + (ulong)(row0 + xr) * k + k0 + xq * 8);
                dst[0] = src[0];
                dst[1] = src[1];
            } else {
                dst[0] = half4(0.0h);
                dst[1] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TK; kk += 8) {
            simdgroup_half8x8 fx[4];
            simdgroup_half8x8 fw[4];
            for (uint a = 0; a < 4; ++a) simdgroup_load(fx[a], sx + (xi + a * 8) * TK + kk, TK);
            for (uint b = 0; b < 4; ++b) simdgroup_load(fw[b], sw + kk * TQ + wj + b * 8, TQ);
            for (uint a = 0; a < 4; ++a)
                for (uint b = 0; b < 4; ++b) simdgroup_multiply_accumulate(acc[a][b], fx[a], fw[b], acc[a][b]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Full tiles store directly; with accumulation (y += ...) the previous values are
    // loaded as fragments and added by a product with the identity. Partial tiles go
    // through threadgroup staging.
    if (row0 + TQ <= m && col0 + TQ <= n) {
        const simdgroup_float8x8 ident = simdgroup_float8x8(1.0f);
        for (uint a = 0; a < 4; ++a)
            for (uint b = 0; b < 4; ++b) {
                device float* o = y + (ulong)(row0 + xi + a * 8) * ldy + col0 + wj + b * 8;
                if (acc_in) {
                    simdgroup_float8x8 prev;
                    simdgroup_load(prev, o, ldy);
                    simdgroup_multiply_accumulate(acc[a][b], prev, ident, acc[a][b]);
                }
                simdgroup_store(acc[a][b], o, ldy);
            }
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;           // [TQ][TQ] floats 16 KB
        for (uint a = 0; a < 4; ++a)
            for (uint b = 0; b < 4; ++b)
                simdgroup_store(acc[a][b], sc + (xi + a * 8) * TQ + wj + b * 8, TQ);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < TQ * TQ; i += 128) {
            const uint r = row0 + i / TQ, c = col0 + i % TQ;
            if (r < m && c < n) {
                device float* o = y + (ulong)r * ldy + c;
                *o = acc_in ? *o + sc[i] : sc[i];
            }
        }
    }
}

// `x` is bound as half-precision activations for these kernels.
#define MMH_KERNEL(NAME, TAG, BM) \
kernel void NAME(LINEAR_ARGS, uint2 tg [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup char shmem[16384]; \
    mmq_impl<TAG, BM>(w, (device const half*)x, y, n, k, m, accumulate, ldy, tg, tid, sg, shmem); }

MMH_KERNEL(mm_q8_0, WQ8, false)
MMH_KERNEL(mm_q4_0, WQ4, false)
MMH_KERNEL(mm_f16, WF16, false)
MMH_KERNEL(mm_q4_k, WQ4K, false)
MMH_KERNEL(mm_q6_k, WQ6K, true)

// ── quantized / F16-weight tiled matrix-matrix, 64x32 output tile ──
// 64 weight rows x 32 tokens per threadgroup, 4 simdgroups of 32 rows x 16 tokens (8
// accumulators), both tiles stored as contiguous 8x8 blocks (as in llama.cpp's classic
// `kernel_mul_mm`). Each thread dequantizes 16 values of one weight row per 32-value
// block, and two blocks are staged per pair of barriers (12 KB). Chosen with
// tests/gemm_bench.rs on an M4: 1-10% faster than the 64x64 tile for Q8_0, Q4_0, Q4_K and
// F16 at any prompt length, and for Q6_K when the 64x64 grid would launch fewer than 256
// threadgroups. Double-buffered tiles, 4 blocks per stage and token-major grid
// order were slower or no different. Rows with an odd number of blocks use the
// one-block-per-stage instantiation (`mml1_*`): a one-block tail step inside the
// two-block kernel made it 5% slower even when never taken.
// One stage of `mml`: NB blocks from block b.
template <typename T, uint NB>
inline void mml_stage(device const uchar* w, device const half* xr, bool tok_ok, uint lr0, uint k, uint b,
                      uint tid, uint sg, threadgroup half* sa, threadgroup half* sb, thread simdgroup_float8x8* mc) {
    const uint il0 = tid % 2, row = tid / 2, sy = row / 8, lx = row % 8, tok = tid / 4;
    float v[NB][16];
    half4 xa[NB], xb[NB];
    for (uint s = 0; s < NB; ++s) {
        load16(w, lr0, k, b + s, il0, v[s], T());
        xa[s] = tok_ok ? ((device const half4*)(xr + (b + s) * 32))[0] : half4(0.0h);
        xb[s] = tok_ok ? ((device const half4*)(xr + (b + s) * 32))[1] : half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < NB; ++s) {
        // sa: block 8*kc + rowblock, element [k%8][row%8]; sb: block 4*kc + tokblock,
        // element [tok%8][k%8]
        threadgroup half* a = sa + s * 64 * 32;
        for (uint i = 0; i < 8; ++i) {
            a[64 * (8 * (2 * il0) + sy) + 8 * i + lx] = half(v[s][i]);
            a[64 * (8 * (2 * il0 + 1) + sy) + 8 * i + lx] = half(v[s][8 + i]);
        }
        threadgroup half4* d = (threadgroup half4*)(sb + s * 32 * 32 + 64 * (4 * (tid % 4) + tok / 8) + 8 * (tok % 8));
        d[0] = xa[s];
        d[1] = xb[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < NB; ++s) {
        threadgroup const half* lsma = sa + s * 64 * 32 + 4 * 64 * (sg % 2);
        threadgroup const half* lsmb = sb + s * 32 * 32 + 2 * 64 * (sg / 2);
        for (uint ik = 0; ik < 4; ++ik) {
            simdgroup_half8x8 ma[4], mb[2];
            for (uint i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8);
            for (uint i = 0; i < 2; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8);
            for (uint i = 0; i < 8; ++i) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
}

// With GLU, the weight is a gate/up pair interleaved in 32-row chunks (rows 64c..64c+31
// gate chunk c, the next 32 up chunk c), so each tile holds matching gate and up
// columns; the epilogue writes silu(gate) * up as half precision to `y` (row stride
// `ldy` halves, column r0/2..) -- the down projection's input -- instead of both F32
// halves. `n` is then a multiple of 64.
template <typename T, uint NB, bool GLU>
inline void mml_impl(device const uchar* w, device const half* x, device float* y,
                     uint n, uint k, uint m, uint acc_in, uint ldy, uint2 tg, uint split, uint splits,
                     uint tid, uint sg, threadgroup half* sa, threadgroup half* sb, threadgroup float* stage) {
    const uint r0 = tg.x * 64, r1 = tg.y * 32;
    // weight staging: thread -> row tid/2, values 16*(tid%2) .. +16 of each 32-block
    const uint lr0 = min(r0 + tid / 2, n - 1);
    // activation staging: thread -> token tid/4, k offset 8*(tid%4)
    const uint tok = tid / 4;
    const bool tok_ok = r1 + tok < m;
    device const half* xr = x + (ulong)min(r1 + tok, m - 1) * k + 8 * (tid % 4);
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // Split-K (grid z): each split sums its share of the blocks and writes a partial
    // [m, n] output to y + split * m * ldy (`split_k_reduce` adds them up).
    const uint nb = k / 32;
    const uint per = (nb / NB + splits - 1) / splits * NB;
    const uint b1 = min(nb, (split + 1) * per);
    y += (ulong)split * m * ldy;
    for (uint b = split * per; b < b1; b += NB) mml_stage<T, NB>(w, xr, tok_ok, lr0, k, b, tid, sg, sa, sb, mc);
    const uint rr = r0 + 32 * (sg & 1), tt = r1 + 16 * (sg >> 1);
    if (GLU) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* st = stage + 32 * (sg & 1) + 16 * (sg >> 1) * 64;   // [32 tok][64 rows]
        for (uint i = 0; i < 8; ++i) simdgroup_store(mc[i], st + 8 * (i % 4) + 8 * 64 * (i / 4), 64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device half* out = (device half*)y + r0 / 2;
        for (uint e = tid; e < 32 * 32; e += 128) {
            const uint t = e / 32, j = e % 32;
            if (r1 + t < m) {
                const float g = stage[t * 64 + j], u = stage[t * 64 + 32 + j];
                out[(ulong)(r1 + t) * ldy + j] = half(g / (1.0f + exp(-g)) * u);
            }
        }
    } else if (r0 + 64 <= n && (r1 + 32 <= m || m % 8 == 0)) {
        // Whole 8x8 fragments store directly (with a token count that is a multiple of 8,
        // each fragment is entirely inside or outside the batch: 16- and 24-token
        // batches need no staging). Accumulation adds the previous values through a
        // product with the identity.
        const simdgroup_float8x8 ident = simdgroup_float8x8(1.0f);
        for (uint i = 0; i < 8; ++i) {
            if (tt + 8 * (i / 4) >= m) continue;
            device float* o = y + (ulong)(tt + 8 * (i / 4)) * ldy + rr + 8 * (i % 4);
            if (acc_in) {
                simdgroup_float8x8 prev;
                simdgroup_load(prev, o, ldy);
                simdgroup_multiply_accumulate(mc[i], prev, ident, mc[i]);
            }
            simdgroup_store(mc[i], o, ldy);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* st = stage + 32 * (sg & 1) + 16 * (sg >> 1) * 64;   // [32 tok][64 rows]
        for (uint i = 0; i < 8; ++i) simdgroup_store(mc[i], st + 8 * (i % 4) + 8 * 64 * (i / 4), 64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 64 * 32; e += 128) {
            const uint t = r1 + e / 64, r = r0 + e % 64;
            if (t < m && r < n) {
                device float* o = y + (ulong)t * ldy + r;
                *o = acc_in ? *o + stage[e] : stage[e];
            }
        }
    }
}
#define MML_KERNEL(NAME, TAG, NB, GLU) \
kernel void NAME(LINEAR_ARGS, uint3 tg [[threadgroup_position_in_grid]], uint3 ntg [[threadgroups_per_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float stage[3 * 64 * 32 / 2];   /* 12 KB: 2 x (64x32 + 32x32) halves */ \
    threadgroup half* sa = (threadgroup half*)stage; \
    threadgroup half* sb = sa + 2 * 64 * 32; \
    mml_impl<TAG, NB, GLU>(w, (device const half*)x, y, n, k, m, accumulate, ldy, tg.xy, tg.z, ntg.z, tid, sg, sa, sb, stage); }
#define MML_ALL(TAG, SUF) MML_KERNEL(mml_##SUF, TAG, 2, false) MML_KERNEL(mml1_##SUF, TAG, 1, false) \
    MML_KERNEL(mmlg_##SUF, TAG, 2, true) MML_KERNEL(mmlg1_##SUF, TAG, 1, true)
MML_ALL(WQ8, q8_0)
MML_ALL(WQ4, q4_0)
MML_ALL(WF16, f16)
MML_ALL(WQ4K, q4_k)
MML_ALL(WQ6K, q6_k)

// ── quantized / F16-weight GEMM for 9-16 tokens: 64 rows x 16 tokens per threadgroup ──
// The `mml` scheme with a 16-token tile: 4 simdgroups of 16 weight rows x 16 tokens (4
// accumulators), so no products are spent on an empty half of a 32-token tile. Per
// Llama-3.2-1B layer on an M4 it is 5-9% faster than `mms` at 12-16 tokens (`mms`
// stays faster at 8 and 17-24 tokens, `mml` from 25). One block per stage was within
// ±4%.
template <typename T, uint NB>
inline void mmh_stage(device const uchar* w, device const half* xr, bool tok_ok, uint lr0, uint k, uint b,
                      uint tid, uint sg, threadgroup half* sa, threadgroup half* sb, thread simdgroup_float8x8* mc) {
    const uint il0 = tid % 2, row = tid / 2, sy = row / 8, lx = row % 8, tok = tid / 8;
    float v[NB][16];
    half4 xa[NB];
    for (uint s = 0; s < NB; ++s) {
        load16(w, lr0, k, b + s, il0, v[s], T());
        xa[s] = tok_ok ? ((device const half4*)(xr + (b + s) * 32))[0] : half4(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < NB; ++s) {
        // sa as in `mml`; sb: block 2*kc + tokblock, element [tok%8][k%8]
        threadgroup half* a = sa + s * 64 * 32;
        for (uint i = 0; i < 8; ++i) {
            a[64 * (8 * (2 * il0) + sy) + 8 * i + lx] = half(v[s][i]);
            a[64 * (8 * (2 * il0 + 1) + sy) + 8 * i + lx] = half(v[s][8 + i]);
        }
        const uint kc = (tid % 8) / 2;
        *(threadgroup half4*)(sb + s * 16 * 32 + 64 * (2 * kc + tok / 8) + 8 * (tok % 8) + 4 * (tid % 2)) = xa[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < NB; ++s) {
        for (uint kc = 0; kc < 4; ++kc) {
            simdgroup_half8x8 ma[2], mb[2];
            for (uint i = 0; i < 2; ++i) simdgroup_load(ma[i], sa + s * 64 * 32 + 64 * (8 * kc + 2 * sg + i), 8);
            for (uint j = 0; j < 2; ++j) simdgroup_load(mb[j], sb + s * 16 * 32 + 64 * (2 * kc + j), 8);
            for (uint j = 0; j < 2; ++j)
                for (uint i = 0; i < 2; ++i) simdgroup_multiply_accumulate(mc[2 * j + i], mb[j], ma[i], mc[2 * j + i]);
        }
    }
}

template <typename T, uint NB>
kernel void mmh(LINEAR_ARGS, uint2 tg [[threadgroup_position_in_grid]], uint2 ntg [[threadgroups_per_grid]],
                uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2 * 64 * 32];   // 8 KB (NB <= 2), reused for output staging
    threadgroup half sb[2 * 16 * 32];
    device const half* xh = (device const half*)x;
    // tg.y = split * token tiles + token tile. Split-K: with several splits, each sums
    // its share of the blocks and writes a partial [m, n] output to y + split * m * ldy
    // (`split_k_reduce` adds them up).
    const uint tiles = (m + 15) / 16, split = tg.y / tiles, splits = ntg.y / tiles;
    const uint r0 = tg.x * 64, r1 = (tg.y % tiles) * 16;
    const uint nb = k / 32;
    const uint per = (nb / NB + splits - 1) / splits * NB;
    const uint b0 = split * per, b1 = min(nb, b0 + per);
    y += (ulong)split * m * ldy;
    // weight staging: thread -> row tid/2, values 16*(tid%2) .. +16 of each 32-block;
    // activations: thread -> token tid/8, k offset 4*(tid%8)
    const uint lr0 = min(r0 + tid / 2, n - 1);
    const uint tok = tid / 8;
    const bool tok_ok = r1 + tok < m;
    device const half* xr = xh + (ulong)min(r1 + tok, m - 1) * k + 4 * (tid % 8);
    simdgroup_float8x8 mc[4];
    for (uint i = 0; i < 4; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint b = b0; b < b1; b += NB) mmh_stage<T, NB>(w, xr, tok_ok, lr0, k, b, tid, sg, sa, sb, mc);
    // mc[2j + i]: tokens r1 + 8j.., rows r0 + 16 sg + 8i..
    if (r0 + 64 <= n && m % 8 == 0) {
        const simdgroup_float8x8 ident = simdgroup_float8x8(1.0f);
        for (uint j = 0; j < 2; ++j) {
            if (r1 + 8 * j >= m) continue;
            for (uint i = 0; i < 2; ++i) {
                device float* o = y + (ulong)(r1 + 8 * j) * ldy + r0 + 16 * sg + 8 * i;
                if (accumulate) {
                    simdgroup_float8x8 prev;
                    simdgroup_load(prev, o, ldy);
                    simdgroup_multiply_accumulate(mc[2 * j + i], prev, ident, mc[2 * j + i]);
                }
                simdgroup_store(mc[2 * j + i], o, ldy);
            }
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* st = (threadgroup float*)sa;   // [16 tok][64 rows]
        for (uint j = 0; j < 2; ++j)
            for (uint i = 0; i < 2; ++i) simdgroup_store(mc[2 * j + i], st + 8 * j * 64 + 16 * sg + 8 * i, 64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 16 * 64; e += 128) {
            const uint t = r1 + e / 64, r = r0 + e % 64;
            if (t < m && r < n) {
                device float* o = y + (ulong)t * ldy + r;
                *o = accumulate ? *o + st[e] : st[e];
            }
        }
    }
}
// Sum of split-K partials (`mmh`, `mml`): y[t, r] (+)= sum over s of part[s][t][r], in order.
kernel void split_k_reduce(device const float* part [[buffer(0)]], device float* y [[buffer(1)]],
                       constant uint& n [[buffer(2)]], constant uint& m [[buffer(3)]],
                       constant uint& accumulate [[buffer(4)]], constant uint& ldy [[buffer(5)]],
                       constant uint& splits [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.x, t = gid.y;
    if (r >= n || t >= m) return;
    float v = 0.0f;
    for (uint s = 0; s < splits; ++s) v += part[((ulong)s * m + t) * n + r];
    device float* o = y + (ulong)t * ldy + r;
    *o = accumulate ? *o + v : v;
}

// Two blocks per stage, or one for rows with an odd number of blocks (`mmh1_*`).
typedef decltype(mmh<WQ8, 2>) mmh_t;
#define MMH_ALL(TAG, SUF) \
    template [[host_name("mmh_" #SUF)]] kernel mmh_t mmh<TAG, 2>; \
    template [[host_name("mmh1_" #SUF)]] kernel mmh_t mmh<TAG, 1>;
MMH_ALL(WQ8, q8_0)
MMH_ALL(WQ4, q4_0)
MMH_ALL(WF16, f16)
MMH_ALL(WQ4K, q4_k)
MMH_ALL(WQ6K, q6_k)

// ── F32 -> F16 conversion of a linear layer's input (prefill) ──
kernel void to_half(device const float4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) { y[i] = half4(x[i]); }


// ── embedding lookup: out[r, c] = table[ids[r], c] ──
template <typename T>
inline void embed_impl(device const uchar* table, device const uint* ids, device float* out,
                       uint h, uint2 gid) {
    const uint c8 = gid.x, r = gid.y;      // c8 indexes groups of 8 columns
    float v[8];
    load8(table, ids[r], h, c8 / 4, c8 % 4, v, T());
    for (uint j = 0; j < 8; ++j) out[(ulong)r * h + c8 * 8 + j] = v[j];
}
#define EMBED_KERNEL(NAME, TAG) \
kernel void NAME(device const uchar* table [[buffer(0)]], device const uint* ids [[buffer(1)]], \
                 device float* out [[buffer(2)]], constant uint& h [[buffer(3)]], \
                 uint2 gid [[thread_position_in_grid]]) { embed_impl<TAG>(table, ids, out, h, gid); }
EMBED_KERNEL(embed_f32, WF32)
EMBED_KERNEL(embed_q8_0, WQ8)
EMBED_KERNEL(embed_q4_0, WQ4)
EMBED_KERNEL(embed_f16, WF16)
EMBED_KERNEL(embed_q4_k, WQ4K)
EMBED_KERNEL(embed_q6_k, WQ6K)

// ── RMSNorm: one threadgroup (256 threads) per output row; optional row gather ──
// `rms_norm_h` / `swiglu_h` write half precision straight into the input buffer of the
// next linear layer when it runs the half-precision GEMM, saving a `to_half` pass.
template <typename OT>
kernel void rms_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device OT* y [[buffer(2)]], device const uint* rows [[buffer(3)]],
                     constant uint& h [[buffer(4)]], constant float& eps [[buffer(5)]],
                     constant uint& gather [[buffer(6)]],
                     uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const uint src = gather ? rows[r] : r;
    device const float* xr = x + (ulong)src * h;
    float s = 0.0f;
    for (uint i = tid; i < h; i += 256) s += xr[i] * xr[i];
    s = simd_sum(s);
    if (lane == 0) partial[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float inv = rsqrt(total / float(h) + eps);
    for (uint i = tid; i < h; i += 256) y[(ulong)r * h + i] = OT(xr[i] * inv * w[i]);
}
template [[host_name("rms_norm")]] kernel decltype(rms_norm<float>) rms_norm<float>;
template [[host_name("rms_norm_h")]] kernel decltype(rms_norm<half>) rms_norm<half>;

// ── RoPE (adjacent pairs) on the first `heads` heads of each row (row stride in floats) ──
kernel void rope(device float* x [[buffer(0)]], device const float* cos_t [[buffer(1)]],
                 device const float* sin_t [[buffer(2)]], device const uint* pos [[buffer(3)]],
                 constant uint& heads [[buffer(4)]], constant uint& hd [[buffer(5)]],
                 constant uint& stride [[buffer(6)]], uint3 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, hh = gid.y, r = gid.z;
    const uint half_d = hd / 2;
    const float c = cos_t[pos[r] * half_d + i];
    const float s = sin_t[pos[r] * half_d + i];
    device float* v = x + (ulong)r * stride + hh * hd + 2 * i;
    const float a = v[0], b = v[1];
    v[0] = a * c - b * s;
    v[1] = a * s + b * c;
}

// ── write K/V rows (from the fused QKV buffer) into the paged cache ──
template <typename KVT>
inline void kv_write_impl(device const float* qkv, uint k_off, uint v_off, uint stride, device KVT* kc,
                          device KVT* vc, device const uint* tables, device const uint* table_off,
                          device const uint* pos, uint nkv, uint hd, uint bs, uint3 gid) {
    const uint d = gid.x, h = gid.y, r = gid.z;
    const uint p = pos[r];
    const uint phys = tables[table_off[r] + p / bs];
    const ulong dst = (ulong)phys * nkv * bs * hd + (ulong)h * bs * hd + (p % bs) * hd + d;
    device const float* row = qkv + (ulong)r * stride + h * hd + d;
    kc[dst] = KVT(row[k_off]);
    vc[dst] = KVT(row[v_off]);
}
// KV cache element type: `_f16` (default) or `_f32` kernels.
#define KV_WRITE_KERNEL(NAME, KVT) \
kernel void NAME(device const float* qkv [[buffer(0)]], constant uint& k_off [[buffer(1)]], \
                 constant uint& v_off [[buffer(2)]], constant uint& stride [[buffer(3)]], \
                 device KVT* kc [[buffer(4)]], device KVT* vc [[buffer(5)]], \
                 device const uint* tables [[buffer(6)]], device const uint* table_off [[buffer(7)]], \
                 device const uint* pos [[buffer(8)]], constant uint& nkv [[buffer(9)]], \
                 constant uint& hd [[buffer(10)]], constant uint& bs [[buffer(11)]], \
                 uint3 gid [[thread_position_in_grid]]) { \
    kv_write_impl<KVT>(qkv, k_off, v_off, stride, kc, vc, tables, table_off, pos, nkv, hd, bs, gid); }
KV_WRITE_KERNEL(kv_write_f32, float)
KV_WRITE_KERNEL(kv_write_f16, half)

#define ATTN_ARGS(KVT) \
    device const float* q [[buffer(0)]], device const KVT* kc [[buffer(1)]], \
    device const KVT* vc [[buffer(2)]], device float* out [[buffer(3)]], \
    device const uint* tables [[buffer(4)]], device const uint* table_off [[buffer(5)]], \
    device const uint* pos [[buffer(6)]], constant uint& nq [[buffer(7)]], \
    constant uint& nkv [[buffer(8)]], constant uint& hd [[buffer(9)]], \
    constant uint& bs [[buffer(10)]], constant float& scale [[buffer(11)]], \
    constant uint& q_stride [[buffer(13)]]

// Loads N consecutive cache values as floats (one vector load for N = 2 or 4).
template <typename T, uint N> struct vload {
    static inline void f(device const T* p, thread float* v) { for (uint i = 0; i < N; ++i) v[i] = float(p[i]); }
};
template <typename T> struct vload<T, 2> {
    static inline void f(device const T* p, thread float* v) { const vec<T, 2> x = *(device const vec<T, 2>*)p; v[0] = x[0]; v[1] = x[1]; }
};
template <typename T> struct vload<T, 4> {
    static inline void f(device const T* p, thread float* v) {
        const vec<T, 4> x = *(device const vec<T, 4>*)p;
        v[0] = x[0]; v[1] = x[1]; v[2] = x[2]; v[3] = x[3];
    }
};

// ── causal paged attention for decode rows, split over the context ("flash decoding") ──
// One threadgroup (4 simdgroups) per (query head, row, context chunk of `chunk`
// positions). Positions are processed in tiles of 32: each lane scores one position
// (keys read 4 values at a time against the query in threadgroup memory), the tile
// updates an online softmax, and values are accumulated with each lane owning HD/32
// output dimensions. Simdgroups take interleaved tiles and are merged at the end. With
// one chunk the result is written directly; otherwise each chunk writes its partial
// (max, sum, unnormalised output) and `attn_reduce` combines them. Splitting keeps the
// GPU busy for single-sequence decode at long context (one chunk per head would be
// only `nq` threadgroups). The head dimension is a template parameter so the per-lane
// accumulators stay in registers.
template <uint HD, typename KVT>
inline void attn_row_impl(device const float* q, device const KVT* kc, device const KVT* vc,
                          device float* out, device const uint* tables, device const uint* table_off,
                          device const uint* pos, uint nq, uint nkv, uint bs, float scale, uint q_stride,
                          device const uint* row_list, device float* part, uint chunk, uint nsplit,
                          threadgroup float* qs, threadgroup float* tm, threadgroup float* ts,
                          threadgroup float* tacc, uint3 tg, uint sg, uint lane) {
    constexpr uint NSG = 4;
    constexpr uint dpl = HD / 32;

    const uint head = tg.x, slot = tg.y, split = tg.z, r = row_list[slot];
    const uint kvh = head / (nq / nkv);
    const uint ctx = pos[r] + 1;
    const uint begin = split * chunk, end = min(ctx, begin + chunk);
    device float* pout = part + ((ulong)(slot * nq + head) * nsplit + split) * (HD + 2);
    if (begin >= ctx) {
        // This row's context is shorter than the longest in the batch: empty partial.
        if (sg == 0 && lane == 0) { pout[0] = -INFINITY; pout[1] = 0.0f; }
        if (sg == 0) for (uint i = 0; i < dpl; ++i) pout[2 + lane + i * 32] = 0.0f;
        return;
    }
    const ulong block_stride = (ulong)nkv * bs * HD;
    const ulong head_off = (ulong)kvh * bs * HD;
    device const uint* table = tables + table_off[r];

    device const float* qr = q + (ulong)r * q_stride + head * HD;
    for (uint i = sg * 32 + lane; i < HD; i += NSG * 32) qs[i] = qr[i] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const float4* q4 = (threadgroup const float4*)qs;

    float m = -INFINITY, s = 0.0f;
    float acc[dpl];
    for (uint i = 0; i < dpl; ++i) acc[i] = 0.0f;

    for (uint t0 = begin + sg * 32; t0 < end; t0 += NSG * 32) {
        const uint t = t0 + lane;
        float score = -INFINITY;
        if (t < end) {
            device const vec<KVT, 4>* kr = (device const vec<KVT, 4>*)(kc + table[t / bs] * block_stride + head_off + (t % bs) * HD);
            float d = 0.0f;
            for (uint j = 0; j < HD / 4; ++j) d += dot(q4[j], float4(kr[j]));
            score = d;
        }
        const float mn = max(m, simd_max(score));
        const float corr = exp(m - mn);
        const float p = (t < end) ? exp(score - mn) : 0.0f;
        s = s * corr + simd_sum(p);
        for (uint i = 0; i < dpl; ++i) acc[i] *= corr;
        const uint cnt = min(32u, end - t0);
        // Each lane owns dims lane*dpl .. lane*dpl+dpl-1 (one vector load per key).
        #define ROW_V_STEP(u) { \
            const float pu = simd_shuffle(p, u); \
            const uint tu = t0 + (u); \
            float v[dpl]; \
            vload<KVT, dpl>::f(vc + table[tu / bs] * block_stride + head_off + (tu % bs) * HD + lane * dpl, v); \
            for (uint i = 0; i < dpl; ++i) acc[i] += pu * v[i]; }
        if (cnt == 32) {
            #pragma unroll(8)
            for (uint u = 0; u < 32; ++u) ROW_V_STEP(u)
        } else {
            for (uint u = 0; u < cnt; ++u) ROW_V_STEP(u)
        }
        #undef ROW_V_STEP
        m = mn;
    }

    if (lane == 0) { tm[sg] = m; ts[sg] = s; }
    for (uint i = 0; i < dpl; ++i) tacc[sg * HD + lane * dpl + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float M = -INFINITY;
        for (uint g = 0; g < NSG; ++g) M = max(M, tm[g]);
        float S = 0.0f;
        for (uint g = 0; g < NSG; ++g) S += ts[g] * exp(tm[g] - M);
        for (uint i = 0; i < dpl; ++i) {
            const uint dim = lane + i * 32;
            float o = 0.0f;
            for (uint g = 0; g < NSG; ++g) o += tacc[g * HD + dim] * exp(tm[g] - M);
            if (nsplit == 1) out[((ulong)r * nq + head) * HD + dim] = o / S;
            else pout[2 + dim] = o;
        }
        if (nsplit > 1 && lane == 0) { pout[0] = M; pout[1] = S; }
    }
}

#define ATTN_ROW_KERNEL(HDV, SFX, KVT) \
kernel void attention_hd##HDV##SFX(ATTN_ARGS(KVT), device const uint* row_list [[buffer(12)]], \
                              device float* part [[buffer(14)]], constant uint& chunk [[buffer(15)]], \
                              constant uint& nsplit [[buffer(16)]], \
                              uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                              uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float qs[HDV]; \
    threadgroup float tm[4]; \
    threadgroup float ts[4]; \
    threadgroup float tacc[4 * HDV]; \
    attn_row_impl<HDV, KVT>(q, kc, vc, out, tables, table_off, pos, nq, nkv, bs, scale, q_stride, row_list, part, \
                            chunk, nsplit, qs, tm, ts, tacc, tg, sg, lane); }
ATTN_ROW_KERNEL(32, _f32, float)
ATTN_ROW_KERNEL(64, _f32, float)
ATTN_ROW_KERNEL(96, _f32, float)
ATTN_ROW_KERNEL(128, _f32, float)
ATTN_ROW_KERNEL(32, _f16, half)
ATTN_ROW_KERNEL(64, _f16, half)
ATTN_ROW_KERNEL(96, _f16, half)
ATTN_ROW_KERNEL(128, _f16, half)

// ── GQA decode attention: one threadgroup per (KV head, row, context chunk) ──
// Same algorithm as `attn_row_impl`, but the threadgroup serves all G query heads that
// share a KV head: each key and value is read once and applied to G queries (the
// per-query-head kernel reads every K/V row G times). Each lane scores one position
// for all G heads; per-head online softmax state and G x HD/32 accumulators per lane.
// Partials use the same layout as `attn_row_impl`, so `attn_reduce` combines them.
template <uint HD, typename KVT, uint G>
inline void attn_gqa_impl(device const float* q, device const KVT* kc, device const KVT* vc,
                          device float* out, device const uint* tables, device const uint* table_off,
                          device const uint* pos, uint nq, uint nkv, uint bs, float scale, uint q_stride,
                          device const uint* row_list, device float* part, uint chunk, uint nsplit,
                          threadgroup float* qs, threadgroup float* tm, threadgroup float* ts,
                          threadgroup float* tacc, uint3 tg, uint sg, uint lane) {
    constexpr uint NSG = 4;
    constexpr uint dpl = HD / 32;
    constexpr uint H4 = HD / 4;

    const uint kvh = tg.x, slot = tg.y, split = tg.z, r = row_list[slot];
    const uint head0 = kvh * G;
    const uint ctx = pos[r] + 1;
    const uint begin = split * chunk, end = min(ctx, begin + chunk);
    if (begin >= ctx) {
        if (sg == 0) {
            for (uint g = 0; g < G; ++g) {
                device float* pout = part + ((ulong)(slot * nq + head0 + g) * nsplit + split) * (HD + 2);
                if (lane == 0) { pout[0] = -INFINITY; pout[1] = 0.0f; }
                for (uint i = 0; i < dpl; ++i) pout[2 + lane + i * 32] = 0.0f;
            }
        }
        return;
    }
    const ulong block_stride = (ulong)nkv * bs * HD;
    const ulong head_off = (ulong)kvh * bs * HD;
    device const uint* table = tables + table_off[r];

    device const float* qr = q + (ulong)r * q_stride + head0 * HD;
    for (uint i = sg * 32 + lane; i < G * HD; i += NSG * 32) qs[i] = qr[i] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const float4* q4 = (threadgroup const float4*)qs;

    float m[G], s[G], acc[G][dpl];
    for (uint g = 0; g < G; ++g) {
        m[g] = -INFINITY; s[g] = 0.0f;
        for (uint i = 0; i < dpl; ++i) acc[g][i] = 0.0f;
    }

    for (uint t0 = begin + sg * 32; t0 < end; t0 += NSG * 32) {
        const uint t = t0 + lane;
        float score[G];
        for (uint g = 0; g < G; ++g) score[g] = -INFINITY;
        if (t < end) {
            device const vec<KVT, 4>* kr = (device const vec<KVT, 4>*)(kc + table[t / bs] * block_stride + head_off + (t % bs) * HD);
            float d[G];
            for (uint g = 0; g < G; ++g) d[g] = 0.0f;
            for (uint j = 0; j < H4; ++j) {
                const float4 kv = float4(kr[j]);
                for (uint g = 0; g < G; ++g) d[g] += dot(q4[g * H4 + j], kv);
            }
            for (uint g = 0; g < G; ++g) score[g] = d[g];
        }
        float p[G];
        for (uint g = 0; g < G; ++g) {
            const float mn = max(m[g], simd_max(score[g]));
            const float corr = exp(m[g] - mn);
            p[g] = (t < end) ? exp(score[g] - mn) : 0.0f;
            s[g] = s[g] * corr + simd_sum(p[g]);
            for (uint i = 0; i < dpl; ++i) acc[g][i] *= corr;
            m[g] = mn;
        }
        const uint cnt = min(32u, end - t0);
        #define GQA_V_STEP(u) { \
            const uint tu = t0 + (u); \
            float v[dpl]; \
            vload<KVT, dpl>::f(vc + table[tu / bs] * block_stride + head_off + (tu % bs) * HD + lane * dpl, v); \
            for (uint g = 0; g < G; ++g) { \
                const float pu = simd_shuffle(p[g], u); \
                for (uint i = 0; i < dpl; ++i) acc[g][i] += pu * v[i]; } }
        if (cnt == 32) {
            #pragma unroll(8)
            for (uint u = 0; u < 32; ++u) GQA_V_STEP(u)
        } else {
            for (uint u = 0; u < cnt; ++u) GQA_V_STEP(u)
        }
        #undef GQA_V_STEP
    }

    for (uint g = 0; g < G; ++g) {
        if (lane == 0) { tm[sg * G + g] = m[g]; ts[sg * G + g] = s[g]; }
        for (uint i = 0; i < dpl; ++i) tacc[(sg * G + g) * HD + lane * dpl + i] = acc[g][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Simdgroup g' < G finalises head g' (the four simdgroups split the heads).
    for (uint g = sg; g < G; g += NSG) {
        const uint head = head0 + g;
        device float* pout = part + ((ulong)(slot * nq + head) * nsplit + split) * (HD + 2);
        float M = -INFINITY;
        for (uint w = 0; w < NSG; ++w) M = max(M, tm[w * G + g]);
        float S = 0.0f;
        for (uint w = 0; w < NSG; ++w) S += ts[w * G + g] * exp(tm[w * G + g] - M);
        for (uint i = 0; i < dpl; ++i) {
            const uint dim = lane + i * 32;
            float o = 0.0f;
            for (uint w = 0; w < NSG; ++w) o += tacc[(w * G + g) * HD + dim] * exp(tm[w * G + g] - M);
            if (nsplit == 1) out[((ulong)r * nq + head) * HD + dim] = o / S;
            else pout[2 + dim] = o;
        }
        if (nsplit > 1 && lane == 0) { pout[0] = M; pout[1] = S; }
    }
}

#define ATTN_GQA_KERNEL(HDV, SFX, KVT, GV) \
kernel void attention_gqa##GV##_hd##HDV##SFX(ATTN_ARGS(KVT), device const uint* row_list [[buffer(12)]], \
                              device float* part [[buffer(14)]], constant uint& chunk [[buffer(15)]], \
                              constant uint& nsplit [[buffer(16)]], \
                              uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                              uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float qs[GV * HDV]; \
    threadgroup float tm[4 * GV]; \
    threadgroup float ts[4 * GV]; \
    threadgroup float tacc[4 * GV * HDV]; \
    attn_gqa_impl<HDV, KVT, GV>(q, kc, vc, out, tables, table_off, pos, nq, nkv, bs, scale, q_stride, row_list, part, \
                                chunk, nsplit, qs, tm, ts, tacc, tg, sg, lane); }
#define ATTN_GQA_ALL(GV) \
ATTN_GQA_KERNEL(32, _f32, float, GV) ATTN_GQA_KERNEL(64, _f32, float, GV) \
ATTN_GQA_KERNEL(96, _f32, float, GV) ATTN_GQA_KERNEL(128, _f32, float, GV) \
ATTN_GQA_KERNEL(32, _f16, half, GV) ATTN_GQA_KERNEL(64, _f16, half, GV) \
ATTN_GQA_KERNEL(96, _f16, half, GV) ATTN_GQA_KERNEL(128, _f16, half, GV)
ATTN_GQA_ALL(2)
ATTN_GQA_ALL(3)
ATTN_GQA_ALL(4)
ATTN_GQA_ALL(8)

// ── combine split-context partials: one threadgroup (hd threads) per (head, row) ──
kernel void attn_reduce(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
                        device const uint* row_list [[buffer(2)]], constant uint& nq [[buffer(3)]],
                        constant uint& hd [[buffer(4)]], constant uint& nsplit [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    const uint head = tg.x, slot = tg.y, r = row_list[slot];
    device const float* p = part + (ulong)(slot * nq + head) * nsplit * (hd + 2);
    float M = -INFINITY;
    for (uint s = 0; s < nsplit; ++s) M = max(M, p[s * (hd + 2)]);
    float S = 0.0f, o = 0.0f;
    for (uint s = 0; s < nsplit; ++s) {
        const float w = exp(p[s * (hd + 2)] - M);
        S += p[s * (hd + 2) + 1] * w;
        o += p[s * (hd + 2) + 2 + d] * w;
    }
    out[((ulong)r * nq + head) * hd + d] = o / S;
}

// ── causal attention for a tile of up to 32 consecutive query rows of one sequence ──
// One threadgroup (4 simdgroups x 8 rows) per (head, tile); prefill. Each step stages
// KB = 32 keys and values of the paged cache in threadgroup memory as half precision,
// shared by the four simdgroups. Scores are computed transposed, S^T = K Q^T (K
// fragments load untransposed; Q^T stays in registers), with two accumulators per
// 8-key group so several products are in flight. The online softmax runs inside the
// simdgroup (4 lanes per row, shuffle reductions); P (half) x V (half) accumulates into
// float output fragments, rescaled only when a row's maximum grows by more than 8 in
// log space (lazy rescaling: probabilities stay below e^8, safe in half). The head
// dimension is a template parameter so fragment arrays stay in registers.
// `KV` is dynamic threadgroup memory: max(2 * KB * HD halves, 4 * 8 * HD floats).
constant constexpr uint ATTN_KB = 32;

template <uint HD, typename KVT>
inline void attn_tile_impl(device const float* q, device const KVT* kc, device const KVT* vc,
                           device float* out, device const uint* tables, device const uint* table_off,
                           device const uint* pos, uint nq, uint nkv, uint bs, float scale, uint q_stride,
                           device const uint2* tiles, threadgroup half* KV,
                           threadgroup float (*Sbuf)[8 * ATTN_KB], threadgroup half (*Pbuf)[8 * ATTN_KB],
                           threadgroup float (*Dbuf)[64],
                           uint2 tg, uint tid, uint sg, uint lane) {
    constexpr uint KB = ATTN_KB, ND = HD / 8, KL = KB / 4;
    threadgroup half* Ks = KV;
    threadgroup half* Vs = KV + KB * HD;

    const uint head = tg.x;
    const uint r0 = tiles[tg.y].x, cnt = tiles[tg.y].y;
    const uint kvh = head / (nq / nkv);
    device const uint* table = tables + table_off[r0];
    const uint nkeys = pos[r0 + cnt - 1] + 1;
    const ulong block_stride = (ulong)nkv * bs * HD;
    const ulong head_off = (ulong)kvh * bs * HD;

    // This simdgroup's rows: r0 + 8*sg .. (+sub). Rows past `cnt` are padding (the q
    // buffer has 32 rows of slack) and are never written.
    const uint sub = (cnt > 8 * sg) ? min(8u, cnt - 8 * sg) : 0;
    const uint my_keys = sub ? pos[r0 + 8 * sg + sub - 1] + 1 : 0;
    const uint row = lane / 4, part = lane % 4;
    const uint my_pos = sub ? pos[r0 + 8 * sg + min(row, sub - 1)] : 0;

    simdgroup_float8x8 QT[ND];
    simdgroup_float8x8 O[ND];
    device const float* qbase = q + (ulong)(r0 + 8 * sg) * q_stride + head * HD;
    for (uint i = 0; i < ND; ++i) {
        simdgroup_load(QT[i], qbase + i * 8, q_stride, ulong2(0, 0), true);
        O[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m_row = -INFINITY, l_row = 0.0f;

    for (uint t0 = 0; t0 < nkeys; t0 += KB) {
        // Stage K and V for keys t0..t0+KB (zeros past the last key).
        for (uint idx = tid; idx < KB * HD / 4; idx += 128) {
            const uint kr = idx / (HD / 4), c4 = idx % (HD / 4);
            const uint t = t0 + kr;
            half4 kv = half4(0.0h), vv = half4(0.0h);
            if (t < nkeys) {
                const ulong off = (ulong)table[t / bs] * block_stride + head_off + (t % bs) * HD + c4 * 4;
                kv = half4(*(device const vec<KVT, 4>*)(kc + off));
                vv = half4(*(device const vec<KVT, 4>*)(vc + off));
            }
            *(threadgroup half4*)(Ks + kr * HD + c4 * 4) = kv;
            *(threadgroup half4*)(Vs + kr * HD + c4 * 4) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t0 < my_keys) {
            // S^T[keys][rows] = K Q^T, stored transposed as S[rows][keys].
            simdgroup_float8x8 st[KB / 8][2];
            for (uint c = 0; c < KB / 8; ++c)
                for (uint h2 = 0; h2 < 2; ++h2) st[c][h2] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint i = 0; i < ND; i += 2) {
                for (uint h2 = 0; h2 < 2 && i + h2 < ND; ++h2) {
                    for (uint c = 0; c < KB / 8; ++c) {
                        simdgroup_half8x8 kf;
                        simdgroup_load(kf, Ks + (c * 8) * HD + (i + h2) * 8, HD);
                        simdgroup_multiply_accumulate(st[c][h2], kf, QT[i + h2], st[c][h2]);
                    }
                }
            }
            const simdgroup_float8x8 ident(1.0f);   // diagonal constructor: identity
            for (uint c = 0; c < KB / 8; ++c) {
                simdgroup_float8x8 t;
                simdgroup_multiply_accumulate(t, ident, st[c][0], st[c][1]);
                simdgroup_store(t, Sbuf[sg] + c * 8, KB, ulong2(0, 0), true);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // Online softmax: lane handles row `row`, keys part*KL .. part*KL+KL.
            float v[KL];
            float mx = -INFINITY;
            for (uint j = 0; j < KL; ++j) {
                const uint t = t0 + part * KL + j;
                v[j] = (t <= my_pos) ? Sbuf[sg][row * KB + part * KL + j] * scale : -INFINITY;
                mx = max(mx, v[j]);
            }
            mx = max(mx, simd_shuffle_xor(mx, 1));
            mx = max(mx, simd_shuffle_xor(mx, 2));
            const bool first = (m_row == -INFINITY);
            const bool grow = simd_any(!first && mx > m_row + 8.0f);
            const float m_new = (first || grow) ? max(m_row, mx) : m_row;
            const float corr = first ? 0.0f : exp(m_row - m_new);
            float sum = 0.0f;
            for (uint j = 0; j < KL; ++j) {
                const float pj = (v[j] == -INFINITY) ? 0.0f : exp(v[j] - m_new);
                Pbuf[sg][row * KB + part * KL + j] = half(pj);
                sum += pj;
            }
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 2);
            l_row = l_row * corr + sum;
            m_row = m_new;
            if (grow) {
                for (uint j = 0; j < 2; ++j) {
                    const uint col = part * 2 + j;
                    Dbuf[sg][row * 8 + col] = (col == row) ? corr : 0.0f;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (grow) {
                simdgroup_float8x8 df;
                simdgroup_load(df, Dbuf[sg], 8);
                for (uint i = 0; i < ND; ++i) simdgroup_multiply(O[i], df, O[i]);
            }
            for (uint c = 0; c < KB / 8; ++c) {
                simdgroup_half8x8 pf;
                simdgroup_load(pf, Pbuf[sg] + c * 8, KB);
                for (uint i = 0; i < ND; ++i) {
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, Vs + (c * 8) * HD + i * 8, HD);
                    simdgroup_multiply_accumulate(O[i], pf, vf, O[i]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalise and write through threadgroup memory (the K/V staging area is free now).
    threadgroup float* Ot = (threadgroup float*)KV + sg * 8 * HD;
    for (uint i = 0; i < ND; ++i) simdgroup_store(O[i], Ot + i * 8, HD);
    if (part == 0) Sbuf[sg][row] = l_row;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < sub * HD; idx += 32) {
        const uint rr = idx / HD, d = idx % HD;
        out[((ulong)(r0 + 8 * sg + rr) * nq + head) * HD + d] = Ot[rr * HD + d] / Sbuf[sg][rr];
    }
}

#define ATTN_TILE_KERNEL(HDV, SFX, KVT) \
kernel void attention_tile32_hd##HDV##SFX(ATTN_ARGS(KVT), device const uint2* tiles [[buffer(12)]], \
                                     threadgroup half* KV [[threadgroup(0)]], \
                                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float Sbuf[4][8 * ATTN_KB]; \
    threadgroup half Pbuf[4][8 * ATTN_KB]; \
    threadgroup float Dbuf[4][64]; \
    attn_tile_impl<HDV, KVT>(q, kc, vc, out, tables, table_off, pos, nq, nkv, bs, scale, q_stride, tiles, KV, \
                        Sbuf, Pbuf, Dbuf, tg, tid, sg, lane); }
ATTN_TILE_KERNEL(32, _f32, float)
ATTN_TILE_KERNEL(64, _f32, float)
ATTN_TILE_KERNEL(96, _f32, float)
ATTN_TILE_KERNEL(128, _f32, float)
ATTN_TILE_KERNEL(32, _f16, half)
ATTN_TILE_KERNEL(64, _f16, half)
ATTN_TILE_KERNEL(96, _f16, half)
ATTN_TILE_KERNEL(128, _f16, half)

// ── SwiGLU on the fused gate/up buffer: out[r, j] = silu(gu[r, j]) * gu[r, inter + j] ──
template <typename OT>
kernel void swiglu(device const float* gu [[buffer(0)]], device OT* out [[buffer(1)]],
                   constant uint& inter [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint r = i / inter, j = i % inter;
    const float a = gu[(ulong)r * 2 * inter + j];
    const float u = gu[(ulong)r * 2 * inter + inter + j];
    out[i] = OT(a / (1.0f + exp(-a)) * u);
}
template [[host_name("swiglu")]] kernel decltype(swiglu<float>) swiglu<float>;
template [[host_name("swiglu_h")]] kernel decltype(swiglu<half>) swiglu<half>;

// As `swiglu` for a gate/up buffer interleaved in 32-column chunks (see `mml_impl`).
template <typename OT>
kernel void swiglu_i(device const float* gu [[buffer(0)]], device OT* out [[buffer(1)]],
                     constant uint& inter [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint r = i / inter, j = i % inter;
    device const float* g = gu + (ulong)r * 2 * inter + (j / 32) * 64 + j % 32;
    const float a = g[0], u = g[32];
    out[i] = OT(a / (1.0f + exp(-a)) * u);
}
template [[host_name("swiglu_i")]] kernel decltype(swiglu_i<float>) swiglu_i<float>;
template [[host_name("swiglu_ih")]] kernel decltype(swiglu_i<half>) swiglu_i<half>;
"#;

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
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WQ4) {
    device const block_q4_0* b = (device const block_q4_0*)w + (ulong)row * (k / 32) + ib;
    device const ushort* p = (device const ushort*)(b->qs + (q % 2) * 8);
    const float d = float(b->d);
    const uint sh = q < 2 ? 0 : 4;
    for (uint j = 0; j < 4; ++j) {
        const ushort c = p[j];
        out[2 * j] = d * (float((c >> sh) & 0x0F) - 8.0f);
        out[2 * j + 1] = d * (float((c >> (8 + sh)) & 0x0F) - 8.0f);
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
MV_KERNEL(mv_q8_0, WQ8)
MV_KERNEL(mv_f16, WF16)

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

// ── Q4_K matrix-vector: 8 lanes per super-block, 4 super-blocks per simdgroup step ──
// Lane t covers 64-value chunk c = t/2, half h = t%2: 16 bytes of qs give 16 values of
// sub-block 2c (low nibbles) and 16 of sub-block 2c+1 (high nibbles). Mins are applied
// through sums of x.
kernel void mv_q4_k(LINEAR_ARGS, uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint nb = k / 256;
    const uint row0 = (tg * 2 + sg) * 4;
    const uint ix = lane / 8, t = lane % 8, c = t / 2, h = t % 2;
    device const block_q4_K* wb = (device const block_q4_K*)w;
    for (uint mi = 0; mi < m; ++mi) {
        device const float* xr = x + (ulong)mi * k;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint ib = ix; ib < nb; ib += 4) {
            float xl[16], xh[16];
            device const float4* pl = (device const float4*)(xr + ib * 256 + c * 64 + h * 16);
            device const float4* ph = (device const float4*)(xr + ib * 256 + c * 64 + 32 + h * 16);
            float sl = 0.0f, shs = 0.0f;
            for (uint v = 0; v < 4; ++v) {
                const float4 a = pl[v], b = ph[v];
                for (uint e = 0; e < 4; ++e) { xl[v * 4 + e] = a[e]; xh[v * 4 + e] = b[e]; }
                sl += a[0] + a[1] + a[2] + a[3];
                shs += b[0] + b[1] + b[2] + b[3];
            }
            for (uint r = 0; r < 4; ++r) {
                const uint row = row0 + r;
                if (row >= n) break;
                device const block_q4_K* b = wb + (ulong)row * nb + ib;
                const float2 sm1 = scale_min_k4(2 * c, b->scales);
                const float2 sm2 = scale_min_k4(2 * c + 1, b->scales);
                const uint4 q = *(device const uint4*)(b->qs + c * 32 + h * 16);
                float al = 0.0f, ah = 0.0f;
                for (uint v = 0; v < 4; ++v) {
                    const uint u = q[v];
                    for (uint e = 0; e < 4; ++e) {
                        const uint byte = (u >> (8 * e)) & 0xFF;
                        al += float(byte & 0x0F) * xl[v * 4 + e];
                        ah += float(byte >> 4) * xh[v * 4 + e];
                    }
                }
                sums[r] += float(b->d) * (sm1.x * al + sm2.x * ah) - float(b->dmin) * (sm1.y * sl + sm2.y * shs);
            }
        }
        for (uint r = 0; r < 4; ++r) {
            const float tot = simd_sum(sums[r]);
            const uint row = row0 + r;
            if (lane == 0 && row < n) {
                device float* o = y + (ulong)mi * ldy + row;
                *o = accumulate ? *o + tot : tot;
            }
        }
    }
}

// ── Q6_K matrix-vector: 8 lanes per super-block ──
// Lane t covers half hf = t/4 and positions l0..l0+8 (l0 = 8*(t%4)) of its four 32-value
// groups: 8 bytes from each ql half, 8 bytes of qh, one scale per group.
kernel void mv_q6_k(LINEAR_ARGS, uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint nb = k / 256;
    const uint row0 = (tg * 2 + sg) * 4;
    const uint ix = lane / 8, t = lane % 8, hf = t / 4, l0 = (t % 4) * 8;
    device const block_q6_K* wb = (device const block_q6_K*)w;
    for (uint mi = 0; mi < m; ++mi) {
        device const float* xr = x + (ulong)mi * k;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint ib = ix; ib < nb; ib += 4) {
            float xg[4][8];
            float sx[4];
            for (uint g = 0; g < 4; ++g) {
                device const float4* p = (device const float4*)(xr + ib * 256 + hf * 128 + g * 32 + l0);
                const float4 a = p[0], b = p[1];
                for (uint e = 0; e < 4; ++e) { xg[g][e] = a[e]; xg[g][4 + e] = b[e]; }
                sx[g] = (a[0] + a[1] + a[2] + a[3]) + (b[0] + b[1] + b[2] + b[3]);
            }
            for (uint r = 0; r < 4; ++r) {
                const uint row = row0 + r;
                if (row >= n) break;
                device const block_q6_K* b = wb + (ulong)row * nb + ib;
                device const uchar* ql = b->ql + hf * 64 + l0;
                device const uchar* qh = b->qh + hf * 32 + l0;
                device const char* sc = b->scales + hf * 8 + l0 / 16;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (uint e = 0; e < 8; ++e) {
                    const uint lo = ql[e], hi = ql[32 + e], hb = qh[e];
                    a0 += float((lo & 0x0F) | ((hb & 3) << 4)) * xg[0][e];
                    a1 += float((hi & 0x0F) | (((hb >> 2) & 3) << 4)) * xg[1][e];
                    a2 += float((lo >> 4) | (((hb >> 4) & 3) << 4)) * xg[2][e];
                    a3 += float((hi >> 4) | ((hb >> 6) << 4)) * xg[3][e];
                }
                // Stored values are q + 32; fold the offset in through sums of x.
                sums[r] += float(b->d) * (float(sc[0]) * (a0 - 32.0f * sx[0]) + float(sc[2]) * (a1 - 32.0f * sx[1])
                                        + float(sc[4]) * (a2 - 32.0f * sx[2]) + float(sc[6]) * (a3 - 32.0f * sx[3]));
            }
        }
        for (uint r = 0; r < 4; ++r) {
            const float tot = simd_sum(sums[r]);
            const uint row = row0 + r;
            if (lane == 0 && row < n) {
                device float* o = y + (ulong)mi * ldy + row;
                *o = accumulate ? *o + tot : tot;
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
// 1 or 4 simdgroups and 1, 4 or 8 blocks per step were up to 20% slower). With 16-32
// tokens, RG = 2 row groups per simdgroup reuse each activation fragment twice
// (-12 to -23% per Llama-3.2-1B layer; RG = 4 was mixed); at 8 tokens RG = 1 is faster
// (and for F16 weights at 16 tokens). NT = 3 token fragments for 17-24 tokens: padding
// 24 tokens to 32 made a layer 19-31% slower.
typedef decltype(mms<WQ8, 1, 2, 2, 1>) mms_t;
#define MMS(TAG, SUF, NT) template [[host_name("mms" #NT "_" SUF)]] kernel mms_t mms<TAG, NT, 2, 2, 1>;
#define MMSR(TAG, SUF, NT, RG) template [[host_name("mms" #NT "r" #RG "_" SUF)]] kernel mms_t mms<TAG, NT, 2, 2, RG>;
#define MMS_ALL(TAG, SUF) MMS(TAG, SUF, 1) MMS(TAG, SUF, 2) MMSR(TAG, SUF, 2, 2) MMSR(TAG, SUF, 3, 2) MMSR(TAG, SUF, 4, 2)
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
                     uint n, uint k, uint m, uint acc_in, uint ldy, uint2 tg, uint tid, uint sg,
                     threadgroup half* sa, threadgroup half* sb, threadgroup float* stage) {
    const uint r0 = tg.x * 64, r1 = tg.y * 32;
    // weight staging: thread -> row tid/2, values 16*(tid%2) .. +16 of each 32-block
    const uint lr0 = min(r0 + tid / 2, n - 1);
    // activation staging: thread -> token tid/4, k offset 8*(tid%4)
    const uint tok = tid / 4;
    const bool tok_ok = r1 + tok < m;
    device const half* xr = x + (ulong)min(r1 + tok, m - 1) * k + 8 * (tid % 4);
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint nb = k / 32;
    for (uint b = 0; b < nb; b += NB) mml_stage<T, NB>(w, xr, tok_ok, lr0, k, b, tid, sg, sa, sb, mc);
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
    } else if (r0 + 64 <= n && r1 + 32 <= m) {
        // Accumulation adds the previous values through a product with the identity.
        const simdgroup_float8x8 ident = simdgroup_float8x8(1.0f);
        for (uint i = 0; i < 8; ++i) {
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
kernel void NAME(LINEAR_ARGS, uint2 tg [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float stage[3 * 64 * 32 / 2];   /* 12 KB: 2 x (64x32 + 32x32) halves */ \
    threadgroup half* sa = (threadgroup half*)stage; \
    threadgroup half* sb = sa + 2 * 64 * 32; \
    mml_impl<TAG, NB, GLU>(w, (device const half*)x, y, n, k, m, accumulate, ldy, tg, tid, sg, sa, sb, stage); }
#define MML_ALL(TAG, SUF) MML_KERNEL(mml_##SUF, TAG, 2, false) MML_KERNEL(mml1_##SUF, TAG, 1, false) \
    MML_KERNEL(mmlg_##SUF, TAG, 2, true) MML_KERNEL(mmlg1_##SUF, TAG, 1, true)
MML_ALL(WQ8, q8_0)
MML_ALL(WQ4, q4_0)
MML_ALL(WF16, f16)
MML_ALL(WQ4K, q4_k)
MML_ALL(WQ6K, q6_k)

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
        for (uint u = 0; u < cnt; ++u) {
            const float pu = simd_shuffle(p, u);
            const uint tu = t0 + u;
            device const KVT* vr = vc + table[tu / bs] * block_stride + head_off + (tu % bs) * HD;
            for (uint i = 0; i < dpl; ++i) acc[i] += pu * float(vr[lane + i * 32]);
        }
        m = mn;
    }

    if (lane == 0) { tm[sg] = m; ts[sg] = s; }
    for (uint i = 0; i < dpl; ++i) tacc[sg * HD + lane + i * 32] = acc[i];
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
        for (uint u = 0; u < cnt; ++u) {
            const uint tu = t0 + u;
            device const KVT* vr = vc + table[tu / bs] * block_stride + head_off + (tu % bs) * HD;
            float v[dpl];
            for (uint i = 0; i < dpl; ++i) v[i] = float(vr[lane + i * 32]);
            for (uint g = 0; g < G; ++g) {
                const float pu = simd_shuffle(p[g], u);
                for (uint i = 0; i < dpl; ++i) acc[g][i] += pu * v[i];
            }
        }
    }

    for (uint g = 0; g < G; ++g) {
        if (lane == 0) { tm[sg * G + g] = m[g]; ts[sg * G + g] = s[g]; }
        for (uint i = 0; i < dpl; ++i) tacc[(sg * G + g) * HD + lane + i * 32] = acc[g][i];
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

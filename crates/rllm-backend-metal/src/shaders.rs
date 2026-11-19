//! Metal Shading Language kernels for the LLaMA executor.
//!
//! Activations are F32. Weights are F32, Q8_0 or Q4_0 in their GGUF block layout.
//! Kernels assume `in_features % 32 == 0` and `head_dim % 32 == 0` (checked on the host).
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

// ── weight access: 8 consecutive values (quarter q of 32-value block ib of a row) ──
struct WF32 {};
struct WQ8 {};
struct WQ4 {};
struct WF16 {};

inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WF32) {
    device const float* r = (device const float*)w + (ulong)row * k + ib * 32 + q * 8;
    for (uint j = 0; j < 8; ++j) out[j] = r[j];
}
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WF16) {
    device const half4* r = (device const half4*)((device const half*)w + (ulong)row * k + ib * 32 + q * 8);
    const float4 a = float4(r[0]), b = float4(r[1]);
    for (uint j = 0; j < 4; ++j) { out[j] = a[j]; out[4 + j] = b[j]; }
}
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WQ8) {
    device const block_q8_0* b = (device const block_q8_0*)w + (ulong)row * (k / 32) + ib;
    const float d = float(b->d);
    for (uint j = 0; j < 8; ++j) out[j] = d * float(b->qs[q * 8 + j]);
}
inline void load8(device const uchar* w, uint row, uint k, uint ib, uint q, thread float* out, WQ4) {
    device const block_q4_0* b = (device const block_q4_0*)w + (ulong)row * (k / 32) + ib;
    const float d = float(b->d);
    if (q < 2) {
        for (uint j = 0; j < 8; ++j) out[j] = d * (float(b->qs[q * 8 + j] & 0x0F) - 8.0f);
    } else {
        for (uint j = 0; j < 8; ++j) out[j] = d * (float(b->qs[(q - 2) * 8 + j] >> 4) - 8.0f);
    }
}

#define LINEAR_ARGS \
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]], constant uint& n [[buffer(3)]], \
    constant uint& k [[buffer(4)]], constant uint& m [[buffer(5)]], \
    constant uint& accumulate [[buffer(6)]]

// ── matrix-vector: y[m, n] (+)= x[m, k] · W[n, k]^T for small m (decode) ──
// Threadgroup = 2 simdgroups; each simdgroup computes 4 output rows.
template <typename T>
inline void mv_impl(device const uchar* w, device const float* x, device float* y,
                    uint n, uint k, uint m, uint acc, uint tg, uint sg, uint lane) {
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
                device float* o = y + (ulong)mi * n + row;
                *o = acc ? *o + t : t;
            }
        }
    }
}

#define MV_KERNEL(NAME, TAG) \
kernel void NAME(LINEAR_ARGS, uint tg [[threadgroup_position_in_grid]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    mv_impl<TAG>(w, x, y, n, k, m, accumulate, tg, sg, lane); }

MV_KERNEL(mv_f32, WF32)
MV_KERNEL(mv_q8_0, WQ8)
MV_KERNEL(mv_q4_0, WQ4)
MV_KERNEL(mv_f16, WF16)

constant constexpr uint BK = 32;

// ── F32-weight tiled matrix-matrix (prefill), 32x32 output tile, full F32 ──
template <typename T>
inline void mm32_impl(device const uchar* w, device const float* x, device float* y,
                      uint n, uint k, uint m, uint acc_in, uint2 tg, uint tid, uint sg,
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
                simdgroup_store(acc[a][b], y + (ulong)(row0 + si + a * 8) * n + col0 + sj + b * 8, n);
    } else {
        for (uint a = 0; a < 2; ++a)
            for (uint b = 0; b < 2; ++b) simdgroup_store(acc[a][b], sc + (si + a * 8) * 32 + sj + b * 8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 32 * 32; i += 128) {
            const uint r = row0 + i / 32, c = col0 + i % 32;
            if (r < m && c < n) {
                device float* o = y + (ulong)r * n + c;
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
    mm32_impl<WF32>(w, x, y, n, k, m, accumulate, tg, tid, sg, sa, sb, sc);
}

// ── quantized-weight tiled matrix-matrix (prefill), 64x64 output tile ──
// 64 activation rows x 64 weight rows per threadgroup (128 threads, 4 simdgroups, each
// 32x32 = 4x4 accumulators). Each k-step dequantizes a 64x32 weight tile into
// half-precision threadgroup memory stored k-major (fragments load untransposed) and
// copies a 64x32 activation tile as float4; products are float x half -> float.
// Chosen by A/B on an M4 against 32x32, 32x64 (with/without prefetch, half activations),
// 8-simdgroup and transposed-output variants.
constant constexpr uint TQ = 64;

template <typename T>
inline void mmq_impl(device const uchar* w, device const float* x, device float* y,
                     uint n, uint k, uint m, uint acc_in, uint2 tg, uint tid, uint sg,
                     threadgroup char* shmem) {
    threadgroup half*  sw = (threadgroup half*)shmem;                // [BK][TQ] halves  4 KB
    threadgroup float* sx = (threadgroup float*)(shmem + 4096);      // [TQ][BK] floats  8 KB
    const uint row0 = tg.y * TQ, col0 = tg.x * TQ;
    const uint xi = (sg / 2) * 32, wj = (sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint a = 0; a < 4; ++a)
        for (uint b = 0; b < 4; ++b) acc[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < k; k0 += BK) {
        for (uint idx = tid; idx < TQ * 4; idx += 128) {
            const uint wr = idx / 4, q = idx % 4;
            float v[8];
            if (col0 + wr < n) load8(w, col0 + wr, k, k0 / 32, q, v, T());
            else for (uint j = 0; j < 8; ++j) v[j] = 0.0f;
            for (uint j = 0; j < 8; ++j) sw[(q * 8 + j) * TQ + wr] = half(v[j]);
        }
        for (uint idx = tid; idx < TQ * 4; idx += 128) {
            const uint xr = idx / 4, xq = idx % 4;
            threadgroup float4* dst = (threadgroup float4*)(sx + xr * BK + xq * 8);
            if (row0 + xr < m) {
                device const float4* src = (device const float4*)(x + (ulong)(row0 + xr) * k + k0 + xq * 8);
                dst[0] = src[0];
                dst[1] = src[1];
            } else {
                dst[0] = 0.0f;
                dst[1] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 fx[4];
            simdgroup_half8x8 fw[4];
            for (uint a = 0; a < 4; ++a) simdgroup_load(fx[a], sx + (xi + a * 8) * BK + kk, BK);
            for (uint b = 0; b < 4; ++b) simdgroup_load(fw[b], sw + kk * TQ + wj + b * 8, TQ);
            for (uint a = 0; a < 4; ++a)
                for (uint b = 0; b < 4; ++b) simdgroup_multiply_accumulate(acc[a][b], fx[a], fw[b], acc[a][b]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Accumulation (y += ...) and partial tiles go through threadgroup staging so the
    // k-loop above stays identical to the plain kernel (keeps accumulators in registers).
    if (!acc_in && row0 + TQ <= m && col0 + TQ <= n) {
        for (uint a = 0; a < 4; ++a)
            for (uint b = 0; b < 4; ++b)
                simdgroup_store(acc[a][b], y + (ulong)(row0 + xi + a * 8) * n + col0 + wj + b * 8, n);
    } else {
        threadgroup float* sc = (threadgroup float*)shmem;           // [TQ][TQ] floats 16 KB
        for (uint a = 0; a < 4; ++a)
            for (uint b = 0; b < 4; ++b)
                simdgroup_store(acc[a][b], sc + (xi + a * 8) * TQ + wj + b * 8, TQ);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < TQ * TQ; i += 128) {
            const uint r = row0 + i / TQ, c = col0 + i % TQ;
            if (r < m && c < n) {
                device float* o = y + (ulong)r * n + c;
                *o = acc_in ? *o + sc[i] : sc[i];
            }
        }
    }
}

#define MMQ_KERNEL(NAME, TAG) \
kernel void NAME(LINEAR_ARGS, uint2 tg [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup char shmem[16384]; \
    mmq_impl<TAG>(w, x, y, n, k, m, accumulate, tg, tid, sg, shmem); }

MMQ_KERNEL(mm_q8_0, WQ8)
MMQ_KERNEL(mm_q4_0, WQ4)
MMQ_KERNEL(mm_f16, WF16)

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

// ── RMSNorm: one threadgroup (256 threads) per output row; optional row gather ──
kernel void rms_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                     device float* y [[buffer(2)]], device const uint* rows [[buffer(3)]],
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
    for (uint i = tid; i < h; i += 256) y[(ulong)r * h + i] = xr[i] * inv * w[i];
}

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
kernel void kv_write(device const float* qkv [[buffer(0)]], constant uint& k_off [[buffer(1)]],
                     constant uint& v_off [[buffer(2)]], constant uint& stride [[buffer(3)]],
                     device float* kc [[buffer(4)]], device float* vc [[buffer(5)]],
                     device const uint* tables [[buffer(6)]], device const uint* table_off [[buffer(7)]],
                     device const uint* pos [[buffer(8)]], constant uint& nkv [[buffer(9)]],
                     constant uint& hd [[buffer(10)]], constant uint& bs [[buffer(11)]],
                     uint3 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, h = gid.y, r = gid.z;
    const uint p = pos[r];
    const uint phys = tables[table_off[r] + p / bs];
    const ulong dst = (ulong)phys * nkv * bs * hd + (ulong)h * bs * hd + (p % bs) * hd + d;
    device const float* row = qkv + (ulong)r * stride + h * hd + d;
    kc[dst] = row[k_off];
    vc[dst] = row[v_off];
}

#define ATTN_ARGS \
    device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], \
    device const float* vc [[buffer(2)]], device float* out [[buffer(3)]], \
    device const uint* tables [[buffer(4)]], device const uint* table_off [[buffer(5)]], \
    device const uint* pos [[buffer(6)]], constant uint& nq [[buffer(7)]], \
    constant uint& nkv [[buffer(8)]], constant uint& hd [[buffer(9)]], \
    constant uint& bs [[buffer(10)]], constant float& scale [[buffer(11)]], \
    constant uint& q_stride [[buffer(13)]]

// ── causal paged attention, one threadgroup (4 simdgroups) per (query head, row) ──
// Positions are processed in tiles of 32: each lane scores one position against the
// query (held in threadgroup memory), the tile updates an online softmax, and values
// are accumulated with each lane owning hd/32 output dimensions. Simdgroups take
// interleaved tiles and are merged at the end. Used for decode rows.
kernel void attention(ATTN_ARGS, device const uint* row_list [[buffer(12)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NSG = 4;
    threadgroup float qs[256];
    threadgroup float tm[NSG];
    threadgroup float ts[NSG];
    threadgroup float tacc[NSG * 256];

    const uint head = tg.x, r = row_list[tg.y];
    const uint kvh = head / (nq / nkv);
    const uint ctx = pos[r] + 1;
    const uint dpl = hd / 32;
    const ulong block_stride = (ulong)nkv * bs * hd;
    const ulong head_off = (ulong)kvh * bs * hd;
    device const uint* table = tables + table_off[r];

    device const float* qr = q + (ulong)r * q_stride + head * hd;
    for (uint i = sg * 32 + lane; i < hd; i += NSG * 32) qs[i] = qr[i] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, s = 0.0f;
    float acc[8];
    for (uint i = 0; i < dpl; ++i) acc[i] = 0.0f;

    for (uint t0 = sg * 32; t0 < ctx; t0 += NSG * 32) {
        const uint t = t0 + lane;
        float score = -INFINITY;
        if (t < ctx) {
            device const float* kr = kc + table[t / bs] * block_stride + head_off + (t % bs) * hd;
            float d = 0.0f;
            for (uint j = 0; j < hd; ++j) d += qs[j] * kr[j];
            score = d;
        }
        const float mn = max(m, simd_max(score));
        const float corr = exp(m - mn);
        const float p = (t < ctx) ? exp(score - mn) : 0.0f;
        s = s * corr + simd_sum(p);
        for (uint i = 0; i < dpl; ++i) acc[i] *= corr;
        const uint cnt = min(32u, ctx - t0);
        for (uint u = 0; u < cnt; ++u) {
            const float pu = simd_shuffle(p, u);
            const uint tu = t0 + u;
            device const float* vr = vc + table[tu / bs] * block_stride + head_off + (tu % bs) * hd;
            for (uint i = 0; i < dpl; ++i) acc[i] += pu * vr[lane + i * 32];
        }
        m = mn;
    }

    if (lane == 0) { tm[sg] = m; ts[sg] = s; }
    for (uint i = 0; i < dpl; ++i) tacc[sg * hd + lane + i * 32] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float M = -INFINITY;
        for (uint g = 0; g < NSG; ++g) M = max(M, tm[g]);
        float S = 0.0f;
        for (uint g = 0; g < NSG; ++g) S += ts[g] * exp(tm[g] - M);
        for (uint i = 0; i < dpl; ++i) {
            const uint dim = lane + i * 32;
            float o = 0.0f;
            for (uint g = 0; g < NSG; ++g) o += tacc[g * hd + dim] * exp(tm[g] - M);
            out[((ulong)r * nq + head) * hd + dim] = o / S;
        }
    }
}

// ── causal attention for a tile of up to 8 consecutive query rows of one sequence ──
// One simdgroup per (head, tile). Keys/values are read once per tile as 8x8 fragments
// straight from the paged cache (8-key groups never cross a block when bs % 8 == 0);
// scores and outputs use simdgroup matrices, with a per-row online softmax. Prefill.
kernel void attention_tile(ATTN_ARGS, device const uint2* tiles [[buffer(12)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float S[64];
    threadgroup float P[64];
    threadgroup float D[64];
    threadgroup float rowm[8];
    threadgroup float rows[8];
    threadgroup float Ot[8 * 128];

    const uint head = tg.x;
    const uint r0 = tiles[tg.y].x, cnt = tiles[tg.y].y;
    const uint kvh = head / (nq / nkv);
    device const uint* table = tables + table_off[r0];
    const uint nkeys = pos[r0 + cnt - 1] + 1;
    const uint nd = hd / 8;
    const ulong block_stride = (ulong)nkv * bs * hd;
    const ulong head_off = (ulong)kvh * bs * hd;

    simdgroup_float8x8 Qf[16];
    simdgroup_float8x8 O[16];
    device const float* qbase = q + (ulong)r0 * q_stride + head * hd;   // rows past cnt are padding
    for (uint i = 0; i < nd; ++i) {
        simdgroup_load(Qf[i], qbase + i * 8, q_stride);
        O[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    if (lane < 8) { rowm[lane] = -INFINITY; rows[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t0 = 0; t0 < nkeys; t0 += 8) {
        const ulong kbase = (ulong)table[t0 / bs] * block_stride + head_off + (t0 % bs) * hd;
        simdgroup_float8x8 sf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint i = 0; i < nd; ++i) {
            simdgroup_float8x8 kf;
            simdgroup_load(kf, kc + kbase + i * 8, hd, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sf, Qf[i], kf, sf);
        }
        simdgroup_store(sf, S, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8) {
            const uint pr = pos[r0 + min(lane, cnt - 1)];
            float vals[8];
            float mx = rowm[lane];
            for (uint j = 0; j < 8; ++j) {
                vals[j] = (t0 + j <= pr) ? S[lane * 8 + j] * scale : -INFINITY;
                mx = max(mx, vals[j]);
            }
            const float corr = (rowm[lane] == -INFINITY) ? 0.0f : exp(rowm[lane] - mx);
            float sum = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                const float p = (vals[j] == -INFINITY) ? 0.0f : exp(vals[j] - mx);
                P[lane * 8 + j] = p;
                sum += p;
            }
            rows[lane] = rows[lane] * corr + sum;
            rowm[lane] = mx;
            for (uint j = 0; j < 8; ++j) D[lane * 8 + j] = (j == lane) ? corr : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 pf, df;
        simdgroup_load(pf, P, 8);
        simdgroup_load(df, D, 8);
        for (uint i = 0; i < nd; ++i) {
            simdgroup_float8x8 vf;
            simdgroup_load(vf, vc + kbase + i * 8, hd);
            simdgroup_multiply(O[i], df, O[i]);
            simdgroup_multiply_accumulate(O[i], pf, vf, O[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < nd; ++i) simdgroup_store(O[i], Ot + i * 8, hd);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < cnt * hd; idx += 32) {
        const uint row = idx / hd, d = idx % hd;
        out[((ulong)(r0 + row) * nq + head) * hd + d] = Ot[row * hd + d] / rows[row];
    }
}

// ── SwiGLU on the fused gate/up buffer: out[r, j] = silu(gu[r, j]) * gu[r, inter + j] ──
kernel void swiglu(device const float* gu [[buffer(0)]], device float* out [[buffer(1)]],
                   constant uint& inter [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint r = i / inter, j = i % inter;
    const float a = gu[(ulong)r * 2 * inter + j];
    const float u = gu[(ulong)r * 2 * inter + inter + j];
    out[i] = a / (1.0f + exp(-a)) * u;
}
"#;

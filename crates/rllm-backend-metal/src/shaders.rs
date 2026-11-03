//! Metal Shading Language (MSL) GPU kernel sources for Apple Silicon hardware acceleration.
//!
//! Includes threadgroup-parallelized kernels for RMSNorm, Rotary Position Embedding (RoPE),
//! and matrix-vector multiplication (GEMV).

/// Embedded MSL shader source code compiled at runtime via `device.new_library_with_source`.
pub const SHADERS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 1. RMSNorm Kernel
kernel void rms_norm(
    device const float* in       [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant uint&      dim      [[buffer(3)]],
    constant float&     eps      [[buffer(4)]],
    uint                tid      [[threadgroup_position_in_grid]],
    uint                lid      [[thread_position_in_threadgroup]],
    uint                threads  [[threads_per_threadgroup]]
) {
    uint row_offset = tid * dim;
    threadgroup float sdata[256];

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = lid; i < dim; i += threads) {
        float val = in[row_offset + i];
        sum_sq += val * val;
    }
    sdata[lid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction in threadgroup
    for (uint s = threads / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sdata[lid] += sdata[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(sdata[0] / float(dim) + eps);
    float inv_rms = 1.0f / rms;

    // Apply normalization and scale
    for (uint i = lid; i < dim; i += threads) {
        out[row_offset + i] = in[row_offset + i] * inv_rms * weight[i];
    }
}

// 2. RoPE Kernel
kernel void rope_embedding(
    device float*       q         [[buffer(0)]],
    device float*       k         [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    constant uint&      head_dim  [[buffer(4)]],
    uint2               pos       [[thread_position_in_grid]]
) {
    uint head_idx = pos.y;
    uint half_dim_idx = pos.x;
    if (half_dim_idx >= head_dim / 2) return;

    uint q_offset = head_idx * head_dim + half_dim_idx * 2;
    float c = cos_table[half_dim_idx];
    float s = sin_table[half_dim_idx];

    // Rotate Q
    float q0 = q[q_offset];
    float q1 = q[q_offset + 1];
    q[q_offset] = q0 * c - q1 * s;
    q[q_offset + 1] = q0 * s + q1 * c;

    // Rotate K
    float k0 = k[q_offset];
    float k1 = k[q_offset + 1];
    k[q_offset] = k0 * c - k1 * s;
    k[q_offset + 1] = k0 * s + k1 * c;
}

// 3. GEMV Matrix-Vector Multiplication: Y = W * X
kernel void gemv_f32(
    device const float* weight [[buffer(0)]],
    device const float* act    [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      k      [[buffer(3)]],
    uint                row    [[thread_position_in_grid]]
) {
    uint offset = row * k;
    float sum = 0.0f;
    for (uint i = 0; i < k; ++i) {
        sum += weight[offset + i] * act[i];
    }
    out[row] = sum;
}
"#;

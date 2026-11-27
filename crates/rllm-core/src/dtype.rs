//! Data types, precisions, and quantization formats.

use serde::{Deserialize, Serialize};

/// Scalar floating-point, integer, and block-quantized memory layouts.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit single-precision IEEE 754 float (4 bytes).
    F32,
    /// 16-bit half-precision IEEE 754 float (2 bytes).
    F16,
    /// 16-bit Brain Floating Point (2 bytes, 8-bit exponent).
    BF16,
    /// 32-bit signed integer (4 bytes).
    I32,
    /// 8-bit signed integer (1 byte).
    I8,
    /// 4-bit block quantization (block size = 32 elements in 18 bytes: 16-bit f16 scale + 16 bytes nibbles).
    Q4_0,
    /// 4-bit K-quant (GGUF Q4_K_M layout with sub-block scales).
    Q4_K,
    /// 8-bit block quantization (block size = 32 elements in 34 bytes: 16-bit f16 scale + 32 signed int8s).
    Q8_0,
    /// 5-bit block quantization (32 elements in 22 bytes: f16 scale, 32 high bits, 16 bytes of nibbles).
    Q5_0,
    /// 6-bit K-quant (256-element super-blocks of 210 bytes with 16 int8 sub-block scales).
    Q6_K,
    /// CPU-internal repacked `Q4_0`: groups of four rows with their blocks interleaved
    /// (four f16 scales, then the four rows' nibbles in 8-byte chunks, stored as signed
    /// 4-bit values). Same size as `Q4_0`; a row count that is a multiple of 4 is required,
    /// and row views must start on a 4-row boundary.
    Q4_0x4,
    /// CPU-internal (x86) repacked `Q8_0`: groups of eight rows; per 32-value block, the
    /// eight f16 scales, then eight 32-byte chunks each holding four consecutive values
    /// of every row. Same size as `Q8_0`; rows a multiple of 8, row views on 8-row
    /// boundaries.
    Q8_0x8,
    /// CPU-internal (x86) repacked `Q4_0`: as [`DType::Q8_0x8`] with four 32-byte chunks
    /// of nibble bytes (byte `i` of a row's block holds values `i` and `16 + i`). Same
    /// size as `Q4_0`.
    Q4_0x8,
    /// CPU-internal (x86) repacked `Q4_K`: groups of eight rows; per 256-value
    /// super-block the rows' scales, unpacked 6-bit sub-block scales and mins, then 4-byte
    /// chunks of all eight rows' nibbles. 148 bytes per row and super-block (`Q4_K`: 144).
    Q4_Kx8,
    /// CPU-internal (x86) repacked `Q6_K`: as [`DType::Q4_Kx8`] with the rows' scales,
    /// sub-block scales, then 4-byte chunks of `ql` and of `qh`. Same size as `Q6_K`.
    Q6_Kx8,
}

impl DType {
    /// Returns the element size in bytes for scalar unquantized formats.
    /// For block-quantized types, returns 0 (use [`DType::storage_bytes`]).
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
            DType::I8 => 1,
            DType::Q4_0 => 0,
            DType::Q4_K => 0,
            DType::Q8_0
            | DType::Q5_0
            | DType::Q6_K
            | DType::Q4_0x4
            | DType::Q8_0x8
            | DType::Q4_0x8
            | DType::Q4_Kx8
            | DType::Q6_Kx8 => 0,
        }
    }

    /// Returns true if this data type is a block-quantized representation.
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            DType::Q4_0
                | DType::Q4_K
                | DType::Q8_0
                | DType::Q5_0
                | DType::Q6_K
                | DType::Q4_0x4
                | DType::Q8_0x8
                | DType::Q4_0x8
                | DType::Q4_Kx8
                | DType::Q6_Kx8
        )
    }

    /// Returns `(block_size, bytes_per_block)` for quantized types.
    pub fn block_info(&self) -> Option<(usize, usize)> {
        match self {
            DType::Q4_0 | DType::Q4_0x4 | DType::Q4_0x8 => Some((32, 18)),
            DType::Q5_0 => Some((32, 22)),
            DType::Q6_K | DType::Q6_Kx8 => Some((256, 210)),
            DType::Q4_Kx8 => Some((256, 148)),
            DType::Q4_K => Some((256, 144)),
            DType::Q8_0 | DType::Q8_0x8 => Some((32, 34)),
            _ => None,
        }
    }

    /// Computes total storage bytes required to store `num_elements` elements.
    pub fn storage_bytes(&self, num_elements: usize) -> usize {
        if let Some((block_size, bytes_per_block)) = self.block_info() {
            let num_blocks = num_elements.div_ceil(block_size);
            num_blocks * bytes_per_block
        } else {
            num_elements * self.size_in_bytes()
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::I8 => write!(f, "i8"),
            DType::Q4_0 => write!(f, "q4_0"),
            DType::Q4_K => write!(f, "q4_k"),
            DType::Q8_0 => write!(f, "q8_0"),
            DType::Q5_0 => write!(f, "q5_0"),
            DType::Q6_K => write!(f, "q6_k"),
            DType::Q4_0x4 => write!(f, "q4_0x4"),
            DType::Q8_0x8 => write!(f, "q8_0x8"),
            DType::Q4_0x8 => write!(f, "q4_0x8"),
            DType::Q4_Kx8 => write!(f, "q4_kx8"),
            DType::Q6_Kx8 => write!(f, "q6_kx8"),
        }
    }
}

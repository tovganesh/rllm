use serde::{Deserialize, Serialize};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I8,
    /// 4-bit block quantization (block size = 32, 18 bytes per block: 16-bit f16 scale + 16 bytes payload)
    Q4_0,
    /// 4-bit K-quant (GGUF Q4_K_M layout)
    Q4_K,
    /// 8-bit block quantization (block size = 32, 34 bytes per block: 16-bit f16 scale + 32 bytes payload)
    Q8_0,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
            DType::I8 => 1,
            DType::Q4_0 => 0, // Block quantized; call block_bytes_and_size instead
            DType::Q4_K => 0,
            DType::Q8_0 => 0,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Q4_0 | DType::Q4_K | DType::Q8_0)
    }

    /// Returns (block_size, bytes_per_block) for quantized types.
    pub fn block_info(&self) -> Option<(usize, usize)> {
        match self {
            DType::Q4_0 => Some((32, 18)),
            DType::Q4_K => Some((256, 144)),
            DType::Q8_0 => Some((32, 34)),
            _ => None,
        }
    }

    /// Calculate total bytes required for a given number of elements.
    pub fn storage_bytes(&self, num_elements: usize) -> usize {
        if let Some((block_size, bytes_per_block)) = self.block_info() {
            let num_blocks = (num_elements + block_size - 1) / block_size;
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
        }
    }
}

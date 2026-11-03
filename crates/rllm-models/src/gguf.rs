//! GGUF binary format parser and deserializer.
//!
//! Implements a zero-copy friendly reader for GGUF model files (v2/v3), parsing header
//! magic, metadata key-value pairs, tensor information, and tensor offsets.

use byteorder::{LittleEndian, ReadBytesExt};
use rllm_core::{
    dtype::DType,
    error::{Result, RllmError},
};
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};

/// Magic header constant for GGUF files (`"GGUF"` in little-endian ASCII: `0x46554747`).
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Metadata and dimension descriptor for an individual tensor within a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Name of the tensor in GGUF naming convention (e.g., `blk.0.attn_q.weight`).
    pub name: String,
    /// Shape of the tensor in row-major order.
    pub shape: Vec<usize>,
    /// Quantized or floating point data type.
    pub dtype: DType,
    /// Byte offset relative to the file's data segment.
    pub offset: u64,
}

/// Parsed representation of a GGUF binary file.
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// GGUF specification format version (usually 2 or 3).
    pub version: u32,
    /// Extracted metadata key-value pairs (e.g., architecture, context length).
    pub metadata: HashMap<String, String>,
    /// Map of tensor name to tensor metadata descriptor.
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Byte offset within the file where raw tensor payload buffers begin.
    pub data_offset: u64,
}

impl GgufFile {
    /// Parses a GGUF file from an in-memory byte slice.
    ///
    /// Validates the magic constant, parses version, metadata key-value attributes,
    /// and tensor descriptors, and aligns the data offset to 32 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`RllmError::ModelError`] if magic mismatch or corrupted structure,
    /// or [`RllmError::QuantizationError`] if an unsupported quantization type is encountered.
    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = Cursor::new(bytes);

        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(RllmError::ModelError(format!(
                "Invalid GGUF magic: expected 0x{:08X}, got 0x{:08X}",
                GGUF_MAGIC, magic
            )));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        let tensor_count = reader.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut reader)?;
            let val_type = reader.read_u32::<LittleEndian>()?;
            let val_str = read_gguf_val(&mut reader, val_type)?;
            metadata.insert(key, val_str);
        }

        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(reader.read_u64::<LittleEndian>()? as usize);
            }
            shape.reverse(); // GGUF stores dimensions in column-major order; reverse to row-major

            let type_id = reader.read_u32::<LittleEndian>()?;
            let dtype = match type_id {
                0 => DType::F32,
                1 => DType::F16,
                2 => DType::Q4_0,
                8 => DType::Q8_0,
                12 => DType::Q4_K,
                other => {
                    return Err(RllmError::QuantizationError(format!(
                        "Unsupported GGUF dtype id: {}",
                        other
                    )))
                }
            };
            let offset = reader.read_u64::<LittleEndian>()?;

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                },
            );
        }

        let current_pos = reader.position();
        let alignment = 32u64;
        let data_offset = (current_pos + alignment - 1) & !(alignment - 1);

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| RllmError::ModelError(e.to_string()))
}

fn read_gguf_val<R: Read + Seek>(reader: &mut R, val_type: u32) -> Result<String> {
    match val_type {
        0 => Ok(reader.read_u8()?.to_string()),
        1 => Ok(reader.read_i8()?.to_string()),
        2 => Ok(reader.read_u16::<LittleEndian>()?.to_string()),
        3 => Ok(reader.read_i16::<LittleEndian>()?.to_string()),
        4 => Ok(reader.read_u32::<LittleEndian>()?.to_string()),
        5 => Ok(reader.read_i32::<LittleEndian>()?.to_string()),
        6 => Ok(reader.read_f32::<LittleEndian>()?.to_string()),
        7 => Ok(reader.read_u8()?.to_string()), // bool
        8 => read_gguf_string(reader),          // string
        9 => {
            // array
            let arr_type = reader.read_u32::<LittleEndian>()?;
            let arr_len = reader.read_u64::<LittleEndian>()? as usize;
            let mut items = Vec::new();
            for _ in 0..arr_len {
                items.push(read_gguf_val(reader, arr_type)?);
            }
            Ok(format!("[{}]", items.join(", ")))
        }
        10 => Ok(reader.read_u64::<LittleEndian>()?.to_string()),
        11 => Ok(reader.read_i64::<LittleEndian>()?.to_string()),
        12 => Ok(reader.read_f64::<LittleEndian>()?.to_string()),
        other => Err(RllmError::ModelError(format!(
            "Unsupported GGUF metadata type id: {}",
            other
        ))),
    }
}

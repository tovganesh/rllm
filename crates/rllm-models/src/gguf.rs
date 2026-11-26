//! GGUF binary format reader.
//!
//! Parses the header, typed metadata key/value pairs and tensor descriptors of GGUF
//! (v2/v3) files, and gives zero-copy access to tensor payloads through a memory map.
//!
//! Layout reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.

use byteorder::{LittleEndian, ReadBytesExt};
use rllm_core::error::{Result, RllmError};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

/// Magic header constant for GGUF files (`"GGUF"` read as a little-endian `u32`).
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Default alignment of the tensor data section when `general.alignment` is absent.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// GGML tensor element types that can appear in a GGUF file.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    BF16,
    /// A type ID this reader does not know about.
    Other(u32),
}

impl GgmlType {
    /// Maps a GGML type ID to a [`GgmlType`].
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    /// `(elements per block, bytes per block)`, or `None` for unknown types.
    pub fn block_layout(&self) -> Option<(usize, usize)> {
        Some(match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::Other(_) => return None,
        })
    }

    /// Number of bytes needed to store `numel` elements of this type.
    pub fn storage_bytes(&self, numel: usize) -> Option<usize> {
        let (block, bytes) = self.block_layout()?;
        if !numel.is_multiple_of(block) {
            return None;
        }
        Some(numel / block * bytes)
    }
}

/// A typed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Integer value as `u64`, for any integer or bool variant.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v as u64),
            Self::U16(v) => Some(v as u64),
            Self::U32(v) => Some(v as u64),
            Self::U64(v) => Some(v),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            Self::Bool(v) => Some(v as u64),
            _ => None,
        }
    }

    /// Numeric value as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(v as f64),
            Self::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    /// Boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// String value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Array elements.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Descriptor of one tensor stored in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g. `blk.0.attn_q.weight`).
    pub name: String,
    /// Shape in row-major order (outermost first). GGUF stores dimensions innermost
    /// first; they are reversed here, so a linear weight is `[out_features, in_features]`.
    pub shape: Vec<usize>,
    /// Element type.
    pub ggml_type: GgmlType,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Parsed GGUF header: metadata, tensor descriptors and data-section offset.
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// Format version (2 or 3).
    pub version: u32,
    /// Typed metadata key/value pairs.
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor descriptors by name.
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// Absolute byte offset of the tensor data section within the file.
    pub data_offset: u64,
}

impl GgufFile {
    /// Parses a GGUF header from an in-memory byte slice (the slice may be the whole
    /// file or just a prefix containing the full header).
    ///
    /// # Errors
    ///
    /// Returns [`RllmError::ModelError`] on a bad magic number, unsupported version,
    /// malformed metadata or truncated input.
    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Cursor::new(bytes);

        let magic = r.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(RllmError::ModelError(format!(
                "Invalid GGUF magic: expected 0x{:08X}, got 0x{:08X}",
                GGUF_MAGIC, magic
            )));
        }
        let version = r.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            return Err(RllmError::ModelError(format!(
                "Unsupported GGUF version {version} (expected 2 or 3)"
            )));
        }
        let tensor_count = r.read_u64::<LittleEndian>()?;
        let kv_count = r.read_u64::<LittleEndian>()?;
        // Guard against absurd counts from corrupted files before allocating.
        let remaining = bytes.len() as u64;
        if tensor_count > remaining || kv_count > remaining {
            return Err(RllmError::ModelError("Corrupted GGUF header counts".to_string()));
        }

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = read_string(&mut r)?;
            let ty = r.read_u32::<LittleEndian>()?;
            let value = read_value(&mut r, ty, 0)?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut r)?;
            let n_dims = r.read_u32::<LittleEndian>()? as usize;
            if n_dims > 8 {
                return Err(RllmError::ModelError(format!(
                    "Tensor '{name}' has {n_dims} dimensions"
                )));
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.read_u64::<LittleEndian>()? as usize);
            }
            shape.reverse();
            let ggml_type = GgmlType::from_id(r.read_u32::<LittleEndian>()?);
            let offset = r.read_u64::<LittleEndian>()?;
            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    shape,
                    ggml_type,
                    offset,
                },
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT)
            .max(1);
        let pos = r.position();
        let data_offset = pos.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Required unsigned integer metadata.
    pub fn get_u64(&self, key: &str) -> Result<u64> {
        self.get(key)
            .and_then(GgufValue::as_u64)
            .ok_or_else(|| RllmError::ModelError(format!("GGUF metadata '{key}' missing or not an integer")))
    }

    /// Required float metadata.
    pub fn get_f64(&self, key: &str) -> Result<f64> {
        self.get(key)
            .and_then(GgufValue::as_f64)
            .ok_or_else(|| RllmError::ModelError(format!("GGUF metadata '{key}' missing or not a number")))
    }

    /// Required string metadata.
    pub fn get_str(&self, key: &str) -> Result<&str> {
        self.get(key)
            .and_then(GgufValue::as_str)
            .ok_or_else(|| RllmError::ModelError(format!("GGUF metadata '{key}' missing or not a string")))
    }
}

/// A GGUF file opened through a read-only memory map.
pub struct MappedGguf {
    /// Parsed header.
    pub header: GgufFile,
    mmap: memmap2::Mmap,
}

impl MappedGguf {
    /// Memory-maps and parses a GGUF file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| RllmError::ModelError(format!("Cannot open model file {}: {e}", path.display())))?;
        // SAFETY: the map is read-only. As with any mmap, the file must not be truncated
        // or modified while it is open.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let header = GgufFile::read_from_bytes(&mmap)?;
        Ok(Self { header, mmap })
    }

    /// Raw bytes of a tensor's payload, validated against the file size.
    pub fn tensor_bytes(&self, info: &GgufTensorInfo) -> Result<&[u8]> {
        let size = info.ggml_type.storage_bytes(info.numel()).ok_or_else(|| {
            RllmError::QuantizationError(format!(
                "Tensor '{}' has unsupported type {:?} or a shape not divisible by its block size",
                info.name, info.ggml_type
            ))
        })?;
        let start = self.header.data_offset as usize + info.offset as usize;
        self.mmap
            .get(start..start + size)
            .ok_or_else(|| RllmError::ModelError(format!("Tensor '{}' extends past end of file", info.name)))
    }

    /// Descriptor of a named tensor.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.header.tensors.get(name)
    }
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = r.read_u64::<LittleEndian>()? as usize;
    if len > (1 << 30) {
        return Err(RllmError::ModelError(format!("GGUF string of {len} bytes")));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    // Some vocabularies contain invalid UTF-8 byte tokens; keep them lossily.
    Ok(String::from_utf8(buf).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
}

fn read_value<R: Read>(r: &mut R, ty: u32, depth: usize) -> Result<GgufValue> {
    Ok(match ty {
        0 => GgufValue::U8(r.read_u8()?),
        1 => GgufValue::I8(r.read_i8()?),
        2 => GgufValue::U16(r.read_u16::<LittleEndian>()?),
        3 => GgufValue::I16(r.read_i16::<LittleEndian>()?),
        4 => GgufValue::U32(r.read_u32::<LittleEndian>()?),
        5 => GgufValue::I32(r.read_i32::<LittleEndian>()?),
        6 => GgufValue::F32(r.read_f32::<LittleEndian>()?),
        7 => GgufValue::Bool(r.read_u8()? != 0),
        8 => GgufValue::String(read_string(r)?),
        9 => {
            if depth > 4 {
                return Err(RllmError::ModelError("GGUF arrays nested too deeply".to_string()));
            }
            let elem_ty = r.read_u32::<LittleEndian>()?;
            let len = r.read_u64::<LittleEndian>()? as usize;
            if len > (1 << 28) {
                return Err(RllmError::ModelError(format!("GGUF array of {len} elements")));
            }
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_value(r, elem_ty, depth + 1)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(r.read_u64::<LittleEndian>()?),
        11 => GgufValue::I64(r.read_i64::<LittleEndian>()?),
        12 => GgufValue::F64(r.read_f64::<LittleEndian>()?),
        other => {
            return Err(RllmError::ModelError(format!(
                "Unsupported GGUF metadata type id {other}"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        b.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn parses_typed_metadata_and_tensor_descriptors() {
        let mut b = Vec::new();
        b.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        b.write_u32::<LittleEndian>(3).unwrap();
        b.write_u64::<LittleEndian>(1).unwrap(); // tensors
        b.write_u64::<LittleEndian>(3).unwrap(); // kv pairs
        put_str(&mut b, "general.architecture");
        b.write_u32::<LittleEndian>(8).unwrap();
        put_str(&mut b, "llama");
        put_str(&mut b, "llama.rope.freq_base");
        b.write_u32::<LittleEndian>(6).unwrap();
        b.write_f32::<LittleEndian>(10000.0).unwrap();
        put_str(&mut b, "tokenizer.ggml.tokens");
        b.write_u32::<LittleEndian>(9).unwrap();
        b.write_u32::<LittleEndian>(8).unwrap();
        b.write_u64::<LittleEndian>(2).unwrap();
        put_str(&mut b, "a");
        put_str(&mut b, "b");
        put_str(&mut b, "blk.0.attn_q.weight");
        b.write_u32::<LittleEndian>(2).unwrap();
        b.write_u64::<LittleEndian>(64).unwrap(); // innermost (in_features)
        b.write_u64::<LittleEndian>(8).unwrap(); // out_features
        b.write_u32::<LittleEndian>(8).unwrap(); // Q8_0
        b.write_u64::<LittleEndian>(0).unwrap();

        let f = GgufFile::read_from_bytes(&b).unwrap();
        assert_eq!(f.get_str("general.architecture").unwrap(), "llama");
        assert_eq!(f.get_f64("llama.rope.freq_base").unwrap(), 10000.0);
        assert_eq!(f.get("tokenizer.ggml.tokens").unwrap().as_array().unwrap().len(), 2);
        let t = &f.tensors["blk.0.attn_q.weight"];
        assert_eq!(t.shape, vec![8, 64]);
        assert_eq!(t.ggml_type, GgmlType::Q8_0);
        assert_eq!(t.ggml_type.storage_bytes(t.numel()), Some(2 * 8 * 34));
        assert_eq!(f.data_offset % GGUF_DEFAULT_ALIGNMENT, 0);
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert!(GgufFile::read_from_bytes(b"NOPE\x03\0\0\0").is_err());
        let mut b = Vec::new();
        b.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        b.write_u32::<LittleEndian>(3).unwrap();
        assert!(GgufFile::read_from_bytes(&b).is_err());
    }
}

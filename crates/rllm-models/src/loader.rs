//! Loading LLaMA-family models from GGUF files.
//!
//! Supported today: `general.architecture = "llama"` (LLaMA 2/3, SmolLM, Mistral-style
//! models exported with the llama architecture). Weight types:
//!
//! | GGUF type | Handling |
//! | :--- | :--- |
//! | `F32` | used directly |
//! | `Q8_0`, `Q4_0` | kept quantized for linear layers and the embedding table |
//! | `F16`, `BF16`, `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K` | dequantized to F32 at load |

use crate::bpe::BpeTokenizer;
use crate::chat::ChatTemplate;
use crate::config::LlamaConfig;
use crate::gguf::{GgmlType, GgufFile, GgufValue, MappedGguf};
use crate::llama::{LlamaAttention, LlamaDecoderLayer, LlamaMlp, LlamaModel};
use rllm_backend_cpu::ops::quant::{
    convert_bf16, convert_f16, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};
use rllm_backend_cpu::ops::rope::precompute_rope_freqs_scaled;
use rllm_core::{
    backend::Backend,
    device::Device,
    dtype::DType,
    error::{Result, RllmError},
    shape::Shape,
    storage::{CpuStorage, Storage},
    tensor::Tensor,
};
use std::path::Path;
use std::sync::Arc;
use tracing::info;

/// Options controlling how a GGUF model is loaded.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Cap on the context window (and RoPE table size). Defaults to the model's
    /// `context_length`.
    pub max_context: Option<usize>,
}

impl LlamaConfig {
    /// Reads a LLaMA-architecture configuration from GGUF metadata.
    pub fn from_gguf(g: &GgufFile, opts: &LoadOptions) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        if arch != "llama" {
            return Err(RllmError::ModelError(format!(
                "Unsupported architecture '{arch}' (only 'llama' is supported)"
            )));
        }
        let key = |k: &str| format!("{arch}.{k}");

        let hidden_size = g.get_u64(&key("embedding_length"))? as usize;
        let num_attention_heads = g.get_u64(&key("attention.head_count"))? as usize;
        let num_key_value_heads = g
            .get(&key("attention.head_count_kv"))
            .and_then(GgufValue::as_u64)
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);
        if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(RllmError::ModelError(format!(
                "hidden size {hidden_size} not divisible by {num_attention_heads} heads"
            )));
        }
        let head_dim = hidden_size / num_attention_heads;
        for k in ["attention.key_length", "rope.dimension_count"] {
            if let Some(v) = g.get(&key(k)).and_then(GgufValue::as_u64) {
                if v as usize != head_dim {
                    return Err(RllmError::ModelError(format!(
                        "{arch}.{k} = {v} differs from hidden/heads = {head_dim}; not supported yet"
                    )));
                }
            }
        }

        let vocab_size = match g.get("tokenizer.ggml.tokens").and_then(GgufValue::as_array) {
            Some(tokens) => tokens.len(),
            None => g.get_u64(&key("vocab_size"))? as usize,
        };
        let context_length = g.get_u64(&key("context_length"))? as usize;
        let max_position_embeddings = opts
            .max_context
            .map_or(context_length, |cap| cap.min(context_length));

        Ok(LlamaConfig {
            hidden_size,
            intermediate_size: g.get_u64(&key("feed_forward_length"))? as usize,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers: g.get_u64(&key("block_count"))? as usize,
            vocab_size,
            rms_norm_eps: g.get_f64(&key("attention.layer_norm_rms_epsilon"))? as f32,
            rope_theta: g
                .get(&key("rope.freq_base"))
                .and_then(GgufValue::as_f64)
                .unwrap_or(10000.0) as f32,
            max_position_embeddings,
        })
    }
}

impl<B: Backend> LlamaModel<B> {
    /// Loads a LLaMA-architecture model from a GGUF file.
    pub fn from_gguf(path: impl AsRef<Path>, backend: Arc<B>, opts: &LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        let gguf = MappedGguf::open(path)?;
        let model = Self::from_mapped(&gguf, backend, opts)?;
        info!("Loaded model weights from {}", path.display());
        Ok(model)
    }

    /// Builds the model from an already opened GGUF file.
    pub fn from_mapped(gguf: &MappedGguf, backend: Arc<B>, opts: &LoadOptions) -> Result<Self> {
        let config = LlamaConfig::from_gguf(&gguf.header, opts)?;
        let dev = backend.device();
        if dev != Device::Cpu {
            return Err(RllmError::ModelError(
                "GGUF loading currently targets the CPU backend only".to_string(),
            ));
        }
        let h = config.hidden_size;
        let head_dim = config.head_dim();
        let kv_dim = config.num_key_value_heads * head_dim;
        let inter = config.intermediate_size;

        let weight = |name: &str, shape: &[usize]| load_tensor(gguf, name, shape, true);
        let norm = |name: &str| load_tensor(gguf, name, &[h], false);

        let embed_tokens = weight("token_embd.weight", &[config.vocab_size, h])?;
        let lm_head = if gguf.tensor_info("output.weight").is_some() {
            weight("output.weight", &[config.vocab_size, h])?
        } else {
            // Tied input/output embeddings.
            embed_tokens.clone()
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = |n: &str| format!("blk.{i}.{n}.weight");
            layers.push(LlamaDecoderLayer {
                input_layernorm: norm(&p("attn_norm"))?,
                self_attn: LlamaAttention {
                    q_proj: weight(&p("attn_q"), &[h, h])?,
                    k_proj: weight(&p("attn_k"), &[kv_dim, h])?,
                    v_proj: weight(&p("attn_v"), &[kv_dim, h])?,
                    o_proj: weight(&p("attn_output"), &[h, h])?,
                },
                post_attention_layernorm: norm(&p("ffn_norm"))?,
                mlp: LlamaMlp {
                    gate_proj: weight(&p("ffn_gate"), &[inter, h])?,
                    up_proj: weight(&p("ffn_up"), &[inter, h])?,
                    down_proj: weight(&p("ffn_down"), &[h, inter])?,
                },
            });
        }
        let final_norm = norm("output_norm.weight")?;

        // Optional per-frequency RoPE scaling factors (LLaMA 3.1+).
        let freq_factors = match gguf.tensor_info("rope_freqs.weight") {
            Some(_) => Some(load_tensor(gguf, "rope_freqs.weight", &[head_dim / 2], false)?.to_vec::<f32>()?),
            None => None,
        };
        let (cos_raw, sin_raw) = precompute_rope_freqs_scaled(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            freq_factors.as_deref(),
        );
        let table_shape = Shape::new(vec![config.max_position_embeddings, head_dim / 2]);
        let cos_table = Tensor::from_f32(&cos_raw, table_shape.clone(), dev)?;
        let sin_table = Tensor::from_f32(&sin_raw, table_shape, dev)?;

        info!(
            "Model: {} layers, hidden {}, {} heads / {} KV heads, vocab {}, context {}",
            config.num_hidden_layers,
            h,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.vocab_size,
            config.max_position_embeddings
        );

        Ok(LlamaModel {
            config,
            backend,
            embed_tokens,
            layers,
            norm: final_norm,
            lm_head,
            cos_table,
            sin_table,
        })
    }
}

/// Loads one tensor, checking its shape. With `keep_quantized`, Q8_0 / Q4_0 payloads
/// are kept as-is; every other type is converted to F32.
pub(crate) fn load_tensor(g: &MappedGguf, name: &str, shape: &[usize], keep_quantized: bool) -> Result<Tensor> {
    let info = g
        .tensor_info(name)
        .ok_or_else(|| RllmError::ModelError(format!("Missing tensor '{name}'")))?;
    if info.shape != shape {
        return Err(RllmError::ModelError(format!(
            "Tensor '{name}' has shape {:?}, expected {:?}",
            info.shape, shape
        )));
    }
    let bytes = g.tensor_bytes(info)?;
    let shape = Shape::new(shape.to_vec());

    let quantized = match info.ggml_type {
        GgmlType::Q8_0 => Some(DType::Q8_0),
        GgmlType::Q4_0 => Some(DType::Q4_0),
        _ => None,
    };
    if let (true, Some(dtype)) = (keep_quantized, quantized) {
        let storage = Storage::Cpu(CpuStorage::from_bytes(bytes.to_vec()));
        return Ok(Tensor::new(shape, dtype, Device::Cpu, storage));
    }

    let mut out = vec![0f32; info.numel()];
    match info.ggml_type {
        GgmlType::F32 => {
            for (b, y) in bytes.as_chunks::<4>().0.iter().zip(out.iter_mut()) {
                *y = f32::from_le_bytes(*b);
            }
        }
        GgmlType::F16 => convert_f16(bytes, &mut out),
        GgmlType::BF16 => convert_bf16(bytes, &mut out),
        GgmlType::Q8_0 => dequantize_q8_0(bytes, &mut out),
        GgmlType::Q4_0 => dequantize_q4_0(bytes, &mut out),
        GgmlType::Q4_1 => dequantize_q4_1(bytes, &mut out),
        GgmlType::Q5_0 => dequantize_q5_0(bytes, &mut out),
        GgmlType::Q5_1 => dequantize_q5_1(bytes, &mut out),
        GgmlType::Q4_K => dequantize_q4_k(bytes, &mut out),
        GgmlType::Q5_K => dequantize_q5_k(bytes, &mut out),
        GgmlType::Q6_K => dequantize_q6_k(bytes, &mut out),
        other => {
            return Err(RllmError::QuantizationError(format!(
                "Tensor '{name}' uses unsupported type {other:?}"
            )))
        }
    }
    Tensor::from_f32(&out, shape, Device::Cpu)
}

/// Everything needed to serve a GGUF model: weights, tokenizer and chat template.
pub struct GgufBundle<B: Backend> {
    /// The decoder model.
    pub model: LlamaModel<B>,
    /// The model's BPE tokenizer.
    pub tokenizer: BpeTokenizer,
    /// The model's chat template (ChatML if the file has none).
    pub chat_template: ChatTemplate,
    /// `general.name` from the file, if present.
    pub name: Option<String>,
}

impl<B: Backend> GgufBundle<B> {
    /// Opens a GGUF file once and loads the model, tokenizer and chat template.
    pub fn load(path: impl AsRef<Path>, backend: Arc<B>, opts: &LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        let gguf = MappedGguf::open(path)?;
        let tokenizer = BpeTokenizer::from_gguf(&gguf.header)?;
        let model = LlamaModel::from_mapped(&gguf, backend, opts)?;
        if tokenizer.vocab_size() != model.config.vocab_size {
            return Err(RllmError::ModelError(format!(
                "Tokenizer has {} tokens but the model expects {}",
                tokenizer.vocab_size(),
                model.config.vocab_size
            )));
        }
        let chat_template = ChatTemplate::from_gguf(&gguf.header, &tokenizer)?;
        let name = gguf
            .header
            .get("general.name")
            .and_then(GgufValue::as_str)
            .map(str::to_string);
        info!("Loaded {} from {}", name.as_deref().unwrap_or("model"), path.display());
        Ok(Self {
            model,
            tokenizer,
            chat_template,
            name,
        })
    }
}

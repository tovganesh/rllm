//! End-to-end checks against llama.cpp on a real GGUF model.
//!
//! Expected outputs in `tests/fixtures/` were produced by llama.cpp (`llama-tokenize`
//! and greedy `llama-completion`) for `SmolLM2-135M-Instruct-Q8_0.gguf`.
//!
//! The model file is not committed. Tests look for it at `$RLLM_TEST_MODEL`, or at
//! `<workspace>/models/SmolLM2-135M-Instruct-Q8_0.gguf`, and are skipped (with a
//! message) if it is absent. To fetch it:
//!
//! ```text
//! curl -L -o models/SmolLM2-135M-Instruct-Q8_0.gguf \
//!   https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf
//! ```

use rllm_backend_cpu::CpuBackend;
use rllm_core::{backend::BlockTable, device::Device, dtype::DType, shape::Shape, tensor::Tensor};
use rllm_models::{ChatMessage, ForwardBatch, GgufBundle, LlamaModel, LoadOptions};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

fn model_path() -> Option<PathBuf> {
    let p = std::env::var_os("RLLM_TEST_MODEL").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/SmolLM2-135M-Instruct-Q8_0.gguf")
    });
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping: model not found at {} (set RLLM_TEST_MODEL)", p.display());
        None
    }
}

fn bundle() -> Option<&'static GgufBundle<CpuBackend>> {
    static B: OnceLock<Option<GgufBundle<CpuBackend>>> = OnceLock::new();
    B.get_or_init(|| {
        let path = model_path()?;
        let opts = LoadOptions { max_context: Some(1024) };
        Some(GgufBundle::load(path, Arc::new(CpuBackend), &opts).expect("model loads"))
    })
    .as_ref()
}

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Greedy generation driven directly through `LlamaModel::forward`.
fn greedy(model: &LlamaModel<CpuBackend>, prompt: &[u32], max_tokens: usize, stops: &[u32]) -> Vec<u32> {
    let c = &model.config;
    let block = 16;
    let blocks = (prompt.len() + max_tokens).div_ceil(block);
    let shape = Shape::new(vec![blocks, c.num_key_value_heads, block, c.head_dim()]);
    let mut k: Vec<Tensor> =
        (0..c.num_hidden_layers).map(|_| Tensor::zeros(shape.clone(), DType::F32, Device::Cpu).unwrap()).collect();
    let mut v = k.clone();
    let table = BlockTable { block_size: block, block_ids: (0..blocks).collect() };

    let positions: Vec<usize> = (0..prompt.len()).collect();
    let mut logits = model
        .forward(
            &ForwardBatch {
                tokens: prompt,
                positions: &positions,
                block_tables: &vec![table.clone(); prompt.len()],
                logits_rows: &[prompt.len() - 1],
            },
            &mut k,
            &mut v,
        )
        .unwrap();
    let mut out = Vec::new();
    loop {
        let row: Vec<f32> = logits.to_vec().unwrap();
        let next = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        if stops.contains(&next) {
            break;
        }
        out.push(next);
        if out.len() == max_tokens {
            break;
        }
        let pos = prompt.len() + out.len() - 1;
        logits = model
            .forward_decode(&[next], &mut k, &mut v, std::slice::from_ref(&table), &[pos + 1])
            .unwrap();
    }
    out
}

#[test]
fn tokenizer_matches_llama_cpp() {
    let Some(b) = bundle() else { return };
    let fx = fixture("smollm2_tokenizer.json");
    let mut failures = Vec::new();
    for case in fx["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got = b.tokenizer.encode(text, true, true);
        if got != expected {
            failures.push(format!("{text:?}\n  expected {expected:?}\n  got      {got:?}"));
        }
        // Lossless round trip when special tokens are kept.
        assert_eq!(b.tokenizer.decode(&got, false), text, "round trip of {text:?}");
    }
    assert!(failures.is_empty(), "{} tokenizer mismatches:\n{}", failures.len(), failures.join("\n"));
}

fn check_greedy(b: &GgufBundle<CpuBackend>, fixture_name: &str) {
    let fx = fixture(fixture_name);
    for case in fx["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        let expected = case["completion"].as_str().unwrap();
        let max_tokens = case["max_tokens"].as_u64().unwrap() as usize;
        let ids = b.tokenizer.encode(prompt, true, true);
        let out = greedy(&b.model, &ids, max_tokens, &b.tokenizer.stop_token_ids);
        let text = b.tokenizer.decode(&out, true);
        assert_eq!(text, expected, "[{fixture_name}] greedy completion of {prompt:?}");
    }
}

#[test]
fn greedy_generation_matches_llama_cpp() {
    let Some(b) = bundle() else { return };
    check_greedy(b, "smollm2_greedy.json");
}

/// Other weight formats, produced locally from the Q8_0 file with
/// `llama-quantize --allow-requantize <q8_0.gguf> models/SmolLM2-135M-Instruct-<TYPE>.requant.gguf <TYPE>`.
/// Each variant is skipped if its file is absent.
#[test]
fn greedy_generation_matches_llama_cpp_across_weight_formats() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    for variant in ["Q4_0", "Q4_K_M", "Q5_K_M", "F16"] {
        let path = dir.join(format!("SmolLM2-135M-Instruct-{variant}.requant.gguf"));
        if !path.exists() {
            eprintln!("skipping {variant}: {} not found", path.display());
            continue;
        }
        let b = GgufBundle::load(&path, Arc::new(CpuBackend), &LoadOptions { max_context: Some(1024) })
            .unwrap_or_else(|e| panic!("{variant} failed to load: {e}"));
        check_greedy(&b, &format!("smollm2_greedy_{variant}.json"));
    }
}

#[test]
fn chat_template_renders_model_format() {
    let Some(b) = bundle() else { return };
    let rendered = b
        .chat_template
        .render(&[ChatMessage::new("user", "What is the boiling point of water at sea level?")], true)
        .unwrap();
    // The SmolLM2 template injects its default system prompt when none is given; this
    // matches the prompt used for the chat case in the greedy fixture.
    let fx = fixture("smollm2_greedy.json");
    let chat_case = fx["cases"].as_array().unwrap().iter().find(|c| c["prompt"].as_str().unwrap().starts_with("<|im_start|>")).unwrap();
    assert_eq!(rendered, chat_case["prompt"].as_str().unwrap());
    assert!(b.tokenizer.stop_token_ids.contains(&b.tokenizer.token_id("<|im_end|>").unwrap()));
}

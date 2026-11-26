//! Metal executor checked against llama.cpp's Metal backend (`-ngl 99`) on SmolLM2-135M
//! and, optionally, Llama-3.2-1B-Instruct. Skipped when the model files (see
//! `crates/rllm-models/tests/real_model.rs`) or a Metal device are absent. Llama 3.2:
//!
//! ```text
//! curl -L -o models/Llama-3.2-1B-Instruct-Q8_0.gguf \
//!   https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
//! ```
#![cfg(target_os = "macos")]

use rllm_backend_cpu::CpuBackend;
use rllm_backend_metal::MetalLlama;
use rllm_core::backend::BlockTable;
use rllm_models::{ForwardBatch, GgufBundle, LanguageModel, LoadOptions};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> Value {
    let p = root().join("crates/rllm-models/tests/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn model_file(name: &str) -> Option<PathBuf> {
    let file = if name == "Q8_0" {
        "SmolLM2-135M-Instruct-Q8_0.gguf".to_string()
    } else {
        format!("SmolLM2-135M-Instruct-{name}.requant.gguf")
    };
    let p = root().join("models").join(file);
    if p.exists() && MetalLlama::is_available() {
        Some(p)
    } else {
        eprintln!("skipping {name}: model file or Metal device missing");
        None
    }
}

#[test]
fn perplexity_matches_llama_cpp_metal() {
    let fx = fixture("smollm2_perplexity_metal.json");
    let ctx = fx["ctx"].as_u64().unwrap() as usize;
    let text = std::fs::read_to_string(root().join(fx["text"].as_str().unwrap())).unwrap();
    for (name, expected) in fx["ppl"].as_object().unwrap() {
        let Some(path) = model_file(name) else { continue };
        let b = GgufBundle::load(
            &path,
            Arc::new(CpuBackend),
            &LoadOptions {
                max_context: Some(ctx),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let gpu = MetalLlama::from_cpu_model(&b.model).unwrap();
        let tokens = b.tokenizer.encode(&text, true, false);
        let bos = if b.tokenizer.add_bos {
            b.tokenizer.bos_token_id
        } else {
            None
        };
        let got = rllm_models::perplexity(&gpu, &tokens, ctx, bos).unwrap().ppl;
        let expected = expected.as_f64().unwrap();
        let rel = (got - expected) / expected;
        eprintln!(
            "[{name}] Metal PPL rllm {got:.4} vs llama.cpp {expected:.4} ({:+.2}%)",
            rel * 100.0
        );
        assert!(rel.abs() < 0.01, "[{name}] {got:.4} vs {expected:.4}");
    }
}

#[test]
fn greedy_generation_matches_llama_cpp_metal() {
    if let Some(path) = model_file("Q8_0") {
        check_greedy(&path, "smollm2_greedy_metal.json");
    }
    let llama = root().join("models/Llama-3.2-1B-Instruct-Q8_0.gguf");
    if llama.exists() && MetalLlama::is_available() {
        check_greedy(&llama, "llama32_greedy_metal.json");
    } else {
        eprintln!("skipping Llama 3.2: {} not found", llama.display());
    }
}

fn check_greedy(path: &std::path::Path, fixture_name: &str) {
    let b = GgufBundle::load(
        path,
        Arc::new(CpuBackend),
        &LoadOptions {
            max_context: Some(1024),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let gpu = MetalLlama::from_cpu_model(&b.model).unwrap();
    let block = 16;
    let blocks = 1024 / block;
    let table = BlockTable {
        block_size: block,
        block_ids: (0..blocks).collect(),
    };
    for case in fixture(fixture_name)["cases"].as_array().unwrap() {
        gpu.init_kv_cache(blocks, block).unwrap();
        let prompt = b.tokenizer.encode(case["prompt"].as_str().unwrap(), true, true);
        let max_tokens = case["max_tokens"].as_u64().unwrap() as usize;
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let mut logits = gpu
            .forward_cached(&ForwardBatch {
                tokens: &prompt,
                positions: &positions,
                block_tables: &vec![table.clone(); prompt.len()],
                logits_rows: &[prompt.len() - 1],
            })
            .unwrap();
        let mut out = Vec::new();
        loop {
            let next = logits.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
            if b.tokenizer.stop_token_ids.contains(&next) {
                break;
            }
            out.push(next);
            if out.len() == max_tokens {
                break;
            }
            let pos = prompt.len() + out.len() - 1;
            logits = gpu
                .forward_cached(&ForwardBatch {
                    tokens: &[next],
                    positions: &[pos],
                    block_tables: std::slice::from_ref(&table),
                    logits_rows: &[0],
                })
                .unwrap();
        }
        assert_eq!(
            b.tokenizer.decode(&out, true),
            case["completion"].as_str().unwrap(),
            "[{fixture_name}] prompt {:?}",
            case["prompt"]
        );
    }
}

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
    let p = std::env::var_os("RLLM_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
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
        let opts = LoadOptions {
            max_context: Some(1024),
            ..LoadOptions::default()
        };
        Some(GgufBundle::load(path, Arc::new(CpuBackend), &opts).expect("model loads"))
    })
    .as_ref()
}

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Greedy generation driven directly through `LlamaModel::forward`.
fn greedy(model: &LlamaModel<CpuBackend>, prompt: &[u32], max_tokens: usize, stops: &[u32]) -> Vec<u32> {
    let c = &model.config;
    let block = 16;
    let blocks = (prompt.len() + max_tokens).div_ceil(block);
    let shape = Shape::new(vec![blocks, c.num_key_value_heads, block, c.head_dim()]);
    let mut k: Vec<Tensor> = (0..c.num_hidden_layers)
        .map(|_| Tensor::zeros(shape.clone(), DType::F32, Device::Cpu).unwrap())
        .collect();
    let mut v = k.clone();
    let table = BlockTable {
        block_size: block,
        block_ids: (0..blocks).collect(),
    };

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
        let expected: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = b.tokenizer.encode(text, true, true);
        if got != expected {
            failures.push(format!("{text:?}\n  expected {expected:?}\n  got      {got:?}"));
        }
        // Lossless round trip when special tokens are kept.
        assert_eq!(b.tokenizer.decode(&got, false), text, "round trip of {text:?}");
    }
    assert!(
        failures.is_empty(),
        "{} tokenizer mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
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

/// Teacher-forced check: run llama.cpp's completion through rllm in one prefill and
/// require that, at every position, llama.cpp's chosen token is rllm's argmax or within
/// `tol` (logit units) of it. Returns the largest observed gap.
///
/// Exact token equality is too strict across engines: llama.cpp generated these tokens
/// one at a time with Q8-quantized activations, while a batched prefill (here, and in
/// llama.cpp) uses F32 activations. On near-ties the two disagree by ~0.15 logits; even
/// llama.cpp's own Metal and CPU backends pick different tokens at such a position.
/// The gap grows with weight-quantization noise, so the tolerance is 0.25 logits for
/// 8/16-bit weights and 0.5 for 4/5-bit formats. Real kernel bugs move logits by whole
/// units and still fail; `perplexity_matches_llama_cpp_across_weight_formats` is the
/// primary quantitative check.
fn check_teacher_forced(b: &GgufBundle<CpuBackend>, fixture_name: &str, tol: f32) -> f32 {
    let fx = fixture(fixture_name);
    let c = &b.model.config;
    let mut worst = 0f32;
    for case in fx["cases"].as_array().unwrap() {
        let prompt = b.tokenizer.encode(case["prompt"].as_str().unwrap(), true, true);
        let completion = b.tokenizer.encode(case["completion"].as_str().unwrap(), false, false);
        let mut all = prompt.clone();
        all.extend_from_slice(&completion);
        let n = all.len();
        let block = 16;
        let blocks = n.div_ceil(block);
        let shape = Shape::new(vec![blocks, c.num_key_value_heads, block, c.head_dim()]);
        let mut k: Vec<Tensor> = (0..c.num_hidden_layers)
            .map(|_| Tensor::zeros(shape.clone(), DType::F32, Device::Cpu).unwrap())
            .collect();
        let mut v = k.clone();
        let table = BlockTable {
            block_size: block,
            block_ids: (0..blocks).collect(),
        };
        let positions: Vec<usize> = (0..n).collect();
        let rows: Vec<usize> = (prompt.len() - 1..n - 1).collect();
        let logits = b
            .model
            .forward(
                &ForwardBatch {
                    tokens: &all,
                    positions: &positions,
                    block_tables: &vec![table; n],
                    logits_rows: &rows,
                },
                &mut k,
                &mut v,
            )
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for (i, row) in logits.chunks(c.vocab_size).enumerate() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let gap = max - row[completion[i] as usize];
            worst = worst.max(gap);
            assert!(
                gap <= tol,
                "[{fixture_name}] position {i} of {:?}: llama.cpp token is {gap:.4} below rllm's argmax",
                case["prompt"]
            );
        }
    }
    worst
}

#[test]
fn llama_cpp_greedy_tokens_are_rllm_argmax_across_weight_formats() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut variants = vec![(
        "Q8_0",
        dir.join("SmolLM2-135M-Instruct-Q8_0.gguf"),
        "smollm2_greedy.json".to_string(),
    )];
    for q in ["Q4_0", "Q4_K_M", "Q5_K_M", "F16"] {
        variants.push((
            q,
            dir.join(format!("SmolLM2-135M-Instruct-{q}.requant.gguf")),
            format!("smollm2_greedy_{q}.json"),
        ));
    }
    for (name, path, fx) in variants {
        if !path.exists() {
            eprintln!("skipping {name}: {} not found", path.display());
            continue;
        }
        let b = GgufBundle::load(
            &path,
            Arc::new(CpuBackend),
            &LoadOptions {
                max_context: Some(1024),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let tol = if matches!(name, "Q8_0" | "F16") { 0.25 } else { 0.5 };
        let worst = check_teacher_forced(&b, &fx, tol);
        eprintln!("[{name}] largest gap between llama.cpp's token and rllm's argmax: {worst:.4} logits");
    }
}

#[test]
fn chat_template_renders_model_format() {
    let Some(b) = bundle() else { return };
    let rendered = b
        .chat_template
        .render(
            &[ChatMessage::new(
                "user",
                "What is the boiling point of water at sea level?",
            )],
            true,
        )
        .unwrap();
    // The SmolLM2 template injects its default system prompt when none is given; this
    // matches the prompt used for the chat case in the greedy fixture.
    let fx = fixture("smollm2_greedy.json");
    let chat_case = fx["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["prompt"].as_str().unwrap().starts_with("<|im_start|>"))
        .unwrap();
    assert_eq!(rendered, chat_case["prompt"].as_str().unwrap());
    assert!(b
        .tokenizer
        .stop_token_ids
        .contains(&b.tokenizer.token_id("<|im_end|>").unwrap()));
}

/// Perplexity on a fixed text must be within 1% of llama.cpp (CPU, same chunking) for
/// every weight format. This is the standard quantitative check for inference kernels.
#[test]
fn perplexity_matches_llama_cpp_across_weight_formats() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fx = fixture("smollm2_perplexity.json");
    let ctx = fx["ctx"].as_u64().unwrap() as usize;
    let text = std::fs::read_to_string(root.join(fx["text"].as_str().unwrap())).unwrap();
    for (name, expected) in fx["ppl"].as_object().unwrap() {
        let file = if name == "Q8_0" {
            "SmolLM2-135M-Instruct-Q8_0.gguf".to_string()
        } else {
            format!("SmolLM2-135M-Instruct-{name}.requant.gguf")
        };
        let path = root.join("models").join(file);
        if !path.exists() {
            eprintln!("skipping {name}: {} not found", path.display());
            continue;
        }
        let b = GgufBundle::load(
            &path,
            Arc::new(CpuBackend),
            &LoadOptions {
                max_context: Some(ctx),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let tokens = b.tokenizer.encode(&text, true, false);
        let bos = if b.tokenizer.add_bos {
            b.tokenizer.bos_token_id
        } else {
            None
        };
        let got = rllm_models::perplexity(&b.model, &tokens, ctx, bos).unwrap().ppl;
        let expected = expected.as_f64().unwrap();
        let rel = (got - expected) / expected;
        eprintln!(
            "[{name}] PPL rllm {got:.4} vs llama.cpp {expected:.4} ({:+.2}%)",
            rel * 100.0
        );
        assert!(
            rel.abs() < 0.01,
            "[{name}] perplexity {got:.4} differs from llama.cpp {expected:.4} by {:.2}%",
            rel * 100.0
        );
    }
}

/// Llama 3.2 (llama-bpe pre-tokenizer, 128k vocab, BOS added by default) against
/// llama-tokenize. Skipped when `models/Llama-3.2-1B-Instruct-Q8_0.gguf` is absent.
#[test]
fn llama32_tokenizer_matches_llama_cpp() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Llama-3.2-1B-Instruct-Q8_0.gguf");
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let g = rllm_models::MappedGguf::open(&path).unwrap();
    let tok = rllm_models::BpeTokenizer::from_gguf(&g.header).unwrap();
    assert!(tok.add_bos, "Llama 3 tokenizers add BOS by default");
    assert!(tok.stop_token_ids.contains(&tok.token_id("<|eot_id|>").unwrap()));
    assert!(tok.stop_token_ids.contains(&tok.token_id("<|end_of_text|>").unwrap()));
    let mut failures = Vec::new();
    for case in fixture("llama32_tokenizer.json")["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok.encode(text, true, true);
        if got != expected {
            failures.push(format!("{text:?}\n  expected {expected:?}\n  got      {got:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

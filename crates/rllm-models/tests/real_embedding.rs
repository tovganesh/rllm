//! Embedding models checked against llama.cpp.
//!
//! Fixtures come from `scripts/gen_tokenizer_fixture.py` and
//! `scripts/gen_embedding_fixture.py`. Model files are looked up in `<workspace>/models/`
//! and the tests are skipped when absent:
//!
//! ```text
//! curl -L -o models/bge-small-en-v1.5-q8_0.gguf \
//!   https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf
//! curl -L -o models/all-MiniLM-L6-v2-Q8_0.gguf \
//!   https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf
//! ```

use rllm_backend_cpu::CpuBackend;
use rllm_models::{EmbeddingTokenizer, TextEmbedder};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

fn model(file: &str) -> Option<TextEmbedder<CpuBackend>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models").join(file);
    if !p.exists() {
        eprintln!("skipping: {} not found", p.display());
        return None;
    }
    Some(TextEmbedder::from_gguf(&p, Arc::new(CpuBackend)).expect("embedding model loads"))
}

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
fn wordpiece_tokenizer_matches_llama_cpp() {
    let Some(e) = model("bge-small-en-v1.5-q8_0.gguf") else { return };
    let EmbeddingTokenizer::WordPiece(tok) = &e.tokenizer else { panic!("expected WordPiece") };
    let mut failures = Vec::new();
    for case in fixture("bge_small_tokenizer.json")["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text, true, true);
        if got != expected {
            failures.push(format!("{text:?}\n  expected {expected:?}\n  got      {got:?}"));
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

fn check_embeddings(file: &str, fixture_name: &str) {
    let Some(e) = model(file) else { return };
    let mut min_cos = 1.0f32;
    for case in fixture(fixture_name)["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected: Vec<f32> = case["embedding"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let got = e.embed_text(text).unwrap();
        assert_eq!(got.len(), expected.len());
        let cos = cosine(&got, &expected);
        assert!(cos > 0.999, "[{file}] cosine {cos} for {text:?}");
        min_cos = min_cos.min(cos);
    }
    eprintln!("[{file}] minimum cosine similarity to llama.cpp: {min_cos:.6}");
}

#[test]
fn bge_cls_pooled_embeddings_match_llama_cpp() {
    check_embeddings("bge-small-en-v1.5-q8_0.gguf", "bge_small_embeddings.json");
}

#[test]
fn minilm_mean_pooled_embeddings_match_llama_cpp() {
    check_embeddings("all-MiniLM-L6-v2-Q8_0.gguf", "minilm_embeddings.json");
}

#[test]
fn semantically_related_texts_score_higher() {
    let Some(e) = model("bge-small-en-v1.5-q8_0.gguf") else { return };
    let q = e.embed_text("How do plants make their food?").unwrap();
    let related = e.embed_text("Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.").unwrap();
    let unrelated = e.embed_text("The capital of France is Paris.").unwrap();
    assert!(cosine(&q, &related) > cosine(&q, &unrelated) + 0.1);
}

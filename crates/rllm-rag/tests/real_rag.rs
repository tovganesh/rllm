//! RAG with real models: bge-small-en-v1.5 embeddings and a SmolLM2 answer generator.
//! Skipped when the model files are absent (see `crates/rllm-models/tests/*.rs` for the
//! download commands).

use rllm_backend_cpu::CpuBackend;
use rllm_models::{GgufBundle, LoadOptions, TextEmbedder};
use rllm_rag::{LlmGenerator, RagEngine};
use rllm_scheduler::{ContinuousBatchScheduler, EngineConfig, EngineHandle, SamplingParams, SchedulerConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

const DOCS: [(&str, &str); 5] = [
    ("water", "Water boils at 100 degrees Celsius (212 Fahrenheit) at sea level; at higher altitudes the boiling point is lower because air pressure is lower."),
    ("plants", "Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide to produce glucose and release oxygen."),
    ("paris", "Paris is the capital and most populous city of France, located on the Seine river."),
    ("rust", "The Rust programming language guarantees memory safety without a garbage collector through its ownership and borrowing system."),
    ("reef", "The Great Barrier Reef, off the coast of Queensland, Australia, is the world's largest coral reef system."),
];

fn engine(with_llm: bool) -> Option<RagEngine<CpuBackend>> {
    let emb_path = models_dir().join("bge-small-en-v1.5-q8_0.gguf");
    let llm_path = models_dir().join("SmolLM2-135M-Instruct-Q8_0.gguf");
    if !emb_path.exists() || (with_llm && !llm_path.exists()) {
        eprintln!("skipping: model files not found in {}", models_dir().display());
        return None;
    }
    let backend = Arc::new(CpuBackend);
    let embedder = Arc::new(TextEmbedder::from_gguf(&emb_path, backend.clone()).unwrap());
    let generator = with_llm.then(|| {
        let b = GgufBundle::load(&llm_path, backend, &LoadOptions { max_context: Some(2048), ..LoadOptions::default() }).unwrap();
        let scheduler = ContinuousBatchScheduler::new(
            SchedulerConfig { max_batch_size: 4, block_size: 16, total_blocks: 256, ..SchedulerConfig::default() },
            Arc::new(b.model),
        )
        .unwrap();
        LlmGenerator {
            engine: EngineHandle::start(Arc::new(scheduler), EngineConfig::default()),
            tokenizer: Arc::new(b.tokenizer),
            chat_template: Arc::new(b.chat_template),
        }
    });
    let e = RagEngine::new(embedder, generator);
    for (id, text) in DOCS {
        e.ingest_document(id, text, HashMap::new()).unwrap();
    }
    Some(e)
}

#[test]
fn paraphrased_queries_retrieve_the_right_passage() {
    let Some(e) = engine(false) else { return };
    for (query, expected) in [
        ("How do plants turn light into energy?", "plants"),
        ("Which city is France's capital?", "paris"),
        ("How does Rust avoid memory bugs without GC?", "rust"),
        ("Where is the largest reef in the world?", "reef"),
        ("At what temperature does water start boiling?", "water"),
    ] {
        let r = e.retrieve(query, 2, None).unwrap();
        assert!(r.is_relevant, "{query:?} should be relevant");
        assert_eq!(r.citations[0].doc_id, expected, "top passage for {query:?}");
        assert!(r.citations[0].dense_similarity.unwrap() >= e.min_dense_similarity);
    }
}

#[test]
fn unrelated_queries_are_flagged() {
    let Some(e) = engine(false) else { return };
    for query in ["Who won the 1998 football World Cup?", "What is a good recipe for chocolate cake?"] {
        let r = e.retrieve(query, 2, None).unwrap();
        assert!(!r.is_relevant, "{query:?} should be flagged; got {:?}", r.citations.first().map(|c| (&c.doc_id, c.dense_similarity, c.relevance)));
    }
}

#[tokio::test]
async fn llm_generates_a_grounded_answer() {
    let Some(e) = engine(true) else { return };
    let sampling = SamplingParams { temperature: 0.0, max_tokens: 80, ..SamplingParams::default() };
    let r = e.query("At what temperature does water boil at sea level?", 2, sampling).await.unwrap();
    eprintln!("answer: {:?}\nwarning: {:?}\ncited: {:?}", r.answer, r.warning, r.citations.iter().map(|c| (&c.doc_id, c.cited_in_answer)).collect::<Vec<_>>());
    assert!(r.is_relevant);
    assert_eq!(r.answer_source, "llm");
    assert!(r.generated_tokens > 0);
    assert_eq!(r.citations[0].doc_id, "water");
    assert!(r.answer.contains("100"), "answer should use the source: {:?}", r.answer);
}

#[test]
fn calibrated_thresholds_hold_for_both_embedding_models() {
    let backend = Arc::new(CpuBackend);
    let related = ["How do leaves make sugar?", "What powers plant growth?", "How tall is the tower in the French capital?", "Is Rust garbage collected?"];
    let unrelated = ["Who won the 1998 World Cup?", "Best chocolate cake recipe", "How do I change a car tyre?", "What is the stock price of Apple?"];
    for file in ["bge-small-en-v1.5-q8_0.gguf", "all-MiniLM-L6-v2-Q8_0.gguf"] {
        let path = models_dir().join(file);
        if !path.exists() {
            eprintln!("skipping {file}");
            continue;
        }
        let e = RagEngine::new(Arc::new(TextEmbedder::from_gguf(&path, backend.clone()).unwrap()), None);
        let docs = [
            ("photosynthesis", "Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide to produce glucose and release oxygen."),
            ("paris", "Paris is the capital and most populous city of France. It is known for the Eiffel Tower, completed in 1889."),
            ("rust", "The Rust programming language guarantees memory safety without a garbage collector through its ownership and borrowing system."),
        ];
        for (id, text) in docs {
            e.ingest_document(id, text, HashMap::new()).unwrap();
        }
        for q in related {
            assert!(e.retrieve(q, 1, None).unwrap().is_relevant, "[{file}] {q:?} should be relevant");
        }
        for q in unrelated {
            assert!(!e.retrieve(q, 1, None).unwrap().is_relevant, "[{file}] {q:?} should be flagged");
        }
    }
}

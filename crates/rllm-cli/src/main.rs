//! Command-line interface and binary entrypoint for the `rllm` inference and RAG engines.
//!
//! Subcommands:
//! - `serve`: Launches the OpenAI-compatible HTTP service in `all`, `inference`, or `rag` decoupled mode.
//! - `bench`: Runs the inference performance benchmark suite against target architectures (e.g. `edge-4gb`).
//! - `rag-demo`: Runs an interactive retrieval-augmented generation pipeline with citations and relevance verification.

mod benchmark;

use benchmark::{run_comprehensive_benchmark, BenchmarkOptions};
use clap::{Parser, Subcommand};
use rllm_backend_cpu::CpuBackend;
use rllm_models::{BertConfig, BertEmbeddingModel, LlamaConfig, LlamaModel};
use rllm_rag::RagEngine;
use rllm_scheduler::{ContinuousBatchScheduler, SamplingParams, SchedulerConfig};
use rllm_server::{run_server, AppState};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "rllm")]
#[command(about = "Enterprise-Grade LLM Inference & High-Performance RAG Engine in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the OpenAI-compatible HTTP & RAG Server
    Serve {
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: SocketAddr,

        /// Service mode to run: 'all', 'inference', or 'rag'
        #[arg(long, default_value = "all")]
        service: String,

        /// Remote inference URL (e.g., http://inference:8000) for decoupled RAG container
        #[arg(long, env = "INFERENCE_URL")]
        inference_url: Option<String>,

        #[arg(long, default_value = "edge-4gb")]
        model: String,

        #[arg(long, default_value_t = 32)]
        max_batch_size: usize,
    },

    /// Run high-throughput inference performance benchmark
    Bench {
        /// Target model architecture preset: 'edge-4gb' (~400M params, fits in 4GB), 'llama-1b', or 'tiny'
        #[arg(long, default_value = "edge-4gb")]
        preset: String,

        #[arg(long, default_value_t = 4)]
        batch_size: usize,

        #[arg(long, default_value_t = 64)]
        prompt_tokens: usize,

        #[arg(long, default_value_t = 16)]
        generate_tokens: usize,

        #[arg(long, default_value_t = 1)]
        warmup: usize,
    },

    /// Run end-to-end RAG interactive demonstration with citations and relevance verification
    RagDemo {
        #[arg(long, default_value = "What makes rllm enterprise grade and edge compatible?")]
        query: String,

        /// Number of top citations to retrieve
        #[arg(long, default_value_t = 4)]
        top_k: usize,

        /// Minimum relevance threshold (0.0 to 1.0) for qualifying citations
        #[arg(long)]
        min_relevance: Option<f32>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            addr,
            service,
            inference_url,
            model,
            max_batch_size,
        } => {
            use rllm_server::ServiceMode;
            let mode = match service.to_lowercase().as_str() {
                "inference" => ServiceMode::Inference,
                "rag" => ServiceMode::Rag,
                _ => ServiceMode::All,
            };

            println!("🚀 Initializing rllm server in [{:?}] mode", mode);
            let backend = Arc::new(CpuBackend);

            // 1. Initialize inference scheduler if mode is All or Inference
            let scheduler = if mode == ServiceMode::All || mode == ServiceMode::Inference {
                let llama_cfg = match model.as_str() {
                    "edge-4gb" => LlamaConfig::edge_4gb(),
                    "1b" => LlamaConfig::llama_3_2_1b(),
                    _ => LlamaConfig::tiny_test_config(),
                };
                let llama_model = Arc::new(LlamaModel::new_random(llama_cfg, backend.clone())?);
                let sched_cfg = SchedulerConfig {
                    max_batch_size,
                    block_size: 16,
                    total_blocks: 1024,
                };
                Some(Arc::new(ContinuousBatchScheduler::new(sched_cfg, llama_model)?))
            } else {
                None
            };

            // 2. Initialize RAG engine if mode is All or Rag
            let rag_engine = if mode == ServiceMode::All || mode == ServiceMode::Rag {
                let bert_cfg = BertConfig::tiny_test_config();
                let bert_model =
                    Arc::new(BertEmbeddingModel::new_random(bert_cfg.clone(), backend.clone())?);

                // Use the local scheduler if present, or create a minimal one for RAG prompt driving
                let dummy_scheduler = if let Some(ref s) = scheduler {
                    s.clone()
                } else {
                    let dummy_cfg = LlamaConfig::tiny_test_config();
                    let dummy_model = Arc::new(LlamaModel::new_random(dummy_cfg, backend.clone())?);
                    let sched_cfg = SchedulerConfig {
                        max_batch_size: 2,
                        block_size: 16,
                        total_blocks: 64,
                    };
                    Arc::new(ContinuousBatchScheduler::new(sched_cfg, dummy_model)?)
                };

                Some(Arc::new(RagEngine::new(
                    bert_model,
                    dummy_scheduler,
                    bert_cfg.hidden_size,
                )))
            } else {
                None
            };

            let state = Arc::new(AppState {
                scheduler,
                rag_engine,
                inference_url: inference_url.clone(),
                model_name: model,
                mode,
            });

            println!("⚡ Serving API at http://{}", addr);
            println!("   - Health: http://{}/health", addr);
            if mode == ServiceMode::All || mode == ServiceMode::Inference {
                println!("   - Chat:   http://{}/v1/chat/completions", addr);
                println!("   - Embed:  http://{}/v1/embeddings", addr);
            }
            if mode == ServiceMode::All || mode == ServiceMode::Rag {
                println!("   - RAG:    http://{}/v1/rag/query", addr);
                println!("   - Index:  http://{}/v1/rag/index", addr);
                if let Some(ref url) = inference_url {
                    println!("   - Remote Inference Target: {}", url);
                }
            }

            run_server(addr, state).await?;
        }

        Commands::Bench {
            preset,
            batch_size,
            prompt_tokens,
            generate_tokens,
            warmup,
        } => {
            let opts = BenchmarkOptions {
                preset,
                batch_size,
                prompt_tokens,
                generate_tokens,
                warmup,
            };
            run_comprehensive_benchmark(opts)?;
        }

        Commands::RagDemo { query, top_k, min_relevance } => {
            println!("📚 Initializing rllm High-Performance RAG Demonstration");
            let backend = Arc::new(CpuBackend);

            let bert_cfg = BertConfig::tiny_test_config();
            let bert_model =
                Arc::new(BertEmbeddingModel::new_random(bert_cfg.clone(), backend.clone())?);

            let llama_cfg = LlamaConfig::tiny_test_config();
            let llama_model = Arc::new(LlamaModel::new_random(llama_cfg, backend)?);

            let sched_cfg = SchedulerConfig {
                max_batch_size: 4,
                block_size: 16,
                total_blocks: 512,
            };
            let scheduler = Arc::new(ContinuousBatchScheduler::new(sched_cfg, llama_model)?);
            let rag_engine = RagEngine::new(bert_model, scheduler, bert_cfg.hidden_size);

            println!("📥 Ingesting enterprise knowledge base...");
            let docs = vec![
                (
                    "doc_arch",
                    "rllm is an enterprise grade inference engine built in Rust from ground up. It implements continuous batching, PagedAttention, and a Radix tree prefix cache to eliminate memory fragmentation and minimize TTFT.",
                ),
                (
                    "doc_edge",
                    "For smartphone and edge deployments on iOS and Android, rllm supports Q4_0, Q4_K_M, and Q8_0 block quantization. This allows 7B/8B parameter models to run within 2-4 GB of unified memory.",
                ),
                (
                    "doc_cluster",
                    "On Apple Silicon and GPU clusters, rllm leverages Metal Shading Language (MSL) and CUDA kernels with unified memory zero-copy tensor passing and FlashAttention.",
                ),
                (
                    "doc_rag",
                    "The rllm high-performance RAG engine features an in-memory SIMD HNSW vector store, Okapi BM25 lexical search, reciprocal rank fusion (RRF), cross-encoder reranking, and prefix-aware KV cache reuse.",
                ),
            ];

            for (id, text) in docs {
                let mut meta = HashMap::new();
                meta.insert("source".to_string(), "knowledge_base".to_string());
                rag_engine.ingest_document(id, text, meta)?;
            }

            println!("\n🔍 Executing RAG Query: \"{}\"", query);
            let start = Instant::now();
            let mut sampling = SamplingParams::default();
            sampling.max_tokens = 32;

            let resp = rag_engine
                .query_with_threshold(&query, top_k, sampling, min_relevance)
                .await?;
            let elapsed = start.elapsed();

            println!("⏱️  Query Latency: {:.2?}", elapsed);
            println!("📊 Confidence Score: {:.2}", resp.confidence_score);

            if resp.is_relevant {
                println!("✅ Status: RELEVANT (Grounded)");
                println!("✨ Synthesized Answer:\n{}\n", resp.answer);
                println!("⚡ Autoregressive Inference: {} tokens generated & decoded via continuous batch scheduler", resp.generated_tokens);
                println!("📑 Grounded Citations ({} retrieved):", resp.citations.len());
                for (i, cit) in resp.citations.iter().enumerate() {
                    println!("\n   [{}] Document: {} (Relevance: {:.2})", i + 1, cit.doc_id, cit.relevance);
                    if !cit.matched_keywords.is_empty() {
                        println!("       Matched Keywords: [{}]", cit.matched_keywords.join(", "));
                    }
                    println!("       Exact Grounded Quote: \"{}\"", cit.quote);
                    println!("       Full Passage Context: \"{}\"", cit.passage);
                }
            } else {
                println!("⚠️  Status: UNRELATED / LOW RELEVANCE (Flagged)");
                if let Some(ref w) = resp.warning {
                    println!("⚠️  Warning: {}", w);
                }
                println!("✨ Refusal Answer: {}", resp.answer);
                println!("💡 Hallucination avoided: Generation short-circuited to save compute and maintain truthfulness.");
            }
        }
    }

    Ok(())
}

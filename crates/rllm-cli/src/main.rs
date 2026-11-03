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
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: SocketAddr,

        #[arg(long, default_value = "llama-3-8b")]
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

    /// Run end-to-end RAG interactive demonstration with citations
    RagDemo {
        #[arg(long, default_value = "What makes rllm enterprise grade and edge compatible?")]
        query: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            addr,
            model,
            max_batch_size,
        } => {
            println!("🚀 Initializing rllm engine for model: {}", model);
            let backend = Arc::new(CpuBackend);

            let llama_cfg = LlamaConfig::tiny_test_config();
            let llama_model = Arc::new(LlamaModel::new_random(llama_cfg, backend.clone())?);

            let bert_cfg = BertConfig::tiny_test_config();
            let bert_model =
                Arc::new(BertEmbeddingModel::new_random(bert_cfg.clone(), backend.clone())?);

            let sched_cfg = SchedulerConfig {
                max_batch_size,
                block_size: 16,
                total_blocks: 1024,
            };
            let scheduler = Arc::new(ContinuousBatchScheduler::new(sched_cfg, llama_model)?);

            let rag_engine = Arc::new(RagEngine::new(
                bert_model,
                scheduler.clone(),
                bert_cfg.hidden_size,
            ));

            let state = Arc::new(AppState {
                scheduler,
                rag_engine,
                model_name: model,
            });

            println!("⚡ Serving OpenAI-compatible API at http://{}", addr);
            println!("   - Health: http://{}/health", addr);
            println!("   - Chat:   http://{}/v1/chat/completions", addr);
            println!("   - Embed:  http://{}/v1/embeddings", addr);
            println!("   - RAG:    http://{}/v1/rag/query", addr);

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

        Commands::RagDemo { query } => {
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
                total_blocks: 256,
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

            let resp = rag_engine.query(&query, 2, sampling).await?;
            let elapsed = start.elapsed();

            println!("\n✨ Answer: {}", resp.answer);
            println!("⏱️  Query Latency: {:.2?}", elapsed);
            println!("📑 Retrieved Citations:");
            for (i, cit) in resp.citations.iter().enumerate() {
                println!(
                    "   [{}] Document: {} (Relevance: {:.2})\n       \"{}\"",
                    i + 1,
                    cit.doc_id,
                    cit.relevance,
                    cit.text
                );
            }
        }
    }

    Ok(())
}

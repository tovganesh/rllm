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
use rllm_models::{
    BertConfig, BertEmbeddingModel, ChatMessage, GgufBundle, LlamaConfig, LlamaModel, LoadOptions,
    StreamDecoder,
};
use rllm_rag::RagEngine;
use rllm_scheduler::{ContinuousBatchScheduler, SamplingParams, SchedulerConfig, Sequence};
use std::io::Write;
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

        /// Path to a GGUF model file, or a synthetic preset: 'edge-4gb', '1b', 'tiny'
        #[arg(long, default_value = "edge-4gb")]
        model: String,

        #[arg(long, default_value_t = 32)]
        max_batch_size: usize,

        /// Cap on the model context window (GGUF models)
        #[arg(long, default_value_t = 4096)]
        max_context: usize,

        /// Total KV-cache capacity in tokens, shared by all concurrent sequences
        #[arg(long, default_value_t = 16384)]
        kv_cache_tokens: usize,
    },

    /// Generate text from a GGUF model
    Generate {
        /// Path to a GGUF model file (LLaMA architecture, byte-level BPE tokenizer)
        #[arg(long)]
        model: std::path::PathBuf,

        /// User prompt (or raw text with --raw)
        #[arg(long)]
        prompt: String,

        /// Optional system message (chat mode only)
        #[arg(long)]
        system: Option<String>,

        /// Feed the prompt as raw text instead of applying the model's chat template
        #[arg(long)]
        raw: bool,

        #[arg(long, default_value_t = 256)]
        max_tokens: usize,

        /// Sampling temperature; 0 selects greedy decoding
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        #[arg(long, default_value_t = 40)]
        top_k: usize,

        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f32,

        /// Cap on the model context window
        #[arg(long, default_value_t = 4096)]
        max_context: usize,
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
            max_context,
            kv_cache_tokens,
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
            let mut tokenizer = None;
            let mut chat_template = None;
            let mut model_name = model.clone();
            let scheduler = if mode == ServiceMode::All || mode == ServiceMode::Inference {
                let llama_model = if model.ends_with(".gguf") {
                    let bundle = GgufBundle::load(
                        &model,
                        backend.clone(),
                        &LoadOptions { max_context: Some(max_context) },
                    )?;
                    model_name = std::path::Path::new(&model)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or(model.clone());
                    tokenizer = Some(Arc::new(bundle.tokenizer));
                    chat_template = Some(Arc::new(bundle.chat_template));
                    Arc::new(bundle.model)
                } else {
                    let llama_cfg = match model.as_str() {
                        "edge-4gb" => LlamaConfig::edge_4gb(),
                        "1b" => LlamaConfig::llama_3_2_1b(),
                        _ => LlamaConfig::tiny_test_config(),
                    };
                    Arc::new(LlamaModel::new_random(llama_cfg, backend.clone())?)
                };
                let block_size = 16;
                let sched_cfg = SchedulerConfig {
                    max_batch_size,
                    block_size,
                    total_blocks: kv_cache_tokens.div_ceil(block_size).max(1),
                    ..SchedulerConfig::default()
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
                        ..SchedulerConfig::default()
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
                tokenizer,
                chat_template,
                rag_engine,
                inference_url: inference_url.clone(),
                model_name,
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

        Commands::Generate {
            model,
            prompt,
            system,
            raw,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_context,
        } => {
            run_generate(GenerateArgs {
                model,
                prompt,
                system,
                raw,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_context,
            })?;
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
                ..SchedulerConfig::default()
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
            let sampling = SamplingParams {
                max_tokens: 32,
                ..SamplingParams::default()
            };

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

struct GenerateArgs {
    model: std::path::PathBuf,
    prompt: String,
    system: Option<String>,
    raw: bool,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    max_context: usize,
}

/// `rllm generate`: load a GGUF model and stream a completion to stdout.
fn run_generate(args: GenerateArgs) -> anyhow::Result<()> {
    let load_start = Instant::now();
    let bundle = GgufBundle::load(
        &args.model,
        Arc::new(CpuBackend),
        &LoadOptions { max_context: Some(args.max_context) },
    )?;
    let load_time = load_start.elapsed();
    let tokenizer = &bundle.tokenizer;

    let prompt_text = if args.raw {
        args.prompt.clone()
    } else {
        let mut messages = Vec::new();
        if let Some(sys) = &args.system {
            messages.push(ChatMessage::new("system", sys));
        }
        messages.push(ChatMessage::new("user", &args.prompt));
        bundle.chat_template.render(&messages, true)?
    };
    let prompt_tokens = tokenizer.encode(&prompt_text, true, true);
    let prompt_len = prompt_tokens.len();

    let block_size = 16;
    let needed = (prompt_len + args.max_tokens + 1).min(bundle.model.config.max_position_embeddings);
    let scheduler = ContinuousBatchScheduler::new(
        SchedulerConfig {
            max_batch_size: 1,
            block_size,
            total_blocks: needed.div_ceil(block_size),
            max_prefill_tokens: prompt_len.max(1),
        },
        Arc::new(bundle.model),
    )?;

    let sampling = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        max_tokens: args.max_tokens,
        stop_tokens: tokenizer.stop_token_ids.clone(),
    };
    let seq = Sequence::new(1, prompt_tokens, sampling, block_size, None);

    let mut decoder = StreamDecoder::default();
    let mut stdout = std::io::stdout().lock();
    let start = Instant::now();
    let mut first_token_at = None;
    let generated = scheduler.generate(seq, |tok| {
        first_token_at.get_or_insert_with(|| start.elapsed());
        let text = decoder.push(tokenizer, tok, true);
        let _ = stdout.write_all(text.as_bytes());
        let _ = stdout.flush();
    })?;
    let _ = writeln!(stdout, "{}", decoder.finish());
    let total = start.elapsed();

    let ttft = first_token_at.unwrap_or(total);
    let decode_tokens = generated.len().saturating_sub(1);
    let decode_secs = (total - ttft).as_secs_f64();
    eprintln!(
        "\n[rllm] load {:.2?} | prompt {} tokens, prefill {:.2?} ({:.1} tok/s) | generated {} tokens, decode {:.1} tok/s",
        load_time,
        prompt_len,
        ttft,
        prompt_len as f64 / ttft.as_secs_f64().max(1e-9),
        generated.len(),
        if decode_secs > 0.0 { decode_tokens as f64 / decode_secs } else { 0.0 },
    );
    Ok(())
}

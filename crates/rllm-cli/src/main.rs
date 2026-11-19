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
    BertConfig, ChatMessage, GgufBundle, LanguageModel, LlamaConfig, LlamaModel, LoadOptions,
    StreamDecoder, TextEmbedder,
};
use rllm_rag::{LlmGenerator, RagEngine};
use rllm_scheduler::{
    ContinuousBatchScheduler, EngineConfig, EngineHandle, SamplingParams, SchedulerConfig, Sequence,
    SpeculationConfig,
};
use std::io::Write;
use rllm_server::{run_server, AppState, ServerLimits, ServerMetrics};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "rllm")]
#[command(about = "Enterprise-Grade LLM Inference & High-Performance RAG Engine in Rust", long_about = None)]
struct Cli {
    /// CPU worker threads (default: performance cores on Apple Silicon, otherwise all
    /// cores; RAYON_NUM_THREADS is also honoured)
    #[arg(long, global = true)]
    threads: Option<usize>,

    /// Device for the LLM: `cpu`, or `metal` (Apple GPU, macOS only). Embedding models
    /// always run on the CPU.
    #[arg(long, global = true, default_value = "cpu")]
    device: String,

    /// KV cache storage type for the LLM: `f16` (default, as in llama.cpp: half the
    /// memory and bandwidth) or `f32`.
    #[arg(long, global = true, default_value = "f16")]
    kv_dtype: String,

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
        #[arg(long, env = "RLLM_MODEL", default_value = "edge-4gb")]
        model: String,

        #[arg(long, default_value_t = 32)]
        max_batch_size: usize,

        /// Cap on the model context window (GGUF models)
        #[arg(long, default_value_t = 4096)]
        max_context: usize,

        /// Total KV-cache capacity in tokens, shared by all concurrent sequences
        #[arg(long, default_value_t = 16384)]
        kv_cache_tokens: usize,

        /// Path to a BERT-architecture GGUF embedding model (e.g. bge-small-en-v1.5).
        /// Without it, a synthetic embedder is used and dense retrieval is meaningless.
        #[arg(long, env = "RLLM_EMBEDDING_MODEL")]
        embedding_model: Option<std::path::PathBuf>,

        /// Minimum query/passage cosine similarity for a passage to count as relevant.
        /// Defaults to the calibrated value for known embedding models (bge 0.6, MiniLM 0.3).
        #[arg(long)]
        min_dense_similarity: Option<f32>,

        /// Maximum requests waiting for admission; further requests get HTTP 429
        #[arg(long, default_value_t = 256)]
        max_waiting: usize,

        /// Largest max_tokens a request may ask for
        #[arg(long, default_value_t = 4096)]
        max_tokens_limit: usize,

        /// Prompt-lookup speculative decoding (off by default): propose up to N draft
        /// tokens per step copied from earlier context and verify them in one forward
        /// pass. Helps when output repeats the prompt (code edits, quoting, RAG).
        #[arg(long, default_value_t = 0)]
        lookup_draft: usize,

        /// Longest suffix n-gram matched by --lookup-draft (tries shorter ones down to 1)
        #[arg(long, default_value_t = 3)]
        lookup_ngram: usize,
    },

    /// Generate text from a GGUF model
    Generate {
        /// Path to a GGUF model file (LLaMA architecture, byte-level BPE tokenizer)
        #[arg(long)]
        model: std::path::PathBuf,

        /// User prompt (or raw text with --raw)
        #[arg(long, required_unless_present = "prompt_file", conflicts_with = "prompt_file")]
        prompt: Option<String>,

        /// Read the prompt from a file instead of --prompt
        #[arg(long)]
        prompt_file: Option<std::path::PathBuf>,

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

        /// Prompt-lookup speculative decoding (off by default): propose up to N draft
        /// tokens per step copied from earlier context and verify them in one forward
        /// pass. Helps when output repeats the prompt (code edits, quoting, RAG).
        #[arg(long, default_value_t = 0)]
        lookup_draft: usize,

        /// Longest suffix n-gram matched by --lookup-draft (tries shorter ones down to 1)
        #[arg(long, default_value_t = 3)]
        lookup_ngram: usize,
    },

    /// Measure perplexity of a GGUF model on a text file (llama.cpp's protocol:
    /// non-overlapping chunks of --ctx tokens, scoring the second half of each chunk)
    Perplexity {
        #[arg(long)]
        model: std::path::PathBuf,

        /// UTF-8 text file to score
        #[arg(long)]
        file: std::path::PathBuf,

        /// Tokens per chunk
        #[arg(long, default_value_t = 128)]
        ctx: usize,
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

        /// Benchmark a real GGUF model instead of a synthetic preset, mirroring
        /// llama-bench: pp = one forward pass over --prompt-tokens tokens, tg = time per
        /// generated token over --generate-tokens tokens (batch size 1).
        #[arg(long)]
        model: Option<std::path::PathBuf>,

        /// Repetitions per measurement for --model (the median is reported)
        #[arg(long, default_value_t = 3)]
        repetitions: usize,

        /// With --model: tokens of context already in the KV cache before each pp and tg
        /// test (like llama-bench's `-d`), to measure long-context behaviour.
        #[arg(long, default_value_t = 0)]
        depth: usize,

        /// With --model: measure time to first token through the scheduler for prompts
        /// of --prompt-tokens tokens whose first N tokens are shared with the previous
        /// request (prefix cache warm) versus not (cold).
        #[arg(long)]
        shared_prefix: Option<usize>,
    },

    /// Run end-to-end RAG interactive demonstration with citations and relevance verification
    RagDemo {
        #[arg(long, default_value = "At what temperature does water boil at sea level?")]
        query: String,

        /// Number of passages to retrieve
        #[arg(long, default_value_t = 3)]
        top_k: usize,

        /// Minimum lexical relevance (0.0 to 1.0) for qualifying citations
        #[arg(long)]
        min_relevance: Option<f32>,

        /// GGUF LLM that writes the answer (otherwise the answer is extractive)
        #[arg(long)]
        model: Option<std::path::PathBuf>,

        /// GGUF BERT embedding model (otherwise a synthetic embedder is used)
        #[arg(long)]
        embedding_model: Option<std::path::PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let threads = rllm_backend_cpu::threads::init_thread_pool(cli.threads);
    tracing::info!("Using {threads} CPU worker threads");
    let device = LlmDevice::parse(&cli.device)?;
    let kv_dtype = match cli.kv_dtype.as_str() {
        "f16" => rllm_core::dtype::DType::F16,
        "f32" => rllm_core::dtype::DType::F32,
        other => anyhow::bail!("unknown --kv-dtype '{other}' (expected f16 or f32)"),
    };
    let _ = KV_DTYPE.set(kv_dtype);

    match cli.command {
        Commands::Serve {
            addr,
            service,
            inference_url,
            model,
            max_batch_size,
            max_context,
            kv_cache_tokens,
            embedding_model,
            min_dense_similarity,
            max_waiting,
            max_tokens_limit,
            lookup_draft,
            lookup_ngram,
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
                        &load_opts(max_context),
                    )?;
                    model_name = std::path::Path::new(&model)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or(model.clone());
                    tokenizer = Some(Arc::new(bundle.tokenizer));
                    chat_template = Some(Arc::new(bundle.chat_template));
                    device.executor(bundle.model)?
                } else {
                    let llama_cfg = match model.as_str() {
                        "edge-4gb" => LlamaConfig::edge_4gb(),
                        "1b" => LlamaConfig::llama_3_2_1b(),
                        _ => LlamaConfig::tiny_test_config(),
                    };
                    device.executor(LlamaModel::new_random(llama_cfg, backend.clone())?)?
                };
                let block_size = 16;
                let sched_cfg = SchedulerConfig {
                    max_batch_size,
                    block_size,
                    total_blocks: kv_cache_tokens.div_ceil(block_size).max(1),
                    speculation: speculation(lookup_draft, lookup_ngram),
                    ..SchedulerConfig::default()
                };
                let scheduler = Arc::new(ContinuousBatchScheduler::new(sched_cfg, llama_model)?);
                Some(EngineHandle::start(scheduler, EngineConfig { max_waiting }))
            } else {
                None
            };

            // 2. Embedding model (serves /v1/embeddings and RAG retrieval).
            let embedder = Arc::new(load_embedder(embedding_model.as_deref(), backend.clone())?);

            // 3. RAG engine. A locally loaded GGUF model writes the answers; otherwise the
            //    remote INFERENCE_URL does, or answers are extractive.
            let rag_engine = if mode == ServiceMode::All || mode == ServiceMode::Rag {
                let generator = match (&scheduler, &tokenizer, &chat_template) {
                    (Some(s), Some(t), Some(c)) if inference_url.is_none() => Some(LlmGenerator {
                        engine: s.clone(),
                        tokenizer: t.clone(),
                        chat_template: c.clone(),
                    }),
                    _ => None,
                };
                let mut engine = RagEngine::new(embedder.clone(), generator);
                if let Some(t) = min_dense_similarity {
                    engine = engine.with_dense_threshold(t);
                }
                Some(Arc::new(engine))
            } else {
                None
            };

            let state = Arc::new(AppState {
                engine: scheduler,
                tokenizer,
                chat_template,
                embedder: Some(embedder),
                rag_engine,
                inference_url: inference_url.clone(),
                model_name,
                mode,
                limits: ServerLimits {
                    max_tokens: max_tokens_limit,
                    ..ServerLimits::default()
                },
                metrics: Arc::new(ServerMetrics::default()),
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
            prompt_file,
            system,
            raw,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_context,
            lookup_draft,
            lookup_ngram,
        } => {
            let prompt = match (prompt, prompt_file) {
                (Some(p), _) => p,
                (None, Some(f)) => std::fs::read_to_string(&f)
                    .map_err(|e| anyhow::anyhow!("reading {}: {e}", f.display()))?,
                (None, None) => anyhow::bail!("--prompt or --prompt-file is required"),
            };
            run_generate(GenerateArgs {
                device,
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
                speculation: speculation(lookup_draft, lookup_ngram),
            })?;
        }

        Commands::Perplexity { model, file, ctx } => {
            run_perplexity(&model, &file, ctx, device)?;
        }

        Commands::Bench {
            preset,
            batch_size,
            prompt_tokens,
            generate_tokens,
            warmup,
            model,
            repetitions,
            depth,
            shared_prefix,
        } => {
            if let Some(path) = model {
                match shared_prefix {
                    Some(shared) => run_prefix_bench(&path, prompt_tokens, shared, repetitions.max(1), device)?,
                    None => run_model_bench(&path, prompt_tokens, generate_tokens, depth, repetitions.max(1), device)?,
                }
                return Ok(());
            }
            let opts = BenchmarkOptions {
                preset,
                batch_size,
                prompt_tokens,
                generate_tokens,
                warmup,
            };
            run_comprehensive_benchmark(opts)?;
        }

        Commands::RagDemo {
            query,
            top_k,
            min_relevance,
            model,
            embedding_model,
        } => {
            let backend = Arc::new(CpuBackend);
            let embedder = Arc::new(load_embedder(embedding_model.as_deref(), backend.clone())?);
            let generator = match model {
                Some(path) => {
                    let b = GgufBundle::load(&path, backend.clone(), &load_opts(4096))?;
                    let scheduler = ContinuousBatchScheduler::new(
                        SchedulerConfig {
                            max_batch_size: 1,
                            block_size: 16,
                            total_blocks: 256,
                            ..SchedulerConfig::default()
                        },
                        device.executor(b.model)?,
                    )?;
                    Some(LlmGenerator {
                        engine: EngineHandle::start(Arc::new(scheduler), EngineConfig::default()),
                        tokenizer: Arc::new(b.tokenizer),
                        chat_template: Arc::new(b.chat_template),
                    })
                }
                None => None,
            };
            let rag_engine = RagEngine::new(embedder.clone(), generator);

            let docs = [
                ("water", "Water boils at 100 degrees Celsius (212 Fahrenheit) at sea level. At higher altitudes the boiling point is lower because air pressure is lower; in Denver water boils at about 95 degrees Celsius."),
                ("photosynthesis", "Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide to produce glucose and release oxygen. It takes place mainly in the chloroplasts of leaf cells."),
                ("paris", "Paris is the capital and most populous city of France, located on the Seine river. It is known for the Eiffel Tower, completed in 1889."),
                ("rust", "The Rust programming language guarantees memory safety without a garbage collector through its ownership and borrowing system, checked at compile time."),
                ("reef", "The Great Barrier Reef, off the coast of Queensland, Australia, is the world's largest coral reef system, stretching for over 2,300 kilometres."),
            ];
            println!("Embedder: {}{}", embedder.name, if embedder.is_synthetic() { " (synthetic: dense retrieval disabled)" } else { "" });
            println!("Indexing {} documents...", docs.len());
            for (id, text) in docs {
                rag_engine.ingest_document(id, text, HashMap::new())?;
            }

            println!("\nQuery: {query}");
            let start = Instant::now();
            let sampling = SamplingParams {
                temperature: 0.0,
                max_tokens: 128,
                ..SamplingParams::default()
            };
            let resp = rag_engine
                .query_with_threshold(&query, top_k, sampling, min_relevance)
                .await?;
            println!("Latency: {:.2?}", start.elapsed());
            println!(
                "Relevant: {} | lexical confidence {:.2} | max dense similarity {}",
                resp.is_relevant,
                resp.confidence_score,
                resp.max_dense_similarity.map_or("n/a".to_string(), |d| format!("{d:.2}"))
            );
            println!("\nAnswer ({}, {} tokens generated):\n{}", resp.answer_source, resp.generated_tokens, resp.answer);
            if let Some(w) = &resp.warning {
                println!("\nWarning: {w}");
            }
            for (i, c) in resp.citations.iter().enumerate() {
                println!(
                    "\n[{}] {} (lexical {:.2}, dense {}, cited: {})\n    {}",
                    i + 1,
                    c.doc_id,
                    c.relevance,
                    c.dense_similarity.map_or("n/a".to_string(), |d| format!("{d:.2}")),
                    c.cited_in_answer,
                    c.passage
                );
            }
        }
    }

    Ok(())
}

struct GenerateArgs {
    device: LlmDevice,
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
    speculation: SpeculationConfig,
}

/// Prompt-lookup settings from `--lookup-draft` / `--lookup-ngram`.
fn speculation(max_draft_tokens: usize, max_ngram: usize) -> SpeculationConfig {
    SpeculationConfig {
        max_draft_tokens,
        min_ngram: 1,
        max_ngram: max_ngram.max(1),
        adaptive: true,
    }
}

/// `rllm generate`: load a GGUF model and stream a completion to stdout.
fn run_generate(args: GenerateArgs) -> anyhow::Result<()> {
    let load_start = Instant::now();
    let bundle = GgufBundle::load(
        &args.model,
        Arc::new(CpuBackend),
        &load_opts(args.max_context),
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
            speculation: args.speculation.clone(),
        },
        args.device.executor(bundle.model)?,
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
    if args.speculation.enabled() {
        let m = &scheduler.metrics;
        let drafted = m.draft_tokens.load(std::sync::atomic::Ordering::Relaxed);
        let accepted = m.accepted_draft_tokens.load(std::sync::atomic::Ordering::Relaxed);
        let passes = m.forward_passes.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(1);
        eprintln!(
            "[rllm] prompt lookup: {accepted}/{drafted} draft tokens accepted ({:.0}%), {:.2} tokens per decode pass",
            100.0 * accepted as f64 / drafted.max(1) as f64,
            decode_tokens as f64 / passes.max(1) as f64,
        );
    }
    Ok(())
}

/// Loads a GGUF embedding model, or a synthetic one when no path is given.
fn load_embedder(path: Option<&std::path::Path>, backend: Arc<CpuBackend>) -> anyhow::Result<TextEmbedder<CpuBackend>> {
    Ok(match path {
        Some(p) => TextEmbedder::from_gguf(p, backend)?,
        None => TextEmbedder::synthetic(BertConfig::tiny_test_config(), backend)?,
    })
}

/// `rllm perplexity`: mirrors `llama-perplexity` so results can be compared directly.
fn run_perplexity(model: &std::path::Path, file: &std::path::Path, ctx: usize, device: LlmDevice) -> anyhow::Result<()> {
    let bundle = GgufBundle::load(model, Arc::new(CpuBackend), &load_opts(ctx))?;
    let text = std::fs::read_to_string(file)?;
    let tokens = bundle.tokenizer.encode(&text, true, false);
    let bos = if bundle.tokenizer.add_bos { bundle.tokenizer.bos_token_id } else { None };
    let start = Instant::now();
    let lm = device.executor(bundle.model)?;
    let p = rllm_models::perplexity(lm.as_ref(), &tokens, ctx, bos)?;
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{} chunks of {} tokens, {} scored: PPL = {:.4} ({:.2}s, {:.0} tok/s)",
        p.chunks,
        ctx,
        p.scored_tokens,
        p.ppl,
        secs,
        (p.chunks * ctx) as f64 / secs
    );
    Ok(())
}

/// Where the LLM runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlmDevice {
    Cpu,
    Metal,
}

impl LlmDevice {
    fn parse(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "metal" | "gpu" => Ok(Self::Metal),
            other => anyhow::bail!("unknown --device '{other}' (expected cpu or metal)"),
        }
    }

    /// Wraps a CPU-loaded model for this device.
    fn executor(self, model: LlamaModel<CpuBackend>) -> anyhow::Result<Arc<dyn LanguageModel>> {
        match self {
            Self::Cpu => Ok(Arc::new(model)),
            Self::Metal => {
                #[cfg(target_os = "macos")]
                {
                    let gpu = rllm_backend_metal::MetalLlama::from_cpu_model(&model)?;
                    tracing::info!("LLM running on {}", gpu.device_name());
                    Ok(Arc::new(gpu))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = model;
                    anyhow::bail!("--device metal is only available on macOS")
                }
            }
        }
    }
}

/// `rllm bench --model`: llama-bench-style prompt-processing and generation throughput,
/// optionally after `depth` tokens of context (llama-bench `-d`).
fn run_model_bench(path: &std::path::Path, pp: usize, tg: usize, depth: usize, reps: usize, device: LlmDevice) -> anyhow::Result<()> {
    use rllm_core::backend::BlockTable;
    use rllm_models::ForwardBatch;

    let ctx = depth + pp.max(tg) + 1;
    let bundle = GgufBundle::load(path, Arc::new(CpuBackend), &load_opts(ctx.max(16)))?;
    let vocab = bundle.model.config.vocab_size as u32;
    let lm = device.executor(bundle.model)?;
    let block = 16;
    let blocks = ctx.div_ceil(block);
    lm.init_kv_cache(blocks, block)?;
    let table = BlockTable { block_size: block, block_ids: (0..blocks).collect() };
    // Deterministic pseudo-random tokens, as llama-bench uses random tokens.
    let token = |i: usize| ((i as u32).wrapping_mul(2654435761) >> 8) % vocab;
    let prefill = |start: usize, n: usize| -> anyhow::Result<()> {
        let tokens: Vec<u32> = (start..start + n).map(token).collect();
        let positions: Vec<usize> = (start..start + n).collect();
        let tables = vec![table.clone(); n];
        lm.forward_cached(&ForwardBatch { tokens: &tokens, positions: &positions, block_tables: &tables, logits_rows: &[n - 1] })?;
        Ok(())
    };
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };

    // Context of `depth` tokens (in chunks, as a server would prefill), then a warm-up
    // pass (shader/pipeline caches, page faults).
    let mut done = 0;
    while done < depth {
        let n = (depth - done).min(2048);
        prefill(done, n)?;
        done += n;
    }
    if pp > 0 {
        prefill(depth, pp)?;
    }

    let mut pp_rates = Vec::new();
    for _ in 0..reps {
        if pp == 0 {
            break;
        }
        let t = Instant::now();
        prefill(depth, pp)?;
        pp_rates.push(pp as f64 / t.elapsed().as_secs_f64());
    }
    let mut tg_rates = Vec::new();
    for _ in 0..reps {
        if tg == 0 {
            break;
        }
        let t = Instant::now();
        for i in 0..tg {
            let tok = [(i as u32 * 7919) % vocab];
            let pos = depth + i;
            lm.forward_cached(&ForwardBatch { tokens: &tok, positions: &[pos], block_tables: std::slice::from_ref(&table), logits_rows: &[0] })?;
        }
        tg_rates.push(tg as f64 / t.elapsed().as_secs_f64());
    }
    println!("| model | device | test | t/s |");
    println!("| --- | --- | --- | ---: |");
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let at = if depth > 0 { format!(" @ d{depth}") } else { String::new() };
    if !pp_rates.is_empty() {
        println!("| {name} | {} | pp{pp}{at} | {:.1} |", lm.device_name(), median(pp_rates));
    }
    if !tg_rates.is_empty() {
        println!("| {name} | {} | tg{tg}{at} | {:.1} |", lm.device_name(), median(tg_rates));
    }
    Ok(())
}

/// `rllm bench --model --shared-prefix N`: time to first token through the scheduler
/// for a `prompt`-token request whose first `shared` tokens a previous request already
/// computed (warm prefix cache), versus a request with nothing cached (cold).
fn run_prefix_bench(path: &std::path::Path, prompt: usize, shared: usize, reps: usize, device: LlmDevice) -> anyhow::Result<()> {
    anyhow::ensure!(shared < prompt, "--shared-prefix must be smaller than --prompt-tokens");
    let bundle = GgufBundle::load(path, Arc::new(CpuBackend), &load_opts(prompt + 16))?;
    let vocab = bundle.model.config.vocab_size as u32;
    let lm = device.executor(bundle.model)?;
    let block_size = 16;
    let tokens = |seed: u32, n: usize| -> Vec<u32> {
        (0..n as u32).map(|i| (i.wrapping_add(seed.wrapping_mul(7919)).wrapping_mul(2654435761) >> 8) % vocab).collect()
    };
    let one_token = SamplingParams { temperature: 0.0, max_tokens: 1, stop_tokens: Vec::new(), ..SamplingParams::default() };
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let (mut cold, mut warm) = (Vec::new(), Vec::new());
    let mut cached = 0;
    for r in 0..=reps {
        let scheduler = ContinuousBatchScheduler::new(
            SchedulerConfig {
                max_batch_size: 1,
                block_size,
                total_blocks: (3 * prompt).div_ceil(block_size) + 4,
                max_prefill_tokens: prompt + 1,
                ..SchedulerConfig::default()
            },
            lm.clone(),
        )?;
        let prefix = tokens(1000 + r as u32, shared);
        let ttft = |suffix_seed: u32| -> anyhow::Result<f64> {
            let mut p = prefix.clone();
            p.extend(tokens(suffix_seed, prompt - shared));
            scheduler.submit_sequence(Sequence::new(scheduler.next_sequence_id(), p, one_token.clone(), block_size, None))?;
            let t = Instant::now();
            scheduler.step()?;
            let dt = t.elapsed().as_secs_f64();
            while scheduler.has_work() {
                scheduler.step()?;
            }
            Ok(dt)
        };
        let c = ttft(1)?;
        let before = scheduler.metrics.cached_prompt_tokens.load(std::sync::atomic::Ordering::Relaxed);
        let w = ttft(2)?;
        cached = scheduler.metrics.cached_prompt_tokens.load(std::sync::atomic::Ordering::Relaxed) - before;
        if r > 0 {
            // The first round only warms up pipelines and page tables.
            cold.push(c);
            warm.push(w);
        }
    }
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let (c, w) = (median(cold) * 1e3, median(warm) * 1e3);
    println!("| model | device | prompt | shared prefix | cached tokens | TTFT cold ms | TTFT warm ms | speed-up |");
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    println!("| {name} | {} | {prompt} | {shared} | {cached} | {c:.1} | {w:.1} | {:.2}x |", lm.device_name(), c / w);
    Ok(())
}

/// KV cache storage type chosen with `--kv-dtype`.
static KV_DTYPE: std::sync::OnceLock<rllm_core::dtype::DType> = std::sync::OnceLock::new();

/// Load options for the LLM: context cap plus the `--kv-dtype` setting.
fn load_opts(max_context: usize) -> LoadOptions {
    LoadOptions {
        max_context: Some(max_context),
        kv_dtype: KV_DTYPE.get().copied().unwrap_or(rllm_core::dtype::DType::F16),
        ..LoadOptions::default()
    }
}

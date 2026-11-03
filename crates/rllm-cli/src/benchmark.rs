use rllm_backend_cpu::{
    ops::{
        gemm::gemv_f32,
        norm::rms_norm_f32,
        paged_attn::paged_attention_decode_single,
        quant::{gemv_q4_0_f32, quantize_f32_to_q4_0},
    },
    CpuBackend,
};
use rllm_core::{
    backend::BlockTable,
    dtype::DType,
    error::Result,
};
use rllm_models::{LlamaConfig, LlamaModel};
use rllm_scheduler::{
    ContinuousBatchScheduler, SamplingParams, SchedulerConfig, Sequence,
};
use std::sync::Arc;
use std::time::Instant;

pub struct BenchmarkOptions {
    pub preset: String,
    pub batch_size: usize,
    pub prompt_tokens: usize,
    pub generate_tokens: usize,
    pub warmup: usize,
}

pub struct LatencyStats {
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

fn calculate_percentiles(mut latencies: Vec<f64>) -> LatencyStats {
    if latencies.is_empty() {
        return LatencyStats {
            mean_ms: 0.0,
            p50_ms: 0.0,
            p90_ms: 0.0,
            p99_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
        };
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let count = latencies.len();
    let sum: f64 = latencies.iter().sum();
    let mean_ms = sum / count as f64;
    let min_ms = latencies[0];
    let max_ms = latencies[count - 1];

    let p50_ms = latencies[(count as f64 * 0.50) as usize];
    let p90_ms = latencies[((count as f64 * 0.90) as usize).min(count - 1)];
    let p99_ms = latencies[((count as f64 * 0.99) as usize).min(count - 1)];

    LatencyStats {
        mean_ms,
        p50_ms,
        p90_ms,
        p99_ms,
        min_ms,
        max_ms,
    }
}

pub fn run_comprehensive_benchmark(opts: BenchmarkOptions) -> Result<()> {
    println!("══════════════════════════════════════════════════════════════════════");
    println!("          rllm High-Performance Inference Benchmark Suite             ");
    println!("══════════════════════════════════════════════════════════════════════\n");

    let config = match opts.preset.as_str() {
        "1b" | "llama-1b" => LlamaConfig::llama_3_2_1b(),
        "tiny" => LlamaConfig::tiny_test_config(),
        _ => LlamaConfig::edge_4gb(),
    };

    let params_count = config.num_parameters();
    let (weights_fp32, kv_fp32, total_fp32) =
        config.estimated_memory_mb(DType::F32, opts.prompt_tokens + opts.generate_tokens, opts.batch_size);
    let (weights_q4, _, total_q4) =
        config.estimated_memory_mb(DType::Q4_0, opts.prompt_tokens + opts.generate_tokens, opts.batch_size);

    println!("📋 Target Model Architecture & 4GB Fit Analysis:");
    println!("   • Preset:              {}", opts.preset);
    println!("   • Parameter Count:     {:.2} Million", params_count as f64 / 1_000_000.0);
    println!("   • Hidden Size:         {} (Intermediate: {})", config.hidden_size, config.intermediate_size);
    println!("   • Attention Heads:     {} (KV Heads: {}, GQA {:.1}:1)",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_attention_heads as f64 / config.num_key_value_heads as f64
    );
    println!("   • Layers:              {}", config.num_hidden_layers);
    println!("   • Vocab Size:          {}", config.vocab_size);
    println!("\n💾 Memory Footprint Profile (4.0 GB Target Envelope):");
    println!("   ┌────────────────────────┬─────────────┬─────────────┬────────────────┐");
    println!("   │ Format                 │ Weights     │ KV Cache    │ Total RAM (%4G)│");
    println!("   ├────────────────────────┼─────────────┼─────────────┼────────────────┤");
    println!("   │ FP32 (Unquantized)     │ {:>7.1} MB  │ {:>7.1} MB  │ {:>6.1} MB ({:>4.1}%)│",
        weights_fp32, kv_fp32, total_fp32, (total_fp32 / 4096.0) * 100.0);
    println!("   │ Q4_0 (4-bit Block)     │ {:>7.1} MB  │ {:>7.1} MB  │ {:>6.1} MB ({:>4.1}%)│",
        weights_q4, kv_fp32, total_q4, (total_q4 / 4096.0) * 100.0);
    println!("   └────────────────────────┴─────────────┴─────────────┴────────────────┘");
    println!("   💡 Headroom on 4GB Device: {:.1} MB available for OS & user applications.\n",
        4096.0 - total_fp32);

    // ──────────────────────────────────────────────────────────────────
    // 1. Kernel Micro-Benchmarks (GEMV, Quantized GEMV, RoPE, PagedAttn)
    // ──────────────────────────────────────────────────────────────────
    println!("⚡ Micro-Kernel Performance Benchmarks:");
    benchmark_micro_kernels(&config)?;

    // ──────────────────────────────────────────────────────────────────
    // 2. End-to-End LLM Generation Benchmark
    // ──────────────────────────────────────────────────────────────────
    println!("\n🚀 End-to-End Inference Generation Benchmark:");
    println!("   Batch Size: {}, Context Prompt: {} tokens, Generated: {} tokens",
        opts.batch_size, opts.prompt_tokens, opts.generate_tokens);

    let backend = Arc::new(CpuBackend);
    let model = Arc::new(LlamaModel::new_random(config.clone(), backend)?);

    let sched_cfg = SchedulerConfig {
        max_batch_size: opts.batch_size,
        block_size: 16,
        total_blocks: 2048,
    };
    let scheduler = Arc::new(ContinuousBatchScheduler::new(sched_cfg, model)?);

    let mut sampling = SamplingParams::default();
    sampling.max_tokens = opts.generate_tokens;
    sampling.stop_tokens.clear(); // Ensure deterministic run

    let prompt: Vec<u32> = (0..opts.prompt_tokens).map(|i| (i % 500 + 1) as u32).collect();

    // Warmup iterations
    for w in 0..opts.warmup {
        let seq = Sequence::new(
            (1000 + w) as u64,
            prompt.clone(),
            sampling.clone(),
            scheduler.config.block_size,
            None,
        );
        scheduler.submit_sequence(seq)?;
        while scheduler.num_running() > 0 || scheduler.num_waiting() > 0 {
            let _ = scheduler.step()?;
        }
    }

    // Submit benchmark batch
    for i in 0..opts.batch_size {
        let seq = Sequence::new(
            i as u64,
            prompt.clone(),
            sampling.clone(),
            scheduler.config.block_size,
            None,
        );
        scheduler.submit_sequence(seq)?;
    }

    let start_time = Instant::now();
    let mut step_latencies = Vec::new();
    let mut ttft = None;
    let mut total_tokens = 0;

    loop {
        let step_start = Instant::now();
        let gen = scheduler.step()?;
        let step_elapsed = step_start.elapsed().as_secs_f64() * 1000.0;

        if ttft.is_none() && !gen.is_empty() {
            ttft = Some(start_time.elapsed());
        }

        if !gen.is_empty() {
            step_latencies.push(step_elapsed);
            total_tokens += gen.len();
        }

        if scheduler.num_running() == 0 && scheduler.num_waiting() == 0 {
            break;
        }
    }

    let total_elapsed = start_time.elapsed();
    let ttft_val = ttft.unwrap_or_default();
    let stats = calculate_percentiles(step_latencies);
    let tokens_per_sec = total_tokens as f64 / total_elapsed.as_secs_f64();

    println!("\n📊 Latency & Throughput Results:");
    println!("   • Total Tokens Generated:   {}", total_tokens);
    println!("   • Time to First Token (TTFT): {:.2?}", ttft_val);
    println!("   • Total Execution Time:     {:.2?}", total_elapsed);
    println!("   • Generation Throughput:    {:.1} tokens/sec", tokens_per_sec);
    println!("\n⏱️  Per-Token Latency Percentiles (Autoregressive Decode):");
    println!("   • Mean Latency:             {:.2} ms / token", stats.mean_ms);
    println!("   • Median (P50):             {:.2} ms / token", stats.p50_ms);
    println!("   • P90:                      {:.2} ms / token", stats.p90_ms);
    println!("   • P99:                      {:.2} ms / token", stats.p99_ms);
    println!("   • Min / Max:                {:.2} ms / {:.2} ms", stats.min_ms, stats.max_ms);

    // ──────────────────────────────────────────────────────────────────
    // 3. Batch Scaling Efficiency
    // ──────────────────────────────────────────────────────────────────
    println!("\n📈 Batch Scaling Efficiency Analysis:");
    run_batch_scaling_analysis(&config)?;

    // ──────────────────────────────────────────────────────────────────
    // 4. Radix Prefix Cache Acceleration
    // ──────────────────────────────────────────────────────────────────
    println!("\n⚡ Radix-Tree Prefix Cache Benchmark:");
    run_prefix_cache_benchmark(&config)?;

    println!("\n══════════════════════════════════════════════════════════════════════");
    println!("                    Benchmark Complete                                ");
    println!("══════════════════════════════════════════════════════════════════════");
    Ok(())
}

fn benchmark_micro_kernels(config: &LlamaConfig) -> Result<()> {
    let k = config.hidden_size;
    let n = config.intermediate_size;

    // 1. GEMV F32 vs Q4_0
    let weights_f32 = vec![1.0f32; n * k];
    let act_f32 = vec![1.0f32; k];
    let mut out_f32 = vec![0.0f32; n];

    let iters = 20;

    let start = Instant::now();
    for _ in 0..iters {
        gemv_f32(&weights_f32, &act_f32, &mut out_f32, n, k);
    }
    let f32_dur = start.elapsed() / (iters as u32);

    let mut q4_bytes = vec![0u8; (n * k / 32) * 18];
    for row in 0..n {
        let row_src = &weights_f32[row * k..(row + 1) * k];
        let row_dst = &mut q4_bytes[row * (k / 32 * 18)..(row + 1) * (k / 32 * 18)];
        quantize_f32_to_q4_0(row_src, row_dst);
    }

    let start = Instant::now();
    for _ in 0..iters {
        gemv_q4_0_f32(&q4_bytes, &act_f32, &mut out_f32, n, k);
    }
    let q4_dur = start.elapsed() / (iters as u32);

    let speedup = f32_dur.as_secs_f64() / q4_dur.as_secs_f64();
    println!("   • GEMV [{}, {}]:", n, k);
    println!("     - FP32 GEMV Latency:      {:.2?}", f32_dur);
    println!("     - Q4_0 Quantized GEMV:    {:.2?}", q4_dur);
    println!("     - Quantization Speedup:   {:.2}x faster (memory bandwidth savings)", speedup);

    // 2. RMSNorm
    let mut norm_out = vec![0.0f32; k];
    let weight = vec![1.0f32; k];
    let start = Instant::now();
    for _ in 0..50 {
        rms_norm_f32(&act_f32, &weight, &mut norm_out, 1, k, 1e-5);
    }
    let norm_dur = start.elapsed() / 50;
    println!("   • RMSNorm [{}]:              {:.2?}", k, norm_dur);

    // 3. PagedAttention decode step
    let block_size = 16;
    let num_blocks = 64;
    let head_dim = config.head_dim();
    let num_heads_q = config.num_attention_heads;
    let num_heads_kv = config.num_key_value_heads;

    let q = vec![1.0f32; num_heads_q * head_dim];
    let k_cache = vec![1.0f32; num_blocks * num_heads_kv * block_size * head_dim];
    let v_cache = vec![1.0f32; num_blocks * num_heads_kv * block_size * head_dim];
    let block_table = BlockTable {
        block_size,
        block_ids: (0..num_blocks).collect(),
    };
    let mut attn_out = vec![0.0f32; num_heads_q * head_dim];

    let start = Instant::now();
    for _ in 0..iters {
        paged_attention_decode_single(
            &q,
            &k_cache,
            &v_cache,
            &block_table,
            block_size * num_blocks,
            num_heads_q,
            num_heads_kv,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            &mut attn_out,
        );
    }
    let attn_dur = start.elapsed() / (iters as u32);
    println!("   • PagedAttention (1024 ctx): {:.2?}", attn_dur);

    Ok(())
}

fn run_batch_scaling_analysis(config: &LlamaConfig) -> Result<()> {
    let backend = Arc::new(CpuBackend);
    let model = Arc::new(LlamaModel::new_random(config.clone(), backend)?);

    let batch_sizes = [1, 2, 4, 8];
    println!("   ┌────────────┬──────────────────┬─────────────────┬──────────────┐");
    println!("   │ Batch Size │ Total Tokens/Sec │ Per-Seq Latency │ Scaling Eff. │");
    println!("   ├────────────┼──────────────────┼─────────────────┼──────────────┤");

    let mut base_tps = 0.0;

    for &bs in &batch_sizes {
        let sched_cfg = SchedulerConfig {
            max_batch_size: bs,
            block_size: 16,
            total_blocks: 1024,
        };
        let scheduler = ContinuousBatchScheduler::new(sched_cfg, model.clone())?;

        let mut sampling = SamplingParams::default();
        sampling.max_tokens = 8;
        sampling.stop_tokens.clear();

        for i in 0..bs {
            let seq = Sequence::new(
                i as u64,
                vec![10, 20, 30, 40],
                sampling.clone(),
                scheduler.config.block_size,
                None,
            );
            scheduler.submit_sequence(seq)?;
        }

        let start = Instant::now();
        let mut tokens_gen = 0;
        loop {
            let gen = scheduler.step()?;
            tokens_gen += gen.len();
            if scheduler.num_running() == 0 && scheduler.num_waiting() == 0 {
                break;
            }
        }
        let dur = start.elapsed();
        let tps = tokens_gen as f64 / dur.as_secs_f64();
        let per_seq_lat = (dur.as_secs_f64() * 1000.0) / 8.0;

        if bs == 1 {
            base_tps = tps;
            println!("   │ {:>10} │ {:>13.1} t/s │ {:>12.2} ms │ {:>11.1}x │",
                bs, tps, per_seq_lat, 1.0);
        } else {
            let scaling = tps / base_tps.max(1e-6);
            println!("   │ {:>10} │ {:>13.1} t/s │ {:>12.2} ms │ {:>11.2}x │",
                bs, tps, per_seq_lat, scaling);
        }
    }
    println!("   └────────────┴──────────────────┴─────────────────┴──────────────┘");
    Ok(())
}

fn run_prefix_cache_benchmark(config: &LlamaConfig) -> Result<()> {
    let backend = Arc::new(CpuBackend);
    let model = Arc::new(LlamaModel::new_random(config.clone(), backend)?);

    let sched_cfg = SchedulerConfig {
        max_batch_size: 2,
        block_size: 16,
        total_blocks: 512,
    };
    let scheduler = ContinuousBatchScheduler::new(sched_cfg, model)?;

    let mut sampling = SamplingParams::default();
    sampling.max_tokens = 4;
    sampling.stop_tokens.clear();

    // 64-token shared system prompt
    let shared_prompt: Vec<u32> = (0..64).map(|i| (i % 500 + 1) as u32).collect();

    // Request 1: Cold start (uncached)
    let seq1 = Sequence::new(
        1,
        shared_prompt.clone(),
        sampling.clone(),
        scheduler.config.block_size,
        None,
    );
    let start_cold = Instant::now();
    scheduler.submit_sequence(seq1)?;
    while scheduler.num_running() > 0 || scheduler.num_waiting() > 0 {
        let _ = scheduler.step()?;
    }
    let cold_dur = start_cold.elapsed();

    // Request 2: Warm hit (full prefix match from Radix tree!)
    let seq2 = Sequence::new(
        2,
        shared_prompt.clone(),
        sampling.clone(),
        scheduler.config.block_size,
        None,
    );
    let start_warm = Instant::now();
    scheduler.submit_sequence(seq2)?;
    while scheduler.num_running() > 0 || scheduler.num_waiting() > 0 {
        let _ = scheduler.step()?;
    }
    let warm_dur = start_warm.elapsed();

    println!("   • Cold Prefill + Generation: {:.2?}", cold_dur);
    println!("   • Warm Prefix Hit + Gen:     {:.2?}", warm_dur);
    let speedup = cold_dur.as_secs_f64() / warm_dur.as_secs_f64().max(1e-9);
    println!("   • Radix Cache Speedup:       {:.2}x faster prefill turnaround!", speedup);

    Ok(())
}

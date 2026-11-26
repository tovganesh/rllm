//! Behavioural tests for `ContinuousBatchScheduler`.
//!
//! Greedy generations from the scheduler (paged KV cache, prefix reuse, batching,
//! preemption) are compared with a reference that recomputes a full, cache-free
//! prefill of the whole sequence for every generated token.

use rllm_backend_cpu::CpuBackend;
use rllm_core::{backend::BlockTable, device::Device, dtype::DType, shape::Shape, tensor::Tensor};
use rllm_models::{ForwardBatch, LlamaConfig, LlamaModel};
use rllm_scheduler::{ContinuousBatchScheduler, SamplingParams, SchedulerConfig, Sequence};
use std::collections::HashMap;
use std::sync::Arc;

const BLOCK: usize = 4;

fn config() -> LlamaConfig {
    LlamaConfig {
        hidden_size: 32,
        intermediate_size: 48,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        num_hidden_layers: 2,
        vocab_size: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
    }
}

fn model() -> Arc<LlamaModel<CpuBackend>> {
    Arc::new(LlamaModel::new_seeded(config(), Arc::new(CpuBackend), 7).unwrap())
}

fn scheduler(total_blocks: usize, max_batch_size: usize) -> ContinuousBatchScheduler {
    ContinuousBatchScheduler::new(
        SchedulerConfig {
            max_batch_size,
            block_size: BLOCK,
            total_blocks,
            ..SchedulerConfig::default()
        },
        model(),
    )
    .unwrap()
}

fn greedy(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        stop_tokens: Vec::new(),
        ..SamplingParams::default()
    }
}

fn seq(id: u64, prompt: &[u32], max_tokens: usize) -> Sequence {
    Sequence::new(id, prompt.to_vec(), greedy(max_tokens), BLOCK, None)
}

/// Steps the scheduler until idle and returns every sequence's generated tokens.
fn run_to_completion(s: &ContinuousBatchScheduler) -> HashMap<u64, Vec<u32>> {
    let mut out: HashMap<u64, Vec<u32>> = HashMap::new();
    for _ in 0..1000 {
        for (id, tok) in s.step().unwrap() {
            out.entry(id).or_default().push(tok);
        }
        if s.num_running() == 0 && s.num_waiting() == 0 {
            return out;
        }
    }
    panic!("scheduler did not finish within 1000 steps");
}

/// Greedy continuation computed by re-running a full prefill for every new token.
fn reference_greedy(m: &LlamaModel<CpuBackend>, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    let c = &m.config;
    let mut tokens = prompt.to_vec();
    let mut out = Vec::new();
    while out.len() < max_tokens && tokens.len() < c.max_position_embeddings {
        let n = tokens.len();
        let blocks = n.div_ceil(BLOCK);
        let shape = Shape::new(vec![blocks, c.num_key_value_heads, BLOCK, c.head_dim()]);
        let mut k: Vec<Tensor> = (0..c.num_hidden_layers)
            .map(|_| Tensor::zeros(shape.clone(), DType::F32, Device::Cpu).unwrap())
            .collect();
        let mut v = k.clone();
        let table = BlockTable {
            block_size: BLOCK,
            block_ids: (0..blocks).collect(),
        };
        let positions: Vec<usize> = (0..n).collect();
        let logits = m
            .forward(
                &ForwardBatch {
                    tokens: &tokens,
                    positions: &positions,
                    block_tables: &vec![table; n],
                    logits_rows: &[n - 1],
                },
                &mut k,
                &mut v,
            )
            .unwrap();
        let row: Vec<f32> = logits.to_vec().unwrap();
        let next = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        out.push(next);
        tokens.push(next);
    }
    out
}

const PROMPT_A: [u32; 13] = [3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9, 32, 38];

#[test]
fn batched_generation_matches_full_recompute() {
    let m = model();
    let s = scheduler(64, 8);
    let prompts: Vec<Vec<u32>> = vec![PROMPT_A.to_vec(), vec![1, 2], vec![60, 50, 40, 30, 20, 10, 0]];
    for (i, p) in prompts.iter().enumerate() {
        s.submit_sequence(seq(i as u64, p, 6)).unwrap();
    }
    let out = run_to_completion(&s);
    for (i, p) in prompts.iter().enumerate() {
        assert_eq!(out[&(i as u64)], reference_greedy(&m, p, 6), "sequence {i}");
    }
}

#[test]
fn prefix_cache_hit_reuses_blocks_and_preserves_output() {
    let m = model();
    let total_blocks = 64;
    let s = scheduler(total_blocks, 4);
    let expected = reference_greedy(&m, &PROMPT_A, 3);

    // Cold run: 13 prompt + 2 fed-back output tokens computed -> 3 full blocks cached.
    s.submit_sequence(seq(1, &PROMPT_A, 3)).unwrap();
    assert_eq!(run_to_completion(&s)[&1], expected);
    assert_eq!(s.num_cached_blocks(), 3);
    assert_eq!(s.num_free_blocks(), total_blocks - 3);

    // Warm run with the same prompt: 12 tokens (3 blocks) come from the cache, so only
    // one fresh block is allocated for the remaining prompt token.
    s.submit_sequence(seq(2, &PROMPT_A, 3)).unwrap();
    let first = s.step().unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(s.num_free_blocks(), total_blocks - 3 - 1);
    let mut warm = vec![first[0].1];
    warm.extend(run_to_completion(&s).remove(&2).unwrap_or_default());
    assert_eq!(warm, expected, "warm output must equal cold output");

    // A prompt that shares only the first two blocks then diverges.
    let mut diverging = PROMPT_A[..8].to_vec();
    diverging.extend_from_slice(&[1, 1, 1]);
    s.submit_sequence(seq(3, &diverging, 4)).unwrap();
    assert_eq!(run_to_completion(&s)[&3], reference_greedy(&m, &diverging, 4));

    // At idle every block is either free or owned by the cache.
    assert_eq!(s.num_free_blocks() + s.num_cached_blocks(), total_blocks);
}

#[test]
fn running_sequences_share_their_prefix_with_later_requests() {
    // A second request with the same prompt, submitted while the first is still
    // generating, must reuse the first one's prompt blocks.
    let m = model();
    let total_blocks = 64;
    let s = scheduler(total_blocks, 4);
    let expected = reference_greedy(&m, &PROMPT_A, 8);

    s.submit_sequence(seq(1, &PROMPT_A, 8)).unwrap();
    let mut out: HashMap<u64, Vec<u32>> = HashMap::new();
    for (id, tok) in s.step().unwrap() {
        out.entry(id).or_default().push(tok);
    }
    // After the prefill step the 3 full prompt blocks are already cached.
    assert_eq!(s.num_cached_blocks(), 3);
    assert_eq!(s.num_running(), 1);

    let free_before = s.num_free_blocks();
    s.submit_sequence(seq(2, &PROMPT_A, 8)).unwrap();
    for (id, tok) in s.step().unwrap() {
        out.entry(id).or_default().push(tok);
    }
    // Sequence 2 needed one fresh block (for its last prompt token); sequence 1 may
    // have grown by at most one block in the same step.
    assert!(free_before - s.num_free_blocks() <= 2, "prefix blocks were not shared");
    for (id, toks) in run_to_completion(&s) {
        out.entry(id).or_default().extend(toks);
    }
    assert_eq!(out[&1], expected);
    assert_eq!(out[&2], expected);
    assert_eq!(s.num_free_blocks() + s.num_cached_blocks(), total_blocks);
}

#[test]
fn cached_blocks_survive_pool_pressure_without_corruption() {
    // Regression test: the cache used to keep pointing at blocks that had been returned
    // to the free list, so later sequences overwrote cached KV entries.
    let m = model();
    let total_blocks = 8;
    let s = scheduler(total_blocks, 1);
    let expected_a = reference_greedy(&m, &PROMPT_A, 3);

    s.submit_sequence(seq(1, &PROMPT_A, 3)).unwrap();
    assert_eq!(run_to_completion(&s)[&1], expected_a);

    // Unrelated sequences that need most of the pool, forcing evictions.
    for i in 0..4u64 {
        let p: Vec<u32> = (0..18).map(|t| ((t * 7 + i as usize * 11) % 64) as u32).collect();
        s.submit_sequence(seq(10 + i, &p, 5)).unwrap();
        assert_eq!(run_to_completion(&s)[&(10 + i)], reference_greedy(&m, &p, 5));
        assert_eq!(s.num_free_blocks() + s.num_cached_blocks(), total_blocks);
    }

    // Re-running A must still produce the reference output, whether or not its
    // blocks survived in the cache.
    s.submit_sequence(seq(2, &PROMPT_A, 3)).unwrap();
    assert_eq!(run_to_completion(&s)[&2], expected_a);
}

#[test]
fn preemption_under_kv_pressure_preserves_outputs() {
    let m = model();
    // 6 blocks x 4 tokens = 24 slots, but 3 sequences x 12 tokens = 36 are needed.
    let s = scheduler(6, 3);
    let prompts: Vec<Vec<u32>> = (0..3u32)
        .map(|i| (0..6).map(|t| (t * 5 + i * 13) % 64).collect())
        .collect();
    for (i, p) in prompts.iter().enumerate() {
        s.submit_sequence(seq(i as u64, p, 6)).unwrap();
    }
    let out = run_to_completion(&s);
    assert!(s.num_preemptions() > 0, "test must actually exercise preemption");
    for (i, p) in prompts.iter().enumerate() {
        let got = &out[&(i as u64)];
        // A preempted sequence re-emits nothing for tokens it already produced; the
        // recorded stream must still be exactly the greedy continuation.
        assert_eq!(got, &reference_greedy(&m, p, 6), "sequence {i}");
    }
    assert_eq!(s.num_free_blocks() + s.num_cached_blocks(), 6);
}

#[test]
fn submit_rejects_invalid_sequences() {
    let s = scheduler(4, 2);
    assert!(s.submit_sequence(seq(1, &[], 4)).is_err(), "empty prompt");
    assert!(s.submit_sequence(seq(2, &[64], 4)).is_err(), "token outside vocabulary");
    let too_long: Vec<u32> = vec![1; 128];
    assert!(
        s.submit_sequence(seq(3, &too_long, 4)).is_err(),
        "prompt >= context window"
    );
    let too_big_for_pool: Vec<u32> = vec![1; 16]; // needs 5 blocks of 4, pool has 4
    assert!(
        s.submit_sequence(seq(4, &too_big_for_pool, 4)).is_err(),
        "prompt > KV pool"
    );
    assert_eq!(s.num_waiting(), 0);
}

#[test]
fn generation_stops_at_context_window() {
    let s = scheduler(64, 1);
    let prompt: Vec<u32> = (0..120).map(|t| (t % 64) as u32).collect();
    s.submit_sequence(seq(1, &prompt, 50)).unwrap();
    let out = run_to_completion(&s);
    assert_eq!(
        out[&1].len(),
        128 - 120,
        "stops when the sequence fills max_position_embeddings"
    );
}

#[test]
fn generate_returns_when_its_own_sequence_finishes() {
    let m = model();
    let s = scheduler(64, 4);
    // A long-running sequence shares the scheduler with the one we wait for.
    s.submit_sequence(seq(99, &[1, 2, 3], 40)).unwrap();
    let mut streamed = Vec::new();
    let out = s.generate(seq(1, &PROMPT_A, 5), |t| streamed.push(t)).unwrap();
    assert_eq!(out, reference_greedy(&m, &PROMPT_A, 5));
    assert_eq!(streamed, out);
    assert_eq!(s.num_running(), 1, "the other sequence is still running");
}

// ───────────────────────────── engine thread ─────────────────────────────

use rllm_scheduler::{EngineConfig, EngineHandle, FinishReason, SequenceEvent, SubmitError};

fn collect(stream: &mut rllm_scheduler::GenerationStream) -> (Vec<u32>, SequenceEvent) {
    let mut toks = Vec::new();
    loop {
        match stream.blocking_next().expect("stream ended without a terminal event") {
            SequenceEvent::Token(t) => toks.push(t),
            other => return (toks, other),
        }
    }
}

#[test]
fn engine_serves_concurrent_requests_from_many_threads() {
    let m = model();
    let engine = EngineHandle::start(Arc::new(scheduler(128, 8)), EngineConfig::default());
    let prompts: Vec<Vec<u32>> = (0..6u32)
        .map(|i| (0..(3 + i as usize * 2)).map(|t| (t as u32 * 7 + i * 5) % 64).collect())
        .collect();
    let handles: Vec<_> = prompts
        .iter()
        .cloned()
        .map(|p| {
            let engine = engine.clone();
            std::thread::spawn(move || {
                let mut s = engine.submit(p, greedy(6)).unwrap();
                collect(&mut s)
            })
        })
        .collect();
    for (p, h) in prompts.iter().zip(handles) {
        let (toks, end) = h.join().unwrap();
        assert_eq!(end, SequenceEvent::Finished(FinishReason::Length));
        assert_eq!(toks, reference_greedy(&m, p, 6));
    }
    let metrics = &engine.scheduler().metrics;
    assert_eq!(metrics.finished_length.load(std::sync::atomic::Ordering::Relaxed), 6);
    assert_eq!(metrics.generated_tokens.load(std::sync::atomic::Ordering::Relaxed), 36);
    engine.shutdown();
}

#[test]
fn dropping_the_stream_cancels_and_frees_blocks() {
    let total = 64;
    let engine = EngineHandle::start(Arc::new(scheduler(total, 4)), EngineConfig::default());
    let mut s = engine.submit(PROMPT_A.to_vec(), greedy(100)).unwrap();
    // Receive a couple of tokens, then disconnect.
    for _ in 0..2 {
        assert!(matches!(s.blocking_next(), Some(SequenceEvent::Token(_))));
    }
    drop(s);
    let sched = engine.scheduler().clone();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while sched.num_running() > 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(sched.num_running(), 0, "cancelled sequence must be retired");
    assert_eq!(
        sched.metrics.finished_abort.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(sched.num_free_blocks() + sched.num_cached_blocks(), total);
    engine.shutdown();
}

#[test]
fn engine_rejects_invalid_and_overload() {
    let engine = EngineHandle::start(Arc::new(scheduler(64, 1)), EngineConfig { max_waiting: 0 });
    assert!(matches!(
        engine.submit(vec![1, 2], greedy(4)),
        Err(SubmitError::Overloaded { .. })
    ));
    engine.shutdown();
    assert_eq!(engine.submit(vec![1], greedy(1)).err(), Some(SubmitError::ShutDown));

    let engine = EngineHandle::start(Arc::new(scheduler(64, 1)), EngineConfig::default());
    assert!(matches!(engine.submit(vec![], greedy(4)), Err(SubmitError::Invalid(_))));
    assert!(matches!(
        engine.submit(vec![1], greedy(0)),
        Err(SubmitError::Invalid(_))
    ));
    engine.shutdown();
}

#[test]
fn oversized_sequence_gets_a_terminal_event_instead_of_looping() {
    // 4 blocks x 4 tokens = 16 slots; a 10-token prompt asking for 50 tokens outgrows it.
    let engine = EngineHandle::start(Arc::new(scheduler(4, 1)), EngineConfig::default());
    let mut s = engine.submit((0..10).collect(), greedy(50)).unwrap();
    let (toks, end) = collect(&mut s);
    assert_eq!(end, SequenceEvent::Finished(FinishReason::Length));
    assert!(toks.len() < 50);
    engine.shutdown();
}

#[test]
fn f16_kv_cache_tracks_f32_and_keeps_greedy_output() {
    // F16 keys/values (the default for loaded models) change logits only slightly: the
    // scheduler's greedy output must equal the F32-cache reference on this model.
    let mut m = LlamaModel::new_seeded(config(), Arc::new(CpuBackend), 7).unwrap();
    m.kv_dtype = DType::F16;
    let s = ContinuousBatchScheduler::new(
        SchedulerConfig {
            max_batch_size: 4,
            block_size: BLOCK,
            total_blocks: 64,
            ..SchedulerConfig::default()
        },
        Arc::new(m),
    )
    .unwrap();
    let reference = model();
    let prompts: Vec<Vec<u32>> = vec![PROMPT_A.to_vec(), vec![60, 50, 40, 30, 20, 10, 0]];
    for (i, p) in prompts.iter().enumerate() {
        s.submit_sequence(seq(i as u64, p, 8)).unwrap();
    }
    let out = run_to_completion(&s);
    for (i, p) in prompts.iter().enumerate() {
        assert_eq!(out[&(i as u64)], reference_greedy(&reference, p, 8), "sequence {i}");
    }
}

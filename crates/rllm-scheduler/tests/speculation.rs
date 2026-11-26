//! Prompt-lookup speculation against an exact stub model.
//!
//! The stub keeps a paged *token* store in place of a KV cache: a forward pass writes
//! every row's token at its (block, offset) slot, then computes each requested row's
//! next-token distribution from the tokens it reads back through the row's block
//! table. Any bookkeeping error — a stale rejected draft, a wrong position, a block
//! shared or freed too early — changes the context the stub reads and so the output.
//!
//! Most tests verify every lookup match at full length (`adaptive: false`) so they
//! don't depend on timing; one runs the adaptive cost model, with each pass sleeping a
//! fixed time like a memory-bound decode whose cost barely depends on the row count.

use parking_lot::Mutex;
use rllm_core::error::Result;
use rllm_models::{ForwardBatch, LanguageModel, LlamaConfig};
use rllm_scheduler::{ContinuousBatchScheduler, SamplingParams, SchedulerConfig, Sequence, SpeculationConfig};
use std::sync::atomic::Ordering;
use std::sync::Arc;

const VOCAB: usize = 64;
const BLOCK: usize = 4;

/// Next token "intended" by the stub: a copy of the token 6 positions back, except
/// at every 11th position, which gets a new token. Earlier n-grams therefore predict
/// the continuation most of the time, but not always.
fn intended(ctx: &[u32]) -> u32 {
    let n = ctx.len();
    if n.is_multiple_of(11) || n < 6 {
        return ((n * 13 + 5) % VOCAB) as u32;
    }
    ctx[n - 6]
}

struct StubModel {
    config: LlamaConfig,
    /// Logit of the intended token (all others are 0).
    peak: f32,
    /// Fixed time each pass sleeps.
    pass_us: u64,
    store: Mutex<Vec<Option<u32>>>,
}

impl StubModel {
    fn new(peak: f32) -> Arc<Self> {
        Self::with_pass_time(peak, 0)
    }

    fn with_pass_time(peak: f32, pass_us: u64) -> Arc<Self> {
        Arc::new(Self {
            config: LlamaConfig {
                hidden_size: 8,
                intermediate_size: 8,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                num_hidden_layers: 1,
                vocab_size: VOCAB,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_position_embeddings: 512,
            },
            peak,
            pass_us,
            store: Mutex::new(Vec::new()),
        })
    }
}

impl LanguageModel for StubModel {
    fn config(&self) -> &LlamaConfig {
        &self.config
    }

    fn device_name(&self) -> String {
        "stub".into()
    }

    fn init_kv_cache(&self, num_blocks: usize, block_size: usize) -> Result<()> {
        assert_eq!(block_size, BLOCK);
        *self.store.lock() = vec![None; num_blocks * block_size];
        Ok(())
    }

    fn forward_cached(&self, batch: &ForwardBatch<'_>) -> Result<Vec<f32>> {
        std::thread::sleep(std::time::Duration::from_micros(self.pass_us));
        let mut store = self.store.lock();
        let slot =
            |table: &rllm_core::backend::BlockTable, pos: usize| table.block_ids[pos / BLOCK] * BLOCK + pos % BLOCK;
        for (r, &tok) in batch.tokens.iter().enumerate() {
            store[slot(&batch.block_tables[r], batch.positions[r])] = Some(tok);
        }
        let mut out = vec![0.0; batch.logits_rows.len() * VOCAB];
        for (i, &r) in batch.logits_rows.iter().enumerate() {
            let table = &batch.block_tables[r];
            let ctx: Vec<u32> = (0..=batch.positions[r])
                .map(|p| store[slot(table, p)].expect("row reads a slot that was never written"))
                .collect();
            out[i * VOCAB + intended(&ctx) as usize] = self.peak;
        }
        Ok(out)
    }
}

fn run(
    model: Arc<StubModel>,
    total_blocks: usize,
    max_batch_size: usize,
    speculation: SpeculationConfig,
    prompts: &[Vec<u32>],
    params: &SamplingParams,
) -> (Vec<Vec<u32>>, Arc<ContinuousBatchScheduler>) {
    let sched = Arc::new(
        ContinuousBatchScheduler::new(
            SchedulerConfig {
                max_batch_size,
                block_size: BLOCK,
                total_blocks,
                max_prefill_tokens: 64,
                speculation,
            },
            model,
        )
        .unwrap(),
    );
    let mut rxs = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit_sequence(Sequence::new(i as u64 + 1, p.clone(), params.clone(), BLOCK, Some(tx)))
            .unwrap();
        rxs.push(rx);
    }
    while sched.has_work() {
        sched.step().unwrap();
    }
    let outs = rxs
        .into_iter()
        .map(|mut rx| {
            let mut toks = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                if let rllm_scheduler::SequenceEvent::Token(t) = ev {
                    toks.push(t);
                }
            }
            toks
        })
        .collect();
    (outs, sched)
}

fn prompts() -> Vec<Vec<u32>> {
    let shared: Vec<u32> = (0..10).map(|i| (i * 7 % VOCAB) as u32).collect();
    let mut a = shared.clone();
    a.extend([3, 9, 27]);
    let mut b = shared;
    b.extend([3, 9, 28, 1]);
    vec![a, b, vec![5, 6], vec![40, 41, 42, 43, 44, 45, 46, 47, 48]]
}

/// Speculation that verifies every match at full length (no timing dependence).
fn fixed(max_draft_tokens: usize) -> SpeculationConfig {
    SpeculationConfig {
        adaptive: false,
        ..SpeculationConfig::prompt_lookup(max_draft_tokens)
    }
}

fn greedy(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        stop_tokens: Vec::new(),
        ..SamplingParams::default()
    }
}

#[test]
fn greedy_speculation_matches_plain_decoding() {
    let params = greedy(80);
    let (plain, plain_sched) = run(StubModel::new(10.0), 256, 4, fixed(0), &prompts(), &params);
    for draft in [1, 4, 15] {
        let (spec, sched) = run(StubModel::new(10.0), 256, 4, fixed(draft), &prompts(), &params);
        assert_eq!(spec, plain, "draft {draft}");
        let m = &sched.metrics;
        let accepted = m.accepted_draft_tokens.load(Ordering::Relaxed);
        assert!(
            accepted > 0 && accepted < m.draft_tokens.load(Ordering::Relaxed),
            "draft {draft}"
        );
        assert!(
            m.forward_passes.load(Ordering::Relaxed) < plain_sched.metrics.forward_passes.load(Ordering::Relaxed),
            "draft {draft}"
        );
        assert_eq!(m.generated_tokens.load(Ordering::Relaxed), 4 * 80);
        // Accepted drafts keep their KV entries: no token is computed twice.
        let prompt_tokens: usize = prompts().iter().map(Vec::len).sum();
        assert_eq!(
            m.prefill_tokens.load(Ordering::Relaxed) + m.cached_prompt_tokens.load(Ordering::Relaxed),
            prompt_tokens as u64,
            "draft {draft}"
        );
    }
    assert_eq!(plain_sched.metrics.draft_tokens.load(Ordering::Relaxed), 0);
}

#[test]
fn adaptive_draft_lengths_keep_greedy_output() {
    let params = greedy(80);
    let (plain, _) = run(StubModel::new(10.0), 256, 4, fixed(0), &prompts(), &params);
    let model = StubModel::with_pass_time(10.0, 300);
    let (spec, sched) = run(model, 256, 4, SpeculationConfig::prompt_lookup(15), &prompts(), &params);
    assert_eq!(spec, plain);
    // Drafting is cheap for this stub; the cost model should use it (timing-dependent,
    // so only a weak check).
    assert!(sched.metrics.draft_tokens.load(Ordering::Relaxed) > 0);
}

#[test]
fn greedy_speculation_survives_preemption_and_prefix_sharing() {
    // A pool too small for all four sequences: drafts are dropped and sequences
    // preempted and recomputed (reusing prefix-cached blocks).
    let params = greedy(40);
    let (plain, _) = run(StubModel::new(10.0), 256, 4, fixed(0), &prompts(), &params);
    let (spec, sched) = run(StubModel::new(10.0), 30, 4, fixed(8), &prompts(), &params);
    assert_eq!(spec, plain);
    assert!(sched.num_preemptions() > 0);
    assert!(sched.metrics.accepted_draft_tokens.load(Ordering::Relaxed) > 0);
}

#[test]
fn speculation_respects_max_tokens_and_stop_tokens() {
    let mut params = greedy(23);
    let (plain, _) = run(StubModel::new(10.0), 256, 4, fixed(0), &prompts(), &params);
    let (spec, _) = run(StubModel::new(10.0), 256, 4, fixed(15), &prompts(), &params);
    assert_eq!(spec, plain);
    assert!(spec.iter().all(|o| o.len() == 23));

    // Stop on a token that appears mid-stream in the plain output.
    params.max_tokens = 80;
    let stop = plain[0][17];
    params.stop_tokens = vec![stop];
    let (plain, _) = run(StubModel::new(10.0), 256, 4, fixed(0), &prompts(), &params);
    let (spec, _) = run(StubModel::new(10.0), 256, 4, fixed(15), &prompts(), &params);
    assert_eq!(spec, plain);
    assert!(spec[0].len() <= 18 && spec[0].last() == Some(&stop));
}

#[test]
fn sampling_distribution_is_unchanged() {
    // The intended token gets probability e^L / (e^L + 63) = 0.7.
    let peak = (0.7f32 * 63.0 / 0.3).ln();
    let params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        max_tokens: 200,
        stop_tokens: Vec::new(),
    };
    let prompts: Vec<Vec<u32>> = (0..16).map(|i| vec![i as u32, (i * 3 % 64) as u32, 7]).collect();
    let rate = |draft: usize| {
        let (outs, sched) = run(StubModel::new(peak), 2048, 16, fixed(draft), &prompts, &params);
        let (mut hit, mut n) = (0usize, 0usize);
        for (p, o) in prompts.iter().zip(&outs) {
            let mut ctx = p.clone();
            for &t in o {
                hit += usize::from(t == intended(&ctx));
                n += 1;
                ctx.push(t);
            }
        }
        (
            hit as f64 / n as f64,
            sched.metrics.accepted_draft_tokens.load(Ordering::Relaxed),
        )
    };
    // 3200 samples: standard deviation 0.008.
    let (plain, _) = rate(0);
    let (spec, accepted) = rate(8);
    assert!((plain - 0.7).abs() < 0.04, "plain {plain}");
    assert!((spec - 0.7).abs() < 0.04, "speculative {spec}");
    assert!(accepted > 100, "{accepted}");
}

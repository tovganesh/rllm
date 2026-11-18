//! Numerical reference tests for `LlamaModel`.
//!
//! The engine's paged, batched forward pass is compared against a deliberately naive
//! reference implementation written with plain loops: dense causal attention over the
//! whole sequence, no KV cache, no paging, no backend kernels. Weights are seeded
//! pseudo-random values, so layout, indexing, RoPE-position or masking mistakes change
//! the logits and fail these tests.

use rllm_backend_cpu::CpuBackend;
use rllm_core::{backend::BlockTable, device::Device, dtype::DType, shape::Shape, tensor::Tensor};
use rllm_models::{ForwardBatch, LlamaConfig, LlamaModel};
use std::sync::Arc;

const TOL: f32 = 2e-4;

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

fn model() -> LlamaModel<CpuBackend> {
    LlamaModel::new_seeded(config(), Arc::new(CpuBackend), 42).unwrap()
}

// ───────────────────────────── naive reference ─────────────────────────────

fn f32s(t: &Tensor) -> Vec<f32> {
    t.to_vec().unwrap()
}

/// `y = W x` for `W` stored `[out, in]`.
fn linear(w: &[f32], x: &[f32], out: usize) -> Vec<f32> {
    let inp = x.len();
    (0..out)
        .map(|j| (0..inp).map(|i| w[j * inp + i] * x[i]).sum())
        .collect()
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

/// Interleaved-pair RoPE with angles computed independently of the engine's tables.
fn rope(x: &mut [f32], head_dim: usize, pos: usize, theta: f32) {
    for head in x.chunks_mut(head_dim) {
        for i in 0..head_dim / 2 {
            let freq = 1.0f64 / (theta as f64).powf(2.0 * i as f64 / head_dim as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            let (s, c) = (s as f32, c as f32);
            let (a, b) = (head[2 * i], head[2 * i + 1]);
            head[2 * i] = a * c - b * s;
            head[2 * i + 1] = a * s + b * c;
        }
    }
}

/// Logits for every position of `tokens`, computed without any caching.
fn reference_logits(m: &LlamaModel<CpuBackend>, tokens: &[u32]) -> Vec<Vec<f32>> {
    let c = &m.config;
    let (h, hd) = (c.hidden_size, c.head_dim());
    let (nq, nkv) = (c.num_attention_heads, c.num_key_value_heads);
    let kv_dim = nkv * hd;
    let embed = f32s(&m.embed_tokens);

    let mut x: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| embed[t as usize * h..(t as usize + 1) * h].to_vec())
        .collect();

    for layer in &m.layers {
        let (wq, wk, wv, wo) = (
            f32s(&layer.self_attn.q_proj),
            f32s(&layer.self_attn.k_proj),
            f32s(&layer.self_attn.v_proj),
            f32s(&layer.self_attn.o_proj),
        );
        let (ln1, ln2) = (f32s(&layer.input_layernorm), f32s(&layer.post_attention_layernorm));
        let (wg, wu, wd) = (
            f32s(&layer.mlp.gate_proj),
            f32s(&layer.mlp.up_proj),
            f32s(&layer.mlp.down_proj),
        );

        let mut qs = Vec::new();
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for (t, xt) in x.iter().enumerate() {
            let xn = rms_norm(xt, &ln1, c.rms_norm_eps);
            let mut q = linear(&wq, &xn, h);
            let mut k = linear(&wk, &xn, kv_dim);
            rope(&mut q, hd, t, c.rope_theta);
            rope(&mut k, hd, t, c.rope_theta);
            qs.push(q);
            ks.push(k);
            vs.push(linear(&wv, &xn, kv_dim));
        }

        for t in 0..x.len() {
            let mut attn = vec![0.0f32; h];
            for head in 0..nq {
                let kvh = head / (nq / nkv);
                let q = &qs[t][head * hd..(head + 1) * hd];
                let scores: Vec<f32> = (0..=t)
                    .map(|s| {
                        let k = &ks[s][kvh * hd..(kvh + 1) * hd];
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (s, e) in exps.iter().enumerate() {
                    let v = &vs[s][kvh * hd..(kvh + 1) * hd];
                    for d in 0..hd {
                        attn[head * hd + d] += e / sum * v[d];
                    }
                }
            }
            let proj = linear(&wo, &attn, h);
            for d in 0..h {
                x[t][d] += proj[d];
            }

            let xn = rms_norm(&x[t], &ln2, c.rms_norm_eps);
            let g = linear(&wg, &xn, c.intermediate_size);
            let u = linear(&wu, &xn, c.intermediate_size);
            let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let ffn = linear(&wd, &a, h);
            for d in 0..h {
                x[t][d] += ffn[d];
            }
        }
    }

    let norm = f32s(&m.norm);
    let lm = f32s(&m.lm_head);
    x.iter()
        .map(|xt| linear(&lm, &rms_norm(xt, &norm, c.rms_norm_eps), c.vocab_size))
        .collect()
}

// ───────────────────────────── engine helpers ─────────────────────────────

struct Cache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

fn cache(m: &LlamaModel<CpuBackend>, num_blocks: usize, block_size: usize) -> Cache {
    let shape = Shape::new(vec![
        num_blocks,
        m.config.num_key_value_heads,
        block_size,
        m.config.head_dim(),
    ]);
    let mk = || {
        (0..m.config.num_hidden_layers)
            .map(|_| Tensor::zeros(shape.clone(), DType::F32, Device::Cpu).unwrap())
            .collect::<Vec<_>>()
    };
    Cache { k: mk(), v: mk() }
}

fn rows(t: &Tensor) -> Vec<Vec<f32>> {
    let vocab = *t.dims().last().unwrap();
    f32s(t).chunks(vocab).map(|r| r.to_vec()).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    let max_err = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < TOL, "{what}: max abs error {max_err} exceeds {TOL}");
}

const TOKENS: [u32; 11] = [5, 17, 3, 42, 8, 8, 61, 0, 29, 13, 50];

// ───────────────────────────────── tests ─────────────────────────────────

#[test]
fn prefill_matches_reference_at_every_position() {
    let m = model();
    let expected = reference_logits(&m, &TOKENS);

    let block_size = 4;
    // Deliberately scrambled, non-contiguous physical blocks.
    let table = BlockTable { block_size, block_ids: vec![6, 1, 4] };
    let mut c = cache(&m, 8, block_size);
    let positions: Vec<usize> = (0..TOKENS.len()).collect();
    let tables = vec![table; TOKENS.len()];
    let all_rows: Vec<usize> = (0..TOKENS.len()).collect();

    let logits = m
        .forward(
            &ForwardBatch {
                tokens: &TOKENS,
                positions: &positions,
                block_tables: &tables,
                logits_rows: &all_rows,
            },
            &mut c.k,
            &mut c.v,
        )
        .unwrap();

    for (pos, (got, want)) in rows(&logits).iter().zip(&expected).enumerate() {
        assert_close(got, want, &format!("prefill position {pos}"));
    }
}

#[test]
fn prefill_then_decode_matches_reference() {
    let m = model();
    let expected = reference_logits(&m, &TOKENS);
    let block_size = 4;
    let table = BlockTable { block_size, block_ids: vec![2, 7, 0] };
    let mut c = cache(&m, 8, block_size);

    // Prefill the first 5 tokens, asking only for the last row's logits.
    let prompt = 5;
    let positions: Vec<usize> = (0..prompt).collect();
    let logits = m
        .forward(
            &ForwardBatch {
                tokens: &TOKENS[..prompt],
                positions: &positions,
                block_tables: &vec![table.clone(); prompt],
                logits_rows: &[prompt - 1],
            },
            &mut c.k,
            &mut c.v,
        )
        .unwrap();
    assert_eq!(logits.dims(), &[1, m.config.vocab_size]);
    assert_close(&rows(&logits)[0], &expected[prompt - 1], "prefill last row");

    // Decode the remaining tokens one at a time through the KV cache.
    for pos in prompt..TOKENS.len() {
        let logits = m
            .forward_decode(&[TOKENS[pos]], &mut c.k, &mut c.v, std::slice::from_ref(&table), &[pos + 1])
            .unwrap();
        assert_close(&rows(&logits)[0], &expected[pos], &format!("decode position {pos}"));
    }
}

#[test]
fn mixed_batch_of_two_sequences_matches_reference() {
    let m = model();
    let seq_a: &[u32] = &TOKENS;
    let seq_b: &[u32] = &[9, 33, 1, 60, 21, 7];
    let exp_a = reference_logits(&m, seq_a);
    let exp_b = reference_logits(&m, seq_b);

    let block_size = 4;
    let table_a = BlockTable { block_size, block_ids: vec![5, 0, 9] };
    let table_b = BlockTable { block_size, block_ids: vec![3, 8] };
    let mut c = cache(&m, 10, block_size);

    // Step 1: prefill A[..7] and B[..4] together in one call.
    let (pa, pb) = (7, 4);
    let mut tokens = seq_a[..pa].to_vec();
    tokens.extend_from_slice(&seq_b[..pb]);
    let mut positions: Vec<usize> = (0..pa).collect();
    positions.extend(0..pb);
    let mut tables = vec![table_a.clone(); pa];
    tables.extend(vec![table_b.clone(); pb]);
    let logits = m
        .forward(
            &ForwardBatch {
                tokens: &tokens,
                positions: &positions,
                block_tables: &tables,
                logits_rows: &[pa - 1, pa + pb - 1],
            },
            &mut c.k,
            &mut c.v,
        )
        .unwrap();
    let r = rows(&logits);
    assert_close(&r[0], &exp_a[pa - 1], "batched prefill A");
    assert_close(&r[1], &exp_b[pb - 1], "batched prefill B");

    // Step 2: decode both sequences together (different positions per sequence).
    for step in 0..2 {
        let (ia, ib) = (pa + step, pb + step);
        let logits = m
            .forward_decode(
                &[seq_a[ia], seq_b[ib]],
                &mut c.k,
                &mut c.v,
                &[table_a.clone(), table_b.clone()],
                &[ia + 1, ib + 1],
            )
            .unwrap();
        let r = rows(&logits);
        assert_close(&r[0], &exp_a[ia], &format!("batched decode A pos {ia}"));
        assert_close(&r[1], &exp_b[ib], &format!("batched decode B pos {ib}"));
    }
}

#[test]
fn forward_rejects_out_of_range_positions_instead_of_panicking() {
    let m = model();
    let mut c = cache(&m, 2, 4);
    let table = BlockTable { block_size: 4, block_ids: vec![0, 1] };

    // Position beyond the model's context window.
    let err = m.forward_decode(
        &[1],
        &mut c.k,
        &mut c.v,
        std::slice::from_ref(&table),
        &[m.config.max_position_embeddings + 1],
    );
    assert!(err.is_err());

    // Position not covered by the sequence's blocks.
    let err = m.forward_decode(&[1], &mut c.k, &mut c.v, &[table], &[9]);
    assert!(err.is_err());

    // Token outside the vocabulary.
    let table = BlockTable { block_size: 4, block_ids: vec![0] };
    let err = m.forward_decode(&[m.config.vocab_size as u32], &mut c.k, &mut c.v, &[table], &[1]);
    assert!(err.is_err());
}

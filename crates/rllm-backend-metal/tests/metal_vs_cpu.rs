//! Metal executor checked against the CPU executor on seeded synthetic models.
#![cfg(target_os = "macos")]

use rllm_backend_cpu::{
    ops::quant::{quantize_f32_to_q4_0, quantize_f32_to_q4_k, quantize_f32_to_q6_k, quantize_f32_to_q8_0},
    CpuBackend,
};
use rllm_backend_metal::MetalLlama;
use rllm_core::{backend::BlockTable, device::Device, dtype::DType, storage::{CpuStorage, Storage}, tensor::Tensor};
use rllm_models::{ForwardBatch, LanguageModel, LlamaConfig, LlamaModel};
use std::sync::Arc;

fn config() -> LlamaConfig {
    LlamaConfig {
        hidden_size: 64,
        intermediate_size: 128,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_hidden_layers: 2,
        vocab_size: 96,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 256,
    }
}

fn quantize(t: &Tensor, dtype: DType) -> Tensor {
    let data = t.to_vec::<f32>().unwrap();
    let mut bytes = vec![0u8; dtype.storage_bytes(data.len())];
    match dtype {
        DType::Q8_0 => quantize_f32_to_q8_0(&data, &mut bytes),
        DType::Q4_0 => {
            for (src, dst) in data.chunks(32).zip(bytes.chunks_mut(18)) {
                quantize_f32_to_q4_0(src, dst);
            }
        }
        DType::Q4_K => bytes = quantize_f32_to_q4_k(&data),
        DType::Q6_K => bytes = quantize_f32_to_q6_k(&data),
        _ => unreachable!(),
    }
    Tensor::new(t.shape().clone(), dtype, Device::Cpu, Storage::Cpu(CpuStorage::from_bytes(bytes)))
}

/// 256-wide config for K-quants (whole 256-value super-blocks per row).
fn config_k() -> LlamaConfig {
    LlamaConfig { hidden_size: 256, intermediate_size: 512, ..config() }
}

fn model(dtype: DType) -> LlamaModel<CpuBackend> {
    if dtype == DType::Q4_K {
        return model_k_mixed();
    }
    let mut m = LlamaModel::new_seeded(config(), Arc::new(CpuBackend), 11).unwrap();
    if dtype != DType::F32 {
        m.embed_tokens = quantize(&m.embed_tokens, dtype);
        m.lm_head = quantize(&m.lm_head, dtype);
        for l in &mut m.layers {
            for w in [&mut l.self_attn.q_proj, &mut l.self_attn.k_proj, &mut l.self_attn.v_proj, &mut l.self_attn.o_proj, &mut l.mlp.gate_proj, &mut l.mlp.up_proj, &mut l.mlp.down_proj] {
                *w = quantize(w, dtype);
            }
        }
        m.fuse_projections().unwrap();
        m.pack_for_backend().unwrap();
    }
    m
}

/// Q4_K_M-like mix: Q4_K q/k/o/gate/up, Q6_K v/down and embeddings, so fused Q/K/V
/// has mixed formats (two segments on Metal).
fn model_k_mixed() -> LlamaModel<CpuBackend> {
    let mut m = LlamaModel::new_seeded(config_k(), Arc::new(CpuBackend), 11).unwrap();
    m.embed_tokens = quantize(&m.embed_tokens, DType::Q6_K);
    m.lm_head = quantize(&m.lm_head, DType::Q6_K);
    for l in &mut m.layers {
        for w in [&mut l.self_attn.q_proj, &mut l.self_attn.k_proj, &mut l.self_attn.o_proj, &mut l.mlp.gate_proj, &mut l.mlp.up_proj] {
            *w = quantize(w, DType::Q4_K);
        }
        for w in [&mut l.self_attn.v_proj, &mut l.mlp.down_proj] {
            *w = quantize(w, DType::Q6_K);
        }
    }
    m.fuse_projections().unwrap();
    m.pack_for_backend().unwrap();
    m
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

/// Prefill 48 rows of two sequences (tiled kernel), then decode single tokens
/// (matrix-vector kernel), comparing logits at every step. The CPU reference always
/// runs batches of 32+ rows (and requests logits for every row) so that it, like Metal,
/// uses F32 activations; single-row CPU decode would quantize activations to Q8.
/// `prefill_tol` is looser: the Metal prefill GEMM multiplies half-precision weight and
/// activation tiles and the prefill attention stages keys/values as half (as llama.cpp's
/// Metal backend does), whereas the CPU reference uses F32.
fn check(dtype: DType, tol: f32, prefill_tol: f32) {
    check_kv(dtype, DType::F32, tol, prefill_tol);
}

/// As [`check`], with both executors using a `kv` KV cache.
fn check_kv(dtype: DType, kv: DType, tol: f32, prefill_tol: f32) {
    let model = |dtype| {
        let mut m = model(dtype);
        m.kv_dtype = kv;
        m
    };
    if !MetalLlama::is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let cpu = model(dtype);
    let gpu = MetalLlama::from_cpu_model(&cpu).unwrap();
    let block = 16;
    cpu.init_kv_cache(12, block).unwrap();
    gpu.init_kv_cache(12, block).unwrap();
    let ta = BlockTable { block_size: block, block_ids: vec![7, 2, 9, 0] };
    let tb = BlockTable { block_size: block, block_ids: vec![5, 11, 3] };

    let (la, lb) = (33usize, 15usize);
    let mut tokens: Vec<u32> = (0..la as u32).map(|i| (i * 7 + 3) % 96).collect();
    tokens.extend((0..lb as u32).map(|i| (i * 13 + 1) % 96));
    let mut positions: Vec<usize> = (0..la).collect();
    positions.extend(0..lb);
    let mut tables = vec![ta.clone(); la];
    tables.extend(vec![tb.clone(); lb]);
    let rows: Vec<usize> = (0..la + lb).collect();
    let batch = ForwardBatch { tokens: &tokens, positions: &positions, block_tables: &tables, logits_rows: &rows };
    let d = max_rel_diff(&gpu.forward_cached(&batch).unwrap(), &cpu.forward_cached(&batch).unwrap());
    assert!(d < prefill_tol, "{dtype:?} prefill: relative diff {d}");

    // Decode: compare Metal against a CPU *prefill* of the full sequence so that both
    // use F32 activations (single-row CPU decode quantizes activations).
    let mut seq_a: Vec<u32> = tokens[..la].to_vec();
    for step in 0..4 {
        let tok = (step * 11 + 5) as u32 % 96;
        seq_a.push(tok);
        let pos = seq_a.len() - 1;
        let g = gpu
            .forward_cached(&ForwardBatch { tokens: &[tok], positions: &[pos], block_tables: std::slice::from_ref(&ta), logits_rows: &[0] })
            .unwrap();
        let fresh = model(dtype);
        fresh.init_kv_cache(12, block).unwrap();
        let p: Vec<usize> = (0..seq_a.len()).collect();
        let all = fresh
            .forward_cached(&ForwardBatch { tokens: &seq_a, positions: &p, block_tables: &vec![ta.clone(); seq_a.len()], logits_rows: &p })
            .unwrap();
        let vocab = cpu.config.vocab_size;
        let c = &all[pos * vocab..(pos + 1) * vocab];
        let d = max_rel_diff(&g, c);
        assert!(d < tol, "{dtype:?} decode step {step}: relative diff {d}");
    }
}

#[test]
fn metal_matches_cpu_f32() {
    // Prefill attention stages keys/values in half precision (as llama.cpp's Metal flash
    // attention does with its F16 KV cache), hence the looser prefill bound.
    check(DType::F32, 1e-4, 5e-4);
}

#[test]
fn metal_matches_cpu_q8_0() {
    check(DType::Q8_0, 1e-4, 1e-3);
}

#[test]
fn metal_matches_cpu_q4_0() {
    check(DType::Q4_0, 1e-4, 1e-3);
}

#[test]
fn metal_matches_cpu_q4_k_q6_k_mixed() {
    check(DType::Q4_K, 1e-4, 1e-3);
}

/// Prefix reuse on Metal: a sequence whose first two KV blocks were written by another
/// sequence computes only its own suffix, and must produce the same logits as a full
/// prefill of its tokens into fresh blocks.
#[test]
fn metal_prefix_block_reuse_matches_full_prefill() {
    if !MetalLlama::is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let gpu = MetalLlama::from_cpu_model(&model(DType::Q8_0)).unwrap();
    let block = 16;
    gpu.init_kv_cache(12, block).unwrap();
    let run = |tokens: &[u32], start: usize, ids: Vec<usize>| -> Vec<f32> {
        let table = BlockTable { block_size: block, block_ids: ids };
        let positions: Vec<usize> = (start..start + tokens.len()).collect();
        let tables = vec![table; tokens.len()];
        gpu.forward_cached(&ForwardBatch { tokens, positions: &positions, block_tables: &tables, logits_rows: &[tokens.len() - 1] })
            .unwrap()
    };
    let a: Vec<u32> = (0..40u32).map(|i| (i * 7 + 3) % 96).collect();
    let mut b: Vec<u32> = a[..32].to_vec();
    b.extend([5u32, 9, 11, 2, 60, 44]);

    // Sequence A fills blocks 3 and 7 with the shared 32-token prefix (and block 1).
    run(&a, 0, vec![3, 7, 1]);
    // Sequence B reuses blocks 3 and 7 and computes only its 6 suffix tokens in block 9.
    let reused = run(&b[32..], 32, vec![3, 7, 9]);
    // Reference: B from scratch in unrelated blocks.
    let fresh = run(&b, 0, vec![0, 2, 4]);
    let d = max_rel_diff(&reused, &fresh);
    assert!(d < 1e-3, "prefix reuse: relative diff {d}");
}

/// F16 KV cache (the default for loaded models) on both executors. Both round keys and
/// values to half precision at the same point, so they still agree closely.
#[test]
fn metal_matches_cpu_q8_0_f16_kv() {
    check_kv(DType::Q8_0, DType::F16, 1e-3, 1e-3);
}

/// Decode at long context, where Metal splits each row's keys over several threadgroups
/// (chunks of 256) and combines the partials, batched with a short sequence whose
/// later chunks are empty. Compared with fresh CPU prefills of the full sequences.
#[test]
fn metal_split_context_decode_matches_cpu() {
    if !MetalLlama::is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let cfg = LlamaConfig { max_position_embeddings: 1024, ..config() };
    let make = || {
        let mut m = LlamaModel::new_seeded(cfg.clone(), Arc::new(CpuBackend), 11).unwrap();
        m.kv_dtype = DType::F16;
        m
    };
    let gpu = MetalLlama::from_cpu_model(&make()).unwrap();
    let block = 16;
    gpu.init_kv_cache(64, block).unwrap();
    let long: Vec<u32> = (0..700u32).map(|i| (i * 31 + 7) % 96).collect();
    let short: Vec<u32> = (0..20u32).map(|i| (i * 13 + 1) % 96).collect();
    let ta = BlockTable { block_size: block, block_ids: (0..45).collect() };
    let tb = BlockTable { block_size: block, block_ids: (50..52).collect() };
    for (toks, t) in [(&long, &ta), (&short, &tb)] {
        let p: Vec<usize> = (0..toks.len()).collect();
        gpu.forward_cached(&ForwardBatch { tokens: toks, positions: &p, block_tables: &vec![t.clone(); toks.len()], logits_rows: &[0] })
            .unwrap();
    }
    // One decode step for both sequences in the same batch.
    let (na, nb) = (5u32, 9u32);
    let g = gpu
        .forward_cached(&ForwardBatch {
            tokens: &[na, nb],
            positions: &[long.len(), short.len()],
            block_tables: &[ta.clone(), tb.clone()],
            logits_rows: &[0, 1],
        })
        .unwrap();
    let vocab = cfg.vocab_size;
    for (row, (toks, next)) in [(&long, na), (&short, nb)].into_iter().enumerate() {
        let mut all = toks.clone();
        all.push(next);
        let cpu = make();
        cpu.init_kv_cache(64, block).unwrap();
        let p: Vec<usize> = (0..all.len()).collect();
        let t = BlockTable { block_size: block, block_ids: (0..all.len().div_ceil(block)).collect() };
        let c = cpu
            .forward_cached(&ForwardBatch { tokens: &all, positions: &p, block_tables: &vec![t; all.len()], logits_rows: &[all.len() - 1] })
            .unwrap();
        let d = max_rel_diff(&g[row * vocab..(row + 1) * vocab], &c);
        assert!(d < 2e-3, "row {row} (context {}): relative diff {d}", all.len());
    }
}

/// Small batches use dedicated kernels: up to 4 rows (2 for K-quants, 8 for F32
/// weights) the multi-row matrix-vector kernels (`mv2`/`mv4`/`mv8`), then up to 32 rows
/// the small-batch simdgroup-matrix GEMM (`mms`). Prefilling `m` tokens at once must match feeding them
/// one at a time through the single-row kernels, for every weight format (F16 weights
/// are exact roundings of the seeded F32 ones, which Metal uploads as F16). The bound
/// allows for the prefill attention kernel staging keys/values in half precision (see
/// `check`) and `mms` rounding activations to half; a wrong weight or activation index
/// gives differences of order 1.
#[test]
fn metal_small_batches_match_token_by_token() {
    if !MetalLlama::is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let f16_model = || {
        let mut m = LlamaModel::new_seeded(config(), Arc::new(CpuBackend), 11).unwrap();
        let round = |t: &mut Tensor| {
            let v: Vec<f32> = t.to_vec::<f32>().unwrap().iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
            *t = Tensor::from_f32(&v, t.shape().clone(), Device::Cpu).unwrap();
        };
        round(&mut m.embed_tokens);
        round(&mut m.lm_head);
        for l in &mut m.layers {
            for w in [&mut l.self_attn.q_proj, &mut l.self_attn.k_proj, &mut l.self_attn.v_proj, &mut l.self_attn.o_proj, &mut l.mlp.gate_proj, &mut l.mlp.up_proj, &mut l.mlp.down_proj] {
                round(w);
            }
        }
        m.fuse_projections().unwrap();
        m
    };
    let models = [
        ("f32", model(DType::F32)),
        ("f16", f16_model()),
        ("q8_0", model(DType::Q8_0)),
        ("q4_0", model(DType::Q4_0)),
        ("q4_k/q6_k", model(DType::Q4_K)),
    ];
    let mut worst = Vec::new();
    for (name, cpu) in models {
        let mut max_d = 0f32;
        let gpu = MetalLlama::from_cpu_model(&cpu).unwrap();
        let block = 16;
        gpu.init_kv_cache(4, block).unwrap();
        let vocab = cpu.config.vocab_size;
        for m in (2..=8usize).chain([12, 16, 24, 32]) {
            let tokens: Vec<u32> = (0..m as u32).map(|i| (i * 17 + m as u32) % 96).collect();
            let rows: Vec<usize> = (0..m).collect();
            let batched = gpu
                .forward_cached(&ForwardBatch {
                    tokens: &tokens,
                    positions: &rows,
                    block_tables: &vec![BlockTable { block_size: block, block_ids: vec![0, 1] }; m],
                    logits_rows: &rows,
                })
                .unwrap();
            let t1 = BlockTable { block_size: block, block_ids: vec![2, 3] };
            for (i, &tok) in tokens.iter().enumerate() {
                let single = gpu
                    .forward_cached(&ForwardBatch { tokens: &[tok], positions: &[i], block_tables: std::slice::from_ref(&t1), logits_rows: &[0] })
                    .unwrap();
                let d = max_rel_diff(&batched[i * vocab..(i + 1) * vocab], &single);
                assert!(d < 1e-3, "{name} m={m} row {i}: relative diff {d}");
                max_d = max_d.max(d);
            }
        }
        worst.push(format!("{name} {max_d:.1e}"));
    }
    eprintln!("max relative diffs: {}", worst.join(", "));
}

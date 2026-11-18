# `rllm` Benchmarks

> [!IMPORTANT]
> **Previously published end-to-end numbers have been withdrawn.** They were measured on synthetic (constant) weights, with no prompt prefill and with correctness bugs in the decode path that have since been fixed. They do not describe the behaviour of a real model and should not be cited.
>
> Real-model measurements now start in section 1; the synthetic `bench` command is described in section 2.

---

## 1. Real-model measurements (preliminary)

**Correctness first.** For SmolLM2-135M-Instruct, rllm's greedy output matches llama.cpp token-for-token (4 prompts × 32 tokens) in F16, Q8_0, Q4_0, Q4_K_M and Q5_K_M, and its tokenizer matches `llama-tokenize` on a 23-case corpus (`crates/rllm-models/tests/real_model.rs`).

**Throughput.** Apple M4 (10 cores), macOS, CPU only, SmolLM2-135M-Instruct Q8_0, 41-token chat prompt, 60 generated tokens:

| Engine | Prefill (tok/s) | Decode (tok/s) |
| :--- | ---: | ---: |
| rllm (`rllm generate --temperature 0`, 3 runs) | ~280–340 | ~66–75 |
| llama.cpp (`llama-bench -ngl 0 -p 41 -n 60`, 4 threads, BLAS) | ~2,900 | ~360 |

rllm is roughly 9× slower at prefill and 5× slower at decode. The main reasons are scalar quantized dot products (llama.cpp uses SIMD int8 dot products on quantized activations), no tiled GEMM for prefill, and F32 KV cache. These are P3 items. Treat these figures as a single-machine snapshot, not a formal benchmark.

---

## 2. What the `bench` command measures today

`rllm-cli bench` runs four sections. Their current validity:

| Section | What it does | Status |
| :--- | :--- | :--- |
| Memory footprint table | Arithmetic estimate from the preset's parameter count and KV-cache size | **Estimate only.** Not a measured RSS. The presets are not real model configurations. |
| Micro-kernels (F32 GEMV, Q4_0 GEMV, RMSNorm, PagedAttention decode) | Times individual CPU kernels on random-shaped buffers | **Usable as relative kernel timings.** Kernels are un-tuned (scalar Q4_0 inner loop, no tiling), so absolute numbers are far from state of the art. |
| End-to-end decode (TTFT, per-token latency, throughput, batch scaling) | Runs `ContinuousBatchScheduler::step` on a synthetic model | **Mechanically correct, not citable.** Prefill and decode now compute the right thing (verified against a reference implementation), but on synthetic weights with an untiled GEMM. TTFT is the prefill step; decode latency excludes it. |
| Radix prefix cache (cold vs warm) | Submits the same prompt twice | **Meaningful as a relative figure.** The warm run reuses cached KV blocks and only prefills the uncached tail. |

---

## 3. Presets

The presets are synthetic configurations used for sizing. They are **not** the published configurations of any real model.

| Preset | Hidden | Intermediate | Layers | Q heads / KV heads | Vocab | Approx. params |
| :--- | ---: | ---: | ---: | :---: | ---: | ---: |
| `tiny` | 64 | 128 | 2 | 4 / 2 | 1,000 | < 1 M |
| `edge-4gb` | 1,536 | 4,096 | 12 | 12 / 4 | 32,000 | ~400 M |
| `1b` | 2,048 | 8,192 | 16 | 32 / 8 | 32,000 | ~1.1 B |

Run `cargo run --release -p rllm-cli -- bench --preset <name>` to print the exact configuration and the memory estimate.

---

## 4. Planned methodology

Once a real model runs correctly:

1. **Models:** one small LLaMA-family GGUF model in F32/F16, Q8_0 and Q4_0.
2. **Correctness gate:** logits match a reference implementation within tolerance before any timing is reported.
3. **Metrics:** prefill tokens/s, decode tokens/s, TTFT, per-token latency percentiles, peak RSS, at batch sizes 1/2/4/8.
4. **Prefix cache:** cold vs warm TTFT for a shared system prompt of 256–2,048 tokens.
5. **Baseline:** llama.cpp on the same machine and the same model file.
6. **Reporting:** hardware, OS, Rust version, commit hash, and exact commands for every table.

---

## 5. Reproduce

```bash
# Real model (see README for the download command)
cargo run --release -p rllm-cli -- generate --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --prompt "What is the boiling point of water at sea level?" --temperature 0 --max-tokens 60
llama-bench -m models/SmolLM2-135M-Instruct-Q8_0.gguf -ngl 0 -p 41 -n 60

# Synthetic presets
cargo run --release -p rllm-cli -- bench --preset tiny --batch-size 4
cargo run --release -p rllm-cli -- bench --preset edge-4gb --batch-size 4 --prompt-tokens 64 --generate-tokens 16
```

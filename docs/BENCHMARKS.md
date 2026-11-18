# `rllm` Benchmarks

> [!IMPORTANT]
> **Previously published end-to-end numbers have been withdrawn.** They were measured on synthetic (constant) weights, with no prompt prefill and with correctness bugs in the decode path that have since been fixed. They do not describe the behaviour of a real model and should not be cited.
>
> Real-model measurements now start in section 1; the synthetic `bench` command is described in section 2.

---

## 1. Real-model measurements

### Correctness

Checked against llama.cpp on SmolLM2-135M-Instruct (`crates/rllm-models/tests/real_model.rs`):

| Check | Result |
| :--- | :--- |
| Tokenizer vs `llama-tokenize` (23 cases) | identical |
| Greedy decode vs `llama-completion -ngl 0` (Q8_0, 4 prompts × 32 tokens) | identical |
| Perplexity vs `llama-perplexity -ngl 0` (LICENSE text, 16 × 128-token chunks) | F16 −0.00%, Q8_0 +0.35%, Q4_0 −0.59%, Q4_K_M +0.22%, Q5_K_M +0.09% |
| Teacher-forced: llama.cpp's greedy tokens vs rllm's argmax, all formats | every token is rllm's argmax or within 0.25 (8/16-bit) / 0.5 (4/5-bit) logits |

Exact token parity across engines is only expected where both use the same arithmetic. llama.cpp quantizes activations to Q8 for decode and small batches but uses F32 BLAS for batches of 32+ tokens, and its Metal backend never quantizes activations; rllm follows the CPU behaviour. Near-tied tokens can flip between these paths, so perplexity is the primary cross-engine check.

### Throughput

Apple M4 (4 performance + 6 efficiency cores), macOS, CPU only, 4 threads for both engines, SmolLM2-135M-Instruct. llama.cpp: `llama-bench -ngl 0 -p 128,512 -n 128 -r 3` (BLAS via Accelerate). rllm: prefill = `rllm perplexity --ctx {128,512}` (one forward pass per chunk), decode = `rllm generate --raw --max-tokens 128`, median of 3. Tokens/s:

| Weights | llama.cpp pp128 | rllm pp128 | llama.cpp pp512 | rllm pp512 | llama.cpp tg128 | rllm tg128 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 1,918 | **2,252** | 1,308 | **1,888** | **399** | 340 |
| Q4_0 | 1,969 | **2,232** | 1,318 | **1,903** | **568** | 368 |
| Q4_K_M | 1,378 | **2,232** | 1,108 | **1,876** | **364** | ~320 ¹ |
| Q5_K_M | 1,469 | **2,302** | 1,214 | **1,891** | **423** | ~320 ¹ |
| F16 | 1,162 | **2,161** | 1,257 | **1,874** | **132** | 94 |

¹ After the load-time Q8_0 conversion of K-quant weights (previously 135 tok/s); measured in separate runs in the same session.

Progress during P3 on the 41-token chat prompt (Q8_0): prefill 340 → ~1,050 tok/s (and ~1,900 tok/s at 512 tokens), decode 70 → ~340 tok/s.

**What changed and why it mattered** (profiled with macOS `sample`):
1. *Int8 decode kernels.* Activations are quantized to Q8 once per matrix-vector product and multiplied with Q8_0 / Q4_0 weights using the ARM `SDOT` instruction (16 int8 multiply-adds per instruction), as llama.cpp does.
2. *Accelerate GEMM for batches of 32+ rows.* Weights are dequantized and multiplied with `cblas_sgemm`, which runs on Apple's matrix units.
3. *Parallel overhead.* Small element-wise kernels no longer split across threads, matrix-vector products split into ~128 KiB tasks, the forward pass runs inside the Rayon pool (inner kernels use work-stealing instead of injecting jobs and blocking), and the default thread count is the number of performance cores. On this machine 4 threads beat 10 by ~1.6× in decode, because every synchronisation waits for the slowest (efficiency) core.
4. *Grouped-query attention.* All query heads sharing a KV head are processed in one pass over the cache, reading keys and values once instead of `num_heads / num_kv_heads` times.
5. *K-quant weights converted to Q8_0 at load* instead of F32 (2.4× faster decode, ~4× less memory; perplexity within 0.25% of llama.cpp).

**Remaining gaps:** decode is 15–35% behind llama.cpp (it repacks Q4_0 weights for wider SIMD kernels and uses native K-quant kernels and an F16 KV cache). Non-Apple platforms have no vendor BLAS (portable GEMM fallback) and x86 has no SIMD int8 kernels yet (portable scalar fallback). Measurements from this single machine are snapshots, not a formal benchmark; background load affects them noticeably.

### Tuning knobs

- `--threads N` / `RAYON_NUM_THREADS` — worker threads (default: performance cores on Apple Silicon, otherwise all cores).
- `RLLM_TASK_BYTES` — weight bytes per parallel task in matrix-vector products (default 131072).

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
cargo run --release -p rllm-cli -- perplexity --model models/SmolLM2-135M-Instruct-Q8_0.gguf --file LICENSE --ctx 512
cargo run --release -p rllm-cli -- generate --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --raw --prompt "Once upon a time" --temperature 0 --max-tokens 128
llama-bench -m models/SmolLM2-135M-Instruct-Q8_0.gguf -ngl 0 -p 128,512 -n 128
llama-perplexity -m models/SmolLM2-135M-Instruct-Q8_0.gguf -f LICENSE -c 128 -ngl 0

# Synthetic presets
cargo run --release -p rllm-cli -- bench --preset tiny --batch-size 4
cargo run --release -p rllm-cli -- bench --preset edge-4gb --batch-size 4 --prompt-tokens 64 --generate-tokens 16
```

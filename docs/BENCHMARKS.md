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

### Head-to-head on an Apple M4 (latest)

`python3 scripts/bench_vs_llamacpp.py --device {cpu,metal} <models>`: llama-bench (`-ngl 0` / `-ngl 99`) vs `rllm bench --model`, pp = one forward pass over N tokens, tg = per-token decode, 3 repetitions, 4 CPU threads. Ratios above 1.00x mean rllm is faster. Single runs vary by a few percent (llama.cpp's Metal tg128 on SmolLM2 ranged 283–347 tok/s across runs).

| model | device | test | llama.cpp t/s | rllm t/s | rllm / llama.cpp |
| --- | --- | --- | ---: | ---: | ---: |
| SmolLM2-135M-Instruct-Q8_0 | cpu | pp128 | 1,935 | 2,993 | 1.55x |
| SmolLM2-135M-Instruct-Q8_0 | cpu | pp512 | 1,304 | 2,324 | 1.78x |
| SmolLM2-135M-Instruct-Q8_0 | cpu | tg128 | 401 | 372 | 0.93x |
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | pp128 | 1,961 | 2,949 | 1.50x |
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | pp512 | 1,320 | 2,287 | 1.73x |
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | tg128 | 555 | 417 | 0.75x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | pp128 | 1,381 | 3,050 | 2.21x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | pp512 | 1,113 | 2,300 | 2.07x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | tg128 | 447 | 374 | 0.84x |
| SmolLM2-135M-Instruct-F16.requant | cpu | pp128 | 1,207 | 3,048 | 2.53x |
| SmolLM2-135M-Instruct-F16.requant | cpu | pp512 | 1,239 | 2,307 | 1.86x |
| SmolLM2-135M-Instruct-F16.requant | cpu | tg128 | 131 | 121 | 0.92x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp128 | 469 | 410 | 0.87x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp512 | 391 | 520 | 1.33x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | tg128 | 54 | 53 | 0.98x |
| SmolLM2-135M-Instruct-Q8_0 | metal | pp128 | 9,916 | 7,640 | 0.77x |
| SmolLM2-135M-Instruct-Q8_0 | metal | pp512 | 10,858 | 8,162 | 0.75x |
| SmolLM2-135M-Instruct-Q8_0 | metal | tg128 | 347 | 303 | 0.87x |
| SmolLM2-135M-Instruct-Q4_0.requant | metal | pp128 | 10,193 | 7,320 | 0.72x |
| SmolLM2-135M-Instruct-Q4_0.requant | metal | pp512 | 11,058 | 7,756 | 0.70x |
| SmolLM2-135M-Instruct-Q4_0.requant | metal | tg128 | 350 | 303 | 0.87x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | metal | pp128 | 9,518 | 7,632 | 0.80x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | metal | pp512 | 10,365 | 8,146 | 0.79x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | metal | tg128 | 349 | 296 | 0.85x |
| SmolLM2-135M-Instruct-F16.requant | metal | pp128 | 10,129 | 7,426 | 0.73x |
| SmolLM2-135M-Instruct-F16.requant | metal | pp512 | 11,361 | 8,260 | 0.73x |
| SmolLM2-135M-Instruct-F16.requant | metal | tg128 | 223 | 221 | 0.99x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp128 | 1,604 | 1,323 | 0.82x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp512 | 1,657 | 1,388 | 0.84x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | tg128 | 71 | 69 | 0.97x |

**Where rllm is ahead:** CPU prompt processing in every format (1.3–2.5x).
**At parity (within ~3%):** CPU decode on Llama-3.2-1B, Metal decode on Llama-3.2-1B and on F16 weights.
**Behind:**
- Metal prompt processing, 0.70–0.84x. The quantized GEMM (64x64 tiles, half-precision k-major weight tiles) plateaued here after A/B testing nine variants; the remaining gap is inside the k-loop.
- CPU decode of 4-bit weights, Q4_0 0.75x and Q4_K_M 0.84x. llama.cpp repacks Q4_0 weights for wider kernels, and runs K-quants natively at ~4.5 bits per weight while rllm converts them to Q8_0 (8.5 bits).
- CPU prompt processing on 1B at 128 tokens, 0.87x (Accelerate GEMM efficiency at small M).

**What changed in this round** (all measured, with interleaved A/B runs for decisions):
1. Metal quantized GEMM: 32x32 F32 tiles → 64x64 tiles with half-precision k-major weight tiles (1B pp512 996 → 1,388 tok/s). Rejected after A/B: register prefetching, half-precision activations, 8 simdgroups, a transposed Cᵀ formulation (3.6x slower), and a conflict-free weight-write mapping.
2. Metal: Q/K/V and gate/up fused into single matrices, a single RoPE dispatch, and residual adds folded into the output projections (17 → 10 dispatches per layer). A first version initialised GEMM accumulators from `y` and was 3x slower (register spills); the accumulate is now a staged epilogue.
3. Metal: F16 weights stay F16 (decode 136 → 221 tok/s on F16 files).
4. CPU: fused Q/K/V and gate/up for batches under 32 rows (decode +6%), zero-copy via row views; prefill keeps separate projections (fusion cost 10% there).
5. CPU: 2,048-row chunked dequantize+GEMM for prefill (1B pp128 +6%).
6. CPU: four-row Q4_0 decode kernel with the nibble offset folded into activation sums (+5–7%); not used for Q8_0, where it was slower.

### Earlier measurements

#### Throughput

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

#### Metal (Apple GPU, before this round)

Apple M4 (10-core GPU), `rllm --device metal bench --model … --prompt-tokens {128,512} --generate-tokens 128` (median of 3) vs `llama-bench -ngl 99`. Tokens/s:

| Model | Engine | pp128 | pp512 | tg128 |
| :--- | :--- | ---: | ---: | ---: |
| SmolLM2-135M Q8_0 | llama.cpp Metal | 9,879 | 10,838 | 347 |
| | rllm Metal | 6,127 | 6,518 | 270 |
| Llama-3.2-1B Q8_0 | llama.cpp CPU (`-ngl 0`) | 467 | 393 | 54 |
| | rllm CPU | 385 | 510 | 52 |
| | llama.cpp Metal | 1,598 | 1,658 | **67.5** |
| | rllm Metal | 986 | 987 | **67.3** |

For a 135M model, GPU decode is limited by per-token submission overhead and is slower than the CPU for both engines' best settings; at 1B, decode is memory-bound and rllm matches llama.cpp. Prefill on Metal is ~60% of llama.cpp's: profiling showed attention dominating at first (58% of prefill at 512 tokens), which the tiled kernel fixed (pp512 3,500 → 6,500 tok/s on 135M); the remaining gap is the GEMM kernel (a 64×32 tile variant was measured ~4% slower and not adopted).

Correctness on Metal: perplexity vs `llama-perplexity -ngl 99` — Q8_0 +0.02%, Q4_0 +0.02%, F16 +0.00%, Q4_K_M +0.57%, Q5_K_M +0.23% (K-quants are converted to Q8_0 at load; llama.cpp runs them natively); Llama-3.2-1B +0.006%. Greedy output identical to `llama-completion -ngl 99` for SmolLM2-135M and Llama-3.2-1B (4 prompts × 32 tokens each). Metal vs rllm CPU on synthetic models: max relative logit difference < 1e-4 (F32, Q8_0, Q4_0).

`rllm bench --model` mirrors llama-bench: pp = one forward pass over N random tokens, tg = per-token time of N single-token decode steps; `rllm perplexity` additionally spends time on CPU log-softmax and is not a throughput benchmark.

### Tuning knobs

- `--device cpu|metal` — where the LLM runs.
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
# Real model (see README for the download command); add `--device metal` for the GPU
cargo run --release -p rllm-cli -- bench --model models/SmolLM2-135M-Instruct-Q8_0.gguf --prompt-tokens 512 --generate-tokens 128
cargo run --release -p rllm-cli -- perplexity --model models/SmolLM2-135M-Instruct-Q8_0.gguf --file LICENSE --ctx 512
cargo run --release -p rllm-cli -- generate --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --raw --prompt "Once upon a time" --temperature 0 --max-tokens 128
llama-bench -m models/SmolLM2-135M-Instruct-Q8_0.gguf -ngl 0 -p 128,512 -n 128
llama-perplexity -m models/SmolLM2-135M-Instruct-Q8_0.gguf -f LICENSE -c 128 -ngl 0

# Synthetic presets
cargo run --release -p rllm-cli -- bench --preset tiny --batch-size 4
cargo run --release -p rllm-cli -- bench --preset edge-4gb --batch-size 4 --prompt-tokens 64 --generate-tokens 16
```

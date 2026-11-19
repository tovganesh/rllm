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

`python3 scripts/bench_vs_llamacpp.py --device {cpu,metal} <models>`: llama-bench (`-ngl 0` / `-ngl 99`) vs `rllm bench --model`, pp = one forward pass over N tokens, tg = per-token decode, 3 repetitions, 4 CPU threads. Ratios above 1.00x mean rllm is faster. Single runs vary by a few percent (llama.cpp's Metal tg128 on SmolLM2 ranged 283–347 tok/s across runs). The `.requant` Llama-3.2-1B files were made from the Q8_0 file with `llama-quantize --allow-requantize`; both engines load the same file.

#### CPU

| model | device | test | llama.cpp t/s | rllm t/s | rllm / llama.cpp |
| --- | --- | --- | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp128 | 476 | 405 | 0.85x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp512 | 390 | 507 | 1.30x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | tg128 | 54 | 51 | 0.95x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | pp128 | 164 | 393 | 2.39x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | pp512 | 155 | 502 | 3.25x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | tg128 | 78 | 82 | 1.04x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp128 | 489 | 457 | 0.93x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp512 | 395 | 503 | 1.28x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | tg128 | 88 | 87 | 0.99x |
| SmolLM2-135M-Instruct-Q8_0 | cpu | pp128 | 1,881 | 3,033 | 1.61x |
| SmolLM2-135M-Instruct-Q8_0 | cpu | pp512 | 1,293 | 2,311 | 1.79x |
| SmolLM2-135M-Instruct-Q8_0 | cpu | tg128 | 400 | 362 | 0.90x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | pp128 | 1,369 | 3,019 | 2.21x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | pp512 | 1,084 | 2,313 | 2.13x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | tg128 | 439 | 380 | 0.87x |
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | pp128 | 1,912 | 2,958 | 1.55x |
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | pp512 | 1,300 | 2,312 | 1.78x |
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | tg128 | 561 | 505 | 0.90x |
| SmolLM2-135M-Instruct-F16.requant | cpu | pp128 | 1,196 | 3,052 | 2.55x |
| SmolLM2-135M-Instruct-F16.requant | cpu | pp512 | 1,205 | 2,332 | 1.94x |
| SmolLM2-135M-Instruct-F16.requant | cpu | tg128 | 132 | 120 | 0.91x |

#### Metal

| model | device | test | llama.cpp t/s | rllm t/s | rllm / llama.cpp |
| --- | --- | --- | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp128 | 1,599 | 1,390 | 0.87x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp512 | 1,647 | 1,508 | 0.92x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | tg128 | 67 | 64 | 0.96x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | metal | pp128 | 1,546 | 1,327 | 0.86x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | metal | pp512 | 1,592 | 1,436 | 0.90x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | metal | tg128 | 102 | 100 | 0.98x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | pp128 | 1,629 | 1,386 | 0.85x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | pp512 | 1,672 | 1,489 | 0.89x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | tg128 | 108 | 112 | 1.04x |
| SmolLM2-135M-Instruct-Q8_0 | metal | pp128 | 9,468 | 8,359 | 0.88x |
| SmolLM2-135M-Instruct-Q8_0 | metal | pp512 | 10,958 | 9,910 | 0.90x |
| SmolLM2-135M-Instruct-Q8_0 | metal | tg128 | 270 | 331 | 1.23x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | metal | pp128 | 9,538 | 8,273 | 0.87x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | metal | pp512 | 10,273 | 9,732 | 0.95x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | metal | tg128 | 263 | 341 | 1.30x |
| SmolLM2-135M-Instruct-Q4_0.requant | metal | pp128 | 10,249 | 7,294 | 0.71x |
| SmolLM2-135M-Instruct-Q4_0.requant | metal | pp512 | 10,924 | 9,813 | 0.90x |
| SmolLM2-135M-Instruct-Q4_0.requant | metal | tg128 | 349 | 411 | 1.18x |
| SmolLM2-135M-Instruct-F16.requant | metal | pp128 | 8,822 | 8,669 | 0.98x |
| SmolLM2-135M-Instruct-F16.requant | metal | pp512 | 11,272 | 10,475 | 0.93x |
| SmolLM2-135M-Instruct-F16.requant | metal | tg128 | 214 | 236 | 1.10x |

SmolLM2-135M pp128 on Metal is noisy in this script (single runs of the same build vary by up to ~20%).

#### Long context (2,048 tokens of context)

`python3 scripts/bench_vs_llamacpp.py --depth 2048` (llama-bench `-d 2048`, `rllm bench --depth 2048`; 2 repetitions):

| model | device | test | llama.cpp t/s | rllm t/s | rllm / llama.cpp |
| --- | --- | --- | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | pp128 @ d2048 | 1,352 | 1,171 | 0.87x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | pp512 @ d2048 | 1,390 | 1,266 | 0.91x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | tg128 @ d2048 | 101 | 80 | 0.79x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp128 @ d2048 | 1,327 | 1,179 | 0.89x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp512 @ d2048 | 1,381 | 1,282 | 0.93x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | tg128 @ d2048 | 67 | 54 | 0.81x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp128 @ d2048 | 137 | 178 | 1.29x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp512 @ d2048 | 130 | 191 | 1.46x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | tg128 @ d2048 | 45 | 60 | 1.34x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp128 @ d2048 | 137 | 171 | 1.25x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp512 @ d2048 | 124 | 191 | 1.53x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | tg128 @ d2048 | 34 | 42 | 1.21x |

On the CPU, rllm's paged GQA attention keeps it well ahead at depth. On Metal, long-context decode is 0.79–0.81x: the decode attention kernel reads F32 keys/values (llama.cpp: F16) — the next targets are the F16 KV cache and that kernel.

#### Prefix cache: time to first token

`rllm bench --model M --prompt-tokens P --shared-prefix S`: a request whose first S tokens were computed by an earlier request (warm) versus one with nothing cached (cold), through the scheduler; median of 3.

| model | device | prompt | shared prefix | cached tokens | TTFT cold ms | TTFT warm ms | speed-up |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | 512 | 448 | 448 | 996.4 | 282.0 | 3.53x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | 2048 | 1984 | 1984 | 6039.8 | 433.2 | 13.94x |
| Llama-3.2-1B-Instruct-Q8_0 | metal (Apple M4) | 512 | 448 | 448 | 342.0 | 54.5 | 6.27x |
| Llama-3.2-1B-Instruct-Q8_0 | metal (Apple M4) | 2048 | 1984 | 1984 | 1440.3 | 62.4 | 23.08x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | 1024 | 896 | 896 | 2269.9 | 474.1 | 4.79x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal (Apple M4) | 1024 | 896 | 896 | 706.5 | 100.5 | 7.03x |

Completed blocks are now shared as soon as they are computed (previously only when a sequence finished), so concurrent requests with the same system prompt or chat history reuse each other's KV blocks.

**Where rllm is ahead:** CPU prompt processing at 512 tokens in every format (1.3–3.3x), and at 128 tokens on SmolLM2 and K-quant files; CPU decode of Llama-3.2-1B Q4_K_M (1.04x); Metal decode on SmolLM2-135M (1.10–1.30x) and Llama-3.2-1B Q4_0 (1.04x).
**At parity (within ~5%):** decode on Llama-3.2-1B in every format on both CPU (0.95–1.04x) and Metal (0.96–1.04x); Metal prompt processing on SmolLM2 Q4_K_M at 512 tokens (0.95x).
**Behind:**
- Metal prompt processing, 0.85–0.93x (128–512 tokens). The quantized GEMM now runs at 3.0–3.2 TFLOPS on the 1B shapes, ~80% of the 3.9 TFLOPS half-precision `simdgroup_matrix` peak measured on this GPU; prefill attention is ~4% of the time.
- CPU decode on SmolLM2-135M, 0.87–0.91x. At ~2 ms per token across ~120 small matrix-vector products, the profile is dominated by thread wake-up/fork-join latency rather than kernels (llama.cpp's thread pool spins between ops).
- CPU prompt processing on 1B Q8_0 at 128 tokens, 0.85x (Accelerate GEMM efficiency at small M; Q4_0 uses i8mm there, 0.93x).

**What changed in this round** (Metal prefill; Llama-3.2-1B pp512 Q8_0 1,377 → 1,508, Q4_0 1,292 → 1,487, Q4_K_M 1,327 → 1,425 tok/s; SmolLM2 Q8_0 8,132 → 9,868):
1. **GEMM** (`tests/gemm_bench.rs`, an ignored micro-benchmark on the 1B shapes, plus a peak probe: 3.93 TFLOPS for half x half -> float `simdgroup_matrix` products, 1.05 for float x float): k-step 32 → 64; activations converted to half once per linear layer (`to_half`) instead of once per weight tile; one 32-value weight block dequantized per thread with conflict-free tile stores (Q6_K, whose 210-byte blocks are not word-aligned, keeps four threads per block). GEMM time per layer at 512 rows: Q4_0 −11%, F16 −9%, Q8_0 −6%, Q4_K −5%, Q6_K −5%. Rejected: 64x128 tiles with 32 accumulators per simdgroup (register spills, 5.5x slower), 8 simdgroups on 64x128 (mixed), half activations alone (+1%).
2. **Prefill attention** rewritten (3.1x faster at 1,024 tokens: 92 → 29 ms on 1B; was 8% of pp512 time): 32-row query tiles over four simdgroups sharing 32-key K/V blocks staged as half; scores computed transposed (S^T = K Q^T, so K fragments load untransposed) with two accumulators per key group (the previous single chain of 8 dependent products was latency-bound); in-simdgroup online softmax; lazy rescaling; and — the largest single step, 63 → 29 ms — the head dimension as a template parameter, so fragment arrays indexed in loops stay in registers instead of spilling. Found by ablating loads / scores / P·V with temporary shader defines.
3. Decode attention also templated on the head dimension (+2–6% decode; larger on SmolLM2). *Correction:* this was reported as measured "at 2,048 tokens of context", but `rllm bench --model` then always decoded from position 0, so it was a short-context figure. `--depth` now exists; see "Long context" above.

Correctness of this round: Metal perplexity vs `llama-perplexity -ngl 99` at 128 and 512 tokens of context within 0.05% for 1B Q8_0 / Q4_0 and SmolLM2 Q8_0 / F16 (1B Q4_K_M: −0.20% at 128, −0.01% at 512); greedy output identical to `llama-simple -ngl 99` for 1B Q8_0, Q4_K_M and Q4_0. The F32 Metal-vs-CPU prefill bound went from 1e-4 to 5e-4 (attention now stages K/V as half, as llama.cpp's Metal flash attention does).

### Earlier measurements

#### Native 4-bit Metal kernels round (PR #13)

1. **Native Q4_K and Q6_K on Metal** instead of converting to Q8_0 at upload: dedicated matrix-vector kernels (8 lanes per 256-value super-block, 4 rows per simdgroup sharing the activations, Q4_K mins and Q6_K's −32 folded in through sums of x) plus K-quant tile loaders for the prefill GEMM and embedding lookups. Llama-3.2-1B Q4_K_M decode 65 → 97 tok/s (0.63x → 0.96x), with about half the GPU memory. Its tied Q6_K output head alone was 279 MB per token as Q8_0, 215 MB now.
2. **Dedicated Q4_0 matrix-vector kernel:** two lanes per block reading 2-byte words, the −8 offset applied once per block as 8·Σx (the generic kernel had four lanes per block, two of them re-reading the same bytes). 1B Q4_0 decode 85 → 105 tok/s (0.77x → 1.01x).
3. **Mixed-format fusion:** fused Q/K/V and gate/up are now lists of same-format segments; linear kernels take an output row stride (`ldy`) and a buffer offset, so a Q4_K q/k segment and a Q6_K v segment write adjacent columns of one output (no F32 fallback).
4. Q5_0 is expanded to Q8_0 losslessly on Metal too (previously via F32 with a new scale).
5. Cost: 1B Q4_K_M prefill −4% (1,365 → 1,315 tok/s) from decoding K-quant scales in the GEMM tile loader and the extra Q6_K segment dispatch; 2-byte/8-byte vector loads in the loaders recovered ~0.5%.

Correctness of this round: Metal-vs-CPU test on a synthetic Q4_K_M-like model (Q4_K q/k/o/gate/up, Q6_K v/down/embeddings) within 1e-4 for decode and 1e-3 for prefill — a planted −32→−31 bug in the Q6_K kernel fails it at 2e-2. Greedy output on Metal identical to `llama-simple -ngl 99` for 1B Q4_K_M and Q4_0. Metal perplexity vs `llama-perplexity -ngl 99`: 1B Q4_K_M 7.6988 vs 7.7143 (previously 7.7358 as Q8_0), 1B Q4_0 8.9459 vs 8.9436, SmolLM2 Q4_K_M 12.8518 vs 12.8523.

#### Native 4-bit CPU kernels round (PR #12)

1. **Native K-quants.** `Q4_K` and `Q6_K` weights stay as stored (4.5 / 6.6 bits per weight instead of 8.5 as Q8_0) and run SDOT kernels against `Q8_K` activations (256-value super-blocks with per-16 sums), as in llama.cpp. Llama-3.2-1B Q4_K_M decode 52 → 82 tok/s (0.70x → 1.04x), and the model uses ~half the memory.
2. **Q4_0 repacked at load** into interleaved 4-row groups (`DType::Q4_0x4`, nibbles stored as signed 4-bit) through a new `Backend::pack_weight` hook. One 16-byte load feeds two rows of SDOT lanes; the four row results come out of one `vpaddq` and are scaled with one vector of f16 scales. Q4_0 decode kernels: 1.6x faster in isolation; 1B decode 74 → 87 tok/s (0.85x → 0.99x).
3. **i8mm (`SMMLA`) GEMM** on the repacked layout for batches: 3–6x faster than per-row SDOT for 2–31 rows, and used up to 192 rows for large matrices where it beats dequantize + Accelerate (1B Q4_0 pp128 0.80x → 0.93x; pp64 +65%). Small matrices keep the Accelerate path above 32 rows (it measured 25% faster there).
4. **Q5_0 expanded losslessly to Q8_0** (same scale, int8 values in −16..15): a native Q5_0 kernel was written and tested but measured 1.75x slower than the Q8_0 kernel on the M4, even for large matrices, so it is only a fallback. This also lets SmolLM2-135M Q4_K_M layers (Q5_0 + Q8_0 parts) fuse Q/K/V, and makes its Metal decode +17% and perplexity exact.
5. Metal receives the same values as before: `Q4_0x4` is unpacked to Q4_0 and K-quants are converted to Q8_0 at upload (perplexity bit-identical to the previous release for Q4_0 and Llama-3.2-1B Q4_K_M).

Correctness of this round: kernel tests against dequantized references (exact up to float rounding, with random bits for all scale/min fields); decode-path teacher forcing on the 1B and 135M Q4_K_M files picks llama.cpp's token at every position except one near-tie each (0.12 and 0.14 logits); greedy output on 1B Q4_0 identical to `llama-simple -ngl 0`. CPU perplexity vs `llama-perplexity -ngl 0` on LICENSE: 1B Q4_K_M 7.6986 vs 7.7390, 1B Q4_0 8.9458 vs 8.9573, 135M Q4_K_M 12.8534 vs 12.8884, Q4_0 15.6772 vs 15.6971.

#### Metal GEMM, fusion and F16 round (PR #11)

All measured, with interleaved A/B runs for decisions. Before it, Metal pp512 was 0.60x, F16 Metal decode 0.57x and CPU 4-bit decode 0.75–0.84x.

1. Metal quantized GEMM: 32x32 F32 tiles → 64x64 tiles with half-precision k-major weight tiles (1B pp512 996 → 1,388 tok/s). Rejected after A/B: register prefetching, half-precision activations, 8 simdgroups, a transposed Cᵀ formulation (3.6x slower), and a conflict-free weight-write mapping.
2. Metal: Q/K/V and gate/up fused into single matrices, a single RoPE dispatch, and residual adds folded into the output projections (17 → 10 dispatches per layer). A first version initialised GEMM accumulators from `y` and was 3x slower (register spills); the accumulate is now a staged epilogue.
3. Metal: F16 weights stay F16 (decode 136 → 221 tok/s on F16 files).
4. CPU: fused Q/K/V and gate/up for batches under 32 rows (decode +6%), zero-copy via row views; prefill keeps separate projections (fusion cost 10% there).
5. CPU: 2,048-row chunked dequantize+GEMM for prefill (1B pp128 +6%).
6. CPU: four-row Q4_0 decode kernel with the nibble offset folded into activation sums (+5–7%); not used for Q8_0, where it was slower.

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
4. **Prefix cache:** cold vs warm TTFT for a shared system prompt of 256–2,048 tokens. ✅ (`rllm bench --shared-prefix`, see section 1)
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

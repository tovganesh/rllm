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
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | pp128 | 1,945 | 3,014 | 1.55x |
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | pp512 | 1,279 | 2,532 | 1.98x |
| SmolLM2-135M-Instruct-Q4_0.requant | cpu | tg128 | 553 | 581 | 1.05x |
| SmolLM2-135M-Instruct-Q8_0 | cpu | pp128 | 1,902 | 3,047 | 1.60x |
| SmolLM2-135M-Instruct-Q8_0 | cpu | pp512 | 1,284 | 2,510 | 1.95x |
| SmolLM2-135M-Instruct-Q8_0 | cpu | tg128 | 394 | 405 | 1.03x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | pp128 | 1,298 | 3,024 | 2.33x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | pp512 | 1,105 | 2,504 | 2.27x |
| SmolLM2-135M-Instruct-Q4_K_M.requant | cpu | tg128 | 442 | 428 | 0.97x |
| SmolLM2-135M-Instruct-F16.requant | cpu | pp128 | 1,179 | 3,118 | 2.65x |
| SmolLM2-135M-Instruct-F16.requant | cpu | pp512 | 1,218 | 2,504 | 2.06x |
| SmolLM2-135M-Instruct-F16.requant | cpu | tg128 | 127 | 114 | 0.90x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp128 | 484 | 456 | 0.94x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp512 | 386 | 517 | 1.34x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | tg128 | 88 | 89 | 1.01x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | pp128 | 167 | 391 | 2.34x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | pp512 | 154 | 519 | 3.36x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | tg128 | 76 | 82 | 1.08x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp128 | 467 | 408 | 0.87x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp512 | 387 | 528 | 1.36x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | tg128 | 53 | 53 | 1.00x |

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

#### Long context (2,048 and 8,192 tokens of context)

`python3 scripts/bench_vs_llamacpp.py --depth D` (llama-bench `-d D`, `rllm bench --depth D`; 2 repetitions; F16 KV cache on both sides):

| model | device | test | llama.cpp t/s | rllm t/s | rllm / llama.cpp |
| --- | --- | --- | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | pp128 @ d2048 | 1,351 | 1,180 | 0.87x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | pp512 @ d2048 | 1,397 | 1,274 | 0.91x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | tg128 @ d2048 | 104 | 95 | 0.92x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp128 @ d2048 | 1,328 | 1,187 | 0.89x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp512 @ d2048 | 1,378 | 1,291 | 0.94x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | tg128 @ d2048 | 68 | 62 | 0.91x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | pp128 @ d8192 | 900 | 823 | 0.91x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | pp512 @ d8192 | 928 | 884 | 0.95x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | tg128 @ d8192 | 86 | 66 | 0.77x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp128 @ d8192 | 843 | 816 | 0.97x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | pp512 @ d8192 | 867 | 855 | 0.99x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | tg128 @ d8192 | 59 | 48 | 0.81x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp128 @ d2048 | 134 | 190 | 1.42x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp512 @ d2048 | 128 | 213 | 1.67x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | tg128 @ d2048 | 46 | 66 | 1.43x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp128 @ d2048 | 136 | 188 | 1.38x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp512 @ d2048 | 130 | 217 | 1.67x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | tg128 @ d2048 | 34 | 44 | 1.28x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp128 @ d8192 | 44 | 63 | 1.42x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | pp512 @ d8192 | 43 | 67 | 1.55x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | tg128 @ d8192 | 15 | 38 | 2.53x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp128 @ d8192 | 41 | 68 | 1.66x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | pp512 @ d8192 | 41 | 71 | 1.72x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | tg128 @ d8192 | 15 | 29 | 1.97x |

On the CPU rllm is 1.3–2.5x ahead at depth: its paged GQA attention reads each key/value once per KV head, and F16 keys/values are widened in registers. On Metal, prompt processing at depth is 0.87–0.99x and decode 0.91–0.92x at 2k and 0.77–0.81x at 8k (llama.cpp's Metal decode figures vary by ~15% between runs). The remaining 8k gap is the decode attention kernel: query heads that share a KV head (GQA) still read it separately.

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

#### Small batches (2–32 tokens)

One forward pass over N tokens (`rllm bench --model M --prompt-tokens N`, llama-bench `-p N`): the shape of speculative-decoding verification, several sequences decoding together, and short prompts. "before" is the previous release, measured interleaved with this one (mean of 2 rounds of 5 repetitions); llama.cpp 3 repetitions.

| model | device | tokens | llama.cpp t/s | before t/s | rllm t/s | vs before | rllm / llama.cpp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | 2 | 90 | 93 | 100 | 1.08x | 1.12x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | 4 | 209 | 104 | 174 | 1.67x | 0.83x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | 8 | 290 | 118 | 239 | 2.03x | 0.82x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | 16 | 379 | 114 | 328 | 2.86x | 0.86x |
| Llama-3.2-1B-Instruct-Q8_0 | cpu | 32 | 448 | 185 | 375 | 2.02x | 0.84x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | 2 | 99 | 121 | 139 | 1.15x | 1.41x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | 4 | 134 | 132 | 177 | 1.34x | 1.32x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | 8 | 148 | 145 | 205 | 1.42x | 1.39x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | 16 | 119 | 137 | 196 | 1.43x | 1.65x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | cpu | 32 | 167 | 162 | 162 | 1.00x | 0.97x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | 2 | 130 | 171 | 171 | 1.00x | 1.32x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | 4 | 307 | 289 | 290 | 1.00x | 0.94x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | 8 | 336 | 383 | 379 | 0.99x | 1.13x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | 16 | 426 | 426 | 428 | 1.00x | 1.01x |
| Llama-3.2-1B-Instruct-Q4_0.requant | cpu | 32 | 494 | 435 | 434 | 1.00x | 0.88x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | 2 | 132 | 98 | 141 | 1.44x | 1.07x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | 4 | 255 | 115 | 268 | 2.32x | 1.05x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | 8 | 484 | 134 | 483 | 3.62x | 1.00x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | 16 | 858 | 325 | 708 | 2.18x | 0.82x |
| Llama-3.2-1B-Instruct-Q8_0 | metal | 32 | 1,205 | 639 | 857 | 1.34x | 0.71x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | metal | 2 | 182 | 146 | 167 | 1.14x | 0.92x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | metal | 4 | 348 | 169 | 315 | 1.86x | 0.91x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | metal | 8 | 651 | 181 | 634 | 3.50x | 0.97x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | metal | 16 | 893 | 312 | 702 | 2.25x | 0.79x |
| Llama-3.2-1B-Instruct-Q4_K_M.requant | metal | 32 | 1,199 | 614 | 847 | 1.38x | 0.71x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | 2 | 203 | 170 | 209 | 1.23x | 1.03x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | 4 | 399 | 202 | 309 | 1.53x | 0.77x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | 8 | 745 | 218 | 594 | 2.72x | 0.80x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | 16 | 1,033 | 326 | 677 | 2.07x | 0.66x |
| Llama-3.2-1B-Instruct-Q4_0.requant | metal | 32 | 1,283 | 642 | 836 | 1.30x | 0.65x |

On the CPU, K-quants are 1.3–1.65x ahead of llama.cpp up to 16 tokens and Q8_0 is 0.82–0.86x from 4 tokens (llama.cpp repacks Q8_0 for i8mm; rllm interleaves rows on the fly). On Metal rllm is at 0.91–1.07x of llama.cpp up to 8 tokens for Q8_0 and Q4_K_M (Q4_0 0.77–1.03x) and 0.65–0.82x at 16–32, where the small-batch GEMM runs at ~45% of the half-precision peak.

**Where rllm is ahead:** CPU prompt processing at 512 tokens in every format (1.3–3.3x), and at 128 tokens on SmolLM2 and K-quant files; CPU K-quant batches of 2–16 tokens (1.3–1.65x); CPU decode of Llama-3.2-1B Q4_K_M (1.04x) and of F16 files (1.87x); CPU everything at 4k context (1.4–3.2x); Metal decode on SmolLM2-135M (1.10–1.30x) and Llama-3.2-1B Q4_0 (1.04x).
**At parity (within ~5%):** decode on Llama-3.2-1B in every format on both CPU (0.95–1.04x) and Metal (0.96–1.04x); Metal prompt processing on SmolLM2 Q4_K_M at 512 tokens (0.95x).
**Behind:**
- Metal prompt processing on Llama-3.2-1B, 0.94–0.96x (128–512 tokens); SmolLM2 F16 is at parity or ahead (0.99–1.07x). The GEMMs run at 3.2–3.5 TFLOPS (~85% of the 3.9 TFLOPS half-precision `simdgroup_matrix` peak) and take ~94% of the time; llama.cpp's classic GEMM, rebuilt in our benchmark, measures the same. See "Metal prefill GEMM and fused SwiGLU" below.
- CPU prompt processing on 1B Q8_0 at 128 tokens, 0.85x (Accelerate GEMM efficiency at small M; Q4_0 uses i8mm there, 0.93x).
- Small batches on Metal at 16 tokens: 0.87–0.92x (at 8, 24 and 32 tokens rllm is at 0.95–1.05x). See "Metal 9–32-token batches" below.

**Metal 9–32-token batches (PR #25)**

llama.cpp's time on Llama-3.2-1B Q8_0 rises in steps: 16–18 ms for 8–16 tokens, 27 ms for 17–32 and 44 ms for 33–64. The small-batch GEMM (`mms`) grows more steeply. With the faster 64x32 GEMM from PR #24 the crossover moved down. Per layer (`tests/gemm_bench.rs`, now with `GEMM_REPS=200`, because 20 repetitions vary by ±15% at these sizes):

1. **25–32 tokens use the tiled GEMM** (and get the fused SwiGLU): 15–20% faster than `mms` per layer. At 24 tokens `mms` stays 10% faster.
2. **A 16-token tile (`mmh`) for 9–16 tokens.** It is the 64x32 kernel's scheme with 4 simdgroups of 16 weight rows × 16 tokens, so no products go to an empty half of a 32-token tile: 5–9% faster than `mms` at 12–16 tokens. It is used only on matrices of at least 2048 rows (32 threadgroups). On SmolLM2-135M's 576-row matrices it launched 9 threadgroups and was 12–14% slower.
3. **The 64x32 kernel stores whole 8x8 fragments directly** when the token count is a multiple of 8, instead of staging a partly filled tile through threadgroup memory.
4. **The `load8` Q4_0 dequantizer masks nibbles in place** (bit-identical, as in PR #24), for the matrix-vector and `mms` kernels.

End to end, Llama-3.2-1B (tok/s, interleaved A/B against PR #24, mean of 2 rounds):

| tokens | Q8_0 | Q4_K_M | Q4_0 |
| ---: | ---: | ---: | ---: |
| 12 | 585 → 602 (+2.9%) | 582 → 610 (+4.7%) | 621 → 667 (+7.5%) |
| 16 | 768 → 799 (+4.1%) | 779 → 813 (+4.3%) | 826 → 884 (+7.0%) |
| 20 | 768 → 796 (+3.7%) | 779 → 779 | 814 → 836 (+2.7%) |
| 28 | 928 → 1,009 (+8.7%) | 902 → 992 (+9.9%) | 935 → 1,050 (+12.3%) |
| 32 | 1,053 → 1,154 (+9.5%) | 1,028 → 1,136 (+10.5%) | 1,064 → 1,206 (+13.3%) |

Against llama.cpp (`llama-bench -ngl 99`) at 8 / 16 / 24 / 32 tokens: Q8_0 1.04 / 0.91 / 1.05 / 0.97x, Q4_K_M 0.97 / 0.92 / 1.00 / 0.95x, Q4_0 0.96 / 0.87 / 1.04 / 0.95x (previously 0.83–0.89x at 16 and 32 tokens). SmolLM2-135M F16 is unchanged at 8–24 tokens and 11% faster at 32.

Tests: the batch-vs-token-by-token Metal test now runs 2–9, 12, 16, 20, 24, 28 and 32 tokens on a model with a 2112-row gate/up matrix and a 96-wide hidden size (2048 rows for the K-quant mix), so both `mmh` variants (two blocks per stage, and one for odd block counts) run. A second test with a 2048-wide hidden size runs the accumulating output and down projections on `mmh` too. A planted bug in either variant fails them, as does a dropped accumulation in either `mmh` epilogue path.

**Metal prefill GEMM and fused SwiGLU (PR #24)**

Skipping one kernel type at a time on Llama-3.2-1B Q8_0 at 512 tokens (327 ms) put the GEMMs at ~308 ms, prefill attention at 8 ms, SwiGLU at 5–6 ms, RMSNorm at 1.5 ms and RoPE at 1 ms. llama.cpp's whole pass takes 310 ms, and its GEMM kernel, ported into `tests/gemm_bench.rs`, was no faster than ours. Changes:

1. **One 64x32 GEMM for nearly everything.** Each thread now dequantizes 16 values of a weight row per 32-value block (`load16`: Q4_K reads its 16 bytes in one load and decodes the sub-block scale once), and two blocks are staged per pair of threadgroup barriers. Per Llama-3.2-1B layer this is 4–5% faster for Q4_K and 1–2% for Q8_0 than before, and faster than the 64x64 tile at every prompt length for all formats except Q6_K, which keeps the 64x64 tile on large grids. Rows with an odd number of blocks use a one-block-per-stage instantiation: a one-block tail step inside the two-block kernel made Q8_0 5% slower even when it never ran.
2. **Q4_0 nibbles are masked in place** and multiplied by d, d/16, d/256 or d/4096 instead of shifted (llama.cpp's trick; the results are bit-identical). The GEMM is 3–6% faster.
3. **SwiGLU runs in the gate/up GEMM's epilogue.** Gate and up rows are interleaved in 32-row chunks at load, so each 64-row tile holds matching gate and up outputs. The tile writes `silu(gate)·up` in half precision straight into the down projection's input, which removes the 32 MB F32 gate/up buffer per layer at 512 tokens and one dispatch. Decode and small batches read the interleaved buffer with `swiglu_i`.

Interleaved A/B against the previous release (tok/s, mean of 2 rounds):

| model | 128 tokens | 512 tokens |
| --- | ---: | ---: |
| 1B Q8_0 | 1,500 → 1,526 (+1.7%) | 1,566 → 1,606 (+2.6%) |
| 1B Q4_K_M | 1,376 → 1,470 (+6.8%) | 1,471 → 1,542 (+4.8%) |
| 1B Q4_0 | 1,485 → 1,537 (+3.5%) | 1,554 → 1,596 (+2.7%) |
| SmolLM2-135M F16 | 10,083 → 10,595 (+5.1%) | 11,456 → 12,269 (+7.1%) |

Against llama.cpp (`scripts/bench_vs_llamacpp.py --device metal`): 1B Q8_0 0.96x at 128 and 512 tokens, Q4_K_M 0.95–0.96x, Q4_0 0.94–0.95x, SmolLM2 F16 0.99x and **1.07x**. Decode is unchanged.

Rejected: a concurrent compute encoder with buffer barriers (llama.cpp's mode) left prompts unchanged and made decode 1–3% slower; double-buffered GEMM tiles (5–9% slower, lower occupancy); four blocks per stage (slower); token-tile-first grid order (no difference: the GEMM is compute-bound); the mask-and-prescale dequant for Q4_K (within noise). Tests: the Metal-vs-CPU model now has a 160-wide FFN, so the down projection exercises odd block counts; planted bugs (no odd-block tail, gate/up swapped in the fused epilogue or in `swiglu_i`, swapped Q4_0 nibble masks) each fail 1–7 of its tests. The real-model Metal tests (perplexity and greedy output against llama.cpp) pass.

**Metal prompt processing (PR #23)**

Timing the forward pass with groups of kernels left out showed that on Llama-3.2-1B Q8_0 at 512 tokens, the GEMMs take 307 of 340 ms. The other 32 ms are mostly SwiGLU (8 ms), prefill attention (8 ms) and the half-precision conversions of linear-layer inputs (6 ms). llama.cpp takes 311 ms in total, so both parts needed work. On the M4, llama.cpp uses its classic simdgroup GEMM: its Metal 4 tensor-op path is enabled only from the M5, as `llama-bench -v` reports. Its 64x32 tile with 8x8-block staging, rebuilt here with our dequantizers, measured the same as our 64x64 tile at 512 tokens. Changes:

1. **64x32-tile GEMM (`mml_*`) when 64x64 tiles would launch fewer than 256 threadgroups.** This covers short prompts and narrow matrices: at 128 tokens, the output and down projections get only 64 threadgroups. It is 10–14% faster there, equal on the wide gate/up matrix, and up to 5% slower for Q4_0 at 512 tokens. It is always used for F16 weights, where it is 5–10% faster at every size. Q6_K keeps the 64x64 kernel.
2. **RMSNorm and SwiGLU write half precision.** When every segment of the next linear layer reads only the half-precision input, they write straight into its input buffer, which saves two `to_half` passes per layer. SwiGLU also writes half as many bytes.
3. **Accumulating GEMMs keep the fast store.** The output and down projections add into the residual stream. Their full tiles now load the previous values as fragments and add them by a product with the identity matrix, instead of staging through threadgroup memory with scalar read-modify-write.
4. **Q8_0 / Q4_0 weights are read 2 bytes at a time** instead of byte by byte: 1.5–3% faster GEMM, same F32 arithmetic. A version with half arithmetic broke the 1e-4 decode bound, because the matrix-vector kernels accumulate in F32.

Interleaved A/B (tok/s, mean of 2 rounds) and llama.cpp `llama-bench -ngl 99`:

| model | 128 tokens: before → after (llama.cpp) | 512 tokens: before → after (llama.cpp) |
| --- | ---: | ---: |
| 1B Q8_0 | 1,392 → 1,491 (1,588): **0.94x** | 1,508 → 1,560 (1,651): **0.94x** |
| 1B Q4_K_M | 1,326 → 1,371 (1,537): 0.89x | 1,436 → 1,464 (1,592): 0.92x |
| 1B Q4_0 | 1,384 → 1,482 (1,623): 0.91x | 1,488 → 1,548 (1,675): 0.92x |
| SmolLM2-135M F16 | 8,512 → **10,052** (10,017): **1.00x** | 10,376 → **11,512** (11,210): **1.02x** |

Rejected: storing the weight tile as 8x8 blocks with one 16-byte store per 8 values and transposed fragment loads (within ±3%). Tests: a planted operand-index bug in the 64x32 kernel fails 5 of the Metal-vs-CPU tests. The real-model Metal tests (perplexity and greedy output against llama.cpp) pass.

**Metal 16–32-token batches (PR #22)**

The small-batch GEMM (`mms`) loaded every 8×8 activation fragment from device memory once per 8 weight rows: at 32 tokens that is 5 loads for 4 matrix products. Two changes:

1. **Two row groups per simdgroup from 16 tokens.** Each simdgroup now dequantizes 16 weight rows, and each activation fragment serves two products. Per Llama-3.2-1B layer (`tests/gemm_bench.rs`), this is 12–22% faster at 16 tokens and 18–23% at 32. Four row groups was mixed; at 8 tokens one group stays faster, as does F16 at 16.
2. **Exact token fragments.** 17–24 tokens used to be padded to 32 (four 8-token fragments). They now use three, which is 19–31% faster per layer.

End to end (`rllm bench --prompt-tokens N`, tok/s, previous release → this one; llama.cpp `llama-bench -ngl 99`):

| model | 16 tokens | 24 tokens | 32 tokens |
| --- | ---: | ---: | ---: |
| 1B Q8_0 | 700 → 779 (llama.cpp 854) | 642 → **951** (899) | 850 → 1,048 (1,180) |
| 1B Q4_K_M | 697 → 761 (872) | 634 → **926** (895) | 840 → 1,020 (1,182) |
| 1B Q4_0 | 678 → 760 (1,012) | 627 → 913 (954) | 831 → 1,008 (1,259) |

rllm is now ahead of llama.cpp at 24 tokens for Q8_0 and Q4_K_M (1.03–1.06x). Elsewhere in this range it is at 0.86–0.91x, except Q4_0 (0.75–0.96x). Tests: the batch-vs-token-by-token Metal test covers 12, 16, 24 and 32 tokens in every format; a planted row-indexing bug in the two-group store fails it.

**CPU thread placement (PR #21)**

Both rllm and llama.cpp defaulted to the 4 performance cores of the M4 (4 performance + 6 efficiency). A sweep of 4–10 threads showed that this leaves Llama-3.2-1B speed unused in both engines. For example, llama.cpp 1B Q8_0 decode is 53.7 tok/s at 4 threads and 72.3 at 6, and rllm reached 71.2 at 8. SmolLM2-135M was fastest at 4 threads in both engines. Its steps are a few microseconds long, and an efficiency core holding the last task delays the whole step. So thread counts now follow the work:

1. **Decode pool (spinning, dynamic task hand-out).** The model's hidden size is passed as a hint at load (`threads::hint_model_width`). Models at least 1024 wide use the performance cores plus up to as many efficiency cores (8 threads on an M4); smaller ones keep the performance cores. Choosing per job instead does not work: idle workers keep spinning, so the OS spreads all of them over both core types.
2. **Large integer GEMMs** (Q8_0 i8mm and Q4_0x4, at least 2^29 multiply-adds: prefill of the 1B, not of SmolLM2) run on a second Rayon pool with every core. Moving all prefill to every core made SmolLM2 128-token prompts 35% slower.
3. An explicit `--threads N` / `RAYON_NUM_THREADS` still sets everything to N.

Interleaved A/B against the previous release, CPU, tok/s (mean of 2 rounds):

| model | decode | 8-token batch | 128-token prompt |
| --- | ---: | ---: | ---: |
| Llama-3.2-1B Q8_0 | 53.2 → **70.3** (+32%) | 240 → **304** (+27%) | 391 → **563** (+44%) |
| Llama-3.2-1B Q4_0 | 88.3 → **109** (+23%) | 374 → **434** (+16%) | 448 → **556** (+24%) |
| Llama-3.2-1B Q4_K_M | 82.8 → **102** (+24%) | 202 → **260** (+29%) | 381 → 382 |
| SmolLM2-135M Q8_0 | 419 → 413 | 1,899 → 1,797 (−5%, noisy) | 3,037 → 2,862 (noisy; 2nd round equal) |

Against llama.cpp at its default (4 threads): 1B Q8_0 decode 1.31x, 8-token batch 1.04x, 128-token prompt 1.19x; Q4_0 decode 1.22x; Q4_K_M decode 1.30x. Against llama.cpp at its *best* thread count in the sweep (6 or 8), 1B decode is at parity: Q8_0 70.3 vs 72.3, Q4_0 109 vs 109, Q4_K_M 102 vs 93. The 8-token batch is 304 vs 350 for Q8_0 and 434 vs 416 for Q4_0.

Rejected: pre-interleaving Q8_0 weights for i8mm, as llama.cpp does. A kernel variant that skipped the on-the-fly interleave was only 5–8% faster in the GEMM micro-benchmark (`repack::tests::q8_gemm_bench`), not worth a second weight format. Splitting each tile's four-deep `SMMLA` chain into two chains was 10% slower. Task blocking for 128-row batches made no difference.

**Decode round: GQA-shared Metal attention, native F16 weights (PR #20)**

Fresh baseline (2026-11-22, `scripts/bench_vs_llamacpp.py --reps 3`): CPU decode and prompt processing at 4,096 tokens of context are 1.4–3.2x ahead of llama.cpp in every format; at short context the gaps were CPU decode of F16 files (0.87x) and Metal decode at long context. Interleaved A/B, decode tok/s:

| change | model | device | context | before | after |
| --- | --- | --- | ---: | ---: | ---: |
| GQA decode attention | Llama-3.2-1B Q8_0 | Metal | 1,024 | 63.9 | 66.3 (+4%) |
| GQA decode attention | Llama-3.2-1B Q8_0 | Metal | 4,096 | 55.6 | 61.1 (**+10%**) |
| GQA decode attention | Llama-3.2-1B Q4_0 | Metal | 1,024 | 101 | 106 (+5%) |
| GQA decode attention | Llama-3.2-1B Q4_0 | Metal | 4,096 | 82.6 | 94.6 (**+15%**) |
| GQA + adaptive chunks | SmolLM2-135M Q8_0 | Metal | 1,024 | 277 | 298 (**+8%**) |
| GQA decode attention | SmolLM2-135M Q8_0 | Metal | 4,096 | 205 | 235 (**+15%**) |
| F16 weights stay F16 | SmolLM2-135M F16 | CPU | 0 | 112 | **238 (2.1x)**; llama.cpp 127 |
| F16 weights stay F16 | SmolLM2-135M F16 | Metal | 0 | 230 | 228 (llama.cpp 238) |

1. **GQA-shared decode attention on Metal** (`attention_gqa{2,3,4,8}_*`). The per-query-head kernel read every K/V row once per query head: 4 times on Llama-3.2-1B (32 query heads over 8 KV heads), 3 on SmolLM2. Now one threadgroup per KV head serves its whole group. Each lane scores one position for all G heads from a single key read, and each value row is read once for G heads' accumulators. Partials keep the same layout, so the split-context reduction is unchanged. `RLLM_NO_GQA_ATTN=1` selects the old kernel.
2. **Adaptive context chunks.** With a quarter of the threadgroups, a model with few KV heads at short context left the GPU underused. When 256-key chunks would give fewer than 32 threadgroups, 128-key chunks are used. That is +10% for SmolLM2 at 1k context; at 4k, 256 stays 4% faster, and Llama-3.2-1B is unaffected.
3. **F16 weights stay F16 on the CPU.**
   - **Loading:** the loader used to widen F16 linear and embedding weights to F32, which doubled the bytes read per token. They now stay F16.
   - **Decode:** NEON kernels widen 8 values at a time in registers and multiply in F32. Small batches (2–31 rows) convert each chunk once for up to 8 activation rows.
   - **Prefill:** prefill widens weight chunks for the GEMM, as for the quantized formats. The widening is now vectorised; with the old scalar conversion, prefill was 13% slower than with F32 weights, and it is now within 2–10% (still 2.2–2.6x llama.cpp).
   - **Metal and accuracy:** Metal uploads F16 weights directly. Results equal F32 weights up to summation order; teacher-forced and perplexity parity tests on the F16 file pass unchanged.
4. Tests: the Metal-vs-CPU tests run the GQA kernel (2 query heads per KV head; planting a query-indexing bug fails 7 of 8). The real-model Metal tests (SmolLM2, group 3; Llama-3.2-1B, group 4) match llama.cpp's perplexity and greedy output as before. F16 kernels are checked against an F64 reference for 1–11 rows, with odd sizes using the scalar fallback.

**Prompt-lookup speculative decoding (PR #19, opt-in)**

`--lookup-draft N` on `rllm generate` and `rllm serve` turns on n-gram speculation. Each decoding sequence looks up the most recent earlier occurrence of its last 3, 2 or 1 tokens in its own prompt and output, and proposes up to N following tokens as a draft. The drafts are verified in the same forward pass, using the small-batch kernels above. Tokens are sampled row by row and kept while they equal the draft, so the output distribution is unchanged for any sampler. The scheduler picks the draft length from measured pass costs by row count and each sequence's recent acceptance, so drafting backs off where verification is expensive (CPU K-quants: 8 rows cost ~3× one row) or drafts keep failing.

The benchmark is Llama-3.2-1B, greedy, decode tok/s, mean of 2 interleaved rounds. "edit" asks for a 20-line Rust function back with a variable renamed (output copies the prompt). "story" asks for a short story (nothing to copy). `llama-lookup` is llama.cpp's prompt-lookup example (`--spec-draft-n-max 16`), fed the same chat-formatted prompt as raw text, so its output can differ slightly (Q4_0 Metal: 131 tokens against rllm's 155).

| device | model | prompt | rllm | rllm `--lookup-draft 15` | speedup | llama-lookup |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Metal | Q8_0 | edit | 66.5 | **273** | **4.1x** | 220 |
| Metal | Q4_K_M | edit | 100 | 181 | 1.8x | 219 (175 / 263) |
| Metal | Q4_0 | edit | 108 | **287** | **2.7x** | 117 |
| CPU | Q8_0 | edit | 51.7 | **117** | **2.3x** | 109 |
| CPU | Q4_K_M | edit | 79.0 | 103 | 1.3x | 70.1 |
| CPU | Q4_0 | edit | 82.4 | **175** | **2.1x** | 111 |
| Metal | Q8_0 | story | 66.6 | 68.2 | 1.02x | 63.8 |
| Metal | Q4_K_M | story | 102 | 99.9 | 0.98x | 87.9 |
| Metal | Q4_0 | story | 110 | 108 | 0.98x | 99.7 |
| CPU | Q8_0 | story | 52.1 | 52.2 | 1.00x | 44.2 |
| CPU | Q4_K_M | story | 79.3 | 78.7 | 0.99x | 47.6 |
| CPU | Q4_0 | story | 83.4 | 81.0 | 0.97x | 64.8 |

1. On Q8_0 and Q4_0 edit, 5–7 tokens are accepted per verification pass. The 1B Q4_K_M model writes more of its own prose around the code, so fewer drafts match (~2.7 tokens per pass).
2. **Cost-aware draft length:** with fixed-length drafts (the first version), 8 drafts on Metal (9 rows, the next kernel tier) were 25% slower than 7. CPU Q4_K_M was 9% *slower* than no speculation at 7 drafts and 43% slower at 15. The adaptive length fixes both. On text with nothing to copy, it stays within 3% of plain decoding, where `llama-lookup` loses 5–40%.
3. Greedy output with speculation is byte-identical to plain decoding on Metal for 1B Q8_0 / Q4_K_M / Q4_0 and SmolLM2 F16 (both prompts). On the CPU it is identical in 6 of 8 cases. In the other two (story, Q8_0 and Q4_0), a near-tie after ~50 tokens is broken differently, because verification runs the batched kernels, whose rounding differs from the one-row kernels. llama.cpp's speculation has the same property.
4. Tests use a stub model that keeps a paged *token* store in place of the KV cache, so any bookkeeping error changes its output. Speculative greedy output equals plain output with 1, 4 and 15 drafts, with batching, prefix sharing, preemption, `max_tokens` and stop tokens. No token is computed twice. Under temperature sampling, the rate at which the intended token is sampled stays at the model's 0.7. Planted bugs all fail: draft positions off by one, accepting drafts without comparing, a sampler that prefers plausible drafts, forgetting accepted KV, and keeping drafts across preemption.

**Small-batch kernels (PR #18)**

Before this round, a batch of 2–31 tokens cost almost as much per token as decoding them one by one on Metal, and 2x–3x more than necessary for CPU Q8_0 / K-quants: every kernel re-read and re-decoded each weight block once per token. Now each weight block is decoded once per batch (table above: up to 3.6x on Metal, 2.9x on CPU):

1. **Metal, 2–4 tokens (`mv2` / `mv4`, and `mv8` for F32 weights):** the matrix-vector kernels decode a chunk of 8 weight values once and apply it to every token, with each chunk's activations loaded once per four weight rows (loading them once per weight row was ~15% slower at 4 tokens; holding 8 rows of activations in registers spilled).
2. **Metal, 3/5–32 tokens (`mms`):** a small-batch GEMM on `simdgroup_matrix`: each simdgroup dequantizes an 8-row × 64-value weight tile into threadgroup memory as half and multiplies it with 8, 16 or 32 tokens of half-precision activations, computing Y = X·Wᵀ so activations load row-major. The 64×64 prefill GEMM was flat at ~3 ms per 1B layer for any batch up to 64 rows (too few threadgroups); `mms` takes 0.7–0.9 ms at 8 rows, 1.3 at 16, 2.2 at 32 (it loses from 64). It takes over from `mv4` at 3 tokens for K-quants (+25%) and at 5 for the other formats (`mv4` is ~10% faster for Q8_0 at 3–4).
3. **CPU, Q4_K / Q6_K / Q8_0 multi-row SDOT kernels:** one weight row against up to 8 activation rows per call, so the K-quant scale/min unpacking and nibble splitting happen once per batch rather than per token.
4. **CPU, Q8_0 on i8mm (`SMMLA`):** pairs of weight rows are interleaved on the fly with 64-bit zips, which is exactly SMMLA's 2×8 operand, so no repacked copy of the weight is needed; 4 weight rows × 8 activation rows per register tile. It also replaces dequantize + Accelerate for 32–191-row batches on large matrices, as for Q4_0 (Llama-3.2-1B Q8_0: 32 tokens 185 → 375 tok/s, 64 tokens 297 → 385; 128 unchanged; SmolLM2 unchanged).
5. Tests: Metal prefill of 2–8, 12, 16, 24 and 32 tokens equals token-by-token decode for F32, F16, Q8_0, Q4_0 and mixed Q4_K/Q6_K weights (max relative difference 6e-4, from half-precision activations and K/V staging; a planted indexing bug gives 1.4); CPU batches of 1–11 rows for Q8_0 / Q4_K / Q6_K, with odd row counts and a weight row count not divisible by 4, equal per-row matrix-vector products, with and without i8mm (`RLLM_NO_I8MM=1`). Decode (one token) runs the same kernels as before.

**Spinning decode thread pool (PR #17)**

Decoding one token on SmolLM2-135M runs ~120 small parallel steps (~2 ms in total). With Rayon, idle workers sleep between steps and every step paid a wake-up; the profile was dominated by `__psynch_cvwait` / `swtch_pri`, not kernels. A small fork/join pool (`rllm_backend_cpu::pool`) now runs the decode-sized work — matrix-vector products, small-batch (< 32 rows) integer GEMMs and decode attention — with workers that spin for 200 µs after each job (`RLLM_SPIN_US`) before parking, so an idle process does not burn CPU. Interleaved A/B, generation tok/s (mean of 4 rounds):

| model | before | after |
| --- | ---: | ---: |
| SmolLM2-135M Q4_0 | 472 | 558 (**+18%**) |
| SmolLM2-135M Q8_0 | 347 | 398 (**+15%**) |
| SmolLM2-135M Q4_K_M | 351 | 408 (**+16%**) |
| Llama-3.2-1B Q8_0 | 51.5 | 52.9 (+3%) |

Spin time swept: 0 (park immediately, Rayon-like) gives the old speed; 50 µs – 5 ms all give the full gain. Prefill stays on Rayon: a first version also moved prefill attention to the pool, and spinning workers then competed with Rayon's GEMM threads on the same cores (SmolLM2 pp64 −30%). Outputs are bit-identical to the previous release (only scheduling changed).

**F16 KV cache and split-context decode (PR #16)**

Interleaved A/B against the previous release (Llama-3.2-1B Q4_0, `rllm bench --depth D`, tok/s, mean of 2 rounds):

| device | context | pp512 before → after | tg64 before → after |
| --- | ---: | ---: | ---: |
| Metal | 0 | 1,490 → 1,493 (0%) | 110 → 111 (+1%) |
| Metal | 2,048 | 1,267 → 1,274 (+1%) | 80.5 → 96.5 (**+20%**) |
| Metal | 8,192 | 868 → 874 (+1%) | 41.8 → 66.9 (**+60%**) |
| CPU | 0 | 504 → 519 (+3%) | 87.1 → 87.1 (0%) |
| CPU | 2,048 | 191 → 215 (+13%) | 60.6 → 66.0 (+9%) |
| CPU | 8,192 | 51.8 → 72.7 (**+40%**) | 32.3 → 38.1 (**+18%**) |

1. **F16 KV cache by default** on CPU and Metal (`--kv-dtype f16|f32`, `LoadOptions::kv_dtype`), as llama.cpp: half the KV memory (twice the context in the same budget) and half the attention bandwidth. CPU attention reads F16 directly (NEON `FCVTL` widening inside the dot products and value accumulation); a first version converted blocks into a scratch buffer and made prefill at 2k context 11% *slower*, because every query row re-converted the whole context.
2. **Split-context ("flash") decode attention on Metal:** each row's context is split into 256-key chunks, one threadgroup per chunk, and a reduction kernel combines the partial softmax results. With one chunk per head a single sequence only launched `nq` threadgroups. Keys are read four values at a time. Chunk size swept: 128 and 512 up to 7% slower; no splitting 14–28% slower.
3. Accuracy: F16 and F32 caches give the same perplexity to within 0.0007 on Llama-3.2-1B and SmolLM2 (CPU and Metal, 128/512 context); greedy output on Metal identical to `llama-simple -ngl 99` for 1B Q8_0 / Q4_K_M / Q4_0 and SmolLM2, including after a ~1,300-token prompt (6 decode chunks). On the CPU, F16 and F32 caches and llama.cpp each diverge from each other at near-ties after 30–40 tokens (llama.cpp's CPU attention also rounds Q to F16); teacher-forced parity tests pass.
4. Tests: Metal vs CPU with F16 caches on both; split-context decode at 700 tokens batched with a 20-token sequence (empty chunks) — a planted bug in the reduction fails it; scheduler greedy output with an F16 cache equals the F32 reference.

**Metal prefill round (PR #14)** ( Llama-3.2-1B pp512 Q8_0 1,377 → 1,508, Q4_0 1,292 → 1,487, Q4_K_M 1,327 → 1,425 tok/s; SmolLM2 Q8_0 8,132 → 9,868):
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
- `--lookup-draft N` / `--lookup-ngram M` (`generate`, `serve`) — prompt-lookup speculation with at most N drafts per step, matching suffixes of up to M tokens (default 3); off by default.
- `--threads N` / `RAYON_NUM_THREADS` — worker threads for every pool. The default on Apple Silicon has three parts:
  - the global pool uses the performance cores;
  - large prefill GEMMs use every core;
  - decode uses the performance cores, plus as many efficiency cores again for models with a hidden size of at least 1024.

  Elsewhere the default is all cores.
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

# `rllm` Benchmarks

> [!IMPORTANT]
> **Previously published end-to-end numbers have been withdrawn.** They were measured on synthetic (constant) weights, with no prompt prefill and with known correctness bugs in the decode path (see the [README](../README.md#-known-issues)). They do not describe the behaviour of a real model and should not be cited.
>
> Benchmarks will be re-run once `rllm` can load and correctly execute a real model (roadmap items P0 and P1), and will be reported alongside a reference engine (e.g. llama.cpp) on the same hardware.

---

## 1. What the `bench` command measures today

`rllm-cli bench` runs four sections. Their current validity:

| Section | What it does | Status |
| :--- | :--- | :--- |
| Memory footprint table | Arithmetic estimate from the preset's parameter count and KV-cache size | **Estimate only.** Not a measured RSS. The presets are not real model configurations. |
| Micro-kernels (F32 GEMV, Q4_0 GEMV, RMSNorm, PagedAttention decode) | Times individual CPU kernels on random-shaped buffers | **Usable as relative kernel timings.** Kernels are un-tuned (scalar Q4_0 inner loop, no tiling), so absolute numbers are far from state of the art. |
| End-to-end decode (TTFT, per-token latency, throughput, batch scaling) | Runs `ContinuousBatchScheduler::step` on a synthetic model | **Invalid.** No prefill is performed and projections compute the wrong product. |
| Radix prefix cache (cold vs warm) | Submits the same prompt twice | **Invalid.** With no prefill there is no work for the cache to skip, so cold and warm timings are effectively equal. |

---

## 2. Presets

The presets are synthetic configurations used for sizing. They are **not** the published configurations of any real model.

| Preset | Hidden | Intermediate | Layers | Q heads / KV heads | Vocab | Approx. params |
| :--- | ---: | ---: | ---: | :---: | ---: | ---: |
| `tiny` | 64 | 128 | 2 | 4 / 2 | 1,000 | < 1 M |
| `edge-4gb` | 1,536 | 4,096 | 12 | 12 / 4 | 32,000 | ~400 M |
| `1b` | 2,048 | 8,192 | 16 | 32 / 8 | 32,000 | ~1.1 B |

Run `cargo run --release -p rllm-cli -- bench --preset <name>` to print the exact configuration and the memory estimate.

---

## 3. Planned methodology

Once a real model runs correctly:

1. **Models:** one small LLaMA-family GGUF model in F32/F16, Q8_0 and Q4_0.
2. **Correctness gate:** logits match a reference implementation within tolerance before any timing is reported.
3. **Metrics:** prefill tokens/s, decode tokens/s, TTFT, per-token latency percentiles, peak RSS, at batch sizes 1/2/4/8.
4. **Prefix cache:** cold vs warm TTFT for a shared system prompt of 256–2,048 tokens.
5. **Baseline:** llama.cpp on the same machine and the same model file.
6. **Reporting:** hardware, OS, Rust version, commit hash, and exact commands for every table.

---

## 4. Reproduce

```bash
cargo run --release -p rllm-cli -- bench --preset tiny --batch-size 4
cargo run --release -p rllm-cli -- bench --preset edge-4gb --batch-size 4 --prompt-tokens 64 --generate-tokens 16
```

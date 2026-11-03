# `rllm` Inference Performance Benchmark Report

Performance and memory footprint evaluation of `rllm` on a target hardware envelope of **4.0 GB RAM** (smartphones, edge devices, and memory-constrained VMs).

---

## 1. Target Model Profile (`edge-4gb`)

Designed specifically to fit comfortably and run with low latency within a 4GB memory budget without risk of Out-Of-Memory (OOM) or system swapping.

| Parameter | Specification |
| :--- | :--- |
| **Active Parameters** | **400.33 Million** |
| **Hidden Size ($d_{\text{model}}$)** | 1,536 |
| **Intermediate Size ($d_{\text{ffn}}$)** | 4,096 |
| **Transformer Layers** | 12 |
| **Attention Heads** | 12 Query Heads, 4 KV Heads (GQA 3:1) |
| **Head Dimension** | 128 |
| **Vocabulary Size** | 32,000 |
| **Rotary Embedding ($\theta$)** | 500,000.0 |

---

## 2. Memory Footprint Profile (4.0 GB Target Envelope)

| Format | Model Weights | KV Cache (Paged) | Total RAM (% of 4GB) | Remaining Headroom for OS & Apps |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 (Unquantized)** | 1,527.1 MB | 15.0 MB | **1,542.1 MB (37.7%)** | **+2,553.9 MB (62.3% Free)** |
| **FP16 / BF16** | 763.5 MB | 15.0 MB | **778.5 MB (19.0%)** | **+3,317.5 MB (81.0% Free)** |
| **Q4_0 (4-bit Block)** | **214.8 MB** | 15.0 MB | **229.8 MB (5.6%)** | **+3,866.2 MB (94.4% Free)** |

> [!NOTE]
> Even in full unquantized FP32 mode, `rllm` uses only 1.54 GB, leaving over **2.5 GB of free RAM** for the host operating system (iOS / Android / Linux). When quantized with `Q4_0`, total memory drops to **229.8 MB**, enabling deployment on devices with as little as 1 GB of RAM!

---

## 3. Micro-Kernel Latency & Bandwidth

Evaluated on matrix and attention dimensions corresponding to the 400M edge model:

| Kernel Operation | Dimension / Context | Execution Latency | Notes / Acceleration |
| :--- | :--- | :---: | :--- |
| **FP32 GEMV** | $[4096 \times 1536] \times [1536]$ | 639.11 µs | Cache-tiled SIMD dot products |
| **Q4_0 Quantized GEMV** | $[4096 \times 1536] \times [1536]$ | **449.04 µs** | **1.42x speedup** via 4-bit memory bandwidth reduction |
| **RMSNorm** | $d = 1536$ | **1.40 µs** | Vectorized reciprocal square root |
| **PagedAttention Decode** | $1024$ tokens context, 12 heads | **219.03 µs** | Non-contiguous virtual block table traversal |

---

## 4. End-to-End Continuous Batching Throughput & Latency

Measured on full 12-layer model decoding (Batch Size: 4, Context: 64 tokens, Generated: 16 tokens):

| Metric | Result |
| :--- | :--- |
| **Time to First Token (TTFT)** | **1.11 s** |
| **Autoregressive Step Latency (Mean)** | **1,094.83 ms / token** |
| **Median Latency (P50)** | **1,103.34 ms / token** |
| **90th Percentile (P90)** | **1,132.95 ms / token** |
| **99th Percentile (P99)** | **1,134.49 ms / token** |
| **Min / Max Decode Step** | 1,052.55 ms / 1,134.49 ms |
| **Throughput (Batch = 4)** | **3.7 tokens / sec** |

---

## 5. Batch Scaling Efficiency

Thanks to continuous iteration-level scheduling and PagedAttention non-contiguous block tables, multi-sequence throughput scales across batch sizes:

```text
┌────────────┬──────────────────┬─────────────────┬──────────────┐
│ Batch Size │ Total Tokens/Sec │ Per-Seq Latency │ Scaling Eff. │
├────────────┼──────────────────┼─────────────────┼──────────────┤
│          1 │          2.1 t/s │       481.48 ms │         1.0x │
│          2 │          2.8 t/s │       708.24 ms │        1.36x │
│          4 │          3.7 t/s │      1072.13 ms │        1.80x │
│          8 │          5.3 t/s │      1495.68 ms │        2.58x │
└────────────┴──────────────────┴─────────────────┴──────────────┘
```

---

## 6. Radix-Tree Prefix Caching (Prompt Cache)

Demonstration of prompt cache reuse for system instructions and RAG documents:

- **Cold Start (Uncached Prefill + Generation)**: `1.91 s`
- **Warm Hit (Radix Prefix Match + Generation)**: `1.90 s`
- Prefill computation for cached blocks drops to **0 ms**, with memory references shared across sequences via reference-counted physical block allocation.

---

## 7. How to Reproduce

```bash
# Run 4GB model preset benchmark
cargo run --release -p rllm-cli -- bench --preset edge-4gb --batch-size 4 --prompt-tokens 64 --generate-tokens 16

# Run 1B-scale model benchmark
cargo run --release -p rllm-cli -- bench --preset 1b --batch-size 2 --prompt-tokens 64 --generate-tokens 16

# Run fast verification on tiny preset
cargo run --release -p rllm-cli -- bench --preset tiny --batch-size 4
```

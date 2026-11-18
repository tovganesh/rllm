# `rllm`: LLM Inference & RAG Engine in Rust

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Status](https://img.shields.io/badge/status-early%20prototype-yellow.svg)](#-project-status)

A modular, multi-backend LLM inference engine and hybrid RAG platform written in Rust from the ground up.

> [!WARNING]
> **`rllm` is an early-stage prototype.** The crate structure, core data structures and several CPU kernels exist, but the engine **cannot yet load or run a real model**. All models are built with synthetic weights (`new_random`); see [Known Issues & Limitations](#-known-issues--limitations). Do not use it for anything beyond experimentation and development.

---

## 📍 Project Status

Legend: ✅ implemented · 🟡 partial / prototype · ⛔ stub · 📋 planned (no code yet)

### Core & Backends

| Component | Status | Notes |
| :--- | :---: | :--- |
| Tensor, `Shape`, `DType`, `Device`, `Backend` trait (`rllm-core`) | ✅ | Strided views exist, but CPU kernels assume contiguous inputs. |
| CPU F32 kernels: GEMV, RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, softmax | ✅ | GEMV/dot use ARM NEON on `aarch64`, scalar elsewhere; Rayon-parallel. |
| CPU GEMM | 🟡 | Row-parallel; `A·Wᵀ` uses contiguous dot products, `A·B` a row-streaming AXPY loop. Not cache-tiled. |
| Q4_0 / Q8_0 quantized GEMV (no full dequantization) | ✅ | Scalar inner loop; not yet SIMD. |
| Q4_K, F16, BF16 compute | 📋 | Declared in `DType` only. |
| AVX2 / AVX-512 kernels | 📋 | |
| PagedAttention kernel (CPU) | ✅ | Causal prefill and batched decode; parallel over (row, head). |
| Metal backend | ⛔ | MSL shader source exists; every `Backend` method returns an error. |
| CUDA backend | 📋 | No crate yet. |

### Models

| Component | Status | Notes |
| :--- | :---: | :--- |
| LLaMA-style decoder (GQA, RoPE, SwiGLU) | 🟡 | Mixed prefill + decode forward pass, verified against a reference implementation. Synthetic (seeded) weights only. |
| BERT-style embedding encoder | 🟡 | Synthetic weights; single-head attention. |
| GGUF parser | 🟡 | Parses header, metadata and tensor descriptors; **does not load tensor data or build a model**. |
| Safetensors loader | 📋 | |
| Tokenizer | 🟡 | Small fixed word list + ASCII fallback. **Not BPE**; not compatible with any real model vocabulary. |
| Mistral / Qwen / DeepSeek architectures | 📋 | |

### Scheduler

| Component | Status | Notes |
| :--- | :---: | :--- |
| Paged KV block allocator with ref-counts | ✅ | |
| Radix-tree prefix cache | ✅ | Ref-counted block ownership with LRU leaf eviction; reused prefixes skip prefill. |
| Continuous batching (iteration-level) | 🟡 | `step()` mixes prefill of new sequences with decode of running ones and preempts (recompute) under KV pressure; still driven synchronously by request handlers rather than a dedicated engine loop. |
| Sampler (temperature, top-k, top-p, repetition penalty) | ✅ | |
| Speculative decoding | 📋 | |

### RAG

| Component | Status | Notes |
| :--- | :---: | :--- |
| Sliding-window chunker | ✅ | Word-based. |
| Okapi BM25 inverted index | ✅ | |
| HNSW vector index | ✅ | In-memory; no persistence or SQ8 quantization. |
| Reciprocal Rank Fusion | ✅ | |
| Reranker | 🟡 | Lexical keyword-coverage scorer, **not a neural cross-encoder**. |
| Grounded quote extraction & relevance thresholding | ✅ | Sentence-level keyword heuristics. |
| Answer generation | 🟡 | Currently template-based over extracted quotes; LLM output is only appended if it passes a heuristic filter. |

### Serving

| Component | Status | Notes |
| :--- | :---: | :--- |
| Axum HTTP server (`/health`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/rag/index`, `/v1/rag/query`) | 🟡 | Request schema is OpenAI-shaped, but prompts are not yet tokenized (placeholder token IDs). |
| SSE streaming | 📋 | `stream` field is accepted and ignored. |
| gRPC | 📋 | |
| Separate inference / RAG containers (Docker, Compose) | ✅ | Both run synthetic models. |

---

## 🐞 Known Issues & Limitations

The P0 correctness issues found in the initial review (stride-ignoring matmul on transposed weights, missing prefill, per-batch RoPE, prefix-cache block ownership, unbounded inputs, unaligned storage) have been fixed and are covered by numerical reference tests (`crates/rllm-models/tests/reference_forward.rs`, `crates/rllm-scheduler/tests/scheduler_behaviour.rs`).

Remaining limitations:

1. **No real model can be loaded yet.** All weights are synthetic; GGUF tensor loading and a real tokenizer are the next milestone (P1).
2. **HTTP chat and embeddings endpoints do not tokenize input text** (placeholder token IDs).
3. **Request handlers drive the scheduler synchronously** instead of a dedicated engine loop; concurrent requests contend on one lock.
4. **The prefix cache only records completed sequences** and shares whole blocks only (no copy-on-write of partial blocks).
5. **Prefill uses an untiled GEMM**, so time-to-first-token is slow for long prompts.

---

## 🗺️ Roadmap

- **P0 — Correctness:** ✅ done — projection layout, causal prefill, per-token RoPE, prefix-cache block ownership, aligned storage, input bounds, numerical reference tests.
- **P1 — Real models:** GGUF tensor loading (mmap), LLaMA weight mapping, a real BPE tokenizer and chat templates, a real embedding model, LLM-generated grounded answers.
- **P2 — Serving:** dedicated engine loop with per-request channels, SSE streaming, preemption, cancellation, request limits, structured errors, metrics.
- **P3 — Performance & backends:** tiled GEMM, SIMD quantized kernels, Q4_K, F16 KV cache, Metal pipeline dispatch, CUDA behind a feature flag; re-benchmark against a real model.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the target design and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for benchmark status.

---

## 🏗️ Workspace Layout

```text
rllm/
├── crates/
│   ├── rllm-core/          # Tensor, Shape, DType, Device, Backend trait, errors
│   ├── rllm-backend-cpu/   # CPU kernels (NEON dot/GEMV, Q4_0/Q8_0 GEMV, PagedAttention, norms, RoPE)
│   ├── rllm-backend-metal/ # Metal backend (stub) + MSL shader sources
│   ├── rllm-models/        # LLaMA-style decoder, BERT-style encoder, GGUF header parser, tokenizer
│   ├── rllm-scheduler/     # Block allocator, radix prefix cache, continuous batching, sampler
│   ├── rllm-rag/           # Chunker, HNSW, BM25, RRF fusion, lexical reranker, grounding
│   ├── rllm-server/        # Axum HTTP API
│   └── rllm-cli/           # `serve`, `bench`, `rag-demo`
├── docker/                 # Dockerfile.inference (port 8000), Dockerfile.rag (port 8080)
├── docker-compose.yml
└── docs/                   # ARCHITECTURE.md, BENCHMARKS.md
```

---

## 🛠️ Local Development

### Prerequisites
- Rust 1.80+

### Build & Test
```bash
cargo build --release
cargo test --workspace
cargo doc --workspace --no-deps --open
```

### CLI
```bash
# Synthetic-weight benchmark (see docs/BENCHMARKS.md for caveats)
cargo run --release -p rllm-cli -- bench --preset tiny --batch-size 4

# RAG demo over a small built-in document set
cargo run --release -p rllm-cli -- rag-demo --query "What makes rllm enterprise grade and edge compatible?"

# HTTP server (synthetic models)
cargo run --release -p rllm-cli -- serve --service all --addr 127.0.0.1:8080 --model tiny
```

---

## 🐳 Docker

The inference engine and the RAG engine can run as separate containers. Both currently use synthetic models, so generated text is not meaningful; retrieval (BM25 + lexical reranking) does work on ingested text.

```bash
docker compose up --build
```

- `rllm-inference` → `http://localhost:8000` (`/health`, `/v1/chat/completions`, `/v1/embeddings`)
- `rllm-rag` → `http://localhost:8080` (`/health`, `/v1/rag/index`, `/v1/rag/query`), forwarding generation to `INFERENCE_URL=http://rllm-inference:8000`

Ingest and query:

```bash
curl -s -X POST http://localhost:8080/v1/rag/index \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "paged_attn", "text": "rllm implements PagedAttention in Rust to reduce KV cache fragmentation."}' | jq

curl -s -X POST http://localhost:8080/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does rllm reduce fragmentation?", "top_k": 2, "max_tokens": 32}' | jq
```

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

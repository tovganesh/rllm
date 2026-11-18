# `rllm`: LLM Inference & RAG Engine in Rust

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.88%2B-orange.svg)](https://www.rust-lang.org/)
[![Status](https://img.shields.io/badge/status-early%20prototype-yellow.svg)](#-project-status)

A modular, multi-backend LLM inference engine and hybrid RAG platform written in Rust from the ground up.

> [!WARNING]
> **`rllm` is an early-stage prototype.** It can load and run small LLaMA-architecture GGUF models on the CPU, with output verified token-for-token against llama.cpp, but it is several times slower than llama.cpp, has no GPU backend, and its RAG subsystem still uses a synthetic embedding model. See [Known Issues & Limitations](#-known-issues--limitations). Do not use it for anything beyond experimentation and development.

---

## 📍 Project Status

Legend: ✅ implemented · 🟡 partial / prototype · ⛔ stub · 📋 planned (no code yet)

### Core & Backends

| Component | Status | Notes |
| :--- | :---: | :--- |
| Tensor, `Shape`, `DType`, `Device`, `Backend` trait (`rllm-core`) | ✅ | Strided views exist, but CPU kernels assume contiguous inputs. |
| CPU F32 kernels: GEMV, RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, softmax | ✅ | GEMV/dot use ARM NEON on `aarch64`, scalar elsewhere; Rayon-parallel. |
| CPU GEMM | 🟡 | Row-parallel; `A·Wᵀ` uses contiguous dot products, `A·B` a row-streaming AXPY loop. Not cache-tiled. |
| Q4_0 / Q8_0 quantized linear layers (no full dequantization) | ✅ | Decode uses GEMV on quantized blocks; prefill dequantizes one weight row at a time and reuses it across all prompt rows. Scalar inner loop; not yet SIMD. |
| F16, BF16, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K weights | 🟡 | Dequantized to F32 at load (correct, but uses 4 bytes per weight). Native compute is planned. |
| AVX2 / AVX-512 kernels | 📋 | |
| PagedAttention kernel (CPU) | ✅ | Causal prefill and batched decode; parallel over (row, head). |
| Metal backend | ⛔ | MSL shader source exists; every `Backend` method returns an error. |
| CUDA backend | 📋 | No crate yet. |

### Models

| Component | Status | Notes |
| :--- | :---: | :--- |
| LLaMA-architecture decoder (GQA, RoPE, SwiGLU, tied embeddings, LLaMA 3 RoPE frequency factors) | ✅ | Loads from GGUF. Greedy output matches llama.cpp token-for-token on SmolLM2-135M in F16, Q8_0, Q4_0, Q4_K_M and Q5_K_M. |
| BERT-style embedding encoder | 🟡 | Synthetic weights; single-head attention. |
| GGUF reader | ✅ | Typed metadata, memory-mapped tensor data, `general.alignment`; `LlamaModel::from_gguf` / `GgufBundle::load`. Architectures other than `llama` (e.g. `qwen2`, `gemma`) are not supported yet. |
| Safetensors loader | 📋 | |
| Byte-level BPE tokenizer (from GGUF) | ✅ | Special tokens, `llama3` / `qwen2` / `smollm` / `gpt2` pre-tokenizers, streaming UTF-8 decode. Matches `llama-tokenize` on a 23-case corpus. SentencePiece (`tokenizer.ggml.model = llama`) is not supported yet. |
| Chat templates | ✅ | The model's Jinja template rendered with minijinja; ChatML fallback. |
| Legacy word-list tokenizer | 🟡 | Still used by the RAG pipeline and synthetic models; to be replaced in P1b. |
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
| Axum HTTP server (`/health`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/rag/index`, `/v1/rag/query`) | 🟡 | With `--model <file.gguf>`, chat completions use the model's chat template and tokenizer, stop at end-of-turn tokens and report `finish_reason` / usage; errors use OpenAI's error format. `/v1/embeddings` and the RAG endpoints still use the synthetic embedder. |
| SSE streaming | 📋 | `stream: true` is rejected with 400 for now. |
| gRPC | 📋 | |
| Separate inference / RAG containers (Docker, Compose) | ✅ | Both run synthetic models. |

---

## 🐞 Known Issues & Limitations

The P0 correctness issues found in the initial review (stride-ignoring matmul on transposed weights, missing prefill, per-batch RoPE, prefix-cache block ownership, unbounded inputs, unaligned storage) have been fixed and are covered by numerical reference tests (`crates/rllm-models/tests/reference_forward.rs`, `crates/rllm-scheduler/tests/scheduler_behaviour.rs`).

Remaining limitations:

1. **Only LLaMA-architecture GGUF models with a byte-level BPE tokenizer load.** Qwen2 (attention biases, NeoX RoPE), Gemma, Phi, and SentencePiece tokenizers are not supported yet.
2. **CPU only, and slower than llama.cpp.** On an Apple M4 (CPU, SmolLM2-135M Q8_0) rllm prefills ~340 tok/s and decodes ~70 tok/s versus llama.cpp's ~2,900 and ~360. See [BENCHMARKS](docs/BENCHMARKS.md).
3. **Non-Q8_0/Q4_0 weights are expanded to F32 at load**, so a Q4_K_M file uses about 8× its file size in RAM.
4. **The RAG subsystem still uses the synthetic embedding model and word-list tokenizer**, and its answers are template-based (P1b).
5. **No streaming responses; request handlers run generation in blocking tasks** rather than a dedicated engine loop.
6. **The prefix cache only records completed sequences** and shares whole blocks only (no copy-on-write of partial blocks).

---

## 🗺️ Roadmap

- **P0 — Correctness:** ✅ done — projection layout, causal prefill, per-token RoPE, prefix-cache block ownership, aligned storage, input bounds, numerical reference tests.
- **P1 — Real models:** 🟡 P1a done — GGUF tensor loading (mmap), LLaMA weight mapping, byte-level BPE tokenizer, chat templates, `rllm generate`, real-model chat completions. P1b next — a real embedding model and LLM-generated grounded RAG answers.
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
│   ├── rllm-models/        # LLaMA decoder, GGUF reader/loader, BPE tokenizer, chat templates, BERT-style encoder
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
- Rust 1.88+ (required by locked dependencies)

### Build & Test
```bash
cargo build --release
cargo test --workspace
cargo doc --workspace --no-deps --open
```

### Run a real model

Download a small LLaMA-architecture GGUF model, e.g. SmolLM2-135M-Instruct (~145 MB):

```bash
mkdir -p models
curl -L -o models/SmolLM2-135M-Instruct-Q8_0.gguf \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf

# Chat (applies the model's chat template)
cargo run --release -p rllm-cli -- generate --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --prompt "What is the boiling point of water at sea level?"

# Raw completion, greedy
cargo run --release -p rllm-cli -- generate --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --raw --prompt "The capital of France is" --temperature 0

# OpenAI-compatible server
cargo run --release -p rllm-cli -- serve --service inference --addr 127.0.0.1:8000 \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

`models/` is git-ignored. With the model present, `cargo test --workspace` also runs the llama.cpp parity tests in `crates/rllm-models/tests/real_model.rs`.

### CLI
```bash
# Synthetic-weight benchmark (see docs/BENCHMARKS.md for caveats)
cargo run --release -p rllm-cli -- bench --preset tiny --batch-size 4

# RAG demo over a small built-in document set
cargo run --release -p rllm-cli -- rag-demo --query "What makes rllm enterprise grade and edge compatible?"

# HTTP server with a synthetic model (no download needed)
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

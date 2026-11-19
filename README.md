# `rllm`: LLM Inference & RAG Engine in Rust

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.88%2B-orange.svg)](https://www.rust-lang.org/)
[![Status](https://img.shields.io/badge/status-early%20prototype-yellow.svg)](#-project-status)

A modular, multi-backend LLM inference engine and hybrid RAG platform written in Rust from the ground up.

> [!WARNING]
> **`rllm` is an early-stage prototype.** It can load and run small LLaMA-architecture GGUF chat models and BERT-architecture GGUF embedding models on the CPU or an Apple GPU (Metal), with outputs verified against llama.cpp, and answer questions over indexed documents with cited sources. On an Apple M4 its CPU prompt processing is 1.3–3.3x faster than llama.cpp's CPU backend at 512 tokens, decode on a 1B model is at parity with llama.cpp on both CPU and Metal across Q8_0 / Q4_K_M / Q4_0 (0.87–0.91x on CPU for a 135M model), and Metal prompt processing is 0.85–0.95x of llama.cpp's Metal backend. See [Known Issues & Limitations](#-known-issues--limitations). Do not use it for anything beyond experimentation and development.

---

## 📍 Project Status

Legend: ✅ implemented · 🟡 partial / prototype · ⛔ stub · 📋 planned (no code yet)

### Core & Backends

| Component | Status | Notes |
| :--- | :---: | :--- |
| Tensor, `Shape`, `DType`, `Device`, `Backend` trait (`rllm-core`) | ✅ | Strided views exist, but CPU kernels assume contiguous inputs. |
| CPU F32 kernels: GEMV, RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, softmax | ✅ | GEMV/dot use ARM NEON on `aarch64`, scalar elsewhere; Rayon-parallel. |
| CPU GEMM | ✅ / 🟡 | Batches of 32+ rows use Accelerate `cblas_sgemm` on macOS (Apple matrix units); elsewhere a portable row-parallel kernel (not cache-tiled). |
| Q4_0 / Q8_0 quantized linear layers | ✅ | Decode quantizes activations to Q8 and uses ARM `SDOT` int8 dot products (runtime-detected; scalar fallback); batches of 32+ rows dequantize and use the GEMM above. |
| Q4_K, Q6_K, Q5_0 weights (Q4_K_M files) | ✅ | Q4_K / Q6_K run natively on CPU (SDOT kernels on Q8_K activations, as llama.cpp) and Metal (dedicated matrix-vector kernels, K-quant GEMM tiles). Q5_0 is expanded losslessly to Q8_0. On CPU, Q4_0 is repacked into interleaved 4-row groups with SDOT decode and i8mm (`SMMLA`) batched kernels. |
| F16, BF16, Q4_1, Q5_1, Q5_K weights | 🟡 | Q4_1 / Q5_1 / Q5_K are converted to Q8_0 at load; F16 / BF16 become F32 (F16 again on Metal). |
| AVX2 / AVX-512 kernels | 📋 | x86 uses portable scalar int8 kernels. |
| PagedAttention kernel (CPU) | ✅ | Causal prefill and batched decode; parallel over (row, head). |
| Metal executor (`--device metal`) | ✅ | Whole forward pass in one command buffer: F32 / F16 / Q8_0 / Q4_0 / Q4_K / Q6_K matrix-vector and tiled `simdgroup_matrix` GEMM kernels, FlashAttention-style tiled prefill attention (half-precision K/V tiles, in-simdgroup online softmax), GPU-resident paged KV cache. Perplexity within 0.02% of llama.cpp Metal and greedy output identical (SmolLM2-135M, Llama-3.2-1B). macOS only. |
| CUDA backend | 📋 | No crate yet. |

### Models

| Component | Status | Notes |
| :--- | :---: | :--- |
| LLaMA-architecture decoder (GQA, RoPE, SwiGLU, tied embeddings, LLaMA 3 RoPE frequency factors) | ✅ | Loads from GGUF. Greedy output matches llama.cpp token-for-token on SmolLM2-135M Q8_0 and Llama-3.2-1B Q4_0; in the other formats (F16, Q4_0, Q4_K_M, Q5_K_M) llama.cpp's greedy token is rllm's argmax or within a near-tie (teacher-forced tests), and perplexity is within 0.6%. |
| BERT embedding models (bge, MiniLM) | ✅ | Loaded from GGUF with a WordPiece tokenizer; mean or `[CLS]` pooling. Embeddings match llama.cpp (cosine ≥ 0.999999) for bge-small-en-v1.5 and all-MiniLM-L6-v2. |
| WordPiece tokenizer | ✅ | BERT basic tokenization (NFD, lower-casing, accent stripping, CJK splitting) and greedy longest-match. Matches `llama-tokenize` on a 23-case corpus. |
| GGUF reader | ✅ | Typed metadata, memory-mapped tensor data, `general.alignment`; `LlamaModel::from_gguf` / `GgufBundle::load`. Architectures other than `llama` (e.g. `qwen2`, `gemma`) are not supported yet. |
| Safetensors loader | 📋 | |
| Byte-level BPE tokenizer (from GGUF) | ✅ | Special tokens, `llama3` / `qwen2` / `smollm` / `gpt2` pre-tokenizers, streaming UTF-8 decode. Matches `llama-tokenize` on a 23-case corpus. SentencePiece (`tokenizer.ggml.model = llama`) is not supported yet. |
| Chat templates | ✅ | The model's Jinja template rendered with minijinja; ChatML fallback. |
| Legacy word-list tokenizer | 🟡 | Only used by synthetic test models. |
| Mistral / Qwen / DeepSeek architectures | 📋 | |

### Scheduler

| Component | Status | Notes |
| :--- | :---: | :--- |
| Paged KV block allocator with ref-counts | ✅ | |
| Radix-tree prefix cache | ✅ | Block-granular tree, ref-counted blocks, LRU leaf eviction. Blocks are shared as soon as they are computed, so concurrent requests reuse a common system prompt or chat history. Time to first token with a warm 2k-token prefix: 14x faster on CPU, 23x on Metal (Llama-3.2-1B). |
| Continuous batching (iteration-level) | ✅ | A dedicated engine thread steps the scheduler; each step mixes prefill of new sequences with decode of running ones, and preempts (recompute) under KV pressure. Requests receive tokens and a terminal event over their own channel; disconnecting cancels the sequence and frees its KV blocks. 8 concurrent requests run 2.8× faster than sequential on SmolLM2-135M. |
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
| Relevance gating | ✅ | A passage is relevant if its lexical score ≥ 0.25 **or** its embedding cosine similarity ≥ a per-model threshold (bge 0.6, MiniLM 0.3, calibrated on a small set; override with `--min-dense-similarity`). Unrelated queries are refused without generation. |
| Grounded quote extraction | ✅ | Sentence-level keyword heuristics. |
| Answer generation | ✅ | A local or remote LLM answers from numbered source passages; `[n]` citations in the answer are verified, and a warning is returned when the answer cites nothing or cites a non-existent source. Without an LLM the answer is extractive. |

### Serving

| Component | Status | Notes |
| :--- | :---: | :--- |
| Axum HTTP server (`/health`, `/metrics`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/rag/index`, `/v1/rag/query`) | ✅ | With `--model <file.gguf>`, chat completions use the model's chat template and tokenizer, stop at end-of-turn tokens and report `finish_reason` / usage; errors use OpenAI's error format. With `--embedding-model <bert.gguf>`, `/v1/embeddings` and RAG use a real embedding model. |
| SSE streaming | ✅ | OpenAI `chat.completion.chunk` events, `stream_options.include_usage`, `data: [DONE]`; streamed text is identical to non-streamed text. |
| Limits & operations | ✅ | Queue cap with HTTP 429 + `Retry-After` (`--max-waiting`), `max_tokens` cap (`--max-tokens-limit`), parameter validation, 2 MiB body limit, Prometheus `/metrics`, graceful shutdown on SIGTERM/Ctrl-C. No authentication or rate limiting per client. |
| gRPC | 📋 | |
| Separate inference / RAG containers (Docker, Compose) | ✅ | Both run synthetic models. |

---

## 🐞 Known Issues & Limitations

The P0 correctness issues found in the initial review (stride-ignoring matmul on transposed weights, missing prefill, per-batch RoPE, prefix-cache block ownership, unbounded inputs, unaligned storage) have been fixed and are covered by numerical reference tests (`crates/rllm-models/tests/reference_forward.rs`, `crates/rllm-scheduler/tests/scheduler_behaviour.rs`).

Remaining limitations:

1. **Only LLaMA-architecture GGUF models with a byte-level BPE tokenizer load.** Qwen2 (attention biases, NeoX RoPE), Gemma, Phi, and SentencePiece tokenizers are not supported yet.
2. **Performance gaps (Apple M4).** CPU: decode on Llama-3.2-1B is at parity or ahead (Q4_K_M 1.04x, Q4_0 0.99x, Q8_0 0.95x) and prompt processing is ahead except 128-token prompts on Q8_0/Q4_0 (0.85–0.93x); decode on the tiny SmolLM2-135M is 0.87–0.91x (thread-synchronisation overhead). Metal: decode at parity or ahead (0.96–1.30x across formats), prompt processing 0.85–0.95x. No CUDA backend; Metal runs only on macOS. See [BENCHMARKS](docs/BENCHMARKS.md).
3. **Q5_K weights are converted to Q8_0** (no native kernel yet), and F16 / BF16 files use 2× their size on CPU (F32).
4. **Small models rarely write `[n]` citations.** SmolLM2-135M answers correctly from the sources but usually without citation markers; the response then carries a warning. Dense thresholds are calibrated only for bge-small-en-v1.5 and all-MiniLM-L6-v2. The reranker is lexical, not a neural cross-encoder.
5. **No authentication, per-client rate limiting or TLS** — put rllm behind a reverse proxy for anything exposed. A single engine thread serves one model per process.
6. **The prefix cache shares whole blocks only** (no copy-on-write of partial blocks), and only exact prefixes: RAG prompts whose retrieved passages differ reuse just the instructions before them.

---

## 🗺️ Roadmap

- **P0 — Correctness:** ✅ done — projection layout, causal prefill, per-token RoPE, prefix-cache block ownership, aligned storage, input bounds, numerical reference tests.
- **P1 — Real models:** ✅ done — GGUF loading, BPE and WordPiece tokenizers, chat templates, `rllm generate`, real-model chat completions, BERT embedding models, LLM-generated RAG answers with verified citations.
- **P2 — Serving:** ✅ done — dedicated engine thread with per-request channels, SSE streaming, cancellation on disconnect, queue/`max_tokens`/body limits with 429s, structured errors, Prometheus metrics, graceful shutdown.
- **P3 — Performance & backends:** 🟡 CPU and Metal done — int8 `SDOT` kernels, Accelerate GEMM, parallelism tuning, GQA-grouped attention, native Q4_K / Q6_K kernels on CPU and Metal, repacked Q4_0 with i8mm, a Metal executor with tiled attention, perplexity/greedy parity tests against llama.cpp CPU and Metal. Remaining: faster Metal GEMM, lower-latency CPU thread pool, Q5_K kernels, F16 KV cache, x86 SIMD, CUDA.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the target design and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for benchmark status.

---

## 🏗️ Workspace Layout

```text
rllm/
├── crates/
│   ├── rllm-core/          # Tensor, Shape, DType, Device, Backend trait, errors
│   ├── rllm-backend-cpu/   # CPU kernels (NEON dot/GEMV, Q4_0/Q8_0 GEMV, PagedAttention, norms, RoPE)
│   ├── rllm-backend-metal/ # Metal (Apple GPU) LLaMA executor + MSL kernels
│   ├── rllm-models/        # LLaMA decoder, BERT embedder, GGUF reader/loader, BPE + WordPiece tokenizers, chat templates
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

# Same on the Apple GPU (any subcommand accepts --device metal)
cargo run --release -p rllm-cli -- --device metal generate --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --prompt "What is the boiling point of water at sea level?"

# Raw completion, greedy
cargo run --release -p rllm-cli -- generate --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --raw --prompt "The capital of France is" --temperature 0

# OpenAI-compatible server
cargo run --release -p rllm-cli -- serve --service inference --addr 127.0.0.1:8000 \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'

# Streaming (server-sent events); Ctrl-C on the client cancels generation
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true,"stream_options":{"include_usage":true}}'

# Prometheus metrics (queue, KV cache, tokens, TTFT, per-route requests)
curl -s localhost:8000/metrics
```

### Question answering over your documents (RAG)

```bash
curl -L -o models/bge-small-en-v1.5-q8_0.gguf \
  https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf

# Built-in demo corpus
cargo run --release -p rllm-cli -- rag-demo \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --embedding-model models/bge-small-en-v1.5-q8_0.gguf \
  --query "At what temperature does water boil at sea level?"

# Server with chat, embeddings and RAG in one process
cargo run --release -p rllm-cli -- serve --service all --addr 127.0.0.1:8080 \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --embedding-model models/bge-small-en-v1.5-q8_0.gguf
```

`models/` is git-ignored. With the model files present, `cargo test --workspace` also runs the llama.cpp parity tests (`crates/rllm-models/tests/real_model.rs`, `real_embedding.rs`) and the RAG tests in `crates/rllm-rag/tests/real_rag.rs`.

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

The inference engine and the RAG engine can run as separate containers. Without model files both use synthetic models (generated text is meaningless; lexical retrieval still works). To use real models, download them into `./models` (mounted read-only at `/models`) and uncomment the `RLLM_MODEL` / `RLLM_EMBEDDING_MODEL` lines in `docker-compose.yml`.

```bash
docker compose up --build
```

- `rllm-inference` → `http://localhost:8000` (`/health`, `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`)
- `rllm-rag` → `http://localhost:8080` (`/health`, `/v1/rag/index`, `/v1/rag/query`); retrieval runs here and answers are generated by `INFERENCE_URL=http://rllm-inference:8000`

```bash
curl -s -X POST http://localhost:8080/v1/rag/index -H "Content-Type: application/json" \
  -d '{"doc_id": "eiffel", "text": "The Eiffel Tower in Paris was completed in 1889 and is 330 metres tall."}' | jq

curl -s -X POST http://localhost:8080/v1/rag/query -H "Content-Type: application/json" \
  -d '{"query": "How tall is the tower in the French capital?", "top_k": 2, "max_tokens": 64}' | jq
```

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

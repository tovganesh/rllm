# `rllm`: Enterprise-Grade LLM Inference & High-Performance RAG Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Container-2496ED.svg)](docker-compose.yml)

A modular, zero-cost-abstraction, multi-backend LLM inference engine and high-throughput RAG platform written in **Rust from the ground up**.

---

## 📚 Core Documentation & Benchmarks

- **[System Architecture Specification](docs/ARCHITECTURE.md)**: Deep dive into the PagedAttention memory manager, continuous batching scheduler, Radix tree prompt cache, hardware abstraction, and hybrid RAG subsystem.
- **[Inference Performance Benchmarks & 4GB Fit Report](docs/BENCHMARKS.md)**: Comprehensive empirical evaluation, memory footprint breakdown for 4GB hardware envelopes, micro-kernel timings, batch scaling efficiency, and Radix cache speedup metrics.

---

## 🌐 Designed for the Full Hardware Spectrum

- **Smartphone & Edge**: iOS / Android (ARM NEON SIMD, unified memory, Q4_0 / Q4_K / Q8_0 block quantization, zero-copy buffers). Designed to comfortably run within **4 GB RAM** envelopes leaving >2.5 GB headroom for the operating system and apps.
- **Apple Silicon / MLX**: Native Metal Shading Language (MSL) compute shaders, Unified Memory Architecture (UMA) zero-copy tensor passing, SIMDgroup matrix ops.
- **GPU Clusters**: CUDA / cuBLAS / FlashAttention runtime, continuous batching, and multi-node throughput scaling.

---

## 🏗️ Workspace Architecture

```text
rllm/
├── crates/
│   ├── rllm-core/          # Tensor abstraction, Shape, DType (F32/F16/BF16/Q4/Q8), Device & Backend traits
│   ├── rllm-backend-cpu/   # SIMD CPU engine (NEON, AVX2, cache-tiled GEMM, Q4/Q8 GEMV, Rayon)
│   ├── rllm-backend-metal/ # Apple Silicon / Metal compute shaders (MSL GEMV, RMSNorm, RoPE)
│   ├── rllm-models/        # GGUF parser, LLaMA 3.x / Mistral / Qwen, BERT embedding
│   ├── rllm-scheduler/     # PagedAttention KV-Cache, Continuous Batching, Radix-Tree Prefix Cache
│   ├── rllm-rag/           # SIMD HNSW vector store, Okapi BM25, Hybrid RRF, Cross-Encoder Reranker
│   ├── rllm-server/        # OpenAI-compatible REST + SSE Streaming API (Axum)
│   └── rllm-cli/           # CLI tool for serving, benchmarking, and interactive RAG queries
├── docker/
│   ├── Dockerfile.inference # Dedicated LLM Inference Engine container (port 8000)
│   └── Dockerfile.rag       # Dedicated High-Performance RAG container (port 8080)
├── docker-compose.yml       # Multi-container service orchestration
├── docs/
│   ├── ARCHITECTURE.md     # Full architectural specification
│   └── BENCHMARKS.md       # Benchmarking results & memory footprint analysis
└── LICENSE                 # Apache 2.0 License
```

---

## 🚀 Key Technical Innovations

### 1. Unified Hardware Abstraction (`rllm-core`)
- Zero-cost trait dispatch across devices (`Device::Cpu`, `Device::Metal(id)`, `Device::Cuda(id)`).
- Block-quantized data types (`Q4_0`, `Q4_K`, `Q8_0`) with packed bit nibbles for edge footprint minimization.
- Zero-copy tensor slicing, broadcasting, and memory-mapped weight ingestion.

### 2. PagedAttention & Radix Prefix KV Caching (`rllm-scheduler`)
- **Virtual Memory Paging**: KV caches are managed as non-contiguous physical blocks (16 tokens per block), eliminating up to 80% of memory fragmentation.
- **Radix-Tree Prompt Caching**: Common system instructions, few-shot examples, and retrieved RAG context documents form paths in a prefix trie. Matching prefixes skip the prefill phase entirely for **instant Time-To-First-Token (TTFT)**.
- **Continuous (In-Flight) Batching**: Iteration-level scheduling allows new requests to join the forward pass without stalling existing sequence generation.

### 3. Edge & Smartphone Optimized (`rllm-backend-cpu` / `rllm-backend-metal`)
- **Direct Quantized GEMV**: Multiplies Q4_0 and Q8_0 weight matrices directly with F32 activations with zero intermediate dequantization memory overhead (delivering **1.42x speedup** over unquantized execution).
- ARM NEON vectorization for sub-millisecond mobile latency.
- Native Metal MSL compute shaders targeting Apple Silicon unified memory.

### 4. High-Performance Hybrid RAG Subsystem (`rllm-rag`)
- **Direct Engine Integration**: Embedding generation (BERT / ModernBERT / BGE / Nomic) executes directly on the tensor backend without Python or ONNX serialization overhead.
- **Embedded SIMD HNSW Index**: Hierarchical Navigable Small World vector graph with cosine / dot product distance.
- **Okapi BM25 Inverted Index**: Exact keyword, code identifier, and acronym retrieval.
- **Reciprocal Rank Fusion (RRF)**: Combines dense vector and lexical rankings.
- **Cross-Encoder Reranker**: Reranks top candidates before context generation.
- **Prefix-Aware Assembly**: Sorts retrieved documents to maximize Radix tree KV-cache hit rates.

---

## 🐳 Docker Deployment: Separate Containers

`rllm` supports running the **Inference Engine** and the **RAG Engine** in completely decoupled containers, allowing independent scaling, resource isolation, and deployment across heterogeneous infrastructure (e.g. GPU cluster for inference, CPU nodes for vector retrieval).

```text
 ┌────────────────────────────────────────────────────────┐
 │                      Client / App                      │
 └──────────────┬──────────────────────────┬──────────────┘
                │                          │
      POST /v1/chat/completions     POST /v1/rag/query
                │                          │
                ▼                          ▼
 ┌──────────────────────────┐    ┌──────────────────────────┐
 │      rllm-inference      │    │         rllm-rag         │
 │     Container (Port 8000)│    │     Container (Port 8080)│
 │                          │    │                          │
 │  • Continuous Batching   │◄───┤  • SIMD HNSW (Vectors)   │
 │  • PagedAttention        │    │  • Okapi BM25 (Lexical)  │
 │  • Block Quant (Q4/Q8)   │    │  • Hybrid RRF Fusion     │
 │  • OpenAI API Compatible │    │  • Cross-Encoder Rerank  │
 └──────────────────────────┘    └──────────────────────────┘
```

### 1. One-Command Orchestration (Recommended)

Run both services together with Docker Compose:

```bash
docker compose up --build
```

- **`rllm-inference`** listens on `http://localhost:8000`
- **`rllm-rag`** listens on `http://localhost:8080` (configured with `INFERENCE_URL=http://rllm-inference:8000`)

### 2. Standalone Inference Container

Build and run only the LLM inference engine:

```bash
# Build
docker build -f docker/Dockerfile.inference -t rllm-inference:latest .

# Run on port 8000
docker run -d --name rllm-inference -p 8000:8000 rllm-inference:latest
```

Test inference health and chat completion:

```bash
# Health check
curl -s http://localhost:8000/health | jq

# OpenAI-compatible Chat Completion
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain PagedAttention in Rust."}],
    "temperature": 0.7,
    "max_tokens": 64
  }' | jq
```

### 3. Standalone RAG Container

Build and run only the RAG engine, pointing it to an inference container:

```bash
# Build
docker build -f docker/Dockerfile.rag -t rllm-rag:latest .

# Run on port 8080 pointing to the inference host
docker run -d --name rllm-rag \
  -p 8080:8080 \
  -e INFERENCE_URL=http://localhost:8000 \
  rllm-rag:latest
```

Test RAG document ingestion and semantic query:

```bash
# Ingest an enterprise document
curl -s -X POST http://localhost:8080/v1/rag/index \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "rust_paged_attn",
    "text": "rllm implements PagedAttention in Rust, eliminating memory fragmentation and enabling sub-millisecond continuous batching."
  }' | jq

# Query knowledge base with citation synthesis
curl -s -X POST http://localhost:8080/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does rllm eliminate memory fragmentation?",
    "top_k": 2,
    "max_tokens": 32
  }' | jq
```

---

## 🛠️ Local Development & CLI

### Prerequisites
- Rust 1.80+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

### Build
```bash
cargo build --release
```

### Run Tests Across All Workspace Crates
```bash
cargo test --workspace
```

### Build Documentation
```bash
cargo doc --workspace --no-deps --open
```

### Run Performance Benchmarks
```bash
cargo run --release -p rllm-cli -- bench --preset edge-4gb --batch-size 4 --prompt-tokens 64 --generate-tokens 16
```

### Interactive RAG Demonstration
```bash
cargo run --release -p rllm-cli -- rag-demo --query "What makes rllm enterprise grade and edge compatible?"
```

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

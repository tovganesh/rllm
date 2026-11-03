# `rllm`: Enterprise-Grade LLM Inference & High-Performance RAG Engine

A modular, zero-cost-abstraction, multi-backend LLM inference engine and high-throughput RAG platform written in **Rust from the ground up**.

Designed to scale across the full hardware spectrum:
- **Smartphone & Edge**: iOS / Android (ARM NEON, unified memory, Q4_0 / Q4_K / Q8_0 block quantization, zero-copy buffers).
- **Apple Silicon / MLX**: Native Metal Shading Language (MSL) compute shaders, Unified Memory Architecture (UMA), SIMDgroup matrix ops.
- **GPU Clusters**: NVIDIA CUDA / cuBLAS / FlashAttention runtime, continuous batching, and multi-node throughput scaling.

---

## 🏗️ System Architecture

```text
rllm/
├── crates/
│   ├── rllm-core/          # Tensor abstraction, Shape, DType (F32/F16/BF16/Q4/Q8), Device & Backend traits
│   ├── rllm-backend-cpu/   # SIMD CPU engine (NEON, AVX2, cache-tiled GEMM, Q4/Q8 GEMV, Rayon)
│   ├── rllm-backend-metal/ # Apple Silicon / Metal compute shaders (MSL GEMV, RMSNorm, RoPE)
│   ├── rllm-models/        # GGUF & Safetensors loaders, LLaMA 3.x / Mistral / Qwen, BERT embedding
│   ├── rllm-scheduler/     # PagedAttention KV-Cache, Continuous Batching, Radix-Tree Prefix Cache
│   ├── rllm-rag/           # SIMD HNSW vector store, Okapi BM25, Hybrid RRF, Cross-Encoder Reranker
│   ├── rllm-server/        # OpenAI-compatible REST + SSE Streaming API (Axum)
│   └── rllm-cli/           # CLI tool for serving, benchmarking, and interactive RAG queries
```

---

## 🚀 Key Technical Innovations

### 1. Unified Hardware Abstraction (`rllm-core`)
- Zero-cost trait dispatch across devices (`Device::Cpu`, `Device::Metal(id)`, `Device::Cuda(id)`).
- Block-quantized data types (`Q4_0`, `Q4_K`, `Q8_0`) with packed bit nibbles for edge footprint minimization.
- Zero-copy tensor slicing, broadcasting, and memory-mapped weight ingestion.

### 2. PagedAttention & Radix Prefix KV Caching (`rllm-scheduler`)
- **Virtual Memory Paging**: KV caches are managed as non-contiguous physical blocks (16 tokens per block), eliminating up to 80% of memory fragmentation.
- **Radix-Tree Prompt Caching**: Common system instructions, few-shot examples, and retrieved RAG context documents form paths in a prefix trie. Matching prefixes skip the prefill phase entirely for **0 ms Time-To-First-Token (TTFT)**!
- **Continuous (In-Flight) Batching**: Iteration-level scheduling allows new requests to join the forward pass without stalling existing sequence generation.

### 3. Edge & Smartphone Optimized (`rllm-backend-cpu` / `rllm-backend-metal`)
- Direct Quantized GEMV: Multiplies Q4_0 and Q8_0 weight matrices directly with F32 activations with zero intermediate dequantization memory overhead.
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

## 🛠️ Getting Started

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

---

## 💻 CLI Usage

### 1. Run Performance Benchmark
Measures Time To First Token (TTFT), token generation throughput, and continuous batching scheduling:
```bash
cargo run --release -p rllm-cli -- bench --batch-size 4 --prompt-tokens 64 --generate-tokens 32
```

### 2. Interactive High-Performance RAG Demo
Indexes enterprise knowledge documents into the hybrid SIMD HNSW + BM25 store and answers queries with citations:
```bash
cargo run --release -p rllm-cli -- rag-demo --query "What makes rllm enterprise grade and edge compatible?"
```

### 3. Launch OpenAI-Compatible API Server
```bash
cargo run --release -p rllm-cli -- serve --addr 127.0.0.1:8080 --model llama-3-8b
```

#### Endpoints:
- `GET  /health` - Cluster health & active KV cache metrics
- `POST /v1/chat/completions` - OpenAI-compatible chat completion
- `POST /v1/embeddings` - OpenAI-compatible dense vector embeddings
- `POST /v1/rag/index` - Ingest and index documents into hybrid store
- `POST /v1/rag/query` - End-to-end RAG query with citations


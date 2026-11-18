# `rllm` Architecture

This document describes the **target architecture** of `rllm` and marks, for each subsystem, how much of it exists today. For a component-by-component status table and the list of known bugs, see the [README](../README.md#-project-status).

Legend: ✅ implemented · 🟡 partial / prototype · ⛔ stub · 📋 planned (no code yet)

---

## 1. Layered Design

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Serving Layer"]
        CLI["rllm-cli ✅"]
        HTTP["OpenAI-shaped REST API (Axum) 🟡"]
        SSE["SSE streaming 📋"]
        GRPC["gRPC (Tonic) 📋"]
    end

    subgraph RagEngine ["RAG Engine (rllm-rag)"]
        Chunker["Word-window chunker ✅"]
        HybridIndex["HNSW ✅ + BM25 ✅ + RRF ✅"]
        Reranker["Lexical reranker 🟡 (neural cross-encoder 📋)"]
        Grounding["Grounded quote extraction ✅"]
    end

    subgraph SchedulerLayer ["Scheduling (rllm-scheduler)"]
        Batcher["Continuous batching 🟡"]
        Paged["Paged KV block allocator ✅"]
        Radix["Radix prefix cache ✅"]
        SpecDec["Speculative decoding 📋"]
    end

    subgraph ModelLayer ["Models (rllm-models)"]
        Weights["GGUF loader ✅ · Safetensors 📋"]
        Llama["LLaMA-architecture decoder ✅ (GGUF)"]
        Embed["BERT-style encoder 🟡 (synthetic weights)"]
        Tok["Byte-level BPE tokenizer ✅ · chat templates ✅"]
    end

    subgraph ComputeLayer ["Backends"]
        Trait["Backend trait (rllm-core) ✅"]
        CPU["rllm-backend-cpu ✅ (NEON; AVX2/AVX-512 📋)"]
        Metal["rllm-backend-metal ⛔"]
        CUDA["rllm-backend-cuda 📋"]
    end

    ClientLayer --> RagEngine
    ClientLayer --> SchedulerLayer
    RagEngine --> SchedulerLayer
    SchedulerLayer --> ModelLayer
    ModelLayer --> ComputeLayer
```

---

## 2. Target Hardware Tiers

These are design goals. Today only the CPU backend executes; there is no Metal or CUDA execution path.

| Tier | Target platforms | Intended techniques | Today |
| :--- | :--- | :--- | :--- |
| Edge / smartphone | iOS, Android (ARM64) | Q4/Q8 block quantization, mmap weights, NEON, Metal | NEON F32 dot/GEMV and scalar Q4_0/Q8_0 GEMV on CPU |
| Workstation / unified memory | Apple Silicon | Metal compute pipelines, zero-copy unified memory | MSL source only; no dispatch |
| GPU cluster | NVIDIA data-center GPUs | CUDA, FlashAttention, tensor parallelism | Not started |

---

## 3. Crates

```text
crates/
├── rllm-core/          # Tensor, Shape, DType, Device, Storage, Backend trait, errors
├── rllm-backend-cpu/   # CPU kernels
├── rllm-backend-metal/ # Metal backend stub + MSL shaders
├── rllm-models/        # LlamaModel, BertEmbeddingModel, GgufFile, Tokenizer, configs
├── rllm-scheduler/     # BlockAllocator, PrefixCache, ContinuousBatchScheduler, Sampler, Sequence
├── rllm-rag/           # TextChunker, HnswIndex, Bm25Index, HybridIndex, reranker, grounding, RagEngine
├── rllm-server/        # Axum router and handlers
└── rllm-cli/           # serve / bench / rag-demo
```

A `rllm-backend-cuda` crate is planned but does not exist.

---

## 4. Subsystems

### 4.1 Tensor & Backend Abstraction (`rllm-core`) — ✅

- `Tensor` holds a shape, strides, dtype, device, byte offset and a `Storage` (`Cpu`, `Metal`, `Cuda`). Only `Cpu` storage holds data; `Metal`/`Cuda` variants are placeholders.
- `reshape` and `transpose` are zero-copy views. `as_f32_slice` rejects non-contiguous views; `contiguous()` materialises them. CPU `matmul` recognises a transposed contiguous weight (`W.transpose(0, 1)`) and computes `A·Wᵀ` with row dot products without copying.
- `CpuStorage` is 64-byte aligned.
- `Backend` is an object-safe trait with: `matmul`, `rms_norm`, `rope`, `swiglu`, `softmax`, `embedding`, `paged_attention`, `gemv_quantized`.
- `DType` declares `F32, F16, BF16, I32, I8, Q4_0, Q4_K, Q8_0`. Only `F32` compute and `Q4_0`/`Q8_0` GEMV are implemented.

### 4.2 CPU Backend (`rllm-backend-cpu`) — ✅ / 🟡

- F32 dot product with ARM NEON FMA on `aarch64`; 4-way unrolled scalar elsewhere.
- GEMV parallelised over output rows with Rayon. GEMM is a naive row-parallel loop (🟡; tiling/packing planned).
- Q4_0 (32 elements / 18 bytes) and Q8_0 (32 / 34 bytes) GEMV operate directly on quantized blocks without dequantizing the full matrix. Inner loops are scalar.
- PagedAttention decode: per-head logits over a block table, softmax, weighted V accumulation; GQA supported.

### 4.3 Models (`rllm-models`) — ✅ decoder · 🟡 embedder

- `LlamaModel::forward` takes a flattened batch of rows, each with its own token, absolute position and block table: embedding → N × (RMSNorm, QKV, per-row RoPE, KV write for every row, causal PagedAttention, O-proj, RMSNorm, SwiGLU MLP) → final norm → LM head for selected rows only. All prompt tokens of a sequence = causal prefill; one token per sequence = batched decode; both can be mixed. `forward_decode` is a convenience wrapper.
- Numerical tests compare it against a naive dense reference implementation.
- **GGUF loading.** `MappedGguf` memory-maps the file and parses typed metadata and tensor descriptors. `LlamaConfig::from_gguf` reads `llama.*` hyper-parameters; `LlamaModel::from_gguf` maps `token_embd`, `blk.N.{attn_q,attn_k,attn_v,attn_output,attn_norm,ffn_gate,ffn_up,ffn_down,ffn_norm}`, `output_norm` and `output` (falling back to tied embeddings) onto layers, and applies `rope_freqs` factors when present. Linear weights in Q8_0 / Q4_0 stay quantized and go through `Backend::linear`; F16, BF16, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K are dequantized to F32 at load. `GgufBundle::load` returns the model, tokenizer and chat template from one file.
- **Tokenizer.** `BpeTokenizer` implements llama.cpp-compatible byte-level BPE from `tokenizer.ggml.{tokens,token_type,merges,pre}`: special-token splitting, per-model pre-tokenizer regexes applied in sequence, rank-ordered merges, and streaming UTF-8 decode (`StreamDecoder`). Stop tokens come from EOS/EOT metadata plus known end-of-turn markers.
- **Chat templates.** `ChatTemplate` renders the model's Jinja `tokenizer.chat_template` with minijinja (Python-compatible methods, `raise_exception`, `bos_token`, `eos_token`), falling back to ChatML.
- **Verification.** Besides the dense reference test, `tests/real_model.rs` compares tokenization and greedy generation with llama.cpp fixtures for SmolLM2-135M in five weight formats.
- `BertEmbeddingModel::embed` runs a simplified encoder with mean pooling and L2 normalisation, still with synthetic weights (P1b).
- The legacy word-list `Tokenizer` remains for the RAG pipeline and synthetic models until P1b.

### 4.4 Scheduling (`rllm-scheduler`) — 🟡

- **Paged KV cache:** per-layer K/V tensors of shape `[total_blocks, num_kv_heads, block_size, head_dim]`; `BlockAllocator` hands out block IDs from a free list with reference counts.
- **Radix prefix cache:** a token trie that records a physical block ID at every `block_size` boundary; `match_prefix` returns the longest fully cached block-aligned prefix. The cache holds one allocator reference per cached block, so cached blocks are never on the free list; when the pool is exhausted, least-recently-used leaf blocks not used by any sequence are evicted. Matching happens at admission and always leaves at least one token to compute. Only full blocks are shared (no copy-on-write of partial blocks yet).
- **Continuous batching:** `step()` admits waiting sequences while blocks are available, ensures capacity for the next token, runs one batched decode, samples, and retires finished sequences. Each step prefills newly admitted sequences (subject to `max_prefill_tokens`) together with decoding running ones; under KV pressure the newest sequence is preempted and later recomputed. Target design: a dedicated engine loop serving requests through per-sequence channels.
- **Sampler:** greedy, temperature, top-k, top-p, repetition penalty.

### 4.5 RAG (`rllm-rag`)

- **Chunking** ✅ — sliding window over whitespace-delimited words (default 128 words, 32 overlap).
- **HNSW** ✅ — in-memory multi-layer graph, cosine distance (NEON dot on `aarch64`). Persistence and SQ8 vector quantization are planned.
- **BM25** ✅ — Okapi BM25 (`k1 = 1.2`, `b = 0.75`) over a lowercase alphanumeric tokenisation.
- **Hybrid fusion** ✅ — Reciprocal Rank Fusion (`k = 60`) of dense and BM25 rankings.
- **Reranking** 🟡 — `CrossEncoderReranker` is currently a lexical scorer (content-keyword coverage, term density, exact-phrase bonus). A neural cross-encoder is planned.
- **Grounding & relevance** ✅ — sentence-level quote extraction keyed on non-stop-word query terms; queries whose best score is below `min_relevance_threshold` (default 0.25) are returned with `is_relevant: false` and a `warning`, without invoking generation.
- **Answer generation** 🟡 — today the answer is assembled from extracted quotes by a template, and model output is appended only if a heuristic deems it coherent. Target design: the LLM generates the answer from the grounded prompt, and citations are verified against the output.
- **Prefix-aware context ordering** 📋 — ordering retrieved chunks to maximise radix-cache hits is a design goal and is not implemented.

### 4.6 Serving (`rllm-server`) — 🟡

- Service modes: `all`, `inference`, `rag`. In `rag` mode with `INFERENCE_URL` set, retrieval runs locally and generation is delegated to the inference container's `/v1/chat/completions`.
- Handlers currently drive `scheduler.step()` inline until the scheduler is idle; this is a placeholder for a dedicated engine task.
- Chat and embeddings endpoints do not yet tokenize input text (placeholder token IDs).
- SSE streaming, gRPC, authentication, request limits and metrics are planned.

---

## 5. Known Issues

See the [README](../README.md#-known-issues--limitations). The P0 correctness issues are fixed; the main remaining gap is that no real model or tokenizer can be loaded yet.

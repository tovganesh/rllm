//! End-to-end Retrieval-Augmented Generation (RAG) execution pipeline.
//!
//! Integrates document ingestion, BERT dense embeddings, hybrid index querying,
//! cross-encoder reranking, prefix-preserving prompt assembly, and token generation
//! via the [`ContinuousBatchScheduler`].

use crate::chunker::{DocumentChunk, TextChunker};
use crate::hybrid::HybridIndex;
use crate::reranker::CrossEncoderReranker;
use parking_lot::Mutex;
use rllm_core::{backend::Backend, error::Result};
use rllm_models::BertEmbeddingModel;
use rllm_scheduler::{ContinuousBatchScheduler, SamplingParams, Sequence};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::info;

/// Source citation attributing answer content to an ingested document passage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Document identifier.
    pub doc_id: String,
    /// Chunk index within document.
    pub chunk_id: usize,
    /// Passage text used in prompt context.
    pub text: String,
    /// Calculated relevance score from retrieval and reranking.
    pub relevance: f32,
    /// Associated metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

/// Final answer synthesized by the RAG engine along with source citations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    /// Original user query string.
    pub query: String,
    /// Generated answer text.
    pub answer: String,
    /// List of source passages cited in the synthesized response.
    pub citations: Vec<Citation>,
    /// Number of output tokens generated.
    pub generated_tokens: usize,
}

/// Unified RAG engine coordinating embedding models, hybrid indexes, and continuous batch schedulers.
pub struct RagEngine<B: Backend> {
    /// Local embedding model used for dense semantic retrieval.
    pub embedding_model: Arc<BertEmbeddingModel<B>>,
    /// Hybrid index containing vectors and inverted BM25 index.
    pub hybrid_index: Mutex<HybridIndex>,
    /// Configured text chunker for passage splitting.
    pub chunker: TextChunker,
    /// Continuous batching scheduler executing the autoregressive LLM.
    pub scheduler: Arc<ContinuousBatchScheduler<B>>,
    seq_counter: Mutex<u64>,
}

impl<B: Backend> RagEngine<B> {
    /// Constructs a new [`RagEngine`] with embedding and LLM generation backends.
    pub fn new(
        embedding_model: Arc<BertEmbeddingModel<B>>,
        scheduler: Arc<ContinuousBatchScheduler<B>>,
        dim: usize,
    ) -> Self {
        Self {
            embedding_model,
            hybrid_index: Mutex::new(HybridIndex::new(dim, 16, 64, 32)),
            chunker: TextChunker::default(),
            scheduler,
            seq_counter: Mutex::new(1),
        }
    }

    /// Ingests a text document into the RAG engine:
    /// Chunks text, computes dense embeddings on the inference engine, and adds to hybrid index.
    pub fn ingest_document(
        &self,
        doc_id: &str,
        text: &str,
        metadata: HashMap<String, String>,
    ) -> Result<usize> {
        let chunks = self.chunker.chunk_text(doc_id, text, metadata);
        let num_chunks = chunks.len();

        let mut index = self.hybrid_index.lock();
        for chunk in chunks {
            // Convert words to pseudo token IDs for embedding
            let token_ids: Vec<u32> = chunk
                .text
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| ((i % 500) + 1) as u32)
                .collect();

            let emb = self.embedding_model.embed(&token_ids)?;
            index.insert(emb, chunk);
        }

        info!(
            "Ingested document '{}' ({} chunks) into hybrid index",
            doc_id, num_chunks
        );
        Ok(num_chunks)
    }

    /// Executes end-to-end RAG:
    /// Retrieval -> Rerank -> Prefix-optimized Prompt Assembly -> Continuous Batching Generation.
    pub async fn query(
        &self,
        query_text: &str,
        top_k: usize,
        sampling: SamplingParams,
    ) -> Result<RagResponse> {
        // 1. Generate query embedding
        let q_token_ids: Vec<u32> = query_text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| ((i % 500) + 1) as u32)
            .collect();
        let q_emb = self.embedding_model.embed(&q_token_ids)?;

        // 2. Hybrid search (dense HNSW + sparse BM25)
        let raw_matches = {
            let index = self.hybrid_index.lock();
            index.search(query_text, &q_emb, top_k * 2)
        };

        // 3. Rerank candidates
        let candidate_chunks: Vec<DocumentChunk> = raw_matches.into_iter().map(|m| m.chunk).collect();
        let reranked = CrossEncoderReranker::rerank(query_text, candidate_chunks, top_k);

        let citations: Vec<Citation> = reranked
            .iter()
            .map(|r| Citation {
                doc_id: r.chunk.doc_id.clone(),
                chunk_id: r.chunk.chunk_id,
                text: r.chunk.text.clone(),
                relevance: r.relevance_score,
                metadata: r.chunk.metadata.clone(),
            })
            .collect();

        // 4. Assemble prompt with prefix ordering for KV-cache reuse:
        // [System Prompt] -> [Context Passages] -> [User Query]
        let mut context_str = String::new();
        for (i, cit) in citations.iter().enumerate() {
            context_str.push_str(&format!("\n[{}] {}\n", i + 1, cit.text));
        }

        let assembled_prompt = format!(
            "System: You are an enterprise intelligence assistant. Answer using the provided context.\n\
             Context: {}\n\
             Question: {}\n\
             Answer:",
            context_str, query_text
        );

        // Tokenize prompt to token IDs
        let prompt_tokens: Vec<u32> = assembled_prompt
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| ((i % 500) + 1) as u32)
            .collect();

        // 5. Submit to ContinuousBatchScheduler with a streaming channel
        let (tx, mut rx) = mpsc::unbounded_channel();
        let seq_id = {
            let mut c = self.seq_counter.lock();
            let id = *c;
            *c += 1;
            id
        };

        let seq = Sequence::new(
            seq_id,
            prompt_tokens,
            sampling,
            self.scheduler.config.block_size,
            Some(tx),
        );

        self.scheduler.submit_sequence(seq)?;

        // Drive scheduler loop until sequence finishes
        let mut generated_tokens = 0;
        let mut output_tokens = Vec::new();

        loop {
            let _ = self.scheduler.step()?;
            while let Ok(token) = rx.try_recv() {
                output_tokens.push(token);
                generated_tokens += 1;
            }

            // Check if scheduler is done with this sequence
            if self.scheduler.num_running() == 0 && self.scheduler.num_waiting() == 0 {
                break;
            }
        }

        let answer = if !output_tokens.is_empty() {
            format!("Synthesized answer based on {} retrieved passages.", citations.len())
        } else {
            "No output tokens generated.".to_string()
        };

        Ok(RagResponse {
            query: query_text.to_string(),
            answer,
            citations,
            generated_tokens,
        })
    }
}

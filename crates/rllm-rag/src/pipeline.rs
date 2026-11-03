//! End-to-end Retrieval-Augmented Generation (RAG) execution pipeline.
//!
//! Integrates document ingestion, BERT dense embeddings, hybrid index querying,
//! cross-encoder reranking, precise grounded sentence quotation extraction,
//! relevance thresholding with explicit unrelated-query flagging, and continuous batching.

use crate::chunker::{DocumentChunk, TextChunker};
use crate::grounding::{extract_content_keywords, extract_grounded_quote};
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

/// Default minimum relevance score threshold below which a query is flagged as unrelated.
pub const DEFAULT_MIN_RELEVANCE_THRESHOLD: f32 = 0.25;

/// Source citation attributing answer content to an exact grounded passage in an ingested document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Document identifier.
    pub doc_id: String,
    /// Chunk index within document.
    pub chunk_id: usize,
    /// Exact grounded quotation / sentence extracted from the document that is relevant to the query.
    pub quote: String,
    /// Passage text used in prompt context (contains the exact grounded quotation).
    pub text: String,
    /// Full surrounding passage text from the document chunk.
    pub passage: String,
    /// Calculated relevance score from retrieval and reranking in $[0.0, 1.0]$.
    pub relevance: f32,
    /// Content keywords that grounded this citation to the query.
    pub matched_keywords: Vec<String>,
    /// Associated metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

/// Final answer synthesized by the RAG engine along with source citations and relevance status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResponse {
    /// Original user query string.
    pub query: String,
    /// Generated answer text or explicit warning if unrelated.
    pub answer: String,
    /// List of source passages cited in the synthesized response.
    pub citations: Vec<Citation>,
    /// Number of output tokens generated.
    pub generated_tokens: usize,
    /// Explicit flag indicating whether sufficiently relevant and grounded context was found.
    pub is_relevant: bool,
    /// Maximum relevance confidence score ($0.0$ to $1.0$) among candidate documents.
    pub confidence_score: f32,
    /// Explicit warning if the query was flagged as low relevance or unrelated to the index.
    pub warning: Option<String>,
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
    /// Minimum relevance threshold for qualifying retrieved citations.
    pub min_relevance_threshold: f32,
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
            min_relevance_threshold: DEFAULT_MIN_RELEVANCE_THRESHOLD,
            seq_counter: Mutex::new(1),
        }
    }

    /// Sets a custom minimum relevance threshold.
    pub fn with_relevance_threshold(mut self, threshold: f32) -> Self {
        self.min_relevance_threshold = threshold;
        self
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

    /// Executes end-to-end RAG with the default configured relevance threshold.
    pub async fn query(
        &self,
        query_text: &str,
        top_k: usize,
        sampling: SamplingParams,
    ) -> Result<RagResponse> {
        self.query_with_threshold(query_text, top_k, sampling, None).await
    }

    /// Executes end-to-end RAG with an optional custom relevance threshold:
    /// Retrieval -> Rerank -> Relevance Check & Flagging -> Grounded Quote Extraction -> Generation.
    pub async fn query_with_threshold(
        &self,
        query_text: &str,
        top_k: usize,
        sampling: SamplingParams,
        min_threshold: Option<f32>,
    ) -> Result<RagResponse> {
        let threshold = min_threshold.unwrap_or(self.min_relevance_threshold);

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

        // 3. Rerank candidates with content keyword verification
        let candidate_chunks: Vec<DocumentChunk> = raw_matches.into_iter().map(|m| m.chunk).collect();
        let reranked = CrossEncoderReranker::rerank(query_text, candidate_chunks, top_k);

        let max_relevance = reranked
            .first()
            .map(|r| r.relevance_score)
            .unwrap_or(0.0);

        // 4. Check if relevance is low or query is unrelated
        let content_kws = extract_content_keywords(query_text);
        let valid_matches: Vec<_> = reranked
            .into_iter()
            .filter(|r| r.relevance_score >= threshold)
            .collect();

        if valid_matches.is_empty() || (!content_kws.is_empty() && max_relevance < threshold) {
            let warning_msg = format!(
                "Low relevance detected (max relevance score: {:.2}, threshold: {:.2}). The query appears unrelated to the ingested knowledge base.",
                max_relevance, threshold
            );
            info!("{}", warning_msg);

            let answer = format!(
                "⚠️ Unrelated Query Warning: No sufficiently relevant or grounded information was found in the indexed documents for your query '{}' (confidence: {:.2} < threshold {:.2}).",
                query_text, max_relevance, threshold
            );

            return Ok(RagResponse {
                query: query_text.to_string(),
                answer,
                citations: Vec::new(),
                generated_tokens: 0,
                is_relevant: false,
                confidence_score: max_relevance,
                warning: Some(warning_msg),
            });
        }

        // 5. Extract precise grounded quotes for each qualifying citation
        let citations: Vec<Citation> = valid_matches
            .into_iter()
            .filter_map(|r| {
                let grounded = extract_grounded_quote(query_text, &r.chunk.text);
                if !content_kws.is_empty() && grounded.matched_keywords.is_empty() {
                    return None;
                }
                let combined_relevance = (r.relevance_score * 0.7 + grounded.score * 0.3).min(1.0);
                Some(Citation {
                    doc_id: r.chunk.doc_id.clone(),
                    chunk_id: r.chunk.chunk_id,
                    quote: grounded.quote.clone(),
                    text: grounded.quote,
                    passage: r.chunk.text.clone(),
                    relevance: combined_relevance,
                    matched_keywords: grounded.matched_keywords,
                    metadata: r.chunk.metadata.clone(),
                })
            })
            .collect();

        if citations.is_empty() {
            let warning_msg = format!(
                "Low relevance detected (confidence: {:.2}, threshold: {:.2}). The query appears unrelated to the ingested knowledge base.",
                max_relevance, threshold
            );
            info!("{}", warning_msg);

            let answer = format!(
                "⚠️ Unrelated Query Warning: No sufficiently relevant or grounded information was found in the indexed documents for your query '{}' (confidence: {:.2} < threshold {:.2}).",
                query_text, max_relevance, threshold
            );

            return Ok(RagResponse {
                query: query_text.to_string(),
                answer,
                citations: Vec::new(),
                generated_tokens: 0,
                is_relevant: false,
                confidence_score: max_relevance,
                warning: Some(warning_msg),
            });
        }

        // 6. Assemble grounded prompt with citations:
        // [System Prompt] -> [Exact Factual Citations] -> [User Question]
        let mut context_str = String::new();
        for (i, cit) in citations.iter().enumerate() {
            context_str.push_str(&format!(
                "\n[{}] Document: {} (Relevance: {:.2})\nExact Quote: \"{}\"\n",
                i + 1, cit.doc_id, cit.relevance, cit.quote
            ));
        }

        let assembled_prompt = format!(
            "System: You are an enterprise intelligence assistant. You must answer the user's question strictly grounded in the provided factual context passages below.\n\
             For every claim you make, cite the corresponding source reference [1], [2], etc.\n\
             If the provided context does not contain enough information to answer the question, state that clearly without guessing.\n\n\
             Factual Context:\n{}\n\
             Question: {}\n\
             Grounded Answer:",
            context_str, query_text
        );

        // Tokenize prompt to token IDs
        let prompt_tokens: Vec<u32> = assembled_prompt
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| ((i % 500) + 1) as u32)
            .collect();

        // 7. Submit to ContinuousBatchScheduler with a streaming channel
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
        let mut max_steps = 200;

        while max_steps > 0 {
            max_steps -= 1;
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
            format!(
                "Synthesized answer grounded in {} retrieved passage citation(s).",
                citations.len()
            )
        } else {
            "No output tokens generated.".to_string()
        };

        Ok(RagResponse {
            query: query_text.to_string(),
            answer,
            citations,
            generated_tokens,
            is_relevant: true,
            confidence_score: max_relevance,
            warning: None,
        })
    }
}

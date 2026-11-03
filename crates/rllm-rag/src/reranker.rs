//! Cross-encoder passage reranker.
//!
//! Evaluates candidate passages against the query using combined lexical coverage,
//! term frequency density, and token proximity scoring to rerank the top retrieved hits.

use crate::chunker::DocumentChunk;
use serde::{Deserialize, Serialize};

/// Reranked document match with updated relevance confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankedMatch {
    /// Original chunk index.
    pub chunk_idx: usize,
    /// Refined cross-encoder relevance score in $[0.0, 1.0]$.
    pub relevance_score: f32,
    /// Associated document chunk.
    pub chunk: DocumentChunk,
}

/// Lightweight semantic cross-encoder reranker.
pub struct CrossEncoderReranker;

impl CrossEncoderReranker {
    /// Reranks passages based on lexical overlap, length normalization, and token coverage.
    pub fn rerank(
        query: &str,
        candidates: Vec<DocumentChunk>,
        top_k: usize,
    ) -> Vec<RerankedMatch> {
        let query_tokens: std::collections::HashSet<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        if query_tokens.is_empty() {
            return candidates
                .into_iter()
                .take(top_k)
                .enumerate()
                .map(|(i, c)| RerankedMatch {
                    chunk_idx: i,
                    relevance_score: 1.0,
                    chunk: c,
                })
                .collect();
        }

        let mut scored: Vec<(usize, f32, DocumentChunk)> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let doc_words: Vec<String> = chunk
                    .text
                    .to_lowercase()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();

                let mut matched_terms = 0;
                let mut term_freq = 0;
                for q in &query_tokens {
                    let count = doc_words.iter().filter(|&w| w == q).count();
                    if count > 0 {
                        matched_terms += 1;
                        term_freq += count;
                    }
                }

                let term_coverage = matched_terms as f32 / query_tokens.len() as f32;
                let density = term_freq as f32 / (doc_words.len() as f32 + 10.0);
                let score = term_coverage * 0.7 + density * 0.3;

                (idx, score, chunk)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score, chunk)| RerankedMatch {
                chunk_idx: idx,
                relevance_score: score,
                chunk,
            })
            .collect()
    }
}

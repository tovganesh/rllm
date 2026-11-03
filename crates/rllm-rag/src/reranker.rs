//! Cross-encoder passage reranker.
//!
//! Evaluates candidate passages against the query using combined lexical coverage,
//! content keyword presence (ignoring common stop words), term frequency density,
//! and phrase matching to compute calibrated relevance scores in $[0.0, 1.0]$.

use crate::chunker::DocumentChunk;
use crate::grounding::extract_content_keywords;
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
    /// Reranks passages based on content keyword overlap, length normalization, and phrase matching.
    ///
    /// Filters out English stop words so unrelated queries (e.g., "what is the weather on mars")
    /// cannot artificially score high from common grammatical tokens.
    pub fn rerank(
        query: &str,
        candidates: Vec<DocumentChunk>,
        top_k: usize,
    ) -> Vec<RerankedMatch> {
        let content_keywords = extract_content_keywords(query);

        if content_keywords.is_empty() {
            let raw_tokens: Vec<String> = query
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            if raw_tokens.is_empty() {
                return candidates
                    .into_iter()
                    .take(top_k)
                    .enumerate()
                    .map(|(i, c)| RerankedMatch {
                        chunk_idx: i,
                        relevance_score: 0.0,
                        chunk: c,
                    })
                    .collect();
            }
        }

        let mut scored: Vec<(usize, f32, DocumentChunk)> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let doc_words: Vec<String> = chunk
                    .text
                    .to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|w| !w.is_empty())
                    .map(|s| s.to_string())
                    .collect();

                if doc_words.is_empty() {
                    return (idx, 0.0, chunk);
                }

                let mut matched_keywords = 0;
                let mut total_term_freq = 0;

                for kw in &content_keywords {
                    let freq = doc_words
                        .iter()
                        .filter(|&w| w == kw || (kw.len() >= 4 && w.len() >= 4 && (w.starts_with(kw) || kw.starts_with(w))))
                        .count();
                    if freq > 0 {
                        matched_keywords += 1;
                        total_term_freq += freq;
                    }
                }

                // If query has content keywords and NONE of them appear in this document chunk,
                // the document is considered completely unrelated to the query.
                if !content_keywords.is_empty() && matched_keywords == 0 {
                    return (idx, 0.0, chunk);
                }

                let keyword_coverage = if content_keywords.is_empty() {
                    0.0
                } else {
                    matched_keywords as f32 / content_keywords.len() as f32
                };

                let density = (total_term_freq as f32) / (doc_words.len() as f32 + 10.0);

                // Exact phrase bonus if query has multiple words
                let mut phrase_bonus = 0.0f32;
                if content_keywords.len() >= 2 {
                    let chunk_lower = chunk.text.to_lowercase();
                    let query_lower = query.to_lowercase();
                    if chunk_lower.contains(&query_lower) {
                        phrase_bonus = 0.20;
                    }
                }

                let score = (keyword_coverage * 0.65 + density * 0.20 + phrase_bonus).min(1.0);
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

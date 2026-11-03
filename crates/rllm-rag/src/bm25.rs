//! BM25 Okapi lexical sparse inverted index.
//!
//! Provides tokenization, inverted postings, document frequency weighting,
//! and BM25 scoring for exact keyword and term frequency matching.

use crate::chunker::DocumentChunk;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A matched document chunk returned from BM25 lexical search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Match {
    /// Internal index of the matched document chunk.
    pub chunk_idx: usize,
    /// BM25 Okapi relevance score.
    pub score: f32,
    /// Document chunk contents and metadata.
    pub chunk: DocumentChunk,
}

/// Inverted index implementing BM25 Okapi lexical information retrieval.
pub struct Bm25Index {
    /// Term frequency saturation parameter ($k_1 \approx 1.2\text{--}2.0$).
    pub k1: f32,
    /// Document length normalization parameter ($b \approx 0.75$).
    pub b: f32,
    chunks: Vec<DocumentChunk>,
    // term -> list of (doc_idx, term_frequency)
    inverted_index: HashMap<String, Vec<(usize, u32)>>,
    doc_lengths: Vec<usize>,
    total_tokens: usize,
}

impl Bm25Index {
    /// Constructs a new [`Bm25Index`] with the given $k_1$ and $b$ tuning hyperparameters.
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            chunks: Vec::new(),
            inverted_index: HashMap::new(),
            doc_lengths: Vec::new(),
            total_tokens: 0,
        }
    }

    /// Number of document passages indexed.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// True if no documents have been added to the index.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Lowercases and splits text on non-alphanumeric character boundaries.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Inserts a [`DocumentChunk`] into the inverted index.
    pub fn insert(&mut self, chunk: DocumentChunk) {
        let doc_idx = self.chunks.len();
        let tokens = Self::tokenize(&chunk.text);
        let len = tokens.len();

        let mut term_counts: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *term_counts.entry(token).or_insert(0) += 1;
        }

        for (term, count) in term_counts {
            self.inverted_index
                .entry(term)
                .or_default()
                .push((doc_idx, count));
        }

        self.doc_lengths.push(len);
        self.total_tokens += len;
        self.chunks.push(chunk);
    }

    /// Searches the index using BM25 scoring for terms present in the query string.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<Bm25Match> {
        let query_tokens = Self::tokenize(query);
        if query_tokens.is_empty() || self.chunks.is_empty() {
            return Vec::new();
        }

        let n = self.chunks.len() as f32;
        let avg_dl = self.total_tokens as f32 / n;
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in &query_tokens {
            if let Some(postings) = self.inverted_index.get(term) {
                let df = postings.len() as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(doc_idx, tf) in postings {
                    let dl = self.doc_lengths[doc_idx] as f32;
                    let tf_f = tf as f32;
                    let denom = tf_f + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl));
                    let term_score = idf * (tf_f * (self.k1 + 1.0)) / denom;

                    *scores.entry(doc_idx).or_insert(0.0) += term_score;
                }
            }
        }

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        ranked
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| Bm25Match {
                chunk_idx: idx,
                score,
                chunk: self.chunks[idx].clone(),
            })
            .collect()
    }
}

use crate::chunker::DocumentChunk;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Match {
    pub chunk_idx: usize,
    pub score: f32,
    pub chunk: DocumentChunk,
}

pub struct Bm25Index {
    pub k1: f32,
    pub b: f32,
    chunks: Vec<DocumentChunk>,
    // term -> list of (doc_idx, term_frequency)
    inverted_index: HashMap<String, Vec<(usize, u32)>>,
    doc_lengths: Vec<usize>,
    total_tokens: usize,
}

impl Bm25Index {
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

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

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


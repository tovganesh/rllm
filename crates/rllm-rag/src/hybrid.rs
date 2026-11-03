//! Hybrid retrieval combining dense vector search and sparse BM25.
//!
//! Uses Reciprocal Rank Fusion (RRF) to merge candidate rankings from dense HNSW
//! vector similarity and sparse BM25 keyword matching into a single unified result.

use crate::bm25::Bm25Index;
use crate::chunker::DocumentChunk;
use crate::hnsw::HnswIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A matched passage produced by hybrid dense-sparse search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridMatch {
    /// Internal chunk index.
    pub chunk_idx: usize,
    /// Fused Reciprocal Rank Fusion (RRF) score.
    pub rrf_score: f32,
    /// Dense vector cosine similarity score if matched in HNSW.
    pub dense_score: Option<f32>,
    /// Sparse BM25 score if matched in inverted index.
    pub bm25_score: Option<f32>,
    /// Associated document chunk.
    pub chunk: DocumentChunk,
}

/// Unified index combining [`HnswIndex`] and [`Bm25Index`] via Reciprocal Rank Fusion.
pub struct HybridIndex {
    /// Dense vector approximate nearest neighbor index.
    pub hnsw: HnswIndex,
    /// Sparse BM25 inverted lexical index.
    pub bm25: Bm25Index,
    /// Scaling weight applied to dense vector RRF scores.
    pub dense_weight: f32,
    /// Scaling weight applied to sparse BM25 RRF scores.
    pub bm25_weight: f32,
    /// Smoothing constant $k$ in RRF formula ($RRF(d) = \sum \frac{w}{k + r_i}$).
    pub rrf_k: f32,
}

impl HybridIndex {
    /// Constructs a new [`HybridIndex`].
    pub fn new(dim: usize, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            hnsw: HnswIndex::new(dim, m, ef_construction, ef_search),
            bm25: Bm25Index::new(1.2, 0.75),
            dense_weight: 1.0,
            bm25_weight: 1.0,
            rrf_k: 60.0,
        }
    }

    /// Inserts a chunk and its corresponding embedding into both indices.
    pub fn insert(&mut self, vector: Vec<f32>, chunk: DocumentChunk) {
        self.bm25.insert(chunk.clone());
        self.hnsw.insert(vector, chunk);
    }

    /// Number of passages indexed.
    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    /// True if the index contains no passages.
    pub fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
    }

    /// Performs hybrid search over dense vector embeddings and lexical text, fusing ranks via RRF.
    pub fn search(
        &self,
        query_text: &str,
        query_vector: &[f32],
        top_k: usize,
    ) -> Vec<HybridMatch> {
        let dense_results = self.hnsw.search(query_vector, top_k * 2);
        let bm25_results = self.bm25.search(query_text, top_k * 2);

        let mut rrf_scores: HashMap<usize, f32> = HashMap::new();
        let mut dense_scores: HashMap<usize, f32> = HashMap::new();
        let mut bm25_scores: HashMap<usize, f32> = HashMap::new();
        let mut chunks: HashMap<usize, DocumentChunk> = HashMap::new();

        // 1. Accumulate dense ranks
        for (rank, hit) in dense_results.into_iter().enumerate() {
            let rrf = self.dense_weight / (self.rrf_k + (rank + 1) as f32);
            *rrf_scores.entry(hit.chunk_idx).or_insert(0.0) += rrf;
            dense_scores.insert(hit.chunk_idx, hit.score);
            chunks.entry(hit.chunk_idx).or_insert(hit.chunk);
        }

        // 2. Accumulate BM25 ranks
        for (rank, hit) in bm25_results.into_iter().enumerate() {
            let rrf = self.bm25_weight / (self.rrf_k + (rank + 1) as f32);
            *rrf_scores.entry(hit.chunk_idx).or_insert(0.0) += rrf;
            bm25_scores.insert(hit.chunk_idx, hit.score);
            chunks.entry(hit.chunk_idx).or_insert(hit.chunk);
        }

        let mut fused: Vec<(usize, f32)> = rrf_scores.into_iter().collect();
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        fused
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| HybridMatch {
                chunk_idx: idx,
                rrf_score: score,
                dense_score: dense_scores.get(&idx).copied(),
                bm25_score: bm25_scores.get(&idx).copied(),
                chunk: chunks.remove(&idx).unwrap(),
            })
            .collect()
    }
}

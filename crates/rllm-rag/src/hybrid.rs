use crate::bm25::Bm25Index;
use crate::chunker::DocumentChunk;
use crate::hnsw::HnswIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridMatch {
    pub chunk_idx: usize,
    pub rrf_score: f32,
    pub dense_score: Option<f32>,
    pub bm25_score: Option<f32>,
    pub chunk: DocumentChunk,
}

pub struct HybridIndex {
    pub hnsw: HnswIndex,
    pub bm25: Bm25Index,
    pub dense_weight: f32,
    pub bm25_weight: f32,
    pub rrf_k: f32,
}

impl HybridIndex {
    pub fn new(dim: usize, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            hnsw: HnswIndex::new(dim, m, ef_construction, ef_search),
            bm25: Bm25Index::new(1.2, 0.75),
            dense_weight: 1.0,
            bm25_weight: 1.0,
            rrf_k: 60.0,
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>, chunk: DocumentChunk) {
        self.bm25.insert(chunk.clone());
        self.hnsw.insert(vector, chunk);
    }

    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
    }

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


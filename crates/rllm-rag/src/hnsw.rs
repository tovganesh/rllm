use crate::chunker::DocumentChunk;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let mut sum_v = unsafe { vdupq_n_f32(0.0) };
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                sum_v = vfmaq_f32(sum_v, va, vb);
            }
            i += 4;
        }
        let mut sum = unsafe { vaddvq_f32(sum_v) };
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        let mut i = 0;
        while i + 4 <= len {
            sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            i += 4;
        }
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMatch {
    pub chunk_idx: usize,
    pub score: f32,
    pub chunk: DocumentChunk,
}

#[derive(Copy, Clone, PartialEq)]
struct Candidate {
    id: usize,
    dist: f32, // distance = 1.0 - similarity
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct HnswIndex {
    pub dim: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub ml: f64,
    vectors: Vec<Vec<f32>>,
    chunks: Vec<DocumentChunk>,
    // layers[level][node_id] = list of neighbor IDs
    layers: Vec<Vec<Vec<usize>>>,
    entry_point: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn new(dim: usize, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let ml = 1.0 / (m as f64).ln();
        Self {
            dim,
            m,
            ef_construction,
            ef_search,
            ml,
            vectors: Vec::new(),
            chunks: Vec::new(),
            layers: Vec::new(),
            entry_point: None,
            max_level: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>().max(1e-9);
        (-r.ln() * self.ml) as usize
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let sim = cosine_similarity(a, b);
        1.0 - sim
    }

    pub fn insert(&mut self, vector: Vec<f32>, chunk: DocumentChunk) {
        assert_eq!(vector.len(), self.dim);
        let id = self.vectors.len();
        self.vectors.push(vector.clone());
        self.chunks.push(chunk);

        let node_level = self.random_level();
        while self.layers.len() <= node_level {
            self.layers.push(Vec::new());
        }
        for lvl in 0..=node_level {
            if self.layers[lvl].len() <= id {
                self.layers[lvl].resize(id + 1, Vec::new());
            }
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_level = node_level;
            return;
        }

        let mut curr_entry = self.entry_point.unwrap();
        let mut curr_dist = self.distance(&vector, &self.vectors[curr_entry]);

        // 1. Search greedily from top level down to node_level + 1
        for lvl in (node_level + 1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if curr_entry < self.layers[lvl].len() {
                    for &neighbor in &self.layers[lvl][curr_entry] {
                        let d = self.distance(&vector, &self.vectors[neighbor]);
                        if d < curr_dist {
                            curr_dist = d;
                            curr_entry = neighbor;
                            changed = true;
                        }
                    }
                }
            }
        }

        // 2. Insert connections from min(node_level, max_level) down to level 0
        let mut enter_pts = vec![curr_entry];
        for lvl in (0..=node_level.min(self.max_level)).rev() {
            let candidates = self.search_layer(&vector, &enter_pts, self.ef_construction, lvl);
            let m_max = if lvl == 0 { 2 * self.m } else { self.m };
            let neighbors: Vec<usize> = candidates.iter().take(m_max).map(|c| c.id).collect();

            // Link bidirectionally
            self.layers[lvl][id] = neighbors.clone();
            for &n_id in &neighbors {
                self.layers[lvl][n_id].push(id);
                if self.layers[lvl][n_id].len() > m_max {
                    // Prune furthest neighbor
                    let mut n_dists: Vec<(usize, f32)> = self.layers[lvl][n_id]
                        .iter()
                        .map(|&other| (other, self.distance(&self.vectors[n_id], &self.vectors[other])))
                        .collect();
                    n_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    n_dists.truncate(m_max);
                    self.layers[lvl][n_id] = n_dists.into_iter().map(|(o, _)| o).collect();
                }
            }

            enter_pts = neighbors;
        }

        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(id);
        }
    }

    fn search_layer(
        &self,
        query: &[f32],
        enter_points: &[usize],
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // max-heap
        let mut nearest = BinaryHeap::new();

        for &ep in enter_points {
            let d = self.distance(query, &self.vectors[ep]);
            visited.insert(ep);
            candidates.push(std::cmp::Reverse(Candidate { id: ep, dist: d }));
            nearest.push(Candidate { id: ep, dist: d });
        }

        while let Some(std::cmp::Reverse(curr)) = candidates.pop() {
            if let Some(furthest) = nearest.peek() {
                if curr.dist > furthest.dist {
                    break;
                }
            }

            if curr.id < self.layers[level].len() {
                for &neighbor in &self.layers[level][curr.id] {
                    if visited.insert(neighbor) {
                        let d = self.distance(query, &self.vectors[neighbor]);
                        if nearest.len() < ef || d < nearest.peek().unwrap().dist {
                            candidates.push(std::cmp::Reverse(Candidate { id: neighbor, dist: d }));
                            nearest.push(Candidate { id: neighbor, dist: d });
                            if nearest.len() > ef {
                                nearest.pop();
                            }
                        }
                    }
                }
            }
        }

        // into_sorted_vec() returns elements in ascending order (smallest distance first)
        nearest.into_sorted_vec()
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<VectorMatch> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let mut curr_entry = self.entry_point.unwrap();
        let mut curr_dist = self.distance(query, &self.vectors[curr_entry]);

        for lvl in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if curr_entry < self.layers[lvl].len() {
                    for &neighbor in &self.layers[lvl][curr_entry] {
                        let d = self.distance(query, &self.vectors[neighbor]);
                        if d < curr_dist {
                            curr_dist = d;
                            curr_entry = neighbor;
                            changed = true;
                        }
                    }
                }
            }
        }

        let ef = self.ef_search.max(top_k);
        let candidates = self.search_layer(query, &[curr_entry], ef, 0);

        candidates
            .into_iter()
            .take(top_k)
            .map(|c| VectorMatch {
                chunk_idx: c.id,
                score: 1.0 - c.dist, // return similarity
                chunk: self.chunks[c.id].clone(),
            })
            .collect()
    }
}

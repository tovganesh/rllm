//! Radix-tree prefix cache for KV-cache sharing between sequences.
//!
//! Stores token prefixes mapped to the physical KV blocks that hold their keys and
//! values. When a new sequence shares a block-aligned prefix with a cached one, those
//! blocks are reused and the matching tokens are not recomputed.
//!
//! ## Ownership
//!
//! The cache holds **one allocator reference** to every block it records.
//! [`PrefixCache::insert_prefix`] returns the blocks it newly recorded so the caller can
//! take that reference, and [`PrefixCache::evict_one`] returns the block it dropped so
//! the caller can release it. A cached block is therefore never on the free list.

use std::collections::HashMap;

/// A node of the prefix tree: one full KV block, keyed (in its parent) by the
/// `block_size` tokens it holds.
#[derive(Debug, Default)]
struct Node {
    /// Physical block holding these tokens' keys and values (unused for the root).
    block_id: usize,
    /// Logical clock of the last match or insert that touched this node.
    last_access: u64,
    /// Children keyed by the next block's tokens.
    children: HashMap<Box<[u32]>, Node>,
}

/// Block-granular radix tree mapping token prefixes to the KV blocks that hold them.
///
/// Every node is one full block, so lookups cost one hash probe per block (not per
/// token) and the tree has at most as many nodes as there are KV blocks.
#[derive(Debug)]
pub struct PrefixCache {
    /// Block size in tokens for physical block alignment.
    pub block_size: usize,
    root: Node,
    clock: u64,
    num_blocks: usize,
}

impl PrefixCache {
    /// Constructs a new [`PrefixCache`] with the specified block size.
    pub fn new(block_size: usize) -> Self {
        Self { block_size, root: Node::default(), clock: 0, num_blocks: 0 }
    }

    /// Number of physical blocks currently recorded (and referenced) by the cache.
    pub fn num_cached_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Finds the longest cached block-aligned prefix of `tokens`.
    ///
    /// Returns `(number of tokens covered, physical block IDs in order)` and marks the
    /// matched path as recently used. The caller must take its own reference to every
    /// returned block before using it.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<usize>) {
        self.clock += 1;
        let now = self.clock;
        let mut curr = &mut self.root;
        let mut blocks = Vec::new();
        for chunk in tokens.chunks_exact(self.block_size) {
            match curr.children.get_mut(chunk) {
                Some(child) => {
                    child.last_access = now;
                    blocks.push(child.block_id);
                    curr = child;
                }
                None => break,
            }
        }
        (blocks.len() * self.block_size, blocks)
    }

    /// Records the full blocks of `tokens`, whose KV entries live in `block_ids`.
    ///
    /// Only complete blocks are recorded. Where a block-aligned prefix is already cached
    /// (possibly under a different physical block) the existing entry is kept.
    ///
    /// Returns the block IDs newly recorded; the caller must add one allocator reference
    /// to each of them on behalf of the cache.
    pub fn insert_prefix(&mut self, tokens: &[u32], block_ids: &[usize]) -> Vec<usize> {
        let mut newly_cached = Vec::new();
        self.clock += 1;
        let now = self.clock;
        let mut curr = &mut self.root;
        for (chunk, &bid) in tokens.chunks_exact(self.block_size).zip(block_ids) {
            let child = curr.children.entry(chunk.into()).or_insert_with(|| {
                newly_cached.push(bid);
                Node { block_id: bid, last_access: now, children: HashMap::new() }
            });
            child.last_access = now;
            curr = child;
        }
        self.num_blocks += newly_cached.len();
        newly_cached
    }

    /// Evicts the least-recently-used *leaf* block (one with no cached descendants)
    /// for which `can_evict` returns true, and returns its block ID. The caller must
    /// release the cache's allocator reference to it.
    ///
    /// Typical use passes `|b| allocator.ref_count(b) == 1`, i.e. only blocks no running
    /// sequence is using.
    pub fn evict_one(&mut self, can_evict: impl Fn(usize) -> bool) -> Option<usize> {
        let mut best: Option<(u64, Vec<Box<[u32]>>)> = None;
        let mut path = Vec::new();
        find_lru_leaf(&self.root, &mut path, &can_evict, &mut best);
        let (_, victim) = best?;
        let (last, parents) = victim.split_last()?;
        let mut node = &mut self.root;
        for key in parents {
            node = node.children.get_mut(key)?;
        }
        let removed = node.children.remove(last)?;
        self.num_blocks -= 1;
        Some(removed.block_id)
    }
}

/// Finds the least-recently-used evictable leaf below `node`.
fn find_lru_leaf(
    node: &Node,
    path: &mut Vec<Box<[u32]>>,
    can_evict: &impl Fn(usize) -> bool,
    best: &mut Option<(u64, Vec<Box<[u32]>>)>,
) {
    for (key, child) in &node.children {
        path.push(key.clone());
        if child.children.is_empty() {
            if can_evict(child.block_id) && best.as_ref().is_none_or(|(t, _)| child.last_access < *t) {
                *best = Some((child.last_access, path.clone()));
            }
        } else {
            find_lru_leaf(child, path, can_evict, best);
        }
        path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_reports_only_newly_recorded_blocks() {
        let mut c = PrefixCache::new(2);
        assert_eq!(c.insert_prefix(&[1, 2, 3, 4, 5], &[10, 11, 12]), vec![10, 11]);
        // Same prefix under different physical blocks: existing entries are kept.
        assert_eq!(c.insert_prefix(&[1, 2, 3, 4], &[20, 21]), Vec::<usize>::new());
        // Extending the prefix records only the new block.
        assert_eq!(c.insert_prefix(&[1, 2, 3, 4, 5, 6], &[20, 21, 22]), vec![22]);
        assert_eq!(c.num_cached_blocks(), 3);
        assert_eq!(c.match_prefix(&[1, 2, 3, 4, 5, 6, 7]), (6, vec![10, 11, 22]));
    }

    #[test]
    fn eviction_takes_lru_leaves_and_respects_predicate() {
        let mut c = PrefixCache::new(2);
        c.insert_prefix(&[1, 2, 3, 4], &[10, 11]); // chain 10 -> 11
        c.insert_prefix(&[5, 6], &[20]);
        c.match_prefix(&[1, 2, 3, 4]); // chain A is now more recent than B

        // Leaves are 11 and 20; 20 is least recently used.
        assert_eq!(c.evict_one(|_| true), Some(20));
        // 10 has a cached descendant, so 11 must go before it.
        assert_eq!(c.evict_one(|_| true), Some(11));
        // Predicate can protect blocks that sequences still use.
        assert_eq!(c.evict_one(|b| b != 10), None);
        assert_eq!(c.evict_one(|_| true), Some(10));
        assert_eq!(c.num_cached_blocks(), 0);
        assert_eq!(c.match_prefix(&[1, 2, 3, 4]), (0, vec![]));
    }
}

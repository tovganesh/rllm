//! Radix-tree prefix cache for prompt KV-cache sharing.
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

/// A node in the prefix tree (one node per token).
#[derive(Debug)]
pub struct RadixNode {
    /// Token ID associated with this node.
    pub token: u32,
    /// Physical block ID if this node ends a full block.
    pub block_id: Option<usize>,
    /// Logical clock of the last match or insert that touched this node.
    pub last_access: u64,
    /// Child nodes keyed by the next token ID.
    pub children: HashMap<u32, RadixNode>,
}

impl RadixNode {
    /// Constructs a new empty node for a given token.
    pub fn new(token: u32) -> Self {
        Self {
            token,
            block_id: None,
            last_access: 0,
            children: HashMap::new(),
        }
    }
}

/// Radix-tree prefix cache for prompt key-value block reuse.
#[derive(Debug)]
pub struct PrefixCache {
    /// Block size in tokens for physical block alignment.
    pub block_size: usize,
    root: RadixNode,
    clock: u64,
    num_blocks: usize,
}

impl PrefixCache {
    /// Constructs a new [`PrefixCache`] with the specified block size.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            root: RadixNode::new(0),
            clock: 0,
            num_blocks: 0,
        }
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
        let mut matched_blocks = Vec::new();
        let mut matched_tokens = 0;

        for (i, &token) in tokens.iter().enumerate() {
            match curr.children.get_mut(&token) {
                Some(child) => {
                    curr = child;
                    curr.last_access = now;
                    if (i + 1) % self.block_size == 0 {
                        match curr.block_id {
                            Some(bid) => {
                                matched_blocks.push(bid);
                                matched_tokens = i + 1;
                            }
                            None => break,
                        }
                    }
                }
                None => break,
            }
        }

        (matched_tokens, matched_blocks)
    }

    /// Records the full blocks of `tokens`, whose KV entries live in `block_ids`.
    ///
    /// Only complete blocks are recorded. Where a block-aligned prefix is already cached
    /// (possibly under a different physical block) the existing entry is kept.
    ///
    /// Returns the block IDs newly recorded; the caller must add one allocator reference
    /// to each of them on behalf of the cache.
    pub fn insert_prefix(&mut self, tokens: &[u32], block_ids: &[usize]) -> Vec<usize> {
        let blocks_to_insert = (tokens.len() / self.block_size).min(block_ids.len());
        let mut newly_cached = Vec::new();
        if blocks_to_insert == 0 {
            return newly_cached;
        }
        self.clock += 1;
        let now = self.clock;

        let mut curr = &mut self.root;
        for (i, &token) in tokens[..blocks_to_insert * self.block_size].iter().enumerate() {
            curr = curr
                .children
                .entry(token)
                .or_insert_with(|| RadixNode::new(token));
            curr.last_access = now;

            if (i + 1) % self.block_size == 0 && curr.block_id.is_none() {
                let bid = block_ids[i / self.block_size];
                curr.block_id = Some(bid);
                newly_cached.push(bid);
            }
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
        let mut candidates = Vec::new();
        let mut path = Vec::new();
        collect_leaf_blocks(&self.root, &mut path, &mut candidates);

        let (_, victim_path, block_id) = candidates
            .into_iter()
            .filter(|(_, _, b)| can_evict(*b))
            .min_by_key(|(t, _, _)| *t)?;

        remove_block(&mut self.root, &victim_path);
        self.num_blocks -= 1;
        Some(block_id)
    }
}

/// Collects `(last_access, token path, block_id)` for every block node that has no
/// block-holding descendants. Returns whether `node`'s subtree holds any block.
fn collect_leaf_blocks(
    node: &RadixNode,
    path: &mut Vec<u32>,
    out: &mut Vec<(u64, Vec<u32>, usize)>,
) -> bool {
    let mut subtree_has_block = false;
    for (&tok, child) in &node.children {
        path.push(tok);
        subtree_has_block |= collect_leaf_blocks(child, path, out);
        path.pop();
    }
    if let Some(bid) = node.block_id {
        if !subtree_has_block {
            out.push((node.last_access, path.clone(), bid));
        }
        return true;
    }
    subtree_has_block
}

/// Clears the block at the end of `path` and prunes nodes left with no block and no
/// children. Returns true if `node` itself should be removed by its parent.
fn remove_block(node: &mut RadixNode, path: &[u32]) -> bool {
    match path.split_first() {
        None => node.block_id = None,
        Some((tok, rest)) => {
            let remove_child = node
                .children
                .get_mut(tok)
                .map(|child| remove_block(child, rest))
                .unwrap_or(false);
            if remove_child {
                node.children.remove(tok);
            }
        }
    }
    node.block_id.is_none() && node.children.is_empty()
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

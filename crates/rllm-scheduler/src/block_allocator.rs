//! PagedAttention physical block allocator and reference counter.
//!
//! Manages fixed-size physical KV-cache blocks, supporting O(1) allocation,
//! deallocation, reference counting for prefix sharing, and automated table resizing.

use parking_lot::Mutex;
use rllm_core::{
    backend::BlockTable,
    error::{Result, RllmError},
};
use std::sync::Arc;

/// Fixed-size physical memory block allocator for paged attention key-value caches.
///
/// Tracks free physical block IDs in a stack-like pool and maintains reference counts
/// across blocks to allow zero-copy prefix sharing across multiple sequences.
#[derive(Debug)]
pub struct BlockAllocator {
    /// Number of token slots held within each physical block (typically 16 or 32).
    pub block_size: usize,
    /// Total count of physical blocks allocated in GPU/CPU VRAM.
    pub total_blocks: usize,
    free_blocks: Vec<usize>,
    ref_counts: Vec<usize>,
}

impl BlockAllocator {
    /// Constructs a new [`BlockAllocator`] with the specified block size and total block count.
    pub fn new(block_size: usize, total_blocks: usize) -> Self {
        let free_blocks = (0..total_blocks).rev().collect();
        let ref_counts = vec![0; total_blocks];
        Self {
            block_size,
            total_blocks,
            free_blocks,
            ref_counts,
        }
    }

    /// Returns the number of unallocated physical blocks remaining.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Allocates an unused physical block ID and initializes its reference count to 1.
    ///
    /// # Errors
    ///
    /// Returns [`RllmError::OutOfMemory`] when all blocks are currently in use.
    pub fn allocate_block(&mut self) -> Result<usize> {
        let block_id = self.free_blocks.pop().ok_or_else(|| {
            RllmError::OutOfMemory {
                requested_bytes: self.block_size * 4,
                device: "kv_cache".to_string(),
            }
        })?;
        self.ref_counts[block_id] = 1;
        Ok(block_id)
    }

    /// Increments the reference count of an **allocated** physical block (e.g. when a
    /// sequence or the prefix cache starts sharing it).
    ///
    /// # Panics
    ///
    /// Panics if the block is currently free: sharing a free block would let the
    /// allocator hand it out again while it is still referenced.
    pub fn inc_ref(&mut self, block_id: usize) {
        assert!(
            self.ref_counts[block_id] > 0,
            "inc_ref on free KV block {}",
            block_id
        );
        self.ref_counts[block_id] += 1;
    }

    /// Current reference count of a block (0 means free).
    pub fn ref_count(&self, block_id: usize) -> usize {
        self.ref_counts.get(block_id).copied().unwrap_or(0)
    }

    /// Decrements the reference count of a block, recycling it back into the free pool when it hits 0.
    pub fn free_block(&mut self, block_id: usize) {
        if block_id >= self.total_blocks {
            return;
        }
        if self.ref_counts[block_id] > 0 {
            self.ref_counts[block_id] -= 1;
            if self.ref_counts[block_id] == 0 {
                self.free_blocks.push(block_id);
            }
        }
    }

    /// Releases all blocks referenced in a sequence's [`BlockTable`] and empties it.
    pub fn free_table(&mut self, table: &mut BlockTable) {
        for &block_id in &table.block_ids {
            self.free_block(block_id);
        }
        table.block_ids.clear();
    }

    /// Ensures that the provided [`BlockTable`] contains sufficient physical blocks
    /// to accommodate `token_count` total tokens.
    ///
    /// # Errors
    ///
    /// Returns [`RllmError::OutOfMemory`] if insufficient free blocks remain to satisfy capacity.
    pub fn ensure_capacity(
        &mut self,
        table: &mut BlockTable,
        token_count: usize,
    ) -> Result<()> {
        let needed_blocks = token_count.div_ceil(self.block_size);
        while table.block_ids.len() < needed_blocks {
            let block_id = self.allocate_block()?;
            table.block_ids.push(block_id);
        }
        Ok(())
    }
}

/// Thread-safe shared block allocator wrapped in an `Arc<Mutex<BlockAllocator>>`.
pub type SharedBlockAllocator = Arc<Mutex<BlockAllocator>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "inc_ref on free KV block")]
    fn inc_ref_on_free_block_panics() {
        let mut a = BlockAllocator::new(4, 2);
        a.inc_ref(0);
    }

    #[test]
    fn shared_block_is_freed_only_after_last_reference() {
        let mut a = BlockAllocator::new(4, 2);
        let b = a.allocate_block().unwrap();
        a.inc_ref(b);
        a.free_block(b);
        assert_eq!(a.num_free_blocks(), 1);
        a.free_block(b);
        assert_eq!(a.num_free_blocks(), 2);
    }
}

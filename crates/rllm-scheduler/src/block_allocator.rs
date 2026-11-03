use parking_lot::Mutex;
use rllm_core::{
    backend::BlockTable,
    error::{Result, RllmError},
};
use std::sync::Arc;

#[derive(Debug)]
pub struct BlockAllocator {
    pub block_size: usize,
    pub total_blocks: usize,
    free_blocks: Vec<usize>,
    ref_counts: Vec<usize>,
}

impl BlockAllocator {
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

    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

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

    pub fn inc_ref(&mut self, block_id: usize) {
        self.ref_counts[block_id] += 1;
    }

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

    pub fn free_table(&mut self, table: &BlockTable) {
        for &block_id in &table.block_ids {
            self.free_block(block_id);
        }
    }

    /// Allocate or append blocks needed to hold `token_count` total tokens
    pub fn ensure_capacity(
        &mut self,
        table: &mut BlockTable,
        token_count: usize,
    ) -> Result<()> {
        let needed_blocks = (token_count + self.block_size - 1) / self.block_size;
        while table.block_ids.len() < needed_blocks {
            let block_id = self.allocate_block()?;
            table.block_ids.push(block_id);
        }
        Ok(())
    }
}

pub type SharedBlockAllocator = Arc<Mutex<BlockAllocator>>;


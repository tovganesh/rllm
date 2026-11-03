use std::collections::HashMap;

#[derive(Debug)]
pub struct RadixNode {
    pub token: u32,
    pub block_id: Option<usize>,
    pub children: HashMap<u32, RadixNode>,
}

impl RadixNode {
    pub fn new(token: u32) -> Self {
        Self {
            token,
            block_id: None,
            children: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct PrefixCache {
    pub block_size: usize,
    root: RadixNode,
}

impl PrefixCache {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            root: RadixNode::new(0),
        }
    }

    /// Finds the longest matching prefix of cached blocks for a token sequence.
    /// Returns: (number of tokens cached, list of physical block IDs).
    pub fn match_prefix(&self, tokens: &[u32]) -> (usize, Vec<usize>) {
        let mut curr = &self.root;
        let mut matched_blocks = Vec::new();
        let mut matched_tokens = 0;

        for (i, &token) in tokens.iter().enumerate() {
            if let Some(child) = curr.children.get(&token) {
                curr = child;
                if (i + 1) % self.block_size == 0 {
                    if let Some(bid) = curr.block_id {
                        matched_blocks.push(bid);
                        matched_tokens = i + 1;
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }

        (matched_tokens, matched_blocks)
    }

    /// Inserts a token sequence associated with allocated physical block IDs.
    pub fn insert_prefix(&mut self, tokens: &[u32], block_ids: &[usize]) {
        let num_complete_blocks = tokens.len() / self.block_size;
        let blocks_to_insert = num_complete_blocks.min(block_ids.len());
        if blocks_to_insert == 0 {
            return;
        }

        let mut curr = &mut self.root;
        for (i, &token) in tokens.iter().enumerate() {
            let block_idx = i / self.block_size;
            if block_idx >= blocks_to_insert {
                break;
            }

            curr = curr
                .children
                .entry(token)
                .or_insert_with(|| RadixNode::new(token));

            if (i + 1) % self.block_size == 0 {
                curr.block_id = Some(block_ids[block_idx]);
            }
        }
    }
}


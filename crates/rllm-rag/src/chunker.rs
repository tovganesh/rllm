use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub chunk_id: usize,
    pub doc_id: String,
    pub text: String,
    pub token_count: usize,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct TextChunker {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl Default for TextChunker {
    fn default() -> Self {
        Self {
            chunk_size: 128,
            chunk_overlap: 32,
        }
    }
}

impl TextChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        assert!(chunk_overlap < chunk_size);
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    /// Chunks a text document into sliding window passages.
    pub fn chunk_text(
        &self,
        doc_id: &str,
        text: &str,
        metadata: std::collections::HashMap<String, String>,
    ) -> Vec<DocumentChunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let step = self.chunk_size - self.chunk_overlap;
        let mut start = 0;
        let mut chunk_id = 0;

        while start < words.len() {
            let end = (start + self.chunk_size).min(words.len());
            let chunk_words = &words[start..end];
            let chunk_str = chunk_words.join(" ");

            chunks.push(DocumentChunk {
                chunk_id,
                doc_id: doc_id.to_string(),
                text: chunk_str,
                token_count: chunk_words.len(),
                metadata: metadata.clone(),
            });

            chunk_id += 1;
            if end == words.len() {
                break;
            }
            start += step;
        }

        chunks
    }
}


//! Document chunking and sliding window passage extraction.
//!
//! Splits long raw texts into overlapping chunks with metadata tracking for passage retrieval.

use serde::{Deserialize, Serialize};

/// An individual chunk/passage extracted from an ingested document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Zero-based sequential chunk index within the parent document.
    pub chunk_id: usize,
    /// Parent document identifier.
    pub doc_id: String,
    /// Extracted text content of the passage.
    pub text: String,
    /// Approximate word/token count in the chunk.
    pub token_count: usize,
    /// Arbitrary key-value metadata associated with the passage.
    pub metadata: std::collections::HashMap<String, String>,
}

/// Sliding window text partitioner with configurable passage length and overlap.
#[derive(Debug, Clone)]
pub struct TextChunker {
    /// Maximum number of whitespace-delimited words per chunk.
    pub chunk_size: usize,
    /// Number of overlapping words shared between consecutive chunks.
    pub chunk_overlap: usize,
}

impl Default for TextChunker {
    /// Default passage size of 128 words with 32-word overlap.
    fn default() -> Self {
        Self {
            chunk_size: 128,
            chunk_overlap: 32,
        }
    }
}

impl TextChunker {
    /// Constructs a new [`TextChunker`].
    ///
    /// # Panics
    ///
    /// Panics if `chunk_overlap >= chunk_size`.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        assert!(chunk_overlap < chunk_size, "chunk_overlap must be less than chunk_size");
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

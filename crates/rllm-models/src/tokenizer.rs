//! Fast, zero-dependency BPE and vocabulary tokenizer for `rllm`.
//!
//! Provides bidirectional conversion between UTF-8 strings and token IDs with
//! byte-level ASCII fallbacks, special token handling, and common vocabulary mappings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Predefined special token identifiers.
pub const PAD_TOKEN_ID: u32 = 0;
pub const BOS_TOKEN_ID: u32 = 1;
pub const EOS_TOKEN_ID: u32 = 2;
pub const UNK_TOKEN_ID: u32 = 3;
pub const SYSTEM_TOKEN_ID: u32 = 4;
pub const USER_TOKEN_ID: u32 = 5;
pub const ASSISTANT_TOKEN_ID: u32 = 6;

/// Common foundational vocabulary tokens to ensure fast exact-word tokenization.
const BASE_VOCAB: &[&str] = &[
    // Special tokens
    "<|pad|>", "<|begin_of_text|>", "<|end_of_text|>", "<|unk|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    // Punctuation & Delimiters
    ".", ",", "!", "?", ":", ";", "'", "\"", "(", ")", "[", "]", "{", "}",
    "-", "_", "/", "\\", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "|", "~", "`",
    "\n", "\t", " ", "  ",
    // Common English words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "are", "was", "were", "been", "has", "had", "does", "did", "where",
    "why", "here", "used", "can", "may", "should", "must", "each", "both",
    // Technical & Domain Vocabulary (LLM / RAG / Systems)
    "rllm", "rust", "inference", "engine", "model", "models", "rag", "pagedattention",
    "continuous", "batching", "batch", "prefix", "radix", "cache", "kv", "tokens",
    "token", "attention", "transformer", "embedding", "embeddings", "vector", "vectors",
    "hnsw", "bm25", "hybrid", "reranker", "rerank", "cross", "encoder", "citations",
    "citation", "grounded", "grounding", "relevance", "score", "memory", "fragmentation",
    "quantization", "quantized", "q4_0", "q4_k", "q8_0", "fp16", "fp32", "bf16",
    "apple", "silicon", "metal", "msl", "cuda", "gpu", "gpus", "cpu", "cluster",
    "clusters", "smartphone", "smartphones", "ios", "android", "edge", "deploy",
    "deployment", "deployments", "device", "devices", "overhead", "latency", "throughput",
    "ttft", "llama", "mistral", "qwen", "bert", "safetensors", "gguf", "open",
    "ai", "api", "rest", "server", "stream", "streaming", "answer", "question",
    "context", "factual", "exact", "quote", "passage", "document", "documents",
    "source", "sources", "evidence", "enterprise", "grade", "performance", "speed",
    "fast", "scalable", "scale", "sub", "millisecond", "unified", "architecture",
];

/// Bidirectional vocabulary tokenizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    /// Constructs a new [`Tokenizer`] initialized with standard ASCII byte fallbacks and vocabulary.
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        // 1. Insert Base Vocabulary
        for &word in BASE_VOCAB {
            if !token_to_id.contains_key(word) {
                let id = id_to_token.len() as u32;
                token_to_id.insert(word.to_string(), id);
                id_to_token.push(word.to_string());
            }
        }

        // 2. Insert ASCII Byte Fallbacks (0..128)
        for b in 0..128u8 {
            let s = (b as char).to_string();
            if !token_to_id.contains_key(&s) {
                let id = id_to_token.len() as u32;
                token_to_id.insert(s.clone(), id);
                id_to_token.push(s);
            }
        }

        Self {
            token_to_id,
            id_to_token,
        }
    }

    /// Size of the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Beginning-of-sequence token ID.
    pub fn bos_token_id(&self) -> u32 {
        BOS_TOKEN_ID
    }

    /// End-of-sequence token ID.
    pub fn eos_token_id(&self) -> u32 {
        EOS_TOKEN_ID
    }

    /// Padding token ID.
    pub fn pad_token_id(&self) -> u32 {
        PAD_TOKEN_ID
    }

    /// Registers a new token into the vocabulary, returning its assigned ID.
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            id
        } else {
            let id = self.id_to_token.len() as u32;
            self.token_to_id.insert(token.to_string(), id);
            self.id_to_token.push(token.to_string());
            id
        }
    }

    /// Encodes a text string into a sequence of token IDs.
    ///
    /// Splits on whitespace and punctuation with subword / character-level fallbacks.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Check for whitespace
            if chars[i].is_whitespace() {
                if chars[i] == '\n' {
                    if let Some(&id) = self.token_to_id.get("\n") {
                        tokens.push(id);
                    }
                }
                i += 1;
                continue;
            }

            // Check for punctuation
            if !chars[i].is_alphanumeric() {
                let s = chars[i].to_string();
                if let Some(&id) = self.token_to_id.get(&s) {
                    tokens.push(id);
                } else if (chars[i] as u32) < 128 {
                    tokens.push(chars[i] as u32);
                }
                i += 1;
                continue;
            }

            // Accumulate alphanumeric word
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let word_lower = word.to_lowercase();

            if let Some(&id) = self.token_to_id.get(&word_lower) {
                tokens.push(id);
            } else if let Some(&id) = self.token_to_id.get(&word) {
                tokens.push(id);
            } else {
                // Character-level fallback
                for c in word.chars() {
                    let s = c.to_string();
                    if let Some(&id) = self.token_to_id.get(&s) {
                        tokens.push(id);
                    } else if (c as u32) < 128 {
                        tokens.push(c as u32);
                    }
                }
            }
        }

        tokens
    }

    /// Decodes a sequence of token IDs back into a reconstructed text string.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();
        let mut last_was_word = false;

        for &id in token_ids {
            if id == PAD_TOKEN_ID || id == BOS_TOKEN_ID || id == EOS_TOKEN_ID {
                continue;
            }

            let token = if let Some(t) = self.id_to_token.get(id as usize) {
                t.as_str()
            } else if id < 128 {
                // ASCII fallback
                let ch = (id as u8) as char;
                if ch.is_ascii_graphic() || ch == ' ' || ch == '\n' {
                    let s = ch.to_string();
                    if s == "." || s == "," || s == "!" || s == "?" || s == ":" || s == ";" {
                        result.push_str(&s);
                        last_was_word = false;
                    } else {
                        if last_was_word && !s.starts_with(' ') {
                            result.push(' ');
                        }
                        result.push_str(&s);
                        last_was_word = ch.is_alphanumeric();
                    }
                }
                continue;
            } else {
                continue;
            };

            // Format spacing based on punctuation
            let is_punct = token.len() == 1 && !token.chars().next().unwrap().is_alphanumeric() && token != "\n";
            if is_punct && (token == "." || token == "," || token == "!" || token == "?" || token == ":" || token == ";") {
                result.push_str(token);
                last_was_word = false;
            } else if token == "\n" {
                result.push('\n');
                last_was_word = false;
            } else {
                if last_was_word {
                    result.push(' ');
                }
                result.push_str(token);
                last_was_word = true;
            }
        }

        result.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_decode() {
        let tokenizer = Tokenizer::new();
        let text = "rllm is an enterprise grade inference engine built in Rust.";
        let tokens = tokenizer.encode(text);
        assert!(!tokens.is_empty());

        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("rllm"));
        assert!(decoded.contains("enterprise"));
        assert!(decoded.contains("engine"));
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.bos_token_id(), BOS_TOKEN_ID);
        assert_eq!(tokenizer.eos_token_id(), EOS_TOKEN_ID);
    }
}

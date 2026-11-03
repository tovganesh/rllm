//! OpenAI-compatible and RAG HTTP API payload schemas and serializable types.
//!
//! Provides request and response types for `/v1/chat/completions`, `/v1/embeddings`,
//! `/v1/rag/index`, and `/v1/rag/query`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An individual message in a chat completion prompt or response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender: `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

/// Request body for OpenAI-compatible `/v1/chat/completions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier to invoke.
    pub model: Option<String>,
    /// List of dialogue messages comprising the conversation context.
    pub messages: Vec<ChatMessage>,
    /// Sampling temperature ($0.0 \le T \le 2.0$).
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff ($0.0 < p \le 1.0$).
    pub top_p: Option<f32>,
    /// Maximum number of output tokens to generate.
    pub max_tokens: Option<usize>,
    /// Whether to stream server-sent events (SSE).
    pub stream: Option<bool>,
}

/// A single choice/candidate in a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index within the completion.
    pub index: usize,
    /// Generated assistant message.
    pub message: ChatMessage,
    /// Completion termination reason (`"stop"`, `"length"`).
    pub finish_reason: String,
}

/// Response payload returned by `/v1/chat/completions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique completion identifier.
    pub id: String,
    /// Object type descriptor (`"chat.completion"`).
    pub object: String,
    /// Unix epoch timestamp in seconds.
    pub created: u64,
    /// Model name used for completion.
    pub model: String,
    /// Generated response choices.
    pub choices: Vec<ChatChoice>,
    /// Token usage accounting statistics.
    pub usage: Usage,
}

/// Token usage accounting for a completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens processed in the input prompt.
    pub prompt_tokens: usize,
    /// Number of tokens generated in the completion.
    pub completion_tokens: usize,
    /// Total tokens (prompt + completion).
    pub total_tokens: usize,
}

/// Request body for OpenAI-compatible `/v1/embeddings`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Model identifier.
    pub model: Option<String>,
    /// List of input texts to encode.
    pub input: Vec<String>,
}

/// Single vector embedding item in an embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// Zero-based index corresponding to the input list.
    pub index: usize,
    /// Dense floating-point vector embedding.
    pub embedding: Vec<f32>,
}

/// Response body returned by `/v1/embeddings`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Object type (`"list"`).
    pub object: String,
    /// List of generated embeddings.
    pub data: Vec<EmbeddingData>,
    /// Embedding model name.
    pub model: String,
}

/// Request payload to ingest and index a document into the RAG engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagIndexRequest {
    /// Unique document ID.
    pub doc_id: String,
    /// Raw text content of the document.
    pub text: String,
    /// Optional metadata key-value tags.
    pub metadata: Option<HashMap<String, String>>,
}

/// Confirmation response after document indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagIndexResponse {
    /// Status message (e.g., `"ok"`).
    pub status: String,
    /// Ingested document ID.
    pub doc_id: String,
    /// Number of text chunks partitioned and added to the hybrid index.
    pub chunks_indexed: usize,
}

/// Request payload for semantic RAG query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQueryRequest {
    /// User query string.
    pub query: String,
    /// Top-$k$ context passages to retrieve and cite.
    pub top_k: Option<usize>,
    /// Maximum output tokens to generate in the final synthesis.
    pub max_tokens: Option<usize>,
    /// Optional minimum relevance threshold in $[0.0, 1.0]$. If citations fail to meet this,
    /// the response flags `is_relevant = false` and skips generation.
    pub min_relevance: Option<f32>,
}

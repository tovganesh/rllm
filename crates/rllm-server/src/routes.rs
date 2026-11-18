//! Route handlers for OpenAI-compatible REST and RAG endpoints.

use crate::api_types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, EmbeddingData,
    EmbeddingRequest, EmbeddingResponse, RagIndexRequest, RagIndexResponse, RagQueryRequest,
    Usage,
};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use rllm_backend_cpu::CpuBackend;
use rllm_models::{BpeTokenizer, ChatTemplate, TextEmbedder};
use rllm_rag::{grounded_messages, unrelated_response, verify_citations, RagEngine, RagResponse};
use rllm_scheduler::{ContinuousBatchScheduler, SamplingParams, Sequence};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

/// Operating service mode for containerized or standalone deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceMode {
    /// Host both inference and RAG services in a single unified process.
    All,
    /// Host only the LLM inference engine (continuous batching, PagedAttention, completions).
    Inference,
    /// Host only the RAG service (vector store, chunking, hybrid search, reranker).
    Rag,
}

/// Global shared application state across HTTP request handlers.
pub struct AppState {
    /// Optional local scheduler for in-process LLM generation.
    pub scheduler: Option<Arc<ContinuousBatchScheduler<CpuBackend>>>,
    /// Tokenizer of the loaded model. `None` for synthetic models, which fall back to
    /// placeholder token IDs.
    pub tokenizer: Option<Arc<BpeTokenizer>>,
    /// Chat template of the loaded model (used together with `tokenizer`).
    pub chat_template: Option<Arc<ChatTemplate>>,
    /// Embedding model serving `/v1/embeddings` (shared with the RAG engine).
    pub embedder: Option<Arc<TextEmbedder<CpuBackend>>>,
    /// Optional local RAG engine for hybrid retrieval and document indexing.
    pub rag_engine: Option<Arc<RagEngine<CpuBackend>>>,
    /// Optional remote inference URL (e.g., `http://rllm-inference:8000`) for decoupled RAG.
    pub inference_url: Option<String>,
    /// Model name identifier reported in API responses.
    pub model_name: String,
    /// Active service operating mode.
    pub mode: ServiceMode,
}

/// Health and cluster status handler (`GET /health`).
pub async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let running = state.scheduler.as_ref().map(|s| s.num_running()).unwrap_or(0);
    let waiting = state.scheduler.as_ref().map(|s| s.num_waiting()).unwrap_or(0);

    Json(serde_json::json!({
        "status": "healthy",
        "engine": "rllm",
        "service_mode": match state.mode {
            ServiceMode::All => "all",
            ServiceMode::Inference => "inference",
            ServiceMode::Rag => "rag",
        },
        "model": state.model_name,
        "running_sequences": running,
        "waiting_sequences": waiting,
        "remote_inference_url": state.inference_url,
    }))
}

/// Error returned by API handlers, serialised in OpenAI's error format.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let kind = if self.status.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        };
        let body = Json(serde_json::json!({
            "error": { "message": self.message, "type": kind }
        }));
        (self.status, body).into_response()
    }
}

/// OpenAI-compatible model listing (`GET /v1/models`).
pub async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{ "id": state.model_name, "object": "model", "owned_by": "rllm" }]
    }))
}

/// OpenAI-compatible chat completion handler (`POST /v1/chat/completions`).
///
/// With a real model loaded, the conversation is rendered with the model's chat
/// template and tokenized with its tokenizer, and generation stops at the model's
/// end-of-turn tokens. Streaming (`stream: true`) is not supported yet.
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let scheduler = state.scheduler.clone().ok_or_else(|| {
        error!("Chat completions requested but inference scheduler is not enabled in this container");
        ApiError::new(StatusCode::NOT_IMPLEMENTED, "inference is not enabled on this server")
    })?;
    if req.stream == Some(true) {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "stream=true is not supported yet"));
    }
    if req.messages.is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "messages must not be empty"));
    }

    let mut sampling = SamplingParams::default();
    if let Some(t) = req.temperature {
        sampling.temperature = t;
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p;
    }
    if let Some(max_t) = req.max_tokens {
        if max_t == 0 {
            return Err(ApiError::new(StatusCode::BAD_REQUEST, "max_tokens must be >= 1"));
        }
        sampling.max_tokens = max_t;
    }

    let prompt_tokens: Vec<u32> = match (&state.tokenizer, &state.chat_template) {
        (Some(tokenizer), Some(template)) => {
            let messages: Vec<rllm_models::ChatMessage> = req
                .messages
                .iter()
                .map(|m| rllm_models::ChatMessage::new(&m.role, &m.content))
                .collect();
            let prompt = template
                .render(&messages, true)
                .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e.to_string()))?;
            sampling.stop_tokens = tokenizer.stop_token_ids.clone();
            tokenizer.encode(&prompt, true, true)
        }
        // Synthetic model without a tokenizer: placeholder token IDs.
        _ => {
            let mut prompt = String::new();
            for msg in &req.messages {
                prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
            prompt.push_str("assistant: ");
            prompt
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| ((i % 500) + 1) as u32)
                .collect()
        }
    };

    let seq_id = rand::random::<u64>();
    let prompt_len = prompt_tokens.len();
    let stop_tokens = sampling.stop_tokens.clone();
    let seq = Sequence::new(seq_id, prompt_tokens, sampling, scheduler.config.block_size, None);

    let mut generated = tokio::task::spawn_blocking(move || scheduler.generate(seq, |_| {}))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| match e {
            // Prompt validation failures (too long, bad token IDs) are client errors.
            rllm_core::error::RllmError::BackendError(msg) if msg.contains("Prompt") || msg.contains("Token ID") => {
                ApiError::new(StatusCode::BAD_REQUEST, msg)
            }
            other => ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
        })?;

    let finish_reason = if generated.last().is_some_and(|t| stop_tokens.contains(t)) {
        generated.pop();
        "stop"
    } else {
        // max_tokens reached, or the context window / KV capacity ran out.
        "length"
    };
    let completion_tokens = generated.len() + usize::from(finish_reason == "stop");

    let content = match &state.tokenizer {
        Some(tokenizer) => tokenizer.decode(&generated, true),
        None => rllm_models::Tokenizer::default().decode(&generated),
    };

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", seq_id),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    }))
}

/// OpenAI-compatible dense vector embeddings handler (`POST /v1/embeddings`).
pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let embedder = state.embedder.clone().ok_or_else(|| {
        ApiError::new(StatusCode::NOT_IMPLEMENTED, "no embedding model is loaded on this server")
    })?;
    if req.input.is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "input must not be empty"));
    }
    let model = embedder.name.clone();
    let inputs = req.input;
    let data = tokio::task::spawn_blocking(move || {
        inputs
            .iter()
            .enumerate()
            .map(|(index, text)| {
                embedder
                    .embed_text(text)
                    .map(|embedding| EmbeddingData { index, embedding })
            })
            .collect::<rllm_core::error::Result<Vec<_>>>()
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e.to_string()))?;

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model,
    }))
}

/// Document ingestion handler for the RAG hybrid vector store (`POST /v1/rag/index`).
pub async fn rag_index_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagIndexRequest>,
) -> Result<Json<RagIndexResponse>, ApiError> {
    let rag_engine = state.rag_engine.clone().ok_or_else(|| {
        ApiError::new(StatusCode::NOT_IMPLEMENTED, "RAG is not enabled on this server")
    })?;
    if req.text.trim().is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "text must not be empty"));
    }
    let meta = req.metadata.unwrap_or_default();
    let doc_id = req.doc_id.clone();
    let text = req.text;
    let count = tokio::task::spawn_blocking(move || rag_engine.ingest_document(&doc_id, &text, meta))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(RagIndexResponse {
        status: "success".to_string(),
        doc_id: req.doc_id,
        chunks_indexed: count,
    }))
}

/// End-to-end RAG query handler (`POST /v1/rag/query`).
///
/// Retrieval always runs locally. The answer is generated by the remote inference
/// service when `inference_url` is configured, otherwise by the local engine (its LLM
/// generator if one is loaded, or an extractive answer).
pub async fn rag_query_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagQueryRequest>,
) -> Result<Json<RagResponse>, ApiError> {
    let rag_engine = state.rag_engine.clone().ok_or_else(|| {
        ApiError::new(StatusCode::NOT_IMPLEMENTED, "RAG is not enabled on this server")
    })?;
    if req.query.trim().is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "query must not be empty"));
    }
    let top_k = req.top_k.unwrap_or(3).clamp(1, 20);
    let mut sampling = SamplingParams::default();
    if let Some(m) = req.max_tokens {
        sampling.max_tokens = m.max(1);
    }

    let Some(remote_url) = state.inference_url.clone() else {
        let response = rag_engine
            .query_with_threshold(&req.query, top_k, sampling, req.min_relevance)
            .await
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        return Ok(Json(response));
    };

    // Remote generation: retrieve locally, generate on the inference service.
    info!("Executing RAG query with remote inference service at {}", remote_url);
    let local_query = req.query.clone();
    let threshold = req.min_relevance;
    let engine = rag_engine.clone();
    let retrieval = tokio::task::spawn_blocking(move || engine.retrieve(&local_query, top_k, threshold))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if !retrieval.is_relevant {
        return Ok(Json(unrelated_response(
            &req.query,
            &retrieval,
            req.min_relevance.unwrap_or(rag_engine.min_relevance_threshold),
            rag_engine.min_dense_similarity,
        )));
    }

    let mut citations = retrieval.citations;
    let messages = grounded_messages(&req.query, &citations)
        .into_iter()
        .map(|m| ChatMessage { role: m.role, content: m.content })
        .collect();
    let chat_req = ChatCompletionRequest {
        model: None,
        messages,
        temperature: Some(sampling.temperature),
        top_p: Some(sampling.top_p),
        max_tokens: Some(sampling.max_tokens),
        stream: Some(false),
    };
    let endpoint = format!("{}/v1/chat/completions", remote_url.trim_end_matches('/'));
    let resp = reqwest::Client::new()
        .post(&endpoint)
        .json(&chat_req)
        .send()
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, format!("inference service unreachable: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ApiError::new(StatusCode::BAD_GATEWAY, format!("inference service returned {status}: {body}")));
    }
    let chat: ChatCompletionResponse = resp
        .json()
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, format!("invalid inference response: {e}")))?;
    let answer = chat
        .choices
        .first()
        .map(|c| c.message.content.trim().to_string())
        .unwrap_or_default();
    let warning = verify_citations(&answer, &mut citations);

    Ok(Json(RagResponse {
        query: req.query,
        answer,
        answer_source: "llm".to_string(),
        citations,
        generated_tokens: chat.usage.completion_tokens,
        is_relevant: true,
        confidence_score: retrieval.confidence_score,
        max_dense_similarity: retrieval.max_dense_similarity,
        warning,
    }))
}

//! Route handlers for OpenAI-compatible REST and RAG endpoints.

use crate::api_types::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, ChatMessage, EmbeddingData, EmbeddingRequest,
    EmbeddingResponse, RagIndexRequest, RagIndexResponse, RagQueryRequest, Usage,
};
use crate::metrics::ServerMetrics;
use axum::{
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use futures_util::stream::{self, Stream};
use std::convert::Infallible;
use std::time::Instant;
use rllm_backend_cpu::CpuBackend;
use rllm_models::{BpeTokenizer, ChatTemplate, StreamDecoder, TextEmbedder};
use rllm_rag::{grounded_messages, unrelated_response, verify_citations, RagEngine, RagResponse};
use rllm_scheduler::{EngineHandle, FinishReason, GenerationStream, SamplingParams, SequenceEvent, SubmitError};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

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

/// Request limits enforced by the server.
#[derive(Debug, Clone)]
pub struct ServerLimits {
    /// Largest accepted `max_tokens`.
    pub max_tokens: usize,
    /// `max_tokens` used when a request does not set it.
    pub default_max_tokens: usize,
    /// Largest accepted request body in bytes.
    pub max_body_bytes: usize,
    /// Largest number of inputs in one `/v1/embeddings` request.
    pub max_embedding_inputs: usize,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            default_max_tokens: 256,
            max_body_bytes: 2 * 1024 * 1024,
            max_embedding_inputs: 256,
        }
    }
}

/// Global shared application state across HTTP request handlers.
pub struct AppState {
    /// Engine running the local LLM (`None` when inference is not served here).
    pub engine: Option<EngineHandle>,
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
    /// Request limits.
    pub limits: ServerLimits,
    /// HTTP-level metrics.
    pub metrics: Arc<ServerMetrics>,
}

/// Health and cluster status handler (`GET /health`).
pub async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let running = state.engine.as_ref().map(|e| e.scheduler().num_running()).unwrap_or(0);
    let waiting = state.engine.as_ref().map(|e| e.scheduler().num_waiting()).unwrap_or(0);

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

    fn from_submit(e: SubmitError, metrics: &ServerMetrics) -> Self {
        match e {
            SubmitError::Invalid(msg) => Self::new(StatusCode::BAD_REQUEST, msg),
            SubmitError::Overloaded { .. } => {
                metrics.rejected_overloaded.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Self::new(StatusCode::TOO_MANY_REQUESTS, e.to_string())
            }
            SubmitError::ShutDown => Self::new(StatusCode::SERVICE_UNAVAILABLE, e.to_string()),
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
        let mut resp = (self.status, body).into_response();
        if self.status == StatusCode::TOO_MANY_REQUESTS {
            resp.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        resp
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
/// end-of-turn tokens. `stream: true` returns server-sent `chat.completion.chunk`
/// events terminated by `data: [DONE]`. Disconnecting cancels generation.
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let start = Instant::now();
    let engine = state.engine.clone().ok_or_else(|| {
        ApiError::new(StatusCode::NOT_IMPLEMENTED, "inference is not enabled on this server")
    })?;
    if req.messages.is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "messages must not be empty"));
    }

    let mut sampling = SamplingParams {
        max_tokens: state.limits.default_max_tokens,
        ..SamplingParams::default()
    };
    if let Some(t) = req.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(ApiError::new(StatusCode::BAD_REQUEST, "temperature must be between 0 and 2"));
        }
        sampling.temperature = t;
    }
    if let Some(p) = req.top_p {
        if !(p > 0.0 && p <= 1.0) {
            return Err(ApiError::new(StatusCode::BAD_REQUEST, "top_p must be in (0, 1]"));
        }
        sampling.top_p = p;
    }
    if let Some(max_t) = req.max_tokens {
        if max_t == 0 || max_t > state.limits.max_tokens {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("max_tokens must be between 1 and {}", state.limits.max_tokens),
            ));
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

    let stop_tokens = sampling.stop_tokens.clone();
    let generation = engine
        .submit(prompt_tokens, sampling)
        .map_err(|e| ApiError::from_submit(e, &state.metrics))?;
    let ctx = CompletionContext {
        id: format!("chatcmpl-{}", generation.id),
        created: unix_now(),
        model: state.model_name.clone(),
        tokenizer: state.tokenizer.clone(),
        stop_tokens,
        metrics: state.metrics.clone(),
        start,
    };

    if req.stream == Some(true) {
        let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
        let sse = Sse::new(chat_stream(generation, ctx, include_usage)).keep_alive(KeepAlive::default());
        return Ok(sse.into_response());
    }

    let (content, finish_reason, completion_tokens, prompt_len) = collect_completion(generation, &ctx).await?;
    Ok(Json(ChatCompletionResponse {
        id: ctx.id,
        object: "chat.completion".to_string(),
        created: ctx.created,
        model: ctx.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    })
    .into_response())
}

/// Per-request data shared by the streaming and non-streaming paths.
struct CompletionContext {
    id: String,
    created: u64,
    model: String,
    tokenizer: Option<Arc<BpeTokenizer>>,
    stop_tokens: Vec<u32>,
    metrics: Arc<ServerMetrics>,
    start: Instant,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn finish_reason_str(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Stop => "stop",
        FinishReason::Length | FinishReason::Abort => "length",
    }
}

/// Text of one generated token for models without a BPE tokenizer.
fn legacy_token_text(tok: u32) -> String {
    format!("{} ", rllm_models::Tokenizer::default().decode(&[tok]))
}

/// Runs a request to completion: `(content, finish_reason, completion_tokens, prompt_tokens)`.
async fn collect_completion(
    mut generation: GenerationStream,
    ctx: &CompletionContext,
) -> Result<(String, String, usize, usize), ApiError> {
    let mut tokens = Vec::new();
    let reason = loop {
        match generation.next().await {
            Some(SequenceEvent::Token(t)) => {
                if tokens.is_empty() {
                    ctx.metrics.observe_ttft(ctx.start);
                }
                tokens.push(t);
            }
            Some(SequenceEvent::Finished(r)) => break r,
            Some(SequenceEvent::Error(e)) => return Err(ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e)),
            None => return Err(ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly")),
        }
    };
    let completion_tokens = tokens.len();
    if reason == FinishReason::Stop && tokens.last().is_some_and(|t| ctx.stop_tokens.contains(t)) {
        tokens.pop();
    }
    let content = match &ctx.tokenizer {
        Some(t) => t.decode(&tokens, true),
        None => tokens.iter().map(|&t| legacy_token_text(t)).collect::<String>().trim().to_string(),
    };
    Ok((content, finish_reason_str(reason).to_string(), completion_tokens, generation.prompt_tokens))
}

enum StreamPhase {
    Start,
    Generating,
    Usage(Usage),
    Done,
    Closed,
}

struct ChatStreamState {
    generation: GenerationStream,
    ctx: CompletionContext,
    decoder: StreamDecoder,
    include_usage: bool,
    completion_tokens: usize,
    phase: StreamPhase,
}

impl ChatStreamState {
    fn chunk(&self, delta: ChatDelta, finish_reason: Option<&str>) -> Event {
        let chunk = ChatCompletionChunk {
            id: self.ctx.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.ctx.created,
            model: self.ctx.model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta,
                finish_reason: finish_reason.map(str::to_string),
            }],
            usage: None,
        };
        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
    }
}

/// SSE event stream for a streamed chat completion.
fn chat_stream(
    generation: GenerationStream,
    ctx: CompletionContext,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let state = ChatStreamState {
        generation,
        ctx,
        decoder: StreamDecoder::default(),
        include_usage,
        completion_tokens: 0,
        phase: StreamPhase::Start,
    };
    stream::unfold(state, |mut st| async move {
        loop {
            match std::mem::replace(&mut st.phase, StreamPhase::Closed) {
                StreamPhase::Start => {
                    st.phase = StreamPhase::Generating;
                    let ev = st.chunk(ChatDelta { role: Some("assistant".to_string()), content: None }, None);
                    return Some((Ok(ev), st));
                }
                StreamPhase::Generating => match st.generation.next().await {
                    Some(SequenceEvent::Token(t)) => {
                        st.phase = StreamPhase::Generating;
                        if st.completion_tokens == 0 {
                            st.ctx.metrics.observe_ttft(st.ctx.start);
                        }
                        st.completion_tokens += 1;
                        if st.ctx.stop_tokens.contains(&t) {
                            continue;
                        }
                        let text = match &st.ctx.tokenizer {
                            Some(tok) => st.decoder.push(tok, t, true),
                            None => legacy_token_text(t),
                        };
                        if text.is_empty() {
                            continue;
                        }
                        let ev = st.chunk(ChatDelta { role: None, content: Some(text) }, None);
                        return Some((Ok(ev), st));
                    }
                    Some(SequenceEvent::Finished(reason)) => {
                        let tail = st.decoder.finish();
                        let usage = Usage {
                            prompt_tokens: st.generation.prompt_tokens,
                            completion_tokens: st.completion_tokens,
                            total_tokens: st.generation.prompt_tokens + st.completion_tokens,
                        };
                        st.phase = if st.include_usage { StreamPhase::Usage(usage) } else { StreamPhase::Done };
                        let delta = ChatDelta { role: None, content: (!tail.is_empty()).then_some(tail) };
                        let ev = st.chunk(delta, Some(finish_reason_str(reason)));
                        return Some((Ok(ev), st));
                    }
                    Some(SequenceEvent::Error(e)) => {
                        st.phase = StreamPhase::Done;
                        let body = serde_json::json!({ "error": { "message": e, "type": "server_error" } });
                        return Some((Ok(Event::default().data(body.to_string())), st));
                    }
                    None => {
                        st.phase = StreamPhase::Done;
                        continue;
                    }
                },
                StreamPhase::Usage(usage) => {
                    st.phase = StreamPhase::Done;
                    let chunk = ChatCompletionChunk {
                        id: st.ctx.id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: st.ctx.created,
                        model: st.ctx.model.clone(),
                        choices: Vec::new(),
                        usage: Some(usage),
                    };
                    let ev = Event::default().data(serde_json::to_string(&chunk).unwrap_or_default());
                    return Some((Ok(ev), st));
                }
                StreamPhase::Done => {
                    st.phase = StreamPhase::Closed;
                    return Some((Ok(Event::default().data("[DONE]")), st));
                }
                StreamPhase::Closed => return None,
            }
        }
    })
}

/// OpenAI-compatible dense vector embeddings handler (`POST /v1/embeddings`).
pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let embedder = state.embedder.clone().ok_or_else(|| {
        ApiError::new(StatusCode::NOT_IMPLEMENTED, "no embedding model is loaded on this server")
    })?;
    if req.input.is_empty() || req.input.len() > state.limits.max_embedding_inputs {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("input must contain between 1 and {} texts", state.limits.max_embedding_inputs),
        ));
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
        stream_options: None,
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

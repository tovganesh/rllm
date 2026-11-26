//! `rllm-server`: High-throughput, asynchronous OpenAI-compatible HTTP and SSE streaming server.

pub mod api_types;
pub mod metrics;
pub mod routes;

use axum::{
    extract::DefaultBodyLimit,
    middleware,
    routing::{get, post},
    Router,
};
pub use metrics::ServerMetrics;
pub use routes::{AppState, ServerLimits, ServiceMode};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Construct the Axum HTTP router with active routes based on the configured `ServiceMode`.
pub fn create_router(state: Arc<AppState>) -> Router {
    let mut router = Router::new()
        .route("/health", get(routes::health_handler))
        .route("/metrics", get(metrics::metrics_handler))
        .route("/v1/embeddings", post(routes::embeddings_handler));

    if state.mode == ServiceMode::All || state.mode == ServiceMode::Inference {
        router = router
            .route("/v1/chat/completions", post(routes::chat_completions_handler))
            .route("/v1/models", get(routes::models_handler));
    }
    if state.mode == ServiceMode::All || state.mode == ServiceMode::Rag {
        router = router
            .route("/v1/rag/index", post(routes::rag_index_handler))
            .route("/v1/rag/query", post(routes::rag_query_handler));
    }

    router
        .layer(middleware::from_fn_with_state(state.clone(), metrics::track_requests))
        .layer(DefaultBodyLimit::max(state.limits.max_body_bytes))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Binds to `addr` and serves until Ctrl-C / SIGTERM, then stops accepting
/// connections, lets in-flight requests finish and shuts the engine down.
pub async fn run_server(addr: SocketAddr, state: Arc<AppState>) -> Result<(), std::io::Error> {
    let mode_str = match state.mode {
        ServiceMode::All => "Unified (Inference + RAG)",
        ServiceMode::Inference => "Dedicated LLM Inference Engine",
        ServiceMode::Rag => "Dedicated RAG Engine",
    };
    info!("Starting rllm server [{}] on {}", mode_str, addr);
    let app = create_router(state.clone());
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    if let Some(engine) = &state.engine {
        let engine = engine.clone();
        let _ = tokio::task::spawn_blocking(move || engine.shutdown()).await;
    }
    info!("Server stopped");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Shutdown signal received");
}

#[cfg(test)]
mod tests {
    use super::*;
    use api_types::{RagIndexRequest, RagQueryRequest};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use rllm_backend_cpu::CpuBackend;
    use rllm_models::{BertConfig, LlamaConfig, LlamaModel, TextEmbedder};
    use rllm_rag::RagEngine;
    use rllm_scheduler::{ContinuousBatchScheduler, EngineConfig, EngineHandle, SchedulerConfig};
    use tower::ServiceExt;

    fn setup_test_state(mode: ServiceMode) -> Arc<AppState> {
        setup_state_with(mode, 256)
    }

    fn setup_state_with(mode: ServiceMode, max_waiting: usize) -> Arc<AppState> {
        let backend = Arc::new(CpuBackend);
        let embedder = Arc::new(TextEmbedder::synthetic(BertConfig::tiny_test_config(), backend.clone()).unwrap());

        let llama_config = LlamaConfig::tiny_test_config();
        let llama_model = Arc::new(LlamaModel::new_random(llama_config, backend).unwrap());

        let engine = EngineHandle::start(
            Arc::new(
                ContinuousBatchScheduler::new(
                    SchedulerConfig {
                        max_batch_size: 4,
                        block_size: 4,
                        total_blocks: 512,
                        ..SchedulerConfig::default()
                    },
                    llama_model,
                )
                .unwrap(),
            ),
            EngineConfig { max_waiting },
        );

        let rag_engine = Arc::new(RagEngine::new(embedder.clone(), None));

        Arc::new(AppState {
            engine: Some(engine),
            tokenizer: None,
            chat_template: None,
            embedder: Some(embedder),
            rag_engine: Some(rag_engine),
            inference_url: None,
            model_name: "rllm-tiny-test".to_string(),
            mode,
            limits: ServerLimits::default(),
            metrics: Arc::new(ServerMetrics::default()),
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = setup_test_state(ServiceMode::All);
        let app = create_router(state);

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_rag_index_and_query_endpoints() {
        let state = setup_test_state(ServiceMode::Rag);
        let app = create_router(state);

        // 1. Index document
        let index_req = RagIndexRequest {
            doc_id: "doc_edge".to_string(),
            text: "rllm delivers sub-millisecond local inference on smartphones and macs".to_string(),
            metadata: None,
        };

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/rag/index")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&index_req).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // 2. Query document
        let query_req = RagQueryRequest {
            query: "where does rllm deliver sub-millisecond inference?".to_string(),
            top_k: Some(1),
            max_tokens: Some(2),
            min_relevance: None,
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/rag/query")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&query_req).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let rag: rllm_rag::RagResponse = serde_json::from_slice(&body).unwrap();
        assert!(rag.is_relevant);
        assert_eq!(rag.answer_source, "extractive");
        assert_eq!(rag.citations[0].doc_id, "doc_edge");
    }

    #[tokio::test]
    async fn test_embeddings_endpoint_and_error_bodies() {
        let app = create_router(setup_test_state(ServiceMode::All));
        let post = |uri: &str, body: &str| {
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let resp = app
            .clone()
            .oneshot(post("/v1/embeddings", r#"{"input":["hello world","second"]}"#))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
        assert_eq!(body["data"].as_array().unwrap().len(), 2);
        assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 64);

        let resp = app
            .clone()
            .oneshot(post("/v1/rag/query", r#"{"query":"   "}"#))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value =
            serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    fn post_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn body_string(resp: axum::response::Response) -> String {
        String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_streaming_matches_non_streaming_and_ends_with_done() {
        let app = create_router(setup_test_state(ServiceMode::Inference));
        let msgs = serde_json::json!([{"role": "user", "content": "hello there general kenobi"}]);

        let resp = app
            .clone()
            .oneshot(post_json(
                "/v1/chat/completions",
                serde_json::json!({"messages": msgs, "temperature": 0, "max_tokens": 6}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let full: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        let full_text = full["choices"][0]["message"]["content"].as_str().unwrap().to_string();
        assert_eq!(full["choices"][0]["finish_reason"], "length");
        assert_eq!(full["usage"]["completion_tokens"], 6);

        let resp = app
            .oneshot(post_json(
                "/v1/chat/completions",
                serde_json::json!({"messages": msgs, "temperature": 0, "max_tokens": 6, "stream": true, "stream_options": {"include_usage": true}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let body = body_string(resp).await;
        let events: Vec<&str> = body.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
        assert_eq!(*events.last().unwrap(), "[DONE]");
        let chunks: Vec<serde_json::Value> = events[..events.len() - 1]
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let streamed: String = chunks
            .iter()
            .filter_map(|c| c["choices"].get(0).and_then(|ch| ch["delta"]["content"].as_str()))
            .collect();
        assert_eq!(
            streamed.trim(),
            full_text,
            "streamed text must equal the non-streamed text"
        );
        let finish = chunks
            .iter()
            .find_map(|c| c["choices"].get(0).and_then(|ch| ch["finish_reason"].as_str()))
            .unwrap();
        assert_eq!(finish, "length");
        let usage = &chunks.last().unwrap()["usage"];
        assert_eq!(usage["completion_tokens"], 6);
        assert!(chunks.iter().all(|c| c["id"] == chunks[0]["id"]));
    }

    #[tokio::test]
    async fn test_limits_and_overload() {
        let app = create_router(setup_test_state(ServiceMode::Inference));
        let msgs = serde_json::json!([{"role": "user", "content": "hi"}]);
        for body in [
            serde_json::json!({"messages": msgs, "max_tokens": 0}),
            serde_json::json!({"messages": msgs, "max_tokens": 100000}),
            serde_json::json!({"messages": msgs, "temperature": 5.0}),
            serde_json::json!({"messages": []}),
        ] {
            let resp = app
                .clone()
                .oneshot(post_json("/v1/chat/completions", body.clone()))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{body}");
        }

        let overloaded = create_router(setup_state_with(ServiceMode::Inference, 0));
        let resp = overloaded
            .oneshot(post_json("/v1/chat/completions", serde_json::json!({"messages": msgs})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(resp.headers()["retry-after"], "1");
    }

    #[tokio::test]
    async fn test_metrics_endpoint_reports_activity() {
        let state = setup_test_state(ServiceMode::All);
        let app = create_router(state);
        let resp = app
            .clone()
            .oneshot(post_json(
                "/v1/chat/completions",
                serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = body_string(resp).await;

        let resp = app
            .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let text = body_string(resp).await;
        assert!(
            text.contains("rllm_http_requests_total{route=\"/v1/chat/completions\",status=\"200\"} 1"),
            "{text}"
        );
        assert!(text.contains("rllm_generated_tokens_total 3"), "{text}");
        assert!(
            text.contains("rllm_sequences_finished_total{reason=\"length\"} 1"),
            "{text}"
        );
        assert!(text.contains("rllm_time_to_first_token_seconds_count 1"), "{text}");
        assert!(text.contains("rllm_kv_blocks_free"));
        assert!(text.contains("rllm_draft_tokens_total 0"), "{text}");
        assert!(text.contains("rllm_draft_tokens_accepted_total 0"), "{text}");
    }
}

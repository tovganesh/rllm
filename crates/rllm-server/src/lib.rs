//! `rllm-server`: High-throughput, asynchronous OpenAI-compatible HTTP and SSE streaming server.

pub mod api_types;
pub mod routes;

use axum::{
    routing::{get, post},
    Router,
};
pub use routes::{AppState, ServiceMode};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Construct the Axum HTTP router with active routes based on the configured `ServiceMode`.
pub fn create_router(state: Arc<AppState>) -> Router {
    let mut router = Router::new().route("/health", get(routes::health_handler));

    // Mount LLM Inference endpoints if mode is All or Inference
    if state.mode == ServiceMode::All || state.mode == ServiceMode::Inference {
        router = router
            .route(
                "/v1/chat/completions",
                post(routes::chat_completions_handler),
            )
            .route("/v1/embeddings", post(routes::embeddings_handler));
    }

    // Mount RAG endpoints if mode is All or Rag
    if state.mode == ServiceMode::All || state.mode == ServiceMode::Rag {
        router = router
            .route("/v1/rag/index", post(routes::rag_index_handler))
            .route("/v1/rag/query", post(routes::rag_query_handler));
    }

    router
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Binds to `addr` and runs the HTTP server.
pub async fn run_server(addr: SocketAddr, state: Arc<AppState>) -> Result<(), std::io::Error> {
    let mode_str = match state.mode {
        ServiceMode::All => "Unified (Inference + RAG)",
        ServiceMode::Inference => "Dedicated LLM Inference Engine",
        ServiceMode::Rag => "Dedicated High-Performance RAG Engine",
    };
    info!("Starting rllm server [{}] on {}", mode_str, addr);
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use api_types::{RagIndexRequest, RagQueryRequest};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use rllm_backend_cpu::CpuBackend;
    use rllm_models::{BertConfig, BertEmbeddingModel, LlamaConfig, LlamaModel};
    use rllm_rag::RagEngine;
    use rllm_scheduler::{ContinuousBatchScheduler, SchedulerConfig};
    use tower::ServiceExt;

    fn setup_test_state(mode: ServiceMode) -> Arc<AppState> {
        let backend = Arc::new(CpuBackend);
        let bert_config = BertConfig::tiny_test_config();
        let bert_model =
            Arc::new(BertEmbeddingModel::new_random(bert_config.clone(), backend.clone()).unwrap());

        let llama_config = LlamaConfig::tiny_test_config();
        let llama_model = Arc::new(LlamaModel::new_random(llama_config, backend).unwrap());

        let scheduler = Arc::new(
            ContinuousBatchScheduler::new(
                SchedulerConfig {
                    max_batch_size: 4,
                    block_size: 4,
                    total_blocks: 32,
                },
                llama_model,
            )
            .unwrap(),
        );

        let rag_engine = Arc::new(RagEngine::new(
            bert_model,
            scheduler.clone(),
            bert_config.hidden_size,
        ));

        Arc::new(AppState {
            scheduler: Some(scheduler),
            rag_engine: Some(rag_engine),
            inference_url: None,
            model_name: "rllm-tiny-test".to_string(),
            mode,
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = setup_test_state(ServiceMode::All);
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
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
    }
}

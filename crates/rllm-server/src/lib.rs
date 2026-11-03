pub mod api_types;
pub mod routes;

use axum::{
    routing::{get, post},
    Router,
};
pub use routes::AppState;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health_handler))
        .route(
            "/v1/chat/completions",
            post(routes::chat_completions_handler),
        )
        .route("/v1/embeddings", post(routes::embeddings_handler))
        .route("/v1/rag/index", post(routes::rag_index_handler))
        .route("/v1/rag/query", post(routes::rag_query_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(addr: SocketAddr, state: Arc<AppState>) -> Result<(), std::io::Error> {
    let app = create_router(state);
    info!("Starting rllm OpenAI-compatible API server on {}", addr);
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

    fn setup_test_state() -> Arc<AppState> {
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
            scheduler,
            rag_engine,
            model_name: "rllm-tiny-test".to_string(),
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = setup_test_state();
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
        let state = setup_test_state();
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

//! Prometheus metrics (`GET /metrics`, text exposition format).

use crate::routes::AppState;
use axum::{
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// HTTP-level counters collected by [`track_requests`].
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// `(route, status) -> (count, total seconds)`.
    requests: Mutex<BTreeMap<(String, u16), (u64, f64)>>,
    in_flight: AtomicI64,
    /// Time to first token of chat completions, in microseconds.
    ttft_us_sum: AtomicU64,
    ttft_count: AtomicU64,
    /// Requests rejected because the waiting queue was full.
    pub rejected_overloaded: AtomicU64,
}

impl ServerMetrics {
    /// Records a time-to-first-token observation.
    pub fn observe_ttft(&self, start: Instant) {
        self.ttft_us_sum.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// Middleware counting requests per matched route and status code. For streamed
/// responses the duration covers the time until response headers are sent.
pub async fn track_requests(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "unmatched".to_string());
    let metrics = &state.metrics;
    metrics.in_flight.fetch_add(1, Ordering::Relaxed);
    let start = Instant::now();
    let response = next.run(req).await;
    metrics.in_flight.fetch_sub(1, Ordering::Relaxed);
    let mut map = metrics.requests.lock();
    let entry = map.entry((route, response.status().as_u16())).or_default();
    entry.0 += 1;
    entry.1 += start.elapsed().as_secs_f64();
    response
}

/// `GET /metrics`.
pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        render(&state),
    )
}

fn metric(out: &mut String, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}");
}

/// Renders all metrics in Prometheus text format.
pub fn render(state: &AppState) -> String {
    let mut out = String::new();
    let m = &state.metrics;

    let requests = m.requests.lock();
    let _ = writeln!(out, "# HELP rllm_http_requests_total HTTP requests by route and status.\n# TYPE rllm_http_requests_total counter");
    for ((route, status), (count, _)) in requests.iter() {
        let _ = writeln!(out, "rllm_http_requests_total{{route=\"{route}\",status=\"{status}\"}} {count}");
    }
    let _ = writeln!(out, "# HELP rllm_http_request_duration_seconds Time until response headers, by route.\n# TYPE rllm_http_request_duration_seconds summary");
    let mut by_route: BTreeMap<&str, (u64, f64)> = BTreeMap::new();
    for ((route, _), (count, secs)) in requests.iter() {
        let e = by_route.entry(route).or_default();
        e.0 += count;
        e.1 += secs;
    }
    for (route, (count, secs)) in by_route {
        let _ = writeln!(out, "rllm_http_request_duration_seconds_sum{{route=\"{route}\"}} {secs:.6}");
        let _ = writeln!(out, "rllm_http_request_duration_seconds_count{{route=\"{route}\"}} {count}");
    }
    drop(requests);
    metric(&mut out, "rllm_http_requests_in_flight", "gauge", "Requests currently being handled.", m.in_flight.load(Ordering::Relaxed));
    metric(&mut out, "rllm_requests_rejected_overloaded_total", "counter", "Requests rejected with 429 because the queue was full.", m.rejected_overloaded.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP rllm_time_to_first_token_seconds Time from request to first generated token.\n# TYPE rllm_time_to_first_token_seconds summary");
    let _ = writeln!(out, "rllm_time_to_first_token_seconds_sum {:.6}", m.ttft_us_sum.load(Ordering::Relaxed) as f64 / 1e6);
    let _ = writeln!(out, "rllm_time_to_first_token_seconds_count {}", m.ttft_count.load(Ordering::Relaxed));

    if let Some(engine) = &state.engine {
        let s = engine.scheduler();
        let sm = &s.metrics;
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        metric(&mut out, "rllm_sequences_running", "gauge", "Sequences in the running batch.", s.num_running());
        metric(&mut out, "rllm_sequences_waiting", "gauge", "Sequences waiting for admission.", s.num_waiting());
        metric(&mut out, "rllm_kv_blocks_total", "gauge", "KV-cache blocks in the pool.", s.config.total_blocks);
        metric(&mut out, "rllm_kv_blocks_free", "gauge", "Free KV-cache blocks.", s.num_free_blocks());
        metric(&mut out, "rllm_kv_blocks_prefix_cached", "gauge", "KV-cache blocks held by the prefix cache.", s.num_cached_blocks());
        metric(&mut out, "rllm_sequences_submitted_total", "counter", "Sequences accepted by the scheduler.", get(&sm.sequences_submitted));
        let _ = writeln!(out, "# HELP rllm_sequences_finished_total Finished sequences by reason.\n# TYPE rllm_sequences_finished_total counter");
        for (reason, c) in [("stop", &sm.finished_stop), ("length", &sm.finished_length), ("abort", &sm.finished_abort), ("error", &sm.failed)] {
            let _ = writeln!(out, "rllm_sequences_finished_total{{reason=\"{reason}\"}} {}", get(c));
        }
        metric(&mut out, "rllm_preemptions_total", "counter", "Sequences preempted for lack of KV blocks.", s.num_preemptions());
        metric(&mut out, "rllm_prompt_tokens_computed_total", "counter", "Prompt tokens run through the model.", get(&sm.prefill_tokens));
        metric(&mut out, "rllm_prompt_tokens_cached_total", "counter", "Prompt tokens reused from the prefix cache.", get(&sm.cached_prompt_tokens));
        metric(&mut out, "rllm_generated_tokens_total", "counter", "Generated tokens.", get(&sm.generated_tokens));
        metric(&mut out, "rllm_forward_passes_total", "counter", "Model forward passes.", get(&sm.forward_passes));
        metric(&mut out, "rllm_forward_seconds_total", "counter", "Time spent in forward passes.", format!("{:.6}", get(&sm.forward_time_us) as f64 / 1e6));
    }
    out
}

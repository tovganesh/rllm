//! Dedicated engine thread driving the continuous batching scheduler.
//!
//! Request handlers never step the scheduler themselves. They call
//! [`EngineHandle::submit`], which validates and enqueues a sequence and returns a
//! [`GenerationStream`] of [`SequenceEvent`]s. A single background thread steps the
//! scheduler whenever there is work and sleeps otherwise.
//!
//! Dropping a `GenerationStream` (e.g. when an HTTP client disconnects) closes its
//! channel; the scheduler notices at its next step, aborts the sequence and frees its
//! KV blocks.

use crate::sampler::SamplingParams;
use crate::scheduler::ContinuousBatchScheduler;
use crate::sequence::{Sequence, SequenceEvent};
use parking_lot::{Condvar, Mutex};
use rllm_core::backend::Backend;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedReceiver;
use tracing::{error, info};

/// Engine limits.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum number of sequences waiting for admission. Further submissions are
    /// rejected with [`SubmitError::Overloaded`].
    pub max_waiting: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { max_waiting: 256 }
    }
}

/// Why a submission was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    /// The request is invalid (empty or over-long prompt, bad token IDs, ...).
    #[error("{0}")]
    Invalid(String),
    /// The waiting queue is full.
    #[error("server overloaded: {waiting} requests already waiting")]
    Overloaded {
        /// Current queue length.
        waiting: usize,
    },
    /// The engine has been shut down.
    #[error("engine is shut down")]
    ShutDown,
}

struct Shared {
    stop: AtomicBool,
    work: Mutex<bool>,
    cond: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Shared {
    fn wake(&self) {
        *self.work.lock() = true;
        self.cond.notify_one();
    }
}

/// Cloneable handle to a running engine.
pub struct EngineHandle<B: Backend> {
    scheduler: Arc<ContinuousBatchScheduler<B>>,
    shared: Arc<Shared>,
    config: EngineConfig,
}

impl<B: Backend> Clone for EngineHandle<B> {
    fn clone(&self) -> Self {
        Self {
            scheduler: self.scheduler.clone(),
            shared: self.shared.clone(),
            config: self.config.clone(),
        }
    }
}

impl<B: Backend> EngineHandle<B> {
    /// Starts the engine thread for `scheduler`.
    pub fn start(scheduler: Arc<ContinuousBatchScheduler<B>>, config: EngineConfig) -> Self {
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            work: Mutex::new(false),
            cond: Condvar::new(),
            thread: Mutex::new(None),
        });
        let thread = {
            let scheduler = scheduler.clone();
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("rllm-engine".to_string())
                .spawn(move || run(scheduler, shared))
                .expect("failed to spawn engine thread")
        };
        *shared.thread.lock() = Some(thread);
        info!("Engine started (max waiting {})", config.max_waiting);
        Self {
            scheduler,
            shared,
            config,
        }
    }

    /// The scheduler driven by this engine (for configuration and gauges).
    pub fn scheduler(&self) -> &Arc<ContinuousBatchScheduler<B>> {
        &self.scheduler
    }

    /// Validates and enqueues a generation request.
    pub fn submit(&self, prompt_tokens: Vec<u32>, sampling: SamplingParams) -> Result<GenerationStream, SubmitError> {
        if self.shared.stop.load(Ordering::Acquire) {
            return Err(SubmitError::ShutDown);
        }
        let waiting = self.scheduler.num_waiting();
        if waiting >= self.config.max_waiting {
            return Err(SubmitError::Overloaded { waiting });
        }
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let id = self.scheduler.next_sequence_id();
        let prompt_len = prompt_tokens.len();
        let seq = Sequence::new(id, prompt_tokens, sampling, self.scheduler.config.block_size, Some(tx));
        self.scheduler
            .submit_sequence(seq)
            .map_err(|e| SubmitError::Invalid(e.to_string()))?;
        self.shared.wake();
        Ok(GenerationStream {
            id,
            prompt_tokens: prompt_len,
            rx,
        })
    }

    /// Stops the engine thread after its current step. Pending sequences are failed.
    pub fn shutdown(&self) {
        if self.shared.stop.swap(true, Ordering::AcqRel) {
            return;
        }
        self.shared.wake();
        if let Some(t) = self.shared.thread.lock().take() {
            let _ = t.join();
        }
        self.scheduler.fail_all("engine shut down");
        info!("Engine stopped");
    }
}

/// Events of one generation request. Dropping it cancels the request.
pub struct GenerationStream {
    /// Sequence ID inside the scheduler.
    pub id: u64,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    rx: UnboundedReceiver<SequenceEvent>,
}

impl GenerationStream {
    /// Next event, or `None` once the sequence is gone.
    pub async fn next(&mut self) -> Option<SequenceEvent> {
        self.rx.recv().await
    }

    /// Blocking variant of [`next`](Self::next) for non-async callers.
    pub fn blocking_next(&mut self) -> Option<SequenceEvent> {
        self.rx.blocking_recv()
    }
}

fn run<B: Backend>(scheduler: Arc<ContinuousBatchScheduler<B>>, shared: Arc<Shared>) {
    while !shared.stop.load(Ordering::Acquire) {
        if scheduler.has_work() {
            if let Err(e) = scheduler.step() {
                error!("Engine step failed: {e}; failing all in-flight sequences");
                scheduler.fail_all(&format!("engine error: {e}"));
            }
            continue;
        }
        let mut work = shared.work.lock();
        if !*work {
            // Timeout guards against a missed wake-up; it costs nothing when idle.
            shared.cond.wait_for(&mut work, Duration::from_millis(100));
        }
        *work = false;
    }
}

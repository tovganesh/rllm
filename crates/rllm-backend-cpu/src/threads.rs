//! Thread-pool sizing for the CPU backend.
//!
//! Kernels synchronise many times per token, so every task waits for the slowest worker.
//! On CPUs with performance and efficiency cores (Apple Silicon), the global Rayon pool
//! uses only the performance cores. The decode pool ([`crate::pool`]) hands out tasks
//! dynamically, so slower cores just take fewer: with enough work per step (matrices of
//! a model with hidden size >= 1024) it also uses up to as many efficiency cores as
//! there are performance cores ([`hint_model_width`]; Llama-3.2-1B decode on an M4,
//! 4 -> 8 threads: Q8_0 +33%, Q4_K_M +24%), while for small models (SmolLM2-135M) the
//! efficiency cores become stragglers and it keeps the performance cores. Large
//! integer GEMMs (prefill), made of many equal tasks that work stealing balances across
//! fast and slow cores, run on a second pool with every core ([`run_large`]):
//! Llama-3.2-1B Q8_0 128-token prompts on an M4 (4+6 cores) 391 -> 564 tok/s. Small
//! ones stay on the performance cores (all cores made SmolLM2-135M 128-token prompts
//! 35% slower).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Worker count for the decode pool, set by [`init_thread_pool`] (0 = unset).
static DECODE_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Whether the thread count was chosen explicitly (`--threads`, `RAYON_NUM_THREADS`);
/// then no extra all-core pool is used.
static EXPLICIT: AtomicBool = AtomicBool::new(false);

/// Multiply-adds from which an integer GEMM uses every core ([`run_large`]).
pub const LARGE_GEMM_MACS: usize = 1 << 29;

/// Number of performance cores on Apple Silicon, if the OS reports it.
#[cfg(target_os = "macos")]
fn macos_performance_cores() -> Option<usize> {
    use std::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn sysctlbyname(name: *const c_char, oldp: *mut c_void, oldlenp: *mut usize, newp: *mut c_void, newlen: usize) -> c_int;
    }
    let mut value: c_int = 0;
    let mut len = std::mem::size_of::<c_int>();
    // SAFETY: valid NUL-terminated name, output buffer of the declared size.
    let rc = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            &mut value as *mut c_int as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value as usize)
}

/// Default worker count: performance cores on Apple Silicon, otherwise the available
/// parallelism reported by the OS.
pub fn default_num_threads() -> usize {
    #[cfg(target_os = "macos")]
    if let Some(n) = macos_performance_cores() {
        return n;
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Efficiency cores on Apple Silicon (0 elsewhere or when unknown).
fn efficiency_cores() -> usize {
    #[cfg(target_os = "macos")]
    if let Some(p) = macos_performance_cores() {
        return all_cores().saturating_sub(p);
    }
    0
}

/// Hidden size from which decode uses efficiency cores too.
pub const WIDE_MODEL_HIDDEN: usize = 1024;

/// Tells the CPU backend the hidden size of the model about to run, before the decode
/// pool starts: models at least [`WIDE_MODEL_HIDDEN`] wide decode on the performance
/// cores plus up to as many efficiency cores. Ignored after an explicit thread count
/// or once the decode pool exists.
pub fn hint_model_width(hidden_size: usize) {
    if EXPLICIT.load(Ordering::Relaxed) || crate::pool::started() {
        return;
    }
    let p = default_num_threads();
    let n = if hidden_size >= WIDE_MODEL_HIDDEN { p + efficiency_cores().min(p) } else { p };
    DECODE_THREADS.store(n, Ordering::Relaxed);
}

/// All cores reported by the OS.
pub fn all_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Threads of the low-latency decode pool: what [`init_thread_pool`] chose, else
/// [`default_num_threads`].
pub fn decode_threads() -> usize {
    match DECODE_THREADS.load(Ordering::Relaxed) {
        0 => default_num_threads(),
        n => n,
    }
}

/// Runs `f` (a GEMM of `macs` multiply-adds whose parallel loops use Rayon) on a pool
/// with every core when it is large enough and the thread count was not set
/// explicitly; otherwise on the current pool.
pub fn run_large<R: Send>(macs: usize, f: impl FnOnce() -> R + Send) -> R {
    static ALL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    if macs < LARGE_GEMM_MACS || EXPLICIT.load(Ordering::Relaxed) {
        return f();
    }
    let pool = ALL.get_or_init(|| {
        let n = all_cores();
        (n > rayon::current_num_threads())
            .then(|| rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("rllm-gemm-{i}")).build().ok())
            .flatten()
    });
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// Threads that [`run_large`] runs a GEMM of `macs` multiply-adds on.
pub fn gemm_threads(macs: usize) -> usize {
    if macs >= LARGE_GEMM_MACS && !EXPLICIT.load(Ordering::Relaxed) {
        all_cores().max(rayon::current_num_threads())
    } else {
        rayon::current_num_threads()
    }
    .max(1)
}

/// Configures the thread pools. Precedence for the global Rayon pool and the decode
/// pool: `threads` argument, then the `RAYON_NUM_THREADS` environment variable, then
/// [`default_num_threads`]. Without an explicit count, large GEMMs also use an all-core
/// pool ([`run_large`]). Returns the Rayon thread count. Has no effect on a pool that
/// already exists.
pub fn init_thread_pool(threads: Option<usize>) -> usize {
    let explicit = threads
        .or_else(|| std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.parse().ok()))
        .filter(|&n| n > 0);
    EXPLICIT.store(explicit.is_some(), Ordering::Relaxed);
    let n = explicit.unwrap_or_else(default_num_threads);
    DECODE_THREADS.store(n, Ordering::Relaxed);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("rllm-cpu-{i}"))
        .build_global();
    rayon::current_num_threads()
}

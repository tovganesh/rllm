//! Thread-pool sizing for the CPU backend.
//!
//! Inference kernels synchronise many times per token, so every task waits for the
//! slowest worker. On CPUs with performance and efficiency cores (Apple Silicon), using
//! only the performance cores is markedly faster than using all cores.

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

/// Configures the global Rayon pool. Precedence: `threads` argument, then the
/// `RAYON_NUM_THREADS` environment variable, then [`default_num_threads`].
/// Returns the number of threads in use. Has no effect if the pool already exists.
pub fn init_thread_pool(threads: Option<usize>) -> usize {
    let n = threads
        .or_else(|| std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.parse().ok()))
        .filter(|&n| n > 0)
        .unwrap_or_else(default_num_threads);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("rllm-cpu-{i}"))
        .build_global();
    rayon::current_num_threads()
}

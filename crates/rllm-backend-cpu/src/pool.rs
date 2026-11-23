//! Low-latency fork/join pool for the decode path.
//!
//! Decoding one token runs ~100–200 small parallel steps (matrix-vector products,
//! attention, element-wise ops), each a few microseconds of work. With Rayon, idle
//! workers go to sleep between steps and every step pays a wake-up; on small models
//! that hand-off dominated the profile. This pool keeps its workers spinning for a
//! bounded time after each job (default 200 µs, `RLLM_SPIN_US`), so back-to-back steps
//! start immediately, then parks them so an idle process does not burn CPU.
//!
//! Protocol: the caller publishes a job (closure pointer and task count) and bumps an
//! epoch; **every** worker joins every epoch, takes task indices from a shared atomic
//! counter (dynamic balancing, so a preempted worker does not stall the step), and
//! reports done once. The caller runs tasks too, then waits until all workers have
//! reported, and only then may publish the next job — so a worker never reads a job
//! that is being rewritten, and the borrowed closure outlives every use.
//!
//! Nested calls (from inside a task) and calls made while another thread is using the
//! pool run inline on the calling thread. Panics in tasks are caught and re-raised on
//! the caller after all workers have finished.

use std::cell::{Cell, UnsafeCell};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

type Task<'a> = dyn Fn(usize) + Sync + 'a;

struct Job {
    f: *const Task<'static>,
    n: usize,
}

struct Pool {
    workers: usize,
    epoch: AtomicU64,
    job: UnsafeCell<Job>,
    next: AtomicUsize,
    done: AtomicUsize,
    panicked: AtomicBool,
    busy: AtomicBool,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
    spin: Duration,
}

// SAFETY: `job` is written only by the single caller holding `busy`, and only while no
// worker is reading it (all workers have reported done for the previous epoch);
// workers read it only between observing a new epoch and reporting done.
unsafe impl Sync for Pool {}
unsafe impl Send for Pool {}

thread_local! {
    /// True on pool workers and on a thread currently running a pool job.
    static IN_POOL: Cell<bool> = const { Cell::new(false) };
}

/// Placeholder job before the first real one.
static NOOP: fn(usize) = |_| {};

static POOL: OnceLock<&'static Pool> = OnceLock::new();

/// Whether the pool's workers have been started (its size is then fixed).
pub fn started() -> bool {
    POOL.get().is_some()
}

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let workers = crate::threads::decode_threads().saturating_sub(1);
        let spin_us = std::env::var("RLLM_SPIN_US").ok().and_then(|v| v.parse().ok()).unwrap_or(200u64);
        let p: &'static Pool = Box::leak(Box::new(Pool {
            workers,
            epoch: AtomicU64::new(0),
            job: UnsafeCell::new(Job { f: &NOOP as &Task<'static> as *const Task<'static>, n: 0 }),
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            spin: Duration::from_micros(spin_us),
        }));
        for i in 0..workers {
            std::thread::Builder::new()
                .name(format!("rllm-decode-{i}"))
                .spawn(move || worker(p))
                .expect("failed to spawn decode pool worker");
        }
        p
    })
}

/// Number of threads that run a job (workers plus the caller).
pub fn num_threads() -> usize {
    pool().workers + 1
}

fn run_tasks(p: &Pool, f: &Task<'_>, n: usize) {
    loop {
        let i = p.next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        f(i);
    }
}

fn worker(p: &'static Pool) {
    IN_POOL.with(|c| c.set(true));
    set_interactive_qos();
    let mut seen = 0u64;
    loop {
        // Wait for the next epoch: spin for `p.spin`, then park.
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            if p.epoch.load(Ordering::Acquire) != seen {
                break;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(256) && start.elapsed() >= p.spin {
                let mut guard = p.lock.lock().unwrap_or_else(|e| e.into_inner());
                p.sleepers.fetch_add(1, Ordering::SeqCst);
                while p.epoch.load(Ordering::SeqCst) == seen {
                    guard = p.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
                }
                p.sleepers.fetch_sub(1, Ordering::SeqCst);
                break;
            }
        }
        seen = p.epoch.load(Ordering::Acquire);
        // SAFETY: the job for epoch `seen` stays valid until this worker reports done.
        let (f, n) = unsafe {
            let job = &*p.job.get();
            (&*job.f, job.n)
        };
        if catch_unwind(AssertUnwindSafe(|| run_tasks(p, f, n))).is_err() {
            p.panicked.store(true, Ordering::Relaxed);
        }
        p.done.fetch_add(1, Ordering::Release);
    }
}

/// Runs `f(i)` for every `i` in `0..n`, spread over the decode pool (the calling
/// thread participates). Returns when all calls have finished.
pub fn for_each_index(n: usize, f: impl Fn(usize) + Sync) {
    if n == 0 {
        return;
    }
    let inline = || (0..n).for_each(&f);
    if n == 1 || IN_POOL.with(|c| c.get()) {
        return inline();
    }
    let p = pool();
    if p.workers == 0 || p.busy.swap(true, Ordering::Acquire) {
        return inline();
    }
    let f_dyn: &Task<'_> = &f;
    // SAFETY: no worker reads the job now (all reported done for the previous epoch
    // before the previous caller released `busy`). The closure's lifetime is erased;
    // it is only used until every worker reports done below.
    unsafe {
        *p.job.get() = Job { f: std::mem::transmute::<&Task<'_>, &Task<'static>>(f_dyn) as *const Task<'static>, n };
    }
    p.next.store(0, Ordering::Relaxed);
    p.done.store(0, Ordering::Relaxed);
    p.epoch.fetch_add(1, Ordering::SeqCst);
    if p.sleepers.load(Ordering::SeqCst) > 0 {
        let _g = p.lock.lock().unwrap_or_else(|e| e.into_inner());
        p.wake.notify_all();
    }
    IN_POOL.with(|c| c.set(true));
    let mine = catch_unwind(AssertUnwindSafe(|| run_tasks(p, f_dyn, n)));
    IN_POOL.with(|c| c.set(false));
    while p.done.load(Ordering::Acquire) < p.workers {
        std::hint::spin_loop();
    }
    let worker_panicked = p.panicked.swap(false, Ordering::Relaxed);
    p.busy.store(false, Ordering::Release);
    if let Err(e) = mine {
        resume_unwind(e);
    }
    if worker_panicked {
        panic!("a decode pool task panicked");
    }
}

/// Splits `data` into chunks of `chunk` elements and runs `f(index, chunk)` for each
/// on the decode pool.
pub fn for_each_chunk_mut<T: Send>(data: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let chunk = chunk.max(1);
    let len = data.len();
    let n = len.div_ceil(chunk);
    let base = SendPtr(data.as_mut_ptr());
    for_each_index(n, |i| {
        let start = i * chunk;
        let end = (start + chunk).min(len);
        // SAFETY: chunks are disjoint and within `data`, which outlives the call.
        let slice = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), end - start) };
        f(i, slice);
    });
}

struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
// SAFETY: used only to hand out disjoint sub-slices (see `for_each_chunk_mut`).
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// On macOS, asks the scheduler to keep this thread on performance cores.
fn set_interactive_qos() {
    #[cfg(target_os = "macos")]
    {
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        extern "C" {
            fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
        }
        // SAFETY: plain FFI call affecting only the calling thread.
        unsafe {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn runs_every_index_exactly_once_across_many_jobs() {
        for n in [0usize, 1, 2, 3, 7, 64, 1000] {
            for _ in 0..50 {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                for_each_index(n, |i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n={n}");
            }
        }
    }

    #[test]
    fn chunks_cover_the_slice_and_nested_calls_run_inline() {
        let mut v = vec![0u32; 1003];
        for_each_chunk_mut(&mut v, 10, |i, c| {
            for (j, x) in c.iter_mut().enumerate() {
                *x = (i * 10 + j) as u32;
            }
            // Nested use runs inline instead of deadlocking.
            let inner = AtomicUsize::new(0);
            for_each_index(3, |_| {
                inner.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(inner.load(Ordering::Relaxed), 3);
        });
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32));
    }

    #[test]
    fn concurrent_callers_and_parking_are_safe() {
        std::thread::scope(|s| {
            for t in 0..4 {
                s.spawn(move || {
                    for r in 0..200 {
                        let sum = AtomicUsize::new(0);
                        for_each_index(37, |i| {
                            sum.fetch_add(i, Ordering::Relaxed);
                        });
                        assert_eq!(sum.load(Ordering::Relaxed), 37 * 36 / 2);
                        if r % 50 == 0 && t == 0 {
                            // Let workers park, then wake them again.
                            std::thread::sleep(Duration::from_millis(3));
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn task_panics_reach_the_caller() {
        let r = catch_unwind(|| {
            for_each_index(16, |i| {
                if i == 11 {
                    panic!("boom");
                }
            })
        });
        assert!(r.is_err());
        // The pool is still usable afterwards.
        let c = AtomicUsize::new(0);
        for_each_index(16, |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(c.load(Ordering::Relaxed), 16);
    }
}

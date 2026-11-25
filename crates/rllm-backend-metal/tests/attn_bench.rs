//! Decode attention micro-benchmark (ignored): times one decode row's attention over a
//! paged F16 KV cache (kernel + split reduce), per layer of SmolLM2-135M and
//! Llama-3.2-1B, cycling through one cache per layer.
//!
//! `cargo test --release -p rllm-backend-metal --test attn_bench -- --ignored --nocapture`
//!
//! `ATTN_KERNELS="attention_gqa{G}_hd64_f16,variant{G}"` (`{G}` = query heads per KV head),
//! `ATTN_EXTRA=path.metal` appends experimental kernels, `ATTN_CTX=4096` context length,
//! `ATTN_CHUNK=256` keys per threadgroup, `ATTN_THREADS=128` per threadgroup, `ATTN_REPS=200`. Prints time per dispatch
//! (attention + reduce), the K/V read rate and the max relative difference against the
//! first kernel.
#![cfg(target_os = "macos")]

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[test]
#[ignore]
fn attn_bench() {
    let device = Device::system_default().unwrap();
    let mut src = rllm_backend_metal::shaders::SHADERS_MSL.to_string();
    if let Ok(p) = std::env::var("ATTN_EXTRA") {
        src.push_str(&std::fs::read_to_string(p).unwrap());
    }
    let lib = device.new_library_with_source(&src, &CompileOptions::new()).unwrap();
    let kernels: Vec<String> =
        std::env::var("ATTN_KERNELS").unwrap_or("attention_gqa{G}_hd64_f16".into()).split(',').map(String::from).collect();
    let ctx: usize = env("ATTN_CTX", 4096);
    let chunk: usize = env("ATTN_CHUNK", 256);
    let reps: usize = env("ATTN_REPS", 200);
    let threads: u64 = env("ATTN_THREADS", 128);
    let bs = 16usize;
    let q = device.new_command_queue();
    let shared = MTLResourceOptions::StorageModeShared;
    let reduce = device.new_compute_pipeline_state_with_function(&lib.get_function("attn_reduce", None).unwrap()).unwrap();
    // (name, query heads, KV heads, head dim, layers). Each layer has its own K/V cache,
    // visited in turn as in a decode step, so reads come from DRAM rather than the GPU's
    // system-level cache.
    for (model, nq, nkv, hd, layers) in [("SmolLM2-135M", 9usize, 3usize, 64usize, 30usize), ("Llama-3.2-1B", 32, 8, 64, 16)] {
        let g = nq / nkv;
        let nblocks = ctx.div_ceil(bs);
        // Pages in a scrambled order, as after allocation churn.
        let mut table: Vec<u32> = (0..nblocks as u32).collect();
        let mut s = 12345u64;
        for i in (1..table.len()).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            table.swap(i, (s >> 33) as usize % (i + 1));
        }
        let kv_len = nblocks * nkv * bs * hd;
        let mk = |seed: usize| -> Vec<half::f16> {
            (0..kv_len).map(|i| half::f16::from_f32((((i * 7919 + seed) % 2003) as f32 - 1001.0) / 1001.0)).collect()
        };
        let (k, v) = (mk(1), mk(7));
        let qv: Vec<f32> = (0..nq * hd).map(|i| ((i * 31 % 997) as f32 - 498.0) / 250.0).collect();
        let buf = |p: *const std::ffi::c_void, n: usize| device.new_buffer_with_data(p, n as u64, shared);
        let kbs: Vec<_> = (0..layers).map(|_| buf(k.as_ptr() as _, kv_len * 2)).collect();
        let vbs: Vec<_> = (0..layers).map(|_| buf(v.as_ptr() as _, kv_len * 2)).collect();
        let qb = buf(qv.as_ptr() as _, qv.len() * 4);
        let tb = buf(table.as_ptr() as _, table.len() * 4);
        let zero = [0u32];
        let toff = buf(zero.as_ptr() as _, 4);
        let rows = zero;
        let rb = buf(rows.as_ptr() as _, 4);
        let pos = [(ctx - 1) as u32];
        let pb = buf(pos.as_ptr() as _, 4);
        let nsplit = ctx.div_ceil(chunk);
        let part = device.new_buffer((nq * nsplit * (hd + 2) * 4) as u64, MTLResourceOptions::StorageModePrivate);
        let mut reference: Option<Vec<f32>> = None;
        let mut line = format!("{model:13} ctx={ctx} chunk={chunk}:");
        for name in &kernels {
            let name = name.replace("{G}", &g.to_string());
            let p = device.new_compute_pipeline_state_with_function(&lib.get_function(&name, None).unwrap()).unwrap();
            let out = device.new_buffer((nq * hd * 4) as u64, shared);
            let run = |reps: usize| {
                let cb = q.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                let u = |e: &metal::ComputeCommandEncoderRef, i: u64, v: u32| e.set_bytes(i, 4, &v as *const u32 as *const _);
                for rep in 0..reps {
                    e.set_compute_pipeline_state(&p);
                    let (kb, vb) = (&kbs[rep % layers], &vbs[rep % layers]);
                    for (i, b) in [&qb, kb, vb, &out, &tb, &toff, &pb].iter().enumerate() {
                        e.set_buffer(i as u64, Some(b), 0);
                    }
                    u(e, 7, nq as u32);
                    u(e, 8, nkv as u32);
                    u(e, 9, hd as u32);
                    u(e, 10, bs as u32);
                    let scale = 1.0f32 / (hd as f32).sqrt();
                    e.set_bytes(11, 4, &scale as *const f32 as *const _);
                    e.set_buffer(12, Some(&rb), 0);
                    u(e, 13, (nq * hd) as u32);
                    e.set_buffer(14, Some(&part), 0);
                    u(e, 15, chunk as u32);
                    u(e, 16, nsplit as u32);
                    let heads = if name.contains("gqa") { nkv } else { nq };
                    e.dispatch_thread_groups(MTLSize::new(heads as u64, 1, nsplit as u64), MTLSize::new(threads, 1, 1));
                    if nsplit > 1 {
                        e.set_compute_pipeline_state(&reduce);
                        e.set_buffer(0, Some(&part), 0);
                        e.set_buffer(1, Some(&out), 0);
                        e.set_buffer(2, Some(&rb), 0);
                        u(e, 3, nq as u32);
                        u(e, 4, hd as u32);
                        u(e, 5, nsplit as u32);
                        e.dispatch_thread_groups(MTLSize::new(nq as u64, 1, 1), MTLSize::new(hd as u64, 1, 1));
                    }
                }
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };
            // Warm up with a full run: the first kernel otherwise pays for the GPU clock ramp.
            run(reps);
            let t = Instant::now();
            run(reps);
            let dt = t.elapsed().as_secs_f64() / reps as f64;
            let o: Vec<f32> = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, nq * hd) }.to_vec();
            let diff = match &reference {
                None => {
                    reference = Some(o);
                    0.0
                }
                Some(r) => {
                    let sc = r.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
                    r.iter().zip(&o).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / sc
                }
            };
            let bytes = 2.0 * (ctx * nkv * hd * 2) as f64;
            line += &format!("  {name} {:.1}us {:.0}GB/s (diff {diff:.1e})", dt * 1e6, bytes / dt / 1e9);
        }
        println!("{line}");
    }
}

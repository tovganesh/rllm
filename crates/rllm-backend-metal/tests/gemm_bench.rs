//! Prefill GEMM micro-benchmark (ignored): times linear kernels on Llama-3.2-1B shapes.
//!
//! `cargo test --release -p rllm-backend-metal --test gemm_bench -- --ignored --nocapture`
//!
//! `GEMM_KERNELS="mm_q4_0:64:64:128,variant:64:32:128"` (name:rows-per-tile:tokens-per-tile:threads),
//! `GEMM_EXTRA=path.metal` appends experimental kernels to the library,
//! `GEMM_FMT=q4_0|q8_0|q4_k|q6_k|f16`, `GEMM_M` = activation rows (default 512). Prints
//! time and TFLOPS per shape and the max relative difference against the first kernel.
#![cfg(target_os = "macos")]

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

fn rnd_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

#[test]
#[ignore]
fn gemm_bench() {
    let device = Device::system_default().unwrap();
    let mut src = rllm_backend_metal::shaders::SHADERS_MSL.to_string();
    if let Ok(p) = std::env::var("GEMM_EXTRA") {
        src.push_str(&std::fs::read_to_string(p).unwrap());
    }
    let lib = device.new_library_with_source(&src, &CompileOptions::new()).unwrap();
    let fmt = std::env::var("GEMM_FMT").unwrap_or("q4_0".into());
    let (bvals, bbytes, doffs): (usize, usize, Vec<usize>) = match fmt.as_str() {
        "q4_0" => (32, 18, vec![0]),
        "q8_0" => (32, 34, vec![0]),
        "q4_k" => (256, 144, vec![0, 2]),
        "q6_k" => (256, 210, vec![208]),
        "f16" => (1, 2, vec![]),
        _ => panic!(),
    };
    let kernels: Vec<(String, u64, u64, u64)> = std::env::var("GEMM_KERNELS")
        .unwrap_or(format!("mm_{fmt}:64:64:128"))
        .split(',')
        .map(|s| {
            let p: Vec<&str> = s.split(':').collect();
            (p[0].to_string(), p[1].parse().unwrap(), p[2].parse().unwrap(), p[3].parse().unwrap())
        })
        .collect();
    let m: usize = std::env::var("GEMM_M").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let shapes = [(3072usize, 2048usize), (2048, 2048), (16384, 2048), (2048, 8192)];
    let q = device.new_command_queue();
    let mut totals = vec![0f64; kernels.len()];
    for (n, k) in shapes {
        let mut w = if fmt == "f16" {
            (0..n * k).flat_map(|i| half::f16::from_f32(((i * 7919 % 2001) as f32 - 1000.0) / 4000.0).to_le_bytes()).collect()
        } else {
            rnd_bytes(n * k / bvals * bbytes, 1)
        };
        for b in 0..n * k / bvals.max(1) {
            for &o in &doffs {
                w[b * bbytes + o..b * bbytes + o + 2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..m * k).map(|i| ((i * 31 % 997) as f32 - 498.0) / 500.0).collect();
        let wb = device.new_buffer_with_data(w.as_ptr() as *const _, w.len() as u64, MTLResourceOptions::StorageModeShared);
        let xb = device.new_buffer_with_data(x.as_ptr() as *const _, (x.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let xh: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
        let xhb = device.new_buffer_with_data(xh.as_ptr() as *const _, (xh.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
        let mut reference: Option<Vec<f32>> = None;
        let mut line = format!("{n:5}x{k:5} m={m}:");
        for (ki, (name, tn, tm, threads)) in kernels.iter().enumerate() {
            let f = lib.get_function(name, None).unwrap();
            let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
            let yb = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
            let run = |reps: usize| {
                let cb = q.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                e.set_compute_pipeline_state(&p);
                e.set_buffer(0, Some(&wb), 0);
                // The quantized/F16 tiled kernels take half-precision activations
                // (`GEMM_FLOAT_X=name1,name2` lists experimental kernels that take F32).
                let float_x = name == "mm_f32" || std::env::var("GEMM_FLOAT_X").unwrap_or_default().split(',').any(|k| k == name);
                e.set_buffer(1, Some(if float_x { &xb } else { &xhb }), 0);
                e.set_buffer(2, Some(&yb), 0);
                let args = [n as u32, k as u32, m as u32, 0u32, n as u32];
                for (i, v) in args.iter().enumerate() {
                    e.set_bytes(3 + i as u64, 4, v as *const u32 as *const _);
                }
                for _ in 0..reps {
                    e.dispatch_thread_groups(
                        MTLSize::new((n as u64).div_ceil(*tn), (m as u64).div_ceil(*tm), 1),
                        MTLSize::new(*threads, 1, 1),
                    );
                }
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };
            run(3);
            let reps = 20;
            let t = Instant::now();
            run(reps);
            let dt = t.elapsed().as_secs_f64() / reps as f64;
            totals[ki] += dt;
            let out: Vec<f32> =
                unsafe { std::slice::from_raw_parts(yb.contents() as *const f32, m * n) }.to_vec();
            let diff = match &reference {
                None => {
                    reference = Some(out);
                    0.0
                }
                Some(r) => {
                    let scale = r.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
                    r.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale
                }
            };
            line += &format!("  {name} {:.0}us {:.2}TF (diff {diff:.1e})", dt * 1e6, 2.0 * (m * n * k) as f64 / dt / 1e12);
        }
        println!("{line}");
    }
    println!("total per layer: {}", kernels.iter().zip(&totals).map(|(k, t)| format!("{} {:.0}us", k.0, t * 1e6)).collect::<Vec<_>>().join("  "));
}

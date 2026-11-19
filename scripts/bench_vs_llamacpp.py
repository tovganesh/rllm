#!/usr/bin/env python3
"""Side-by-side throughput of rllm vs llama.cpp on the same model and device.

Usage: python3 scripts/bench_vs_llamacpp.py [--device cpu|metal] [--reps N] <model.gguf> ...
Runs `llama-bench` (-ngl 0 for cpu, 99 for metal) and `rllm bench --model` with
pp128, pp512 and tg128, and prints tokens/s and the rllm / llama.cpp ratio.
Requires a release build of rllm (`cargo build --release -p rllm-cli`).
"""
import csv, io, os, re, subprocess, sys

args = sys.argv[1:]
device, reps = "cpu", 3
if "--device" in args:
    i = args.index("--device"); device = args[i + 1]; del args[i:i + 2]
if "--reps" in args:
    i = args.index("--reps"); reps = int(args[i + 1]); del args[i:i + 2]
ngl = "0" if device == "cpu" else "99"
rllm = os.path.join(os.path.dirname(__file__), "..", "target", "release", "rllm")

print(f"| model | device | test | llama.cpp t/s | rllm t/s | rllm / llama.cpp |")
print(f"| --- | --- | --- | ---: | ---: | ---: |")
for model in args:
    out = subprocess.run(["llama-bench", "-m", model, "-ngl", ngl, "-p", "128,512", "-n", "128", "-r", str(reps), "-o", "csv"],
                         capture_output=True, text=True).stdout
    ref = {}
    for row in csv.DictReader(io.StringIO(out)):
        key = f"pp{row['n_prompt']}" if int(row["n_gen"]) == 0 else f"tg{row['n_gen']}"
        ref[key] = float(row["avg_ts"])
    ours = {}
    for pp in (128, 512):
        o = subprocess.run([rllm, "--device", device, "bench", "--model", model, "--prompt-tokens", str(pp),
                            "--generate-tokens", "128", "--repetitions", str(reps)], capture_output=True, text=True).stdout
        for test, ts in re.findall(r"\| (pp\d+|tg\d+) \| ([0-9.]+) \|", o):
            ours.setdefault(test, float(ts))  # tg128 from the first run
    name = os.path.basename(model).replace(".gguf", "")
    for test in ("pp128", "pp512", "tg128"):
        a, b = ref.get(test, 0), ours.get(test, 0)
        print(f"| {name} | {device} | {test} | {a:,.0f} | {b:,.0f} | {b / a if a else 0:.2f}x |")
    sys.stdout.flush()

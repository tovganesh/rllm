#!/usr/bin/env python3
"""Record llama.cpp CPU perplexities for rllm's perplexity test.

Usage: python3 scripts/gen_perplexity_fixture.py <out.json> <text file> <ctx> <name>=<model.gguf> ...
Requires `llama-perplexity` (llama.cpp) on PATH.
"""
import json, re, subprocess, sys

out, text, ctx, *models = sys.argv[1:]
results = {}
for spec in models:
    name, path = spec.split("=", 1)
    r = subprocess.run(["llama-perplexity", "-m", path, "-f", text, "-c", ctx, "-b", ctx, "-ngl", "0", "-t", "4"],
                       capture_output=True, text=True)
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", r.stdout + r.stderr)
    if not m:
        sys.exit(f"no perplexity for {name}: {r.stderr[-300:]}")
    results[name] = float(m.group(1))
    print(name, results[name])
json.dump({"generator": f"llama-perplexity -c {ctx} -ngl 0 on {text}", "ctx": int(ctx), "text": text, "ppl": results},
          open(out, "w"), indent=1)

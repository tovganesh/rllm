#!/usr/bin/env python3
"""Generate embedding fixtures with llama.cpp.

Usage: python3 scripts/gen_embedding_fixture.py <model.gguf> <out.json>
Requires `llama-embedding` (llama.cpp) on PATH. Embeddings are L2-normalised.
"""
import json, os, subprocess, sys

model, out = sys.argv[1], sys.argv[2]
texts = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Rust is a systems programming language focused on safety and performance.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
    "How do plants make their food?",
    "naïve café résumé — UNICODE and Punctuation!!!",
    "a",
]
cases = []
for t in texts:
    r = subprocess.run(["llama-embedding", "-m", model, "-p", t, "--embd-normalize", "2",
                        "--embd-output-format", "json", "-t", "4", "-c", "512", "-b", "512", "-ub", "512"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("skipping (llama-embedding failed):", repr(t[:40]), r.stderr.strip().splitlines()[-1:])
        continue
    emb = json.loads(r.stdout)["data"][0]["embedding"]
    cases.append({"text": t, "embedding": [round(x, 6) for x in emb]})
json.dump({"model": os.path.basename(model), "generator": "llama-embedding --embd-normalize 2", "cases": cases},
          open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(len(cases), "cases")

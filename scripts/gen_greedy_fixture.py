#!/usr/bin/env python3
"""Generate greedy-completion fixtures with llama.cpp.

Usage: python3 scripts/gen_greedy_fixture.py <model.gguf> <out.json>
Usage: python3 scripts/gen_greedy_fixture.py [--gpu] <model.gguf> <out.json>
Requires `llama-completion` (llama.cpp) on PATH. Runs llama.cpp on the CPU (`-ngl 0`), or
with all layers offloaded (`-ngl 99`, Metal) when --gpu is given. On CPU,
its CPU kernels quantize activations to Q8 like rllm, whereas the Metal backend does not,
so CPU is the like-for-like reference.
"""
import json, subprocess, sys, os
args = sys.argv[1:]
gpu = "--gpu" in args
args = [a for a in args if a != "--gpu"]
model, out, n = args[0], args[1], 32
ngl = "99" if gpu else "0"
prompts = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Rust is a systems programming language that",
    "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nWhat is the boiling point of water at sea level?<|im_end|>\n<|im_start|>assistant\n",
]
cases = []
for p in prompts:
    r = subprocess.run(['llama-completion', '-m', model, '-p', p, '-n', str(n), '--temp', '0', '--top-k', '1',
                        '-no-cnv', '--no-warmup', '--no-display-prompt', '-t', '4', '-s', '1', '-ngl', ngl],
                       capture_output=True, text=True)
    text = r.stdout
    # llama-completion appends " [end of text]" when generation stops on EOS.
    eos = text.endswith(" [end of text]\n") or text.endswith("[end of text]\n")
    text = text.replace(" [end of text]\n", "").replace("[end of text]\n", "")
    # llama-completion prints "\n\n" after the last token when it stops on -n.
    if not eos and text.endswith("\n\n"):
        text = text[:-2]
    cases.append({"prompt": p, "max_tokens": n, "completion": text, "stopped_on_eos": eos})
    print(repr(p[:40]), '->', repr(text[:90]), 'eos' if eos else '')
json.dump({"model": os.path.basename(model), "generator": f"llama-completion greedy (--temp 0 --top-k 1 -ngl {ngl}), -n {n}", "cases": cases},
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

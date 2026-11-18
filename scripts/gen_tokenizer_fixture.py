#!/usr/bin/env python3
"""Generate tokenizer fixtures with llama.cpp.

Usage: python3 scripts/gen_tokenizer_fixture.py <model.gguf> <out.json>
Requires `llama-tokenize` (llama.cpp) on PATH.
"""
import json, subprocess, sys, tempfile, os
model, out = sys.argv[1], sys.argv[2]
corpus = [
    "The capital of France is",
    "Hello, world!",
    "I'm sure they've done it; we'll see what you'd say. It's John's.",
    "DON'T SHOUT, I'M HERE",
    "Numbers: 1, 12, 123, 1234, 12345 and 3.14159 or 1,000,000.",
    "   leading spaces and trailing   ",
    "multiple    internal     spaces",
    "line one\nline two\n\nline four\r\nwindows\ttabs\t\there",
    "\n\n\n",
    "naïve café résumé façade",
    "日本語のテキストと中文混合",
    "emoji 🦀🚀 and 👨‍👩‍👧 family",
    "fn main() {\n    println!(\"{}\", x + 1);\n}",
    "def f(x):\n\treturn x**2  # square",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "text with <|endoftext|> inside",
    "a<|im_end|>b",
    "URL: https://example.com/path?q=1&r=two#frag",
    "email@domain.org (test) [brackets] {braces}",
    "Ünïcödé ŞPÉÇÏÀL ÇHÄRS",
    "x",
    " ",
    "",
]
cases = []
for text in corpus:
    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as f:
        f.write(text); path = f.name
    r = subprocess.run(['llama-tokenize', '-m', model, '-f', path, '--ids', '--log-disable'],
                       capture_output=True, text=True)
    os.unlink(path)
    line = [l for l in r.stdout.strip().splitlines() if l.startswith('[')]
    ids = json.loads(line[-1]) if line else []
    cases.append({"text": text, "ids": ids})
json.dump({"model": os.path.basename(model), "generator": "llama-tokenize (llama.cpp) default flags: parse special, add BOS per model", "cases": cases},
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(cases), "cases")

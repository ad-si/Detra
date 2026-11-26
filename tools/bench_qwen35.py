# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-lm", "torch", "transformers"]
# ///
"""Benchmark the integer engine against float implementations of Qwen3.5-0.8B.

Usage: uv run tools/bench_qwen35.py <model-dir> <model.dllm> [--torch] [detllm-binary]

Measures model load time, prompt throughput, generation throughput and peak memory
for the same prompts with greedy decoding:

- detllm: the integer engine, all threads and single-threaded (CPU only).
- MLX: float (bfloat16) on the GPU via mlx-lm, loaded from the same checkpoint.
- with --torch: the Hugging Face PyTorch implementation in float32 on the CPU. It
  uses unoptimised fallback kernels for the recurrent layers and is very slow.
"""

import resource
import subprocess
import sys
import time

SHORT = "Write a detailed essay about the history of the Roman Empire."
PARAGRAPH = (
    "The Roman Empire grew from a small city-state on the Tiber into a power that "
    "ruled the Mediterranean for centuries. Its roads, laws and language shaped "
    "Europe long after the western half fell in the fifth century. "
)
LONG = PARAGRAPH * 24 + "Summarise the text above in one sentence."
NEW_TOKENS = 200


def detllm_run(binary, dllm, prompt, max_tokens, threads):
    """Wall time, peak memory (bytes) and token counts of one `detllm trace` run."""
    env = {"DETLLM_THREADS": str(threads)} if threads else {}
    cmd = ["/usr/bin/time", "-l", binary, "trace", "--max-tokens", str(max_tokens), "--model", dllm, "--", prompt]
    start = time.perf_counter()
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env={**__import__("os").environ, **env})
    wall = time.perf_counter() - start
    counts = [len(line.split()) - 1 for line in out.stdout.splitlines()]
    peak = next(int(line.split()[0]) for line in out.stderr.splitlines() if "maximum resident set size" in line)
    tokens = [[int(t) for t in line.split()[1:]] for line in out.stdout.splitlines()]
    return wall, peak, counts[0], counts[1], tokens[1]


def bench_detllm(binary, dllm, threads, label):
    # Differences between runs isolate the phases: an empty generation gives load +
    # prompt time, a longer prompt gives the prompt rate, more tokens the generation rate.
    base, _, n_short, _, _ = detllm_run(binary, dllm, SHORT, 0, threads)
    gen, peak, _, n_new, tokens = detllm_run(binary, dllm, SHORT, NEW_TOKENS, threads)
    long, _, n_long, _, _ = detllm_run(binary, dllm, LONG, 0, threads)
    prompt_tps = (n_long - n_short) / (long - base)
    load = base - n_short / prompt_tps
    report(label, load, prompt_tps, n_new / (gen - base), peak, n_long, n_new)
    return tokens


def report(label, load, prompt_tps, gen_tps, peak, n_prompt, n_new):
    print(f"{label:34s} load {load:6.2f} s   prompt {prompt_tps:8.1f} tok/s ({n_prompt} tokens)   "
          f"generation {gen_tps:7.1f} tok/s ({n_new} tokens)   peak memory {peak / 2**30:5.2f} GiB", flush=True)


def bench_mlx(src):
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    start = time.perf_counter()
    model, tok = load(src)
    mx.eval(model.parameters())
    load_time = time.perf_counter() - start

    def run(prompt, max_tokens):
        # mlx-lm enables thinking by default; the engine's template has it off.
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids, last = [], None
        for last in stream_generate(model, tok, text, max_tokens=max_tokens):
            ids.append(last.token)
        return ids, last

    run(SHORT, 8)  # warm-up: compiles kernels
    ids, gen = run(SHORT, NEW_TOKENS)
    _, long = run(LONG, 1)
    report("MLX float (GPU, bfloat16)", load_time, long.prompt_tps, gen.generation_tps,
           int(gen.peak_memory * 1e9), long.prompt_tokens, gen.generation_tokens)
    return ids


def bench_torch(src):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    start = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.float32).eval()
    load_time = time.perf_counter() - start

    def ids(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", add_special_tokens=False).input_ids

    new = 24
    with torch.no_grad():
        long_ids = ids(LONG)
        start = time.perf_counter()
        model(input_ids=long_ids)
        prompt_time = time.perf_counter() - start
        short_ids = ids(SHORT)
        start = time.perf_counter()
        model.generate(short_ids, max_new_tokens=1, do_sample=False)
        first = time.perf_counter() - start
        start = time.perf_counter()
        out = model.generate(short_ids, max_new_tokens=new, do_sample=False)
        gen_time = time.perf_counter() - start - first
    n_new = out.shape[1] - short_ids.shape[1]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report("PyTorch float32 (CPU, reference)", load_time, long_ids.shape[1] / prompt_time,
           (n_new - 1) / gen_time, peak, long_ids.shape[1], n_new)


def main():
    args = [a for a in sys.argv[1:] if a != "--torch"]
    if len(args) < 2:
        raise SystemExit(__doc__)
    src, dllm = args[0], args[1]
    binary = args[2] if len(args) > 2 else "target/release/detllm"
    ours = bench_detllm(binary, dllm, None, "detllm integer (CPU, all threads)")
    bench_detllm(binary, dllm, 1, "detllm integer (CPU, 1 thread)")
    theirs = bench_mlx(src)
    same = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
    print(f"detllm and MLX agree on the first {same} of {min(len(ours), len(theirs))} generated tokens")
    if "--torch" in sys.argv:
        bench_torch(src)


if __name__ == "__main__":
    main()

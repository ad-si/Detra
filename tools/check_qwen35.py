# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "transformers"]
# ///
"""Compare the integer engine with the Hugging Face float implementation of Qwen3.5.

Usage: uv run tools/check_qwen35.py <model-dir> <model.dllm> [detllm-binary]

Checks, per prompt: the prompt token ids (tokenizer and chat template), the logits of
the first predicted token, and the greedy continuation. This is a fidelity diagnostic:
the float side is not reproducible across machines and small differences are expected.
The reference model runs slowly on a CPU (several minutes per prompt).
"""

import subprocess
import sys
import tempfile

import unicodedata

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

TOKENIZER_CASES = [
    "Hello, world!",
    "I can't believe it's not butter; they'll've said THEY'RE wrong, I'M sure. He'd go",
    "  leading spaces\tand\ttabs  \n\n\n  indented line\r\n  another   \n",
    "x = 3.14159 * 2**10 - 1234567890 // 42",
    "Zoë's café was über cold — brr! 🙂🙃 naïve façade",
    "日本語のテキストと中文文本，以及한국어 텍스트。",
    "Ünïcödé combining: é ä ñ and Devanagari: नमस्ते दुनिया",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # comment\n",
    "URL https://example.com/path?q=1&r=two#frag and e-mail a.b@c.io",
    "Trailing space ",
    " ",
    "\n",
    "a b c　d١٢٣ Ⅷ ½",
    "!!!???...   ---   ***\n\n$$$",
    "'S 'T 'RE 'Ve 'M 'Ll 'D 'x '",
]
PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about deterministic computers.",
    "Explain in two sentences why floating point addition is not associative.",
    "List the first ten prime numbers.",
]
NEW_TOKENS = 32


def trace(binary, dllm, prompt, max_tokens, logits_path=None):
    cmd = [binary, "trace", prompt, "--max-tokens", str(max_tokens), "--model", dllm]
    if logits_path:
        cmd += ["--logits", logits_path]
    lines = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()
    return [[int(t) for t in line.split()[1:]] for line in lines]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dllm = sys.argv[1], sys.argv[2]
    binary = sys.argv[3] if len(sys.argv) > 3 else "target/release/detllm"
    tok = AutoTokenizer.from_pretrained(src)

    def prompt_ids(text):
        messages = [{"role": "user", "content": text}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tok(rendered, add_special_tokens=False).input_ids

    failures = 0
    for case in TOKENIZER_CASES:
        # The engine does not normalise Unicode; the reference applies NFC.
        case = unicodedata.normalize("NFC", case)
        want = prompt_ids(case)
        got, _ = trace(binary, dllm, case, 0)
        ok = got == want
        failures += not ok
        print(f"tokenizer {'ok  ' if ok else 'FAIL'} {case!r}")
        if not ok:
            print("   want", want, "\n   got ", got)

    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.float32)
    model.eval()
    for prompt in PROMPTS:
        want_ids = prompt_ids(prompt)
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            got_ids, got_new = trace(binary, dllm, prompt, NEW_TOKENS, f.name)
            got_logits = np.fromfile(f.name, dtype="<i4").astype(np.float64) / 2.0**20
        with torch.no_grad():
            ids = torch.tensor([want_ids])
            want_logits = model(input_ids=ids).logits[0, -1].double().numpy()
            out = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False)
        want_new = out[0, len(want_ids):].tolist()
        stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
        for i, t in enumerate(want_new):
            if t in stop:
                want_new = want_new[:i]
                break
        diff = np.abs(got_logits - want_logits)
        same = next((i for i, (a, b) in enumerate(zip(got_new, want_new)) if a != b), min(len(got_new), len(want_new)))
        top2 = np.sort(want_logits)[-2:]
        print(f"\nprompt {prompt!r}")
        print(f"  prompt ids equal      {got_ids == want_ids}")
        print(f"  first logits          max |diff| {diff.max():.5f}, mean |diff| {diff.mean():.5f}, "
              f"argmax equal {int(got_logits.argmax()) == int(want_logits.argmax())}, "
              f"float top-2 margin {top2[1] - top2[0]:.3f}")
        print(f"  greedy continuation   {same} of {len(want_new)} tokens equal"
              f"{'' if got_new == want_new else ' (then diverges)'}")
        print(f"  integer engine        {tok.decode(got_new)!r}")
        if got_new != want_new:
            print(f"  float reference       {tok.decode(want_new)!r}")
        failures += got_ids != want_ids
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

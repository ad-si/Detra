# detllm

Bit-exact, integer-only LLM inference engine written in Rust.
No floating point, no dependencies.

The output is a pure function of (model file bytes, prompt bytes, parameters).
The same inputs produce the same logits and tokens
on every machine, build profile, and target.


## How It Works

- No floating point anywhere in the engine crate (enforced by `make lint`).
  Weights are Q24 fixed point, activations and logits Q20,
  the residual stream Q32; probabilities and RoPE sin/cos are Q30.
- Every operation that may overflow uses explicit wrapping arithmetic,
  so debug and release builds agree.
- No dependence on pointer width, endianness, hash-map iteration order,
  or the number of threads (`$DETLLM_THREADS` changes the speed, never the output).
- Sampling (temperature, top-k, top-p) is seeded and also integer-only.
- The identity of a model is the SHA-256 of its converted `.dllm` file.
  Floating point is only used in the offline converters
  (`detllm-convert` and `tools/convert_qwen35.py`).

Two models are supported:

- [Qwen3.5-0.8B] (text part only, thinking mode off).
  This is the model `make install` installs.
- [stories15M] from [llama2.c].
  Small enough to run under emulation, so it is verified on every architecture.

The engine does not apply Unicode normalization:
prompts should be in NFC to match the original Qwen tokenizer.

[Qwen3.5-0.8B]: https://huggingface.co/Qwen/Qwen3.5-0.8B
[stories15M]: https://huggingface.co/karpathy/tinyllamas
[llama2.c]: https://github.com/karpathy/llama2.c


## Usage

```sh
make install
detllm generate "Why is the sky blue?" --max-tokens 100
```

```txt
detllm generate <prompt> [--max-tokens N] [--temperature T] [--top-k K]
                [--top-p P] [--seed S] [--model model.dllm]
detllm info [--model model.dllm]
detllm digest <prompts.txt> [--max-tokens N] [--model model.dllm]
```

Without `--model`, `$DETLLM_MODEL` is used,
or else `$XDG_DATA_HOME/detllm/model.dllm`
(default `~/.local/share/detllm/model.dllm`).
Temperature 0 (the default) is greedy decoding.


## Verification

`detllm digest` runs every prompt in `golden/prompts.txt` greedily
and with seeded sampling and prints a SHA-256
over all hidden states, logits, and tokens.
The `verify` targets compare that output with the files in `golden/`:

| Target                    | Checks                                                               |
| ------------------------- | -------------------------------------------------------------------- |
| `make verify-native`      | Native release build, both models (Qwen also single-threaded)        |
| `make verify-debug`       | Unoptimized build with overflow checks enabled (stories15M)          |
| `make verify-x86_64`      | x86-64 build run under Rosetta, both models                          |
| `make verify-wasm`        | WebAssembly (32-bit) build run under Node's WASI (stories15M)        |
| `make verify-miri`        | Synthetic models on big-endian 64-bit and 32-bit targets             |
| `make verify-docker`      | Emulated big-endian s390x, 32-bit x86, and x86-64 Linux (stories15M) |
| `make verify-docker-qwen` | Qwen3.5-0.8B on emulated big-endian s390x Linux                      |

`make verify` runs the tests, the linter, and the first four.
`make check-qwen` compares the integer engine
with the Hugging Face float implementation of Qwen3.5.
`make setup` installs the required Rust targets and components.
Run `make help` for all targets.


## Related

- [nathanrs/detllm] - Deterministic cross-device inference pipeline
    for Qwen3-0.6B in Python / PyTorch.
    Integer-only from input ids to int32 logits,
    with a cuBLASLt int8 GEMM backend for CUDA and a CPU reference backend.

[nathanrs/detllm]: https://github.com/nathanrs/detllm

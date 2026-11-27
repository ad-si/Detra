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
- Weights that came from 16-bit floats are stored in a lossless 16-bit encoding,
  which halves the file size and the memory traffic without changing any result.
- Every operation that may overflow uses explicit wrapping arithmetic,
  so debug and release builds agree.
- No dependence on pointer width, endianness, hash-map iteration order,
  or the number of threads (`$DETLLM_THREADS` changes the speed, never the output).
  Threads only ever compute independent blocks of wrapping integer sums,
  so neither their number nor their scheduling can affect a result.
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


## Guarantee

For a given model file, prompt (as bytes), and parameters,
every run on every machine ends in exactly one of two ways:

- the same hidden states, logits, and tokens, bit for bit, or
- an error, where a machine cannot run the request at all
  (out of memory, or a model too large for a 32-bit address space).

There is no third outcome: no input can make two machines produce different results.
This holds for arbitrary bytes as the prompt (including invalid UTF-8),
for any sampling parameters, and for any model file
(a damaged file is rejected or runs normally, but never differently per machine).

How this is ensured:

- Every arithmetic operation is either proven unable to overflow,
  or uses explicit wrapping or saturating integer arithmetic.
  The test-suite runs with overflow checks enabled,
  so an unproven overflow fails the tests instead of going unnoticed.
- Adversarial tests with fixed expected digests run on every target:
  random prompt bytes with random and out-of-range parameters,
  weights at the extremes of their range, and corrupted model files.
- `golden/prompts.txt` contains hostile prompts for the real models:
  invalid UTF-8, control characters, special-token text, mixed scripts,
  whitespace only, and a prompt longer than the context window.

This rests on a manual audit of the arithmetic plus the tests above,
not on a machine-checked proof.

Not covered: the offline converters (a model is identified by the hash
of the converted file, not by how it was produced),
and prompt bytes as passed by the shell on platforms
where arguments are not raw bytes (use `--prompt-file` there).


## Usage

```sh
make install
detllm generate "Why is the sky blue?" --max-tokens 100
```

```txt
detllm generate <prompt> [--max-tokens N] [--temperature T] [--top-k K]
                [--top-p P] [--seed S] [--model model.dllm]
detllm generate --prompt-file <path> [...]
detllm info [--model model.dllm]
detllm digest <prompts.txt> [--max-tokens N] [--model model.dllm]
detllm trace <prompt> | --messages <chat.json> [--max-tokens N] [--logits out.bin]
detllm serve [--port N] [--host H] [--allow-origin URL] [--model model.dllm]
```

Without `--model`, `$DETLLM_MODEL` is used,
or else `$XDG_DATA_HOME/detllm/model.dllm`
(default `~/.local/share/detllm/model.dllm`).
Temperature 0 (the default) is greedy decoding.
The text is printed as it is generated.
The prompt is used byte for byte; `--prompt-file` reads it from a file
(every byte counts, including a final newline).
Put `--` before a prompt that starts with dashes.


## Chat Server

```sh
detllm serve                # http://127.0.0.1:8080/v1
```

`detllm serve [--port N] [--host H] [--allow-origin URL] [--model model.dllm]`
serves an OpenAI-compatible API, so any chat front end
that talks to OpenAI-compatible servers can use it
(e.g. Open WebUI, LibreChat, Chatbox, Jan, or the `openai` client libraries).
Set the base URL to `http://127.0.0.1:8080/v1`, any API key, and model `detllm`.

- `POST /v1/chat/completions`: system, user and assistant messages,
  `stream` (server-sent events, with `stream_options.include_usage`),
  `max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`,
  `seed`, `stop`. Unknown fields are ignored.
- `GET /v1/models`, `GET /health`.

Not supported: tools, images, multiple choices (`n` > 1), logprobs, thinking mode.
Message text is always plain text:
it never turns into the control tokens that structure the conversation,
so a message cannot fake another turn.
A conversation longer than the context window is refused, not shortened.

The reply is a function of the model file and the request body only,
and the response contains no timestamps or random ids.
The same conversation with the same settings gives the same reply, byte for byte.
Defaults are greedy decoding and seed 0.
Sampling at any temperature uses the request's seed (0 if it sends none),
so it is reproducible too: "regenerate" in a front end gives the same answer again
unless the front end sends a different seed.

The server runs one generation at a time and keeps snapshots of recent sessions,
so continuing a conversation only computes the new messages.
It listens on localhost only by default.
Browser pages served from localhost may call it;
`--allow-origin` allows another origin (or `*` for any).
There is no authentication, so do not expose it to a network you do not trust.


## Verification

`detllm digest` runs every prompt in `golden/prompts.txt` greedily
and with seeded sampling and prints a SHA-256
over all hidden states, logits, and tokens (or the error).
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
`make test` runs the test-suite natively in debug and release builds;
the x86-64, WASM, Miri, and `verify-docker` targets also run it on their architecture.
`make check-qwen` compares the integer engine
with the Hugging Face float implementation of Qwen3.5,
including the chat template for multi-turn conversations.
`make setup` installs the required Rust targets and components.
Run `make help` for all targets.


## Performance

Qwen3.5-0.8B on an Apple M4 Max (10 performance cores), greedy decoding,
1030-token prompt, 200 generated tokens (`tools/bench_qwen35.py`):

| Runtime                          | Prompt       | Generation  | Peak memory |
| -------------------------------- | ------------ | ----------- | ----------- |
| detllm (integer, CPU)            | ~105 tok/s   | ~65 tok/s   | 1.5 GiB     |
| detllm (integer, CPU, 1 thread)  | ~18 tok/s    | ~12 tok/s   | 1.5 GiB     |
| MLX (bfloat16, GPU)              | ~7800 tok/s  | ~185 tok/s  | 1.5 GiB     |

Generation is limited by memory bandwidth (every token reads all weights once),
prompt processing by integer multiply-accumulate throughput on the CPU.


## Related

- [nathanrs/detllm] - Deterministic cross-device inference pipeline
    for Qwen3-0.6B in Python / PyTorch.
    Integer-only from input ids to int32 logits,
    with a cuBLASLt int8 GEMM backend for CUDA and a CPU reference backend.

[nathanrs/detllm]: https://github.com/nathanrs/detllm

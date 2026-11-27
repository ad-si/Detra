#![deny(clippy::float_arithmetic, clippy::float_cmp)]

use detllm::fixed::{parse_q16, ONE_Q16};
use detllm::chat::{chat_tokens, Role};
use detllm::engine::Session;
use detllm::generate::{feed_prompt, generate, run_digest, sample_tokens, GenParams};
use detllm::server::parse_messages;
use detllm::model::Model;
use detllm::sample::SampleParams;
use detllm::server::{serve, Config};
use detllm::sha256::{hex, Sha256};
use detllm::tokenizer::Tokenizer;
use std::ffi::OsString;
use std::io::{BufReader, Read, Write};
use std::process::ExitCode;
use std::sync::Arc;

const USAGE: &str = "\
Usage:
  detllm generate <prompt> [--max-tokens N] [--temperature T] [--top-k K]
                  [--top-p P] [--seed S] [--model model.dllm]
  detllm generate --prompt-file <path> [...]
  detllm info [--model model.dllm]
  detllm digest <prompts.txt> [--max-tokens N] [--model model.dllm]
  detllm trace <prompt> [--max-tokens N] [--logits out.bin] [--model model.dllm]
  detllm trace --messages <chat.json> [...]
  detllm serve [--port 8080] [--host 127.0.0.1] [--allow-origin URL] [--model model.dllm]

Without --model, $DETLLM_MODEL is used, or else
$XDG_DATA_HOME/detllm/model.dllm (default ~/.local/share/detllm/model.dllm).

generate: temperature 0 (the default) is greedy decoding. T and P are decimals.
          The prompt is used byte for byte; --prompt-file reads it from a file (every
          byte counts, including a final newline). Put `--` before a prompt that
          starts with dashes.
digest:   runs every line of the prompts file greedily and with seeded sampling and
          prints a SHA-256 over all hidden states, logits and tokens (or the error).
          Equal output means bit-exact.
serve:    OpenAI-compatible HTTP API for chat front ends: POST /v1/chat/completions
          (streaming supported), GET /v1/models. Listens on 127.0.0.1:8080 by default.
          Browser pages from localhost may call it; --allow-origin adds another
          origin, or `*` for any. Defaults are greedy decoding and seed 0, so a
          request gives the same reply every time.
trace:    prints the token ids of the prompt and of the greedy continuation, and
          optionally writes the first predicted logits as little-endian i32 (Q20).
          --messages takes a conversation instead, as a JSON list of
          {\"role\": ..., \"content\": ...} objects in the chat API's format.

$DETLLM_THREADS sets the number of threads. It changes the speed, never the output.";

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    positional: Vec<OsString>,
    model: Option<String>,
    logits: Option<String>,
    prompt_file: Option<String>,
    messages: Option<String>,
    max_tokens: Option<u32>,
    sample: SampleParams,
    host: Option<String>,
    port: Option<u16>,
    allow_origin: Option<String>,
}

/// Options are `--name value`. Everything after a bare `--` is positional, so a prompt
/// may start with dashes. Numbers are parsed as fixed-width integers so that the same
/// command line is accepted or rejected identically on 32-bit and 64-bit targets.
fn parse_options(args: &[OsString]) -> Result<Options, String> {
    let mut o = Options {
        positional: Vec::new(),
        model: None,
        logits: None,
        prompt_file: None,
        messages: None,
        max_tokens: None,
        sample: SampleParams::GREEDY,
        host: None,
        port: None,
        allow_origin: None,
    };
    let mut it = args.iter();
    let mut options_ended = false;
    while let Some(arg) = it.next() {
        let name = arg.to_str().filter(|a| !options_ended && a.starts_with("--"));
        let Some(name) = name else {
            o.positional.push(arg.clone());
            continue;
        };
        if name == "--" {
            options_ended = true;
            continue;
        }
        let value = it.next().ok_or(format!("{name} needs a value"))?;
        let value = value.to_str().ok_or(format!("the value of {name} is not valid Unicode"))?;
        let bad = || format!("invalid value for {name}: {value}");
        match name {
            "--model" => o.model = Some(value.to_string()),
            "--logits" => o.logits = Some(value.to_string()),
            "--prompt-file" => o.prompt_file = Some(value.to_string()),
            "--messages" => o.messages = Some(value.to_string()),
            "--max-tokens" => o.max_tokens = Some(value.parse().map_err(|_| bad())?),
            "--temperature" => o.sample.temperature_q16 = parse_q16(value).ok_or_else(bad)?,
            "--top-k" => o.sample.top_k = value.parse::<u32>().map_err(|_| bad())? as usize,
            "--top-p" => {
                let p = parse_q16(value).ok_or_else(bad)?;
                if !(1..=ONE_Q16).contains(&p) {
                    return Err(bad());
                }
                o.sample.top_p_q16 = p;
            }
            "--seed" => o.sample.seed = value.parse().map_err(|_| bad())?,
            "--host" => o.host = Some(value.to_string()),
            "--port" => o.port = Some(value.parse().map_err(|_| bad())?),
            "--allow-origin" => o.allow_origin = Some(value.to_string()),
            _ => return Err(format!("unknown option {name}")),
        }
    }
    Ok(o)
}

/// The prompt as bytes: the positional argument exactly as the OS passed it, or the
/// exact contents of `--prompt-file`.
fn prompt_bytes(o: &Options, positional: &[OsString]) -> Result<Vec<u8>, String> {
    match (positional, &o.prompt_file) {
        ([prompt], None) => Ok(prompt.as_encoded_bytes().to_vec()),
        ([], Some(path)) => std::fs::read(path).map_err(|e| format!("{path}: {e}")),
        _ => Err("give either one prompt argument or --prompt-file".into()),
    }
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Location of the installed model. Only the CLI looks at the environment; the engine
/// itself takes model bytes and nothing else.
fn model_path(explicit: Option<&String>) -> Result<String, String> {
    if let Some(path) = explicit.cloned().or_else(|| env_var("DETLLM_MODEL")) {
        return Ok(path);
    }
    let data_dir = match (env_var("XDG_DATA_HOME"), env_var("HOME")) {
        (Some(dir), _) => dir,
        (None, Some(home)) => format!("{home}/.local/share"),
        (None, None) => return Err("no --model given and neither DETLLM_MODEL nor HOME is set".into()),
    };
    Ok(format!("{data_dir}/detllm/model.dllm"))
}

fn threads() -> Result<usize, String> {
    if cfg!(target_family = "wasm") {
        return Ok(1);
    }
    match env_var("DETLLM_THREADS") {
        Some(v) => match v.parse::<u32>() {
            Ok(n) => Ok(n.clamp(1, 256) as usize),
            Err(_) => Err(format!("invalid DETLLM_THREADS: {v}")),
        },
        None => Ok(performance_cores().unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))),
    }
}

/// Number of performance cores on Apple silicon. Generation is limited by memory
/// bandwidth, and threads on efficiency cores slow it down rather than help.
fn performance_cores() -> Option<usize> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", "hw.perflevel0.physicalcpu"]).output().ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok().filter(|&n| n > 0)
}

/// Passes bytes through while hashing them.
struct HashingReader<R> {
    inner: R,
    hasher: Option<Sha256>,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if let Some(h) = &mut self.hasher {
            h.update(&buf[..n]);
        }
        Ok(n)
    }
}

/// Load a model; with `hash`, also return the SHA-256 of the file.
fn load(explicit: Option<&String>, hash: bool) -> Result<(Arc<Model>, [u8; 32]), String> {
    let path = model_path(explicit)?;
    let file = std::fs::File::open(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut reader = HashingReader {
        inner: BufReader::with_capacity(1 << 20, file),
        hasher: hash.then(Sha256::new),
    };
    let model = Model::read(&mut reader).map_err(|e| format!("{path}: {e}"))?;
    let id = reader.hasher.map_or([0; 32], Sha256::finalize);
    Ok((Arc::new(model), id))
}

fn ids(tokens: &[u32]) -> String {
    tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

fn run(args: &[OsString]) -> Result<(), String> {
    let o = parse_options(args)?;
    let Some((command, positional)) = o.positional.split_first() else {
        return Err(USAGE.into());
    };
    let threads = threads()?;
    let max_tokens = o.max_tokens.map(|n| n as usize);
    match (command.to_str().unwrap_or(""), positional) {
        ("info", []) => {
            let (m, id) = load(o.model.as_ref(), true)?;
            println!("sha256      {}", hex(&id));
            println!("dim         {}", m.dim);
            println!("hidden_dim  {}", m.hidden_dim);
            println!("n_layers    {}", m.layers.len());
            println!("vocab_size  {}", m.vocab_size);
            println!("seq_len     {}", m.seq_len);
            Ok(())
        }
        ("generate", _) => {
            let prompt = prompt_bytes(&o, positional)?;
            let (m, _) = load(o.model.as_ref(), false)?;
            let params = GenParams {
                max_new_tokens: max_tokens.unwrap_or(usize::MAX),
                sample: o.sample,
                threads,
            };
            // Print the text as it is generated. Each step decodes everything shown so
            // far and writes what is new, so the bytes are exactly those of decoding the
            // finished token sequence at once.
            let tokenizer = Tokenizer::new(&m);
            let mut stdout = std::io::stdout();
            let mut written = 0;
            let mut failed = None;
            let mut show = |text: &[u32], generated: &[u32]| {
                let shown: Vec<u32> = if m.echo_prompt { [text, generated].concat() } else { generated.to_vec() };
                let bytes = tokenizer.decode(&shown);
                if failed.is_none() && bytes.len() > written {
                    failed = stdout.write_all(&bytes[written..]).and_then(|()| stdout.flush()).err();
                    written = bytes.len();
                }
            };
            let out = generate(&m, &prompt, &params, |step| show(step.text, step.generated))
                .map_err(|e| e.to_string())?;
            show(&out.tokens[out.text.clone()], &out.tokens[out.generated_from..]);
            if let Some(e) = failed {
                return Err(e.to_string());
            }
            stdout.write_all(b"\n").map_err(|e| e.to_string())
        }
        ("trace", _) => {
            let params = GenParams {
                max_new_tokens: max_tokens.unwrap_or(16),
                sample: SampleParams::GREEDY,
                threads,
            };
            let mut first_logits: Option<Vec<i32>> = None;
            let (tokens, generated_from) = match &o.messages {
                Some(path) if positional.is_empty() && o.prompt_file.is_none() => {
                    let body = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
                    let messages = parse_messages(&body).map_err(|e| format!("{path}: {e}"))?;
                    let (m, _) = load(o.model.as_ref(), false)?;
                    let tokenizer = Tokenizer::new(&m);
                    let list: Vec<(Role, &[u8])> = messages.iter().map(|(r, c)| (*r, c.as_bytes())).collect();
                    let mut tokens = chat_tokens(&m, &tokenizer, &list).map_err(|e| e.to_string())?;
                    let prompt_len = tokens.len();
                    let mut session = Session::new(Arc::clone(&m), threads);
                    let want_logits = params.max_new_tokens > 0 && prompt_len < m.seq_len;
                    feed_prompt(&mut session, &tokens, want_logits, |_, logits| {
                        if !logits.is_empty() {
                            first_logits = Some(logits.to_vec());
                        }
                    })
                    .map_err(|e| e.to_string())?;
                    sample_tokens(&mut session, &mut tokens, &params, |_, _| true).map_err(|e| e.to_string())?;
                    (tokens, prompt_len)
                }
                Some(_) => return Err("give either --messages or a prompt".into()),
                None => {
                    let prompt = prompt_bytes(&o, positional)?;
                    let (m, _) = load(o.model.as_ref(), false)?;
                    let out = generate(&m, &prompt, &params, |step| {
                        if first_logits.is_none() && !step.logits.is_empty() {
                            first_logits = Some(step.logits.to_vec());
                        }
                    })
                    .map_err(|e| e.to_string())?;
                    (out.tokens, out.generated_from)
                }
            };
            println!("prompt {}", ids(&tokens[..generated_from]));
            println!("generated {}", ids(&tokens[generated_from..]));
            if let (Some(path), Some(logits)) = (&o.logits, &first_logits) {
                let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))?;
            }
            Ok(())
        }
        ("digest", [prompts_path]) => {
            let (m, id) = load(o.model.as_ref(), true)?;
            let prompts = std::fs::read(prompts_path)
                .map_err(|e| format!("{}: {e}", prompts_path.to_string_lossy()))?;
            let max_new_tokens = max_tokens.unwrap_or(48);
            println!("model {}", hex(&id));
            let mut all = Sha256::new();
            all.update(&id);
            for (i, line) in prompts.split(|&b| b == b'\n').enumerate() {
                if line.is_empty() {
                    continue;
                }
                let sampled = SampleParams {
                    temperature_q16: 52_429, // 0.8
                    top_k: 40,
                    top_p_q16: 58_982, // 0.9
                    seed: i as u64,
                };
                for (mode, sample) in [("greedy", SampleParams::GREEDY), ("sample", sampled)] {
                    let params = GenParams { max_new_tokens, sample, threads };
                    // A rejected prompt is an outcome too: it must be rejected everywhere.
                    match run_digest(&m, line, &params) {
                        Ok((digest, out)) => {
                            all.update(&digest);
                            println!("{i:03} {mode} {:3} tokens {}", out.tokens.len(), hex(&digest));
                        }
                        Err(e) => {
                            all.update(e.0.as_bytes());
                            println!("{i:03} {mode} error: {e}");
                        }
                    }
                }
            }
            println!("total {}", hex(&all.finalize()));
            Ok(())
        }
        ("serve", []) => {
            let (m, id) = load(o.model.as_ref(), true)?;
            if m.chat.is_none() {
                return Err("this model has no chat template, so it cannot be served".into());
            }
            let addr = format!("{}:{}", o.host.as_deref().unwrap_or("127.0.0.1"), o.port.unwrap_or(8080));
            let listener = std::net::TcpListener::bind(&addr).map_err(|e| format!("{addr}: {e}"))?;
            let name = "detllm".to_string();
            eprintln!("Serving model `{name}` (sha256 {}) at http://{addr}/v1", &hex(&id)[..16]);
            serve(listener, Config { model: m, model_id: id, name, threads, allow_origin: o.allow_origin, log: true });
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

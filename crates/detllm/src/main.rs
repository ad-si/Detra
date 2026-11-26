use detllm::fixed::{parse_q16, ONE_Q16};
use detllm::generate::{generate, run_digest, GenParams};
use detllm::model::Model;
use detllm::sample::SampleParams;
use detllm::sha256::{hex, Sha256};
use detllm::tokenizer::Tokenizer;
use std::io::{BufReader, Read, Write};
use std::process::ExitCode;
use std::sync::Arc;

const USAGE: &str = "\
Usage:
  detllm generate <prompt> [--max-tokens N] [--temperature T] [--top-k K]
                  [--top-p P] [--seed S] [--model model.dllm]
  detllm info [--model model.dllm]
  detllm digest <prompts.txt> [--max-tokens N] [--model model.dllm]
  detllm trace <prompt> [--max-tokens N] [--logits out.bin] [--model model.dllm]

Without --model, $DETLLM_MODEL is used, or else
$XDG_DATA_HOME/detllm/model.dllm (default ~/.local/share/detllm/model.dllm).

generate: temperature 0 (the default) is greedy decoding. T and P are decimals.
digest:   runs every line of the prompts file greedily and with seeded sampling and
          prints a SHA-256 over all hidden states, logits and tokens.
          Equal output means bit-exact.
trace:    prints the token ids of the prompt and of the greedy continuation, and
          optionally writes the first predicted logits as little-endian i32 (Q20).

$DETLLM_THREADS sets the number of threads. It changes the speed, never the output.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    positional: Vec<String>,
    model: Option<String>,
    logits: Option<String>,
    max_tokens: Option<usize>,
    sample: SampleParams,
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        positional: Vec::new(),
        model: None,
        logits: None,
        max_tokens: None,
        sample: SampleParams::GREEDY,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if !arg.starts_with("--") {
            o.positional.push(arg.clone());
            continue;
        }
        let value = it.next().ok_or(format!("{arg} needs a value"))?;
        let bad = || format!("invalid value for {arg}: {value}");
        match arg.as_str() {
            "--model" => o.model = Some(value.clone()),
            "--logits" => o.logits = Some(value.clone()),
            "--max-tokens" => o.max_tokens = Some(value.parse().map_err(|_| bad())?),
            "--temperature" => o.sample.temperature_q16 = parse_q16(value).ok_or_else(bad)?,
            "--top-k" => o.sample.top_k = value.parse().map_err(|_| bad())?,
            "--top-p" => {
                let p = parse_q16(value).ok_or_else(bad)?;
                if !(1..=ONE_Q16).contains(&p) {
                    return Err(bad());
                }
                o.sample.top_p_q16 = p;
            }
            "--seed" => o.sample.seed = value.parse().map_err(|_| bad())?,
            _ => return Err(format!("unknown option {arg}")),
        }
    }
    Ok(o)
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
        Some(v) => v.parse().map_err(|_| format!("invalid DETLLM_THREADS: {v}")),
        None => Ok(std::thread::available_parallelism().map_or(1, |n| n.get())),
    }
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

fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first() else {
        return Err(USAGE.into());
    };
    let o = parse_options(&args[1..])?;
    let threads = threads()?;
    match (command.as_str(), o.positional.as_slice()) {
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
        ("generate", [prompt]) => {
            let (m, _) = load(o.model.as_ref(), false)?;
            let params = GenParams {
                max_new_tokens: o.max_tokens.unwrap_or(usize::MAX),
                sample: o.sample,
                threads,
            };
            let out = generate(&m, prompt.as_bytes(), &params, |_| {}).map_err(|e| e.to_string())?;
            let tokenizer = Tokenizer::new(&m);
            let shown: Vec<u32> = if m.echo_prompt {
                out.tokens[out.text.clone()].iter().chain(&out.tokens[out.generated_from..]).copied().collect()
            } else {
                out.tokens[out.generated_from..].to_vec()
            };
            let mut text = tokenizer.decode(&shown);
            text.push(b'\n');
            std::io::stdout().write_all(&text).map_err(|e| e.to_string())
        }
        ("trace", [prompt]) => {
            let (m, _) = load(o.model.as_ref(), false)?;
            let params = GenParams {
                max_new_tokens: o.max_tokens.unwrap_or(16),
                sample: SampleParams::GREEDY,
                threads,
            };
            let mut first_logits: Option<Vec<i32>> = None;
            let out = generate(&m, prompt.as_bytes(), &params, |step| {
                if first_logits.is_none() && !step.logits.is_empty() {
                    first_logits = Some(step.logits.to_vec());
                }
            })
            .map_err(|e| e.to_string())?;
            println!("prompt {}", ids(&out.tokens[..out.generated_from]));
            println!("generated {}", ids(&out.tokens[out.generated_from..]));
            if let (Some(path), Some(logits)) = (&o.logits, &first_logits) {
                let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))?;
            }
            Ok(())
        }
        ("digest", [prompts_path]) => {
            let (m, id) = load(o.model.as_ref(), true)?;
            let prompts =
                std::fs::read(prompts_path).map_err(|e| format!("{prompts_path}: {e}"))?;
            let max_new_tokens = o.max_tokens.unwrap_or(48);
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
                    let (digest, out) =
                        run_digest(&m, line, &params).map_err(|e| e.to_string())?;
                    all.update(&digest);
                    println!("{i:03} {mode} {:3} tokens {}", out.tokens.len(), hex(&digest));
                }
            }
            println!("total {}", hex(&all.finalize()));
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

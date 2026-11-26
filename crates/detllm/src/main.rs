use detllm::fixed::{parse_q16, ONE_Q16};
use detllm::generate::{generate, run_digest, GenParams};
use detllm::model::Model;
use detllm::sample::SampleParams;
use detllm::sha256::{hex, sha256, Sha256};
use detllm::tokenizer::Tokenizer;
use std::io::Write;
use std::process::ExitCode;

const USAGE: &str = "\
Usage:
  detllm generate <prompt> [--max-tokens N] [--temperature T] [--top-k K]
                  [--top-p P] [--seed S] [--model model.dllm]
  detllm info [--model model.dllm]
  detllm digest <prompts.txt> [--max-tokens N] [--model model.dllm]

Without --model, $DETLLM_MODEL is used, or else
$XDG_DATA_HOME/detllm/model.dllm (default ~/.local/share/detllm/model.dllm).
generate: temperature 0 (the default) is greedy decoding. T and P are decimals.
digest:   runs every line of the prompts file greedily and with seeded sampling and
          prints a SHA-256 over all logits and tokens. Equal output means bit-exact.";

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
    max_tokens: usize,
    sample: SampleParams,
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        positional: Vec::new(),
        model: None,
        max_tokens: 0,
        sample: SampleParams::GREEDY,
    };
    let mut max_tokens = None;
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
            "--max-tokens" => max_tokens = Some(value.parse().map_err(|_| bad())?),
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
    o.max_tokens = max_tokens.unwrap_or(usize::MAX);
    Ok(o)
}

/// Location of the installed model. Only the CLI looks at the environment; the engine
/// itself takes model bytes and nothing else.
fn default_model_path() -> Result<String, String> {
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    if let Some(path) = var("DETLLM_MODEL") {
        return Ok(path);
    }
    let data_dir = match (var("XDG_DATA_HOME"), var("HOME")) {
        (Some(dir), _) => dir,
        (None, Some(home)) => format!("{home}/.local/share"),
        (None, None) => return Err("no model path given and neither DETLLM_MODEL nor HOME is set".into()),
    };
    Ok(format!("{data_dir}/detllm/model.dllm"))
}

fn load_or_default(path: Option<&String>) -> Result<(Model, [u8; 32]), String> {
    match path {
        Some(path) => load(path),
        None => load(&default_model_path()?),
    }
}

fn load(path: &str) -> Result<(Model, [u8; 32]), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("{path}: {e}"))?;
    Ok((model, sha256(&bytes)))
}

fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first() else {
        return Err(USAGE.into());
    };
    let o = parse_options(&args[1..])?;
    match (command.as_str(), o.positional.as_slice()) {
        ("info", []) => {
            let (m, id) = load_or_default(o.model.as_ref())?;
            println!("sha256      {}", hex(&id));
            println!("dim         {}", m.dim);
            println!("hidden_dim  {}", m.hidden_dim);
            println!("n_layers    {}", m.n_layers);
            println!("n_heads     {}", m.n_heads);
            println!("vocab_size  {}", m.vocab_size);
            println!("seq_len     {}", m.seq_len);
            Ok(())
        }
        ("generate", [prompt]) => {
            let (m, _) = load_or_default(o.model.as_ref())?;
            let params = GenParams { max_new_tokens: o.max_tokens, sample: o.sample };
            let tokens =
                generate(&m, prompt.as_bytes(), &params, |_| {}).map_err(|e| e.to_string())?;
            let mut text = Tokenizer::new(&m).decode(&tokens);
            text.push(b'\n');
            std::io::stdout().write_all(&text).map_err(|e| e.to_string())
        }
        ("digest", [prompts_path]) => {
            let (m, id) = load_or_default(o.model.as_ref())?;
            let prompts =
                std::fs::read(prompts_path).map_err(|e| format!("{prompts_path}: {e}"))?;
            let max_new_tokens = if o.max_tokens == usize::MAX { 48 } else { o.max_tokens };
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
                    let params = GenParams { max_new_tokens, sample };
                    let (digest, tokens) =
                        run_digest(&m, line, &params).map_err(|e| e.to_string())?;
                    all.update(&digest);
                    println!("{i:03} {mode} {:3} tokens {}", tokens.len(), hex(&digest));
                }
            }
            println!("total {}", hex(&all.finalize()));
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

//! Offline tooling. This is the only place where floating point is allowed.
//!
//! `convert` only uses exact IEEE operations (multiplication by a power of two and
//! round-to-nearest-even), so it is reproducible, but the engine never relies on that:
//! the identity of a model is the hash of the converted file.
//!
//! `compare` runs a plain float forward pass next to the integer engine to measure how
//! much the fixed-point arithmetic changes the model. It is a diagnostic and its float
//! side is not reproducible across machines.

use detra::generate::{generate, GenParams};
use detra::model::{Attention, Layer, Mixer, Model, Tensor, Vocab};
use std::sync::Arc;
use detra::sample::{argmax, SampleParams};
use detra::sha256::{hex, sha256};
use std::process::ExitCode;

const USAGE: &str = "\
Usage:
  detra-convert convert <llama2c-model.bin> <tokenizer.bin> <out.dllm>
  detra-convert compare <llama2c-model.bin> <model.dllm> <prompt> [max-tokens]";

const BOS_ID: u32 = 1;
const EOS_ID: u32 = 2;
const BYTE_BASE: u32 = 3;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let strs: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match strs.as_slice() {
        ["convert", model, tokenizer, out] => convert(model, tokenizer, out),
        ["compare", ckpt, model, prompt] => compare(ckpt, model, prompt, 100),
        ["compare", ckpt, model, prompt, n] => match n.parse() {
            Ok(n) => compare(ckpt, model, prompt, n),
            Err(_) => Err(format!("invalid max-tokens: {n}")),
        },
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// A llama2.c checkpoint: 7 x i32 header, then f32 tensors grouped by kind.
struct Checkpoint {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    vocab_size: usize,
    seq_len: usize,
    tok_emb: Vec<f32>,
    rms_att: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    rms_ffn: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    rms_final: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    wcls: Option<Vec<f32>>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.bytes.len() - self.pos {
            return Err("file is truncated".into());
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn i32(&mut self) -> Result<i32, String> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.i32()? as u32))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let b = self.take(n * 4)?;
        Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }
}

fn read_checkpoint(path: &str) -> Result<Checkpoint, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let mut r = Reader { bytes: &bytes, pos: 0 };
    let mut header = [0i32; 7];
    for h in &mut header {
        *h = r.i32()?;
    }
    let [dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab, seq_len] = header;
    if header.iter().enumerate().any(|(i, &v)| if i == 5 { v == 0 } else { v <= 0 }) {
        return Err(format!("{path}: implausible header {header:?}"));
    }
    if n_kv_heads != n_heads {
        return Err("checkpoints with n_kv_heads != n_heads are not supported".into());
    }
    let shared = vocab > 0;
    let (d, h, l, v, s) = (
        dim as usize,
        hidden_dim as usize,
        n_layers as usize,
        vocab.unsigned_abs() as usize,
        seq_len as usize,
    );
    let half_head = d / n_heads as usize / 2;
    let ckpt = Checkpoint {
        dim: d,
        hidden_dim: h,
        n_layers: l,
        n_heads: n_heads as usize,
        vocab_size: v,
        seq_len: s,
        tok_emb: r.f32s(v * d)?,
        rms_att: r.f32s(l * d)?,
        wq: r.f32s(l * d * d)?,
        wk: r.f32s(l * d * d)?,
        wv: r.f32s(l * d * d)?,
        wo: r.f32s(l * d * d)?,
        rms_ffn: r.f32s(l * d)?,
        w1: r.f32s(l * h * d)?,
        w2: r.f32s(l * d * h)?,
        w3: r.f32s(l * h * d)?,
        rms_final: r.f32s(d)?,
        rope_cos: r.f32s(s * half_head)?,
        rope_sin: r.f32s(s * half_head)?,
        wcls: if shared { None } else { Some(r.f32s(v * d)?) },
    };
    if r.pos != bytes.len() {
        return Err(format!("{path}: {} unexpected trailing bytes", bytes.len() - r.pos));
    }
    Ok(ckpt)
}

/// `round_ties_even(x * 2^frac_bits)`, rejecting anything that does not fit in `i32`.
fn to_fixed(name: &str, values: &[f32], frac_bits: i32) -> Result<Vec<i32>, String> {
    let scale = 2f64.powi(frac_bits);
    values
        .iter()
        .map(|&x| {
            let y = (x as f64 * scale).round_ties_even();
            if y.is_finite() && y >= i32::MIN as f64 && y <= i32::MAX as f64 {
                Ok(y as i32)
            } else {
                Err(format!("{name}: value {x} does not fit in Q{frac_bits}"))
            }
        })
        .collect()
}

fn read_tokenizer(path: &str, vocab_size: usize) -> Result<(Vec<Vec<u8>>, Vec<u32>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let mut r = Reader { bytes: &bytes, pos: 0 };
    let _max_token_length = r.i32()?;
    let mut tokens = Vec::new();
    let mut scores = Vec::new();
    for _ in 0..vocab_size {
        let score = r.f32()?;
        if score.is_nan() {
            return Err(format!("{path}: token score is NaN"));
        }
        let len = r.i32()?;
        if len < 0 {
            return Err(format!("{path}: negative token length"));
        }
        scores.push(score);
        tokens.push(r.take(len as usize)?.to_vec());
    }
    if r.pos != bytes.len() {
        return Err(format!("{path}: {} unexpected trailing bytes", bytes.len() - r.pos));
    }
    // The byte tokens are stored as the text "<0xHH>"; replace them by the raw byte.
    for b in 0..256usize {
        let id = BYTE_BASE as usize + b;
        let expected = format!("<0x{b:02X}>");
        if tokens.get(id).map(Vec::as_slice) != Some(expected.as_bytes()) {
            return Err(format!("{path}: token {id} is not the byte token {expected}"));
        }
        tokens[id] = vec![b as u8];
    }
    // Scores only matter by their order: replace them by dense integer ranks.
    let mut order: Vec<usize> = (0..vocab_size).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    let mut ranks = vec![0u32; vocab_size];
    let mut rank = 0u32;
    for (i, &id) in order.iter().enumerate() {
        if i > 0 && scores[id] != scores[order[i - 1]] {
            rank += 1;
        }
        ranks[id] = rank;
    }
    Ok((tokens, ranks))
}

fn convert(ckpt_path: &str, tokenizer_path: &str, out_path: &str) -> Result<(), String> {
    let c = read_checkpoint(ckpt_path)?;
    let (tokens, ranks) = read_tokenizer(tokenizer_path, c.vocab_size)?;
    let (d, h) = (c.dim, c.hidden_dim);
    let q24 = |name: &str, v: &[f32]| to_fixed(name, v, 24);
    // Matrices are stored packed when that is lossless (rare for float32 checkpoints).
    let matrix = |name: &str, v: &[f32]| to_fixed(name, v, 24).map(Tensor::pack);
    let mut layers = Vec::new();
    for l in 0..c.n_layers {
        layers.push(Layer {
            norm1: q24("rms_att", &c.rms_att[l * d..(l + 1) * d])?,
            mixer: Mixer::Attention(Attention {
                wq: matrix("wq", &c.wq[l * d * d..(l + 1) * d * d])?,
                wk: matrix("wk", &c.wk[l * d * d..(l + 1) * d * d])?,
                wv: matrix("wv", &c.wv[l * d * d..(l + 1) * d * d])?,
                wo: matrix("wo", &c.wo[l * d * d..(l + 1) * d * d])?,
                q_norm: Vec::new(),
                k_norm: Vec::new(),
            }),
            norm2: q24("rms_ffn", &c.rms_ffn[l * d..(l + 1) * d])?,
            w_gate: matrix("w1", &c.w1[l * h * d..(l + 1) * h * d])?,
            w_down: matrix("w2", &c.w2[l * d * h..(l + 1) * d * h])?,
            w_up: matrix("w3", &c.w3[l * h * d..(l + 1) * h * d])?,
        });
    }
    let model = Model {
        dim: d,
        hidden_dim: h,
        vocab_size: c.vocab_size,
        seq_len: c.seq_len,
        rms_eps_q32: 42_950, // ceil(1e-5 * 2^32)
        n_heads: c.n_heads,
        n_kv_heads: c.n_heads,
        head_dim: d / c.n_heads,
        rotary_dim: d / c.n_heads,
        rope_halves: false,
        qk_norm: false,
        out_gate: false,
        lin_heads: 0,
        lin_k_dim: 0,
        lin_v_dim: 0,
        conv_kernel: 0,
        echo_prompt: true,
        trim_prompt: false,
        tok_emb: matrix("tok_emb", &c.tok_emb)?,
        layers,
        final_norm: q24("rms_final", &c.rms_final)?,
        rope_cos: to_fixed("rope_cos", &c.rope_cos, 30)?,
        rope_sin: to_fixed("rope_sin", &c.rope_sin, 30)?,
        wcls: c.wcls.as_deref().map(|w| matrix("wcls", w)).transpose()?,
        tokens,
        vocab: Vocab::Scored { byte_base: BYTE_BASE, ranks },
        prefix: vec![BOS_ID],
        suffix: Vec::new(),
        stop: vec![BOS_ID, EOS_ID],
        chat: None,
    };
    let bytes = model.to_bytes().map_err(|e| e.to_string())?;
    std::fs::write(out_path, &bytes).map_err(|e| format!("{out_path}: {e}"))?;
    println!("{}  {out_path}", hex(&sha256(&bytes)));
    Ok(())
}

fn f_matvec(out: &mut [f32], w: &[f32], x: &[f32]) {
    for (o, row) in out.iter_mut().zip(w.chunks_exact(x.len())) {
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

fn f_rmsnorm(out: &mut [f32], x: &[f32], w: &[f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5;
    let inv = 1.0 / ss.sqrt();
    for ((o, xv), wv) in out.iter_mut().zip(x).zip(w) {
        *o = wv * (inv * xv);
    }
}

/// Float forward pass over a fixed token sequence, returning the logits of every step.
fn float_logits(c: &Checkpoint, tokens: &[u32]) -> Vec<Vec<f32>> {
    let (d, h, hs) = (c.dim, c.hidden_dim, c.dim / c.n_heads);
    let half = hs / 2;
    let mut keys = vec![0f32; c.n_layers * c.seq_len * d];
    let mut vals = vec![0f32; c.n_layers * c.seq_len * d];
    let (mut x, mut xb, mut xb2, mut q) = (vec![0f32; d], vec![0f32; d], vec![0f32; d], vec![0f32; d]);
    let (mut hb, mut hb2) = (vec![0f32; h], vec![0f32; h]);
    let mut att = vec![0f32; c.seq_len];
    let mut all = Vec::new();
    for (pos, &token) in tokens.iter().enumerate() {
        let t = token as usize;
        x.copy_from_slice(&c.tok_emb[t * d..(t + 1) * d]);
        for l in 0..c.n_layers {
            let base = l * c.seq_len * d;
            let cur = base + pos * d;
            f_rmsnorm(&mut xb, &x, &c.rms_att[l * d..(l + 1) * d]);
            f_matvec(&mut q, &c.wq[l * d * d..(l + 1) * d * d], &xb);
            f_matvec(&mut keys[cur..cur + d], &c.wk[l * d * d..(l + 1) * d * d], &xb);
            f_matvec(&mut vals[cur..cur + d], &c.wv[l * d * d..(l + 1) * d * d], &xb);
            for i in (0..d).step_by(2) {
                let fc = c.rope_cos[pos * half + (i % hs) / 2];
                let fs = c.rope_sin[pos * half + (i % hs) / 2];
                let (q0, q1) = (q[i], q[i + 1]);
                q[i] = q0 * fc - q1 * fs;
                q[i + 1] = q0 * fs + q1 * fc;
                let (k0, k1) = (keys[cur + i], keys[cur + i + 1]);
                keys[cur + i] = k0 * fc - k1 * fs;
                keys[cur + i + 1] = k0 * fs + k1 * fc;
            }
            for head in 0..c.n_heads {
                let o = head * hs;
                let a = &mut att[..=pos];
                for (t, av) in a.iter_mut().enumerate() {
                    let k = &keys[base + t * d + o..][..hs];
                    *av = q[o..o + hs].iter().zip(k).map(|(a, b)| a * b).sum::<f32>()
                        / (hs as f32).sqrt();
                }
                let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for av in a.iter_mut() {
                    *av = (*av - max).exp();
                    sum += *av;
                }
                for i in 0..hs {
                    xb[o + i] = a
                        .iter()
                        .enumerate()
                        .map(|(t, av)| av / sum * vals[base + t * d + o + i])
                        .sum();
                }
            }
            f_matvec(&mut xb2, &c.wo[l * d * d..(l + 1) * d * d], &xb);
            for (xv, r) in x.iter_mut().zip(&xb2) {
                *xv += r;
            }
            f_rmsnorm(&mut xb, &x, &c.rms_ffn[l * d..(l + 1) * d]);
            f_matvec(&mut hb, &c.w1[l * h * d..(l + 1) * h * d], &xb);
            f_matvec(&mut hb2, &c.w3[l * h * d..(l + 1) * h * d], &xb);
            for (a, b) in hb.iter_mut().zip(&hb2) {
                *a = *a * (1.0 / (1.0 + (-*a).exp())) * b;
            }
            f_matvec(&mut xb, &c.w2[l * d * h..(l + 1) * d * h], &hb);
            for (xv, r) in x.iter_mut().zip(&xb) {
                *xv += r;
            }
        }
        f_rmsnorm(&mut xb, &x, &c.rms_final);
        let mut logits = vec![0f32; c.vocab_size];
        f_matvec(&mut logits, c.wcls.as_deref().unwrap_or(&c.tok_emb), &xb);
        all.push(logits);
    }
    all
}

fn compare(ckpt_path: &str, model_path: &str, prompt: &str, max_new_tokens: usize) -> Result<(), String> {
    let c = read_checkpoint(ckpt_path)?;
    let bytes = std::fs::read(model_path).map_err(|e| format!("{model_path}: {e}"))?;
    let model = Arc::new(Model::from_bytes(&bytes).map_err(|e| e.to_string())?);
    let params = GenParams { max_new_tokens, sample: SampleParams::GREEDY, threads: 1 };
    // The engine only computes logits from the last prompt position onward.
    let mut int_logits: Vec<Vec<i32>> = Vec::new();
    let out = generate(&model, prompt.as_bytes(), &params, |step| {
        if !step.logits.is_empty() {
            int_logits.push(step.logits.to_vec());
        }
    })
    .map_err(|e| e.to_string())?;
    let first = out.generated_from - 1;
    let float = float_logits(&c, &out.tokens[..first + int_logits.len()]).split_off(first);

    let mut agree = 0;
    let mut max_diff = 0f64;
    let mut sum_diff = 0f64;
    let mut count = 0f64;
    for (il, fl) in int_logits.iter().zip(&float) {
        let f_best = fl
            .iter()
            .enumerate()
            .fold(0, |best, (i, &v)| if v > fl[best] { i } else { best });
        if argmax(il) as usize == f_best {
            agree += 1;
        }
        for (&i, &f) in il.iter().zip(fl) {
            let diff = (i as f64 / 1048576.0 - f as f64).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
            count += 1.0;
        }
    }
    println!("steps compared         {}", int_logits.len());
    println!("argmax agreement       {agree}/{}", int_logits.len());
    println!("max |logit difference| {max_diff:.6}");
    println!("mean |logit difference| {:.6}", sum_diff / count);
    Ok(())
}

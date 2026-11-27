//! Conversion of a Hugging Face Qwen3.5 checkpoint (the language model only) to the
//! DLLM format, in integer arithmetic only, so that the converted file is the same on
//! every machine.
//!
//! A checkpoint is a `config.json`, a `tokenizer.json` and a single `.safetensors`
//! file. The vision tower and the multi-token-prediction head are ignored.
//!
//! - Weights are scaled by 2^24 and rounded to the nearest integer, ties to even.
//!   Matrices are stored packed when every value allows it.
//! - `exp(A_log)` and the RoPE tables are computed with [`crate::exact`] and correctly
//!   rounded. The RoPE angle of position `p` and frequency `j` is the double-precision
//!   product `p * f_j`, where `f_j = theta^(-2j / rotary_dim)` rounded to a double
//!   (the first models were converted in double precision, and this reproduces them).
//! - The tokenizer's character classes come from [`crate::unicode_classes`], and the
//!   template token ids are encoded with the engine's own tokenizer.

use crate::exact;
use crate::json::{self, Value};
use crate::model::{Attention, Chat, Delta, Layer, Mixer, Model, Tensor, Turn, Vocab};
use crate::tokenizer::Tokenizer;
use crate::unicode_classes::CLASSES;
use crate::{err, Error, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Context length of converted models (or less, if the model supports less).
pub const SEQ_LEN: usize = 4096;

const PREFIX: &str = "<|im_start|>user\n";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const STOP: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];
/// Chat template (`chat_template.jinja` without tools and thinking): per role the text
/// before and after the message content, then the generation prompt.
const CHAT: [(&str, &str); 3] = [
    ("<|im_start|>system\n", "<|im_end|>\n"),
    ("<|im_start|>user\n", "<|im_end|>\n"),
    ("<|im_start|>assistant\n", "<|im_end|>\n"),
];
const GENERATION: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))
}

fn member<'a>(v: &'a Value, name: &str) -> Result<&'a Value> {
    v.get(name).ok_or_else(|| Error(format!("`{name}` is missing")))
}

fn int(v: &Value, name: &str) -> Result<usize> {
    match member(v, name)? {
        Value::Number(n) if !n.is_empty() && n.len() <= 12 && n.bytes().all(|b| b.is_ascii_digit()) => {
            n.parse().map_err(|_| Error(format!("`{name}` is out of range")))
        }
        _ => err(format!("`{name}` is not a non-negative integer")),
    }
}

fn flag(v: &Value, name: &str) -> Result<bool> {
    match member(v, name)? {
        Value::Bool(b) => Ok(*b),
        _ => err(format!("`{name}` is not a boolean")),
    }
}

/// A non-negative JSON number as an exact fraction `(numerator, denominator)`.
fn fraction(v: &Value, name: &str) -> Result<(u128, u128)> {
    let bad = || Error(format!("`{name}` is not a supported number"));
    let Value::Number(text) = member(v, name)? else { return Err(bad()) };
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(i) => (&text[..i], text[i + 1..].parse::<i32>().map_err(|_| bad())?),
        None => (text.as_str(), 0),
    };
    let (whole, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{frac}");
    if digits.len() > 30 || !digits.bytes().all(|b| b.is_ascii_digit()) || !(-30..=30).contains(&exponent) {
        return Err(bad());
    }
    let n: u128 = digits.parse().map_err(|_| bad())?;
    let exponent = exponent - frac.len() as i32;
    let pow = |e: i32| 10u128.checked_pow(e.unsigned_abs()).ok_or_else(bad);
    if exponent >= 0 {
        Ok((n.checked_mul(pow(exponent)?).ok_or_else(bad)?, 1))
    } else {
        Ok((n, pow(exponent)?))
    }
}

/// The value of the IEEE single with bits `bits` times 2^`frac`, rounded to the
/// nearest integer (ties to even). `None` if it is not finite or not below 2^62.
fn scaled_f32(bits: u32, frac: u32) -> Option<i64> {
    let exp = (bits >> 23) & 0xFF;
    let man = (bits & 0x7F_FFFF) as u64;
    if exp == 0xFF {
        return None;
    }
    let (m, e) = if exp == 0 { (man, -149) } else { (man | 0x80_0000, exp as i32 - 150) };
    let shift = e + frac as i32;
    let magnitude = if m == 0 {
        0
    } else if shift >= 0 {
        if shift > 37 {
            return None;
        }
        m << shift
    } else if shift < -40 {
        0
    } else {
        let d = (-shift) as u32;
        let (q, rem, half) = (m >> d, m & ((1 << d) - 1), 1u64 << (d - 1));
        q + (rem > half || (rem == half && q & 1 == 1)) as u64
    };
    Some(if bits >> 31 == 1 { -(magnitude as i64) } else { magnitude as i64 })
}

/// The value of the IEEE single with bits `bits` in Q96, truncated. `None` if it is not
/// finite or its magnitude is 2^24 or more.
fn f32_to_q96(bits: u32) -> Option<i128> {
    let exp = (bits >> 23) & 0xFF;
    let man = (bits & 0x7F_FFFF) as i128;
    if exp == 0xFF || exp >= 127 + 24 {
        return None;
    }
    let (m, e) = if exp == 0 { (man, -149) } else { (man | 0x80_0000, exp as i32 - 150) };
    let shift = e + exact::FRAC as i32;
    let magnitude = if shift >= 0 { m << shift } else if shift <= -64 { 0 } else { m >> -shift };
    Some(if bits >> 31 == 1 { -magnitude } else { magnitude })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dtype {
    Bf16,
    Single,
}

/// One tensor of the checkpoint, as IEEE single bits.
struct Raw {
    name: String,
    shape: Vec<usize>,
    dtype: Dtype,
    bytes: Vec<u8>,
}

impl Raw {
    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    fn bits(&self) -> impl Iterator<Item = u32> + '_ {
        let (halves, singles) = match self.dtype {
            Dtype::Bf16 => (self.bytes.as_chunks::<2>().0, [].as_slice()),
            Dtype::Single => ([].as_slice(), self.bytes.as_chunks::<4>().0),
        };
        let halves = halves.iter().map(|c| (u16::from_le_bytes(*c) as u32) << 16);
        halves.chain(singles.iter().map(|c| u32::from_le_bytes(*c)))
    }

    fn expect(&self, shape: &[usize]) -> Result<()> {
        if self.shape == shape {
            Ok(())
        } else {
            err(format!("{}: shape {:?}, expected {:?}", self.name, self.shape, shape))
        }
    }

    /// The same values viewed with another shape of the same size.
    fn reshaped(mut self, shape: &[usize]) -> Result<Raw> {
        if self.len() != shape.iter().product::<usize>() {
            return err(format!("{}: shape {:?} cannot be viewed as {:?}", self.name, self.shape, shape));
        }
        self.shape = shape.to_vec();
        Ok(self)
    }

    /// Values times 2^`frac` plus `offset`, which must stay below 2^31 in magnitude.
    fn fixed(&self, shape: &[usize], frac: u32, offset: i64) -> Result<Vec<i32>> {
        self.expect(shape)?;
        self.bits()
            .map(|b| {
                scaled_f32(b, frac)
                    .map(|v| v + offset)
                    .filter(|v| v.unsigned_abs() < 1 << 31)
                    .map(|v| v as i32)
                    .ok_or_else(|| Error(format!("{}: value does not fit in Q{frac}", self.name)))
            })
            .collect()
    }

    /// A Q24 matrix, packed if every value can be.
    fn matrix(&self, shape: &[usize]) -> Result<Tensor> {
        self.expect(shape)?;
        let values = || self.bits().map(|b| scaled_f32(b, 24).filter(|v| v.unsigned_abs() < 1 << 31).map(|v| v as i32));
        let mut packable = true;
        for v in values() {
            let Some(v) = v else { return err(format!("{}: value does not fit in Q24", self.name)) };
            packable &= Tensor::pack_value(v).is_some();
        }
        Ok(if packable {
            Tensor::Packed(values().map(|v| Tensor::pack_value(v.unwrap_or(0)).unwrap_or(0)).collect())
        } else {
            Tensor::Wide(values().map(|v| v.unwrap_or(0)).collect())
        })
    }
}

struct SafeTensors {
    file: File,
    base: u64,
    size: u64,
    header: Value,
}

impl SafeTensors {
    fn open(path: &Path) -> Result<SafeTensors> {
        let fail = |e: std::io::Error| Error(format!("{}: {e}", path.display()));
        let mut file = File::open(path).map_err(fail)?;
        let size = file.metadata().map_err(fail)?.len();
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(fail)?;
        let len = u64::from_le_bytes(len);
        if len > 1 << 27 || 8 + len > size {
            return err(format!("{}: invalid safetensors header", path.display()));
        }
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header).map_err(fail)?;
        let header = json::parse(&header).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Ok(SafeTensors { file, base: 8 + len, size, header })
    }

    fn get(&mut self, name: &str) -> Result<Raw> {
        let fail = |what: &str| Error(format!("{name}: {what}"));
        let info = self.header.get(name).ok_or_else(|| fail("tensor is missing"))?;
        let dtype = match info.get("dtype").and_then(Value::as_str) {
            Some("BF16") => Dtype::Bf16,
            Some("F32") => Dtype::Single,
            other => return Err(fail(&format!("unsupported dtype {other:?}"))),
        };
        let Some(Value::Array(dims)) = info.get("shape") else { return Err(fail("invalid shape")) };
        let shape = dims
            .iter()
            .map(|d| match d {
                Value::Number(n) => n.parse::<u32>().map(|n| n as usize).map_err(|_| fail("invalid shape")),
                _ => Err(fail("invalid shape")),
            })
            .collect::<Result<Vec<usize>>>()?;
        let Some(Value::Array(offsets)) = info.get("data_offsets") else { return Err(fail("invalid offsets")) };
        let offset = |i: usize| match offsets.get(i) {
            Some(Value::Number(n)) => n.parse::<u64>().map_err(|_| fail("invalid offsets")),
            _ => Err(fail("invalid offsets")),
        };
        let (start, end) = (offset(0)?, offset(1)?);
        let width: u64 = if dtype == Dtype::Bf16 { 2 } else { 4 };
        let size = shape.iter().try_fold(width, |a, &d| a.checked_mul(d as u64)).ok_or_else(|| fail("too large"))?;
        if shape.is_empty() || end < start || end - start != size || end > self.size - self.base {
            return Err(fail("data does not match its shape"));
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact((end - start) as usize).map_err(|_| fail("does not fit in memory"))?;
        self.file.seek(SeekFrom::Start(self.base + start)).map_err(|e| fail(&e.to_string()))?;
        (&mut self.file).take(end - start).read_to_end(&mut bytes).map_err(|e| fail(&e.to_string()))?;
        if bytes.len() as u64 != end - start {
            return Err(fail("file is truncated"));
        }
        Ok(Raw { name: name.to_string(), shape, dtype, bytes })
    }
}

/// Inverse of the GPT-2 byte-to-character table of byte-level BPE vocabularies.
fn byte_chars() -> [char; 256] {
    let direct = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut chars = ['\0'; 256];
    let mut next = 256;
    for b in 0..256u32 {
        chars[b as usize] = if direct(b) {
            char::from_u32(b).unwrap_or('\0')
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap_or('\0')
        };
    }
    chars
}

/// RoPE tables `(cos, sin)` in Q30, `[seq_len, rotary_dim / 2]`.
fn rope_tables(theta: u32, rotary_dim: usize, seq_len: usize) -> Result<(Vec<i32>, Vec<i32>)> {
    let fail = || Error("RoPE table entry cannot be rounded reliably".into());
    let ln_theta = exact::ln(theta);
    let freqs = (0..rotary_dim / 2)
        .map(|j| {
            let x = -(2 * j as i128 * ln_theta) / rotary_dim as i128;
            exact::exp(x).and_then(exact::round_double).ok_or_else(fail)
        })
        .collect::<Result<Vec<_>>>()?;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for p in 0..seq_len {
        for &f in &freqs {
            let angle = exact::double_to_fixed(exact::mul_double(p as u64, f)).ok_or_else(fail)?;
            if angle > exact::ONE << 20 {
                return Err(fail());
            }
            let (c, s) = exact::cos_sin(angle);
            cos.push(exact::round_fixed(c, 30).ok_or_else(fail)? as i32);
            sin.push(exact::round_fixed(s, 30).ok_or_else(fail)? as i32);
        }
    }
    Ok((cos, sin))
}

/// `exp(A_log)` in Q24.
fn a_exp(raw: &Raw, heads: usize) -> Result<Vec<i32>> {
    raw.expect(&[heads])?;
    raw.bits()
        .map(|b| {
            let too_large = || Error(format!("{}: value does not fit in Q24", raw.name));
            let x = f32_to_q96(b).ok_or_else(too_large)?;
            // e^x for x < -1000 is far below 2^-25 and rounds to 0.
            let (m, k) = if x < -1000 * exact::ONE { (0, 0) } else { exact::exp(x).ok_or_else(too_large)? };
            if k > 24 {
                return Err(too_large());
            }
            let v = if k >= 0 { m << k } else { m >> (-k).min(127) };
            exact::round_fixed(v, 24)
                .filter(|v| v.unsigned_abs() < 1 << 31)
                .map(|v| v as i32)
                .ok_or_else(|| Error(format!("{}: value cannot be rounded reliably", raw.name)))
        })
        .collect()
}

/// The files of a checkpoint.
pub struct Checkpoint {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

impl Checkpoint {
    /// The checkpoint in a directory with `config.json`, `tokenizer.json` and
    /// `model.safetensors`.
    pub fn in_dir(dir: &Path) -> Checkpoint {
        Checkpoint {
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
            weights: dir.join("model.safetensors"),
        }
    }
}

/// Convert a checkpoint. `progress` is called with a short description of each step.
pub fn convert(files: &Checkpoint, mut progress: impl FnMut(&str)) -> Result<Model> {
    let config = json::parse(&read_file(&files.config)?)?;
    let cfg = member(&config, "text_config")?;
    let unsupported = |what: &str| Error(format!("unsupported model configuration: {what}"));
    if !flag(cfg, "tie_word_embeddings")? {
        return Err(unsupported("separate output embeddings"));
    }
    if flag(cfg, "attention_bias")? {
        return Err(unsupported("attention bias"));
    }
    if member(cfg, "hidden_act")?.as_str() != Some("silu") {
        return Err(unsupported("activation other than silu"));
    }
    let (dim, hidden, n_layers) = (int(cfg, "hidden_size")?, int(cfg, "intermediate_size")?, int(cfg, "num_hidden_layers")?);
    let (vocab_size, head_dim) = (int(cfg, "vocab_size")?, int(cfg, "head_dim")?);
    let (n_heads, n_kv) = (int(cfg, "num_attention_heads")?, int(cfg, "num_key_value_heads")?);
    let lin_heads = int(cfg, "linear_num_key_heads")?;
    if int(cfg, "linear_num_value_heads")? != lin_heads {
        return Err(unsupported("different numbers of key and value heads"));
    }
    let (kd, vd, kernel) =
        (int(cfg, "linear_key_head_dim")?, int(cfg, "linear_value_head_dim")?, int(cfg, "linear_conv_kernel_dim")?);
    // The engine's limits, checked before anything is allocated from these sizes.
    if [dim, hidden, n_layers, n_heads, n_kv, lin_heads, kd, vd, kernel].iter().any(|&n| n > 1 << 20)
        || vocab_size > 1 << 24
        || head_dim > 1 << 14
    {
        return Err(unsupported("dimension too large"));
    }
    let rope = member(cfg, "rope_parameters")?;
    let (num, den) = if rope.get("partial_rotary_factor").is_some() { fraction(rope, "partial_rotary_factor")? } else { (1, 1) };
    let rotary_dim = num
        .checked_mul(head_dim as u128)
        .and_then(|n| usize::try_from(n / den).ok())
        .ok_or_else(|| unsupported("partial_rotary_factor"))?;
    if rotary_dim < 2 || !rotary_dim.is_multiple_of(2) || rotary_dim > head_dim {
        return Err(unsupported("rotary dimension"));
    }
    let (num, den) = fraction(rope, "rope_theta")?;
    let theta = u32::try_from(num / den).ok().filter(|&t| num % den == 0 && t >= 2).ok_or_else(|| unsupported("rope_theta"))?;
    let (num, den) = fraction(cfg, "rms_norm_eps")?;
    // round(eps * 2^32), ties up. A tie needs eps to be an odd multiple of 2^-33.
    let eps = num.checked_mul(1 << 33).map(|n| (n / den).div_ceil(2));
    let rms_eps_q32 = eps.and_then(|e| u32::try_from(e).ok()).filter(|&e| e > 0).ok_or_else(|| unsupported("rms_norm_eps"))?;
    let Some(Value::Array(types)) = cfg.get("layer_types") else { return Err(unsupported("layer_types")) };
    let layer_types = types
        .iter()
        .map(|t| match t.as_str() {
            Some("full_attention") => Ok(0),
            Some("linear_attention") => Ok(1),
            _ => Err(unsupported("layer type")),
        })
        .collect::<Result<Vec<u32>>>()?;
    if layer_types.len() != n_layers {
        return Err(unsupported("layer_types does not match num_hidden_layers"));
    }
    let seq_len = SEQ_LEN.min(int(cfg, "max_position_embeddings")?);
    let (key_dim, value_dim) = (lin_heads * kd, lin_heads * vd);
    let conv_dim = 2 * key_dim + value_dim;
    let out_gate = flag(cfg, "attn_output_gate")?;

    let mut st = SafeTensors::open(&files.weights)?;
    let lm = "model.language_model.";
    progress("embeddings");
    let tok_emb = st.get(&format!("{lm}embed_tokens.weight"))?.matrix(&[vocab_size, dim])?;
    let norm = |st: &mut SafeTensors, name: &str, n: usize| st.get(name)?.fixed(&[n], 24, 1 << 24);
    let mut layers = Vec::new();
    for (i, &kind) in layer_types.iter().enumerate() {
        progress(&format!("layer {} of {n_layers}", i + 1));
        let p = format!("{lm}layers.{i}.");
        let norm1 = norm(&mut st, &format!("{p}input_layernorm.weight"), dim)?;
        let mixer = if kind == 0 {
            let a = format!("{p}self_attn.");
            Mixer::Attention(Attention {
                wq: st.get(&format!("{a}q_proj.weight"))?.matrix(&[n_heads * head_dim * (1 + out_gate as usize), dim])?,
                wk: st.get(&format!("{a}k_proj.weight"))?.matrix(&[n_kv * head_dim, dim])?,
                wv: st.get(&format!("{a}v_proj.weight"))?.matrix(&[n_kv * head_dim, dim])?,
                wo: st.get(&format!("{a}o_proj.weight"))?.matrix(&[dim, n_heads * head_dim])?,
                q_norm: norm(&mut st, &format!("{a}q_norm.weight"), head_dim)?,
                k_norm: norm(&mut st, &format!("{a}k_norm.weight"), head_dim)?,
            })
        } else {
            let g = format!("{p}linear_attn.");
            Mixer::Delta(Delta {
                w_qkv: st.get(&format!("{g}in_proj_qkv.weight"))?.matrix(&[conv_dim, dim])?,
                w_z: st.get(&format!("{g}in_proj_z.weight"))?.matrix(&[value_dim, dim])?,
                w_b: st.get(&format!("{g}in_proj_b.weight"))?.matrix(&[lin_heads, dim])?,
                w_a: st.get(&format!("{g}in_proj_a.weight"))?.matrix(&[lin_heads, dim])?,
                conv: st.get(&format!("{g}conv1d.weight"))?.reshaped(&[conv_dim, kernel])?.fixed(&[conv_dim, kernel], 24, 0)?,
                a_exp: a_exp(&st.get(&format!("{g}A_log"))?, lin_heads)?,
                dt_bias: st.get(&format!("{g}dt_bias"))?.fixed(&[lin_heads], 24, 0)?,
                gnorm: st.get(&format!("{g}norm.weight"))?.fixed(&[vd], 24, 0)?,
                w_out: st.get(&format!("{g}out_proj.weight"))?.matrix(&[dim, value_dim])?,
            })
        };
        layers.push(Layer {
            norm1,
            mixer,
            norm2: norm(&mut st, &format!("{p}post_attention_layernorm.weight"), dim)?,
            w_gate: st.get(&format!("{p}mlp.gate_proj.weight"))?.matrix(&[hidden, dim])?,
            w_up: st.get(&format!("{p}mlp.up_proj.weight"))?.matrix(&[hidden, dim])?,
            w_down: st.get(&format!("{p}mlp.down_proj.weight"))?.matrix(&[dim, hidden])?,
        });
    }
    let final_norm = norm(&mut st, &format!("{lm}norm.weight"), dim)?;
    drop(st);
    progress("RoPE tables");
    let (rope_cos, rope_sin) = rope_tables(theta, rotary_dim, seq_len)?;

    progress("tokenizer");
    let tok = json::parse(&read_file(&files.tokenizer)?)?;
    let tok_model = member(&tok, "model")?;
    let normalizer = tok.get("normalizer").map(Value::to_json);
    if member(tok_model, "type")?.as_str() != Some("BPE") || normalizer.as_deref() != Some(r#"{"type":"NFC"}"#) {
        return err("unsupported tokenizer");
    }
    let chars = byte_chars();
    let mut char_byte = BTreeMap::new();
    for (b, &c) in chars.iter().enumerate() {
        char_byte.insert(c, b as u8);
    }
    let Value::Object(vocab_entries) = member(tok_model, "vocab")? else { return err("tokenizer vocab is not an object") };
    let mut tokens = vec![Vec::new(); vocab_size];
    let mut special = vec![true; vocab_size];
    let mut vocab: BTreeMap<&str, u32> = BTreeMap::new();
    for (text, id) in vocab_entries {
        let id = match id {
            Value::Number(n) => n.parse::<u32>().ok().filter(|&id| (id as usize) < vocab_size),
            _ => None,
        }
        .ok_or_else(|| Error(format!("token {text:?} has an invalid id")))?;
        tokens[id as usize] =
            text.chars().map(|c| char_byte.get(&c).copied()).collect::<Option<Vec<u8>>>().ok_or_else(|| {
                Error(format!("token {text:?} is not byte-level"))
            })?;
        special[id as usize] = false;
        vocab.insert(text, id);
    }
    let Value::Array(added_entries) = member(&tok, "added_tokens")? else { return err("added_tokens is not a list") };
    let mut added: Vec<(Vec<u8>, u32)> = Vec::new();
    for a in added_entries {
        let id = int(a, "id")?;
        let content = member(a, "content")?.as_str().ok_or_else(|| Error("added token without content".into()))?;
        if id >= vocab_size {
            return err(format!("added token {content:?} is outside the vocabulary"));
        }
        let is_special = a.get("special").is_some() && flag(a, "special")?;
        tokens[id] = content.as_bytes().to_vec();
        special[id] = is_special;
        added.push((content.as_bytes().to_vec(), id as u32));
    }
    let Value::Array(merge_entries) = member(tok_model, "merges")? else { return err("merges is not a list") };
    let lookup = |text: &str| vocab.get(text).copied().ok_or_else(|| Error(format!("merge refers to unknown token {text:?}")));
    let mut merges = Vec::with_capacity(merge_entries.len());
    for m in merge_entries {
        let pair = match m {
            Value::String(s) => s.split_once(' ').filter(|(_, b)| !b.contains(' ')),
            Value::Array(items) => match items.as_slice() {
                [Value::String(a), Value::String(b)] => Some((a.as_str(), b.as_str())),
                _ => None,
            },
            _ => None,
        };
        let (a, b) = pair.ok_or_else(|| Error("invalid merge".into()))?;
        merges.push([lookup(a)?, lookup(b)?, lookup(&format!("{a}{b}"))?]);
    }
    let byte_tokens = chars.iter().map(|c| lookup(&c.to_string())).collect::<Result<Vec<u32>>>()?;

    let mut model = Model {
        dim,
        hidden_dim: hidden,
        vocab_size,
        seq_len,
        rms_eps_q32,
        n_heads,
        n_kv_heads: n_kv,
        head_dim,
        rotary_dim,
        rope_halves: true,
        qk_norm: true,
        out_gate,
        lin_heads,
        lin_k_dim: kd,
        lin_v_dim: vd,
        conv_kernel: kernel,
        echo_prompt: false,
        trim_prompt: true,
        tok_emb,
        layers,
        final_norm,
        rope_cos,
        rope_sin,
        wcls: None,
        tokens,
        vocab: Vocab::Bpe { special, merges, byte_tokens, classes: CLASSES.to_vec() },
        prefix: Vec::new(),
        suffix: Vec::new(),
        stop: Vec::new(),
        chat: None,
    };
    let (prefix, suffix, stop, chat) = {
        let tokenizer = Tokenizer::new(&model);
        let encode = |text: &str| encode_template(&tokenizer, &added, text);
        let stop = STOP
            .iter()
            .map(|t| added.iter().find(|(c, _)| c == t.as_bytes()).map(|&(_, id)| id))
            .collect::<Option<Vec<u32>>>()
            .ok_or_else(|| Error("stop token is missing".into()))?;
        let turn = |(open, close): (&str, &str)| Ok::<_, Error>(Turn { open: encode(open)?, close: encode(close)? });
        let chat = Chat { system: turn(CHAT[0])?, user: turn(CHAT[1])?, assistant: turn(CHAT[2])?, generation: encode(GENERATION)? };
        (encode(PREFIX)?, encode(SUFFIX)?, stop, chat)
    };
    model.prefix = prefix;
    model.suffix = suffix;
    model.stop = stop;
    model.chat = Some(chat);
    model.validate()?;
    Ok(model)
}

/// Token ids of template text: added tokens are matched first (longest at each
/// position), the text between them is encoded normally.
fn encode_template(tokenizer: &Tokenizer, added: &[(Vec<u8>, u32)], text: &str) -> Result<Vec<u32>> {
    let bytes = text.as_bytes();
    let (mut out, mut start, mut i) = (Vec::new(), 0, 0);
    while i < bytes.len() {
        let hit = added.iter().filter(|(c, _)| !c.is_empty() && bytes[i..].starts_with(c)).max_by_key(|(c, _)| c.len());
        match hit {
            Some((content, id)) => {
                out.extend(tokenizer.encode(&bytes[start..i])?);
                out.push(*id);
                i += content.len();
                start = i;
            }
            None => i += 1,
        }
    }
    out.extend(tokenizer.encode(&bytes[start..])?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_singles() {
        let one = 0x3F80_0000;
        assert_eq!(scaled_f32(one, 24), Some(1 << 24));
        assert_eq!(scaled_f32(one | 1 << 31, 24), Some(-(1 << 24)));
        assert_eq!(scaled_f32(0x8000_0000, 24), Some(0));
        // 1.5 * 2^-25 * 2^24 = 0.75 -> 1; 2^-25 * 2^24 = 0.5 -> 0 (ties to even);
        // 3 * 2^-25 * 2^24 = 1.5 -> 2.
        assert_eq!(scaled_f32(0x3340_0000, 24), Some(1));
        assert_eq!(scaled_f32(0x3300_0000, 24), Some(0));
        assert_eq!(scaled_f32(0x33C0_0000, 24), Some(2));
        assert_eq!(scaled_f32(0x0000_0001, 24), Some(0));
        assert_eq!(scaled_f32(0x7F80_0000, 24), None);
        assert_eq!(scaled_f32(0x7FC0_0000, 24), None);
        assert_eq!(scaled_f32(0x7F00_0000, 24), None);
        assert_eq!(f32_to_q96(one), Some(exact::ONE));
        assert_eq!(f32_to_q96(0xC000_0000), Some(-2 * exact::ONE));
        assert_eq!(f32_to_q96(0x4B80_0000), None);
    }

    #[test]
    fn parses_fractions() {
        let v = |text: &str| fraction(&json::parse(format!("{{\"x\": {text}}}").as_bytes()).unwrap(), "x");
        assert_eq!(v("1e-06"), Ok((1, 1_000_000)));
        assert_eq!(v("0.25"), Ok((25, 100)));
        assert_eq!(v("10000000"), Ok((10_000_000, 1)));
        assert_eq!(v("1.5E3"), Ok((1_500, 1)));
        assert!(v("-1").is_err());
        assert!(v("1e99").is_err());
    }

    #[test]
    fn maps_bytes_to_characters() {
        let chars = byte_chars();
        assert_eq!(chars[b'a' as usize], 'a');
        assert_eq!(chars[b' ' as usize], '\u{120}');
        assert_eq!(chars[b'\n' as usize], '\u{10A}');
        assert_eq!(chars[0xAD], '\u{143}');
        let mut sorted = chars.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 256);
    }

    #[test]
    fn rope_tables_start_at_the_identity() {
        let (cos, sin) = rope_tables(10_000, 8, 3).unwrap();
        assert_eq!(&cos[..4], &[1 << 30; 4]);
        assert_eq!(&sin[..4], &[0; 4]);
        // Position 1, frequency 0: angle 1.
        assert_eq!((cos[4], sin[4]), (580_145_183, 903_522_590));
    }
}

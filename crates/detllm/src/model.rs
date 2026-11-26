//! Model file format (`.dllm`, version 1). All integers are little-endian.
//!
//! ```text
//! magic        4 bytes  "DLLM"
//! header       11 x u32 version, dim, hidden_dim, n_layers, n_heads, vocab_size,
//!                       seq_len, bos_id, eos_id, byte_base, shared_classifier (0|1)
//! tok_emb      i32[vocab_size * dim]                 Q16
//! per layer:
//!   rms_att    i32[dim]                              Q16
//!   wq wk wv wo  i32[dim * dim] each, row-major      Q16
//!   rms_ffn    i32[dim]                              Q16
//!   w1         i32[hidden_dim * dim]                 Q16
//!   w2         i32[dim * hidden_dim]                 Q16
//!   w3         i32[hidden_dim * dim]                 Q16
//! rms_final    i32[dim]                              Q16
//! rope_cos     i32[seq_len * head_size / 2]          Q30
//! rope_sin     i32[seq_len * head_size / 2]          Q30
//! wcls         i32[vocab_size * dim]                 Q16, only if shared_classifier = 0
//! per token:   u32 rank, u32 len, u8[len] bytes
//! ```
//!
//! Token ids below `byte_base` are special tokens. Ids `byte_base .. byte_base + 256`
//! are the raw byte tokens 0x00..0xFF (their `bytes` field is that single byte).
//! All later ids are regular tokens; a larger `rank` wins a merge.
//!
//! The identity of a model is the SHA-256 of its file.

use crate::{err, Result};

pub const MAGIC: [u8; 4] = *b"DLLM";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    pub rms_att: Vec<i32>,
    pub wq: Vec<i32>,
    pub wk: Vec<i32>,
    pub wv: Vec<i32>,
    pub wo: Vec<i32>,
    pub rms_ffn: Vec<i32>,
    pub w1: Vec<i32>,
    pub w2: Vec<i32>,
    pub w3: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub bos_id: u32,
    pub eos_id: u32,
    pub byte_base: u32,
    pub tok_emb: Vec<i32>,
    pub layers: Vec<Layer>,
    pub rms_final: Vec<i32>,
    pub rope_cos: Vec<i32>,
    pub rope_sin: Vec<i32>,
    /// `None` means the classifier shares `tok_emb`.
    pub wcls: Option<Vec<i32>>,
    pub tokens: Vec<Vec<u8>>,
    pub ranks: Vec<u32>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.bytes.len() - self.pos {
            return err("model file is truncated");
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32s(&mut self, n: usize) -> Result<Vec<i32>> {
        let Some(len) = n.checked_mul(4) else {
            return err("tensor size overflows");
        };
        let b = self.take(len)?;
        Ok(b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect())
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i32s(out: &mut Vec<u8>, values: &[i32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

impl Model {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// First id of the regular (mergeable) tokens.
    pub fn first_regular(&self) -> u32 {
        self.byte_base + 256
    }

    pub fn classifier(&self) -> &[i32] {
        self.wcls.as_deref().unwrap_or(&self.tok_emb)
    }

    /// Check every structural invariant the engine relies on.
    pub fn validate(&self) -> Result<()> {
        let limits = [
            ("dim", self.dim, 1 << 16),
            ("hidden_dim", self.hidden_dim, 1 << 20),
            ("n_layers", self.n_layers, 1 << 10),
            ("n_heads", self.n_heads, 1 << 10),
            ("vocab_size", self.vocab_size, 1 << 24),
            ("seq_len", self.seq_len, 1 << 20),
        ];
        for (name, value, max) in limits {
            if value == 0 || value > max {
                return err(format!("{name} = {value} is outside 1..={max}"));
            }
        }
        if !self.dim.is_multiple_of(self.n_heads) {
            return err("dim is not a multiple of n_heads");
        }
        let hs = self.head_size();
        if !hs.is_multiple_of(2) {
            return err("head size is odd");
        }
        let (d, h, v) = (self.dim, self.hidden_dim, self.vocab_size);
        let mul = |a: usize, b: usize| a.checked_mul(b).ok_or(crate::Error("size overflow".into()));
        let check = |name: &str, len: usize, want: usize| {
            if len == want {
                Ok(())
            } else {
                err(format!("{name} has {len} elements, expected {want}"))
            }
        };
        check("tok_emb", self.tok_emb.len(), mul(v, d)?)?;
        check("layers", self.layers.len(), self.n_layers)?;
        for l in &self.layers {
            check("rms_att", l.rms_att.len(), d)?;
            check("wq", l.wq.len(), mul(d, d)?)?;
            check("wk", l.wk.len(), mul(d, d)?)?;
            check("wv", l.wv.len(), mul(d, d)?)?;
            check("wo", l.wo.len(), mul(d, d)?)?;
            check("rms_ffn", l.rms_ffn.len(), d)?;
            check("w1", l.w1.len(), mul(h, d)?)?;
            check("w2", l.w2.len(), mul(d, h)?)?;
            check("w3", l.w3.len(), mul(h, d)?)?;
        }
        check("rms_final", self.rms_final.len(), d)?;
        check("rope_cos", self.rope_cos.len(), mul(self.seq_len, hs / 2)?)?;
        check("rope_sin", self.rope_sin.len(), mul(self.seq_len, hs / 2)?)?;
        if let Some(w) = &self.wcls {
            check("wcls", w.len(), mul(v, d)?)?;
        }
        check("tokens", self.tokens.len(), v)?;
        check("ranks", self.ranks.len(), v)?;
        if self.bos_id as usize >= v || self.eos_id as usize >= v {
            return err("bos_id or eos_id is outside the vocabulary");
        }
        if self.bos_id >= self.byte_base || self.eos_id >= self.byte_base {
            return err("bos_id and eos_id must be special tokens (below byte_base)");
        }
        if self.byte_base as usize + 256 > v {
            return err("byte tokens do not fit in the vocabulary");
        }
        for b in 0..256usize {
            if self.tokens[self.byte_base as usize + b] != [b as u8] {
                return err(format!("byte token {b} does not hold its byte"));
            }
        }
        for t in &self.tokens {
            if t.len() > 1 << 16 {
                return err("token is longer than 65536 bytes");
            }
        }
        Ok(())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Model> {
        let mut r = Reader { bytes, pos: 0 };
        if r.take(4)? != MAGIC {
            return err("not a DLLM model file");
        }
        let version = r.u32()?;
        if version != VERSION {
            return err(format!("unsupported model version {version}"));
        }
        let dim = r.u32()? as usize;
        let hidden_dim = r.u32()? as usize;
        let n_layers = r.u32()? as usize;
        let n_heads = r.u32()? as usize;
        let vocab_size = r.u32()? as usize;
        let seq_len = r.u32()? as usize;
        let bos_id = r.u32()?;
        let eos_id = r.u32()?;
        let byte_base = r.u32()?;
        let shared = match r.u32()? {
            0 => false,
            1 => true,
            other => return err(format!("invalid shared_classifier flag {other}")),
        };
        // Reject absurd headers before sizing any allocation from them.
        for (name, value, max) in [
            ("dim", dim, 1usize << 16),
            ("hidden_dim", hidden_dim, 1 << 20),
            ("n_layers", n_layers, 1 << 10),
            ("n_heads", n_heads, 1 << 10),
            ("vocab_size", vocab_size, 1 << 24),
            ("seq_len", seq_len, 1 << 20),
        ] {
            if value == 0 || value > max {
                return err(format!("{name} = {value} is outside 1..={max}"));
            }
        }
        if !dim.is_multiple_of(n_heads) {
            return err("dim is not a multiple of n_heads");
        }
        let half_head = dim / n_heads / 2;
        let sz = |a: usize, b: usize| a.checked_mul(b).ok_or(crate::Error("size overflow".into()));

        let tok_emb = r.i32s(sz(vocab_size, dim)?)?;
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            layers.push(Layer {
                rms_att: r.i32s(dim)?,
                wq: r.i32s(sz(dim, dim)?)?,
                wk: r.i32s(sz(dim, dim)?)?,
                wv: r.i32s(sz(dim, dim)?)?,
                wo: r.i32s(sz(dim, dim)?)?,
                rms_ffn: r.i32s(dim)?,
                w1: r.i32s(sz(hidden_dim, dim)?)?,
                w2: r.i32s(sz(dim, hidden_dim)?)?,
                w3: r.i32s(sz(hidden_dim, dim)?)?,
            });
        }
        let rms_final = r.i32s(dim)?;
        let rope_cos = r.i32s(sz(seq_len, half_head)?)?;
        let rope_sin = r.i32s(sz(seq_len, half_head)?)?;
        let wcls = if shared { None } else { Some(r.i32s(sz(vocab_size, dim)?)?) };
        let mut tokens = Vec::new();
        let mut ranks = Vec::new();
        for _ in 0..vocab_size {
            ranks.push(r.u32()?);
            let len = r.u32()? as usize;
            tokens.push(r.take(len)?.to_vec());
        }
        if r.pos != bytes.len() {
            return err("trailing bytes after the end of the model");
        }
        let model = Model {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            vocab_size,
            seq_len,
            bos_id,
            eos_id,
            byte_base,
            tok_emb,
            layers,
            rms_final,
            rope_cos,
            rope_sin,
            wcls,
            tokens,
            ranks,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        for v in [
            VERSION,
            self.dim as u32,
            self.hidden_dim as u32,
            self.n_layers as u32,
            self.n_heads as u32,
            self.vocab_size as u32,
            self.seq_len as u32,
            self.bos_id,
            self.eos_id,
            self.byte_base,
            self.wcls.is_none() as u32,
        ] {
            put_u32(&mut out, v);
        }
        put_i32s(&mut out, &self.tok_emb);
        for l in &self.layers {
            for t in [&l.rms_att, &l.wq, &l.wk, &l.wv, &l.wo, &l.rms_ffn, &l.w1, &l.w2, &l.w3] {
                put_i32s(&mut out, t);
            }
        }
        put_i32s(&mut out, &self.rms_final);
        put_i32s(&mut out, &self.rope_cos);
        put_i32s(&mut out, &self.rope_sin);
        if let Some(w) = &self.wcls {
            put_i32s(&mut out, w);
        }
        for (token, rank) in self.tokens.iter().zip(&self.ranks) {
            put_u32(&mut out, *rank);
            put_u32(&mut out, token.len() as u32);
            out.extend_from_slice(token);
        }
        Ok(out)
    }
}

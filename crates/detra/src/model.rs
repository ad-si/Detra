//! Model file format (`.dllm`, version 5). Integers are little-endian `u32`/`i32`
//! unless noted.
//!
//! ```text
//! magic "DLLM"
//! header (u32 each):
//!   version, dim, hidden_dim, n_layers, vocab_size, seq_len, rms_eps_q32,
//!   n_heads, n_kv_heads, head_dim, rotary_dim, rope_halves, qk_norm, out_gate,
//!   lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel,
//!   shared_classifier, tokenizer_kind, echo_prompt, trim_prompt
//! layer_types   u32[n_layers]            0 = attention, 1 = gated delta
//! tok_emb       [vocab_size, dim]        Q24
//! per layer:
//!   norm1       [dim]                    Q24
//!   attention:  wq [q_rows, dim], wk [kv_rows, dim], wv [kv_rows, dim],
//!               wo [dim, n_heads * head_dim]; if qk_norm: q_norm, k_norm [head_dim]
//!   delta:      w_qkv [2K + V, dim], w_z [V, dim], w_b, w_a [lin_v_heads, dim],
//!               conv [2K + V, conv_kernel], a_exp, dt_bias [lin_v_heads],
//!               gnorm [lin_v_dim], w_out [dim, V]
//!   norm2       [dim]
//!   w_gate, w_up [hidden_dim, dim], w_down [dim, hidden_dim]
//! final_norm    [dim]
//! rope_cos, rope_sin  [seq_len, rotary_dim / 2]      Q30, each within [-2^30, 2^30]
//! wcls          [vocab_size, dim]        only if shared_classifier = 0
//! tokenizer     (see below)
//! prefix, suffix, stop: each `u32 count` followed by that many token ids
//! has_chat      u32; if 1, seven more token lists: system open, system close,
//!               user open, user close, assistant open, assistant close, generation
//! ```
//!
//! All tensors hold Q24 values unless noted, matrices row-major `[rows, cols]`.
//! Vectors (norm weights, `conv`, `a_exp`, `dt_bias`, RoPE tables) are plain `i32`.
//! Every matrix (`tok_emb`, `wcls` and all `w*` tensors) starts with a `u32` encoding:
//! 0 = `i32` values; 1 = packed `i16` values, where `p` stands for the value
//! `(p >> 5) << (p & 31)` (arithmetic right shift, then a left shift in 32 bits).
//! Packing is lossless for the values it is used for and only saves space and time:
//! the two encodings of the same values give the same results.
//! `q_rows = n_heads * head_dim * (1 + out_gate)`, `kv_rows = n_kv_heads * head_dim`,
//! `K = lin_k_heads * lin_k_dim`, `V = lin_v_heads * lin_v_dim`. Value head `h` of a gated
//! delta layer uses query and key head `h / (lin_v_heads / lin_k_heads)`.
//! Norm weights are the effective multipliers (any `1 + w` is already applied).
//!
//! Version 4 files are read as well: they have no `lin_v_heads` field, which then equals
//! `lin_k_heads`.
//!
//! Tokenizer kind 0 (scored merges): `u32 byte_base`, then per token `u32 rank,
//! u32 len, bytes`. Kind 1 (byte-level BPE): per token `u32 special, u32 len, bytes`;
//! `u32 n_merges` and per merge `left, right, result` in priority order;
//! `u32[256]` byte token ids; `u32 n_ranges` and per range `first, last, class` with
//! class 1 = letter, 2 = mark, 3 = number, 4 = whitespace.
//!
//! A prompt is run as `prefix ++ encode(text) ++ suffix`, where leading and trailing
//! whitespace is removed from `text` first if `trim_prompt` is set; generation ends at
//! any `stop` token. A conversation is run as `open ++ encode(content) ++ close` for
//! every message with the lists of its role, followed by `generation`; the content is
//! trimmed in the same way. The identity of a model is the SHA-256 of its file.

use crate::{err, Error, Result};
use std::io::{Read, Write};

pub const MAGIC: [u8; 4] = *b"DLLM";
pub const VERSION: u32 = 5;
/// The previous version, which is still read.
const VERSION_4: u32 = 4;

/// A matrix of Q24 values in one of the two encodings of the file format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tensor {
    Wide(Vec<i32>),
    Packed(Vec<i16>),
}

impl Tensor {
    /// The value a packed element stands for.
    #[inline(always)]
    pub fn unpack(p: i16) -> i32 {
        ((p >> 5) as i32) << (p & 31)
    }

    /// The packed form of `v`, if it has one: an 11-bit signed mantissa times a power
    /// of two.
    pub(crate) fn pack_value(v: i32) -> Option<i16> {
        if v == 0 {
            return Some(0);
        }
        let shift = v.trailing_zeros();
        let mantissa = v >> shift;
        (-1024..=1023).contains(&mantissa).then_some(((mantissa << 5) | shift as i32) as i16)
    }

    /// Store `values` packed if every one of them can be, otherwise as they are.
    pub fn pack(values: Vec<i32>) -> Tensor {
        match values.iter().map(|&v| Tensor::pack_value(v)).collect::<Option<Vec<i16>>>() {
            Some(packed) => Tensor::Packed(packed),
            None => Tensor::Wide(values),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Tensor::Wide(v) => v.len(),
            Tensor::Packed(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> i32 {
        match self {
            Tensor::Wide(v) => v[i],
            Tensor::Packed(v) => Tensor::unpack(v[i]),
        }
    }
}

impl From<Vec<i32>> for Tensor {
    fn from(values: Vec<i32>) -> Tensor {
        Tensor::Wide(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attention {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub q_norm: Vec<i32>,
    pub k_norm: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delta {
    pub w_qkv: Tensor,
    pub w_z: Tensor,
    pub w_b: Tensor,
    pub w_a: Tensor,
    pub conv: Vec<i32>,
    pub a_exp: Vec<i32>,
    pub dt_bias: Vec<i32>,
    pub gnorm: Vec<i32>,
    pub w_out: Tensor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mixer {
    Attention(Attention),
    Delta(Delta),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    pub norm1: Vec<i32>,
    pub mixer: Mixer,
    pub norm2: Vec<i32>,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
}

pub const CLASS_LETTER: u32 = 1;
pub const CLASS_MARK: u32 = 2;
pub const CLASS_NUMBER: u32 = 3;
pub const CLASS_SPACE: u32 = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Vocab {
    Scored { byte_base: u32, ranks: Vec<u32> },
    Bpe { special: Vec<bool>, merges: Vec<[u32; 3]>, byte_tokens: Vec<u32>, classes: Vec<[u32; 3]> },
}

/// The tokens around one message of a conversation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Turn {
    pub open: Vec<u32>,
    pub close: Vec<u32>,
}

/// Chat template: a conversation is `open ++ encode(content) ++ close` for every
/// message, followed by `generation`, after which the model writes the reply.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Chat {
    pub system: Turn,
    pub user: Turn,
    pub assistant: Turn,
    pub generation: Vec<u32>,
}

impl Chat {
    fn lists(&self) -> [&Vec<u32>; 7] {
        let Chat { system, user, assistant, generation } = self;
        [&system.open, &system.close, &user.open, &user.close, &assistant.open, &assistant.close, generation]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rms_eps_q32: u32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    /// RoPE pairs are `(i, i + rotary_dim / 2)` if true, `(2i, 2i + 1)` otherwise.
    pub rope_halves: bool,
    pub qk_norm: bool,
    pub out_gate: bool,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    pub echo_prompt: bool,
    pub trim_prompt: bool,
    pub tok_emb: Tensor,
    pub layers: Vec<Layer>,
    pub final_norm: Vec<i32>,
    pub rope_cos: Vec<i32>,
    pub rope_sin: Vec<i32>,
    /// `None` means the classifier shares `tok_emb`.
    pub wcls: Option<Tensor>,
    pub tokens: Vec<Vec<u8>>,
    pub vocab: Vocab,
    pub prefix: Vec<u32>,
    pub suffix: Vec<u32>,
    pub stop: Vec<u32>,
    /// `None` if the model cannot hold conversations.
    pub chat: Option<Chat>,
}

struct Reader<R: Read> {
    inner: R,
    buf: Vec<u8>,
}

impl<R: Read> Reader<R> {
    fn fill(&mut self, out: &mut [u8]) -> Result<()> {
        self.inner.read_exact(out).map_err(|e| Error(format!("cannot read model: {e}")))
    }

    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.fill(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        self.fill(&mut out)?;
        Ok(out)
    }

    fn i32s(&mut self, n: usize) -> Result<Vec<i32>> {
        const CHUNK: usize = 1 << 18;
        let mut out = Vec::new();
        out.try_reserve_exact(n).map_err(|_| Error("model does not fit in memory".into()))?;
        self.buf.resize(CHUNK * 4, 0);
        let mut buf = std::mem::take(&mut self.buf);
        let mut left = n;
        while left > 0 {
            let take = left.min(CHUNK);
            self.fill(&mut buf[..take * 4])?;
            out.extend(buf[..take * 4].as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)));
            left -= take;
        }
        self.buf = buf;
        Ok(out)
    }

    fn tensor(&mut self, n: usize) -> Result<Tensor> {
        match self.u32()? {
            0 => Ok(Tensor::Wide(self.i32s(n)?)),
            1 => {
                const CHUNK: usize = 1 << 18;
                let mut out = Vec::new();
                out.try_reserve_exact(n).map_err(|_| Error("model does not fit in memory".into()))?;
                let mut buf = vec![0u8; CHUNK * 2];
                let mut left = n;
                while left > 0 {
                    let take = left.min(CHUNK);
                    self.fill(&mut buf[..take * 2])?;
                    out.extend(buf[..take * 2].as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)));
                    left -= take;
                }
                Ok(Tensor::Packed(out))
            }
            other => err(format!("unknown tensor encoding {other}")),
        }
    }

    fn ids(&mut self, limit: usize) -> Result<Vec<u32>> {
        let n = self.u32()? as usize;
        if n > limit {
            return err("token list is too long");
        }
        (0..n).map(|_| self.u32()).collect()
    }

    fn flag(&mut self, name: &str) -> Result<bool> {
        match self.u32()? {
            0 => Ok(false),
            1 => Ok(true),
            other => err(format!("invalid {name} flag {other}")),
        }
    }
}

/// Buffered little-endian output for [`Model::write`].
struct Writer<'w, W: Write> {
    inner: &'w mut W,
    buf: Vec<u8>,
}

impl<W: Write> Writer<'_, W> {
    const CAPACITY: usize = 1 << 20;

    fn flush_if_full(&mut self) -> Result<()> {
        if self.buf.len() >= Self::CAPACITY {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.write_all(&self.buf).map_err(|e| Error(format!("cannot write model: {e}")))?;
        self.buf.clear();
        Ok(())
    }

    fn bytes(&mut self, data: &[u8]) -> Result<()> {
        self.buf.extend_from_slice(data);
        self.flush_if_full()
    }

    fn u32(&mut self, v: u32) -> Result<()> {
        self.bytes(&v.to_le_bytes())
    }

    fn i32s(&mut self, values: &[i32]) -> Result<()> {
        for chunk in values.chunks(Self::CAPACITY / 4) {
            self.buf.extend(chunk.iter().flat_map(|v| v.to_le_bytes()));
            self.flush_if_full()?;
        }
        Ok(())
    }

    fn tensor(&mut self, tensor: &Tensor) -> Result<()> {
        match tensor {
            Tensor::Wide(values) => {
                self.u32(0)?;
                self.i32s(values)
            }
            Tensor::Packed(values) => {
                self.u32(1)?;
                for chunk in values.chunks(Self::CAPACITY / 2) {
                    self.buf.extend(chunk.iter().flat_map(|v| v.to_le_bytes()));
                    self.flush_if_full()?;
                }
                Ok(())
            }
        }
    }

    fn ids(&mut self, ids: &[u32]) -> Result<()> {
        self.u32(ids.len() as u32)?;
        for &id in ids {
            self.u32(id)?;
        }
        Ok(())
    }
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(Error("tensor size overflows".into()))
}

fn check(name: &str, len: usize, want: usize) -> Result<()> {
    if len == want {
        Ok(())
    } else {
        err(format!("{name} has {len} elements, expected {want}"))
    }
}

impl Model {
    pub fn q_rows(&self) -> usize {
        self.n_heads * self.head_dim * (1 + self.out_gate as usize)
    }

    pub fn kv_rows(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn att_out(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn lin_k(&self) -> usize {
        self.lin_k_heads * self.lin_k_dim
    }

    pub fn lin_v(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }

    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k() + self.lin_v()
    }

    pub fn classifier(&self) -> &Tensor {
        self.wcls.as_ref().unwrap_or(&self.tok_emb)
    }

    /// Check the header fields. Run before any size is derived from them.
    fn validate_header(&self, n_layers: usize) -> Result<()> {
        let limits = [
            ("dim", self.dim, 1, 1 << 16),
            ("hidden_dim", self.hidden_dim, 1, 1 << 20),
            ("n_layers", n_layers, 1, 1 << 10),
            ("vocab_size", self.vocab_size, 1, 1 << 24),
            ("seq_len", self.seq_len, 1, 1 << 20),
            ("n_heads", self.n_heads, 1, 1 << 10),
            ("n_kv_heads", self.n_kv_heads, 1, 1 << 10),
            ("head_dim", self.head_dim, 2, 1 << 14),
            ("rotary_dim", self.rotary_dim, 2, 1 << 14),
            ("lin_k_heads", self.lin_k_heads, 0, 1 << 10),
            ("lin_v_heads", self.lin_v_heads, 0, 1 << 10),
            ("lin_k_dim", self.lin_k_dim, 0, 1 << 12),
            ("lin_v_dim", self.lin_v_dim, 0, 1 << 12),
            ("conv_kernel", self.conv_kernel, 0, 64),
        ];
        for (name, value, min, max) in limits {
            if value < min || value > max {
                return err(format!("{name} = {value} is outside {min}..={max}"));
            }
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return err("n_heads is not a multiple of n_kv_heads");
        }
        // Also rejects value heads without key heads (only 0 is a multiple of 0).
        if !self.lin_v_heads.is_multiple_of(self.lin_k_heads) {
            return err("lin_v_heads is not a multiple of lin_k_heads");
        }
        if !self.rotary_dim.is_multiple_of(2) || self.rotary_dim > self.head_dim {
            return err("rotary_dim must be even and at most head_dim");
        }
        if self.rms_eps_q32 == 0 {
            return err("rms_eps_q32 must not be zero");
        }
        Ok(())
    }

    /// Check every structural invariant the engine relies on.
    pub fn validate(&self) -> Result<()> {
        self.validate_header(self.layers.len())?;
        let (d, h, v) = (self.dim, self.hidden_dim, self.vocab_size);
        check("tok_emb", self.tok_emb.len(), mul(v, d)?)?;
        for l in &self.layers {
            check("norm1", l.norm1.len(), d)?;
            match &l.mixer {
                Mixer::Attention(a) => {
                    check("wq", a.wq.len(), mul(self.q_rows(), d)?)?;
                    check("wk", a.wk.len(), mul(self.kv_rows(), d)?)?;
                    check("wv", a.wv.len(), mul(self.kv_rows(), d)?)?;
                    check("wo", a.wo.len(), mul(d, self.att_out())?)?;
                    let norm = if self.qk_norm { self.head_dim } else { 0 };
                    check("q_norm", a.q_norm.len(), norm)?;
                    check("k_norm", a.k_norm.len(), norm)?;
                }
                Mixer::Delta(g) => {
                    if self.lin_k_heads == 0 || self.lin_v_heads == 0 || self.lin_k_dim == 0 || self.lin_v_dim == 0 || self.conv_kernel == 0 {
                        return err("gated delta layer present but its dimensions are zero");
                    }
                    check("w_qkv", g.w_qkv.len(), mul(self.conv_dim(), d)?)?;
                    check("w_z", g.w_z.len(), mul(self.lin_v(), d)?)?;
                    check("w_b", g.w_b.len(), mul(self.lin_v_heads, d)?)?;
                    check("w_a", g.w_a.len(), mul(self.lin_v_heads, d)?)?;
                    check("conv", g.conv.len(), mul(self.conv_dim(), self.conv_kernel)?)?;
                    check("a_exp", g.a_exp.len(), self.lin_v_heads)?;
                    check("dt_bias", g.dt_bias.len(), self.lin_v_heads)?;
                    check("gnorm", g.gnorm.len(), self.lin_v_dim)?;
                    check("w_out", g.w_out.len(), mul(d, self.lin_v())?)?;
                }
            }
            check("norm2", l.norm2.len(), d)?;
            check("w_gate", l.w_gate.len(), mul(h, d)?)?;
            check("w_up", l.w_up.len(), mul(h, d)?)?;
            check("w_down", l.w_down.len(), mul(d, h)?)?;
        }
        check("final_norm", self.final_norm.len(), d)?;
        let rope = mul(self.seq_len, self.rotary_dim / 2)?;
        check("rope_cos", self.rope_cos.len(), rope)?;
        check("rope_sin", self.rope_sin.len(), rope)?;
        // The rotation relies on |cos|, |sin| <= 1 to stay inside 64 bits.
        if self.rope_cos.iter().chain(&self.rope_sin).any(|v| !(-(1 << 30)..=1 << 30).contains(v)) {
            return err("RoPE table entry is outside [-1, 1]");
        }
        if let Some(w) = &self.wcls {
            check("wcls", w.len(), mul(v, d)?)?;
        }
        check("tokens", self.tokens.len(), v)?;
        if self.tokens.iter().any(|t| t.len() > 1 << 16) {
            return err("token is longer than 65536 bytes");
        }
        match &self.vocab {
            Vocab::Scored { byte_base, ranks } => {
                check("ranks", ranks.len(), v)?;
                if *byte_base as usize + 256 > v {
                    return err("byte tokens do not fit in the vocabulary");
                }
                for b in 0..256usize {
                    if self.tokens[*byte_base as usize + b] != [b as u8] {
                        return err(format!("byte token {b} does not hold its byte"));
                    }
                }
            }
            Vocab::Bpe { special, merges, byte_tokens, classes } => {
                check("special", special.len(), v)?;
                check("byte_tokens", byte_tokens.len(), 256)?;
                for (b, &id) in byte_tokens.iter().enumerate() {
                    if self.tokens.get(id as usize).map(Vec::as_slice) != Some(&[b as u8]) {
                        return err(format!("byte token {b} does not hold its byte"));
                    }
                }
                if merges.iter().flatten().any(|&id| id as usize >= v) {
                    return err("merge refers to a token outside the vocabulary");
                }
                let mut next = 0u32;
                for &[first, last, class] in classes {
                    if first < next || last < first || !(1..=4).contains(&class) || last > 0x10FFFF {
                        return err("character class ranges are not sorted, disjoint and valid");
                    }
                    next = last + 1;
                }
            }
        }
        let chat = self.chat.iter().flat_map(Chat::lists).flatten();
        if self.prefix.iter().chain(&self.suffix).chain(&self.stop).chain(chat).any(|&id| id as usize >= v) {
            return err("prompt template refers to a token outside the vocabulary");
        }
        Ok(())
    }

    pub fn from_bytes(mut bytes: &[u8]) -> Result<Model> {
        Model::read(&mut bytes)
    }

    /// Read a model from a stream. The stream must end exactly where the model ends.
    pub fn read(inner: impl Read) -> Result<Model> {
        let mut r = Reader { inner, buf: Vec::new() };
        if r.bytes(4)? != MAGIC {
            return err("not a DLLM model file");
        }
        let version = r.u32()?;
        if version != VERSION && version != VERSION_4 {
            return err(format!("unsupported model version {version} (expected {VERSION})"));
        }
        let mut m = Model {
            dim: r.u32()? as usize,
            hidden_dim: r.u32()? as usize,
            vocab_size: 0,
            seq_len: 0,
            rms_eps_q32: 0,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: 0,
            rotary_dim: 0,
            rope_halves: false,
            qk_norm: false,
            out_gate: false,
            lin_k_heads: 0,
            lin_v_heads: 0,
            lin_k_dim: 0,
            lin_v_dim: 0,
            conv_kernel: 0,
            echo_prompt: false,
            trim_prompt: false,
            tok_emb: Tensor::Wide(Vec::new()),
            layers: Vec::new(),
            final_norm: Vec::new(),
            rope_cos: Vec::new(),
            rope_sin: Vec::new(),
            wcls: None,
            tokens: Vec::new(),
            vocab: Vocab::Scored { byte_base: 0, ranks: Vec::new() },
            prefix: Vec::new(),
            suffix: Vec::new(),
            stop: Vec::new(),
            chat: None,
        };
        let n_layers = r.u32()? as usize;
        m.vocab_size = r.u32()? as usize;
        m.seq_len = r.u32()? as usize;
        m.rms_eps_q32 = r.u32()?;
        m.n_heads = r.u32()? as usize;
        m.n_kv_heads = r.u32()? as usize;
        m.head_dim = r.u32()? as usize;
        m.rotary_dim = r.u32()? as usize;
        m.rope_halves = r.flag("rope_halves")?;
        m.qk_norm = r.flag("qk_norm")?;
        m.out_gate = r.flag("out_gate")?;
        m.lin_k_heads = r.u32()? as usize;
        m.lin_v_heads = if version == VERSION_4 { m.lin_k_heads } else { r.u32()? as usize };
        m.lin_k_dim = r.u32()? as usize;
        m.lin_v_dim = r.u32()? as usize;
        m.conv_kernel = r.u32()? as usize;
        let shared = r.flag("shared_classifier")?;
        let tokenizer_kind = r.u32()?;
        m.echo_prompt = r.flag("echo_prompt")?;
        m.trim_prompt = r.flag("trim_prompt")?;
        m.validate_header(n_layers)?;
        let layer_types = (0..n_layers).map(|_| r.u32()).collect::<Result<Vec<u32>>>()?;

        let (d, h, v) = (m.dim, m.hidden_dim, m.vocab_size);
        m.tok_emb = r.tensor(mul(v, d)?)?;
        for layer_type in layer_types {
            let norm1 = r.i32s(d)?;
            let mixer = match layer_type {
                0 => {
                    let norm = if m.qk_norm { m.head_dim } else { 0 };
                    Mixer::Attention(Attention {
                        wq: r.tensor(mul(m.q_rows(), d)?)?,
                        wk: r.tensor(mul(m.kv_rows(), d)?)?,
                        wv: r.tensor(mul(m.kv_rows(), d)?)?,
                        wo: r.tensor(mul(d, m.att_out())?)?,
                        q_norm: r.i32s(norm)?,
                        k_norm: r.i32s(norm)?,
                    })
                }
                1 => Mixer::Delta(Delta {
                    w_qkv: r.tensor(mul(m.conv_dim(), d)?)?,
                    w_z: r.tensor(mul(m.lin_v(), d)?)?,
                    w_b: r.tensor(mul(m.lin_v_heads, d)?)?,
                    w_a: r.tensor(mul(m.lin_v_heads, d)?)?,
                    conv: r.i32s(mul(m.conv_dim(), m.conv_kernel)?)?,
                    a_exp: r.i32s(m.lin_v_heads)?,
                    dt_bias: r.i32s(m.lin_v_heads)?,
                    gnorm: r.i32s(m.lin_v_dim)?,
                    w_out: r.tensor(mul(d, m.lin_v())?)?,
                }),
                other => return err(format!("unknown layer type {other}")),
            };
            m.layers.push(Layer {
                norm1,
                mixer,
                norm2: r.i32s(d)?,
                w_gate: r.tensor(mul(h, d)?)?,
                w_up: r.tensor(mul(h, d)?)?,
                w_down: r.tensor(mul(d, h)?)?,
            });
        }
        m.final_norm = r.i32s(d)?;
        let rope = mul(m.seq_len, m.rotary_dim / 2)?;
        m.rope_cos = r.i32s(rope)?;
        m.rope_sin = r.i32s(rope)?;
        if !shared {
            m.wcls = Some(r.tensor(mul(v, d)?)?);
        }

        let token = |r: &mut Reader<_>| -> Result<(u32, Vec<u8>)> {
            let field = r.u32()?;
            let len = r.u32()? as usize;
            if len > 1 << 16 {
                return err("token is longer than 65536 bytes");
            }
            Ok((field, r.bytes(len)?))
        };
        m.vocab = match tokenizer_kind {
            0 => {
                let byte_base = r.u32()?;
                let mut ranks = Vec::new();
                for _ in 0..v {
                    let (rank, bytes) = token(&mut r)?;
                    ranks.push(rank);
                    m.tokens.push(bytes);
                }
                Vocab::Scored { byte_base, ranks }
            }
            1 => {
                let mut special = Vec::new();
                for _ in 0..v {
                    let (flag, bytes) = token(&mut r)?;
                    if flag > 1 {
                        return err("invalid special-token flag");
                    }
                    special.push(flag == 1);
                    m.tokens.push(bytes);
                }
                let n_merges = r.u32()? as usize;
                if n_merges > 1 << 24 {
                    return err("too many merges");
                }
                let mut merges = Vec::new();
                for _ in 0..n_merges {
                    merges.push([r.u32()?, r.u32()?, r.u32()?]);
                }
                let byte_tokens = (0..256).map(|_| r.u32()).collect::<Result<Vec<u32>>>()?;
                let n_ranges = r.u32()? as usize;
                if n_ranges > 1 << 20 {
                    return err("too many character class ranges");
                }
                let mut classes = Vec::new();
                for _ in 0..n_ranges {
                    classes.push([r.u32()?, r.u32()?, r.u32()?]);
                }
                Vocab::Bpe { special, merges, byte_tokens, classes }
            }
            other => return err(format!("unknown tokenizer kind {other}")),
        };
        m.prefix = r.ids(1 << 16)?;
        m.suffix = r.ids(1 << 16)?;
        m.stop = r.ids(1 << 16)?;
        if r.flag("has_chat")? {
            let mut list = || r.ids(1 << 16);
            m.chat = Some(Chat {
                system: Turn { open: list()?, close: list()? },
                user: Turn { open: list()?, close: list()? },
                assistant: Turn { open: list()?, close: list()? },
                generation: list()?,
            });
        }
        let mut probe = [0u8; 1];
        match r.inner.read(&mut probe) {
            Ok(0) => {}
            Ok(_) => return err("trailing bytes after the end of the model"),
            Err(e) => return err(format!("cannot read model: {e}")),
        }
        m.validate()?;
        Ok(m)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.write(&mut out)?;
        Ok(out)
    }

    /// Write the model in the file format to `inner`.
    pub fn write(&self, inner: &mut impl Write) -> Result<()> {
        self.validate()?;
        let out = &mut Writer { inner, buf: Vec::with_capacity(Writer::<Vec<u8>>::CAPACITY * 2) };
        out.bytes(&MAGIC)?;
        let tokenizer_kind = match self.vocab {
            Vocab::Scored { .. } => 0,
            Vocab::Bpe { .. } => 1,
        };
        for v in [
            VERSION,
            self.dim as u32,
            self.hidden_dim as u32,
            self.layers.len() as u32,
            self.vocab_size as u32,
            self.seq_len as u32,
            self.rms_eps_q32,
            self.n_heads as u32,
            self.n_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.rope_halves as u32,
            self.qk_norm as u32,
            self.out_gate as u32,
            self.lin_k_heads as u32,
            self.lin_v_heads as u32,
            self.lin_k_dim as u32,
            self.lin_v_dim as u32,
            self.conv_kernel as u32,
            self.wcls.is_none() as u32,
            tokenizer_kind,
            self.echo_prompt as u32,
            self.trim_prompt as u32,
        ] {
            out.u32(v)?;
        }
        for l in &self.layers {
            out.u32(matches!(l.mixer, Mixer::Delta(_)) as u32)?;
        }
        out.tensor(&self.tok_emb)?;
        for l in &self.layers {
            out.i32s(&l.norm1)?;
            match &l.mixer {
                Mixer::Attention(a) => {
                    for t in [&a.wq, &a.wk, &a.wv, &a.wo] {
                        out.tensor(t)?;
                    }
                    out.i32s(&a.q_norm)?;
                    out.i32s(&a.k_norm)?;
                }
                Mixer::Delta(g) => {
                    for t in [&g.w_qkv, &g.w_z, &g.w_b, &g.w_a] {
                        out.tensor(t)?;
                    }
                    for t in [&g.conv, &g.a_exp, &g.dt_bias, &g.gnorm] {
                        out.i32s(t)?;
                    }
                    out.tensor(&g.w_out)?;
                }
            }
            out.i32s(&l.norm2)?;
            for t in [&l.w_gate, &l.w_up, &l.w_down] {
                out.tensor(t)?;
            }
        }
        out.i32s(&self.final_norm)?;
        out.i32s(&self.rope_cos)?;
        out.i32s(&self.rope_sin)?;
        if let Some(w) = &self.wcls {
            out.tensor(w)?;
        }
        match &self.vocab {
            Vocab::Scored { byte_base, ranks } => {
                out.u32(*byte_base)?;
                for (token, rank) in self.tokens.iter().zip(ranks) {
                    out.u32(*rank)?;
                    out.u32(token.len() as u32)?;
                    out.bytes(token)?;
                }
            }
            Vocab::Bpe { special, merges, byte_tokens, classes } => {
                for (token, flag) in self.tokens.iter().zip(special) {
                    out.u32(*flag as u32)?;
                    out.u32(token.len() as u32)?;
                    out.bytes(token)?;
                }
                out.u32(merges.len() as u32)?;
                for v in merges.iter().flatten() {
                    out.u32(*v)?;
                }
                for v in byte_tokens {
                    out.u32(*v)?;
                }
                out.u32(classes.len() as u32)?;
                for v in classes.iter().flatten() {
                    out.u32(*v)?;
                }
            }
        }
        out.ids(&self.prefix)?;
        out.ids(&self.suffix)?;
        out.ids(&self.stop)?;
        out.u32(self.chat.is_some() as u32)?;
        for list in self.chat.iter().flat_map(Chat::lists) {
            out.ids(list)?;
        }
        out.flush()
    }
}

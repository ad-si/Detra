//! End-to-end determinism tests on small synthetic models. The expected digests are
//! constants: every platform, compiler, optimization level and thread count must
//! reproduce them.

use detllm::generate::{run_digest, GenParams};
use detllm::model::{Attention, Delta, Layer, Mixer, Model, Vocab};
use detllm::sample::{Rng, SampleParams};
use detllm::sha256::{hex, sha256};
use detllm::tokenizer::Tokenizer;
use std::sync::Arc;

struct Weights(Rng);

impl Weights {
    /// Uniform in [-0.5, 0.5) in Q24.
    fn q24(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| (self.0.next_u64() % (1 << 24)) as i32 - (1 << 23)).collect()
    }

    /// Uniform in [0, 1) in Q24.
    fn positive(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| (self.0.next_u64() % (1 << 24)) as i32).collect()
    }

    /// Uniform in [-1, 1) in Q30.
    fn q30(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| (self.0.next_u64() % (1 << 31)) as i64 as i32 ^ i32::MIN).collect()
    }
}

/// Llama-style model: attention only, adjacent-pair RoPE, scored-merge tokenizer.
fn llama_like() -> Model {
    let (dim, hidden_dim, n_layers, n_heads, seq_len) = (16usize, 40usize, 2usize, 2usize, 48usize);
    let mut w = Weights(Rng::new(0xD11A));
    let mut tokens: Vec<Vec<u8>> = vec![b"<unk>".to_vec(), b"<s>".to_vec(), b"</s>".to_vec()];
    tokens.extend((0..=255u8).map(|b| vec![b]));
    let regular: &[&str] = &[
        " ", "a", "b", "c", "d", "e", "h", "n", "o", "r", "s", "t", "\u{e9}", "th", "he", " t",
        "the", " the", "an", "and", " and", "er", "on", "st", "es",
    ];
    tokens.extend(regular.iter().map(|t| t.as_bytes().to_vec()));
    let vocab_size = tokens.len();
    let head_dim = dim / n_heads;
    let layers = (0..n_layers)
        .map(|_| Layer {
            norm1: vec![1 << 24; dim],
            mixer: Mixer::Attention(Attention {
                wq: w.q24(dim * dim),
                wk: w.q24(dim * dim),
                wv: w.q24(dim * dim),
                wo: w.q24(dim * dim),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
            }),
            norm2: vec![1 << 24; dim],
            w_gate: w.q24(hidden_dim * dim),
            w_up: w.q24(hidden_dim * dim),
            w_down: w.q24(dim * hidden_dim),
        })
        .collect();
    Model {
        dim,
        hidden_dim,
        vocab_size,
        seq_len,
        rms_eps_q32: 42_950,
        n_heads,
        n_kv_heads: n_heads,
        head_dim,
        rotary_dim: head_dim,
        rope_halves: false,
        qk_norm: false,
        out_gate: false,
        lin_heads: 0,
        lin_k_dim: 0,
        lin_v_dim: 0,
        conv_kernel: 0,
        echo_prompt: true,
        trim_prompt: false,
        tok_emb: w.q24(vocab_size * dim),
        layers,
        final_norm: vec![1 << 24; dim],
        rope_cos: w.q30(seq_len * head_dim / 2),
        rope_sin: w.q30(seq_len * head_dim / 2),
        wcls: Some(w.q24(vocab_size * dim)),
        tokens,
        vocab: Vocab::Scored { byte_base: 3, ranks: (0..vocab_size as u32).map(|i| i * 7 % 13).collect() },
        prefix: vec![1],
        suffix: Vec::new(),
        stop: vec![1, 2],
    }
}

const T_TH: u32 = 256;
const T_THE: u32 = 257;
const T_SPACE_THE: u32 = 258;
const T_AN: u32 = 259;
const T_AND: u32 = 260;
const T_START: u32 = 261;
const T_END: u32 = 262;

/// Qwen3.5-style model: gated delta and gated attention layers, grouped keys/values,
/// query/key norm, partial split-half RoPE, tied classifier, byte-level BPE tokenizer.
fn qwen_like() -> Model {
    let (dim, hidden_dim, seq_len) = (16usize, 24usize, 40usize);
    let (n_heads, n_kv_heads, head_dim, rotary_dim) = (4usize, 2usize, 8usize, 4usize);
    let (lin_heads, lin_k_dim, lin_v_dim, conv_kernel) = (2usize, 4usize, 6usize, 3usize);
    let (key_dim, value_dim) = (lin_heads * lin_k_dim, lin_heads * lin_v_dim);
    let conv_dim = 2 * key_dim + value_dim;
    let mut w = Weights(Rng::new(0x35));

    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    for t in ["th", "the", " the", "an", "and", "<|start|>", "<|end|>"] {
        tokens.push(t.as_bytes().to_vec());
    }
    let vocab_size = tokens.len();
    let mut special = vec![false; vocab_size];
    special[T_START as usize] = true;
    special[T_END as usize] = true;
    let b = |c: u8| c as u32;
    let merges = vec![
        [b(b't'), b(b'h'), T_TH],
        [T_TH, b(b'e'), T_THE],
        [b(b' '), T_THE, T_SPACE_THE],
        [b(b'a'), b(b'n'), T_AN],
        [T_AN, b(b'd'), T_AND],
    ];
    let classes = vec![
        [0x09, 0x0D, 4],
        [0x20, 0x20, 4],
        [0x30, 0x39, 3],
        [0x41, 0x5A, 1],
        [0x61, 0x7A, 1],
        [0xE9, 0xE9, 1],
        [0x301, 0x301, 2],
    ];

    let mlp = |w: &mut Weights| (w.q24(hidden_dim * dim), w.q24(hidden_dim * dim), w.q24(dim * hidden_dim));
    let mut layers = Vec::new();
    for kind in [1, 0, 1] {
        let mixer = if kind == 0 {
            Mixer::Attention(Attention {
                wq: w.q24(n_heads * head_dim * 2 * dim),
                wk: w.q24(n_kv_heads * head_dim * dim),
                wv: w.q24(n_kv_heads * head_dim * dim),
                wo: w.q24(dim * n_heads * head_dim),
                q_norm: w.positive(head_dim),
                k_norm: w.positive(head_dim),
            })
        } else {
            Mixer::Delta(Delta {
                w_qkv: w.q24(conv_dim * dim),
                w_z: w.q24(value_dim * dim),
                w_b: w.q24(lin_heads * dim),
                w_a: w.q24(lin_heads * dim),
                conv: w.q24(conv_dim * conv_kernel),
                a_exp: w.positive(lin_heads),
                dt_bias: w.q24(lin_heads),
                gnorm: w.positive(lin_v_dim),
                w_out: w.q24(dim * value_dim),
            })
        };
        let (w_gate, w_up, w_down) = mlp(&mut w);
        layers.push(Layer { norm1: w.positive(dim), mixer, norm2: w.positive(dim), w_gate, w_up, w_down });
    }
    Model {
        dim,
        hidden_dim,
        vocab_size,
        seq_len,
        rms_eps_q32: 4_295,
        n_heads,
        n_kv_heads,
        head_dim,
        rotary_dim,
        rope_halves: true,
        qk_norm: true,
        out_gate: true,
        lin_heads,
        lin_k_dim,
        lin_v_dim,
        conv_kernel,
        echo_prompt: false,
        trim_prompt: true,
        tok_emb: w.q24(vocab_size * dim),
        layers,
        final_norm: w.positive(dim),
        rope_cos: w.q30(seq_len * rotary_dim / 2),
        rope_sin: w.q30(seq_len * rotary_dim / 2),
        wcls: None,
        tokens,
        vocab: Vocab::Bpe { special, merges, byte_tokens: (0..256).collect(), classes },
        prefix: vec![T_START],
        suffix: vec![b(b'\n')],
        stop: vec![T_END],
    }
}

#[test]
fn file_format_roundtrip() {
    for (model, digest) in [
        (llama_like(), "433a4bff984672e124904a9431ec4f7ee68abfac8854480f57bcd8f83d8a064d"),
        (qwen_like(), "f21c3104b5fdf8717b48fa9bf7848b27120d25882de3392c75beaba64361351a"),
    ] {
        let bytes = model.to_bytes().unwrap();
        assert_eq!(Model::from_bytes(&bytes).unwrap(), model);
        assert!(Model::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(Model::from_bytes(&extra).is_err());
        assert_eq!(hex(&sha256(&bytes)), digest);
    }
}

#[test]
fn scored_tokenizer() {
    let model = llama_like();
    let tok = Tokenizer::new(&model);
    let id = |s: &str| model.tokens.iter().rposition(|t| t == s.as_bytes()).unwrap() as u32;
    assert_eq!(tok.encode(b"").unwrap(), Vec::<u32>::new());
    // 'x' is not a regular token and falls back to its byte token, which never merges.
    let text = "the x and \u{e9}\n".as_bytes();
    let ids = tok.encode(text).unwrap();
    assert!(ids.contains(&(3 + b'x' as u32)));
    assert!(ids.contains(&(3 + b'\n' as u32)));
    assert!(ids.contains(&id("\u{e9}")));
    assert_eq!(tok.decode(&ids), text);
    // Invalid UTF-8 is handled bytewise.
    let raw = [0xFFu8, 0x80, 0x80, 0x80, 0x80, b'a'];
    assert_eq!(tok.decode(&tok.encode(&raw).unwrap()), raw);
    assert_eq!(tok.trim(b"  the \n"), b"the");
    assert_eq!(tok.trim(b" "), b"");
}

#[test]
fn bpe_tokenizer() {
    let model = qwen_like();
    let tok = Tokenizer::new(&model);
    let enc = |s: &str| tok.encode(s.as_bytes()).unwrap();
    let b = |s: &str| s.bytes().map(u32::from).collect::<Vec<u32>>();
    assert_eq!(enc(""), Vec::<u32>::new());
    // Pieces: "the", " the"; merges never cross a piece boundary.
    assert_eq!(enc("the the"), [T_THE, T_SPACE_THE]);
    // "and1" splits into "and" and the single number "1"; "1and" likewise.
    assert_eq!(enc("and1"), [T_AND, b'1' as u32]);
    assert_eq!(enc("12and"), [b'1' as u32, b'2' as u32, T_AND]);
    // A run of spaces keeps its last space for the following word.
    assert_eq!(enc("the   the"), [vec![T_THE], b("  "), vec![T_SPACE_THE]].concat());
    // Whitespace up to the last newline is one piece; contractions split off.
    assert_eq!(enc("an \n\n the'll"), [vec![T_AN], b(" \n\n"), vec![T_SPACE_THE], b("'ll")].concat());
    // Punctuation takes one leading space and trailing newlines.
    assert_eq!(enc("the ?!\nthe"), [vec![T_THE], b(" ?!\n"), vec![T_THE]].concat());
    // A mark continues a word; invalid UTF-8 bytes are pieces of their own.
    let text = "the\u{301}n \u{e9}the";
    assert_eq!(enc(text), [vec![T_THE], b("\u{301}n"), b(" \u{e9}"), vec![T_THE]].concat());
    let raw = [b't', b'h', b'e', 0xFF, 0x80, b' ', b't', b'h', b'e'];
    let ids = tok.encode(&raw).unwrap();
    assert_eq!(ids, [T_THE, 0xFF, 0x80, T_SPACE_THE]);
    assert_eq!(tok.decode(&ids), raw);
    assert_eq!(tok.decode(&[T_START, T_THE, T_END]), b"the");
    assert_eq!(tok.trim(b" \n\tthe  the \r\n"), b"the  the");
    assert_eq!(tok.trim(b" \n "), b"");
    assert_eq!(tok.trim(b""), b"");
}

fn digests(model: Model, prompt: &[u8], threads: usize) -> (String, String) {
    let model = Arc::new(model);
    let greedy = GenParams { max_new_tokens: 14, sample: SampleParams::GREEDY, threads };
    let sampled = GenParams {
        max_new_tokens: 14,
        sample: SampleParams { temperature_q16: 52_429, top_k: 20, top_p_q16: 58_982, seed: 42 },
        threads,
    };
    let (g, g_out) = run_digest(&model, prompt, &greedy).unwrap();
    let (s, s_out) = run_digest(&model, prompt, &sampled).unwrap();
    assert!(g_out.tokens.len() > g_out.generated_from + 4, "greedy run stopped early");
    assert_ne!(g_out.tokens, s_out.tokens);
    (hex(&g), hex(&s))
}

#[test]
fn llama_like_digests_are_fixed() {
    let (greedy, sampled) = digests(llama_like(), b"the and then", 1);
    assert_eq!(greedy, "03383a67f92334d76b856ce3f577b199f1972092a8b751d88094b4062923d299");
    assert_eq!(sampled, "a70b49a84f1badaff7506ef0ea01848eadc21b0dbf6461d1102cd44852b3a2c4");
}

#[test]
fn qwen_like_digests_are_fixed() {
    let (greedy, sampled) = digests(qwen_like(), b"the and 12 then", 1);
    assert_eq!(greedy, "6f923f5d40f223fb88151a42d6b9d04453f2a74f10a801e2dea516b888ddf02c");
    assert_eq!(sampled, "e82d4b41ca83716262ffdb70d597ee50433fb5e120a85764b59b28eab0c2e85d");
}

/// Threads are unavailable on some targets (and slow under Miri); everywhere else the
/// thread count must not change a single bit.
#[test]
#[cfg(not(any(miri, target_family = "wasm")))]
fn thread_count_does_not_change_the_result() {
    // Large enough for the pool to actually split the matrices.
    let mut model = llama_like();
    let mut w = Weights(Rng::new(7));
    let dim = model.dim;
    model.hidden_dim = 3000;
    for layer in &mut model.layers {
        layer.w_gate = w.q24(3000 * dim);
        layer.w_up = w.q24(3000 * dim);
        layer.w_down = w.q24(dim * 3000);
    }
    let one = digests(model.clone(), b"the and then", 1);
    for threads in [2, 3, 8] {
        assert_eq!(digests(model.clone(), b"the and then", threads), one);
    }
}

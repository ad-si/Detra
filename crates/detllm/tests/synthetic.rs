//! End-to-end determinism tests on small synthetic models. The expected digests are
//! constants: every platform, compiler, optimization level and thread count must
//! reproduce them.

use detllm::generate::{run_digest, GenParams};
use detllm::model::{Attention, Delta, Layer, Mixer, Model, Tensor, Vocab};
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

    /// Uniform in [-1, 1] in Q30.
    fn q30(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| ((self.0.next_u64() % ((1 << 31) + 1)) as i64 - (1 << 30)) as i32).collect()
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
                wq: w.q24(dim * dim).into(),
                wk: w.q24(dim * dim).into(),
                wv: w.q24(dim * dim).into(),
                wo: w.q24(dim * dim).into(),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
            }),
            norm2: vec![1 << 24; dim],
            w_gate: w.q24(hidden_dim * dim).into(),
            w_up: w.q24(hidden_dim * dim).into(),
            w_down: w.q24(dim * hidden_dim).into(),
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
        tok_emb: w.q24(vocab_size * dim).into(),
        layers,
        final_norm: vec![1 << 24; dim],
        rope_cos: w.q30(seq_len * head_dim / 2),
        rope_sin: w.q30(seq_len * head_dim / 2),
        wcls: Some(w.q24(vocab_size * dim).into()),
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

    let mlp = |w: &mut Weights| -> (Tensor, Tensor, Tensor) {
        (w.q24(hidden_dim * dim).into(), w.q24(hidden_dim * dim).into(), w.q24(dim * hidden_dim).into())
    };
    let mut layers = Vec::new();
    for kind in [1, 0, 1] {
        let mixer = if kind == 0 {
            Mixer::Attention(Attention {
                wq: w.q24(n_heads * head_dim * 2 * dim).into(),
                wk: w.q24(n_kv_heads * head_dim * dim).into(),
                wv: w.q24(n_kv_heads * head_dim * dim).into(),
                wo: w.q24(dim * n_heads * head_dim).into(),
                q_norm: w.positive(head_dim),
                k_norm: w.positive(head_dim),
            })
        } else {
            Mixer::Delta(Delta {
                w_qkv: w.q24(conv_dim * dim).into(),
                w_z: w.q24(value_dim * dim).into(),
                w_b: w.q24(lin_heads * dim).into(),
                w_a: w.q24(lin_heads * dim).into(),
                conv: w.q24(conv_dim * conv_kernel),
                a_exp: w.positive(lin_heads),
                dt_bias: w.q24(lin_heads),
                gnorm: w.positive(lin_v_dim),
                w_out: w.q24(dim * value_dim).into(),
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
        tok_emb: w.q24(vocab_size * dim).into(),
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
        (llama_like(), "8e8d0f0debfaafd7db3aa37c85362a4ac133cac8c5e7ad5a645a686511508b77"),
        (qwen_like(), "4ce64dc06c92c4ed0c2771f6611aa64112632391696f38b64c45bf23efa379c7"),
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
    assert_eq!(greedy, "6a6aef2d12dcbd14731b961735ece5801b1d78fc605686803607bd47ab7e7842");
    assert_eq!(sampled, "dc18f273b0bcb9e11ccfcc9757b73e6277364198c40fbc0919b59b0c842903fa");
}

#[test]
fn qwen_like_digests_are_fixed() {
    let (greedy, sampled) = digests(qwen_like(), b"the and 12 then", 1);
    assert_eq!(greedy, "9d2900a1c1c82ca84eb0c1bafd2634355545b5146d9c8659f0e7b84c952d055b");
    assert_eq!(sampled, "036a2be44cba63eb3804ba85f6ca31dffe34e62a30bc9ec240da7cd5551b36f9");
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
        layer.w_gate = w.q24(3000 * dim).into();
        layer.w_up = w.q24(3000 * dim).into();
        layer.w_down = w.q24(dim * 3000).into();
    }
    let one = digests(model.clone(), b"the and then", 1);
    for threads in [2, 3, 8] {
        assert_eq!(digests(model.clone(), b"the and then", threads), one);
    }
}

/// Every weight tensor of a model except the RoPE tables. Matrices must be unpacked.
fn weight_tensors(model: &mut Model) -> Vec<&mut Vec<i32>> {
    fn wide(t: &mut Tensor) -> &mut Vec<i32> {
        match t {
            Tensor::Wide(v) => v,
            Tensor::Packed(_) => panic!("expected an unpacked tensor"),
        }
    }
    let mut all = vec![wide(&mut model.tok_emb), &mut model.final_norm];
    if let Some(w) = &mut model.wcls {
        all.push(wide(w));
    }
    for layer in &mut model.layers {
        all.extend([&mut layer.norm1, &mut layer.norm2]);
        all.extend([wide(&mut layer.w_gate), wide(&mut layer.w_up), wide(&mut layer.w_down)]);
        match &mut layer.mixer {
            Mixer::Attention(a) => {
                all.extend([wide(&mut a.wq), wide(&mut a.wk), wide(&mut a.wv), wide(&mut a.wo)]);
                all.extend([&mut a.q_norm, &mut a.k_norm]);
            }
            Mixer::Delta(g) => {
                all.extend([wide(&mut g.w_qkv), wide(&mut g.w_z), wide(&mut g.w_b), wide(&mut g.w_a)]);
                all.extend([&mut g.conv, &mut g.a_exp, &mut g.dt_bias, &mut g.gnorm]);
                all.push(wide(&mut g.w_out));
            }
        }
    }
    all
}

/// Every matrix of a model.
fn matrices(model: &mut Model) -> Vec<&mut Tensor> {
    let mut all = vec![&mut model.tok_emb];
    all.extend(model.wcls.as_mut());
    for layer in &mut model.layers {
        all.extend([&mut layer.w_gate, &mut layer.w_up, &mut layer.w_down]);
        match &mut layer.mixer {
            Mixer::Attention(a) => all.extend([&mut a.wq, &mut a.wk, &mut a.wv, &mut a.wo]),
            Mixer::Delta(g) => all.extend([&mut g.w_qkv, &mut g.w_z, &mut g.w_b, &mut g.w_a, &mut g.w_out]),
        }
    }
    all
}

/// Round every matrix value to 8 significant bits, as weights converted from 16-bit
/// floats are, so that the matrices can be packed.
fn with_packable_weights(mut model: Model) -> Model {
    for tensor in matrices(&mut model) {
        let Tensor::Wide(values) = tensor else { panic!("expected an unpacked tensor") };
        for v in values.iter_mut() {
            let drop = (32 - v.unsigned_abs().leading_zeros()).saturating_sub(8);
            *v = (*v >> drop) << drop;
        }
    }
    model
}

/// The same model with every matrix stored packed.
fn packed(mut model: Model) -> Model {
    for tensor in matrices(&mut model) {
        let Tensor::Wide(values) = std::mem::replace(tensor, Tensor::Wide(Vec::new())) else {
            panic!("expected an unpacked tensor")
        };
        *tensor = Tensor::pack(values);
        assert!(matches!(tensor, Tensor::Packed(_)), "weights should be packable");
    }
    model
}

/// Packing is only an encoding: the packed model gives the same results, through the
/// single-position kernel (generation) and the multi-position one (prompt blocks).
#[test]
fn packed_and_unpacked_models_agree() {
    for model in [with_packable_weights(llama_like()), with_packable_weights(qwen_like())] {
        let twin = packed(model.clone());
        assert_ne!(twin.to_bytes().unwrap(), model.to_bytes().unwrap());
        assert_eq!(Model::from_bytes(&twin.to_bytes().unwrap()).unwrap(), twin);
        assert_eq!(digests(twin.clone(), b"the and 12 then the", 1), digests(model.clone(), b"the and 12 then the", 1));
        // Miri never reuses stack addresses, so a long run exhausts a 32-bit address space.
        let cases = if cfg!(miri) { 4 } else { 40 };
        assert_eq!(fuzz_digest(twin, 21, cases), fuzz_digest(model, 21, cases));
    }
    // Values that cannot be packed stay as they are, and every packed value decodes.
    assert_eq!(Tensor::pack(vec![1025]), Tensor::Wide(vec![1025]));
    let values = vec![0, 1, -1, 1023, -1024, 1023 << 21, -1024 << 21, i32::MIN, 3 << 30];
    let Tensor::Packed(p) = Tensor::pack(values.clone()) else { panic!("values should be packable") };
    assert_eq!(p.iter().map(|&p| Tensor::unpack(p)).collect::<Vec<_>>(), values);
}

/// Replace all weights by values chosen to provoke overflow: the extremes of `i32`,
/// values around zero, and arbitrary bit patterns.
fn with_extreme_weights(mut model: Model, seed: u64) -> Model {
    let mut rng = Rng::new(seed);
    let mut pick = |limit: i32| -> i32 {
        match rng.next_u64() % 8 {
            0 => limit,
            1 => -limit,
            2 => 0,
            3 => 1,
            4 => -1,
            _ => ((rng.next_u64() % (2 * limit as u64 + 1)) as i64 - limit as i64) as i32,
        }
    };
    for tensor in weight_tensors(&mut model) {
        for v in tensor.iter_mut() {
            *v = match pick(i32::MAX) {
                i32::MAX => i32::MAX,
                v if v == -i32::MAX => i32::MIN,
                v => v,
            };
        }
    }
    for v in model.rope_cos.iter_mut().chain(model.rope_sin.iter_mut()) {
        *v = pick(1 << 30);
    }
    model
}

/// A prompt built to exercise the tokenizers: arbitrary bytes (mostly invalid UTF-8),
/// words from the vocabulary, whitespace runs, multi-byte characters and contractions.
fn random_prompt(rng: &mut Rng) -> Vec<u8> {
    const PARTS: &[&str] = &[
        "the", " the", "and", "an", "th", " ", "  ", "\n", "\r\n", "\t", "'ll", "'S", "'", "1", "23", "?!", "...",
        "\u{e9}", "e\u{301}", "\u{3000}", "\u{1F642}", "x", "THE", "<|start|>", "<|end|>", "<s>", "</s>", "\0",
    ];
    let mut out = Vec::new();
    for _ in 0..rng.next_u64() % 12 {
        match rng.next_u64() % 4 {
            0 => out.extend((0..rng.next_u64() % 5).map(|_| rng.next_u64() as u8)),
            _ => out.extend_from_slice(PARTS[(rng.next_u64() % PARTS.len() as u64) as usize].as_bytes()),
        }
    }
    out
}

fn random_params(rng: &mut Rng) -> GenParams {
    const TEMPERATURES: &[i64] = &[0, 0, 1, 6_554, 52_429, 65_536, 1 << 30, i64::MAX, -5];
    const TOP_K: &[usize] = &[0, 1, 2, 7, 100_000];
    const TOP_P: &[i64] = &[65_536, 58_982, 32_768, 1, 0, -3, 1 << 40];
    let mut choose = |n: usize| (rng.next_u64() % n as u64) as usize;
    GenParams {
        max_new_tokens: choose(7),
        sample: SampleParams {
            temperature_q16: TEMPERATURES[choose(TEMPERATURES.len())],
            top_k: TOP_K[choose(TOP_K.len())],
            top_p_q16: TOP_P[choose(TOP_P.len())],
            seed: [0, 1, u64::MAX, 0xDEAD_BEEF][choose(4)],
        },
        threads: 1,
    }
}

/// One digest over the complete outcome (hidden states, logits, tokens, or the error)
/// of `cases` random prompts with random parameters.
fn fuzz_digest(model: Model, seed: u64, cases: usize) -> String {
    let model = Arc::new(model);
    let mut rng = Rng::new(seed);
    let mut all = detllm::sha256::Sha256::new();
    let mut completed = 0;
    for _ in 0..cases {
        let prompt = random_prompt(&mut rng);
        let params = random_params(&mut rng);
        match run_digest(&model, &prompt, &params) {
            Ok((digest, _)) => {
                all.update(&digest);
                completed += 1;
            }
            Err(e) => all.update(e.0.as_bytes()),
        }
    }
    assert!(completed * 2 > cases, "most random prompts should fit the context window");
    hex(&all.finalize())
}

/// Arbitrary prompt bytes and sampling parameters. In a debug build this also proves
/// that no input reaches an arithmetic overflow that is not explicitly wrapping.
#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn arbitrary_inputs_have_fixed_digests() {
    assert_eq!(fuzz_digest(llama_like(), 1, 300), "c6c246349a7d23a4be7d26d3ded256a4af8ab112795f27514f8cc79ea8580ece");
    assert_eq!(fuzz_digest(qwen_like(), 2, 300), "b3987e59e6ce677d902bff684525945b1c26e7f97a7c824b596cbfe0a66c3d67");
}

/// The same with weights at the extremes of their range, where sums wrap and
/// activations saturate constantly. The result must still be the same everywhere.
#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn extreme_weights_have_fixed_digests() {
    assert_eq!(fuzz_digest(with_extreme_weights(llama_like(), 3), 4, 150), "ce491fe47367739bdc33a18668172ab0f7c3dd7da9f1d94a80e6a1ab5ed7fa6a");
    assert_eq!(fuzz_digest(with_extreme_weights(qwen_like(), 5), 6, 150), "235c0885dc81bb0aab3fa5bc50be288dd7dca1b7ae5e630821a1639b534d80a3");
}

/// Damaged model files must be rejected with an error or run normally, never panic,
/// and the outcome must be the same everywhere.
#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn corrupted_model_files_have_fixed_outcomes() {
    let mut all = detllm::sha256::Sha256::new();
    let (mut loaded, mut rejected) = (0, 0);
    for (model, seed) in [(llama_like(), 11), (qwen_like(), 12)] {
        let bytes = model.to_bytes().unwrap();
        let mut rng = Rng::new(seed);
        for case in 0..400 {
            let mut damaged = bytes.clone();
            match case % 4 {
                // Header and layer types.
                0 => damaged[4 + (rng.next_u64() % 100) as usize] = rng.next_u64() as u8,
                // Tokenizer tables and prompt template at the end of the file.
                1 => {
                    let at = damaged.len() - 1 - (rng.next_u64() % 3000) as usize;
                    damaged[at] = rng.next_u64() as u8;
                }
                // Anywhere, several bytes.
                2 => {
                    for _ in 0..4 {
                        let at = (rng.next_u64() % damaged.len() as u64) as usize;
                        damaged[at] ^= 1 << (rng.next_u64() % 8);
                    }
                }
                // Truncation.
                _ => damaged.truncate((rng.next_u64() % damaged.len() as u64) as usize),
            }
            match Model::from_bytes(&damaged) {
                Ok(model) => {
                    loaded += 1;
                    let params = GenParams { max_new_tokens: 3, sample: SampleParams::GREEDY, threads: 1 };
                    match run_digest(&Arc::new(model), b"the and 12", &params) {
                        Ok((digest, _)) => all.update(&digest),
                        Err(e) => all.update(e.0.as_bytes()),
                    }
                }
                Err(_) => {
                    rejected += 1;
                    all.update(b"rejected");
                }
            }
        }
    }
    assert!(loaded > 100 && rejected > 100, "loaded {loaded}, rejected {rejected}");
    assert_eq!(hex(&all.finalize()), "ebd6adda13fa98fc9109faf561460ca00e586d511f744b21efd5c01f81494a98");
}

//! End-to-end determinism test on a small synthetic model. The expected digests are
//! constants: every platform, compiler and optimization level must reproduce them.

use detllm::generate::{run_digest, GenParams};
use detllm::model::{Layer, Model};
use detllm::sample::{Rng, SampleParams};
use detllm::sha256::{hex, sha256};
use detllm::tokenizer::Tokenizer;

fn synthetic_model() -> Model {
    let (dim, hidden_dim, n_layers, n_heads, seq_len) = (16usize, 40usize, 2usize, 2usize, 48usize);
    let mut rng = Rng::new(0xD11A);
    // Weights uniform in [-0.5, 0.5) in Q16; RoPE entries uniform in [-1, 1) in Q30.
    let mut q16 = |n: usize| -> Vec<i32> {
        (0..n).map(|_| (rng.next_u64() % 65536) as i32 - 32768).collect()
    };
    let mut tokens: Vec<Vec<u8>> = vec![b"<unk>".to_vec(), b"<s>".to_vec(), b"</s>".to_vec()];
    tokens.extend((0..=255u8).map(|b| vec![b]));
    let regular: &[&str] = &[
        " ", "a", "b", "c", "d", "e", "h", "n", "o", "r", "s", "t", "\u{e9}", "th", "he", " t",
        "the", " the", "an", "and", " and", "er", "on", "st", "es",
    ];
    tokens.extend(regular.iter().map(|t| t.as_bytes().to_vec()));
    let vocab_size = tokens.len();
    let ranks = (0..vocab_size as u32).map(|i| i * 7 % 13).collect();
    let layers = (0..n_layers)
        .map(|_| Layer {
            rms_att: vec![65536; dim],
            wq: q16(dim * dim),
            wk: q16(dim * dim),
            wv: q16(dim * dim),
            wo: q16(dim * dim),
            rms_ffn: vec![65536; dim],
            w1: q16(hidden_dim * dim),
            w2: q16(dim * hidden_dim),
            w3: q16(hidden_dim * dim),
        })
        .collect();
    let tok_emb = q16(vocab_size * dim);
    let wcls = Some(q16(vocab_size * dim));
    let half = dim / n_heads / 2;
    let mut rng = Rng::new(0x0A0E);
    let mut q30 = |n: usize| -> Vec<i32> {
        (0..n).map(|_| (rng.next_u64() % (1 << 31)) as i64 as i32 ^ i32::MIN).collect()
    };
    Model {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        vocab_size,
        seq_len,
        bos_id: 1,
        eos_id: 2,
        byte_base: 3,
        tok_emb,
        layers,
        rms_final: vec![65536; dim],
        rope_cos: q30(seq_len * half),
        rope_sin: q30(seq_len * half),
        wcls,
        tokens,
        ranks,
    }
}

#[test]
fn file_format_roundtrip() {
    let model = synthetic_model();
    let bytes = model.to_bytes().unwrap();
    assert_eq!(Model::from_bytes(&bytes).unwrap(), model);
    assert!(Model::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(Model::from_bytes(&extra).is_err());
    assert_eq!(
        hex(&sha256(&bytes)),
        "8a3ee2deaf72b91370dfe0a5e0562cb89560638b99d6f5d518cc345ca72c156d"
    );
}

#[test]
fn tokenizer_merges_and_roundtrips() {
    let model = synthetic_model();
    let tok = Tokenizer::new(&model);
    let id = |s: &str| model.tokens.iter().rposition(|t| t == s.as_bytes()).unwrap() as u32;
    assert_eq!(tok.encode(b"").unwrap(), vec![1]);
    // 'x' is not a regular token and falls back to its byte token, which never merges.
    let text = "the x and \u{e9}\n".as_bytes();
    let ids = tok.encode(text).unwrap();
    assert_eq!(ids[0], 1);
    assert!(ids.contains(&(3 + b'x' as u32)));
    assert!(ids.contains(&(3 + b'\n' as u32)));
    assert!(ids.contains(&id("\u{e9}")));
    assert_eq!(tok.decode(&ids), text);
    // Invalid UTF-8 is handled bytewise.
    let raw = [0xFFu8, 0x80, 0x80, 0x80, 0x80, b'a'];
    assert_eq!(tok.decode(&tok.encode(&raw).unwrap()), raw);
}

#[test]
fn digests_are_fixed() {
    let model = synthetic_model();
    let greedy = GenParams { max_new_tokens: 30, sample: SampleParams::GREEDY };
    let sampled = GenParams {
        max_new_tokens: 30,
        sample: SampleParams { temperature_q16: 52_429, top_k: 20, top_p_q16: 58_982, seed: 42 },
    };
    let (g, g_tokens) = run_digest(&model, b"the and then", &greedy).unwrap();
    let (s, s_tokens) = run_digest(&model, b"the and then", &sampled).unwrap();
    assert!(g_tokens.len() > 8 && s_tokens.len() > 8);
    assert_ne!(g_tokens, s_tokens);
    assert_eq!(hex(&g), "4b33765257bfd41e9cac8ff7cbf39ab71ad0659cdc1090843b9ffa60849a3a49");
    assert_eq!(hex(&s), "e12bb692cc2c9df10913e0770da02d51bffd6815aea2e7a9b1b25ac79568d5c3");
}

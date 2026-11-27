# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "tokenizers"]
# ///
"""Convert a Hugging Face Qwen3.5 text model to the integer-only DLLM format (version 4).

Usage: uv run tools/convert_qwen35.py <model-dir> <out.dllm> [--seq-len N]

Only the language model is converted; the vision tower and the multi-token-prediction
head are ignored. Floating point is used here and nowhere else: weights are scaled by
2^24 and rounded to nearest-even (exact for bfloat16 inputs above 2^-17), while the
RoPE tables and exp(A_log) go through the platform's libm. The engine therefore never
depends on this script being reproducible: a model is identified by the SHA-256 of the
converted file.
"""

import hashlib
import json
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PREFIX = "<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
STOP = ["<|im_end|>", "<|endoftext|>"]
# Chat template (chat_template.jinja without tools and thinking): per role the text
# before and after the message content, then the generation prompt.
CHAT = [
    ("<|im_start|>system\n", "<|im_end|>\n"),
    ("<|im_start|>user\n", "<|im_end|>\n"),
    ("<|im_start|>assistant\n", "<|im_end|>\n"),
]
GENERATION = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Unicode White_Space, which is what `\s` means in the tokenizer's regex.
WHITE_SPACE = [*range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
               0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
CHUNK = 1 << 22


class SafeTensors:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.base = 8 + n

    def get(self, name):
        info = self.header[name]
        start, end = info["data_offsets"]
        raw = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.base + start, shape=(end - start,))
        dtype = info["dtype"]
        if dtype == "BF16":
            values = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            values = raw.view("<f2").astype(np.float32)
        elif dtype == "F32":
            values = raw.view("<f4")
        else:
            raise SystemExit(f"{name}: unsupported dtype {dtype}")
        return values.reshape(info["shape"])


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.sha = hashlib.sha256()

    def raw(self, data):
        self.f.write(data)
        self.sha.update(data)

    def u32(self, *values):
        self.raw(struct.pack(f"<{len(values)}I", *values))

    def fixed(self, name, values, frac_bits=24, shape=None):
        values = np.asarray(values)
        if shape is not None and tuple(values.shape) != tuple(shape):
            raise SystemExit(f"{name}: shape {values.shape}, expected {shape}")
        flat = values.reshape(-1)
        for i in range(0, flat.size, CHUNK):
            scaled = np.rint(flat[i:i + CHUNK].astype(np.float64) * 2.0**frac_bits)
            if not np.all(np.abs(scaled) < 2.0**31):
                raise SystemExit(f"{name}: value does not fit in Q{frac_bits}")
            self.raw(scaled.astype("<i4").tobytes())

    def matrix(self, name, values, shape):
        """A Q24 matrix: packed 16-bit if every value fits, plain 32-bit otherwise.

        Packed element p stands for (p >> 5) << (p & 31): an 11-bit signed mantissa
        times a power of two. Weights converted from bfloat16 always fit.
        """
        values = np.asarray(values)
        if tuple(values.shape) != tuple(shape):
            raise SystemExit(f"{name}: shape {values.shape}, expected {shape}")
        flat = values.reshape(-1)

        def chunks():
            for i in range(0, flat.size, CHUNK):
                scaled = np.rint(flat[i:i + CHUNK].astype(np.float64) * 2.0**24)
                if not np.all(np.abs(scaled) < 2.0**31):
                    raise SystemExit(f"{name}: value does not fit in Q24")
                yield scaled.astype(np.int64)

        def split(v):
            lowest = np.where(v == 0, 1, v & -v)
            shift = np.round(np.log2(lowest)).astype(np.int64)
            return v >> shift, shift

        packable = all(np.all(np.abs(split(v)[0] + 0.5) < 1024) for v in chunks())
        self.u32(1 if packable else 0)
        for v in chunks():
            if packable:
                mantissa, shift = split(v)
                self.raw(((mantissa << 5) | shift).astype("<i2").tobytes())
            else:
                self.raw(v.astype("<i4").tobytes())
        return packable

    def ids(self, values):
        self.u32(len(values), *values)


def byte_decoder():
    """Inverse of the GPT-2 byte-to-character table used by byte-level BPE vocabularies."""
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def class_ranges():
    """Sorted (first, last, class) ranges: 1 letter, 2 mark, 3 number, 4 whitespace."""
    kinds = {"L": 1, "M": 2, "N": 3}
    white = set(WHITE_SPACE)
    ranges = []
    for cp in range(0x110000):
        cls = 4 if cp in white else kinds.get(unicodedata.category(chr(cp))[0], 0)
        if cls == 0:
            continue
        if ranges and ranges[-1][2] == cls and ranges[-1][1] == cp - 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp, cls])
    return ranges


def main():
    args = sys.argv[1:]
    seq_len = 4096
    if "--seq-len" in args:
        i = args.index("--seq-len")
        seq_len = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 2:
        raise SystemExit(__doc__)
    src, out_path = Path(args[0]), args[1]

    cfg = json.loads((src / "config.json").read_text())["text_config"]
    if not cfg["tie_word_embeddings"] or cfg["attention_bias"] or cfg["hidden_act"] != "silu":
        raise SystemExit("unsupported model configuration")
    if cfg["linear_num_key_heads"] != cfg["linear_num_value_heads"]:
        raise SystemExit("different numbers of key and value heads are not supported")
    dim, hidden, n_layers = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    vocab_size, head_dim = cfg["vocab_size"], cfg["head_dim"]
    n_heads, n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    lin_heads, kd, vd = cfg["linear_num_key_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    kernel = cfg["linear_conv_kernel_dim"]
    rope = cfg["rope_parameters"]
    rotary_dim = int(head_dim * rope.get("partial_rotary_factor", 1.0))
    key_dim, value_dim = lin_heads * kd, lin_heads * vd
    conv_dim = 2 * key_dim + value_dim
    layer_types = [1 if t == "linear_attention" else 0 for t in cfg["layer_types"]]
    seq_len = min(seq_len, cfg["max_position_embeddings"])

    st = SafeTensors(src / "model.safetensors")
    w = Writer(out_path)
    w.raw(b"DLLM")
    w.u32(4, dim, hidden, n_layers, vocab_size, seq_len, round(cfg["rms_norm_eps"] * 2**32),
          n_heads, n_kv, head_dim, rotary_dim, 1, 1, int(cfg["attn_output_gate"]),
          lin_heads, kd, vd, kernel, 1, 1, 0, 1)
    w.u32(*layer_types)

    lm = "model.language_model."
    w.matrix("embed", st.get(lm + "embed_tokens.weight"), (vocab_size, dim))
    for i, kind in enumerate(layer_types):
        p = f"{lm}layers.{i}."
        w.fixed("norm1", 1.0 + st.get(p + "input_layernorm.weight").astype(np.float64), shape=(dim,))
        if kind == 0:
            a = p + "self_attn."
            w.matrix("wq", st.get(a + "q_proj.weight"), (n_heads * head_dim * 2, dim))
            w.matrix("wk", st.get(a + "k_proj.weight"), (n_kv * head_dim, dim))
            w.matrix("wv", st.get(a + "v_proj.weight"), (n_kv * head_dim, dim))
            w.matrix("wo", st.get(a + "o_proj.weight"), (dim, n_heads * head_dim))
            w.fixed("q_norm", 1.0 + st.get(a + "q_norm.weight").astype(np.float64), shape=(head_dim,))
            w.fixed("k_norm", 1.0 + st.get(a + "k_norm.weight").astype(np.float64), shape=(head_dim,))
        else:
            g = p + "linear_attn."
            w.matrix("w_qkv", st.get(g + "in_proj_qkv.weight"), (conv_dim, dim))
            w.matrix("w_z", st.get(g + "in_proj_z.weight"), (value_dim, dim))
            w.matrix("w_b", st.get(g + "in_proj_b.weight"), (lin_heads, dim))
            w.matrix("w_a", st.get(g + "in_proj_a.weight"), (lin_heads, dim))
            w.fixed("conv", st.get(g + "conv1d.weight").reshape(conv_dim, kernel), shape=(conv_dim, kernel))
            w.fixed("a_exp", np.exp(st.get(g + "A_log").astype(np.float64)), shape=(lin_heads,))
            w.fixed("dt_bias", st.get(g + "dt_bias"), shape=(lin_heads,))
            w.fixed("gnorm", st.get(g + "norm.weight"), shape=(vd,))
            w.matrix("w_out", st.get(g + "out_proj.weight"), (dim, value_dim))
        w.fixed("norm2", 1.0 + st.get(p + "post_attention_layernorm.weight").astype(np.float64), shape=(dim,))
        w.matrix("w_gate", st.get(p + "mlp.gate_proj.weight"), (hidden, dim))
        w.matrix("w_up", st.get(p + "mlp.up_proj.weight"), (hidden, dim))
        w.matrix("w_down", st.get(p + "mlp.down_proj.weight"), (dim, hidden))
    w.fixed("final_norm", 1.0 + st.get(lm + "norm.weight").astype(np.float64), shape=(dim,))

    inv_freq = float(rope["rope_theta"]) ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    angles = np.arange(seq_len, dtype=np.float64)[:, None] * inv_freq[None, :]
    w.fixed("rope_cos", np.cos(angles), frac_bits=30)
    w.fixed("rope_sin", np.sin(angles), frac_bits=30)

    tok = json.loads((src / "tokenizer.json").read_text())
    if tok["model"]["type"] != "BPE" or tok["normalizer"] != {"type": "NFC"}:
        raise SystemExit("unsupported tokenizer")
    decode = byte_decoder()
    vocab = tok["model"]["vocab"]
    tokens = [b""] * vocab_size
    special = [1] * vocab_size
    for text, idx in vocab.items():
        tokens[idx] = bytes(decode[c] for c in text)
        special[idx] = 0
    for added in tok["added_tokens"]:
        tokens[added["id"]] = added["content"].encode()
        special[added["id"]] = int(added["special"])
    for flag, data in zip(special, tokens):
        w.u32(flag, len(data))
        w.raw(data)
    merges = [m.split(" ") if isinstance(m, str) else m for m in tok["model"]["merges"]]
    w.u32(len(merges))
    w.raw(np.array([[vocab[a], vocab[b], vocab[a + b]] for a, b in merges], dtype="<u4").tobytes())
    encode = {b: c for c, b in decode.items()}
    w.u32(*[vocab[encode[b]] for b in range(256)])
    ranges = class_ranges()
    w.u32(len(ranges))
    w.raw(np.array(ranges, dtype="<u4").tobytes())

    hf = Tokenizer.from_file(str(src / "tokenizer.json"))
    w.ids(hf.encode(PREFIX, add_special_tokens=False).ids)
    w.ids(hf.encode(SUFFIX, add_special_tokens=False).ids)
    w.ids([hf.token_to_id(t) for t in STOP])
    w.u32(1)
    for text in [*(t for turn in CHAT for t in turn), GENERATION]:
        w.ids(hf.encode(text, add_special_tokens=False).ids)
    w.f.close()
    print(f"{w.sha.hexdigest()}  {out_path}")


if __name__ == "__main__":
    main()

//! Reference forward pass. This is the definition other backends must reproduce.
//!
//! A model is a stack of pre-norm blocks. Each block adds a token mixer and then a
//! SiLU-gated MLP to the residual stream. The mixer is either causal attention
//! (optional per-head query/key RMSNorm, RoPE on the first `rotary_dim` dimensions of
//! each head, grouped keys/values, optional sigmoid output gate) or a gated delta
//! layer (causal depthwise convolution, then a per-head recurrent associative memory).
//!
//! Number formats: the residual stream is `i64` Q32, every other activation `i32` Q20,
//! weights Q24, unit vectors / gates / probabilities Q30, recurrent state `i64` Q32.
//! A matrix product accumulates Q44 in a wrapping `i64`; results that become
//! activations are clamped to the `i32` range.
//!
//! Several consecutive positions can be processed in one call. This is purely a matter
//! of speed: each position's result is defined by the single-position computation, and
//! processing a sequence in blocks of any size gives the same bits.

use crate::fixed::{
    clamp_i64, exp_neg_q30, isqrt_u128, rshift_round, rshift_round_wide, saturate, saturate_wide,
    sigmoid_q30, sigmoid_q30_wide, silu, softmax_q30, softplus_q30,
};
use crate::model::{Attention, Delta, Mixer, Model};
use crate::pool::{Pool, Weight};
use crate::{err, Result};
use std::sync::{Arc, Mutex};

/// Shift that turns a Q44 product sum into a Q20 activation.
const TO_ACT: u32 = 24;
/// Shift that turns a Q44 product sum into a Q32 residual update.
const TO_RESIDUAL: u32 = 12;
/// Shift that turns a Q44 product sum into a Q30 gate input.
const TO_GATE: u32 = 14;
/// Epsilon of the L2 normalisation in gated delta layers: round(1e-6 * 2^40).
const L2_EPS_Q40: u128 = 1_099_512;

/// Root mean square in Q30 of `n` values with `frac` fractional bits, where `sum_sq` is
/// the sum of their squares and `eps_q32` the model's epsilon:
/// `isqrt(scale(sum_sq + eps * n) / n)`, at least 1, where `scale` shifts from
/// `2 * frac` to 60 fractional bits. All arithmetic wraps in `u128`.
fn rms_q30(sum_sq: u128, n: usize, frac: u32, eps_q32: u32) -> i128 {
    let eps = (eps_q32 as u128) << (2 * frac - 32);
    let num = sum_sq.wrapping_add(eps.wrapping_mul(n as u128));
    let scaled = if 2 * frac <= 60 { num << (60 - 2 * frac) } else { num >> (2 * frac - 60) };
    (isqrt_u128(scaled / n as u128) as i128).max(1)
}

fn sum_sq(values: impl Iterator<Item = i64>) -> u128 {
    values.fold(0u128, |s, v| s.wrapping_add((v as i128 * v as i128) as u128))
}

/// RMSNorm of the residual stream (Q32) into activations (Q20):
/// `out = saturate(trunc(x * w / (rms * 2^6)))`.
fn rmsnorm_residual(out: &mut [i32], x: &[i64], w: &[i32], eps_q32: u32) {
    let rms = rms_q30(sum_sq(x.iter().copied()), x.len(), 32, eps_q32) << 6;
    for ((o, &xv), &wv) in out.iter_mut().zip(x).zip(w) {
        *o = saturate_wide((xv as i128 * wv as i128) / rms);
    }
}

/// RMSNorm of every position of the residual stream.
fn rmsnorm_all(x: &[i64], w: &[i32], eps_q32: u32) -> Vec<i32> {
    let mut out = vec![0i32; x.len()];
    for (o, xv) in out.chunks_exact_mut(w.len()).zip(x.chunks_exact(w.len())) {
        rmsnorm_residual(o, xv, w, eps_q32);
    }
    out
}

/// `saturate(trunc(x * w * 2^6 / rms))`: a Q20 value times a Q24 weight over a Q30
/// divisor, in Q20.
#[inline]
fn normed(x: i64, w: i32, rms: i128) -> i32 {
    saturate_wide(((x as i128 * w as i128) << 6) / rms)
}

/// In-place RMSNorm of one attention head (Q20).
fn rmsnorm_head(v: &mut [i32], w: &[i32], eps_q32: u32) {
    let rms = rms_q30(sum_sq(v.iter().map(|&x| x as i64)), v.len(), 20, eps_q32);
    for (x, &wv) in v.iter_mut().zip(w) {
        *x = normed(*x as i64, wv, rms);
    }
}

/// Rotate the first `2 * cos.len()` dimensions of one head by the position angles.
fn rope(v: &mut [i32], cos: &[i32], sin: &[i32], halves: bool) {
    let half = cos.len();
    for i in 0..half {
        let (ia, ib) = if halves { (i, i + half) } else { (2 * i, 2 * i + 1) };
        let (a, b) = (v[ia] as i64, v[ib] as i64);
        let (c, s) = (cos[i] as i64, sin[i] as i64);
        v[ia] = saturate(rshift_round(a * c - b * s, 30));
        v[ib] = saturate(rshift_round(a * s + b * c, 30));
    }
}

fn activations(values: Vec<i64>) -> Arc<Vec<i32>> {
    Arc::new(values.into_iter().map(saturate).collect())
}

fn add_to(x: &mut [i64], update: &[i64]) {
    for (x, &r) in x.iter_mut().zip(update) {
        *x = x.wrapping_add(r);
    }
}

#[derive(Clone)]
enum LayerState {
    /// Rotated keys and values of every past position, `kv_rows` each.
    Attention { keys: Arc<Vec<i32>>, vals: Arc<Vec<i32>> },
    /// `conv`: the last `conv_kernel - 1` inputs of the convolution, oldest first.
    /// `memory`: per head a `[lin_k_dim, lin_v_dim]` matrix in Q32.
    Delta { conv: Vec<i32>, memory: Vec<Arc<Vec<i64>>> },
}

/// One head of a gated delta layer over `count` consecutive positions.
///
/// `qkv` holds, per position, the convolved queries, keys and values of all heads;
/// `z`, `a` and `b` the gate projections. `mem` is the head's memory, updated in place.
/// Appends `lin_v_dim` outputs per position to `out`.
#[allow(clippy::too_many_arguments)]
fn delta_head(
    m: &Model,
    g: &Delta,
    h: usize,
    count: usize,
    qkv: &[i32],
    z: &[i32],
    a: &[i64],
    b: &[i64],
    mem: &mut [i64],
    out: &mut Vec<i64>,
) {
    let (heads, kd, vd) = (m.lin_heads, m.lin_k_dim, m.lin_v_dim);
    let (key_dim, conv_dim, value_dim) = (m.lin_k(), m.conv_dim(), m.lin_v());
    let mut key = vec![0i64; kd];
    let mut query = vec![0i64; kd];
    let mut kv_mem = vec![0i128; vd];
    let mut delta = vec![0i64; vd];
    let mut read = vec![0i128; vd];
    for t in 0..count {
        let at = &qkv[t * conv_dim..(t + 1) * conv_dim];
        let q_raw = &at[h * kd..(h + 1) * kd];
        let k_raw = &at[key_dim + h * kd..key_dim + (h + 1) * kd];
        let value = &at[2 * key_dim + h * vd..2 * key_dim + (h + 1) * vd];
        let z = &z[t * value_dim + h * vd..t * value_dim + (h + 1) * vd];

        // Unit key, and unit query divided by sqrt(lin_k_dim), both in Q30.
        let norm = |v: &[i32], scale: usize| -> i128 {
            let ss = sum_sq(v.iter().map(|&x| x as i64)).wrapping_add(L2_EPS_Q40);
            (isqrt_u128(ss.wrapping_mul(scale as u128) << 20) as i128).max(1)
        };
        let (k_den, q_den) = (norm(k_raw, 1), norm(q_raw, kd));
        for i in 0..kd {
            key[i] = clamp_i64(((k_raw[i] as i128) << 40) / k_den);
            query[i] = clamp_i64(((q_raw[i] as i128) << 40) / q_den);
        }

        // decay = exp(-a_exp * softplus(a + dt_bias)), beta = sigmoid(b), both Q30.
        let rate = softplus_q30(a[t * heads + h].saturating_add((g.dt_bias[h] as i64) << 6));
        let decay = exp_neg_q30(rshift_round_wide(rate as i128 * g.a_exp[h] as i128, 24)) as i128;
        let beta = sigmoid_q30_wide(b[t * heads + h]) as i128;

        // Decay the memory and read what it currently stores under this key.
        kv_mem.fill(0);
        for (row, &kk) in mem.chunks_exact_mut(vd).zip(&key) {
            for (s, acc) in row.iter_mut().zip(kv_mem.iter_mut()) {
                *s = rshift_round_wide(*s as i128 * decay, 30);
                *acc = acc.wrapping_add(*s as i128 * kk as i128);
            }
        }
        // Move the stored value toward the new one, then read with the query.
        for ((dv, &acc), &v) in delta.iter_mut().zip(&kv_mem).zip(value) {
            let error = ((v as i64) << 12).wrapping_sub(rshift_round_wide(acc, 30));
            *dv = rshift_round_wide(error as i128 * beta, 30);
        }
        read.fill(0);
        for ((row, &kk), &qq) in mem.chunks_exact_mut(vd).zip(&key).zip(&query) {
            for ((s, &dv), acc) in row.iter_mut().zip(&delta).zip(read.iter_mut()) {
                *s = s.wrapping_add(rshift_round_wide(kk as i128 * dv as i128, 30));
                *acc = acc.wrapping_add(*s as i128 * qq as i128);
            }
        }

        // RMSNorm over the head (read out in Q32), scaled by SiLU of the z projection.
        let core: Vec<i64> = read.iter().map(|&acc| rshift_round_wide(acc, 30)).collect();
        let rms = rms_q30(sum_sq(core.iter().copied()), vd, 32, m.rms_eps_q32) << 6;
        for i in 0..vd {
            let n = saturate_wide((core[i] as i128 * g.gnorm[i] as i128) / rms) as i64;
            out.push(saturate(rshift_round(n * silu(z[i]) as i64, 20)) as i64);
        }
    }
}

/// Hidden activation of the MLP from its gate and up projections (Q20):
/// `saturate(round_shift(silu(saturate(gate)) * saturate(up), 20))`.
fn mlp_hidden(gate: i64, up: i64) -> i64 {
    saturate(rshift_round(silu(saturate(gate)) as i64 * saturate(up) as i64, 20)) as i64
}

/// One query head at one position attending to all cached positions up to its own.
/// `head` holds the head's query values followed by its gate values (if any). Appends
/// `head_dim` outputs (Q20).
#[allow(clippy::too_many_arguments)]
fn attention_head(
    m: &Model,
    head: &[i32],
    keys: &[i32],
    vals: &[i32],
    kv: usize,
    visible: usize,
    sqrt_head_q16: i64,
    out: &mut Vec<i64>,
) {
    let (hd, kv_rows) = (m.head_dim, m.kv_rows());
    // Score in Q24: the Q40 dot product over sqrt(head_dim) in Q16.
    let mut att: Vec<i64> = (0..visible)
        .map(|s| {
            let ks = &keys[s * kv_rows + kv..][..hd];
            let mut dot = 0i64;
            for (&qv, &kv) in head[..hd].iter().zip(ks) {
                dot = dot.wrapping_add(qv as i64 * kv as i64);
            }
            dot / sqrt_head_q16
        })
        .collect();
    softmax_q30(&mut att);
    for i in 0..hd {
        // |sum| <= max|v| * sum(att) <= 2^31 * 2^30: cannot overflow.
        let mut acc = 0i64;
        for (s, &w) in att.iter().enumerate() {
            acc += w * vals[s * kv_rows + kv + i] as i64;
        }
        let mut value = rshift_round(acc, 30);
        if m.out_gate {
            value = rshift_round(value * sigmoid_q30(head[hd + i]), 30);
        }
        out.push(saturate(value) as i64);
    }
}

pub struct Session {
    model: Arc<Model>,
    /// Created on first use, so that sessions kept for later hold no threads.
    pool: Option<Pool>,
    threads: usize,
    pos: usize,
    /// `isqrt(head_dim * 2^32)`: sqrt(head_dim) in Q16.
    sqrt_head_q16: i64,
    states: Vec<LayerState>,
    hidden: Vec<i32>,
    logits: Vec<i32>,
}

impl Session {
    /// `threads` only affects speed, never the result.
    pub fn new(model: Arc<Model>, threads: usize) -> Session {
        let states = model
            .layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Attention(_) => {
                    LayerState::Attention { keys: Arc::new(Vec::new()), vals: Arc::new(Vec::new()) }
                }
                Mixer::Delta(_) => LayerState::Delta {
                    conv: vec![0; (model.conv_kernel - 1) * model.conv_dim()],
                    memory: (0..model.lin_heads)
                        .map(|_| Arc::new(vec![0; model.lin_k_dim * model.lin_v_dim]))
                        .collect(),
                },
            })
            .collect();
        Session {
            pool: None,
            threads,
            pos: 0,
            sqrt_head_q16: isqrt_u128((model.head_dim as u128) << 32) as i64,
            states,
            hidden: Vec::new(),
            logits: Vec::new(),
            model,
        }
    }

    fn pool(&self) -> &Pool {
        self.pool.as_ref().expect("the pool is created before the first forward pass")
    }

    /// An independent copy of this session: both continue from
    /// the same position and give the same results as one session fed the same tokens.
    pub fn fork(&self) -> Session {
        Session {
            model: Arc::clone(&self.model),
            pool: None,
            threads: self.threads,
            pos: self.pos,
            sqrt_head_q16: self.sqrt_head_q16,
            states: self.states.clone(),
            hidden: self.hidden.clone(),
            logits: self.logits.clone(),
        }
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Number of tokens fed so far.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Logits (Q20) after the last position fed; empty if they were not requested.
    pub fn logits(&self) -> &[i32] {
        &self.logits
    }

    /// Final normalised hidden states (Q20) of the positions fed by the last call,
    /// `dim` values per position.
    pub fn hidden(&self) -> &[i32] {
        &self.hidden
    }

    /// Feed one token at the next position. Returns the logits (Q20) for the token that
    /// follows it, or an empty slice if `want_logits` is false.
    pub fn step(&mut self, token: u32, want_logits: bool) -> Result<&[i32]> {
        self.feed(&[token], want_logits)
    }

    /// Feed several tokens at the next positions. Equivalent to feeding them one by one;
    /// logits are computed for the last one only.
    pub fn feed(&mut self, tokens: &[u32], want_logits: bool) -> Result<&[i32]> {
        let m = Arc::clone(&self.model);
        let count = tokens.len();
        if tokens.iter().any(|&t| t as usize >= m.vocab_size) {
            return err("token id is outside the vocabulary");
        }
        if count > m.seq_len - self.pos.min(m.seq_len) {
            return err("context window is full");
        }
        self.logits = Vec::new();
        self.hidden = Vec::new();
        if count == 0 {
            return Ok(&self.logits);
        }
        let threads = self.threads;
        self.pool.get_or_insert_with(|| Pool::new(threads));
        let d = m.dim;
        let mut x = vec![0i64; count * d];
        for (xs, &token) in x.chunks_exact_mut(d).zip(tokens) {
            for (i, x) in xs.iter_mut().enumerate() {
                *x = (m.tok_emb.get(token as usize * d + i) as i64) << 8;
            }
        }

        for (l, layer) in m.layers.iter().enumerate() {
            let xn = self.norm(&x, &layer.norm1, count);
            let mixed = match &layer.mixer {
                Mixer::Attention(a) => self.attention(l, a, &xn, count),
                Mixer::Delta(_) => self.delta(l, &xn, count),
            };
            add_to(&mut x, &mixed);

            let xn = self.norm(&x, &layer.norm2, count);
            let hidden =
                self.pool().matmul_pair(&m, (Weight::Gate(l), Weight::Up(l)), &xn, count, TO_ACT, mlp_hidden);
            let hidden = Arc::new(hidden.into_iter().map(|v| v as i32).collect());
            let down = self.pool().matmul(&m, Weight::Down(l), &hidden, count, TO_RESIDUAL);
            add_to(&mut x, &down);
        }

        self.hidden = self.norm(&x, &m.final_norm, count).to_vec();
        if want_logits {
            let last = Arc::new(self.hidden[(count - 1) * d..].to_vec());
            let logits = self.pool().matmul(&m, Weight::Classifier, &last, 1, TO_ACT);
            self.logits = logits.into_iter().map(saturate).collect();
        }
        self.pos += count;
        Ok(&self.logits)
    }

    /// RMSNorm of every position of the residual stream, in parallel when there are
    /// several positions.
    fn norm(&self, x: &[i64], w: &[i32], count: usize) -> Arc<Vec<i32>> {
        if count < 4 {
            return Arc::new(rmsnorm_all(x, w, self.model.rms_eps_q32));
        }
        let (x, w, eps) = (Arc::new(x.to_vec()), Arc::new(w.to_vec()), self.model.rms_eps_q32);
        let dim = w.len();
        let values = self.pool().map(count, true, true, move |t, out| {
            let mut normed = vec![0i32; dim];
            rmsnorm_residual(&mut normed, &x[t * dim..(t + 1) * dim], &w, eps);
            out.extend(normed.iter().map(|&v| v as i64));
        });
        Arc::new(values.into_iter().map(|v| v as i32).collect())
    }

    /// Causal attention for `count` new positions. Returns the update of the residual.
    fn attention(&mut self, l: usize, a: &Attention, xn: &Arc<Vec<i32>>, count: usize) -> Vec<i64> {
        let m = Arc::clone(&self.model);
        let hd = m.head_dim;
        let half = m.rotary_dim / 2;
        let q_stride = hd * (1 + m.out_gate as usize);
        let (q_rows, kv_rows, att_out) = (m.q_rows(), m.kv_rows(), m.att_out());

        // Per head, wq yields `head_dim` query values followed by the gate values.
        let mut q: Vec<i32> = self.pool().matmul(&m, Weight::Wq(l), xn, count, TO_ACT).into_iter().map(saturate).collect();
        let mut k: Vec<i32> = self.pool().matmul(&m, Weight::Wk(l), xn, count, TO_ACT).into_iter().map(saturate).collect();
        let v = activations(self.pool().matmul(&m, Weight::Wv(l), xn, count, TO_ACT));
        for t in 0..count {
            let pos = self.pos + t;
            let cos = &m.rope_cos[pos * half..(pos + 1) * half];
            let sin = &m.rope_sin[pos * half..(pos + 1) * half];
            for head in q[t * q_rows..(t + 1) * q_rows].chunks_exact_mut(q_stride) {
                if m.qk_norm {
                    rmsnorm_head(&mut head[..hd], &a.q_norm, m.rms_eps_q32);
                }
                rope(&mut head[..hd], cos, sin, m.rope_halves);
            }
            for head in k[t * kv_rows..(t + 1) * kv_rows].chunks_exact_mut(hd) {
                if m.qk_norm {
                    rmsnorm_head(head, &a.k_norm, m.rms_eps_q32);
                }
                rope(head, cos, sin, m.rope_halves);
            }
        }
        let LayerState::Attention { keys, vals } = &mut self.states[l] else {
            unreachable!("layer state does not match the layer type");
        };
        // No other reference exists between calls, so this extends the caches in place.
        Arc::make_mut(keys).extend_from_slice(&k);
        Arc::make_mut(vals).extend_from_slice(&v);

        // Every (position, head) pair is independent of the others.
        let group = m.n_heads / m.n_kv_heads;
        let (n_heads, first_pos, sqrt_head_q16) = (m.n_heads, self.pos, self.sqrt_head_q16);
        let (q, keys, vals, model) = (Arc::new(q), Arc::clone(keys), Arc::clone(vals), Arc::clone(&m));
        let parallel = count * n_heads * (first_pos + count) * hd >= 1 << 16;
        let out = self.pool().map(count * n_heads, parallel, true, move |block, out| {
            let (t, h) = (block / n_heads, block % n_heads);
            let head = &q[t * q_rows + h * q_stride..t * q_rows + (h + 1) * q_stride];
            let kv = (h / group) * hd;
            attention_head(&model, head, &keys, &vals, kv, first_pos + t + 1, sqrt_head_q16, out);
        });
        debug_assert_eq!(out.len(), count * att_out);
        let out = Arc::new(out.into_iter().map(|v| v as i32).collect());
        self.pool().matmul(&m, Weight::Wo(l), &out, count, TO_RESIDUAL)
    }

    /// Gated delta layer for `count` new positions. Returns the update of the residual.
    fn delta(&mut self, l: usize, xn: &Arc<Vec<i32>>, count: usize) -> Vec<i64> {
        let m = Arc::clone(&self.model);
        let (heads, vd) = (m.lin_heads, m.lin_v_dim);
        let (conv_dim, kernel, value_dim) = (m.conv_dim(), m.conv_kernel, m.lin_v());

        let mixed = activations(self.pool().matmul(&m, Weight::Qkv(l), xn, count, TO_ACT));
        let z = activations(self.pool().matmul(&m, Weight::Z(l), xn, count, TO_ACT));
        let b = Arc::new(self.pool().matmul(&m, Weight::B(l), xn, count, TO_GATE));
        let a = Arc::new(self.pool().matmul(&m, Weight::A(l), xn, count, TO_GATE));

        let pool = self.pool.as_ref().expect("the pool is created before the first forward pass");
        let LayerState::Delta { conv, memory } = &mut self.states[l] else {
            unreachable!("layer state does not match the layer type");
        };
        // Causal depthwise convolution over the last `kernel` inputs, then SiLU.
        // `history` is the remembered inputs followed by the new ones. Channels are
        // independent of each other, so blocks of channels run in parallel.
        let mut history = std::mem::take(conv);
        history.extend_from_slice(&mixed);
        *conv = history[count * conv_dim..].to_vec();
        const CHANNELS: usize = 128;
        let (history, model) = (Arc::new(history), Arc::clone(&m));
        let by_channel = pool.map(conv_dim.div_ceil(CHANNELS), true, true, move |block, out| {
            let Mixer::Delta(g) = &model.layers[l].mixer else {
                unreachable!("layer {l} is not a gated delta layer");
            };
            for c in block * CHANNELS..((block + 1) * CHANNELS).min(conv_dim) {
                let w = &g.conv[c * kernel..(c + 1) * kernel];
                for t in 0..count {
                    let mut acc = 0i64;
                    for (j, &wj) in w.iter().enumerate() {
                        acc = acc.wrapping_add(wj as i64 * history[(t + j) * conv_dim + c] as i64);
                    }
                    out.push(silu(saturate(rshift_round(acc, TO_ACT))) as i64);
                }
            }
        });
        let mut qkv = vec![0i32; count * conv_dim];
        for (c, values) in by_channel.chunks_exact(count).enumerate() {
            for (t, &v) in values.iter().enumerate() {
                qkv[t * conv_dim + c] = v as i32;
            }
        }

        // The heads are independent of each other, so they run in parallel. A head
        // reads the old memory and builds the new one in a buffer of its own, which
        // makes it a pure function; the first finished copy of each head is kept.
        let updated: Arc<Vec<Mutex<Option<Vec<i64>>>>> = Arc::new((0..heads).map(|_| Mutex::new(None)).collect());
        let (qkv, old, new, model) = (Arc::new(qkv), memory.clone(), Arc::clone(&updated), Arc::clone(&m));
        let by_head = pool.map(heads, true, true, move |h, out| {
            let Mixer::Delta(g) = &model.layers[l].mixer else {
                unreachable!("layer {l} is not a gated delta layer");
            };
            let mut mem = old[h].to_vec();
            delta_head(&model, g, h, count, &qkv, &z, &a, &b, &mut mem, out);
            new[h].lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(mem);
        });
        for (current, slot) in memory.iter_mut().zip(updated.iter()) {
            let mem = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
            *current = Arc::new(mem.expect("every head stores its memory before it is counted as finished"));
        }
        let mut out = vec![0i32; count * value_dim];
        for (h, values) in by_head.chunks_exact(count * vd).enumerate() {
            for (t, head) in values.chunks_exact(vd).enumerate() {
                for (o, &v) in out[t * value_dim + h * vd..].iter_mut().zip(head) {
                    *o = v as i32;
                }
            }
        }
        self.pool().matmul(&m, Weight::Out(l), &Arc::new(out), count, TO_RESIDUAL)
    }
}

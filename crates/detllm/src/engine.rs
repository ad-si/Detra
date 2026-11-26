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

use crate::fixed::{
    exp_neg_q30, isqrt_u128, rshift_round, rshift_round_wide, saturate, sigmoid_q30,
    sigmoid_q30_wide, silu, softmax_q30, softplus_q30,
};
use crate::model::{Attention, Delta, Mixer, Model};
use crate::pool::{Pool, Weight};
use crate::{err, Result};
use std::sync::Arc;

/// Shift that turns a Q44 product sum into a Q20 activation.
const TO_ACT: u32 = 24;
/// Shift that turns a Q44 product sum into a Q32 residual update.
const TO_RESIDUAL: u32 = 12;
/// Shift that turns a Q44 product sum into a Q30 gate input.
const TO_GATE: u32 = 14;

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
        *o = saturate(((xv as i128 * wv as i128) / rms) as i64);
    }
}

/// `saturate(trunc(x * w * 2^6 / rms))`: a Q20 value times a Q24 weight over a Q30
/// divisor, in Q20.
#[inline]
fn normed(x: i64, w: i32, rms: i128) -> i32 {
    saturate((((x as i128 * w as i128) << 6) / rms) as i64)
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

fn activations(values: Vec<i64>) -> Vec<i32> {
    values.into_iter().map(saturate).collect()
}

enum LayerState {
    /// Rotated keys and values of every past position, `kv_rows` each.
    Attention { keys: Vec<i32>, vals: Vec<i32> },
    /// `conv`: the last `conv_kernel - 1` inputs of the convolution, oldest first.
    /// `memory`: per head a `[lin_k_dim, lin_v_dim]` matrix in Q32.
    Delta { conv: Vec<i32>, memory: Vec<i64> },
}

pub struct Session {
    model: Arc<Model>,
    pool: Pool,
    pos: usize,
    /// `isqrt(head_dim * 2^32)`: sqrt(head_dim) in Q16.
    sqrt_head_q16: i64,
    states: Vec<LayerState>,
    x: Vec<i64>,
    xn: Vec<i32>,
    att: Vec<i64>,
    logits: Vec<i32>,
}

impl Session {
    /// `threads` only affects speed, never the result.
    pub fn new(model: Arc<Model>, threads: usize) -> Session {
        let states = model
            .layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Attention(_) => LayerState::Attention { keys: Vec::new(), vals: Vec::new() },
                Mixer::Delta(_) => LayerState::Delta {
                    conv: vec![0; (model.conv_kernel - 1) * model.conv_dim()],
                    memory: vec![0; model.lin_heads * model.lin_k_dim * model.lin_v_dim],
                },
            })
            .collect();
        Session {
            pool: Pool::new(threads),
            pos: 0,
            sqrt_head_q16: isqrt_u128((model.head_dim as u128) << 32) as i64,
            states,
            x: vec![0; model.dim],
            xn: vec![0; model.dim],
            att: Vec::new(),
            logits: Vec::new(),
            model,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Logits (Q20) of the last step; empty if they were not requested.
    pub fn logits(&self) -> &[i32] {
        &self.logits
    }

    /// Final normalised hidden state (Q20) of the last step.
    pub fn hidden(&self) -> &[i32] {
        &self.xn
    }

    /// Feed one token at the next position. Returns the logits (Q20) for the token that
    /// follows it, or an empty slice if `want_logits` is false.
    pub fn step(&mut self, token: u32, want_logits: bool) -> Result<&[i32]> {
        let m = Arc::clone(&self.model);
        if token as usize >= m.vocab_size {
            return err("token id is outside the vocabulary");
        }
        if self.pos >= m.seq_len {
            return err("context window is full");
        }
        let d = m.dim;
        let emb = &m.tok_emb[token as usize * d..(token as usize + 1) * d];
        for (x, &e) in self.x.iter_mut().zip(emb) {
            *x = (e as i64) << 8;
        }

        for (l, layer) in m.layers.iter().enumerate() {
            rmsnorm_residual(&mut self.xn, &self.x, &layer.norm1, m.rms_eps_q32);
            let mixed = match &layer.mixer {
                Mixer::Attention(a) => self.attention(l, a),
                Mixer::Delta(g) => self.delta(l, g),
            };
            for (x, r) in self.x.iter_mut().zip(mixed) {
                *x = x.wrapping_add(r);
            }

            rmsnorm_residual(&mut self.xn, &self.x, &layer.norm2, m.rms_eps_q32);
            let gate = self.pool.matvec(&m, Weight::Gate(l), &self.xn, TO_ACT);
            let up = self.pool.matvec(&m, Weight::Up(l), &self.xn, TO_ACT);
            let hidden: Vec<i32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| saturate(rshift_round(silu(saturate(g)) as i64 * saturate(u) as i64, 20)))
                .collect();
            let down = self.pool.matvec(&m, Weight::Down(l), &hidden, TO_RESIDUAL);
            for (x, r) in self.x.iter_mut().zip(down) {
                *x = x.wrapping_add(r);
            }
        }

        rmsnorm_residual(&mut self.xn, &self.x, &m.final_norm, m.rms_eps_q32);
        self.logits = if want_logits {
            activations(self.pool.matvec(&m, Weight::Classifier, &self.xn, TO_ACT))
        } else {
            Vec::new()
        };
        self.pos += 1;
        Ok(&self.logits)
    }

    /// Causal attention over all positions so far. Returns the update of the residual.
    fn attention(&mut self, l: usize, a: &Attention) -> Vec<i64> {
        let m = Arc::clone(&self.model);
        let (hd, pos) = (m.head_dim, self.pos);
        let half = m.rotary_dim / 2;
        let cos = &m.rope_cos[pos * half..(pos + 1) * half];
        let sin = &m.rope_sin[pos * half..(pos + 1) * half];
        let q_stride = hd * (1 + m.out_gate as usize);

        // Per head, wq yields `head_dim` query values followed by the gate values.
        let mut q = activations(self.pool.matvec(&m, Weight::Wq(l), &self.xn, TO_ACT));
        let mut k = activations(self.pool.matvec(&m, Weight::Wk(l), &self.xn, TO_ACT));
        let v = activations(self.pool.matvec(&m, Weight::Wv(l), &self.xn, TO_ACT));
        for head in q.chunks_exact_mut(q_stride) {
            if m.qk_norm {
                rmsnorm_head(&mut head[..hd], &a.q_norm, m.rms_eps_q32);
            }
            rope(&mut head[..hd], cos, sin, m.rope_halves);
        }
        for head in k.chunks_exact_mut(hd) {
            if m.qk_norm {
                rmsnorm_head(head, &a.k_norm, m.rms_eps_q32);
            }
            rope(head, cos, sin, m.rope_halves);
        }
        let LayerState::Attention { keys, vals } = &mut self.states[l] else {
            unreachable!("layer state does not match the layer type");
        };
        keys.extend_from_slice(&k);
        vals.extend_from_slice(&v);

        let kv_rows = m.kv_rows();
        let group = m.n_heads / m.n_kv_heads;
        let mut out = vec![0i32; m.att_out()];
        self.att.resize(pos + 1, 0);
        for h in 0..m.n_heads {
            let head = &q[h * q_stride..(h + 1) * q_stride];
            let kv = (h / group) * hd;
            // Score in Q24: the Q40 dot product over sqrt(head_dim) in Q16.
            for (t, score) in self.att.iter_mut().enumerate() {
                let kt = &keys[t * kv_rows + kv..][..hd];
                let mut dot = 0i64;
                for (&qv, &kv) in head[..hd].iter().zip(kt) {
                    dot = dot.wrapping_add(qv as i64 * kv as i64);
                }
                *score = dot / self.sqrt_head_q16;
            }
            softmax_q30(&mut self.att);
            for i in 0..hd {
                // |sum| <= max|v| * sum(att) <= 2^31 * 2^30: cannot overflow.
                let mut acc = 0i64;
                for (t, &w) in self.att.iter().enumerate() {
                    acc += w * vals[t * kv_rows + kv + i] as i64;
                }
                let mut value = rshift_round(acc, 30);
                if m.out_gate {
                    value = rshift_round(value * sigmoid_q30(head[hd + i]), 30);
                }
                out[h * hd + i] = value as i32;
            }
        }
        self.pool.matvec(&m, Weight::Wo(l), &out, TO_RESIDUAL)
    }

    /// Gated delta layer. Returns the update of the residual.
    fn delta(&mut self, l: usize, g: &Delta) -> Vec<i64> {
        let m = Arc::clone(&self.model);
        let (heads, kd, vd) = (m.lin_heads, m.lin_k_dim, m.lin_v_dim);
        let (key_dim, conv_dim, kernel) = (m.lin_k(), m.conv_dim(), m.conv_kernel);

        let mixed = activations(self.pool.matvec(&m, Weight::Qkv(l), &self.xn, TO_ACT));
        let z = activations(self.pool.matvec(&m, Weight::Z(l), &self.xn, TO_ACT));
        let b = self.pool.matvec(&m, Weight::B(l), &self.xn, TO_GATE);
        let a = self.pool.matvec(&m, Weight::A(l), &self.xn, TO_GATE);

        let LayerState::Delta { conv, memory } = &mut self.states[l] else {
            unreachable!("layer state does not match the layer type");
        };
        // Causal depthwise convolution over the last `kernel` inputs, then SiLU.
        let mut qkv = vec![0i32; conv_dim];
        for (c, out) in qkv.iter_mut().enumerate() {
            let w = &g.conv[c * kernel..(c + 1) * kernel];
            let mut acc = w[kernel - 1] as i64 * mixed[c] as i64;
            for j in 0..kernel - 1 {
                acc = acc.wrapping_add(w[j] as i64 * conv[j * conv_dim + c] as i64);
            }
            *out = silu(saturate(rshift_round(acc, TO_ACT)));
        }
        if kernel > 1 {
            conv.copy_within(conv_dim.., 0);
            conv[(kernel - 2) * conv_dim..].copy_from_slice(&mixed);
        }

        let mut out = vec![0i32; m.lin_v()];
        let mut key = vec![0i64; kd];
        let mut query = vec![0i64; kd];
        let mut kv_mem = vec![0i128; vd];
        let mut delta = vec![0i64; vd];
        let mut read = vec![0i128; vd];
        for h in 0..heads {
            let q_raw = &qkv[h * kd..(h + 1) * kd];
            let k_raw = &qkv[key_dim + h * kd..key_dim + (h + 1) * kd];
            let value = &qkv[2 * key_dim + h * vd..2 * key_dim + (h + 1) * vd];
            let mem = &mut memory[h * kd * vd..(h + 1) * kd * vd];

            // Unit key, and unit query divided by sqrt(lin_k_dim), both in Q30. The
            // epsilon of this normalisation is 1e-6, the same as the model's RMS epsilon
            // only by coincidence of the reference implementation; it is a constant here.
            let norm = |v: &[i32], scale: usize| -> i128 {
                let ss = sum_sq(v.iter().map(|&x| x as i64)).wrapping_add(L2_EPS_Q40);
                (isqrt_u128(ss.wrapping_mul(scale as u128) << 20) as i128).max(1)
            };
            let (k_den, q_den) = (norm(k_raw, 1), norm(q_raw, kd));
            for i in 0..kd {
                key[i] = (((k_raw[i] as i128) << 40) / k_den) as i64;
                query[i] = (((q_raw[i] as i128) << 40) / q_den) as i64;
            }

            // decay = exp(-a_exp * softplus(a + dt_bias)), beta = sigmoid(b), both Q30.
            let rate = softplus_q30(a[h].saturating_add((g.dt_bias[h] as i64) << 6));
            let decay = exp_neg_q30(rshift_round_wide(rate as i128 * g.a_exp[h] as i128, 24)) as i128;
            let beta = sigmoid_q30_wide(b[h]) as i128;

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
                let n = saturate(((core[i] as i128 * g.gnorm[i] as i128) / rms) as i64) as i64;
                out[h * vd + i] = saturate(rshift_round(n * silu(z[h * vd + i]) as i64, 20));
            }
        }
        self.pool.matvec(&m, Weight::Out(l), &out, TO_RESIDUAL)
    }
}

/// Epsilon of the L2 normalisation in gated delta layers: round(1e-6 * 2^40).
const L2_EPS_Q40: u128 = 1_099_512;

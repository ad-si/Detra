//! Reference forward pass of a Llama-style decoder (RMSNorm, RoPE, SiLU-gated MLP).
//! Scalar and single-threaded: this is the definition other backends must reproduce.

use crate::fixed::{isqrt_u128, rshift_round, silu, softmax_q30, RMS_EPS_Q32};
use crate::model::Model;
use crate::{err, Result};

/// `out[r] = round_shift(sum_c w[r][c] * x[c], 16)` truncated to `i32`.
/// The sum wraps in `i64`; wrapping addition is associative, so any evaluation order
/// (SIMD, threads) gives the same bits.
fn matvec(out: &mut [i32], w: &[i32], x: &[i32]) {
    let n = x.len();
    for (o, row) in out.iter_mut().zip(w.chunks_exact(n)) {
        let mut acc = 0i64;
        for (&a, &b) in row.iter().zip(x) {
            acc = acc.wrapping_add(a as i64 * b as i64);
        }
        *o = rshift_round(acc, 16) as i32;
    }
}

/// `out[i] = trunc(x[i] * w[i] * 2^14 / rms)`, with
/// `rms = isqrt((sum(x^2) / n + RMS_EPS_Q32) * 2^28)` in Q30. Exact: nothing can overflow.
fn rmsnorm(out: &mut [i32], x: &[i32], w: &[i32]) {
    let mut ss = 0u128;
    for &v in x {
        let v = v as i64;
        ss += (v * v) as u128;
    }
    let mean = ss / x.len() as u128;
    let rms = isqrt_u128((mean + RMS_EPS_Q32) << 28) as i128;
    for ((o, &xv), &wv) in out.iter_mut().zip(x).zip(w) {
        let num = ((xv as i64 * wv as i64) as i128) << 14;
        *o = (num / rms) as i32;
    }
}

pub struct Session<'m> {
    model: &'m Model,
    pos: usize,
    /// `isqrt(head_size * 2^32)`: sqrt(head_size) in Q16.
    sqrt_head_q16: i64,
    key_cache: Vec<i32>,
    val_cache: Vec<i32>,
    x: Vec<i32>,
    xb: Vec<i32>,
    xb2: Vec<i32>,
    q: Vec<i32>,
    hb: Vec<i32>,
    hb2: Vec<i32>,
    att: Vec<i64>,
    logits: Vec<i32>,
}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model) -> Session<'m> {
        let (d, h) = (model.dim, model.hidden_dim);
        let cache = model.n_layers * model.seq_len * d;
        Session {
            model,
            pos: 0,
            sqrt_head_q16: isqrt_u128((model.head_size() as u128) << 32) as i64,
            key_cache: vec![0; cache],
            val_cache: vec![0; cache],
            x: vec![0; d],
            xb: vec![0; d],
            xb2: vec![0; d],
            q: vec![0; d],
            hb: vec![0; h],
            hb2: vec![0; h],
            att: vec![0; model.seq_len],
            logits: vec![0; model.vocab_size],
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Feed one token at the next position and return the logits (Q16) for the token
    /// that follows it.
    pub fn step(&mut self, token: u32) -> Result<&[i32]> {
        let m = self.model;
        if token as usize >= m.vocab_size {
            return err("token id is outside the vocabulary");
        }
        if self.pos >= m.seq_len {
            return err("context window is full");
        }
        let (d, hs, pos) = (m.dim, m.head_size(), self.pos);
        let half = hs / 2;
        let cos = &m.rope_cos[pos * half..(pos + 1) * half];
        let sin = &m.rope_sin[pos * half..(pos + 1) * half];

        self.x.copy_from_slice(&m.tok_emb[token as usize * d..(token as usize + 1) * d]);

        for (l, layer) in m.layers.iter().enumerate() {
            let base = l * m.seq_len * d;
            let cur = base + pos * d;

            rmsnorm(&mut self.xb, &self.x, &layer.rms_att);
            matvec(&mut self.q, &layer.wq, &self.xb);
            matvec(&mut self.key_cache[cur..cur + d], &layer.wk, &self.xb);
            matvec(&mut self.val_cache[cur..cur + d], &layer.wv, &self.xb);

            // RoPE: rotate each (even, odd) pair of q and k by the position angle.
            for i in (0..d).step_by(2) {
                let c = cos[(i % hs) / 2] as i64;
                let s = sin[(i % hs) / 2] as i64;
                let (q0, q1) = (self.q[i] as i64, self.q[i + 1] as i64);
                self.q[i] = rshift_round(q0 * c - q1 * s, 30) as i32;
                self.q[i + 1] = rshift_round(q0 * s + q1 * c, 30) as i32;
                let (k0, k1) = (self.key_cache[cur + i] as i64, self.key_cache[cur + i + 1] as i64);
                self.key_cache[cur + i] = rshift_round(k0 * c - k1 * s, 30) as i32;
                self.key_cache[cur + i + 1] = rshift_round(k0 * s + k1 * c, 30) as i32;
            }

            for h in 0..m.n_heads {
                let q = &self.q[h * hs..(h + 1) * hs];
                let att = &mut self.att[..=pos];
                for (t, a) in att.iter_mut().enumerate() {
                    let k = &self.key_cache[base + t * d + h * hs..][..hs];
                    let mut dot = 0i64;
                    for (&qv, &kv) in q.iter().zip(k) {
                        dot = dot.wrapping_add(qv as i64 * kv as i64);
                    }
                    *a = dot / self.sqrt_head_q16;
                }
                softmax_q30(att);
                for i in 0..hs {
                    // |sum| <= max|v| * sum(att) <= 2^31 * 2^30: cannot overflow.
                    let mut acc = 0i64;
                    for (t, &a) in att.iter().enumerate() {
                        acc += a * self.val_cache[base + t * d + h * hs + i] as i64;
                    }
                    self.xb[h * hs + i] = rshift_round(acc, 30) as i32;
                }
            }

            matvec(&mut self.xb2, &layer.wo, &self.xb);
            for (x, &r) in self.x.iter_mut().zip(&self.xb2) {
                *x = x.wrapping_add(r);
            }

            rmsnorm(&mut self.xb, &self.x, &layer.rms_ffn);
            matvec(&mut self.hb, &layer.w1, &self.xb);
            matvec(&mut self.hb2, &layer.w3, &self.xb);
            for (a, &b) in self.hb.iter_mut().zip(&self.hb2) {
                *a = rshift_round(silu(*a) as i64 * b as i64, 16) as i32;
            }
            matvec(&mut self.xb, &layer.w2, &self.hb);
            for (x, &r) in self.x.iter_mut().zip(&self.xb) {
                *x = x.wrapping_add(r);
            }
        }

        rmsnorm(&mut self.xb, &self.x, &m.rms_final);
        matvec(&mut self.logits, m.classifier(), &self.xb);
        self.pos += 1;
        Ok(&self.logits)
    }
}

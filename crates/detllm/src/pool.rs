//! Matrix-vector products, optionally spread over worker threads.
//!
//! Every output row is an independent wrapping integer sum, so the result does not
//! depend on the number of threads or on how the rows are divided among them.

use crate::fixed::rshift_round;
use crate::model::{Mixer, Model};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;

/// Selects one weight matrix of a model.
#[derive(Clone, Copy, Debug)]
pub enum Weight {
    Wq(usize),
    Wk(usize),
    Wv(usize),
    Wo(usize),
    Qkv(usize),
    Z(usize),
    B(usize),
    A(usize),
    Out(usize),
    Gate(usize),
    Up(usize),
    Down(usize),
    Classifier,
}

impl Weight {
    fn get(self, m: &Model) -> &[i32] {
        let attention = |l: usize| match &m.layers[l].mixer {
            Mixer::Attention(a) => a,
            Mixer::Delta(_) => unreachable!("layer {l} is not an attention layer"),
        };
        let delta = |l: usize| match &m.layers[l].mixer {
            Mixer::Delta(d) => d,
            Mixer::Attention(_) => unreachable!("layer {l} is not a gated delta layer"),
        };
        match self {
            Weight::Wq(l) => &attention(l).wq,
            Weight::Wk(l) => &attention(l).wk,
            Weight::Wv(l) => &attention(l).wv,
            Weight::Wo(l) => &attention(l).wo,
            Weight::Qkv(l) => &delta(l).w_qkv,
            Weight::Z(l) => &delta(l).w_z,
            Weight::B(l) => &delta(l).w_b,
            Weight::A(l) => &delta(l).w_a,
            Weight::Out(l) => &delta(l).w_out,
            Weight::Gate(l) => &m.layers[l].w_gate,
            Weight::Up(l) => &m.layers[l].w_up,
            Weight::Down(l) => &m.layers[l].w_down,
            Weight::Classifier => m.classifier(),
        }
    }
}

/// `out[r] = round_shift(sum_c w[r][c] * x[c], shift)` for the given rows. The sum
/// wraps in `i64`.
fn rows(w: &[i32], x: &[i32], shift: u32) -> Vec<i64> {
    w.chunks_exact(x.len())
        .map(|row| {
            let mut acc = 0i64;
            for (&a, &b) in row.iter().zip(x) {
                acc = acc.wrapping_add(a as i64 * b as i64);
            }
            rshift_round(acc, shift)
        })
        .collect()
}

type Job = Box<dyn FnOnce() + Send>;

pub struct Pool {
    workers: Vec<Sender<Job>>,
}

/// Below this many multiply-adds a product is computed on the calling thread.
const PARALLEL_THRESHOLD: usize = 1 << 17;

impl Pool {
    /// A pool that uses `threads` threads in total (the caller counts as one).
    pub fn new(threads: usize) -> Pool {
        let mut workers = Vec::new();
        for _ in 1..threads.max(1) {
            let (tx, rx) = channel::<Job>();
            let spawned = std::thread::Builder::new().spawn(move || {
                for job in rx {
                    job();
                }
            });
            if spawned.is_ok() {
                workers.push(tx);
            }
        }
        Pool { workers }
    }

    /// `weight · x`, with each row sum shifted right by `shift` bits (rounded).
    pub fn matvec(&self, model: &Arc<Model>, weight: Weight, x: &[i32], shift: u32) -> Vec<i64> {
        let w = weight.get(model);
        let n_rows = w.len() / x.len();
        let parts = self.workers.len() + 1;
        if parts == 1 || w.len() < PARALLEL_THRESHOLD || n_rows < parts {
            return rows(w, x, shift);
        }
        let x = Arc::new(x.to_vec());
        let cols = x.len();
        let bounds = |p: usize| p * n_rows / parts;
        let (tx, rx) = channel::<(usize, Vec<i64>)>();
        for (p, worker) in self.workers.iter().enumerate() {
            let (model, x, tx) = (Arc::clone(model), Arc::clone(&x), tx.clone());
            let (from, to) = (bounds(p + 1), bounds(p + 2));
            let job: Job = Box::new(move || {
                let w = weight.get(&model);
                let _ = tx.send((p + 1, rows(&w[from * cols..to * cols], &x, shift)));
            });
            worker.send(job).expect("worker thread is gone");
        }
        drop(tx);
        let mut chunks = vec![Vec::new(); parts];
        chunks[0] = rows(&w[..bounds(1) * cols], &x, shift);
        for _ in 1..parts {
            let (p, chunk) = rx.recv().expect("worker thread is gone");
            chunks[p] = chunk;
        }
        chunks.concat()
    }
}

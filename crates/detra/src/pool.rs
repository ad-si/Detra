//! Thread pool for the forward pass.
//!
//! Work is split into independent blocks whose results do not depend on which thread
//! computes them or in what order: every output is a wrapping integer sum over its own
//! inputs. The number of threads therefore changes the speed and nothing else.
//!
//! Workers spin briefly between parallel regions (a forward pass starts hundreds of
//! short ones per token) and sleep when no work arrives.

use crate::fixed::rshift_round;
use crate::model::{Mixer, Model, Tensor};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread::JoinHandle;

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
    fn get(self, m: &Model) -> &Tensor {
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

/// `round_shift(sum_c row[c] * x[c], shift)`. The sum wraps in `i64`.
#[inline]
fn dot(row: &[i32], x: &[i32], shift: u32) -> i64 {
    let mut acc = 0i64;
    for (&a, &b) in row.iter().zip(x) {
        acc = acc.wrapping_add(a as i64 * b as i64);
    }
    rshift_round(acc, shift)
}

/// [`dot`] for a packed row.
#[inline]
fn dot_packed(row: &[i16], x: &[i32], shift: u32) -> i64 {
    let mut acc = 0i64;
    for (&a, &b) in row.iter().zip(x) {
        acc = acc.wrapping_add(Tensor::unpack(a) as i64 * b as i64);
    }
    rshift_round(acc, shift)
}

/// Rows `from..to` of a matrix with `cols` columns, ready for repeated products.
enum Rows<'a> {
    Wide(&'a [i32]),
    Packed(&'a [i16]),
    /// Packed rows decoded once because they are about to be used several times.
    Decoded(Vec<i32>),
}

impl<'a> Rows<'a> {
    fn new(tensor: &'a Tensor, from: usize, to: usize, cols: usize, decode: bool) -> Rows<'a> {
        match tensor {
            Tensor::Wide(w) => Rows::Wide(&w[from * cols..to * cols]),
            Tensor::Packed(w) if decode => {
                Rows::Decoded(w[from * cols..to * cols].iter().map(|&p| Tensor::unpack(p)).collect())
            }
            Tensor::Packed(w) => Rows::Packed(&w[from * cols..to * cols]),
        }
    }

    #[inline]
    fn dot(&self, r: usize, cols: usize, x: &[i32], shift: u32) -> i64 {
        match self {
            Rows::Wide(w) => dot(&w[r * cols..(r + 1) * cols], x, shift),
            Rows::Decoded(w) => dot(&w[r * cols..(r + 1) * cols], x, shift),
            Rows::Packed(w) => dot_packed(&w[r * cols..(r + 1) * cols], x, shift),
        }
    }
}

/// Finished blocks of one thread: block index and values.
type BlockResults = Vec<(usize, Vec<i64>)>;

/// A task receives the index of the thread running it (0 is the caller).
type Task = Arc<dyn Fn(usize) + Send + Sync>;

struct Shared {
    /// Incremented once per parallel region (and once at shutdown).
    generation: AtomicU64,
    shutdown: AtomicBool,
    task: RwLock<Option<Task>>,
    /// Guards going to sleep; the generation only changes while it is held.
    sleep: Mutex<()>,
    wake: Condvar,
}

/// How long a thread spins while waiting before it sleeps or yields. Long enough to
/// bridge the gaps between the parallel regions of a forward pass. This only affects
/// scheduling, never a result.
const SPIN_TIME: Duration = Duration::from_millis(2);

/// Spin until `ready` returns true or `SPIN_TIME` has passed. Returns whether it did.
fn spin_until(ready: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    loop {
        for _ in 0..256 {
            if ready() {
                return true;
            }
            std::hint::spin_loop();
        }
        if start.elapsed() >= SPIN_TIME {
            return false;
        }
    }
}

/// A worker helps with whatever region is current when it notices one. A region does
/// not wait for workers, only for its blocks, so a worker that is asleep or late simply
/// finds nothing left to do.
fn worker(shared: Arc<Shared>, index: usize) {
    let mut seen = 0u64;
    loop {
        if !spin_until(|| shared.generation.load(Ordering::SeqCst) != seen) {
            let mut guard = shared.sleep.lock().unwrap_or_else(|e| e.into_inner());
            while shared.generation.load(Ordering::SeqCst) == seen {
                guard = shared.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
        }
        seen = shared.generation.load(Ordering::SeqCst);
        if shared.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let task = shared.task.read().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(task) = task {
            task(index);
        }
    }
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

/// Blocks per thread a product is divided into. Many, so that a thread on a slow or
/// busy core holds the others up only briefly.
const BLOCKS_PER_THREAD: usize = 8;
/// Bytes of input vectors processed per pass over a block of rows.
const TILE_BYTES: usize = 48 << 10;
/// Below this many multiply-adds a product is computed on the calling thread.
const PARALLEL_THRESHOLD: usize = 1 << 16;

impl Pool {
    /// A pool that uses `threads` threads in total (the caller counts as one).
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            task: RwLock::new(None),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
        });
        let mut workers = Vec::new();
        for _ in 1..threads.max(1) {
            let shared = Arc::clone(&shared);
            let index = workers.len() + 1;
            if let Ok(handle) = std::thread::Builder::new().spawn(move || worker(shared, index)) {
                workers.push(handle);
            }
        }
        Pool { shared, workers }
    }

    /// Offer `task` to the workers and run it on the caller too. Returns once the
    /// caller's own run is over; the task must track its completion itself.
    fn region(&self, task: Task) {
        *self.shared.task.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::clone(&task));
        {
            // The generation changes under the lock so that a worker about to sleep
            // cannot miss it.
            let _guard = self.shared.sleep.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.generation.fetch_add(1, Ordering::SeqCst);
        }
        self.shared.wake.notify_all();
        task(0);
        *self.shared.task.write().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Run `f(block, out)` for every block index below `blocks`, on any thread in any
    /// order; `f` appends the block's values to `out`. Returns all values concatenated
    /// in block order. With `parallel` false everything runs on the caller.
    ///
    /// If `pure` is true, `f` must depend on nothing but its block index and have no
    /// side effects. A thread that runs out of unclaimed blocks then recomputes blocks
    /// other threads are still working on, so that a slow or paused thread never holds
    /// up the rest; the copies are identical and the first one finished is used.
    pub fn map<F>(&self, blocks: usize, parallel: bool, pure: bool, f: F) -> Vec<i64>
    where
        F: Fn(usize, &mut Vec<i64>) + Send + Sync + 'static,
    {
        if !parallel || self.workers.is_empty() || blocks < 2 {
            let mut out = Vec::new();
            for block in 0..blocks {
                f(block, &mut out);
            }
            return out;
        }
        struct State<F> {
            next: AtomicUsize,
            /// Per block: whether a result for it has been stored.
            finished: Vec<AtomicBool>,
            /// Number of blocks with a stored result.
            stored: AtomicUsize,
            /// One slot per thread, so that storing results never contends.
            results: Vec<Mutex<BlockResults>>,
            f: F,
        }
        impl<F: Fn(usize, &mut Vec<i64>)> State<F> {
            fn run(&self, thread: usize, block: usize) {
                let mut values = Vec::new();
                (self.f)(block, &mut values);
                self.results[thread].lock().unwrap_or_else(|e| e.into_inner()).push((block, values));
                if !self.finished[block].swap(true, Ordering::SeqCst) {
                    self.stored.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
        let state = Arc::new(State {
            next: AtomicUsize::new(0),
            finished: (0..blocks).map(|_| AtomicBool::new(false)).collect(),
            stored: AtomicUsize::new(0),
            results: (0..=self.workers.len()).map(|_| Mutex::new(Vec::new())).collect(),
            f,
        });
        let shared = Arc::clone(&state);
        self.region(Arc::new(move |thread| {
            loop {
                let block = shared.next.fetch_add(1, Ordering::Relaxed);
                if block >= blocks {
                    break;
                }
                shared.run(thread, block);
            }
            // Nothing left to claim: help with blocks that are not finished yet.
            while pure && shared.stored.load(Ordering::SeqCst) < blocks {
                for offset in 0..blocks {
                    let block = (thread * 7 + offset) % blocks;
                    if !shared.finished[block].load(Ordering::SeqCst) {
                        shared.run(thread, block);
                    }
                }
            }
        }));
        while !spin_until(|| state.stored.load(Ordering::SeqCst) == blocks) {
            std::thread::yield_now();
        }
        // A block may have been computed more than once; all copies are equal.
        let mut by_block: Vec<Option<Vec<i64>>> = (0..blocks).map(|_| None).collect();
        for slot in &state.results {
            for (block, values) in std::mem::take(&mut *slot.lock().unwrap_or_else(|e| e.into_inner())) {
                by_block[block].get_or_insert(values);
            }
        }
        by_block.into_iter().flatten().flatten().collect()
    }

    /// Product of a weight matrix `[rows, cols]` with `count` input vectors stored one
    /// after another in `x`. Returns `count` output vectors of `rows` values each, every
    /// row sum shifted right by `shift` bits (rounded).
    pub fn matmul(&self, model: &Arc<Model>, weight: Weight, x: &Arc<Vec<i32>>, count: usize, shift: u32) -> Vec<i64> {
        self.matmul_with(model, weight, None, x, count, shift, |a, _| a)
    }

    /// Like [`Pool::matmul`] for two matrices of the same shape at once: output `i` is
    /// `combine(a_i, b_i)` of the two shifted row sums.
    pub fn matmul_pair(
        &self,
        model: &Arc<Model>,
        (a, b): (Weight, Weight),
        x: &Arc<Vec<i32>>,
        count: usize,
        shift: u32,
        combine: fn(i64, i64) -> i64,
    ) -> Vec<i64> {
        self.matmul_with(model, a, Some(b), x, count, shift, combine)
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_with(
        &self,
        model: &Arc<Model>,
        weight: Weight,
        second: Option<Weight>,
        x: &Arc<Vec<i32>>,
        count: usize,
        shift: u32,
        combine: fn(i64, i64) -> i64,
    ) -> Vec<i64> {
        let cols = x.len() / count;
        let n_rows = weight.get(model).len() / cols;
        let rows_per_block = n_rows.div_ceil((self.workers.len() + 1) * BLOCKS_PER_THREAD).max(1);
        let blocks = n_rows.div_ceil(rows_per_block);
        let parallel = n_rows * cols * count >= PARALLEL_THRESHOLD;
        // Input vectors handled per pass over a block of rows: few enough to stay in
        // the innermost cache together with one weight row.
        let tile = (TILE_BYTES / (cols * 4)).max(1);
        let (model, x) = (Arc::clone(model), Arc::clone(x));
        // Each block yields, per input vector, the values of its rows.
        let values = self.map(blocks, parallel, true, move |block, out| {
            let from = block * rows_per_block;
            let to = (from + rows_per_block).min(n_rows);
            // With several input vectors a packed row is decoded once and reused.
            let w = Rows::new(weight.get(&model), from, to, cols, count > 1);
            let w2 = second.map(|s| Rows::new(s.get(&model), from, to, cols, count > 1));
            let (base, rows) = (out.len(), to - from);
            out.resize(base + rows * count, 0);
            for first in (0..count).step_by(tile) {
                let last = (first + tile).min(count);
                for r in 0..rows {
                    for t in first..last {
                        let v = &x[t * cols..(t + 1) * cols];
                        let a = w.dot(r, cols, v, shift);
                        out[base + t * rows + r] = match &w2 {
                            Some(w2) => combine(a, w2.dot(r, cols, v, shift)),
                            None => a,
                        };
                    }
                }
            }
        });
        if blocks == 1 {
            return values;
        }
        let mut out = vec![0i64; count * n_rows];
        let mut offset = 0;
        for block in 0..blocks {
            let from = block * rows_per_block;
            let rows = (from + rows_per_block).min(n_rows) - from;
            for t in 0..count {
                out[t * n_rows + from..][..rows].copy_from_slice(&values[offset + t * rows..][..rows]);
            }
            offset += rows * count;
        }
        out
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let _guard = self.shared.sleep.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.shutdown.store(true, Ordering::SeqCst);
            self.shared.generation.fetch_add(1, Ordering::SeqCst);
        }
        self.shared.wake.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

//! Token selection. Greedy, or seeded sampling done entirely in integers.

use crate::fixed::{exp_neg, ACT_FRAC, ONE_Q16};

/// SplitMix64. The stream is a pure function of the seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `0..n` by rejection (no modulo bias). `n` must be non-zero.
    pub fn below(&mut self, n: u64) -> u64 {
        let rem = (u64::MAX % n + 1) % n; // 2^64 mod n
        let last_accepted = u64::MAX - rem;
        loop {
            let v = self.next_u64();
            if v <= last_accepted {
                return v % n;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleParams {
    /// Temperature in Q16. 0 selects greedy decoding and ignores the other fields.
    pub temperature_q16: i64,
    /// Keep only the `top_k` most likely tokens. 0 disables the limit.
    pub top_k: usize,
    /// Nucleus threshold in Q16, in `1..=65536`. 65536 disables it.
    pub top_p_q16: i64,
    pub seed: u64,
}

impl SampleParams {
    pub const GREEDY: SampleParams =
        SampleParams { temperature_q16: 0, top_k: 0, top_p_q16: ONE_Q16, seed: 0 };
}

/// Index of the largest logit; the lowest index wins ties.
pub fn argmax(logits: &[i32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Pick the next token.
///
/// 1. Order tokens by (logit descending, id ascending); keep the first `top_k`.
/// 2. Weight of token i: `exp(-((logit_max - logit_i) * 2^16 / temperature))` (Q30).
///    Logits are Q20 and the temperature is Q16.
/// 3. Keep the shortest prefix whose weight sum `c` satisfies
///    `c * 2^16 >= top_p * total`.
/// 4. Draw `r` uniformly below the kept sum and take the first token whose cumulative
///    weight exceeds `r`.
pub fn sample(logits: &[i32], p: &SampleParams, rng: &mut Rng) -> u32 {
    if p.temperature_q16 <= 0 {
        return argmax(logits);
    }
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        logits[b as usize].cmp(&logits[a as usize]).then(a.cmp(&b))
    });
    if p.top_k > 0 && p.top_k < order.len() {
        order.truncate(p.top_k);
    }
    let max = logits[order[0] as usize] as i64;
    let mut weights: Vec<u64> = order
        .iter()
        .map(|&id| {
            let diff = max - logits[id as usize] as i64;
            exp_neg(diff * ONE_Q16 / p.temperature_q16, ACT_FRAC) as u64
        })
        .collect();
    // The first weight is exactly 2^30, so the total is never zero.
    let mut total: u64 = weights.iter().sum();
    if p.top_p_q16 < ONE_Q16 {
        let need = p.top_p_q16.max(1) as u128 * total as u128;
        let mut cum = 0u64;
        let mut keep = weights.len();
        for (i, &w) in weights.iter().enumerate() {
            cum += w;
            if (cum as u128) << 16 >= need {
                keep = i + 1;
                break;
            }
        }
        weights.truncate(keep);
        total = cum;
    }
    let r = rng.below(total);
    let mut cum = 0u64;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if r < cum {
            return order[i];
        }
    }
    order[weights.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_values() {
        // Published SplitMix64 outputs for seed 0.
        let mut r = Rng::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(r.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn argmax_prefers_lowest_id() {
        assert_eq!(argmax(&[1, 5, 5, 2]), 1);
        assert_eq!(argmax(&[i32::MIN; 3]), 0);
    }

    #[test]
    fn sampling_respects_filters() {
        let logits = [0, 10 << 20, 9 << 20, -(5 << 20)];
        let mut rng = Rng::new(7);
        let top1 = SampleParams { temperature_q16: ONE_Q16, top_k: 1, top_p_q16: ONE_Q16, seed: 7 };
        let tiny_p = SampleParams { temperature_q16: ONE_Q16, top_k: 0, top_p_q16: 1, seed: 7 };
        let any = SampleParams { temperature_q16: ONE_Q16 * 4, top_k: 2, top_p_q16: ONE_Q16, seed: 7 };
        let mut seen = [0u32; 4];
        for _ in 0..200 {
            assert_eq!(sample(&logits, &top1, &mut rng), 1);
            assert_eq!(sample(&logits, &tiny_p, &mut rng), 1);
            seen[sample(&logits, &any, &mut rng) as usize] += 1;
        }
        assert!(seen[1] > 0 && seen[2] > 0 && seen[0] == 0 && seen[3] == 0);
    }
}

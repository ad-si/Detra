//! Fixed-point primitives. Every function is total and defined purely on integers.
//!
//! Formats (value = n / 2^bits):
//! - Q20: activations and logits, stored as `i32`.
//! - Q24: weights, stored as `i32`; attention scores, held in `i64`.
//! - Q30: probabilities, sigmoid outputs, unit vectors, RoPE sin/cos.
//! - Q32: the residual stream and the recurrent state of gated delta layers, `i64`.

/// Fractional bits of activations and logits.
pub const ACT_FRAC: u32 = 20;
/// 1.0 in Q16 (used for sampling parameters).
pub const ONE_Q16: i64 = 1 << 16;
/// 1.0 in Q30.
pub const ONE_Q30: i64 = 1 << 30;
/// round(ln(2) * 2^30).
pub const LN2_Q30: i64 = 744_261_118;
/// Number of Taylor terms used by [`exp_neg_q30`].
pub const EXP_TERMS: i64 = 12;
/// `exp_neg_q30` returns exactly 0 from this input (32.0) on.
pub const EXP_CUTOFF_Q30: i64 = 32 << 30;
/// Number of series terms used by [`ln1p_q30`].
pub const LN1P_TERMS: i64 = 10;

/// `floor((a + 2^(s-1)) / 2^s)` with wrapping addition: round half toward +infinity.
#[inline(always)]
pub fn rshift_round(a: i64, s: u32) -> i64 {
    a.wrapping_add(1i64 << (s - 1)) >> s
}

/// [`rshift_round`] on 128-bit values, truncated to `i64`.
#[inline(always)]
pub fn rshift_round_wide(a: i128, s: u32) -> i64 {
    (a.wrapping_add(1i128 << (s - 1)) >> s) as i64
}

/// Clamp to the `i32` range.
#[inline(always)]
pub fn saturate(v: i64) -> i32 {
    v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Clamp a 128-bit value to the `i32` range.
#[inline(always)]
pub fn saturate_wide(v: i128) -> i32 {
    v.clamp(i32::MIN as i128, i32::MAX as i128) as i32
}

/// Clamp a 128-bit value to the `i64` range.
#[inline(always)]
pub fn clamp_i64(v: i128) -> i64 {
    v.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Floor of the square root.
pub const fn isqrt_u128(n: u128) -> u128 {
    let mut x = n;
    let mut r = 0u128;
    let mut bit = 1u128 << 126;
    while bit > x {
        bit >>= 2;
    }
    while bit != 0 {
        if x >= r + bit {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    r
}

/// Floor of the cube root, for `n < 2^108`.
pub const fn icbrt_u128(n: u128) -> u128 {
    let mut lo = 0u128;
    let mut hi = 1u128 << 36;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if mid * mid * mid <= n {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `exp(-n)` for `n >= 0`, Q30 in and out. Negative inputs are treated as 0.
///
/// Definition: `k = n / LN2_Q30`, `r = n % LN2_Q30`. Evaluate `t = exp(-r)` by the
/// Horner recurrence `t <- 2^30 - ((r * t) >> 30) / i` for `i = EXP_TERMS down to 1`,
/// starting at `t = 2^30`. The result is `t >> k`, and 0 for `n >= EXP_CUTOFF_Q30`.
pub fn exp_neg_q30(n: i64) -> i64 {
    if n <= 0 {
        return ONE_Q30;
    }
    if n >= EXP_CUTOFF_Q30 {
        return 0;
    }
    let k = n / LN2_Q30;
    let r = n % LN2_Q30;
    let mut t = ONE_Q30;
    let mut i = EXP_TERMS;
    while i >= 1 {
        t = ONE_Q30 - ((r * t) >> 30) / i;
        i -= 1;
    }
    t >> k
}

/// `exp(-n)` for `n >= 0` given with `frac` fractional bits (`frac <= 30`); output Q30.
#[inline]
pub fn exp_neg(n: i64, frac: u32) -> i64 {
    if n <= 0 {
        return ONE_Q30;
    }
    if n >= EXP_CUTOFF_Q30 >> (30 - frac) {
        return 0;
    }
    exp_neg_q30(n << (30 - frac))
}

/// `1 / (1 + e)` if `non_negative`, else `e / (1 + e)`, with `e = exp(-|x|)` in Q30.
fn sigmoid_from_exp(e: i64, non_negative: bool) -> i64 {
    let den = ONE_Q30 + e;
    if non_negative {
        (ONE_Q30 << 30) / den
    } else {
        (e << 30) / den
    }
}

/// Logistic function. Input Q20, output Q30.
pub fn sigmoid_q30(x: i32) -> i64 {
    sigmoid_from_exp(exp_neg((x as i64).abs(), ACT_FRAC), x >= 0)
}

/// Logistic function. Input Q30, output Q30.
pub fn sigmoid_q30_wide(x: i64) -> i64 {
    sigmoid_from_exp(exp_neg_q30(x.saturating_abs()), x >= 0)
}

/// `x * sigmoid(x)` in Q20.
#[inline]
pub fn silu(x: i32) -> i32 {
    rshift_round(x as i64 * sigmoid_q30(x), 30) as i32
}

/// `ln(1 + u)` for `u` in `[0, 1]`, Q30 in and out (inputs are clamped to that range).
///
/// Definition: `t = (u << 30) / (2^31 + u)`, `s = (t * t) >> 30`, and with `p = 0`,
/// `p <- 2^30 / (2i + 1) + ((s * p) >> 30)` for `i = LN1P_TERMS down to 0`.
/// The result is `2 * ((t * p) >> 30)`  (the series `2 * atanh(u / (2 + u))`).
pub fn ln1p_q30(u: i64) -> i64 {
    let u = u.clamp(0, ONE_Q30);
    let t = (u << 30) / (2 * ONE_Q30 + u);
    let s = (t * t) >> 30;
    let mut p = 0i64;
    let mut i = LN1P_TERMS;
    while i >= 0 {
        p = ONE_Q30 / (2 * i + 1) + ((s * p) >> 30);
        i -= 1;
    }
    2 * ((t * p) >> 30)
}

/// `ln(1 + exp(x))`, Q30 in and out: `max(x, 0) + ln1p(exp(-|x|))`.
pub fn softplus_q30(x: i64) -> i64 {
    x.max(0).saturating_add(ln1p_q30(exp_neg_q30(x.saturating_abs())))
}

/// In-place softmax. Input: scores in Q24. Output: probabilities in Q30 whose sum is
/// at most 2^30. An empty slice is left untouched.
pub fn softmax_q30(a: &mut [i64]) {
    let Some(&max) = a.iter().max() else { return };
    let mut sum = 0i64;
    for v in a.iter_mut() {
        *v = exp_neg(max.saturating_sub(*v), 24);
        sum += *v;
    }
    // The maximum element contributes exactly 2^30, so `sum > 0`.
    for v in a.iter_mut() {
        *v = (*v << 30) / sum;
    }
}

/// Parse a non-negative decimal such as `0.8` into Q16, truncating toward zero.
/// At most 9 fractional digits and an integer part below 2^31.
pub fn parse_q16(s: &str) -> Option<i64> {
    let (int_s, frac_s) = match s.split_once('.') {
        Some((i, f)) => (i, f),
        None => (s, ""),
    };
    if int_s.is_empty() && frac_s.is_empty() {
        return None;
    }
    if frac_s.len() > 9 || int_s.len() > 10 {
        return None;
    }
    let digits = |d: &str| -> Option<i64> {
        let mut v = 0i64;
        for c in d.bytes() {
            if !c.is_ascii_digit() {
                return None;
            }
            v = v * 10 + (c - b'0') as i64;
        }
        Some(v)
    };
    let int = digits(int_s)?;
    if int >= 1 << 31 {
        return None;
    }
    let frac = digits(frac_s)?;
    let scale = 10i64.pow(frac_s.len() as u32);
    Some((int << 16) + (frac << 16) / scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_Q20: i32 = 1 << 20;

    #[test]
    fn isqrt() {
        assert_eq!(isqrt_u128(0), 0);
        assert_eq!(isqrt_u128(1), 1);
        assert_eq!(isqrt_u128(15), 3);
        assert_eq!(isqrt_u128(16), 4);
        assert_eq!(isqrt_u128(u128::MAX), u64::MAX as u128);
        assert_eq!(isqrt_u128(1 << 64), 1 << 32);
    }

    #[test]
    fn icbrt() {
        assert_eq!(icbrt_u128(0), 0);
        assert_eq!(icbrt_u128(26), 2);
        assert_eq!(icbrt_u128(27), 3);
        assert_eq!(icbrt_u128(1 << 105), 1 << 35);
    }

    #[test]
    fn rounding_and_saturation() {
        assert_eq!(rshift_round(3, 1), 2);
        assert_eq!(rshift_round(-3, 1), -1);
        assert_eq!(rshift_round_wide(3, 1), 2);
        assert_eq!(rshift_round_wide(-3, 1), -1);
        assert_eq!(saturate(i64::MAX), i32::MAX);
        assert_eq!(saturate(i64::MIN), i32::MIN);
        assert_eq!(saturate(-7), -7);
        assert_eq!(saturate_wide(i128::MAX), i32::MAX);
        assert_eq!(saturate_wide(i128::MIN), i32::MIN);
        assert_eq!(clamp_i64(i128::MAX), i64::MAX);
        assert_eq!(clamp_i64(i128::MIN), i64::MIN);
        assert_eq!(clamp_i64(-7), -7);
    }

    #[test]
    fn exp_values() {
        assert_eq!(exp_neg_q30(0), ONE_Q30);
        assert_eq!(exp_neg_q30(-5), ONE_Q30);
        assert_eq!(exp_neg_q30(EXP_CUTOFF_Q30), 0);
        assert_eq!(exp_neg_q30(i64::MAX), 0);
        assert_eq!(exp_neg(i64::MAX, 16), 0);
        assert_eq!(exp_neg(1 << 16, 16), exp_neg_q30(ONE_Q30));
        // exp(-1) * 2^30 = 395007542.2, exp(-10) * 2^30 = 48747.9
        assert!((exp_neg_q30(ONE_Q30) - 395_007_542).abs() < 64);
        assert!((exp_neg_q30(10 * ONE_Q30) - 48_748).abs() < 4);
        let mut prev = ONE_Q30;
        for i in 1..4000 {
            let v = exp_neg_q30(i * 9_830_400);
            assert!(v <= prev);
            prev = v;
        }
    }

    #[test]
    fn sigmoid_and_silu() {
        assert_eq!(sigmoid_q30(0), ONE_Q30 / 2);
        assert_eq!(sigmoid_q30_wide(0), ONE_Q30 / 2);
        assert_eq!(sigmoid_q30_wide(i64::MIN), 0);
        assert_eq!(sigmoid_q30_wide(i64::MAX), ONE_Q30);
        assert_eq!(silu(0), 0);
        // silu(1) = 0.7310586, silu(-1) = -0.2689414
        assert!((silu(ONE_Q20) - 766_572).abs() <= 4);
        assert!((silu(-ONE_Q20) + 282_006).abs() <= 4);
        assert_eq!(silu(i32::MIN), 0);
        assert_eq!(silu(i32::MAX), i32::MAX);
    }

    #[test]
    fn softplus_and_log() {
        assert_eq!(ln1p_q30(0), 0);
        // ln(2) * 2^30 = 744261117.95
        assert!((ln1p_q30(ONE_Q30) - LN2_Q30).abs() < 64);
        assert!((softplus_q30(0) - LN2_Q30).abs() < 64);
        // softplus(1) = 1.3132616875, softplus(-1) = 0.3132616875
        assert!((softplus_q30(ONE_Q30) - 1_410_103_992).abs() < 64);
        assert!((softplus_q30(-ONE_Q30) - 336_362_168).abs() < 64);
        assert_eq!(softplus_q30(40 * ONE_Q30), 40 * ONE_Q30);
        assert_eq!(softplus_q30(i64::MIN), 0);
        assert_eq!(softplus_q30(i64::MAX), i64::MAX);
    }

    #[test]
    fn softmax() {
        let mut a = [0i64, 0, 0, 0];
        softmax_q30(&mut a);
        assert_eq!(a, [ONE_Q30 / 4; 4]);
        let mut b = [i64::MIN, 0, i64::MAX];
        softmax_q30(&mut b);
        assert_eq!(b, [0, 0, ONE_Q30]);
    }

    #[test]
    fn parse() {
        assert_eq!(parse_q16("0"), Some(0));
        assert_eq!(parse_q16("1"), Some(65536));
        assert_eq!(parse_q16("0.5"), Some(32768));
        assert_eq!(parse_q16(".25"), Some(16384));
        assert_eq!(parse_q16("0.8"), Some(52428));
        assert_eq!(parse_q16("1e3"), None);
        assert_eq!(parse_q16("-1"), None);
        assert_eq!(parse_q16(""), None);
    }
}

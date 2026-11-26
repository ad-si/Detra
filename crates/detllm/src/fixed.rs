//! Fixed-point primitives. Every function is total and defined purely on integers.
//!
//! Formats:
//! - Q16: weights, activations, logits. Stored as `i32`, value = n / 2^16.
//! - Q30: probabilities, sigmoid outputs, RoPE sin/cos. value = n / 2^30.
//! - Q32: products of two Q16 values, held in `i64`.

/// 1.0 in Q16.
pub const ONE_Q16: i64 = 1 << 16;
/// 1.0 in Q30.
pub const ONE_Q30: i64 = 1 << 30;
/// round(ln(2) * 2^30).
pub const LN2_Q30: i64 = 744_261_118;
/// Number of Taylor terms used by [`exp_neg`].
pub const EXP_TERMS: i64 = 12;
/// Inputs at or below `-EXP_CUTOFF_Q16` map to exactly 0.
pub const EXP_CUTOFF_Q16: i64 = 32 << 16;
/// RMSNorm epsilon: ceil(1e-5 * 2^32).
pub const RMS_EPS_Q32: u128 = 42_950;

/// `floor((a + 2^(s-1)) / 2^s)` with wrapping addition: round half toward +infinity.
#[inline(always)]
pub fn rshift_round(a: i64, s: u32) -> i64 {
    a.wrapping_add(1i64 << (s - 1)) >> s
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

/// `exp(x)` for `x <= 0`. Input Q16, output Q30 in `[0, 2^30]`. Positive inputs are
/// treated as 0.
///
/// Definition: let `n = -x * 2^14` (Q30), `k = n / LN2_Q30`, `r = n % LN2_Q30`.
/// Evaluate `t = exp(-r)` by the Horner recurrence
/// `t <- 2^30 - ((r * t) >> 30) / i` for `i = EXP_TERMS down to 1`, starting at `t = 2^30`.
/// The result is `t >> k`.
pub fn exp_neg(x: i64) -> i64 {
    let x = if x > 0 { 0 } else { x };
    if x <= -EXP_CUTOFF_Q16 {
        return 0;
    }
    let n = (-x) << 14;
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

/// Logistic function. Input Q16, output Q30.
pub fn sigmoid_q30(x: i32) -> i64 {
    let x = x as i64;
    let e = exp_neg(-x.abs());
    let den = ONE_Q30 + e;
    if x >= 0 {
        (ONE_Q30 << 30) / den
    } else {
        (e << 30) / den
    }
}

/// `x * sigmoid(x)` in Q16.
#[inline]
pub fn silu(x: i32) -> i32 {
    rshift_round(x as i64 * sigmoid_q30(x), 30) as i32
}

/// In-place softmax. Input: scores in Q16. Output: probabilities in Q30 whose sum is
/// at most 2^30. An empty slice is left untouched.
pub fn softmax_q30(a: &mut [i64]) {
    let Some(&max) = a.iter().max() else { return };
    let mut sum = 0i64;
    for v in a.iter_mut() {
        *v = exp_neg(v.saturating_sub(max));
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
    fn exp_values() {
        assert_eq!(exp_neg(0), ONE_Q30);
        assert_eq!(exp_neg(5), ONE_Q30);
        assert_eq!(exp_neg(-EXP_CUTOFF_Q16), 0);
        assert_eq!(exp_neg(i64::MIN), 0);
        // exp(-1) * 2^30 = 395007542.2, exp(-10) * 2^30 = 48747.9
        assert!((exp_neg(-ONE_Q16) - 395_007_542).abs() < 64);
        assert!((exp_neg(-10 * ONE_Q16) - 48_748).abs() < 4);
        let mut prev = ONE_Q30;
        for i in 1..4000 {
            let v = exp_neg(-i * 600);
            assert!(v <= prev);
            prev = v;
        }
    }

    #[test]
    fn sigmoid_and_silu() {
        assert_eq!(sigmoid_q30(0), ONE_Q30 / 2);
        assert_eq!(silu(0), 0);
        // silu(1) = 0.7310586, silu(-1) = -0.2689414
        assert!((silu(65536) - 47_911).abs() <= 2);
        assert!((silu(-65536) + 17_625).abs() <= 2);
        assert_eq!(silu(i32::MIN), 0);
        assert_eq!(silu(i32::MAX), i32::MAX);
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

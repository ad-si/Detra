//! High-precision integer arithmetic for the few transcendental values a model
//! conversion needs (`exp`, `ln`, `cos`, `sin`), so that conversion is a pure function
//! of the checkpoint bytes, like inference.
//!
//! Values are `i128` with 96 fractional bits (Q96). Every function here is accurate to
//! far better than `SLACK` (2^-76, absolute or, for [`exp`], relative), and results are
//! correctly rounded to their target format: a value that lies within `SLACK` of a
//! rounding boundary cannot be rounded with certainty and is reported as `None` instead.

/// Fractional bits of a Q96 value.
pub const FRAC: u32 = 96;
pub const ONE: i128 = 1 << FRAC;

/// Error bound for rounding decisions, in Q96 units.
const SLACK: i128 = 1 << 20;

/// pi / 2 in Q160, split into its Q96 part and the next 64 bits.
const HALF_PI_HI: i128 = 0x1921fb54442d18469898cc517;
const HALF_PI_LO: i128 = 0x01b839a252049c11;
/// ln 2 in Q160, split the same way.
const LN2_HI: i128 = 0xb17217f7d1cf79abc9e3b398;
const LN2_LO: i128 = 0x03f2f6af40f34326;

/// `a * b` in Q96, truncated toward zero. The caller guarantees `|a * b| < 2^31`.
pub fn mul(a: i128, b: i128) -> i128 {
    const MASK: u128 = u64::MAX as u128;
    let (x, y) = (a.unsigned_abs(), b.unsigned_abs());
    let (x1, x0, y1, y0) = (x >> 64, x & MASK, y >> 64, y & MASK);
    let (p00, p01, p10, p11) = (x0 * y0, x0 * y1, x1 * y0, x1 * y1);
    let mid = (p00 >> 64) + (p01 & MASK) + (p10 & MASK);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    let lo_top = mid & MASK; // bits 64..128 of the product
    assert!(hi >> 95 == 0, "fixed-point product out of range");
    let magnitude = ((hi << 32) | (lo_top >> 32)) as i128;
    if (a < 0) != (b < 0) {
        -magnitude
    } else {
        magnitude
    }
}

/// `k * c` for a constant `c` given in Q160 as `hi` (Q96) and `lo` (the next 64 bits).
fn mul_const(k: i128, hi: i128, lo: i128) -> i128 {
    k * hi + ((k * lo) >> 64)
}

/// `a / b` rounded to the nearest integer (ties up), for `b > 0`.
fn div_nearest(a: i128, b: i128) -> i128 {
    (2 * a + b).div_euclid(2 * b)
}

/// atanh(z) for `0 <= z <= 1/3` (Q96).
fn atanh(z: i128) -> i128 {
    let z2 = mul(z, z);
    let (mut term, mut sum, mut k) = (z, 0, 1);
    while term != 0 {
        sum += term / k;
        term = mul(term, z2);
        k += 2;
    }
    sum
}

/// ln(n) in Q96 for `n >= 1`.
pub fn ln(n: u32) -> i128 {
    assert!(n >= 1);
    let k = 31 - n.leading_zeros();
    let p = 1i128 << k;
    // n = 2^k * m with 1 <= m < 2, and ln m = 2 atanh((m - 1) / (m + 1)).
    let z = ((n as i128 - p) << FRAC) / (n as i128 + p);
    mul_const(k as i128, LN2_HI, LN2_LO) + 2 * atanh(z)
}

/// e^x as `(m, k)` with `e^x = m * 2^k` and `m` in Q96 between 0.7 and 1.5, so that `m`
/// carries the full relative precision. `None` if `x` is outside [-1000, 1000].
pub fn exp(x: i128) -> Option<(i128, i32)> {
    if x.abs() > 1000 * ONE {
        return None;
    }
    let k = div_nearest(x, LN2_HI);
    let r = x - mul_const(k, LN2_HI, LN2_LO);
    let (mut term, mut sum, mut n) = (ONE, ONE, 1);
    loop {
        term = mul(term, r) / n;
        if term == 0 {
            break;
        }
        sum += term;
        n += 1;
    }
    Some((sum, k as i32))
}

/// (cos a, sin a) in Q96 for `0 <= a <= 2^20` (Q96).
pub fn cos_sin(a: i128) -> (i128, i128) {
    assert!((0..=ONE << 20).contains(&a));
    let k = div_nearest(a, HALF_PI_HI);
    let r = a - mul_const(k, HALF_PI_HI, HALF_PI_LO);
    let r2 = mul(r, r);
    let series = |first: i128, start: i128| {
        let (mut term, mut sum, mut i) = (first, 0, start);
        while term != 0 {
            sum += term;
            term = -mul(term, r2) / ((i + 1) * (i + 2));
            i += 2;
        }
        sum
    };
    let (c, s) = (series(ONE, 0), series(r, 1));
    match k.rem_euclid(4) {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

/// `v` (Q96) rounded to a multiple of 2^-bits, ties to even, as an integer count of
/// 2^-bits. `None` if `v` is too close to a rounding boundary.
pub fn round_fixed(v: i128, bits: u32) -> Option<i128> {
    let shift = FRAC - bits;
    let q = v >> shift;
    let rem = v - (q << shift);
    let half = 1i128 << (shift - 1);
    if rem != 0 && (rem - half).abs() <= SLACK {
        return None;
    }
    Some(if rem > half { q + 1 } else { q })
}

/// `m * 2^k` (as returned by [`exp`]) rounded to the nearest IEEE double, as an integer
/// significand `2^52 <= s < 2^53` and exponent `e` with value `s * 2^e`.
/// `None` if the value is too close to a rounding boundary.
pub fn round_double((m, k): (i128, i32)) -> Option<(u64, i32)> {
    if m <= 0 {
        return None;
    }
    let shift = (127 - m.leading_zeros()) - 52;
    let q = m >> shift;
    let rem = m - (q << shift);
    let half = 1i128 << (shift - 1);
    if (rem - half).abs() <= SLACK {
        return None;
    }
    let (mut s, mut e) = ((q + (rem > half) as i128) as u64, k - FRAC as i32 + shift as i32);
    if s == 1 << 53 {
        s >>= 1;
        e += 1;
    }
    Some((s, e))
}

/// `n * (s * 2^e)` rounded to the nearest IEEE double (ties to even), exactly as a
/// floating-point multiplication of the two values does.
pub fn mul_double(n: u64, (s, e): (u64, i32)) -> (u64, i32) {
    let p = n as u128 * s as u128;
    if p == 0 {
        return (0, 0);
    }
    let bits = 128 - p.leading_zeros();
    if bits <= 53 {
        return ((p << (53 - bits)) as u64, e - (53 - bits) as i32);
    }
    let shift = bits - 53;
    let (q, rem, half) = (p >> shift, p & ((1 << shift) - 1), 1u128 << (shift - 1));
    let mut q = q + (rem > half || (rem == half && q & 1 == 1)) as u128;
    let mut e = e + shift as i32;
    if q == 1 << 53 {
        q >>= 1;
        e += 1;
    }
    (q as u64, e)
}

/// `s * 2^e` in Q96. `None` if it does not fit or would lose bits.
pub fn double_to_fixed((s, e): (u64, i32)) -> Option<i128> {
    let shift = e + FRAC as i32;
    if s == 0 {
        Some(0)
    } else if shift >= 0 {
        (shift <= 63).then(|| (s as i128) << shift)
    } else {
        let s = s as i128;
        let d = (-shift) as u32;
        (d < 64 && s & ((1 << d) - 1) == 0).then(|| s >> d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference series for the constants, computed independently of the tables.
    fn atan_inv(n: i128) -> i128 {
        let (mut term, mut sum, mut k, mut sign) = (ONE / n, 0, 1, 1);
        while term != 0 {
            sum += sign * (term / k);
            term /= n * n;
            k += 2;
            sign = -sign;
        }
        sum
    }

    #[test]
    fn constants_match_their_series() {
        let half_pi = (16 * atan_inv(5) - 4 * atan_inv(239)) / 2;
        assert!((half_pi - HALF_PI_HI).abs() < 64);
        assert!((2 * atanh(ONE / 3) - LN2_HI).abs() < 64);
    }

    #[test]
    fn multiplies() {
        assert_eq!(mul(3 * ONE, ONE / 2), 3 * ONE / 2);
        assert_eq!(mul(-3 * ONE, ONE / 2), -3 * ONE / 2);
        assert_eq!(mul(ONE << 15, ONE << 15), ONE << 30);
        assert_eq!(mul(1, 1), 0);
    }

    #[test]
    fn elementary_functions() {
        assert_eq!(ln(1), 0);
        assert!((ln(2) - LN2_HI).abs() < 4);
        assert!((ln(1 << 20) - 20 * LN2_HI).abs() < 64);
        assert_eq!(exp(0), Some((ONE, 0)));
        // e^(ln 8) = 8.
        let (m, k) = exp(ln(8)).unwrap();
        assert!(((m << k) - 8 * ONE).abs() < 1 << 10);
        assert_eq!(exp(1001 * ONE), None);
        assert_eq!(cos_sin(0), (ONE, 0));
        let (c, s) = cos_sin(2 * HALF_PI_HI);
        assert!((c + ONE).abs() < 1 << 10 && s.abs() < 1 << 10);
        // cos(1) * 2^30 = 580145183.414..., sin(1) * 2^30 = 903522590.070...
        let (c, s) = cos_sin(ONE);
        assert_eq!(round_fixed(c, 30), Some(580_145_183));
        assert_eq!(round_fixed(s, 30), Some(903_522_590));
        // 3 / 2 pi: cos = 0, sin = -1.
        let (c, s) = cos_sin(3 * HALF_PI_HI);
        assert!(c.abs() < 1 << 10 && (s + ONE).abs() < 1 << 10);
    }

    #[test]
    fn rounds() {
        let half = ONE >> 31; // half of 2^-30
        assert_eq!(round_fixed(ONE + half + (1 << 30), 30), Some((1 << 30) + 1));
        assert_eq!(round_fixed(ONE + half - (1 << 30), 30), Some(1 << 30));
        assert_eq!(round_fixed(ONE + half - 1, 30), None);
        assert_eq!(round_fixed(ONE + half, 30), None);
        assert_eq!(round_fixed(-(ONE >> 2), 30), Some(-(1 << 28)));
        assert_eq!(round_double((ONE, 0)), Some((1 << 52, -52)));
        assert_eq!(round_double((ONE, -3)), Some((1 << 52, -55)));
        assert_eq!(mul_double(3, (1 << 52, -52)), (3 << 51, -51));
        // (2^53 - 1) * 3 needs rounding: 3 * 2^53 - 3 -> nearest multiple of 4, ties to even.
        assert_eq!(mul_double(3, ((1 << 53) - 1, 0)), ((3 << 51) - 1, 2));
        assert_eq!(double_to_fixed((1 << 52, -52)), Some(ONE));
        assert_eq!(double_to_fixed((3, -97)), None);
    }
}

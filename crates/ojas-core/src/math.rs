//! Scalar special functions shared by the CPU oracle and host-side model code.

/// GELU, exact form: `0.5x(1 + erf(x/√2))`, PyTorch's `nn.GELU()` default and the
/// `gelu` activation of ModernBERT. Evaluated in f64 through [`erf`];
/// `ojas_cpu::cpu_math` re-exports it as the oracle for the Metal `ffn_act(g, 3u)`.
pub fn gelu_erf(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// The error function in f64, accurate to ~1e-14 over the real line.
///
/// The Maclaurin series for |x| < 2.5, where its largest term stays below 10^2 so
/// f64 cancellation costs at most two digits; beyond that `1 - erfc(x)`, with erfc
/// from its continued fraction (modified Lentz), which converges fastest exactly
/// where the series stops being usable. Independent of the Metal kernel's
/// Abramowitz-Stegun approximation, so the two can check each other.
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 2.5 {
        let (x2, mut term, mut sum) = (a * a, a, a);
        let mut n = 0.0f64;
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -x2 / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else {
        // erfc(a) = exp(-a²)/√π · 1/(a + (1/2)/(a + 1/(a + (3/2)/(a + 2/(a + ...)))))
        let tiny = 1e-300f64;
        let mut f = a;
        let (mut c, mut d) = (a, 0.0f64);
        for k in 1..200 {
            let an = k as f64 * 0.5;
            d = a + an * d;
            d = if d.abs() < tiny { 1.0 / tiny } else { 1.0 / d };
            c = a + an / c;
            if c.abs() < tiny { c = tiny; }
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 { break; }
        }
        1.0 - (-a * a).exp() / (f * std::f64::consts::PI.sqrt())
    };
    r.copysign(x)
}

#[cfg(test)]
mod erf_tests {
    use super::{erf, gelu_erf};

    /// Reference values from Abramowitz & Stegun Table 7.1 and mpmath at 20 digits.
    const TABLE: &[(f64, f64)] = &[
        (0.0, 0.0),
        (0.1, 0.112_462_916_018_284_9),
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (1.3, 0.934_007_944_940_652_4),
        (2.0, 0.995_322_265_018_952_7),
        (2.4999, 0.999_592_830_099_666_5),
        (2.5, 0.999_593_047_982_555),
        (3.0, 0.999_977_909_503_001_4),
        (4.0, 0.999_999_984_582_742_1),
        (6.0, 1.0),
    ];

    #[test]
    fn matches_reference_values_on_both_branches() {
        for &(x, want) in TABLE {
            for (xs, ws) in [(x, want), (-x, -want)] {
                let got = erf(xs);
                assert!((got - ws).abs() < 1e-13, "erf({xs}) = {got}, want {ws}");
            }
        }
    }

    #[test]
    fn gelu_erf_matches_closed_form_points() {
        // 0.5·x·(1 + erf(x/√2)) at x = 1, -1, 3.
        for (x, want) in [(1.0f32, 0.841_344_746_068_542_9f64), (-1.0, -0.158_655_253_931_457),
                          (3.0, 2.995_950_305_905_11)] {
            let got = gelu_erf(x) as f64;
            assert!((got - want).abs() < 1e-6, "gelu_erf({x}) = {got}, want {want}");
        }
    }
}

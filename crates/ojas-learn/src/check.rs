//! Gradient checking on the CPU reference: analytic gradients from the tape
//! against central finite differences of the loss.

use crate::cpu::Cpu;
use crate::tape::{Tape, Var};

/// Worst gradient mismatch found by [`gradcheck`].
#[derive(Debug)]
pub struct Mismatch {
    pub input: usize,
    pub index: usize,
    pub analytic: f32,
    pub numeric: f32,
    /// |analytic − numeric| / max(1, |numeric|)
    pub err: f32,
}

/// Check d loss / d input for every element of every input. `f` builds the
/// loss from the inputs (as tape vars) and must be deterministic.
pub fn gradcheck(inputs: &[(Vec<f32>, Vec<usize>)], eps: f32, f: impl Fn(&mut Tape<Cpu>, &[Var]) -> Var) -> Mismatch {
    let be = Cpu;
    let loss_at = |vals: &[(Vec<f32>, Vec<usize>)]| -> f64 {
        let mut t = Tape::new(&be);
        let vs: Vec<Var> = vals.iter().map(|(d, s)| t.input(d, s)).collect();
        let l = f(&mut t, &vs);
        t.value(l)[0] as f64
    };
    let mut t = Tape::new(&be);
    let vs: Vec<Var> = inputs.iter().map(|(d, s)| t.var(d, s)).collect();
    let l = f(&mut t, &vs);
    t.backward(l).expect("backward");
    let mut worst = Mismatch { input: 0, index: 0, analytic: 0.0, numeric: 0.0, err: 0.0 };
    for (k, v) in vs.iter().enumerate() {
        let analytic = t.grad_vec(*v).unwrap_or_else(|| vec![0.0; inputs[k].0.len()]);
        for i in 0..inputs[k].0.len() {
            let mut plus = inputs.to_vec();
            plus[k].0[i] += eps;
            let mut minus = inputs.to_vec();
            minus[k].0[i] -= eps;
            let numeric = ((loss_at(&plus) - loss_at(&minus)) / (2.0 * eps as f64)) as f32;
            let err = (analytic[i] - numeric).abs() / numeric.abs().max(1.0);
            if err > worst.err {
                worst = Mismatch { input: k, index: i, analytic: analytic[i], numeric, err };
            }
        }
    }
    worst
}

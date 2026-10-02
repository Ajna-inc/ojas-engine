//! Document cache: a span of a prompt (a page, a document) kept as its attention KV
//! rows, so a later prompt holding the same span at another position reuses it.
//!
//! The prompt-prefix cache is exact because it only reuses what follows an identical
//! prefix. A document reused after different text is not exact, and this cache is
//! opt-in for that reason:
//!
//! - Keys are stored rotated. Rotating the rotary dims again by the position
//!   difference gives the key at the new position, up to f16 rounding. Values carry
//!   no position and are copied as they are.
//! - The rows were computed after the earlier text, so the decoder recomputes the
//!   first tokens of the span in the new context.
//! - Recurrent state cannot be moved at all. The decoder rebuilds it by processing
//!   the end of the span again, starting from the state after the new prefix, so the
//!   span's head reaches the recurrent layers only through attention.
//!
//! This module holds the spans and does the rotation; placing them is the decoder's.

use super::prefix_cache::{chain, ROOT};
use std::collections::HashMap;
use std::sync::Arc;

struct Doc {
    tokens: Box<[u32]>,
    /// Position of the span's first token when its rows were computed.
    base: usize,
    kv: Arc<Vec<u8>>,
    last_used: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DocStats {
    pub(crate) docs: usize,
    pub(crate) bytes: usize,
    pub(crate) budget: usize,
    pub(crate) hits: u64,
    pub(crate) reused_tokens: u64,
}

/// Spans by content, least recently used evicted first, within a byte budget.
pub(crate) struct DocCache {
    docs: HashMap<u64, Doc>,
    budget: usize,
    used: usize,
    clock: u64,
    hits: u64,
    reused_tokens: u64,
}

impl DocCache {
    pub(crate) fn new(budget: usize) -> Self {
        DocCache { docs: HashMap::new(), budget, used: 0, clock: 0, hits: 0, reused_tokens: 0 }
    }

    pub(crate) fn enabled(&self) -> bool { self.budget > 0 }

    pub(crate) fn stats(&self) -> DocStats {
        DocStats { docs: self.docs.len(), bytes: self.used, budget: self.budget, hits: self.hits,
                   reused_tokens: self.reused_tokens }
    }

    /// The rows cached for exactly `tokens`, and the position they were computed at.
    pub(crate) fn get(&mut self, tokens: &[u32]) -> Option<(Arc<Vec<u8>>, usize)> {
        self.clock += 1;
        let clock = self.clock;
        let doc = self.docs.get_mut(&chain(ROOT, tokens)).filter(|d| *d.tokens == *tokens)?;
        doc.last_used = clock;
        Some((doc.kv.clone(), doc.base))
    }

    pub(crate) fn contains(&self, tokens: &[u32]) -> bool {
        self.docs.get(&chain(ROOT, tokens)).is_some_and(|d| *d.tokens == *tokens)
    }

    /// Count a reuse of `tokens` cached tokens.
    pub(crate) fn record(&mut self, tokens: usize) {
        self.hits += 1;
        self.reused_tokens += tokens as u64;
    }

    /// Keep the rows of `tokens`, computed with its first token at `base`. False when
    /// they cannot fit even with everything else evicted.
    pub(crate) fn insert(&mut self, tokens: &[u32], base: usize, kv: Vec<u8>) -> bool {
        if kv.len() > self.budget { return false; }
        let key = chain(ROOT, tokens);
        if let Some(old) = self.docs.remove(&key) { self.used -= old.kv.len(); }
        while self.used + kv.len() > self.budget {
            let oldest = *self.docs.iter().min_by_key(|(_, d)| d.last_used).unwrap().0;
            self.used -= self.docs.remove(&oldest).unwrap().kv.len();
        }
        self.clock += 1;
        self.used += kv.len();
        self.docs.insert(key, Doc { tokens: tokens.into(), base, kv: Arc::new(kv), last_used: self.clock });
        true
    }
}

/// How one layer's keys are rotated: `rd` of each head's `hd` dims, paired NeoX
/// style as `(j, j + rd/2)`, at frequency `base^(-2j/rd)`. The convention of the
/// `rope_qk_store` kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rope {
    pub(crate) hd: usize,
    pub(crate) rd: usize,
    pub(crate) base: f32,
}

/// Rotate f16 key rows (`kvdim` halves each) by `delta` positions.
pub(crate) fn rotate_keys(rows: &mut [u8], kvdim: usize, rope: Rope, delta: i64) {
    if delta == 0 { return; }
    let half_rd = rope.rd / 2;
    let turn: Vec<(f32, f32)> = (0..half_rd).map(|j| {
        let freq = 1.0 / (rope.base as f64).powf(2.0 * j as f64 / rope.rd as f64);
        let (s, c) = (delta as f64 * freq).sin_cos();
        (s as f32, c as f32)
    }).collect();
    let (halves, _) = rows.as_chunks_mut::<2>();
    for row in halves.chunks_exact_mut(kvdim) {
        for head in row.chunks_exact_mut(rope.hd) {
            for (j, &(s, c)) in turn.iter().enumerate() {
                let x0 = half::f16::from_le_bytes(head[j]).to_f32();
                let x1 = half::f16::from_le_bytes(head[j + half_rd]).to_f32();
                head[j] = half::f16::from_f32(x0 * c - x1 * s).to_le_bytes();
                head[j + half_rd] = half::f16::from_f32(x0 * s + x1 * c).to_le_bytes();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROPE: Rope = Rope { hd: 8, rd: 4, base: 10_000.0 };

    fn halves(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect() }

    fn floats(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<2>().0.iter().map(|&h| half::f16::from_le_bytes(h).to_f32()).collect()
    }

    /// The kernel's rotation of one head at `pos`, in f32.
    fn rope_at(head: &[f32], pos: f64) -> Vec<f32> {
        let mut out = head.to_vec();
        for j in 0..ROPE.rd / 2 {
            let a = pos / (ROPE.base as f64).powf(2.0 * j as f64 / ROPE.rd as f64);
            let (s, c) = (a.sin() as f32, a.cos() as f32);
            let (x0, x1) = (head[j], head[j + ROPE.rd / 2]);
            out[j] = x0 * c - x1 * s;
            out[j + ROPE.rd / 2] = x0 * s + x1 * c;
        }
        out
    }

    #[test]
    fn rotating_by_the_difference_moves_a_key_to_the_new_position() {
        let raw: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
        let at = |pos: f64| -> Vec<f32> { raw.chunks(ROPE.hd).flat_map(|h| rope_at(h, pos)).collect() };
        let mut rows = halves(&at(40.0));
        rotate_keys(&mut rows, 16, ROPE, 1200 - 40);
        for (got, want) in floats(&rows).iter().zip(at(1200.0)) {
            assert!((got - want).abs() < 4e-3, "{got} vs {want}");
        }
        let unrotated = floats(&rows)[ROPE.rd..ROPE.hd].to_vec();
        assert_eq!(unrotated, floats(&halves(&raw))[ROPE.rd..ROPE.hd], "dims past rd are not rotated");
    }

    #[test]
    fn rotating_back_restores_the_rows() {
        let v: Vec<f32> = (0..32).map(|i| (i as f32 * 1.3).cos()).collect();
        let mut rows = halves(&v);
        rotate_keys(&mut rows, 16, ROPE, 517);
        rotate_keys(&mut rows, 16, ROPE, -517);
        for (got, want) in floats(&rows).iter().zip(floats(&halves(&v))) {
            assert!((got - want).abs() < 4e-3);
        }
    }

    #[test]
    fn spans_are_found_by_content_and_evicted_oldest_first() {
        let mut c = DocCache::new(30);
        let (a, b, d) = ([1, 2, 3], [4, 5], [6]);
        assert!(c.insert(&a, 10, vec![1; 10]) && c.insert(&b, 20, vec![2; 10]) && c.insert(&d, 0, vec![3; 10]));
        assert_eq!(c.get(&a).map(|(kv, base)| (kv[0], base)), Some((1, 10)));
        assert!(c.insert(&[7, 8], 5, vec![4; 10]), "evicts b, the least recently used");
        assert!(c.get(&b).is_none() && c.get(&a).is_some() && c.get(&d).is_some());
        assert!(c.get(&[1, 2]).is_none(), "a span matches only its exact tokens");
        assert!(!c.insert(&[9], 0, vec![0; 31]));
        assert_eq!(c.stats().bytes, 30);
    }
}

//! The images of a prompt: where their rows go, and how those rows reach the model.
//!
//! An image is a run of placeholder ids in the prompt. Its rows come from the
//! model's own vision tower and enter through `Model::prefill_rows`, beside the
//! text rows of the same chunk. An image is encoded when prompt processing first
//! reaches it, on the thread that owns the model, so a prompt whose images the
//! prefix cache already covers never runs the tower.

use ojas_core::Model;
use std::sync::OnceLock;

/// One image: the prompt rows it fills and the pixels they are encoded from.
pub struct PromptImage {
    /// Prompt position of its first row.
    pub at: usize,
    /// Rows it fills.
    pub tokens: usize,
    /// Preprocessed pixels, planar CHW, and their size: `Model::encode_image`'s input.
    pub pixels: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// The tower's rows, once encoded; `None` when it failed.
    rows: OnceLock<Option<Vec<f32>>>,
}

impl PromptImage {
    pub fn new(at: usize, tokens: usize, pixels: Vec<f32>, width: usize, height: usize) -> Self {
        PromptImage { at, tokens, pixels, width, height, rows: OnceLock::new() }
    }

    /// The image's rows, `tokens * hidden_dim` values, encoding it on first use.
    fn rows<M: Model + ?Sized>(&self, model: &M) -> Option<&[f32]> {
        self.rows.get_or_init(|| match model.encode_image(&self.pixels, self.width, self.height) {
            Some(Ok(rows)) if rows.len() == self.tokens * model.hidden_dim() && rows.iter().all(|v| v.is_finite()) => Some(rows),
            Some(Ok(rows)) => {
                tracing::warn!("the vision tower returned {} values for {} rows, or values that are not finite", rows.len(), self.tokens);
                None
            }
            Some(Err(e)) => {
                tracing::warn!("the vision tower failed: {e:#}");
                None
            }
            None => None,
        }).as_deref()
    }
}

/// What a prompt with images adds to its token ids.
pub struct PromptImages {
    /// One entry per prompt token, for `Model::set_prefix_keys`: what the row holds.
    pub keys: Vec<u32>,
    /// A rotary coordinate per prompt token, when the model places rows by one.
    pub pos3: Option<Vec<[u32; 4]>>,
    /// The end of the last image. Chunks that start before it are prefilled as rows;
    /// the rest of the prompt is ordinary text.
    pub until: usize,
    pub images: Vec<PromptImage>,
}

impl PromptImages {
    /// Prefill `tokens`, the prompt's rows from `base` on, their images encoded as
    /// needed. False when an image could not be encoded or the model has no path
    /// for such rows.
    pub(crate) fn prefill<M: Model + ?Sized>(&self, model: &M, tokens: &[u32], base: usize) -> bool {
        let (end, d) = (base + tokens.len(), model.hidden_dim());
        if end > self.keys.len() || self.pos3.as_ref().is_some_and(|p| end > p.len()) { return false; }
        let mut given = Vec::new();
        for img in &self.images {
            let (lo, hi) = (img.at.max(base), (img.at + img.tokens).min(end));
            if lo >= hi { continue; }
            let Some(rows) = img.rows(model) else { return false };
            given.push((lo - base, &rows[(lo - img.at) * d..(hi - img.at) * d]));
        }
        model.prefill_rows(tokens, &self.keys[base..end], base, &given, self.pos3.as_deref().map(|p| &p[base..end]))
    }
}

/// Start a prompt's prefix reuse: by key for a prompt with images, by id otherwise.
/// Returns the leading tokens of `pre` already in place. A model that cannot reuse
/// by key starts such a prompt as an unrelated sequence, since its ids match any
/// other image's.
pub(crate) fn reuse_start<M: Model + ?Sized>(model: &M, images: Option<&PromptImages>, pre: &[u32]) -> usize {
    let keyed = model.set_prefix_keys(images.map_or(&[], |i| &i.keys));
    if images.is_some() && !keyed {
        model.reset_session();
        return 0;
    }
    model.reuse_prefix_len(pre).min(pre.len())
}

/// Prefill one chunk of a prompt, as rows where it lies among the images.
pub(crate) fn prefill_chunk<M: Model + ?Sized>(model: &M, images: Option<&PromptImages>, tokens: &[u32], base: usize) -> bool {
    match images {
        Some(i) if base < i.until => i.prefill(model, tokens, base),
        _ => {
            model.prefill(tokens, base);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// Records what reaches the model: each `prefill_rows` call as (base, rows,
    /// given runs with their first value), and each plain prefill.
    #[derive(Default)]
    struct Recorder {
        keyed: bool,
        encodes: Cell<usize>,
        keys: RefCell<Vec<u32>>,
        rows: RefCell<Vec<(usize, usize, Vec<(usize, usize, f32)>)>>,
        text: RefCell<Vec<(usize, usize)>>,
        resets: Cell<usize>,
    }

    impl Model for Recorder {
        fn n_layers(&self) -> usize { 1 }
        fn hidden_dim(&self) -> usize { 2 }
        fn forward_id(&self, _: u32, _: usize) -> u32 { 0 }
        fn prefill(&self, t: &[u32], base: usize) { self.text.borrow_mut().push((base, t.len())); }
        fn reset_session(&self) { self.resets.set(self.resets.get() + 1); }
        fn reuse_prefix_len(&self, _: &[u32]) -> usize { 4 }
        fn set_prefix_keys(&self, keys: &[u32]) -> bool {
            *self.keys.borrow_mut() = keys.to_vec();
            self.keyed
        }
        fn encode_image(&self, img: &[f32], w: usize, h: usize) -> Option<anyhow::Result<Vec<f32>>> {
            self.encodes.set(self.encodes.get() + 1);
            // Row r holds [first pixel + r, 0].
            Some(Ok((0..w * h).flat_map(|r| [img[0] + r as f32, 0.0]).collect()))
        }
        fn prefill_rows(&self, t: &[u32], keys: &[u32], base: usize, given: &[(usize, &[f32])], pos3: Option<&[[u32; 4]]>) -> bool {
            assert!(keys.len() == t.len() && pos3.is_some_and(|p| p.len() == t.len()));
            self.rows.borrow_mut().push((base, t.len(), given.iter().map(|&(at, x)| (at, x.len() / 2, x[0])).collect()));
            true
        }
    }

    /// Ten rows: an image of 2x2 at 2..6 and one of 1x2 at 7..9.
    fn prompt() -> PromptImages {
        PromptImages {
            keys: (0..10).collect(), pos3: Some(vec![[0; 4]; 10]), until: 9,
            images: vec![PromptImage::new(2, 4, vec![10.0; 12], 2, 2), PromptImage::new(7, 2, vec![20.0; 6], 1, 2)],
        }
    }

    #[test]
    fn each_chunk_carries_the_image_rows_that_fall_inside_it() {
        let (m, p) = (Recorder::default(), prompt());
        assert!(prefill_chunk(&m, Some(&p), &[0; 4], 0));
        assert!(prefill_chunk(&m, Some(&p), &[0; 4], 4));
        assert!(prefill_chunk(&m, Some(&p), &[0; 1], 8));
        assert!(prefill_chunk(&m, Some(&p), &[0; 1], 9));
        // Rows 2..4 of the first chunk are the image's first two; the second chunk
        // begins with its last two and holds the second image's first row at 7.
        assert_eq!(*m.rows.borrow(), vec![(0, 4, vec![(2, 2, 10.0)]), (4, 4, vec![(0, 2, 12.0), (3, 1, 20.0)]),
                                          (8, 1, vec![(0, 1, 21.0)])]);
        assert_eq!(*m.text.borrow(), vec![(9, 1)], "past the last image the prompt is text");
        assert_eq!(m.encodes.get(), 2, "an image is encoded once, however many chunks it spans");
    }

    #[test]
    fn an_image_no_chunk_reaches_is_never_encoded() {
        let (m, p) = (Recorder::default(), prompt());
        assert!(prefill_chunk(&m, Some(&p), &[0; 2], 6));
        assert_eq!(m.encodes.get(), 1, "only the second image lies in rows 6..8");
    }

    #[test]
    fn a_model_without_keys_starts_an_image_prompt_from_nothing() {
        let (m, p) = (Recorder::default(), prompt());
        assert_eq!(reuse_start(&m, Some(&p), &[0; 9]), 0);
        assert_eq!((m.resets.get(), m.keys.borrow().len()), (1, 10));
        // A prompt of text clears the keys and reuses by id.
        assert_eq!(reuse_start(&m, None, &[0; 9]), 4);
        assert!(m.keys.borrow().is_empty());
        let keyed = Recorder { keyed: true, ..Default::default() };
        assert_eq!((reuse_start(&keyed, Some(&p), &[0; 9]), keyed.resets.get()), (4, 0));
    }

    #[test]
    fn a_failed_encode_or_a_short_prompt_fails_the_chunk() {
        struct NoTower;
        impl Model for NoTower {
            fn n_layers(&self) -> usize { 1 }
            fn hidden_dim(&self) -> usize { 2 }
            fn forward_id(&self, _: u32, _: usize) -> u32 { 0 }
            fn prefill(&self, _: &[u32], _: usize) {}
        }
        assert!(!prefill_chunk(&NoTower, Some(&prompt()), &[0; 4], 0));
        assert!(!prefill_chunk(&Recorder::default(), Some(&prompt()), &[0; 4], 8), "rows past the prompt's keys");
    }
}

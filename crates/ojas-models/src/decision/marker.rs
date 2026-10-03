//! The marker readout: an encoder whose head scores each option at the marker
//! token that opens it.
//!
//! The rendered prompt is
//!
//! ```text
//! [CLS] question [SEP] ([MASK] option)* [SEP] state [SEP]
//! ```
//!
//! with the file's own special tokens. The question and the options are cut to the
//! head's token budget (`decision.max_head_tokens`) the way the model was trained:
//! each option to at most 48 tokens after its marker, then all of them evenly when
//! they leave the question fewer than 16 tokens, and the question to what remains.
//! Every question of a request is one sequence, and all of them run in one GPU pass
//! ([`DecoderGpu::marker_head_forward`]); the host applies the scorer's last
//! projection (`cls.output`).

use super::json::Json;
use super::template::{OptionView, PromptInput, Template};
use super::{invalid, read_f32, Question, Request, Scored};
use crate::decoder::DecoderGpu;
use anyhow::{ensure, Context, Result};
use ojas_formats::gguf::Gguf;
use ojas_metal::MetalGpu;
use ojas_tokenize::Bpe;

/// Most tokens of an option's text.
const MAX_OPTION_TOKENS: usize = 48;
/// Head tokens the options leave the question before they are shrunk.
const MIN_QUESTION_TOKENS: usize = 16;
/// Most tokens in one GPU pass over several requests. Past a few thousand rows the
/// f32 activations outgrow the caches and the GEMMs turn memory-bound, so larger
/// passes answer fewer tokens per second.
const PASS_TOKENS: usize = 4096;

/// Token sequences, one per question, and each one's option marker positions.
type Sequences = (Vec<Vec<u32>>, Vec<Vec<usize>>);

pub(super) struct MarkerHead<'a> {
    dec: DecoderGpu<'a>,
    tok: Bpe,
    template: Template,
    marker: u32,
    sep: u32,
    /// The marker token's text, which the request's strings may not contain.
    marker_text: String,
    max_head_tokens: usize,
    max_positions: usize,
    out_w: Vec<f32>,
    out_b: f32,
}

impl<'a> MarkerHead<'a> {
    pub(super) fn load(gpu: &'a MetalGpu, g: &mut Gguf, template: Template, prefix: &str) -> Result<Self> {
        let id = |k: &str| g.meta_u32(&format!("tokenizer.ggml.{k}")).with_context(|| format!("no tokenizer.ggml.{k}"));
        let (marker, sep) = (id("mask_token_id")?, id("seperator_token_id")?);
        let max_head_tokens = g.meta_u32(&format!("{prefix}max_head_tokens")).unwrap_or(0) as usize;
        ensure!(max_head_tokens > 0, "no valid {prefix}max_head_tokens");
        let tok = Bpe::from_gguf(g);
        let marker_text = tok.decode(marker as usize);
        ensure!(!marker_text.is_empty(), "the marker token has no text");
        let out_w = read_f32(g, "cls.output.weight")?;
        let out_b = read_f32(g, "cls.output.bias")?;

        // The encoder keeps no KV cache, so the decoder context is minimal. Weights are
        // held in f16: every token goes through batched GEMMs, which the file's
        // quantized blocks would serve one row at a time, and requantizing them per row
        // moves the probabilities further from the file's own numbers.
        // An encoder takes no projector or draft head; files beside it belong to
        // other models.
        g.sidecars = false;
        let dec = DecoderGpu::load(gpu, g, 64, 0, None, None)?;
        ensure!(dec.has_marker_head(), "the encoder loaded without its decision head");
        let d = dec.text_encoder_width().context("not a text encoder")?;
        ensure!(out_w.len() == d && out_b.len() == 1, "cls.output must project {d} values to one score");
        let max_positions = dec.text_encoder_max_positions().unwrap_or(usize::MAX);
        Ok(MarkerHead { dec, tok, template, marker, sep, marker_text, max_head_tokens, max_positions, out_w, out_b: out_b[0] })
    }

    /// `q`'s token sequence and the positions of its option markers.
    fn sequence(&self, state: &Json, q: &Question) -> Result<(Vec<u32>, Vec<usize>)> {
        let clean = |j: &Json| j.map_strings(&|s: &str| s.replace(&self.marker_text, " "));
        let (instructions, state) = (clean(&q.instructions), clean(state));
        let options: Vec<(String, Json)> = q.options.iter()
            .map(|o| (o.key.replace(&self.marker_text, " "), clean(&o.description))).collect();
        let id = q.id.replace(&self.marker_text, " ");
        let prompt = self.template.render(&PromptInput {
            id: &id, kind: q.kind.name(), instructions: &instructions, state: &state,
            options: options.iter().map(|(key, description)| OptionView { key, description, label: None }).collect(),
            images: Vec::new(),
        })?;
        let tokens: Vec<u32> = self.tok.encode(&prompt).into_iter().map(|t| t as u32).collect();
        let (ids, markers) = self.fit(&tokens, q.options.len())?;
        if ids.len() > self.max_positions {
            return Err(invalid(format!("questions.{}: the prompt is {} tokens, this model reads at most {}",
                q.id, ids.len(), self.max_positions)));
        }
        Ok((ids, markers))
    }

    /// Cut the question and options of a rendered prompt to the head's budget.
    fn fit(&self, tokens: &[u32], n_options: usize) -> Result<(Vec<u32>, Vec<usize>)> {
        let found: Vec<usize> = (0..tokens.len()).filter(|&i| tokens[i] == self.marker).collect();
        ensure!(found.len() == n_options && found[0] >= 2 && tokens[found[0] - 1] == self.sep
                && tokens.last() == Some(&self.sep), "unexpected layout of the decision prompt");
        let head_end = found[0] - 1;
        let opts_end = found[n_options - 1] + tokens[found[n_options - 1]..].iter().position(|&t| t == self.sep)
            .context("unexpected layout of the decision prompt")?;
        ensure!(opts_end + 1 < tokens.len(), "unexpected layout of the decision prompt");

        let mut options: Vec<&[u32]> = (0..n_options)
            .map(|i| &tokens[found[i]..if i + 1 < n_options { found[i + 1] } else { opts_end }]).collect();
        let cap = |options: &mut Vec<&[u32]>, most: usize| -> usize {
            for o in options.iter_mut() { *o = &o[..o.len().min(most)]; }
            options.iter().map(|o| o.len()).sum()
        };
        let mut used = cap(&mut options, MAX_OPTION_TOKENS + 1);
        if used + MIN_QUESTION_TOKENS > self.max_head_tokens {
            used = cap(&mut options, (self.max_head_tokens.saturating_sub(MIN_QUESTION_TOKENS) / n_options).max(4));
        }
        let question_tokens = self.max_head_tokens.saturating_sub(used).max(8);

        let mut ids = Vec::with_capacity(tokens.len());
        ids.push(tokens[0]);
        ids.extend_from_slice(&tokens[1..head_end.min(1 + question_tokens)]);
        ids.push(self.sep);
        let mut markers = Vec::with_capacity(n_options);
        for o in &options {
            markers.push(ids.len());
            ids.extend_from_slice(o);
        }
        ids.extend_from_slice(&tokens[opts_end..]);
        Ok((ids, markers))
    }

    fn sequences(&self, state: &Json, questions: &[Question]) -> Result<Sequences> {
        Ok(questions.iter().map(|q| self.sequence(state, q)).collect::<Result<Vec<_>>>()?.into_iter().unzip())
    }

    /// Each request's option scores, in one option order. The questions of every
    /// request run together, in GPU passes of at most [`PASS_TOKENS`] tokens; a
    /// request is never split across passes, and one that fails to build fails alone.
    pub(super) fn scores(&self, reqs: &[&Request]) -> Vec<Result<Scored>> {
        let built: Vec<Result<Sequences>> = reqs.iter().map(|r| self.sequences(&r.state, &r.questions)).collect();
        let mut out: Vec<Option<Result<Scored>>> = built.iter().map(|_| None).collect();
        let mut pass: Vec<usize> = Vec::new();
        let mut pass_tokens = 0;
        for (i, b) in built.iter().enumerate() {
            let Ok((seqs, _)) = b else { continue };
            let tokens: usize = seqs.iter().map(Vec::len).sum();
            if !pass.is_empty() && pass_tokens + tokens > PASS_TOKENS {
                self.run(reqs, &built, &pass, &mut out);
                pass.clear();
                pass_tokens = 0;
            }
            pass.push(i);
            pass_tokens += tokens;
        }
        if !pass.is_empty() { self.run(reqs, &built, &pass, &mut out); }
        built.into_iter().zip(out).map(|(b, o)| match (b, o) {
            (Err(e), _) => Err(e),
            (Ok(_), Some(r)) => r,
            (Ok(_), None) => unreachable!("every built request is in a pass"),
        }).collect()
    }

    /// One GPU pass over the requests `pass` names. The pass's GPU time is shared
    /// among them by token count.
    fn run(&self, reqs: &[&Request], built: &[Result<Sequences>], pass: &[usize], out: &mut [Option<Result<Scored>>]) {
        let (mut seqs, mut markers, mut qtypes) = (Vec::new(), Vec::new(), Vec::new());
        for &i in pass {
            let Ok((s, m)) = &built[i] else { continue };
            seqs.extend(s.iter().cloned());
            markers.extend(m.iter().cloned());
            qtypes.extend(reqs[i].questions.iter().map(|q| q.kind.index()));
        }
        let total: usize = seqs.iter().map(Vec::len).sum();
        match self.dec.marker_head_forward(&seqs, &qtypes, &markers) {
            Err(e) => for &i in pass { out[i] = Some(Err(anyhow::anyhow!("{e:#}"))); },
            Ok(fwd) => {
                let mut hidden = fwd.scorer_hidden.into_iter();
                for &i in pass {
                    let Ok((s, _)) = &built[i] else { continue };
                    let tokens: usize = s.iter().map(Vec::len).sum();
                    let scores = hidden.by_ref().take(s.len()).map(|hs| vec![hs.iter().map(|h| {
                        self.out_b + h.iter().zip(&self.out_w).map(|(a, w)| a * w).sum::<f32>()
                    }).collect()]).collect();
                    out[i] = Some(Ok(Scored { scores, tokens, reused: 0, gpu_s: fwd.gpu_s * tokens as f64 / total as f64 }));
                }
            }
        }
    }

    /// Per-category GPU time of the encoder over the request's sequences.
    pub(super) fn profile(&self, state: &Json, questions: &[Question]) -> Result<Vec<(String, f64)>> {
        self.dec.profile_text(&self.sequences(state, questions)?.0)
    }
}

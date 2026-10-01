//! Constrained output for the CLI and the server: building the per-model token
//! vocabulary and the per-request grammar processor, and naming why generation
//! ended.

use crate::backend::ModelInfo;
use anyhow::Result;
use ojas_grammar::{GrammarProcessor, OutputFormat, TokenVocab};
use ojas_infer::FinishReason;
use ojas_tokenize::Bpe;
use std::sync::Arc;

/// Every token's bytes, shared by all constrained requests on one model.
pub fn vocab(bpe: &Bpe, info: &ModelInfo) -> Arc<TokenVocab> {
    Arc::new(TokenVocab::from_bpe(bpe, info.vocab, &info.eog))
}

/// The processor enforcing `format`, if one was asked for.
pub fn processor(format: Option<&OutputFormat>, vocab: &Arc<TokenVocab>) -> Result<Option<GrammarProcessor>> {
    let Some(f) = format else { return Ok(None) };
    Ok(Some(GrammarProcessor::new(Arc::new(f.compile()?), vocab.clone())))
}

/// OpenAI's `finish_reason`: "length" when the token budget ran out, else "stop".
pub fn finish_reason(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Length => "length",
        _ => "stop",
    }
}

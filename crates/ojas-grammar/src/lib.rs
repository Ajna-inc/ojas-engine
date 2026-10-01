//! Grammar-constrained decoding for Ojas.
//!
//! - [`Grammar::parse`] compiles a GBNF grammar.
//! - [`Grammar::from_json_schema`] compiles a JSON Schema, and [`Grammar::json`]
//!   accepts any JSON object, for OpenAI-style `response_format`.
//! - [`GrammarProcessor`] applies a grammar to generation as an
//!   [`ojas_infer::LogitProcessor`], over a [`TokenVocab`] built once per model.
//!
//! ```ignore
//! use ojas_grammar::{Grammar, GrammarProcessor, TokenVocab};
//! use ojas_infer::{EngineCore, FinishReason, StopMatcher};
//! use std::sync::Arc;
//!
//! // Once per model.
//! let vocab = Arc::new(TokenVocab::from_bpe(&bpe, n_vocab, &eog_ids));
//! // Per request.
//! let grammar = Arc::new(Grammar::from_json_schema(&schema)?);
//! let mut json = GrammarProcessor::new(grammar, vocab.clone());
//! let gen = core.generate_ex(&prompt_ids, 512, None, Some(&mut json), &mut |_, _| {}, &mut |_| true);
//! assert_eq!(gen.finish, FinishReason::Complete); // the document closed
//! ```
//!
//! Any other per-token rule is an [`ojas_infer::LogitProcessor`] of its own: edit
//! the logits in `process`, or reject candidates in `allows`. Stop strings are
//! applied to the decoded text with [`ojas_infer::StopMatcher`].

mod gbnf;
mod matcher;
mod schema;

pub use gbnf::Grammar;
pub use matcher::{GrammarProcessor, State, TokenVocab};

use anyhow::{Context, Result};

/// The output constraint a request asks for, in the forms the CLI and the server
/// accept.
#[derive(Clone, Debug)]
pub enum OutputFormat {
    /// Any JSON object (OpenAI `response_format: {"type": "json_object"}`).
    JsonObject,
    /// JSON matching a schema (`response_format: {"type": "json_schema", ...}`).
    JsonSchema(serde_json::Value),
    /// A GBNF grammar.
    Grammar(String),
}

impl OutputFormat {
    pub fn compile(&self) -> Result<Grammar> {
        match self {
            OutputFormat::JsonObject => Ok(Grammar::json()),
            OutputFormat::JsonSchema(s) => Grammar::from_json_schema(s).context("compiling the JSON schema"),
            OutputFormat::Grammar(g) => Grammar::parse(g).context("compiling the grammar"),
        }
    }
}

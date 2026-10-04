//! Training an encoder decision model from rewards, with no dataset: tasks made on
//! demand ([`tasks`]), judged by any `/v1/systemone` server ([`teacher`]), trained by
//! advantage-weighted policy gradients over each question's options ([`train`]), and
//! saved as the GGUF the decision server loads ([`export`]).

pub mod export;
pub mod memory;
pub mod tasks;
pub mod teacher;
pub mod train;

pub use crate::models::modern_bert::{LearnDecision, MarkerSeq, ModernBert};

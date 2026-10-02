//! ojas-tokenize — BPE (GPT2/Qwen/Llama/SPM families) + SentencePiece unigram
//! + chat templates. Vocab comes from ojas-formats GGUF metadata.

pub mod sentencepiece;
pub mod tokenizer;

pub use sentencepiece::Sp;
pub use tokenizer::{byte_maps, chat_eos, chat_template, chat_transcript, chatml, eog_token_ids, transcript_boundaries, transcript_span, Bpe};

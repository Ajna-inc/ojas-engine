//! ojas-core — the engine's contract layer: traits and wire types only, with no device
//! code, no model code and no heavy dependencies. Every other ojas crate implements or
//! consumes these.
//!
//! `Learner` and `Span` are peers of `Model`, so a model on any device can serve, learn
//! while serving, and join a pipeline swarm.

use anyhow::Result;

pub mod cancel;
pub mod device_fault;
pub mod config;
pub mod conv;
pub mod iq_grids;
pub mod iq_tables;
pub mod kernel;
pub mod logging;
pub mod quant_src;
pub use kernel::{Caps, KernelRuntime, KernelSpec, Manifest, Tier, VariantSet};

// ================================ device =====================================

/// A compute device (Metal, CUDA, CPU): runtime-compiled kernel families +
/// opaque buffers/encoders.
pub trait Device {
    type Buf;
    type Pipeline;
    type Enc;

    fn name(&self) -> String;
    /// Kernel source for a family key ("qwen", "attn", "moe", "train"), in this
    /// device's dialect. `None` = the family is not implemented here.
    fn kernel_source(&self, family: &str) -> Option<&'static str>;
    fn pipeline(&self, source: &str, entry: &str) -> Result<Self::Pipeline>;
    fn alloc(&self, len_f32: usize) -> Self::Buf;
    fn upload(&self, data: &[f32]) -> Self::Buf;
    fn upload_f16(&self, data: &[f32]) -> Self::Buf;
    fn read(&self, buf: &Self::Buf, out: &mut [f32]);
}

// ================================ model ======================================

/// Decode-time model interface (serving).
pub trait Model {
    /// Maximum number of input positions the allocated state can hold.
    fn context_capacity(&self) -> usize { usize::MAX }
    /// Number of input positions a committed MTP call may evaluate.
    fn mtp_verify_width(&self) -> usize { 2 }
    fn n_layers(&self) -> usize;
    fn hidden_dim(&self) -> usize;
    fn prefill(&self, tokens: &[u32], base_pos: usize);
    /// Prefill a span whose residual-stream rows are supplied directly rather than
    /// gathered from `token_embd` — how a vision encoder's output reaches the decoder.
    ///
    /// `x` is `tokens.len() * hidden_dim()` f32, row-major: row i is the residual
    /// stream at position `base_pos + i`, used verbatim, with no embedding scale, no
    /// normalization and no lookup. The rows need not be embeddings, so a caller
    /// targeting an architecture that scales gathered embeddings (Gemma's sqrt(d))
    /// applies that scale itself.
    ///
    /// `tokens` carries the placeholder ids for those positions. They are never
    /// looked up; they keep bounds checks, session bookkeeping and id-based logs
    /// correct, and keep a mixed text/image prompt in one position space. Those ids
    /// are identical across different images, so an implementation must mark an
    /// injected span non-reusable, or `reuse_prefix_len` will match one page's cache
    /// against another page's ids.
    ///
    /// `pos3` is the per-row M-RoPE coordinate `(t, h, w, e)`: one position per rope
    /// section, in the order `{arch}.rope.dimension_sections` declares the section
    /// sizes. It is four wide because that is the kernel descriptor's width and
    /// because llama.cpp's HunyuanVL XD-RoPE carries the image index in the fourth
    /// stream (`tools/mtmd/mtmd.cpp`, `pos.z = image_idx`). Following llama.cpp:
    ///
    /// * text row at sequence position `p` — `(p, p, p, 0)` (`src/llama-graph.cpp`,
    ///   `llm_graph_input_pos::set_input`). Every section reads the same number, so a
    ///   sectioned kernel reproduces plain rope bit-for-bit and text-only models are
    ///   unaffected.
    /// * image row `i` of an `nx` x `ny` patch grid starting at `pos_0` —
    ///   `(pos_0, pos_0 + i/nx, pos_0 + i%nx, 0)`: one shared temporal position plus
    ///   the patch's row and column (`mtmd_image_tokens_get_decoder_pos`,
    ///   `MTMD_POS_TYPE_MROPE`). The span consumes `max(nx, ny)` sequence positions,
    ///   not `nx*ny` (`mtmd_image_tokens_get_n_pos`), so following text starts at
    ///   `pos_0 + max(nx, ny)` and the caller advances that counter.
    ///
    /// `base_pos + i` is the KV cache row and is always contiguous, so an image span
    /// occupies `tokens.len()` consecutive slots however its coordinates are
    /// numbered; `pos3` moves only the rope angle. `None` means ordinary text rows,
    /// each at scalar position `base_pos + i`, byte-identical to the id path.
    ///
    /// Returns false when this model has no injection path (the default), so the
    /// caller can fall back to an id prefill or refuse the request instead of
    /// silently prefilling nothing. Also false for a `Some(pos3)` the model cannot
    /// honour: dropping the coordinates and prefilling anyway would look like
    /// success.
    fn prefill_embeds(&self, _tokens: &[u32], _x: &[f32], _base_pos: usize,
                      _pos3: Option<&[[u32; 4]]>) -> bool { false }
    /// Width of the rows this model's own vision tower produces, or `None` when it
    /// has none — the capability probe for the three methods below.
    ///
    /// It is `hidden_dim()` by construction (a projector that disagreed would come
    /// from a different checkpoint, and `mmproj::validate` refuses one), but it is
    /// reported rather than assumed so a host can check.
    ///
    /// A vision tower is not a second model: it shares the decoder's device, weight
    /// arena, `projm` precision flag and scratch buffers, so on the Metal backend
    /// the tower is the decoder object. `Model` is the only surface a host such as
    /// `ojas ocr` holds (`backend::with_model` hands out `&dyn Model` and nothing
    /// more), so a tower not reachable from here is not reachable at all.
    fn vision_width(&self) -> Option<usize> { None }
    /// Rows a `width` x `height` image will produce — known before encoding, so a
    /// caller can build the prompt and check the context before paying for the
    /// tower. `None` when this model has no vision tower, or when it has one but
    /// cannot encode that geometry.
    fn vision_tokens(&self, _w: usize, _h: usize) -> Option<usize> { None }
    /// Encode one already-preprocessed image: `img` is planar CHW f32
    /// (`img[c*H*W + y*W + x]`), the resize/normalize policy is the caller's, and
    /// the result is `vision_tokens(w, h) * vision_width()` f32 row-major — the
    /// input [`Model::prefill_embeds`] takes, used verbatim.
    ///
    /// `None` means this model has no tower, so a host can fall back to a portable
    /// CPU encoder; `Some(Err(_))` means it has one and the encode failed (bad
    /// image geometry, device fault), which must be reported rather than routed
    /// around. Flattening the two would make a device fault look like a missing
    /// capability.
    fn encode_image(&self, _img: &[f32], _w: usize, _h: usize) -> Option<Result<Vec<f32>>> { None }
    /// Cross-turn KV-prefix reuse: how many leading tokens of `full_prompt` are
    /// already resident and valid in the cache from a previous turn, so the caller
    /// can prefill from that offset instead of position 0. Default 0 = re-prefill
    /// everything. A non-zero return promises that the cached rows for `[0, n)` are
    /// bit-identical to what a fresh prefill of the same tokens would write; the
    /// caller relies on that for correctness.
    fn reuse_prefix_len(&self, _full_prompt: &[u32]) -> usize { 0 }
    /// Greedy-decode one token. u32::MAX = cancelled.
    fn forward_id(&self, token: u32, pos: usize) -> u32;
    fn forward_logits(&self, _token: u32, _pos: usize) -> Option<Vec<f32>> { None }
    /// Batched forward returning logits for every position — the speculative-verify
    /// primitive. `tokens[i]` is decoded at `base_pos + i`, and the weights are read
    /// once for the whole batch, so verifying M drafted tokens costs about one
    /// forward's bandwidth instead of M.
    ///
    /// `None` = this model cannot verify safely (recurrent state would need
    /// rollback, or the batched path is not implemented for its architecture), so
    /// callers must fall back to single-token decode.
    fn forward_batch_logits(&self, _tokens: &[u32], _base_pos: usize) -> Option<Vec<Vec<f32>>> { None }
    /// Batched forward returning only each position's argmax id — what a greedy
    /// speculative verify actually needs. Avoids copying vocab*M floats to the host.
    fn forward_batch_ids(&self, _tokens: &[u32], _base_pos: usize) -> Option<Vec<u32>> { None }
    /// Batched forward returning (argmax id, top-8 ids) per position. The top-8 are
    /// the Token Recycling drafter's food; None = fall back to forward_batch_ids.
    fn forward_batch_topk(&self, _tokens: &[u32], _base_pos: usize) -> Option<(Vec<u32>, Vec<u32>)> { None }

    /// Whether a NextN/MTP draft block is loaded and usable as a drafter.
    fn has_mtp(&self) -> bool { false }

    /// One self-speculative step from `cur` at `pos`, returning the committed
    /// tokens: one on rejection, up to `mtp_verify_width()` on full acceptance.
    /// The model owns the draft/verify protocol and its rollback; a rejected draft
    /// costs time, never correctness, because verify re-runs the real stack.
    fn mtp_step_committed(&self, _cur: u32, _pos: usize) -> Option<Vec<u32>> { None }
    /// Single-token greedy step that also reports the top-8 ids (recycled into the
    /// drafter during the stretches where no draft fires). None = unsupported.
    fn forward_id_topk(&self, _token: u32, _pos: usize) -> Option<(u32, [u32; 8])> { None }

    // ---- sequence slots: B independent sequences decoded per step ------------
    //
    // These five amortize weight bandwidth: a decode step reads every weight once,
    // and at B sequences it still reads them once while serving B tokens. On an
    // M2 Max the weight-read term measures 1.84x at B=4 (Q4L) and the whole token
    // ~2.0x. `max_slots()` is a small number rather than an open dial because the
    // Q4L GEMV family regresses past 4 rows on a register-occupancy cliff (1.84x
    // at 4, 1.52x at 6), under all three kernel routings.
    //
    // A slot is an independent sequence, not a token of one sequence — the
    // distinction from `forward_batch_ids`, which decodes M consecutive positions
    // of one sequence and shares its recurrent state and KV rows. Slots share the
    // weights and nothing else: each carries its own recurrent state, its own KV
    // rows and its own position counter.

    /// Slots this backend can decode concurrently. 1 = no batching, the default.
    ///
    /// A caller must treat this as a hard cap, not a hint: slot indices are direct
    /// offsets into state allocated at load time, so a slot at or past this bound
    /// has no state to address.
    fn max_slots(&self) -> usize { 1 }
    /// Decode one token for each of several independent sequences in one step.
    ///
    /// `steps` is `(slot, token, pos)`: the slot whose state to advance, the token
    /// to feed it, and the position to feed it at. All three are per-slot because
    /// slots are unrelated sequences — two slots decoding at different positions is
    /// the normal case.
    ///
    /// Returns the argmax per slot in the same order as `steps`, indexed by the
    /// entry's place in `steps` rather than by its slot number, so a caller that
    /// passed slots out of order or passed a subset reads its answers back
    /// positionally.
    ///
    /// `None` = unsupported, and the caller falls back to per-sequence
    /// `forward_id`. A backend may support this for some architectures and not
    /// others, and returns `None` for a shape it cannot batch rather than failing
    /// the request.
    fn decode_slots(&self, _steps: &[(usize, u32, usize)]) -> Option<Vec<u32>> { None }
    /// Zero slot `s`'s recurrent + KV state — `reset_session` scoped to one slot.
    ///
    /// A host serving unrelated requests through slots must call this before
    /// reusing a slot. Recurrent state is a running accumulation and does not
    /// self-heal the way positional KV does: a slot reused without a reset answers
    /// the next request from the middle of the previous one.
    fn reset_slot(&self, _s: usize) {}
    /// Prefill into a specific slot. [`Model::prefill`] and
    /// [`Model::prefill_embeds`] target slot 0, so `prefill_slot(0, ..)` and
    /// `prefill` are the same operation; the slot argument is what makes the other
    /// B-1 sequences reachable.
    ///
    /// False = this backend cannot prefill that slot (no slot support, an
    /// architecture its batched graph does not cover, or `s >= max_slots()`), so
    /// the caller must not then decode it.
    fn prefill_slot(&self, _s: usize, _tokens: &[u32], _base_pos: usize) -> bool { false }
    /// [`Model::prefill_embeds`] into a specific slot: how a vision encoder's rows
    /// enter one slot of a batch. Same row and coordinate convention as
    /// `prefill_embeds`, including the meaning of `None` vs `Some(pos3)`.
    fn prefill_embeds_slot(&self, _s: usize, _tokens: &[u32], _x: &[f32], _base_pos: usize,
                           _pos3: Option<&[[u32; 4]]>) -> bool { false }

    /// Begin an unrelated sequence: drop recurrent (SSM/GDN) state, any MTP carry,
    /// and the cross-turn reuse bookkeeping.
    ///
    /// A host that serves independent requests from one resident model must call
    /// this between them. Dense KV is positional and self-heals under a full
    /// re-prefill, so a purely dense model does not notice, but a recurrent one
    /// carries its state forward and answers the next request from the middle of
    /// the previous one — fluent nonsense rather than an error. The default is a
    /// no-op: a stateless model has nothing to reset.
    fn reset_session(&self) {}
}

/// A boxed model is a model, so a host can pick its backend at runtime — Metal on
/// Apple, CPU elsewhere — and still drive one `EngineCore`.
///
/// Every method is forwarded, including the ones with defaults. Forwarding only
/// the required methods compiles and then answers the optional ones with their
/// defaults: `has_mtp` goes false and `mtp_step_committed` goes `None`, so
/// speculative decoding disables itself for every boxed backend with no error.
/// A method added to `Model` must be added here too.
impl<T: Model + ?Sized> Model for Box<T> {
    fn context_capacity(&self) -> usize { (**self).context_capacity() }
    fn mtp_verify_width(&self) -> usize { (**self).mtp_verify_width() }
    fn n_layers(&self) -> usize { (**self).n_layers() }
    fn hidden_dim(&self) -> usize { (**self).hidden_dim() }
    fn prefill(&self, tokens: &[u32], base_pos: usize) { (**self).prefill(tokens, base_pos) }
    fn prefill_embeds(&self, tokens: &[u32], x: &[f32], base_pos: usize, pos3: Option<&[[u32; 4]]>) -> bool {
        (**self).prefill_embeds(tokens, x, base_pos, pos3)
    }
    fn vision_width(&self) -> Option<usize> { (**self).vision_width() }
    fn vision_tokens(&self, w: usize, h: usize) -> Option<usize> { (**self).vision_tokens(w, h) }
    fn encode_image(&self, img: &[f32], w: usize, h: usize) -> Option<Result<Vec<f32>>> {
        (**self).encode_image(img, w, h)
    }
    fn reuse_prefix_len(&self, full_prompt: &[u32]) -> usize { (**self).reuse_prefix_len(full_prompt) }
    fn forward_id(&self, token: u32, pos: usize) -> u32 { (**self).forward_id(token, pos) }
    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> { (**self).forward_logits(token, pos) }
    fn forward_batch_logits(&self, tokens: &[u32], base_pos: usize) -> Option<Vec<Vec<f32>>> {
        (**self).forward_batch_logits(tokens, base_pos)
    }
    fn forward_batch_ids(&self, tokens: &[u32], base_pos: usize) -> Option<Vec<u32>> {
        (**self).forward_batch_ids(tokens, base_pos)
    }
    fn forward_batch_topk(&self, tokens: &[u32], base_pos: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        (**self).forward_batch_topk(tokens, base_pos)
    }
    fn has_mtp(&self) -> bool { (**self).has_mtp() }
    fn mtp_step_committed(&self, cur: u32, pos: usize) -> Option<Vec<u32>> {
        (**self).mtp_step_committed(cur, pos)
    }
    fn forward_id_topk(&self, token: u32, pos: usize) -> Option<(u32, [u32; 8])> {
        (**self).forward_id_topk(token, pos)
    }
    fn max_slots(&self) -> usize { (**self).max_slots() }
    fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Option<Vec<u32>> {
        (**self).decode_slots(steps)
    }
    fn reset_slot(&self, s: usize) { (**self).reset_slot(s) }
    fn prefill_slot(&self, s: usize, tokens: &[u32], base_pos: usize) -> bool {
        (**self).prefill_slot(s, tokens, base_pos)
    }
    fn prefill_embeds_slot(&self, s: usize, tokens: &[u32], x: &[f32], base_pos: usize,
                           pos3: Option<&[[u32; 4]]>) -> bool {
        (**self).prefill_embeds_slot(s, tokens, x, base_pos, pos3)
    }
    fn reset_session(&self) { (**self).reset_session() }
}

/// A borrowed model is a model, so an engine can be built on a backend it does
/// not own. Same forwarding rule as `Box`: every method, defaults included.
impl<T: Model + ?Sized> Model for &T {
    fn context_capacity(&self) -> usize { (**self).context_capacity() }
    fn mtp_verify_width(&self) -> usize { (**self).mtp_verify_width() }
    fn n_layers(&self) -> usize { (**self).n_layers() }
    fn hidden_dim(&self) -> usize { (**self).hidden_dim() }
    fn prefill(&self, tokens: &[u32], base_pos: usize) { (**self).prefill(tokens, base_pos) }
    fn prefill_embeds(&self, tokens: &[u32], x: &[f32], base_pos: usize, pos3: Option<&[[u32; 4]]>) -> bool {
        (**self).prefill_embeds(tokens, x, base_pos, pos3)
    }
    fn vision_width(&self) -> Option<usize> { (**self).vision_width() }
    fn vision_tokens(&self, w: usize, h: usize) -> Option<usize> { (**self).vision_tokens(w, h) }
    fn encode_image(&self, img: &[f32], w: usize, h: usize) -> Option<Result<Vec<f32>>> {
        (**self).encode_image(img, w, h)
    }
    fn reuse_prefix_len(&self, full_prompt: &[u32]) -> usize { (**self).reuse_prefix_len(full_prompt) }
    fn forward_id(&self, token: u32, pos: usize) -> u32 { (**self).forward_id(token, pos) }
    fn forward_logits(&self, token: u32, pos: usize) -> Option<Vec<f32>> { (**self).forward_logits(token, pos) }
    fn forward_batch_logits(&self, tokens: &[u32], base_pos: usize) -> Option<Vec<Vec<f32>>> {
        (**self).forward_batch_logits(tokens, base_pos)
    }
    fn forward_batch_ids(&self, tokens: &[u32], base_pos: usize) -> Option<Vec<u32>> {
        (**self).forward_batch_ids(tokens, base_pos)
    }
    fn forward_batch_topk(&self, tokens: &[u32], base_pos: usize) -> Option<(Vec<u32>, Vec<u32>)> {
        (**self).forward_batch_topk(tokens, base_pos)
    }
    fn has_mtp(&self) -> bool { (**self).has_mtp() }
    fn mtp_step_committed(&self, cur: u32, pos: usize) -> Option<Vec<u32>> {
        (**self).mtp_step_committed(cur, pos)
    }
    fn forward_id_topk(&self, token: u32, pos: usize) -> Option<(u32, [u32; 8])> {
        (**self).forward_id_topk(token, pos)
    }
    fn max_slots(&self) -> usize { (**self).max_slots() }
    fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Option<Vec<u32>> {
        (**self).decode_slots(steps)
    }
    fn reset_slot(&self, s: usize) { (**self).reset_slot(s) }
    fn prefill_slot(&self, s: usize, tokens: &[u32], base_pos: usize) -> bool {
        (**self).prefill_slot(s, tokens, base_pos)
    }
    fn prefill_embeds_slot(&self, s: usize, tokens: &[u32], x: &[f32], base_pos: usize,
                           pos3: Option<&[[u32; 4]]>) -> bool {
        (**self).prefill_embeds_slot(s, tokens, x, base_pos, pos3)
    }
    fn reset_session(&self) { (**self).reset_session() }
}

/// Pipeline-stage execution: the decentralized worker primitive.
pub trait Span {
    /// Run layers [lo, hi) at `pos`. Stage 0 embeds `token`; the last stage
    /// (do_head) returns the argmax token id; middle stages leave the hidden
    /// in place for `read_hidden`.
    fn forward_span(&self, token: u32, pos: usize, lo: usize, hi: usize,
                    do_embed: bool, do_head: bool) -> u32;
    fn read_hidden(&self) -> Vec<f32>;
    fn write_hidden(&self, h: &[f32]);
}

// ================================ learner ====================================

/// Training-capable model slice: serve-and-learn
/// (probe -> gate -> backward, activations resident between calls).
pub trait Learner {
    /// Forward layers [lo, n_layers) with loss. lo==0 embeds `tokens`;
    /// lo>0 injects `x_in` (t*d f32 from the upstream worker).
    /// Activations stay resident: call `bwd_from` after a gate decision.
    fn step_core(&mut self, tokens: &[u32], x_in: Option<&[f32]>, targets: &[u32],
                 lo: usize, do_bwd: bool) -> Result<(f32, usize)>;
    /// Backward + optimizer down to layer `lo` (the worker boundary).
    fn bwd_from(&mut self, lo: usize) -> Result<()>;
    /// Pure span forward for middle workers: layers [lo, hi), returns x[hi].
    fn fwd_span_range(&mut self, tokens: &[u32], x_in: Option<&[f32]>,
                      lo: usize, hi: usize) -> Result<Vec<f32>>;
    fn save_ckpt(&self, path: &str) -> Result<()>;
    fn load_ckpt(&mut self, path: &str) -> Result<()>;
}

/// Surprise gate: decide which served requests warrant an online update.
#[derive(Clone, Debug)]
pub struct Gate {
    pub mu: f32,
    pub sdev: f32,
    pub n_seen: u32,
    pub n_upd: u32,
    pub k_sigma: f32,
    pub warmup: u32,
}

impl Gate {
    pub fn new(k_sigma: f32) -> Self {
        Gate { mu: f32::NAN, sdev: 0.5, n_seen: 0, n_upd: 0, k_sigma, warmup: 10 }
    }
    /// Observe a request's loss; returns true if this request should trigger learning.
    pub fn observe(&mut self, loss: f32) -> bool {
        if self.mu.is_nan() { self.mu = loss; }
        let surprised = self.n_seen < self.warmup || loss > self.mu + self.k_sigma * self.sdev;
        self.sdev = 0.98 * self.sdev + 0.02 * (loss - self.mu).abs();
        self.mu = 0.98 * self.mu + 0.02 * loss;
        self.n_seen += 1;
        if surprised { self.n_upd += 1; }
        surprised
    }
}

// ================================ wire =======================================

/// Swarm wire protocol v1:
/// frame = [u8 kind][u32 id][u32 T][T u32 tokens][u8 has_x][T*D f32 x?]
/// reply = [u8 4][u32 id][f32 loss][u8 updated]
pub mod wire {
    use anyhow::Result;
    use std::io::{Read, Write};

    pub const SERVE: u8 = 1;
    pub const VAL: u8 = 2;
    pub const STOP: u8 = 3;
    pub const REPLY: u8 = 4;
    pub const BLOB: u8 = 5;

    pub fn send(w: &mut impl Write, kind: u8, id: u32, toks: &[u32], x: Option<&[f32]>) -> Result<()> {
        w.write_all(&[kind])?;
        w.write_all(&id.to_le_bytes())?;
        w.write_all(&(toks.len() as u32).to_le_bytes())?;
        for t in toks { w.write_all(&t.to_le_bytes())?; }
        w.write_all(&[x.is_some() as u8])?;
        if let Some(xs) = x {
            let b = unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) };
            w.write_all(b)?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn recv(r: &mut impl Read, d: usize) -> Result<(u8, u32, Vec<u32>, Option<Vec<f32>>)> {
        let mut b1 = [0u8; 1];
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b1)?;
        if b1[0] == STOP { return Ok((STOP, 0, vec![], None)); }
        let kind = b1[0];
        r.read_exact(&mut b4)?; let id = u32::from_le_bytes(b4);
        r.read_exact(&mut b4)?; let t = u32::from_le_bytes(b4) as usize;
        // Length prefix is attacker-controllable; cap it so a garbage frame can't drive
        // a multi-GB allocation (t and t*d below).
        const MAX_TOKENS: usize = 1 << 20;
        if t > MAX_TOKENS { anyhow::bail!("wire: frame token count {t} exceeds cap {MAX_TOKENS}"); }
        let mut toks = vec![0u32; t];
        for i in 0..t { r.read_exact(&mut b4)?; toks[i] = u32::from_le_bytes(b4); }
        r.read_exact(&mut b1)?;
        let x = if b1[0] == 1 {
            let mut buf = vec![0f32; t * d];
            let bb = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 4) };
            r.read_exact(bb)?;
            Some(buf)
        } else { None };
        Ok((kind, id, toks, x))
    }

    pub fn send_reply(w: &mut impl Write, id: u32, loss: f32, updated: bool) -> Result<()> {
        w.write_all(&[REPLY])?;
        w.write_all(&id.to_le_bytes())?;
        w.write_all(&loss.to_le_bytes())?;
        w.write_all(&[updated as u8])?;
        w.flush()?;
        Ok(())
    }

    pub fn recv_reply(r: &mut impl Read) -> Result<(u32, f32, bool)> {
        let mut b1 = [0u8; 1];
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b1)?;
        anyhow::ensure!(b1[0] == REPLY, "bad reply frame");
        r.read_exact(&mut b4)?; let id = u32::from_le_bytes(b4);
        r.read_exact(&mut b4)?; let loss = f32::from_le_bytes(b4);
        r.read_exact(&mut b1)?;
        Ok((id, loss, b1[0] == 1))
    }

    /// Blob frame: `[u8 5][u32 id][u64 len][len bytes]`. The DiLoCo path carries
    /// whole parameter vectors, which do not fit the token/activation frame above;
    /// it runs on its own socket and never interleaves with it.
    pub fn send_blob(w: &mut impl Write, id: u32, bytes: &[u8]) -> Result<()> {
        w.write_all(&[BLOB])?;
        w.write_all(&id.to_le_bytes())?;
        w.write_all(&(bytes.len() as u64).to_le_bytes())?;
        w.write_all(bytes)?;
        w.flush()?;
        Ok(())
    }

    pub fn recv_blob(r: &mut impl Read) -> Result<(u32, Vec<u8>)> {
        let mut b1 = [0u8; 1];
        let mut b4 = [0u8; 4];
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b1)?;
        anyhow::ensure!(b1[0] == BLOB, "bad blob frame");
        r.read_exact(&mut b4)?; let id = u32::from_le_bytes(b4);
        r.read_exact(&mut b8)?; let n = u64::from_le_bytes(b8) as usize;
        // Same reasoning as MAX_TOKENS: the length prefix is attacker-controllable.
        const MAX_BLOB: usize = 4 << 30;
        anyhow::ensure!(n <= MAX_BLOB, "wire: blob of {n} bytes exceeds cap {MAX_BLOB}");
        let mut v = vec![0u8; n];
        r.read_exact(&mut v)?;
        Ok((id, v))
    }

    /// `send_blob` over an f32 vector (host-endian, matching the activation path).
    pub fn send_vec(w: &mut impl Write, id: u32, v: &[f32]) -> Result<()> {
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        send_blob(w, id, b)
    }

    pub fn recv_vec(r: &mut impl Read) -> Result<(u32, Vec<f32>)> {
        let (id, b) = recv_blob(r)?;
        anyhow::ensure!(b.len() % 4 == 0, "blob of {} bytes is not f32-aligned", b.len());
        let mut v = vec![0f32; b.len() / 4];
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len()); }
        Ok((id, v))
    }
}

// ================================ diloco =====================================

/// DiLoCo outer loop. Inner optimization is local and ordinary; only the
/// accumulated parameter difference crosses the network, once every `inner` steps:
///
/// ```text
/// delta_i    = theta_start - theta_i
/// theta_next = theta_start - outer_lr * mean(delta_i)
/// ```
///
/// The hub holds no model and runs no kernels — it is f32 arithmetic over a
/// socket, so it builds and runs on every platform, including those with no
/// training backend.
pub mod diloco {
    use crate::wire;
    use anyhow::Result;
    use std::net::{TcpListener, TcpStream};

    /// Round id reserved to mean "run is over, disconnect".
    pub const DONE: u32 = u32::MAX;

    /// `theta -= outer_lr * mean(deltas)`, in place.
    pub fn apply(theta: &mut [f32], deltas: &[Vec<f32>], outer_lr: f32) -> Result<()> {
        anyhow::ensure!(!deltas.is_empty(), "aggregate: no deltas");
        for d in deltas {
            anyhow::ensure!(d.len() == theta.len(),
                "aggregate: delta has {} params, base has {}", d.len(), theta.len());
        }
        let k = deltas.len() as f32;
        for (i, t) in theta.iter_mut().enumerate() {
            let mut s = 0f32;
            for d in deltas { s += d[i]; }
            *t -= outer_lr * s / k;
        }
        Ok(())
    }

    /// Aggregation hub for `n` workers over `rounds`.
    ///
    /// Join: each worker sends its architecture signature as the blob id, with its
    /// own parameters as payload. All signatures must agree — a delta computed
    /// against a different base or tensor order is meaningless rather than merely
    /// noisy, so a mismatch ends the run instead of being averaged in.
    pub fn hub(listen: u16, n: usize, rounds: usize, outer_lr: f32, out: Option<&str>) -> Result<()> {
        anyhow::ensure!(n > 0, "hub needs at least one worker");
        let srv = TcpListener::bind(("0.0.0.0", listen))?;
        let mut peers: Vec<TcpStream> = Vec::with_capacity(n);
        let (mut sig, mut theta) = (None::<u32>, Vec::new());
        while peers.len() < n {
            let (mut s, addr) = srv.accept()?;
            let (peer_sig, w) = wire::recv_vec(&mut s)?;
            match sig {
                None => {
                    tracing::info!(target: "diloco", "base from {addr}: sig {peer_sig:08x}, {} params", w.len());
                    (sig, theta) = (Some(peer_sig), w);
                }
                Some(want) => anyhow::ensure!(peer_sig == want && w.len() == theta.len(),
                    "worker {addr} has sig {peer_sig:08x}/{} params, run is {want:08x}/{}",
                    w.len(), theta.len()),
            }
            peers.push(s);
        }
        tracing::info!(target: "diloco", "{n} workers joined; {rounds} rounds, outer_lr {outer_lr}");

        for round in 0..rounds {
            for p in peers.iter_mut() { wire::send_vec(p, round as u32, &theta)?; }
            let mut deltas = Vec::with_capacity(n);
            for p in peers.iter_mut() {
                let (r, d) = wire::recv_vec(p)?;
                anyhow::ensure!(r == round as u32, "delta for round {r}, expected {round}");
                deltas.push(d);
            }
            let norm = deltas.iter()
                .map(|d| d.iter().map(|x| x * x).sum::<f32>().sqrt())
                .sum::<f32>() / (n as f32);
            apply(&mut theta, &deltas, outer_lr)?;
            tracing::info!(target: "diloco", "round {round}: {n} deltas, mean |delta| {norm:.5}");
        }
        for p in peers.iter_mut() { let _ = wire::send_vec(p, DONE, &[]); }
        if let Some(path) = out {
            let b = unsafe { std::slice::from_raw_parts(theta.as_ptr() as *const u8, theta.len() * 4) };
            std::fs::write(path, b)?;
            tracing::info!(target: "diloco", "wrote {} params to {path}", theta.len());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A backend chosen at runtime is boxed, and the speculative path runs
    /// entirely through `Model`'s optional methods. If `impl Model for Box<T>`
    /// forgets one, the default answers instead of the real backend and MTP turns
    /// itself off with no error. This pins the forwarding.
    #[test]
    fn boxing_a_model_forwards_the_optional_methods_too() {
        struct Speculative {
            reset_calls: std::cell::Cell<usize>,
            /// (tokens, x floats, base_pos, pos3 rows) of the last prefill_embeds.
            /// The rows' contents are asserted inside the impl below.
            embed_spans: std::cell::Cell<(usize, usize, usize, usize)>,
            /// Slot bookkeeping: the last `reset_slot` index, and the last
            /// `prefill_slot` / `prefill_embeds_slot` (slot, tokens, base_pos).
            reset_slots: std::cell::Cell<usize>,
            slot_spans: std::cell::Cell<(usize, usize, usize)>,
        }
        impl Model for Speculative {
            fn n_layers(&self) -> usize { 3 }
            fn hidden_dim(&self) -> usize { 8 }
            fn prefill(&self, _t: &[u32], _p: usize) {}
            fn forward_id(&self, _t: u32, _p: usize) -> u32 { 7 }
            // The default `false` reads as "this backend cannot inject rows", so a
            // missed forward sends every vision prompt down the text-id path (or has
            // it refused) with nothing in the logs to say why.
            fn prefill_embeds(&self, t: &[u32], x: &[f32], p: usize, p3: Option<&[[u32; 4]]>) -> bool {
                // Pin the contents, not just the row count: widened, truncated or
                // transposed coordinate rows keep the count and change the rope
                // angles, which is invisible downstream.
                assert_eq!(p3, Some(&[[0u32, 1, 2, 3], [4, 5, 6, 7]][..]),
                    "prefill_embeds: pos3 rows must arrive verbatim");
                self.embed_spans.set((t.len(), x.len(), p, p3.map_or(0, |v| v.len())));
                true
            }
            // The vision trio. `vision_width` is the capability probe a host
            // branches on, so a missed forward hides the model's own tower and
            // selects the portable CPU oracle instead — ~500x slower, a page in
            // minutes rather than milliseconds.
            fn vision_width(&self) -> Option<usize> { Some(1024) }
            fn vision_tokens(&self, w: usize, h: usize) -> Option<usize> { Some(w * h / 4) }
            fn encode_image(&self, img: &[f32], w: usize, h: usize) -> Option<Result<Vec<f32>>> {
                // Pin the arguments too: swapping w and h still returns the right
                // row count on a square image and transposes every page.
                assert_eq!((img.len(), w, h), (12, 4, 3), "encode_image args must arrive verbatim");
                Some(Ok(vec![0.25; w * h / 4 * 1024]))
            }
            // Everything below has a default that would mask a missing forward.
            fn context_capacity(&self) -> usize { 128 }
            fn mtp_verify_width(&self) -> usize { 4 }
            fn reuse_prefix_len(&self, p: &[u32]) -> usize { p.len() }
            fn has_mtp(&self) -> bool { true }
            fn mtp_step_committed(&self, _c: u32, _p: usize) -> Option<Vec<u32>> { Some(vec![1, 2]) }
            fn forward_logits(&self, _t: u32, _p: usize) -> Option<Vec<f32>> { Some(vec![0.5]) }
            fn forward_batch_ids(&self, _t: &[u32], _p: usize) -> Option<Vec<u32>> { Some(vec![9]) }
            fn forward_id_topk(&self, _t: u32, _p: usize) -> Option<(u32, [u32; 8])> { Some((7, [0; 8])) }
            fn reset_session(&self) { self.reset_calls.set(self.reset_calls.get() + 1); }
            // The slot quintet. Every default reads as "this backend does not batch
            // sequences", so a missed forward drops a measured ~2x on the OCR decode
            // path rather than failing.
            fn max_slots(&self) -> usize { 4 }
            fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Option<Vec<u32>> {
                // Pin the triples, not the length: slot, token and pos are all
                // per-entry and all integers, so swapping slot for pos returns the
                // right number of answers and decodes at the wrong places.
                assert_eq!(steps, &[(0usize, 11u32, 5usize), (2, 12, 9)][..],
                    "decode_slots: (slot, token, pos) triples must arrive verbatim");
                Some(vec![21, 22])
            }
            fn reset_slot(&self, s: usize) { self.reset_slots.set(self.reset_slots.get() + 1 + s); }
            fn prefill_slot(&self, s: usize, t: &[u32], p: usize) -> bool {
                // Pin the arguments here rather than after the call: the test drives
                // `&dyn Model`, which cannot reach these cells, and dropping `s`
                // prefills slot 0 every time, collapsing the batch onto one
                // sequence's state.
                assert_eq!((s, t, p), (3, &[1u32, 2, 3][..], 7),
                    "prefill_slot: (slot, tokens, base_pos) must arrive verbatim");
                self.slot_spans.set((s, t.len(), p));
                true
            }
            fn prefill_embeds_slot(&self, s: usize, t: &[u32], x: &[f32], p: usize,
                                   p3: Option<&[[u32; 4]]>) -> bool {
                assert_eq!((s, t, x.len(), p, p3), (2, &[5u32, 6][..], 8, 4,
                    Some(&[[0u32, 1, 2, 3], [4, 5, 6, 7]][..])),
                    "prefill_embeds_slot: slot, ids, rows, base_pos and pos3 must arrive verbatim");
                self.slot_spans.set((s, t.len(), p));
                true
            }
        }
        // Both wrappers the CLI uses: an owned boxed backend, and a borrowed one
        // handed to an engine that does not own it.
        let boxed: Box<dyn Model> = Box::new(Speculative {
            reset_calls: Default::default(), embed_spans: Default::default(),
            reset_slots: Default::default(), slot_spans: Default::default() });
        let owned = Speculative {
            reset_calls: Default::default(), embed_spans: Default::default(),
            reset_slots: Default::default(), slot_spans: Default::default() };
        let borrowed: &dyn Model = &owned;
        // `borrowed` does not exercise `impl Model for &T`: `&dyn Model` has a
        // vtable pointing straight at `Speculative`, so calls bypass the blanket impl
        // and deleting a forward from it left this test green (verified by mutation).
        // `&&dyn Model` coerces through the blanket impl, so its vtable is the one
        // under test. Both rows stay: `&dyn` is the shape
        // `EngineCore::new(&wrapper as &dyn Model)` builds, `&&dyn` covers the
        // forwarding.
        let nested: &dyn Model = &borrowed;
        for (what, m) in [("Box", &boxed as &dyn Model), ("&", borrowed), ("&&", nested)] {
            assert!(m.has_mtp(), "{what}: must report MTP, not the default false");
            assert_eq!(m.mtp_step_committed(0, 0), Some(vec![1, 2]), "{what}");
            assert_eq!(m.mtp_verify_width(), 4, "{what}");
            assert_eq!(m.context_capacity(), 128, "{what}");
            assert_eq!(m.reuse_prefix_len(&[1, 2, 3]), 3, "{what}");
            assert_eq!(m.forward_logits(0, 0), Some(vec![0.5]), "{what}");
            assert_eq!(m.forward_batch_ids(&[1], 0), Some(vec![9]), "{what}");
            assert_eq!(m.forward_id_topk(0, 0).map(|(t, _)| t), Some(7), "{what}");
            // Precomputed-row prefill (the vision seam). Assert the real answer and
            // that the arguments arrived intact: dropping `pos3` or swapping `x` for
            // `tokens` still returns true.
            assert!(m.prefill_embeds(&[5, 6], &[0.0; 8], 3, Some(&[[0, 1, 2, 3], [4, 5, 6, 7]])), "{what}");
            // `ojas ocr` branches on `vision_width()` to pick the model's own
            // encoder over the CPU oracle, so the default `None` runs the slow path
            // rather than failing.
            assert_eq!(m.vision_width(), Some(1024), "{what}: must report its tower, not the default None");
            assert_eq!(m.vision_tokens(64, 64), Some(1024), "{what}");
            // `anyhow::Error` is not `PartialEq`, so unwrap both layers rather than
            // comparing the nested Option<Result<_>>. `None` is "no tower",
            // `Some(Err(_))` is "the tower failed", and neither may collapse into
            // the other.
            let rows = m.encode_image(&[0.0; 12], 4, 3)
                .expect("{what}: encode_image must reach the backend, not answer None")
                .expect("{what}: the backend returned Ok, so the wrapper must too");
            assert_eq!((rows.len(), rows[0]), (3 * 1024, 0.25),
                "{what}: encode_image rows must come back unchanged");
            assert_eq!(m.forward_id(0, 0), 7, "{what}");
            assert_eq!(m.n_layers(), 3, "{what}");
            assert_eq!(m.hidden_dim(), 8, "{what}");
            // Swallowing this leaves a served recurrent model carrying state between
            // unrelated requests, with no error anywhere.
            m.reset_session();
            // `max_slots`'s default 1 and `decode_slots`' default `None` both read as
            // "cannot batch sequences": a ~2x loss on the OCR decode path, not a
            // failure.
            assert_eq!(m.max_slots(), 4, "{what}: must report its slot count, not the default 1");
            assert_eq!(m.decode_slots(&[(0, 11, 5), (2, 12, 9)]), Some(vec![21, 22]), "{what}");
            // Both default to `false`. Assert the real answer and that the slot
            // index survived: dropping `s` prefills slot 0 every time and collapses
            // the batch onto one sequence's state.
            assert!(m.prefill_slot(3, &[1, 2, 3], 7), "{what}");
            assert!(m.prefill_embeds_slot(2, &[5, 6], &[0.0; 8], 4,
                    Some(&[[0, 1, 2, 3], [4, 5, 6, 7]])), "{what}");
            m.reset_slot(2);
        }
        // Twice: once through the bare trait object, once through `impl Model for &T`.
        assert_eq!(owned.reset_calls.get(), 2, "&T must forward reset_session to the backend");
        // Two rows reach `owned` (`&dyn` and `&&dyn`); the Box row has its own
        // instance. `reset_slot` adds `1 + s`, so slot 2 twice is 6; dropping the
        // index would give 2 and reset slot 0 every time.
        assert_eq!(owned.reset_slots.get(), 6, "&T must forward reset_slot's slot index");
        // The last slot prefill through `owned` was `prefill_embeds_slot(2, 2 ids, 4)`.
        assert_eq!(owned.slot_spans.get(), (2, 2, 4),
            "&T must forward the slot, token count AND base_pos of a slot prefill");
        assert_eq!(owned.embed_spans.get(), (2, 8, 3, 2),
            "&T must forward prefill_embeds' tokens, rows, base_pos AND pos3 unchanged");
    }

    #[test]
    fn gate_warmup_then_selective() {
        let mut g = Gate::new(0.5);
        for _ in 0..10 { assert!(g.observe(2.0)); }       // warmup always learns
        for _ in 0..50 { g.observe(2.0); }                 // routine traffic settles mu
        assert!(!g.observe(2.0));                          // routine -> skip
        assert!(g.observe(5.0));                           // surprising -> learn
    }

    #[test]
    fn wire_roundtrip() {
        let mut buf = vec![];
        wire::send(&mut buf, wire::SERVE, 7, &[1, 2, 3], Some(&[0.5f32; 6])).unwrap();
        let (k, id, toks, x) = wire::recv(&mut buf.as_slice(), 2).unwrap();
        assert_eq!((k, id, toks.len(), x.unwrap().len()), (wire::SERVE, 7, 3, 6));
    }

    #[test]
    fn blob_roundtrip() {
        let mut buf = vec![];
        wire::send_vec(&mut buf, 3, &[1.0, -2.5, 4.25]).unwrap();
        let (id, v) = wire::recv_vec(&mut buf.as_slice()).unwrap();
        assert_eq!((id, v), (3, vec![1.0, -2.5, 4.25]));
    }

    /// Hand-computed reference for the outer step: the aggregate math is checked
    /// against a small vector example rather than inferred from matching shapes.
    #[test]
    fn diloco_outer_step() {
        // theta_start 10; workers land on 8 and 6, so deltas are 2 and 4, mean 3.
        // outer_lr 0.5 -> 10 - 0.5*3 = 8.5.
        let mut theta = vec![10.0f32, 10.0];
        diloco::apply(&mut theta, &[vec![2.0, 2.0], vec![4.0, 4.0]], 0.5).unwrap();
        assert_eq!(theta, vec![8.5, 8.5]);

        // outer_lr 1.0 with one worker is plain replacement by that worker's result.
        let mut theta = vec![10.0f32];
        diloco::apply(&mut theta, &[vec![4.0]], 1.0).unwrap();
        assert_eq!(theta, vec![6.0]);

        // A delta of the wrong length is refused, never padded or truncated.
        assert!(diloco::apply(&mut vec![1.0, 2.0], &[vec![1.0]], 0.5).is_err());
        assert!(diloco::apply(&mut vec![1.0], &[], 0.5).is_err());
    }
}

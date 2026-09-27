//! Continuous batching: keep B pages in flight through a batched decoder.
//!
//! Decode is 88% of an OCR page, and batching B=4 sequences through the M-row GEMV
//! family is worth ~2.0x on the token (bracket 1.7x-2.3x, ceiling 2.24x). A static
//! batch of 4 gives 20-25% of that back: a surya page emits 2100-3200 tokens
//! (measured over 131 pages), so a fixed round idles up to three of its four slots
//! while the longest page finishes.
//!
//! This module is pure — no Metal types, no `ojas_core::Model`, no image decoding, no
//! stdout. The decoder is reached through [`SlotModel`] (three methods) and the pages
//! through [`PageSource`] (two); both are implemented by a software mock in the tests
//! below as well as by the real thing in `ocr.rs`. The batched decoder needs a GPU,
//! `OJAS_SLOTS>1` and a supported architecture to run at all, so testing only through
//! it would mean testing on one machine in one configuration; here the
//! ragged-completion behaviour is 22 unit tests that run with no Metal.
//!
//! Three invariants the tests pin:
//!
//! 1. Every admitted page is completed exactly once, whatever order the slots free
//!    in, with a reason (EOG, budget, repetition guard, skip).
//! 2. `reset_slot` precedes every admission, including the first. Every image token is
//!    the literal id 11, so a prefix match over ids matches across different pages; a
//!    slot handed to page N while still holding page N-1's state transcribes the wrong
//!    page with nothing in the output to say so. [`drive`] is the single place that
//!    resets, so the invariant is one line and one test rather than a rule callers
//!    must remember.
//! 3. A slot's token stream depends only on that slot's own state. The step assembly
//!    hands the decoder `(slot, token, pos)` and nothing else, and a slot's `pos`
//!    counts only its own tokens, so page X decodes identically whether it ran alone,
//!    first in a batch, or fourth. Greedy OCR output must be byte-identical across
//!    batch composition, and [`Scheduler`] is where that is preserved or lost.

use std::collections::VecDeque;

use anyhow::{ensure, Context, Result};

// ---------------------------------------------------------------------------
// the two seams
// ---------------------------------------------------------------------------

/// The batched-decode half of the decoder contract.
///
/// `ojas_core::Model`'s slot methods — `max_slots`/`reset_slot`/`decode_slots` —
/// restated as a trait this crate can implement for a mock. A separate trait rather
/// than a `Model` bound so the scheduler is testable without a GPU: the mock below is
/// a `RefCell`-and-`Vec` machine that need not implement a 30-method model interface
/// to answer three questions.
///
/// `ocr.rs` holds the one adapter that forwards these to the real model.
pub trait SlotModel {
    /// How many sequences this decoder can hold at once. 1 = unbatched, and the
    /// caller must take its sequential path.
    fn max_slots(&self) -> usize;
    /// Drop slot `s`'s KV rows and recurrent state. Mandatory between pages.
    fn reset_slot(&self, s: usize);
    /// One decode step for every `(slot, token, pos)` in `steps`, returning one argmax
    /// token id per entry in the same order. `None` means this decoder cannot batch,
    /// which the caller must treat as an error rather than a fallback: by then the
    /// pages are already prefilled into slots.
    fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Option<Vec<u32>>;
}

/// What a page needs to start decoding in a slot: the token that drives its
/// first step, and how long its prompt was.
///
/// `prompt_len` rather than a position, because there is one right answer
/// (`pos = prompt_len - 1`, the last prompt token's own row, since `EngineCore`
/// prefills `ids[..len-1]` and forwards the final id by hand) and two fields would be
/// two chances to disagree about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seed {
    pub first_token: u32,
    pub prompt_len: usize,
}

/// The pages, and what to do with a finished one.
///
/// [`PageSource::admit`] is where everything impure happens in a real run — read the
/// image, run the ViT, prefill the slot — which is why it sits behind a trait.
/// Admission is synchronous and exclusive: a new page's ViT encode (0.7-2.2 s) and its
/// ~4 500-token prefill stall every other slot while they run. The scheduler does not
/// hide that cost; [`Stats::rounds`] and the caller's clock account for it.
pub trait PageSource {
    /// Prepare page `page` and prefill it into slot `s`, which has just been reset.
    /// `Ok(None)` means the page could not be prepared (an unreadable scan) and must be
    /// skipped without failing the batch; `Err` means the engine is wrong and the run
    /// must stop.
    fn admit(&mut self, slot: usize, page: usize) -> Result<Option<Seed>>;
    /// One page has finished. Called exactly once per admitted page, including
    /// skipped ones.
    fn finished(&mut self, done: Done) -> Result<()>;
}

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

/// When a queued page may enter a slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Admission {
    /// Fill any free slot immediately — continuous batching.
    #[default]
    Continuous,
    /// Admit only when every slot is free, so fixed batches run to completion. The A/B
    /// control for the 20-25% ragged-tail loss. `OJAS_OCR_ADMIT=static` selects it.
    StaticRounds,
}

/// Per-slot completion policy. Mirrors what `EngineCore::generate_with` and
/// `cmds::stream` do for one sequence, because the batched path must stop pages
/// at exactly the same tokens the sequential path stops them at.
#[derive(Clone, Debug)]
pub struct SchedCfg {
    /// `-n/--predict`.
    pub n_predict: usize,
    /// The decoder's context capacity, which caps `n_predict` the way
    /// `EngineCore` caps it: `capacity - prompt_len + 1`.
    pub ctx: usize,
    /// The model's full end-of-generation set (`ModelInfo::eog`). Empty falls
    /// back to `primary` alone, as `EngineCore::is_stop` does.
    pub eog: Vec<u32>,
    /// `EngineCore::eos` — the template's terminator.
    pub primary: Option<u32>,
    /// The second stop id `cmds::stream` checks in its own callback, because a
    /// templated turn can end at either the template marker or the GGUF's EOS.
    pub also_stop: Option<u32>,
    /// The repetition guard, as a function pointer so this module does not own a
    /// second copy of `ocr::loop_period`. `None` disables it.
    pub loop_probe: Option<fn(&[u32]) -> Option<usize>>,
    pub admission: Admission,
}

impl Default for SchedCfg {
    fn default() -> Self {
        SchedCfg {
            n_predict: 128,
            ctx: usize::MAX,
            eog: Vec::new(),
            primary: None,
            also_stop: None,
            loop_probe: None,
            admission: Admission::Continuous,
        }
    }
}

/// How many tokens a page whose prompt is `prompt_len` may produce, under `-n`
/// and a context of `ctx`.
///
/// `EngineCore::generate_with`'s own clamp, lifted out so both drivers share one
/// definition: the batched path sizes a slot's budget with it, the sequential path
/// decides whether a page came back truncated. The `saturating_add(1)` is
/// `EngineCore`'s and means an overlong prompt still gets one token rather than none.
/// Copied rather than improved, because both paths must truncate in the same place.
pub(crate) fn budget_for(n_predict: usize, ctx: usize, prompt_len: usize) -> usize {
    n_predict.min(ctx.saturating_sub(prompt_len).saturating_add(1))
}

impl SchedCfg {
    /// Whether `t` ends generation. Same rule as `EngineCore::is_stop` plus
    /// `stream`'s secondary: the full EOG set when there is one, `primary`
    /// otherwise, and `also_stop` either way.
    fn is_stop(&self, t: u32) -> bool {
        if Some(t) == self.also_stop {
            return true;
        }
        if self.eog.is_empty() {
            Some(t) == self.primary
        } else {
            self.eog.contains(&t)
        }
    }

    /// This run's budget for a page of `prompt_len` prompt tokens.
    fn budget(&self, prompt_len: usize) -> usize {
        budget_for(self.n_predict, self.ctx, prompt_len)
    }
}

// ---------------------------------------------------------------------------
// results
// ---------------------------------------------------------------------------

/// Why a page stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Finish {
    /// An end-of-generation id, carried so the log distinguishes a page that ended
    /// from a page that ran out of room.
    Stop(u32),
    /// `-n` (or the context) ran out, which on a real page means the transcription is
    /// truncated.
    Budget,
    /// The repetition guard fired, with the block period it found.
    Looped(usize),
    /// The page could never be prepared — an unreadable scan. No slot time was spent on
    /// it.
    Skipped,
}

/// One completed page.
#[derive(Clone, Debug)]
pub struct Done {
    pub slot: usize,
    pub page: usize,
    pub finish: Finish,
    /// The tokens to emit, excluding any stop id: `EngineCore` pushes a stop token into
    /// its output but never passes it to the token callback, so `stream` never prints
    /// it. Byte-identical output across the two paths requires reproducing that, not
    /// just the token count.
    pub tokens: Vec<u32>,
    /// Tokens the model produced, stop id included — what the budget counts, and what
    /// `EngineCore::generate_with` would have returned.
    pub produced: usize,
    /// Decode rounds this page was resident for. A live slot steps in every round, so
    /// this equals `produced` today; it is carried separately so a policy that ever
    /// holds a slot back for a round (a priority scheme, a prefill stall) shows up in
    /// the log.
    pub rounds: usize,
}

/// What the run cost, in scheduler terms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// `decode_slots` calls.
    pub rounds: usize,
    /// Sum over rounds of the number of slots that had work: the numerator of
    /// occupancy.
    pub slot_steps: usize,
    /// Tokens produced across every page.
    pub tokens: usize,
    pub admitted: usize,
    pub completed: usize,
    pub skipped: usize,
}

impl Stats {
    /// Mean live slots per round, out of `n_slots`. 1.0 = no slot ever idled;
    /// 0.75 on B=4 is the ragged tail the static policy leaves on the floor.
    pub fn occupancy(&self, n_slots: usize) -> f64 {
        if self.rounds == 0 || n_slots == 0 {
            return 0.0;
        }
        self.slot_steps as f64 / (self.rounds * n_slots) as f64
    }
}

// ---------------------------------------------------------------------------
// the slot table
// ---------------------------------------------------------------------------

/// A slot with a page in it.
#[derive(Clone, Debug)]
struct Live {
    page: usize,
    /// The token that drives the next step.
    cur: u32,
    /// The cache row `cur` sits at. Advances once per step, so a slot's positions count
    /// only its own tokens, which makes a page's output independent of who else is in
    /// the batch.
    pos: usize,
    budget: usize,
    produced: usize,
    /// Emitted tokens (no stop id), which is also what the repetition guard reads:
    /// `stream` calls the guard with exactly the tokens it printed.
    gen: Vec<u32>,
    rounds: usize,
}

#[derive(Clone, Debug)]
enum Slot {
    Free,
    /// Handed out by [`Scheduler::next_admission`] and not yet seeded. The state
    /// exists so the admission loop can be a `while let` without handing the same
    /// slot to two pages.
    Reserved(usize),
    Live(Live),
}

/// The slot table, the queue, and the completion policy. Drives nothing on its own:
/// [`drive`] is the loop, and every method here is a pure state transition over `Vec`s
/// and `usize`s.
pub struct Scheduler {
    cfg: SchedCfg,
    slots: Vec<Slot>,
    queue: VecDeque<usize>,
    done: Vec<Done>,
    stats: Stats,
    /// [`Admission::StaticRounds`] only: a fill window is open because every slot was
    /// free, so the whole batch may be loaded before the first step. It closes at the
    /// first decode round, which is what makes the policy run a fixed batch to
    /// completion rather than one page at a time.
    fill_open: bool,
}

impl Scheduler {
    /// `n_slots` slots (at least one) and the page indices to run, in order.
    pub fn new(n_slots: usize, pages: impl IntoIterator<Item = usize>, cfg: SchedCfg) -> Scheduler {
        Scheduler {
            cfg,
            slots: vec![Slot::Free; n_slots.max(1)],
            queue: pages.into_iter().collect(),
            done: Vec::new(),
            stats: Stats::default(),
            fill_open: true,
        }
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Nothing running, nothing reserved, nothing queued.
    pub fn idle(&self) -> bool {
        self.queue.is_empty() && self.slots.iter().all(|s| matches!(s, Slot::Free))
    }

    fn live_or_reserved(&self) -> usize {
        self.slots.iter().filter(|s| !matches!(s, Slot::Free)).count()
    }

    /// The admission policy: the lowest free slot and the head of the queue, or
    /// `None`.
    ///
    /// Lowest free slot rather than round-robin, so a given queue produces the same
    /// slot assignment on every run; a batch that is not reproducible cannot be
    /// bisected when one page in a thousand comes out wrong.
    ///
    /// Under [`Admission::StaticRounds`] the whole batch is filled while every slot is
    /// free and nothing more is admitted until the last of them drains.
    pub fn next_admission(&mut self) -> Option<(usize, usize)> {
        if self.queue.is_empty() {
            return None;
        }
        if self.cfg.admission == Admission::StaticRounds {
            // A new fixed batch may only start once the previous one has fully drained;
            // inside the window every free slot is filled.
            if self.live_or_reserved() == 0 {
                self.fill_open = true;
            }
            if !self.fill_open {
                return None;
            }
        }
        let slot = self.slots.iter().position(|s| matches!(s, Slot::Free))?;
        let page = self.queue.pop_front()?;
        self.slots[slot] = Slot::Reserved(page);
        self.stats.admitted += 1;
        Some((slot, page))
    }

    /// The page is prefilled; start decoding it.
    ///
    /// A zero budget (`-n 0`, or a prompt that fills the context) completes the page
    /// immediately rather than entering the step assembly with nothing to do, matching
    /// `EngineCore`, which returns an empty output.
    pub fn seed(&mut self, slot: usize, page: usize, seed: Seed) {
        self.expect_reserved(slot, page, "seed");
        assert!(seed.prompt_len >= 1, "a page prompt cannot be empty");
        let budget = self.cfg.budget(seed.prompt_len);
        if budget == 0 {
            self.slots[slot] = Slot::Free;
            self.finish(slot, page, Finish::Budget, Vec::new(), 0, 0);
            return;
        }
        self.slots[slot] = Slot::Live(Live {
            page,
            cur: seed.first_token,
            pos: seed.prompt_len - 1,
            budget,
            produced: 0,
            gen: Vec::new(),
            rounds: 0,
        });
    }

    /// The page could not be prepared. Frees the slot and accounts for it, so the
    /// exactly-once invariant covers unreadable scans too.
    pub fn skip(&mut self, slot: usize, page: usize) {
        self.expect_reserved(slot, page, "skip");
        self.slots[slot] = Slot::Free;
        self.stats.skipped += 1;
        self.finish(slot, page, Finish::Skipped, Vec::new(), 0, 0);
    }

    fn expect_reserved(&self, slot: usize, page: usize, what: &str) {
        match self.slots.get(slot) {
            Some(Slot::Reserved(p)) if *p == page => {}
            other => panic!("{what}: slot {slot} is not reserved for page {page} ({other:?})"),
        }
    }

    /// The step for this round: one `(slot, token, pos)` per live slot, ascending
    /// by slot.
    ///
    /// Ascending rather than table-iteration order because the batched GEMV reduces over
    /// M rows: a stable row order keeps a page's logits bit-identical from run to run,
    /// so the transcription does not depend on scheduling.
    pub fn plan(&self) -> Vec<(usize, u32, usize)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Live(l) => Some((i, l.cur, l.pos)),
                _ => None,
            })
            .collect()
    }

    /// Apply one round's outputs: advance every live slot and complete the ones
    /// that stopped.
    ///
    /// `out[k]` is the token the decoder produced for `steps[k]`. The completion ladder
    /// is `EngineCore`'s, in `EngineCore`'s order — stop id, then budget, then the guard
    /// — because matching the sequential path means matching where generation stops as
    /// well as what it emits.
    pub fn ingest(&mut self, steps: &[(usize, u32, usize)], out: &[u32]) -> Result<()> {
        ensure!(
            steps.len() == out.len(),
            "decode_slots returned {} tokens for {} steps",
            out.len(),
            steps.len()
        );
        self.stats.rounds += 1;
        self.stats.slot_steps += steps.len();
        self.fill_open = false;
        for (&(slot, token, pos), &t) in steps.iter().zip(out) {
            let cfg = &self.cfg;
            let Some(Slot::Live(l)) = self.slots.get_mut(slot) else {
                anyhow::bail!("decode_slots produced a token for idle slot {slot}");
            };
            ensure!(
                l.cur == token && l.pos == pos,
                "slot {slot} stepped with ({token},{pos}) but holds ({},{})",
                l.cur,
                l.pos
            );
            l.produced += 1;
            l.rounds += 1;
            l.pos += 1;
            l.cur = t;
            self.stats.tokens += 1;
            // A stop id is produced but never emitted, as `EngineCore` pushes it into
            // `out` and returns before the token callback sees it.
            let finish = if cfg.is_stop(t) {
                Some(Finish::Stop(t))
            } else {
                l.gen.push(t);
                if l.produced >= l.budget {
                    Some(Finish::Budget)
                } else {
                    cfg.loop_probe.and_then(|p| p(&l.gen)).map(Finish::Looped)
                }
            };
            if let Some(f) = finish {
                let (page, tokens, produced, rounds) =
                    (l.page, std::mem::take(&mut l.gen), l.produced, l.rounds);
                self.slots[slot] = Slot::Free;
                self.finish(slot, page, f, tokens, produced, rounds);
            }
        }
        Ok(())
    }

    fn finish(
        &mut self,
        slot: usize,
        page: usize,
        finish: Finish,
        tokens: Vec<u32>,
        produced: usize,
        rounds: usize,
    ) {
        self.stats.completed += 1;
        self.done.push(Done { slot, page, finish, tokens, produced, rounds });
    }

    /// Hand back the pages that finished since the last call, in completion
    /// order.
    pub fn take_done(&mut self) -> Vec<Done> {
        std::mem::take(&mut self.done)
    }
}

// ---------------------------------------------------------------------------
// the loop
// ---------------------------------------------------------------------------

/// Run a whole batch: admit, step, complete, repeat until the queue and the
/// slots are empty.
///
/// The only place `reset_slot` is called, and it is called before every admission
/// (invariant 2 in the module header). A driver that reset inside its own `admit` would
/// be one forgotten call away from transcribing page N as page N-1 with no error
/// anywhere.
pub fn drive<M, P>(m: &M, src: &mut P, sched: &mut Scheduler) -> Result<()>
where
    M: SlotModel + ?Sized,
    P: PageSource + ?Sized,
{
    loop {
        while let Some((slot, page)) = sched.next_admission() {
            // Before `admit`, because `admit` prefills: the reset must land on the slot
            // while it still holds only the previous page.
            m.reset_slot(slot);
            match src.admit(slot, page)? {
                Some(seed) => sched.seed(slot, page, seed),
                None => sched.skip(slot, page),
            }
            for d in sched.take_done() {
                src.finished(d)?;
            }
        }
        let steps = sched.plan();
        if steps.is_empty() {
            // An empty plan on a non-idle scheduler would spin forever, so it is an
            // error rather than a clean exit.
            ensure!(
                sched.idle(),
                "scheduler stalled with {} pages queued and no live slot",
                sched.queue.len()
            );
            return Ok(());
        }
        let out = m
            .decode_slots(&steps)
            .context("the decoder refused a batched step after the pages were prefilled into \
                     slots (Model::decode_slots returned None)")?;
        sched.ingest(&steps, &out)?;
        for d in sched.take_done() {
            src.finished(d)?;
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    // ---- the mock decoder --------------------------------------------------

    /// What the mock saw, in order. `reset` must precede the matching `prefill` for
    /// every page, which is checked rather than assumed.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Ev {
        Reset(usize),
        Prefill(usize, usize), // (slot, page)
        Round(Vec<(usize, u32, usize)>),
        Done(usize, usize), // (page, emitted tokens)
    }

    /// A software decoder implementing the slot interface.
    ///
    /// Its rule: a slot's next token is a function of that slot's current token alone.
    /// Page `p` is seeded with `1000*(p+1)` and the mock answers `cur + 1` until the page
    /// has emitted `len[p]` tokens, then [`EOS`]. A page's token stream is therefore
    /// determined by the page and nothing else — no slot index, no batch composition, no
    /// round number — the property a real greedy decoder must have and the one the
    /// composition test below measures.
    ///
    /// It also checks the scheduler's arithmetic from the other side: every step's `pos`
    /// must be the position that slot is actually at, so a scheduler sharing a position
    /// counter between slots fails here rather than in a 3 000-token transcription.
    struct Mock {
        slots: usize,
        /// Output length per page.
        len: Vec<usize>,
        /// Prompt length per page, so `pos` can be predicted.
        prompt: Vec<usize>,
        log: RefCell<Vec<Ev>>,
        /// Rounds are logged only where a test reads them; the 131-page
        /// occupancy run is 89 000 of them.
        log_rounds: bool,
        /// Per-slot (page, tokens produced so far, expected pos).
        state: RefCell<Vec<Option<(usize, usize, usize)>>>,
    }

    const EOS: u32 = 2;
    const BASE: u32 = 1000;

    /// Page `p`'s `n`-th token. Distinct per page, so a slot carrying the wrong page's
    /// token is caught on the spot rather than at the end of a transcription.
    fn tok(page: usize, n: usize) -> u32 {
        BASE * (page as u32 + 1) + n as u32
    }

    impl Mock {
        fn new(slots: usize, len: Vec<usize>, prompt_len: usize) -> Mock {
            let prompt = vec![prompt_len; len.len()];
            Mock {
                slots,
                len,
                prompt,
                log: RefCell::new(Vec::new()),
                log_rounds: true,
                state: RefCell::new(vec![None; slots]),
            }
        }
        fn seed_of(&self, page: usize) -> Seed {
            Seed { first_token: tok(page, 0), prompt_len: self.prompt[page] }
        }
        fn log(&self) -> Vec<Ev> {
            self.log.borrow().clone()
        }
    }

    impl SlotModel for Mock {
        fn max_slots(&self) -> usize {
            self.slots
        }
        fn reset_slot(&self, s: usize) {
            self.log.borrow_mut().push(Ev::Reset(s));
            self.state.borrow_mut()[s] = None;
        }
        fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Option<Vec<u32>> {
            if self.log_rounds {
                self.log.borrow_mut().push(Ev::Round(steps.to_vec()));
            }
            let mut st = self.state.borrow_mut();
            let mut out = Vec::with_capacity(steps.len());
            for &(slot, token, pos) in steps {
                let (page, n, want_pos) = st[slot].expect("a step for a slot with no page");
                assert_eq!(pos, want_pos, "slot {slot} stepped at the wrong position");
                assert_eq!(token, tok(page, n), "slot {slot} stepped with the wrong token");
                st[slot] = Some((page, n + 1, pos + 1));
                // One more token of this page's own stream, then its terminator.
                out.push(if n + 1 >= self.len[page] { EOS } else { tok(page, n + 1) });
            }
            Some(out)
        }
    }

    /// The pages side: hands out seeds, records completions. `bad` names pages
    /// that cannot be prepared.
    struct Pages<'m> {
        m: &'m Mock,
        bad: Vec<usize>,
        done: Vec<Done>,
    }

    impl PageSource for Pages<'_> {
        fn admit(&mut self, slot: usize, page: usize) -> Result<Option<Seed>> {
            self.m.log.borrow_mut().push(Ev::Prefill(slot, page));
            if self.bad.contains(&page) {
                return Ok(None);
            }
            self.m.state.borrow_mut()[slot] =
                Some((page, 0, self.m.prompt[page].saturating_sub(1)));
            Ok(Some(self.m.seed_of(page)))
        }
        fn finished(&mut self, done: Done) -> Result<()> {
            self.m.log.borrow_mut().push(Ev::Done(done.page, done.tokens.len()));
            self.done.push(done);
            Ok(())
        }
    }

    fn cfg(n_predict: usize) -> SchedCfg {
        SchedCfg { n_predict, primary: Some(EOS), ..SchedCfg::default() }
    }

    /// Run `len.len()` pages through `slots` slots and return what happened:
    /// the event log, the completions, and the cost.
    fn run(slots: usize, len: Vec<usize>, cfg: SchedCfg) -> (Vec<Ev>, Vec<Done>, Stats) {
        run_with(slots, len, cfg, Vec::new())
    }

    fn run_with(
        slots: usize,
        len: Vec<usize>,
        cfg: SchedCfg,
        bad: Vec<usize>,
    ) -> (Vec<Ev>, Vec<Done>, Stats) {
        let n_pages = len.len();
        let m = Mock::new(slots, len, 9);
        let mut src = Pages { m: &m, bad, done: Vec::new() };
        let mut s = Scheduler::new(slots, 0..n_pages, cfg);
        drive(&m, &mut src, &mut s).expect("the mock never fails a step");
        (m.log(), src.done, s.stats())
    }

    /// Page `p`'s emitted stream, as the mock defines it: `len[p]` tokens with
    /// the terminator excluded (the last token the mock produces is EOS).
    fn expected(page: usize, len: usize) -> Vec<u32> {
        (1..len).map(|i| tok(page, i)).collect()
    }

    // ---- admission ---------------------------------------------------------

    /// A free slot takes the head of the queue, and nothing else does.
    #[test]
    fn a_free_slot_is_filled_from_the_queue() {
        let mut s = Scheduler::new(2, 0..3, cfg(16));
        assert_eq!(s.next_admission(), Some((0, 0)));
        assert_eq!(s.next_admission(), Some((1, 1)));
        // Both slots are reserved now, so page 2 waits.
        assert_eq!(s.next_admission(), None, "a reserved slot is not free");
        s.seed(0, 0, Seed { first_token: BASE, prompt_len: 9 });
        s.seed(1, 1, Seed { first_token: 2 * BASE, prompt_len: 9 });
        assert_eq!(s.next_admission(), None, "a live slot is not free either");
        assert_eq!(s.plan(), vec![(0, BASE, 8), (1, 2 * BASE, 8)]);
    }

    /// All slots busy: the step is the batch, and no page is admitted until one
    /// of them frees.
    #[test]
    fn all_slots_busy_admits_nothing() {
        let mut s = Scheduler::new(4, 0..8, cfg(64));
        for i in 0..4 {
            let (slot, page) = s.next_admission().unwrap();
            s.seed(slot, page, Seed { first_token: BASE * (page as u32 + 1), prompt_len: 9 });
            assert_eq!((slot, page), (i, i));
        }
        assert_eq!(s.next_admission(), None);
        assert_eq!(s.plan().len(), 4);
        // Four queued pages, four in flight, nothing dropped.
        assert_eq!(s.stats().admitted, 4);
        assert!(!s.idle());
    }

    /// A slot that finishes mid-round is refilled on the next admission pass, not at the
    /// end of the round of four.
    #[test]
    fn a_slot_that_finishes_mid_round_is_refilled_immediately() {
        // Page 0 is one token long; pages 1..4 are long.
        let (log, done, _) = run(2, vec![1, 5, 5], cfg(64));
        // Slot 0 takes page 0, slot 1 takes page 1, page 0 ends on round 1, and
        // page 2 is prefilled into slot 0 before round 2.
        let first_round = log.iter().position(|e| matches!(e, Ev::Round(_))).unwrap();
        let page2 = log.iter().position(|e| *e == Ev::Prefill(0, 2)).unwrap();
        let second_round = log
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e, Ev::Round(_)))
            .nth(1)
            .unwrap()
            .0;
        assert!(first_round < page2 && page2 < second_round,
                "page 2 must enter slot 0 between round 1 and round 2: {log:?}");
        // And the reset landed on slot 0 before that prefill, not after.
        let reset = log[..page2].iter().rposition(|e| *e == Ev::Reset(0)).unwrap();
        assert!(reset > first_round, "slot 0 must be reset after page 0 left it: {log:?}");
        assert_eq!(done.len(), 3);
    }

    /// Ragged pages: the continuous policy never idles a slot while work is queued, and
    /// the static policy idles exactly the tail. The gap is the 20-25% the throughput
    /// model predicts, reproduced here as arithmetic.
    #[test]
    fn continuous_admission_beats_static_rounds_on_ragged_pages() {
        // Real pages are 2100–3200 tokens; the same 1:1.5 spread, scaled down.
        let len = vec![21, 32, 24, 30, 22, 31, 25, 28];
        let total: usize = len.iter().sum();
        let (_, dc, sc) = run(4, len.clone(), cfg(4096));
        let (_, ds, ss) = run(
            4,
            len.clone(),
            SchedCfg { admission: Admission::StaticRounds, ..cfg(4096) },
        );
        // Both transcribe every page identically: the policy changes when a page runs,
        // never what it produces.
        for a in &dc {
            let b = ds.iter().find(|d| d.page == a.page).expect("every page completes");
            assert_eq!(a.tokens, b.tokens, "page {} differs between policies", a.page);
        }
        assert_eq!(sc.tokens, total, "every token of every page is produced once");
        assert_eq!(ss.tokens, total);
        // The only difference is idle slots, i.e. rounds.
        assert!(sc.rounds < ss.rounds, "continuous {} vs static {}", sc.rounds, ss.rounds);
        assert!(
            sc.occupancy(4) > ss.occupancy(4) + 0.04,
            "continuous {:.3} vs static {:.3} occupancy",
            sc.occupancy(4),
            ss.occupancy(4)
        );
        // Pinned so a regression in the admission policy shows up as a number. A static
        // batch of 4 over these eight pages costs 63 rounds at 0.845 occupancy (it waits
        // for 32 then for 31); continuous costs 60 at 0.887.
        assert_eq!((sc.rounds, ss.rounds), (60, 63));
        assert!(sc.occupancy(4) > 0.88, "occupancy {:.3}", sc.occupancy(4));
        assert!(ss.occupancy(4) < 0.85, "static occupancy {:.3}", ss.occupancy(4));
        // Continuous cannot do better than never idling a slot while work is queued; the
        // residue is the true tail, when fewer than four pages are left.
        assert!(sc.rounds >= total.div_ceil(4), "no policy beats a full batch");
    }

    /// The same comparison at the measured page-length distribution: 131 pages,
    /// 2100-3200 output tokens, shuffled, B=4.
    ///
    /// A static batch of 4 costs 12% more decode rounds here (100 649 vs 89 807) at 0.878
    /// occupancy against 0.984, not the 20-25% the short-batch estimate projects. 20-25%
    /// is what the ragged tail costs when a batch is only a few rounds long; over a
    /// hundred-page volume the static policy amortizes its own tail and the recoverable
    /// loss is ~12%. Since decode is 88% of a page, that is ~10% of a volume.
    #[test]
    fn ragged_pages_at_the_measured_distribution_cost_12_percent_statically() {
        // xorshift64*, the same generator `ojas-infer` samples with, so the page
        // lengths are a fixed, reproducible draw rather than a chosen sequence.
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let len: Vec<usize> = (0..131)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                2100 + (x % 1101) as usize
            })
            .collect();
        let total: usize = len.iter().sum();
        assert_eq!((len.iter().copied().min(), len.iter().copied().max()), (Some(2108), Some(3200)));

        let run_quiet = |admission: Admission| -> Stats {
            let mut m = Mock::new(4, len.clone(), 4220);
            m.log_rounds = false;
            let mut src = Pages { m: &m, bad: Vec::new(), done: Vec::new() };
            let mut s = Scheduler::new(
                4,
                0..len.len(),
                SchedCfg { n_predict: 4000, admission, ..cfg(4000) },
            );
            drive(&m, &mut src, &mut s).unwrap();
            assert_eq!(src.done.len(), len.len());
            s.stats()
        };
        let c = run_quiet(Admission::Continuous);
        let st = run_quiet(Admission::StaticRounds);
        assert_eq!((c.tokens, st.tokens), (total, total), "same work either way");
        assert_eq!((c.rounds, st.rounds), (89_807, 100_649));
        assert!((c.occupancy(4) - 0.9836).abs() < 5e-4, "{:.4}", c.occupancy(4));
        assert!((st.occupancy(4) - 0.8777).abs() < 5e-4, "{:.4}", st.occupancy(4));
        // The recoverable loss, as one number.
        let saved = 1.0 - c.rounds as f64 / st.rounds as f64;
        assert!((0.10..0.13).contains(&saved), "continuous saves {:.1}%", saved * 100.0);
    }

    /// B=1 must stay exactly the sequential path: one page at a time, in queue order, one
    /// triple per step.
    #[test]
    fn a_single_slot_is_the_sequential_path() {
        let (log, done, stats) = run(1, vec![4, 2, 3], cfg(64));
        assert_eq!(done.iter().map(|d| d.page).collect::<Vec<_>>(), vec![0, 1, 2],
                   "one slot cannot reorder pages");
        for e in log {
            if let Ev::Round(s) = e {
                assert_eq!(s.len(), 1, "a one-slot batch is one step");
            }
        }
        assert_eq!(stats.occupancy(1), 1.0);
        assert_eq!(stats.rounds, 4 + 2 + 3);
    }

    // ---- completion --------------------------------------------------------

    /// Every admitted page completes exactly once, with a reason, whatever order
    /// the slots free in.
    #[test]
    fn every_admitted_page_completes_exactly_once() {
        for slots in [1usize, 2, 3, 4, 7] {
            let len = vec![5, 1, 9, 3, 12, 2, 7, 4, 6, 8];
            let (_, done, stats) = run(slots, len.clone(), cfg(4096));
            let mut pages: Vec<usize> = done.iter().map(|d| d.page).collect();
            pages.sort_unstable();
            assert_eq!(pages, (0..len.len()).collect::<Vec<_>>(), "slots={slots}");
            assert_eq!(stats.admitted, len.len());
            assert_eq!(stats.completed, len.len());
            assert_eq!(stats.skipped, 0);
            for d in &done {
                assert_eq!(d.finish, Finish::Stop(EOS), "page {} slots={slots}", d.page);
                assert_eq!(d.produced, len[d.page], "page {}", d.page);
                assert_eq!(d.tokens, expected(d.page, len[d.page]));
            }
            assert_eq!(stats.tokens, len.iter().sum::<usize>());
        }
    }

    /// A page's emitted text must be the same whether it ran alone, first in a batch or
    /// fourth, and whatever batch it shares. Checked across four batch widths and a
    /// rotated queue, so slot index, neighbours and admission order all vary.
    #[test]
    fn a_pages_output_is_independent_of_batch_composition() {
        let len = vec![6, 3, 11, 2, 8, 5, 9, 4];
        let alone: Vec<Vec<u32>> = (0..len.len()).map(|p| expected(p, len[p])).collect();
        for slots in [1usize, 2, 3, 4, 8] {
            let (_, done, _) = run(slots, len.clone(), cfg(4096));
            for d in &done {
                assert_eq!(d.tokens, alone[d.page],
                           "page {} at B={slots} slot {}", d.page, d.slot);
            }
        }
        // Same pages, different order: page 3 now leads and page 0 is fourth.
        let n = len.len();
        let m = Mock::new(4, len.clone(), 9);
        let mut src = Pages { m: &m, bad: Vec::new(), done: Vec::new() };
        let order: Vec<usize> = (0..n).map(|i| (i + 3) % n).collect();
        let mut s = Scheduler::new(4, order, cfg(4096));
        drive(&m, &mut src, &mut s).unwrap();
        for d in &src.done {
            assert_eq!(d.tokens, alone[d.page], "page {} after reordering", d.page);
        }
    }

    /// A stop id is produced but never emitted: `EngineCore` pushes it into its output and
    /// returns before the token callback runs, so `stream` never prints it. The batched
    /// path must match, or every page gains a stray `<|im_end|>`.
    #[test]
    fn a_stop_id_is_not_emitted() {
        let (_, done, stats) = run(2, vec![3, 3], cfg(64));
        for d in &done {
            assert_eq!(d.produced, 3);
            assert_eq!(d.tokens.len(), 2, "the terminator is produced, not emitted");
            assert!(!d.tokens.contains(&EOS));
        }
        assert_eq!(stats.tokens, 6, "the terminator still costs a step");
    }

    /// The secondary stop id `cmds::stream` checks in its own callback stops a
    /// slot the same way the primary does.
    #[test]
    fn the_secondary_stop_id_also_ends_a_page() {
        let mut s = Scheduler::new(1, 0..1, SchedCfg { also_stop: Some(77), ..cfg(64) });
        let (slot, page) = s.next_admission().unwrap();
        s.seed(slot, page, Seed { first_token: BASE, prompt_len: 4 });
        s.ingest(&s.plan(), &[77]).unwrap();
        let done = s.take_done();
        assert_eq!(done[0].finish, Finish::Stop(77));
        assert!(done[0].tokens.is_empty());
    }

    /// `-n` caps a page, and the cap is reported as truncation rather than as a
    /// clean finish.
    #[test]
    fn the_predict_budget_truncates_a_page() {
        let (_, done, _) = run(2, vec![50, 50], cfg(7));
        for d in &done {
            assert_eq!(d.finish, Finish::Budget);
            assert_eq!(d.produced, 7);
            assert_eq!(d.tokens.len(), 7, "no terminator, so every token is emitted");
        }
    }

    /// The context caps `-n` the way `EngineCore::generate_with` caps it:
    /// `capacity - prompt_len + 1`. A page whose prompt already fills the context
    /// produces nothing at all and must still complete.
    #[test]
    fn the_context_caps_the_budget_the_way_the_engine_does() {
        let c = SchedCfg { n_predict: 1000, ctx: 100, ..cfg(1000) };
        assert_eq!(c.budget(90), 11);
        assert_eq!(c.budget(100), 1);
        // `saturating_add(1)` is `EngineCore`'s, and it means an overlong prompt still
        // gets one token rather than zero. Copied so both paths truncate in the same
        // place.
        assert_eq!(c.budget(101), 1);
        let mut s = Scheduler::new(1, 0..1, c);
        let (slot, page) = s.next_admission().unwrap();
        s.seed(slot, page, Seed { first_token: BASE, prompt_len: 90 });
        s.ingest(&s.plan(), &[BASE + 1]).unwrap();
        assert_eq!(s.plan(), vec![(0, BASE + 1, 90)], "11 tokens of room, one spent");
    }

    /// `-n 0` is the only way a page can have no budget at all. It must complete rather
    /// than sit in a slot the step assembly never visits.
    #[test]
    fn a_page_with_no_budget_completes_immediately() {
        let mut s = Scheduler::new(1, 0..1, SchedCfg { n_predict: 0, ..cfg(0) });
        let (slot, page) = s.next_admission().unwrap();
        s.seed(slot, page, Seed { first_token: BASE, prompt_len: 9 });
        let done = s.take_done();
        assert_eq!(done.len(), 1, "a page with no budget still completes");
        assert_eq!(done[0].finish, Finish::Budget);
        assert!(done[0].tokens.is_empty());
        assert!(s.plan().is_empty());
        assert!(s.idle());
    }

    /// The repetition guard is per slot: the looping page stops and its neighbours carry
    /// on. One bad page burning the whole `-n` budget is the 271 s outlier in the server
    /// log; stalling the other three slots too would be worse.
    #[test]
    fn the_repetition_guard_stops_one_slot_only() {
        // The mock emits ascending ids, so a real n-gram probe would never fire; this uses
        // a probe keyed on a fixed length instead. What is tested here is the plumbing,
        // not `loop_period`, which has its own tests in `ocr.rs`.
        fn probe(gen: &[u32]) -> Option<usize> {
            // Page 1's ids start at 2000: stop that page after 3 tokens.
            (gen.len() == 3 && gen[0] / BASE == 2).then_some(9)
        }
        let c = SchedCfg { loop_probe: Some(probe), ..cfg(64) };
        let (_, done, _) = run(2, vec![10, 10], c);
        let p0 = done.iter().find(|d| d.page == 0).unwrap();
        let p1 = done.iter().find(|d| d.page == 1).unwrap();
        assert_eq!(p1.finish, Finish::Looped(9));
        assert_eq!(p1.tokens.len(), 3, "the token that tripped the guard is still emitted");
        assert_eq!(p0.finish, Finish::Stop(EOS), "the neighbour is unaffected");
        assert_eq!(p0.produced, 10);
    }

    /// The real guard, through the real function, on a real loop shape.
    #[test]
    fn the_real_loop_probe_wires_up() {
        fn probe(gen: &[u32]) -> Option<usize> {
            crate::ocr::loop_period(gen, crate::ocr::LOOP_MAX_PERIOD, crate::ocr::LOOP_MIN_RUN)
        }
        let mut s = Scheduler::new(1, 0..1, SchedCfg { loop_probe: Some(probe), ..cfg(4096) });
        let (slot, page) = s.next_admission().unwrap();
        s.seed(slot, page, Seed { first_token: 5, prompt_len: 4 });
        // Feed the same token back 24 times: `LOOP_MIN_RUN` at period 1.
        let mut fired = None;
        for _ in 0..64 {
            let steps = s.plan();
            if steps.is_empty() {
                break;
            }
            s.ingest(&steps, &[5]).unwrap();
            if let Some(d) = s.take_done().pop() {
                fired = Some(d);
            }
        }
        let d = fired.expect("24 identical tokens must trip the guard");
        assert_eq!(d.finish, Finish::Looped(1));
        assert_eq!(d.tokens.len(), crate::ocr::LOOP_MIN_RUN);
    }

    // ---- skips and resets --------------------------------------------------

    /// An unreadable page frees its slot for the next one and is still reported exactly
    /// once: a thousand-page batch must not lose a page or die on one.
    #[test]
    fn a_page_that_cannot_be_prepared_is_skipped_and_its_slot_reused() {
        let (log, done, stats) = run_with(2, vec![4, 4, 4, 4], cfg(64), vec![1, 2]);
        assert_eq!(stats.admitted, 4);
        assert_eq!(stats.completed, 4);
        assert_eq!(stats.skipped, 2);
        let skipped: Vec<usize> = done
            .iter()
            .filter(|d| d.finish == Finish::Skipped)
            .map(|d| d.page)
            .collect();
        assert_eq!(skipped, vec![1, 2]);
        for d in done.iter().filter(|d| d.finish != Finish::Skipped) {
            assert_eq!(d.produced, 4);
        }
        // A skipped page costs no slot time.
        assert_eq!(stats.tokens, 8);
        assert!(log.contains(&Ev::Prefill(1, 3)), "page 3 reused slot 1: {log:?}");
    }

    /// Invariant 2: every admission is preceded by a reset of that slot, the first page
    /// included. Without it a slot handed to page N still holds page N-1's KV rows, and
    /// since every image token is id 11 the result is a fluent transcription of the wrong
    /// page.
    #[test]
    fn every_admission_is_preceded_by_a_reset_of_its_slot() {
        let (log, _, _) = run(3, vec![2, 5, 1, 4, 3, 6], cfg(64));
        let mut resets = vec![0usize; 3];
        let mut prefills = vec![0usize; 3];
        for e in &log {
            match e {
                Ev::Reset(s) => resets[*s] += 1,
                Ev::Prefill(s, _) => {
                    prefills[*s] += 1;
                    assert_eq!(resets[*s], prefills[*s],
                               "slot {s} was prefilled without a fresh reset: {log:?}");
                }
                _ => {}
            }
        }
        assert_eq!(prefills.iter().sum::<usize>(), 6, "one prefill per page");
    }

    // ---- step assembly -----------------------------------------------------

    /// Positions are per slot and advance once per step. A shared counter would
    /// put page 2 at page 1's position and the page would decode against the
    /// wrong cache rows.
    #[test]
    fn positions_advance_per_slot_not_per_round() {
        let mut s = Scheduler::new(2, 0..2, cfg(64));
        let (s0, p0) = s.next_admission().unwrap();
        s.seed(s0, p0, Seed { first_token: BASE, prompt_len: 10 });
        let (s1, p1) = s.next_admission().unwrap();
        s.seed(s1, p1, Seed { first_token: 2 * BASE, prompt_len: 400 });
        assert_eq!(s.plan(), vec![(0, BASE, 9), (1, 2 * BASE, 399)]);
        s.ingest(&s.plan(), &[BASE + 1, 2 * BASE + 1]).unwrap();
        assert_eq!(s.plan(), vec![(0, BASE + 1, 10), (1, 2 * BASE + 1, 400)]);
        s.ingest(&s.plan(), &[BASE + 2, 2 * BASE + 2]).unwrap();
        assert_eq!(s.plan(), vec![(0, BASE + 2, 11), (1, 2 * BASE + 2, 401)]);
    }

    /// A decoder answering the wrong number of tokens, or answering for an empty slot,
    /// would otherwise show up as one page wearing another's tokens. Both are refused.
    #[test]
    fn a_mismatched_round_is_an_error_not_a_guess() {
        let mut s = Scheduler::new(2, 0..2, cfg(64));
        for _ in 0..2 {
            let (slot, page) = s.next_admission().unwrap();
            s.seed(slot, page, Seed { first_token: BASE * (page as u32 + 1), prompt_len: 9 });
        }
        let steps = s.plan();
        assert!(s.ingest(&steps, &[7]).is_err(), "two steps, one token");
        assert!(
            s.ingest(&[(1, BASE, 8)], &[7]).is_err(),
            "slot 1 holds page 1's token, not page 0's"
        );
        let mut empty = Scheduler::new(1, 0..0, cfg(64));
        assert!(empty.ingest(&[(0, 1, 0)], &[7]).is_err(), "slot 0 is idle");
    }

    /// A decoder that cannot batch after all must fail the run rather than silently
    /// transcribe nothing: by the time `decode_slots` is called the pages are already
    /// prefilled into slots and there is no sequential path left to fall back to. The
    /// fallback decision is made earlier, on `max_slots()`.
    #[test]
    fn a_decoder_that_refuses_a_batched_step_fails_the_run() {
        struct NoBatch;
        impl SlotModel for NoBatch {
            fn max_slots(&self) -> usize {
                4
            }
            fn reset_slot(&self, _s: usize) {}
            fn decode_slots(&self, _s: &[(usize, u32, usize)]) -> Option<Vec<u32>> {
                None
            }
        }
        struct One;
        impl PageSource for One {
            fn admit(&mut self, _s: usize, _p: usize) -> Result<Option<Seed>> {
                Ok(Some(Seed { first_token: BASE, prompt_len: 4 }))
            }
            fn finished(&mut self, _d: Done) -> Result<()> {
                Ok(())
            }
        }
        let mut s = Scheduler::new(4, 0..2, cfg(64));
        let err = drive(&NoBatch, &mut One, &mut s).unwrap_err();
        assert!(format!("{err:#}").contains("decode_slots"), "{err:#}");
    }

    /// An empty batch is not an error and not a spin.
    #[test]
    fn no_pages_is_a_no_op() {
        let m = Mock::new(4, Vec::new(), 9);
        let mut src = Pages { m: &m, bad: Vec::new(), done: Vec::new() };
        let mut s = Scheduler::new(4, 0..0, cfg(64));
        drive(&m, &mut src, &mut s).unwrap();
        assert_eq!(s.stats(), Stats::default());
        assert!(s.idle());
        assert!(m.log().is_empty(), "nothing to reset, nothing to step");
        assert!(src.done.is_empty());
    }

    /// Occupancy is slots-with-work over slots-available.
    #[test]
    fn occupancy_counts_idle_slots() {
        let mut st = Stats { rounds: 10, slot_steps: 30, ..Stats::default() };
        assert_eq!(st.occupancy(4), 0.75);
        st.slot_steps = 40;
        assert_eq!(st.occupancy(4), 1.0);
        assert_eq!(Stats::default().occupancy(4), 0.0, "no rounds, no occupancy");
    }
}

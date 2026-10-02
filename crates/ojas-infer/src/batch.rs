//! Several generations at once, through the model's sequence slots.
//!
//! Each request holds one slot from admission to its last token. A step admits
//! waiting requests into free slots, processes one chunk of one prompt, then decodes
//! one token for every slot that is generating, all of them in one `decode_slots`
//! pass that reads the weights once. Prompt chunks and decode steps alternate, so a
//! long new prompt delays the replies already streaming by one chunk per token
//! rather than by the whole prompt.
//!
//! Every token is decoded through `decode_slots`, whether the request is alone or
//! not, so its output does not depend on what else is running: a greedy request is
//! byte-identical alone or beside three others. Token choice is the `Picker` the
//! single-sequence path uses. Speculative decoding belongs to that path alone.
//!
//! When every slot is busy, the waiting request whose prompt the prefix cache can
//! restore furthest goes next; one that has waited `max_wait` goes first instead,
//! oldest first, so none starves.

use crate::{FinishReason, Generation, LogitProcessor, Picker, SampleOpts};
use ojas_core::Model;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Prompt tokens processed per step for one request, between decode steps.
const PREFILL_CHUNK: usize = 256;

/// A generation to run.
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    /// `None` or a temperature of 0 for greedy decoding.
    pub opts: Option<SampleOpts>,
    pub processor: Option<Box<dyn LogitProcessor>>,
    /// Ids that end the generation; emitted as its last token.
    pub stop: Vec<u32>,
    /// Ids never chosen.
    pub banned: Vec<u32>,
    /// Message boundaries in the prompt, for the prefix cache.
    pub marks: Vec<usize>,
    /// Token ranges the document cache may serve (`Model::set_prefix_docs`).
    pub docs: Vec<(usize, usize)>,
    /// Whether the prompt may be restored from the prefix cache.
    pub reuse: bool,
}

/// What happens to a request, reported as it happens.
#[derive(Debug)]
pub enum Event {
    /// Prompt tokens processed so far, of the total.
    Prefill(usize, usize),
    Token(u32),
    Done(Generation),
}

/// Receives each request's events; returning false ends that request.
pub type Sink<'s> = dyn FnMut(u64, Event) -> bool + 's;

struct Waiting {
    id: u64,
    req: Request,
    arrived: Instant,
}

enum Phase {
    /// Processing the prompt; this many of its tokens (all but the last) are done.
    Prompt(usize),
    Decode,
}

struct Running {
    id: u64,
    req: Request,
    picker: Picker,
    phase: Phase,
    /// The token to feed next, and the position to feed it at.
    cur: u32,
    pos: usize,
    out: Vec<u32>,
    cached_tokens: usize,
    prompt_cache: Option<ojas_core::PrefixRestore>,
}

impl Running {
    /// Whether the next token is chosen on the host from logits rather than taken
    /// as the device's argmax.
    fn picks_on_host(&self) -> bool {
        self.picker.sampled() || !self.req.banned.is_empty() || self.req.processor.is_some()
    }
}

pub struct Batch<'m, M: Model + ?Sized> {
    model: &'m M,
    slots: Vec<Option<Running>>,
    waiting: VecDeque<Waiting>,
    max_wait: Duration,
    /// The slot whose prompt the next prefill step serves, rotated so two long
    /// prompts progress together.
    prompt_turn: usize,
}

impl<'m, M: Model + ?Sized> Batch<'m, M> {
    pub fn new(model: &'m M, max_wait: Duration) -> Self {
        Batch {
            model, slots: (0..model.max_slots()).map(|_| None).collect(), waiting: VecDeque::new(), max_wait,
            prompt_turn: 0,
        }
    }

    pub fn submit(&mut self, id: u64, req: Request) {
        self.waiting.push_back(Waiting { id, req, arrived: Instant::now() });
    }

    /// Whether nothing is running or waiting.
    pub fn is_idle(&self) -> bool { self.waiting.is_empty() && self.slots.iter().all(Option::is_none) }

    /// A slot no request holds, for work outside the batch between steps.
    pub fn free_slot(&self) -> Option<usize> { self.slots.iter().position(Option::is_none) }

    /// Requests running and waiting.
    pub fn load(&self) -> (usize, usize) { (self.slots.iter().flatten().count(), self.waiting.len()) }

    /// Admit, process one prompt chunk, and decode one token for every generating
    /// slot.
    pub fn step(&mut self, sink: &mut Sink<'_>) {
        self.admit(sink);
        self.prompt_step(sink);
        self.decode_step(sink);
    }

    fn admit(&mut self, sink: &mut Sink<'_>) {
        while let Some(slot) = self.slots.iter().position(Option::is_none) {
            let Some(i) = self.next_waiting() else { return };
            let w = self.waiting.remove(i).unwrap();
            self.start(slot, w, sink);
        }
    }

    /// The waiting request to admit next: the oldest one past `max_wait`, else the
    /// one the prefix cache can restore furthest, earliest first among equals.
    fn next_waiting(&self) -> Option<usize> {
        let now = Instant::now();
        if let Some(i) = self.waiting.iter().position(|w| now - w.arrived >= self.max_wait) { return Some(i); }
        let cached = |w: &Waiting| self.model.cached_prefix_len(&w.req.prompt[..w.req.prompt.len().saturating_sub(1)]);
        (0..self.waiting.len()).rev().max_by_key(|&i| cached(&self.waiting[i]))
    }

    fn start(&mut self, slot: usize, w: Waiting, sink: &mut Sink<'_>) {
        let Waiting { id, mut req, .. } = w;
        let capacity = self.model.context_capacity();
        if req.prompt.is_empty() || req.max_tokens == 0 || req.prompt.len() > capacity {
            let finish = if req.prompt.len() > capacity { FinishReason::NoLogits } else { FinishReason::Length };
            sink(id, Event::Done(Generation { tokens: Vec::new(), finish, cached_tokens: 0, prompt_cache: None }));
            return;
        }
        req.max_tokens = req.max_tokens.min(capacity.saturating_sub(req.prompt.len()).saturating_add(1));
        let pre = &req.prompt[..req.prompt.len() - 1];
        let model = self.model;
        let (mut start, mut prompt_cache) = (0, None);
        let addressed = model.with_slot(slot, &mut || {
            model.set_prefix_reuse(req.reuse);
            if !req.reuse { model.reset_session(); }
            model.set_prefix_marks(&req.marks);
            model.set_prefix_docs(&req.docs, req.reuse);
            start = model.reuse_prefix_len(pre).min(pre.len());
            prompt_cache = model.prefix_cache_stats().map(|s| s.last);
            if pre.is_empty() { model.prefill(&[], 0); }
        });
        assert!(addressed, "slot {slot} is within max_slots, so the model addresses it");
        let total = pre.len();
        let running = Running {
            id, picker: Picker::new(&req.prompt, req.opts.as_ref()),
            phase: if start < total { Phase::Prompt(start) } else { Phase::Decode },
            cur: *req.prompt.last().unwrap(), pos: total, out: Vec::new(),
            cached_tokens: prompt_cache.map_or(start, |r| r.reused_tokens), prompt_cache, req,
        };
        self.slots[slot] = Some(running);
        if !sink(id, Event::Prefill(start, total)) { self.finish(slot, FinishReason::Caller, sink); }
    }

    /// Process the next chunk of one prompt still in progress.
    fn prompt_step(&mut self, sink: &mut Sink<'_>) {
        let n = self.slots.len();
        let Some(slot) = (0..n).map(|k| (self.prompt_turn + k) % n)
            .find(|&s| matches!(self.slots[s].as_ref().map(|r| &r.phase), Some(Phase::Prompt(_))))
        else { return };
        self.prompt_turn = (slot + 1) % n;
        let model = self.model;
        let r = self.slots[slot].as_mut().unwrap();
        let Phase::Prompt(done) = r.phase else { unreachable!() };
        let total = r.req.prompt.len() - 1;
        let end = (done + PREFILL_CHUNK).min(total);
        let chunk = &r.req.prompt[done..end];
        model.with_slot(slot, &mut || model.prefill(chunk, done));
        r.phase = if end == total { Phase::Decode } else { Phase::Prompt(end) };
        let id = r.id;
        if ojas_core::device_fault::is_faulted() {
            self.finish(slot, FinishReason::Fault, sink);
        } else if !sink(id, Event::Prefill(end, total)) {
            self.finish(slot, FinishReason::Caller, sink);
        }
    }

    /// Decode one token for every slot past its prompt, in one pass.
    fn decode_step(&mut self, sink: &mut Sink<'_>) {
        let slots: Vec<usize> = (0..self.slots.len())
            .filter(|&s| matches!(self.slots[s].as_ref().map(|r| &r.phase), Some(Phase::Decode)))
            .collect();
        if slots.is_empty() { return; }
        let steps: Vec<(usize, u32, usize)> = slots.iter()
            .map(|&s| self.slots[s].as_ref().map(|r| (s, r.cur, r.pos)).unwrap())
            .collect();
        let ids = self.model.decode_slots(&steps);
        let faulted = ojas_core::device_fault::is_faulted();
        for (i, &slot) in slots.iter().enumerate() {
            let outcome = match (&ids, faulted) {
                (_, true) => Err(FinishReason::Fault),
                (None, _) => Err(FinishReason::NoLogits),
                (Some(ids), _) => self.choose(slot, i, ids[i]),
            };
            match outcome {
                Ok(t) => self.emit(slot, t, sink),
                Err(finish) => self.finish(slot, finish, sink),
            }
        }
    }

    /// Slot `slot`'s next token: entry `i` of the last decode step, `argmax` its
    /// device argmax.
    fn choose(&mut self, slot: usize, i: usize, argmax: u32) -> Result<u32, FinishReason> {
        let r = self.slots[slot].as_mut().unwrap();
        if !r.picks_on_host() { return Ok(argmax); }
        let mut logits = self.model.slot_logits(i).ok_or(FinishReason::NoLogits)?;
        r.picker.pick(&mut logits, &r.req.banned, r.req.processor.as_deref_mut()).ok_or(FinishReason::NoAllowedToken)
    }

    /// Append a decoded token and end the request if it is done, checking in the
    /// single-sequence path's order: a stop id, the caller, a finished processor,
    /// then the length limit.
    fn emit(&mut self, slot: usize, t: u32, sink: &mut Sink<'_>) {
        let r = self.slots[slot].as_mut().unwrap();
        r.out.push(t);
        r.cur = t;
        r.pos += 1;
        let finish = if r.req.stop.contains(&t) {
            Some(FinishReason::Stop)
        } else if !sink(r.id, Event::Token(t)) {
            Some(FinishReason::Caller)
        } else if r.req.processor.as_ref().is_some_and(|p| p.finished()) {
            Some(FinishReason::Complete)
        } else {
            r.picker.note(t);
            (r.out.len() >= r.req.max_tokens).then_some(FinishReason::Length)
        };
        if let Some(finish) = finish { self.finish(slot, finish, sink); }
    }

    fn finish(&mut self, slot: usize, finish: FinishReason, sink: &mut Sink<'_>) {
        let r = self.slots[slot].take().unwrap();
        sink(r.id, Event::Done(Generation {
            tokens: r.out, finish, cached_tokens: r.cached_tokens, prompt_cache: r.prompt_cache,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;

    const VOCAB: usize = 64;
    const STOP: u32 = 0;

    /// A slotted model whose next token depends only on the slot's own history: the
    /// sum of every token fed to the slot, so a slot that saw another slot's tokens
    /// or skipped one answers differently.
    struct Mock {
        slots: usize,
        cur: Cell<usize>,
        state: RefCell<Vec<u64>>,
        cached: HashMap<usize, usize>,
        calls: RefCell<Vec<String>>,
    }

    impl Mock {
        fn new(slots: usize) -> Self {
            Mock { slots, cur: Cell::new(0), state: RefCell::new(vec![0; slots]), cached: HashMap::new(),
                   calls: RefCell::new(Vec::new()) }
        }
        fn next(state: u64) -> u32 { (state * 7 % 61) as u32 }
    }

    impl Model for Mock {
        fn n_layers(&self) -> usize { 1 }
        fn hidden_dim(&self) -> usize { 1 }
        fn context_capacity(&self) -> usize { 4096 }
        fn prefill(&self, tokens: &[u32], base: usize) {
            let s = self.cur.get();
            if base == 0 { self.state.borrow_mut()[s] = 0; }
            self.state.borrow_mut()[s] += tokens.iter().map(|&t| t as u64).sum::<u64>();
            self.calls.borrow_mut().push(format!("prefill {s} {base}+{}", tokens.len()));
        }
        fn forward_id(&self, _: u32, _: usize) -> u32 { unreachable!("batched requests decode through slots") }
        fn max_slots(&self) -> usize { self.slots }
        fn with_slot(&self, s: usize, f: &mut dyn FnMut()) -> bool {
            self.cur.set(s);
            f();
            self.cur.set(0);
            true
        }
        fn reuse_prefix_len(&self, _: &[u32]) -> usize { 0 }
        fn cached_prefix_len(&self, tokens: &[u32]) -> usize { self.cached.get(&tokens.len()).copied().unwrap_or(0) }
        fn decode_slots(&self, steps: &[(usize, u32, usize)]) -> Option<Vec<u32>> {
            self.calls.borrow_mut().push(format!("decode {}", steps.len()));
            let mut state = self.state.borrow_mut();
            Some(steps.iter().map(|&(s, t, _)| {
                state[s] += t as u64;
                Self::next(state[s])
            }).collect())
        }
        fn slot_logits(&self, i: usize) -> Option<Vec<f32>> {
            let last = self.calls.borrow().iter().rev().find_map(|c| c.strip_prefix("decode ").map(str::to_owned));
            assert!(last.is_some() && i < last.unwrap().parse::<usize>().unwrap());
            Some((0..VOCAB).map(|t| -((t as f32) - 9.0).abs()).collect())
        }
    }

    fn request(prompt: Vec<u32>, max_tokens: usize) -> Request {
        Request { prompt, max_tokens, opts: None, processor: None, stop: vec![STOP], banned: Vec::new(),
                  marks: Vec::new(), docs: Vec::new(), reuse: true }
    }

    /// Run `reqs` to completion and return each one's generation by id.
    fn run(model: &Mock, reqs: Vec<Request>) -> HashMap<u64, Generation> {
        let mut batch = Batch::new(model, Duration::from_secs(3600));
        for (id, r) in reqs.into_iter().enumerate() { batch.submit(id as u64, r); }
        let mut done = HashMap::new();
        while !batch.is_idle() {
            batch.step(&mut |id, e| {
                if let Event::Done(g) = e { done.insert(id, g); }
                true
            });
        }
        done
    }

    fn prompts() -> Vec<Vec<u32>> {
        vec![(1..700).map(|i| i % 50 + 1).collect(), vec![3, 4, 5], (1..300).map(|i| i % 9 + 2).collect(), vec![8; 40]]
    }

    #[test]
    fn output_does_not_depend_on_what_else_is_running() {
        let alone: Vec<Vec<u32>> = prompts().into_iter()
            .map(|p| run(&Mock::new(4), vec![request(p, 30)]).remove(&0).unwrap().tokens)
            .collect();
        for slots in [1, 2, 4] {
            let together = run(&Mock::new(slots), prompts().into_iter().map(|p| request(p, 30)).collect());
            for (id, tokens) in alone.iter().enumerate() {
                assert_eq!(&together[&(id as u64)].tokens, tokens, "request {id} with {slots} slots");
            }
        }
    }

    #[test]
    fn long_prompts_are_processed_between_decode_steps() {
        let model = Mock::new(2);
        run(&model, vec![request(vec![1, 2], 20), request((0..1000).map(|i| i % 9 + 1).collect(), 2)]);
        let calls = model.calls.borrow();
        let first_chunk = calls.iter().position(|c| c.starts_with("prefill 1 0+256")).unwrap();
        let last_chunk = calls.iter().rposition(|c| c.starts_with("prefill 1 768+")).unwrap();
        let decodes_between = calls[first_chunk..last_chunk].iter().filter(|c| c.starts_with("decode")).count();
        assert_eq!(decodes_between, 3, "one decode step after each chunk: {calls:?}");
    }

    #[test]
    fn a_full_batch_admits_the_longest_cached_prompt_first_unless_one_is_overdue() {
        let mut model = Mock::new(1);
        model.cached.insert(9, 256);
        let reqs = || vec![request(vec![5; 4], 1), request(vec![6; 10], 1), request(vec![7; 6], 1)];
        let mut order = Vec::new();
        let mut batch = Batch::new(&model, Duration::from_secs(3600));
        for (id, r) in reqs().into_iter().enumerate() { batch.submit(id as u64, r); }
        while !batch.is_idle() {
            batch.step(&mut |id, e| {
                if matches!(e, Event::Prefill(..)) && !order.contains(&id) { order.push(id); }
                true
            });
        }
        assert_eq!(order, vec![1, 0, 2]);
        let mut batch = Batch::new(&model, Duration::ZERO);
        for (id, r) in reqs().into_iter().enumerate() { batch.submit(id as u64, r); }
        let mut first = None;
        batch.step(&mut |id, _| {
            first.get_or_insert(id);
            true
        });
        assert_eq!(first, Some(0), "every request is overdue, so the oldest goes first");
    }

    #[test]
    fn requests_end_on_a_stop_id_the_length_limit_or_the_caller() {
        let model = Mock::new(2);
        let gens = run(&model, vec![request(vec![30, 31], 1000), request(vec![2], 3)]);
        assert_eq!(gens[&0].finish, FinishReason::Stop);
        assert_eq!(gens[&0].tokens.last(), Some(&STOP));
        assert_eq!((gens[&1].finish, gens[&1].tokens.len()), (FinishReason::Length, 3));
        let mut batch = Batch::new(&model, Duration::from_secs(3600));
        batch.submit(7, request(vec![4, 5], 100));
        let mut seen = 0;
        let mut finish = None;
        while !batch.is_idle() {
            batch.step(&mut |_, e| match e {
                Event::Token(_) => { seen += 1; seen < 2 }
                Event::Done(g) => { finish = Some((g.finish, g.tokens.len())); true }
                Event::Prefill(..) => true,
            });
        }
        assert_eq!(finish, Some((FinishReason::Caller, 2)));
    }

    #[test]
    fn a_processor_constrains_through_the_slot_logits() {
        struct OnlyOdd(usize);
        impl LogitProcessor for OnlyOdd {
            fn allows(&mut self, t: u32) -> bool { t % 2 == 1 }
            fn accept(&mut self, _: u32) { self.0 += 1; }
            fn finished(&self) -> bool { self.0 == 3 }
        }
        let model = Mock::new(2);
        let mut req = request(vec![1, 2, 3], 50);
        req.processor = Some(Box::new(OnlyOdd(0)));
        let gens = run(&model, vec![req, request(vec![9], 4)]);
        assert_eq!(gens[&0].tokens, vec![9, 9, 9], "the logits peak at 9, which is odd");
        assert_eq!(gens[&0].finish, FinishReason::Complete);
    }
}

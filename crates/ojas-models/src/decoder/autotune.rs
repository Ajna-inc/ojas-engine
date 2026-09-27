#![allow(clippy::too_many_arguments)]
use super::*;
use objc::{msg_send, sel, sel_impl};
use metal::MTLSize;
use super::dispatch::q4l_tg;
use std::ffi::c_void;
 // re-export

impl<'a> DecoderGpu<'a> {
    /// Device-adaptive autotune (run once at load). Picks the fastest GEMV config per
    /// matmul shape by micro-benchmarking on this GPU, so tile sizes and split-K adapt
    /// across Apple GPUs (7-core M1 to 80-core Ultra) instead of being hardcoded for
    /// one chip. SIMD width is 32 on all Apple GPUs; thread counts are clamped to the
    /// device's max_threads_per_threadgroup.
    ///
    /// Compact signature of the shapes that make a pseudo-keyed plan model-specific.
    /// Real (K,N) entries are model-independent — a 2048x2048 gemv is the same matmul
    /// whatever produced it — so they stay in the shared namespace and keep cross-model
    /// reuse. The pseudo keys ((0,1) ffn_gu, (0,2) attention, (0,3) ffn_gu_q4l) are
    /// not: each stands for a whole fused kernel whose cost depends on this model's
    /// dims. Keyed by device alone, the second model to load inherits the first's plan:
    /// measured, a 9B model tunes (0,3)=128 alone but loaded the 3B's 256 from a shared
    /// cache and never tuned its own.
    pub(crate) fn arch_tag(&self) -> String {
        format!("a{}-{}-{}-{}-{}", self.d, self.arch.ffn, self.arch.hd,
                self.arch.n_head, self.arch.n_kv)
    }

    pub(crate) fn autotune(&mut self) {
        // cache key includes precision — q4 and q8 have different optimal configs.
        // "#q4v2" version tag: bump when q4 kernels change so stale caches (with configs
        // for the old kernels) are ignored instead of applied wrongly.
        let dev = if self.wt.q4 { format!("{}#q4v2", self.gpu.name()) } else { self.gpu.name() };
        let ew = self.gpu.device.max_threads_per_threadgroup().width;
        let d = self.d as u32;
        let ffn = self.arch.ffn as u32;
        let vocab = self.arch.vocab as u32;
        // distinct GEMV shapes used by mm(): (K, N, representative layer-0 weight)
        let qdim = (self.arch.n_head * self.arch.hd) as u32;       // o_proj input (Qwen3: qdim ≠ d)
        let shapes: [(u32, u32, String); 3] = [
            (qdim, d, "blk.0.attn_output.weight".into()), // o_proj
            (ffn, d, "blk.0.ffn_down.weight".into()),    // ffn_down (long K)
            (d, vocab, "token_embd.weight".into()),      // lm_head
        ];
        // Load cached plans (keyed by device+shape) → tune once per device, reuse.
        let cache = load_tune_cache();
        let atag = self.arch_tag();
        let pdev = format!("{dev}#{atag}");     // namespace for pseudo keys
        let mut tuned = false;
        let mut tuned_pseudo = false;
        let mut get = |key: (u32, u32), tune: &dyn Fn() -> GemvPlan| -> GemvPlan {
            let ns = if key.0 == 0 { &pdev } else { &dev };   // pseudo keys are arch-scoped
            if let Some(&p) = cache.get(&(ns.clone(), key.0, key.1)) { p }
            else { if key.0 == 0 { tuned_pseudo = true; } else { tuned = true; } tune() }
        };
        let q4 = self.wt.q4;
        for (k, n, wname) in shapes.iter() {
            // skip shapes whose representative tensor is absent (MoE has no dense
            // ffn_down; hybrid blk.0 has no attn_output) — mm() falls back fine.
            if !(if q4 { self.wt.w4.contains_key(wname) } else { self.wt.w8.contains_key(wname) || self.wt.w16.contains_key(wname) }) { continue; }
            let p = get((*k, *n), &|| if q4 { self.tune_gemv_q4(*k, *n, wname) } else { self.tune_gemv(*k, *n, wname) });
            self.tune.gemv_plan.insert((*k, *n), p);
        }
        // ffn_gu (pseudo-key (0,1)) — precision-specific tuner.
        if if q4 { self.wt.w4.contains_key("blk.0.ffn_gate.weight") } else { self.wt.w8.contains_key("blk.0.ffn_gate.weight") || self.wt.w16.contains_key("blk.0.ffn_gate.weight") } {
            let p = get((0, 1), &|| if q4 { self.tune_ffn_gu_q4() } else { self.tune_ffn_gu() });
            self.tune.gemv_plan.insert((0, 1), p);
        }
        // attention (pseudo-key (0,2)) — same kernel regardless of weight precision.
        let p = get((0, 2), &|| self.tune_attn());
        self.tune.gemv_plan.insert((0, 2), p);
        drop(get);
        // ---- Q4L family: threadgroup size per shape ------------------------
        // The winner is set by threadgroups-produced against threadgroups-needed, so
        // it cannot be a constant across Apple GPUs (the hardcoded `n >= 4096 -> 4,
        // long-K narrow-N -> 8` rule it replaced was measured on one M2 Max).
        //
        // Cached under its own device namespace, which keeps the TSV format unchanged
        // and avoids a key collision: `dev` carries the "#q4v2" tag only when wt.q4,
        // which prec=3 (Q4L) is not, so a Q8-tuned plan and a Q4L run would share a
        // key. "#q4l1" is the version tag — bump it when a Q4L kernel changes shape so
        // stale plans are ignored rather than applied to a kernel they were never
        // measured on.
        let q4ldev = format!("{dev}#q4l1");
        let q4lpdev = format!("{q4ldev}#{atag}");
        let layers = |suffix: &str| -> Vec<String> {
            (0..self.arch.n_layers).map(|l| format!("blk.{l}.{suffix}"))
                .filter(|nm| self.wt.w4l.contains_key(nm)).collect()
        };
        let qdim0 = (self.arch.n_head * self.arch.hd) as u32;
        let mut q4l_tuned = false;
        // gemv_q4l shapes, mirroring mm()'s w4l branch
        for (k, n, suffix) in [(qdim0, d, "attn_output.weight"), (ffn, d, "ffn_down.weight")] {
            let names = layers(suffix);
            if names.is_empty() { continue; }
            let t = match cache.get(&(q4ldev.clone(), k, n)) {
                Some(&p) => p.threads as u32,
                None => { q4l_tuned = true; self.tune_q4l_gemv(k, n, &names) }
            };
            self.tune.q4l_tg.insert((k, n), t);
        }
        // fused ffn_gu_q4l (pseudo-key (0,3))
        {
            let (g, u) = (layers("ffn_gate.weight"), layers("ffn_up.weight"));
            if !g.is_empty() && !u.is_empty() {
                let t = match cache.get(&(q4lpdev.clone(), 0, 3)) {
                    Some(&p) => p.threads as u32,
                    None => { q4l_tuned = true; self.tune_q4l_ffn_gu(&g, &u) }
                };
                self.tune.q4l_tg.insert((0, 3), t);
            }
        }
        if q4l_tuned {
            let mk = |pseudo: bool| -> HashMap<(u32, u32), GemvPlan> {
                self.tune.q4l_tg.iter().filter(|((k, _), _)| (*k == 0) == pseudo)
                    .map(|(&kn, &t)| (kn, GemvPlan { ksplit: false, threads: t as u64 })).collect()
            };
            save_tune_cache(&q4ldev, &mk(false));
            save_tune_cache(&q4lpdev, &mk(true));
            tracing::debug!(target: "autotune", "{q4ldev}: tuned Q4L threadgroups {:?}", self.tune.q4l_tg);
        }

        let xvdev = format!("{dev}#xv1");
        let qdim = (self.arch.n_head * self.arch.hd) as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        // A representative tensor per shape, and not blk.0: mixed-precision GGUFs
        // (Q4_K_S ships some layers' ffn_down / attn_v as Q6_K) leave layer 0 out of
        // w4l while most layers are in it, so anchoring on 0 skips three of the four
        // shapes. Take the first layer that has one.
        let nl = self.arch.n_layers;
        let xv_shapes: [(u32, u32, &str); 4] = [
            (d, qdim, "attn_q.weight"),
            (d, kvdim, "attn_k.weight"),
            (d, ffn, "ffn_gate.weight"),
            (ffn, d, "ffn_down.weight"),
        ];
        let mut xv_tuned = false;
        for (k, n, suffix) in xv_shapes.iter() {
            let names: Vec<String> = (0..nl).map(|l| format!("blk.{l}.{suffix}"))
                .filter(|nm| self.wt.w4l.contains_key(nm)).collect();
            if names.is_empty() { continue; }
            if self.tune.xv_plan.contains_key(&(*k, *n)) { continue; }
            let nx = match cache.get(&(xvdev.clone(), *k, *n)) {
                Some(&p) => p.threads as u32,
                None => { xv_tuned = true; self.tune_xv(*k, *n, &names) }
            };
            self.tune.xv_plan.insert((*k, *n), nx);
        }
        if xv_tuned {
            let asplan: HashMap<(u32, u32), GemvPlan> = self.tune.xv_plan.iter()
                .map(|(&kn, &nx)| (kn, GemvPlan { ksplit: false, threads: nx as u64 })).collect();
            save_tune_cache(&xvdev, &asplan);
            tracing::debug!(target: "autotune", "{xvdev}: tuned verify lanes/row {:?}", self.tune.xv_plan);
        }
        if tuned_pseudo {
            let ps: HashMap<(u32, u32), GemvPlan> = self.tune.gemv_plan.iter()
                .filter(|((k, _), _)| *k == 0).map(|(&kn, &p)| (kn, p)).collect();
            save_tune_cache(&pdev, &ps);
        }
        if tuned || tuned_pseudo {
            let shp: HashMap<(u32, u32), GemvPlan> = self.tune.gemv_plan.iter()
                .filter(|((k, _), _)| *k != 0).map(|(&kn, &p)| (kn, p)).collect();
            save_tune_cache(&dev, &shp);
            tracing::debug!(target: "autotune", "{dev} (max {ew} threads/tg): tuned {:?}", self.tune.gemv_plan);
        } else {
            tracing::debug!(target: "autotune", "{dev}: loaded cached plans {:?}", self.tune.gemv_plan);
        }
    }

    /// Pick the candidate with the lowest GPU time. Warms up all, then interleaves
    /// candidates across rounds (so thermal drift biases none) and takes each one's min.
    pub(crate) fn pick_best<T: Copy>(&self, cands: &[T], run: impl Fn(T) -> f64) -> T {
        for &c in cands { run(c); run(c); }
        let mut best = vec![f64::INFINITY; cands.len()];
        for _ in 0..7 {
            for (i, &c) in cands.iter().enumerate() { best[i] = best[i].min(run(c)); }
        }
        let ok = plausible(&best);
        let mut bi = usize::MAX;
        for i in 0..cands.len() {
            if !ok[i] { continue; }
            if bi == usize::MAX || best[i] < best[bi] { bi = i; }
        }
        cands[if bi == usize::MAX { 0 } else { bi }]
    }

    /// Time `reps` dispatches of a kernel in one command buffer (ms). `setup` binds
    /// buffers/bytes on the encoder; `grid`/`threads` are the dispatch dims.
    pub(crate) fn time_kernel(&self, kern: &str, grid: u64, threads: u64, reps: u32,
                   setup: impl Fn(&metal::ComputeCommandEncoderRef)) -> f64 {
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.p[kern]);
        setup(&enc);
        for _ in 0..reps { enc.dispatch_thread_groups(MTLSize::new(grid, 1, 1), MTLSize::new(threads, 1, 1)); }
        enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "autotune probe");
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        (ge - gs) * 1e3
    }

    /// Benchmark candidate GEMV configs for one (K,N) shape; return the fastest.
    pub(crate) fn tune_gemv(&self, k: u32, n: u32, wname: &str) -> GemvPlan {
        let w = &self.wt.w8[wname];
        let s = &self.wt.scale8[wname];
        // Clamp to each kernel's own limit, not the device's — see pipe_max().
        let cap_p = self.pipe_max("gemv_q8") as u64;
        let cap_s = self.pipe_max("gemv_q8_ksplit") as u64;
        let mut cands: Vec<GemvPlan> = Vec::new();
        for &t in &[128u64, 256, 512, 1024] {
            if t <= self.tune.max_tg.min(cap_p) { cands.push(GemvPlan { ksplit: false, threads: t }); }
        }
        if k >= 1024 {
            for &t in &[128u64, 256, 512] {
                if t <= self.tune.max_tg.min(cap_s) { cands.push(GemvPlan { ksplit: true, threads: t }); }
            }
        }
        if cands.is_empty() { return GemvPlan { ksplit: false, threads: 64 }; }
        self.pick_best(&cands, |pl| {
            let kern = if pl.ksplit { "gemv_q8_ksplit" } else { "gemv_q8" };
            let rows = (pl.threads / 32).max(1) as u32;
            let (grid, per) = if pl.ksplit { (n as u64, pl.threads) } else { (((n + rows - 1) / rows) as u64, pl.threads) };
            self.time_kernel(kern, grid, per, 40, |enc| {
                enc.set_buffer(0, Some(&self.st.act), 0); // x (garbage ok; timing only). act ≥ max K.
                enc.set_buffer(1, Some(w), 0);
                enc.set_buffer(2, Some(&self.st.tmp), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(5, Some(s), 0);
            })
        })
    }

    /// Q4 analog of tune_gemv (gemv_q4 / gemv_q4_ksplit, nibbles + f16 scale, no min).
    /// The q4 fast GEMV is a fixed-config qmv_fast kernel (64 threads); autotuning the
    /// thread count regressed, because an isolated micro-benchmark mispredicts the real
    /// interleaved forward. Returns the fixed config.
    pub(crate) fn tune_gemv_q4(&self, _k: u32, _n: u32, _wname: &str) -> GemvPlan {
        GemvPlan { ksplit: false, threads: 64 }
    }

    /// Tune ffn_gu (fused gate/up SwiGLU, K=d N=ffn) thread count → GemvPlan{threads}.
    pub(crate) fn tune_ffn_gu(&self) -> GemvPlan {
        let (k, n) = (self.d as u32, self.arch.ffn as u32);
        let wg = &self.wt.w8["blk.0.ffn_gate.weight"];
        let wu = &self.wt.w8["blk.0.ffn_up.weight"];
        let sg = &self.wt.scale8["blk.0.ffn_gate.weight"];
        let su = &self.wt.scale8["blk.0.ffn_up.weight"];
        let cap = self.pipe_max("ffn_gu_q8") as u64;
        let cands: Vec<u64> = [128u64, 256, 512, 1024].into_iter()
            .filter(|&t| t <= self.tune.max_tg.min(cap)).collect();
        if cands.is_empty() { return GemvPlan { ksplit: false, threads: 64 }; }
        let threads = self.pick_best(&cands, |t| {
            let rows = (t / 32).max(1) as u32;
            let grid = ((n + rows - 1) / rows) as u64;
            self.time_kernel("ffn_gu_q8", grid, t, 30, |enc| {
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(wg), 0); enc.set_buffer(2, Some(wu), 0); enc.set_buffer(3, Some(&self.st.act), 0);
                enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(6, Some(sg), 0); enc.set_buffer(7, Some(su), 0);
                let a = self.arch.gelu as u32; enc.set_bytes(8, 4, &a as *const u32 as *const c_void);
            })
        });
        GemvPlan { ksplit: false, threads }
    }

    /// ffn_gu_q4 is a fixed-config qmv_fast kernel (64 threads); autotuning regressed. Stub.
    pub(crate) fn tune_ffn_gu_q4(&self) -> GemvPlan {
        GemvPlan { ksplit: false, threads: 64 }
    }

    /// Rejects implausibly fast candidates the same way `pick_best` does: a dispatch
    /// past the pipeline's thread limit does not run and times at ~0. See `plausible`.

    /// `pick_best` with hysteresis: a candidate must beat `pref` by more than
    /// `margin` to displace it, so the tuner does not chase noise — without it the
    /// same shape picked 256 on one run and 512 on the next when the candidates were
    /// within a percent of each other, churning the cache and making every A/B against
    /// it unrepeatable. On a near-tie the incumbent wins.
    pub(crate) fn pick_best_pref<T: Copy + PartialEq>(&self, cands: &[T], pref: T, margin: f64,
                                                      run: impl Fn(T) -> f64) -> T {
        for &c in cands { run(c); run(c); }
        let mut best = vec![f64::INFINITY; cands.len()];
        for _ in 0..7 {
            for (i, &c) in cands.iter().enumerate() { best[i] = best[i].min(run(c)); }
        }
        let ok = plausible(&best);
        let mut bi = usize::MAX;
        for i in 0..cands.len() {
            if !ok[i] { continue; }
            if bi == usize::MAX || best[i] < best[bi] { bi = i; }
        }
        if bi == usize::MAX { return pref; }        // nothing plausible: keep the default
        if let Some(pi) = cands.iter().position(|&c| c == pref) {
            if ok[pi] && best[pi] <= best[bi] * (1.0 + margin) { return pref; }
        }
        cands[bi]
    }

    /// Max threads/threadgroup this pipeline supports. The device limit (1024 on
    /// Apple) is an upper bound; a register-heavy kernel's own limit is lower, and
    /// dispatching past it does not run at all, which a tuner reads as infinitely
    /// fast — an early Q4L tuner picked 1024 for every shape because o_proj measured
    /// 0.000 ms on an invalid dispatch. Clamp candidates to this, never to max_tg.
    pub(crate) fn pipe_max(&self, kern: &str) -> u32 {
        self.p.get(kern)
            .map(|p| p.max_total_threads_per_threadgroup() as u32)
            .unwrap_or(64)
    }

    /// Time a Q4L kernel across the actual per-layer weights (one dispatch per layer,
    /// so nothing stays L2-resident) at a given threadgroup size. `bind` binds
    /// the per-layer buffers for layer `i`.
    pub(crate) fn time_q4l(&self, kern: &str, grid: u64, threads: u64, reps: usize,
                           bind: &dyn Fn(&metal::ComputeCommandEncoderRef, usize)) -> f64 {
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.p[kern]);
        for i in 0..reps {
            bind(&enc, i);
            enc.dispatch_thread_groups(MTLSize::new(grid, 1, 1), MTLSize::new(threads, 1, 1));
        }
        enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "autotune probe");
        let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        (ge - gs) * 1e3
    }

    /// Threadgroup size for gemv_q4l on one (K,N). A single hardcoded 256 for every
    /// shape measured flat on an M2 Max (7.365/7.384/7.349 ms for 64/128/256), but the
    /// rows-per-threadgroup this sets decides whether a small-N matvec can fill the
    /// GPU, so it is tuned per device rather than fixed.
    pub(crate) fn tune_q4l_gemv(&self, k: u32, n: u32, names: &[String]) -> u32 {
        let cap = self.pipe_max("gemv_q4l").min(self.tune.max_tg as u32);
        let cands: Vec<u32> = [64u32, 128, 256, 512, 1024]
            .into_iter().filter(|&t| t <= cap).collect();
        if cands.len() < 2 { return q4l_tg().min(cap.max(32)); }
        self.pick_best_pref(&cands, q4l_tg().min(cap), 0.02, |t| {
            let rows = t / 32 * 4;
            let grid = ((n + rows - 1) / rows) as u64;
            self.time_q4l("gemv_q4l", grid, t as u64, names.len().min(36), &|enc, i| {
                let nm = &names[i];
                enc.set_buffer(0, Some(&self.st.act), 0);
                enc.set_buffer(1, Some(&self.wt.w4l[nm]), 0);
                enc.set_buffer(2, Some(&self.st.tmp), 0);
                enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(5, Some(&self.wt.q4l_a[nm]), 0);
                enc.set_buffer(6, Some(&self.wt.q4l_b[nm]), 0);
            })
        })
    }

    /// Threadgroup size for the fused ffn_gu_q4l (gate+up SwiGLU), the largest
    /// single category in Q4L decode.
    pub(crate) fn tune_q4l_ffn_gu(&self, gates: &[String], ups: &[String]) -> u32 {
        let (k, n) = (self.d as u32, self.arch.ffn as u32);
        let act = self.arch.gelu as u32;
        let cap = self.pipe_max("ffn_gu_q4l").min(self.tune.max_tg as u32);
        let cands: Vec<u32> = [64u32, 128, 256, 512, 1024]
            .into_iter().filter(|&t| t <= cap).collect();
        if cands.len() < 2 { return q4l_tg().min(cap.max(32)); }
        let reps = gates.len().min(ups.len()).min(36);
        if reps == 0 { return q4l_tg(); }
        self.pick_best_pref(&cands, q4l_tg().min(cap), 0.02, |t| {
            let rows = t / 32 * 4;
            let grid = ((n + rows - 1) / rows) as u64;
            self.time_q4l("ffn_gu_q4l", grid, t as u64, reps, &|enc, i| {
                let (g, u) = (&gates[i], &ups[i]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w4l[g]), 0);
                enc.set_buffer(2, Some(&self.wt.w4l[u]), 0);
                enc.set_buffer(3, Some(&self.st.act), 0);
                enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &n as *const u32 as *const c_void);
                enc.set_buffer(6, Some(&self.wt.q4l_a[g]), 0);
                enc.set_buffer(7, Some(&self.wt.q4l_b[g]), 0);
                enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                enc.set_buffer(9, Some(&self.wt.q4l_a[u]), 0);
                enc.set_buffer(10, Some(&self.wt.q4l_b[u]), 0);
            })
        })
    }

    /// Pick lanes-per-row for the lane-partitioned verify matvec on one (K,N).
    /// Candidates: 0 = the row-blocked gemv_m4_q4l, else gemv_x<nx>_4_q4l. M=4 is
    /// the drafter's operating point, so that is what is timed.
    pub(crate) fn tune_xv(&self, k: u32, n: u32, names: &[String]) -> u32 {
        let m = 4u32;
        let mut cands: Vec<u32> = vec![0];
        for nx in [4u32, 8, 16] {
            if k % 32 == 0 && self.p.contains_key(&format!("gemv_x{nx}_4_q4l")) { cands.push(nx); }
        }
        if cands.len() == 1 { return 0; }
        let t = cands.iter().map(|&nx| {
            let kn = if nx == 0 { "gemv_m4_q4l".to_string() } else { format!("gemv_x{nx}_4_q4l") };
            self.pipe_max(&kn)
        }).min().unwrap_or(64).min(q4l_tg());
        // Cycle real layer weights, one per rep. Re-dispatching the same matrix leaves
        // it resident in L2, which makes every candidate look bandwidth-free and picks
        // the wrong one — the trap already recorded on tune_gemv_q4. Without cycling a
        // first pass disagreed with the profiler on 2 of 4 shapes; with it all 4 agree.
        self.pick_best_pref(&cands, 0u32, 0.02, |nx| {
            let (kern, rows) = if nx == 0 {
                ("gemv_m4_q4l".to_string(), t / 64 * 4)
            } else {
                (format!("gemv_x{nx}_4_q4l"), t / nx)
            };
            let grid = ((n + rows - 1) / rows) as u64;
            let cb = self.gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.p[&kern]);
            enc.set_buffer(0, Some(&self.st.act), 0);  // x (garbage ok; timing only)
            enc.set_buffer(2, Some(&self.st.tmp), 0);
            enc.set_bytes(3, 4, &k as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.set_bytes(7, 4, &m as *const u32 as *const c_void);
            let ac = 0u32;
            enc.set_bytes(8, 4, &ac as *const u32 as *const c_void);
            for i in 0..names.len().min(36) {
                let nm = &names[i];
                enc.set_buffer(1, Some(&self.wt.w4l[nm]), 0);
                enc.set_buffer(5, Some(&self.wt.q4l_a[nm]), 0);
                enc.set_buffer(6, Some(&self.wt.q4l_b[nm]), 0);
                enc.dispatch_thread_groups(MTLSize::new(grid, 1, 1), MTLSize::new(t as u64, 1, 1));
            }
            enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "autotune probe");
            let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
            (ge - gs) * 1e3
        })
    }

    /// Tune attention thread count (one threadgroup per head, threads split scores/V).
    pub(crate) fn tune_attn(&self) -> GemvPlan {
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        let seq = 64u32; // representative decode context
        // parallel-hd attention benefits from more simdgroups (positions) — try up to 1024.
        let cap = self.pipe_max("attention_short") as u64;
        let cands: Vec<u64> = [64u64, 128, 256, 512, 1024].into_iter()
            .filter(|&t| t <= self.tune.max_tg.min(cap)).collect();
        if cands.is_empty() { return GemvPlan { ksplit: false, threads: 64 }; }
        let threads = self.pick_best(&cands, |t| {
            self.time_kernel("attention_short", self.arch.n_head as u64, t, 60, |enc| {
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.kcache[0]), 0);
                enc.set_buffer(2, Some(&self.st.vcache[0]), 0);
                enc.set_buffer(3, Some(&self.st.attn), 0);
                enc.set_bytes(4, 4, &hd as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &kvdim as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &seq as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &group as *const u32 as *const c_void);
                enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
            })
        });
        GemvPlan { ksplit: false, threads }
    }
}

/// Reject candidates whose time is implausibly small. A dispatch that exceeds the
/// pipeline's thread limit does not execute and times at ~0, which a minimum-picker
/// reads as infinitely fast: an early Q4L tuner selected 1024 threads for every shape
/// and produced a kernel that wrote nothing (o_proj measured 0.000 ms while the decode
/// floor appeared to drop from 7.29 to 5.50 ms). `pipe_max()` prevents that at the
/// source; this is the backstop. A genuine candidate is never 10x faster than the
/// median of the field.
fn plausible(times: &[f64]) -> Vec<bool> {
    let mut sorted: Vec<f64> = times.iter().copied().filter(|t| t.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() { return vec![true; times.len()]; }
    let med = sorted[sorted.len() / 2];
    times.iter().map(|&t| t.is_finite() && t > 1e-4 && t >= med / 10.0).collect()
}

#[cfg(test)]
mod tests {
    use super::plausible;

    #[test]
    fn rejects_a_dispatch_that_never_ran() {
        // one candidate exceeded the pipeline's thread limit, did not execute,
        // and timed at zero.
        let t = [0.0, 1.70, 1.72, 1.75];
        assert_eq!(plausible(&t), vec![false, true, true, true]);
    }

    #[test]
    fn keeps_a_genuine_spread() {
        // 2.3x between best and worst is ordinary for a threadgroup sweep and must
        // survive: the guard is for zeros, not for real wins.
        let t = [1.00, 1.45, 2.30, 1.10];
        assert_eq!(plausible(&t), vec![true; 4]);
    }

    #[test]
    fn all_zero_field_is_not_silently_accepted() {
        // if every candidate failed to run, none is plausible and the caller
        // keeps its default rather than picking arbitrarily.
        let t = [0.0, 0.0, 0.0];
        assert_eq!(plausible(&t), vec![false, false, false]);
    }
}

#![allow(clippy::too_many_arguments)]
use super::*;
use objc::{msg_send, sel, sel_impl};
use metal::MTLSize;
use std::ffi::c_void;
 // re-export

/// Wall-time buckets for the streamed Flash graph. GPU time is a
/// subset of submit/wait time, not an additional additive wall-time bucket.
#[derive(Clone, Copy, Debug, Default)]
pub struct FlashTargetTiming {
    /// Diagnostic only: separate route/expert/tail command buffers. Changes scheduling.
    pub split_stages: bool,
    pub route_gpu_s: f64,
    pub expert_gpu_s: f64,
    pub tail_gpu_s: f64,
    pub wall_s: f64,
    pub gather_s: f64,
    pub gather_setup_s: f64,
    pub gather_copy_s: f64,
    pub direct_expert_bytes: u64,
    pub gather_read_s: f64,
    pub gather_admit_s: f64,
    pub encode_s: f64,
    pub submit_wait_s: f64,
    pub gpu_s: f64,
    pub commands: u64,
    pub invalid_gpu_timestamps: u64,
}

/// Nested wall-time events. Scalar includes catch-up; MTP includes draft,
/// target verification, catch-up and rollback. GPU time is inside wall time.
#[derive(Debug)]
pub struct FlashTraceEvent {
    pub name: &'static str,
    pub position: usize,
    pub rows: usize,
    pub start_s: f64,
    pub seconds: f64,
    pub hits: u64,
    pub lookups: u64,
    pub cache_bytes: usize,
    pub target: FlashTargetTiming,
}
pub(crate) struct FlashTrace {
    origin: std::time::Instant,
    events: Vec<FlashTraceEvent>,
}
pub(crate) type FlashTraceStart = (std::time::Instant, u64, u64);

impl<'a> DecoderGpu<'a> {
    /// Diagnostic A/B: retain the same Metal cache while toggling hit copies.
    /// Call only between completed requests, then reset/prefill before measuring.
    pub fn flash_direct_probe_mode(&mut self, enabled: bool) {
        assert!(self.arch.qwen4exp.is_some());
        assert!(self.strm.expert_cache.borrow().metal_device.is_some(),
            "load with OJAS_FLASH_DIRECT_EXPERTS=1 for a shared-cache comparison");
        self.cfg.flash_direct_experts = enabled;
    }

    pub fn flash_trace_begin(&self) {
        assert!(self.arch.qwen4exp.is_some());
        assert!(!self.cfg.glm_dbg, "Flash tracing requires stable gather counters; unset GLM debug");
        *self.flash_trace.borrow_mut() = Some(FlashTrace { origin: std::time::Instant::now(), events: Vec::with_capacity(4096) });
    }
    pub fn flash_trace_end(&self) -> Vec<FlashTraceEvent> {
        self.flash_trace.borrow_mut().take().expect("trace was not started").events
    }
    pub(crate) fn flash_trace_start(&self) -> Option<FlashTraceStart> {
        if self.flash_trace.borrow().is_some() {
            Some((std::time::Instant::now(), self.strm.gather_hits.get(), self.strm.gather_reads.get()))
        } else { None }
    }
    pub(crate) fn flash_trace_finish(&self, start: Option<FlashTraceStart>, name: &'static str,
        position: usize, rows: usize, target: FlashTargetTiming) {
        if let Some((time, hits, lookups)) = start {
            let seconds = time.elapsed().as_secs_f64();
            if let Some(trace) = self.flash_trace.borrow_mut().as_mut() {
                trace.events.push(FlashTraceEvent { name, position, rows,
                    start_s: time.duration_since(trace.origin).as_secs_f64(), seconds,
                    hits: self.strm.gather_hits.get() - hits,
                    lookups: self.strm.gather_reads.get() - lookups,
                    cache_bytes: self.strm.expert_cache.borrow().bytes(), target });
            }
        }
    }

    /// Diagnostic policy switch for paired requests on one loaded Flash model.
    /// Load with OJAS_MTP_PREFIX=1 to reserve the per-row snapshot capacity.
    pub fn flash_prefix_probe_mode(&mut self, enabled: bool) {
        assert!(self.arch.qwen4exp.is_some() && (!enabled || self.sp.snapshot_rows > 1));
        self.cfg.mtp_prefix = enabled;
        self.sp.verified_rows.set(0);
    }

    pub fn mtp_snapshot_bytes(&self) -> u64 {
        self.sp.conv_snap.iter().chain(&self.sp.ssm_snap).map(|b| b.length()).sum()
    }
    /// Profile real target verification without adding dispatches or barriers.
    /// Draft catch-up preserves normal state but is outside the target timers.
    /// An optional copy-thread override allows interleaved A/B requests on one
    /// loaded model; the load-time setting is restored before returning.
    pub fn flash_target_profile(&mut self, tokens: &[u32], pos: usize, copy_threads: Option<usize>) -> (Vec<u32>, FlashTargetTiming) {
        self.flash_target_profile_impl(tokens, pos, copy_threads, false)
    }

    /// Diagnostic switch for paired GPU kernel comparisons on one loaded model.
    pub fn flash_q8_probe_mode(&mut self, enabled: bool) {
        assert!(self.arch.qwen4exp.is_some());
        self.cfg.flash_q8_cooperative = enabled;
    }

    /// Separates route, expert and tail GPU timing by adding command boundaries.
    /// Use normal target profiling for final performance comparisons.
    pub fn flash_stage_profile(&mut self, tokens: &[u32], pos: usize) -> (Vec<u32>, FlashTargetTiming) {
        self.flash_target_profile_impl(tokens, pos, None, true)
    }

    fn flash_target_profile_impl(&mut self, tokens: &[u32], pos: usize, copy_threads: Option<usize>, split_stages: bool) -> (Vec<u32>, FlashTargetTiming) {
        assert!(self.arch.qwen4exp.is_some() && self.strm.stream && self.has_mtp());
        assert!(!tokens.is_empty() && tokens.len() <= MAXM);
        let saved = self.cfg.expert_copy_threads;
        if let Some(n) = copy_threads {
            assert!((1..=8).contains(&n));
            self.cfg.expert_copy_threads = n;
        }
        let stats = std::cell::RefCell::new(FlashTargetTiming { split_stages, ..Default::default() });
        let start = std::time::Instant::now();
        self.forward_chunk_qwen4exp_timed(tokens, pos, true, Some(&stats));
        stats.borrow_mut().wall_s = start.elapsed().as_secs_f64();
        self.cfg.expert_copy_threads = saved;
        self.qwen4exp_mtp_catchup(tokens, pos);
        let out = unsafe { std::slice::from_raw_parts(self.st.tmp.contents() as *const u32, tokens.len()) }.to_vec();
        (out, stats.into_inner())
    }

    /// Copy verification logits before any subsequent model call overwrites scratch.
    /// Intended for independent release checks; requires drafting disabled.
    pub fn flash_target_logits(&self, rows: usize) -> Vec<f32> {
        assert!(self.cfg.no_spec && self.arch.qwen4exp.is_some() && (1..=MAXM).contains(&rows));
        assert!((rows * self.arch.vocab * 4) as u64 <= self.st.logits.length());
        unsafe { std::slice::from_raw_parts(self.st.logits.contents().cast::<f32>(), rows * self.arch.vocab) }.to_vec()
    }

    /// Independent FP64 check of the Flash NextN combiner using the loaded Q8_0
    /// projection and real committed hidden state. Diagnostic only; no KV writes.
    /// Returns (RMSE, max absolute error, RMSE if all projected streams were pooled).
    pub fn flash_mtp_projection_reference(&self, token: u32) -> (f64, f64, f64) {
        let mc = self.sp.mtp.expect("draft weights required");
        let hc = self.arch.qwen4exp.as_ref().unwrap().hc_mult as usize;
        let d = self.d;
        assert_eq!(mc.hnorm_len, d * hc);
        let name = |s: &str| format!("blk.{}.{}", mc.layer, s);
        let proj = name("nextn.eh_proj.weight");
        assert_eq!(self.wt.w_qtype[&proj], 8, "reference probe requires native Q8_0");
        let offset = self.sp.hrow.get() * d * hc * 4;
        let cb = self.gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.qwen4exp_mtp_project(enc, &[token], &self.sp.mtp_h, offset as u64);
        enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "profiling probe");
        let view = |b: &metal::Buffer, off: usize, n: usize| unsafe {
            std::slice::from_raw_parts(b.contents().cast::<f32>().add(off), n)
        };
        let emb = view(&self.st.x, 0, d);
        let hidden = view(&self.sp.mtp_h, offset / 4, d * hc);
        let en = view(&self.wt.w32[&name("nextn.enorm.weight")], 0, d);
        let hn = view(&self.wt.w32[&name("nextn.hnorm.weight")], 0, d * hc);
        let actual = view(&self.st.hc_res, 0, d * hc);
        let rms = |x: &[f32]| (x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / d as f64 + f64::from(self.arch.eps)).sqrt();
        let er = rms(emb);
        let mut cat = vec![0.0f64; hc * 2 * d];
        for c in 0..hc {
            let hr = rms(&hidden[c*d..(c+1)*d]);
            for j in 0..d {
                cat[c*2*d+j] = f64::from(emb[j]) / er * f64::from(en[j]);
                cat[c*2*d+d+j] = f64::from(hidden[c*d+j]) / hr * f64::from(hn[c*d+j]);
            }
        }
        let wb = &self.wt.wq[&proj];
        let wo = self.wt.w_off.get(&proj).copied().unwrap_or(0) as usize;
        let bytes = unsafe { std::slice::from_raw_parts(wb.contents().cast::<u8>().add(wo), d * (2*d/32) * 34) };
        let mut expected = vec![0.0f64; d * hc];
        for out in 0..d {
            for block in 0..2*d/32 {
                let start = (out*(2*d/32)+block)*34;
                let scale = half::f16::from_bits(u16::from_le_bytes([bytes[start], bytes[start+1]])).to_f64();
                for j in 0..32 {
                    let w = scale * f64::from(bytes[start+2+j] as i8);
                    for c in 0..hc { expected[c*d+out] += w * cat[c*2*d+block*32+j]; }
                }
            }
        }
        let mut sq = 0.0f64;
        let mut max = 0.0f64;
        let mut pooled = 0.0f64;
        for j in 0..d {
            let mean = (0..hc).map(|c| expected[c*d+j]).sum::<f64>() / hc as f64;
            for c in 0..hc {
                let err = f64::from(actual[c*d+j]) - expected[c*d+j];
                sq += err*err; max = max.max(err.abs());
                pooled += (mean - expected[c*d+j]).powi(2);
            }
        }
        ((sq/(d*hc) as f64).sqrt(), max, (pooled/(d*hc) as f64).sqrt())
    }

    /// Draft from the current committed target hidden row (including batched prefill).
    pub fn flash_draft_current(&self, token: u32, pos: usize) -> u32 {
        self.flash_draft_probe(token, pos, self.sp.hrow.get(), false)
    }

    /// Diagnostic single Flash draft step, including optional self-hidden chaining.
    /// Mutates draft scratch/KV state just like the production drafter.
    pub fn flash_draft_probe(&self, token: u32, pos: usize, hrow: usize, chain: bool) -> u32 {
        assert!(self.arch.qwen4exp.is_some() && self.has_mtp());
        assert!(pos < self.st.max_seq && hrow < MAXM && (token as usize) < self.arch.vocab);
        self.mtp_draft_qwen4exp(token, pos, hrow, true, chain)
    }

    /// The normal Flash verify sequence with target and draft-catch-up wall timers.
    /// Returns the same rows/state as `mtp_verify_n`; not a production fast path.
    pub fn flash_verify_probe(&self, tokens: &[u32], pos: usize) -> (Vec<u32>, f64, f64) {
        assert!(self.arch.qwen4exp.is_some() && self.has_mtp());
        assert!(!tokens.is_empty() && tokens.len() <= MAXM);
        let start = std::time::Instant::now();
        self.forward_chunk_qwen4exp(tokens, pos, true);
        let target_s = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        self.qwen4exp_mtp_catchup(tokens, pos);
        let catchup_s = start.elapsed().as_secs_f64();
        let out = unsafe { std::slice::from_raw_parts(self.st.tmp.contents() as *const u32, tokens.len()) }.to_vec();
        (out, target_s, catchup_s)
    }

    /// Read `npos` positions of a layer's K and V cache as f32 (they are stored
    /// f16). Debug accessor for comparing the two writers, rope_qk_store (per
    /// token) and rope_qk_store_m (per chunk).
    pub fn dump_kv(&self, layer: usize, pos0: usize, npos: usize) -> (Vec<f32>, Vec<f32>) {
        let kvdim = self.arch.n_kv * self.arch.hd;
        let rd = |b: &metal::Buffer| -> Vec<f32> {
            let p = b.contents() as *const half::f16;
            (0..npos * kvdim)
                .map(|i| unsafe { *p.add(pos0 * kvdim + i) }.to_f32())
                .collect()
        };
        (rd(&self.st.kcache[layer]), rd(&self.st.vcache[layer]))
    }

    /// Total accumulated GPU execution time (seconds) across all forwards.
    pub fn gpu_seconds(&self) -> f64 {
        self.gpu_s.get()
    }

    /// L2 norm of a NextN combiner input row. Zero means the verify tail never
    /// wrote it, which leaves the combiner with only its embedding half.
    pub fn mtp_h_norm(&self, row: usize) -> f32 {
        let d = self.d;
        let v = unsafe { std::slice::from_raw_parts(self.sp.mtp_h.contents() as *const f32, 2 * d) };
        v[row * d..(row + 1) * d].iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Expert-gather counters since load: (hits, reads, bytes resident in the LRU).
    /// The hit rate dominates a streamed model's throughput.
    pub fn gather_stats(&self) -> (u64, u64, usize) {
        (self.strm.gather_hits.get(), self.strm.gather_reads.get(),
         self.strm.expert_cache.borrow().bytes())
    }

    /// Reset the gather counters so a phase can be measured on its own.
    pub fn gather_stats_reset(&self) {
        self.strm.gather_hits.set(0);
        self.strm.gather_reads.set(0);
    }

    /// Per-category GPU-time profiler. Dispatches each kernel type as many times
    /// as it runs per token (all layers) in one command buffer and times it,
    /// isolating each category's cost. The sum is the no-gap compute floor; the
    /// difference from the real per-token GPU time is inter-kernel idle gaps.
    pub fn profile(&self, pos: usize) {
        let d = self.d as u32;
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        let nh = self.arch.n_head as u32;
        let pos_u = pos as u32;
        let off = pos as u32 * kvdim;
        let totq = nh * hd / 2;
        let totk = self.arch.n_kv as u32 * hd / 2;
        let nl = self.arch.n_layers;
        let pl = |l: usize, s: &str| format!("blk.{}.{}", l % nl, s); // cycle real layers → uncached weights
        // Hybrid archs: on qwen35 only every 4th layer is attention; the rest are
        // Gated-DeltaNet blocks with no attn_* tensors. Cycling `l % nl` through
        // every layer for an attention category asks for a weight that does not
        // exist, so attention categories cycle the attention layers and SSM
        // categories the SSM ones, each reporting its own layer count.
        let attn_ls: Vec<usize> = (0..nl).filter(|&l| !self.arch.layers[l].is_ssm).collect();
        let ssm_ls: Vec<usize> = (0..nl).filter(|&l| self.arch.layers[l].is_ssm).collect();
        let n_attn = attn_ls.len().max(1);
        let al = |i: usize, s: &str| format!("blk.{}.{}", attn_ls[i % attn_ls.len()], s);
        if !ssm_ls.is_empty() {
            tracing::debug!(target: "profile", "hybrid: {} attention layers, {} SSM layers", attn_ls.len(), ssm_ls.len());
        }

        // Times a closure that encodes `reps` dispatches (rep i uses layer i%nl so
        // weights aren't L2-cached across reps → real device bandwidth).
        // `wbytes` is what one pass over `wname` must read, from the tensor's real
        // shape and representation, so a category can be reported in GB/s rather
        // than milliseconds.
        let wbytes = |wname: &str| -> f64 {
            let Some(&(k, n)) = self.wt.wshape.get(wname) else { return 0.0 };
            let elems = k as f64 * n as f64;
            let rows = n as f64;
            match self.wt.repr(wname) {
                Repr::F32 => elems * 4.0,
                Repr::F16 => elems * 2.0,
                // Q8: one byte per weight plus one f32 scale per row.
                Repr::Q8 => elems + rows * 4.0,
                // Q4 (symmetric, per-32 block): nibbles + one f16 scale per block.
                Repr::Q4 => elems * (0.5 + 2.0 / 32.0),
                // Q4L: Q4_K's values, nibbles contiguous + two f16 side arrays per block.
                Repr::Q4L => elems * (0.5 + 4.0 / 32.0),
                Repr::Q4K => elems * (144.0 / 256.0),
                Repr::Q6K => elems * (210.0 / 256.0),
                // Native: the GGUF block table is authoritative.
                Repr::Native => match self.wt.w_qtype.get(wname)
                    .and_then(|&t| ojas_metal::kernels::nat::format_of(t)) {
                    Some(f) => elems * f.block_bytes as f64 / f.weights as f64,
                    None => 0.0,
                },
                _ => 0.0,
            }
        };
        // The encoder type must match the real path: `forward_span` opens a
        // concurrent encoder on this arch (span.rs), so independent dispatches
        // overlap there. A serial encoder measures a serialization the decoder does
        // not pay, and biases categories unequally — roughly doubling the reading
        // for the small independent pairs (ssm_alpha/ssm_beta) while barely
        // affecting a category that already saturates the machine.
        let conc = self.arch.ssm.is_some() && !self.cfg.serial;
        fn mk_enc(cb: &metal::CommandBufferRef, conc: bool) -> &metal::ComputeCommandEncoderRef {
            if conc { cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent) }
            else { cb.new_compute_command_encoder() }
        }
        // Each category is timed twice and both numbers are reported:
        //
        //   iso  — every rep back to back in a serial encoder, so nothing overlaps:
        //          what one dispatch of this kernel costs standing alone.
        //   ovl  — the same reps in the concurrent encoder the decoder opens, with
        //          no barriers between them, so they overlap freely.
        //
        // The spread is the headroom a lone dispatch leaves unused, and for the
        // 4-bit matvecs it is large: a Q4 ffn_gu moves 3.6x fewer bytes than the f16
        // one for the same launch, so a single dispatch never reaches steady state
        // and `iso` reads far below peak while `ovl` reaches the same GB/s f16 does.
        // The real graph sits between, at a point set by how many independent
        // dispatches its barriers leave in flight.
        let time_cat_b = |name: &str, reps: usize, bytes: f64,
                          f: &dyn Fn(&metal::ComputeCommandEncoderRef, usize)| {
            let run = |c: bool| -> f64 {
                for _ in 0..2 {
                    let cb = self.gpu.command_buffer();
                    let enc = mk_enc(cb, c);
                    for i in 0..reps { f(&enc, i); }
                    enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "profiling probe");
                }
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let cb = self.gpu.command_buffer();
                    let enc = mk_enc(cb, c);
                    for i in 0..reps { f(&enc, i); }
                    enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "profiling probe");
                    let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
                    best = best.min((ge - gs) * 1e3);
                }
                best
            };
            let iso = run(false);
            let ovl = if conc { run(true) } else { iso };
            let bw = |ms: f64| if bytes > 0.0 { format!("{:5.0}", bytes / 1e6 / ms) } else { "    -".into() };
            tracing::debug!(target: "ojas",
                "  {name:<12} {reps:>3}d {:7.2} MB | iso {iso:7.3} ms {} GB/s | ovl {ovl:7.3} ms {} GB/s | x{:.2}",
                bytes / 1e6, bw(iso), bw(ovl), iso / ovl.max(1e-9));
            iso
        };
        let time_cat = |name: &str, reps: usize, f: &dyn Fn(&metal::ComputeCommandEncoderRef, usize)| {
            time_cat_b(name, reps, 0.0, f)
        };

        // pipeline-switch cost probe: N trivial dispatches, no-switch vs switch-every.
        {
            let n = 960usize; // ~ dispatches in 5 tokens' worth
            let probe = |switch: bool| -> f64 {
                let mut best = f64::INFINITY;
                for _ in 0..7 {
                    let cb = self.gpu.command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    for i in 0..n {
                        let pn = if switch { if i & 1 == 0 { "probe0" } else { "probe1" } } else { "probe0" };
                        enc.set_compute_pipeline_state(&self.p[pn]);
                        enc.set_buffer(0, Some(&self.st.tmp), 0);
                        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                    }
                    enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "profiling probe");
                    let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
                    best = best.min((ge - gs) * 1e3);
                }
                best
            };
            let same = probe(false);
            let sw = probe(true);
            tracing::debug!(target: "probe", "{n} trivial dispatches: no-switch {same:.3} ms ({:.3} us/disp), switch-every {sw:.3} ms ({:.3} us/disp)",
                same / n as f64 * 1e3, sw / n as f64 * 1e3);
            tracing::debug!(target: "probe", "=> pipeline-switch overhead ≈ {:.3} us/switch", (sw - same) / n as f64 * 1e3);
        }
        tracing::debug!(target: "profile", "per-category GPU time (one token's worth, real per-layer weights):");
        let mut sum = 0.0;
        // GEMV time and GEMV bytes, accumulated separately from `sum` so the report
        // can state bandwidth over the time that actually reads weights.
        let mut gemv_ms = 0.0f64;
        let mut b_ssm = 0.0f64;
        sum += time_cat("rmsnorm", 2 * nl + 1, &|enc, i| {
            self.enc_reduce(enc, "rmsnorm", &[(&self.st.x, 0), (&self.wt.w32[&pl(i, "attn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], 1, 256);
        });
        // Mirror graph_decode's selection. Calling the q8 variant unconditionally
        // panics at prec=2/3, where these weights live in w4/w4k instead.
        let b_qkv: f64 = attn_ls.iter().map(|&l| ["attn_q.weight","attn_k.weight","attn_v.weight"]
    .iter().map(|t| wbytes(&format!("blk.{l}.{t}"))).sum::<f64>()).sum();
        gemv_ms += { let t = time_cat_b("qkv", n_attn, b_qkv, &|enc, i| {
            let p = |s: &str| al(i, s);
            // Must mirror graph_decode's `qkv_sep`, including the None case, or a
            // variant whose weights live in another map panics with "no entry found
            // for key". `fused_repr` returns None when q/k/v are different types,
            // which mixed-type (UD-style) quants ship as a matter of course.
            let qkv_repr = self.wt.fused_repr(
                &[&p("attn_q.weight"), &p("attn_k.weight"), &p("attn_v.weight")]);
            let qkv_sep = match qkv_repr {
                None | Some(Repr::Native) => true,
                Some(Repr::Q4) | Some(Repr::Q20) => !self.arch.qkv_bias,
                _ => false,
            };
            if qkv_sep {
                // Three separate matvecs — `mm()` resolves each weight
                // independently, so they serve any mix.
                self.mm(enc, "plain", &p("attn_q.weight"), &self.st.h, &self.st.q, d, d, None);
                self.mm(enc, "plain", &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim, None);
                self.mm(enc, "plain", &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim, None);
            } else if qkv_repr == Some(Repr::Q4L) {
                self.qkv_q4l(enc, &p, &self.st.h, d, d, kvdim);
            } else if self.wt.q4 {
                self.qkv_q4(enc, &p, &self.st.h, d, d, kvdim);
            } else if self.wt.q8 {
                self.qkv(enc, &p, &self.st.h, d, d, kvdim);
            } else {
                self.qkv_f16(enc, &p, &self.st.h, d, d, kvdim);
            }
        }); sum += t; t };
        sum += time_cat("rope+store", n_attn, &|enc, i| {
            self.enc_reduce(enc, "rope_qk_store",
                &[(&self.st.q, 0), (&self.st.k, 1), (&self.st.v, 2), (&self.st.kcache[attn_ls[i % attn_ls.len()]], 3), (&self.st.vcache[attn_ls[i % attn_ls.len()]], 4)],
                &[(5, hd), (6, pos_u), (8, totq), (9, totk), (10, kvdim), (11, off), (12, self.arch.rope_neox as u32)], &[(7, self.arch.rope_base)],
                (((totq + totk + kvdim) + 63) / 64) as u64, 64);
        });
        // Match the real path: threads come from autotune (capped at 256 for the
        // streaming kernel), not a hardcoded 64. Profiling with 64 understates by 4x.
        sum += time_cat("attention", n_attn, &|enc, i| {
            // Mirror graph_decode: KV split across workgroups + merge.
            let seq = (pos + 1) as u32;
            let nh = self.arch.n_head as u32;
            let nwg = if seq < 32 { 0 } else {
                let base = (64 / nh.max(1)).max(seq / 1024);
                base.clamp(2, ojas_metal::kernels::attn::ATTN_NWG as u32).min((seq / 16).max(2))
            };
            if nwg == 0 {
                self.enc_reduce(enc, "attention_short",
                    &[(&self.st.q, 0), (&self.st.kcache[attn_ls[i % attn_ls.len()]], 1), (&self.st.vcache[attn_ls[i % attn_ls.len()]], 2), (&self.st.attn, 3)],
                    &[(4, hd), (5, kvdim), (6, seq), (7, group)], &[(8, scale)], nh as u64, 256);
            } else {
                enc.set_compute_pipeline_state(&self.p["attention_part"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.kcache[attn_ls[i % attn_ls.len()]]), 0);
                enc.set_buffer(2, Some(&self.st.vcache[attn_ls[i % attn_ls.len()]]), 0);
                enc.set_buffer(3, Some(&self.st.attn_part), 0);
                for (bi, v) in [(4u32, hd), (5, kvdim), (6, seq), (7, group), (9, nwg)] {
                    enc.set_bytes(bi as u64, 4, &v as *const u32 as *const c_void);
                }
                enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, nwg as u64, 1), MTLSize::new(256, 1, 1));
                enc.set_compute_pipeline_state(&self.p["attention_merge"]);
                enc.set_buffer(0, Some(&self.st.attn_part), 0);
                enc.set_buffer(1, Some(&self.st.attn), 0);
                enc.set_bytes(2, 4, &hd as *const u32 as *const c_void);
                enc.set_bytes(3, 4, &nwg as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
        });
        // K is qdim (n_head*head_dim), not d. They coincide on qwen2 and diverge on
        // qwen3/qwen35, where passing d makes o_proj consume half the attention
        // output.
        let b_o: f64 = attn_ls.iter().map(|&l| wbytes(&format!("blk.{l}.attn_output.weight"))).sum();
        gemv_ms += { let t = time_cat_b("o_proj", n_attn, b_o, &|enc, i| {
            let l = attn_ls[i % attn_ls.len()];
            self.mm(enc, "accum", &al(i, "attn_output.weight"), &self.st.attn, &self.st.x,
                    self.arch.layers[l].qdim, d, None);
        }); sum += t; t };
        // SSM/Gated-DeltaNet layers carry their own projections: on this 27B,
        // 5.56 B parameters (attn_qkv 52 M + attn_gate 31 M + ssm_out 31 M per
        // layer x 48), about 30% of the model. Leaving them uncategorised shows up
        // as a ~17 ms gap between the category sum and measured decode.
        if !ssm_ls.is_empty() {
            let sl = |i: usize, t: &str| format!("blk.{}.{}", ssm_ls[i % ssm_ls.len()], t);
            b_ssm = ssm_ls.iter().map(|&l| ["attn_qkv.weight","attn_gate.weight","ssm_out.weight",
        "ssm_alpha.weight","ssm_beta.weight"].iter()
        .map(|t| wbytes(&format!("blk.{l}.{t}"))).sum::<f64>()).sum();
            // Split the five SSM projections by size. ssm_alpha/ssm_beta are
            // (K=1024, N=dt_rank=16) on this model: 0.02 Mw against attn_qkv's
            // 6.29, yet they are two full dispatches per SSM layer (36 per token),
            // and at 8 rows per threadgroup a whole matvec is two threadgroups.
            // Lumped in with the big projections their cost is invisible.
            let big = ["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"];
            let tiny = ["ssm_alpha.weight", "ssm_beta.weight"];
            let bof = |ts: &[&str]| -> f64 { ssm_ls.iter().map(|&l| ts.iter()
                .map(|t| wbytes(&format!("blk.{l}.{t}"))).sum::<f64>()).sum() };
            let (b_big, b_tiny) = (bof(&big), bof(&tiny));
            b_ssm = b_big + b_tiny;
            let run = |names: &'static [&'static str]| move |enc: &metal::ComputeCommandEncoderRef, i: usize| {
                for t in names {
                    let n = sl(i, t);
                    if let Some(&(k, nn)) = self.wt.wshape.get(&n) {
                        self.mm(enc, "plain", &n, &self.st.h, &self.st.act, k, nn, None);
                    }
                }
            };
            gemv_ms += { let t = time_cat_b("ssm_proj", ssm_ls.len(), b_big,
                &run(&["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"])); sum += t; t };
            gemv_ms += { let t = time_cat_b("ssm_ab_proj", ssm_ls.len(), b_tiny,
                &run(&["ssm_alpha.weight", "ssm_beta.weight"])); sum += t; t };
            // The SSM math: conv1d + ssm_ab + the delta-net recurrence + the gated
            // norm. Four dispatches per SSM layer, 72 per token; omitting them
            // leaves the mixer arithmetic out of the category sum.
            if let Some(sc) = self.arch.ssm {
                let (s_st, hk, hv) = (sc.d_state, sc.n_group, sc.dt_rank);
                let d_inner = sc.d_inner;
                let conv_ch = d_inner + 2 * hk * s_st;
                let conv_k = sc.conv_kernel;
                let head_v = d_inner / hv;
                let kmap_div: u32 = self.cfg.moe_kmap_div as u32;
                sum += time_cat("ssm_math", ssm_ls.len(), &|enc, i| {
                    let l = ssm_ls[i % ssm_ls.len()];
                    let p = |s: &str| format!("blk.{l}.{s}");
                    self.enc_reduce(enc, "ssm_ab", &[(&self.st.ssm_gate, 0), (&self.st.ssm_beta, 1),
                        (&self.wt.w32[&p("ssm_dt.bias")], 2), (&self.wt.w32[&p("ssm_a")], 3)],
                        &[(4, hv), (5, hv)], &[], ((hv + 63) / 64) as u64, 64);
                    enc.set_compute_pipeline_state(&self.p["conv1d_decode"]);
                    enc.set_buffer(0, Some(&self.st.ssm_qkv), 0);
                    enc.set_buffer(1, Some(&self.st.conv_state[l]), 0);
                    enc.set_buffer(2, Some(&self.wt.w32[&p("ssm_conv1d.weight")]), 0);
                    enc.set_bytes(3, 4, &conv_ch as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &conv_k as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(((conv_ch + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                    enc.set_compute_pipeline_state(&self.p["deltanet_fused"]);
                    enc.set_buffer(0, Some(&self.st.ssm_state[l]), 0);
                    enc.set_buffer(1, Some(&self.st.ssm_qkv), 0);
                    enc.set_buffer(2, Some(&self.st.ssm_gate), 0);
                    enc.set_buffer(3, Some(&self.st.ssm_beta), 0);
                    enc.set_buffer(4, Some(&self.st.ssm_o), 0);
                    for (bi, v) in [(5u32, s_st), (6, hk), (7, hv), (8, conv_ch), (9, 1)] {
                        enc.set_bytes(bi as u64, 4, &v as *const u32 as *const c_void);
                    }
                    enc.set_bytes(10, 4, &self.arch.eps as *const f32 as *const c_void);
                    enc.set_buffer(11, Some(&self.st.ssm_state[l]), 0);
                    enc.set_bytes(12, 4, &u32::MAX as *const u32 as *const c_void);
                    enc.set_bytes(13, 4, &kmap_div as *const u32 as *const c_void);
                    enc.set_bytes(14, 4, &0u32 as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new((s_st / 4) as u64, hv as u64, 1), MTLSize::new(128, 1, 1));
                    enc.set_compute_pipeline_state(&self.p["gated_rmsnorm"]);
                    enc.set_buffer(0, Some(&self.st.ssm_o), 0);
                    enc.set_buffer(1, Some(&self.wt.w32[&p("ssm_norm.weight")]), 0);
                    enc.set_buffer(2, Some(&self.st.ssm_z), 0);
                    enc.set_bytes(3, 4, &head_v as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
                    enc.set_bytes(5, 4, &d_inner as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &0u32 as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(hv as u64, 1, 1), MTLSize::new(32, 1, 1));
                });
            }
        }
        let b_gu: f64 = (0..nl).map(|l| wbytes(&format!("blk.{l}.ffn_gate.weight"))
        + wbytes(&format!("blk.{l}.ffn_up.weight"))).sum();
        gemv_ms += { let t = time_cat_b("ffn_gu", nl, b_gu, &|enc, i| {
            // Mirror the real dispatch (graph_decode): native K-quant pair when both
            // operands landed natively, Q8 otherwise. Indexing w8 unconditionally
            // panics at prec=3.
            let (gate_n, up_n) = (pl(i, "ffn_gate.weight"), pl(i, "ffn_up.weight"));
            let nffn = self.arch.ffn as u32;
            let act = self.arch.gelu as u32;
            // Phrased as graph_decode phrases it: negative against the set of reprs
            // a fused kernel can serve, not positive against Native. The positive
            // form misses None, which `fused_repr` returns when gate and up are
            // different types — the normal case in mixed-type (UD-style) quants.
            if !matches!(self.wt.fused_repr(&[&gate_n, &up_n]),
                Some(Repr::Q4L) | Some(Repr::Q4) | Some(Repr::Q8) | Some(Repr::Q20)
                | Some(Repr::F16) | Some(Repr::Q4K) | Some(Repr::Q6K)) {
                // No fused gate/up kernel can serve this pair, so mirror
                // graph_decode — two matvecs plus the split activation.
                self.mm(enc, "plain", &gate_n, &self.st.h, &self.st.gate, d, nffn, None);
                self.mm(enc, "plain", &up_n, &self.st.h, &self.st.up, d, nffn, None);
                enc.set_compute_pipeline_state(&self.p["ffn_gu_split"]);
                enc.set_buffer(0, Some(&self.st.gate), 0);
                enc.set_buffer(1, Some(&self.st.up), 0);
                enc.set_buffer(2, Some(&self.st.act), 0);
                enc.set_bytes(3, 4, &nffn as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &act as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((nffn + 255) / 256) as u64, 1, 1),
                                           MTLSize::new(256, 1, 1));
                return;   // not a pure if/else chain — a second one follows
            } else if self.wt.w4l.contains_key(&gate_n) && self.wt.w4l.contains_key(&up_n) {
                enc.set_compute_pipeline_state(&self.p["ffn_gu_q4l"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w4l[&gate_n]), 0);
                enc.set_buffer(2, Some(&self.wt.w4l[&up_n]), 0);
                enc.set_buffer(3, Some(&self.st.act), 0);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                enc.set_buffer(6, Some(&self.wt.q4l_a[&gate_n]), 0);
                enc.set_buffer(7, Some(&self.wt.q4l_b[&gate_n]), 0);
                enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
                enc.set_buffer(9, Some(&self.wt.q4l_a[&up_n]), 0);
                enc.set_buffer(10, Some(&self.wt.q4l_b[&up_n]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((nffn + 31) / 32) as u64, 1, 1), MTLSize::new(256, 1, 1));
                return;
            }
            let native = if self.wt.w4k.contains_key(&gate_n) && self.wt.w4k.contains_key(&up_n) {
                Some(("ffn_gu_q4k", &self.wt.w4k))
            } else if self.wt.w6k.contains_key(&gate_n) && self.wt.w6k.contains_key(&up_n) {
                Some(("ffn_gu_q6k", &self.wt.w6k))
            } else {
                None
            };
            if let Some((kernel, map)) = native {
                enc.set_compute_pipeline_state(&self.p[kernel]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&map[&gate_n]), self.wt.w_off.get(&gate_n).copied().unwrap_or(0));
                enc.set_buffer(2, Some(&map[&up_n]), self.wt.w_off.get(&up_n).copied().unwrap_or(0));
                enc.set_buffer(3, Some(&self.st.act), 0);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
            } else if self.wt.q4 {
                enc.set_compute_pipeline_state(&self.p["ffn_gu_q4"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w4[&gate_n]), 0);
                enc.set_buffer(2, Some(&self.wt.w4[&up_n]), 0);
                enc.set_buffer(3, Some(&self.st.act), 0);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                enc.set_buffer(6, Some(&self.wt.scale4[&gate_n]), 0);
                enc.set_buffer(7, Some(&self.wt.scale4[&up_n]), 0);
                enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
            } else if !self.wt.q8 && self.wt.w16.contains_key(&gate_n) {
                // F16 (prec=4 streams the GGUF's own f16 straight through). Without
                // this branch prec=4 falls through to the Q8 assert below and panics.
                enc.set_compute_pipeline_state(&self.p["ffn_gu_f16"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w16[&gate_n]), 0);
                enc.set_buffer(2, Some(&self.wt.w16[&up_n]), 0);
                enc.set_buffer(3, Some(&self.st.act), 0);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &act as *const u32 as *const c_void);
            } else {
                assert!(self.wt.w8.contains_key(&gate_n),
                    "ffn_gu profiled a path that cannot serve {gate_n} (wq={}, w4l={}, w16={}) — the \
                     dispatch chain here must mirror graph_decode or the profile is fiction",
                    self.wt.wq.contains_key(&gate_n), self.wt.w4l.contains_key(&gate_n),
                    self.wt.w16.contains_key(&gate_n));
                enc.set_compute_pipeline_state(&self.p["ffn_gu_q8"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w8[&gate_n]), 0);
                enc.set_buffer(2, Some(&self.wt.w8[&up_n]), 0);
                enc.set_buffer(3, Some(&self.st.act), 0);
                enc.set_bytes(4, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nffn as *const u32 as *const c_void);
                enc.set_buffer(6, Some(&self.wt.scale8[&gate_n]), 0);
                enc.set_buffer(7, Some(&self.wt.scale8[&up_n]), 0);
                enc.set_bytes(8, 4, &act as *const u32 as *const c_void);
            }
            // Launch geometry must mirror graph_decode's, not just the kernel name.
            // `ffn_gu_q4` derives its row from `ts` (out_row = tgid*(ts/32*4) +
            // sgid*4), so at 256 threads a threadgroup covers 32 rows; a grid sized
            // for 8 gives four times the threadgroups, three quarters of them
            // returning immediately.
            if native.is_some() {
                enc.dispatch_thread_groups(MTLSize::new(((nffn + 1) / 2) as u64, 1, 1), MTLSize::new(64, 1, 1));
            } else if self.wt.q4 {
                enc.dispatch_thread_groups(MTLSize::new((nffn / 8) as u64, 1, 1), MTLSize::new(64, 1, 1));
            } else {
                enc.dispatch_thread_groups(MTLSize::new(((nffn + 7) / 8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
        }); sum += t; t };
        let b_dn: f64 = (0..nl).map(|l| wbytes(&format!("blk.{l}.ffn_down.weight"))).sum();
        gemv_ms += { let t = time_cat_b("ffn_down", nl, b_dn, &|enc, i| { self.mm(enc, "accum", &pl(i, "ffn_down.weight"), &self.st.act, &self.st.x, self.arch.ffn as u32, d, None); }); sum += t; t };
        let b_lm: f64 = wbytes(&self.arch.lm_head);
        gemv_ms += { let t = time_cat_b("lm_head", 1, b_lm, &|enc, _| { self.mm(enc, "plain", &self.arch.lm_head, &self.st.h, &self.st.logits, d, self.arch.vocab as u32, None); }); sum += t; t };
        // Diagnostic, not part of the floor: the same lm_head matvec chopped into
        // 36 ffn_gu-sized dispatches. If the one-dispatch version's bandwidth
        // advantage over ffn_gu comes from dispatch granularity — short dispatches
        // paying drain and never reaching DRAM steady state — this collapses to
        // ffn_gu-class GB/s; if it holds ~389 GB/s, the difference is the kernel.
        if self.wt.q8 && self.wt.w8.contains_key("token_embd.weight") {
            let w = &self.wt.w8["token_embd.weight"];
            let sc = &self.wt.scale8["token_embd.weight"];
            let plan = self.tune.gemv_plan.get(&(d, self.arch.vocab as u32)).copied()
                .unwrap_or(GemvPlan { ksplit: false, threads: 128 });
            let rows_tg = (plan.threads / 32).max(1) as u32;
            let nsl = 36u32;
            let srows = (self.arch.vocab as u32 / nsl) / rows_tg * rows_tg; // rows per slice
            time_cat("lm_head/36", nsl as usize, &|enc, i| {
                let off = i as u64 * srows as u64;
                enc.set_compute_pipeline_state(&self.p["gemv_q8"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(w), off * d as u64);
                enc.set_buffer(2, Some(&self.st.logits), off * 4);
                enc.set_bytes(3, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &srows as *const u32 as *const c_void);
                enc.set_buffer(5, Some(sc), off * 4);
                enc.dispatch_thread_groups(MTLSize::new((srows / rows_tg) as u64, 1, 1), MTLSize::new(plan.threads as u64, 1, 1));
            });
        }
        // Weight bytes the GEMV categories move, over the time those categories
        // take — not over the whole token, which is the naive weights/total-time
        // division.
        let wb = b_qkv + b_o + b_ssm + b_gu + b_dn + b_lm;
        tracing::debug!(target: "profile",
            "weight-reading GEMVs: {:.1} MB/token in {gemv_ms:.3} ms (iso) = {:.0} GB/s. \
             Compare the `ovl` column: the SAME kernels overlapped reach the f16 rate, \
             so a shortfall here is dispatch granularity, not the inner loop.",
            wb / 1e6, wb / 1e6 / gemv_ms);
        tracing::debug!(target: "profile",
            "sum of categories (no-gap compute floor): {sum:.3} ms/token  \
             (GEMV {gemv_ms:.3} ms = {:.0}%, everything else {:.3} ms = {:.0}%)",
            gemv_ms / sum * 100.0, sum - gemv_ms, (sum - gemv_ms) / sum * 100.0);
    }

    /// Per-category GPU-time profiler for the batched (prefill) path.
    ///
    /// Decode is bandwidth-bound, dominated by reading weights once per token.
    /// Prefill at M=256 does 256x the arithmetic against the same weight reads, so
    /// it should be compute-bound and GEMM-dominated; when it is not, this says
    /// which kernel is holding the pass.
    ///
    /// Each category is dispatched `n_layers` times in one command buffer, cycling
    /// real layer weights so nothing is L2-resident across reps.
    pub fn profile_batch(&self, m: u32, pos: usize) {
        let d = self.d as u32;
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let group = (self.arch.n_head / self.arch.n_kv) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        let nh = self.arch.n_head as u32;
        let qdim = nh * hd;
        let ff = self.arch.ffn as u32;
        let nl = self.arch.n_layers;
        // Same hybrid handling as profile(): attention categories must cycle the
        // attention layers, or a Gated-DeltaNet layer is asked for attn_* tensors
        // it does not have.
        let attn_ls: Vec<usize> = (0..nl).filter(|&l| !self.arch.layers[l].is_ssm).collect();
        let n_attn = attn_ls.len().max(1);
        let bp = pos as u32;
        let aq = nh * hd;
        let ak = self.arch.n_kv as u32 * hd;
        let pl = |l: usize, s: &str| format!("blk.{}.{}", l % nl, s);
        // Attention categories index this, not pl(): on a hybrid, blk.0 is a
        // Gated-DeltaNet layer carrying attn_qkv/attn_gate and no attn_q at all.
        let al = |i: usize, s: &str| format!("blk.{}.{}", attn_ls[i % attn_ls.len()], s);

        let time_cat = |name: &str, reps: usize, f: &dyn Fn(&metal::ComputeCommandEncoderRef, usize)| -> f64 {
            for _ in 0..2 {
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                for i in 0..reps { f(&enc, i); }
                enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "profiling probe");
            }
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                for i in 0..reps { f(&enc, i); }
                enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "profiling probe");
                let (gs, ge): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
                best = best.min((ge - gs) * 1e3);
            }
            tracing::debug!(target: "ojas", "  {name:<14} {reps:>3} dispatches: {best:8.3} ms   ({:.4} ms/call)", best / reps as f64);
            best
        };

        tracing::debug!(target: "profile-batch", "M={m} pos={pos} d={d} ff={ff} layers={nl}");
        let mut sum = 0.0;
        // GEMM flops per category, for a TFLOP/s reading below.
        let mut gflop = 0.0f64;
        let mm_flops = |k: u32, n: u32| 2.0 * (k as f64) * (n as f64) * (m as f64) * (nl as f64) / 1e9;
        // Per-category efficiency: the aggregate hides which GEMM is weak, and they
        // differ by nearly 2x. A category is slow either because the kernel is slow
        // or because its N is too narrow to fill the GPU; only the per-shape
        // reading separates those.
        let mut eff: Vec<(&str, f64, f64, u32, u32)> = vec![];

        sum += time_cat("rmsnorm_m", 2 * nl, &|enc, i| {
            self.enc_reduce(enc, "rmsnorm_m", &[(&self.st.x, 0), (&self.wt.w32[&pl(i, "attn_norm.weight")], 1), (&self.st.h, 2)], &[(3, d)], &[(4, self.arch.eps)], m as u64, 256);
        });
        let t = time_cat("qkv (3 gemm)", n_attn, &|enc, i| {
            self.gemm_named(enc, &al(i, "attn_q.weight"), &self.st.h, &self.st.q, d, qdim, m, false);
            self.gemm_named(enc, &al(i, "attn_k.weight"), &self.st.h, &self.st.k, d, kvdim, m, false);
            self.gemm_named(enc, &al(i, "attn_v.weight"), &self.st.h, &self.st.v, d, kvdim, m, false);
        });
        sum += t; let f = mm_flops(d, qdim) + 2.0 * mm_flops(d, kvdim); gflop += f;
        eff.push(("qkv", f, t, d, qdim + 2*kvdim));
        // qkv bias: three separate epilogue dispatches per layer plus a barrier.
        // Rigel (arXiv 2606.12765) measures epilogue fusion at +6.5-12.9% in
        // cache-resident regimes; this category prices it before a fused store
        // exists.
        if self.wt.w32.contains_key(&pl(0, "attn_q.bias")) {
            sum += time_cat("qkv_bias (3)", n_attn, &|enc, i| {
                let p = |s: &str| pl(i, s);
                self.enc_reduce(enc, "add_rowbias_m", &[(&self.st.q, 0), (&self.wt.w32[&p("attn_q.bias")], 1)],
                    &[(2, qdim), (3, m*qdim)], &[], ((m*qdim + 63)/64) as u64, 64);
                self.enc_reduce(enc, "add_rowbias_m", &[(&self.st.k, 0), (&self.wt.w32[&p("attn_k.bias")], 1)],
                    &[(2, kvdim), (3, m*kvdim)], &[], ((m*kvdim + 63)/64) as u64, 64);
                self.enc_reduce(enc, "add_rowbias_m", &[(&self.st.v, 0), (&self.wt.w32[&p("attn_v.bias")], 1)],
                    &[(2, kvdim), (3, m*kvdim)], &[], ((m*kvdim + 63)/64) as u64, 64);
            });
        }
        sum += time_cat("rope_store_m", n_attn, &|enc, i| {
            self.enc_reduce(enc, "rope_qk_store_m",
                &[(&self.st.q,0),(&self.st.k,1),(&self.st.v,2),(&self.st.kcache[i % nl],3),(&self.st.vcache[i % nl],4),(&self.st.k,14)],
                &[(5,hd),(6,bp),(8,aq),(9,ak),(10,kvdim),(11,m),(12,self.arch.rope_neox as u32),(13,hd)], &[(7,self.arch.rope_base)],
                (((m*(aq+ak+kvdim))+63)/64) as u64, 64);
        });
        // Mirror forward_batch_impl's selection, or the profile measures a kernel
        // the prefill path does not run.
        let use_fat = self.gpu.native_reduce && hd == 128 && m >= 8
            && self.p.contains_key("attn_prefill_fat")
            // Opt-in: correct (attn_gate cos=1.000000) but slower than
            // attention_m_mma (16.8 vs 10.6 ms at M=256): register-resident Q
            // (16 half frags) + O (32 f32/lane) caps occupancy harder than the old
            // kernel's transposed device loads cost it. C=32 and C=64 measured
            // within 0.3% of each other, so the block size is not the issue.
            // Kept behind its numeric gate for a config that spends fewer
            // registers.
                && std::env::var("OJAS_ATTN_FAT").is_ok();
        let use_mma = self.gpu.native_reduce && hd <= 256 && hd % 64 == 0 && m >= 2;
        sum += time_cat(if use_fat { "attention (fat)" } else if use_mma { "attention (mma)" } else { "attention (short)" }, n_attn, &|enc, i| {
            if use_fat {
                enc.set_compute_pipeline_state(&self.p["attn_prefill_fat"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.kcache[attn_ls[i % attn_ls.len()]]), 0);
                enc.set_buffer(2, Some(&self.st.vcache[attn_ls[i % attn_ls.len()]]), 0);
                enc.set_buffer(3, Some(&self.st.attn), 0);
                for (bi, v) in [(4u32, hd), (5, kvdim), (6, bp), (7, group), (9, nh), (10, m)] {
                    enc.set_bytes(bi as u64, 4, &v as *const u32 as *const c_void);
                }
                enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, ((m + 31) / 32) as u64, 1), MTLSize::new(256, 1, 1));
            } else if use_mma {
                // Mirror forward_batch_impl's device-Q selection, including the
                // q_to_half convert, or the profile prices a kernel nobody runs.
                let dq = std::env::var("OJAS_ATTN_DQ").is_ok()
                    && self.p.contains_key("attention_m_mma_dq");
                if dq {
                    let total = ((m + 31) / 32) * 32 * qdim;
                    let valid = m * qdim;
                    self.enc_reduce(enc, "q_to_half", &[(&self.st.q, 0), (&self.st.qh, 1)],
                        &[(2, valid), (3, total)], &[], ((total + 63) / 64) as u64, 64);
                }
                let kname = if ojas_metal::kernels::attn::ATTN_HD_SPECIAL.contains(&hd) {
                    format!("attention_m_mma{}_{hd}", if dq { "_dq" } else { "" })
                } else if dq { "attention_m_mma_dq".to_string() } else { "attention_m_mma".to_string() };
                enc.set_compute_pipeline_state(&self.p[&kname]);
                enc.set_buffer(0, Some(if dq { &self.st.qh } else { &self.st.q }), 0);
                enc.set_buffer(1, Some(&self.st.kcache[attn_ls[i % attn_ls.len()]]), 0);
                enc.set_buffer(2, Some(&self.st.vcache[attn_ls[i % attn_ls.len()]]), 0);
                enc.set_buffer(3, Some(&self.st.attn), 0);
                for (bi, v) in [(4u32, hd), (5, kvdim), (6, bp), (7, group), (9, nh), (10, m)] {
                    enc.set_bytes(bi as u64, 4, &v as *const u32 as *const c_void);
                }
                enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, ((m + 31) / 32) as u64, 1), MTLSize::new(256, 1, 1));
            } else {
                self.enc_reduce(enc, "attention_m_short",
                    &[(&self.st.q,0),(&self.st.kcache[i % nl],1),(&self.st.vcache[i % nl],2),(&self.st.attn,3)],
                    &[(4,hd),(5,kvdim),(6,bp),(7,group),(9,nh)], &[(8,scale)], (m*nh) as u64, 64);
            }
        });
        let t = time_cat("o_proj gemm", n_attn, &|enc, i| {
            // o_proj contracts over qdim (n_head*hd), not d — they coincide on many
            // models but not on GQA/odd-head-dim ones (e.g. qwen3-0.6B).
            self.gemm_named(enc, &al(i, "attn_output.weight"), &self.st.attn, &self.st.x, qdim, d, m, true);
        });
        sum += t; let f = mm_flops(d, d); gflop += f;
        eff.push(("o_proj", f, t, d, d));
        let t = time_cat("ffn_gu (2 gemm)", nl, &|enc, i| {
            let gn = pl(i, "ffn_gate.weight");
            let un = pl(i, "ffn_up.weight");
            // Mirror forward_batch_impl: fused fat kernel when eligible.
            if self.gpu.native_reduce && ff % 64 == 0 && d % 32 == 0 && m >= 8
                && self.p.contains_key("ffn_gu_fat")
                && self.wt.w4l.contains_key(&gn) && self.wt.w4l.contains_key(&un)
                // Opt-in: fused measured 99.8 vs 92.0 ms — the 25% fragment-load
                // saving costs more in registers (64 accumulator floats/lane) and
                // dual-tile shmem than it returns. Kept for a config that frees
                // registers elsewhere.
                && std::env::var("OJAS_FFN_FAT").is_ok()
            {
                enc.set_compute_pipeline_state(&self.p["ffn_gu_fat"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w4l[&gn]), 0);
                enc.set_buffer(2, Some(&self.wt.w4l[&un]), 0);
                enc.set_buffer(3, Some(&self.st.gate), 0);
                enc.set_buffer(4, Some(&self.st.up), 0);
                enc.set_bytes(5, 4, &d as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &ff as *const u32 as *const c_void);
                enc.set_buffer(7, Some(&self.wt.q4l_a[&gn]), 0);
                enc.set_buffer(8, Some(&self.wt.q4l_b[&gn]), 0);
                enc.set_buffer(9, Some(&self.wt.q4l_a[&un]), 0);
                enc.set_buffer(10, Some(&self.wt.q4l_b[&un]), 0);
                enc.set_bytes(11, 4, &m as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(((m + 63) / 64) as u64, (ff / 64) as u64, 1), MTLSize::new(128, 1, 1));
            } else {
                self.gemm_named(enc, &gn, &self.st.h, &self.st.gate, d, ff, m, false);
                self.gemm_named(enc, &un, &self.st.h, &self.st.up, d, ff, m, false);
            }
        });
        sum += t; let f = 2.0 * mm_flops(d, ff); gflop += f;
        eff.push(("ffn_gu", f, t, d, ff));
        sum += time_cat("silu_mul", nl, &|enc, _| {
            self.enc_reduce(enc, "silu_mul", &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)], &[(3, m*ff)], &[], ((m*ff + 63)/64) as u64, 64);
        });
        let t = time_cat("ffn_down gemm", nl, &|enc, i| {
            self.gemm_named(enc, &pl(i, "ffn_down.weight"), &self.st.act, &self.st.x, ff, d, m, true);
        });
        sum += t; let f = mm_flops(ff, d); gflop += f;
        eff.push(("ffn_down", f, t, ff, d));

        // lm_head + argmax: once per chunk, not per layer, but not a rounding error
        // at verify-sized M. The batched head runs the MMA GEMM, whose M tile is 32
        // rows, so an M=4 verify computes 32 and discards 28 on the largest matrix
        // in the model. Reported separately so the per-layer floor above stays
        // comparable across M, and totalled below.
        let vocab = self.arch.vocab as u32;
        let head = if self.wt.w8.contains_key(&self.arch.lm_head) {
            time_cat("lm_head (gemm)", 1, &|enc, _| {
                self.gemm8(enc, &self.st.h, &self.wt.w8[&self.arch.lm_head],
                    &self.wt.scale8[&self.arch.lm_head], &self.st.logits, d, vocab, m, false);
            })
        } else { 0.0 };
        let am = time_cat("argmax_m", 1, &|enc, _| {
            self.enc_reduce(enc, "argmax_m", &[(&self.st.logits, 0), (&self.st.tmp, 1)],
                &[(2, vocab), (3, m)], &[], m as u64, self.tune.max_tg.min(1024));
        });
        tracing::debug!(target: "profile-batch", "sum of categories: {sum:.3} ms for {m} tokens  => {:.1} tok/s (no-gap floor)",
            m as f64 / (sum / 1e3));
        tracing::debug!(target: "profile-batch", "+ head {:.3} ms (lm_head {head:.3} + argmax {am:.3}) => TOTAL {:.3} ms",
            head + am, sum + head + am);
        tracing::debug!(target: "profile-batch", "GEMM flops {:.1} GFLOP", gflop);
        // Per-GEMM efficiency plus the tile grid each shape produces. `tiles` is
        // the threadgroup count of one 64-wide-N, 32-tall-M tile pass: a category
        // below the best TFLOP/s with few tiles is starved for parallelism rather
        // than slow, and wants a narrower tile or a split.
        let best = eff.iter().map(|e| e.1 / (e.2 / 1e3) / 1e3).fold(0.0f64, f64::max);
        tracing::debug!(target: "profile-batch", "per-GEMM efficiency (best in-house = {best:.2} TFLOP/s):");
        tracing::debug!(target: "ojas", "    {:<10} {:>8} {:>9} {:>8} {:>7} {:>10}", "category", "ms", "TFLOP/s", "tiles", "of best", "recover ms");
        let mut recover = 0.0;
        for (name, f, t, kk, nn) in &eff {
            let tf = f / (t / 1e3) / 1e3;
            let tiles = ((m + 31) / 32) * (nn / 64).max(1);
            let rec = t - f / best / 1e3 * 1e3;
            recover += rec.max(0.0);
            tracing::debug!(target: "ojas", "    {:<10} {:>8.2} {:>9.2} {:>8} {:>6.0}% {:>10.2}   (k={kk} n={nn})",
                name, t, tf, tiles, 100.0 * tf / best, rec.max(0.0));
        }
        tracing::debug!(target: "profile-batch", "recoverable at in-house parity: {recover:.2} ms => floor {:.1} tok/s",
            m as f64 / ((sum - recover) / 1e3));
        tracing::debug!(target: "profile-batch", "NOTE: per-layer sum excludes embed.");
    }

}

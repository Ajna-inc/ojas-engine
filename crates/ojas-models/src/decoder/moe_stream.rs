#![allow(clippy::too_many_arguments)]
use super::*;
use std::ffi::c_void;
use ojas_core::cancel::STREAM_CANCEL;
 // re-export

impl<'a> DecoderGpu<'a> {
    /// Join any in-flight background prefetch so the staging buffers are fully written and
    /// safe for the main thread to read. Take-then-join keeps the RefCell borrow short (the
    /// join happens with no borrow held, so the background thread can never observe the cache).
    pub(crate) fn dbuf_join(&self) {
        let h = self.strm.expert_cache.borrow_mut().dbuf_take_handle();
        if let Some(h) = h { let _ = h.join(); }
    }

    /// Per-stream L2 norms of the hyper-connection residual, for localizing a
    /// numerical fault without a reference implementation: a stream that blows up or
    /// collapses identifies the layer. Gated on OJAS_HC_TRACE because it reads back
    /// GPU memory between layers.
    pub(crate) fn hc_trace(&self, tag: &str) {
        if !self.cfg.hc_trace { return; }
        let Some(q) = self.arch.qwen4exp.as_ref() else { return };
        let (d, hc) = (self.d, q.hc_mult as usize);
        let v = unsafe { std::slice::from_raw_parts(self.st.hc_res.contents() as *const f32, d * hc) };
        let mut out = String::new();
        let mut bad = 0usize;
        for s in 0..hc {
            let sl = &v[s * d..(s + 1) * d];
            let l2 = sl.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mx = sl.iter().fold(0f32, |a, x| a.max(x.abs()));
            bad += sl.iter().filter(|x| !x.is_finite()).count();
            out.push_str(&format!(" s{s}:l2={l2:.3e},max={mx:.3e}"));
        }
        // `h` is the block output the layer just scattered into the streams.
        let hv = unsafe { std::slice::from_raw_parts(self.st.h.contents() as *const f32, d) };
        let hl2 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
        // The HC-mixed read view every block consumes. Pre-norm makes it the
        // scale-invariant input, so drift here points at hc_mix rather than at the
        // block that read it.
        let mv = unsafe { std::slice::from_raw_parts(self.st.hc_mixed.contents() as *const f32, d) };
        let ml2 = mv.iter().map(|x| x * x).sum::<f32>().sqrt();
        tracing::info!(target: "hc", "{tag}{out} |h|={hl2:.3e} |mixed|={ml2:.3e}{}", if bad > 0 { format!(" NONFINITE={bad}") } else { String::new() });
    }

    /// First layer index that routes experts: DeepSeek/GLM lead with a few dense
    /// FFN blocks, every other MoE arch routes from layer 0.
    ///
    /// Whether this architecture implements the Route/Experts split the streamed
    /// driver needs. Same predicates `encode_phase` dispatches on, so the two cannot
    /// drift.
    pub(crate) fn phase_split(&self) -> bool {
        self.arch.mla.is_some() || self.arch.qwen4exp.is_some()
    }

    /// Any layer held resident (partial or full). 0 = pure streaming.
    pub(crate) fn resident_any(&self) -> bool { self.strm.resident_layers > 0 }
    /// Whether layer `l`'s experts are wired resident (gather-free).
    pub(crate) fn is_resident(&self, l: usize) -> bool { l < self.strm.resident_layers }

    pub(crate) fn moe_leading_dense(&self) -> usize {
        self.arch.mla.as_ref().map(|m| m.leading_dense as usize).unwrap_or(0)
    }

    pub(crate) fn direct_expert_format(&self, l: usize) -> Option<u32> {
        let ty = |name| self.strm.stream_meta.get(&format!("blk.{l}.{name}")).map(|m| m.3);
        let gate = ty("ffn_gate_exps.weight")?;
        (matches!(gate, 21 | 23) && ty("ffn_up_exps.weight") == Some(gate)
            && ty("ffn_down_exps.weight") == Some(20)).then_some(gate)
    }

    pub(crate) fn direct_expert_resources(&self, enc: &metal::ComputeCommandEncoderRef) {
        let live = self.strm.direct_live.borrow();
        if self.cfg.flash_expert_pool {
            // All pooled allocations share one backing buffer. Declare it once.
            if let Some(allocation) = live.first() {
                enc.use_resource(&allocation.buffer, metal::MTLResourceUsage::Read);
            }
        } else {
            for allocation in live.iter() {
                enc.use_resource(&allocation.buffer, metal::MTLResourceUsage::Read);
            }
        }
        for buffer in [&self.strm.moe_gs, &self.strm.moe_us, &self.strm.moe_ds] {
            enc.use_resource(buffer, metal::MTLResourceUsage::Read);
        }
    }

    pub(crate) fn gather_experts(&self, l: usize) { self.gather_experts_m(l, 1) }

    /// Pack layer `l`'s routed experts into the GPU scratch, for the first `mtok`
    /// rows of `moe_idx`.
    ///
    /// The M tokens' top-k sets overlap heavily — consecutive tokens route to
    /// largely the same experts — so the scratch holds their *union*, packed once,
    /// and `moe_slot` records where each token's k-th pick landed. Gathering
    /// per token instead would re-read shared experts and need M times the scratch.
    /// At mtok=1 the union is the token's own top-k in order, so `moe_slot` comes
    /// out as the identity and this is byte-for-byte the decode gather.
    pub(crate) fn gather_experts_m(&self, l: usize, mtok: usize) {
        let trace = self.flash_trace_start();
        let stats = std::cell::RefCell::new(FlashTargetTiming::default());
        self.gather_experts_m_timed(l, mtok, if trace.is_some() { Some(&stats) } else { None });
        self.flash_trace_finish(trace, "gather", l, mtok, stats.into_inner());
    }

    pub(crate) fn gather_experts_m_timed(&self, l: usize, mtok: usize,
        timing: Option<&std::cell::RefCell<FlashTargetTiming>>) {
        let mut phase = timing.map(|_| std::time::Instant::now());
        let mut tick = |bucket: fn(&mut FlashTargetTiming) -> &mut f64| {
            if let Some(stats) = timing {
                *bucket(&mut stats.borrow_mut()) += phase.unwrap().elapsed().as_secs_f64();
                phase = Some(std::time::Instant::now());
            }
        };
        let (mg, m) = match (&self.wt.mapped, self.arch.moe) { (Some(mg), Some(m)) => (mg, m), _ => return };
        let (nu, ne) = (m.n_used as usize, m.n_expert as u64);
        // n_layers + 1 slots: the NextN/MTP draft block routes its own experts from
        // slot n_layers, and `l` here can be that block.
        let ids = unsafe { std::slice::from_raw_parts(self.ms.moe_idx.contents() as *const u32, (self.arch.n_layers + 1) * MAXM * nu) };
        let base = l * MAXM * nu;
        // union of the M tokens' picks, in first-seen order, plus the per-pick slot map
        let mut where_e = vec![u32::MAX; ne as usize];
        let mut union: Vec<u32> = Vec::with_capacity(mtok * nu);
        let mut slots: Vec<u32> = vec![0; mtok * nu];
        for t in 0..mtok {
            for j in 0..nu {
                let e = ids[base + t * nu + j];
                if e as u64 >= ne { continue; }
                if where_e[e as usize] == u32::MAX {
                    where_e[e as usize] = union.len() as u32;
                    union.push(e);
                }
                slots[t * nu + j] = where_e[e as usize];
            }
        }
        assert!(union.len() <= self.strm.gather_cap,
            "layer {l}: {} tokens routed {} distinct experts, scratch holds {} \
             (raise --ubatch-size or lower the batch)", mtok, union.len(), self.strm.gather_cap);
        unsafe { std::ptr::copy_nonoverlapping(slots.as_ptr(), self.strm.moe_slot.contents() as *mut u32, slots.len()); }
        if ojas_core::config::var("OJAS_LAYER_DUMP").is_ok() {
            let wg = unsafe { std::slice::from_raw_parts(self.ms.moe_wgt.contents() as *const f32, nu) };
            let sh = unsafe { std::slice::from_raw_parts(self.st.tmp.contents() as *const f32, self.d) };
            let shl2 = sh.iter().map(|v| v * v).sum::<f32>().sqrt();
            tracing::trace!(target: "rdump", "l={l} idx={:?} wgt={:?} sh_l2={shl2:.4} sh={:?}", &ids[base..base + nu], wg, &sh[..4]);
        }
        // First-divergence dump. At one execution point per layer it captures the
        // three things that separate a router defect from an upstream activation
        // difference: the activation entering the expert computation, the routing
        // coefficients, and the complete selected indices (not a signature).
        //
        // It also validates the slot map the consumer dereferences: `moe_slot[row,
        // pick]` must be in range and must name the union entry equal to
        // `moe_idx[row, pick]`. A correct table with a wrong slot map reads the wrong
        // expert while every byte-level check passes.
        if let Ok(path) = ojas_core::config::var("OJAS_DIVERGE_DUMP") {
            use std::io::Write as _;
            let wg = unsafe { std::slice::from_raw_parts(self.ms.moe_wgt.contents() as *const f32, mtok * nu) };
            let act = unsafe { std::slice::from_raw_parts(self.st.tmp.contents() as *const f32, self.d) };
            let act_l2: f64 = act.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
            let act_hash = act.iter().fold(2166136261u32, |h, v| (h ^ v.to_bits()).wrapping_mul(16777619));
            let mut bad_slots = Vec::new();
            for t in 0..mtok {
                for j in 0..nu {
                    let want = ids[base + t * nu + j];
                    if want as u64 >= ne { continue; }
                    let slot = slots[t * nu + j] as usize;
                    if slot >= union.len() || union[slot] != want {
                        bad_slots.push(format!("row{t}.pick{j}:slot{slot}->{:?}!={want}",
                            union.get(slot)));
                    }
                }
            }
            let seq = self.strm.gather_reads.get();
            let rec = format!(
                "{{\"seq\":{seq},\"layer\":{l},\"mtok\":{mtok},\"union_len\":{},\
                 \"act_l2\":{act_l2:.9},\"act_hash\":{act_hash},\
                 \"idx\":{:?},\"wgt\":{:?},\"slots\":{:?},\"bad_slots\":{:?}}}\n",
                union.len(), &ids[base..base + mtok * nu], wg, &slots[..mtok * nu], bad_slots);
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let _ = f.write_all(rec.as_bytes());
            }
        }
        // Make the prefetch (if any) that ran during the previous layer's expert GEMMs visible.
        // After the join the background thread is gone, so reading the staging buffers below is
        // race-free. When OJAS_MOE_DBUF unset no prefetch was ever spawned → dbuf_join is a no-op
        // and dbuf_staged_ptr always returns None, so this path is byte-identical to the serial one.
        self.dbuf_join();
        // All callers wait for the preceding GPU work before gathering. Keep
        // this batch's buffer clones alive even if LFU admission evicts a hit.
        self.strm.direct_live.borrow_mut().clear();
        let instrument = super::audit::instrumented_layer() == Some(l);
        if instrument { self.strm.expert_hash_records.borrow_mut().clear(); }
        let direct = self.cfg.flash_direct_experts && self.arch.qwen4exp.is_some()
            && self.direct_expert_format(l).is_some();
        self.strm.direct_layer.set(direct.then_some(l));
        let tens = [(0u8, "ffn_gate_exps.weight", &self.strm.moe_gs), (1u8, "ffn_up_exps.weight", &self.strm.moe_us), (2u8, "ffn_down_exps.weight", &self.strm.moe_ds)];
        tick(|s| &mut s.gather_setup_s);
        // Pass 1: serial cache lookup, then serial or scoped-parallel copies to
        // scratch. Cache hits copy from RAM; prefetched experts memcpy from staging
        // (also RAM speed — the disk read already happened off the critical path);
        // anything still absent is queued for a synchronous pread. `newborn` = experts
        // materialized this call (from staging or disk) that must be admitted to the LRU.
        let mut misses: Vec<GatherJob> = Vec::with_capacity(3 * union.len());
        let copy_threads = self.cfg.expert_copy_threads.clamp(1, 8);
        let mut copies: Vec<(usize, usize, usize)> = Vec::new(); // (src, dst, len)
        let mut newborn: Vec<(u64, usize, usize)> = Vec::with_capacity(3 * union.len()); // (key, dst, len)
        {
            let mut cache = self.strm.expert_cache.borrow_mut();
            for (kind, s, scratch) in tens {
                if let Some(&(part, abs, rawlen, _)) = self.strm.stream_meta.get(&format!("blk.{l}.{s}")) {
                    let stride = rawlen / ne;
                    let dstb = scratch.contents() as *mut u8;
                    for (slot, &eu) in union.iter().enumerate() {
                        let e = eu as u64;
                        let key = ((l as u64) << 40) | (e << 8) | kind as u64;
                        if kind == 0 && (self.cfg.expert_stats.is_some() || self.cfg.moe_skew) {
                            *self.strm.expert_stats.borrow_mut().entry((l as u32, e as u32)).or_insert(0) += 1;
                        }
                        let dst = unsafe { dstb.add(slot * stride as usize) };
                        let table = self.strm.direct_tables[kind as usize].contents() as *mut u64;
                        if direct {
                            // A miss is read into scratch by the normal path below.
                            unsafe { *table.add(slot) = scratch.gpu_address() + slot as u64 * stride; }
                        }
                        self.strm.gather_reads.set(self.strm.gather_reads.get() + 1);
                        if let Some(src) = cache.get(key) {
                            self.strm.gather_hits.set(self.strm.gather_hits.get() + 1);
                            if direct {
                                if let Some(buffer) = cache.metal_buffer(key) {
                                    unsafe { *table.add(slot) = buffer.address(); }
                                    self.strm.direct_live.borrow_mut().push(buffer);
                                    if let Some(stats) = timing { stats.borrow_mut().direct_expert_bytes += stride; }
                                    continue;
                                }
                            }
                            if copy_threads == 1 {
                                unsafe { std::ptr::copy_nonoverlapping(src, dst, stride as usize); }
                            } else {
                                copies.push((src as usize, dst as usize, stride as usize));
                            }
                        } else if let Some(src) = cache.dbuf_staged_ptr(key) {
                            // Prefetch hit: staged bytes are pread from the same fd/offset the miss
                            // path below would use → byte-identical. Skips the blocking disk read.
                            if copy_threads == 1 {
                                unsafe { std::ptr::copy_nonoverlapping(src, dst, stride as usize); }
                            } else {
                                copies.push((src as usize, dst as usize, stride as usize));
                            }
                            newborn.push((key, dst as usize, stride as usize));
                        } else {
                            misses.push(GatherJob { key, fd: mg.fd(part), off: (abs + e * stride) as i64, len: stride as usize, dst: dst as usize });
                        }
                    }
                }
            }
        }
        if !copies.is_empty() {
            // SAFETY: the prefetch was joined above. get() only touches counters;
            // no cache insertion, eviction or staging refill occurs until this
            // scope has joined. Each component/union slot has a disjoint scratch
            // destination, and the previous GPU command has completed. Workers
            // receive only byte ranges, never the cache or a Metal encoder.
            let nth = copy_threads.min(copies.len());
            let chunk_size = copies.len().div_ceil(nth);
            let copy = |jobs: &[(usize, usize, usize)]| {
                for &(src, dst, len) in jobs {
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len); }
                }
            };
            std::thread::scope(|scope| {
                let mut chunks = copies.chunks(chunk_size);
                let local = chunks.next().unwrap();
                for chunk in chunks { scope.spawn(move || copy(chunk)); }
                copy(local);
            });
        }
        tick(|s| &mut s.gather_copy_s);
        // Pass 2 (parallel pread) — read directly into the GPU scratch (no transient Vec).
        // A per-miss Vec alloc would churn ~13GB/token, pressuring macOS to compress/evict the
        // resident skeleton → the attention GEMVs would then re-read 17GB/token of skeleton at
        // disk speed (the route-phase blowup). Positioned pread also bypasses the mmap VMA lock.
        if !misses.is_empty() {
            let nth = self.cfg.gather_threads.min(misses.len());
            let cs = (misses.len() + nth - 1) / nth.max(1);
            std::thread::scope(|sc| {
                for chunk in misses.chunks(cs.max(1)) {
                    sc.spawn(move || {
                        for job in chunk {
                            unsafe { libc::pread(job.fd, job.dst as *mut c_void, job.len, job.off); }
                        }
                    });
                }
            });
        }
        tick(|s| &mut s.gather_read_s);
        // Pass 3 (only if caching): admit the freshly-materialized experts (disk misses
        // and staging hits) into the LRU from the scratch bytes. A staging hit is a miss
        // whose read happened early, so output is byte-identical (the true expert bytes
        // always land in moe_gs); only the LFU eviction order may differ slightly
        // (staged experts are admitted in two passes vs one), which affects future cache
        // hit-rate/telemetry, never the tokens produced.
        if self.strm.expert_cache.borrow().budget() > 0 {
            let mut cache = self.strm.expert_cache.borrow_mut();
            for job in &misses {
                let mut b = vec![0u8; job.len];
                unsafe { std::ptr::copy_nonoverlapping(job.dst as *const u8, b.as_mut_ptr(), job.len); }
                cache.insert(job.key, b.into_boxed_slice());
            }
            for &(key, dst, len) in &newborn {
                let mut b = vec![0u8; len];
                unsafe { std::ptr::copy_nonoverlapping(dst as *const u8, b.as_mut_ptr(), len); }
                cache.insert(key, b.into_boxed_slice());
            }
        }
        tick(|s| &mut s.gather_admit_s);
        if instrument {
            self.record_expert_table(l, &union);
        }
    }

    /// Diagnostic: resolve each slot the way the consumer resolves it, and record the
    /// hash of the bytes that should be there.
    ///
    /// Buffer selection must mirror `qwen4exp_moe_m_experts`: a direct layer reads
    /// `direct_tables[kind][slot]`, everything else reads `scratch + slot*stride`.
    /// Reading the direct table unconditionally is wrong on a scratch layer — the
    /// gather only writes those entries when `direct` is set, so they hold stale
    /// addresses from whichever direct layer last ran, and interpreting them with this
    /// layer's stride fabricates mismatches. Layer 47's down weights are Q8_0, which
    /// disqualifies direct addressing.
    ///
    /// The expected hash comes from the model file, so a slot only agrees when the GPU
    /// reads the routed expert's real bytes.
    fn record_expert_table(&self, l: usize, union: &[u32]) {
        let Some(moe) = self.arch.moe.as_ref() else { return };
        let Some(mapped) = self.wt.mapped.as_ref() else { return };
        let ne = moe.n_expert as u64;
        let mut records = self.strm.expert_hash_records.borrow_mut();
        let live = self.strm.direct_live.borrow();
        // Exactly the consumer's test.
        let direct = self.strm.direct_layer.get() == Some(l);
        let names = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];
        let scratches = [&self.strm.moe_gs, &self.strm.moe_us, &self.strm.moe_ds];
        let addr_out = self.st.expert_hash_addr.contents() as *mut u64;
        for (kind, name) in names.iter().enumerate() {
            let Some(&(part, abs, rawlen, ty)) = self.strm.stream_meta.get(&format!("blk.{l}.{name}")) else { continue };
            let stride = (rawlen / ne) as usize;
            let need = stride as u64;
            let scratch = scratches[kind];
            let (s_gpu, s_cpu, s_len) = (scratch.gpu_address(), scratch.contents() as *const u8, scratch.length());
            let table = self.strm.direct_tables[kind].contents() as *const u64;
            for (slot, &expert) in union.iter().enumerate() {
                // As the consumer resolves it.
                let address = if direct {
                    unsafe { *table.add(slot) }
                } else {
                    s_gpu + slot as u64 * stride as u64
                };
                unsafe { *addr_out.add(kind * 1024 + slot) = address; }
                let cpu = live.iter()
                    .find(|a| address >= a.address() && address + need <= a.address() + a.len as u64)
                    .map(|a| unsafe { a.contents().add((address - a.address()) as usize) as *const u8 })
                    .or_else(|| {
                        (address >= s_gpu && address + need <= s_gpu + s_len)
                            .then(|| unsafe { s_cpu.add((address - s_gpu) as usize) })
                    });
                let mut source = vec![0u8; stride];
                let got = unsafe {
                    libc::pread(mapped.fd(part), source.as_mut_ptr() as *mut c_void, stride,
                                (abs + expert as u64 * stride as u64) as i64)
                };
                let expected = if got == stride as isize {
                    super::audit::sample_hash(source.as_ptr(), stride, super::audit::HASH_SAMPLES)
                } else { 0 };
                records.push(super::audit::SlotRecord {
                    layer: l as u32, kind: kind as u8, slot: slot as u32, expert,
                    address, offset_in_tensor: expert as u64 * stride as u64, len: stride,
                    expected_hash: expected,
                    resolved: cpu.is_some() && got == stride as isize,
                    direct, ggml_type: ty,
                });
            }
        }
    }

    pub(crate) fn dbuf_prefetch(&self, l: usize) { self.dbuf_prefetch_m(l, 1) }

    /// As above, predicting for `mtok` rows. A batched pass gathers the union of its
    /// rows' picks, so prefetching only row 0 leaves exactly the experts the extra
    /// rows added — the ones the union had to go to disk for — unstaged.
    pub(crate) fn dbuf_prefetch_m(&self, l: usize, mtok: usize) {
        if l >= self.arch.n_layers { return; }
        let ld = self.moe_leading_dense();
        if l < ld { return; } // dense layer: no experts to gather
        let (mg, m) = match (&self.wt.mapped, self.arch.moe) { (Some(mg), Some(m)) => (mg, m), _ => return };
        let (nu, ne) = (m.n_used as usize, m.n_expert as u64);
        let ids = unsafe { std::slice::from_raw_parts(self.ms.moe_idx.contents() as *const u32, (self.arch.n_layers + 1) * MAXM * nu) };
        let base = l * MAXM * nu;
        let sizes = [self.strm.moe_gs.length() as usize, self.strm.moe_us.length() as usize, self.strm.moe_ds.length() as usize];
        let tens = [(0usize, "ffn_gate_exps.weight"), (1usize, "ffn_up_exps.weight"), (2usize, "ffn_down_exps.weight")];
        // Candidate = predicted-routed expert not already cache-resident (a resident one is a
        // guaranteed cache hit next layer, so staging it would waste a disk read). (key,kind,fd,off,len)
        let mut cand: Vec<(u64, usize, i32, i64, usize)> = Vec::with_capacity(3 * mtok * nu);
        {
            let cache = self.strm.expert_cache.borrow();
            let mut seen = vec![false; ne as usize];
            for (kind, s) in tens {
                if let Some(&(part, abs, rawlen, _)) = self.strm.stream_meta.get(&format!("blk.{l}.{s}")) {
                    let stride = rawlen / ne;
                    seen.iter_mut().for_each(|v| *v = false);
                    for j in 0..mtok * nu {
                        let e = ids[base + j] as u64;
                        if e >= ne || seen[e as usize] { continue; }
                        seen[e as usize] = true;
                        let key = ((l as u64) << 40) | (e << 8) | kind as u64;
                        if cache.contains(key) { continue; }
                        cand.push((key, kind, mg.fd(part), (abs + e * stride) as i64, stride as usize));
                    }
                }
            }
        }
        if cand.is_empty() { return; }
        let jobs = self.strm.expert_cache.borrow_mut().dbuf_stage_jobs(sizes, &cand);
        if jobs.is_empty() { return; }
        // dst pointers alias the staging Vecs, which are allocated once and never resized, so the
        // pointers stay valid for the thread's lifetime; the handle is joined before staging is
        // read or re-staged. Positioned pread with an explicit offset shares no file cursor → the
        // parallel reads are independent, and they touch only staging (never moe_gs / the cache).
        let nth = self.cfg.gather_threads.max(1);
        let handle = std::thread::spawn(move || {
            let n = nth.min(jobs.len()).max(1);
            let cs = (jobs.len() + n - 1) / n;
            std::thread::scope(|sc| {
                for chunk in jobs.chunks(cs.max(1)) {
                    sc.spawn(move || {
                        for &(fd, off, len, dst) in chunk {
                            unsafe { libc::pread(fd, dst as *mut c_void, len, off); }
                        }
                    });
                }
            });
        });
        self.strm.expert_cache.borrow_mut().dbuf_set_handle(handle);
    }

    /// OJAS_MOE_SKEW: routing-skew telemetry for sizing the expert cache. From the (layer,expert)
    /// routed-selection census (the same machinery OJAS_EXPERT_STATS dumps), emit to stderr:
    ///  - working-set W: distinct (layer,expert) slots touched this session;
    ///  - routing entropy H (bits) and its normalized form H/log2(W) (1.0 = uniform, 0 = one-hot);
    ///  - oracle-hit@S: the steady-state hit rate a cache of the S hottest slots would achieve
    ///    (frequency oracle = keep-most-frequent, which the LFU policy approximates) — read off
    ///    the S where the curve flattens to pick the cache size.
    pub(crate) fn emit_skew_stats(&self, pos: usize) {
        let stats = self.strm.expert_stats.borrow();
        if stats.is_empty() { return; }
        let total: u64 = stats.values().sum();
        if total == 0 { return; }
        let w = stats.len();
        let mut counts: Vec<u64> = stats.values().copied().collect();
        let mut h = 0.0f64;
        for &c in &counts { let p = c as f64 / total as f64; if p > 0.0 { h -= p * p.log2(); } }
        let hnorm = if w > 1 { h / (w as f64).log2() } else { 0.0 };
        counts.sort_unstable_by(|a, b| b.cmp(a)); // descending frequency
        let (n_expert, n_used) = self.arch.moe.map(|m| (m.n_expert as usize, m.n_used as usize)).unwrap_or((0, 1));
        let slot_list = [64usize, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
        let mut oracle = String::new();
        for &s in &slot_list {
            if s > w { break; }
            let hit: u64 = counts[..s.min(counts.len())].iter().sum();
            oracle.push_str(&format!(" @{}={:.1}%", s, 100.0 * hit as f64 / total as f64));
        }
        // rough RAM to hold the 1024 hottest (layer,expert) gate slots at the avg gate stride.
        let avg_stride = self.strm.moe_gs.length() as f64 / n_used.max(1) as f64;
        let gb_1024 = 1024.0 * avg_stride / 1e9;
        tracing::debug!(target: "moe-skew",
            "pos{pos} working_set={w} slots (of {n_expert} experts) accesses={total} entropy={h:.2}b norm={hnorm:.3} | oracle-hit{oracle} | ~{gb_1024:.1}GB@1024-gate-slots"
        );
    }

    /// Lookahead prefetch (pre-gate): madvise(WILLNEED) the routed experts of layers
    /// [l_start, l_end) using the routing already in moe_idx. During a token, moe_idx
    /// still holds the previous token's routing for any layer not yet computed — a
    /// strong predictor (~90% overlap for decode). Issuing it before the GPU faults
    /// turns the per-page fault storm into ~8 sequential per-expert reads per layer.
    #[allow(dead_code)] // disk-stream prefetch API; not yet wired into the decode loop
    pub(crate) fn stream_prefetch_routed(&self, l_start: usize, l_end: usize) {
        let (mg, m) = match (&self.wt.mapped, self.arch.moe) { (Some(mg), Some(m)) => (mg, m), _ => return };
        let (nu, ne) = (m.n_used as usize, m.n_expert);
        let ids = unsafe { std::slice::from_raw_parts(self.ms.moe_idx.contents() as *const u32, self.arch.n_layers * MAXM * nu) };
        for l in l_start..l_end.min(self.arch.n_layers) {
            for s in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"] {
                let name = format!("blk.{l}.{s}");
                if let Some(&(part, abs, rawlen, _)) = self.strm.stream_meta.get(&name) {
                    let stride = rawlen / ne as u64;
                    for j in 0..nu {
                        let e = ids[l * MAXM * nu + j];
                        if e < ne { mg.willneed(part, abs + e as u64 * stride, stride, false); }
                    }
                }
            }
        }
    }

    /// Stream path: create no-copy Metal buffers for the expert tensors of layers
    /// [l_start, l_end) from the mmap'd shards. Only a chunk's worth of expert bytes are
    /// mapped at once, staying under Metal's ~77GB working set (the full 439GB can't be
    /// mapped simultaneously). Buffers are released by stream_clear() after the chunk runs.
    #[allow(dead_code)]
    pub(crate) fn stream_bind(&self, l_start: usize, l_end: usize) {
        if !self.strm.stream { return; }
        let mg = match &self.wt.mapped { Some(m) => m, None => return };
        let mut bufs = self.strm.stream_bufs.borrow_mut();
        for l in l_start..l_end {
            for s in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"] {
                let name = format!("blk.{l}.{s}");
                if let Some(&(part, abs, rawlen, _)) = self.strm.stream_meta.get(&name) {
                    let (buf, off) = mg.buffer(self.gpu, part, abs, rawlen);
                    bufs.insert(name, (buf, off));
                }
            }
        }
    }
    #[allow(dead_code)]
    pub(crate) fn stream_clear(&self) { self.strm.stream_bufs.borrow_mut().clear(); }

    /// Scalar disk-streamed decode with partial/full residency, for any architecture
    /// that implements the Route/Experts split (see `encode_span_phase`).
    ///
    /// The layers run in small ranges, each in its own command buffer, to bound
    /// Metal's resident set (the no-copy expert buffers are only wired for the chunk
    /// that references them). A run of consecutive resident layers is encoded
    /// Full-phase in one command buffer (no gather); a streamed layer keeps
    /// route -> CPU gather -> experts. Reduces the ~49 command buffers/token toward
    /// `groups + streamed_layers`. The KV cache and residual stream (self.st.x)
    /// persist across command buffers, so results match the single-buffer forward.
    pub(crate) fn forward_id_partial(&self, token: u32, pos: usize) -> u32 {
        use std::sync::atomic::Ordering;
        if let Some(rs) = &self.strm.expert_residency { rs.renew(); }
        let seq = (pos + 1) as u32;
        let n = self.arch.n_layers;
        let d = self.d as u32;
        let hd = self.arch.hd as u32;
        let kvdim = (self.arch.n_kv * self.arch.hd) as u32;
        let grp = (self.arch.n_head / self.arch.n_kv.max(1)) as u32;
        let scale = 1.0 / (self.arch.hd as f32).sqrt();
        let g = self.cfg.flash_resident_group.max(1);
        let mut l = 0usize;
        while l < n {
            if STREAM_CANCEL.load(Ordering::Relaxed) { return u32::MAX; }
            if self.is_resident(l) {
                // Consecutive resident layers, Full phase, one command buffer.
                let mut hi = l;
                while hi < n && self.is_resident(hi) && (hi - l) < g { hi += 1; }
                let last = hi == n;
                objc::rc::autoreleasepool(|| {
                    let cb = self.gpu.command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    self.encode_span_phase(&enc, token, pos, d, hd, kvdim, grp, scale, seq,
                                           l, hi, l == 0, last, MoePhase::Full);
                    if last {
                        self.bar(&enc);
                        self.enc_reduce(&enc, "argmax", &[(&self.st.logits, 0), (&self.st.tmp, 1)],
                                        &[(2, self.arch.vocab as u32)], &[], 1, self.tune.max_tg.min(1024));
                    }
                    enc.end_encoding();
                    let _ = ojas_metal::commit_and_wait_checked(cb, "qwen4exp resident");
                });
                l = hi;
            } else {
                // Streamed layer: route, gather, experts (two command buffers).
                let last = l + 1 == n;
                objc::rc::autoreleasepool(|| {
                    let cb = self.gpu.command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    self.encode_phase(&enc, token, pos, seq, l, l + 1, l == 0, false, MoePhase::Route);
                    enc.end_encoding();
                    let _ = ojas_metal::commit_and_wait_checked(cb, "expert stream");
                });
                if ojas_core::device_fault::is_faulted() { return u32::MAX; }
                self.gather_experts_m(l, 1);
                objc::rc::autoreleasepool(|| {
                    let cb = self.gpu.command_buffer();
                    let enc = cb.new_compute_command_encoder();
                    self.encode_phase(&enc, token, pos, seq, l, l + 1, false, last, MoePhase::Experts);
                    if last {
                        self.bar(&enc);
                        self.enc_reduce(&enc, "argmax", &[(&self.st.logits, 0), (&self.st.tmp, 1)],
                                        &[(2, self.arch.vocab as u32)], &[], 1, self.tune.max_tg.min(1024));
                    }
                    enc.end_encoding();
                    let _ = ojas_metal::commit_and_wait_checked(cb, "expert stream");
                });
                l += 1;
            }
        }
        if ojas_core::device_fault::is_faulted() { return u32::MAX; }
        unsafe { *(self.st.tmp.contents() as *const u32) }
    }

    pub(crate) fn forward_id_streamed(&self, token: u32, pos: usize) -> u32 {
        // route → gather → compute, per layer. Each MoE layer runs in two command
        // buffers: (1) attention + router (writes moe_idx), then the CPU gathers only
        // the routed experts into packed scratch, then (2) the routed-expert GEMMs over
        // that scratch. Metal thus only ever wires ~n_used experts, not all 256.
        // Each cb is wrapped in its own autorelease pool (command_buffer() is autoreleased).
        let seq = (pos + 1) as u32;
        let n = self.arch.n_layers;
        let ld = self.moe_leading_dense() as u32;
        let dbg = self.cfg.glm_dbg;
        let t_tok = std::time::Instant::now();
        let (mut t_route, mut t_exp, mut t_gather) = (0.0f64, 0.0f64, 0.0f64);
        // Layer L's experts and layer L+1's router have no gather between them, so they
        // share a command buffer: 2n commit+wait round trips become n+1. On this model a
        // round trip measured ~1.16 ms against ~14 ms of GEMV work for the whole token.
        // The per-layer debug hooks read state between the two phases, so they keep the
        // unfused boundary.
        let unfused = ojas_core::config::var("OJAS_LAYER_DUMP").is_ok()
            || self.cfg.reference_trace.is_some()
            || ojas_core::config::var("OJAS_BLOCK_INFLUENCE").is_ok()
            || ojas_core::config::var("OJAS_H2_DUMP").is_ok();
        let run_experts = |pl: usize| {
            objc::rc::autoreleasepool(|| {
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                self.encode_phase(&enc, token, pos, seq, pl, pl + 1, false, pl == n - 1, MoePhase::Experts);
                if pl == n - 1 {
                    self.bar(&enc);
                    self.enc_reduce(&enc, "argmax", &[(&self.st.logits, 0), (&self.st.tmp, 1)], &[(2, self.arch.vocab as u32)], &[], 1, self.tune.max_tg.min(1024));
                }
                enc.end_encoding(); let _ = ojas_metal::commit_and_wait_checked(cb, "expert stream");
            });
        };
        let mut pend: Option<usize> = None;
        for l in 0..n {
            // Mid-forward cancel: a streamed token takes 5-16s, so per-token cancel isn't
            // responsive. The app's Stop sets STREAM_CANCEL → bail between layers (u32::MAX
            // marker; the caller discards it). KV for this pos is partial but never reused
            // (the session ends / next prompt re-prefills).
            if STREAM_CANCEL.load(std::sync::atomic::Ordering::Relaxed) { return u32::MAX; }
            let is_moe = (l as u32) >= ld;
            let last = l == n - 1;
            // OJAS_BLOCK_INFLUENCE=<path>: ShortGPT block influence — cos(x_in,x_out)
            // per layer. Low influence (cos≈1, input≈output) = redundant/droppable.
            let bi_in: Option<Vec<f32>> = if ojas_core::config::var("OJAS_BLOCK_INFLUENCE").is_ok() {
                Some(self.read_hidden())
            } else { None };
            let _tg = std::time::Instant::now();
            // Phase 1: the previous layer's experts, then attention + ffn-norm + router
            // (+shared); dense layers do the full FFN.
            let carry = pend.take();
            objc::rc::autoreleasepool(|| {
                let cb = self.gpu.command_buffer();
                let enc = cb.new_compute_command_encoder();
                if let Some(pl) = carry {
                    self.encode_phase(&enc, token, pos, seq, pl, pl + 1, false, false, MoePhase::Experts);
                }
                let head = last && !is_moe;
                self.encode_phase(&enc, token, pos, seq, l, l + 1, l == 0, head, MoePhase::Route);
                if head {
                    self.bar(&enc);
                    self.enc_reduce(&enc, "argmax", &[(&self.st.logits, 0), (&self.st.tmp, 1)], &[(2, self.arch.vocab as u32)], &[], 1, self.tune.max_tg.min(1024));
                }
                enc.end_encoding();
                // Reads moe_idx synchronously to build its candidate list, so it must run
                // before the commit that overwrites this layer's slot with the real routing.
                if self.cfg.moe_dbuf && is_moe { self.dbuf_prefetch(l); }
                let _ = ojas_metal::commit_and_wait_checked(cb, "expert stream");
            });
            t_route += _tg.elapsed().as_secs_f64();
            // OJAS_H2_DUMP=<layer>:<path>: after the Route cb completes, self.st.h holds this
            // layer's ffn-input (rmsnorm(x)·ffn_norm) — real activation samples X for
            // per-expert healing. One d-vector per token for the target layer.
            if let Ok(spec) = ojas_core::config::var("OJAS_H2_DUMP") {
                if let Some((tl, path)) = spec.split_once(':') {
                    if tl.parse::<usize>() == Ok(l) {
                        use std::io::Write;
                        let h = unsafe { std::slice::from_raw_parts(self.st.h.contents() as *const f32, self.d) };
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                            let bytes = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 4) };
                            let _ = f.write_all(bytes);
                        }
                    }
                }
            }
            self.hc_trace(&format!("L{l}mix"));
            if let (Some(dir), Some(mc)) = (self.cfg.reference_trace.as_ref(), self.arch.moe) {
                std::fs::create_dir_all(dir).expect("create reference trace directory");
                if self.arch.qwen4exp.is_some() && !self.arch.layers[l].is_ssm {
                    let lp=self.arch.layers[l];
                    for (name, buffer, offset, bytes, ext) in [
                        ("Qcur", &self.st.q, 0, lp.qdim as usize*4, "f32"),
                        ("Kstored", &self.st.kcache[l], pos*lp.kvdim as usize*2, lp.kvdim as usize*2, "f16"),
                        ("Vstored", &self.st.vcache[l], pos*lp.kvdim as usize*2, lp.kvdim as usize*2, "f16"),
                        ("attn_gated", &self.st.attn, 0, lp.qdim as usize*4, "f32"),
                        ("attn_output", &self.st.h, 0, self.d*4, "f32"),
                    ] {
                        let data=unsafe { std::slice::from_raw_parts((buffer.contents() as *const u8).add(offset),bytes) };
                        std::fs::write(format!("{dir}/{pos}-{name}-{l}.{ext}"),data).expect("write reference attention trace");
                    }
                }
                for (name, ptr, len) in [
                    ("ffn_moe_logits", self.ms.moe_lg.contents() as *const u8, mc.n_expert as usize * 4),
                    ("ffn_moe_topk", unsafe { (self.ms.moe_idx.contents() as *const u8).add(l * MAXM * mc.n_used as usize * 4) }, mc.n_used as usize * 4),
                ] {
                    let ext = if name.ends_with("topk") { "i32" } else { "f32" };
                    std::fs::write(format!("{dir}/{pos}-{name}-{l}.{ext}"), unsafe { std::slice::from_raw_parts(ptr,len) }).expect("write reference routing trace");
                }
            }
            if is_moe {
                // Phase 2: gather this layer's routed experts (moe_idx now written), run them.
                let _ta = std::time::Instant::now();
                self.gather_experts(l);
                t_gather += _ta.elapsed().as_secs_f64();
                pend = Some(l);
                if unfused || last {
                    let _te = std::time::Instant::now();
                    run_experts(l);
                    pend = None;
                    t_exp += _te.elapsed().as_secs_f64();
                }
                self.hc_trace(&format!("L{l}"));
                if ojas_core::config::var("OJAS_LAYER_DUMP").is_ok() {
                    let a = unsafe { std::slice::from_raw_parts(self.ms.moe_act.contents() as *const f32, 2048) };
                    let al2 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
                    tracing::trace!(target: "adump", "l={l} act_l2={al2:.4} act={:?}", &a[..4]);
                }
            }
            // OJAS_LAYER_DUMP=1: per-layer residual-stream fingerprint (CPU-vs-GPU bisection;
            if let (Some(dir), Some(q)) = (self.cfg.reference_trace.as_ref(), self.arch.qwen4exp.as_ref()) {
                // Explicit diagnostic capture at the completed layer boundary.
                // The unfused path above ensures the next layer has not modified it.
                std::fs::create_dir_all(&dir).expect("create reference trace directory");
                let bytes = unsafe { std::slice::from_raw_parts(self.st.hc_res.contents() as *const u8, self.d * q.hc_mult as usize * 4) };
                std::fs::write(format!("{dir}/{pos}-l_last-{l}.f32"), bytes).expect("write reference layer trace");
            }
            // cpu_glm prints the same lines — diff the two logs to find the divergent layer).
            if ojas_core::config::var("OJAS_LAYER_DUMP").is_ok() {
                let h = self.read_hidden();
                let l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                tracing::trace!(target: "ldump", "pos={pos} l={l} l2={l2:.4} x={:?}", &h[..4]);
            }
            if let (Ok(path), Some(xi)) = (ojas_core::config::var("OJAS_BLOCK_INFLUENCE"), bi_in) {
                let xo = self.read_hidden();
                let (mut dot, mut ni, mut no) = (0f64, 0f64, 0f64);
                for i in 0..self.d { dot += xi[i] as f64 * xo[i] as f64; ni += (xi[i] as f64).powi(2); no += (xo[i] as f64).powi(2); }
                let cos = dot / (ni.sqrt() * no.sqrt() + 1e-12);
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = writeln!(f, "{l} {cos:.6}");
                }
            }
        }
        if dbg { let (h, r) = (self.strm.gather_hits.get(), self.strm.gather_reads.get()); let (ph, pr) = self.strm.expert_cache.borrow_mut().dbuf_report(); tracing::trace!(target: "tok", "{pos} total={:.2}s route={:.2}s exp={:.2}s gather={:.2}s | cache {:.0}% ({}/{}) prefetch {}/{}", t_tok.elapsed().as_secs_f64(), t_route, t_exp, t_gather, if r>0 {100.0*h as f64/r as f64} else {0.0}, h, r, ph, pr); self.strm.gather_hits.set(0); self.strm.gather_reads.set(0); }
        if self.cfg.moe_skew && pos % 8 == 0 { self.emit_skew_stats(pos); }
        if let Some(sp) = &self.cfg.expert_stats {
            if pos % 8 == 0 {
                let stats = self.strm.expert_stats.borrow();
                let mut out = String::with_capacity(stats.len() * 16);
                out.push_str("layer,expert,hits\n");
                let mut rows: Vec<_> = stats.iter().collect();
                rows.sort();
                for (&(l, e), &c) in rows { out.push_str(&format!("{l},{e},{c}\n")); }
                let _ = std::fs::write(sp, out);
            }
        }
        // (no mmap prefetch here — the gather path preads only the routed experts; a mmap
        // WILLNEED would double-read the disk and pressure-evict the mlock'd skeleton.)
        if false {
            let lg = unsafe { std::slice::from_raw_parts(self.st.logits.contents() as *const f32, self.arch.vocab) };
            let (mut mx, mut ai) = (f32::MIN, 0usize);
            for (i, &v) in lg.iter().enumerate() { if v > mx { mx = v; ai = i; } }
            let nan = lg.iter().filter(|x| x.is_nan()).count();
            tracing::trace!(target: "glm-dbg", "pos{pos} logits max={mx:.4} nan={nan} argmax={ai}");
        }
        unsafe { *(self.st.tmp.contents() as *const u32) }
    }

    /// Routing telemetry readback: [n_layers × n_expert] router logits from the
    /// last forward (OJAS_ROUTE_STATS=1), plus (n_layers, n_expert, n_used).
    pub fn route_stats(&self) -> Option<(Vec<f32>, usize, usize, usize)> {
        let (rl, m) = (self.ms.route_lg.as_ref()?, self.arch.moe?);
        let ne = m.n_expert as usize;
        let v = unsafe { std::slice::from_raw_parts(rl.contents() as *const f32, self.arch.n_layers * ne) }.to_vec();
        Some((v, self.arch.n_layers, ne, m.n_used as usize))
    }

}

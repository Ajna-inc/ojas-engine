//! MoE FFN training block (router + grouped experts, fwd/bwd).
use ojas_metal::{MBuf, MetalGpu};
use ojas_core::Device as _;
use metal::MTLSize;


use crate::kit::{g1, g2, Kit};
use std::cell::{OnceCell, RefCell};

// Muon scratch: momentum (f32, contiguous ne*e*d; NS applied per-expert slice) plus shared NS
// scratch, ~60MB. Allocated lazily on the first opt_step_muon; healing never touches it.
struct MuonScratch {
    mwg: MBuf, mwu: MBuf, mwd: MBuf, msg: MBuf, msu: MBuf, msd: MBuf,
    nsx: MBuf, nst: MBuf, nsa: MBuf, nsaa: MBuf, nsb: MBuf, nsbx: MBuf, nrm: MBuf,
}

// gather/scatter scratch (gathered layout, nr = t_max*k rows): top-k compute instead of all-ne.
// Allocated lazily on the first forward_gs/backward_gs/route_counts.
struct GsScratch {
    gidx: MBuf, tok2: MBuf, gg: MBuf, rexp: MBuf,
    gath: MBuf, gl: MBuf, ul: MBuf, act: MBuf, eo: MBuf, shared: MBuf,
    deo: MBuf, dgr: MBuf, dgath: MBuf, dact: MBuf, dgl: MBuf, dul: MBuf, dgd: MBuf,
    // GPU routing scratch (t_moe_route_*): per-expert count/fill (u32, atomics)
    // and the exclusive-prefix offset [ne+1].
    cnt: MBuf, off: MBuf, fill: MBuf,
    // m-tile map for t_mm_grp_xwT (worst-case nr/32 + ne + 1 slots, GPU-built)
    texp: MBuf, trow0: MBuf, tmend: MBuf,
}

/// Uninitialized f32 device buffer (no MetalGpu handle needed — lazy scratch).
fn draw(dev: &metal::Device, n: usize) -> MBuf {
    MBuf { buf: dev.new_buffer((n * 4) as u64, metal::MTLResourceOptions::StorageModeShared), len: n }
}
fn dzero(dev: &metal::Device, n: usize) -> MBuf {
    let b = draw(dev, n);
    unsafe { std::ptr::write_bytes(b.buf.contents() as *mut u8, 0, n * 4) };
    b
}

pub struct MoeFfn {
    pub d: usize, pub e: usize, pub ne: usize, pub k: usize, pub t_max: usize,
    dev: metal::Device,
    // experts are contiguous: wg/wu/wd are one buffer each [ne*e*d] (expert ex at offset
    // ex*e*d), so one AdamW and one sumsq cover all experts instead of one dispatch per expert.
    pub wr: MBuf, pub wg: MBuf, pub wu: MBuf, pub wd: MBuf,
    pub sg: MBuf, pub su: MBuf, pub sd: MBuf,
    logits: MBuf, gates: MBuf, topk: MBuf,
    gl: Vec<MBuf>, ul: Vec<MBuf>, oe: Vec<MBuf>, gls: MBuf, uls: MBuf, act: MBuf,
    dact: MBuf, dgl: MBuf, dul: MBuf, dtmp: MBuf, dgate: MBuf, dlog: MBuf,
    pub gwr: MBuf, pub gwg: MBuf, pub gwu: MBuf, pub gwd: MBuf,
    pub gsg: MBuf, pub gsu: MBuf, pub gsd: MBuf,
    // optimizer states: router f32 (m,v); experts+shared 8-bit (mh f16, vq u8, vs scales)
    orm: MBuf, orv: MBuf,
    owg: (MBuf, MBuf, MBuf), owu: (MBuf, MBuf, MBuf), owd: (MBuf, MBuf, MBuf),
    osg: (MBuf, MBuf, MBuf), osu: (MBuf, MBuf, MBuf), osd: (MBuf, MBuf, MBuf),
    muon: OnceCell<MuonScratch>,
    gs: OnceCell<GsScratch>,
    // (count, offset) from the last forward_gs routing; backward_gs and route_counts reuse it
    // instead of re-running the identical routing.
    route_cache: RefCell<Option<(Vec<usize>, Vec<usize>)>>,
}

impl MoeFfn {
    pub fn new(gpu: &MetalGpu, d: usize, e: usize, ne: usize, k: usize, t_max: usize, seed: u64) -> MoeFfn {
        Self::build(gpu, d, e, ne, k, t_max, seed, true)
    }

    /// Like `new` but skips the random weight init, zero-filling instead, for callers that
    /// immediately overwrite every weight buffer (pack/repack paths). Saves ~9.4M LCG draws and
    /// f16 conversions per construction.
    pub fn new_empty(gpu: &MetalGpu, d: usize, e: usize, ne: usize, k: usize, t_max: usize) -> MoeFfn {
        Self::build(gpu, d, e, ne, k, t_max, 0, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(gpu: &MetalGpu, d: usize, e: usize, ne: usize, k: usize, t_max: usize, seed: u64, init: bool) -> MoeFfn {
        let mut rng = seed;
        let mut nxt = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 };
        let sc = (d as f32).powf(-0.5); let se = (e as f32).powf(-0.5);
        let mkf = |n: usize, s: f32, r: &mut dyn FnMut() -> f32| gpu.upload(&(0..n).map(|_| r()*s).collect::<Vec<f32>>());
        let mkh = |n: usize, s: f32, r: &mut dyn FnMut() -> f32| gpu.upload_f16(&(0..n).map(|_| r()*s).collect::<Vec<f32>>());
        let a = |n: usize| gpu.alloc(n);
        let zero = |b: &MBuf| unsafe { std::ptr::write_bytes(b.buf.contents() as *mut u8, 0, b.len*4) };
        let c8 = |n: usize| { let x = (gpu.alloc(n.div_ceil(2)), gpu.alloc(n.div_ceil(4)), gpu.alloc(n.div_ceil(256))); zero(&x.0); zero(&x.1); zero(&x.2); x };
        let f2 = |n: usize| { let x = (gpu.alloc(n), gpu.alloc(n)); zero(&x.0); zero(&x.1); x };
        let (orm, orv) = f2(ne*d);
        // weight init in a fixed order (wr, wg, wu, wd, sg, su, sd) so `new` keeps the same
        // RNG stream across versions; `new_empty` zero-fills instead.
        let (wr, wg, wu, wd, sg, su, sd) = if init {
            (mkf(ne*d, 0.3, &mut nxt),
             mkh(ne*e*d, sc, &mut nxt), mkh(ne*e*d, sc, &mut nxt), mkh(ne*d*e, se, &mut nxt),   // contiguous (same RNG order as ne separate)
             mkh(e*d, sc, &mut nxt), mkh(e*d, sc, &mut nxt), mkh(d*e, se, &mut nxt))
        } else {
            let zf = |n: usize| { let b = gpu.alloc(n); zero(&b); b };
            (zf(ne*d),
             gpu.alloc_f16_zeroed(ne*e*d), gpu.alloc_f16_zeroed(ne*e*d), gpu.alloc_f16_zeroed(ne*d*e),
             gpu.alloc_f16_zeroed(e*d), gpu.alloc_f16_zeroed(e*d), gpu.alloc_f16_zeroed(d*e))
        };
        MoeFfn {
            d, e, ne, k, t_max,
            dev: gpu.device.clone(),
            orm, orv,
            owg: c8(ne*e*d), owu: c8(ne*e*d), owd: c8(ne*d*e),
            osg: c8(e*d), osu: c8(e*d), osd: c8(d*e),
            wr, wg, wu, wd, sg, su, sd,
            logits: a(t_max*ne), gates: a(t_max*ne), topk: a(t_max*k),
            gl: (0..ne).map(|_| a(t_max*e)).collect(), ul: (0..ne).map(|_| a(t_max*e)).collect(),
            oe: (0..ne).map(|_| a(t_max*d)).collect(), gls: a(t_max*e), uls: a(t_max*e), act: a(t_max*e),
            dact: a(t_max*e), dgl: a(t_max*e), dul: a(t_max*e), dtmp: a(t_max*d), dgate: a(t_max*ne), dlog: a(t_max*ne),
            gwr: a(ne*d), gwg: a(ne*e*d), gwu: a(ne*e*d), gwd: a(ne*d*e), gsg: a(e*d), gsu: a(e*d), gsd: a(d*e),
            muon: OnceCell::new(),
            gs: OnceCell::new(),
            route_cache: RefCell::new(None),
        }
    }

    fn muon_scratch(&self) -> &MuonScratch {
        self.muon.get_or_init(|| {
            let (d, e, ne) = (self.d, self.e, self.ne);
            let a = |n: usize| draw(&self.dev, n);
            let z = |n: usize| dzero(&self.dev, n);
            MuonScratch {
                mwg: z(ne*e*d), mwu: z(ne*e*d), mwd: z(ne*d*e),
                msg: z(e*d), msu: z(e*d), msd: z(d*e),
                nsx: a(e*d), nst: a(e*d), nsa: a(e.max(d)*e.max(d)), nsaa: a(e.max(d)*e.max(d)), nsb: a(e.max(d)*e.max(d)), nsbx: a(e*d), nrm: a(1),
            }
        })
    }

    fn gs_scratch(&self) -> &GsScratch {
        self.gs.get_or_init(|| {
            let (d, e, ne, k, t_max) = (self.d, self.e, self.ne, self.k, self.t_max);
            let a = |n: usize| draw(&self.dev, n);
            GsScratch {
                gidx: a(t_max*k), tok2: a(t_max*k), gg: a(t_max*k), rexp: a(t_max*k),
                gath: a(t_max*k*d), gl: a(t_max*k*e), ul: a(t_max*k*e), act: a(t_max*k*e), eo: a(t_max*k*d), shared: a(t_max*d),
                deo: a(t_max*k*d), dgr: a(t_max*k), dgath: a(t_max*k*d), dact: a(t_max*k*e), dgl: a(t_max*k*e), dul: a(t_max*k*e), dgd: a(t_max*ne),
                cnt: a(ne), off: a(ne + 1), fill: a(ne),
                texp: a((t_max*k).div_ceil(32) + ne + 1), trow0: a((t_max*k).div_ceil(32) + ne + 1), tmend: a((t_max*k).div_ceil(32) + ne + 1),
            }
        })
    }

    /// Fully encoder-based top-k forward: no CPU readbacks, one launch per grouped expert GEMM
    /// (t_mm_grp_xwT over a GPU-built tile map). Same math as forward_gs/forward, and safe to
    /// encode inside any open encoder (for instance the trainer's fused per-8-layer forward).
    /// Requires e % 64 == 0 and d % 64 == 0.
    pub fn forward_gs_enc(&self, kit: &Kit, enc: &metal::ComputeCommandEncoderRef, h2: &MBuf, out: &MBuf, t: usize) {
        let gs = self.gs_scratch();
        let (du, eu, neu, tu, ku) = (self.d as u32, self.e as u32, self.ne as u32, t as u32, self.k as u32);
        let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
        let el = |n: usize| g1(n.div_ceil(256));
        let gf = |o: usize, m: usize| g2(m.div_ceil(32), o.div_ceil(64));
        let nr = t * self.k;
        let nmt = nr.div_ceil(32) + self.ne + 1;
        // router + gate + routing + tile map (all device-side)
        kit.d(enc, "t_gemm_xwT", &[(h2, 0), (&self.wr, 0), (&self.logits, 0)], &[du, neu, tu], g2(self.ne.div_ceil(8), t.div_ceil(16)), tg256);
        kit.d(enc, "t_moe_gate_fwd", &[(&self.logits, 0), (&self.gates, 0), (&self.topk, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
        self.encode_route(kit, enc, t);
        kit.d(enc, "t_moe_tilemap", &[(&gs.off, 0), (&gs.texp, 0), (&gs.trow0, 0), (&gs.tmend, 0)], &[neu, nmt as u32], g1(1), tg256);
        // shared expert over all T
        kit.d(enc, "t_mm_xwT_h", &[(h2, 0), (&self.sg, 0), (&self.gls, 0)], &[du, eu, tu], gf(self.e, t), tg128);
        kit.d(enc, "t_mm_xwT_h", &[(h2, 0), (&self.su, 0), (&self.uls, 0)], &[du, eu, tu], gf(self.e, t), tg128);
        kit.d(enc, "t_swiglu_fwd", &[(&self.gls, 0), (&self.uls, 0), (&self.act, 0)], &[(t*self.e) as u32], el(t*self.e), tg256);
        kit.d(enc, "t_mm_xwT_h", &[(&self.act, 0), (&self.sd, 0), (&gs.shared, 0)], &[eu, du, tu], gf(self.d, t), tg128);
        // gather + grouped expert GEMMs: one launch each for gate/up/down
        kit.d(enc, "t_moe_gather", &[(&gs.gath, 0), (h2, 0), (&gs.gidx, 0)], &[du, (nr*self.d) as u32], el(nr*self.d), tg256);
        kit.d(enc, "t_mm_grp_xwT", &[(&gs.gath, 0), (&self.wg, 0), (&gs.gl, 0), (&gs.texp, 0), (&gs.trow0, 0), (&gs.tmend, 0)],
              &[du, eu], g2(nmt, self.e.div_ceil(64)), tg128);
        kit.d(enc, "t_mm_grp_xwT", &[(&gs.gath, 0), (&self.wu, 0), (&gs.ul, 0), (&gs.texp, 0), (&gs.trow0, 0), (&gs.tmend, 0)],
              &[du, eu], g2(nmt, self.e.div_ceil(64)), tg128);
        kit.d(enc, "t_swiglu_fwd", &[(&gs.gl, 0), (&gs.ul, 0), (&gs.act, 0)], &[(nr*self.e) as u32], el(nr*self.e), tg256);
        kit.d(enc, "t_mm_grp_xwT", &[(&gs.act, 0), (&self.wd, 0), (&gs.eo, 0), (&gs.texp, 0), (&gs.trow0, 0), (&gs.tmend, 0)],
              &[eu, du], g2(nmt, self.d.div_ceil(64)), tg128);
        kit.d(enc, "t_moe_scatter_k", &[(out, 0), (&gs.shared, 0), (&gs.eo, 0), (&gs.tok2, 0), (&gs.gg, 0)], &[du, ku, (t*self.d) as u32], el(t*self.d), tg256);
    }

    /// Encode GPU routing (t_moe_route_count/offset/scatter) from self.topk / self.gates into the
    /// gs scratch maps (gidx/tok2/gg/rexp + cnt/off). Must be encoded after t_moe_gate_fwd in the
    /// same (or a completed) command buffer. Intra-expert positions are atomic-fill order, which
    /// may differ from CPU token order; gidx/tok2/gg/rexp stay mutually consistent either way.
    fn encode_route(&self, kit: &Kit, enc: &metal::ComputeCommandEncoderRef, t: usize) {
        let gs = self.gs_scratch();
        let (ne, k) = (self.ne, self.k);
        let na = t * k;
        let tg256 = MTLSize::new(256, 1, 1);
        // t_fill writes 0.0f (bit pattern 0) — zeroes the u32 count/fill arrays
        kit.d(enc, "t_fill", &[(&gs.cnt, 0)], &[ne as u32], g1(ne.div_ceil(256)), tg256);
        kit.d(enc, "t_fill", &[(&gs.fill, 0)], &[ne as u32], g1(ne.div_ceil(256)), tg256);
        kit.d(enc, "t_moe_route_count", &[(&gs.cnt, 0), (&self.topk, 0)], &[na as u32], g1(na.div_ceil(256)), tg256);
        kit.d(enc, "t_moe_route_offset", &[(&gs.cnt, 0), (&gs.off, 0)], &[ne as u32], g1(1), tg256);
        kit.d(enc, "t_moe_route_scatter",
              &[(&self.topk, 0), (&self.gates, 0), (&gs.off, 0), (&gs.fill, 0),
                (&gs.gidx, 0), (&gs.tok2, 0), (&gs.gg, 0), (&gs.rexp, 0)],
              &[ne as u32, k as u32, t as u32], g1(na.div_ceil(256)), tg256);
    }

    /// Read back just the per-expert count array (ne u32s) after the routing kernels completed;
    /// offset is the same prefix sum the GPU built.
    fn read_route(&self) -> (Vec<usize>, Vec<usize>) {
        let gs = self.gs_scratch();
        let ne = self.ne;
        let cnt: Vec<usize> = unsafe { std::slice::from_raw_parts(gs.cnt.buf.contents() as *const u32, ne) }
            .iter().map(|&c| c as usize).collect();
        let mut offset = vec![0usize; ne + 1];
        for e in 0..ne { offset[e + 1] = offset[e] + cnt[e]; }
        (cnt, offset)
    }

    /// Standalone GPU routing in its own command buffer, for when backward_gs runs without a
    /// cached routing from forward_gs.
    fn gs_route_gpu(&self, kit: &Kit, gpu: &MetalGpu, t: usize) -> (Vec<usize>, Vec<usize>) {
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_route(kit, enc, t);
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        self.read_route()
    }

    /// Gather/scatter forward: computes only the top-k and shared experts per token rather than
    /// all ne. Commits internally, since it reads the routing back.
    /// out[t,d] = shared(h2) + sum_topk gate*expert(h2).
    pub fn forward_gs(&self, kit: &Kit, gpu: &MetalGpu, h2: &MBuf, out: &MBuf, t: usize) {
        let gs = self.gs_scratch();
        let (du, eu, neu, tu, ku) = (self.d as u32, self.e as u32, self.ne as u32, t as u32, self.k as u32);
        let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
        let el = |n: usize| g1(n.div_ceil(256)); let gf = |o: usize, m: usize| g2(m.div_ceil(32), o.div_ceil(64));
        // router GEMM + gate + GPU routing in one command buffer; only the ne-u32 count array
        // comes back to the CPU, to size the per-expert dispatches
        { let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
          kit.d(enc, "t_gemm_xwT", &[(h2, 0), (&self.wr, 0), (&self.logits, 0)], &[du, neu, tu], g2(self.ne.div_ceil(8), t.div_ceil(16)), tg256);
          kit.d(enc, "t_moe_gate_fwd", &[(&self.logits, 0), (&self.gates, 0), (&self.topk, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
          self.encode_route(kit, enc, t);
          enc.end_encoding(); cb.commit(); cb.wait_until_completed(); }
        let (count, offset) = self.read_route();
        *self.route_cache.borrow_mut() = Some((count.clone(), offset.clone()));
        let nr = t*self.k;
        let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        // shared expert over all T
        kit.d(enc, "t_mm_xwT_h", &[(h2, 0), (&self.sg, 0), (&self.gls, 0)], &[du, eu, tu], gf(self.e, t), tg128);
        kit.d(enc, "t_mm_xwT_h", &[(h2, 0), (&self.su, 0), (&self.uls, 0)], &[du, eu, tu], gf(self.e, t), tg128);
        kit.d(enc, "t_swiglu_fwd", &[(&self.gls, 0), (&self.uls, 0), (&self.act, 0)], &[(t*self.e) as u32], el(t*self.e), tg256);
        kit.d(enc, "t_mm_xwT_h", &[(&self.act, 0), (&self.sd, 0), (&gs.shared, 0)], &[eu, du, tu], gf(self.d, t), tg128);
        // gather tokens into per-expert groups, per-expert SwiGLU on the slice
        kit.d(enc, "t_moe_gather", &[(&gs.gath, 0), (h2, 0), (&gs.gidx, 0)], &[du, (nr*self.d) as u32], el(nr*self.d), tg256);
        let w16 = |ex: usize| (ex*self.e*self.d*2) as u64;   // f16 weight offset for contiguous experts
        for ex in 0..self.ne { let m = count[ex]; if m == 0 { continue; }
            let (od, oe) = ((offset[ex]*self.d*4) as u64, (offset[ex]*self.e*4) as u64);
            kit.d(enc, "t_mm_xwT_h", &[(&gs.gath, od), (&self.wg, w16(ex)), (&gs.gl, oe)], &[du, eu, m as u32], gf(self.e, m), tg128);
            kit.d(enc, "t_mm_xwT_h", &[(&gs.gath, od), (&self.wu, w16(ex)), (&gs.ul, oe)], &[du, eu, m as u32], gf(self.e, m), tg128);
            kit.d(enc, "t_swiglu_fwd", &[(&gs.gl, oe), (&gs.ul, oe), (&gs.act, oe)], &[(m*self.e) as u32], el(m*self.e), tg256);
            kit.d(enc, "t_mm_xwT_h", &[(&gs.act, oe), (&self.wd, w16(ex)), (&gs.eo, od)], &[eu, du, m as u32], gf(self.d, m), tg128); }
        kit.d(enc, "t_moe_scatter_k", &[(out, 0), (&gs.shared, 0), (&gs.eo, 0), (&gs.tok2, 0), (&gs.gg, 0)], &[du, ku, (t*self.d) as u32], el(t*self.d), tg256);
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    }

    /// Gather/scatter backward: mirrors the permutation for the grad path and fills d_h2 and every
    /// weight grad (router, per-expert wg/wu/wd on gathered slices, shared). Call forward_gs first.
    pub fn backward_gs(&self, kit: &Kit, gpu: &MetalGpu, h2: &MBuf, dout: &MBuf, dh2: &MBuf, t: usize) {
        let gs = self.gs_scratch();
        let (du, eu, neu, tu, ku) = (self.d as u32, self.e as u32, self.ne as u32, t as u32, self.k as u32);
        let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
        let el = |n: usize| g1(n.div_ceil(256)); let gf = |o: usize, m: usize| g2(m.div_ceil(32), o.div_ceil(64));
        // reuse the (count, offset) and GPU maps from forward_gs's routing, since a step's
        // backward routes identically; rebuild from self.topk only if absent
        let cached = self.route_cache.borrow().clone();
        let (count, offset) = match cached {
            Some(co) => co,
            None => {
                let co = self.gs_route_gpu(kit, gpu, t);
                *self.route_cache.borrow_mut() = Some(co.clone());
                co
            }
        };
        let nr = t*self.k;
        let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
        kit.d(enc, "t_moe_gather_scaled", &[(&gs.deo, 0), (dout, 0), (&gs.gidx, 0), (&gs.gg, 0)], &[du, (nr*self.d) as u32], el(nr*self.d), tg256);
        kit.d(enc, "t_moe_dgate_gs", &[(&gs.dgr, 0), (&gs.eo, 0), (dout, 0), (&gs.gidx, 0)], &[du, nr as u32], g1(nr.div_ceil(256)), tg256);
        kit.d(enc, "t_fill", &[(dh2, 0)], &[(t*self.d) as u32], el(t*self.d), tg256);
        // shared expert backward (full T) + weight grads
        kit.d(enc, "t_mm_dx_h", &[(dout, 0), (&self.sd, 0), (&self.dact, 0)], &[eu, du, 0u32, tu], gf(self.e, t), tg128);
        kit.d(enc, "t_swiglu_fwd", &[(&self.gls, 0), (&self.uls, 0), (&self.act, 0)], &[(t*self.e) as u32], el(t*self.e), tg256);
        kit.d(enc, "t_mm_dw", &[(dout, 0), (&self.act, 0), (&self.gsd, 0)], &[eu, du, tu], g2(self.e.div_ceil(32), self.d.div_ceil(64)), tg128);
        kit.d(enc, "t_swiglu_bwd", &[(&self.dact, 0), (&self.gls, 0), (&self.uls, 0), (&self.dgl, 0), (&self.dul, 0)], &[(t*self.e) as u32], el(t*self.e), tg256);
        kit.d(enc, "t_mm_dw", &[(&self.dgl, 0), (h2, 0), (&self.gsg, 0)], &[du, eu, tu], g2(self.d.div_ceil(32), self.e.div_ceil(64)), tg128);
        kit.d(enc, "t_mm_dx_h", &[(&self.dgl, 0), (&self.sg, 0), (dh2, 0)], &[du, eu, 1u32, tu], gf(self.d, t), tg128);
        kit.d(enc, "t_mm_dw", &[(&self.dul, 0), (h2, 0), (&self.gsu, 0)], &[du, eu, tu], g2(self.d.div_ceil(32), self.e.div_ceil(64)), tg128);
        kit.d(enc, "t_mm_dx_h", &[(&self.dul, 0), (&self.su, 0), (dh2, 0)], &[du, eu, 1u32, tu], gf(self.d, t), tg128);
        // per-expert backward on gathered slices: input grads (bdgath) and weight grads
        let w16 = |ex: usize| (ex*self.e*self.d*2) as u64; let g32 = |ex: usize| (ex*self.e*self.d*4) as u64;
        for ex in 0..self.ne { let m = count[ex]; if m == 0 { continue; }
            let (od, oe) = ((offset[ex]*self.d*4) as u64, (offset[ex]*self.e*4) as u64);
            kit.d(enc, "t_mm_dx_h", &[(&gs.deo, od), (&self.wd, w16(ex)), (&gs.dact, oe)], &[eu, du, 0u32, m as u32], gf(self.e, m), tg128);
            kit.d(enc, "t_swiglu_fwd", &[(&gs.gl, oe), (&gs.ul, oe), (&gs.act, oe)], &[(m*self.e) as u32], el(m*self.e), tg256);   // recompute act
            kit.d(enc, "t_mm_dw", &[(&gs.deo, od), (&gs.act, oe), (&self.gwd, g32(ex))], &[eu, du, m as u32], g2(self.e.div_ceil(32), self.d.div_ceil(64)), tg128);
            kit.d(enc, "t_swiglu_bwd", &[(&gs.dact, oe), (&gs.gl, oe), (&gs.ul, oe), (&gs.dgl, oe), (&gs.dul, oe)], &[(m*self.e) as u32], el(m*self.e), tg256);
            kit.d(enc, "t_mm_dw", &[(&gs.dgl, oe), (&gs.gath, od), (&self.gwg, g32(ex))], &[du, eu, m as u32], g2(self.d.div_ceil(32), self.e.div_ceil(64)), tg128);
            kit.d(enc, "t_mm_dx_h", &[(&gs.dgl, oe), (&self.wg, w16(ex)), (&gs.dgath, od)], &[du, eu, 0u32, m as u32], gf(self.d, m), tg128);
            kit.d(enc, "t_mm_dw", &[(&gs.dul, oe), (&gs.gath, od), (&self.gwu, g32(ex))], &[du, eu, m as u32], g2(self.d.div_ceil(32), self.e.div_ceil(64)), tg128);
            kit.d(enc, "t_mm_dx_h", &[(&gs.dul, oe), (&self.wu, w16(ex)), (&gs.dgath, od)], &[du, eu, 1u32, m as u32], gf(self.d, m), tg128); }
        kit.d(enc, "t_moe_scatter_dh2", &[(dh2, 0), (&gs.dgath, 0), (&gs.tok2, 0)], &[du, ku, (t*self.d) as u32], el(t*self.d), tg256);
        // router grad
        kit.d(enc, "t_fill", &[(&gs.dgd, 0)], &[(t*self.ne) as u32], el(t*self.ne), tg256);
        kit.d(enc, "t_moe_scatter_dgate", &[(&gs.dgd, 0), (&gs.dgr, 0), (&gs.gidx, 0), (&gs.rexp, 0)], &[neu, nr as u32], g1(nr.div_ceil(256)), tg256);
        kit.d(enc, "t_moe_gate_bwd", &[(&gs.dgd, 0), (&self.gates, 0), (&self.topk, 0), (&self.dlog, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
        kit.d(enc, "t_gemm_dw", &[(&self.dlog, 0), (h2, 0), (&self.gwr, 0)], &[du, neu, tu], g2(self.d.div_ceil(1024), self.ne.div_ceil(8)), tg256);
        kit.d(enc, "t_gemm_dx", &[(&self.dlog, 0), (&self.wr, 0), (dh2, 0)], &[du, neu, 1u32, tu], g2(self.d.div_ceil(1024), t.div_ceil(8)), tg256);
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
    }

    fn gf(&self, o: usize, t: usize) -> MTLSize { g2(t.div_ceil(32), o.div_ceil(64)) }

    /// out[T,d] = sum_topk gate*expert(h2) + shared(h2).  h2, out are caller buffers (post-norm / ffn-out).
    pub fn forward(&self, kit: &Kit, enc: &metal::ComputeCommandEncoderRef, h2: &MBuf, out: &MBuf, t: usize) {
        let (du, eu, neu, tu, ku) = (self.d as u32, self.e as u32, self.ne as u32, t as u32, self.k as u32);
        let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
        let el = |n: usize| g1(n.div_ceil(256));
        kit.d(enc, "t_gemm_xwT", &[(h2, 0), (&self.wr, 0), (&self.logits, 0)], &[du, neu, tu], g2(self.ne.div_ceil(8), t.div_ceil(16)), tg256);
        kit.d(enc, "t_moe_gate_fwd", &[(&self.logits, 0), (&self.gates, 0), (&self.topk, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
        let swig = |wg: &MBuf, wgo: u64, wu: &MBuf, wuo: u64, wd: &MBuf, wdo: u64, gl: &MBuf, ul: &MBuf, o: &MBuf| {
            kit.d(enc, "t_mm_xwT_h", &[(h2, 0), (wg, wgo), (gl, 0)], &[du, eu, tu], self.gf(self.e, t), tg128);
            kit.d(enc, "t_mm_xwT_h", &[(h2, 0), (wu, wuo), (ul, 0)], &[du, eu, tu], self.gf(self.e, t), tg128);
            kit.d(enc, "t_swiglu_fwd", &[(gl, 0), (ul, 0), (&self.act, 0)], &[(t*self.e) as u32], el(t*self.e), tg256);
            kit.d(enc, "t_mm_xwT_h", &[(&self.act, 0), (wd, wdo), (o, 0)], &[eu, du, tu], self.gf(self.d, t), tg128);
        };
        swig(&self.sg, 0, &self.su, 0, &self.sd, 0, &self.gls, &self.uls, out);      // out = shared
        let w16 = |i: usize| (i*self.e*self.d*2) as u64;   // f16 byte offset for expert i
        for i in 0..self.ne {
            swig(&self.wg, w16(i), &self.wu, w16(i), &self.wd, w16(i), &self.gl[i], &self.ul[i], &self.oe[i]);
            kit.d(enc, "t_moe_acc", &[(out, 0), (&self.oe[i], 0), (&self.gates, 0)], &[du, neu, i as u32, (t*self.d) as u32], el(t*self.d), tg256);
        }
    }

    /// d_h2[T,d] = d(out)/d(h2). Fills gwr/gwg/gwu/gwd/gs* weight grads. Call forward() first.
    pub fn backward(&self, kit: &Kit, enc: &metal::ComputeCommandEncoderRef, h2: &MBuf, dout: &MBuf, dh2: &MBuf, t: usize) {
        let (du, eu, neu, tu, ku) = (self.d as u32, self.e as u32, self.ne as u32, t as u32, self.k as u32);
        let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
        let el = |n: usize| g1(n.div_ceil(256));
        kit.d(enc, "t_fill", &[(dh2, 0)], &[(t*self.d) as u32], el(t*self.d), tg256);
        for i in 0..self.ne { kit.d(enc, "t_moe_dgate", &[(&self.dgate, 0), (&self.oe[i], 0), (dout, 0)], &[du, neu, i as u32, tu], g1(t.div_ceil(256)), tg256); }
        // expert bwd (d_o = gate*dout) + expert weight grads
        let ebwd = |wg: &MBuf, wgo: u64, wu: &MBuf, wuo: u64, wd: &MBuf, wdo: u64, gl: &MBuf, ul: &MBuf, dob: &MBuf, gwd: &MBuf, gwdo: u64, gwg: &MBuf, gwgo: u64, gwu: &MBuf, gwuo: u64| {
            kit.d(enc, "t_mm_dx_h", &[(dob, 0), (wd, wdo), (&self.dact, 0)], &[eu, du, 0u32, tu], self.gf(self.e, t), tg128);
            kit.d(enc, "t_swiglu_fwd", &[(gl, 0), (ul, 0), (&self.act, 0)], &[(t*self.e) as u32], el(t*self.e), tg256);   // recompute act for dwd
            kit.d(enc, "t_mm_dw", &[(dob, 0), (&self.act, 0), (gwd, gwdo)], &[eu, du, tu], g2(self.e.div_ceil(32), self.d.div_ceil(64)), tg128);
            kit.d(enc, "t_swiglu_bwd", &[(&self.dact, 0), (gl, 0), (ul, 0), (&self.dgl, 0), (&self.dul, 0)], &[(t*self.e) as u32], el(t*self.e), tg256);
            kit.d(enc, "t_mm_dw", &[(&self.dgl, 0), (h2, 0), (gwg, gwgo)], &[du, eu, tu], g2(self.d.div_ceil(32), self.e.div_ceil(64)), tg128);
            kit.d(enc, "t_mm_dx_h", &[(&self.dgl, 0), (wg, wgo), (dh2, 0)], &[du, eu, 1u32, tu], self.gf(self.d, t), tg128);
            kit.d(enc, "t_mm_dw", &[(&self.dul, 0), (h2, 0), (gwu, gwuo)], &[du, eu, tu], g2(self.d.div_ceil(32), self.e.div_ceil(64)), tg128);
            kit.d(enc, "t_mm_dx_h", &[(&self.dul, 0), (wu, wuo), (dh2, 0)], &[du, eu, 1u32, tu], self.gf(self.d, t), tg128);
        };
        let w16 = |i: usize| (i*self.e*self.d*2) as u64;   // f16 weight offset
        let g32 = |i: usize| (i*self.e*self.d*4) as u64;   // f32 grad offset
        for i in 0..self.ne {
            kit.d(enc, "t_moe_rowscale", &[(&self.dtmp, 0), (dout, 0), (&self.gates, 0)], &[du, neu, i as u32, (t*self.d) as u32], el(t*self.d), tg256);
            ebwd(&self.wg, w16(i), &self.wu, w16(i), &self.wd, w16(i), &self.gl[i], &self.ul[i], &self.dtmp, &self.gwd, g32(i), &self.gwg, g32(i), &self.gwu, g32(i));
        }
        ebwd(&self.sg, 0, &self.su, 0, &self.sd, 0, &self.gls, &self.uls, dout, &self.gsd, 0, &self.gsg, 0, &self.gsu, 0);   // shared (gate 1)
        kit.d(enc, "t_moe_gate_bwd", &[(&self.dgate, 0), (&self.gates, 0), (&self.topk, 0), (&self.dlog, 0)], &[tu, neu, ku], g1(t.div_ceil(256)), tg256);
        kit.d(enc, "t_gemm_dw", &[(&self.dlog, 0), (h2, 0), (&self.gwr, 0)], &[du, neu, tu], g2(self.d.div_ceil(1024), self.ne.div_ceil(8)), tg256);
        kit.d(enc, "t_gemm_dx", &[(&self.dlog, 0), (&self.wr, 0), (dh2, 0)], &[du, neu, 1u32, tu], g2(self.d.div_ceil(1024), t.div_ceil(8)), tg256);
    }

    /// AdamW update of every MoE weight from its g* grad buffer; call after backward(). The router
    /// uses f32 AdamW, the experts and shared expert 8-bit-moment AdamW (t_adamw_8h, f16 weights).
    pub fn opt_step(&self, kit: &Kit, enc: &metal::ComputeCommandEncoderRef, lr: f32, wd: f32, step: u32) {
        let (b1, b2) = (0.9f32, 0.999f32);
        let (bc1, bc2) = (1.0/(1.0 - b1.powi(step as i32 + 1)), 1.0/(1.0 - b2.powi(step as i32 + 1)));
        let fc = [lr, b1, b2, bc1, bc2, wd];
        let tg256 = MTLSize::new(256, 1, 1);
        kit.df(enc, "t_adamw", &[(&self.wr, 0), (&self.gwr, 0), (&self.orm, 0), (&self.orv, 0)], &fc, &[(self.ne*self.d) as u32], g1((self.ne*self.d).div_ceil(256)), tg256);
        let c8 = |w: &MBuf, g: &MBuf, o: &(MBuf, MBuf, MBuf)| {
            kit.df(enc, "t_adamw_8h", &[(w, 0), (g, 0), (&o.0, 0), (&o.1, 0), (&o.2, 0)], &fc, &[w.len as u32, step], g1(w.len.div_ceil(2048)), tg256);
        };
        // contiguous experts: one AdamW per weight type covers all ne experts
        c8(&self.wg, &self.gwg, &self.owg); c8(&self.wu, &self.gwu, &self.owu); c8(&self.wd, &self.gwd, &self.owd);
        c8(&self.sg, &self.gsg, &self.osg); c8(&self.su, &self.gsu, &self.osu); c8(&self.sd, &self.gsd, &self.osd);
    }

    /// Router-only AdamW ("trigger training"): updates which experts fire (the router `wr`) while
    /// the expert and shared weights stay frozen. The backward pass still fills every grad; this
    /// optimizes `wr` alone.
    pub fn opt_step_router(&self, kit: &Kit, enc: &metal::ComputeCommandEncoderRef, lr: f32, wd: f32, step: u32) {
        let (b1, b2) = (0.9f32, 0.999f32);
        let (bc1, bc2) = (1.0/(1.0 - b1.powi(step as i32 + 1)), 1.0/(1.0 - b2.powi(step as i32 + 1)));
        let fc = [lr, b1, b2, bc1, bc2, wd];
        let tg256 = MTLSize::new(256, 1, 1);
        kit.df(enc, "t_adamw", &[(&self.wr, 0), (&self.gwr, 0), (&self.orm, 0), (&self.orv, 0)], &fc, &[(self.ne*self.d) as u32], g1((self.ne*self.d).div_ceil(256)), tg256);
    }

    /// Per-expert token counts from the last forward_gs; the mean over ne is the experts-fired
    /// sparsity signal used as a trigger-training reward term. Served from the routing cache,
    /// falling back to a CPU count over self.topk (t*k u32s, shared memory) if no forward_gs
    /// has run.
    pub fn route_counts(&self, _gpu: &MetalGpu, t: usize) -> Vec<usize> {
        if let Some((count, _)) = &*self.route_cache.borrow() {
            return count.clone();
        }
        let topk: &[u32] = unsafe { std::slice::from_raw_parts(self.topk.buf.contents() as *const u32, t * self.k) };
        let mut count = vec![0usize; self.ne];
        for &e in topk { count[e as usize] += 1; }
        count
    }

    /// Muon update for the 2D expert and shared weights (momentum -> Newton-Schulz orthogonalize
    /// -> scaled update); the router stays on AdamW. ~2x fewer steps than AdamW.
    pub fn opt_step_muon(&self, kit: &Kit, enc: &metal::ComputeCommandEncoderRef, lr: f32, wd: f32, beta: f32, step: u32) {
        let ms = self.muon_scratch();
        let tg256 = MTLSize::new(256, 1, 1);
        let (na, nb, nc) = (3.4445f32, -4.7750f32, 2.0315f32);
        let ns5 = |r: usize, cc: usize| {
            for _ in 0..5 {
                kit.d(enc, "t_gemm_xwT", &[(&ms.nsx, 0), (&ms.nsx, 0), (&ms.nsa, 0)], &[cc as u32, r as u32, r as u32], g2(r.div_ceil(8), r.div_ceil(16)), tg256);
                kit.d(enc, "t_gemm_dx", &[(&ms.nsa, 0), (&ms.nsa, 0), (&ms.nsaa, 0)], &[r as u32, r as u32, 0, r as u32], g2(r.div_ceil(1024), r.div_ceil(8)), tg256);
                kit.df(enc, "t_lincomb2", &[(&ms.nsb, 0), (&ms.nsa, 0), (&ms.nsaa, 0)], &[nb, nc], &[(r*r) as u32], g1((r*r).div_ceil(256)), tg256);
                kit.d(enc, "t_gemm_dx", &[(&ms.nsb, 0), (&ms.nsx, 0), (&ms.nsbx, 0)], &[cc as u32, r as u32, 0, r as u32], g2(cc.div_ceil(1024), r.div_ceil(8)), tg256);
                kit.df(enc, "t_lincomb2", &[(&ms.nsx, 0), (&ms.nsx, 0), (&ms.nsbx, 0)], &[na, 1.0], &[(r*cc) as u32], g1((r*cc).div_ceil(256)), tg256);
            }
        };
        let muon_one = |w: &MBuf, wo: u64, g: &MBuf, go: u64, mom: &MBuf, momo: u64, out: usize, inn: usize| {
            let n = (out*inn) as u32; let el = g1((out*inn).div_ceil(256));
            kit.df(enc, "t_lincomb2", &[(mom, momo), (mom, momo), (g, go)], &[beta, 1.0], &[n], el, tg256);      // mom = beta*mom + g
            let (r, cc);
            if out <= inn {
                kit.d(enc, "t_sumsq", &[(mom, momo), (&ms.nrm, 0)], &[n], g1(1), tg256);
                kit.d(enc, "t_scale_rnorm", &[(&ms.nsx, 0), (mom, momo), (&ms.nrm, 0)], &[n], el, tg256);
                r = out; cc = inn;
            } else {
                kit.d(enc, "t_transpose", &[(&ms.nst, 0), (mom, momo)], &[out as u32, inn as u32], el, tg256);
                kit.d(enc, "t_sumsq", &[(&ms.nst, 0), (&ms.nrm, 0)], &[n], g1(1), tg256);
                kit.d(enc, "t_scale_rnorm", &[(&ms.nsx, 0), (&ms.nst, 0), (&ms.nrm, 0)], &[n], el, tg256);
                r = inn; cc = out;
            }
            ns5(r, cc);
            let scale = lr * (out.max(inn) as f32).sqrt();
            if out <= inn {
                kit.df(enc, "t_muon_update", &[(w, wo), (&ms.nsx, 0)], &[scale], &[n], el, tg256);
            } else {
                kit.d(enc, "t_transpose", &[(&ms.nst, 0), (&ms.nsx, 0)], &[inn as u32, out as u32], el, tg256);
                kit.df(enc, "t_muon_update", &[(w, wo), (&ms.nst, 0)], &[scale], &[n], el, tg256);
            }
        };
        let w16 = |i: usize| (i*self.e*self.d*2) as u64; let g32 = |i: usize| (i*self.e*self.d*4) as u64;
        for i in 0..self.ne {
            muon_one(&self.wg, w16(i), &self.gwg, g32(i), &ms.mwg, g32(i), self.e, self.d);
            muon_one(&self.wu, w16(i), &self.gwu, g32(i), &ms.mwu, g32(i), self.e, self.d);
            muon_one(&self.wd, w16(i), &self.gwd, g32(i), &ms.mwd, g32(i), self.d, self.e);
        }
        muon_one(&self.sg, 0, &self.gsg, 0, &ms.msg, 0, self.e, self.d);
        muon_one(&self.su, 0, &self.gsu, 0, &ms.msu, 0, self.e, self.d);
        muon_one(&self.sd, 0, &self.gsd, 0, &ms.msd, 0, self.d, self.e);
        // router: AdamW
        let (b1, b2) = (0.9f32, 0.999f32);
        let (bc1, bc2) = (1.0/(1.0 - b1.powi(step as i32 + 1)), 1.0/(1.0 - b2.powi(step as i32 + 1)));
        kit.df(enc, "t_adamw", &[(&self.wr, 0), (&self.gwr, 0), (&self.orm, 0), (&self.orv, 0)], &[lr, b1, b2, bc1, bc2, wd], &[(self.ne*self.d) as u32], g1((self.ne*self.d).div_ceil(256)), tg256);
    }
}

/// Encode the dense SwiGLU FFN on GPU: out[t,d] = swiglu(h2·wgᵀ, h2·wuᵀ)·wdᵀ. wg/wu are f16
/// [n_int,d] and wd is f16 [d,n_int] (the trainer's resident layout); gl/ul/act are [t*n_int] f32
/// scratch, out is [t*d]. The same dispatches as the trainer's dense fwd_ffn, serving as the GPU
/// teacher for healing.
#[allow(clippy::too_many_arguments)]
pub fn encode_dense_ffn(kit: &Kit, enc: &metal::ComputeCommandEncoderRef,
    h2: &MBuf, wg: &MBuf, wu: &MBuf, wd: &MBuf,
    gl: &MBuf, ul: &MBuf, act: &MBuf, out: &MBuf,
    t: usize, d: usize, n_int: usize) {
    let (du, f32_, t32) = (d as u32, n_int as u32, t as u32);
    let (tg128, tg256) = (MTLSize::new(128, 1, 1), MTLSize::new(256, 1, 1));
    let el = |n: usize| g1(n.div_ceil(256));
    kit.d(enc, "t_mm_xwT_h", &[(h2, 0), (wg, 0), (gl, 0)], &[du, f32_, t32], g2(t.div_ceil(32), n_int.div_ceil(64)), tg128);
    kit.d(enc, "t_mm_xwT_h", &[(h2, 0), (wu, 0), (ul, 0)], &[du, f32_, t32], g2(t.div_ceil(32), n_int.div_ceil(64)), tg128);
    kit.d(enc, "t_swiglu_fwd", &[(gl, 0), (ul, 0), (act, 0)], &[(t * n_int) as u32], el(t * n_int), tg256);
    kit.d(enc, "t_mm_xwT_h", &[(act, 0), (wd, 0), (out, 0)], &[f32_, du, t32], g2(t.div_ceil(32), d.div_ceil(64)), tg128);
}

/// Encode `dst[i] = a*x[i] + b*y[i]` over n elements (t_lincomb2). Public so out-of-crate healers
/// can fuse the MSE loss gradient (dout = (2/n)·out − (2/n)·target) into the same command buffer
/// as forward/backward.
pub fn encode_lincomb2(kit: &Kit, enc: &metal::ComputeCommandEncoderRef, dst: &MBuf, x: &MBuf, y: &MBuf, a: f32, b: f32, n: usize) {
    kit.df(enc, "t_lincomb2", &[(dst, 0), (x, 0), (y, 0)], &[a, b], &[n as u32], g1(n.div_ceil(256)), MTLSize::new(256, 1, 1));
}

/// Encode `out[0] = Σ src[i]²` over n elements (t_sumsq, single-threadgroup reduction), for a
/// GPU-side loss readback without an extra forward.
pub fn encode_sumsq(kit: &Kit, enc: &metal::ComputeCommandEncoderRef, src: &MBuf, out: &MBuf, n: usize) {
    kit.d(enc, "t_sumsq", &[(src, 0), (out, 0)], &[n as u32], g1(1), MTLSize::new(256, 1, 1));
}


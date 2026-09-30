#![allow(clippy::too_many_arguments)]
use super::*;
use metal::MTLSize;
use std::ffi::c_void;
 // re-export

impl<'a> DecoderGpu<'a> {
    /// The M-row projection dispatch the chunk graph routes through: `y[M,n] =
    /// x[M,k] @ W^T`, choosing among the native / F32 / F16 / m==2 / MMA-tile / M-row
    /// Q4 kernels by weight representation and shape.
    ///
    /// Shared with the slot-decode graph: the M dimension of these kernels indexes
    /// `x[m*K + k]` and reads nothing else about `m`, so the M rows can be M tokens
    /// of one sequence (prefill) or one token of each of M sequences (slots).
    pub(crate) fn chunk_gemm(&self, enc: &metal::ComputeCommandEncoderRef, m: u32, accum: bool,
                             wname: &str, x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32) {
        let ints = |enc: &metal::ComputeCommandEncoderRef, vals: &[(u64, u32)]| {
            for (idx, v) in vals { enc.set_bytes(*idx, 4, v as *const u32 as *const c_void); }
        };

            self.check_shape(wname, k, n); // qwen35 chunk projections bypass gemm_named
            // Native quant: the code below indexes w4/scale4 directly, so a tensor
            // kept in its own format has to be served first. The batched native
            // kernel covers every M, including the m==2 MTP case.
            if self.wt.repr(wname) == Repr::Native {
                self.nat_batched(enc, wname, x, y, k, n, m, accum);
                return;
            }
            // F32-kept tensors (dynamic-quant GGUFs: ssm_alpha/beta): per-row f32 GEMV fallback.
            if !self.wt.w4.contains_key(wname) {
                if let Some(w) = self.wt.w32.get(wname) {
                    for row in 0..m as u64 {
                        enc.set_compute_pipeline_state(&self.p[if accum { "gemv_w32_accum" } else { "gemv_w32" }]);
                        enc.set_buffer(0, Some(x), row * (k as u64) * 4);
                        enc.set_buffer(1, Some(w), 0);
                        enc.set_buffer(2, Some(y), row * (n as u64) * 4);
                        ints(enc, &[(3, k), (4, n)]);
                        enc.dispatch_thread_groups(MTLSize::new(((n + 7)/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }
                    return;
                }
                if let Some(w) = self.wt.w16.get(wname) {
                    // Tiled f16 GEMM where the shape allows it; the per-row GEMV loop
                    // below is 4.5x slower (1022 -> 4664 tok/s on a 4000-token prefill).
                    // ssm_alpha/ssm_beta (N=16) fail n%64 and keep the loop.
                    if n % 64 == 0 && k % 32 == 0 && m >= 8 && self.gpu.native_reduce
                        && self.p.contains_key("gemm_mm_f16") {
                        self.gemm16_off(enc, x, 0, w, y, k, n, m, accum);
                        return;
                    }
                    for row in 0..m as u64 {
                        enc.set_compute_pipeline_state(&self.p[if accum { "gemv_accum" } else { "gemv_f16" }]);
                        enc.set_buffer(0, Some(x), row*k as u64*4);
                        enc.set_buffer(1, Some(w), 0);
                        enc.set_buffer(2, Some(y), row*n as u64*4);
                        ints(enc, &[(3,k),(4,n)]);
                        enc.dispatch_thread_groups(MTLSize::new(((n+7)/8) as u64,1,1),MTLSize::new(256,1,1));
                    }
                    return;
                }
                assert!(self.gemm_named(enc,wname,x,y,k,n,m,accum),"missing MTP matrix dispatch: {wname}");
                return;
            }
            // m==2 (MTP verify): compile-time two-token GEMV — weight-reuse with full
            // gemv parallelism (the MMA tile kernel is 4-5× slower at M=2).
            if m == 2 {
                enc.set_compute_pipeline_state(&self.p[if accum { "gemv_m2_q4_accum" } else { "gemv_m2_q4" }]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
                enc.set_buffer(2, Some(y), 0);
                ints(enc, &[(3, k), (4, n)]);
                enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((n + 7)/8) as u64, 1, 1), MTLSize::new(64, 1, 1));
                return;
            }
            // MMA (simdgroup-matrix) GEMM — the reference kernel_mul_mm style. Handles
            // any m ≤ 32 (token rows ≥ m are zero-padded). Needs N%64==0; the tiny
            // N=32 alpha/beta projections fall through to the scalar kernels below.
            if n % 64 == 0 && k % 32 == 0 && m >= 8 && self.gpu.native_reduce {
                enc.set_compute_pipeline_state(&self.p["gemm_mm_q4"]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
                enc.set_buffer(2, Some(y), 0);
                ints(enc, &[(3, k), (4, n), (6, accum as u32), (7, m)]);
                enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((m+31)/32) as u64, (n/64) as u64, 1), MTLSize::new(128, 1, 1));
                return;
            } else {
                enc.set_compute_pipeline_state(&self.p[if accum { "gemv_m_q4_accum" } else { "gemv_m_q4" }]);
                enc.set_buffer(0, Some(x), 0);
                enc.set_buffer(1, Some(&self.wt.w4[wname]), 0);
                enc.set_buffer(2, Some(y), 0);
                ints(enc, &[(3, k), (4, n), (6, m)]);
                enc.set_buffer(5, Some(&self.wt.scale4[wname]), 0);
            }
            enc.dispatch_thread_groups(MTLSize::new(((n + 7)/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Row-wise RMSNorm over M rows of `st.x` into `st.h`. Shared with the slot graph.
    pub(crate) fn rmsnorm_rows(&self, enc: &metal::ComputeCommandEncoderRef, m: u32, w: &metal::Buffer) {
        let d = self.d as u32;
        let ints = |enc: &metal::ComputeCommandEncoderRef, vals: &[(u64, u32)]| {
            for (idx, v) in vals { enc.set_bytes(*idx, 4, v as *const u32 as *const c_void); }
        };

            enc.set_compute_pipeline_state(&self.p["rmsnorm_m"]);
            enc.set_buffer(0, Some(&self.st.x), 0);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(&self.st.h), 0);
            ints(enc, &[(3, d)]);
            enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// One qwen35 prefill chunk (M ≤ MAXM tokens), split across command buffers every
    /// `prefill_cb_layers` layers.
    /// Mirrors the qwen35 branch of encode_forward with M-token batched kernels;
    /// activations are [M, dim] row-major throughout.
    pub(crate) fn forward_chunk(&self, tokens: &[u32], base_pos: usize, verify: bool) {
        self.forward_chunk_enc(None, tokens, base_pos, verify, false);
    }

    /// Core chunk encoder. `ext_enc`: encode into an existing encoder/cb (the
    /// draft+verify fused step); `row1_gpu`: row 1 embeds the token id from tmp[2]
    /// (the draft's argmax) instead of tokens[1], with no CPU round-trip.
    ///
    /// Embeds `tokens` itself. The `do_embed` form below is the same encoder with the
    /// embedding gather skipped; this wrapper keeps the existing call sites
    /// (`forward_chunk`, `spec.rs`'s fused draft+verify) on the signature they
    /// already pass.
    pub(crate) fn forward_chunk_enc(&self, ext: Option<&metal::ComputeCommandEncoderRef>, tokens: &[u32], base_pos: usize, verify: bool, row1_gpu: bool) {
        self.forward_chunk_enc_embed(ext, tokens, base_pos, verify, row1_gpu, true, None)
    }

    /// `forward_chunk_enc` plus `do_embed`, the sixth flag.
    ///
    /// `do_embed = false` means `self.st.x` already holds the chunk's `[M, d]`
    /// row-major residual rows, written by the host — legal because `st.x` is
    /// `StorageModeShared` (`load.rs`), so a host store is visible to the next
    /// command buffer. That is the seam a vision encoder's output enters through
    /// (`prefill_embeds`); `tokens` is then only the placeholder ids, kept for the
    /// bounds assert and the session bookkeeping, never gathered. Everything
    /// downstream of the gather is byte-for-byte the same graph, so injected rows and
    /// gathered rows are indistinguishable to the 32 layers that follow.
    ///
    /// `pos3` is the sectioned-M-RoPE argument: `Some` carries one `(t,h,w,e)`
    /// coordinate per row of this chunk, already sliced by the caller (indexed by
    /// `m`, not by `base_pos + m`). `None`, or an architecture that declares no rope
    /// sections, leaves the rope dispatch on the scalar `base_pos + m`. See
    /// [`ojas_core::Model::prefill_embeds`] for the coordinate convention.
    pub(crate) fn forward_chunk_enc_embed(&self, ext: Option<&metal::ComputeCommandEncoderRef>, tokens: &[u32], base_pos: usize, verify: bool, row1_gpu: bool, do_embed: bool, pos3: Option<&[[u32; 4]]>) {
        debug_assert!(do_embed || !row1_gpu, "row1_gpu embeds row 1 — it needs do_embed");
        assert!(!tokens.is_empty() && tokens.len() <= MAXM
            && base_pos.checked_add(tokens.len()).is_some_and(|n| n <= self.st.max_seq)
            && tokens.iter().all(|&t| (t as usize) < self.arch.vocab), "chunk exceeds model bounds");
        let sc = self.arch.ssm.unwrap();
        let m = tokens.len() as u32;
        let d = self.d as u32;
        // Which sequence slot this chunk prefills into. The whole graph binds its
        // recurrent state and KV rows at these offsets; they are 0 for slot 0.
        let (kv_o, conv_o, ssm_o) = (|l: usize| self.kv_off(l), |l: usize| self.conv_off(l), |l: usize| self.ssm_off(l));
        // The MTP rollback snapshots are sized by `snapshot_rows`, not by slots — they
        // belong to the speculative protocol, which runs on one sequence. Prefilling a
        // verify chunk into a non-zero slot would write another slot's snapshot.
        debug_assert!(!verify || self.cur_slot.get() == 0,
            "MTP verify snapshots have no slot dimension — prefill slot 0 or disable MTP");
        let f4 = std::mem::size_of::<f32>() as u64;
        // ---- sectioned M-RoPE ---------------------------------------------------
        // Two conditions, both necessary: the caller has to have coordinates, and the
        // architecture has to declare the section split that says what to do with
        // them. surya-2/qwen35 declares [11,11,10,0] against n_rot 64 and is IMROPE
        // (interleaved t h w t h w …, llama.cpp LLM_ARCH_QWEN35 ->
        // LLAMA_ROPE_TYPE_IMROPE); the contiguous layout would be silently wrong and
        // would still pass a t==h==w degeneracy test, so the mode is named rather
        // than assumed. With no sections declared the coordinates are ignored.
        let mrope = match pos3 {
            Some(p3) => {
                assert_eq!(p3.len(), tokens.len(), "pos3 must carry one (t,h,w,e) per chunk row");
                sc.mrope_sections.iter().any(|&s| s != 0)
            }
            None => false,
        };
        let neox_arg = if mrope {
            // The mode rides in the upper bits of `neox` (see the kernel comment in
            // ops.rs): a fresh scalar slot would be read from a stale argument-table
            // entry at every call site that never enables M-RoPE.
            ojas_metal::kernels::ops::rope_mode(true, ojas_metal::kernels::ops::MROPE_INTERLEAVED)
        } else {
            ojas_metal::kernels::ops::rope_mode(true, ojas_metal::kernels::ops::MROPE_OFF)  // == 1, what every call site passes
        };
        if mrope {
            // [s0,s1,s2,s3] then (t,h,w,e) per row. st.mpos is StorageModeShared and
            // sized 4 + 4*MAXM u32 (load.rs), so this host store is visible to the
            // command buffer encoded below, the same seam st.x uses. It is
            // preallocated rather than made per dispatch because an external encoder's
            // command buffer is committed elsewhere, and
            // set_unretained_command_buffers(true) would not keep a temporary alive.
            let desc = ojas_metal::kernels::ops::mrope_desc(sc.mrope_sections, pos3.unwrap());
            debug_assert!(desc.len() * 4 <= self.st.mpos.length() as usize, "mpos descriptor overruns its buffer");
            unsafe { std::ptr::copy_nonoverlapping(desc.as_ptr(), self.st.mpos.contents() as *mut u32, desc.len()); }
        }
        // concurrent dispatch (reference-style): independent kernels within a stage
        // overlap; bar() marks the real data dependencies (this path is qwen35-only,
        // so bar() is always active here).
        // Without an external encoder the chunk owns its command buffers, split by
        // layer (`pass.rs`).
        let mut pass = ext.is_none().then(|| super::pass::SplitPass::new(self.gpu, true, self.cfg.prefill_cb_layers, m));
        let mut enc = match ext {
            Some(e) => e.to_owned(),
            None => pass.as_mut().unwrap().open(),
        };
        let ints = |enc: &metal::ComputeCommandEncoderRef, vals: &[(u64, u32)]| {
            for (idx, v) in vals { enc.set_bytes(*idx, 4, v as *const u32 as *const c_void); }
        };
        // embed each token into its x row (per-token kernel + row byte-offset).
        // Skipped entirely when the caller supplied the rows: x already holds them.
        for (i, &t) in tokens.iter().enumerate() {
            if !do_embed { break; }
            if row1_gpu && i == 1 {
                // draft-chained verify: token id comes from tmp[2] on the GPU
                enc.set_compute_pipeline_state(&self.p["embed_q4_id"]);
                enc.set_buffer(0, Some(&self.wt.w4["token_embd.weight"]), 0);
                enc.set_buffer(1, Some(&self.st.x), (i as u64) * (d as u64) * f4);
                ints(&enc, &[(2, d), (5, 0)]);
                enc.set_buffer(3, Some(&self.sp.mtp_tok), 0);
                enc.set_buffer(4, Some(&self.wt.scale4["token_embd.weight"]), 0);
                enc.dispatch_thread_groups(MTLSize::new(((d + 63)/64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                continue;
            }
            self.embed_named_off(&enc, "token_embd.weight", t, d, (i as u64)*(d as u64)*f4);
        }
        self.bar(&enc); // x (embeddings) ready
        // The M-row projection and norm dispatches live as methods (`chunk_gemm`,
        // `rmsnorm_rows`) so the slot-decode graph reuses the same routing rather than
        // growing a second copy that drifts. These closures only bind `m`.
        let gemm = |enc: &metal::ComputeCommandEncoderRef, accum: bool, wname: &str, x: &metal::Buffer, y: &metal::Buffer, k: u32, n: u32| {
            self.chunk_gemm(enc, m, accum, wname, x, y, k, n)
        };
        let rmsnorm_m = |enc: &metal::ComputeCommandEncoderRef, w: &metal::Buffer| {
            self.rmsnorm_rows(enc, m, w)
        };
        for l in 0..self.arch.n_layers {
            if let Some(pass) = pass.as_mut() { pass.layer(l, &mut enc); }
            let p = |s: &str| format!("blk.{l}.{s}");
            let lp = self.arch.layers[l];
            rmsnorm_m(&enc, &self.wt.w32[&p("attn_norm.weight")]);
            self.bar(&enc); // h ready for the projections
            if lp.is_ssm {
                let (s_st, hk, hv) = (sc.d_state, sc.n_group, sc.dt_rank);
                let d_inner = sc.d_inner; let conv_ch = d_inner + 2*hk*s_st; let conv_k = sc.conv_kernel;
                let head_v = d_inner / hv;
                gemm(&enc, false, &p("attn_qkv.weight"), &self.st.h, &self.st.ssm_qkv, d, conv_ch);
                gemm(&enc, false, &p("attn_gate.weight"), &self.st.h, &self.st.ssm_z, d, d_inner);
                gemm(&enc, false, &p("ssm_alpha.weight"), &self.st.h, &self.st.ssm_gate, d, hv);
                gemm(&enc, false, &p("ssm_beta.weight"), &self.st.h, &self.st.ssm_beta, d, hv);
                self.bar(&enc); // qkv/z/alpha/beta projections done (ran concurrently)
                self.enc_reduce(&enc, "ssm_ab", &[(&self.st.ssm_gate, 0), (&self.st.ssm_beta, 1), (&self.wt.w32[&p("ssm_dt.bias")], 2), (&self.wt.w32[&p("ssm_a")], 3)], &[(4, m*hv), (5, hv)], &[], ((m*hv + 63)/64) as u64, 64);
                {
                    enc.set_compute_pipeline_state(&self.p["conv1d_prefill"]);
                    enc.set_buffer(0, Some(&self.st.ssm_qkv), 0);
                    enc.set_buffer(1, Some(&self.st.conv_state[l]), conv_o(l));
                    enc.set_buffer(2, Some(&self.wt.w32[&p("ssm_conv1d.weight")]), 0);
                    ints(&enc, &[(3, conv_ch), (4, conv_k), (5, m)]);
                    enc.set_buffer(6, Some(if verify { &self.sp.conv_snap[l] } else { &self.st.conv_state[l] }),
                                   if verify { 0 } else { conv_o(l) });
                    ints(&enc, &[(7, if verify { 0 } else { u32::MAX })]);
                    enc.dispatch_thread_groups(MTLSize::new(((conv_ch + 63)/64) as u64, 1, 1), MTLSize::new(64, 1, 1));
                }
                self.bar(&enc); // ssm_ab + conv done (ran concurrently)
                enc.set_compute_pipeline_state(&self.p["deltanet_fused"]);
                enc.set_buffer(0, Some(&self.st.ssm_state[l]), ssm_o(l));
                enc.set_buffer(1, Some(&self.st.ssm_qkv), 0);
                enc.set_buffer(2, Some(&self.st.ssm_gate), 0);
                enc.set_buffer(3, Some(&self.st.ssm_beta), 0);
                enc.set_buffer(4, Some(&self.st.ssm_o), 0);
                ints(&enc, &[(5, s_st), (6, hk), (7, hv), (8, conv_ch), (9, m)]);
                enc.set_bytes(10, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.set_buffer(11, Some(if verify { &self.sp.ssm_snap[l] } else { &self.st.ssm_state[l] }),
                                if verify { 0 } else { ssm_o(l) });
                // OJAS_KMAP_DIV=1 selects the grouped value->key head mapping.
                ints(&enc, &[(12, if verify { 0 } else { u32::MAX }),
                             (13, self.cfg.moe_kmap_div as u32), (14, 0)]);
                enc.dispatch_thread_groups(MTLSize::new((s_st/4) as u64, hv as u64, 1), MTLSize::new(128, 1, 1));
                self.bar(&enc); // deltanet done
                enc.set_compute_pipeline_state(&self.p["gated_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.ssm_o), 0);
                enc.set_buffer(1, Some(&self.wt.w32[&p("ssm_norm.weight")]), 0);
                enc.set_buffer(2, Some(&self.st.ssm_z), 0);
                ints(&enc, &[(3, head_v), (5, d_inner), (6, 0)]);   // qwen35: silu gate
                enc.set_bytes(4, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(hv as u64, m as u64, 1), MTLSize::new(32, 1, 1));
                self.bar(&enc); // gated norm done
                gemm(&enc, true, &p("ssm_out.weight"), &self.st.ssm_o, &self.st.x, d_inner, d);
            } else {
                let (hd, kvdim, qdim) = (lp.head_dim, lp.kvdim, lp.qdim);
                let group = lp.n_head / lp.n_kv.max(1);
                gemm(&enc, false, &p("attn_q.weight"), &self.st.h, &self.st.ssm_qkv, d, 2*qdim);
                gemm(&enc, false, &p("attn_k.weight"), &self.st.h, &self.st.k, d, kvdim);
                gemm(&enc, false, &p("attn_v.weight"), &self.st.h, &self.st.v, d, kvdim);
                self.bar(&enc); // q/k/v projections done (ran concurrently)
                self.enc_reduce(&enc, "qgate_split", &[(&self.st.ssm_qkv, 0), (&self.st.q, 1)], &[(2, hd), (3, qdim), (4, m)], &[], ((m*qdim + 63)/64) as u64, 64);
                let (nq, nk) = (lp.n_head, lp.n_kv);
                self.bar(&enc); // q split done
                enc.set_compute_pipeline_state(&self.p["qk_rmsnorm"]);
                enc.set_buffer(0, Some(&self.st.q), 0);
                enc.set_buffer(1, Some(&self.st.k), 0);
                enc.set_buffer(2, Some(&self.wt.w32[&p("attn_q_norm.weight")]), 0);
                enc.set_buffer(3, Some(&self.wt.w32[&p("attn_k_norm.weight")]), 0);
                ints(&enc, &[(4, hd), (5, nq), (6, nk)]);
                enc.set_bytes(7, 4, &self.arch.eps as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new((nq + nk) as u64, m as u64, 1), MTLSize::new(32, 1, 1));
                self.bar(&enc); // qk-norm done
                let (aq, ak) = (qdim/2, kvdim/2);
                // Buffer 14 is `st.mpos` whether or not M-RoPE is on: at mode 0 the
                // kernel never dereferences it, but Metal API Validation asserts on a
                // declared-and-unbound buffer regardless, so the real descriptor
                // buffer is bound rather than a dummy.
                //
                // `rope_qk_store_m` writes the cache at `(base_pos + m) * kvdim`
                // relative to the bound base, so a sequence slot is a buffer offset
                // and the kernel needs no slot argument of its own.
                self.enc_reduce_off(&enc, "rope_qk_store_m",
                    &[(&self.st.q, 0, 0), (&self.st.k, 1, 0), (&self.st.v, 2, 0),
                      (&self.st.kcache[l], 3, kv_o(l)), (&self.st.vcache[l], 4, kv_o(l)), (&self.st.mpos, 14, 0)],
                    &[(5, hd), (6, base_pos as u32), (8, aq), (9, ak), (10, kvdim), (11, m), (12, neox_arg), (13, sc.n_rot)], &[(7, lp.rope_base)],
                    ((m*(aq + ak + kvdim) + 63)/64) as u64, 64);
                self.bar(&enc); // rope + cache store done
                if self.arch.sparse_budget.is_some() {
                    // keep page min/max metadata current for the pages this chunk touched
                    let pg0 = (base_pos / ojas_metal::kernels::attn::PAGE) as u32;
                    let npg = ((base_pos + m as usize + ojas_metal::kernels::attn::PAGE - 1) / ojas_metal::kernels::attn::PAGE) as u32 - pg0;
                    self.enc_reduce_off(&enc, "page_minmax",
                        &[(&self.st.kcache[l], 0, kv_o(l)), (&self.st.pmeta[l], 1, 0)],
                        &[(2, kvdim), (3, pg0), (4, (base_pos + m as usize) as u32)], &[],
                        npg as u64, 256);
                    self.bar(&enc); // metadata current before attention reads scores
                }
                if base_pos + tokens.len() <= 512 || (hd <= 256 && hd % 64 != 0) {
                    self.enc_reduce_off(&enc, "attention_m_short",
                        &[(&self.st.q, 0, 0), (&self.st.kcache[l], 1, kv_o(l)), (&self.st.vcache[l], 2, kv_o(l)), (&self.st.attn, 3, 0)],
                        &[(4, hd), (5, kvdim), (6, base_pos as u32), (7, group), (9, lp.n_head)], &[(8, lp.scale)],
                        (m * lp.n_head) as u64, 256);
                } else if hd > 256 || !self.gpu.native_reduce {
                    self.enc_reduce_off(&enc, "attention_m",
                        &[(&self.st.q, 0, 0), (&self.st.kcache[l], 1, kv_o(l)), (&self.st.vcache[l], 2, kv_o(l)), (&self.st.attn, 3, 0)],
                        &[(4, hd), (5, kvdim), (6, base_pos as u32), (7, group), (9, lp.n_head)], &[(8, lp.scale)],
                        (m * lp.n_head) as u64, 256);
                } else {
                    // MMA flash-attention: 32 queries/tg, simdgroup-matrix Q·K^T and P·V
                    // (hd-specialized pipeline when available — loops fully unrolled)
                    let kname = if ojas_metal::kernels::attn::ATTN_HD_SPECIAL.contains(&hd) {
                        format!("attention_m_mma_{hd}")
                    } else { "attention_m_mma".to_string() };
                    enc.set_compute_pipeline_state(&self.p[&kname]);
                    enc.set_buffer(0, Some(&self.st.q), 0);
                    enc.set_buffer(1, Some(&self.st.kcache[l]), kv_o(l));
                    enc.set_buffer(2, Some(&self.st.vcache[l]), kv_o(l));
                    enc.set_buffer(3, Some(&self.st.attn), 0);
                    ints(&enc, &[(4, hd), (5, kvdim), (6, base_pos as u32), (7, group), (9, lp.n_head), (10, m)]);
                    enc.set_bytes(8, 4, &lp.scale as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(lp.n_head as u64, ((m + 31)/32) as u64, 1), MTLSize::new(256, 1, 1));
                }
                self.bar(&enc); // attention done
                self.enc_reduce(&enc, "gate_mul_sigmoid", &[(&self.st.attn, 0), (&self.st.ssm_qkv, 1)], &[(2, hd), (3, qdim), (4, m)], &[], ((m*qdim + 63)/64) as u64, 64);
                self.bar(&enc); // gate applied
                gemm(&enc, true, &p("attn_output.weight"), &self.st.attn, &self.st.x, qdim, d);
            }
            // FFN: post_attention_norm pre-norm, SwiGLU via two GEMMs + silu_mul
            self.bar(&enc); // mixer residual in x
            rmsnorm_m(&enc, &self.wt.w32[&p("post_attention_norm.weight")]);
            self.bar(&enc); // h ready for FFN
            if let Some(mo) = self.arch.moe {
                // Batched MoE FFN (the reference mul_mm_id shape): every row's router,
                // expert SwiGLU and down-proj is one grid-wide dispatch, so
                // threadgroups hitting the same expert run concurrently and its weight
                // stream is read once through the cache instead of once per token.
                // 4 barriers per layer per chunk, not 4 per row.
                let (ne, nu, fe, fs) = (mo.n_expert, mo.n_used, mo.ffn_exp, mo.ffn_shexp);
                self.check_moe(&p, d, fe);
                let ib = |enc: &metal::ComputeCommandEncoderRef, idx: u64, v: u32| {
                    enc.set_bytes(idx, 4, &v as *const u32 as *const c_void);
                };
                // stage 1 (concurrent): router logits + shexp scalar gates + shexp gate/up
                enc.set_compute_pipeline_state(&self.p["gemv_w32_m"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w32[&p("ffn_gate_inp.weight")]), 0);
                enc.set_buffer(2, Some(&self.ms.moe_blg), 0);
                ib(&enc, 3, d); ib(&enc, 4, ne);
                enc.dispatch_thread_groups(MTLSize::new(((ne + 7)/8) as u64, m as u64, 1), MTLSize::new(256, 1, 1));
                enc.set_compute_pipeline_state(&self.p["gemv_w32_m"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w32[&p("ffn_gate_inp_shexp.weight")]), 0);
                enc.set_buffer(2, Some(&self.ms.moe_bsh), 0);
                ib(&enc, 3, d); ib(&enc, 4, 1);
                enc.dispatch_thread_groups(MTLSize::new(1, m as u64, 1), MTLSize::new(32, 1, 1));
                gemm(&enc, false, &p("ffn_gate_shexp.weight"), &self.st.h, &self.ms.moe_bg, d, fs);
                gemm(&enc, false, &p("ffn_up_shexp.weight"), &self.st.h, &self.ms.moe_bu, d, fs);
                self.bar(&enc); // logits + shexp projections done
                // stage 2 (concurrent): per-row top-k + shexp SwiGLU activation
                enc.set_compute_pipeline_state(&self.p["moe_topk_m"]);
                enc.set_buffer(0, Some(&self.ms.moe_blg), 0);
                enc.set_buffer(1, Some(&self.ms.moe_bidx), 0);
                enc.set_buffer(2, Some(&self.ms.moe_bwgt), 0);
                ib(&enc, 3, ne); ib(&enc, 4, nu);
                enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(32, 1, 1));
                self.enc_reduce(&enc, "silu_mul", &[(&self.ms.moe_bg, 0), (&self.ms.moe_bu, 1), (&self.ms.moe_bg, 2)],
                    &[(3, m*fs)], &[], ((m*fs + 63)/64) as u64, 64);
                self.bar(&enc); // routing tables + shexp act ready
                // stage 3 (concurrent): shexp down + all rows' routed expert SwiGLUs
                gemm(&enc, false, &p("ffn_down_shexp.weight"), &self.ms.moe_bg, &self.ms.moe_btmp, fs, d);
                enc.set_compute_pipeline_state(&self.p["moe_gu_q4_m"]);
                enc.set_buffer(0, Some(&self.st.h), 0);
                enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_gate_exps.weight")]), 0);
                enc.set_buffer(2, Some(&self.wt.w4[&p("ffn_up_exps.weight")]), 0);
                enc.set_buffer(3, Some(&self.ms.moe_bact), 0);
                ib(&enc, 4, d); ib(&enc, 5, fe);
                enc.set_buffer(6, Some(&self.wt.scale4[&p("ffn_gate_exps.weight")]), 0);
                enc.set_buffer(7, Some(&self.wt.scale4[&p("ffn_up_exps.weight")]), 0);
                enc.set_buffer(8, Some(&self.ms.moe_bidx), 0);
                ib(&enc, 9, nu);
                enc.dispatch_thread_groups(MTLSize::new(((fe + 7)/8) as u64, (m*nu) as u64, 1), MTLSize::new(64, 1, 1));
                self.bar(&enc); // expert activations + shexp down done
                // stage 4: weighted expert down + shexp add into x rows
                enc.set_compute_pipeline_state(&self.p["moe_down_q4_m"]);
                enc.set_buffer(0, Some(&self.ms.moe_bact), 0);
                enc.set_buffer(1, Some(&self.wt.w4[&p("ffn_down_exps.weight")]), 0);
                enc.set_buffer(2, Some(&self.st.x), 0);
                ib(&enc, 3, fe); ib(&enc, 4, d);
                enc.set_buffer(5, Some(&self.wt.scale4[&p("ffn_down_exps.weight")]), 0);
                enc.set_buffer(6, Some(&self.ms.moe_bidx), 0);
                enc.set_buffer(7, Some(&self.ms.moe_bwgt), 0);
                ib(&enc, 8, nu);
                enc.set_buffer(9, Some(&self.ms.moe_btmp), 0);
                enc.set_buffer(10, Some(&self.ms.moe_bsh), 0);
                enc.dispatch_thread_groups(MTLSize::new(((d + 7)/8) as u64, m as u64, 1), MTLSize::new(64, 1, 1));
                self.bar(&enc); // layer output in x
            } else {
            let nffn = self.arch.ffn as u32;
            // SwiGLU tile fusion loses to this concurrent 2-GEMM + silu_mul pipeline:
            // a silu-in-up epilogue and a dual-stream gate+up MMA kernel measured
            // 320/322 against 343 tok/s, because the fusions either serialize the
            // projections or double accumulator pressure.
            gemm(&enc, false, &p("ffn_gate.weight"), &self.st.h, &self.st.gate, d, nffn);
            gemm(&enc, false, &p("ffn_up.weight"), &self.st.h, &self.st.up, d, nffn);
            self.bar(&enc); // gate + up done (ran concurrently)
            self.enc_reduce(&enc, "silu_mul", &[(&self.st.gate, 0), (&self.st.up, 1), (&self.st.act, 2)], &[(3, m*nffn)], &[], ((m*nffn + 63)/64) as u64, 64);
            self.bar(&enc); // SwiGLU act ready
            gemm(&enc, true, &p("ffn_down.weight"), &self.st.act, &self.st.x, nffn, d);
            self.bar(&enc); // layer output in x
            }
        }
        if self.sp.mtp.is_some() {
            self.bar(&enc);
            self.enc_reduce(&enc, "copy_buf", &[(&self.sp.mtp_h, 0), (&self.st.x, 1)], &[(2, m*d)], &[], ((m*d + 63)/64) as u64, 64);
        }
        if verify {
            // MTP verify tail: capture pre-output-norm hidden rows (next draft input),
            // then logits + argmax per row (tmp[row] = predicted token id).
            self.bar(&enc);
            rmsnorm_m(&enc, &self.wt.w32["output_norm.weight"]);
            self.bar(&enc);
            let lm = self.arch.lm_head.clone();
            gemm(&enc, false, &lm, &self.st.h, &self.st.logits, d, self.arch.vocab as u32);
            self.bar(&enc);
            for row in 0..m as u64 {
                enc.set_compute_pipeline_state(&self.p["argmax"]);
                enc.set_buffer(0, Some(&self.st.logits), row * (self.arch.vocab as u64) * 4);
                enc.set_buffer(1, Some(&self.st.tmp), row * 4);
                enc.set_bytes(2, 4, &(self.arch.vocab as u32) as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(self.tune.max_tg.min(1024), 1, 1));
            }
        }
        if let Some(pass) = pass {
            let cbs = pass.command_buffers();
            let gpu = pass.finish(&enc, "chunked prefill");
            self.gpu_s.set(self.gpu_s.get() + gpu);
            if self.cfg.prefill_dbg {
                tracing::trace!(target: "prefill", "chunk M={} command buffers={} gpu={:.1}ms", m, cbs, gpu * 1e3);
            }
        }
    }

}

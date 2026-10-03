//! Bidirectional (non-causal) MMA flash-attention, gated against the streaming oracle.
//!
//! A ViT tower attends every patch to every other patch, and the only kernel with
//! that shape was `attention_m_bidir` (`attn_core.rs:262`): one threadgroup per
//! (query, head), so each of the 16384x12 threadgroups streams that head's entire K
//! and V — ~824 GB and ~2 s per layer at 16384 tokens.
//! `attention_m_mma_bidir_<hd>` is `attention_m_mma` with its two causality sites
//! rewritten, amortizing one KV pass over 32 queries for 32x less traffic.
//!
//! That rewrite is string replacement over the causal kernel's source (this repo has
//! no Metal function constants — `lib.rs:110`/`:160` both pass `None` — so source
//! rewriting is the only specialization idiom). A missed site yields a kernel that
//! compiles, runs at MMA speed and returns causal numbers: lower-triangular attention
//! inside an encoder, which reads downstream as a mediocre model rather than a bug.
//! Only a numeric oracle catches it.
//!
//! So every output row is compared against `attention_m_bidir` at ragged shapes
//! (`nq`=1 tails, a T past 4096, a query count below the KV length), K/V rows past
//! `total` are poisoned with large random values so a column mask that over-reads
//! produces nonsense instead of harmless zeros, and the output buffer sits between
//! guard regions so an over-store is caught even when the numbers agree.

// The Metal device is macOS-only (`ojas-metal/src/lib.rs`); only its `kernels` source
// table builds elsewhere. These tests drive a real `MetalGpu`.
#![cfg(target_os = "macos")]

use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use ojas_core::Device;
use ojas_metal::MetalGpu;
use std::ffi::c_void;

/// ViT-like geometry: 12 heads, head_dim 64, no GQA grouping.
const NH: usize = 12;
const GROUP: usize = 1;
// Head widths: every one in `ATTN_BIDIR_HD`, the dims the MMA kernels are built for.

/// (queries, kv positions). Ragged on purpose: 1 and 3 exercise the tail store
/// path with a single valid row, 129 and 4225 leave a 1-row block after the
/// 32-query tiles, 4225 is past the `sc[4096]` cap that keeps the score-array
/// kernels out of this regime, and (33, 200) proves `total` is decoupled from the
/// query count now that it no longer derives from `base_pos + q0 + nq`.
const SHAPES: &[(usize, usize)] = &[(1, 1), (3, 3), (17, 17), (64, 64), (129, 129), (33, 200), (4225, 4225)];

/// Amplitudes chosen so the logits have std ~4 (`scale*sqrt(hd*var_q*var_k)`):
/// a flat softmax would make every row the mean of V and the row comparison
/// blind to an index mix-up. `assert_rows_distinct` enforces that property.
const QAMP: f32 = 8.0;
const KAMP: f32 = 1.5;
const VAMP: f32 = 1.0;
/// K/V rows past `total`: the block loop rounds up to 8 positions, so the kernel
/// reads up to 7 rows it must then mask away. Large and random, so an unmasked
/// read dominates the softmax for effectively every row.
const POISON: f32 = 6.0;
const KV_PAD: usize = 64;

/// f32 guard values on each side of every output buffer (256 B, so the offset stays
/// well past Metal's 4 B setBuffer alignment).
const GUARD: usize = 64;
const GUARD_FILL: f32 = 12345.0;

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn upload_f32(gpu: &MetalGpu, v: &[f32]) -> Buffer {
    gpu.device.new_buffer_with_data(v.as_ptr() as *const c_void, (v.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared)
}

fn upload_f16(gpu: &MetalGpu, v: &[f32]) -> Buffer {
    let h: Vec<half::f16> = v.iter().map(|&x| half::f16::from_f32(x)).collect();
    gpu.device.new_buffer_with_data(h.as_ptr() as *const c_void, (h.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared)
}

fn host(buf: &Buffer, n: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }
}

/// One attention dispatch. Buffer slots are the family's shared contract
/// (q, kc, vc, out; 4=hd, 5=kvdim, 6=total, 7=group, 8=scale, 9=n_head), and the
/// bidirectional kernels differ only in reading slot 6 as the whole KV length.
#[allow(clippy::too_many_arguments)]
fn run(gpu: &MetalGpu, pipe: &ComputePipelineState, q: &Buffer, k: &Buffer, v: &Buffer,
       out: &Buffer, scale: f32, ints: &[(u64, u32)], grid: (u64, u64)) {
    let cb = gpu.command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(q), 0);
    enc.set_buffer(1, Some(k), 0);
    enc.set_buffer(2, Some(v), 0);
    enc.set_buffer(3, Some(out), (GUARD * 4) as u64);
    for &(i, x) in ints { enc.set_bytes(i, 4, &x as *const u32 as *const c_void); }
    enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new(grid.0, grid.1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

/// Rows this comparison must be able to tell apart, i.e. where a token/head mix-up
/// moves the numbers. Below this a peaked softmax over a handful of positions
/// legitimately collapses neighbouring queries onto the same V row; those shapes are
/// here for the ragged tail path, not for index sensitivity.
const DISTINCT_MIN_T: usize = 16;

/// The gate is a per-row cosine, which only means something if the rows differ: a
/// uniform softmax makes every output the mean of V, and a kernel that shuffled tokens
/// or heads would still score 0.9999. Checks both neighbours — r+1 is the next head of
/// the same token, r+NH the same head of the next token.
///
/// Graded on the mean rather than the worst pair: with a peaked softmax, two adjacent
/// queries landing on the same dominant key happens, and a shuffle is caught by the
/// rows that do differ.
fn assert_rows_distinct(label: &str, o: &[f32], rows: usize, hd: usize) {
    let cos = |a: &[f32], b: &[f32]| {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for i in 0..hd {
            d += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        d / (na.sqrt() * nb.sqrt()).max(1e-30)
    };
    let row = |r: usize| &o[r * hd..(r + 1) * hd];
    let (mut sum, mut cnt) = (0f64, 0usize);
    for r in 0..rows.min(512) {
        for s in [r + 1, r + NH] {
            if s >= rows { continue; }
            sum += cos(row(r), row(s));
            cnt += 1;
        }
    }
    let mean = sum / cnt.max(1) as f64;
    println!("{label}: neighbour-row cos mean={mean:.4} over {cnt} pairs (sensitivity check)");
    assert!(mean < 0.9, "{label}: neighbouring oracle rows average cos={mean:.4} — the softmax is \
        too flat for the row gate to detect a token/head index mix-up");
}

/// Per-row cosine against the oracle, worst row reported either way (a passing
/// margin is as useful to see as a failing one).
fn compare(label: &str, want: &[f32], got: &[f32], rows: usize, hd: usize) {
    assert!(got.iter().all(|v| v.is_finite()), "{label}: non-finite output");
    let (mut worst_cos, mut worst_row) = (f64::INFINITY, 0usize);
    let (mut worst_abs, mut worst_i) = (0f64, 0usize);
    for r in 0..rows {
        let (a, b) = (&want[r * hd..(r + 1) * hd], &got[r * hd..(r + 1) * hd]);
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for i in 0..hd {
            d += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
            let ad = (a[i] - b[i]).abs() as f64;
            if ad > worst_abs { worst_abs = ad; worst_i = r * hd + i; }
        }
        let c = d / (na.sqrt() * nb.sqrt()).max(1e-30);
        if c < worst_cos { worst_cos = c; worst_row = r; }
    }
    println!("{label}: worst row cos={worst_cos:.6} at row {worst_row} (tok {} head {}), \
        max|diff|={worst_abs:.5} at {worst_i} (oracle {:.4} vs mma {:.4})",
        worst_row / NH, worst_row % NH, want[worst_i], got[worst_i]);
    assert!(worst_cos >= 0.9995,
        "{label}: row {worst_row} (tok {} head {}) cos={worst_cos:.6} < 0.9995; \
         max|diff|={worst_abs:.5} at element {worst_i} (oracle {:.4} vs mma {:.4})",
        worst_row / NH, worst_row % NH, want[worst_i], got[worst_i]);
}

fn assert_guards(label: &str, buf: &Buffer, n: usize) {
    let all = host(buf, GUARD * 2 + n);
    assert!(all[..GUARD].iter().all(|&v| v == GUARD_FILL), "{label}: stored BEFORE the output region");
    assert!(all[GUARD + n..].iter().all(|&v| v == GUARD_FILL), "{label}: stored PAST the output region");
}

#[test]
fn mma_bidir_matches_streaming_bidir_oracle() {
    let gpu = match MetalGpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("attn_bidir: no Metal device ({e}); skipping"); return; }
    };
    if !gpu.native_reduce {
        eprintln!("attn_bidir: simdgroup_matrix needs Apple7/Mac2; skipping");
        return;
    }
    let oracle_src = ojas_metal::kernels::source_of("attention_m_bidir").expect("attention_m_bidir");
    let oracle = gpu.pipeline(oracle_src, "attention_m_bidir").expect("bidir oracle pipeline");
    for &hd in ojas_metal::kernels::attn::ATTN_BIDIR_HD {
    let (hd, kvdim) = (hd as usize, (NH / GROUP) * hd as usize);
        let mma_entry = format!("attention_m_mma_bidir_{hd}");
        let dq_entry = format!("attention_m_mma_dq_bidir_{hd}");
        let build = || -> anyhow::Result<(ComputePipelineState, ComputePipelineState)> {
            Ok((gpu.pipeline(&ojas_metal::kernels::attn::attn_mma_bidir_src(hd as u32), &mma_entry)?,
                gpu.pipeline(&ojas_metal::kernels::attn::attn_mma_dq_bidir_src(hd as u32), &dq_entry)?))
        };
        let (mma, dq) = match build() {
            Ok(p) => p,
            Err(e) => { eprintln!("attn_bidir: MMA pipeline unavailable ({e}); skipping"); return; }
        };

        let scale = 1.0f32 / (hd as f32).sqrt();
        for &(m, total) in SHAPES {
            let mut seed = 0xC0FFEEu32 ^ ((total as u32) << 5) ^ (m as u32);
            let q: Vec<f32> = (0..m * NH * hd).map(|_| lcg(&mut seed) * QAMP).collect();
            let kv = |seed: &mut u32, amp: f32| -> Vec<f32> {
                (0..(total + KV_PAD) * kvdim)
                    .map(|i| lcg(seed) * if i < total * kvdim { amp } else { POISON })
                    .collect()
            };
            let kb = upload_f16(&gpu, &kv(&mut seed, KAMP));
            let vb = upload_f16(&gpu, &kv(&mut seed, VAMP));
            // f16-Q twin reads fragments straight from device for all 32 rows of every
            // tile, so the row count is padded exactly as `q_to_half` pads it.
            let padm = m.div_ceil(32) * 32;
            let mut qpad = vec![0f32; padm * NH * hd];
            qpad[..q.len()].copy_from_slice(&q);
            let qb = upload_f32(&gpu, &q);
            let qhb = upload_f16(&gpu, &qpad);

            let n = m * NH * hd;
            let guarded = || upload_f32(&gpu, &vec![GUARD_FILL; GUARD * 2 + n]);
            let (ob, mb, db) = (guarded(), guarded(), guarded());
            let base: [(u64, u32); 5] = [(4, hd as u32), (5, kvdim as u32), (6, total as u32),
                (7, GROUP as u32), (9, NH as u32)];
            let mut mma_ints = base.to_vec();
            mma_ints.push((10, m as u32));                 // mtok
            let tiles = m.div_ceil(32) as u64;
            run(&gpu, &oracle, &qb, &kb, &vb, &ob, scale, &base, ((m * NH) as u64, 1));
            run(&gpu, &mma, &qb, &kb, &vb, &mb, scale, &mma_ints, (NH as u64, tiles));
            run(&gpu, &dq, &qhb, &kb, &vb, &db, scale, &mma_ints, (NH as u64, tiles));

            let rows = m * NH;
            let want = &host(&ob, GUARD + n)[GUARD..];
            assert!(want.iter().all(|v| v.is_finite()), "oracle produced non-finite output");
            if total >= DISTINCT_MIN_T { assert_rows_distinct(&format!("hd={hd} m={m} T={total}"), want, rows, hd); }
            compare(&format!("{mma_entry} m={m} T={total}"), want, &host(&mb, GUARD + n)[GUARD..], rows, hd);
            compare(&format!("{dq_entry} m={m} T={total}"), want, &host(&db, GUARD + n)[GUARD..], rows, hd);
            for (label, buf) in [("oracle", &ob), (mma_entry.as_str(), &mb), (dq_entry.as_str(), &db)] {
                assert_guards(&format!("{label} m={m} T={total}"), buf, n);
            }
        }
    }
}

/// The rewrite must not leave a causal kernel wearing a bidirectional name, and
/// it must not collide with an existing entry point. Cheap enough to run without
/// a GPU, which is where a stale-source regression is most likely to be noticed.
#[test]
fn bidir_source_drops_every_causality_site() {
    for &hd in ojas_metal::kernels::attn::ATTN_BIDIR_HD {
        for src in [ojas_metal::kernels::attn::attn_mma_bidir_src(hd),
                    ojas_metal::kernels::attn::attn_mma_dq_bidir_src(hd)] {
            assert!(!src.contains("base_pos"), "hd={hd}: causal bound survived the rewrite");
            assert!(src.contains("uint maxseq = total;"), "hd={hd}: block-loop bound not rewritten");
            assert!(src.contains("uint myseq = total;"), "hd={hd}: row column mask not rewritten");
            assert!(src.contains(&format!("const uint hd = {hd}u;")), "hd={hd}: head dim not pinned");
            assert_eq!(src.matches("kernel void ").count(), 1, "hd={hd}: sliced more than one kernel");
        }
        for entry in [format!("attention_m_mma_bidir_{hd}"), format!("attention_m_mma_dq_bidir_{hd}")] {
            assert!(ojas_metal::kernels::source_of(&entry).is_none(),
                "{entry} collides with a family kernel name");
        }
    }
}

/// Packed variable-length batches with an optional symmetric window: the shape a
/// text-encoder request takes (several questions' sequences in one row buffer,
/// ModernBERT's local layers seeing `|i - j| <= 64`).
///
/// The oracle is f64 attention on the host over the f16-rounded K and V the kernel
/// reads, restricted to each row's own sequence and window. Neighbouring sequences are
/// real data rather than padding, so a kernel that let a query see across a sequence
/// boundary produces a plausible but wrong row, which the per-row cosine catches. K/V
/// rows past the packed total are poisoned as in the tests above.
#[test]
fn bidir_span_matches_per_sequence_oracle() {
    let gpu = match MetalGpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("attn_bidir: no Metal device ({e}); skipping"); return; }
    };
    let src = ojas_metal::kernels::attn::attn_bidir_span_src();
    let entry = ojas_metal::kernels::attn::ATTN_BIDIR_SPAN;
    let span_pipe = gpu.pipeline(&src, entry).expect("span pipeline");
    for &hd in ojas_metal::kernels::attn::ATTN_BIDIR_HD {
    let (hd, kvdim) = (hd as usize, (NH / GROUP) * hd as usize);
        let scale = 1.0f32 / (hd as f32).sqrt();
        // Lengths: a single-token sequence, lengths on both sides of the 129-key window,
        // and one long enough that most of its rows see a clipped window.
        let lens = [115usize, 1, 33, 200, 129, 64, 31, 32];
        let total: usize = lens.iter().sum();
        for window in [None, Some(64usize)] {
            let mut seed = 0x5EA1u32 ^ window.unwrap_or(0) as u32;
            let q: Vec<f32> = (0..total * NH * hd).map(|_| lcg(&mut seed) * QAMP).collect();
            let kv = |seed: &mut u32, amp: f32| -> Vec<f32> {
                (0..(total + KV_PAD) * kvdim)
                    .map(|i| lcg(seed) * if i < total * kvdim { amp } else { POISON })
                    .collect()
            };
            let (k, v) = (kv(&mut seed, KAMP), kv(&mut seed, VAMP));
            let round = |x: &[f32]| -> Vec<f64> { x.iter().map(|&a| half::f16::from_f32(a).to_f64()).collect() };
            let (kr, vr) = (round(&k), round(&v));

            let mut spans: Vec<[u32; 2]> = Vec::with_capacity(total);
            let mut start = 0usize;
            for &len in &lens {
                for i in start..start + len {
                    let (lo, hi) = match window {
                        None => (start, start + len),
                        Some(w) => (i.saturating_sub(w).max(start), (i + w + 1).min(start + len)),
                    };
                    spans.push([lo as u32, hi as u32]);
                }
                start += len;
            }

            let mut want = vec![0f32; total * NH * hd];
            for (i, &[lo, hi]) in spans.iter().enumerate() {
                for h in 0..NH {
                    let qi = &q[(i * NH + h) * hd..(i * NH + h + 1) * hd];
                    let logits: Vec<f64> = (lo as usize..hi as usize).map(|j| {
                        let kj = &kr[j * kvdim + h * hd..j * kvdim + (h + 1) * hd];
                        qi.iter().zip(kj).map(|(&a, &b)| a as f64 * b).sum::<f64>() * scale as f64
                    }).collect();
                    let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
                    let w: Vec<f64> = logits.iter().map(|&l| (l - mx).exp()).collect();
                    let z: f64 = w.iter().sum();
                    for c in 0..hd {
                        let acc: f64 = (lo as usize..hi as usize).zip(&w)
                            .map(|(j, &p)| p * vr[j * kvdim + h * hd + c]).sum();
                        want[(i * NH + h) * hd + c] = (acc / z) as f32;
                    }
                }
            }

            let (qb, kb, vb) = (upload_f32(&gpu, &q), upload_f16(&gpu, &k), upload_f16(&gpu, &v));
            let flat: Vec<u32> = spans.iter().flatten().copied().collect();
            let sb = gpu.device.new_buffer_with_data(flat.as_ptr() as *const c_void, (flat.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared);
            let n = total * NH * hd;
            let ob = upload_f32(&gpu, &vec![GUARD_FILL; GUARD * 2 + n]);
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&span_pipe);
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kb), 0);
            enc.set_buffer(2, Some(&vb), 0);
            enc.set_buffer(3, Some(&ob), (GUARD * 4) as u64);
            for (i, x) in [(4u64, hd as u32), (5, kvdim as u32), (6, total as u32), (7, GROUP as u32), (9, NH as u32)] {
                enc.set_bytes(i, 4, &x as *const u32 as *const c_void);
            }
            enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
            enc.set_buffer(10, Some(&sb), 0);
            enc.dispatch_thread_groups(MTLSize::new((total * NH) as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let label = format!("{entry} lens={lens:?} window={window:?}");
            assert_rows_distinct(&label, &want, total * NH, hd);
            compare(&label, &want, &host(&ob, GUARD + n)[GUARD..], total * NH, hd);
            assert_guards(&label, &ob, n);

            // The MMA twin on the same batch. It reads K/V in 8-row blocks past a tile's
            // key range, so its K/V are zero past the packed total (the contract), and it
            // takes one descriptor per 32-row query tile inside a sequence.
            if !gpu.native_reduce { continue; }
            let mma_name = ojas_metal::kernels::attn::attn_mma_span_name(hd as u32);
            let mma = gpu.pipeline(&ojas_metal::kernels::attn::attn_mma_span_src(hd as u32), &mma_name)
                .expect("mma span pipeline");
            let zero_pad = |v: &[f32]| { let mut z = v[..total * kvdim].to_vec(); z.resize((total + KV_PAD) * kvdim, 0.0); z };
            let (kz, vz) = (upload_f16(&gpu, &zero_pad(&k)), upload_f16(&gpu, &zero_pad(&v)));
            let mut tiles: Vec<u32> = Vec::new();
            let mut start = 0usize;
            for &len in &lens {
                for q0 in (start..start + len).step_by(32) {
                    let nq = 32.min(start + len - q0);
                    let (klo, khi) = (spans[q0][0], spans[q0 + nq - 1][1]);
                    tiles.extend_from_slice(&[q0 as u32, nq as u32, klo, khi]);
                }
                start += len;
            }
            let tb = gpu.device.new_buffer_with_data(tiles.as_ptr() as *const c_void, (tiles.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared);
            let mo = upload_f32(&gpu, &vec![GUARD_FILL; GUARD * 2 + n]);
            let cb = gpu.command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&mma);
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kz), 0);
            enc.set_buffer(2, Some(&vz), 0);
            enc.set_buffer(3, Some(&mo), (GUARD * 4) as u64);
            for (i, x) in [(4u64, hd as u32), (5, kvdim as u32), (6, 0), (7, GROUP as u32), (9, NH as u32), (10, total as u32)] {
                enc.set_bytes(i, 4, &x as *const u32 as *const c_void);
            }
            enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
            enc.set_buffer(11, Some(&tb), 0);
            enc.set_buffer(12, Some(&sb), 0);
            enc.dispatch_thread_groups(MTLSize::new(NH as u64, (tiles.len() / 4) as u64, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let label = format!("{mma_name} lens={lens:?} window={window:?}");
            compare(&label, &want, &host(&mo, GUARD + n)[GUARD..], total * NH, hd);
            assert_guards(&label, &mo, n);
        }
    }
}

/// The span rewrite slices exactly one kernel, moves both loop bounds, and does not
/// shadow a family kernel.
#[test]
fn bidir_span_source_moves_both_bounds() {
    let src = ojas_metal::kernels::attn::attn_bidir_span_src();
    assert_eq!(src.matches("kernel void ").count(), 1, "sliced more than one kernel");
    assert!(src.contains("uint seq = sp.y;"), "upper key bound not rewritten");
    assert!(src.contains("t = sp.x + sgid"), "lower key bound not rewritten");
    assert!(ojas_metal::kernels::source_of(ojas_metal::kernels::attn::ATTN_BIDIR_SPAN).is_none(),
        "the span kernel collides with a family kernel name");
}

/// `attention_m_bidir` against an f64 softmax attention over the same f16 K/V, for
/// head widths that are and are not multiples of the 32-lane value chunk (a vision
/// tower's 72-wide heads among them): every dimension of every head must be produced.
#[test]
fn streaming_bidir_matches_f64_reference_for_any_head_width() {
    let gpu = match MetalGpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("attn_bidir: no Metal device ({e}); skipping"); return; }
    };
    let src = ojas_metal::kernels::source_of("attention_m_bidir").expect("attention_m_bidir");
    let pipe = gpu.pipeline(src, "attention_m_bidir").expect("attention_m_bidir pipeline");
    let (nh, m, total) = (2usize, 5usize, 37usize);
    for hd in [64usize, 72, 40] {
        let kvdim = nh * hd;
        let mut seed = 0xBEEF ^ hd as u32;
        let q: Vec<f32> = (0..m * nh * hd).map(|_| lcg(&mut seed)).collect();
        let k: Vec<f32> = (0..total * kvdim).map(|_| lcg(&mut seed)).collect();
        let v: Vec<f32> = (0..total * kvdim).map(|_| lcg(&mut seed)).collect();
        let n = m * nh * hd;
        let ob = upload_f32(&gpu, &vec![GUARD_FILL; GUARD * 2 + n]);
        let scale = 1.0 / (hd as f32).sqrt();
        let ints = [(4, hd as u32), (5, kvdim as u32), (6, total as u32), (7, 1), (9, nh as u32)];
        run(&gpu, &pipe, &upload_f32(&gpu, &q), &upload_f16(&gpu, &k), &upload_f16(&gpu, &v), &ob, scale, &ints, ((m * nh) as u64, 1));
        assert_guards(&format!("hd={hd}"), &ob, n);
        let got = &host(&ob, GUARD + n)[GUARD..];
        let h = |x: f32| half::f16::from_f32(x).to_f64();
        let mut worst = 0f64;
        for t in 0..m {
            for head in 0..nh {
                let qr = &q[(t * nh + head) * hd..][..hd];
                let logits: Vec<f64> = (0..total).map(|j| {
                    (0..hd).map(|i| qr[i] as f64 * h(k[j * kvdim + head * hd + i])).sum::<f64>() * scale as f64
                }).collect();
                let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
                let sum: f64 = e.iter().sum();
                for i in 0..hd {
                    let want: f64 = (0..total).map(|j| e[j] / sum * h(v[j * kvdim + head * hd + i])).sum();
                    worst = worst.max((got[(t * nh + head) * hd + i] as f64 - want).abs());
                }
            }
        }
        assert!(worst < 1e-4, "hd={hd}: worst absolute error {worst:.3e}");
    }
}

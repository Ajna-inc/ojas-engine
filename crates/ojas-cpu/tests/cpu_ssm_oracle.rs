//! CpuSsm as the oracle for the surya-2 (qwen35) GPU ports.
//!
//! The oracle has to run at the file's own weight precision, rope from
//! sectioned `(t,h,w,e)` coordinates exactly as the Metal kernel does, and take
//! an image span as injected rows. Each is pinned on a small synthetic qwen35
//! written to a real GGUF (so the production loader is exercised); the
//! text-level checks repeat on surya-2 itself when its weights are present
//! (`OJAS_SURYA_GGUF=<surya-2.gguf>`, optional `OJAS_SURYA_MMPROJ`, or the HF
//! cache snapshot of `datalab-to/surya-ocr-2-gguf`); without them they print
//! `skip:` and pass.

use ojas_core::Model;
use ojas_cpu::cpu_ssm::{
    image_pos3, mrope_sel, rope_partial, rope_partial_m, text_pos3, write_trace_dir, CpuSsm, CpuSsmOpts,
    RopeAt, TraceCfg, MROPE_INTERLEAVED, MROPE_OFF, MROPE_SECTIONS, MROPE_VISION,
};
use ojas_formats::gguf::Gguf;
use std::path::{Path, PathBuf};

// ============================ tiny GGUF writer ==============================

enum Kv { U32(u32), F32(f32), Str(String), I32Arr(Vec<i32>), StrArr(Vec<String>) }

/// (name, dims [ne0 = cols, ne1 = rows], ggml type 0/1, raw bytes)
type T = (String, Vec<u64>, u32, Vec<u8>);

fn wstr(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn write_gguf(path: &Path, kvs: &[(&str, Kv)], tensors: &[T]) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        wstr(&mut b, k);
        match v {
            Kv::U32(x) => { b.extend_from_slice(&4u32.to_le_bytes()); b.extend_from_slice(&x.to_le_bytes()); }
            Kv::F32(x) => { b.extend_from_slice(&6u32.to_le_bytes()); b.extend_from_slice(&x.to_le_bytes()); }
            Kv::Str(s) => { b.extend_from_slice(&8u32.to_le_bytes()); wstr(&mut b, s); }
            Kv::I32Arr(a) => {
                b.extend_from_slice(&9u32.to_le_bytes());
                b.extend_from_slice(&5u32.to_le_bytes());
                b.extend_from_slice(&(a.len() as u64).to_le_bytes());
                for x in a { b.extend_from_slice(&x.to_le_bytes()); }
            }
            Kv::StrArr(a) => {
                b.extend_from_slice(&9u32.to_le_bytes());
                b.extend_from_slice(&8u32.to_le_bytes());
                b.extend_from_slice(&(a.len() as u64).to_le_bytes());
                for s in a { wstr(&mut b, s); }
            }
        }
    }
    let mut off = 0u64;
    let mut offs = Vec::new();
    for (name, dims, ty, data) in tensors {
        wstr(&mut b, name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims { b.extend_from_slice(&d.to_le_bytes()); }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&off.to_le_bytes());
        offs.push(off);
        off += (data.len() as u64 + 31) & !31;
    }
    while b.len() % 32 != 0 { b.push(0); }
    for ((_, _, _, data), o) in tensors.iter().zip(offs) {
        let base = b.len() as u64;
        debug_assert!(base % 32 == 0);
        let _ = o;
        b.extend_from_slice(data);
        while b.len() % 32 != 0 { b.push(0); }
    }
    std::fs::write(path, b).unwrap();
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Synthetic qwen35 shape: 4 layers (1 and 3 attention), GQA 4/2, hd 16, 12 rope
/// dims = 6 pairs split [2,2,2,0] — every pair lands in a t/h/w section in both
/// the contiguous and the interleaved layouts, like surya-2's [11,11,10,0].
const D: usize = 64;
const VOCAB: usize = 50;

fn synth_gguf(tag: &str, sections: Option<[i32; 4]>) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cpu_ssm_oracle_{tag}.gguf"));
    let (n_layers, n_head, n_kv, hd, n_rot, ffn) = (4usize, 4usize, 2usize, 16usize, 12u32, 96usize);
    let (s_st, h_k, h_v, d_inner, conv_k) = (16usize, 2usize, 4usize, 64usize, 4usize);
    let conv_ch = 2 * h_k * s_st + d_inner;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut ts: Vec<T> = Vec::new();
    let f16t = |ts: &mut Vec<T>, rng: &mut Rng, name: String, cols: usize, rows: usize, amp: f32| {
        let data: Vec<u8> = (0..cols * rows)
            .flat_map(|_| half::f16::from_f32(rng.f() * amp).to_bits().to_le_bytes()).collect();
        ts.push((name, vec![cols as u64, rows as u64], 1, data));
    };
    let f32t = |ts: &mut Vec<T>, name: String, dims: Vec<u64>, v: Vec<f32>| {
        ts.push((name, dims, 0, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    };
    let g = |k: usize| 1.0 / (k as f32).sqrt() * 1.7;
    for l in 0..n_layers {
        let p = |s: &str| format!("blk.{l}.{s}");
        let ones = |rng: &mut Rng, n: usize| (0..n).map(|_| 1.0 + 0.2 * rng.f()).collect::<Vec<f32>>();
        let v = ones(&mut rng, D); f32t(&mut ts, p("attn_norm.weight"), vec![D as u64], v);
        let v = ones(&mut rng, D); f32t(&mut ts, p("post_attention_norm.weight"), vec![D as u64], v);
        f16t(&mut ts, &mut rng, p("ffn_gate.weight"), D, ffn, g(D));
        f16t(&mut ts, &mut rng, p("ffn_up.weight"), D, ffn, g(D));
        f16t(&mut ts, &mut rng, p("ffn_down.weight"), ffn, D, g(ffn));
        if (l + 1) % 2 == 0 {
            f16t(&mut ts, &mut rng, p("attn_q.weight"), D, 2 * n_head * hd, g(D));
            f16t(&mut ts, &mut rng, p("attn_k.weight"), D, n_kv * hd, g(D));
            f16t(&mut ts, &mut rng, p("attn_v.weight"), D, n_kv * hd, g(D));
            f16t(&mut ts, &mut rng, p("attn_output.weight"), n_head * hd, D, g(n_head * hd));
            let v = ones(&mut rng, hd); f32t(&mut ts, p("attn_q_norm.weight"), vec![hd as u64], v);
            let v = ones(&mut rng, hd); f32t(&mut ts, p("attn_k_norm.weight"), vec![hd as u64], v);
        } else {
            f16t(&mut ts, &mut rng, p("attn_qkv.weight"), D, conv_ch, g(D));
            f16t(&mut ts, &mut rng, p("attn_gate.weight"), D, d_inner, g(D));
            f16t(&mut ts, &mut rng, p("ssm_alpha.weight"), D, h_v, g(D));
            f16t(&mut ts, &mut rng, p("ssm_beta.weight"), D, h_v, g(D));
            f16t(&mut ts, &mut rng, p("ssm_out.weight"), d_inner, D, g(d_inner));
            let v = (0..h_v).map(|_| rng.f() * 0.5).collect(); f32t(&mut ts, p("ssm_dt.bias"), vec![h_v as u64], v);
            let v = (0..h_v).map(|_| -(0.5 + rng.f().abs())).collect(); f32t(&mut ts, p("ssm_a"), vec![h_v as u64], v);
            let v = (0..conv_ch * conv_k).map(|_| rng.f() * 0.5).collect();
            f32t(&mut ts, p("ssm_conv1d.weight"), vec![conv_k as u64, conv_ch as u64], v);
            let v = ones(&mut rng, d_inner / h_v); f32t(&mut ts, p("ssm_norm.weight"), vec![(d_inner / h_v) as u64], v);
        }
    }
    let v = (0..D).map(|_| 1.0 + 0.2 * rng.f()).collect(); f32t(&mut ts, "output_norm.weight".into(), vec![D as u64], v);
    f16t(&mut ts, &mut rng, "token_embd.weight".into(), D, VOCAB, 1.0);
    f16t(&mut ts, &mut rng, "output.weight".into(), D, VOCAB, g(D) * 3.0);

    let mut kvs = vec![
        ("general.architecture", Kv::Str("qwen35".into())),
        ("qwen35.embedding_length", Kv::U32(D as u32)),
        ("qwen35.block_count", Kv::U32(n_layers as u32)),
        ("qwen35.attention.head_count", Kv::U32(n_head as u32)),
        ("qwen35.attention.head_count_kv", Kv::U32(n_kv as u32)),
        ("qwen35.attention.key_length", Kv::U32(hd as u32)),
        ("qwen35.feed_forward_length", Kv::U32(ffn as u32)),
        ("qwen35.ssm.state_size", Kv::U32(s_st as u32)),
        ("qwen35.ssm.group_count", Kv::U32(h_k as u32)),
        ("qwen35.ssm.time_step_rank", Kv::U32(h_v as u32)),
        ("qwen35.ssm.inner_size", Kv::U32(d_inner as u32)),
        ("qwen35.ssm.conv_kernel", Kv::U32(conv_k as u32)),
        ("qwen35.full_attention_interval", Kv::U32(2)),
        ("qwen35.rope.dimension_count", Kv::U32(n_rot)),
        ("qwen35.rope.freq_base", Kv::F32(10000.0)),
        ("qwen35.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
        ("tokenizer.ggml.tokens", Kv::StrArr((0..VOCAB).map(|i| format!("t{i}")).collect())),
    ];
    if let Some(s) = sections { kvs.push(("qwen35.rope.dimension_sections", Kv::I32Arr(s.to_vec()))); }
    write_gguf(&path, &kvs, &ts);
    path
}

fn load(path: &Path, exact: bool) -> CpuSsm {
    let mut g = Gguf::open(path.to_str().unwrap()).unwrap();
    CpuSsm::load_with(&mut g, CpuSsmOpts { exact }).unwrap()
}

fn prompt(n: usize) -> Vec<u32> { (0..n).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect() }

fn embeds(m: &CpuSsm, toks: &[u32]) -> Vec<f32> {
    toks.iter().flat_map(|&t| m.embed_row(t as usize)).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
}

/// Top-1 minus top-2 logit.
fn margin(v: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::MIN, f32::MIN);
    for &x in v { if x > a { b = a; a = x; } else if x > b { b = x; } }
    a - b
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) { ab += (*x as f64) * (*y as f64); aa += (*x as f64).powi(2); bb += (*y as f64).powi(2); }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// Prefill `toks` by id and return the logits of every prompt row (via the trace).
fn prompt_logits(m: &CpuSsm, toks: &[u32]) -> Vec<Vec<f32>> {
    m.reset_session();
    m.trace_start(TraceCfg { hidden: false, logits: true, rows: None });
    m.prefill(toks, 0);
    m.trace_take().into_iter().map(|r| r.logits.unwrap()).collect()
}

// ============================ rope unit checks ==============================

#[test]
fn imrope_sector_map_matches_surya_layout() {
    // [11,11,10,0] over 32 pairs: t = {0,3,..,30}, h = {1,4,..,31}, w = {2,5,..,29}; e never.
    let pos = [100, 200, 300, 400];
    for j in 0..32 {
        let (p, je) = mrope_sel([11, 11, 10, 0], MROPE_INTERLEAVED, pos, j);
        assert_eq!(je, j, "IMROPE keeps the plain exponent index");
        assert_eq!(p, pos[j % 3], "pair {j}");
    }
    // Contiguous: first 11 t, next 11 h, last 10 w.
    for j in 0..32 {
        let (p, je) = mrope_sel([11, 11, 10, 0], MROPE_SECTIONS, pos, j);
        assert_eq!(je, j);
        assert_eq!(p, pos[if j < 11 { 0 } else if j < 22 { 1 } else { 2 }], "pair {j}");
    }
    // Vision: theta restarts per section.
    for j in 0..32 {
        let (_, je) = mrope_sel([8, 8, 8, 8], MROPE_VISION, pos, j);
        assert_eq!(je, j % 8);
    }
    // No sections declared: t stream, plain theta.
    assert_eq!(mrope_sel([0; 4], MROPE_INTERLEAVED, pos, 5), (100, 5));
}

#[test]
fn sectioned_rope_with_equal_streams_is_bit_identical_to_scalar() {
    let mut rng = Rng(7);
    let base: Vec<f32> = (0..256).map(|_| rng.f()).collect();
    for p in [0usize, 1, 17, 555, 4095, 70000] {
        let mut a = base.clone();
        rope_partial(&mut a, 64, p, 1e7);
        for mode in [MROPE_OFF, MROPE_SECTIONS, MROPE_INTERLEAVED] {
            let mut b = base.clone();
            rope_partial_m(&mut b, 64, [p as u32, p as u32, p as u32, 0], [11, 11, 10, 0], mode, 1e7);
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "mode {mode} pos {p}");
        }
        if p > 0 {
            let mut c = base.clone();
            rope_partial_m(&mut c, 64, [p as u32; 4], [8, 8, 8, 8], MROPE_VISION, 1e7);
            assert!(a != c, "VISION must not degenerate to plain rope");
        }
    }
    // helpers follow the llama.cpp conventions
    assert_eq!(text_pos3(5, 2), vec![[5, 5, 5, 0], [6, 6, 6, 0]]);
    assert_eq!(image_pos3(10, 3, 2), vec![[10, 10, 10, 0], [10, 10, 11, 0], [10, 10, 12, 0],
                                          [10, 11, 10, 0], [10, 11, 11, 0], [10, 11, 12, 0]]);
}

// ============================ synthetic model ===============================

#[test]
fn exact_vs_default_argmax_agreement_synthetic() {
    let path = synth_gguf("modes", Some([2, 2, 2, 0]));
    let exact = load(&path, true);
    let q8 = load(&path, false);
    assert!(exact.is_exact() && !q8.is_exact());
    let toks = prompt(40);
    let le = prompt_logits(&exact, &toks);
    let lq = prompt_logits(&q8, &toks);
    assert_eq!(le.len(), toks.len());
    let agree = le.iter().zip(&lq).filter(|(a, b)| argmax(a) == argmax(b)).count();
    let min_cos = le.iter().zip(&lq).map(|(a, b)| cosine(a, b)).fold(1.0, f64::min);
    let differ = le.iter().zip(&lq).any(|(a, b)| a != b);
    eprintln!("synthetic exact vs int8: argmax {agree}/{} min cosine {min_cos:.6}", toks.len());
    assert!(differ, "exact and int8 logits are identical — the mode switch did nothing");
    let confident: Vec<usize> = (0..le.len()).filter(|&i| margin(&le[i]) > 1.0).collect();
    assert!(!confident.is_empty());
    assert!(confident.iter().all(|&i| argmax(&le[i]) == argmax(&lq[i])), "int8 flipped a confident argmax");
    assert!(min_cos > 0.99, "int8 drifted too far from exact: min cosine {min_cos}");
    assert!(agree * 100 >= toks.len() * 85, "argmax agreement {agree}/{}", toks.len());
}

#[test]
fn mrope_text_positions_equal_scalar_path() {
    let path = synth_gguf("mrope", Some([2, 2, 2, 0]));
    let mut m = load(&path, true);
    let toks = prompt(30);
    let (span, probe) = (&toks[..29], toks[29]);
    m.prefill(span, 0);
    let want = m.forward_logits(probe, 29).unwrap();
    let x = embeds(&m, span);
    for mode in [MROPE_SECTIONS, MROPE_INTERLEAVED] {
        m.set_mrope([2, 2, 2, 0], mode);
        m.reset_session();
        assert!(m.prefill_embeds(span, &x, 0, Some(&text_pos3(0, span.len()))));
        let got = m.forward_logits(probe, 29).unwrap();
        assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
            "mode {mode}: (p,p,p,0) coordinates must reproduce scalar rope bit-for-bit");
    }
    // Negative control: VISION restarts theta, so the same coordinates must move the logits.
    m.set_mrope([2, 2, 2, 0], MROPE_VISION);
    m.reset_session();
    assert!(m.prefill_embeds(span, &x, 0, Some(&text_pos3(0, span.len()))));
    assert!(m.forward_logits(probe, 29).unwrap() != want, "VISION mode ignored");
    // In the default (IMROPE) mode, different h/w streams must move them too.
    m.set_mrope([2, 2, 2, 0], MROPE_INTERLEAVED);
    m.reset_session();
    let p3: Vec<[u32; 4]> = (0..span.len() as u32).map(|p| [p, p + 3, p + 5, 0]).collect();
    assert!(m.prefill_embeds(span, &x, 0, Some(&p3)));
    assert!(m.forward_logits(probe, 29).unwrap() != want, "h/w coordinates ignored");
}

#[test]
fn prefill_embeds_with_token_rows_reproduces_token_prefill() {
    let path = synth_gguf("inject", Some([2, 2, 2, 0]));
    for exact in [true, false] {
        let m = load(&path, exact);
        // > one CHUNK (128) so the chunk-offset arithmetic is exercised.
        let toks = prompt(200);
        let (span, probe) = (&toks[..199], toks[199]);
        m.reset_session();
        m.prefill(span, 0);
        let want = m.forward_logits(probe, 199).unwrap();

        // One-row-at-a-time reference: the batched layer-major prefill must equal it.
        m.reset_session();
        for (i, &t) in span.iter().enumerate() { m.forward_logits(t, i); }
        let serial = m.forward_logits(probe, 199).unwrap();
        assert!(want.iter().zip(&serial).all(|(a, b)| a.to_bits() == b.to_bits()),
            "exact={exact}: chunked prefill differs from token-by-token forward");

        let x = embeds(&m, span);
        m.reset_session();
        assert!(m.prefill_embeds(span, &x, 0, None));
        let got = m.forward_logits(probe, 199).unwrap();
        assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
            "exact={exact}: injected token rows must reproduce the id prefill bit-for-bit");

        // Split across two calls (text | injected | text), cache rows contiguous.
        m.reset_session();
        m.prefill(&span[..50], 0);
        assert!(m.prefill_embeds(&span[50..170], &x[50 * D..170 * D], 50, None));
        m.prefill(&span[170..], 170);
        let split = m.forward_logits(probe, 199).unwrap();
        assert!(want.iter().zip(&split).all(|(a, b)| a.to_bits() == b.to_bits()),
            "exact={exact}: mixed prefill differs");

        // Negative control: one swapped row (in the second chunk) must move the logits.
        let mut bad = x.clone();
        bad[140 * D..141 * D].copy_from_slice(&m.embed_row((span[140] as usize + 1) % VOCAB));
        m.reset_session();
        assert!(m.prefill_embeds(span, &bad, 0, None));
        assert!(m.forward_logits(probe, 199).unwrap() != want, "exact={exact}: injected rows ignored");
    }
}

#[test]
fn prefill_embeds_refuses_coordinates_without_sections() {
    let path = synth_gguf("nosect", None);
    let m = load(&path, true);
    assert_eq!(m.mrope().0, [0; 4]);
    let x = vec![0f32; D];
    assert!(m.prefill_embeds(&[0], &x, 0, None), "rows without coordinates are fine");
    assert!(!m.prefill_embeds(&[0], &x, 0, Some(&[[0, 0, 0, 0]])), "coordinates it cannot honour must be refused");
    // The ocr capability probe on a model that does declare them:
    let m = load(&synth_gguf("probe", Some([2, 2, 2, 0])), true);
    assert!(m.prefill_embeds(&[0], &x, 0, Some(&[[0, 0, 0, 0]])));
    assert_eq!(m.vision_width(), None, "no tower attached");
}

#[test]
fn trace_hook_records_every_boundary() {
    let path = synth_gguf("trace", Some([2, 2, 2, 0]));
    let m = load(&path, true);
    let toks = prompt(12);
    m.trace_start(TraceCfg { hidden: true, logits: true, rows: Some(4..12) });
    m.prefill(&toks[..6], 0);
    let img = image_pos3(6, 3, 2);
    let x: Vec<f32> = (0..6 * D).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    assert!(m.prefill_embeds(&[0; 6], &x, 6, Some(&img)));
    let lg = m.forward_logits(toks[6], 12).unwrap();
    let rows = m.trace_take();
    // rows 4..12 of the prefill span (4,5 text; 6..11 image); row 12 is outside the range.
    assert_eq!(rows.iter().map(|r| r.row).collect::<Vec<_>>(), (4..12).collect::<Vec<_>>());
    for r in &rows {
        assert_eq!(r.hidden.len(), m.n_layers() + 2);
        assert!(r.hidden.iter().all(|h| h.len() == D));
        assert_eq!(r.logits.as_ref().unwrap().len(), VOCAB);
    }
    assert!(rows[2].injected && !rows[1].injected);
    assert_eq!(rows[3].rope, RopeAt::Sect([6, 6, 7, 0]));
    assert_eq!(rows[3].hidden[0], x[D..2 * D].to_vec(), "hidden[0] is the injected row verbatim");
    assert_eq!(lg.len(), VOCAB);

    // A traced forward's logits equal an untraced one's.
    m.reset_session();
    m.prefill(&toks[..6], 0);
    let plain = m.forward_logits(toks[6], 6).unwrap();
    m.reset_session();
    m.trace_start(TraceCfg { hidden: true, logits: true, rows: None });
    m.prefill(&toks[..6], 0);
    let traced = m.forward_logits(toks[6], 6).unwrap();
    let rows2 = m.trace_take();
    assert_eq!(plain, traced);
    assert_eq!(rows2.last().unwrap().logits.as_ref().unwrap(), &traced);
    // forward_trace (single token) agrees with the hook's hidden states.
    m.reset_session();
    m.prefill(&toks[..6], 0);
    let (ft, fl) = m.forward_trace(toks[6] as usize, 6);
    assert_eq!(fl, traced);
    assert_eq!(&ft, &rows2.last().unwrap().hidden);

    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cpu_ssm_trace_dump");
    let _ = std::fs::remove_dir_all(&dir);
    write_trace_dir(&rows, &dir).unwrap();
    assert_eq!(std::fs::metadata(dir.join("hidden_000.f32")).unwrap().len() as usize, rows.len() * D * 4);
    assert_eq!(std::fs::metadata(dir.join("logits.f32")).unwrap().len() as usize, rows.len() * VOCAB * 4);
    assert_eq!(std::fs::metadata(dir.join("rows.u32")).unwrap().len() as usize, rows.len() * 6 * 4);
}

// ============================== surya-2 =====================================

fn find_surya(file: &str, env: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(h) = std::env::var("HF_HUB_CACHE") { roots.push(h.into()); }
    if let Ok(h) = std::env::var("HF_HOME") { roots.push(Path::new(&h).join("hub")); }
    if let Ok(h) = std::env::var("HOME") { roots.push(Path::new(&h).join(".cache/huggingface/hub")); }
    roots.push("/data/dev-cache/hf/hub".into());
    for r in roots {
        let snaps = r.join("models--datalab-to--surya-ocr-2-gguf/snapshots");
        for e in std::fs::read_dir(&snaps).into_iter().flatten().flatten() {
            let p = e.path().join(file);
            // a symlink to a still-downloading blob does not resolve; skip it
            if std::fs::metadata(&p).is_ok() { return Some(p); }
        }
    }
    None
}

fn surya() -> Option<PathBuf> {
    let p = find_surya("surya-2.gguf", "OJAS_SURYA_GGUF");
    if p.is_none() { eprintln!("skip: surya-2.gguf not found (set OJAS_SURYA_GGUF)"); }
    p
}

fn surya_prompt(path: &Path) -> Vec<u32> {
    let g = Gguf::open(path.to_str().unwrap()).unwrap();
    let bpe = ojas_tokenize::Bpe::from_gguf(&g);
    let ids: Vec<u32> = bpe.encode("<div data-label=\"Text\">The quick brown fox jumps over the lazy dog. \
        Invoice 2026-09-27, total 1,234.50 INR.</div>").into_iter().map(|t| t as u32).collect();
    assert!(ids.len() > 16, "tokenizer produced {} ids", ids.len());
    ids
}

#[test]
fn surya_exact_vs_default_argmax() {
    let Some(path) = surya() else { return };
    let toks = surya_prompt(&path);
    let exact = load(&path, true);
    assert_eq!(exact.mrope().0, [11, 11, 10, 0], "surya-2 declares IMROPE sections");
    let le = prompt_logits(&exact, &toks);
    drop(exact);
    let q8 = load(&path, false);
    let lq = prompt_logits(&q8, &toks);
    let agree = le.iter().zip(&lq).filter(|(a, b)| argmax(a) == argmax(b)).count();
    let min_cos = le.iter().zip(&lq).map(|(a, b)| cosine(a, b)).fold(1.0, f64::min);
    eprintln!("surya-2 exact vs int8 over {} prompt rows: argmax {agree} agree, min cosine {min_cos:.6}", toks.len());
    for (i, (a, b)) in le.iter().zip(&lq).enumerate() {
        if argmax(a) != argmax(b) { eprintln!("  row {i}: exact top1 {} margin {:.3} | int8 top1 {} margin {:.3}", argmax(a), margin(a), argmax(b), margin(b)); }
    }
    let confident: Vec<usize> = (0..le.len()).filter(|&i| margin(&le[i]) > 1.0).collect();
    let conf_agree = confident.iter().filter(|&&i| argmax(&le[i]) == argmax(&lq[i])).count();
    eprintln!("  rows with exact margin > 1.0: {conf_agree}/{} agree", confident.len());
    // Measured 2026-09-27: 75/83 agree, min cosine 0.987, every flip a near-tie
    // (exact margin < 0.45), all 41 rows with margin > 1 agree. The int8 repack
    // of weights and per-vector activations is why the default mode is not an
    // oracle; the confident rows are where the two must still agree.
    assert_eq!(conf_agree, confident.len(), "int8 flipped a confident argmax");
    assert!(min_cos > 0.98, "min cosine {min_cos}");
    assert!(agree * 100 >= toks.len() * 85, "argmax agreement {agree}/{}", toks.len());
}

#[test]
fn surya_text_mrope_and_injection_match_token_prefill() {
    let Some(path) = surya() else { return };
    let toks = surya_prompt(&path);
    let n = toks.len() - 1;
    let (span, probe) = (&toks[..n], toks[n]);
    let m = load(&path, true);
    m.prefill(span, 0);
    let want = m.forward_logits(probe, n).unwrap();
    let x = embeds(&m, span);
    m.reset_session();
    assert!(m.prefill_embeds(span, &x, 0, Some(&text_pos3(0, n))));
    let got = m.forward_logits(probe, n).unwrap();
    assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
        "surya-2: injected token rows at (p,p,p,0) must equal the id prefill bit-for-bit");
}

#[test]
fn surya_image_and_text_prefill_runs() {
    let Some(path) = surya() else { return };
    let Some(mmproj) = find_surya("surya-2-mmproj.gguf", "OJAS_SURYA_MMPROJ") else {
        eprintln!("skip: surya-2-mmproj.gguf not found (set OJAS_SURYA_MMPROJ)");
        return;
    };
    let mut m = load(&path, true);
    let mut g = Gguf::open(mmproj.to_str().unwrap()).unwrap();
    m.attach_vit(ojas_cpu::CpuVit::load(&mut g).unwrap()).unwrap();
    let (w, h) = (128usize, 64usize); // 8x4 patches -> 4x2 merged rows
    let n_img = m.vision_tokens(w, h).unwrap();
    assert_eq!(n_img, 8);
    let img: Vec<f32> = (0..3 * w * h).map(|i| ((i * 37 % 255) as f32 / 127.5) - 1.0).collect();
    let rows = m.encode_image(&img, w, h).unwrap().unwrap();
    assert_eq!(rows.len(), n_img * m.hidden_dim());

    let text = surya_prompt(&path);
    let head = &text[..5];
    let tail = &text[5..];
    let (nx, ny) = (4usize, 2usize);
    let run = |p3: &[[u32; 4]]| -> Vec<f32> {
        m.reset_session();
        m.prefill(head, 0);
        assert!(m.prefill_embeds(&vec![0u32; n_img], &rows, 5, Some(p3)));
        // text after the span: cache rows continue at 5 + n_img
        m.prefill(&tail[..tail.len() - 1], 5 + n_img);
        m.forward_logits(tail[tail.len() - 1], 5 + n_img + tail.len() - 1).unwrap()
    };
    let a = run(&image_pos3(5, nx, ny));
    assert!(a.iter().all(|v| v.is_finite()));
    let b = run(&text_pos3(5, n_img));
    assert!(a != b, "image coordinates must reach the rope");
    let _ = MROPE_OFF;
}

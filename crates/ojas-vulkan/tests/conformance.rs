//! Conformance of the Vulkan kernels.
//!
//! 1. Every compiled kernel loads on every Vulkan device of this machine.
//! 2. The PyTorch golden cases (the same files ojas-cuda is held to, from
//!    `scripts/release/torch_kernel_golden.py`) on every device: copies,
//!    adds, pooling and upsampling must be bit-exact; kernels that call
//!    `exp` or sum products may differ by a few ULP (Vulkan leaves `exp`
//!    precision to the driver; CUDA and torch share `expf`).
//!
//! Devices: all of them, or `OJAS_VK_DEVICES=llvmpipe,radv,nvidia` (index or
//! name parts). Golden data: `OJAS_TORCH_GOLDEN` or `target/torch_golden`.
//! Tests pass (with a note) where there is no Vulkan or no golden data.

use ojas_core::{Device, KernelRuntime};
use ojas_vulkan::{VkBuf, VkGpu};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn gpus() -> Vec<VkGpu> {
    let all = ojas_vulkan::devices();
    let want: Vec<usize> = match std::env::var("OJAS_VK_DEVICES") {
        Ok(s) => s.split(',').filter_map(|p| ojas_vulkan::select(Some(p)).ok()).collect(),
        Err(_) => all.iter().map(|d| d.index).collect(),
    };
    let mut out = vec![];
    for i in want {
        match VkGpu::new(i) {
            Ok(mut g) => {
                g.ensure_family("cnn").expect("load cnn family");
                g.ensure_family("vk").expect("load vk test family");
                out.push(g);
            }
            Err(e) => eprintln!("skip Vulkan device {i}: {e:#}"),
        }
    }
    if out.is_empty() {
        eprintln!("no usable Vulkan device: nothing to test");
    }
    out
}

#[test]
fn every_kernel_loads_on_every_device() {
    for g in gpus() {
        for (name, _) in ojas_vulkan::kernel_table() {
            if name.contains("_cm") && !g.info().coopmat_f16_16x16x16 {
                continue; // matrix-unit kernels load only where they run
            }
            assert!(g.has_kernel(name), "{}: {name} not loaded", g.name());
            // pipeline creation compiles the SPIR-V for this device
            g.dispatch(&g.begin(), name, &[], &[], [0, 0, 0], [256, 1, 1]).unwrap();
        }
        eprintln!("{}: {} kernels load", g.name(), ojas_vulkan::kernel_table().len());
    }
}

struct Golden {
    dir: PathBuf,
    f16: bool,
}

impl Golden {
    fn tensor(&self, case: &Value, key: &str) -> (Vec<f32>, Vec<usize>) {
        let t = &case["tensors"][key];
        let bytes = std::fs::read(self.dir.join(t["file"].as_str().unwrap())).unwrap();
        let data = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let shape = t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
        (data, shape)
    }
    fn has(&self, case: &Value, key: &str) -> bool {
        !case["tensors"][key].is_null()
    }
    fn up(&self, g: &VkGpu, data: &[f32]) -> VkBuf {
        if self.f16 { g.upload_f16(data) } else { g.upload(data) }
    }
    fn zeros(&self, g: &VkGpu, n: usize) -> VkBuf {
        self.up(g, &vec![0.0; n])
    }
}

fn launch(g: &VkGpu, enc: &<VkGpu as Device>::Enc, name: &str, bufs: &[&VkBuf], consts: &[u32], n: usize) {
    let bufs: Vec<(&VkBuf, u64)> = bufs.iter().map(|b| (*b, 0)).collect();
    g.dispatch(enc, name, &bufs, consts, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1]).unwrap();
}

/// Distance in units in the last place of the storage type (NaN == NaN).
fn ulps(a: f32, b: f32, f16: bool) -> u32 {
    if (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let key = |v: f32| -> i64 {
        if f16 {
            let b = half::f16::from_f32(v).to_bits() as i64;
            if b & 0x8000 != 0 { -(b & 0x7fff) } else { b }
        } else {
            let b = v.to_bits() as i64;
            if b & 0x8000_0000 != 0 { -(b & 0x7fff_ffff) } else { b }
        }
    };
    (key(a) - key(b)).unsigned_abs().min(u32::MAX as u64) as u32
}

/// A result torch leaves subnormal that the device flushed to zero (GPUs
/// flush f32 denormals; no inference output lives there).
fn flushed(got: f32, want: f32) -> bool {
    want != 0.0 && want.abs() < f32::MIN_POSITIVE && (got == 0.0 || got.abs() < f32::MIN_POSITIVE)
}

/// llvmpipe's `fma` is a software routine: not fused, and NaN where IEEE gives ±inf. Its sums
/// of products are checked on finite values, to 1e-6 of the case's largest magnitude
/// (cancellation makes ULPs meaningless there).
fn soft_fma(g: &VkGpu) -> bool {
    g.info().driver.to_lowercase().contains("llvmpipe")
}

/// Run one golden case: (differing elements, max ULP, elements).
fn run_case(g: &VkGpu, gd: &Golden, case: &Value) -> (usize, u32, usize) {
    let k = |base: &str| if gd.f16 { format!("{base}_f16") } else { base.to_string() };
    let p = |key: &str| case[key].as_u64().unwrap() as u32;
    let enc = g.begin();
    let (out, want): (Vec<(VkBuf, usize)>, Vec<Vec<f32>>) = match case["kernel"].as_str().unwrap() {
        "cnn_bias_act" => {
            let (x, _) = gd.tensor(case, "x");
            let (b, _) = gd.tensor(case, "bias");
            let (xd, bd) = (gd.up(g, &x), gd.up(g, &b));
            launch(g, &enc, &k("cnn_bias_act"), &[&xd, &bd], &[x.len() as u32, p("plane"), p("ch"), p("act")], x.len());
            (vec![(xd, x.len())], vec![gd.tensor(case, "y").0])
        }
        "cnn_act" => {
            let (x, _) = gd.tensor(case, "x");
            let (xd, yd) = (gd.up(g, &x), gd.zeros(g, x.len()));
            launch(g, &enc, &k("cnn_act"), &[&xd, &yd], &[x.len() as u32, p("act")], x.len());
            (vec![(yd, x.len())], vec![gd.tensor(case, "y").0])
        }
        "cnn_add" => {
            let (a, _) = gd.tensor(case, "a");
            let (b, _) = gd.tensor(case, "b");
            let (ad, bd, yd) = (gd.up(g, &a), gd.up(g, &b), gd.zeros(g, a.len()));
            launch(g, &enc, &k("cnn_add"), &[&ad, &bd, &yd], &[a.len() as u32], a.len());
            (vec![(yd, a.len())], vec![gd.tensor(case, "y").0])
        }
        kind @ ("cnn_concat" | "cnn_split") => {
            let lens: Vec<u32> = case["axis_lens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let total: u32 = lens.iter().sum();
            let inner = p("inner");
            let concat = kind == "cnn_concat";
            let (whole, wshape) = gd.tensor(case, if concat { "y" } else { "x" });
            let outer = wshape[0] as u32;
            let wd = if concat { gd.zeros(g, whole.len()) } else { gd.up(g, &whole) };
            let mut parts = vec![];
            let mut off = 0u32;
            for (i, &len) in lens.iter().enumerate() {
                let n = (outer * len * inner) as usize;
                if concat {
                    let src = gd.up(g, &gd.tensor(case, &format!("in{i}")).0);
                    launch(g, &enc, &k("cnn_axis_copy"), &[&src, &wd], &[n as u32, len, inner, len, 0, total, off], n);
                    parts.push(src);
                } else {
                    let dst = gd.zeros(g, n);
                    launch(g, &enc, &k("cnn_axis_copy"), &[&wd, &dst], &[n as u32, len, inner, total, off, len, 0], n);
                    parts.push(dst);
                }
                off += len;
            }
            if concat {
                (vec![(wd, whole.len())], vec![whole])
            } else {
                let want = (0..lens.len()).map(|i| gd.tensor(case, &format!("out{i}")).0).collect::<Vec<_>>();
                let out = parts.into_iter().zip(&want).map(|(b, w)| (b, w.len())).collect();
                (out, want)
            }
        }
        "cnn_upsample_nearest" => {
            let (x, xs) = gd.tensor(case, "x");
            let (y, ys) = gd.tensor(case, "y");
            let scale = (1.0f64 / p("scale") as f64) as f32;
            let (xd, yd) = (gd.up(g, &x), gd.zeros(g, y.len()));
            launch(g, &enc, &k("cnn_upsample_nearest"), &[&xd, &yd], &[y.len() as u32, xs[2] as u32, xs[3] as u32, ys[2] as u32, ys[3] as u32, scale.to_bits(), scale.to_bits()], y.len());
            (vec![(yd, y.len())], vec![y])
        }
        "cnn_maxpool" => {
            let (x, xs) = gd.tensor(case, "x");
            let (y, ys) = gd.tensor(case, "y");
            let (xd, yd) = (gd.up(g, &x), gd.zeros(g, y.len()));
            let (kk, s, pad, dil) = (p("k"), p("s"), p("pad"), p("dil"));
            launch(g, &enc, &k("cnn_maxpool"), &[&xd, &yd], &[y.len() as u32, xs[1] as u32, xs[2] as u32, xs[3] as u32, ys[2] as u32, ys[3] as u32, kk, kk, s, s, pad, pad, dil, dil], y.len());
            (vec![(yd, y.len())], vec![y])
        }
        "cnn_softmax" => {
            let (x, _) = gd.tensor(case, "x");
            let (dim, inner) = (p("dim"), p("inner"));
            let slices = x.len() / dim as usize;
            let (xd, yd) = (gd.up(g, &x), gd.zeros(g, x.len()));
            launch(g, &enc, &k("cnn_softmax"), &[&xd, &yd], &[slices as u32, dim, inner], slices);
            (vec![(yd, x.len())], vec![gd.tensor(case, "y").0])
        }
        "cnn_dwconv" => {
            let (x, xs) = gd.tensor(case, "x");
            let (w, ws) = gd.tensor(case, "w");
            let (y, ys) = gd.tensor(case, "y");
            let has_bias = gd.has(case, "b");
            let (xd, wd, yd) = (gd.up(g, &x), gd.up(g, &w), gd.zeros(g, y.len()));
            let bd = if has_bias { gd.up(g, &gd.tensor(case, "b").0) } else { gd.zeros(g, 1) };
            let (kk, s, pad, dil) = (p("k"), p("s"), p("pad"), p("dil"));
            let out_ch = ws[0] as u32;
            launch(g, &enc, &k("cnn_dwconv"), &[&xd, &wd, &bd, &yd],
                   &[y.len() as u32, out_ch, out_ch / xs[1] as u32, xs[3] as u32, xs[2] as u32, ys[3] as u32, ys[2] as u32, kk, kk, s, s, pad, pad, dil, dil, has_bias as u32],
                   y.len());
            (vec![(yd, y.len())], vec![y])
        }
        other => panic!("unknown golden kernel {other}"),
    };
    g.submit(enc).unwrap();
    let (mut bad, mut worst, mut total) = (0, 0u32, 0);
    for ((buf, n), want) in out.iter().zip(&want) {
        let mut got = vec![0.0; *n];
        g.read(buf, &mut got);
        let dump = std::env::var("OJAS_VK_DUMP").is_ok_and(|d| case["name"].as_str().unwrap().contains(&d));
        let soft = soft_fma(g) && case["kernel"] == "cnn_dwconv";
        // unfused products + cancellation: judge llvmpipe against the case's scale
        let scale = want.iter().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()));
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            // (and near f32::MAX an unfused product overflows where the fused one does not)
            if (soft && (!b.is_finite() || b.abs() > f32::MAX / 4.0 || (a - b).abs() <= 1e-6 * scale)) || flushed(*a, *b) {
                continue;
            }
            let u = ulps(*a, *b, gd.f16);
            if u > 0 {
                if dump && bad < 6 {
                    eprintln!("      [{i}] got {a:e} want {b:e} ({u} ULP)");
                }
                bad += 1;
                worst = worst.max(u);
            }
        }
        total += n;
    }
    (bad, worst, total)
}

/// Kernels whose result is fully determined by IEEE rounding of +, max and
/// copies: any difference from torch is a bug.
fn must_be_exact(g: &VkGpu, case: &Value) -> bool {
    match case["kernel"].as_str().unwrap() {
        "cnn_add" | "cnn_concat" | "cnn_split" | "cnn_upsample_nearest" | "cnn_maxpool" => true,
        // fused multiply-add in the reference and on hardware with FMA
        "cnn_dwconv" => !soft_fma(g),
        "cnn_bias_act" | "cnn_act" => case["act"].as_u64() == Some(0) || case["act"].as_u64() == Some(3),
        _ => false,
    }
}

/// ULP bound for kernels that use `exp` or (on llvmpipe) sums of products.
/// Vulkan allows `exp` 3 + 2|x| ULP; measured worst case here: 28 (f32
/// softmax), 1 in f16 — the precision inference runs at.
fn ulp_bound(f16: bool) -> u32 {
    if f16 { 1 } else { 32 }
}

#[test]
fn cnn_matches_torch() {
    let dir = std::env::var("OJAS_TORCH_GOLDEN").map(PathBuf::from).unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/torch_golden"));
    let Ok(manifest) = std::fs::read_to_string(dir.join("manifest.json")) else {
        eprintln!("no golden data at {} (run scripts/release/torch_kernel_golden.py): skipped", dir.display());
        return;
    };
    let manifest: Value = serde_json::from_str(&manifest).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    let mut failures = vec![];
    for g in gpus() {
        eprintln!("== {} — {} cases vs torch {}", g.name(), cases.len(), manifest["torch"]);
        for case in cases {
            let gd = Golden { dir: dir.clone(), f16: case["dtype"] == "f16" };
            let (bad, worst, total) = run_case(&g, &gd, case);
            let name = case["name"].as_str().unwrap();
            let exact = must_be_exact(&g, case);
            let ok = if exact { bad == 0 } else { worst <= ulp_bound(gd.f16) };
            if bad > 0 {
                eprintln!("   {name:28} {bad:6} of {total:7} differ, max {worst} ULP{}", if ok { "" } else { "  <-- FAIL" });
            }
            if !ok {
                failures.push(format!("{}: {name}: {bad} of {total} differ (max {worst} ULP, {})", g.info().name, if exact { "must be exact" } else { "over bound" }));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// Which rounding the fp16 bias+activation golden follows (diagnostic:
/// `OJAS_VK_WHY=1 cargo test --test conformance why_bias_act -- --nocapture`).
#[test]
fn why_bias_act() {
    if std::env::var("OJAS_VK_WHY").is_err() {
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/torch_golden");
    let m: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    for name in ["bias_act1_f16", "bias_act2_f16", "bias_act0_f16"] {
        let Some(case) = m["cases"].as_array().unwrap().iter().find(|c| c["name"] == name) else { continue };
        let gd = Golden { dir: dir.clone(), f16: true };
        let (x, _) = gd.tensor(case, "x");
        let (b, _) = gd.tensor(case, "bias");
        let (y, _) = gd.tensor(case, "y");
        let (plane, ch, act) = (case["plane"].as_u64().unwrap() as usize, case["ch"].as_u64().unwrap() as usize, case["act"].as_u64().unwrap() as i32);
        let h = |v: f32| half::f16::from_f32(v).to_f32();
        let f = |v: f32| match act { 1 => v / (1.0 + (-v).exp()), 2 => 1.0 / (1.0 + (-v).exp()), _ => v };
        let (mut round_sum, mut float_sum) = (0, 0);
        for i in 0..x.len() {
            let s = x[i] + b[(i / plane) % ch];
            if h(f(h(s))) != y[i] { round_sum += 1; }
            if h(f(s)) != y[i] { float_sum += 1; }
        }
        eprintln!("{name}: sum rounded to f16 first -> {round_sum} differ; activation on the float sum -> {float_sum} differ (of {})", x.len());
        for g in gpus() {
            let (xd, bd) = (gd.up(&g, &x), gd.up(&g, &b));
            let enc = g.begin();
            launch(&g, &enc, "cnn_bias_act_f16", &[&xd, &bd], &[x.len() as u32, plane as u32, ch as u32, act as u32], x.len());
            g.submit(enc).unwrap();
            let mut got = vec![0.0; x.len()];
            g.read(&xd, &mut got);
            let (mut eq_round, mut eq_float, mut rtz_sum) = (0, 0, 0);
            for i in 0..x.len() {
                let s = x[i] + b[(i / plane) % ch];
                if got[i].to_bits() == h(f(h(s))).to_bits() { eq_round += 1; }
                if got[i].to_bits() == h(f(s)).to_bits() { eq_float += 1; }
                // round-toward-zero of the sum
                let tz = { let r = h(s); if r.abs() > s.abs() { half::f16::from_bits(half::f16::from_f32(s).to_bits() - 1).to_f32() } else { r } };
                if got[i].to_bits() == h(f(tz)).to_bits() { rtz_sum += 1; }
            }
            eprintln!("   {}: equals rounded-sum ref {eq_round}, float-sum ref {eq_float}, RTZ-sum ref {rtz_sum} (of {})", g.info().name, x.len());
        }
    }
}

/// The shaders' binary16 rounding equals IEEE round-to-nearest-even
/// (`half::f16::from_f32`) on every device: edge cases (ties, the subnormal
/// boundary, 65504/65520, inf, NaN) plus a million random bit patterns.
#[test]
fn f16_rounding_is_round_to_nearest_even() {
    let mut xs: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 65504.0, 65519.99, 65520.0, -65520.0, 1e9, f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
        6.1035156e-5, 6.1035156e-5 * 0.999, 5.9604645e-8, 2.9802322e-8, 2.9802326e-8, 8.940697e-8, 1.0e-30, -3.0e-6];
    // exact ties between neighbouring halves, both parities
    for h in [0x3c00u16, 0x3c01, 0x0400, 0x03ff, 0x0001, 0x7bfe, 0x8401] {
        let a = half::f16::from_bits(h).to_f32();
        let b = half::f16::from_bits(h + 1).to_f32();
        xs.push((a + b) / 2.0);
    }
    let mut r = 0x1234_5678u32;
    for _ in 0..1_000_000 {
        r ^= r << 13;
        r ^= r >> 17;
        r ^= r << 5;
        xs.push(f32::from_bits(r & 0xc7ff_ffff)); // exponents up to ~2^16, both signs
    }
    for g in gpus() {
        let xd = g.upload(&xs);
        let yd = g.upload_f16(&vec![0.0; xs.len()]);
        let enc = g.begin();
        launch(&g, &enc, "vk_round_f16_f16", &[&xd, &yd], &[xs.len() as u32], xs.len());
        g.submit(enc).unwrap();
        let mut raw = vec![0u8; xs.len() * 2];
        g.read_bytes(&yd, 0, &mut raw).unwrap();
        let mut bad = vec![];
        for (i, x) in xs.iter().enumerate() {
            let got = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
            let want = half::f16::from_f32(*x).to_bits();
            let both_nan = half::f16::from_bits(got).is_nan() && x.is_nan();
            if got != want && !both_nan {
                bad.push(format!("{x:e}: got {got:#06x} want {want:#06x}"));
            }
        }
        assert!(bad.is_empty(), "{}: {} of {} roundings differ, e.g. {:?}", g.info().name, bad.len(), xs.len(), &bad[..bad.len().min(5)]);
        eprintln!("{}: {} f32 -> f16 roundings match IEEE RNE", g.info().name, xs.len());
    }
}

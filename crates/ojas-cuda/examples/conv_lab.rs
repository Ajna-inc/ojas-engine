//! Kernel lab: compile an fp16 conv kernel variant from a file (entry `lab_conv`,
//! same signature and launch as `cnn_conv_f16`) under several `#define` knob
//! sets, and time it on representative yolo11n layers.
//!
//! `conv_lab <src.c> <batch> [-DNAME=VAL,...]...`  (each arg = one variant)

use std::time::Instant;

use ojas_core::{Device, KernelRuntime};
use ojas_cpu::cpu_cnn::{conv2d, Act, ConvShape};
use ojas_cuda::CudaGpu;

// (cin, cout, h, w, k, s, p)
const LAYERS: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
    (64, 64, 80, 80, 3, 1, 1),
    (256, 64, 20, 20, 3, 1, 1),
    (96, 128, 80, 80, 1, 1, 0),
    (64, 64, 160, 160, 3, 2, 1),
    (16, 32, 320, 320, 3, 2, 1),
    (192, 128, 40, 40, 1, 1, 0),
    (128, 128, 80, 80, 3, 2, 1),
    (384, 256, 20, 20, 1, 1, 0),
];

fn tvec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }).collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let src = std::fs::read_to_string(&args[1])?;
    let n: usize = args[2].parse()?;
    let variants: Vec<String> = if args.len() > 3 { args[3..].to_vec() } else { vec![String::new()] };
    let g = CudaGpu::new(0)?;
    for v in &variants {
        let defs: String = v.split(';').filter(|d| !d.is_empty()).map(|d| {
            let d = d.trim_start_matches("-D");
            let (k, val) = d.split_once('=').unwrap_or((d, ""));
            format!("#define {k} {val}\n")
        }).collect();
        let full: &'static str = Box::leak(format!("{defs}{src}").into_boxed_str());
        let f = g.pipeline(full, "lab_conv")?;
        let mut line = format!("{:<40}", if v.is_empty() { "baseline" } else { v });
        let mut total = 0.0;
        for &(cin, cout, h, w, k, s, p) in LAYERS {
            let (oh, ow) = ((h + 2 * p - k) / s + 1, (w + 2 * p - k) / s + 1);
            let j = cin * k * k;
            let j16 = j.div_ceil(16) * 16;
            let c16 = j16 / 16;
            let groups = cout.div_ceil(128) * 8;
            let wt = tvec(cout * j, 7, 1.0 / (j as f32).sqrt());
            // weight matrix [groups*16 rows x j16 cols] as f16 bits, then lane words
            // KRSC patch order j' = (ky*k + kx)*cin + ic  <-  NCHW j = ic*k*k + ky*k + kx
            let wbits = |r: usize, c: usize| -> u32 {
                if r >= cout || c >= j { return 0; }
                let (kyx, ic) = (c / cin, c % cin);
                half::f16::from_f32(wt[r * j + ic * k * k + kyx]).to_bits() as u32
            };
            let mut wpk = vec![0u32; groups * c16 * 32 * 4];
            let transposed = std::env::var("TRANSPOSED").is_ok();
            let ogroups = cout.div_ceil(8).div_ceil(8) * 8 + 8;
            for gi in 0..groups {
                for ci in 0..c16 {
                    for lane in 0..32 {
                        let (rr, qq) = (lane / 4, lane % 4);
                        let (r0, r1) = (gi * 16 + rr, gi * 16 + rr + 8);
                        let (k0, k1) = (ci * 16 + 2 * qq, ci * 16 + 8 + 2 * qq);
                        let o = ((gi * c16 + ci) * 32 + lane) * 4;
                        wpk[o] = wbits(r0, k0) | wbits(r0, k0 + 1) << 16;
                        wpk[o + 1] = wbits(r1, k0) | wbits(r1, k0 + 1) << 16;
                        wpk[o + 2] = wbits(r0, k1) | wbits(r0, k1 + 1) << 16;
                        wpk[o + 3] = wbits(r1, k1) | wbits(r1, k1 + 1) << 16;
                    }
                }
            }
            if transposed {
                // wtk[((og*c16 + c)*32 + lane)*2 + w]: output R of group og, taps {2Q.. | 2Q+8..}
                wpk = vec![0u32; ogroups * c16 * 64];
                for og in 0..ogroups { for ci in 0..c16 { for lane in 0..32 {
                    let (rr, qq) = (lane / 4, lane % 4);
                    let oc = og * 8 + rr;
                    let (k0, k1) = (ci * 16 + 2 * qq, ci * 16 + 8 + 2 * qq);
                    let o = ((og * c16 + ci) * 32 + lane) * 2;
                    wpk[o] = wbits(oc, k0) | wbits(oc, k0 + 1) << 16;
                    wpk[o + 1] = wbits(oc, k1) | wbits(oc, k1 + 1) << 16;
                } } }
            }
            // zero-padded NHWC input (+1 spare row/col so word reads stay in bounds)
            let x = tvec(n * cin * h * w, 3, 1.0);
            let (hp, wpw) = (h + 2 * p + 1, w + 2 * p + 1);
            let img = hp * wpw * cin;
            let mut xp = vec![0.0f32; n * img];
            for i in 0..n {
                for ic in 0..cin {
                    for r in 0..h {
                        for cc in 0..w {
                            xp[i * img + ((r + p) * wpw + cc + p) * cin + ic] = x[((i * cin + ic) * h + r) * w + cc];
                        }
                    }
                }
            }
            let opix = oh * ow;
            let npix = std::env::var("NPIX").ok().and_then(|v| v.parse().ok()).unwrap_or(64usize);
            let tpi = opix.div_ceil(npix);
            let mut tpo = vec![0u32; tpi * npix];
            for pp in 0..opix {
                tpo[pp] = (((pp / ow) * s * wpw + (pp % ow) * s) * cin) as u32;
            }
            let bits = |v: &[u32]| v.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>();
            let lane_b = std::env::var("LANE_B").is_ok();
            let xbuf: Vec<f32> = if !lane_b { xp.iter().map(|&v| v).collect() } else {
                // bw[((blk*c16 + c)*8 + tt)*32 + lane)*2 + w] = halves (X[j][pix], X[j+1][pix])
                let blocks = n * tpi;
                let hb = |i: usize, pix: usize, jj: usize| -> u32 {
                    if jj >= j || pix >= opix { return 0; }
                    let po = tpo[pix] as usize;
                    let (kyx, ic) = (jj / cin, jj % cin);
                    let (ky, kx) = (kyx / k, kyx % k);
                    half::f16::from_f32(xp[i * img + po + (ky * wpw + kx) * cin + ic]).to_bits() as u32
                };
                let mut bw = vec![0u32; blocks * c16 * 8 * 64];
                for blk in 0..blocks {
                    let (i, t) = (blk / tpi, blk % tpi);
                    for c in 0..c16 { for tt in 0..8 { for lane in 0..32 {
                        let (rr, qq) = (lane / 4, lane % 4);
                        let pix = t * npix + tt * 8 + rr;
                        let j0 = c * 16 + 2 * qq;
                        let o = (((blk * c16 + c) * 8 + tt) * 32 + lane) * 2;
                        bw[o] = hb(i, pix, j0) | hb(i, pix, j0 + 1) << 16;
                        bw[o + 1] = hb(i, pix, j0 + 8) | hb(i, pix, j0 + 9) << 16;
                    } } }
                }
                bits(&bw)
            };
            let (xd, wd, yd, jd) = (
                if lane_b { g.upload(&xbuf) } else { g.upload_f16(&xp) },
                g.upload(&bits(&wpk)),
                g.upload_f16(&vec![0.0; n * cout * oh * ow]),
                g.upload(&bits(&tpo)),
            );
            // adaptive warps (v8+): lg = log2(groups per block), groups = ceil(cout/16)
            let gcount = cout.div_ceil(16);
            let lg = if std::env::var("ADAPT").is_ok() { (gcount.next_power_of_two().min(8)).trailing_zeros() as usize } else { 3 };
            let (ngb, tpb) = (1usize << lg, 8usize >> lg);
            let consts = [img, wpw * cin, k * cin, cout, opix, c16, npix, n, lg].map(|c| c as u32);
            let tt_groups: usize = std::env::var("TTG").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
            let grid = if transposed {
                [(n * tpi) as u32, cout.div_ceil(8).div_ceil(tt_groups) as u32, 1]
            } else if std::env::var("ADAPT").is_ok() {
                [(n * tpi).div_ceil(tpb) as u32, gcount.div_ceil(ngb) as u32, 1]
            } else {
                [(n * tpi) as u32, cout.div_ceil(128) as u32, 1]
            };
            let unfold_src = std::env::var("UNFOLD").ok().map(|p| &*Box::leak(std::fs::read_to_string(p).unwrap().into_boxed_str()));
            let unfold = unfold_src.map(|src| g.pipeline(src, "lab_unfold").unwrap());
            let xpd = g.upload_f16(&xp);
            let unfold_consts = [img, wpw * cin, k * cin, c16, tpi, n * tpi, npix, opix, j].map(|c| c as u32);
            let go = || {
                if let Some(uf) = &unfold {
                    let grid = if std::env::var("UNFOLD_TILE").is_ok() { [(n * tpi) as u32, 1, 1] } else { [(n * tpi * c16) as u32, 1, 1] };
                    g.dispatch_pipeline(uf, &[(&xpd, 0), (&xd, 0), (&jd, 0)], &unfold_consts, grid, [256, 1, 1], 0).unwrap();
                }
                g.dispatch_pipeline(&f, &[(&xd, 0), (&wd, 0), (&yd, 0), (&jd, 0)], &consts, grid, [256, 1, 1], 0).unwrap();
                g.submit(g.begin()).unwrap();
            };
            for _ in 0..3 { go(); }
            let mut ts: Vec<f64> = (0..11).map(|_| { let t = Instant::now(); go(); t.elapsed().as_secs_f64() * 1e3 }).collect();
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // image 0 vs the CPU oracle on the same f16-rounded inputs
            let r16 = |v: &[f32]| v.iter().map(|&a| half::f16::from_f32(a).to_f32()).collect::<Vec<_>>();
            let plane = cout * oh * ow;
            let mut nhwc = vec![0.0f32; n * plane];
            g.read(&yd, &mut nhwc);
            let mut got = vec![0.0f32; n * plane];
            for i in 0..n { for pp in 0..oh * ow { for oc in 0..cout {
                got[(i * cout + oc) * oh * ow + pp] = nhwc[(i * oh * ow + pp) * cout + oc];
            } } }
            let sh = ConvShape { n: 1, cin, h, w, cout, kh: k, kw: k, group: 1, stride: [s; 2], pads: [p; 4], dilation: [1, 1] };
            let mut want = vec![0.0f32; plane];
            conv2d(&r16(&x[..cin * h * w]), &r16(&wt), None, &sh, Act::None, 4, &mut want);
            let err = got[..plane].iter().zip(r16(&want)).map(|(a, b)| (a - b).abs() / b.abs().max(1e-2)).fold(0.0f32, f32::max);
            let last = &got[(n - 1) * plane..];
            let tail_ok = last.iter().any(|v| *v != 0.0);
            let mark = if err < 2e-3 && tail_ok { "" } else { "!" };
            line += &format!(" {:>7.3}{mark:1}", ts[5]);
            total += ts[5];
        }
        println!("{line}  | sum {total:.3} ms");
    }
    Ok(())
}

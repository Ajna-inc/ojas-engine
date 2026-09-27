//! Isolated conv microbenchmark. Min-of-N timing so a loaded machine still
//! reveals kernel capability. `cargo run -p ojas-cpu --release --example conv_bench`

use ojas_cpu::cpu_cnn::{conv2d, Act, ConvShape};

fn bench(name: &str, s: &ConvShape, threads: usize, reps: usize) {
    let cin_g = s.cin / s.group;
    let x: Vec<f32> = (0..s.n * s.cin * s.h * s.w).map(|i| (i % 17) as f32 * 0.1).collect();
    let w: Vec<f32> = (0..s.cout * cin_g * s.kh * s.kw).map(|i| (i % 13) as f32 * 0.01).collect();
    let bias: Vec<f32> = (0..s.cout).map(|i| i as f32 * 0.001).collect();
    let (oh, ow) = s.out_hw();
    let mut out = vec![0.0f32; s.n * s.cout * oh * ow];
    // warmup
    conv2d(&x, &w, Some(&bias), s, Act::Silu, threads, &mut out);
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        conv2d(&x, &w, Some(&bias), s, Act::Silu, threads, &mut out);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let flops = 2.0 * (s.n * s.cout * oh * ow) as f64 * (cin_g * s.kh * s.kw) as f64;
    println!(
        "{name:<28} t={threads}  min {:8.3} ms   {:6.1} GFLOP/s",
        best * 1e3,
        flops / best / 1e9
    );
}

fn main() {
    let shapes = [
        ("stem 3->16 k3s2 320", ConvShape { n: 1, cin: 3, h: 640, w: 640, cout: 16, kh: 3, kw: 3, group: 1, stride: [2, 2], pads: [1, 1, 1, 1], dilation: [1, 1] }),
        ("mid 64x64 k3 80", ConvShape { n: 1, cin: 64, h: 80, w: 80, cout: 64, kh: 3, kw: 3, group: 1, stride: [1, 1], pads: [1, 1, 1, 1], dilation: [1, 1] }),
        ("mid 128x128 k3 40", ConvShape { n: 1, cin: 128, h: 40, w: 40, cout: 128, kh: 3, kw: 3, group: 1, stride: [1, 1], pads: [1, 1, 1, 1], dilation: [1, 1] }),
        ("deep 256x256 k3 20", ConvShape { n: 1, cin: 256, h: 20, w: 20, cout: 256, kh: 3, kw: 3, group: 1, stride: [1, 1], pads: [1, 1, 1, 1], dilation: [1, 1] }),
        ("pw 256->128 1x1 40", ConvShape { n: 1, cin: 256, h: 40, w: 40, cout: 128, kh: 1, kw: 1, group: 1, stride: [1, 1], pads: [0; 4], dilation: [1, 1] }),
    ];
    for (name, s) in &shapes {
        for threads in [1, 8] {
            bench(name, s, threads, 15);
        }
    }
}

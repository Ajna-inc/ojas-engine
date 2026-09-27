//! Wall-clock decomposition of a real prefill: engine wall vs GPU busy vs chunks.
//!
//! profile_prefill times each kernel category in its own serial command buffer — the
//! no-gap floor. Under the concurrent encoder that floor no longer predicts the wall,
//! since overlap hides some categories entirely. This measures the real pass:
//!   wall      = what the user waits
//!   gpu busy  = GPUEndTime - GPUStartTime summed over command buffers
//!   the difference = CPU encode + submit + inter-buffer gaps
//! Compared against the server's first_token_ms, the remainder is tokenizer +
//! template + queue + the +1 decode forward + SSE.
//!
//! usage: prefill_bench <gguf> [n_tokens] [prec]

use anyhow::Result;
use ojas_core::Model;

fn main() -> Result<()> {
    let gguf = std::env::args().nth(1).expect("usage: prefill_bench <gguf> [n] [prec]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let prec: u8 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let gpu = ojas_metal::MetalGpu::new()?;
    let mut g = ojas_formats::gguf::Gguf::open(&gguf)?;
    let m = ojas_models::decoder::DecoderGpu::load(&gpu, &mut g, 4096, prec, None, None)?;
    let toks: Vec<u32> = (0..n).map(|i| 100u32 + (i as u32 % 50)).collect();
    // warm
    m.prefill(&toks, 0);

    let one = |m: &ojas_models::decoder::DecoderGpu| -> (f64, f64) {
        let g0 = m.gpu_seconds();
        let t0 = std::time::Instant::now();
        m.prefill(&toks, 0);
        (t0.elapsed().as_secs_f64(), m.gpu_seconds() - g0)
    };

    // Throttle guard: this machine derates under sustained GPU load with no pmset
    // thermal warning — a 10-run series once drifted 1403 -> 990 tok/s monotonically, in
    // GPU-busy as well as wall, swamping every A/B taken across it. The detector is a
    // control that repeats: the first and last samples are compared and the run is
    // declared void if they disagree. Do not quote a wall-clock number from a void run.
    let (first_wall, first_gpu) = one(&m);
    // Keep the pair together: a gpu time from one sample against a wall from another
    // yields a negative off-gpu.
    let (mut best_wall, mut best_gpu) = (first_wall, first_gpu);
    let mut last_wall = first_wall;
    for _ in 0..5 {
        let (wall, gpu_s) = one(&m);
        last_wall = wall;
        if wall < best_wall { best_wall = wall; best_gpu = gpu_s; }
    }
    let drift = (last_wall - first_wall) / first_wall * 100.0;

    println!(
        "prefill {n} tokens: wall {:.1} ms ({:.0} tok/s) | gpu busy {:.1} ms | off-gpu {:.1} ms ({:.0}%)",
        best_wall * 1e3, n as f64 / best_wall,
        best_gpu * 1e3,
        (best_wall - best_gpu) * 1e3,
        (best_wall - best_gpu) / best_wall * 100.0
    );
    println!("control drift first->last: {drift:+.1}%  {}",
        if drift.abs() > 3.0 { "<<< VOID: machine is derating, re-run cool" } else { "(ok)" });
    Ok(())
}

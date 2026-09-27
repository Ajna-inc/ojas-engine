//! Forward timing of the unoptimized training backend, not of a deployment export.
//! passenger_bench coco.pth
use std::time::Instant;
use ojas_learn::{passenger, Tape};
use ojas_learn::cuda::Cuda;
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("passenger_bench coco.pth"))?;
    let be = Cuda::new(0)?;
    for stage in [1, 2, 3] {
        let st = passenger::init_stage(&be, &path, 1, stage)?;
        let params: usize = st.params.values().map(|p| p.shape.iter().product::<usize>()).sum();
        for size in [160, 224] {
            let input = vec![0.5; 3 * size * size];
            let mut times = vec![];
            for i in 0..60 {
                let now = Instant::now();
                let mut t = Tape::new(&be);
                let x = t.input(&input, &[1, 3, size, size]);
                let logits = passenger::forward(&mut t, &st, x)?;
                let values = t.value(logits); // synchronize by downloading output
                anyhow::ensure!(values.iter().all(|v| v.is_finite()), "nonfinite logits");
                if i >= 10 { times.push(now.elapsed().as_secs_f64() * 1000.); }
            }
            times.sort_by(f64::total_cmp);
            println!("{}", serde_json::json!({"stage":stage,"size":size,"parameters":params,"fp32_parameter_bytes":params*4,"p50_ms":times[25],"p95_ms":times[47],"iterations":50,"scope":"training tape backend, batch 1, upload+forward+download, no image preprocessing; not optimized deployment export"}));
        }
    }
    Ok(())
}

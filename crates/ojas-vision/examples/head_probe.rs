//! Debug probe: run a detector's graph on an image through its own preprocessing
//! and print raw head statistics (no decode).

use anyhow::Result;
use ojas_vision::exec_cpu::CpuExecutor;
use std::collections::HashMap;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = ojas_formats::onnx::load(&args[1])?;
    let mut g = ojas_vision::import(&model, &HashMap::new())?;
    ojas_vision::passes::optimize(&mut g);
    let (w, h, rgb) = ojas_cpu::vit_preprocess::decode_rgb8_path(&args[2])?;
    let target = g.shape(g.inputs[0])[2];
    let mut input = vec![0f32; 3 * target * target];
    let lb = ojas_vision::pre::letterbox_yolox_bgr_into(&rgb, w, h, target, &mut input);
    println!("letterbox: scale {:.3} pads ({},{})  input min/max {:.1}/{:.1}",
        lb.scale, lb.pad_x, lb.pad_y,
        input.iter().cloned().fold(f32::MAX, f32::min),
        input.iter().cloned().fold(f32::MIN, f32::max));
    let mut exec = CpuExecutor::new(&g, 8);
    let out = exec.run(&g, &[&input])?.remove(0);
    let rows = out.len() / 6;
    let mut max_obj = f32::MIN;
    let mut over = 0;
    for a in 0..rows {
        let obj = out[a * 6 + 4];
        if obj > max_obj {
            max_obj = obj;
        }
        if obj > 0.02 {
            over += 1;
        }
    }
    println!("rows {rows}  max obj {max_obj:.4}  rows>0.02 {over}");
    Ok(())
}

//! Read a fixed-length text line with the native Ojas CPU ONNX executor.
//!
//! cargo run --release -p ojas-vision --example read_fixed_text -- model.onnx image.png

use std::{path::Path, thread};

use anyhow::{ensure, Context, Result};
use ojas_vision::exec_cpu::CpuExecutor;

const WIDTH: usize = 200;
const HEIGHT: usize = 50;
const PATCH: usize = 4;
const ROWS: usize = HEIGHT / PATCH;
const COLS: usize = WIDTH / PATCH;
const CHARACTERS: usize = 6;
const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const SHIFTS: [(isize, isize); 9] = [
    (0, 0),
    (-2, 0),
    (2, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

fn load_patches(path: &Path) -> Result<Vec<f32>> {
    let image = image::open(path)
        .with_context(|| format!("could not decode {}", path.display()))?
        .into_rgb8();
    ensure!(
        image.dimensions() == (WIDTH as u32, HEIGHT as u32),
        "{} must be {WIDTH} × {HEIGHT}",
        path.display()
    );

    let mut patches = Vec::with_capacity(SHIFTS.len() * PATCH * PATCH * ROWS * COLS);
    for (dx, dy) in SHIFTS {
        for patch_y in 0..PATCH {
            for patch_x in 0..PATCH {
                for row in 0..ROWS {
                    for column in 0..COLS {
                        let x = (column * PATCH + patch_x) as isize - dx;
                        let y = (row * PATCH + patch_y) as isize - dy;
                        let red = if (0..WIDTH as isize).contains(&x)
                            && (0..HEIGHT as isize).contains(&y)
                        {
                            image.get_pixel(x as u32, y as u32)[0]
                        } else {
                            255
                        };
                        patches.push((1.0 - f32::from(red) / 205.0).max(0.0));
                    }
                }
            }
        }
    }
    Ok(patches)
}

fn decode(probabilities: &[f32]) -> (String, f32) {
    let mut text = String::with_capacity(CHARACTERS);
    let mut score = 1.0;

    for row in probabilities.chunks_exact(ALPHABET.len()) {
        let mut best = 0;
        for (index, &value) in row.iter().enumerate().skip(1) {
            if value > row[best] {
                best = index;
            }
        }
        text.push(ALPHABET[best] as char);
        score *= row[best];
    }
    (text, score)
}

fn main() -> Result<()> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        paths.len() >= 2,
        "usage: read_fixed_text MODEL.onnx IMAGE.png [MORE.png ...]"
    );

    let model = ojas_formats::onnx::parse(&std::fs::read(&paths[0])?)?;
    let mut graph = ojas_vision::import(&model, &Default::default())?;
    ojas_vision::passes::optimize(&mut graph);
    ensure!(
        graph.inputs.len() == 1
            && graph.shape(graph.inputs[0]) == [SHIFTS.len(), PATCH * PATCH, ROWS, COLS],
        "unexpected model input, use the Ojas-specific OCR export"
    );
    ensure!(
        graph.outputs.len() == 2
            && graph.shape(graph.outputs[0]) == [1, CHARACTERS, ALPHABET.len()]
            && graph.shape(graph.outputs[1]) == [1],
        "unexpected model outputs, expected character probabilities and a review cutoff"
    );

    let threads = thread::available_parallelism().map_or(1, |count| count.get().min(4));
    let mut executor = CpuExecutor::new(&graph, threads);
    for path in &paths[1..] {
        let path = Path::new(path);
        let patches = load_patches(path)?;
        let outputs = executor.run(&graph, &[&patches])?;
        let (text, score) = decode(&outputs[0]);
        let status = if score < outputs[1][0] {
            "REVIEW"
        } else {
            "READ"
        };
        let filename = path.file_name().unwrap_or(path.as_os_str());
        println!(
            "{}\t{text}\t{score:.5}\t{status}",
            filename.to_string_lossy()
        );
    }
    Ok(())
}

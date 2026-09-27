//! Markdown tables from the `score_detections.py --json` files that `training/field/audit_models.sh`
//! writes (`<preset>.<set>.<size>.json`): one row per model × input size, mAP per test set plus the
//! mean of the three field sets and the UVH-26 val score, then per-set detail (AP50, AP75,
//! small / medium / large, AR100) and per-class AP on the field sets.
//!
//! `audit_table <scores_dir> <preset@size> [<preset@size> ...]`
use serde_json::Value;

const SETS: [&str; 3] = ["day", "night", "newcam"];

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 3, "audit_table <scores_dir> <preset@size> ...");
    let dir = std::path::Path::new(&a[1]);
    // Python's json writes NaN for a class with no ground truth, which strict JSON rejects: read it as null
    let load = |p: &str, s: &str, set: &str| -> Option<Value> { serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{p}.{set}.{s}.json"))).ok()?.replace("NaN", "null")).ok() };
    let f = |v: &Option<Value>, k: &str| v.as_ref().and_then(|v| v[k].as_f64()).map_or("—".into(), |x| format!("{x:.3}"));

    println!("| model | size | day | night | new cameras | **Field mean** | UVH-26 val |\n|---|---:|---:|---:|---:|---:|---:|");
    for m in &a[2..] {
        let (p, s) = m.split_once('@').unwrap();
        let g: Vec<Option<Value>> = SETS.iter().map(|set| load(p, s, set)).collect();
        let mean = g.iter().map(|v| v.as_ref().and_then(|v| v["mAP"].as_f64())).collect::<Option<Vec<f64>>>().map(|v| v.iter().sum::<f64>() / v.len() as f64);
        println!("| {p} | {s} | {} | {} | {} | **{}** | {} |", f(&g[0], "mAP"), f(&g[1], "mAP"), f(&g[2], "mAP"), mean.map_or("—".into(), |x| format!("{x:.3}")), f(&load(p, s, "uvh"), "mAP"));
    }
    for set in SETS {
        println!("\n**{set}** — detail\n\n| model | size | mAP | AP50 | AP75 | small | medium | large | AR100 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|");
        for m in &a[2..] {
            let (p, s) = m.split_once('@').unwrap();
            let v = load(p, s, set);
            println!("| {p} | {s} | {} | {} | {} | {} | {} | {} | {} |", f(&v, "mAP"), f(&v, "AP50"), f(&v, "AP75"), f(&v, "AP_small"), f(&v, "AP_medium"), f(&v, "AP_large"), f(&v, "AR100"));
        }
    }
    // per-class AP, mean over the three field sets where the class has ground truth
    let classes = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"];
    println!("\n**per class** — AP averaged over the Field sets that contain the class\n\n| model | size | {} |\n|---|---:|{}", classes.join(" | "), "---:|".repeat(classes.len()));
    for m in &a[2..] {
        let (p, s) = m.split_once('@').unwrap();
        let g: Vec<Value> = SETS.iter().filter_map(|set| load(p, s, set)).collect();
        let cells: Vec<String> = classes
            .iter()
            .map(|c| {
                let v: Vec<f64> = g.iter().filter_map(|v| v["per_class"][c]["AP"].as_f64()).filter(|x| x.is_finite() && *x >= 0.0).collect();
                if v.is_empty() { "—".into() } else { format!("{:.2}", v.iter().sum::<f64>() / v.len() as f64) }
            })
            .collect();
        println!("| {p} | {s} | {} |", cells.join(" | "));
    }
    Ok(())
}

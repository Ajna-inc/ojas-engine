//! Class agreement between two annotations of the same images.
//!
//! UVH-26 ships two labellings of one validation set: MV (majority vote) and ST (STAPLE
//! consensus). The published detector is trained on MV and scored on ST, so wherever the two
//! labellings disagree on a box, a model that reproduces MV perfectly is scored wrong on ST: the
//! MV→ST agreement rate is a ceiling on ST accuracy for any MV-trained model.
//!
//! Boxes are matched on the same image at IoU ≥ 0.7, both directions, one match per box. Reports
//! overall agreement, agreement within the passenger family, and the full confusion between the
//! two labellings.
//! `label_agreement <a.json> <b.json> [class ids, e.g. 1,2,3,4]`
use std::collections::HashMap;

const NAMES: [&str; 15] = ["-", "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"];

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax1, ay1, bx1, by1) = (a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]);
    let iw = (ax1.min(bx1) - a[0].max(b[0])).max(0.0);
    let ih = (ay1.min(by1) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = a[2] * a[3] + b[2] * b[3] - i;
    if u <= 0.0 { 0.0 } else { i / u }
}

fn load(path: &str) -> anyhow::Result<HashMap<i64, Vec<(usize, [f32; 4])>>> {
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut by: HashMap<i64, Vec<(usize, [f32; 4])>> = HashMap::new();
    for a in j["annotations"].as_array().into_iter().flatten() {
        let img = a["image_id"].as_i64().unwrap_or(-1);
        let c = a["category_id"].as_u64().unwrap_or(0) as usize;
        let b = a["bbox"].as_array().map(|v| [v[0].as_f64().unwrap_or(0.0) as f32, v[1].as_f64().unwrap_or(0.0) as f32, v[2].as_f64().unwrap_or(0.0) as f32, v[3].as_f64().unwrap_or(0.0) as f32]);
        if let Some(b) = b {
            by.entry(img).or_default().push((c, b));
        }
    }
    Ok(by)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() > 2, "label_agreement <a.json> <b.json> [class ids]");
    let focus: Vec<usize> = a.get(3).map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect()).unwrap_or_else(|| vec![1, 2, 3, 4]);
    let (la, lb) = (load(&a[1])?, load(&a[2])?);
    let name = |p: &str| p.rsplit('/').next().unwrap_or(p).to_string();
    let (na, nb) = (name(&a[1]), name(&a[2]));

    let mut confusion: HashMap<(usize, usize), usize> = HashMap::new();
    let (mut images, mut matched, mut agree) = (0usize, 0usize, 0usize);
    let (mut only_a, mut only_b) = (0usize, 0usize);
    for (img, boxes_a) in &la {
        let Some(boxes_b) = lb.get(img) else { continue };
        images += 1;
        // greedy one-to-one at IoU ≥ 0.7, best pairs first
        let mut pairs: Vec<(f32, usize, usize)> = vec![];
        for (i, (_, ba)) in boxes_a.iter().enumerate() {
            for (j, (_, bb)) in boxes_b.iter().enumerate() {
                let v = iou(*ba, *bb);
                if v >= 0.7 {
                    pairs.push((v, i, j));
                }
            }
        }
        pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
        let (mut ua, mut ub) = (vec![false; boxes_a.len()], vec![false; boxes_b.len()]);
        for (_, i, j) in pairs {
            if ua[i] || ub[j] {
                continue;
            }
            ua[i] = true;
            ub[j] = true;
            matched += 1;
            let (ca, cb) = (boxes_a[i].0, boxes_b[j].0);
            *confusion.entry((ca, cb)).or_default() += 1;
            if ca == cb {
                agree += 1;
            }
        }
        only_a += ua.iter().filter(|u| !**u).count();
        only_b += ub.iter().filter(|u| !**u).count();
    }
    println!("# {na} vs {nb}\n");
    println!("{images} shared images; {matched} boxes matched at IoU ≥ 0.7; {only_a} boxes only in {na}, {only_b} only in {nb}\n");
    println!("**All classes: {agree} / {matched} agree = {:.2} %**\n", 100.0 * agree as f64 / matched.max(1) as f64);

    // within the focus family: boxes both labellings put in the family
    let in_focus = |c: usize| focus.contains(&c);
    let (mut fam_total, mut fam_agree) = (0usize, 0usize);
    let mut per_class: HashMap<usize, (usize, usize)> = HashMap::new(); // a-label → (n, agree)
    for (&(ca, cb), &n) in &confusion {
        if in_focus(ca) && in_focus(cb) {
            fam_total += n;
            let e = per_class.entry(ca).or_default();
            e.0 += n;
            if ca == cb {
                fam_agree += n;
                e.1 += n;
            }
        }
    }
    let fam_names: Vec<&str> = focus.iter().map(|c| NAMES[*c]).collect();
    println!("**Within {{{}}}: {fam_agree} / {fam_total} agree = {:.2} %** — the ceiling on {nb}-scored subtype accuracy for a model that learned {na} perfectly\n", fam_names.join(", "), 100.0 * fam_agree as f64 / fam_total.max(1) as f64);
    println!("| {na} label | boxes | {nb} agrees | leaks to |");
    println!("|---|---:|---:|---|");
    for c in &focus {
        let (n, ag) = per_class.get(c).copied().unwrap_or((0, 0));
        let mut leaks: Vec<(usize, usize)> = confusion.iter().filter(|((x, y), _)| *x == *c && *y != *c && in_focus(*y)).map(|((_, y), n)| (*y, *n)).collect();
        leaks.sort_by_key(|x| std::cmp::Reverse(x.1));
        let l: Vec<String> = leaks.iter().map(|(y, n)| format!("{} {n}", NAMES[*y])).collect();
        println!("| {} | {n} | {:.1} % | {} |", NAMES[*c], 100.0 * ag as f64 / n.max(1) as f64, l.join(", "));
    }
    // and leaving the family altogether
    let out_of_family: usize = confusion.iter().filter(|((x, y), _)| in_focus(*x) && !in_focus(*y)).map(|(_, n)| n).sum();
    println!("\n{out_of_family} boxes {na} calls a family member that {nb} calls something outside the family.");
    Ok(())
}

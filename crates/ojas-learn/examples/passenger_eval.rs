//! Paired classification diagnostic and fixed residual fusion on actual detector boxes.
//! passenger_eval detector.pth expert.safetensors annotations.json image_root out.json [images=256] [skip=0]
//!
//! `PASSENGER_PRIOR=h,s,u,m` (class counts or frequencies) multiplies the expert's
//! probabilities by that prior before argmax, abstention and fusion — logit
//! adjustment for an expert trained with a balanced sampler, which otherwise
//! predicts as if the four classes were equally common. `PASSENGER_PRIOR_POWER`
//! (default 1) scales it. `PASSENGER_ALPHA` (default 0.25) and `PASSENGER_ABSTAIN`
//! (default 0.6) override the fusion weight and the expert-confidence floor below
//! which fusion abstains. Unset, behaviour is unchanged.
use anyhow::{ensure, Result};
use ojas_learn::{passenger, Tape};
use ojas_learn::cuda::Cuda;
use ojas_learn::data::{load_coco, loader};
use ojas_learn::models::rtdetr::{Config, RtDetr, Store};
use ojas_learn::eval::{coco_map, postprocess};
use std::sync::Arc;

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let inter = (a[2].min(b[2])-a[0].max(b[0])).max(0.)*(a[3].min(b[3])-a[1].max(b[1])).max(0.);
    inter / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter).max(1e-8)
}

/// Bounded residual, applied only inside the predicted passenger family. Centred log probabilities
/// do not assume that detector sigmoid logits and classifier logits share a calibrated scale.
fn fuse(row: &mut [f32], p: [f32; 4], alpha: f32) { fuse_with(row, p, alpha, 0.6) }

fn fuse_with(row: &mut [f32], p: [f32; 4], alpha: f32, abstain: f32) {
    if alpha == 0. || p.iter().copied().fold(0., f32::max) < abstain { return; }
    let logs = p.map(|v| v.max(1e-6).ln());
    let mean = logs.iter().sum::<f32>() / 4.;
    let evidence = logs.map(|v| (v - mean).clamp(-2., 2.));
    let center = evidence.iter().sum::<f32>() / 4.;
    for c in 0..4 { row[c + 1] += alpha * (evidence[c] - center); }
}

/// Optional class prior from the environment, normalised to sum to one.
fn prior_from_env() -> Result<Option<([f32; 4], f32)>> {
    let Ok(spec) = std::env::var("PASSENGER_PRIOR") else { return Ok(None) };
    let v: Vec<f32> = spec.split(',').map(|x| x.trim().parse::<f32>()).collect::<Result<_, _>>()?;
    ensure!(v.len() == 4 && v.iter().all(|x| *x > 0.), "PASSENGER_PRIOR needs four positive numbers");
    let sum: f32 = v.iter().sum();
    let power: f32 = std::env::var("PASSENGER_PRIOR_POWER").ok().map(|s| s.parse()).transpose()?.unwrap_or(1.);
    Ok(Some(([v[0] / sum, v[1] / sum, v[2] / sum, v[3] / sum], power)))
}

/// p ∝ p · prior^power, renormalised.
fn apply_prior(p: [f32; 4], prior: &[f32; 4], power: f32) -> [f32; 4] {
    let mut q = [0.; 4];
    for c in 0..4 { q[c] = p[c] * prior[c].powf(power); }
    let sum: f32 = q.iter().sum();
    q.map(|x| x / sum.max(1e-12))
}

fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let prior = prior_from_env()?;
    let alpha: f32 = std::env::var("PASSENGER_ALPHA").ok().map(|s| s.parse()).transpose()?.unwrap_or(0.25);
    let abstain: f32 = std::env::var("PASSENGER_ABSTAIN").ok().map(|s| s.parse()).transpose()?.unwrap_or(0.6);
    if alpha != 0.25 || abstain != 0.6 { eprintln!("fusion alpha {alpha} abstain {abstain}"); }
    if let Some((pr, pw)) = &prior { eprintln!("expert prior {pr:?} power {pw}"); }
    ensure!(a.len()>=6,"passenger_eval detector.pth expert.safetensors annotations.json image_root out.json [images]");
    ensure!(!std::path::Path::new(&a[5]).exists(), "output already exists");
    let limit: usize = a.get(6).map(|s| s.parse()).transpose()?.unwrap_or(256);
    let skip: usize = a.get(7).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let mut samples = load_coco(a[3].as_ref(),a[4].as_ref())?;
    // a fixed pseudo-random image subset, independent of checkpoint and predictions
    samples.sort_by_key(|s| (s.image_id as u64).wrapping_mul(6364136223846793005).wrapping_add(20260920));
    ensure!(skip < samples.len() && limit > 0, "empty evaluation selection");
    samples.drain(..skip);
    samples.truncate(limit);
    let samples = Arc::new(samples);
    let be = Cuda::new(0)?;
    let detector = Store::from_tensors(&be,&ojas_formats::pth::load(&std::fs::read(&a[1])?)?,"ema.module.");
    let expert = Store::from_safetensors(&be,&a[2],"model.")?;
    let model = RtDetr {cfg:Config::r18vd(15),st:&detector,train:false,var:Default::default()};
    let (mut total,mut matched,mut baseline,mut candidate,mut calls,mut changed,mut fixed,mut broken)=(0,0,0,0,0,0,0,0);
    let mut before=[[0usize;15];4];
    let mut after=[[0usize;15];4];
    let mut per_image=vec![];
    let mut proposals=vec![];
    let (mut all_gt, mut detector_dets, mut fused_dets) = (vec![], vec![], vec![]);
    let (mut fused_correct, mut fused_fixed, mut fused_broken) = (0usize, 0usize, 0usize);
    let (mut expert_matched, mut expert_correct, mut expert_baseline_correct) = (0usize, 0usize, 0usize);
    let mut expert_confusion = [[0usize; 4]; 4];
    for batch in loader(samples.clone(),(0..samples.len()).collect(),8,640,false,12,0) {
        let batch=batch?;
        let mut t=Tape::new(&be);
        let x=t.input(&batch.images,&[batch.samples.len(),3,640,640]);
        let o=model.forward(&mut t,x,None)?;
        let (logits,boxes)=(t.value(o.logits[0]),t.value(o.boxes[0]));
        drop(t);
        for (bi,&si) in batch.samples.iter().enumerate() {
            let sample=&samples[si];
            let image=image::open(&sample.path)?.to_rgb8();
            let original_logits = &logits[bi*300*15..(bi+1)*300*15];
            let original_boxes = &boxes[bi*300*4..(bi+1)*300*4];
            let mut fused_logits = original_logits.to_vec();
            let mut predictions=vec![];
            for q in 0..300 {
                let row=&logits[(bi*300+q)*15..(bi*300+q+1)*15];
                let label=(1..15).max_by(|&i,&j|row[i].total_cmp(&row[j])).unwrap();
                let score=1./(1.+(-row[label]).exp());
                if score<0.3 {continue;}
                let b=&boxes[(bi*300+q)*4..(bi*300+q+1)*4];
                let xyxy=[(b[0]-b[2]/2.)*sample.width as f32,(b[1]-b[3]/2.)*sample.height as f32,(b[0]+b[2]/2.)*sample.width as f32,(b[1]+b[3]/2.)*sample.height as f32];
                predictions.push((q,label,score,xyxy,label,label));
            }
            let mut crop_inputs=vec![];
            let mut raw_expert = std::collections::HashMap::new();
            let mut crop_indices=vec![];
            for (i,(_,label,_,b,_,_)) in predictions.iter().enumerate() {
                let (w,h)=(b[2]-b[0],b[3]-b[1]);
                if !(1..=4).contains(label) || w<48. || h<48. {continue;}
                let x0=(b[0]-0.15*w).clamp(0.,image.width() as f32) as u32;
                let y0=(b[1]-0.15*h).clamp(0.,image.height() as f32) as u32;
                let x1=(b[2]+0.15*w).clamp(0.,image.width() as f32) as u32;
                let y1=(b[3]+0.15*h).clamp(0.,image.height() as f32) as u32;
                if x1<=x0 || y1<=y0 {continue;}
                let crop=image::imageops::crop_imm(&image,x0,y0,x1-x0,y1-y0).to_image();
                crop_inputs.push(passenger::preprocess(&crop,224,false,1.));
                crop_indices.push(i);
            }
            calls+=crop_indices.len();
            for (inputs,indices) in crop_inputs.chunks(16).zip(crop_indices.chunks(16)) {
                let images:Vec<_>=inputs.iter().flatten().copied().collect();
                let mut ct=Tape::new(&be);
                let x=ct.input(&images,&[indices.len(),3,224,224]);
                let y=passenger::forward(&mut ct,&expert,x)?;
                let probs=passenger::probabilities(&ct.value(y));
                for (&i,p) in indices.iter().zip(probs) {
                    ensure!(p.iter().all(|v|v.is_finite()),"nonfinite expert output");
                    let p = match &prior { Some((pr, pw)) => apply_prior(p, pr, *pw), None => p };
                    let c=(0..4).max_by(|&i,&j|p[i].total_cmp(&p[j])).unwrap();
                    if p[c]>=0.6 {predictions[i].4=c+1;}
                    let q = predictions[i].0;
                    raw_expert.insert(q, c + 1);
                    fuse_with(&mut fused_logits[q*15..(q+1)*15], p, alpha, abstain);
                    proposals.push(serde_json::json!({"image_id":sample.image_id,"query":predictions[i].0,"detector_class":predictions[i].1,"expert_probabilities":p,"routed_class":predictions[i].4}));
                }
            }
            for prediction in &mut predictions {
                let row = &fused_logits[prediction.0*15..(prediction.0+1)*15];
                prediction.5 = (1..15).max_by(|&i,&j|row[i].total_cmp(&row[j])).unwrap();
            }
            predictions.sort_by(|a,b|b.2.total_cmp(&a.2));
            let gt:Vec<_>=sample.boxes.iter().map(|&(c,b)|(c,[b[0],b[1],b[0]+b[2],b[1]+b[3]])).collect();
            detector_dets.push(postprocess(original_logits, original_boxes, 15, 300, sample.width as f32, sample.height as f32));
            fused_dets.push(postprocess(&fused_logits, original_boxes, 15, 300, sample.width as f32, sample.height as f32));
            all_gt.push(gt.clone());
            total+=gt.iter().filter(|g|(1..=4).contains(&g.0)).count();
            let mut used=vec![false;gt.len()];
            let (mut im_matched,mut im_before,mut im_after,mut im_fused)=(0,0,0,0);
            let (mut im_expert_matched,mut im_expert_correct,mut im_expert_baseline)=(0,0,0);
            for (q,label,_,b,new_label,fused_label) in predictions {
                let mut best=(0.5,None);
                for (j,g) in gt.iter().enumerate() {
                    if !used[j] {let overlap=iou(b,g.1);if overlap>=best.0 {best=(overlap,Some(j));}}
                }
                if let Some(j)=best.1 {
                    used[j]=true;
                    let c=gt[j].0;
                    if (1..=4).contains(&c) {
                        if let Some(&raw_label) = raw_expert.get(&q) {
                            expert_confusion[c - 1][raw_label - 1] += 1;
                            expert_matched += 1; im_expert_matched += 1;
                            expert_correct += usize::from(raw_label == c); im_expert_correct += usize::from(raw_label == c);
                            expert_baseline_correct += usize::from(label == c); im_expert_baseline += usize::from(label == c);
                        }
                        matched+=1; im_matched+=1;
                        baseline+=usize::from(label==c); im_before+=usize::from(label==c);
                        candidate+=usize::from(new_label==c); im_after+=usize::from(new_label==c);
                        changed+=usize::from(label!=new_label);
                        fixed+=usize::from(label!=c && new_label==c);
                        broken+=usize::from(label==c && new_label!=c);
                        before[c-1][label]+=1;after[c-1][new_label]+=1;
                        im_fused+=usize::from(fused_label==c);
                        fused_correct+=usize::from(fused_label==c);
                        fused_fixed+=usize::from(label!=c && fused_label==c);
                        fused_broken+=usize::from(label==c && fused_label!=c);
                    }
                }
            }
            per_image.push(serde_json::json!({"id":sample.image_id,"matched":im_matched,"before":im_before,"after":im_after,"fused":im_fused,
                "expert_matched":im_expert_matched,"expert_correct":im_expert_correct,"expert_baseline_correct":im_expert_baseline}));
        }
        eprintln!("{} / {} images",per_image.len(),samples.len());
    }
    let base_ap = coco_map(&all_gt, &detector_dets);
    let fusion_ap = coco_map(&all_gt, &fused_dets);
    let mut per_class_ap = vec![];
    for c in 1..15 {
        let gt: Vec<Vec<_>> = all_gt.iter().map(|v|v.iter().filter(|x|x.0==c).cloned().collect()).collect();
        let base: Vec<Vec<_>> = detector_dets.iter().map(|v|v.iter().filter(|x|x.label==c).cloned().collect()).collect();
        let fused: Vec<Vec<_>> = fused_dets.iter().map(|v|v.iter().filter(|x|x.label==c).cloned().collect()).collect();
        per_class_ap.push(serde_json::json!({"class":c,"gt":gt.iter().map(Vec::len).sum::<usize>(),"detector_ap":coco_map(&gt,&base).ap,"fused_ap":coco_map(&gt,&fused).ap}));
    }
    let result=serde_json::json!({"images":samples.len(),"selection_skip":skip,"selection_limit":limit,"passenger_gt":total,"localized":matched,"detector_correct":baseline,"routed_correct":candidate,
        "detector_accuracy":baseline as f64/matched.max(1) as f64,"routed_accuracy":candidate as f64/matched.max(1) as f64,
        "expert_calls":calls,"changed_matched":changed,"fixed":fixed,"broken":broken,"before_confusion":before,"after_confusion":after,"per_image":per_image,"proposals":proposals,
        "fused_correct":fused_correct,"fused_fixed":fused_fixed,"fused_broken":fused_broken,
        "expert_matched":expert_matched,"expert_correct":expert_correct,"expert_baseline_correct":expert_baseline_correct,
        "expert_confusion":expert_confusion,
        "expert_prior":prior.map(|(p, w)| serde_json::json!({"prior":p,"power":w})),
        "detector_ap":base_ap.ap,"fused_ap":fusion_ap.ap,"detector_ap50":base_ap.ap50,"fused_ap50":fusion_ap.ap50,"per_class_ap":per_class_ap,
        "fusion_alpha":alpha,"fusion_abstain":abstain,
        "fusion_contract":format!("alpha={alpha}; log probabilities centered, clipped [-2,2], recentered; mean passenger logits preserved, family maximum not preserved; only expert confidence>={abstain}; no fitted temperature"),
        "scope":"AP: all classes/images and unmatched proposals, unchanged top300 query-by-class; paired top1: old score>=0.3 matching at IoU>=0.5, unchanged boxes/order; inference routing: predicted passenger only, min48px"});
    std::fs::write(&a[5],serde_json::to_vec_pretty(&result)?)?;
    println!("passenger GT {total} localized {matched}; detector {baseline} correct; expert routing {candidate} correct; fixed {fixed} broken {broken}; calls {calls}");
    println!("fixed fusion: {fused_correct} correct; fixed {fused_fixed} broken {fused_broken}; AP {:.4} -> {:.4}", base_ap.ap, fusion_ap.ap);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fusion_control_abstention_and_class_isolation() {
        let base = [0.3f32; 15];
        let mut row = base;
        fuse(&mut row, [0.8,0.1,0.05,0.05], 0.);
        assert_eq!(row,base);
        fuse(&mut row, [0.4,0.3,0.2,0.1], 0.25);
        assert_eq!(row,base);
        fuse(&mut row, [0.8,0.1,0.05,0.05], 0.25);
        assert_eq!(row[0],base[0]);
        assert_eq!(&row[5..],&base[5..]);
        assert!(row[1]>base[1]);
        assert!((row[1..5].iter().sum::<f32>()-base[1..5].iter().sum::<f32>()).abs()<1e-6);
    }
    #[test]
    fn prior_reweights_and_renormalises() {
        let p = apply_prior([0.25; 4], &[0.4, 0.3, 0.2, 0.1], 1.);
        assert!((p.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert!((p[0] - 0.4).abs() < 1e-6 && (p[3] - 0.1).abs() < 1e-6);
        assert_eq!(apply_prior([0.1, 0.2, 0.3, 0.4], &[0.4, 0.3, 0.2, 0.1], 0.), [0.1, 0.2, 0.3, 0.4]);
    }
}

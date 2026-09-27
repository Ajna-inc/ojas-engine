//! Plate-read gating over vehicle tracks: read a vehicle's plate a few times
//! while it is in view, then stop — instead of running plate detection and
//! OCR on every vehicle in every frame.
//!
//! `PlateGate` is tracker-agnostic: it only needs a stable track id per vehicle
//! box, so a deployment can drive it from whatever tracker it runs, and
//! `IouTracker` is a small stand-in for benches and examples.
//!
//! A track is settled once its two best reads agree at `agree_conf`, or after
//! `max_attempts` reads. Attempts are spaced by `min_interval_s`, and one is
//! allowed early when the vehicle box has grown by `grow` (closer vehicle, bigger
//! plate). Candidates per camera frame are capped, widest first.

use std::collections::HashMap;

use crate::yolo::Detection;

/// Gate tuning. The defaults are a 0.5 s re-attempt interval, 8 crops per frame,
/// and the early stop described above.
#[derive(Debug, Clone, Copy)]
pub struct GateCfg {
    /// seconds between attempts on one track
    pub min_interval_s: f64,
    /// ... unless the box width grew by this fraction since the last attempt
    pub grow: f32,
    /// attempts before a track is settled on its best read (or given up)
    pub max_attempts: u32,
    /// two reads agreeing at this mean confidence settle the track
    pub agree_conf: f32,
    /// a single read at this confidence settles the track
    pub single_conf: f32,
    /// vehicles narrower than this (pixels) are not attempted
    pub min_vehicle_w: f32,
    /// attempts per camera frame, widest vehicles first
    pub max_per_frame: usize,
}

impl Default for GateCfg {
    fn default() -> Self {
        GateCfg {
            min_interval_s: 0.5,
            grow: 0.25,
            max_attempts: 5,
            agree_conf: 0.85,
            single_conf: 0.97,
            min_vehicle_w: 60.0,
            max_per_frame: 8,
        }
    }
}

/// Where a track stands.
#[derive(Debug, Clone, PartialEq)]
pub enum Settled {
    /// read: text and its confidence
    Read { text: String, conf: f32, attempts: u32 },
    /// attempts exhausted without a usable read
    Unread { attempts: u32 },
}

#[derive(Debug, Default)]
struct Entry {
    attempts: u32,
    last_t: f64,
    last_w: f32,
    /// (text, mean confidence) of every non-empty read
    reads: Vec<(String, f32)>,
    settled: Option<Settled>,
}

impl Entry {
    /// Settled when two reads of one text reach `agree_conf` (mean of the
    /// two best), one read reaches `single_conf`, or attempts run out.
    fn settle(&mut self, cfg: &GateCfg) {
        let mut best: HashMap<&str, (f32, f32)> = HashMap::new(); // text -> two best confs
        for (t, c) in &self.reads {
            let e = best.entry(t.as_str()).or_insert((0.0, 0.0));
            if *c > e.0 {
                *e = (*c, e.0);
            } else if *c > e.1 {
                e.1 = *c;
            }
        }
        let pick = best.iter().max_by(|a, b| (a.1 .0 + a.1 .1).total_cmp(&(b.1 .0 + b.1 .1)));
        if let Some((t, &(c1, c2))) = pick {
            if (c2 > 0.0 && (c1 + c2) / 2.0 >= cfg.agree_conf) || c1 >= cfg.single_conf {
                let conf = if c2 > 0.0 { (c1 + c2) / 2.0 } else { c1 };
                self.settled = Some(Settled::Read { text: t.to_string(), conf, attempts: self.attempts });
                return;
            }
        }
        if self.attempts >= cfg.max_attempts {
            // best single read if any, else unread
            let top = self.reads.iter().max_by(|a, b| a.1.total_cmp(&b.1)).cloned();
            self.settled = Some(match top {
                Some((text, conf)) => Settled::Read { text, conf, attempts: self.attempts },
                None => Settled::Unread { attempts: self.attempts },
            });
        }
    }
}

/// Per-camera gate state.
#[derive(Debug, Default)]
pub struct PlateGate {
    pub cfg: GateCfg,
    tracks: HashMap<u64, Entry>,
}

impl PlateGate {
    pub fn new(cfg: GateCfg) -> Self {
        PlateGate { cfg, tracks: HashMap::new() }
    }

    /// Which of this frame's tracked vehicles `(track id, box)` to read now
    /// (indices into `cands`), at time `t` seconds. Counts as an attempt.
    pub fn select(&mut self, t: f64, cands: &[(u64, Detection)]) -> Vec<usize> {
        let cfg = self.cfg;
        let mut want: Vec<(usize, f32)> = cands
            .iter()
            .enumerate()
            .filter_map(|(i, (id, d))| {
                let w = d.x1 - d.x0;
                if w < cfg.min_vehicle_w {
                    return None;
                }
                let e = self.tracks.get(id);
                let due = match e {
                    None => true,
                    Some(e) => {
                        e.settled.is_none()
                            && (t - e.last_t >= cfg.min_interval_s || w >= e.last_w * (1.0 + cfg.grow))
                    }
                };
                due.then_some((i, w))
            })
            .collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1));
        want.truncate(cfg.max_per_frame);
        for &(i, w) in &want {
            let e = self.tracks.entry(cands[i].0).or_default();
            e.attempts += 1;
            e.last_t = t;
            e.last_w = w;
        }
        want.into_iter().map(|(i, _)| i).collect()
    }

    /// The reads of one attempt on track `id` (empty: no plate / no text).
    /// Returns the settlement when this attempt settled the track.
    pub fn record(&mut self, id: u64, reads: &[(String, f32)]) -> Option<Settled> {
        let cfg = self.cfg;
        let e = self.tracks.get_mut(&id)?;
        if e.settled.is_some() {
            return None;
        }
        e.reads.extend(reads.iter().filter(|(t, _)| !t.is_empty()).cloned());
        e.settle(&cfg);
        e.settled.clone()
    }

    /// Track `id` left view: its settlement, or its best read so far if it
    /// never settled (None if it was never attempted).
    pub fn end(&mut self, id: u64) -> Option<Settled> {
        let e = self.tracks.remove(&id)?;
        e.settled.or_else(|| {
            let top = e.reads.iter().max_by(|a, b| a.1.total_cmp(&b.1)).cloned();
            Some(match top {
                Some((text, conf)) => Settled::Read { text, conf, attempts: e.attempts },
                None => Settled::Unread { attempts: e.attempts },
            })
        })
    }

    /// Tracks currently held (attempted, not ended).
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

/// Minimal IoU tracker, a stand-in for a real tracker in benches:
/// constant-velocity prediction, greedy association by IoU, tracks confirmed after
/// `min_hits` matches and dropped after `max_misses`.
#[derive(Debug)]
pub struct IouTracker {
    pub min_iou: f32,
    pub min_hits: u32,
    pub max_misses: u32,
    next_id: u64,
    tracks: Vec<Trk>,
}

#[derive(Debug, Clone, Copy)]
struct Trk {
    id: u64,
    b: [f32; 4],
    v: [f32; 4],
    hits: u32,
    misses: u32,
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = iw * ih;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u > 0.0 { i / u } else { 0.0 }
}

impl Default for IouTracker {
    fn default() -> Self {
        IouTracker { min_iou: 0.3, min_hits: 2, max_misses: 5, next_id: 1, tracks: vec![] }
    }
}

impl IouTracker {
    /// Associate this frame's detections. Returns, per detection, the id of
    /// the confirmed track it belongs to (None: unconfirmed), and the ids of
    /// confirmed tracks that ended this frame.
    pub fn update(&mut self, dets: &[Detection]) -> (Vec<Option<u64>>, Vec<u64>) {
        let boxes: Vec<[f32; 4]> = dets.iter().map(|d| [d.x0, d.y0, d.x1, d.y1]).collect();
        let pred: Vec<[f32; 4]> = self.tracks.iter().map(|t| std::array::from_fn(|k| t.b[k] + t.v[k])).collect();
        let mut pairs: Vec<(f32, usize, usize)> = vec![];
        for (ti, p) in pred.iter().enumerate() {
            for (di, b) in boxes.iter().enumerate() {
                let o = iou(p, b);
                if o >= self.min_iou {
                    pairs.push((o, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut tused, mut dused) = (vec![false; self.tracks.len()], vec![false; dets.len()]);
        let mut owner = vec![None; dets.len()];
        for (_, ti, di) in pairs {
            if tused[ti] || dused[di] {
                continue;
            }
            tused[ti] = true;
            dused[di] = true;
            let t = &mut self.tracks[ti];
            let b = boxes[di];
            t.v = std::array::from_fn(|k| 0.5 * t.v[k] + 0.5 * (b[k] - t.b[k]));
            t.b = b;
            t.hits += 1;
            t.misses = 0;
            if t.hits >= self.min_hits {
                owner[di] = Some(t.id);
            }
        }
        let mut ended = vec![];
        for (ti, t) in self.tracks.iter_mut().enumerate() {
            if !tused[ti] {
                t.misses += 1;
                t.b = std::array::from_fn(|k| t.b[k] + t.v[k]);
            }
        }
        let (min_hits, max_misses) = (self.min_hits, self.max_misses);
        self.tracks.retain(|t| {
            let keep = t.misses <= max_misses;
            if !keep && t.hits >= min_hits {
                ended.push(t.id);
            }
            keep
        });
        for (di, b) in boxes.iter().enumerate() {
            if !dused[di] {
                self.tracks.push(Trk { id: self.next_id, b: *b, v: [0.0; 4], hits: 1, misses: 0 });
                self.next_id += 1;
            }
        }
        (owner, ended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, w: f32) -> Detection {
        Detection { class: 2, score: 0.9, x0: x, y0: 100.0, x1: x + w, y1: 100.0 + w * 0.6, keypoints: None }
    }

    #[test]
    fn tracker_follows_a_moving_box_and_ends_it() {
        let mut tr = IouTracker::default();
        let mut id = None;
        for f in 0..10 {
            let (own, ended) = tr.update(&[det(100.0 + f as f32 * 8.0, 200.0)]);
            assert!(ended.is_empty());
            if f >= 1 {
                let o = own[0].expect("confirmed after 2 hits");
                assert!(id.is_none_or(|i| i == o));
                id = Some(o);
            }
        }
        let mut ended = vec![];
        for _ in 0..7 {
            ended.extend(tr.update(&[]).1);
        }
        assert_eq!(ended, vec![id.unwrap()]);
    }

    #[test]
    fn gate_stops_after_two_agreeing_reads() {
        let mut g = PlateGate::new(GateCfg::default());
        let c = [(7u64, det(0.0, 200.0))];
        assert_eq!(g.select(0.0, &c), vec![0]);
        assert_eq!(g.record(7, &[("GJ01AB1234".into(), 0.9)]), None);
        assert!(g.select(0.2, &c).is_empty(), "interval not elapsed, box not grown");
        assert_eq!(g.select(0.6, &c), vec![0]);
        match g.record(7, &[("GJ01AB1234".into(), 0.92)]) {
            Some(Settled::Read { text, conf, attempts: 2 }) => {
                assert_eq!(text, "GJ01AB1234");
                assert!((conf - 0.91).abs() < 1e-6);
            }
            s => panic!("not settled: {s:?}"),
        }
        assert!(g.select(5.0, &c).is_empty(), "settled tracks are not read again");
    }

    #[test]
    fn gate_gives_up_and_caps_per_frame() {
        let cfg = GateCfg { max_attempts: 2, max_per_frame: 2, ..Default::default() };
        let mut g = PlateGate::new(cfg);
        let c: Vec<(u64, Detection)> = (0..4).map(|i| (i, det(i as f32 * 300.0, 100.0 + i as f32 * 10.0))).collect();
        let pick = g.select(0.0, &c);
        assert_eq!(pick, vec![3, 2], "widest first, capped");
        g.record(3, &[]);
        g.select(1.0, &c);
        assert_eq!(g.record(3, &[]), Some(Settled::Unread { attempts: 2 }));
        // a grown box is due before the interval
        let mut g = PlateGate::new(GateCfg::default());
        g.select(0.0, &[(1, det(0.0, 100.0))]);
        assert_eq!(g.select(0.1, &[(1, det(0.0, 130.0))]), vec![0]);
    }
}

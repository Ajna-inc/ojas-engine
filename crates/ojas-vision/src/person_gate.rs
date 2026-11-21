//! Lane gating over person tracks: run a lane (appearance embedding, pose, image–text
//! embedding) on a tracked person a few times while they are in view, then stop — and, for the
//! embedding lanes, keep the few embeddings of the track worth matching against.
//!
//! Like `PlateGate` (`track_gate.rs`) this is tracker-agnostic host logic: it needs a stable
//! track id per box and nothing of the device, so it builds without a GPU backend. The lanes
//! themselves are `person.rs`, which re-exports everything here.

use crate::yolo::Detection;

/// Per-lane gate tuning: run a lane on a track a few times while it is in view, spaced out,
/// until enough good samples are in, then stop.
#[derive(Debug, Clone, Copy)]
pub struct PersonGateCfg {
    /// seconds between attempts on one track
    pub min_interval_s: f64,
    /// ... or earlier when the box height grew by this fraction (closer, better crop)
    pub grow: f32,
    /// samples of quality ≥ `good` that settle the track
    pub want: u32,
    pub good: f32,
    /// attempts before giving up
    pub max_attempts: u32,
    /// boxes shorter than this are not attempted
    pub min_h: f32,
    /// attempts per camera frame, tallest first
    pub max_per_frame: usize,
}

impl PersonGateCfg {
    pub fn reid() -> Self {
        PersonGateCfg { min_interval_s: 0.5, grow: 0.25, want: 3, good: 0.5, max_attempts: 6, min_h: 60.0, max_per_frame: 8 }
    }
    pub fn pose() -> Self {
        PersonGateCfg { min_interval_s: 0.2, grow: 0.25, want: 6, good: 0.5, max_attempts: 12, min_h: 100.0, max_per_frame: 8 }
    }
    pub fn clip() -> Self {
        PersonGateCfg { min_interval_s: 1.0, grow: 0.3, want: 1, good: 0.5, max_attempts: 3, min_h: 60.0, max_per_frame: 4 }
    }
}

#[derive(Debug, Default)]
struct GateEntry {
    attempts: u32,
    good: u32,
    last_t: f64,
    last_h: f32,
    done: bool,
    gallery: TrackGallery,
}

/// Per-camera, per-lane gate over tracked people (any tracker: it only needs a stable id).
#[derive(Debug)]
pub struct PersonGate {
    pub cfg: PersonGateCfg,
    /// keep the best embeddings of each track ([`PersonGate::record_embedding`]); `None`: the
    /// gate only counts
    pub gallery: Option<GalleryCfg>,
    tracks: std::collections::HashMap<u64, GateEntry>,
}

impl PersonGate {
    pub fn new(cfg: PersonGateCfg) -> Self {
        PersonGate { cfg, gallery: None, tracks: Default::default() }
    }

    /// A gate that also keeps each track's gallery: the embedding lanes (ReID, CLIP).
    pub fn with_gallery(cfg: PersonGateCfg, gallery: GalleryCfg) -> Self {
        PersonGate { cfg, gallery: Some(gallery), tracks: Default::default() }
    }

    /// Which of this frame's tracked people `(track id, box)` to run now (indices into
    /// `cands`), at time `t` seconds. Counts as an attempt.
    pub fn select(&mut self, t: f64, cands: &[(u64, Detection)]) -> Vec<usize> {
        let cfg = self.cfg;
        let mut want: Vec<(usize, f32)> = cands
            .iter()
            .enumerate()
            .filter_map(|(i, (id, d))| {
                let h = d.y1 - d.y0;
                if h < cfg.min_h {
                    return None;
                }
                let due = match self.tracks.get(id) {
                    None => true,
                    Some(e) => !e.done && (t - e.last_t >= cfg.min_interval_s || h >= e.last_h * (1.0 + cfg.grow)),
                };
                due.then_some((i, h))
            })
            .collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1));
        want.truncate(cfg.max_per_frame);
        for &(i, h) in &want {
            let e = self.tracks.entry(cands[i].0).or_default();
            e.attempts += 1;
            e.last_t = t;
            e.last_h = h;
        }
        want.into_iter().map(|(i, _)| i).collect()
    }

    /// The quality of one attempt's sample on track `id` (0–1; e.g. the box height over 200 px
    /// times the detector score, or the mean keypoint score). Returns true when the track just
    /// settled — enough good samples, or attempts exhausted.
    pub fn record(&mut self, id: u64, quality: f32) -> bool {
        let cfg = self.cfg;
        let Some(e) = self.tracks.get_mut(&id) else { return false };
        if e.done {
            return false;
        }
        e.good += (quality >= cfg.good) as u32;
        if e.good >= cfg.want || e.attempts >= cfg.max_attempts {
            e.done = true;
            return true;
        }
        false
    }

    /// [`PersonGate::record`] for an embedding lane: the same quality also ranks `embedding`
    /// (the lane's vector for this attempt's crop) in the track's gallery, stamped with the
    /// attempt's time. Without a gallery this is `record`.
    pub fn record_embedding(&mut self, id: u64, quality: f32, embedding: &[f32]) -> bool {
        if let (Some(g), Some(e)) = (self.gallery, self.tracks.get_mut(&id)) {
            e.gallery.offer(&g, e.last_t, quality, embedding);
        }
        self.record(id, quality)
    }

    /// The gallery of track `id` so far, best first (empty: none kept, or not attempted) — for
    /// a caller that publishes at settlement instead of waiting for the track to end.
    pub fn views(&self, id: u64) -> &[GalleryView] {
        self.tracks.get(&id).map(|e| e.gallery.views()).unwrap_or(&[])
    }

    /// Track `id` left view: (attempts, good samples) if it was ever attempted. Drops its
    /// gallery; use [`PersonGate::end_gallery`] to take it.
    pub fn end(&mut self, id: u64) -> Option<(u32, u32)> {
        self.tracks.remove(&id).map(|e| (e.attempts, e.good))
    }

    /// Track `id` left view: (attempts, good samples, its gallery best first) if it was ever
    /// attempted. Everything held for the track is released.
    pub fn end_gallery(&mut self, id: u64) -> Option<(u32, u32, Vec<GalleryView>)> {
        self.tracks.remove(&id).map(|e| (e.attempts, e.good, e.gallery.into_views()))
    }

    pub fn len(&self) -> usize {
        self.tracks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

// ---------------------------------------------------------------------------------------------
// Gallery: the few embeddings of a track worth matching against.

/// Per-track gallery tuning.
#[derive(Debug, Clone, Copy)]
pub struct GalleryCfg {
    /// embeddings kept per track; 1 keeps the single best crop's, 0 keeps none
    pub k: usize,
    /// two crops whose embeddings reach this cosine are one view of the person
    pub same_view: f32,
    /// samples below this quality are not kept (0: keep whatever the track produced)
    pub min_quality: f32,
}

impl Default for GalleryCfg {
    fn default() -> Self {
        GalleryCfg { k: 4, same_view: 0.85, min_quality: 0.0 }
    }
}

/// One kept sample of a track.
#[derive(Debug, Clone, PartialEq)]
pub struct GalleryView {
    /// the quality it was recorded with
    pub quality: f32,
    /// time of the attempt that produced it (seconds, the gate's clock)
    pub t: f64,
    /// L2-normalised, as the lane emits it
    pub embedding: Vec<f32>,
}

/// The best few embeddings of one track: at most `k`, no two of the same view.
///
/// The rule, per offered sample:
/// - it is the same view as one or more kept samples (cosine ≥ `same_view`): it replaces all
///   of them if its quality beats every one, else it is dropped;
/// - it is a new view and there is room: it is kept;
/// - it is a new view and the gallery is full: it replaces the lowest-quality kept sample if
///   its quality is higher, else it is dropped.
///
/// So the kept samples are always pairwise below `same_view` — a hundred crops of a person
/// walking towards the camera fill one slot with the best of them, and the back or side view
/// that shows up for a second takes its own — and each slot holds the best crop seen of its
/// view. This is decided online from the kept samples alone: memory is `k` vectors per track
/// and an offer costs `k` dot products, where farthest-point selection over the candidates
/// would have to hold every candidate until the track ends (and picks outliers, a bad crop
/// being far from everything). With `k` = 1 it is the best-quality crop's embedding. Equal
/// quality keeps the earlier sample.
#[derive(Debug, Clone, Default)]
pub struct TrackGallery {
    /// best first; equal quality, earlier first
    views: Vec<GalleryView>,
}

impl TrackGallery {
    /// Offer the embedding of a crop taken at `t` with `quality` (higher is better, whatever
    /// scale the caller scores crops on). Returns true when it was kept. A vector the lane
    /// did not normalise is normalised here; an empty, zero or non-finite one, or one of
    /// another length than those kept, is dropped.
    pub fn offer(&mut self, cfg: &GalleryCfg, t: f64, quality: f32, embedding: &[f32]) -> bool {
        if cfg.k == 0 || quality.is_nan() || quality < cfg.min_quality {
            return false;
        }
        if self.views.first().is_some_and(|v| v.embedding.len() != embedding.len()) {
            return false;
        }
        let n = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !n.is_finite() || n <= 0.0 {
            return false;
        }
        let same: Vec<bool> = self.views.iter().map(|v| v.embedding.iter().zip(embedding).map(|(a, b)| a * b).sum::<f32>() / n >= cfg.same_view).collect();
        if same.contains(&true) {
            if self.views.iter().zip(&same).any(|(v, &s)| s && v.quality >= quality) {
                return false;
            }
            let mut it = same.iter();
            self.views.retain(|_| !*it.next().unwrap());
        } else if self.views.len() >= cfg.k {
            if self.views.last().is_some_and(|v| v.quality >= quality) {
                return false;
            }
            self.views.truncate(cfg.k - 1);
        }
        // the lane's own vectors are kept bit for bit
        let embedding = if (n - 1.0).abs() <= 1e-4 { embedding.to_vec() } else { embedding.iter().map(|v| v / n).collect() };
        let at = self.views.partition_point(|v| v.quality >= quality);
        self.views.insert(at, GalleryView { quality, t, embedding });
        true
    }

    /// The kept samples, best first.
    pub fn views(&self) -> &[GalleryView] {
        &self.views
    }

    pub fn into_views(self) -> Vec<GalleryView> {
        self.views
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16;

    /// A unit vector of view `axis`, perturbed by `seed`: two seeds of one axis are near
    /// duplicates (cosine ≈ 0.99), two axes are far apart (cosine ≈ 0).
    fn view(axis: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(axis as u32 * 40503 + 1);
        let mut v: Vec<f32> = (0..DIM)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.06
            })
            .collect();
        v[axis] += 1.0;
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn axis_of(v: &GalleryView) -> usize {
        (0..DIM).max_by(|&a, &b| v.embedding[a].total_cmp(&v.embedding[b])).unwrap()
    }

    fn person(h: f32) -> Detection {
        Detection { class: 0, score: 0.9, x0: 10.0, y0: 10.0, x1: 10.0 + h * 0.4, y1: 10.0 + h, keypoints: None }
    }

    /// A gate that attempts every frame and never settles, so the gallery sees every sample.
    fn open_gate(k: usize) -> PersonGate {
        let cfg = PersonGateCfg { min_interval_s: 0.0, want: u32::MAX, max_attempts: u32::MAX, max_per_frame: usize::MAX, ..PersonGateCfg::reid() };
        PersonGate::with_gallery(cfg, GalleryCfg { k, ..Default::default() })
    }

    #[test]
    fn synthetic_views_are_what_the_tests_assume() {
        assert!(cos(&view(0, 1), &view(0, 2)) > 0.95);
        assert!(cos(&view(0, 1), &view(1, 1)).abs() < 0.2);
    }

    #[test]
    fn fewer_crops_than_k_are_all_kept_best_first() {
        let cfg = GalleryCfg { k: 4, ..Default::default() };
        let mut g = TrackGallery::default();
        assert!(g.offer(&cfg, 0.0, 0.4, &view(0, 1)));
        assert!(g.offer(&cfg, 1.0, 0.9, &view(1, 1)));
        assert!(g.offer(&cfg, 2.0, 0.6, &view(2, 1)));
        let q: Vec<f32> = g.views().iter().map(|v| v.quality).collect();
        assert_eq!(q, vec![0.9, 0.6, 0.4]);
        assert_eq!(g.views().iter().map(|v| v.t).collect::<Vec<_>>(), vec![1.0, 2.0, 0.0]);
        assert_eq!(g.views()[0].embedding, view(1, 1), "a lane vector is kept as emitted");
    }

    #[test]
    fn near_duplicates_share_a_slot_and_distinct_views_survive() {
        let cfg = GalleryCfg { k: 3, ..Default::default() };
        let mut g = TrackGallery::default();
        // 200 crops of the front, quality rising and falling, with a side and a back view of
        // lower quality in between
        for i in 0..200u32 {
            let q = 0.6 + 0.3 * (1.0 - (i as f32 - 120.0).abs() / 120.0);
            g.offer(&cfg, i as f64, q, &view(0, i));
            if i == 40 {
                g.offer(&cfg, 40.5, 0.35, &view(1, 0));
            }
            if i == 150 {
                g.offer(&cfg, 150.5, 0.3, &view(2, 0));
            }
        }
        let mut axes: Vec<usize> = g.views().iter().map(axis_of).collect();
        axes.sort();
        assert_eq!(axes, vec![0, 1, 2], "one slot per view, however many crops the front had");
        assert_eq!(g.views()[0].t, 120.0, "the front slot holds its best crop");
        assert!((g.views()[0].quality - 0.9).abs() < 1e-6);
        for (i, a) in g.views().iter().enumerate() {
            for b in &g.views()[i + 1..] {
                assert!(cos(&a.embedding, &b.embedding) < cfg.same_view);
            }
        }
    }

    #[test]
    fn a_full_gallery_trades_its_worst_view_for_a_better_new_one() {
        let cfg = GalleryCfg { k: 2, ..Default::default() };
        let mut g = TrackGallery::default();
        g.offer(&cfg, 0.0, 0.5, &view(0, 0));
        g.offer(&cfg, 1.0, 0.7, &view(1, 0));
        assert!(!g.offer(&cfg, 2.0, 0.4, &view(2, 0)), "worse than everything kept");
        assert!(!g.offer(&cfg, 3.0, 0.5, &view(2, 0)), "a tie keeps the earlier sample");
        assert!(g.offer(&cfg, 4.0, 0.6, &view(2, 0)));
        assert_eq!(g.views().iter().map(axis_of).collect::<Vec<_>>(), vec![1, 2]);
        // a same-view sample that is worse changes nothing, a better one takes the slot
        assert!(!g.offer(&cfg, 5.0, 0.65, &view(1, 9)));
        assert!(g.offer(&cfg, 6.0, 0.95, &view(2, 9)));
        assert_eq!(g.views().iter().map(|v| (axis_of(v), v.t)).collect::<Vec<_>>(), vec![(2, 6.0), (1, 1.0)]);
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn a_sample_between_two_kept_views_replaces_both_or_neither() {
        let cfg = GalleryCfg { k: 3, same_view: 0.6, ..Default::default() };
        let mut g = TrackGallery::default();
        let (a, b) = (view(0, 0), view(1, 0));
        let mid: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect(); // cosine ≈ 0.7 to both
        g.offer(&cfg, 0.0, 0.5, &a);
        g.offer(&cfg, 1.0, 0.8, &b);
        assert!(!g.offer(&cfg, 2.0, 0.7, &mid), "beats one of the two only");
        assert_eq!(g.len(), 2);
        assert!(g.offer(&cfg, 3.0, 0.9, &mid));
        assert_eq!(g.len(), 1);
        let n: f32 = g.views()[0].embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "normalised on the way in");
    }

    #[test]
    fn k_of_one_is_the_best_crop() {
        let cfg = GalleryCfg { k: 1, ..Default::default() };
        let mut g = TrackGallery::default();
        let qs = [0.3f32, 0.8, 0.5, 0.8, 0.1, 0.79];
        for (i, q) in qs.iter().enumerate() {
            // same view or another: the single slot follows quality alone
            g.offer(&cfg, i as f64, *q, &view(i % 3, i as u32));
            assert_eq!(g.len(), 1);
        }
        assert_eq!(g.views()[0], GalleryView { quality: 0.8, t: 1.0, embedding: view(1, 1) });
    }

    #[test]
    fn unusable_samples_are_dropped() {
        let cfg = GalleryCfg { k: 2, min_quality: 0.3, ..Default::default() };
        let mut g = TrackGallery::default();
        assert!(!g.offer(&cfg, 0.0, 0.2, &view(0, 0)), "below the floor");
        assert!(!g.offer(&cfg, 0.0, f32::NAN, &view(0, 0)));
        assert!(!g.offer(&cfg, 0.0, 0.9, &[]));
        assert!(!g.offer(&cfg, 0.0, 0.9, &[0.0; DIM]));
        let mut bad = view(0, 0);
        bad[3] = f32::NAN;
        assert!(!g.offer(&cfg, 0.0, 0.9, &bad));
        assert!(g.offer(&cfg, 0.0, 0.9, &view(0, 0)));
        assert!(!g.offer(&cfg, 0.0, 0.9, &[1.0; DIM / 2]), "another model's vector");
        assert!(!TrackGallery::default().offer(&GalleryCfg { k: 0, ..Default::default() }, 0.0, 0.9, &view(0, 0)));
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn the_gate_feeds_the_gallery_and_releases_it_at_track_end() {
        let mut gate = open_gate(2);
        let c = [(7u64, person(180.0))];
        for (i, (axis, q)) in [(0, 0.6f32), (0, 0.7), (1, 0.4), (0, 0.65)].into_iter().enumerate() {
            assert_eq!(gate.select(i as f64, &c), vec![0]);
            assert!(!gate.record_embedding(7, q, &view(axis, i as u32)));
        }
        assert_eq!(gate.views(7).iter().map(|v| (axis_of(v), v.t)).collect::<Vec<_>>(), vec![(0, 1.0), (1, 2.0)], "stamped with the attempt's time");
        assert!(gate.views(8).is_empty());
        // a track the gate never attempted holds nothing
        assert!(!gate.record_embedding(8, 0.9, &view(0, 0)));
        assert_eq!(gate.len(), 1);
        let (attempts, good, views) = gate.end_gallery(7).unwrap();
        assert_eq!((attempts, good), (4, 3));
        assert_eq!(views.iter().map(|v| v.quality).collect::<Vec<_>>(), vec![0.7, 0.4]);
        assert!(gate.is_empty());
        assert!(gate.end_gallery(7).is_none());
        assert!(gate.views(7).is_empty());
    }

    #[test]
    fn the_gate_settles_as_before_with_a_gallery() {
        let cfg = PersonGateCfg { min_interval_s: 0.0, ..PersonGateCfg::reid() };
        let (mut plain, mut kept) = (PersonGate::new(cfg), PersonGate::with_gallery(cfg, GalleryCfg::default()));
        let c = [(1u64, person(180.0))];
        let mut settled = vec![];
        for i in 0..5usize {
            let (a, b) = (plain.select(i as f64, &c), kept.select(i as f64, &c));
            assert_eq!(a, b);
            if a.is_empty() {
                continue;
            }
            let (a, b) = (plain.record(1, 0.8), kept.record_embedding(1, 0.8, &view(i, 0)));
            assert_eq!(a, b);
            settled.push(a);
        }
        assert_eq!(settled, vec![false, false, true], "want = 3 good samples, then no more attempts");
        assert_eq!(kept.views(1).len(), 3);
        // without a gallery record_embedding is record, and end drops a gallery unseen
        assert!(plain.views(1).is_empty());
        assert_eq!(plain.end_gallery(1), Some((3, 3, vec![])));
        assert_eq!(kept.end(1), Some((3, 3)));
        assert!(kept.is_empty());
    }

    #[test]
    fn many_concurrent_tracks_stay_separate_and_bounded() {
        const TRACKS: u64 = 500;
        let k = 3;
        let mut gate = open_gate(k);
        let cands: Vec<(u64, Detection)> = (0..TRACKS).map(|id| (id, person(100.0 + id as f32))).collect();
        for f in 0..40u32 {
            let picked = gate.select(f as f64, &cands);
            assert_eq!(picked.len(), TRACKS as usize);
            for i in picked {
                let id = cands[i].0;
                // track `id` shows five views (id, id+1, …) in turn, interleaved with the others
                let turn = (f % 5) as usize;
                let axis = (id as usize + turn) % DIM;
                gate.record_embedding(id, 0.5 + (f % 7) as f32 * 0.05 + turn as f32 * 0.01, &view(axis, f + id as u32 * 131));
            }
            assert!((0..TRACKS).all(|id| gate.views(id).len() <= k));
        }
        assert_eq!(gate.len(), TRACKS as usize);
        for id in 0..TRACKS {
            let (attempts, _, views) = gate.end_gallery(id).unwrap();
            assert_eq!(attempts, 40);
            assert_eq!(views.len(), k);
            assert!(views.windows(2).all(|w| w[0].quality >= w[1].quality));
            let mut axes: Vec<usize> = views.iter().map(axis_of).collect();
            axes.sort();
            axes.dedup();
            assert_eq!(axes.len(), k, "track {id}: {k} different views");
            assert!(axes.iter().all(|a| (0..5).any(|d| (id as usize + d) % DIM == *a)), "track {id} holds only its own views");
        }
        assert!(gate.is_empty());
    }
}

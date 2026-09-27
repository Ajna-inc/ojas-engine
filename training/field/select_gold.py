#!/usr/bin/env python3
"""Pick the frames a human labels as the Field gold test set.

300 frames is one person-day, so the choice matters more than the count. The sample is
stratified by camera, by time of day, and by traffic density (the ensemble's box count, which
needs no labels), so the set covers the empty road and the jam, early light and full sun —
instead of 300 frames of the same busy minute. Frames are drawn evenly across each stratum's
time span, never consecutively.

  select_gold.py <frames_coco.json> <dets.json> --n 300 --out gold.json
"""
import argparse, collections, json, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("dets")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed)

    coco = json.load(open(a.frames))
    images = {im["id"]: im for im in coco["images"]}
    count = collections.Counter(d["image_id"] for d in json.load(open(a.dets)))

    # density bands from the distribution itself, so "busy" means busy for these cameras
    counts = sorted(count.get(i, 0) for i in images)
    q1, q2 = counts[len(counts) // 3], counts[2 * len(counts) // 3]
    def band(n):
        return "quiet" if n <= q1 else ("medium" if n <= q2 else "busy")

    strata = collections.defaultdict(list)
    for i, im in images.items():
        strata[(im.get("camera", "?"), band(count.get(i, 0)))].append(i)

    per = max(1, a.n // max(len(strata), 1))
    picked = []
    for key, ids in sorted(strata.items()):
        ids.sort(key=lambda i: images[i]["utc_ms"])
        step = max(1, len(ids) // per)  # spread across the stratum's time span
        picked += ids[:: step][:per]
    random.shuffle(picked)
    picked = sorted(picked[: a.n])

    out = {"images": [images[i] for i in picked], "categories": coco["categories"], "annotations": [],
           "info": {"selected_from": a.frames, "density_from": a.dets, "n": len(picked),
                    "strata": len(strata), "density_thresholds": [q1, q2], "seed": a.seed}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(picked)} frames from {len(strata)} strata (density thresholds {q1}/{q2} boxes) → {a.out}")
    per_cam = collections.Counter(images[i].get("camera", "?") for i in picked)
    per_band = collections.Counter(band(count.get(i, 0)) for i in picked)
    print("  per camera:", dict(per_cam))
    print("  per density band:", dict(per_band))
    print(f"  median boxes per chosen frame: {sorted(count.get(i, 0) for i in picked)[len(picked) // 2]}")


if __name__ == "__main__":
    main()

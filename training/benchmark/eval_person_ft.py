#!/usr/bin/env python3
"""Score a vehicle+person fine-tune against the vehicle-only model it started from, on the same
852 gold frames.

Dumps detections with dump_detections.py's own runner (the new checkpoint is registered as a preset
at runtime, so that file is not edited), then prints per set (day / night / day_new):
- vehicle mAP over the 14 UVH classes for both models (the new model must not be lower),
- person AP, recall at score 0.4 overall and for people >= 60 px tall, precision at 0.4
  (unsure person regions are crowd = ignored).

    eval_person_ft.py --weights <best_stg1.pth> --config <yml> --val <val_with_persons.json>
                      --images <img root> --gold-frames <person_r0/gold_frames.json> --out <dir>
                      [--base-preset ours_field_ft3_s] [--size 640]
"""
import argparse
import collections
import contextlib
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_detections as dd  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

PERSON = 15


def run_dump(preset, ann, images, out, size):
    if os.path.exists(out):
        return
    argv = sys.argv
    sys.argv = ["dump_detections.py", "--preset", preset, "--ann", ann, "--images", images, "--out", out, "--size", str(size)]
    try:
        dd.main()
    finally:
        sys.argv = argv


def coco_ap(gt, dets, cats, img_ids):
    g = COCO()
    g.dataset = {"images": [i for i in gt["images"] if i["id"] in img_ids], "categories": gt["categories"],
                 "annotations": [a for a in gt["annotations"] if a["image_id"] in img_ids]}
    with contextlib.redirect_stdout(io.StringIO()):
        g.createIndex()
        d = g.loadRes([x for x in dets if x["image_id"] in img_ids and x["category_id"] in cats] or
                      [{"image_id": next(iter(img_ids)), "category_id": cats[0], "bbox": [0, 0, 1, 1], "score": 0.0}])
        e = COCOeval(g, d, "bbox")
        e.params.catIds = cats
        e.params.imgIds = sorted(img_ids)
        e.evaluate()
        e.accumulate()
        e.summarize()
    return e.stats[0]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def person_recall(gt, dets, img_ids, thr=0.4):
    gtb = collections.defaultdict(list)
    crowd = collections.defaultdict(list)
    for a in gt["annotations"]:
        if a["category_id"] != PERSON or a["image_id"] not in img_ids:
            continue
        (crowd if a.get("iscrowd") else gtb)[a["image_id"]].append(a["bbox"])
    pd = collections.defaultdict(list)
    for d in dets:
        if d["category_id"] == PERSON and d["score"] >= thr and d["image_id"] in img_ids:
            pd[d["image_id"]].append(d["bbox"])
    tp = n = tp60 = n60 = fp = nd = 0
    for img in img_ids:
        G = gtb[img]
        used = set()
        for p in sorted(pd[img], key=lambda b: -b[2] * b[3]):
            best, bi = 0.5, -1
            for k, q in enumerate(G):
                if k not in used and iou(p, q) >= best:
                    best, bi = iou(p, q), k
            nd += 1
            if bi >= 0:
                used.add(bi)
            elif not any(iou(p, c) > 0 and (min(p[0] + p[2], c[0] + c[2]) - max(p[0], c[0])) * (min(p[1] + p[3], c[1] + c[3]) - max(p[1], c[1])) >= 0.5 * p[2] * p[3] for c in crowd[img]):
                fp += 1
        for k, q in enumerate(G):
            n += 1
            tp += k in used
            if q[3] >= 60:
                n60 += 1
                tp60 += k in used
    return tp / max(1, n), tp60 / max(1, n60), 1 - fp / max(1, nd), n, n60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gold-frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-preset", default="ours_field_ft3_s")
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dd.PRESETS["_person_ft"] = dict(family="deim", repo=f"{dd.TRAIN}/DEIM", config=a.config, weights=a.weights, offset=0)
    new = os.path.join(a.out, f"person_ft_{a.size}.dets.json")
    base = os.path.join(a.out, f"{a.base_preset}_{a.size}.dets.json")
    run_dump("_person_ft", a.val, a.images, new, a.size)
    run_dump(a.base_preset, a.val, a.images, base, a.size)

    gt = json.load(open(a.val))
    # the set of each val frame, by timestamp (val.json names carry it; gold_frames.json carries set + utc)
    sets_by_utc = {str(i["utc_ms"]): i["set"] for i in json.load(open(a.gold_frames))["images"]}
    sets = collections.defaultdict(set)
    for i in gt["images"]:
        sets[sets_by_utc[re.findall(r"(\d{13})", i["file_name"])[-1]]].add(i["id"])
    sets["all"] = {i["id"] for i in gt["images"]}
    vehicle = [c["id"] for c in gt["categories"] if c["id"] != PERSON]
    dn = json.load(open(new))
    db = json.load(open(base))
    print(f"{'set':8s} {'veh mAP base':>12s} {'veh mAP new':>11s} {'person AP':>9s} {'rec@.4':>7s} {'rec>=60px':>9s} {'prec@.4':>8s}  people/>=60px")
    for s in ["day", "day_new", "night", "all"]:
        ids = sets[s]
        vb = coco_ap(gt, db, vehicle, ids)
        vn = coco_ap(gt, dn, vehicle, ids)
        pa = coco_ap(gt, dn, [PERSON], ids)
        r, r60, pr, n, n60 = person_recall(gt, dn, ids)
        print(f"{s:8s} {vb:12.3f} {vn:11.3f} {pa:9.3f} {r:7.1%} {r60:9.1%} {pr:8.1%}  {n}/{n60}")


if __name__ == "__main__":
    main()

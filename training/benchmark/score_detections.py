#!/usr/bin/env python3
"""Score COCO detections with one evaluator for every model, and print the numbers a detector
comparison actually needs: mAP, AP50, AP75, AP by object size, recall, and AP per class — so a
headline gain that comes from sacrificing a rare class is visible.

  score_detections.py <ann.json> <detections.json> [--label NAME] [--json OUT.json]
"""
import argparse, contextlib, io, json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NAMES = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler",
         "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ann")
    p.add_argument("dets")
    p.add_argument("--label", default="")
    p.add_argument("--json", dest="out")
    a = p.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.ann)
        dt = gt.loadRes(a.dets)
        e = COCOeval(gt, dt, "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    out = {"label": a.label, "ann": a.ann, "detections": a.dets,
           "mAP": s[0], "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4],
           "AP_large": s[5], "AR100": s[8], "AR_small": s[9], "per_class": {}}

    # per-class AP@[.5:.95], the same accumulation restricted to one category
    for i, cid in enumerate(sorted(gt.getCatIds())):
        pr = e.eval["precision"][:, :, i, 0, 2]
        pr = pr[pr > -1]
        n = len(gt.getAnnIds(catIds=[cid]))
        name = NAMES[cid - 1] if 1 <= cid <= len(NAMES) else str(cid)
        out["per_class"][name] = {"AP": float(np.mean(pr)) if pr.size else float("nan"), "gt": n}

    print(f"{a.label or a.dets}")
    print(f"  mAP {s[0]:.4f}  AP50 {s[1]:.4f}  AP75 {s[2]:.4f} | small {s[3]:.4f} medium {s[4]:.4f} "
          f"large {s[5]:.4f} | AR100 {s[8]:.4f}")
    print("  per class: " + "  ".join(f"{k} {v['AP']:.3f}" for k, v in out["per_class"].items()))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

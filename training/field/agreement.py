#!/usr/bin/env python3
"""How far two detectors disagree on frames nobody has labelled — the domain gap, measured
without ground truth, and the queue for the human who labels the gold set.

Matches each model's boxes one-to-one at IoU ≥ 0.5 and reports: boxes only model A found, boxes
only B found, boxes both found with the same class, and boxes both found with *different*
classes. Run it on Field frames and on UVH validation frames and compare the two: the rise in
disagreement is the part of the domain gap that needs no labels to see.

  agreement.py <frames_coco.json> <A.dets.json> <B.dets.json> [--out queue.jsonl] [--iou 0.5]
"""
import argparse, collections, json

NAMES = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler",
         "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"]


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def by_image(dets):
    d = collections.defaultdict(list)
    for x in dets:
        d[x["image_id"]].append(x)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", help="write the disagreeing boxes as a review queue")
    args = ap.parse_args()

    images = {im["id"]: im for im in json.load(open(args.frames))["images"]}
    A, B = by_image(json.load(open(args.a))), by_image(json.load(open(args.b)))
    only_a = only_b = same = diff = 0
    confusion = collections.Counter()
    per_camera = collections.defaultdict(lambda: [0, 0])  # [matched, disagreeing]
    queue = []

    for img_id in images:
        a_list = sorted(A.get(img_id, []), key=lambda x: -x["score"])
        b_list = sorted(B.get(img_id, []), key=lambda x: -x["score"])
        used = set()
        for x in a_list:
            best, best_j = args.iou, None
            for j, y in enumerate(b_list):
                if j in used:
                    continue
                v = iou(x["bbox"], y["bbox"])
                if v >= best:
                    best, best_j = v, j
            if best_j is None:
                only_a += 1
                continue
            used.add(best_j)
            y = b_list[best_j]
            cam = images[img_id].get("camera", "?")
            per_camera[cam][0] += 1
            if x["category_id"] == y["category_id"]:
                same += 1
            else:
                diff += 1
                per_camera[cam][1] += 1
                confusion[(x["category_id"], y["category_id"])] += 1
                if args.out:
                    queue.append({"image_id": img_id, "file_name": images[img_id]["file_name"],
                                  "bbox": x["bbox"], "a": x["category_id"], "b": y["category_id"],
                                  "a_score": x["score"], "b_score": y["score"]})
        only_b += len(b_list) - len(used)

    matched = same + diff
    print(f"frames {len(images)}")
    print(f"matched boxes           {matched}")
    print(f"  same class            {same}  ({100 * same / max(matched, 1):.1f} %)")
    print(f"  different class       {diff}  ({100 * diff / max(matched, 1):.1f} %)")
    print(f"found only by A         {only_a}")
    print(f"found only by B         {only_b}")
    print(f"agreement over all boxes {100 * same / max(same + diff + only_a + only_b, 1):.1f} %")
    print("\nwhere they disagree (A → B):")
    for (ca, cb), n in confusion.most_common(12):
        na = NAMES[ca - 1] if 1 <= ca <= 14 else ca
        nb = NAMES[cb - 1] if 1 <= cb <= 14 else cb
        print(f"  {na:16} → {nb:16} {n}")
    if len(per_camera) > 1:
        print("\nper camera (matched, % disagreeing):")
        for cam, (m, d) in sorted(per_camera.items()):
            print(f"  {cam:24} {m:7}  {100 * d / max(m, 1):5.1f} %")
    if args.out:
        with open(args.out, "w") as f:
            for q in queue:
                f.write(json.dumps(q) + "\n")
        print(f"\nreview queue: {len(queue)} boxes → {args.out}")


if __name__ == "__main__":
    main()

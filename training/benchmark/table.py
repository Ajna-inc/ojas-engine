#!/usr/bin/env python3
"""Print the benchmark scores as markdown: one row per model, one column per split, plus the
per-class table on the headline split. `table.py <benchmark_dir> [--split uvh_val_st]`"""
import argparse, glob, json, os

ORDER = ["ours_deim_dfine_s", "iisc_rtdetrv2_s", "iisc_rtdetrv2_x", "iisc_yolo11_s", "iisc_yolo11_x", "bmd_dfine_x"]
SPLITS = ["uvh_val_st", "uvh_val_mv", "uvh_val_st_clean"]

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--split", default="uvh_val_st")
a = ap.parse_args()

scores = {}
for f in glob.glob(os.path.join(a.dir, "*.score.json")):
    m, s = os.path.basename(f).replace(".score.json", "").split("__")
    scores.setdefault(m, {})[s] = json.load(open(f))

print(f"| model | {' | '.join(SPLITS)} |")
print("|---|" + "---:|" * len(SPLITS))
for m in ORDER + [k for k in scores if k not in ORDER]:
    if m not in scores:
        continue
    cells = [f"**{scores[m][s]['mAP']:.4f}**" if s in scores[m] else "—" for s in SPLITS]
    print(f"| {m} | {' | '.join(cells)} |")

print(f"\nOn {a.split}: AP50 / AP75 / AP-small / AR100\n")
print("| model | AP50 | AP75 | AP-small | AR100 |")
print("|---|---:|---:|---:|---:|")
for m in ORDER + [k for k in scores if k not in ORDER]:
    d = scores.get(m, {}).get(a.split)
    if d:
        print(f"| {m} | {d['AP50']:.4f} | {d['AP75']:.4f} | {d['AP_small']:.4f} | {d['AR100']:.4f} |")

classes = None
for m in ORDER:
    d = scores.get(m, {}).get(a.split)
    if d:
        classes = classes or list(d["per_class"])
if classes:
    print(f"\nPer-class AP@[.5:.95] on {a.split} (gt count in the header)\n")
    hdr = [f"{c} ({scores[ORDER[0]][a.split]['per_class'][c]['gt']})" if ORDER[0] in scores and a.split in scores[ORDER[0]] else c for c in classes]
    print("| model | " + " | ".join(hdr) + " |")
    print("|---|" + "---:|" * len(classes))
    for m in ORDER + [k for k in scores if k not in ORDER]:
        d = scores.get(m, {}).get(a.split)
        if d:
            print(f"| {m} | " + " | ".join(f"{d['per_class'][c]['AP']:.3f}" for c in classes) + " |")

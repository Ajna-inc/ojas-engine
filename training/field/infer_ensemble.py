#!/usr/bin/env python3
"""Run several detectors over the extracted Field frames and write one COCO detection file per
model, plus a COCO `images` list for them.

These frames have no ground truth, so nothing here is an accuracy measurement. The outputs are
(1) pre-labels a human corrects into the gold set, (2) the raw material for track voting, and
(3) the model-agreement statistics that quantify the domain gap without any labels.

  infer_ensemble.py --frames frames.jsonl --out DIR [--presets a,b] [--limit N] [--score 0.3]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark"))
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dump_detections import PRESETS, build_detr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--presets", default="ours_deim_dfine_s,iisc_rtdetrv2_x")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames = [json.loads(l) for l in open(a.frames)]
    if a.limit:
        frames = frames[: a.limit]
    # one stable integer id per frame: camera folder and wall-clock are the identity
    images = [{"id": i, "file_name": f["path"], "width": f["width"], "height": f["height"],
               "camera": f["camera"], "utc_ms": f["utc_ms"]} for i, f in enumerate(frames)]
    json.dump({"images": images, "categories": [{"id": i + 1, "name": n} for i, n in enumerate(
        ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler",
         "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"])], "annotations": []},
        open(f"{a.out}/frames_coco.json", "w"))
    print(f"{len(images)} frames → {a.out}/frames_coco.json", flush=True)

    import torchvision.transforms as T
    tf = T.Compose([T.Resize((a.size, a.size)), T.ToTensor()])
    for name in a.presets.split(","):
        out_path = f"{a.out}/{name}.dets.json"
        if os.path.exists(out_path):
            print(f"{name}: already done", flush=True)
            continue
        p = PRESETS[name]
        model, post, params = build_detr(p)
        dets, t0 = [], time.time()
        with torch.no_grad():
            for im in images:
                img = Image.open(im["file_name"]).convert("RGB")
                x = tf(img)[None].cuda()
                sizes = torch.tensor([[img.width, img.height]], device="cuda")
                labels, boxes, scores = post(model(x), sizes)
                for l, b, s in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
                    if s < a.score:
                        continue
                    dets.append({"image_id": im["id"], "category_id": int(l) + p["offset"],
                                 "bbox": [round(b[0], 1), round(b[1], 1), round(b[2] - b[0], 1), round(b[3] - b[1], 1)],
                                 "score": round(s, 4)})
                if (im["id"] + 1) % 2000 == 0:
                    print(f"  {name} {im['id'] + 1}/{len(images)} ({len(dets)} boxes, {time.time() - t0:.0f}s)", flush=True)
        json.dump(dets, open(out_path, "w"))
        print(f"{name}: {len(dets)} boxes over {len(images)} frames, {params / 1e6:.1f}M params, {time.time() - t0:.0f}s → {out_path}", flush=True)
        del model, post
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

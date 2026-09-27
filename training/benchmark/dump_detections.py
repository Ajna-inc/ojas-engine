#!/usr/bin/env python3
"""Run one checkpoint over one prepared COCO split and write plain COCO detections.

Each model family needs its own loader, but they all see the same images at the same input size
and emit the same format, so `score_detections.py` scores them with one evaluator. The published
tables use each repo's own evaluation and are not comparable across families.

  dump_detections.py --preset NAME --ann A.json --images DIR --out D.json [--size 640] [--limit N]

Presets carry the repo, config, checkpoint and the class-id mapping onto the canonical UVH ids
1..14 (BMD-45 models are 13-class 0-based, so they shift by 1). The RT-DETRv2-X overrides follow
`evidence/teacher_benchmark_2026_09_20`, where they were verified to load strictly.
"""
import argparse, json, os, sys, time

import torch
from PIL import Image

TRAIN = "/data/dev-cache/train"
BASE = "/data/datasets/iisc-aim"

PRESETS = {
    # our Step 2 model
    "ours_deim_dfine_s": dict(family="deim", repo=f"{TRAIN}/DEIM",
                              config=f"{TRAIN}/DEIM/configs/uvh/step2_deim_dfine_s.yml",
                              weights=f"{TRAIN}/out/step2_deim_dfine_s/best_stg2.pth", offset=0),
    # the same run's last eight checkpoints averaged (see soup.py)
    "ours_soup": dict(family="deim", repo=f"{TRAIN}/DEIM",
                      config=f"{TRAIN}/DEIM/configs/uvh/step2_deim_dfine_s.yml",
                      weights=f"{TRAIN}/out/step2_deim_dfine_s/soup_final.pth", offset=0),
    # our Step 3 production model (3.20 M), trained on a rented GPU; mirrored to the HDD
    "ours_ojas_n32": dict(family="deim", repo=f"{TRAIN}/DEIM",
                          config=f"{TRAIN}/DEIM/configs/uvh/step3_ojas_n32.yml",
                          weights=f"{TRAIN}/out/cloud_step3_ojas_n32/workspace/out/step3_ojas_n32/best_stg2.pth", offset=0),
    # Step 3b: the P3 model, multi-scale (576-992) fine-tune on the pod; latest synced checkpoint
    "ours_step3b": dict(family="deim", repo=f"{TRAIN}/DEIM",
                        config=f"{TRAIN}/DEIM/configs/uvh/step3b_ojas_n32_p3.yml",
                        weights=f"{TRAIN}/out/cloud_step3b_ojas_n32_p3/workspace/out/step3b_ojas_n32_p3/last.pth", offset=0),
    # Step 3b final: the best checkpoint of the run (epoch 28; 0.5480 at 640 in DEIM's own eval)
    "ours_step3b_best": dict(family="deim", repo=f"{TRAIN}/DEIM",
                             config=f"{TRAIN}/DEIM/configs/uvh/step3b_ojas_n32_p3.yml",
                             weights=f"{TRAIN}/out/cloud_step3b_ojas_n32_p3/workspace/out/step3b_ojas_n32_p3/best_stg1.pth", offset=0),
    # Step 3b fine-tuned on the reviewed Field frames only (deim/field_ft.yml; best = epoch 23)
    "ours_field_ft": dict(family="deim", repo=f"{TRAIN}/DEIM",
                            config=f"{TRAIN}/DEIM/configs/uvh/step3b_ojas_n32_p3.yml",
                            weights=f"{TRAIN}/out/cloud_field_ft/workspace/out/field_ft/best_stg1.pth", offset=0),
    # the 10 M Step-2 model fine-tuned on the same Field frames (deim/field_ft_s.yml)
    "ours_field_ft_s": dict(family="deim", repo=f"{TRAIN}/DEIM",
                              config=f"{TRAIN}/DEIM/configs/uvh/step2_deim_dfine_s.yml",
                              weights=f"{TRAIN}/out/cloud_field_ft_s/workspace/out/field_ft_s/best_stg1.pth", offset=0),
    # Field fine-tune 2 (rounds 2–5, 16,548 frames, 33 views): 3.39 M locally, 10 M on a rented GPU
    "ours_field_ft2": dict(family="deim", repo=f"{TRAIN}/DEIM",
                             config=f"{TRAIN}/DEIM/configs/uvh/step3b_ojas_n32_p3.yml",
                             weights=f"{TRAIN}/out/field_ft2_local/best_stg1.pth", offset=0),
    "ours_field_ft2_s": dict(family="deim", repo=f"{TRAIN}/DEIM",
                               config=f"{TRAIN}/DEIM/configs/uvh/step2_deim_dfine_s.yml",
                               weights=f"{TRAIN}/out/cloud_field_ft2_s/workspace/out/field_ft2_s/best_stg1.pth", offset=0),
    # Field fine-tune 3 (rebalanced: original day 40 %, night ×2 40 %, new cameras ½ 20 %)
    "ours_field_ft3": dict(family="deim", repo=f"{TRAIN}/DEIM",
                             config=f"{TRAIN}/DEIM/configs/uvh/step3b_ojas_n32_p3.yml",
                             weights=f"{TRAIN}/out/field_ft3_local/best_stg1.pth", offset=0),
    "ours_field_ft3_s": dict(family="deim", repo=f"{TRAIN}/DEIM",
                               config=f"{TRAIN}/DEIM/configs/uvh/step2_deim_dfine_s.yml",
                               weights=f"{TRAIN}/out/cloud_field_ft3_s/workspace/out/field_ft3_s/best_stg1.pth", offset=0),
    # vehicle + person mix 1 (16 classes: UVH 1..14 + person 15): all Field frames with people + UVH replay
    "ours_mix1_s": dict(family="deim", repo=f"{TRAIN}/DEIM",
                        config=f"{TRAIN}/DEIM/configs/uvh/step2_deim_dfine_s.yml", num_classes=16,
                        weights=f"{TRAIN}/out/cloud_field_mix1_s/workspace/out/field_mix1_s/best_stg1.pth", offset=0),
    "ours_mix2_s": dict(family="deim", repo=f"{TRAIN}/DEIM",
                        config=f"{TRAIN}/DEIM/configs/uvh/step2_deim_dfine_s.yml", num_classes=16,
                        weights=f"{TRAIN}/out/cloud_field_mix2_s/workspace/out/field_mix2_s/best_stg1.pth", offset=0),
    "ours_mix1": dict(family="deim", repo=f"{TRAIN}/DEIM",
                      config=f"{TRAIN}/DEIM/configs/uvh/step3b_ojas_n32_p3.yml", num_classes=16,
                      weights=f"{TRAIN}/out/field_mix1_local/best_stg1.pth", offset=0),
    # IISc's released UVH-26 models
    "iisc_rtdetrv2_s": dict(family="rtdetrv2", repo=f"{TRAIN}/RT-DETR/rtdetrv2_pytorch",
                            config=f"{TRAIN}/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
                            weights=f"{BASE}/models-UVH-26/weights/RT-DETRv2-S/UVH-26-MV-RT-DETRv2-S.pth",
                            num_classes=15, offset=0),
    "iisc_rtdetrv2_x": dict(family="rtdetrv2", repo=f"{TRAIN}/RT-DETR/rtdetrv2_pytorch",
                            config=f"{TRAIN}/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
                            weights=f"{BASE}/models-UVH-26/weights/RT-DETRv2-X/UVH-26-MV-RT-DETRv2-X.pth",
                            num_classes=15, offset=0, wide=True),
    "iisc_yolo11_s": dict(family="yolo", weights=f"{BASE}/models-UVH-26/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt", offset=1),
    "iisc_yolo11_x": dict(family="yolo", weights=f"{BASE}/models-UVH-26/weights/YOLOv11-X/UVH-26-MV-YOLOv11-X.pt", offset=1),
    # trained on BMD-45 only: cross-dataset, honest only on the de-leaked splits
    "bmd_dfine_x": dict(family="dfine", repo=f"{TRAIN}/D-FINE",
                        config=f"{TRAIN}/D-FINE/configs/dfine/dfine_hgnetv2_x_coco.yml",
                        weights=f"{BASE}/models-BMD-45/weights/D-FINE/best_stg1.pth",
                        num_classes=13, offset=1),
}


def build_detr(p, size=640):
    sys.path.insert(0, p["repo"])
    if p["family"] == "deim":
        from engine.core import YAMLConfig
    else:
        from src.core import YAMLConfig
    cfg = YAMLConfig(p["config"])
    y = cfg.yaml_cfg
    if "num_classes" in p:
        y["num_classes"] = p["num_classes"]
    y["remap_mscoco_category"] = False
    for k in ("PResNet", "HGNetv2"):
        if k in y:
            y[k]["pretrained"] = False
    ck = torch.load(p["weights"], map_location="cpu", weights_only=False)
    state = ck["ema"]["module"] if "ema" in ck else ck["model"]
    if p.get("wide"):  # RT-DETRv2-X widens the encoder and decoder
        width = state["encoder.input_proj.0.conv.weight"].shape[0]
        y["HybridEncoder"].update(hidden_dim=width, dim_feedforward=2048 if width == 384 else 1024)
        y["RTDETRTransformerv2"]["feat_channels"] = [width] * 3
    if p["family"] == "deim":  # our checkpoint trained with the D-FINE activations
        y.setdefault("DFINETransformer", {}).update(activation="relu", mlp_act="relu")
    if size != 640:  # another input size: the anchor grid and positions are rebuilt for it
        y["eval_spatial_size"] = [size, size]
        state = {k: v for k, v in state.items() if not k.endswith(("anchors", "valid_mask"))}
    model = cfg.model
    missing, unexpected = model.load_state_dict(state, strict=(size == 640))
    assert not unexpected and all(k.endswith(("anchors", "valid_mask")) for k in missing), (missing, unexpected)
    return model.deploy().cuda().eval(), cfg.postprocessor.deploy().cuda().eval(), sum(x.numel() for x in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score-thresh", type=float, default=0.001)
    a = ap.parse_args()
    p = PRESETS[a.preset]

    images = json.load(open(a.ann))["images"]
    if a.limit:
        images = images[: a.limit]
    dets, t0 = [], time.time()

    if p["family"] == "yolo":
        from ultralytics import YOLO
        model = YOLO(p["weights"])
        params = sum(x.numel() for x in model.model.parameters())
        for n, im in enumerate(images):
            r = model.predict(os.path.join(a.images, im["file_name"]), imgsz=a.size,
                              conf=a.score_thresh, verbose=False, device=0, max_det=300)[0]
            for b, c, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(), r.boxes.conf.cpu().tolist()):
                dets.append({"image_id": im["id"], "category_id": int(c) + p["offset"],
                             "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
            if (n + 1) % 1000 == 0:
                print(f"  {n + 1}/{len(images)}", flush=True)
    else:
        model, post, params = build_detr(p, a.size)
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((a.size, a.size)), T.ToTensor()])
        with torch.no_grad():
            for n, im in enumerate(images):
                img = Image.open(os.path.join(a.images, im["file_name"])).convert("RGB")
                x = tf(img)[None].cuda()
                sizes = torch.tensor([[img.width, img.height]], device="cuda")
                labels, boxes, scores = post(model(x), sizes)
                for l, b, s in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
                    if s < a.score_thresh:
                        continue
                    dets.append({"image_id": im["id"], "category_id": int(l) + p["offset"],
                                 "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
                if (n + 1) % 1000 == 0:
                    print(f"  {n + 1}/{len(images)}", flush=True)

    json.dump(dets, open(a.out, "w"))
    print(f"{a.preset}: {len(images)} images, {len(dets)} detections, {params / 1e6:.2f}M params, "
          f"{time.time() - t0:.0f}s → {a.out}")


if __name__ == "__main__":
    main()

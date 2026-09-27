#!/usr/bin/env python3
"""Stitch one trained detector's backbone to another's encoder and decoder through learned 1×1
adapters, and train only what has to be trained.

Weights cannot be spliced directly — two networks trained apart put their features in different
bases — so the stitch is a learned connector, which is what the model-stitching literature
actually does (Bansal et al. 2021; SN-Netv2). The source and target stay frozen; the adapters
map source channels and strides onto what the target expects.

  stitch.py --arm T1 --epochs 2 [--train-decoder] [--limit N]

Arms:
  T1  RT-DETRv2-X backbone (r101)  → our D-FINE-S encoder + decoder, adapters only
  T2  same, adapters + decoder trainable
  T4  BMD D-FINE-X backbone        → our D-FINE-S encoder + decoder, adapters only
"""
import argparse, json, os, sys, time

import torch
import torch.nn as nn

TRAIN = "/data/dev-cache/train"
BASE = "/data/datasets/iisc-aim"
OUT = f"{TRAIN}/out/stitch"

ARMS = {
    "T1": dict(source="rtdetrv2_x", train_decoder=False),
    "T2": dict(source="rtdetrv2_x", train_decoder=True),
    "T4": dict(source="bmd_dfine_x", train_decoder=False),
}


def load_source(name):
    """The frozen feature extractor: returns (module, out_channels, strides)."""
    if name == "rtdetrv2_x":
        sys.path.insert(0, f"{TRAIN}/RT-DETR/rtdetrv2_pytorch")
        from src.core import YAMLConfig
        cfg = YAMLConfig(f"{TRAIN}/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml")
        y = cfg.yaml_cfg
        y["num_classes"], y["remap_mscoco_category"], y["PResNet"]["pretrained"] = 15, False, False
        ck = torch.load(f"{BASE}/models-UVH-26/weights/RT-DETRv2-X/UVH-26-MV-RT-DETRv2-X.pth", map_location="cpu", weights_only=False)
        state = ck["ema"]["module"]
        width = state["encoder.input_proj.0.conv.weight"].shape[0]
        y["HybridEncoder"].update(hidden_dim=width, dim_feedforward=2048 if width == 384 else 1024)
        y["RTDETRTransformerv2"]["feat_channels"] = [width] * 3
        model = cfg.model
        model.load_state_dict(state, strict=True)
        return model.backbone, [512, 1024, 2048], [8, 16, 32]
    if name == "bmd_dfine_x":
        sys.path.insert(0, f"{TRAIN}/D-FINE")
        from src.core import YAMLConfig
        cfg = YAMLConfig(f"{TRAIN}/D-FINE/configs/dfine/dfine_hgnetv2_x_coco.yml")
        y = cfg.yaml_cfg
        y["num_classes"], y["remap_mscoco_category"], y["HGNetv2"]["pretrained"] = 13, False, False
        ck = torch.load(f"{BASE}/models-BMD-45/weights/D-FINE/best_stg1.pth", map_location="cpu", weights_only=False)
        model = cfg.model
        model.load_state_dict(ck["ema"]["module"], strict=True)
        return model.backbone, [512, 1024, 2048], [8, 16, 32]
    raise ValueError(name)


def load_target():
    """Our Step 2 model: its encoder, decoder, criterion and the channels it expects."""
    sys.path.insert(0, f"{TRAIN}/DEIM")
    from engine.core import YAMLConfig
    cfg = YAMLConfig(f"{TRAIN}/DEIM/configs/uvh/step2_deim_dfine_s.yml")
    y = cfg.yaml_cfg
    y["HGNetv2"]["pretrained"] = False
    ck = torch.load(f"{TRAIN}/out/step2_deim_dfine_s/best_stg2.pth", map_location="cpu", weights_only=False)
    model = cfg.model
    model.load_state_dict(ck["ema"]["module"], strict=True)
    return cfg, model


class Stitched(nn.Module):
    """source backbone → 1×1 adapters → target encoder → target decoder."""

    def __init__(self, source, src_ch, target, tgt_ch):
        super().__init__()
        self.source = source.eval().requires_grad_(False)
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.Conv2d(s, t, 1, bias=False), nn.BatchNorm2d(t), nn.SiLU()) for s, t in zip(src_ch, tgt_ch)
        )
        self.encoder = target.encoder
        self.decoder = target.decoder

    def forward(self, x, targets=None):
        with torch.no_grad():
            feats = self.source(x)
        feats = [a(f) for a, f in zip(self.adapters, feats)]
        return self.decoder(self.encoder(feats), targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="images per epoch (0 = all)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval", metavar="ANN", help="score this arm on a prepared split instead of training")
    ap.add_argument("--images", default="/data/dev-cache/train/data/images/")
    ap.add_argument("--dets", help="where to write the detections")
    a = ap.parse_args()
    arm = ARMS[a.arm]
    os.makedirs(OUT, exist_ok=True)

    cfg, target = load_target()
    tgt_ch = cfg.yaml_cfg["HybridEncoder"]["in_channels"]
    source, src_ch, _ = load_source(arm["source"])
    model = Stitched(source, src_ch, target, tgt_ch).cuda()

    for p in model.encoder.parameters():
        if p.is_floating_point():
            p.requires_grad_(False)
    for p in model.decoder.parameters():
        if p.is_floating_point():  # integer buffers cannot carry gradients
            p.requires_grad_(arm["train_decoder"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"{a.arm}: source {arm['source']} frozen, adapters {src_ch}→{tgt_ch}, "
          f"decoder {'trainable' if arm['train_decoder'] else 'frozen'}; "
          f"{sum(p.numel() for p in trainable) / 1e6:.2f}M trainable of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    if a.eval:
        import json
        from PIL import Image
        import torchvision.transforms as T
        ck = torch.load(f"{OUT}/{a.arm}.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        post = cfg.postprocessor.deploy().cuda().eval()
        tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        images = json.load(open(a.eval))["images"]
        dets = []
        with torch.no_grad():
            for n, im in enumerate(images):
                img = Image.open(os.path.join(a.images, im["file_name"])).convert("RGB")
                x = tf(img)[None].cuda()
                sizes = torch.tensor([[img.width, img.height]], device="cuda")
                labels, boxes, scores = post(model(x), sizes)
                for l, b, sc in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
                    if sc < 0.001:
                        continue
                    dets.append({"image_id": im["id"], "category_id": int(l),
                                 "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": sc})
                if (n + 1) % 1000 == 0:
                    print(f"  {n + 1}/{len(images)}", flush=True)
        out = a.dets or f"{OUT}/{a.arm}.dets.json"
        json.dump(dets, open(out, "w"))
        print(f"{a.arm}: {len(dets)} detections over {len(images)} images → {out}")
        return

    criterion = cfg.criterion.cuda()
    loader = cfg.train_dataloader
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        model.train()
        model.source.eval()
        for images, targets in loader:
            images = images.cuda()
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda"):
                out = model(images, targets)
                loss = sum(criterion(out, targets).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 0.1)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
            if a.limit and step * loader.batch_size >= a.limit * (epoch + 1):
                break
        torch.save({"model": model.state_dict(), "arm": a.arm, "epoch": epoch}, f"{OUT}/{a.arm}.pth")
        print(f"epoch {epoch} done, saved {OUT}/{a.arm}.pth", flush=True)

    json.dump({"arm": a.arm, "source": arm["source"], "trainable_M": sum(p.numel() for p in trainable) / 1e6,
               "total_M": sum(p.numel() for p in model.parameters()) / 1e6, "steps": step},
              open(f"{OUT}/{a.arm}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

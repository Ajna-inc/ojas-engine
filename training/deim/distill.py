#!/usr/bin/env python3
"""Distil our trained D-FINE-S teacher into the 3.2 M student.

Two models trained apart do not share a coordinate system, so their weights cannot be merged —
but the teacher's *outputs* transfer into any architecture. Two channels are used here:

* **response** — the teacher's 300 predictions are matched one-to-one to the student's by box
  cost, and the student is pulled toward the teacher's class distribution (KL over the 15 logits)
  and its boxes (L1 + GIoU). This is the channel that carries the teacher's judgement on the
  fine classes, which is where 64 % of the error lives.
* **feature** — captured with forward hooks, since the model returns only the decoder's output:
  the student's encoder maps are projected to the teacher's width by a learned 1×1
  conv and matched with a smooth-L1, on the strides both models have (16 and 32). The student is
  narrower (96 vs 256), so the projection is part of the loss, not part of the model.

The ordinary DEIM supervised loss runs throughout; the distillation weight anneals to zero over
the final epochs so the student finishes on the real labels.

  distill.py --student configs/uvh/step3_ojas_n32.yml --out DIR [--alpha 1.0] [--epochs 72]
"""
import argparse, json, math, os, sys, time

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

DEIM = "/data/dev-cache/train/DEIM"
TRAIN = "/data/dev-cache/train"
sys.path.insert(0, DEIM)


def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou_matrix(a, b):
    """Generalised IoU between two sets of cxcywh boxes, normalised coordinates."""
    a, b = box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b)
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    lt_e = torch.min(a[:, None, :2], b[None, :, :2])
    rb_e = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh_e = (rb_e - lt_e).clamp(min=0)
    enclosing = (wh_e[..., 0] * wh_e[..., 1]).clamp(min=1e-7)
    return iou - (enclosing - union) / enclosing


def response_loss(student, teacher, top_k=100, temperature=2.0):
    """Match the teacher's most confident predictions to the student's and pull them together."""
    s_logits, s_boxes = student["pred_logits"], student["pred_boxes"]
    t_logits, t_boxes = teacher["pred_logits"], teacher["pred_boxes"]
    kl_total, box_total, n = 0.0, 0.0, 0
    for i in range(s_logits.shape[0]):
        keep = t_logits[i].sigmoid().max(-1).values.topk(min(top_k, t_logits.shape[1])).indices
        tb, tl = t_boxes[i][keep], t_logits[i][keep]
        with torch.no_grad():
            cost = torch.cdist(tb, s_boxes[i], p=1) - giou_matrix(tb, s_boxes[i])
            rows, cols = linear_sum_assignment(cost.float().cpu().numpy())
        rows = torch.as_tensor(rows, device=s_boxes.device)
        cols = torch.as_tensor(cols, device=s_boxes.device)
        sl, sb = s_logits[i][cols], s_boxes[i][cols]
        # soft targets over the class axis, temperature-scaled as usual
        kl = F.kl_div(F.log_softmax(sl / temperature, -1), F.softmax(tl[rows] / temperature, -1),
                      reduction="batchmean") * temperature ** 2
        box = F.l1_loss(sb, tb[rows]) + (1 - torch.diag(giou_matrix(sb, tb[rows]))).mean()
        kl_total, box_total, n = kl_total + kl, box_total + box, n + 1
    return kl_total / max(n, 1), box_total / max(n, 1)


class FeatureBridge(torch.nn.Module):
    """1×1 projections from the student's encoder width to the teacher's, per shared stride."""

    def __init__(self, s_dim, t_dim, levels):
        super().__init__()
        self.proj = torch.nn.ModuleList(torch.nn.Conv2d(s_dim, t_dim, 1) for _ in range(levels))

    def forward(self, s_feats, t_feats):
        loss = 0.0
        for p, s, t in zip(self.proj, s_feats, t_feats):
            if s.shape[-2:] != t.shape[-2:]:
                continue
            loss = loss + F.smooth_l1_loss(p(s), t.detach())
        return loss / max(len(self.proj), 1)


def build(config, weights=None, train=True):
    from engine.core import YAMLConfig
    cfg = YAMLConfig(config)
    y = cfg.yaml_cfg
    if "HGNetv2" in y:
        y["HGNetv2"]["pretrained"] = False
    model = cfg.model
    if weights:
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"], strict=True)
    return cfg, (model.train() if train else model.eval())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=f"{DEIM}/configs/uvh/step3_ojas_n32.yml")
    ap.add_argument("--teacher", default=f"{DEIM}/configs/uvh/step2_deim_dfine_s.yml")
    ap.add_argument("--teacher-weights", default=f"{TRAIN}/out/step2_deim_dfine_s/best_stg2.pth")
    ap.add_argument("--student-init", default=f"{TRAIN}/weights/dfine_n_coco.pth")
    ap.add_argument("--out", default=f"{TRAIN}/out/step4_distilled_n32")
    ap.add_argument("--epochs", type=int, default=72)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the response loss")
    ap.add_argument("--beta", type=float, default=0.5, help="weight of the feature loss")
    ap.add_argument("--anneal-last", type=int, default=12, help="epochs over which distillation fades to zero")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scfg, student = build(a.student, train=True)
    tcfg, teacher = build(a.teacher, a.teacher_weights, train=False)
    student, teacher = student.cuda(), teacher.cuda().requires_grad_(False)
    if a.student_init and not a.resume:
        ck = torch.load(a.student_init, map_location="cpu", weights_only=False)
        state = ck["ema"]["module"] if "ema" in ck else ck["model"]
        got = student.load_state_dict(state, strict=False)
        print(f"student init: {len(state) - len(got.missing_keys)} tensors matched, "
              f"{len(got.missing_keys)} missing (the narrowed encoder/decoder)", flush=True)

    bridge = FeatureBridge(scfg.yaml_cfg["HybridEncoder"]["hidden_dim"],
                           tcfg.yaml_cfg.get("HybridEncoder", {}).get("hidden_dim", 256),
                           len(scfg.yaml_cfg["DFINETransformer"]["feat_strides"])).cuda()

    # the model returns only the decoder output, so the encoder maps are taken with hooks
    feats = {}
    student.encoder.register_forward_hook(lambda m, i, o: feats.__setitem__("s", o))
    teacher.encoder.register_forward_hook(lambda m, i, o: feats.__setitem__("t", o))

    criterion = scfg.criterion.cuda()
    loader = scfg.train_dataloader
    ema = scfg.ema.cuda() if scfg.yaml_cfg.get("use_ema") else None
    params = list(student.parameters()) + list(bridge.parameters())
    opt = torch.optim.AdamW(params, lr=scfg.yaml_cfg["optimizer"]["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")
    start = 0
    if a.resume and os.path.exists(f"{a.out}/last.pth"):
        ck = torch.load(f"{a.out}/last.pth", map_location="cpu", weights_only=False)
        student.load_state_dict(ck["model"]); bridge.load_state_dict(ck["bridge"])
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start = ck["epoch"] + 1
        print(f"resumed at epoch {start}", flush=True)

    print(f"student {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M · "
          f"teacher {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M · "
          f"bridge {sum(p.numel() for p in bridge.parameters()) / 1e6:.2f}M (training only)", flush=True)

    for epoch in range(start, a.epochs):
        student.train()
        teacher.eval()
        # distillation fades out so the student lands on the real labels
        w = 1.0 if epoch < a.epochs - a.anneal_last else max(0.0, (a.epochs - epoch) / a.anneal_last)
        t0, seen = time.time(), 0
        for i, (samples, targets) in enumerate(loader):
            samples = samples.cuda()
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            with torch.autocast("cuda"):
                with torch.no_grad():
                    t_out = teacher(samples)
                s_out = student(samples, targets=targets)
                loss = sum(criterion(s_out, targets).values())
                if w > 0:
                    kl, box = response_loss(s_out, t_out)
                    loss = loss + w * a.alpha * (kl + box)
                    if "s" in feats and "t" in feats:
                        loss = loss + w * a.beta * bridge(feats["s"], feats["t"])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 0.1)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(student)
            seen += samples.shape[0]
            if i % 100 == 0:
                print(f"epoch {epoch} step {i}/{len(loader)} loss {loss.item():.3f} "
                      f"distil_w {w:.2f} ({time.time() - t0:.0f}s)", flush=True)
        sched.step()
        torch.save({"model": student.state_dict(), "bridge": bridge.state_dict(),
                    "ema": {"module": ema.module.state_dict()} if ema is not None else None,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "epoch": epoch},
                   f"{a.out}/last.pth")
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s ({seen} images)", flush=True)

    json.dump({"student": a.student, "teacher": a.teacher_weights, "epochs": a.epochs,
               "alpha": a.alpha, "beta": a.beta}, open(f"{a.out}/distill.json", "w"), indent=1)


if __name__ == "__main__":
    main()

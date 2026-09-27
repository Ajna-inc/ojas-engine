#!/usr/bin/env python3
"""Add a `person` class to a Field vehicle checkpoint.

Reads the vehicle checkpoint (never writes it) and writes a NEW weights file whose class head has one
more row: UVH ids 0..14 stay where they are, id 15 = person. The new row is copied from the COCO
D-FINE weights' `person` row (COCO label 0 with remap_mscoco_category) in every class-dependent layer:

- decoder.enc_score_head        [nc, 256] / [nc]
- decoder.dec_score_head.{i}    [nc, 256] / [nc]          (one per decoder layer)
- decoder.denoising_class_embed [nc + 1, 256]             (the last row is the padding class and stays last)

Everything else (backbone, encoder, decoder, box heads) is the vehicle model's EMA weights, unchanged.
The source checkpoint's sha256 is printed before and after so a run log proves it was not modified.

    add_person_class.py <vehicle best_stg1.pth> <coco dfine_*_obj2coco.pth> <out.pth> [--coco-row 0]
    add_person_class.py <vehicle best_stg1.pth> fresh <out.pth>

`fresh` in place of the COCO file is for a model whose head has no COCO twin (the 3.39 M ojas model:
hidden 96, 4 decoder layers): the new score rows are drawn like the existing rows (normal with their
std, per layer) and their bias is D-FINE's prior-probability init (p = 0.01, bias -4.595); the new
denoising embedding row is drawn with the existing rows' std. The class is then learned from data only.
"""
import math
import argparse
import hashlib

import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights(ckpt):
    return ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vehicle")
    ap.add_argument("coco")
    ap.add_argument("out")
    ap.add_argument("--coco-row", type=int, default=0, help="COCO person row (0 with remapped categories)")
    a = ap.parse_args()
    if a.out in (a.vehicle, a.coco):
        raise SystemExit("refusing to overwrite an input")
    before = sha256(a.vehicle)
    print(f"source {a.vehicle}\n  sha256 before {before}")

    sd = dict(weights(torch.load(a.vehicle, map_location="cpu", weights_only=False)))
    nc = sd["decoder.enc_score_head.weight"].shape[0]
    r = a.coco_row
    if a.coco == "fresh":
        g = torch.Generator().manual_seed(15)
        prior = -math.log((1 - 0.01) / 0.01)

        def fresh(k, w):
            if k.endswith(".bias"):
                return torch.full((1,), prior, dtype=w.dtype)
            return (torch.randn((1,) + tuple(w.shape[1:]), generator=g) * w[:nc].float().std()).to(w.dtype)

        coco = {k: fresh(k, v) for k, v in sd.items()
                if k.startswith(("decoder.enc_score_head.", "decoder.dec_score_head.")) or k == "decoder.denoising_class_embed.weight"}
        r = 0
    else:
        coco = weights(torch.load(a.coco, map_location="cpu", weights_only=False))
    grown = []
    for k in list(sd):
        if k.startswith("decoder.enc_score_head.") or k.startswith("decoder.dec_score_head."):
            src = coco[k]
            sd[k] = torch.cat([sd[k], src[r : r + 1].to(sd[k].dtype)], 0)
            grown.append((k, tuple(sd[k].shape)))
        elif k == "decoder.denoising_class_embed.weight":
            w = sd[k]
            assert w.shape[0] == nc + 1, f"{k}: expected {nc + 1} rows, got {w.shape[0]}"
            sd[k] = torch.cat([w[:nc], coco[k][r : r + 1].to(w.dtype), w[nc:]], 0)
            grown.append((k, tuple(sd[k].shape)))
    for k, s in grown:
        print(f"  {k} -> {s}")
    assert len(grown) >= 3, "no class-dependent layers found"
    torch.save({"model": sd}, a.out)

    after = sha256(a.vehicle)
    print(f"  sha256 after  {after}  {'unchanged' if after == before else 'CHANGED!'}")
    assert after == before
    print(f"wrote {a.out}: {nc} -> {nc + 1} classes (row {nc} = COCO row {r}, person)")


if __name__ == "__main__":
    main()

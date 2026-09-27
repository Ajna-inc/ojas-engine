#!/usr/bin/env python3
"""Grow the trained Step-3 model (strides 16, 32) into the Step-3b layout (strides 8, 16, 32)
without losing what it learned, so the P3 model can be fine-tuned instead of trained from scratch.

The encoder's blocks are indexed by position, and adding a finer level shifts the bottom-up path:

    step 3 (2 levels)                       step 3b (3 levels)
    input_proj.0   (512 ch, stride 16)  ->  input_proj.1
    input_proj.1   (1024 ch, stride 32) ->  input_proj.2
    lateral_convs.0, fpn_blocks.0 (32→16)   unchanged (the top-down path starts at the top)
    downsample_convs.0, pan_blocks.0 (16→32)  ->  downsample_convs.1, pan_blocks.1
    encoder.encoder.0 (the transformer, on the coarsest level)  unchanged

New and randomly initialised: input_proj.0 (256 ch, stride 8), lateral_convs.1 and fpn_blocks.1
(16→8), downsample_convs.0 and pan_blocks.0 (8→16). The backbone and the whole decoder keep their
weights; the decoder's 12 sampling points per head are re-split [6, 6] → [3, 6, 3] across levels
with the same tensor shapes, so they start from a trained, if re-assigned, layout.

  grow_p3.py <step3 best_stg2.pth> <out.pth>
"""
import sys

import torch

DEIM = "/data/dev-cache/train/DEIM"
CFG = f"{DEIM}/configs/uvh/step3b_ojas_n32_p3.yml"

RENAME = [("encoder.input_proj.1.", "encoder.input_proj.2."),
          ("encoder.input_proj.0.", "encoder.input_proj.1."),
          ("encoder.downsample_convs.0.", "encoder.downsample_convs.1."),
          ("encoder.pan_blocks.0.", "encoder.pan_blocks.1.")]


def rename(k):
    for a, b in RENAME:
        if k.startswith(a):
            return b + k[len(a):]
    return k


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sys.path.insert(0, DEIM)
    from engine.core import YAMLConfig
    cfg = YAMLConfig(CFG)
    y = cfg.yaml_cfg
    y["HGNetv2"]["pretrained"] = False
    y.setdefault("DFINETransformer", {}).update(activation="relu", mlp_act="relu")
    model = cfg.model
    target = model.state_dict()

    ck = torch.load(src, map_location="cpu", weights_only=False)
    old = ck["ema"]["module"] if "ema" in ck else ck["model"]
    grown, carried, skipped = dict(target), 0, []
    for k, v in old.items():
        nk = rename(k)
        if nk.endswith(("anchors", "valid_mask", "num_points_scale")):
            continue  # depend on the level layout; the new model's own are correct
        if nk in target and target[nk].shape == v.shape:
            grown[nk] = v
            carried += 1
        else:
            skipped.append(k)
    fresh = [k for k in target if k not in {rename(k2) for k2 in old}]
    model.load_state_dict(grown, strict=True)
    total = sum(v.numel() for v in target.values() if v.dtype.is_floating_point)
    kept = sum(grown[rename(k)].numel() for k in old if rename(k) in target and rename(k) not in
               ("anchors", "valid_mask") and target[rename(k)].shape == old[k].shape)
    print(f"carried {carried} tensors ({kept / 1e6:.2f} M of {total / 1e6:.2f} M values); "
          f"skipped {len(skipped)}: {skipped[:6]}; new {len(fresh)} tensors, e.g. {fresh[:4]}")
    # a DEIM `-t` (tuning) checkpoint: model weights only, training starts fresh from them
    torch.save({"model": grown}, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()

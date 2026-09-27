#!/usr/bin/env python3
"""Average the EMA weights of several checkpoints from one run into a "soup".

This is the one kind of weight mixing that reliably works: checkpoints from a *single* training
run share an initialisation and a loss basin, so their average is a valid model and usually a
slightly better one than any member. Weights from independently trained models do not share a
basis — neurons are permuted — so splicing or averaging across runs destroys the function unless
the neurons are aligned first.

  soup.py <out.pth> <checkpoint.pth> [<checkpoint.pth> ...] [--uniform|--last N]
"""
import argparse, glob, os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--last", type=int, default=0, help="keep only the N newest of the given files")
    a = ap.parse_args()

    files = sorted(sum((glob.glob(c) for c in a.checkpoints), []))
    if a.last:
        files = files[-a.last:]
    assert len(files) >= 2, f"need at least two checkpoints, got {files}"

    acc, n, template = None, 0, None
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        state = ck["ema"]["module"] if "ema" in ck else ck["model"]
        if acc is None:
            template = ck
            acc = {k: v.clone().to(torch.float64) if v.is_floating_point() else v.clone() for k, v in state.items()}
        else:
            assert set(acc) == set(state), f"{f}: different parameter names"
            for k, v in state.items():
                if v.is_floating_point():
                    acc[k] += v.to(torch.float64)
                # integer buffers (num_batches_tracked) keep the first checkpoint's value
        n += 1
        print(f"  + {os.path.basename(f)}")

    soup = {k: (v / n).to(torch.float32) if v.is_floating_point() else v for k, v in acc.items()}
    # written in the shape the trainers load: model weights and an EMA slot holding the same
    out = {"model": soup, "ema": {"module": soup, "decay": template.get("ema", {}).get("decay", 0.9999) if isinstance(template.get("ema"), dict) else 0.9999},
           "soup_of": files}
    torch.save(out, a.out)
    print(f"averaged {n} checkpoints → {a.out}")


if __name__ == "__main__":
    main()

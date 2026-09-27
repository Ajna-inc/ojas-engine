#!/usr/bin/env python3
"""Per-layer cuDNN timings for a model's convolutions — the numbers the ojas
`cnn` conv kernels have to beat.

Reads the layer list written by the conv-layer survey (one JSON object per
distinct Conv: cin, cout, h, w, k, s, p, g, count) and times
`F.conv2d(x, w)` (no bias: the bias + activation epilogue is a separate
kernel on both sides) with `cudnn.benchmark = True`, i.e. cuDNN's autotuned
best, per batch and dtype. Output: JSON with per-layer median ms and the
count-weighted model total.

  ~/.venvs/ojas-vision/bin/python scripts/release/cudnn_conv_layers.py \
      target/baseline/yolo11n_conv_layers.json --batch 1 8 32
"""
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("layers")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--dtype", nargs="+", default=["f16", "f32"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False  # f32 must stay f32
    layers = json.load(open(args.layers))
    dts = {"f16": torch.float16, "f32": torch.float32}
    rows = []
    for tag in args.dtype:
        for b in args.batch:
            total = 0.0
            for L in layers:
                x = torch.randn(b, L["cin"], L["h"], L["w"], device="cuda", dtype=dts[tag])
                w = torch.randn(L["cout"], L["cin"] // L["g"], L["k"], L["k"], device="cuda", dtype=dts[tag])
                ms = time_ms(lambda: F.conv2d(x, w, None, L["s"], L["p"], 1, L["g"]), args.warmup, args.iters)
                total += ms * L["count"]
                rows.append({**L, "dtype": tag, "batch": b, "ms": round(ms, 4)})
            print(f"{tag} b{b}: cuDNN conv total {total:.2f} ms", flush=True)
            rows.append({"dtype": tag, "batch": b, "total_ms": round(total, 3)})
    print(json.dumps({"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
                      "gpu": torch.cuda.get_device_name(0), "rows": rows}))


if __name__ == "__main__":
    main()

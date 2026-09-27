#!/usr/bin/env python3
"""Measure every benchmarked model's forward time on this GPU under identical conditions —
same stack (PyTorch), same precision, same input size, same batch, same machine — because a
latency read from one repo's README and another's ONNX export is not a comparison.

  latency.py [--presets a,b] [--batches 1,8] [--size 640] [--fp16] [--iters 100]

Reports parameters, p50 and p95 of the forward pass (synchronised), and the per-image cost at
each batch. Preprocessing, decode and postprocessing are excluded and named as excluded.
"""
import argparse, json, os, statistics, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_detections import PRESETS, build_detr  # noqa: E402


def measure(fwd, x, iters, warmup=20):
    for _ in range(warmup):
        fwd(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fwd(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return statistics.median(ts), ts[int(0.95 * len(ts)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="ours_deim_dfine_s,iisc_rtdetrv2_s,iisc_rtdetrv2_x,iisc_yolo11_s,iisc_yolo11_x")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json", dest="out")
    a = ap.parse_args()
    batches = [int(b) for b in a.batches.split(",")]
    rows = []

    for name in a.presets.split(","):
        p = PRESETS[name]
        try:
            if p["family"] == "yolo":
                from ultralytics import YOLO
                m = YOLO(p["weights"]).model.cuda().eval()
                params = sum(x.numel() for x in m.parameters())
                if a.fp16:
                    m = m.half()
                fwd = lambda x: m(x)  # noqa: E731
            else:
                model, post, params = build_detr(p)
                if a.fp16:
                    model = model.half()
                fwd = lambda x: model(x)  # noqa: E731
        except Exception as e:
            print(f"{name}: SKIPPED ({type(e).__name__}: {e})")
            continue
        row = {"model": name, "params_M": params / 1e6, "precision": "fp16" if a.fp16 else "fp32", "size": a.size}
        for b in batches:
            x = torch.randn(b, 3, a.size, a.size, device="cuda", dtype=torch.half if a.fp16 else torch.float)
            with torch.no_grad():
                try:
                    p50, p95 = measure(fwd, x, a.iters)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row[f"batch{b}"] = None
                    continue
            row[f"batch{b}"] = {"p50_ms": p50, "p95_ms": p95, "per_image_ms": p50 / b, "fps": 1000 * b / p50}
        rows.append(row)
        cells = "  ".join(f"b{b} {row[f'batch{b}']['p50_ms']:6.2f} ms ({row[f'batch{b}']['per_image_ms']:5.2f}/img, {row[f'batch{b}']['fps']:6.1f} fps)"
                          for b in batches if row.get(f"batch{b}"))
        print(f"{name:22} {params / 1e6:6.2f}M  {cells}", flush=True)
        del fwd
        torch.cuda.empty_cache()

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    print("\nforward pass only: no decode, no letterbox/preprocessing, no postprocessing or NMS;"
          f" {'fp16' if a.fp16 else 'fp32'} PyTorch on this GPU, not an exported engine")


if __name__ == "__main__":
    main()

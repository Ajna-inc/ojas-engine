#!/usr/bin/env python3
"""DEIM / D-FINE checkpoint + yml → the ONNX files the engine imports, with parity proven.

Steps (nothing else touches the graph):

  1. Build the model exactly as the benchmarks do (`training/benchmark/dump_detections.py::build_detr`):
     the yml's model, `num_classes` from the yml, `pretrained: False` for the backbone, the EMA weights,
     `DFINETransformer: {activation: relu, mlp_act: relu}` (our checkpoints carry the D-FINE
     activations, see deim/README.md), `model.deploy().eval()` on the CPU. The input size is the
     yml's `eval_spatial_size` (640 for every model we train) unless `--size` says otherwise, in which
     case the anchor grid is rebuilt for it (`anchors` / `valid_mask` dropped from the state dict).
  2. `torch.onnx.export` (the TorchScript exporter, `dynamo=False`, opset 17 — LayerNormalization is
     opset 17 and it is what torch 2.8 emits without the dynamo path) with the batch symbolic
     (`dynamic_axes`, dim_param `batch`) and the trace run at batch 2 (`--trace-batch`): the decoder
     has a Python branch `if memory.shape[0] > 1: anchors.repeat(...)`, so a batch-1 trace bakes the
     un-repeated anchors in and the graph then fails at batch N in GatherElements (upstream D-FINE
     traces at batch 32 for the same reason). Input `images` [N,3,S,S] RGB 0–1 stretched (no
     letterbox, no mean/std: the trainer's `ConvertPILImage`); outputs `logits` [N,Q,C] (raw,
     sigmoid on the consumer's side) and `boxes` [N,Q,4] cxcywh normalised by the input square. The
     yml's `num_classes: 15` leaves column 0 (no UVH id 0) dead; the export drops it, so `logits`
     has one column per name in `<out>.classes.json` (`--keep-class0` keeps it; the self-check
     prints the column's largest score to show it never wins).
  3. `onnxslim` — the plain export carries ~1,100 shape-math nodes (Shape/Gather/Concat/Range/
     Where...) and a 2-D constant Slice in the HGNetv2 stem's pad (the exporter's `_prepare_onnx_
     paddings`: Reshape → Slice(step -1) → Transpose) that the engine's importer rejects (`const
     Slice: only 1-D supported`); slimming folds all of it into ~1,000 real nodes with ten `Shape`
     nodes left for the batch. This is `<out>-dyn.onnx`, the file that goes into `models/`; the
     importer binds `batch` (`import_check … batch=1`) and folds the rest.
  4. The static batch-1 file, `<out>.onnx`: bind the batch to 1 in the dynamic graph and slim again
     (every Shape op folds; MatMuls become Gemms). It is what `person_survey` and `graph_run` take,
     since they bind nothing. Rewriting a *slimmed* static export to a dynamic one afterwards
     (`scripts/release/dynamic_batch.py`) does not work for these graphs: slimming folds the batch
     into the sequence dims ([400,1,256] → [-1,256], Gemm pre/post reshapes, [8,32,80,80] for
     grid_sample), so the leading-1 → 0 rewrite produces wrong shapes — hence the order above.
  5. Self-check on `--check` gold frames (PIL `Resize((S,S))` + `ToTensor`, the benchmark's
     preprocessing): the PyTorch model vs onnxruntime on the static file and on the dynamic file at
     batch 1 and at batch N; max |Δ| of scores (sigmoid) and boxes must be under `--tol` (1e-3);
     raw logits sit ~2e-3 apart on the 10 M model, which is 5e-4 on a probability. Random
     inputs are useless for this: the top-300 query selection flips on near-ties and whole rows
     permute (|Δ| ≈ 1.5 on noise, 5e-4 on frames).

  export_onnx.py --config CFG.yml --weights ck.pth --out DIR/name [--size 640] [--check 3]

The engine side of the parity story is `crates/ojas-vision/examples/detr_parity.rs`: it writes the
engine's preprocessed input tensors (`<id>.in.f32`) and decoded detections, and this script's `ref`
mode runs onnxruntime on exactly those tensors so the two decode the same numbers:

  export_onnx.py ref --onnx name.onnx --dir WORK      # writes <id>.ref.0.f32 (logits), .ref.1.f32 (boxes)
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEIM = "/data/dev-cache/train/DEIM"
GOLD = "/data/dev-cache/field-dataset/review_gold/reviewed_gold.json"
UVH14 = ["Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", "Two-wheeler",
         "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Others"]


def build_model(config, weights, repo, size=None):
    """The benchmark's loader (dump_detections.build_detr) on the CPU: deploy-mode model + input size."""
    import torch
    sys.path.insert(0, repo)
    from engine.core import YAMLConfig
    cfg = YAMLConfig(config)
    y = cfg.yaml_cfg
    y["remap_mscoco_category"] = False
    for k in ("PResNet", "HGNetv2"):
        if k in y:
            y[k]["pretrained"] = False
    y.setdefault("DFINETransformer", {}).update(activation="relu", mlp_act="relu")
    eval_size = int(y.get("eval_spatial_size", [640, 640])[0])
    size = size or eval_size
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    state = ck["ema"]["module"] if "ema" in ck else ck["model"]
    if size != eval_size:  # another input size: the anchor grid and positions are rebuilt for it
        y["eval_spatial_size"] = [size, size]
        state = {k: v for k, v in state.items() if not k.endswith(("anchors", "valid_mask"))}
    model = cfg.model
    missing, unexpected = model.load_state_dict(state, strict=(size == eval_size))
    assert not unexpected and all(k.endswith(("anchors", "valid_mask")) for k in missing), (missing, unexpected)
    model = model.deploy().eval()
    params = sum(x.numel() for x in model.parameters())
    return model, size, int(y["num_classes"]), params, ck.get("last_epoch", ck.get("epoch"))


class Head(object):
    """`torch.nn.Module` wrapper built lazily so `ref` mode needs no torch."""

    def __new__(cls, model, drop_class0):
        import torch

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, images):
                out = self.m(images)
                logits, boxes = out["pred_logits"], out["pred_boxes"]
                if drop_class0:
                    logits = logits[:, :, 1:]
                return logits, boxes

        return M().eval()


def load_frames(path, n, size):
    """The benchmark's preprocessing: PIL stretch to the input square, 0–1 RGB."""
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    images = json.load(open(path))["images"][:n]
    return [(im["file_name"], tf(Image.open(im["file_name"]).convert("RGB"))[None]) for im in images]


def det_agreement(wl, wb, gl, gb, conf=0.3):
    """Decoded detections (top class per query, sigmoid ≥ conf): the worst IoU at which every
    reference detection finds a same-class partner; 0 when one is missing."""
    def dets(l, b):
        k = l.argmax(-1)
        s = 1 / (1 + np.exp(-l.max(-1)))
        return [(int(k[q]), b[q]) for q in np.nonzero(s >= conf)[0]]
    def iou(a, b):
        ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
        bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
        i = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
        return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)
    ours = dets(gl, gb)
    worst = 1.0
    for k, b in dets(wl, wb):
        worst = min(worst, max([iou(b, ob) for ok, ob in ours if ok == k] or [0.0]))
    return float(worst)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(a):
    import onnx
    import onnxruntime as ort
    import onnxslim
    import torch

    torch.manual_seed(0)
    model, size, num_classes, params, epoch = build_model(a.config, a.weights, a.repo, a.size)
    drop0 = not a.keep_class0
    head = Head(model, drop0)
    classes = num_classes - 1 if drop0 else num_classes
    print(f"model: {params / 1e6:.2f} M params, {num_classes} logit columns → {classes} exported, "
          f"input {size}, checkpoint epoch {epoch}", flush=True)

    raw, slim_path, dyn_path = a.out + "-raw.onnx", a.out + ".onnx", a.out + "-dyn.onnx"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # traced at batch 2: the decoder's `if memory.shape[0] > 1: anchors.repeat(...)` is a Python branch,
    # and a batch-1 trace bakes the un-repeated anchors in (GatherElements then fails at batch N)
    x = torch.rand(a.trace_batch, 3, size, size)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(head, (x,), raw, input_names=["images"], output_names=["logits", "boxes"],
                          opset_version=a.opset, do_constant_folding=True, dynamo=False,
                          dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}, "boxes": {0: "batch"}})
    m = onnx.load(raw)
    print(f"torch.onnx.export: {len(m.graph.node)} nodes, opset {a.opset}, batch symbolic, {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    dyn = onnxslim.slim(m)
    onnx.checker.check_model(dyn)
    onnx.save(dyn, dyn_path)
    outs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in dyn.graph.output]
    print(f"onnxslim (batch symbolic): {len(dyn.graph.node)} nodes, outputs {outs}, {time.time() - t0:.0f} s → {dyn_path}", flush=True)
    assert [n for n, _ in outs] == ["logits", "boxes"] and outs[0][1] == ["batch", outs[0][1][1], classes] and outs[1][1][2] == 4, outs
    if not a.keep_raw:
        os.remove(raw)

    # the static batch-1 file: bind the batch and slim again — every remaining Shape op folds away
    st = onnx.load(dyn_path)
    for vi in list(st.graph.input) + list(st.graph.output):
        vi.type.tensor_type.shape.dim[0].dim_value = 1
    del st.graph.value_info[:]
    st = onnxslim.slim(st)
    onnx.checker.check_model(st)
    onnx.save(st, slim_path)
    print(f"onnxslim (batch 1): {len(st.graph.node)} nodes → {slim_path}", flush=True)

    # self-check: PyTorch vs onnxruntime (static; dynamic at batch 1 and batch N) on gold frames.
    # Rows are compared permutation-invariantly: the decoder keeps the top 300 of 8400 anchors, and a
    # near-tie at the cutoff makes the two runtimes pick or order queries differently (a row-aligned
    # |Δ| then reads ~1.7 on one frame in three while every detection is identical).
    frames = load_frames(a.frames, a.check, size)
    xs = torch.cat([t for _, t in frames])
    with torch.no_grad():
        ref = [o.numpy() for o in head(xs)]
    worst, worst_iou, class0 = 0.0, 1.0, 0.0
    if drop0:
        with torch.no_grad():
            full = model(xs)["pred_logits"]
        class0 = float(torch.sigmoid(full[:, :, 0]).max())
    for name, path in (("static", slim_path), ("dyn", dyn_path)):
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        runs = [(s.run(None, {"images": xs[i:i + 1].numpy()}), [r[i:i + 1] for r in ref]) for i in range(len(frames))]
        if name == "dyn":
            runs.append((s.run(None, {"images": xs.numpy()}), ref))
        for got, want in runs:
            aligned = max(float(np.abs(g - w).max()) for g, w in zip(got, want))
            rows = 0.0
            for b in range(want[0].shape[0]):
                sig = lambda l: 1 / (1 + np.exp(-l))  # noqa: E731 — the tolerance is on scores and boxes
                w = np.concatenate([sig(want[0][b]), want[1][b]], -1)  # [Q, C+4]
                g = np.concatenate([sig(got[0][b]), got[1][b]], -1)
                nearest = np.abs(w[:, None, :] - g[None, :, :]).max(-1).min(-1)  # per PyTorch row: closest ORT row
                rows = max(rows, float(nearest.max()))
                worst_iou = min(worst_iou, det_agreement(want[0][b], want[1][b], got[0][b], got[1][b]))
            worst = max(worst, rows)
            print(f"  {name:<6} batch {got[0].shape[0]}: max |Δ| vs PyTorch {rows:.2e} on scores and boxes (row-aligned raw {aligned:.2e})", flush=True)
    top = [float(1 / (1 + np.exp(-ref[0][i].max()))) for i in range(len(frames))]
    print(f"self-check on {len(frames)} gold frames: worst |Δ| {worst:.2e} (tol {a.tol}), detections ≥ 0.3 all matched at IoU ≥ {worst_iou:.4f}; "
          f"top scores {['%.3f' % t for t in top]}" + (f"; dropped column 0 peaks at {class0:.4f}" if drop0 else ""), flush=True)
    assert worst < a.tol, f"onnxruntime differs from PyTorch by {worst}"

    if drop0 and classes == len(UVH14):
        json.dump(UVH14, open(a.out + ".classes.json", "w"))
    for p in (slim_path, dyn_path):
        print(f"{sha256(p)}  {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")


def reference(a):
    """onnxruntime on the engine's own input tensors (`<id>.in.f32`, [1,3,S,S] little-endian f32)."""
    import onnxruntime as ort
    s = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    shape = [d if isinstance(d, int) else 1 for d in s.get_inputs()[0].shape]
    files = sorted(f for f in os.listdir(a.dir) if f.endswith(".in.f32"))
    for f in files:
        x = np.fromfile(os.path.join(a.dir, f), dtype=np.float32).reshape(shape)
        outs = s.run(None, {"images": x})
        for i, o in enumerate(outs):
            o.astype(np.float32).tofile(os.path.join(a.dir, f[:-len(".in.f32")] + f".ref.{i}.f32"))
    print(f"{len(files)} inputs → reference outputs in {a.dir} ({a.onnx})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    e = sub.add_parser("export", help="checkpoint → slim + dynamic ONNX (the default)")
    e.add_argument("--config", required=True)
    e.add_argument("--weights", required=True)
    e.add_argument("--out", required=True, help="path prefix: writes <out>.onnx, <out>-dyn.onnx, <out>.classes.json")
    e.add_argument("--repo", default=DEIM)
    e.add_argument("--size", type=int, default=0, help="input square; default: the yml's eval_spatial_size")
    e.add_argument("--opset", type=int, default=17)
    e.add_argument("--keep-class0", action="store_true")
    e.add_argument("--keep-raw", action="store_true")
    e.add_argument("--frames", default=GOLD)
    e.add_argument("--check", type=int, default=3)
    e.add_argument("--trace-batch", type=int, default=2, help="dummy batch for the trace (must be > 1)")
    e.add_argument("--tol", type=float, default=1e-3)
    r = sub.add_parser("ref", help="onnxruntime outputs for the engine's dumped inputs")
    r.add_argument("--onnx", required=True)
    r.add_argument("--dir", required=True)
    argv = sys.argv[1:]
    if argv and argv[0] not in ("export", "ref", "-h", "--help"):
        argv = ["export"] + argv
    a = ap.parse_args(argv)
    if a.mode == "ref":
        reference(a)
    else:
        export(a)


if __name__ == "__main__":
    main()

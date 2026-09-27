#!/usr/bin/env python3
"""Golden per-node dumps + CPU baseline timings for ONNX models, via onnxruntime.

Reference tooling only (never in the runtime path). Two jobs:

  dump   Run the model once on a deterministic input and write every
         intermediate tensor as raw little-endian f32 next to a manifest.json.
         The ojas-vision CPU executor is gated against these files.
  bench  Median wall time per forward on the ORT CPU EP - the number the
         ojas-vision CPU path is measured against.

Graph optimizations are DISABLED for dumps so intermediates correspond 1:1 to
the untransformed graph; bench runs with optimizations ON (beat ORT at its best).
"""
import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def sanitize(name: str, used: set) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "_"
    base, i = s, 1
    while s in used:
        s = f"{base}.{i}"
        i += 1
    used.add(s)
    return s


def bind_dims(model: onnx.ModelProto, binds: dict) -> None:
    """Fix dim_params (batch, height, ...) to concrete values; error on unbound."""
    unbound = []
    for vi in list(model.graph.input):
        tt = vi.type.tensor_type
        for d in tt.shape.dim:
            if d.dim_param:
                if d.dim_param in binds:
                    v = binds[d.dim_param]
                    d.ClearField("dim_param")
                    d.dim_value = v
                else:
                    unbound.append((vi.name, d.dim_param))
    if unbound:
        sys.exit(f"unbound dim_params {unbound}; pass --bind name=value")


def make_inputs(sess: ort.InferenceSession, seed: int, supplied: dict) -> dict:
    rng = np.random.default_rng(seed)
    feeds = {}
    for i in sess.get_inputs():
        if i.name in supplied:
            arr = np.fromfile(supplied[i.name], dtype=np.float32)
            arr = arr.reshape(i.shape)
        else:
            shape = [d if isinstance(d, int) else 1 for d in i.shape]
            if "float" not in i.type:
                sys.exit(f"input {i.name} has type {i.type}; supply it with --input")
            # image-like range [0,1): what a normalised frame feeds the net
            arr = rng.random(shape, dtype=np.float32)
        feeds[i.name] = arr
    return feeds


def expose_all_outputs(model: onnx.ModelProto) -> list:
    """Add every node output to graph.output so ORT returns intermediates."""
    existing = {o.name for o in model.graph.output}
    inits = {t.name for t in model.graph.initializer}
    added = []
    for node in model.graph.node:
        for out in node.output:
            if out and out not in existing and out not in inits:
                model.graph.output.append(onnx.helper.make_empty_tensor_value_info(out))
                existing.add(out)
                added.append((out, node.op_type))
    return added


def cmd_dump(args, model_bytes: bytes) -> None:
    model = onnx.load_from_string(model_bytes)
    binds = dict(kv.split("=") for kv in args.bind)
    bind_dims(model, {k: int(v) for k, v in binds.items()})
    producers = {out: n.op_type for n in model.graph.node for out in n.output}
    expose_all_outputs(model)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 1  # bit-stable reference
    sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])

    feeds = make_inputs(sess, args.seed, dict(kv.split("=") for kv in args.input))
    names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(names, feeds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    used, manifest = set(), {}
    for name, arr in list(feeds.items()) + list(zip(names, outputs)):
        entry = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if name in producers:
            entry["op_type"] = producers[name]
        if arr.dtype in (np.float32, np.float16, np.float64):
            f = sanitize(name, used) + ".f32"
            arr.astype(np.float32).tofile(out_dir / f)
            entry["file"] = f
        else:
            f = sanitize(name, used) + ".i64"
            arr.astype(np.int64).tofile(out_dir / f)
            entry["file"] = f
        manifest[name] = entry
    meta = {
        "model": args.model,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "onnxruntime": ort.__version__,
        "seed": args.seed,
        "bind": binds,
        "inputs": sorted(feeds),
        "graph_outputs_original": [o.name for o in onnx.load_from_string(model_bytes).graph.output],
        "tensors": manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"dumped": len(manifest), "dir": str(out_dir)}))


def cmd_bench(args, model_bytes: bytes) -> None:
    results = {}
    for threads in args.threads:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = threads
        sess = ort.InferenceSession(model_bytes, opts, providers=["CPUExecutionProvider"])
        feeds = make_inputs(sess, args.seed, dict(kv.split("=") for kv in args.input))
        for _ in range(args.warmup):
            sess.run(None, feeds)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            sess.run(None, feeds)
            times.append((time.perf_counter() - t0) * 1e3)
        results[threads] = {
            "median_ms": round(statistics.median(times), 3),
            "p90_ms": round(sorted(times)[int(len(times) * 0.9)], 3),
            "min_ms": round(min(times), 3),
        }
    print(json.dumps({
        "model": args.model,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "onnxruntime": ort.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "per_threads": results,
    }, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["dump", "bench"])
    p.add_argument("model")
    p.add_argument("--out", default="golden/onnx", help="dump: output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input", action="append", default=[], metavar="NAME=FILE.f32",
                   help="supply an input tensor from a raw little-endian f32 file")
    p.add_argument("--bind", action="append", default=[], metavar="DIM=VALUE",
                   help="bind a symbolic dim_param, e.g. batch=1")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 8],
                   help="bench: intra-op thread counts to measure")
    args = p.parse_args()
    model_bytes = Path(args.model).read_bytes()
    if args.mode == "dump":
        cmd_dump(args, model_bytes)
    else:
        cmd_bench(args, model_bytes)


if __name__ == "__main__":
    main()

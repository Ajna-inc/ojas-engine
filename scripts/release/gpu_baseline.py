#!/usr/bin/env python3
"""Python-on-GPU baselines for ojas-vision's CUDA backend.

Reference tooling only (never in the runtime path). Before any `cnn` CUDA
kernel is written, this records what the usual Python stacks do on the same
GPU, with the same ONNX file, so the Ojas numbers have something to beat:

  ort-cuda   onnxruntime CUDA EP, fp32 (same graph Ojas imports)
  ort-cuda16 onnxruntime CUDA EP on an fp16-converted graph (keep_io_types)
  ort-trt    onnxruntime TensorRT EP, fp16 (the practical ceiling; opt-in via
             `--only ... ort-trt`, needs the 4 GB `tensorrt-cu12` wheel)
  torch      the Ultralytics nn.Module on CUDA, fp32 and fp16 (`--pt`)

Every ORT row is timed two ways:
  e2e     host array in -> host array out (what a naive worker pays, PCIe included)
  device  IO-bound: input already on the GPU, output left there (kernel time)

and every row is parity-checked against the ORT CPU EP on the same input
(cosine on the first output), so a fast-but-wrong provider cannot pass.

Batch comes from the model's input shape; for a dynamic export pass
`--bind batch=N`. Output is one JSON document on stdout.

  python3 scripts/release/gpu_baseline.py models/yolo11n.onnx --pt models/yolo11n.pt
"""
import argparse
import contextlib
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from onnx_reference import bind_dims  # noqa: E402


def gpu_info() -> dict:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
        name, driver, mem = [s.strip() for s in q.split(",")]
        return {"gpu": name, "driver": driver, "memory": mem}
    except (OSError, subprocess.CalledProcessError, IndexError) as e:
        sys.exit(f"nvidia-smi failed ({e}); the NVIDIA driver is not loaded")


def stats(times_ms: list) -> dict:
    s = sorted(times_ms)
    return {
        "median_ms": round(statistics.median(s), 3),
        "p90_ms": round(s[int(len(s) * 0.9)], 3),
        "min_ms": round(s[0], 3),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def timed(fn, warmup: int, iters: int) -> list:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def ort_session(model_bytes: bytes, providers: list) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_bytes, opts, providers=providers)


def bench_ort(label, model_bytes, providers, x, want, args) -> dict:
    t0 = time.perf_counter()
    try:
        sess = ort_session(model_bytes, providers)
    except Exception as e:  # missing EP libraries -> recorded, not fatal
        return {"label": label, "error": str(e).splitlines()[0]}
    active = sess.get_providers()[0]
    if active != providers[0][0]:
        return {"label": label, "error": f"{providers[0][0]} not active (got {active})"}
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    got = sess.run(out_names, {in_name: x})[0]  # first run also builds TRT engines
    setup_s = round(time.perf_counter() - t0, 2)

    e2e = timed(lambda: sess.run(out_names, {in_name: x}), args.warmup, args.iters)

    x_dev = ort.OrtValue.ortvalue_from_numpy(x, "cuda", args.device)
    io = sess.io_binding()
    io.bind_ortvalue_input(in_name, x_dev)
    for n in out_names:
        io.bind_output(n, "cuda", args.device)
    device = timed(lambda: sess.run_with_iobinding(io), args.warmup, args.iters)

    batch = x.shape[0]
    med = statistics.median(device)
    return {
        "label": label,
        "provider": active,
        "setup_s": setup_s,
        "e2e": stats(e2e),
        "device": stats(device),
        "fps_device": round(batch * 1e3 / med, 1),
        "cosine_vs_cpu": round(cosine(got, want), 7),
    }


def bench_torch(pt_path, x, want, args) -> list:
    import torch
    from ultralytics import YOLO

    with contextlib.redirect_stdout(sys.stderr):  # fuse() prints a summary; stdout is the JSON
        net = YOLO(pt_path).model.to(f"cuda:{args.device}").eval().fuse()
    rows = []
    for half in (False, True):
        m = net.half() if half else net.float()
        xt = torch.from_numpy(x).to(f"cuda:{args.device}")
        xt = xt.half() if half else xt

        @torch.inference_mode()
        def run():
            y = m(xt)
            torch.cuda.synchronize()
            return y

        y = run()
        y = y[0] if isinstance(y, (list, tuple)) else y
        device = timed(run, args.warmup, args.iters)
        rows.append({
            "label": "torch-fp16" if half else "torch-fp32",
            "torch": torch.__version__,
            "device": stats(device),
            "fps_device": round(x.shape[0] * 1e3 / statistics.median(device), 1),
            "cosine_vs_cpu": round(cosine(y.float().cpu().numpy(), want), 7),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--pt", help="Ultralytics .pt for the torch rows (needs `ultralytics`)")
    p.add_argument("--bind", action="append", default=[], metavar="DIM=VALUE")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--trt-cache", default="target/trt_cache")
    p.add_argument("--only", nargs="+", default=["ort-cuda", "ort-cuda16", "torch"])
    args = p.parse_args()

    model_bytes = Path(args.model).read_bytes()
    model = onnx.load_from_string(model_bytes)
    bind_dims(model, {k: int(v) for k, v in (kv.split("=") for kv in args.bind)})
    model_bytes_bound = model.SerializeToString()
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    x = np.random.default_rng(args.seed).random(shape, dtype=np.float32)

    cpu = ort_session(model_bytes_bound, ["CPUExecutionProvider"])
    want = cpu.run(None, {cpu.get_inputs()[0].name: x})[0]

    cuda = ("CUDAExecutionProvider", {"device_id": args.device, "cudnn_conv_algo_search": "EXHAUSTIVE"})
    rows = []
    if "ort-cuda" in args.only:
        rows.append(bench_ort("ort-cuda-fp32", model_bytes_bound, [cuda, "CPUExecutionProvider"], x, want, args))
    if "ort-cuda16" in args.only:
        from onnxconverter_common import float16
        # Stale value_info types clash with the casts the converter inserts
        # around blocked ops; drop them and let ORT re-infer.
        src = onnx.ModelProto()
        src.CopyFrom(model)
        del src.graph.value_info[:]
        m16 = float16.convert_float_to_float16(src, keep_io_types=True, op_block_list=["Resize"]).SerializeToString()
        rows.append(bench_ort("ort-cuda-fp16", m16, [cuda, "CPUExecutionProvider"], x, want, args))
    if "ort-trt" in args.only:
        Path(args.trt_cache).mkdir(parents=True, exist_ok=True)
        trt = ("TensorrtExecutionProvider", {
            "device_id": args.device, "trt_fp16_enable": True,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": args.trt_cache,
        })
        rows.append(bench_ort("ort-trt-fp16", model_bytes_bound, [trt, cuda, "CPUExecutionProvider"], x, want, args))
    if "torch" in args.only and args.pt:
        rows.extend(bench_torch(args.pt, x, want, args))

    print(json.dumps({
        "model": args.model,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "input_shape": shape,
        "host": {**gpu_info(), "cpu": platform.processor() or platform.machine(),
                 "python": platform.python_version(), "onnxruntime": ort.__version__},
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()

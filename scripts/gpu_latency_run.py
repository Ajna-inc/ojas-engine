#!/usr/bin/env python3
"""Measure available batch-one detector artifacts on this machine's CUDA backend."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[1]
out = repo / 'evidence/gpu_latency_2026_09_20'
out.mkdir(exist_ok=False)
cuda = Path('/usr/local/lib/python3/site-packages/nvidia')
env = dict(os.environ, LD_LIBRARY_PATH=f'{cuda}/cuda_nvrtc/lib:{cuda}/cublas/lib', OJAS_CUDA_INCLUDE=str(cuda / 'cuda_runtime/include'))
models = [repo / 'models/yolo11n.onnx',
          Path('/data/dev-cache/detr/rtdetr_v2_r18_slim.onnx'),
          Path('/data/dev-cache/detr/dfine_s_slim.onnx')]
(out / 'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
for model in models:
    command = [str(repo / 'target/release/examples/gpu_latency'), str(model), '100']
    print(f'Benchmarking {model.name}', flush=True)
    run = subprocess.run(command, env=env, capture_output=True, text=True)
    (out / f'{model.stem}.log').write_text(run.stdout + run.stderr)
    if run.returncode:
        print(f'FAILED {model.name}: {run.stderr[-1000:]}', flush=True)
        continue
    result = json.loads(run.stdout.strip().splitlines()[-1])
    result['sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
    result['command'] = command
    result['binary_sha256'] = hashlib.sha256((repo / 'target/release/examples/gpu_latency').read_bytes()).hexdigest()
    (out / f'{model.stem}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)

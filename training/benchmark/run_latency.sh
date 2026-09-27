#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# waits for the accuracy benchmark to release the GPU, then measures every model's forward time
set -u
O="/data/dev-cache/train/out/benchmark"
until grep -q "BENCH DONE" "/data/dev-cache/train/out/benchmark.log" 2>/dev/null; do sleep 120; done
export PYTHONUNBUFFERED=1
echo "== fp32"; python3 $REPO/training/benchmark/latency.py --json "$O/latency_fp32.json"
echo "== fp16"; python3 $REPO/training/benchmark/latency.py --fp16 --json "$O/latency_fp16.json"
echo "== LATENCY DONE"

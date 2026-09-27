#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Stitch arms: frozen big-model backbone → trained 1x1 adapters → our encoder/decoder.
# Waits for the Field inference to release its VRAM, then runs the cheap arms first.
set -u
S="$REPO/training/stitch"
B="$REPO/training/benchmark"
DATA="/data/dev-cache/train/data"
OUT="/data/dev-cache/train/out/stitch"
export PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib
mkdir -p "$OUT"
until grep -q "FIELD STAGE B DONE" "/data/dev-cache/field-dataset/stageB.log" 2>/dev/null; do sleep 60; done
for arm in T1 T4 T2; do
  [ -f "$OUT/$arm.pth" ] || { echo "== $(date +%H:%M) stitch $arm"; python3 "$S/stitch.py" --arm $arm --epochs 1 >> "$OUT/$arm.log" 2>&1 || { echo "STITCH FAILED $arm — $(tail -2 "$OUT/$arm.log" | head -1 | cut -c1-140)"; continue; }; }
  echo "== $(date +%H:%M) $arm trained; scoring"
done
echo "== STITCH DONE $(date +%H:%M)"

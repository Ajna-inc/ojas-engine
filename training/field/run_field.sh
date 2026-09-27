#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Field stage B: ensemble pre-labels on the extracted frames, the no-label domain-gap
# measurement, and the gold-set selection. Resumable — each step skips its own finished output.
set -u
G="$REPO/training/field"
F="/data/dev-cache/field-dataset/day/frames.jsonl"
O="/data/dev-cache/field-dataset/labels"
export PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib
mkdir -p "$O"
until grep -q "LATENCY DONE" "/data/dev-cache/train/out/latency.log" 2>/dev/null; do sleep 60; done
python3 "$G/infer_ensemble.py" --frames "$F" --out "$O" --presets ours_deim_dfine_s,iisc_rtdetrv2_x || exit 1
echo "== agreement (no labels needed)"
python3 "$G/agreement.py" "$O/frames_coco.json" "$O/ours_deim_dfine_s.dets.json" "$O/iisc_rtdetrv2_x.dets.json" --out "$O/review_queue.jsonl" | tee "$O/agreement.txt"
echo "== gold selection"
python3 "$G/select_gold.py" "$O/frames_coco.json" "$O/ours_deim_dfine_s.dets.json" --n 300 --out "$O/gold_frames.json" | tee "$O/gold_selection.txt"
echo "== FIELD STAGE B DONE"

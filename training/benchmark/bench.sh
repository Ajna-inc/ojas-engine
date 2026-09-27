#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# One accuracy benchmark for every detector we can run, scored by one evaluator on the same
# prepared splits. RESUMABLE: rerun it — a model×split pair whose
# detections and score already exist is skipped, so a crash or a reboot costs one pair.
#
# Splits: uvh_val_st (the benchmark), uvh_val_mv (IISc's own label convention),
# uvh_val_st_clean (ST val minus every frame that appears in BMD-45 train — the only split on
# which a BMD-trained model can be scored honestly).
set -u
B="$REPO/training/benchmark"
DATA="/data/dev-cache/train/data"
OUT="/data/dev-cache/train/out/benchmark"; mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
until grep -qE "STEP2 DONE|TRAIN FAILED" "/data/dev-cache/train/out/step2_pipeline.log" 2>/dev/null; do sleep 120; done

MODELS="ours_deim_dfine_s iisc_rtdetrv2_s iisc_rtdetrv2_x iisc_yolo11_s iisc_yolo11_x bmd_dfine_x"
SPLITS="uvh_val_st uvh_val_mv uvh_val_st_clean"

for m in $MODELS; do
  for s in $SPLITS; do
    # a BMD-trained model on a split that still contains its training frames would be meaningless
    [ "$m" = "bmd_dfine_x" ] && [ "$s" != "uvh_val_st_clean" ] && continue
    d="$OUT/${m}__${s}.dets.json"; r="$OUT/${m}__${s}.score.json"
    if [ ! -f "$r" ]; then
      if [ ! -f "$d" ]; then
        echo "== $(date +%H:%M) detect $m on $s"
        python3 "$B/dump_detections.py" --preset "$m" --ann "$DATA/annotations/$s.json" \
          --images "$DATA/images/" --out "$d" > "$OUT/${m}__${s}.log" 2>&1 || {
            echo "DETECT FAILED $m $s — $(tail -2 "$OUT/${m}__${s}.log" | head -1 | cut -c1-140)"; rm -f "$d"; continue; }
      fi
      python3 "$B/score_detections.py" "$DATA/annotations/$s.json" "$d" --label "$m on $s" --json "$r" \
        >> "$OUT/${m}__${s}.log" 2>&1 || { echo "SCORE FAILED $m $s"; continue; }
    fi
    python3 - "$r" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"BENCH {d['label']}  mAP {d['mAP']:.4f}  AP50 {d['AP50']:.4f}  AP75 {d['AP75']:.4f}  "
      f"small {d['AP_small']:.4f}  AR100 {d['AR100']:.4f}")
EOF
  done
done
echo "== $(date +%H:%M) BENCH DONE"
python3 "$B/table.py" "$OUT" || true

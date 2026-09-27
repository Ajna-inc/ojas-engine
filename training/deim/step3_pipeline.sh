#!/bin/bash
# Step 2 on the official DEIM trainer. RESUMABLE — rerun after any interruption:
# waits for Step 1c to release the GPU, resumes from last.pth if present, retries a crashed trainer, skips
# finished evaluations. Scores best.pth and last.pth on ST val, MV val, the de-leaked ST val and de-leaked BMD val.
set -u
R="/data/dev-cache/train/DEIM"
DATA="/data/dev-cache/train/data"
OUT="/data/dev-cache/train/out"
INIT="/data/dev-cache/train/weights/dfine_n_coco.pth"
LOGS="$OUT/step3_logs"; mkdir -p "$LOGS"
cd "$R"
export PYTHONUNBUFFERED=1   # the trainer prints through a block-buffered stdout when redirected
until grep -q "LATENCY DONE" "$OUT/latency.log" 2>/dev/null; do sleep 120; done
ap() { grep -oE "Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = [0-9.]+" "$1" 2>/dev/null | tail -1 | awk '{print $NF}'; }
evaluate() {  # <label> <config> <weights> <split-json-stem>
  local label=$1 cfg=$2 w=$3 split=$4 log="$LOGS/eval_$1_on_$4.log"
  if [ -z "$(ap "$log")" ]; then
    python3 train.py -c "$cfg" -r "$w" --test-only -u val_dataloader.dataset.ann_file="$DATA/annotations/$split.json" > "$log" 2>&1
  fi
  echo "RESULT $label $split AP $(ap "$log")"
}
for arm in ojas_n32; do
  cfg="configs/uvh/step3_$arm.yml"; dir="$OUT/step3_$arm"
  if [ ! -f "$dir/DONE" ]; then
    for attempt in 1 2 3 4 5 6 7 8; do
      if [ -f "$dir/last.pth" ]; then
        echo "== $(date +%d/%H:%M) $arm resume from last.pth (attempt $attempt, last_epoch $(python3 -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['last_epoch'])" "$dir/last.pth" 2>/dev/null))"
        python3 train.py -c "$cfg" -r "$dir/last.pth" --use-amp --seed 1 >> "$LOGS/train_$arm.log" 2>&1 && { touch "$dir/DONE"; break; }
      else
        echo "== $(date +%d/%H:%M) $arm start from $(basename "$INIT") (attempt $attempt)"
        python3 train.py -c "$cfg" -t "$INIT" --use-amp --seed 1 >> "$LOGS/train_$arm.log" 2>&1 && { touch "$dir/DONE"; break; }
      fi
      echo "   train.py exited $? — $(tail -1 "$LOGS/train_$arm.log" | cut -c1-160)"; sleep 60
    done
    [ -f "$dir/DONE" ] || { echo "TRAIN FAILED $arm after 8 attempts"; continue; }
  fi
  grep -oE "best_stat: .*" "$LOGS/train_$arm.log" | tail -1
  for ck in best_stg2 last; do for split in uvh_val_st uvh_val_mv uvh_val_st_clean bmd_val_clean; do evaluate "${arm}_${ck}" "$cfg" "$dir/$ck.pth" $split; done; done
done
echo "== $(date +%d/%H:%M) STEP3 DONE"

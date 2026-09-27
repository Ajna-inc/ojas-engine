#!/bin/bash
# Step 1 on the official RT-DETRv2 trainer. RESUMABLE: rerun this
# script after any interruption and it continues — finished stages are skipped by their
# outputs, an interrupted arm resumes from its last.pth (model, EMA, optimizer, schedulers,
# scaler, epoch are all in it; the trainer saves it every epoch), and a crashed train.py is
# retried from last.pth up to 5 times.
#   0. sanity: IISc's own S checkpoint on the resized ST/MV val (must land near 0.5629 / 0.6179)
#   1. arm st: warm start, ST labels, cosine 12 epochs   2. arm mv: same, MV labels (control)
#   3. each arm's best.pth and last.pth scored on both val splits → RESULT lines.
set -u
R="/data/dev-cache/train/RT-DETR/rtdetrv2_pytorch"
DATA="/data/dev-cache/train/data"
OUT="/data/dev-cache/train/out"
IISC="/data/datasets/iisc-aim/models-UVH-26/weights/RT-DETRv2-S/UVH-26-MV-RT-DETRv2-S.pth"
LOGS="$OUT/step1_logs"; mkdir -p "$LOGS"
cd "$R"
until [ -f "$DATA/annotations/uvh_val_st.json" ] && [ -f "$DATA/annotations/uvh_train_st.json" ] && [ -f "$DATA/annotations/uvh_val_mv.json" ]; do sleep 60; done

ap() { grep -oE "Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = [0-9.]+" "$1" 2>/dev/null | tail -1 | awk '{print $NF}'; }
# evaluate <label> <config> <weights-flag> <weights> <split>  — skipped when its log already holds an AP
evaluate() {
  local label=$1 cfg=$2 flag=$3 w=$4 split=$5 log="$LOGS/eval_$1_on_$5.log"
  if [ -z "$(ap "$log")" ]; then
    python3 tools/train.py -c "$cfg" $flag "$w" --test-only -u val_dataloader.dataset.ann_file="$DATA/annotations/uvh_val_$split.json" > "$log" 2>&1
  fi
  echo "RESULT $label ${split}_val AP $(ap "$log")"
}

echo "== $(date +%H:%M) sanity"
for split in st mv; do evaluate sanity_iisc_s configs/uvh/step1_recipe_s_st.yml -t "$IISC" $split; done

for arm in st mv; do
  cfg="configs/uvh/step1_recipe_s_$arm.yml"; dir="$OUT/step1_recipe_s_$arm"
  if [ ! -f "$dir/DONE" ]; then
    for attempt in 1 2 3 4 5; do
      if [ -f "$dir/last.pth" ]; then
        echo "== $(date +%H:%M) arm $arm resume from last.pth (attempt $attempt, last_epoch $(python3 -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['last_epoch'])" "$dir/last.pth" 2>/dev/null))"
        python3 tools/train.py -c "$cfg" -r "$dir/last.pth" --use-amp --seed 1 >> "$LOGS/train_$arm.log" 2>&1 && { touch "$dir/DONE"; break; }
      else
        echo "== $(date +%H:%M) arm $arm start from IISc weights (attempt $attempt)"
        python3 tools/train.py -c "$cfg" -t "$IISC" --use-amp --seed 1 >> "$LOGS/train_$arm.log" 2>&1 && { touch "$dir/DONE"; break; }
      fi
      echo "   train.py exited $? — $(tail -1 "$LOGS/train_$arm.log" | cut -c1-160)"; sleep 30
    done
    [ -f "$dir/DONE" ] || { echo "TRAIN FAILED $arm after 5 attempts"; continue; }
  fi
  grep -oE "best_stat: .*" "$LOGS/train_$arm.log" | tail -1
  for ck in best last; do for split in st mv; do evaluate "arm_${arm}_${ck}" "$cfg" -r "$dir/$ck.pth" $split; done; done
done
echo "== $(date +%H:%M) STEP1 DONE"

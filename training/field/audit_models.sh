#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Score every model version on every test set with one evaluator, writing one JSON per (model, size,
# set) into field-dataset/ft_eval/scores/.
# Sets: Field day (261, cam01/05/09), night (371, 16 night views), new-camera day (220, 11 cameras),
# and UVH-26 ST val (4,339, Bengaluru) to measure what the Field fine-tunes forgot.
set -u
cd $REPO/training/benchmark
B="/data/dev-cache/field-dataset"; T="/data/dev-cache/train/data"
O="$B/ft_eval/audit"; mkdir -p "$O/scores"
declare -A FR=( [day]="$B/labels/gold_frames.json" [night]="$B/grid_r4/test_frames.json" [newcam]="$B/morning5/test_frames.json" [uvh]="$T/annotations/uvh_val_st.json" )
declare -A GT=( [day]="$B/review_gold/reviewed_gold.json" [night]="$B/grid_r4/review_test/reviewed_gold.json" [newcam]="$B/morning5/review_test/reviewed_gold.json" [uvh]="$T/annotations/uvh_val_st.json" )
declare -A IM=( [day]=/ [night]=/ [newcam]=/ [uvh]="$T/images" )
for job in "$@"; do
  set -- $job; p=$1; s=$2
  for set in day night newcam uvh; do
    d="$O/$p.$set.$s.json"; j="$O/scores/$p.$set.$s.json"
    [ -f "$j" ] && continue
    [ -f "$d" ] || python3 dump_detections.py --preset $p --ann "${FR[$set]}" --images "${IM[$set]}" --out "$d" --size $s > /dev/null 2>&1 || { echo "DUMP FAILED $p $set $s"; continue; }
    python3 score_detections.py "${GT[$set]}" "$d" --label "$p@$s $set" --json "$j" | sed -n 2p | sed "s#^#$p@$s $set #"
  done
done
echo "== AUDIT DONE"

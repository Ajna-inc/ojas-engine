#!/bin/bash
# Local Field fine-tune 3 (rebalanced) of the 3.39 M model (deim/field_ft3_local.yml) on the RTX 3060.
# Resumable: after a crash or reboot, running this again continues from last.pth.
set -u
ulimit -n "$(ulimit -Hn)"
T="/data/dev-cache/train"
cd "$T/DEIM"
export PYTHONUNBUFFERED=1
CFG=configs/uvh/field_ft3_local.yml
OUT="$T/out/field_ft3_local"
INIT="$T/out/field_ft2_local/best_stg1.pth"
for attempt in 1 2 3 4 5 6; do
  if [ -f "$OUT/last.pth" ]; then
    echo "== $(date +%H:%M) resume from last.pth (attempt $attempt)"
    python3 train.py -c $CFG -r "$OUT/last.pth" --use-amp --seed 1 && break
  else
    echo "== $(date +%H:%M) start from field_ft2 best_stg1 (attempt $attempt)"
    python3 train.py -c $CFG -t "$INIT" --use-amp --seed 1 && break
  fi
  echo "== train.py exited $?"; sleep 30
done
echo "== TRAINING DONE $(date +%H:%M)"

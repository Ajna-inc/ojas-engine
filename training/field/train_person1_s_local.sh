#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Person fine-tune 1 of the 10 M model (deim/field_person1_s_local.yml) on the local RTX 3060.
# Resumable: after a crash or reboot, running this again continues from last.pth.
# Reads only: the init weights and data prepared for this run; the fine-tune 3 checkpoints are never an output.
set -u
ulimit -n "$(ulimit -Hn)"
T="/data/dev-cache/train"
cd "$T/DEIM"
export PYTHONUNBUFFERED=1
CFG=configs/uvh/field_person1_s_local.yml
OUT="$T/out/field_person1_s"
INIT="$OUT/init/ft3_s_plus_person.pth"
cp $REPO/training/deim/field_person1_s_local.yml "$T/DEIM/$CFG"
for attempt in 1 2 3 4 5 6; do
  if [ -f "$OUT/last.pth" ]; then
    echo "== $(date +%H:%M) resume from last.pth (attempt $attempt)"
    python3 train.py -c $CFG -r "$OUT/last.pth" --use-amp --seed 1 && break
  else
    echo "== $(date +%H:%M) start from ft3_s + person (attempt $attempt)"
    python3 train.py -c $CFG -t "$INIT" --use-amp --seed 1 && break
  fi
  echo "== train.py exited $?"; sleep 30
done
echo "== TRAINING DONE $(date +%H:%M)"

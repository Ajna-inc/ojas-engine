#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Vehicle + person mix 1 of the 3.39 M model (deim/field_mix1_local.yml) on the local RTX 3060.
# Needs the ojas DEIM patch (training/deim/patches/unlabelled_cats.patch). Resumable: after a crash or
# reboot, running this again continues from last.pth. The fine-tune 3 checkpoint is only read.
set -u
ulimit -n "$(ulimit -Hn)"
T="/data/dev-cache/train"
cd "$T/DEIM"
export PYTHONUNBUFFERED=1
grep -q "_labelled_mask" engine/deim/deim_criterion.py || { echo "DEIM patch missing — not starting"; exit 1; }
CFG=configs/uvh/field_mix1_local.yml
OUT="$T/out/field_mix1_local"
INIT="$OUT/init/ft3_plus_person.pth"
cp $REPO/training/deim/field_mix1_local.yml "$T/DEIM/$CFG"
for attempt in 1 2 3 4 5 6; do
  if [ -f "$OUT/last.pth" ]; then
    echo "== $(date +%H:%M) resume from last.pth (attempt $attempt)"
    python3 train.py -c $CFG -r "$OUT/last.pth" --use-amp --seed 1 && break
  else
    echo "== $(date +%H:%M) start from fine-tune 3 + fresh person row (attempt $attempt)"
    python3 train.py -c $CFG -t "$INIT" --use-amp --seed 1 && break
  fi
  echo "== train.py exited $?"; sleep 30
done
echo "== TRAINING DONE $(date +%H:%M)"

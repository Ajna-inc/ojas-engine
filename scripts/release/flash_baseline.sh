#!/bin/sh
# Flash baseline matrix under the corrected harness.
#
# Every run: alternating paired cycles against llama.cpp, token ids and counts
# checked on every repetition, sampled peak RSS from process start.
#
# Axes: expert cache (16/32 GiB) x expert path (scratch / pooled-direct)
#       x decode mode (plain / MTP depth 1), then generation length and context.
set -eu

MODEL=/models/qwen38-flash-next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
OUT=${OUT:-/tmp/flash-baseline}
CYCLES=${CYCLES:-2}
REPS=${REPS:-3}
mkdir -p "$OUT"

# Shared by every ojas configuration. Copy threads 8 was measured at 1.20-1.25x
# over 4 in paired alternation on this 8-performance-core M2 Max.
BASE="OJAS_EXPERT_COPY_THREADS=8,OJAS_FLASH_Q8_COOPERATIVE=1"

run() { # name, ojas-env, ctx, n_predict
  name=$1; env_extra=$2; ctx=$3; npred=$4
  if [ -f "$OUT/$name.json" ]; then
    echo "== $name (already done, skipping)"
    return 0
  fi
  echo "== $name  ctx=$ctx n=$npred cycles=$CYCLES"
  python3 scripts/release/engine_bench.py \
    --model "$MODEL" --label "$name" \
    --ctx "$ctx" --n-predict "$npred" --reps "$REPS" --cycles "$CYCLES" \
    --llama-extra "--load-mode none --lazy-mode on -fa off" \
    --ojas-env "$BASE,$env_extra" \
    --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1 || echo "   FAILED (see $OUT/$name.log)"
  tail -14 "$OUT/$name.log" | sed 's/^/   /'
}

POOLED="OJAS_FLASH_DIRECT_EXPERTS=1,OJAS_FLASH_EXPERT_POOL=1"

# --- core matrix: cache x path x mode, ctx 2048, 64 tokens -------------------
run flash-c16-scratch-plain "OJAS_EXPERT_CACHE_GB=16,OJAS_NO_SPEC=1"            2048 64
run flash-c16-scratch-mtp   "OJAS_EXPERT_CACHE_GB=16,OJAS_MTP_DRAFT=1"          2048 64
run flash-c16-pooled-plain  "OJAS_EXPERT_CACHE_GB=16,$POOLED,OJAS_NO_SPEC=1"    2048 64
run flash-c16-pooled-mtp    "OJAS_EXPERT_CACHE_GB=16,$POOLED,OJAS_MTP_DRAFT=1"  2048 64
run flash-c32-scratch-plain "OJAS_EXPERT_CACHE_GB=32,OJAS_NO_SPEC=1"            2048 64
run flash-c32-scratch-mtp   "OJAS_EXPERT_CACHE_GB=32,OJAS_MTP_DRAFT=1"          2048 64
run flash-c32-pooled-plain  "OJAS_EXPERT_CACHE_GB=32,$POOLED,OJAS_NO_SPEC=1"    2048 64
run flash-c32-pooled-mtp    "OJAS_EXPERT_CACHE_GB=32,$POOLED,OJAS_MTP_DRAFT=1"  2048 64

# --- longer generation ------------------------------------------------------
run flash-c32-pooled-mtp-n192   "OJAS_EXPERT_CACHE_GB=32,$POOLED,OJAS_MTP_DRAFT=1" 2048 192
run flash-c32-pooled-plain-n192 "OJAS_EXPERT_CACHE_GB=32,$POOLED,OJAS_NO_SPEC=1"   2048 192

# --- different context ------------------------------------------------------
run flash-c32-pooled-mtp-ctx512 "OJAS_EXPERT_CACHE_GB=32,$POOLED,OJAS_MTP_DRAFT=1" 512  64

echo "done; results in $OUT"

#!/bin/bash
# Run ojas with a memory guard, so an over-budget configuration cannot take the
# host down with it.
#
# Two things make a run dangerous on a machine smaller than the model:
#   - pinning experts (OJAS_WIRE_MAX_PCT / OJAS_GRAPH_RESERVE_GB) — pinned pages
#     cannot be evicted, and the engine's own budget check is what normally
#     refuses this;
#   - the expert pool (OJAS_FLASH_EXPERT_POOL) — its budget is one dirty Metal
#     allocation, so unlike mmap'd weights the OS cannot reclaim it under
#     pressure. Measured on a 96 GB M2 Max: 73.8 GB wired, 7.9 GB left.
# Both are unset here; pass --allow-unsafe to keep whatever is in the environment.
#
# The guard sends SIGTERM only. SIGKILL bypasses Drop, so endResidency never runs
# and the wired pages leak until reboot (measured: 53 GB).
set -uo pipefail

FLOOR_GB=${FLOOR_GB:-8}
BIN=${OJAS_BIN:-./target/release/ojas}
ALLOW_UNSAFE=0
if [ "${1:-}" = "--allow-unsafe" ]; then ALLOW_UNSAFE=1; shift; fi

if [ $# -eq 0 ]; then
  echo "usage: $0 [--allow-unsafe] <ojas args...>   (e.g. run model.gguf -p 'hi' -n 64)" >&2
  exit 2
fi
[ -x "$BIN" ] || { echo "no ojas binary at $BIN (cargo build --release -p ojas-cli)" >&2; exit 2; }

if [ "$ALLOW_UNSAFE" = 0 ]; then
  unset OJAS_WIRE_MAX_PCT OJAS_GRAPH_RESERVE_GB OJAS_FLASH_EXPERT_POOL \
        OJAS_FLASH_DIRECT_EXPERTS OJAS_MOE_DBUF OJAS_PIN_GB
fi

PG=$(vm_stat | sed -n 's/.*page size of \([0-9]*\).*/\1/p')
avail_gb() {
  vm_stat | awk -v pg="$PG" '
    /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages purgeable/{p=$3}
    END {gsub(/\./,"",f); gsub(/\./,"",i); gsub(/\./,"",s); gsub(/\./,"",p)
         printf "%.2f", (f+i+s+p)*pg/1e9}'
}

"$BIN" "$@" &
PID=$!

(
  while kill -0 "$PID" 2>/dev/null; do
    A=$(avail_gb)
    if awk "BEGIN{exit !($A < $FLOOR_GB)}"; then
      echo "" >&2
      echo "safe-run: only ${A} GB available (floor ${FLOOR_GB} GB) — stopping ojas before the host stalls." >&2
      kill -TERM "$PID" 2>/dev/null
      for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
      exit 0
    fi
    sleep 1
  done
) &
GUARD=$!

wait "$PID"; RC=$?
kill "$GUARD" 2>/dev/null; wait "$GUARD" 2>/dev/null
exit "$RC"

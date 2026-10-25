#!/usr/bin/env bash
# Runs grids of bench.py on one toolchain.
#   ./run_grids.sh [TC=stable] ["N D B"=all main grids] [REPS=5] [TAG]
# Samples go to results/<grid>-<TC>[-TAG].jsonl, logs to logs/. Set BENCH_CPUS (e.g. 0-11) to pin to the P-cores.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TC=${1:-stable}; GRIDS=${2:-"N D B X S U I"}; REPS=${3:-5}; TAG=${4:+-$4}
cd "$R" && mkdir -p results logs
for g in $GRIDS; do
  python3 bench.py --grid "$g" --reps "$REPS" --tc "$TC" --out "results/$g-$TC$TAG.jsonl" > "logs/bench-$g-$TC$TAG.log" 2>&1
  echo "$(date +%T) grid $g ($TC$TAG) done" | tee -a logs/progress.log
done

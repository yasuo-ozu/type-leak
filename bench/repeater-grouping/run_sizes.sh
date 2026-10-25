#!/usr/bin/env bash
# Size-only pass (no timing): the .rlib and .rmeta of both crates, to results/<grid>-<TC>-sizes.jsonl.
#   ./run_sizes.sh [TC=stable] ["N D B X S"=all size grids] [JOBS=4]
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TC=${1:-stable}; GRIDS=${2:-"N D B X S"}; JOBS=${3:-4}
cd "$R" && mkdir -p results logs
for g in $GRIDS; do
  python3 bench.py --grid "$g" --tc "$TC" --sizes --jobs "$JOBS" > "logs/sizes-$g-$TC.log" 2>&1
  echo "$(date +%T) sizes $g ($TC) done" | tee -a logs/progress.log
done

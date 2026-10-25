#!/usr/bin/env bash
# The runs on other rustc versions (rustup toolchains must be installed):
#   grid T on nightly 1.85.0 1.75.0 1.56, grid V (version scan, check only), grids D and K on 1.56 and 1.75.0,
#   grid M (every variant, check only) on 1.56 for the MSRV.
#   ./run_toolchains.sh [REPS=3]
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPS=${1:-3}
cd "$R" && mkdir -p results logs
note() { echo "$(date +%T) $*" | tee -a logs/progress.log; }
for tc in nightly 1.85.0 1.75.0 1.56; do
  python3 bench.py --grid T --reps "$REPS" --tc $tc --out results/T-$tc.jsonl > logs/bench-T-$tc.log 2>&1
  note "grid T ($tc) done"
done
for tc in 1.56.0 1.58.1 1.60.0 1.63.0 1.65.0 1.66.0 1.70.0 1.72.0 1.74.0 1.76.0 1.78.0 1.80.0 1.82.0 1.84.0 1.86.0 \
          1.88.0 1.90.0 1.92.0 1.94.0 1.96.0 1.98.0 1.99.0 nightly; do
  python3 bench.py --grid V --reps 2 --tc $tc --phases check --out results/V-$tc.jsonl > logs/bench-V-$tc.log 2>&1
  note "grid V ($tc) done"
done
for tc in 1.56 1.75.0; do
  for g in D K; do
    python3 bench.py --grid $g --reps "$REPS" --tc $tc --out results/$g-$tc.jsonl > logs/bench-$g-$tc.log 2>&1
    note "grid $g ($tc) done"
  done
done
python3 bench.py --grid M --reps 1 --tc 1.56 --phases check --out results/M-1.56.jsonl > logs/bench-M-1.56.log 2>&1
note "grid M (1.56) done"

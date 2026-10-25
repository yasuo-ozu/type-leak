#!/usr/bin/env bash
# -Ztime-passes of the Leaker's and the Referrer's crate (needs the work/ directories of grid S).
#   ./run_passes.sh
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$R" && mkdir -p results logs
W=$R/work/S
for tc in stable nightly; do
  python3 passes.py --tc $tc --reps 3 --out results/passes-$tc.jsonl \
    $W/S-E5000-n10/{ctl,base,multi4,multi8,multi16} $W/S-E1000-n10/{ctl,base,multi4,multi8,multi16} > logs/passes-$tc.log 2>&1
done
python3 passes.py --tc 1.56 --reps 2 --out results/passes-1.56.jsonl \
  $W/S-E1000-n10/{ctl,base,multi4,multi8,multi16} > logs/passes-1.56.log 2>&1

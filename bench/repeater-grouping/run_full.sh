#!/usr/bin/env bash
# The whole sequence behind the README (about 2 hours on an idle machine), then the tables.
# Set BENCH_CPUS=0-11 (P-core threads of a hybrid CPU, see lscpu -e) for stable CPU times.
set -eu
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$R"
./run_grids.sh stable "N D B X S U I W" 5   # n sweep, corpus-like, complexity, filler cost, scaling, uses, shared ID, stable column of W
./run_toolchains.sh 5                       # other rustc versions, version scan, MSRV, nightly defaults
./run_passes.sh
./run_sizes.sh stable "N D B X S"   # artifact sizes (no timing)
./run_errdemo.sh
python3 make_report.py

#!/usr/bin/env bash
# The whole sequence behind the README (about 1.5 hours on an idle machine), then the tables.
# Set BENCH_CPUS=0-11 (P-core threads of a hybrid CPU, see lscpu -e) for stable CPU times.
set -eu
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$R"
./run_grids.sh stable "A B D" 5 quiet        # the entries-per-context sweep, complexity, corpus-like
./run_grids.sh stable "C K U R I S T" 3      # sizes, control, uses, raw tuples, shared ID, scaling, stable column of T
./run_toolchains.sh 3                        # other rustc versions, version scan, MSRV
./run_passes.sh
./run_errdemo.sh
python3 make_report.py

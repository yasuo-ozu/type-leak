#!/usr/bin/env bash
# What rustc says about wrong Repeater impls and uses, per encoding (scenarios S1-S4 of errdemo.py).
#   ./run_errdemo.sh [LINES=40]      output: logs/errdemo.log
# TASKSET="taskset -c 12-19" runs the compiles on other cores than a benchmark in progress.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$R" && mkdir -p logs && python3 errdemo.py
for s in S1 S2 S3 S4; do for v in base multi4 multi8; do
  echo "=================== $s $v"
  (cd "$R/errdemo/$s-$v" && rm -rf target && ${TASKSET:-} cargo +stable check --offline -q --color never 2>&1 | head -"${1:-40}")
done; done > logs/errdemo.log
echo "wrote logs/errdemo.log"

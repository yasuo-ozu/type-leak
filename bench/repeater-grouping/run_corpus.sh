#!/usr/bin/env bash
# Entries per context of the property-test corpus. Needs a type-leak checkout of the error-recovery branch with
# prop-hist-logging.patch applied (see README.md). Runs the property test for seeds 1-3 (60 cases each) and writes
# corpus/hist-seed<N>.tsv; then run `python3 corpus_hist.py`.
#   ./run_corpus.sh PATH_TO_PATCHED_CHECKOUT      (SEEDS="1 2 3" CASES=60 are the defaults)
set -eu
R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${1:?usage: run_corpus.sh PATH_TO_PATCHED_CHECKOUT}" && pwd)
mkdir -p "$R/corpus" "$R/logs"
export CARGO_TARGET_DIR=$R/work/target-corpus
cd "$REPO"
for seed in ${SEEDS:-1 2 3}; do
  rm -f "$R/corpus/hist-seed$seed.tsv"
  TYPE_LEAK_HIST_FILE=$R/corpus/hist-seed$seed.tsv TYPE_LEAK_PROP_CASES=${CASES:-60} TYPE_LEAK_PROP_SEED=$seed \
    cargo +stable test -q ${CARGO_ARGS:-} --test prop_cross_crate -- --ignored --nocapture > "$R/logs/corpus-seed$seed.log" 2>&1
  echo "seed $seed: $(wc -l < "$R/corpus/hist-seed$seed.tsv") entries"
done

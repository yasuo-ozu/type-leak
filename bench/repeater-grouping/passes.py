#!/usr/bin/env python3
"""passes.py: -Ztime-passes of the Leaker and the Referrer crate of generated workspaces.

  passes.py --tc nightly --out results/passes-nightly.jsonl DIR [DIR..]
For the Referrer the Leaker is built first (not timed). Wall seconds of the passes, pinned with BENCH_CPUS,
the best of --reps runs (min per pass is not meaningful, so the run with the lowest total is kept).
"""
import argparse, json, os, re, shutil, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perfcount import pin_from_env  # noqa: E402

pin_from_env()  # BENCH_CPUS, as in measure.py
LINE = re.compile(r"time:\s+([\d.]+);\s+rss:.*?\t(\S+)\s*$")

def timed(d, crate, tc, build_dep):
    env = dict(os.environ, CARGO_INCREMENTAL="0", RUSTC_BOOTSTRAP="1", CARGO_TERM_COLOR="never")
    shutil.rmtree(os.path.join(d, "target"), ignore_errors=True)
    if build_dep:
        subprocess.run(["cargo", "+" + tc, "build", "--offline", "-q", "-p", "leaker"], cwd=d, env=env, check=True)
    p = subprocess.run(["cargo", "+" + tc, "rustc", "--offline", "-q", "-p", crate, "--lib", "--",
                        "-Ztime-passes"], cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    passes = {}
    for line in p.stdout.splitlines():
        m = LINE.search(line)
        if m:
            passes[m.group(2)] = passes.get(m.group(2), 0.0) + float(m.group(1))
    return passes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tc", default="nightly")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for d in a.dirs:
        for crate in ("leaker", "referrer"):
            runs = [timed(d, crate, a.tc, crate == "referrer") for _ in range(a.reps)]
            best = min(runs, key=lambda r: r.get("total", 1e9))
            rec = dict(dir=d, crate=crate, tc=a.tc, passes=best, totals=[r.get("total") for r in runs],
                       meta=json.load(open(os.path.join(d, "meta.json"))))
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            top = sorted(best.items(), key=lambda kv: -kv[1])[:8]
            print(d.split("work/")[-1], crate, " ".join("%s=%.2f" % kv for kv in top), flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""make_report.py [--out DIR]: (env BENCH_RESULTS=DIR reads the samples from another directory) writes the text tables of every results/*.jsonl into DIR (default: report/).

Per results file: CPU / wall seconds and instruction counts. Then the summaries of summaries.py.
The report/ directory in the repository holds the tables of the run that the README quotes; a new run
overwrites them when written to the default directory.
"""
import glob, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
results = os.environ.get("BENCH_RESULTS", os.path.join(HERE, "results"))  # BENCH_RESULTS: tables of another run
out_dir = os.path.join(HERE, "report")
if "--out" in sys.argv:
    out_dir = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
os.makedirs(out_dir, exist_ok=True)


def run(*args):
    env = dict(os.environ, BENCH_RESULTS=results)
    return subprocess.run([sys.executable] + list(args), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                          cwd=HERE, env=env).stdout


for path in sorted(glob.glob(os.path.join(results, "*.jsonl"))):
    name = os.path.basename(path)[:-6]
    if name.startswith(("M-", "F-", "V-", "passes-")):
        continue
    if name.endswith("-sizes"):  # size-only passes: bench.py --sizes
        text = run("report.py", path, "--sizes")
        with open(os.path.join(out_dir, name + ".txt"), "w") as f:
            f.write("# %s\n%s\n" % (name, text))
        print("wrote", name + ".txt")
        continue
    cpu = run("report.py", path, "--full")
    ins = run("report.py", path, "--metric", "ginstr")
    with open(os.path.join(out_dir, name + ".txt"), "w") as f:
        f.write("# %s\n\n## CPU seconds / wall seconds (user+sys of cargo and children; median over the reps)\n%s\n\n"
                "## User-space instructions, billions (median)\n%s\n" % (name, cpu, ins))
    print("wrote", name + ".txt")
for what in ("nsweep", "corpus", "complexity", "toolchains", "fillers", "scaling", "defaults", "versions", "msrv", "passes", "sizes"):
    text = run("summaries.py", what)
    if text.strip():
        with open(os.path.join(out_dir, what + ".txt"), "w") as f:
            f.write(text)
        print("wrote", what + ".txt")
log = os.path.join(HERE, "logs", "errdemo.log")
if os.path.exists(log):
    shutil.copy(log, os.path.join(out_dir, "errdemo.txt"))
    print("wrote errdemo.txt")

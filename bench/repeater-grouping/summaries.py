#!/usr/bin/env python3
"""summaries.py: the compact tables of the README, from results/*.jsonl.

  summaries.py matrix       variant x (entries per context): % change of `check` CPU vs base  (grid A, quiet)
  summaries.py toolchains   the same configurations on several rustc versions                (grid T)
  summaries.py versions     scan over rustc versions                                         (grid V)
  summaries.py msrv         which variants compile with `cargo +1.56 check`                  (grid M)
  summaries.py passes       -Ztime-passes                                                    (passes.py)
"""
import glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from report import load, stats, med  # noqa: E402

R = os.environ.get("BENCH_RESULTS", os.path.join(HERE, "results"))


def matrix(path="A-stable-quiet.jsonl"):
    cfgs = [("A-E1000-n%d" % n, "n=%d" % n) for n in (1, 2, 4, 10, 50, 1000)]
    variants = ["multi4", "multi8", "multi16", "multi32", "tup4", "tup8", "tup16", "tup32", "nest8", "nest16", "nest32"]
    st = stats(load([os.path.join(R, path)]))
    print("E=1000, three generics with bounds: % change of `cargo check` CPU against the baseline,\n"
          "by entries per context n.\n")
    for what in ("L", "R", "L+R"):
        hdr = ["variant (%s check CPU vs base)" % what] + [c[1] for c in cfgs]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
        for v in variants:
            cells = [v]
            for cfg, _ in cfgs:
                e, b = st[cfg][v], st[cfg]["base"]

                def g(x):
                    l, r = med(x["check_leaker"]["cpu"]), med(x["check_referrer"]["cpu"])
                    return {"L": l, "R": r, "L+R": l + r}[what]
                cells.append("%+.0f%%" % (100 * (g(e) / g(b) - 1)))
            print("| " + " | ".join(cells) + " |")
        print()
    print("base absolute (check CPU s, L / R): " + ", ".join("%s: %.2f / %.2f" % (
        label, med(st[cfg]["base"]["check_leaker"]["cpu"]), med(st[cfg]["base"]["check_referrer"]["cpu"]))
        for cfg, label in cfgs))


def toolchains():
    tcs = ["1.56", "1.75.0", "1.85.0", "stable", "nightly"]
    names = {"stable": "1.99"}
    variants = ["base", "multi4", "multi8", "multi16", "tup8", "tup16", "nest16"]
    if not any(os.path.exists(os.path.join(R, "T-%s.jsonl" % tc)) for tc in tcs):
        return
    for cfg in ("T-E1000-n10", "T-E5000-n10", "T-corpus-E1000"):
        print("\n%s: check CPU seconds, Leaker / Referrer (and %% vs base of L+R)\n" % cfg)
        print("| rustc | " + " | ".join(variants) + " |")
        print("|---|" + "---|" * len(variants))
        for tc in tcs:
            p = os.path.join(R, "T-%s.jsonl" % tc)
            if not os.path.exists(p):
                continue
            st = stats(load([p]))[cfg]
            b = st["base"]
            bt = med(b["check_leaker"]["cpu"]) + med(b["check_referrer"]["cpu"])
            cells = []
            for v in variants:
                e = st[v]
                l, r = med(e["check_leaker"]["cpu"]), med(e["check_referrer"]["cpu"])
                cells.append("%.2f / %.2f (%+.0f%%)" % (l, r, 100 * ((l + r) / bt - 1)))
            print("| %s | %s |" % (names.get(tc, tc), " | ".join(cells)))


def versions():
    rows = []
    for f in glob.glob(os.path.join(R, "V-*.jsonl")):
        tc = os.path.basename(f)[2:-6]
        d = {}
        for r in load([f]):
            for v in r["variants"]:
                d.setdefault(v, []).append((r["check_leaker"]["cpu"], r["check_referrer"]["cpu"]))
        rows.append((tc, d))

    if not rows:
        return

    def key(tc):
        m = re.match(r"(\d+)\.(\d+)", tc)
        return (200, 0) if tc == "nightly" else (int(m.group(1)), int(m.group(2)))
    print("E=2000, 10 entries/context, three generics; `cargo check` CPU seconds (median of the reps), Leaker / Referrer\n")
    print("| rustc | ctl | base | multi8 | tup8 |")
    print("|---|---|---|---|---|")
    for tc, d in sorted(rows, key=lambda x: key(x[0])):
        cells = []
        for v in ("ctl", "base", "multi8", "tup8"):
            xs = d[v]
            cells.append("%.2f / %.2f" % (statistics.median(x[0] for x in xs), statistics.median(x[1] for x in xs)))
        print("| %s | %s |" % (tc, " | ".join(cells)))


def msrv():
    for f in sorted(glob.glob(os.path.join(R, "M-1.56*.jsonl"))):
        print("%s: `cargo +1.56 check`, exit codes of the Leaker and the Referrer crate (0 = builds)\n" % os.path.basename(f))
        res = {}
        for r in load([f]):
            for v in r["variants"]:
                res.setdefault(r["cfg"], {})[v] = (r["check_leaker"]["rc"], r["check_referrer"]["rc"])
        for cfg, d in res.items():
            bad = [v for v, (a, b) in d.items() if a or b]
            print("%s: %d variants, %d fail%s" % (cfg, len(d), len(bad), (": " + ", ".join(bad)) if bad else ""))
            print("  " + " ".join(sorted(d)))


def passes():
    if not glob.glob(os.path.join(R, "passes-*.jsonl")):
        return
    print("-Ztime-passes, wall seconds of the best of N runs. type_check_crate contains coherence_checking and the\n"
          "wf checks; MIR_borrow_checking contains the type check of the fn bodies on stable/nightly.\n")
    for tc in ("stable", "nightly", "1.56"):
        p = os.path.join(R, "passes-%s.jsonl" % tc)
        if not os.path.exists(p):
            continue
        print("== rustc %s\n" % tc)
        print("| config | variant | crate | total | type_check_crate | coherence_checking | wf_checking | MIR_borrow_checking | parse+resolve | metadata+link |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in (json.loads(l) for l in open(p)):
            d = r["dir"].replace("\\", "/").split("/")
            cfg, var = d[-2], d[-1]
            ps = r["passes"]
            g = lambda k: ps.get(k, 0.0)
            print("| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
                cfg, var, r["crate"], g("total"), g("type_check_crate"), g("coherence_checking"), g("wf_checking"),
                g("MIR_borrow_checking"),
                g("parse_crate") + g("resolve_crate") + g("configure_and_expand") + g("expand_crate"),
                g("generate_crate_metadata") + g("link")))
        print()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"matrix": matrix, "toolchains": toolchains, "versions": versions, "msrv": msrv, "passes": passes}.get(what)
    if not fn:
        sys.exit(__doc__)
    fn(*sys.argv[2:])

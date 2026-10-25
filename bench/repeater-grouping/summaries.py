#!/usr/bin/env python3
"""summaries.py: the compact tables of the README, from results/*.jsonl.

  summaries.py nsweep       change against base by entries per context n and K, break-even n     (grid N)
  summaries.py corpus       K against the corpus-like crates, mean and range over 3 draws         (grid D)
  summaries.py complexity   K against the context complexity                                       (grid B)
  summaries.py toolchains   n=4, n=10 and corpus-like on 1.56, 1.60, 1.63 and stable               (grid W)
  summaries.py fillers      cost of one filler `type TypeJ = ();`                                  (grid X)
  summaries.py scaling      K against the number of entries                                        (grid S)
  summaries.py defaults     associated type defaults, nightly only                                 (grid Z)
  summaries.py versions     scan over rustc versions                                               (grid V)
  summaries.py msrv         which variants compile with `cargo +1.56 check`                        (grid M)
  summaries.py passes       -Ztime-passes                                                          (passes.py)

All changes are of the median CPU time of 5 repetitions (or of the instruction counts, where it says so).
"""
import glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from report import load, stats, med  # noqa: E402

R = os.environ.get("BENCH_RESULTS", os.path.join(HERE, "results"))
KS = [2, 3, 4, 5, 6, 8, 12, 16, 32]
NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 50]
WHATS = ("L", "R", "L+R")


def grid(name, tc="stable", tag=""):
    p = os.path.join(R, "%s-%s%s.jsonl" % (name, tc, tag))
    return stats(load([p])) if os.path.exists(p) else None


def val(e, what, kind, metric="cpu"):
    l, r = med(e[kind + "_leaker"][metric]), med(e[kind + "_referrer"][metric])
    return {"L": l, "R": r, "L+R": l + r}[what]


def pct(e, b, what, kind, metric="cpu"):
    return 100.0 * (val(e, what, kind, metric) / val(b, what, kind, metric) - 1)


def s(x):
    return "%+.0f%%" % x if abs(x) >= 0.5 else "0%"


def break_even(ns, ps, tol=1.0):
    """Smallest measured n from which the change stays <= tol percent, and where the change crosses 0
    (linear interpolation between the measured points)."""
    idx = None
    for i in range(len(ns)):
        if max(ps[i:]) <= tol:
            idx = i
            break
    if idx is None:
        return None, None
    if idx == 0:
        return ns[0], float(ns[0])
    n0, n1, p0, p1 = ns[idx - 1], ns[idx], ps[idx - 1], ps[idx]
    cross = n0 + (0 - p0) * (n1 - n0) / (p1 - p0) if p1 != p0 else float(n1)
    return ns[idx], max(float(n0), min(cross, float(n1)))


def nsweep():
    st = grid("N")
    if not st:
        return
    cfgs = ["N-n%02d" % n for n in NS]
    ks = [k for k in KS if "multi%d" % k in st[cfgs[0]]]
    print("E=1000 (within 1%), three generics with bounds, rustc 1.99 stable. Change of the median CPU time of\n"
          "multiK against base, by entries per context n. Base: Leaker / Referrer `check` seconds:\n")
    print("  " + ", ".join("n=%d: %.2f / %.2f" % (n, val(st[c]["base"], "L", "check"), val(st[c]["base"], "R", "check"))
                           for n, c in zip(NS, cfgs)) + "\n")
    for kind in ("check", "build"):
        for what in WHATS:
            hdr = ["K (%s, %s)" % (what, kind)] + ["n=%d" % n for n in NS]
            print("| " + " | ".join(hdr) + " |")
            print("|" + "|".join(["---"] * len(hdr)) + "|")
            for k in ks:
                cells = ["multi%d" % k] + [s(pct(st[c]["multi%d" % k], st[c]["base"], what, kind)) for c in cfgs]
                print("| " + " | ".join(cells) + " |")
            print()
    for metric, label in (("cpu", "CPU time"), ("ginstr", "instruction counts")):
        print("Break-even n by %s: the smallest measured n from which multiK is not slower than base (within 1%%\n"
              "for every larger measured n), and the n where the interpolated curve crosses 0.\n" % label)
        cols = [("L", "check"), ("L+R", "check"), ("L", "build"), ("L+R", "build"), ("R", "check")]
        hdr = ["K"] + ["%s %s" % c for c in cols]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
        for k in ks:
            cells = ["multi%d" % k]
            for what, kind in cols:
                ps = [pct(st[c]["multi%d" % k], st[c]["base"], what, kind, metric) for c in cfgs]
                n, cross = break_even(NS, ps)
                cells.append("never (>50)" if n is None else ("n>=%d (~%.1f)" % (n, cross) if n > 1 else "all n"))
            print("| " + " | ".join(cells) + " |")
        print()
    # the best K for each n (smallest L+R check)
    print("Best K for each n (smallest L+R `check` CPU; base counts as K=1):\n")
    print("| n | best K | L+R check vs base | best K for L only | L vs base |")
    print("|---|---|---|---|---|")
    for n, c in zip(NS, cfgs):
        cand = {k: val(st[c]["multi%d" % k], "L+R", "check") for k in ks}
        cand[1] = val(st[c]["base"], "L+R", "check")
        kb = min(cand, key=cand.get)
        candl = {k: val(st[c]["multi%d" % k], "L", "check") for k in ks}
        candl[1] = val(st[c]["base"], "L", "check")
        kl = min(candl, key=candl.get)
        print("| %d | %s | %s | %s | %s |" % (n, kb, s(100 * (cand[kb] / cand[1] - 1)), kl,
                                              s(100 * (candl[kl] / candl[1] - 1))))


def corpus():
    g = grid("D")
    if not g:
        return
    st = {(e, seed): g["D-E%d-s%d" % (e, seed)] for e in (50, 200, 1000) for seed in (1, 2, 3)}
    print("Corpus-like crates (context sizes drawn from the corpus histogram, mixed contexts, 6.5% unsized), rustc 1.99.\n"
          "Change against base; mean over 3 draws (seeds 1-3) and [min, max]. Base: Leaker / Referrer seconds.\n")
    for kind in ("check", "build"):
        for e in (50, 200, 1000):
            runs = [st[(e, sd)] for sd in (1, 2, 3)]
            bl = statistics.mean(val(r["base"], "L", kind) for r in runs)
            br = statistics.mean(val(r["base"], "R", kind) for r in runs)
            print("E=%d, `%s` (base %.3f / %.3f; entries %s)\n" % (
                e, kind, bl, br, "/".join(str(r["base"]["meta"]["E"]) for r in runs)))
            hdr = ["change"] + ["multi%d" % k for k in KS]
            print("| " + " | ".join(hdr) + " |")
            print("|" + "|".join(["---"] * len(hdr)) + "|")
            for what in WHATS:
                cells = [what]
                for k in KS:
                    ps = [pct(r["multi%d" % k], r["base"], what, kind) for r in runs]
                    cells.append("%s [%s, %s]" % (s(statistics.mean(ps)), s(min(ps)), s(max(ps))))
                print("| " + " | ".join(cells) + " |")
            print()
    print("Best K by the mean L+R `check` change: " + "; ".join("E=%d: %s" % (e, _best(st, e)) for e in (50, 200, 1000)))


def _best(st, e):
    runs = [st[(e, sd)] for sd in (1, 2, 3)]
    res = {k: statistics.mean(pct(r["multi%d" % k], r["base"], "L+R", "check") for r in runs) for k in KS}
    kb = min(res, key=res.get)
    return "K=%d (%s)" % (kb, s(res[kb]))


def complexity():
    st = grid("B")
    if not st:
        return
    print("Context complexity, E=1000, 10 entries per context, rustc 1.99. Change of `check` CPU against base,\n"
          "L / R / L+R. Base: Leaker / Referrer `check` seconds.\n")
    hdr = ["context (base L / R)"] + ["multi%d" % k for k in KS]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for cx in ("none", "one", "three", "lt"):
        c = st["B-%s-E1000-n10" % cx]
        cells = ["%s (%.2f / %.2f)" % (cx, val(c["base"], "L", "check"), val(c["base"], "R", "check"))]
        for k in KS:
            e = c["multi%d" % k]
            cells.append("%s / %s / %s" % tuple(s(pct(e, c["base"], w, "check")) for w in WHATS))
        print("| " + " | ".join(cells) + " |")


def toolchains():
    tcs = [("1.56", "1.56"), ("1.60.0", "1.60"), ("1.63.0", "1.63"), ("stable", "1.99")]
    data = {}
    for tc, label in tcs:
        g = grid("W", tc)
        if g:
            data[label] = g
    if not data:
        return
    labels = [l for _, l in tcs if l in data]
    print("Other compilers. Change of `check` CPU against base, L / R / L+R. Base: Leaker / Referrer seconds.\n")
    for cfg, title in (("W-E1000-n4", "E=1000, 4 entries per context"), ("W-E1000-n10", "E=1000, 10 entries per context"),
                       ("W-corpus-E1000", "corpus-like E=1000 (one draw)")):
        print("%s\n" % title)
        hdr = ["K"] + ["rustc %s (base %.2f / %.2f)" % (
            l, val(data[l][cfg]["base"], "L", "check"), val(data[l][cfg]["base"], "R", "check")) for l in labels]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
        for k in KS:
            cells = ["multi%d" % k]
            for l in labels:
                c = data[l][cfg]
                cells.append("%s / %s / %s" % tuple(s(pct(c["multi%d" % k], c["base"], w, "check")) for w in WHATS))
            print("| " + " | ".join(cells) + " |")
        print()
        best = {l: min(KS, key=lambda k: val(data[l][cfg]["multi%d" % k], "L+R", "check")) for l in labels}
        print("best K by L+R: " + ", ".join("%s: K=%d" % (l, best[l]) for l in labels) + "\n")
    print("The same with `build` (L+R change):\n")
    for cfg in ("W-E1000-n4", "W-E1000-n10", "W-corpus-E1000"):
        print("%s:" % cfg)
        for l in labels:
            print("  rustc %s: " % l + " ".join(
                "K%d %s" % (k, s(pct(data[l][cfg]["multi%d" % k], data[l][cfg]["base"], "L+R", "build"))) for k in KS))


def fillers():
    st = grid("X")
    if not st:
        return
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32]
    print("Cost of a filler `type TypeJ = ();`: 1000 singleton contexts (one entry each), K = 1..8, 12, 16, 32.\n"
          "Leaker `check` CPU (the Referrer does not change). Slope: least squares over K = 1..8; per filler =\n"
          "slope / 1000 impls; per impl = (base - ctl) / 1000.\n")
    for metric, unit in (("cpu", "ms"), ("ginstr", "M instructions")):
        hdr = ["context (%s)" % unit, "ctl", "base"] + ["K=%d" % k for k in ks] + ["per filler", "per impl", "filler / impl"]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
        for cx in ("none", "three", "lt"):
            c = st["X-%s-n1" % cx]
            ys = {k: val(c["multi%d" % k], "L", "check", metric) * 1000.0 for k in ks}
            ctl, base = (val(c[v], "L", "check", metric) * 1000.0 for v in ("ctl", "base"))
            xs = [k for k in ks if k <= 8]
            mx, my = statistics.mean(xs), statistics.mean(ys[k] for k in xs)
            slope = sum((x - mx) * (ys[x] - my) for x in xs) / sum((x - mx) ** 2 for x in xs)
            per_filler, per_impl = slope / 1000.0, (base - ctl) / 1000.0
            u = "us" if metric == "cpu" else "k instr"
            cells = [cx, "%.0f" % ctl, "%.0f" % base] + ["%.0f" % ys[k] for k in ks]
            cells += ["%.2f %s" % (per_filler * 1000.0, u), "%.2f %s" % (per_impl * 1000.0, u), "%.2f" % (per_filler / per_impl)]
            print("| " + " | ".join(cells) + " |")
        print()


def scaling():
    st = grid("S")
    if not st:
        return
    ks = [2, 4, 8, 16, 32]
    print("Number of entries, 10 per context, three generics, rustc 1.99. `check` CPU seconds (L / R) and the change of\n"
          "L+R against base.\n")
    hdr = ["E", "ctl", "base"] + ["multi%d" % k for k in ks]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for e in (100, 200, 500, 1000, 2000, 5000, 10000):
        c = st["S-E%d-n10" % e]
        cells = [str(e)]
        for v in ["ctl", "base"] + ["multi%d" % k for k in ks]:
            x = c[v]
            cell = "%.2f / %.2f" % (val(x, "L", "check"), val(x, "R", "check"))
            if v.startswith("multi"):
                cell += " (%s)" % s(pct(x, c["base"], "L+R", "check"))
            cells.append(cell)
        print("| " + " | ".join(cells) + " |")


def defaults():
    st = grid("Z", "nightly")
    if not st:
        return
    print("Associated type defaults (`multiKd`, #![feature(associated_type_defaults)], nightly only; an impl defines only\n"
          "the types it needs), rustc nightly. `check` CPU, change against base, Leaker / Referrer.\n")
    ks = [4, 8, 16, 32]
    hdr = ["config (base L / R s)"] + ["multi%d" % k for k in ks] + ["multi%dd" % k for k in ks]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for cfg in st:
        c = st[cfg]
        cells = ["%s (%.2f / %.2f)" % (cfg, val(c["base"], "L", "check"), val(c["base"], "R", "check"))]
        for v in ["multi%d" % k for k in ks] + ["multi%dd" % k for k in ks]:
            cells.append("%s / %s" % (s(pct(c[v], c["base"], "L", "check")), s(pct(c[v], c["base"], "R", "check"))))
        print("| " + " | ".join(cells) + " |")


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
    vs = ["ctl", "base", "multi4", "multi8", "multi16"]
    print("E=2000, 10 entries/context, three generics; `cargo check` CPU seconds (median of the reps), Leaker / Referrer\n")
    print("| rustc | " + " | ".join(vs) + " |")
    print("|---|" + "---|" * len(vs))
    for tc, d in sorted(rows, key=lambda x: key(x[0])):
        cells = []
        for v in vs:
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
            print("  " + " ".join(sorted(d, key=lambda v: (len(v), v))))


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
    fn = {"nsweep": nsweep, "corpus": corpus, "complexity": complexity, "toolchains": toolchains,
          "fillers": fillers, "scaling": scaling, "defaults": defaults, "versions": versions, "msrv": msrv,
          "passes": passes}.get(what)
    if not fn:
        sys.exit(__doc__)
    fn(*sys.argv[2:])

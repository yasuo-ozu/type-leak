#!/usr/bin/env python3
"""report.py: markdown tables from results/*.jsonl (the samples of bench.py).

  report.py results/A-stable.jsonl [--full] [--metric cpu|ginstr]
Per configuration: one row per variant with the median over the repetitions.
"""
import argparse, collections, json, statistics, sys

PH = ["build_leaker", "build_referrer", "check_leaker", "check_referrer"]


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            rows.append(json.loads(line))
    return rows


def med(xs):
    return statistics.median(xs)


def stats(rows):
    """{cfg: {variant: {phase: {metric: [values]}}, 'meta': ...}}"""
    out = collections.OrderedDict()
    for r in rows:
        c = out.setdefault(r["cfg"], collections.OrderedDict())
        for v in r["variants"]:
            e = c.setdefault(v, dict(meta=None, key=r["key"], n=0, load=[]))
            e["n"] += 1
            e["load"].append(float(r["load"][0]))
            e["key"] = r["key"]
            for ph in PH:
                if ph not in r:
                    continue
                x = r[ph]
                if x["rc"]:
                    e.setdefault("fail", set()).add(ph)
                d = e.setdefault(ph, collections.defaultdict(list))
                for m in ("cpu", "wall", "ginstr"):
                    if x.get(m) is not None:
                        d[m].append(x[m])
            if v == r["variants"][0] or e["meta"] is None:
                e["meta"] = r["meta"]
                e["meta"] = dict(r["meta"], variant=v)
    return out


def vorder(v):
    import re
    m = re.match(r"^([a-z]+)(\d*)(d?)$", v)
    kind = ["ctl", "base", "multi"].index(m.group(1))
    return (kind, m.group(3), int(m.group(2) or 0))


def fmt(x, nd=2):
    return ("%." + str(nd) + "f") % x


def table(cfg, vs, metric, full):
    base = vs.get("base")
    names = sorted(vs, key=vorder)
    lines = []
    hdr = ["variant", "impls", "assoc/impl", "L build", "R build", "L check", "R check",
           "build L+R", "check L+R"]
    if full:
        hdr += ["spread (max/min) of build L+R"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")

    def tot(e, kind, m):
        return med([a + b for a, b in zip(e[kind + "_leaker"][m], e[kind + "_referrer"][m])])

    seen = {}
    for v in names:
        e = vs[v]
        meta = e["meta"]
        cells = [v]
        note = ""
        if e["key"] in seen:
            note = " (= %s)" % seen[e["key"]]
        else:
            seen[e["key"]] = v
        cells[0] = v + note
        cells += [str(meta["impls"]), str(meta["assoc_per_impl"])]
        for ph in PH:
            cpu = med(e[ph]["cpu"])
            wall = med(e[ph]["wall"])
            cells.append("%s / %s" % (fmt(cpu), fmt(wall)))
        for kind in ("build", "check"):
            t = tot(e, kind, "cpu")
            if base:
                bt = tot(base, kind, "cpu")
                cells.append("%s (%+.0f%%)" % (fmt(t), 100.0 * (t / bt - 1)))
            else:
                cells.append(fmt(t))
        if full:
            xs = [a + b for a, b in zip(e["build_leaker"]["cpu"], e["build_referrer"]["cpu"])]
            cells.append("%.2f (n=%d)" % (max(xs) / min(xs), len(xs)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def has_instr(vs):
    return all(e[ph]["ginstr"] for e in vs.values() for ph in PH if ph in e)


def table_instr(cfg, vs):
    """Instruction counts (billions, user space, P-cores): deterministic to ~0.1%."""
    base = vs.get("base")
    names = sorted(vs, key=vorder)
    hdr = ["variant", "L build", "R build", "L check", "R check", "L+R build", "L+R check", "L %", "R %", "L+R %"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    seen = {}
    g = lambda e, ph: med(e[ph]["ginstr"])
    for v in names:
        e = vs[v]
        note = ""
        if e["key"] in seen:
            note = " (= %s)" % seen[e["key"]]
        else:
            seen[e["key"]] = v
        lb, rb, lc, rc = (g(e, ph) for ph in PH)
        cells = [v + note, fmt(lb), fmt(rb), fmt(lc), fmt(rc), fmt(lb + rb), fmt(lc + rc)]
        if base:
            bl = (g(base, "check_leaker"), g(base, "check_referrer"))
            cells += ["%+.0f%%" % (100 * (lc / bl[0] - 1)), "%+.0f%%" % (100 * (rc / bl[1] - 1)),
                      "%+.0f%%" % (100 * ((lc + rc) / (bl[0] + bl[1]) - 1))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--metric", default="cpu")
    a = ap.parse_args()
    st = stats(load(a.files))
    for cfg, vs in st.items():
        some = next(iter(vs.values()))
        m = some["meta"]
        print("\n### %s  (E=%d, C=%d, cx=%s, unsized=%d, n reps=%d, load %.0f..%.0f)\n" % (
            cfg, m["E"], m["C"], m["cx"], m["n_unsized"], some["n"], min(some["load"]), max(some["load"])))
        if a.metric == "ginstr":
            if not has_instr(vs):
                print("(no instruction counts: perf_event_open was not available)")
                continue
            print("cells: billions of user-space instructions (median); % vs base for check builds\n")
            print(table_instr(cfg, vs))
        else:
            print("cells: CPU s / wall s, median over reps; totals: CPU s (vs base)\n")
            print(table(cfg, vs, a.metric, a.full))


if __name__ == "__main__":
    main()

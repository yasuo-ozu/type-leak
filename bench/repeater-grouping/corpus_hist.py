#!/usr/bin/env python3
"""Histogram of entries per context in the property-test corpus (corpus/hist-seed*.tsv).

  corpus_hist.py [GLOB]      prints the histogram, writes corpus/sizes.json (entries per impl header and item,
                             the sizes that gen.py --dist corpus draws from)

The TSV files come from run_corpus.sh (the property test with prop-hist-logging.patch applied): one line per
entry: crate, item id, entry index, ContextId, impl header | args | self | where, entry type.
"""
import collections, glob, os, re, statistics, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))

def load(path):
    rows = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 6:
            continue
        pkg, id_, ix, ctx, key, ty = f[:6]
        rows[(pkg, int(id_), int(ix))] = (ctx, key, ty)
    return rows

def bucket(n):
    for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 10**9)]:
        if lo <= n <= hi:
            return "%d" % lo if lo == hi else ("%d-%d" % (lo, hi) if hi < 10**9 else "%d+" % lo)

ORDER = ["1", "2", "3", "4", "5-8", "9-16", "17-32", "33-64", "65+"]

def header(key):
    # key = impl_generics | args | self_ty | where
    parts = key.split(" | ")
    return parts

def classify(key):
    ig, args, selfty, wh = (key.split(" | ") + ["", "", "", ""])[:4]
    ig = ig.replace("__S : ? :: core :: marker :: Sized", "").strip()
    has_lt = "'" in args.split(",")[0] if args.startswith("(&") or "(& '" in args else ("& '" in args)
    n_args = 0 if args.strip() == "()" else args.count(",") + (0 if args.strip().endswith(",)") else 1)
    n_args = max(0, n_args)
    has_bound = ":" in ig
    return args.strip() == "()", has_lt, has_bound

def main(files):
    ctx_sizes = []          # entries per (pkg, id, ContextId)
    hdr_sizes = []          # entries per (pkg, id, impl header + args)
    hdr_sizes_pkg = []      # per (pkg, header incl. ID dropped): all items of a crate together
    per_pkg = collections.defaultdict(lambda: dict(E=0, items=set(), ctxs=set(), hdrs=set()))
    cx_kind = collections.Counter()
    entry_kind = collections.Counter()
    total = 0
    for fpath in files:
        rows = load(fpath)
        by_ctx = collections.defaultdict(list)
        by_hdr = collections.defaultdict(list)
        by_hdr_pkg = collections.defaultdict(list)
        for (pkg, id_, ix), (ctx, key, ty) in rows.items():
            tag = fpath.split("-")[-1].split(".")[0] + ":" + pkg
            by_ctx[(tag, id_, ctx)].append(ix)
            by_hdr[(tag, id_, key)].append(ix)
            by_hdr_pkg[(tag, key)].append(ix)
            p = per_pkg[tag]
            p["E"] += 1; p["items"].add(id_); p["ctxs"].add((id_, ctx)); p["hdrs"].add((id_, key))
            total += 1
            t = ty.strip()
            k = "dyn" if t.startswith("dyn ") or "dyn " in t.split("<")[0] else ("str/slice" if t in ("str", "[ u8 ]") or t.startswith("[") else "other")
            entry_kind[k] += 1
        for k, v in by_ctx.items():
            ctx_sizes.append(len(v))
        for k, v in by_hdr.items():
            hdr_sizes.append(len(v))
            cx_kind[classify(k[2])] += 0
        for k, v in by_hdr_pkg.items():
            hdr_sizes_pkg.append(len(v))
        seen = set()
        for (tag, id_, key), v in by_hdr.items():
            c = classify(key)
            cx_kind[c] += 1
    def hist(sizes):
        c = collections.Counter(bucket(n) for n in sizes)
        tot_e = sum(sizes)
        ce = collections.Counter()
        for n in sizes:
            ce[bucket(n)] += n
        return c, ce, tot_e
    out = []
    for name, sizes in [("entries per ContextId (per item)", ctx_sizes),
                        ("entries per impl header+args (per item; same ID)", hdr_sizes),
                        ("entries per impl header+args (per crate; ID ignored)", hdr_sizes_pkg)]:
        c, ce, tot_e = hist(sizes)
        out.append("\n%s: %d groups, %d entries, mean %.2f, median %s, max %d" % (
            name, len(sizes), tot_e, statistics.mean(sizes), statistics.median(sizes), max(sizes)))
        out.append("  %-7s %8s %8s %10s %10s" % ("size", "groups", "%groups", "entries", "%entries"))
        for b in ORDER:
            out.append("  %-7s %8d %7.1f%% %10d %9.1f%%" % (b, c[b], 100.0 * c[b] / len(sizes), ce[b], 100.0 * ce[b] / tot_e))
        # what share of impls would a K-grouping save: sum(ceil(n/K)) / sum(n)
        for K in (2, 4, 8, 16, 32):
            impls = sum(-(-n // K) for n in sizes)
            out.append("  K=%-2d: %d impls instead of %d (%.1f%%)" % (K, impls, tot_e, 100.0 * impls / tot_e))
    es = sorted(p["E"] for p in per_pkg.values())
    out.append("\nper Leaker crate (one case, %d crates): E median %s, mean %.0f, max %d; items(ids) median %s; contexts(ids x ContextId) median %s; distinct headers median %s" % (
        len(es), statistics.median(es), statistics.mean(es), max(es),
        statistics.median(len(p["items"]) for p in per_pkg.values()),
        statistics.median(len(p["ctxs"]) for p in per_pkg.values()),
        statistics.median(len(p["hdrs"]) for p in per_pkg.values())))
    out.append("entry kinds (rough): %s" % dict(entry_kind))
    n = sum(cx_kind.values())
    out.append("contexts by shape (impl header + args groups, per item): total %d" % n)
    for (nogen, lt, bound), v in sorted(cx_kind.items(), key=lambda kv: -kv[1]):
        out.append("   no-generics=%s lifetime=%s bounds=%s: %d (%.1f%%)" % (nogen, lt, bound, v, 100.0 * v / n))
    print("\n".join(out))
    json.dump(hdr_sizes, open(os.path.join(HERE, "corpus", "sizes.json"), "w"))


def per_item(files):
    """Entries per item (one macro invocation) and contexts per item."""
    items = collections.defaultdict(lambda: collections.defaultdict(list))
    for fpath in files:
        for (pkg, id_, ix), (ctx, key, ty) in load(fpath).items():
            items[(fpath, pkg, id_)][ctx].append(ix)
    sizes = sorted(sum(len(v) for v in d.values()) for d in items.values())
    nctx = sorted(len(d) for d in items.values())
    big = [d for d in items.values() if sum(len(v) for v in d.values()) >= 9]
    cs = [len(v) for d in big for v in d.values()]
    print("\nper item: %d items, entries median %s, mean %.1f, p90 %d, p99 %d, max %d; contexts median %s, mean %.2f, max %d" % (
        len(sizes), statistics.median(sizes), statistics.mean(sizes), sizes[int(.9 * len(sizes))],
        sizes[int(.99 * len(sizes))], sizes[-1], statistics.median(nctx), statistics.mean(nctx), nctx[-1]))
    print("items with >= 9 entries: %d, their contexts: %d, %.0f%% hold one entry" % (
        len(big), len(cs), 100.0 * sum(1 for n in cs if n == 1) / len(cs)))


files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "corpus", "hist-seed*.tsv")))
if not files:
    sys.exit("no TSV files: run run_corpus.sh first")
main(files)
per_item(files)

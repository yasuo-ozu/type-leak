#!/usr/bin/env python3
"""bench.py: interleaved, repeated clean-build timing of generated Repeater variants.

  bench.py --grid A --reps 3 --tc stable [--out results/A-stable.jsonl]

A grid is a list of configurations; a configuration is (name, generator arguments, variants).
Variants whose generated crates are byte-identical are measured once (their `key`).
For each rep, for each configuration, the variants are shuffled and measured one after the other,
so that a change of the machine's load hits all the variants of a configuration alike.
One line of JSON per sample is appended to --out as soon as it is measured; a rerun skips the
samples already there.

Grids (see grids()):
  A  entries per context (E=1000, 1..1000 per context)      B  context complexity (none, one, three, lt)
  C  number of entries (E=200, 5000)                       D  corpus-like (sizes drawn from corpus/sizes.json)
  I  one ID for all contexts                               K  control: ctl against base
  M  every variant (MSRV: --tc 1.56 --phases check)        Q  quick smoke test
  R  raw tuples against *const-wrapped ones                S  scaling with E (100 .. 10000)
  T  other rustc versions                                  U  which uses in the Referrer matter
  V  rustc version scan (--phases check)                   F  empty crates (the floor)
"""
import argparse, hashlib, json, os, random, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure import measure  # noqa: E402

ALL = ["base",
       "multi4", "multi8", "multi16", "multi32",
       "tup4", "tup8", "tup16", "tup32",
       "tup8p", "tup16p", "tup32p",
       "nest4", "nest8", "nest16", "nest32"]
MAIN = ["base", "multi4", "multi8", "multi16", "multi32", "tup4", "tup8", "tup16", "tup32",
        "nest4", "nest8", "nest16", "nest32"]
SMALL = ["base", "multi8", "multi16", "tup8", "tup16", "tup16p", "nest8", "nest16"]


def cfg(name, E, C, cx="three", dist="uniform", unsized=0.12, seed=1, variants=None, extra=None):
    return dict(name=name, E=E, C=C, cx=cx, dist=dist, unsized=unsized, seed=seed,
                variants=variants or MAIN, extra=extra or [])


def grids():
    g = {}
    # A: entries per context sweep at E=1000, three generics with bounds
    g["A"] = [cfg("A-E1000-n%d" % n, 1000, 1000 // n) for n in (1, 2, 4, 10, 50, 1000)]
    # B: context complexity at E=1000, 10 entries per context
    g["B"] = [cfg("B-%s-E1000-n10" % cx, 1000, 100, cx=cx) for cx in ("none", "one", "three", "lt")]
    # C: number of entries, 10 per context
    g["C"] = ([cfg("C-E200-n10", 200, 20, variants=ALL)] +
              [cfg("C-E5000-n10", 5000, 500, variants=SMALL)] +
              [cfg("C-E5000-n1", 5000, 5000, variants=["base"]),
               cfg("C-E5000-n4", 5000, 1250, variants=SMALL)])
    # D: corpus-like (sizes drawn from the corpus, mixed complexity, 6.5% unsized)
    g["D"] = [cfg("D-E%d" % e, e, 1, cx="mixed", dist="corpus", unsized=0.065, variants=ALL) for e in (50, 200, 1000)]
    # K: the control (no Repeater at all: the Referrer writes the original types; the Leaker has no impls)
    g["K"] = [cfg("K-E50-corpus", 50, 1, cx="mixed", dist="corpus", unsized=0.065, variants=["ctl", "base"]),
              cfg("K-E200-corpus", 200, 1, cx="mixed", dist="corpus", unsized=0.065, variants=["ctl", "base"]),
              cfg("K-E1000-corpus", 1000, 1, cx="mixed", dist="corpus", unsized=0.065, variants=["ctl", "base"]),
              cfg("K-E1000-n10", 1000, 100, variants=["ctl", "base"]),
              cfg("K-E5000-n10", 5000, 500, variants=["ctl", "base"])]
    # U: which uses of the entries in the Referrer's crate matter (E=1000, 10 per context)
    U = ["ctl", "base", "multi8", "tup8", "nest8"]
    g["U"] = [cfg("U-sig", 1000, 100, variants=U, extra=["--uses", "sig"]),
              cfg("U-impl", 1000, 100, variants=U, extra=["--uses", "impl"]),
              cfg("U-wit", 1000, 100, variants=U, extra=["--uses", "wit"])]
    # R: raw tuples (no `*const`, unsized entries ungrouped) against the wrapped ones; with and without unsized entries
    Rv = ["base", "tup8", "tup8r", "tup16", "tup16r", "nest8", "nest8r", "nest16", "nest16r", "multi8"]
    g["R"] = [cfg("R-uns12-E1000-n10", 1000, 100, variants=Rv),
              cfg("R-uns0-E1000-n10", 1000, 100, unsized=0.0, variants=Rv)]
    # T: other rustc versions (the interesting variants only)
    Tv = ["base", "multi4", "multi8", "multi16", "tup4", "tup8", "tup16", "nest8", "nest16"]
    g["T"] = [cfg("T-E1000-n10", 1000, 100, variants=Tv),
              cfg("T-E1000-n2", 1000, 500, variants=Tv),
              cfg("T-E5000-n10", 5000, 500, variants=Tv),
              cfg("T-corpus-E1000", 1000, 1, cx="mixed", dist="corpus", unsized=0.065, variants=Tv)]
    # I: one ID for all the contexts (a big item: NTH counts over all its contexts)
    Iv = ["base", "multi4", "multi8", "multi16", "tup8", "tup16", "nest8"]
    g["I"] = [cfg("I-E1000-n10", 1000, 100, variants=Iv, extra=["--ids", "shared"]),
              cfg("I-E5000-n10", 5000, 500, variants=Iv, extra=["--ids", "shared"])]
    # S: scaling with the number of entries (10 per context)
    Sv = ["ctl", "base", "multi8", "multi16", "tup8", "tup16", "nest16"]
    g["S"] = [cfg("S-E%d-n10" % e, e, e // 10, variants=Sv) for e in (100, 200, 500, 1000, 2000, 5000, 10000)]
    # M: every variant on one configuration (used with --tc 1.56 --phases check for the MSRV)
    g["M"] = [cfg("M-E1000-n10", 1000, 100, variants=ALL + ["tup8r", "tup16r", "nest8r", "nest16r", "ctl"]),
              cfg("M-corpus-E200", 200, 1, cx="mixed", dist="corpus", unsized=0.065,
                  variants=ALL + ["tup8r", "tup16r", "nest8r", "nest16r", "ctl"])]
    # V: rustc version scan (use with --phases check): where the quadratic part comes from and goes away
    g["V"] = [cfg("V-E2000-n10", 2000, 200, variants=["ctl", "base", "multi8", "tup8"])]
    # Q: the quick smoke test (about 10 s): E=200, 10 per context
    g["Q"] = [cfg("Q-E200-n10", 200, 20, variants=["ctl", "base", "multi8", "tup8"])]
    # F: the floor (empty crates)
    g["F"] = [cfg("F-floor", 0, 0, variants=["base", "multi8", "multi32", "tup8", "nest8"])]
    return g


def gen(c, variant, root):
    d = os.path.join(root, c["name"], variant)
    cmd = [sys.executable, os.path.join(HERE, "gen.py"), "--out", d, "--variant", variant,
           "--E", str(c["E"]), "--C", str(c["C"]), "--cx", c["cx"], "--dist", c["dist"],
           "--unsized", str(c["unsized"]), "--seed", str(c["seed"])] + c["extra"]
    meta = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout
    h = hashlib.sha1()
    for p in ("leaker/src/lib.rs", "referrer/src/lib.rs"):
        h.update(open(os.path.join(d, p), "rb").read())
    meta = json.load(open(os.path.join(d, "meta.json")))
    return d, h.hexdigest()[:10], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tc", default="stable")
    ap.add_argument("--out", default="", help="JSONL file (default: results/<grid>-<tc>.jsonl)")
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--only", default="", help="substring filter on configuration names")
    ap.add_argument("--variants", default="", help="comma list overriding the variants of each configuration")
    ap.add_argument("--phases", default="build,check")
    ap.add_argument("--env", default="", help="KEY=VAL,.. extra environment for cargo")
    ap.add_argument("--tag", default="", help="stored in each sample")
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(HERE, "results", "%s-%s.jsonl" % (a.grid, a.tc))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cfgs = [c for c in grids()[a.grid] if a.only in c["name"]]
    if a.variants:
        for c in cfgs:
            c["variants"] = a.variants.split(",")
    env_extra = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
    phases = tuple(a.phases.split(","))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["cfg"], r["key"], r["rep"]))
    # generate
    plan = {}
    for c in cfgs:
        keys = {}
        for v in c["variants"]:
            d, key, meta = gen(c, v, os.path.join(a.work, a.grid))
            keys.setdefault(key, []).append((v, d, meta))
        plan[c["name"]] = keys
        print("%s: %d variants, %d distinct" % (c["name"], len(c["variants"]), len(keys)), flush=True)
    for rep in range(a.reps):
        for c in cfgs:
            keys = list(plan[c["name"]].items())
            random.Random("%s-%d" % (c["name"], rep)).shuffle(keys)
            for key, vs in keys:
                if (c["name"], key, rep) in done:
                    continue
                v, d, meta = vs[0]
                t0 = time.time()
                res = measure(d, a.tc, phases, env_extra)
                rec = dict(grid=a.grid, cfg=c["name"], key=key, variants=[x[0] for x in vs], rep=rep, tc=a.tc,
                           tag=a.tag, meta=meta, ts=time.strftime("%H:%M:%S"), **res)
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                bad = [k for k, x in res.items() if isinstance(x, dict) and x.get("rc")]
                print("%s rep%d %-24s %s load=%s%s" % (
                    c["name"], rep, ",".join(x[0] for x in vs)[:24],
                    " ".join("%s=%.2f" % (k.replace("_leaker", "L").replace("_referrer", "R"), x["cpu"])
                             for k, x in res.items() if isinstance(x, dict)),
                    res["load"][0], " FAILED " + str(bad) if bad else ""), flush=True)


if __name__ == "__main__":
    main()

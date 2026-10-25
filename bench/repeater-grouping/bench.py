#!/usr/bin/env python3
"""bench.py: interleaved, repeated clean-build timing of generated Repeater variants.

  bench.py --grid A --reps 3 --tc stable [--out results/A-stable.jsonl]

A grid is a list of configurations; a configuration is (name, generator arguments, variants).
Variants whose generated crates are byte-identical are measured once (their `key`).
For each rep, for each configuration, the variants are shuffled and measured one after the other,
so that a change of the machine's load hits all the variants of a configuration alike.
One line of JSON per sample is appended to --out as soon as it is measured; a rerun skips the
samples already there. `--sizes` is a size-only pass: it builds each distinct variant once, in parallel
(`--jobs`), and records only the artifact sizes (.rlib after a build, .rmeta after a check) and their
sha1 to results/<grid>-<tc>-sizes.jsonl. Nothing is timed.

Grids (see grids()), all with variants ctl, base and multiK unless said otherwise:
  N  entries per context n = 1..50 at E=1000 (K = 2..32)   D  corpus-like E = 50, 200, 1000 (3 seeds each)
  B  context complexity (none, one, three, lt)             W  other rustc: E=1000 n=4, n=10, corpus-like (--tc)
  X  filler cost: n=1, K = 1..8 (and 12, 16, 32)           S  scaling with E (100 .. 10000)
  U  which uses in the Referrer matter                     I  one ID for all contexts
  M  every variant (MSRV: --tc 1.56 --phases check)       V  rustc version scan (--phases check)
  Z  associated type defaults, NIGHTLY ONLY (--tc nightly) F  empty crates (the floor)
  Q  quick smoke test
"""
import argparse, hashlib, json, os, random, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure import measure, measure_sizes  # noqa: E402

KS = [2, 3, 4, 5, 6, 8, 12, 16, 32]
MK = ["multi%d" % k for k in KS]
MAIN = ["ctl", "base"] + MK
NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 50]


def cfg(name, E, C, cx="three", dist="uniform", unsized=0.12, seed=1, variants=None, extra=None):
    return dict(name=name, E=E, C=C, cx=cx, dist=dist, unsized=unsized, seed=seed,
                variants=variants or MAIN, extra=extra or [])


def grids():
    g = {}
    # N: entries per context n at E=1000 (E is n * round(1000 / n), within 1% of 1000), three generics with bounds
    g["N"] = [cfg("N-n%02d" % n, n * round(1000 / n), round(1000 / n), variants=MAIN) for n in NS]
    # D: corpus-like (sizes drawn from the corpus histogram, mixed contexts, 6.5% unsized), three draws each
    g["D"] = [cfg("D-E%d-s%d" % (e, s), e, 1, cx="mixed", dist="corpus", unsized=0.065, seed=s, variants=MAIN)
              for e in (50, 200, 1000) for s in (1, 2, 3)]
    # B: context complexity at E=1000, 10 entries per context
    g["B"] = [cfg("B-%s-E1000-n10" % cx, 1000, 100, cx=cx, variants=MAIN) for cx in ("none", "one", "three", "lt")]
    # W: other rustc versions (--tc 1.56, 1.60.0, 1.63.0, stable)
    g["W"] = [cfg("W-E1000-n4", 1000, 250, variants=MAIN),
              cfg("W-E1000-n10", 1000, 100, variants=MAIN),
              cfg("W-corpus-E1000", 1000, 1, cx="mixed", dist="corpus", unsized=0.065, variants=MAIN)]
    # X: the cost of a filler `type TypeJ = ();`: singleton contexts, K = 1..8 (and 12, 16, 32), three kinds of context
    Xv = ["ctl", "base"] + ["multi%d" % k for k in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32)]
    g["X"] = [cfg("X-%s-n1" % cx, 1000, 1000, cx=cx, variants=Xv) for cx in ("none", "three", "lt")]
    # S: scaling with the number of entries (10 per context)
    Sv = ["ctl", "base", "multi2", "multi4", "multi8", "multi16", "multi32"]
    g["S"] = [cfg("S-E%d-n10" % e, e, e // 10, variants=Sv) for e in (100, 200, 500, 1000, 2000, 5000, 10000)]
    # U: which uses of the entries in the Referrer's crate matter (E=1000, 10 per context)
    U = ["ctl", "base", "multi4", "multi8", "multi16"]
    g["U"] = [cfg("U-sig", 1000, 100, variants=U, extra=["--uses", "sig"]),
              cfg("U-impl", 1000, 100, variants=U, extra=["--uses", "impl"]),
              cfg("U-wit", 1000, 100, variants=U, extra=["--uses", "wit"])]
    # I: one ID for all the contexts (a big item: NTH counts over all its contexts)
    Iv = ["base", "multi4", "multi8", "multi16"]
    g["I"] = [cfg("I-E1000-n10", 1000, 100, variants=Iv, extra=["--ids", "shared"]),
              cfg("I-E5000-n10", 5000, 500, variants=Iv, extra=["--ids", "shared"])]
    # M: every K on two configurations (used with --tc 1.56 --phases check for the MSRV)
    Mv = ["ctl", "base"] + ["multi%d" % k for k in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32)]
    g["M"] = [cfg("M-E1000-n10", 1000, 100, variants=Mv),
              cfg("M-corpus-E200", 200, 1, cx="mixed", dist="corpus", unsized=0.065, variants=Mv)]
    # V: rustc version scan (use with --phases check): where the quadratic part comes from and goes away
    g["V"] = [cfg("V-E2000-n10", 2000, 200, variants=["ctl", "base", "multi4", "multi8", "multi16"])]
    # Z: associated type defaults (`multiKd`, nightly only): an impl defines only the types it needs
    Zv = ["ctl", "base"] + ["multi%d" % k for k in (4, 8, 16, 32)] + ["multi%dd" % k for k in (4, 8, 16, 32)]
    g["Z"] = [cfg("Z-E1000-n%d" % n, n * round(1000 / n), round(1000 / n), variants=Zv) for n in (1, 2, 4, 10)] + \
             [cfg("Z-corpus-E1000", 1000, 1, cx="mixed", dist="corpus", unsized=0.065, variants=Zv)]
    # Q: the quick smoke test (a few seconds): E=200, 10 per context
    g["Q"] = [cfg("Q-E200-n10", 200, 20, variants=["ctl", "base", "multi4", "multi8"])]
    # F: the floor (empty crates)
    g["F"] = [cfg("F-floor", 0, 0, variants=["base", "multi8", "multi32"])]
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


def size_pass(a, cfgs, plan, env_extra):
    """Builds each distinct variant of each configuration once and records the artifact sizes (deterministic)."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    done = set()
    if os.path.exists(a.out):
        done = {(r["cfg"], r["key"]) for r in map(json.loads, open(a.out))}
    jobs = [(c["name"], key, vs) for c in cfgs for key, vs in plan[c["name"]].items() if (c["name"], key) not in done]
    lock = threading.Lock()

    def one(job):
        name, key, vs = job
        v, d, meta = vs[0]
        res = measure_sizes(d, a.tc, env_extra, remap=not a.no_remap)
        rec = dict(grid=a.grid, cfg=name, key=key, variants=[x[0] for x in vs], tc=a.tc, meta=meta, **res)
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            bad = [k for k, x in res.items() if x["rc"] or x["size"] is None]
            print("%s %-24s %s%s" % (name, ",".join(x[0] for x in vs)[:24],
                                     " ".join("%s=%s" % (k.replace("_leaker", "L").replace("_referrer", "R"), x["size"])
                                              for k, x in res.items()), " FAILED " + str(bad) if bad else ""), flush=True)

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(one, jobs))


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
    ap.add_argument("--sizes", action="store_true",
                    help="size-only pass: build each distinct variant once, no timing, write results/<grid>-<tc>-sizes.jsonl")
    ap.add_argument("--jobs", type=int, default=4, help="parallel builds of the size-only pass")
    ap.add_argument("--no-remap", action="store_true", help="size-only pass: do not map the workspace path to /ws")
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(HERE, "results", "%s-%s%s.jsonl" % (a.grid, a.tc, "-sizes" if a.sizes else ""))
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
    if a.sizes:
        size_pass(a, cfgs, plan, env_extra)
        return
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

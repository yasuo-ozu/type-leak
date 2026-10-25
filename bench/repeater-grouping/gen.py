#!/usr/bin/env python3
"""Generates a Leaker crate and a Referrer crate for one Repeater encoding variant.

Usage:
  gen.py --out DIR --variant base|ctl|multiK|multiKd [--E 1000 --C 100 --cx three --seed 1 ...]

The set of entries (their types, contexts and order) depends only on
(E, C or --sizes, cx, seed, unsized fraction), never on the variant, so that all variants of a
configuration encode the same entries.

Variants
  base      one impl per entry, `type Type = Ty;`
  ctl       control without any Repeater: no impls, the Referrer writes the original types
  multiK    the Repeater has K associated types Type0..Type{K-1} (all `?Sized`); one impl carries up to K
            entries of one context; the impl must define all K types (associated type defaults are
            unstable), so the unused ones are `()`. K=1 is `base` with the type named Type0.
  multiKd   NIGHTLY ONLY, an experiment: like multiK, but the trait gives every type the default `()`
            (`#![feature(associated_type_defaults)]`), so an impl defines only the types it needs.
            It shows what defaults would bring; it cannot be used on stable.

Earlier versions also had tuple encodings (`type Type = (T0, T1, ..)` plus indexing impls). They were
dropped: the second projection made the Referrer's crate 20-90% slower on stable.
"""
import argparse
import os
import random
import sys

NN = 12  # number of types in each family of the pool


class Cx:
    """The generics of a context."""

    def __init__(self, name):
        self.name = name
        if name == "none":
            self.impl_g, self.use_g, self.tys, self.lts = [], [], [], []
            self.enc = "()"
        elif name == "one":
            self.impl_g, self.use_g = ["T"], ["T"]
            self.tys, self.lts = ["T"], []
            self.enc = "(T,)"
        elif name == "three":
            self.impl_g = ["T: Clone + Into<U>", "U: Clone", "V: Default"]
            self.use_g = list(self.impl_g)
            self.tys, self.lts = ["T", "U", "V"], []
            self.enc = "(T, U, V)"
        elif name == "lt":
            self.impl_g = ["'a", "'b", "T: Clone"]
            self.use_g = ["'a", "'b: 'a", "T: 'a + Clone"]
            self.tys, self.lts = ["T"], ["'a", "'b"]
            self.enc = "(&'a (), &'b (), T, &'a &'b (), &'a T)"
        else:
            raise SystemExit("unknown cx " + name)

    def names(self):
        """Arguments that name the generics, for `Tr<..>`."""
        return [g.split(":")[0] for g in self.use_g]


def leaf(rng, cx, M):
    if cx.tys and rng.random() < 0.6:
        return rng.choice(cx.tys)
    return "%sN%d" % (M, rng.randrange(NN))


def path(rng, cx, M, arity=None):
    if arity is None:
        arity = rng.choice([0, 1, 1, 2, 3])
    if arity == 0:
        return "%sN%d" % (M, rng.randrange(NN))
    args = ", ".join(leaf(rng, cx, M) for _ in range(arity))
    return "%sA%d_%d<%s>" % (M, arity, rng.randrange(NN), args)


def gen_entry(rng, cx, M, unsized):
    """Returns the type text with M as the prefix of the Leaker's module (`crate::m::`)."""
    if unsized:
        r = rng.random()
        if r < 0.6:
            return "(dyn %sD%d<%s> + 'static)" % (M, rng.randrange(NN), leaf(rng, cx, M))
        if r < 0.85:
            return "%sSl<%s>" % (M, leaf(rng, cx, M))
        return "%sSt" % M
    r = rng.random()
    if r < 0.55:
        return path(rng, cx, M)
    if r < 0.65:
        return "%sA1_%d<%s>" % (M, rng.randrange(NN), path(rng, cx, M, rng.choice([0, 2])))
    if r < 0.80:
        if cx.name == "lt":
            return "&'a %s" % path(rng, cx, M)
        return "&'static %sN%d" % (M, rng.randrange(NN))
    if r < 0.90:
        return "(%s, %s)" % (path(rng, cx, M), path(rng, cx, M))
    if cx.name == "lt":
        return "%sL1_%d<'a, T>" % (M, rng.randrange(NN))
    return path(rng, cx, M)


def sizes_for(E, C, dist, rng, sizes):
    if sizes:
        return list(sizes)
    if E == 0 or C == 0:
        return []
    if dist == "corpus":
        import json
        emp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "sizes.json")))
        out, tot = [], 0
        while tot < E:
            n = min(rng.choice(emp), E - tot)
            out.append(n)
            tot += n
        return out
    base = E // C
    rem = E - base * C
    out = [base + (1 if i < rem else 0) for i in range(C)]
    if dist == "skew":
        # a geometric-ish skew with the same total
        w = [rng.paretovariate(1.2) for _ in range(C)]
        s = sum(w)
        out = [max(1, int(round(E * x / s))) for x in w]
    return [x for x in out if x > 0]


def make_entries(args):
    """Returns the contexts: a list of (cx, [(ty, unsized)])."""
    rng = random.Random(args.seed * 1000003 + 17)
    sizes = sizes_for(args.E, args.C, args.dist, rng, args.sizes_list)
    if args.cx == "mixed":
        names = []
        # the shares of the property-test corpus: none 42, one 14, three (bounds) 19, lt 25
        for i in range(len(sizes)):
            r = rng.random()
            names.append("none" if r < 0.42 else "one" if r < 0.56 else "three" if r < 0.75 else "lt")
    else:
        names = [args.cx] * len(sizes)
    ctxs = []
    for n, name in zip(sizes, names):
        cx = Cx(name)
        ents = []
        for _ in range(n):
            u = rng.random() < args.unsized
            ents.append((gen_entry(rng, cx, "{M}", u), u))
        ctxs.append((cx, ents))
    return ctxs


# ---------------------------------------------------------------- the Leaker's crate

def pool():
    out = ["pub mod m {"]
    for i in range(NN):
        out.append("    pub struct N%d;" % i)
        out.append("    pub struct A1_%d<T>(pub T);" % i)
        out.append("    pub struct A2_%d<T, U>(pub T, pub U);" % i)
        out.append("    pub struct A3_%d<T, U, V>(pub T, pub U, pub V);" % i)
        out.append("    pub struct L1_%d<'a, T: 'a>(pub &'a T);" % i)
        out.append("    pub trait D%d<T> {}" % i)
    out.append("    pub type Sl<T> = [T];")
    out.append("    pub type St = str;")
    out.append("}")
    return "\n".join(out)


def gparams(gs):
    return ", ".join(gs)


def impl_header(cx, trait_args):
    gs = cx.impl_g + ["__S: ?Sized"]
    return "impl<%s> Repeater<%s> for __S" % (gparams(gs), trait_args)


class Plan:
    """What a variant writes for the entries: the impls and the use-site type of each entry."""

    def __init__(self, variant, ctxs, shared_ids=False):
        self.variant = variant
        self.ctxs = ctxs
        self.impls = []
        self.uses = {}  # (context index, entry index) -> use-site type, `{X}` stands for the host
        self.shared_ids = shared_ids
        self.parse(variant)

    def parse(self, v):
        import re
        m = re.match(r"^(base|ctl|multi)(\d*)(d?)$", v)
        if not m or (m.group(1) == "multi") != bool(m.group(2)) or (m.group(3) and m.group(1) != "multi"):
            raise SystemExit("bad variant %r: base, ctl, multiK or multiKd" % v)
        self.kind = m.group(1)
        self.K = int(m.group(2)) if m.group(2) else 1
        self.defaults = m.group(3) == "d"

    def build(self, M_leak):
        """M_leak is the module prefix in the Leaker (`crate::m::`)."""
        kind, K = self.kind, self.K
        impls = []
        uses = {}
        nth_run = 0
        last_len = 0
        for ci, (cx, ents) in enumerate(self.ctxs):
            nth_run += len(impls) - last_len
            last_len = len(impls)
            ID = 1000 if self.shared_ids else 1000 + ci
            off = nth_run if self.shared_ids else 0
            enc = cx.enc
            if kind == "ctl":
                continue
            if kind == "base":
                for ei, (ty, u) in enumerate(ents):
                    t = ty.replace("{M}", M_leak)
                    impls.append(
                        "%s { type Type = %s; }"
                        % (impl_header(cx, "%s, %du64, %dusize" % (enc, ID, ei + off)), t)
                    )
                    uses[(ci, ei)] = ("<%s as Repeater<%s, %du64, %dusize>>::Type" % ("{X}", enc, ID, ei + off))
                continue
            gi = 0
            for start in range(0, len(ents), K):
                chunk = ents[start: start + K]
                tys = [t.replace("{M}", M_leak) for t, _ in chunk]
                n_defined = len(tys) if self.defaults else K
                body = "".join(
                    "type Type%d = %s; " % (j, tys[j] if j < len(tys) else "()") for j in range(n_defined)
                )
                impls.append("%s { %s}" % (impl_header(cx, "%s, %du64, %dusize" % (enc, ID, gi + off)), body))
                for j in range(len(chunk)):
                    uses[(ci, start + j)] = "<{X} as Repeater<%s, %du64, %dusize>>::Type%d" % (enc, ID, gi + off, j)
                gi += 1
        self.impls = impls
        self.uses = uses

    def counts(self):
        multi = self.kind == "multi"
        return dict(
            impls=len(self.impls),
            assoc_per_impl=self.K if multi else 1,
            assoc_defined=(sum(self.impl_types(i) for i in self.impls) if multi else len(self.impls)),
        )

    @staticmethod
    def impl_types(impl):
        return impl.count(" type Type")


def repeater_def(kind, K, defaults=False):
    if kind == "multi":
        if defaults:
            types = " ".join("type Type%d: ?Sized = ();" % j for j in range(K))
        else:
            types = " ".join("type Type%d: ?Sized;" % j for j in range(K))
    else:
        types = "type Type: ?Sized;"
    return "pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> { %s }" % types


def sig_fn(name, cx, ty_in, ty_out, is_unsized):
    g = gparams((["'q"] if is_unsized else []) + cx.use_g)
    gg = "<%s>" % g if g else ""
    if is_unsized:
        return "pub fn %s%s(x: &'q %s) -> &'q %s { x }" % (name, gg, ty_in, ty_out)
    return "pub fn %s%s(x: %s) -> %s { x }" % (name, gg, ty_in, ty_out)


def write_crates(args, outdir):
    ctxs = make_entries(args)
    plan = Plan(args.variant, ctxs, shared_ids=(args.ids == "shared"))
    plan.build("crate::m::")
    uses_on = set(args.uses.split(","))

    # ---- the Leaker's crate
    L = []
    L.append("#![allow(warnings)]")
    if plan.defaults:
        L.append("#![feature(associated_type_defaults)]")
    L.append(repeater_def(plan.kind, plan.K, plan.defaults))
    L.append(pool())
    # the traits declared in the Leaker's crate whose methods take and return the entries
    if "impl" in uses_on:
        for ci, (cx, ents) in enumerate(ctxs):
            gg = "<%s>" % gparams(cx.use_g) if cx.use_g else ""
            L.append("pub trait Tr%d%s {" % (ci, gg))
            for ei, (ty, u) in enumerate(ents):
                t = ty.replace("{M}", "crate::m::")
                if u:
                    L.append("    fn m%d<'q>(&self, x: &'q %s) -> &'q %s;" % (ei, t, t))
                else:
                    L.append("    fn m%d(&self, x: %s) -> %s;" % (ei, t, t))
            L.append("}")
    L.extend(plan.impls)
    lk = "\n".join(L) + "\n"

    # ---- the Referrer's crate
    R = []
    R.append("#![allow(warnings)]")
    R.append("use ::leaker::Repeater;")
    X = "()"
    for ci, (cx, ents) in enumerate(ctxs):
        names = cx.names()
        gg = "<%s>" % gparams(cx.use_g) if cx.use_g else ""
        targs = "<%s>" % ", ".join(names) if names else ""
        if "impl" in uses_on:
            R.append("pub struct Host%d;" % ci)
            R.append("impl%s ::leaker::Tr%d%s for Host%d {" % (gg, ci, targs, ci))
        meths = []
        fns = []
        for ei, (ty, u) in enumerate(ents):
            orig = ty.replace("{M}", "::leaker::m::")
            p = orig if plan.kind == "ctl" else plan.uses[(ci, ei)].replace("{X}", X)
            if "impl" in uses_on:
                if u:
                    meths.append("    fn m%d<'q>(&self, x: &'q %s) -> &'q %s { x }" % (ei, p, p))
                else:
                    meths.append("    fn m%d(&self, x: %s) -> %s { x }" % (ei, p, p))
            if "sig" in uses_on:
                fns.append(sig_fn("f%d_%d" % (ci, ei), cx, p, p, u))
            if "wit" in uses_on:
                fns.append(sig_fn("w%d_%d" % (ci, ei), cx, p, orig, u))
        if "impl" in uses_on:
            R.extend(meths)
            R.append("}")
        R.extend(fns)
    rf = "\n".join(R) + "\n"

    os.makedirs(os.path.join(outdir, "leaker", "src"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "referrer", "src"), exist_ok=True)
    with open(os.path.join(outdir, "Cargo.toml"), "w") as f:
        f.write('[workspace]\nmembers = ["leaker", "referrer"]\nresolver = "2"\n\n'
                '[profile.dev]\ndebug = 0\n')
    with open(os.path.join(outdir, "leaker", "Cargo.toml"), "w") as f:
        f.write('[package]\nname = "leaker"\nversion = "0.0.0"\nedition = "2018"\npublish = false\n')
    with open(os.path.join(outdir, "referrer", "Cargo.toml"), "w") as f:
        f.write('[package]\nname = "referrer"\nversion = "0.0.0"\nedition = "2018"\npublish = false\n\n'
                '[dependencies]\nleaker = { path = "../leaker" }\n')
    with open(os.path.join(outdir, "leaker", "src", "lib.rs"), "w") as f:
        f.write(lk)
    with open(os.path.join(outdir, "referrer", "src", "lib.rs"), "w") as f:
        f.write(rf)
    meta = dict(
        variant=args.variant, E=sum(len(e) for _, e in ctxs), C=len(ctxs), cx=args.cx, seed=args.seed,
        unsized=args.unsized, uses=args.uses, dist=args.dist,
        n_unsized=sum(1 for _, e in ctxs for _, u in e if u),
        **plan.counts(),
    )
    import json
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--E", type=int, default=1000)
    ap.add_argument("--C", type=int, default=100)
    ap.add_argument("--cx", default="three", choices=["none", "one", "three", "lt", "mixed"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--unsized", type=float, default=0.12)
    ap.add_argument("--dist", default="uniform", choices=["uniform", "skew", "corpus"])
    ap.add_argument("--sizes", default="", help="explicit comma list of entries per context")
    ap.add_argument("--uses", default="sig,impl,wit")
    ap.add_argument("--ids", default="per-ctx", choices=["per-ctx", "shared"],
                    help="per-ctx: each context is an item with its own ID; shared: one ID, NTH counts over all contexts")
    args = ap.parse_args()
    args.sizes_list = [int(x) for x in args.sizes.split(",") if x] if args.sizes else None
    meta = write_crates(args, args.out)
    print(meta)


if __name__ == "__main__":
    main()

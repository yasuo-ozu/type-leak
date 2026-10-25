#!/usr/bin/env python3
"""Generates a Leaker crate and a Referrer crate for one Repeater encoding variant.

Usage:
  gen.py --out DIR --variant base|multiK|tupK|tupKp|nestK [--E 1000 --C 100 --cx three --seed 1 ...]

The set of entries (their types, contexts and order) depends only on
(E, C or --sizes, cx, seed, unsized fraction), never on the variant, so that all variants of a
configuration encode the same entries.

Variants
  base      one impl per entry, `type Type = Ty;`
  multiK    the Repeater has K associated types Type0..Type{K-1}; one impl carries up to K entries of
            one context, the unused types are `()`.
  tupK      `type Type = (*const T0, *const T1, ..)` for up to K entries of one context; the use site
            indexes it with `<.. as Repeater<(), IDX, j>>::Type`, with one indexing impl per
            (arity, index) pair that is used, emitted once in the Leaker's crate. The elements are
            wrapped in `*const` so that unsized entries (and entries whose Sizedness the Leaker
            cannot know) can be elements; the indexing impl strips the pointer.
  tupKp     like tupK, with the arity padded up to a power of two (`*const ()` fillers), so
            the arities in use are 1, 2, 4, .. and the indexing impls are few.
  nestK     nested pairs `(*const T0, (*const T1, (.., ())))`: the indexing impl of index j works for
            any length, so K impls are emitted in all.
  A group of one entry in tup/nest is written directly (`type Type = Ty`), as the baseline does
  (`--no-direct1` turns that off).
"""
import argparse
import os
import random
import sys

NN = 12  # number of types in each family of the pool
IDX = "18446744073709551615u64"  # reserved ID of the indexing impls
IDX_PTR = "18446744073709551614u64"


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


def groups_of(ents, K, pad_pow2=False):
    """Chunks the entries of one context into groups of at most K."""
    return [ents[i: i + K] for i in range(0, len(ents), K)]


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class Plan:
    """What a variant writes for the entries: impls and the use-site type of each entry."""

    def __init__(self, variant, ctxs, direct1=True, shared_ids=False):
        self.variant = variant
        self.ctxs = ctxs
        self.impls = []  # (text)
        self.use = {}  # (ctx index, entry index) -> function(M, x_prefix) -> type text at use site
        self.index_impls = []
        self.assoc_per_impl = []
        self.nimpls_entries = 0
        self.arities = set()
        self.kind = "base"
        self.K = 1
        self.direct1 = direct1
        self.shared_ids = shared_ids
        self.parse(variant)

    def parse(self, v):
        import re
        m = re.match(r"^(base|ctl|multi|tup|nest)(\d*)(p?)(r?)$", v)
        if not m:
            raise SystemExit("bad variant " + v)
        self.kind = m.group(1)
        self.K = int(m.group(2)) if m.group(2) else 1
        self.pad = m.group(3) == "p"
        self.raw = m.group(4) == "r"

    def build(self, M_leak):
        """M_leak is the module prefix in the Leaker (`crate::m::`)."""
        kind, K = self.kind, self.K
        impls = []
        used_arities = set()
        max_nest = 0
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
            if self.raw and kind in ("tup", "nest"):
                # raw elements (no `*const`): the unsized entries are ungrouped
                sized = [i for i, (t, u) in enumerate(ents) if not u]
                uns = [i for i, (t, u) in enumerate(ents) if u]
                groups = [sized[k: k + K] for k in range(0, len(sized), K)] + [[i] for i in uns]
                for g in groups:
                    tys = [ents[i][0].replace("{M}", M_leak) for i in g]
                    hdr = impl_header(cx, "%s, %du64, %dusize" % (enc, ID, gi + off))
                    if len(g) == 1:
                        impls.append("%s { type Type = %s; }" % (hdr, tys[0]))
                        uses[(ci, g[0])] = "<{X} as Repeater<%s, %du64, %dusize>>::Type" % (enc, ID, gi + off)
                    else:
                        n = len(g)
                        if kind == "tup":
                            tup = "(%s)" % ", ".join(tys)
                            used_arities.add(n)
                        else:
                            tup = "()"
                            for p in reversed(tys):
                                tup = "(%s, %s)" % (p, tup)
                            max_nest = max(max_nest, n)
                        impls.append("%s { type Type = %s; }" % (hdr, tup))
                        for j, i in enumerate(g):
                            uses[(ci, i)] = (
                                "<<{X} as Repeater<%s, %du64, %dusize>>::Type as Repeater<(), %s, %dusize>>::Type"
                                % (enc, ID, gi + off, IDX, j)
                            )
                    gi += 1
                continue
            for start in range(0, len(ents), K):
                chunk = ents[start: start + K]
                tys = [t.replace("{M}", M_leak) for t, _ in chunk]
                if kind == "multi":
                    body = "".join(
                        "type Type%d = %s; " % (j, tys[j] if j < len(tys) else "()") for j in range(K)
                    )
                    impls.append("%s { %s}" % (impl_header(cx, "%s, %du64, %dusize" % (enc, ID, gi + off)), body))
                    for j in range(len(chunk)):
                        uses[(ci, start + j)] = "<{X} as Repeater<%s, %du64, %dusize>>::Type%d" % (enc, ID, gi + off, j)
                else:
                    if len(chunk) == 1 and self.direct1:
                        impls.append(
                            "%s { type Type = %s; }" % (impl_header(cx, "%s, %du64, %dusize" % (enc, ID, gi + off)), tys[0])
                        )
                        uses[(ci, start)] = "<{X} as Repeater<%s, %du64, %dusize>>::Type" % (enc, ID, gi + off)
                    else:
                        n = len(chunk)
                        if self.pad:
                            n = next_pow2(n)
                        ptys = ["*const %s" % t for t in tys] + ["*const ()"] * (n - len(tys))
                        if kind == "tup":
                            tup = "(%s,)" % ", ".join(ptys) if n == 1 else "(%s)" % ", ".join(ptys)
                            used_arities.add(n)
                        else:
                            tup = "()"
                            for p in reversed(ptys):
                                tup = "(%s, %s)" % (p, tup)
                            max_nest = max(max_nest, len(chunk))
                        impls.append(
                            "%s { type Type = %s; }" % (impl_header(cx, "%s, %du64, %dusize" % (enc, ID, gi + off)), tup)
                        )
                        for j in range(len(chunk)):
                            uses[(ci, start + j)] = (
                                "<<{X} as Repeater<%s, %du64, %dusize>>::Type as Repeater<(), %s, %dusize>>::Type"
                                % (enc, ID, gi + off, IDX, j)
                            )
                gi += 1
        self.impls = impls
        self.index_impls = []
        if kind == "tup" and self.raw:
            for n in sorted(used_arities):
                names = ["A%d" % i for i in range(n)]
                for j in range(n):
                    self.index_impls.append(
                        "impl<%s> Repeater<(), %s, %dusize> for (%s) { type Type = %s; }"
                        % (", ".join(names), IDX, j, ", ".join(names), names[j])
                    )
        elif kind == "nest" and self.raw:
            for j in range(max_nest):
                names = ["A%d" % i for i in range(j + 1)]
                tup = "R"
                for a in reversed(names):
                    tup = "(%s, %s)" % (a, tup)
                self.index_impls.append(
                    "impl<%s, R> Repeater<(), %s, %dusize> for %s { type Type = %s; }"
                    % (", ".join(names), IDX, j, tup, names[j])
                )
        elif kind == "tup":
            for n in sorted(used_arities):
                names = ["A%d" % i for i in range(n)]
                tup = "(%s,)" % ", ".join("*const " + a for a in names) if n == 1 else "(%s)" % ", ".join("*const " + a for a in names)
                for j in range(n):
                    self.index_impls.append(
                        "impl<%s> Repeater<(), %s, %dusize> for %s { type Type = %s; }"
                        % (", ".join(a + ": ?Sized" for a in names), IDX, j, tup, names[j])
                    )
        elif kind == "nest":
            for j in range(max_nest):
                names = ["A%d" % i for i in range(j + 1)]
                tup = "R"
                for a in reversed(names):
                    tup = "(*const %s, %s)" % (a, tup)
                self.index_impls.append(
                    "impl<%s, R> Repeater<(), %s, %dusize> for %s { type Type = %s; }"
                    % (", ".join(a + ": ?Sized" for a in names), IDX, j, tup, names[j])
                )
        self.uses = uses
        self.used_arities = sorted(used_arities)
        self.max_nest = max_nest

    def counts(self):
        return dict(
            impls=len(self.impls),
            index_impls=len(self.index_impls),
            assoc_per_impl=self.K if self.kind == "multi" else 1,
        )


def repeater_def(variant_kind, K):
    if variant_kind == "multi":
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
    plan = Plan(args.variant, ctxs, direct1=not args.no_direct1, shared_ids=(args.ids == "shared"))
    plan.build("crate::m::")
    uses_on = set(args.uses.split(","))

    # ---- the Leaker's crate
    L = []
    L.append("#![allow(warnings)]")
    L.append(repeater_def(plan.kind, plan.K))
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
    L.extend(plan.index_impls)
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
        arities=plan.used_arities, max_nest=plan.max_nest,
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
    ap.add_argument("--no-direct1", action="store_true")
    ap.add_argument("--ids", default="per-ctx", choices=["per-ctx", "shared"],
                    help="per-ctx: each context is an item with its own ID; shared: one ID, NTH counts over all contexts")
    args = ap.parse_args()
    args.sizes_list = [int(x) for x in args.sizes.split(",") if x] if args.sizes else None
    meta = write_crates(args, args.out)
    print(meta)


if __name__ == "__main__":
    main()

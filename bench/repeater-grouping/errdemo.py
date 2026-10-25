#!/usr/bin/env python3
"""errdemo.py: what rustc says when a Repeater use or a Repeater impl is wrong, per encoding.

Writes errdemo/<scenario>-<variant>/{leaker,referrer} and prints nothing; run_errdemo.sh compiles them.
Entries of the context `<T: Clone>`: 0 `W<T>`, 1 `Bnd<T>` (needs T: Copy, so it is wrong there in S1), 2 `N`.
"""
import os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "errdemo")
ENT = ["W<T>", "Bnd<T>", "N"]
POOL = "pub struct W<T>(pub T); pub struct Bnd<T: Copy>(pub T); pub struct N;"
CTX = "T: Clone"

def leaker(variant, entries):
    if variant == "base":
        tr = "pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> { type Type: ?Sized; }"
        impls = ["impl<T: Clone, __S: ?Sized> Repeater<(T,), 1u64, %dusize> for __S { type Type = %s; }" % (i, "crate::" + e) for i, e in enumerate(entries)]
    else:  # multiK
        k = int(variant[len("multi"):])
        tr = "pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> { %s }" % " ".join(
            "type Type%d: ?Sized;" % j for j in range(k))
        ts = ["crate::" + e for e in entries] + ["()"] * (k - len(entries))
        impls = ["impl<T: Clone, __S: ?Sized> Repeater<(T,), 1u64, 0usize> for __S { %s }" % " ".join("type Type%d = %s;" % (j, t) for j, t in enumerate(ts))]
    return "%s\n%s\n%s\n" % (tr, POOL, "\n".join(impls))

def use(variant, j, nth_override=None):
    if variant == "base":
        return "<() as ::leaker::Repeater<(T,), 1u64, %dusize>>::Type" % (j if nth_override is None else nth_override)
    return "<() as ::leaker::Repeater<(T,), 1u64, 0usize>>::Type%d" % j

def write(name, lk, rf):
    d = os.path.join(OUT, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d + "/leaker/src"); os.makedirs(d + "/referrer/src")
    open(d + "/Cargo.toml", "w").write('[workspace]\nmembers = ["leaker", "referrer"]\nresolver = "2"\n')
    open(d + "/leaker/Cargo.toml", "w").write('[package]\nname = "leaker"\nversion = "0.0.0"\nedition = "2018"\n')
    open(d + "/referrer/Cargo.toml", "w").write('[package]\nname = "referrer"\nversion = "0.0.0"\nedition = "2018"\n[dependencies]\nleaker = { path = "../leaker" }\n')
    open(d + "/leaker/src/lib.rs", "w").write(lk)
    open(d + "/referrer/src/lib.rs", "w").write(rf)

for v in ("base", "multi4", "multi8"):
    # S1: the leaked type of entry 1 is not well-formed in the context (Leaker side)
    write("S1-" + v, leaker(v, ENT), "")
    good = leaker(v, [ENT[0], "N", "W<T>"])  # entries 0 `W<T>`, 1 `N`, 2 `W<T>`
    # S2: the use site lacks the context's bound
    write("S2-" + v, good, "pub fn f<T>(x: %s) {}\n" % use(v, 0))
    # S3: the witness expects another type (entry 1 is N, the witness says W<T>)
    write("S3-" + v, good, "pub fn w<T: Clone>(x: %s) -> ::leaker::W<T> { x }\n" % use(v, 1))
    # S4: a use of an entry that does not exist
    if v == "base":
        u = use(v, 0, nth_override=9)
    else:  # one past the last type of the trait
        u = "<() as ::leaker::Repeater<(T,), 1u64, 0usize>>::Type%d" % (int(v[len("multi"):]) + 3)
    write("S4-" + v, good, "pub fn f<T: Clone>(x: %s) {}\n" % u)

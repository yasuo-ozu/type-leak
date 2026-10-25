# Repeater grouping benchmark

Does grouping the entries of one context into fewer Repeater impls reduce compile time?

Today the Leaker emits one impl per entry:

```rust
impl<'x, Y: Bound, __S: ?Sized> Repeater<(&'x (), Y), ID, NTH> for __S { type Type = Leaked<'x, Y>; }
```

A big item makes many such impls, all blanket impls of one trait. This directory generates Leaker and
Referrer crates for several encodings and measures clean `cargo build` and `cargo check` of both.
Nothing here is part of the published package (`bench` is in `exclude`). No library code is changed.

Grouping is possible only for entries of the same context: same generics and bounds, so the same impl
header and the same first argument.

## Variants

| variant | impls | use site |
|---|---|---|
| `base` | one impl per entry, `type Type` | `<X as Repeater<Args, ID, NTH>>::Type` |
| `multiK` | the Repeater has `Type0..Type{K-1}` (all `?Sized`); one impl carries up to K entries of one context; unused types are `()` (associated type defaults are unstable) | `<X as Repeater<Args, ID, G>>::TypeJ` |
| `tupK` | `type Type = (*const T0, *const T1, ..)` for up to K entries; the arity is exact | `<<X as Repeater<Args, ID, G>>::Type as Repeater<(), IDX, J>>::Type` |
| `tupKp` | like `tupK`, arity padded to a power of two with `*const ()` | same |
| `nestK` | nested pairs `(*const T0, (*const T1, (.., ())))` | same, with the indexing impls below |
| `tupKr`, `nestKr` | like `tupK`/`nestK` without `*const`; unsized entries stay ungrouped | same |
| `ctl` | control: no Repeater. The Leaker has no impls, the Referrer writes the original types | the original type |

- `IDX` is `u64::MAX`, reserved. The indexing impls are Repeater impls too, so the Repeater stays the only
  shared item: `impl<A0: ?Sized, A1: ?Sized, A2: ?Sized> Repeater<(), IDX, 1> for (*const A0, *const A1, *const A2) { type Type = A1; }`.
  `tupK` needs one per (arity, index) pair in use. `nestK` needs K in all, for any length:
  `impl<A0: ?Sized, A1: ?Sized, R> Repeater<(), IDX, 1> for (*const A0, (*const A1, R)) { type Type = A1; }`.
- **Unsized entries.** A tuple element must be `Sized` (only the last may not). The Leaker works on syntax, so
  it cannot know whether `a::Alias` or `Wrap<dyn Tr>` is `Sized`. Every element is therefore `*const Ty`,
  which is always `Sized`. The indexing impl strips the pointer in the same projection step. `tupKr` and
  `nestKr` measure the cheaper variant without it.
- **Single-entry groups** in `tup`/`nest` are written directly (`type Type = Ty`), as `base` does.
- Entries of different contexts are never grouped. Each context has its own ID, as if it were its own item.
  With one ID for all contexts (grid `I`) the results are about the same.

## What is generated

`gen.py` writes a workspace with a `leaker` and a `referrer` crate. The entries depend only on
`(E, C or sizes, cx, seed, unsized share)`, never on the variant.

- **Contexts** (`--cx`): `none`; `one` (`T`); `three` (`T: Clone + Into<U>, U: Clone, V: Default`);
  `lt` (`'a, 'b: 'a, T: 'a + Clone`, encoded `(&'a (), &'b (), T, &'a &'b (), &'a T)`); `mixed` (42% none,
  14% one, 19% three, 25% lt, the shares of the corpus).
- **Entry types**: paths into the Leaker (`crate::m::A2_3<T, N5>`), `&'a` paths, tuples, nested paths, and
  12% unsized ones (`dyn D3<T> + 'static`, `[T]` and `str` through type aliases).
- **Leaker**: the Repeater, the module `m` of the types, one trait `Tr_k<G>` per context whose methods
  take and return the entries, and the impls.
- **Referrer**: each entry is used three times, forcing the normalization:
  - a fn `f<G>(x: P) -> P { x }`;
  - a method of `impl<G> Tr_k<G> for Host_k` with the same signature as the Leaker's trait;
  - a witness `w<G>(x: P) -> Orig { x }`, which checks that `P` is the original type.

  `P` is `<() as Repeater<..>>::Type..`. Unsized entries are behind `&'q`.

## Run it

Prerequisites: Linux, `python3` (written with 3.14; standard library only), `rustup` with the toolchains you use
(`stable` is the default; `nightly`, `1.85.0`, `1.75.0`, `1.56` for the other runs). The generated crates have
no dependencies, so everything runs offline.

Optional, for stable numbers:

- `BENCH_CPUS=0-11` pins everything to the given CPUs. On a hybrid Intel CPU pin to the P-cores
  (`lscpu -e`): a process that moves to an E-core takes up to 3x the CPU time.
- Instruction counts use `perf_event_open` (`/proc/sys/kernel/perf_event_paranoid <= 2`; user space only). If
  it is not available the `ginstr` fields are null and the instruction tables are skipped.

```sh
cd bench/repeater-grouping
export BENCH_CPUS=0-11                     # optional

# smoke test, about 10 s: E=200, ctl/base/multi8/tup8
python3 bench.py --grid Q --reps 1 --tc stable
python3 report.py results/Q-stable.jsonl

# one crate pair by hand
python3 gen.py --out /tmp/x --variant tup8 --E 1000 --C 100 --cx three
python3 measure.py /tmp/x --tc stable

# the runs of this README
./run_full.sh                              # 1-2 h on an idle machine; ends with python3 make_report.py
```

The machine should be idle (the load average is recorded in each sample). Each configuration is measured
by interleaving its variants (shuffled per repetition); 3 repetitions, 5 for grids A, B, D.
The median and the spread (max/min over repetitions) are reported. The spread of the runs below
was 1.03 (median), 1.08 (p90), 1.23 (max).

| file | what it does |
|---|---|
| `gen.py` | generator: `--variant base\|ctl\|multiK\|tupK[p][r]\|nestK[r] --E --C --cx --dist uniform\|skew\|corpus --uses sig,impl,wit --ids per-ctx\|shared --unsized` |
| `bench.py` | grids A B C D F I K M Q R S T U V (list at the top of the file): generates, measures, appends one JSON line per sample to `results/<grid>-<tc>.jsonl`; resumable |
| `measure.py` | clean `cargo build` and `cargo check` of the Leaker, then of the Referrer: CPU (user+sys of the children), wall, instructions |
| `perfcount.py` | instruction counter and `BENCH_CPUS` pinning |
| `passes.py` | `-Ztime-passes` of one crate, best of N (uses `RUSTC_BOOTSTRAP=1` on stable) |
| `errdemo.py`, `run_errdemo.sh` | wrong Repeater impls and uses, compiled per encoding (`report/errdemo.txt`) |
| `report.py` | tables of one results file; `make_report.py` writes all of them and the summaries of `summaries.py` into `report/` |
| `corpus_hist.py`, `run_corpus.sh`, `prop-hist-logging.patch` | entries per context of the property-test corpus (below) |
| `run_grids.sh`, `run_toolchains.sh`, `run_passes.sh`, `run_full.sh` | the drivers |

`work/`, `results/`, `logs/` and `errdemo/` are created here and ignored by git.

### The corpus histogram

It needs the property test with one logging statement added to `repeater_impls()` in
`tests/crates/macro/prop.rs`, written as a patch. It was made against the `error-recovery` branch, commit
**1de07a8** (`Check every allow rule after the analysis with clean copies, and generate values`). The file is the same
at c3e3255. The patch also passes `git apply --check` on 0.8.3 (f4da3b3), but the corpus was only
measured on `error-recovery`.

```sh
git worktree add ../type-leak-corpus 1de07a8
git -C ../type-leak-corpus apply "$PWD/prop-hist-logging.patch"
./run_corpus.sh ../type-leak-corpus      # property test, seeds 1-3, 60 cases: corpus/hist-seed*.tsv
python3 corpus_hist.py                   # histogram, and corpus/sizes.json for gen.py --dist corpus
```

## Results

Machine: i7-1280P, pinned to the P-core threads. rustc 1.99.0 stable unless noted.
Times are CPU seconds (user+sys of cargo and its children), median. Wall time is within 1% of CPU
(rustc's front end is single-threaded), and is in `report/`. "L" is the Leaker crate, "R" the Referrer crate.

### Entries per context in the property-test corpus

181 Leaker crates (seeds 1-3, 60 cases each), 10258 entries.

| entries per context | contexts | share of contexts | share of entries |
|---|---|---|---|
| 1 | 4238 | 70.3% | 41.3% |
| 2 | 896 | 14.9% | 17.5% |
| 3 | 410 | 6.8% | 12.0% |
| 4 | 189 | 3.1% | 7.4% |
| 5-8 | 219 | 3.6% | 12.5% |
| 9-16 | 67 | 1.1% | 7.8% |
| 17-32 | 8 | 0.1% | 1.5% |

- **Contexts** (`Referrer::context(ix)`, per item): 6027, mean 1.70, max 24. Grouping by identical impl
  header and arguments gives 5663 groups, mean 1.81, max 40. Perfect grouping would keep 62.8% of the
  impls at K=4 and 58.8% at K=∞.
- **Per crate**: E median 49, mean 57, max 197; 15 items and 32 contexts (median).
- **Per item** (one macro call): median 2 entries and 2 contexts; mean 3.6, p90 7, p99 21, max 46.
  Items with 9 or more entries still have 66% single-entry contexts.
- **Shapes**: no generics 42%, plain generics 14%, bounds 19%, lifetimes 15%, lifetimes and bounds 10.5%.
  About 6.5% of the entries are unsized.

The generated generics are the property test's, so this is a corpus of odd shapes, not of real code.
The "corpus-like" runs below draw their context sizes from this histogram.

### E=1000, 10 entries per context, three generics with bounds

`report/A-stable-quiet.txt`, 5 repetitions. Build and check are `cargo build` and `cargo check` (debug,
`CARGO_INCREMENTAL=0`).

| variant | impls (+index impls) | L build | R build | L check | R check | L+R build | L+R check |
|---|---|---|---|---|---|---|---|
| ctl | 0 | 0.10 | 0.25 | 0.09 | 0.23 | 0.34 (-44%) | 0.32 (-44%) |
| base | 1000 | 0.20 | 0.41 | 0.19 | 0.39 | 0.61 | 0.58 |
| multi4 | 300 | 0.14 | 0.40 | 0.14 | 0.37 | 0.54 (-11%) | 0.50 (-13%) |
| multi8 | 200 | 0.13 | 0.40 | 0.13 | 0.38 | 0.54 (-12%) | 0.51 (-12%) |
| multi16 | 100 | 0.13 | 0.40 | 0.13 | 0.37 | 0.53 (-13%) | 0.50 (-14%) |
| multi32 | 100 | 0.14 | 0.40 | 0.14 | 0.37 | 0.54 (-12%) | 0.52 (-10%) |
| tup4 | 300+6 | 0.13 | 0.56 | 0.12 | 0.52 | 0.68 (+12%) | 0.65 (+12%) |
| tup8 | 200+10 | 0.12 | 0.58 | 0.12 | 0.53 | 0.71 (+16%) | 0.65 (+13%) |
| tup16 | 100+10 | 0.11 | 0.59 | 0.11 | 0.56 | 0.70 (+15%) | 0.67 (+16%) |
| nest8 | 200+8 | 0.12 | 0.58 | 0.12 | 0.55 | 0.71 (+16%) | 0.66 (+15%) |
| nest16 | 100+10 | 0.11 | 0.61 | 0.11 | 0.57 | 0.72 (+18%) | 0.68 (+18%) |

The impls are 53% of this Leaker crate and 41% of the Referrer crate (base minus `ctl`).

### Entries per context (E=1000): change of `cargo check` CPU against `base`

Base (L / R, s): n=1 0.24 / 0.47, n=2 0.21 / 0.42, n=4 0.20 / 0.41, n=10 0.19 / 0.39, n=50 0.18 / 0.37, n=1000 0.19 / 0.37.
`tup`/`nest` equal `base` at n=1 (single entries are written directly).

| variant | L n=1 | n=2 | n=4 | n=10 | n=50 | n=1000 | R n=1 | n=2 | n=4 | n=10 | n=50 | n=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multi4 | +25% | -13% | -31% | -30% | -31% | -34% | +2% | 0% | -6% | -4% | -3% | -5% |
| multi8 | +61% | 0% | -25% | -32% | -35% | -40% | +1% | -2% | -6% | -3% | -4% | -6% |
| multi16 | +125% | +30% | -10% | -35% | -36% | -43% | +2% | -2% | -5% | -4% | -3% | -7% |
| multi32 | +267% | +90% | +16% | -25% | -39% | -43% | +3% | -1% | -6% | -3% | -4% | -5% |
| tup8 | 0% | -25% | -35% | -39% | -41% | -45% | 0% | +33% | +32% | +39% | +42% | +41% |
| tup16 | 0% | -25% | -35% | -44% | -42% | -46% | 0% | +33% | +32% | +46% | +56% | +55% |
| tup32 | 0% | -25% | -35% | -44% | -40% | -44% | 0% | +33% | +32% | +46% | +77% | +81% |
| nest8 | 0% | -25% | -35% | -38% | -41% | -44% | 0% | +34% | +33% | +42% | +46% | +45% |
| nest16 | 0% | -25% | -35% | -43% | -42% | -46% | 0% | +34% | +33% | +48% | +61% | +59% |

The full matrix, including L+R, is `report/matrix-A-stable.txt`. A filler `type TypeJ = ()` costs
0.1-0.3 M instructions, so `multiK` loses on small contexts. Context complexity changes the Leaker gain
little (`report/B-stable-quiet.txt`): multi8 gives -24% (no generics) to -31% (bounds), and in the
Referrer -4% to -14% (lifetimes gain most). The tuples cost +23% (lifetimes) to +53% (no generics) there.

### Number of entries (10 per context, three generics)

| E | variant | L check | R check | L+R check |
|---|---|---|---|---|
| 200 | base | 0.08 | 0.10 | 0.18 |
| 200 | multi8 | 0.07 | 0.10 | 0.17 (-6%) |
| 200 | tup8 | 0.07 | 0.13 | 0.20 (+10%) |
| 1000 | base | 0.19 | 0.39 | 0.58 |
| 1000 | multi8 | 0.13 | 0.38 | 0.51 (-12%) |
| 1000 | tup8 | 0.12 | 0.53 | 0.65 (+13%) |
| 5000 | ctl | 0.26 | 1.10 | 1.36 |
| 5000 | base | 1.24 | 2.15 | 3.40 |
| 5000 | multi8 | 0.56 | 1.86 | 2.43 (-28%) |
| 5000 | multi16 | 0.47 | 1.83 | 2.30 (-32%) |
| 5000 | tup8 | 0.40 | 2.74 | 3.15 (-7%) |
| 5000 | nest16 | 0.35 | 2.88 | 3.23 (-5%) |
| 10000 | base | 3.57 | 4.92 | 8.57 |
| 10000 | multi8 | 1.26 | 3.71 | 4.99 (-42%) |
| 10000 | tup8 | 0.83 | 5.67 | 6.50 (-24%) |

The cost of the impls is superlinear: for the Leaker, base minus `ctl` is 0.10 s at E=1000, 0.98 s at 5000 and
3.1 s at 10000. See `report/C-stable.txt` and `report/S-stable.txt`.

### Corpus-like (context sizes from the histogram, mixed contexts, 6.5% unsized)

`report/D-stable-quiet.txt`. Change of `cargo check` CPU against `base`, L / R.

| E | base L / R | multi4 | multi8 | multi16 | multi32 | tup8 | tup16 | nest16 |
|---|---|---|---|---|---|---|---|---|
| 50 (median crate) | 0.06 / 0.05 | 0% / 0% | +2% / 0% | +5% / +2% | +11% / 0% | 0% / +11% | 0% / +11% | -2% / +9% |
| 200 (largest crate) | 0.08 / 0.10 | -1% / -1% | +4% / -1% | +18% / 0% | +43% / +1% | -3% / +17% | -1% / +17% | -5% / +18% |
| 1000 | 0.19 / 0.38 | -6% / -3% | +7% / -3% | +41% / -3% | +107% / -2% | -20% / +24% | -17% / +26% | -21% / +27% |

At E=50 the Repeater impls cost 0.006 s of the Leaker (`ctl`: 0.057 s) and 0.007 s of the Referrer. On a few real
property-test Leaker crates (76-100 entries) `-Ztime-passes` shows 66-77% of the time in macro
expansion (the proc macro itself) and 0.08-0.11 s in coherence. The 125 s per 60 cases of the property test
is not the impls.

### rustc versions

The cost depends on the compiler more than on the encoding.

`cargo check` CPU, L / R, E=2000, 10 per context (`report/versions.txt` has 23 versions):

| rustc | ctl | base | multi8 | tup8 |
|---|---|---|---|---|
| 1.56 | 0.33 / 0.70 | 9.98 / 5.31 | 0.80 / 1.55 | 0.78 / 2.30 |
| 1.60 | 0.32 / 0.60 | 7.23 / 4.13 | 0.66 / 1.31 | 0.64 / 1.97 |
| 1.63 | 0.29 / 0.62 | 0.71 / 1.35 | 0.38 / 1.23 | 0.36 / 1.89 |
| 1.70 | 0.23 / 0.50 | 0.59 / 1.13 | 0.31 / 1.05 | 0.30 / 1.59 |
| 1.80 | 0.15 / 0.44 | 0.45 / 0.92 | 0.21 / 0.83 | 0.20 / 1.24 |
| 1.99 | 0.13 / 0.44 | 0.40 / 0.78 | 0.22 / 0.73 | 0.18 / 1.06 |
| nightly 1.101 | 0.13 / 0.43 | 0.58 / 0.96 | 0.31 / 0.73 | 0.20 / 1.02 |

Up to 1.60 one blanket impl per entry is quadratic and costs seconds. It is fixed between 1.60 and 1.63.
The same configurations on more compilers (`report/toolchains.txt`), L / R and L+R against base:

| rustc | E=1000, 10 per ctx: multi8 | tup8 | E=5000, 10 per ctx: base L / R | multi8 | tup8 |
|---|---|---|---|---|---|
| 1.56 | -73% | -66% | 61.5 / 27.3 | -91% | -89% |
| 1.75 | -16% | +16% | 1.69 / 2.95 | -39% | -12% |
| 1.85 | -15% | +12% | 1.55 / 2.84 | -37% | -14% |
| 1.99 | -18% | +9% | 1.27 / 2.17 | -30% | -9% |
| nightly | -20% | -6% | 2.50 / 3.44 | -51% | -45% |

On corpus-like E=1000, 1.56 gives multi4 -30% and tup8 -6%; 1.99 gives -4% and +7%.

### Where the time goes (`-Ztime-passes`, wall s, E=5000, 10 per context)

`report/passes.txt`.

| crate, rustc | variant | total | type_check_crate (has coherence) | coherence_checking | MIR_borrow_checking |
|---|---|---|---|---|---|
| L, 1.99 | ctl | 0.25 | 0.08 | 0.08 | 0.00 |
| L, 1.99 | base | 1.42 | 0.79 | 0.72 | 0.02 |
| L, 1.99 | multi8 | 0.62 | 0.26 | 0.24 | 0.01 |
| L, 1.99 | tup8 | 0.44 | 0.16 | 0.14 | 0.00 |
| R, 1.99 | ctl | 1.41 | 0.32 | 0.25 | 0.25 |
| R, 1.99 | base | 2.78 | 1.19 | 0.68 | 0.46 |
| R, 1.99 | multi8 | 2.47 | 0.91 | 0.40 | 0.46 |
| R, 1.99 | tup8 | 3.63 | 1.60 | 0.66 | 0.67 |
| L, nightly | base | 2.54 | 2.05 | 2.02 | 0.02 |
| L, nightly | tup8 | 0.46 | 0.21 | 0.21 | 0.00 |

- **Leaker**: the cost of the impls is coherence checking (blanket impls of one trait are compared
  pairwise). With `ctl` it is 0.08 s, with 5000 impls 0.72 s. Grouping cuts the number of impls.
- **Referrer**: the extra time is type checking of the signatures and bodies that normalize the projection,
  and the comparison of the trait impl methods. `tup`/`nest` pay for a second selection through a big tuple.
- On 1.56 (E=1000): the base Leaker spends 1.39 s in coherence and 0.97 s in `wf_checking` of 2.52 s; the
  Referrer 1.03 s of 1.65 s in `wf_checking`.

### MSRV

All 21 variants (`base`, `ctl`, `multi4..32`, `tup4..32`, `tup*p`, `tup*r`, `nest4..32`, `nest*r`) pass
`cargo +1.56 check` for both crates, on two configurations (E=1000 with 10 per context; corpus-like E=200).
See `report/msrv.txt`. The 1.56 cargo cannot read the lockfile format of newer ones, so `measure.py` deletes
`Cargo.lock` first.

### Error messages

`report/errdemo.txt`, stable.

| scenario | base | multi | tup, nest |
|---|---|---|---|
| a leaked type is not well-formed in the context (Leaker) | E0277 on that type | same | same (the help suggestion prints the whole group impl) |
| the use lacks a bound of the context (Referrer) | one E0277 | one E0277 | two E0277 for one cause; the span is the whole nested projection (120+ characters) |
| type mismatch | same message | same | same (rustc normalizes first) |
| entry that does not exist | `(): Repeater<(T,), 1, 9>` is not satisfied | E0576 "cannot find associated type `Type7`" | the whole tuple type and the list of indexing impls |

### Other checks

- **Uses in the Referrer** (grid U, E=1000, 10 per context). The ordering is the same when only the fn
  signatures, only the trait impl or only the witnesses are written. Referrer `check`, billions of
  instructions (`ctl` / `base` / `multi8` / `tup8`): signatures 0.68 / 1.33 / 1.10 / 1.65, impl 0.71 / 1.44 / 1.21 / 1.86,
  witness 0.68 / 1.20 / 0.97 / 1.38.
- **One ID for all contexts** (grid I): about the same as one ID per context.
- **Unwrapped elements** (grid R, instruction counts). With 12% unsized entries, `tup8r`/`nest8r` are 6-9%
  cheaper in the Referrer than `tup8`/`nest8` and 9-17% dearer in the Leaker (the unsized entries are
  ungrouped). Without unsized entries the Referrer is 4-5% cheaper and the Leaker equal. The `*const` wrapper
  is not what makes the tuples slow.
- **Padding** (`tup16p`): fewer indexing impls (30 against 101 on corpus-like E=1000), but no faster: the
  Referrer is 0-6% slower.
- **Instruction counts** (`ginstr`) give the same ordering of the variants on stable. The ratios differ: the
  pairwise scan runs at a higher instructions-per-second rate than the rest, so instructions overstate
  its share.

## Conclusion

**Does grouping pay?** It depends on the size of the crate and on the compiler.

- **Corpus scale** (a crate has 49 entries at the median, 197 at most; 70% of the contexts hold one entry):
  no. On rustc 1.63 and later the Leaker changes by -5% to +4% with `multi4`, `multi8`, `tup` and `nest`
  (`multi16`/`multi32` lose 5-43%), and the tuple variants make the Referrer 9-18% slower. Even on 1.56 it is
  flat up to E=200 (within 7%).
- **E=1000 and more** on 1.63+: with 10 entries per context the Leaker gains 30-45% at E=1000 and 55-72% at
  E=5000 (stable); on corpus-like contexts 6% (`multi4`) to 21% (`tup`/`nest`) at E=1000. The Referrer gains
  3-15% with `multi` and loses 20-90% with `tup`/`nest`.
- **rustc 1.56-1.60**: the Leaker is 6-46x faster and the Referrer 2-6x, at E=1000-5000, with any variant.

**From how many entries per context?** Measured points (Leaker `check`, E=1000): `multi4` pays at n=2 (-13%);
`multi8` breaks even at n=2 and pays at n=4 (-25%); `multi16` pays at n=4 (-10%); `multi32` loses at n=4
(+16%) and pays at n=10 (-25%). `tup` and `nest` pay in the Leaker from n=2 (-25%) but not in the Referrer.
n=3 and n=5..9 were not measured.

**Which variant?**

- `multi4` to `multi8` is the best all-round choice: it does not slow the Referrer, gains 3-15% there,
  and gains about 30% in the Leaker from 4 entries per context. It loses on single-entry contexts
  (+25% to +61% in the Leaker), which are 70% of the corpus. The corpus-like runs are neutral for `multi4`
  (-6% at E=1000) and negative for `multi8` and up.
- K above 8 only loses on realistic distributions (`multi32` on corpus-like E=1000: Leaker +107%).
- `tup`/`nest` have the best Leaker numbers (-35% to -46%) and cost nothing for single entries, but the
  second projection makes the Referrer 20-90% slower on stable (+27% at E=5000). They win overall only at
  E >= 5000 on stable, on nightly, or on 1.56-1.60.
- `nestK` needs K indexing impls where `tupK` needs 35 (K=8), 101 (K=16) and 229 (K=32) on corpus-like
  E=1000. It is 2-5% slower than `tupK` in the Referrer. Padding does not help.

**What it would mean for the library**

- **Repeater definition**: the user writes exactly K `type TypeJ: ?Sized;`. K must be fixed by the
  library and part of `FORMAT_VERSION`; a Leaker and a Referrer with different K cannot interoperate.
  `repeater_definition` takes K.
- **API**: `visitor(context, |ty, ix|)` and `repeater_impl_with` would hand out `(args, id, group, slot)` instead
  of `ix`. A helper has to build the use-site path, or each macro author reimplements the slot logic.
- **Required-only impls**: a trusting Referrer that writes the impls of `required()` entries only has to
  fill the optional slots with `()`.
- **Unsized entries**: no issue for `multi`, all types are `?Sized`. For `tup`/`nest` every element needs
  `*const`, because sizedness is not known syntactically.
- **Indexing impls** must be emitted once per crate, not per macro call (a second copy conflicts),
  for example by `define_repeater!`. Per-item reserved IDs would cost about 10 impls per item and lose the gain.
- **Failure granularity**: one wrong entry fails the whole group impl, in `multi` and in `tup` alike.
- **Error messages**: `multi` keeps rustc's messages as good as today's. `tup`/`nest` make Referrer-side errors
  longer and duplicated.

**Recommendation.** For rustc 1.63 and later and realistic crates, grouping does not justify a format
change. If it is wanted anyway (very large items, or users on 1.56-1.60), implement `multi4` or `multi8`
and measure it on a real downstream crate first. If only the Leaker crate matters and items are huge
(E >= 5000), the nested pairs are the cheapest.

## Caveats

- The Referrer crates are synthetic: the bodies are `{ x }`, so the Repeater's share of a real Referrer crate
  (41-49% of CPU here) is probably an upper bound.
- One machine, one CPU model. Nightly 1.101 is slower than 1.99 on the baseline at E=5000.
- The corpus is the property test's generator, not real code. It has odd shapes, and the largest
  items are small. A real crate with a 5000-entry item was not available.
- Leaker and Referrer are measured as separate crates. The L+R columns weight them equally; in practice the
  Leaker crate is built once, and each Referrer crate by someone else.
- The cost of running the proc macro (the analysis) is not in these numbers: no proc macro is used here.

## Files

| path | content |
|---|---|
| `report/*.txt` | the tables behind this README (CPU / wall and instructions), by grid and toolchain |
| `corpus/histogram.txt`, `corpus/sizes.json` | the histogram and the sizes that `--dist corpus` draws from |
| `prop-hist-logging.patch` | logging for the corpus run, against error-recovery 1de07a8 |

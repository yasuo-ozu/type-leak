# Repeater grouping benchmark

Does grouping the entries of one context into fewer Repeater impls reduce compile time? And with how
many associated types per impl?

Today the Leaker emits one impl per entry:

```rust
impl<'x, Y: Bound, __S: ?Sized> Repeater<(&'x (), Y), ID, NTH> for __S { type Type = Leaked<'x, Y>; }
```

A big item makes many such impls, all blanket impls of one trait. This directory generates Leaker and
Referrer crates for the encodings below and measures clean `cargo build` and `cargo check` of both.
Nothing here is part of the published package (`bench` is in `exclude`). No library code is changed.

Grouping is possible only for entries of the same context: same generics and bounds, so the same impl
header and the same first argument.

## Variants

| variant | impls | use site |
|---|---|---|
| `base` | one impl per entry, `type Type` | `<X as Repeater<Args, ID, NTH>>::Type` |
| `multiK` | the Repeater has `Type0..Type{K-1}` (all `?Sized`); one impl carries up to K entries of one context | `<X as Repeater<Args, ID, G>>::TypeJ` |
| `ctl` | control: no Repeater. The Leaker has no impls, the Referrer writes the original types | the original type |
| `multiKd` | nightly-only experiment, see "Associated type defaults" | as `multiK` |

```rust
pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> { type Type0: ?Sized; type Type1: ?Sized; /* .. */ }
impl<'x, Y: Bound, __S: ?Sized> Repeater<(&'x (), Y), ID, G> for __S { type Type0 = A<Y>; type Type1 = B<'x>; /* unused: = () */ }
```

- Associated type defaults are unstable, so an impl must define all K types. The unused ones ("fillers") are `()`.
- Tuple encodings (`type Type = (T0, T1, ..)` plus indexing impls) were measured earlier and dropped:
  the second projection made the Referrer crate 20-90% slower on stable.
- Entries of different contexts are never grouped. Each context has its own ID, as if it were its own item.
  With one ID for all contexts (grid `I`) the results are about the same.
- An impl that defines only the types it needs is **not possible on stable**. A trait item without a default
  must be defined by every impl. Under the constraint that the Repeater is the only shared item there is no
  way around it: no second trait, no supertrait, and a GAT indexed by a const (`type Type<const J: usize>`)
  would need a type-level function of `J`, and GATs need 1.65 anyway. `multiKd` measures what defaults
  would bring, on nightly only.

## What is generated

`gen.py` writes a workspace with a `leaker` and a `referrer` crate. The entries depend only on
`(E, C or sizes, cx, seed, unsized share)`, never on the variant.

- **Contexts** (`--cx`): `none`; `one` (`T`); `three` (`T: Clone + Into<U>, U: Clone, V: Default`);
  `lt` (`'a, 'b: 'a, T: 'a + Clone`, encoded `(&'a (), &'b (), T, &'a &'b (), &'a T)`); `mixed` (42% none,
  14% one, 19% three, 25% lt, the shares of the corpus).
- **Entry types**: paths into the Leaker (`crate::m::A2_3<T, N5>`), `&'a` paths, tuples, nested paths, and
  12% unsized ones (`dyn D3<T> + 'static`, `[T]` and `str` through type aliases). All `TypeJ` are `?Sized`.
- **Leaker**: the Repeater, the module `m` of the types, one trait `Tr_k<G>` per context whose methods
  take and return the entries, and the impls.
- **Referrer**: each entry is used three times, forcing the normalization:
  - a fn `f<G>(x: P) -> P { x }`;
  - a method of `impl<G> Tr_k<G> for Host_k` with the same signature as the Leaker's trait;
  - a witness `w<G>(x: P) -> Orig { x }`, which checks that `P` is the original type.

  `P` is `<() as Repeater<..>>::Type..`. Unsized entries are behind `&'q`.

## Run it

Prerequisites: Linux, `python3` (written with 3.14; standard library only), `rustup` with the toolchains you use
(`stable` is the default; `nightly`, `1.63.0`, `1.60.0`, `1.56` for the other runs). The generated crates have
no dependencies, so everything runs offline.

Optional, for stable numbers:

- `BENCH_CPUS=0-11` pins everything to the given CPUs. On a hybrid Intel CPU pin to the P-cores
  (`lscpu -e`): a process that moves to an E-core takes up to 3x the CPU time.
- Instruction counts use `perf_event_open` (`/proc/sys/kernel/perf_event_paranoid <= 2`; user space only). If
  it is not available the `ginstr` fields are null and the instruction tables are skipped.

```sh
cd bench/repeater-grouping
export BENCH_CPUS=0-11                     # optional

# smoke test, a few seconds: E=200, ctl/base/multi4/multi8
python3 bench.py --grid Q --reps 1 --tc stable
python3 report.py results/Q-stable.jsonl

# one crate pair by hand
python3 gen.py --out /tmp/x --variant multi4 --E 1000 --C 100 --cx three
python3 measure.py /tmp/x --tc stable

# the runs of this README
./run_full.sh                              # about 2 h on an idle machine; ends with python3 make_report.py
```

The machine should be idle (the load average is recorded in each sample). Each configuration is measured
by interleaving its variants (shuffled per repetition), 5 repetitions (3 for the version scan). The median is
reported. The spread (max/min over the repetitions) was 1.01-1.16 in the main tables.

| file | what it does |
|---|---|
| `gen.py` | generator: `--variant base\|ctl\|multiK\|multiKd --E --C --cx --dist uniform\|skew\|corpus --uses sig,impl,wit --ids per-ctx\|shared --unsized` |
| `bench.py` | grids N D B W X S U I M V Z F Q (list at the top of the file): generates, measures, appends one JSON line per sample to `results/<grid>-<tc>.jsonl`; resumable |
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
Times are CPU seconds (user+sys of cargo and its children), median of 5. Wall time is within 1% of CPU
(rustc's front end is single-threaded) and is in `report/`. "L" is the Leaker crate, "R" the Referrer crate,
"build" and "check" are `cargo build` and `cargo check` (debug, `CARGO_INCREMENTAL=0`).
A "change" is against `base`, so negative is faster.

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
  header and arguments gives 5663 groups, mean 1.81, max 40.
- **Per crate**: E median 49, mean 57, max 197; 15 items and 32 contexts (median).
- **Per item** (one macro call): median 2 entries and 2 contexts; mean 3.6, p90 7, p99 21, max 46.
  Items with 9 or more entries still have 66% single-entry contexts.
- **Shapes**: no generics 42%, plain generics 14%, bounds 19%, lifetimes 15%, lifetimes and bounds 10.5%.
  About 6.5% of the entries are unsized.

The generated generics are the property test's, so this is a corpus of odd shapes, not of real code.
The "corpus-like" runs below draw their context sizes from this histogram.

### E=1000, 10 entries per context, three generics with bounds

`report/N-stable.txt` (config `N-n10`). The last two columns are L+R.

| variant | impls | L build | R build | L check | R check | L+R build | L+R check |
|---|---|---|---|---|---|---|---|
| ctl | 0 | 0.09 | 0.25 | 0.09 | 0.23 | 0.34 (-43%) | 0.32 (-45%) |
| base | 1000 | 0.19 | 0.41 | 0.19 | 0.39 | 0.61 | 0.58 |
| multi2 | 500 | 0.15 | 0.40 | 0.15 | 0.38 | 0.55 (-9%) | 0.53 (-9%) |
| multi3 | 400 | 0.14 | 0.40 | 0.14 | 0.38 | 0.54 (-10%) | 0.52 (-10%) |
| multi4 | 300 | 0.14 | 0.40 | 0.14 | 0.38 | 0.54 (-11%) | 0.51 (-11%) |
| multi5 | 200 | 0.13 | 0.40 | 0.13 | 0.37 | 0.53 (-13%) | 0.50 (-14%) |
| multi6 | 200 | 0.13 | 0.40 | 0.13 | 0.37 | 0.53 (-13%) | 0.50 (-14%) |
| multi8 | 200 | 0.14 | 0.40 | 0.13 | 0.37 | 0.54 (-11%) | 0.51 (-12%) |
| multi12 | 100 | 0.12 | 0.40 | 0.12 | 0.37 | 0.53 (-13%) | 0.49 (-15%) |
| multi16 | 100 | 0.13 | 0.40 | 0.13 | 0.37 | 0.53 (-13%) | 0.50 (-14%) |
| multi32 | 100 | 0.15 | 0.40 | 0.15 | 0.37 | 0.55 (-9%) | 0.52 (-10%) |

The impls are 53% of this Leaker crate and 41% of the Referrer crate (`base` minus `ctl`).
Grouping does not help the Referrer much: R is 3-8% faster for every K.

### Entries per context n (E=1000, three generics)

E is `n * round(1000 / n)`, within 1% of 1000. Base (L / R `check` seconds): n=1 0.24 / 0.48, n=2 0.21 / 0.43,
n=4 0.20 / 0.42, n=10 0.19 / 0.39, n=50 0.19 / 0.38.

Leaker, change of `check`:

| K | n=1 | n=2 | n=3 | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 | n=10 | n=12 | n=16 | n=24 | n=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | +6% | -22% | -12% | -23% | -15% | -22% | -19% | -22% | -21% | -21% | -22% | -24% | -24% | -24% |
| 3 | +15% | -18% | -29% | -21% | -23% | -28% | -24% | -27% | -31% | -24% | -29% | -27% | -31% | -29% |
| 4 | +23% | -14% | -26% | -34% | -21% | -28% | -32% | -33% | -28% | -28% | -33% | -34% | -34% | -32% |
| 5 | +33% | -11% | -25% | -31% | -34% | -24% | -29% | -31% | -34% | -34% | -31% | -32% | -35% | -35% |
| 6 | +42% | -7% | -20% | -29% | -33% | -35% | -27% | -30% | -33% | -32% | -36% | -35% | -35% | -36% |
| 8 | +59% | +1% | -16% | -26% | -30% | -34% | -36% | -38% | -30% | -30% | -33% | -38% | -39% | -37% |
| 12 | +90% | +15% | -7% | -20% | -24% | -28% | -33% | -34% | -36% | -37% | -39% | -34% | -41% | -38% |
| 16 | +125% | +28% | +2% | -14% | -20% | -25% | -28% | -31% | -34% | -34% | -37% | -41% | -38% | -38% |
| 32 | +262% | +93% | +39% | +14% | +2% | -5% | -14% | -18% | -23% | -23% | -29% | -35% | -40% | -40% |

L+R, change of `check`:

| K | n=1 | n=2 | n=3 | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 | n=10 | n=12 | n=16 | n=24 | n=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | +2% | -9% | -6% | -12% | -7% | -9% | -8% | -9% | -9% | -9% | -10% | -10% | -9% | -10% |
| 3 | +4% | -7% | -12% | -11% | -10% | -11% | -10% | -11% | -13% | -10% | -12% | -12% | -12% | -13% |
| 4 | +7% | -6% | -11% | -16% | -9% | -11% | -13% | -13% | -12% | -11% | -13% | -15% | -13% | -14% |
| 5 | +11% | -5% | -10% | -15% | -14% | -10% | -12% | -13% | -14% | -13% | -13% | -13% | -14% | -15% |
| 6 | +14% | -3% | -9% | -15% | -14% | -14% | -11% | -13% | -14% | -14% | -14% | -15% | -15% | -15% |
| 8 | +20% | -1% | -8% | -14% | -13% | -14% | -15% | -15% | -12% | -12% | -12% | -16% | -15% | -16% |
| 12 | +30% | +4% | -4% | -11% | -11% | -11% | -14% | -15% | -15% | -15% | -16% | -15% | -16% | -16% |
| 16 | +43% | +8% | -2% | -9% | -9% | -11% | -12% | -13% | -15% | -13% | -15% | -17% | -15% | -16% |
| 32 | +89% | +30% | +11% | 0% | -2% | -3% | -7% | -9% | -10% | -10% | -12% | -15% | -16% | -16% |

- The Referrer (R `check`) changes by -1% to -8% for every K and every n >= 2 (mean -3.8%), and by -1% to +1% at n=1.
- `build` is within 1-2 points of `check` everywhere (`report/nsweep.txt` has all six tables).
- The curves are sawtooth in n: a context of n entries uses ceil(n/K) impls, so the last, partly filled impl
  costs its fillers (K=3: n=3 -29%, n=4 -21%; K=4: n=4 -34%, n=5 -21%).

**Break-even n.** The smallest measured n from which `multiK` is never slower than `base` by more than 1%,
and in parentheses where the interpolated curve crosses 0. CPU time; the instruction counts agree within one
step (`report/nsweep.txt`; multi8 n>=3 for L, multi32 n>=5 for L and n>=4 for L+R).

| K | Leaker `check` | L+R `check` | Leaker `build` | L+R `build` |
|---|---|---|---|---|
| 2 | n>=2 (1.2) | n>=2 (1.1) | n>=2 (1.3) | n>=2 (1.2) |
| 3 | n>=2 (1.5) | n>=2 (1.4) | n>=2 (1.5) | n>=2 (1.5) |
| 4 | n>=2 (1.6) | n>=2 (1.5) | n>=2 (1.7) | n>=2 (1.6) |
| 5 | n>=2 (1.8) | n>=2 (1.7) | n>=2 (1.8) | n>=2 (1.7) |
| 6 | n>=2 (1.9) | n>=2 (1.8) | n>=2 (1.9) | n>=2 (1.8) |
| 8 | n>=2 (2.0) | n>=2 (1.9) | n>=2 (2.0) | n>=2 (2.0) |
| 12 | n>=3 (2.7) | n>=3 (2.5) | n>=3 (2.7) | n>=3 (2.5) |
| 16 | n>=4 (3.2) | n>=3 (2.8) | n>=3 (3.0) | n>=3 (2.8) |
| 32 | n>=6 (5.2) | n>=4 (4.0) | n>=6 (5.3) | n>=5 (4.9) |

At n=1 every K loses: L +6% (K=2) to +262% (K=32), L+R +2% to +89%. A rule of thumb from the filler cost
below: the break-even is near n = 1 + 0.2 (K-1), that is 1.6, 2.4, 4.0 and 7.2 for K = 4, 8, 16, 32. The
measured crossings are 1.6, 2.0, 3.2 and 5.2, so the rule is right for small K and pessimistic for large K.

The best K at a given n (smallest L+R `check`) is K=2 at n=2, K=3 at n=3, K=4 at n=4, K=5 at n=5, K=8 for
n = 6..8, K=12 for n = 9..12 and 24..50, K=16 at n=16. K=4 is within 3.6 points of the best L+R at every n >= 2
except n=5 (5.2 points); in the Leaker alone it is up to 13 points behind (n=5: -21% against -34%).

### Cost of a filler

1000 singleton contexts, so every impl carries K-1 fillers `type TypeJ = ();`. Leaker `check` CPU in ms
(`report/fillers.txt`; the Referrer does not change).

| context | ctl | base | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=12 | K=16 | K=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 99 | 171 | 170 | 188 | 207 | 223 | 242 | 258 | 277 | 297 | 372 | 437 | 732 |
| three | 135 | 239 | 240 | 259 | 283 | 299 | 325 | 342 | 360 | 382 | 469 | 539 | 879 |
| lt | 123 | 224 | 224 | 244 | 265 | 288 | 308 | 336 | 352 | 375 | 462 | 549 | 886 |

| context | per filler | per entry impl (`base` - `ctl`) | filler / impl |
|---|---|---|---|
| none | 18 us, 0.26 M instructions | 72 us, 0.87 M | 0.25 (0.30 by instructions) |
| three | 20 us, 0.29 M | 104 us, 1.14 M | 0.19 (0.25) |
| lt | 22 us, 0.30 M | 101 us, 1.10 M | 0.22 (0.27) |

The cost is linear in K (least squares over K=1..8). A filler costs 0.2-0.3 of an entry impl, and about the
same whatever the context. `multi1` equals `base`.

### Corpus-like crates

Context sizes drawn from the corpus histogram (70% singletons), mixed contexts, 6.5% unsized. Three draws
(seeds 1-3) per size; mean change of `check` over the draws. The ranges over the draws are small for K <= 8
(at most 6 points in L) and larger for K >= 12; `report/corpus.txt` has them, `build` and the L / R split.

| E (base L / R) | change | K=2 | K=3 | K=4 | K=5 | K=6 | K=8 | K=12 | K=16 | K=32 |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 (0.061 / 0.052) | L | +1% | 0% | -1% | +1% | +2% | +2% | +3% | +4% | +10% |
| | L+R | 0% | 0% | -1% | 0% | +1% | +1% | +1% | +2% | +6% |
| 200 (0.079 / 0.098) | L | -3% | -3% | -2% | 0% | 0% | +4% | +9% | +16% | +37% |
| | L+R | -1% | -2% | -2% | -1% | -1% | +1% | +3% | +7% | +17% |
| 1000 (0.193 / 0.388) | L | -8% | -8% | -5% | -2% | +2% | +10% | +27% | +45% | +113% |
| | L+R | -4% | -5% | -4% | -4% | -2% | +1% | +7% | +13% | +35% |

For the median crate (E=50) no K changes anything: the Repeater impls are 0.003 s of the Leaker (`ctl` 0.058 s)
and 0.006 s of the Referrer. At E=200 they are 0.014 s and 0.025 s, at E=1000 0.09 s and 0.16 s. Even at E=200 the
best case is -2%. On a few real property-test Leaker crates (76-100 entries) `-Ztime-passes` shows 66-77% of the
time in macro expansion (the proc macro itself) and 0.08-0.11 s in coherence. The 125 s per 60 cases of the
property test is not the impls.

### Context complexity

E=1000, 10 entries per context. Change of `check` in percent, L / R / L+R. Base L / R in brackets.

| context | K=2 | K=3 | K=4 | K=5 | K=6 | K=8 | K=12 | K=16 | K=32 |
|---|---|---|---|---|---|---|---|---|---|
| none (0.15 / 0.28) | -20 / -4 / -10 | -21 / -4 / -10 | -23 / -5 / -12 | -29 / -6 / -14 | -28 / -5 / -13 | -25 / -4 / -12 | -30 / -7 / -15 | -28 / -6 / -14 | -18 / -6 / -10 |
| one (0.16 / 0.30) | -19 / -3 / -8 | -21 / -4 / -10 | -24 / -4 / -11 | -30 / -4 / -13 | -28 / -5 / -13 | -26 / -5 / -12 | -32 / -5 / -15 | -30 / -5 / -14 | -20 / -4 / -9 |
| three (0.19 / 0.39) | -22 / -3 / -9 | -25 / -3 / -10 | -30 / -4 / -12 | -34 / -4 / -14 | -33 / -3 / -13 | -31 / -4 / -13 | -37 / -2 / -14 | -34 / -5 / -14 | -24 / -3 / -10 |
| lt (0.19 / 0.47) | -22 / -10 / -13 | -24 / -11 / -15 | -29 / -13 / -18 | -34 / -13 / -19 | -32 / -14 / -19 | -29 / -15 / -19 | -37 / -15 / -21 | -34 / -15 / -20 | -23 / -16 / -18 |

The complexity changes the size of the gain, not the best K. Lifetime contexts gain most in the Referrer.

### Number of entries (10 per context)

`check` CPU, L / R seconds, and the change of L+R. `report/scaling.txt`.

| E | ctl | base | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---|---|---|---|---|---|
| 100 | 0.06 / 0.06 | 0.07 / 0.07 | 0.06 / 0.07 (-3%) | 0.06 / 0.07 (-2%) | 0.06 / 0.07 (-2%) | 0.06 / 0.07 (-3%) | 0.06 / 0.07 (-2%) |
| 200 | 0.06 / 0.07 | 0.08 / 0.10 | 0.07 / 0.10 (-3%) | 0.07 / 0.10 (-5%) | 0.07 / 0.10 (-6%) | 0.07 / 0.10 (-6%) | 0.07 / 0.10 (-4%) |
| 500 | 0.07 / 0.13 | 0.12 / 0.20 | 0.10 / 0.20 (-6%) | 0.10 / 0.20 (-8%) | 0.09 / 0.20 (-10%) | 0.09 / 0.20 (-11%) | 0.10 / 0.20 (-8%) |
| 1000 | 0.10 / 0.23 | 0.19 / 0.39 | 0.15 / 0.38 (-9%) | 0.14 / 0.38 (-12%) | 0.14 / 0.38 (-12%) | 0.12 / 0.37 (-15%) | 0.14 / 0.38 (-11%) |
| 2000 | 0.13 / 0.45 | 0.39 / 0.80 | 0.27 / 0.75 (-14%) | 0.23 / 0.74 (-18%) | 0.22 / 0.74 (-19%) | 0.20 / 0.73 (-21%) | 0.25 / 0.74 (-17%) |
| 5000 | 0.26 / 1.12 | 1.29 / 2.19 | 0.74 / 1.95 (-23%) | 0.60 / 1.88 (-29%) | 0.57 / 1.86 (-30%) | 0.48 / 1.88 (-32%) | 0.63 / 1.88 (-28%) |
| 10000 | 0.49 / 2.24 | 3.69 / 5.10 | 1.89 / 4.09 (-32%) | 1.41 / 3.88 (-40%) | 1.29 / 3.82 (-42%) | 1.03 / 3.73 (-46%) | 1.39 / 3.79 (-41%) |

The cost of the impls is superlinear in E (Leaker, `base` minus `ctl`: 0.09 s at E=1000, 1.03 s at 5000, 3.2 s at
10000), so the gain grows with E. With 10 entries per context K=16 is the best at every E >= 1000.

### Other compilers

Change of `check` CPU against `base`, L / R / L+R in percent. Base L / R seconds in the header.
`report/toolchains.txt` has `build` too.

**E=1000, 4 entries per context**

| K | 1.56 (2.85 / 1.70) | 1.60 (2.08 / 1.38) | 1.63 (0.41 / 0.68) | 1.99 (0.20 / 0.40) |
|---|---|---|---|---|
| 2 | -66 / -43 / -57 | -63 / -40 / -54 | -19 / -3 / -9 | -22 / -3 / -9 |
| 3 | -65 / -43 / -57 | -63 / -40 / -54 | -18 / -3 / -9 | -18 / -3 / -8 |
| 4 | -83 / -53 / -72 | -79 / -50 / -68 | -27 / -4 / -13 | -32 / -3 / -12 |
| 5 | -83 / -53 / -72 | -79 / -50 / -67 | -26 / -4 / -13 | -31 / -3 / -12 |
| 6 | -83 / -54 / -72 | -79 / -50 / -67 | -26 / -3 / -12 | -29 / -4 / -12 |
| 8 | -82 / -53 / -71 | -79 / -50 / -67 | -25 / -3 / -11 | -24 / -2 / -10 |
| 12 | -82 / -54 / -72 | -78 / -51 / -67 | -23 / -5 / -12 | -19 / -4 / -9 |
| 16 | -81 / -53 / -71 | -78 / -51 / -67 | -20 / -4 / -10 | -12 / -2 / -6 |
| 32 | -80 / -53 / -70 | -76 / -50 / -65 | -11 / -5 / -7 | +14 / -2 / +3 |

**E=1000, 10 entries per context**

| K | 1.56 (2.85 / 1.66) | 1.60 (2.14 / 1.36) | 1.63 (0.40 / 0.65) | 1.99 (0.19 / 0.39) |
|---|---|---|---|---|
| 2 | -66 / -43 / -58 | -64 / -42 / -55 | -20 / -3 / -9 | -22 / -2 / -9 |
| 3 | -74 / -48 / -65 | -72 / -47 / -62 | -21 / -3 / -10 | -25 / -2 / -10 |
| 4 | -81 / -53 / -70 | -77 / -49 / -66 | -25 / -2 / -11 | -29 / -3 / -12 |
| 5 | -85 / -55 / -74 | -82 / -52 / -71 | -28 / -2 / -12 | -35 / -4 / -14 |
| 6 | -85 / -55 / -74 | -82 / -52 / -71 | -28 / -3 / -12 | -33 / -4 / -13 |
| 8 | -85 / -54 / -74 | -82 / -52 / -70 | -27 / -2 / -12 | -30 / -4 / -13 |
| 12 | -88 / -56 / -76 | -85 / -54 / -73 | -31 / -2 / -13 | -36 / -5 / -15 |
| 16 | -88 / -56 / -76 | -85 / -55 / -73 | -30 / -4 / -14 | -35 / -4 / -14 |
| 32 | -87 / -57 / -76 | -84 / -53 / -72 | -27 / -2 / -11 | -25 / -5 / -12 |

**Corpus-like E=1000 (one draw)**

| K | 1.56 (0.93 / 1.21) | 1.60 (0.79 / 1.00) | 1.63 (0.40 / 0.66) | 1.99 (0.19 / 0.39) |
|---|---|---|---|---|
| 2 | -30 / -16 / -22 | -30 / -17 / -23 | -11 / -4 / -7 | -8 / -4 / -6 |
| 3 | -38 / -20 / -28 | -35 / -19 / -26 | -12 / -5 / -8 | -10 / -4 / -6 |
| 4 | -40 / -21 / -29 | -38 / -21 / -28 | -12 / -2 / -6 | -7 / -4 / -5 |
| 5 | -41 / -23 / -31 | -39 / -22 / -30 | -12 / -3 / -6 | -5 / -5 / -5 |
| 6 | -42 / -22 / -31 | -39 / -23 / -30 | -11 / -5 / -7 | -2 / -5 / -4 |
| 8 | -41 / -22 / -30 | -39 / -23 / -30 | -10 / -5 / -7 | +7 / -4 / -1 |
| 12 | -40 / -25 / -31 | -36 / -22 / -28 | -5 / -4 / -4 | +22 / -5 / +4 |
| 16 | -37 / -24 / -30 | -35 / -22 / -28 | -1 / -4 / -3 | +39 / -5 / +10 |
| 32 | -27 / -22 / -24 | -23 / -22 / -23 | +19 / -4 / +5 | +104 / -4 / +32 |

Best K by L+R: n=4: 6 (1.56), 4 (1.60), 4 (1.63), 4 (1.99); n=10: 12, 16, 16, 12; corpus-like: 12, 6, 3, 2.

Before 1.63 the one-impl-per-entry Leaker is quadratic and costs seconds. A scan over 23 versions
(E=2000, 10 per context, `check` L / R seconds; `report/versions.txt`):

| rustc | ctl | base | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|
| 1.56 | 0.33 / 0.71 | 10.35 / 5.41 | 1.32 / 1.81 | 0.81 / 1.58 | 0.51 / 1.43 |
| 1.60 | 0.32 / 0.60 | 7.30 / 4.21 | 1.04 / 1.48 | 0.68 / 1.34 | 0.46 / 1.25 |
| 1.63 | 0.29 / 0.63 | 0.70 / 1.39 | 0.41 / 1.28 | 0.39 / 1.25 | 0.36 / 1.27 |
| 1.70 | 0.24 / 0.51 | 0.58 / 1.15 | 0.34 / 1.05 | 0.32 / 1.06 | 0.30 / 1.07 |
| 1.80 | 0.15 / 0.46 | 0.44 / 0.94 | 0.23 / 0.86 | 0.22 / 0.84 | 0.20 / 0.86 |
| 1.99 | 0.13 / 0.45 | 0.39 / 0.79 | 0.23 / 0.75 | 0.22 / 0.75 | 0.20 / 0.74 |
| nightly 1.101 | 0.13 / 0.44 | 0.61 / 0.96 | 0.32 / 0.77 | 0.31 / 0.73 | 0.26 / 0.71 |

### Associated type defaults (nightly only)

`multiKd` is `multiK` with `#![feature(associated_type_defaults)]`: the trait gives every type the default `()`
and an impl defines only the types it uses. It cannot be used on stable; it shows what defaults would bring.
Nightly 1.101, change of `check`, L / R in percent, against `base` (`report/defaults.txt`):

| config (base L / R) | K=4 | K=8 | K=16 | K=32 | K=4 with defaults | K=8 | K=16 | K=32 |
|---|---|---|---|---|---|---|---|---|
| n=1 (0.30 / 0.51) | +67 / 0 | +163 / 0 | +346 / 0 | +716 / 0 | +1 / -1 | +1 / 0 | -1 / 0 | +2 / -2 |
| n=2 (0.28 / 0.46) | -11 / -8 | +22 / -7 | +91 / -7 | +227 / -8 | -28 / -8 | -27 / -8 | -27 / -10 | -27 / -9 |
| n=4 (0.26 / 0.45) | -40 / -14 | -24 / -15 | 0 / -14 | +55 / -13 | -40 / -15 | -39 / -14 | -39 / -15 | -39 / -14 |
| n=10 (0.25 / 0.43) | -34 / -15 | -34 / -14 | -40 / -15 | -23 / -16 | -36 / -14 | -42 / -14 | -46 / -17 | -46 / -17 |
| corpus-like (0.23 / 0.41) | 0 / -6 | +27 / -7 | +92 / -7 | +216 / -6 | -20 / -6 | -23 / -8 | -23 / -8 | -22 / -8 |

With defaults the filler cost is gone: singletons cost +1% whatever K is. Corpus-like crates gain 20-23% in the
Leaker, against 0% (K=4) and a loss (larger K) without defaults.

### Where the time goes (`-Ztime-passes`, wall s)

`report/passes.txt`. E=5000, 10 per context, rustc 1.99:

| crate | variant | total | type_check_crate (has coherence) | coherence_checking | MIR_borrow_checking |
|---|---|---|---|---|---|
| L | ctl | 0.25 | 0.09 | 0.09 | 0.00 |
| L | base | 1.47 | 0.83 | 0.76 | 0.02 |
| L | multi4 | 0.68 | 0.29 | 0.27 | 0.01 |
| L | multi8 | 0.64 | 0.27 | 0.25 | 0.01 |
| L | multi16 | 0.54 | 0.21 | 0.20 | 0.00 |
| R | ctl | 1.44 | 0.32 | 0.25 | 0.25 |
| R | base | 2.87 | 1.25 | 0.72 | 0.48 |
| R | multi4 | 2.51 | 0.93 | 0.42 | 0.46 |
| R | multi8 | 2.48 | 0.91 | 0.41 | 0.46 |
| R | multi16 | 2.48 | 0.90 | 0.40 | 0.46 |

- **Leaker**: the cost of the impls is coherence checking (blanket impls of one trait are compared
  pairwise): 0.09 s with `ctl`, 0.76 s with 5000 impls. Grouping cuts the number of impls.
- **Referrer**: the extra time is type checking of the signatures and bodies that normalize the projection, and
  the comparison of the trait impl methods.
- On 1.56 (E=1000, 10 per context) the `base` Leaker spends 1.30 s in coherence and 0.91 s in `wf_checking`
  of 2.35 s; multi4 takes 0.31 s in all, multi16 0.13 s. The `base` Referrer spends 0.98 s of 1.56 s in `wf_checking`.

### MSRV

`ctl`, `base` and `multi1..8, 12, 16, 32` (13 variants) pass `cargo +1.56 check` for both crates, on two
configurations (E=1000 with 10 per context; corpus-like E=200). See `report/msrv.txt`. The 1.56 cargo cannot
read the lockfile format of newer ones, so `measure.py` deletes `Cargo.lock` first.

### Error messages

`report/errdemo.txt`, stable, for `base`, `multi4`, `multi8`.

| scenario | base | multiK |
|---|---|---|
| a leaked type is not well-formed in the context (Leaker) | E0277 on that type | E0277 on that `TypeJ`; the help suggestion prints the whole impl |
| the use lacks a bound of the context (Referrer) | one E0277 | one E0277 |
| type mismatch | same message | same |
| entry that does not exist | `(): Repeater<(T,), 1, 9>` is not satisfied | E0576 "cannot find associated type `Type11`" |

The messages stay as good as today's. One wrong entry fails the whole impl.

### Other checks

- **Uses in the Referrer** (grid U, E=1000, 10 per context, L+R `check` change of K=4 / 8 / 16): signatures
  only -22% / -22% / -25%, impl only -18% / -19% / -22%, witness only -23% / -24% / -27%.
- **One ID for all contexts** (grid I): E=1000 -14% / -14% / -16%, E=5000 -32% / -33% / -37% (K=4 / 8 / 16).
- **Instruction counts** give the same break-even n within one step (`report/nsweep.txt`).

## Conclusion

**Does grouping pay?** It depends on the size of the crate and on the compiler.

- **Corpus scale** (a crate has 49 entries at the median, 197 at most; 70% of the contexts hold one entry):
  no. On rustc 1.63 and later the best K changes L+R by -1% (E=50), -2% (E=200) and -5% (E=1000).
- **Large items on 1.63+** (E >= 1000 with 8 or more entries per context, K=4..16): the Leaker gains 28-41% and
  L+R 11-17% at E=1000, L+R 29-46% at E=5000-10000.
- **rustc 1.56-1.60**: the base Leaker is quadratic. K >= 4 cuts L+R by 65-76% at E=1000 (4 or 10 per context)
  and by 80% (K=4) to 88% (K=16) at E=2000. This is where grouping really pays.

**Break-even n per K** (table above): K=2..8 pay from n=2 (L and L+R, `build` and `check`; K=8 only just),
K=12 and 16 from n=3 (L+R) or n=3..4 (L), K=32 from n=4..5 (L+R) or n=6 (L). At n=1 every K loses, by 6% (K=2) to
262% (K=32) in the Leaker. The rule of thumb is n > 1 + 0.2 (K-1), a filler costing 0.2 of an entry impl.

**Which K?**

- **Corpus** (70% singletons): K=2..4. They are the only K that never lose: K=3 is best at E=200 and 1000
  (-2%, -5%), K=4 at E=50 (-1%). K=8 breaks even at E=1000 (+1%), K=16 loses 13%, K=32 loses 35%. On 1.56-1.60
  the best K is 6..12 (about -30%), since there the gain from fewer impls outweighs the fillers.
- **Large crates** (n >= 10, E >= 1000): K=12 or 16 by a small margin (L+R -15% at E=1000, -46% at E=10000
  with K=16). K=4 is within 3-6 points of the best L+R at every E. K=32 pays only from n=10.
- **Robust choice: K=4.** It never loses for n >= 2 (-14% L at n=2, -34% at n=4), costs +23% in the Leaker
  (+7% L+R) on a singleton context, is within 3 points of the best on the corpus and within 3-6 points of the
  best on large crates. K=3 is a little better on the corpus (-5% at E=1000), K=8..16 a little better on large items.
- Nightly with defaults would remove the trade-off: K=16 or 32 and no singleton penalty (-23% on corpus-like
  crates, -46% at n=10). Without defaults the filler cost is what limits K.

**What it would mean for the library**

- **Repeater definition**: the user writes exactly K `type TypeJ: ?Sized;`. K must be fixed by the
  library and part of `FORMAT_VERSION`; a Leaker and a Referrer with different K cannot interoperate.
  `repeater_definition` takes K.
- **API**: `visitor(context, |ty, ix|)` and `repeater_impl_with` would hand out `(args, id, group, slot)` instead
  of `ix`. A helper has to build the use-site path, or each macro author reimplements the slot logic.
- **Required-only impls**: a trusting Referrer that writes the impls of `required()` entries only has to
  fill the optional slots with `()`.
- **Unsized entries**: no issue, all types are `?Sized`.
- **Failure granularity**: one wrong entry fails the whole impl.
- **Error messages**: unchanged in quality (see above).

**Recommendation.** For rustc 1.63 and later, grouping gains at most 5% on crates like the corpus and does not
justify a format change by itself. It is worth it if Leaker crates with items of 1000 or more entries are expected
(-11..-17% L+R, up to -46% at E=10000), or if rustc 1.56-1.60 must stay usable (-65..-88%). Then use
**K=4** (K=3 if only small crates matter, K=8..16 if only large items matter). If `associated_type_defaults`
stabilizes, switch to K=16 with defaults.

## Caveats

- The Referrer crates are synthetic: the bodies are `{ x }`, so the Repeater's share of a real Referrer crate
  (41-49% of CPU here) is probably an upper bound.
- One machine, one CPU model. Nightly 1.101 is slower than 1.99 on the baseline.
- The corpus is the property test's generator, not real code. It has odd shapes, and the largest
  items are small. A real crate with a 5000-entry item was not available.
- n is exact in each context of the n sweep (E is within 1% of 1000); real items mix sizes, which is what the
  corpus-like runs show. The corpus-like runs have three draws per size.
- Leaker and Referrer are measured as separate crates. The L+R columns weight them equally; in practice the
  Leaker crate is built once, and each Referrer crate by someone else.
- The cost of running the proc macro (the analysis) is not in these numbers: no proc macro is used here.

## Files

| path | content |
|---|---|
| `report/*.txt` | the tables behind this README (CPU / wall and instructions), by grid and toolchain |
| `corpus/histogram.txt`, `corpus/sizes.json` | the histogram and the sizes that `--dist corpus` draws from |
| `prop-hist-logging.patch` | logging for the corpus run, against error-recovery 1de07a8 |

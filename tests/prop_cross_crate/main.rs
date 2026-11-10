//! A property test of `type-leak` across crate boundaries.
//!
//! Each case is a generated Leaker's crate (a module of relative types, and random modules,
//! structs, enums, unions, traits and impls with `#[prop_leak]` on them, see `gen.rs`) and a
//! generated Referrer's crate that invokes the macro exported for each item. The macros are those
//! of `tests/crates/macro/prop.rs`, which analyze the item with the matching `intern_*` (or, for
//! some, through a scope: see below), emit the Repeater's implementations, and, in the Referrer's
//! crate, rewrite a copy of the item with `Referrer::visitor(context, ..)`.
//!
//! The property: if the Leaker accepts an item (it may reject it, which is counted, and every
//! rejection must be explained: see below), the Referrer's crate compiles, and every type position
//! of the rewritten copy is exactly the original type.
//! The latter is checked with a witness for each position: the Leaker's crate defines
//! `pub fn w_k<generics>() -> PhantomData<*mut OriginalType>`, and the Referrer's crate has
//! `fn chk_k<generics>() -> PhantomData<*mut RewrittenType> { leaker::w_k::<..>() }`, with the same
//! generics (the rewritten type is the result, as a parameter of a method is, so that the bounds
//! implied by its well-formedness, such as `T: 'a` for `&'a T`, are assumed as in a real copy).
//! `*mut` makes the types invariant: with a bare `PhantomData<T>`, which is covariant, a
//! rewritten type with a shorter lifetime in a covariant position (`&'a u8` for `&'static u8`) would
//! pass for the original, and it must be the same type, lifetimes included.
//! `Self` is what it is at the position (the struct, enum or union, the self type of an impl, or
//! the host type that implements a trait in the Referrer's crate); a position with an elided
//! lifetime has no witness, and in a copy of a module neither has a type that names what the module
//! declares, because the copy declares its own.
//!
//! All the cases are built by one `cargo build --workspace`, so that they compile in parallel.
//! The Leaker's crate must compile by itself: if it does not, it is built again without
//! `type-leak` (`prop_leak_noop`) to tell a bug of the generator from the macro's output.
//!
//! # A run that tests nothing fails
//!
//! A rejection counts as a success of the property, so the test could pass while checking nothing.
//! It does not:
//!
//! - The watchdog. A crate that is not a case (the helper crate `tests/crates/macro`, `type-leak`,
//!   a build script) failing to build, or cargo failing without a compile error, fails the run
//!   (`build_problem()`), and so does a case that built no executable or no Referrer's library
//!   without an error saying why, a binary that fails or prints a line that is not a report, and
//!   an item with no report (`missing_artifacts()`). None of them is read as "rejected".
//! - The acceptance floor. The Leaker must accept at least a share of the items analyzed of each
//!   kind (struct, enum, union, trait, inherent impl, trait impl, module) and of each scope they
//!   are analyzed through, and the total accepted must not be 0 (`Floors`, `FLOORS`: they come
//!   from the rates measured over several seeds, and the failure says which kind is below its
//!   floor and why its items were rejected). A kind with fewer than `TYPE_LEAK_PROP_MIN_SAMPLES`
//!   items is not checked. At least `TYPE_LEAK_PROP_MIN_CHECKED` percent of the type positions of
//!   the accepted items must have been compared with their witness.
//!
//! The generator keeps the shapes that the Leaker rejects (a predicate that names a leaked type
//! that uses a generic, an `impl Trait` argument, ...) in a share of the items (`gen::Gen::cyclic`),
//! so that most items are accepted: an item is rejected with its first such shape, and a module
//! with the first of its dozen items.
//!
//! # Every rejection is explained
//!
//! A rejection counts as a success of the property, so a Leaker that rejected what it should accept
//! would pass. So every error of every rejection must be one of the reasons that the test knows
//! (`family()`: a leaked type in a bound or where predicate that rustc cannot normalize, a bare
//! trait of the prelude where the Leaker does not allow the prelude, `Self::A` where `Self` is a
//! type, the lifetime of a binder that no input keeps, a relative constant that cannot be leaked,
//! ...), and the item must have a shape that gives it: the generator tags each item with the
//! shapes it has (`gen::Item::tags`: `bound_cycle`, `self_assoc`, `binder_no_anchor`, `bare:Fn`,
//! ...). A reason outside the list fails the run ("NOT A KNOWN REASON") with the item and the
//! message, and so does a known one on an item that has no tag for it ("NO SHAPE THAT GIVES IT").
//! Either is a defect of `type-leak`, or a limitation to put in the list with its shape.
//! `TYPE_LEAK_PROP_REJECTIONS=1` prints each rejection with the tags of its item.
//!
//! # Accepting the rejections (`TYPE_LEAK_PROP_ACCEPT`)
//!
//! The Leaker copies the part of the item that conflicts as written, whole, with nothing in it
//! leaked, and goes on: the recovered Referrer (`LeakerResult::into_referrer()`) has the
//! correction it makes, and `LeakerResult::allow(cause)` accepts the conflict by dropping its
//! error. This test did not compile those corrections. With `TYPE_LEAK_PROP_ACCEPT=1` it compiles
//! them, by accepting the causes of the errors:
//!
//! 1. The macro (`prop_leak_accept`) analyzes an item as always. If the Leaker rejects it, it
//!    allows the `cause` of every error, which must drop every error, and takes the recovered
//!    Referrer. The errors are what the rejection is: `family()` explains each as always ("NOT A
//!    KNOWN REASON", "NO SHAPE THAT GIVES IT"). If an error remains, the run fails ("ACCEPT
//!    FAILED", with the errors and those that remain): allowing the causes of all the errors
//!    must leave none.
//! 2. The result is emitted and checked like an accepted item: the Repeater's impls, the
//!    witnesses, the macro for the Referrer's crate, the copy rewritten by `Referrer::visitor`, and
//!    the witness checks. Its status in the report is `acc`. The acceptance floors and the table
//!    are those of the first analysis, where it counts as rejected; the summary has a line for
//!    the items accepted by `allow()`, with the `Reason` and the kind of cause of the errors that
//!    were allowed. A compile failure of such an item is reported with the class "ACCEPT: ..." and
//!    the errors that were allowed. Which crate fails tells where: the Leaker's crate (a Repeater's
//!    impl or the item: the entries or the impls are wrong) or the Referrer's (the copy, or a
//!    witness check: the part that was copied as written cannot be written there, or an entry
//!    that should not be one is replaced).
//! 3. The twin of an item with a bare trait of the prelude (see below) is compared at the first
//!    analysis only. `PRIMITIVE KEPT` does not apply to an item accepted by `allow()`: the part
//!    that is copied as written may hold a primitive type that the Leaker does not allow.
//!
//! The parts that are copied as written name things of the Leaker's crate by the paths they have
//! there (`R0`, `self::R1`, `super::R1`, `super::base::B1`, `crate::base::B1`, an import `I1`),
//! and `allow()` is a promise that they can be written as they are in the copy. So that this does
//! not fail for a reason of no interest, and does not weaken the property for the other items, the
//! copies of the items accepted by `allow()` are made in a Referrer's crate of their own
//! (`tlp_a_NNN`, `gen::accept_referrer()`), and the ordinary Referrer's crate expands them to
//! nothing (`Args::accepted`, the `accept` flag of the invocation). The crate of their own has the
//! modules of the Leaker's crate that the items are in, in the same place of the module tree, with
//! the same imports and a glob import of the Leaker's module (so `R0`, `I1`, `self::` and
//! `super::` mean what they mean in the Leaker's crate), and `base` at its root, the glob import
//! of the Leaker's (so `crate::base::B1` and `super::base::B1` do). The ordinary Referrer's crate
//! has none of them, so that the copy of an ordinary item that keeps a relative path by mistake
//! does not compile there, as before. A separate crate and not a module of the ordinary one,
//! because `crate::base` is the root of the crate: `base` there would let such a copy resolve.
//! A copy of a module analyzed as a whole keeps its imports in the crate of their own. (For an item
//! accepted by `allow()` the test cannot tell a relative path that the Leaker should have leaked
//! and did not from one it copied as written: both resolve. The witnesses still compare the types.)
//!
//! # The copies that do not compile are explained
//!
//! A copy of an item accepted by `allow()` may not build, for the shapes that the docs of
//! `LeakerResult::allow()` list ("What accepting does not make compile"), and the test explains each
//! failure the way it explains a rejection (`accept_families()`, `unexplained_accept()`): the
//! diagnostic is one of the listed failures, if its error code (`E0277`, `E0106`), the errors that
//! were allowed (their `Reason` and the kind and tokens of their cause) and the shapes of the item
//! (its tags) say so. The failure is counted and is no failure of the run. Anything else fails
//! the run: "ACCEPT: NOT A KNOWN FAILURE" for a diagnostic that no listed failure explains, and
//! "ACCEPT: NO SHAPE THAT GIVES IT" for a listed one on an item that has not the shape. Either is a
//! defect, or a limitation to list in the docs of `allow()` and here, with its shape.
//!
//! - (i) `Self::A`, or `Self` as a type, in a predicate whose context has a bound that names a
//!   leaked type: `UnprovableBound`, E0277. Only the cause is copied as written, and the bound stays
//!   leaked. The item has a predicate that names `Self` (`self_in_predicate`) and a bound that names
//!   a leaked type (`bound_cycle`, or a path of `std` that the Leaker does not allow:
//!   `relative_std`).
//! - (ii) The same cause in a trait with a generic whose bound names an associated type that
//!   another bound fixes (`T: Iterator<Item = ..> + Fn(<T as Iterator>::Item) -> u8`): E0277 too,
//!   but rustc cannot implement such a trait with a generic impl at all. The item has the alias
//!   bound (`alias_bound`).
//! - (iii) `Self` with `allow_self = false` (`RelativePath`) in a function pointer whose elided
//!   output takes the lifetime of an input that a path hides: `Visitor::with_self()` replaces
//!   `Self` inside the accepted part, which changes the elision (E0106). The item denies `Self`
//!   (`self_denied`) and has the function pointer (`hidden_self`).
//!
//! The summary counts them by class. The generator avoids none of these shapes: the rejected ones
//! (`gen::Gen::cyclic`) have them.
//!
//! # The copies after the analysis without a rule (`TYPE_LEAK_PROP_FEATURES=allow_stages`)
//!
//! `LeakerConfig::allow_X()` and `LeakerResult::allow_X()` are one rule, but the result judges an
//! analysis made without it, so for a path rule it can leave other errors and other entries (see
//! `tests/allow_stages.rs`). With `allow_stages` the macro of each item takes each rule of
//! `gen::rules()` in turn and makes two copies, and the test builds them, to see whether the
//! difference is more than a number:
//!
//! - after the result: the item is analyzed without the rule (the primitive types and the names of
//!   the prelude allowed, but for the rule `prelude`), the result is given the rule, then
//!   `allow(cause)` for the cause of each error that remains until none does (the nested errors of
//!   an error that was dropped come up: `ErrorItem::nested`; the failures of those are those of the
//!   accepted copies above), and its Referrer copies the item (`__tl_st_<rule>_res_<id>!`);
//! - after the config: the item is analyzed with a config that has the rule, the causes of the
//!   errors that remain are allowed, and its Referrer copies the item (`__tl_st_<rule>_cfg_<id>!`).
//!
//! The rules are `prelude` (`allow_prelude()`, every item), `crate` (`allow_crate()`), `path`
//! (`allow_path(super::base)`: the module `base` as the modules at depth 1 name it), `std`
//! (`allow_std_crates()`, with `relative_std`), `values` (`allow_values()`, with `values`) and
//! `default` (`allow_default()`). A rule is made for the items where it matters: it drops or trims an
//! error of the analysis without it, or the config finds other errors or another Referrer than the
//! result. `values` and `default` are cause and place rules: the config and the result must give
//! identical errors and an equal Referrer (`STAGES DIFFER` fails the run).
//!
//! The Repeater's impls of the copies are written next to those of the ordinary analysis with an `ID`
//! of their own (`Args::repeater_id()`), the witnesses are the ordinary ones (written for a rejected
//! item too), and each copy is built in two crates:
//!
//! - **with globs** (`tlp_<rule>_<side>_glob_NNN`), like the crate of the accepted items
//!   (`gen::accept_referrer()`: the modules of the Leaker's crate, their imports and a glob import)
//!   where the names of the prelude mean the prelude: nothing generated declares one
//!   (`shadowed_prelude_names()`), which is what allowing them promises. A relative path that the
//!   copy keeps by mistake resolves there, so the test cannot tell it from one that is kept as
//!   written;
//! - **clean** (`tlp_<rule>_<side>_clean_NNN`, `gen::clean_referrer()`), for the items that the rule
//!   leaves no error in (`ok`), invoked with `clean` first. The copy is laid out as in the ordinary
//!   Referrer's crate and none of the Leaker's names resolve but the ones that the rule allows, which
//!   the crate makes resolve as the user promised: `base` for `crate::base`, `alloc` for `alloc::`, the
//!   modules of the Leaker with `RLEN` and `ILEN` for the paths of values and defaults. A relative
//!   type that the rule allowed to drop with an error but that the copy kept as written fails to
//!   compile there. (Remove `pub mod base` from the crate of the rule `crate` and it does.)
//!
//! A copy after the result that does not compile is one of
//!
//! - a failure of the docs of `allow()` ((i), (ii), (iii): `accept_families()`, explained by the
//!   errors of the analysis without the rule), counted by class; when the copy after the config
//!   compiles, it is the rule after the analysis that makes the copy fail;
//! - `SHARED`: no listed failure explains it, but the copy after the config fails too, so the rule
//!   after the analysis is not what makes it fail. Not a failure of the run, and printed with what
//!   explains the copy after the config;
//! - `NEW`: only the copy after the result fails, for a reason that the docs of `allow()` do not
//!   list. It fails the run ("STAGES: NEW FAILURE", with the item and the diagnostic). The same for a
//!   clean copy that fails while the clean copy after the config compiles ("STAGES CLEAN: NEW
//!   FAILURE"); there is no documented class for it.
//!
//! The summary has, for each rule, the counts: items compared, items where the rule matters, copies
//! built after the result and after the config, with globs and clean, the failures by kind, and for
//! each item whose number of errors differs its causes and what became of its copy. `STAGES FAILED`
//! is the failure of an error that remains when every cause is allowed, as `ACCEPT FAILED` is.
//!
//! # Bare and absolute spellings
//!
//! Bounds, where clauses, supertraits and trait objects name `Sized`, `?Sized`, `Send`, `Sync`,
//! `Unpin`, `Copy`, `Clone`, `Default`, `Fn`, `FnMut`, `FnOnce`, `Into`, `From`, `AsRef`, `Iterator`,
//! `PartialEq`, `Eq`, `PartialOrd` and `Ord` sometimes bare and sometimes in full
//! (`::core::marker::Sized`), whether the Leaker allows the names of the prelude or not. With it,
//! the macro analyzes the item with the traits written in full too (its twin), and the two must
//! give the same entries, generics and errors; without it, a bare trait in a bound must be
//! rejected as a relative path (`tests/crates/macro/bare.rs`). A bare `?Sized` is the exception (`?`
//! applies to the real `Sized`): the Leaker accepts it without the prelude, which
//! `gen::BARE_RELAXED_SIZED_ALWAYS_OK` says.
//!
//! # What the Referrer needs
//!
//! The macro writes the Repeater's impls of the required entries only (`Referrer::required()`) for
//! an item that the Referrer invokes as trusting, as a downstream crate does, and of all the
//! entries for one it invokes as `strict` (a quarter of the items, `gen::Item::strict`, that the
//! macro is told). A trusting Referrer that replaced an optional entry fails to compile.
//!
//! # Other checks of the macro
//!
//! - The helpers agree: `repeater_impl()` declares `impl_generics()` of the context of its entry
//!   and the parameter of the implementor, `generics()`, `iter_with_generics()`, `required()` and
//!   `must()` say the same, and `required()` comes first (`HELPERS DISAGREE`).
//! - A fifth of the cases do not allow the primitive types: `u8` is an entry then, and the copy
//!   keeps none in a type position (`PRIMITIVE KEPT`).
//!
//! What the macro analyzes besides `intern_*`: a fifth of the structs, enums and unions are
//! analyzed nested in `enter_mod` (`View::visit_item`, its generics are local) or in
//! `enter_generics` (no `Self`), a tenth of the traits nested in `enter_mod`, another tenth with
//! `enter_trait`, and a third of the impls whose generics name no leaked type with `enter_impl`,
//! where only the self type and the arguments of the trait are visited (the shape of a macro that
//! resolves the trait by its name), and only those are compared. What it cannot express: the
//! impls in `enter_mod` (the rewritten self type is the `Self` of the entries, which the nested
//! contexts do not give; the macro cannot write a nested impl's own `Self`), and `enter_trait` for a
//! trait with a generic associated type (the scope does not know its items). A struct, an enum, a
//! union or a trait nested in `enter_mod` names its own `Self` (`nested_self` below): the macro
//! writes its entries with `Visitor::with_nested_self()`, and the check writes the Leaker's type
//! for a `Self` that no entry has.
//!
//! A fifth of the traits analyzed one by one have parameters with defaults that use a parameter with
//! a bound (`trait Tr<T: AsRef<Self>, U = Rel<T>>`, `trait_self_defaults` below, F39). The bound
//! names the trait's `Self` in seven of ten, and the rest is the control (`T: Clone`). Those
//! traits are nested in `enter_mod` in 45% (the Visitor writes the default through the host,
//! `with_nested_self()`), and analyzed as the item itself by `intern_trait` or `enter_trait`
//! otherwise (the Visitor finds the default without `Self`, and the macro picks the host as the
//! implementor for an entry whose bound names `Self`). Of the nested ones, 60% are declared again in
//! the Referrer's crate (`redeclare`, `Args::redeclare` of `tests/crates/macro/prop.rs`), which
//! cannot pass the defaulted parameter to its own default (E0128), and then have a method that
//! names the parameters only. The statistics count them apart ("defaults and `Self`").
//!
//! Environment variables:
//!
//! - `TYPE_LEAK_PROP_CASES` (default 24): the number of cases.
//! - `TYPE_LEAK_PROP_SEED` (default 1): the seed of the generator.
//! - `TYPE_LEAK_PROP_ITEMS` (default 8): the most items in a module.
//! - `TYPE_LEAK_PROP_MIN_ACCEPT`: the acceptance floors, in percent. A number sets every kind and
//!   scope, `name=number` one of them, read in order: `0` turns the floors off (the total accepted
//!   must still be positive), `40,mod=10,enter_impl=70` is 40% but 10% for modules and 70% for
//!   `enter_impl`. The names are the kinds of the table of the summary and the scopes `enter_mod`,
//!   `enter_generics`, `enter_trait` and `enter_impl`.
//! - `TYPE_LEAK_PROP_MIN_SAMPLES` (default 12): the fewest items of a kind for its floor to count.
//! - `TYPE_LEAK_PROP_MIN_CHECKED` (default 30): the least percentage of type positions compared.
//! - `TYPE_LEAK_PROP_REJECTIONS`: if set, the rejections are printed with the tags of their items,
//!   and with the `Reason` and the kind of the cause of each error.
//! - `TYPE_LEAK_PROP_ACCEPT`: if set (and not `0`), the Leaker accepts the items it rejects by
//!   `LeakerResult::allow(cause)` of the cause of every error, and the test compiles the result: see
//!   "Accepting the rejections". The workspace is `ws-<seed>-<cases>-accept`.
//! - `TYPE_LEAK_PROP_FEATURES`: a comma-separated list of constructs that `type-leak` is known to
//!   fail on (see `gen::Features`). The generator avoids them by default, so that the test passes
//!   on the current library, and a new failure is a new finding. With one of `inferred_outlives`,
//!   `impl_self_outlives`, or `all`, the test fails with the reproduction of that finding
//!   (`tests/findings.rs` has each one as a library-level test). `strict_anchors` (F18, about the
//!   items that are invoked as `strict`, a quarter of them) is fixed, and off by default: the items
//!   that have it are not invoked as `strict` unless it is asked for. `unsized_params`,
//!   `bound_cycles`, `impl_assoc_self`, `method_lifetime_paths`, `module_crate_paths`,
//!   `bound_merge_syntax`, `fn_pointer_self`, `elided_fn_generics`, `impl_self_bounds` (F13),
//!   `impl_pred_leaked` (F14), `self_elision` (F15, F23), `implied_outlives` (F16),
//!   `strict_cycles` (F17, about the strict items too), `impl_self_bound_cycle` (F19),
//!   `trait_assoc_bound_cycle` (F20), `iterator_item_cycle` (F21), `supertrait_item` (F25),
//!   `alias_item` (F26), `impl_trait_args` (F27), `hidden_lifetimes` (F28), `projection_outlives` (F30),
//!   `item_predicate_cycle` (F33), `self_projection_outlives` (F34), `gat_where_clause` (F35),
//!   `nested_self` (F36: a struct, an enum, a union or a trait analyzed nested in `enter_mod` that
//!   names its own `Self`, which the Visitor writes with `with_nested_self()`; F37: an entry that
//!   does not name it, and uses a generic whose bound does, is for any type, and the macro passes
//!   the copy's own type; one that names it too is rejected; F38: an absolute type that names it
//!   is no entry, so a strict Referrer copies it as written, with the copy's own `Self`, as a
//!   trusting one does: the macro checks that no optional entry of such an item names `Self`),
//!   `trait_self_defaults` (F39: a trait whose parameter defaults use a generic whose bound names
//!   the trait's `Self`, `trait Tr<T: AsRef<Self>, U = Rel<T>>`, nested in `enter_mod`, where the
//!   Visitor left the default as written, and analyzed as the item itself, where it looks the
//!   default up without `Self`; with the control whose bound names no `Self`; the traits are
//!   counted in the statistics: no shape explains a rejection, so none has a tag in `family()`),
//!   `self_bound_adts` (F41: a struct, an enum or a union nested in `enter_mod` with a generic
//!   whose bound names its own `Self` and a default that uses it; the Leaker must accept it, which
//!   `unexplained()` checks) and `nested_self_entries` (F42: the macro asks for
//!   `LeakerConfig::nested_self_entries()` for the items nested in `enter_mod`, and those of
//!   `self_bound_adts` have a field that names `Self` in a relative type),
//!   `path_named_like_generic` (`self::U` next to a generic `U`), `declared_named_like_generic` (a
//!   module that declares `X`, and `crate::base::B1<X>` in `fn f<X>`) and
//!   `bare_const_in_absolute_path` (a bare `ILEN` next to a leaked argument in a
//!   path of another crate, which was leaked alone as a type: E0573) are fixed findings,
//!   each in a share of the items (`gen::Gen::cyclic`): they are on by default (the Leaker encodes
//!   or rewrites them, records what they need, or rejects and counts what it cannot copy, and the
//!   Referrer names the lifetime of an elided signature where `Self` adds lifetimes to it). The
//!   other findings of `tests/findings.rs` (F22 to F24: the Leaker rejected what compiles, F29,
//!   F31, F32) are not features: a rejection is counted as success, and the test cannot show a
//!   toolchain or a lint.
//!   `allow_stages`, `relative_std` and `values` are no feature of the generator (`gen::Features` is in
//!   the tables, and they are off by default, so the tables do not change):
//!   `allow_stages` makes every macro compare the rules of "The copies after the analysis without a
//!   rule" and builds their copies (the workspace is `ws-<seed>-<cases>-stages`); `relative_std`
//!   writes about a third of the paths of `std` and `core` of the items analyzed one by one without
//!   the leading `::` (some `alloc::`), which the Leaker allows (`allow = "std,core,alloc"`) but for
//!   the rule `std`, and `extern crate alloc;` is in every crate; `values` gives the items values
//!   (`gen::Flags::values`): bodies of functions, methods and default methods, the values of
//!   constants and statics, the defaults of associated constants and of const parameters, with
//!   the types of the Leaker's crate in a `let`, a turbofish and a cast, which the Leaker leaks with
//!   `check_values()` and the witnesses check (`Args::bodies`), and in a share of them a relative path
//!   of a value (`crate::base::bf()`, `RLEN`), which it rejects (the tag `value_path`).
//!
//! The test is ignored by default: it is slow, as it builds the generated crates. Run it with
//!
//! ```text
//! cargo test --test prop_cross_crate -- --ignored
//! TYPE_LEAK_PROP_CASES=200 TYPE_LEAK_PROP_SEED=7 cargo test --test prop_cross_crate -- --ignored
//! ```
//!
//! The sources of every case stay in `target/tmp/prop/ws-<seed>-<cases>/`, and each is a workspace
//! member that builds on its own (`cargo build` there, with `CARGO_TARGET_DIR` as the test has it).

mod gen;
mod json;

use gen::{allow_stages, generate, generate_with, Case, Features, Flags, KINDS, SCOPES};
use json::Json;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::process::Command;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{} is not valid: {:?}", name, value)),
        Err(_) => default,
    }
}

fn case_seed(seed: u64, index: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_u64(seed);
    hasher.write_u64(index as u64);
    hasher.finish()
}

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn leaker_dir(i: usize) -> String {
    format!("c{:03}_l", i)
}

fn referrer_dir(i: usize) -> String {
    format!("c{:03}_r", i)
}

fn leaker_name(i: usize) -> String {
    format!("tlp_l_{:03}", i)
}

fn referrer_name(i: usize) -> String {
    format!("tlp_r_{:03}", i)
}

/// The second Referrer's crate of `TYPE_LEAK_PROP_ACCEPT`.
fn accept_dir(i: usize) -> String {
    format!("c{:03}_a", i)
}

fn accept_name(i: usize) -> String {
    format!("tlp_a_{:03}", i)
}

/// What names a Referrer's crate of `TYPE_LEAK_PROP_FEATURES=allow_stages`: the rule, the side
/// (`res` or `cfg`) and the layout (`glob` or `clean`), see `gen::StageCrate`.
fn stage_suffix(stage: &gen::StageCrate) -> String {
    format!(
        "{}_{}_{}",
        stage.rule,
        stage.side,
        if stage.clean { "clean" } else { "glob" }
    )
}

fn stage_dir(i: usize, stage: &gen::StageCrate) -> String {
    format!("c{:03}_{}", i, stage_suffix(stage))
}

fn stage_name(i: usize, stage: &gen::StageCrate) -> String {
    format!("tlp_{}_{:03}", stage_suffix(stage), i)
}

/// Which crate of `allow_stages` a failure is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct StageKey {
    rule: &'static str,
    side: &'static str,
    clean: bool,
}

impl From<&gen::StageCrate> for StageKey {
    fn from(stage: &gen::StageCrate) -> Self {
        StageKey {
            rule: stage.rule,
            side: stage.side,
            clean: stage.clean,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// With `type-leak`, and the Referrer's crate.
    Real,
    /// `Real`, and an item that the Leaker rejects is accepted by `allow()` of the causes of its
    /// errors (`prop_leak_accept`), and copied in the second Referrer's crate.
    Accept,
    /// Without it, to check the generator.
    Identity,
}

/// Writes the workspace of the cases (the indexes are those of the case list). With `stages` (the
/// feature `allow_stages`) the Leaker's macros also make the copies of each rule, the binary reports
/// them, and each case has the Referrer's crates that build them (`Case::stages`); without it the
/// attribute of the items says `stages = false`.
fn write_workspace(ws: &Path, cases: &[(usize, &Case)], mode: Mode, stages: bool) {
    let _ = std::fs::remove_dir_all(ws);
    std::fs::create_dir_all(ws).unwrap();
    let members: Vec<String> = cases
        .iter()
        .flat_map(|(i, case)| {
            let mut dirs = vec![format!("\"{}\"", leaker_dir(*i))];
            if mode != Mode::Identity {
                dirs.push(format!("\"{}\"", referrer_dir(*i)));
            }
            if mode == Mode::Accept {
                dirs.push(format!("\"{}\"", accept_dir(*i)));
            }
            if mode != Mode::Identity && stages {
                for stage in &case.stages {
                    dirs.push(format!("\"{}\"", stage_dir(*i, stage)));
                }
            }
            dirs
        })
        .collect();
    std::fs::write(
        ws.join("Cargo.toml"),
        format!(
            "[workspace]\nmembers = [{}]\nresolver = \"2\"\n",
            members.join(", ")
        ),
    )
    .unwrap();
    // The same versions as the test crates, if they are locked.
    let lock = root().join("tests/crates/macro/Cargo.lock");
    if lock.exists() {
        std::fs::copy(lock, ws.join("Cargo.lock")).unwrap();
    }
    let macro_path = root().join("tests/crates/macro");
    for (i, case) in cases {
        let dir = ws.join(leaker_dir(*i));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            format!(
                "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
                 [dependencies]\ntype-leak-test-macro = {{ path = {:?} }}\n",
                leaker_name(*i),
                macro_path.display().to_string()
            ),
        )
        .unwrap();
        let attr = match mode {
            Mode::Real => "crate::__macro::prop_leak",
            Mode::Accept => "crate::__macro::prop_leak_accept",
            Mode::Identity => "crate::__macro::prop_leak_noop",
        };
        let leaker = case.leaker.replace(gen::ATTR, attr);
        let leaker = if stages {
            leaker
        } else {
            leaker.replace("stages = true", "stages = false")
        };
        std::fs::write(dir.join("src/lib.rs"), leaker).unwrap();
        if mode == Mode::Identity {
            continue;
        }
        // A binary prints what each macro reported.
        let mut main = String::from("fn main() {\n");
        for item in &case.items {
            writeln!(
                main,
                "    println!(\"{{}}\\u{{1e}}{{}}\", {l}::__tl_report_{id}!(), {l}::__tl_errors_{id}!());",
                l = leaker_name(*i),
                id = item.id
            )
            .unwrap();
            if stages {
                for rule in gen::rules() {
                    writeln!(
                        main,
                        "    println!(\"stages|{rule}|{id}|{{}}\", {l}::__tl_stages_{rule}_{id}!());",
                        l = leaker_name(*i),
                        id = item.id
                    )
                    .unwrap();
                    for side in ["res", "cfg"] {
                        writeln!(
                            main,
                            "    println!(\"stage|{rule}|{side}|{{}}\", {l}::__tl_st_{rule}_{side}_report_{id}!());",
                            l = leaker_name(*i),
                            id = item.id
                        )
                        .unwrap();
                    }
                }
            }
        }
        main.push_str("}\n");
        std::fs::write(dir.join("src/main.rs"), main).unwrap();

        let dir = ws.join(referrer_dir(*i));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            format!(
                "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
                 [dependencies]\n{} = {{ path = \"../{}\" }}\n",
                referrer_name(*i),
                leaker_name(*i),
                leaker_dir(*i)
            ),
        )
        .unwrap();
        std::fs::write(dir.join("src/lib.rs"), &case.referrer).unwrap();

        if mode == Mode::Accept {
            let dir = ws.join(accept_dir(*i));
            std::fs::create_dir_all(dir.join("src")).unwrap();
            std::fs::write(
                dir.join("Cargo.toml"),
                format!(
                    "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
                     [dependencies]\n{} = {{ path = \"../{}\" }}\n",
                    accept_name(*i),
                    leaker_name(*i),
                    leaker_dir(*i)
                ),
            )
            .unwrap();
            std::fs::write(dir.join("src/lib.rs"), &case.accept_referrer).unwrap();
        }

        if stages {
            for stage in &case.stages {
                let dir = ws.join(stage_dir(*i, stage));
                std::fs::create_dir_all(dir.join("src")).unwrap();
                std::fs::write(
                    dir.join("Cargo.toml"),
                    format!(
                        "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
                         [dependencies]\n{} = {{ path = \"../{}\" }}\n",
                        stage_name(*i, stage),
                        leaker_name(*i),
                        leaker_dir(*i)
                    ),
                )
                .unwrap();
                std::fs::write(dir.join("src/lib.rs"), &stage.lib).unwrap();
            }
        }
    }
}

/// The class of the diagnostics of the copy of an item accepted by `allow()`, which
/// `accept_families()` explains if they are the failures that the docs of `allow()` list.
const ACCEPT_COPY: &str =
    "ACCEPT: the copy of an item accepted by allow() does not compile in the Referrer's crate";

/// The class of the diagnostics of a copy of `allow_stages` (the analysis without the rule, the
/// result with the rule and the causes of the errors that remain; or the analysis of a config that
/// has the rule), which is built in a Referrer's crate of the stage copies (`Failure::stage` says
/// which).
const STAGE_COPY: &str =
    "STAGES: the copy made after the analysis without the rule, or after the config's, does not compile in the Referrer's crate";

/// A diagnostic of the build.
struct Diag {
    /// The crate.
    package: String,
    /// The error code of rustc (`E0277`), or empty.
    code: String,
    rendered: String,
    /// The `(file, line)` of its spans and of where the macros of each were invoked.
    lines: Vec<(String, u64)>,
}

struct Build {
    success: bool,
    diags: Vec<Diag>,
    executables: HashMap<String, PathBuf>,
    /// The targets that cargo built (or found fresh), by name.
    artifacts: HashSet<String>,
    /// What cargo printed to stderr.
    stderr: String,
}

fn span_lines(span: &Json, out: &mut Vec<(String, u64)>) {
    if let (Some(file), Some(line)) = (
        span.get("file_name").and_then(Json::str),
        span.get("line_start").and_then(Json::num),
    ) {
        out.push((file.to_string(), line as u64));
    }
    if let Some(expansion) = span.get("expansion").and_then(|e| e.get("span")) {
        span_lines(expansion, out);
    }
}

fn cargo_build(ws: &Path, target: &Path) -> Build {
    let output = Command::new(env!("CARGO"))
        .args([
            "build",
            "--workspace",
            "--keep-going",
            "--message-format=json",
            "--manifest-path",
        ])
        .arg(ws.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", target)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    let mut build = Build {
        success: output.status.success(),
        diags: Vec::new(),
        executables: HashMap::new(),
        artifacts: HashSet::new(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    };
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(json) = Json::parse(line) else {
            continue;
        };
        match json.get("reason").and_then(Json::str) {
            Some("compiler-message") => {
                let message = json.get("message").unwrap();
                if message.get("level").and_then(Json::str) != Some("error") {
                    continue;
                }
                let mut lines = Vec::new();
                // The primary spans first.
                let spans = message.get("spans").map_or(&[][..], Json::arr);
                for primary in [true, false] {
                    for span in spans {
                        if span.get("is_primary").and_then(Json::bool) == Some(primary) {
                            span_lines(span, &mut lines);
                        }
                    }
                }
                build.diags.push(Diag {
                    package: json
                        .path("target.name")
                        .and_then(Json::str)
                        .unwrap_or("")
                        .to_string(),
                    code: message
                        .path("code.code")
                        .and_then(Json::str)
                        .unwrap_or("")
                        .to_string(),
                    rendered: message
                        .get("rendered")
                        .and_then(Json::str)
                        .unwrap_or("")
                        .to_string(),
                    lines,
                });
            }
            Some("compiler-artifact") => {
                if let Some(name) = json.path("target.name").and_then(Json::str) {
                    build.artifacts.insert(name.to_string());
                    if let Some(exe) = json.get("executable").and_then(Json::str) {
                        build
                            .executables
                            .insert(name.to_string(), PathBuf::from(exe));
                    }
                }
            }
            _ => (),
        }
    }
    build
}

/// A Referrer's crate of a case: its package, its directory, the line of each item that it invokes,
/// the class of its diagnostics, and the crate of `allow_stages` it is (`Failure::stage`).
type BuiltCrate<'a> = (
    String,
    String,
    &'a Vec<(usize, usize)>,
    &'static str,
    Option<StageKey>,
);

#[derive(Debug)]
struct Failure {
    case: usize,
    item: Option<usize>,
    class: &'static str,
    /// The error code of rustc, if the failure is a diagnostic of the copy of an item accepted by
    /// `allow()`, which `accept_families()` explains.
    code: String,
    text: String,
    /// The crate of `allow_stages` the diagnostic is in (`STAGE_COPY`).
    stage: Option<StageKey>,
}

/// One error of a rejection (an `ErrorItem`): the message, the tokens it is about, its first note,
/// its `Reason` (the name of the variant) and the kind of its cause (the name of the variant).
#[derive(Debug, Default, Clone)]
struct Rejection {
    message: String,
    tokens: String,
    note: String,
    reason: String,
    kind: String,
}

/// What a macro reported for an item.
#[derive(Debug, Default, Clone)]
struct Report {
    kind: String,
    /// The Leaker accepted the item at the first analysis.
    accepted: bool,
    /// `TYPE_LEAK_PROP_ACCEPT`: the Leaker rejected the item, and allowing the cause of every error
    /// dropped every error (`acc`). It is not `accepted`: the floors count it as rejected.
    allowed: bool,
    /// `TYPE_LEAK_PROP_ACCEPT`: the Leaker rejected the item, and an error remained when the cause
    /// of every error was allowed (`accfail`): `remaining` are the errors that remained.
    accept_failed: bool,
    /// The reason of the rejection (the messages of its errors, on one line), or the error of the
    /// rewrite.
    detail: String,
    /// The errors of the rejection, whatever was allowed.
    rejections: Vec<Rejection>,
    /// The errors that remained when the causes were allowed (`accept_failed`).
    remaining: Vec<Rejection>,
    /// What the macro could not rewrite in the copy of an item it accepted (`accepted` or
    /// `allowed`).
    rewrite_error: Option<String>,
    positions: usize,
    checked: usize,
    entries: usize,
    /// The optional entries whose Repeater impl the macro did not write (the Referrer is
    /// trusting).
    skipped: usize,
    /// The item has bare traits of the prelude and was compared with its twin.
    twin: bool,
    /// `allow_stages`: the rule does not matter for the item, and the macro made no copy (`skip`).
    skip: bool,
    /// `allow_stages`: the errors of the analysis without the rule, which the rule drops or leaves.
    plain: Vec<Rejection>,
}

/// The errors of a rejection, as `prop::rejection_text()` writes them.
fn parse_rejections(errors: &str) -> Vec<Rejection> {
    errors
        .split('\u{1d}')
        .filter(|error| !error.is_empty())
        .map(|error| {
            let mut fields = error.splitn(5, '\u{1f}');
            let mut next = || fields.next().unwrap_or("").to_string();
            Rejection {
                message: next(),
                tokens: next(),
                note: next(),
                reason: next(),
                kind: next(),
            }
        })
        .collect()
}

fn parse_report(line: &str) -> Option<(usize, Report)> {
    // The errors of a rejection follow the report, and the errors that remained of an `accfail`
    // the errors of the rejection. The copy of `allow_stages` has the errors of the analysis without
    // the rule at the end.
    let (line, errors) = line.split_once('\u{1e}').unwrap_or((line, ""));
    let (errors, plain) = errors.split_once('\u{1b}').unwrap_or((errors, ""));
    let (errors, remaining) = errors.split_once('\u{1c}').unwrap_or((errors, ""));
    let mut parts = line.splitn(4, '|');
    let id = parts.next()?.parse().ok()?;
    let kind = parts.next()?.to_string();
    let status = parts.next()?;
    let rest = parts.next().unwrap_or("");
    let mut report = Report {
        kind,
        accepted: status == "ok",
        allowed: status == "acc",
        accept_failed: status == "accfail",
        skip: status == "skip",
        ..Default::default()
    };
    if report.accepted || report.allowed {
        for word in rest.split(' ') {
            match word.split_once('=') {
                Some(("entries", n)) => report.entries = n.parse().unwrap_or(0),
                Some(("positions", n)) => report.positions = n.parse().unwrap_or(0),
                Some(("checked", n)) => report.checked = n.parse().unwrap_or(0),
                Some(("skipped", n)) => report.skipped = n.parse().unwrap_or(0),
                Some(("twin", n)) => report.twin = n == "1",
                _ => (),
            }
        }
        if let Some(at) = rest.find("rewrite-error=") {
            report.rewrite_error = Some(rest[at..].to_string());
            report.detail = rest[at..].to_string();
        }
    }
    report.plain = parse_rejections(plain);
    if !report.accepted {
        report.rejections = parse_rejections(errors);
        report.remaining = parse_rejections(remaining);
        report.detail = report
            .rejections
            .iter()
            .map(|r| r.message.as_str())
            .collect::<Vec<_>>()
            .join(" ; ");
    }
    Some((id, report))
}

/// The first lines of a diagnostic.
fn excerpt(text: &str, max: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = lines[..lines.len().min(max)].join("\n");
    if lines.len() > max {
        write!(out, "\n... ({} more lines)", lines.len() - max).unwrap();
    }
    out
}

/// The errors of a rejection, one per line: the `Reason`, the kind of the cause, the message, the
/// tokens it is about.
fn describe(errors: &[Rejection]) -> String {
    errors
        .iter()
        .map(|r| {
            format!(
                "  {} ({}): {} [tokens: {}]",
                r.reason, r.kind, r.message, r.tokens
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The line of the Leaker's `lib.rs` where the attribute of each item is, with the item's id.
fn attr_lines(case: &Case) -> Vec<(usize, usize)> {
    case.items
        .iter()
        .filter_map(|item| {
            let call = format!("(id = {},", item.id);
            let at = case.leaker.lines().position(|line| line.contains(&call))?;
            Some((at + 1, item.id))
        })
        .collect()
}

/// What the macro of an item counted for a rule with `TYPE_LEAK_PROP_FEATURES=allow_stages`
/// (`__tl_stages_<rule>_<id>!`, see `rule_stage()` of `prop.rs`): the errors of the analysis without
/// the rule, those that remain when the rule is given to the config (`before`), and when it is given
/// to the result (`after`). The path rules cannot be exact (see `tests/allow_stages.rs`), and the
/// difference is a finding, printed with the causes. The cause and place rules (`values`, `default`)
/// are: the config and the result give the same errors and an equal Referrer, or the run fails.
#[derive(Debug)]
struct Stages {
    case: usize,
    item: usize,
    /// The rule drops or trims an error of the analysis without it (always for the rule `prelude`).
    matters: bool,
    plain: usize,
    before: usize,
    after: usize,
    before_causes: String,
    after_causes: String,
    /// The config and the result give the same errors (their text) and an equal Referrer.
    same_errors: bool,
    same_referrer: bool,
}

/// The rules whose config and result give the same errors and Referrer.
fn exact_rule(rule: &str) -> bool {
    matches!(rule, "values" | "default")
}

/// What the Referrer's crates of the stage copies made of one rule, for `Stages::summarize()`: the
/// reports of the macros of the copy of each item after the result and after the config, the
/// documented failures of the copies that do not compile by class, and the verdict of each
/// (`explain_copies()`), and the same for the copies of the config's analysis that they are compared
/// with, and for the clean crates.
struct StageCopies<'a> {
    rule: &'static str,
    reports: &'a BTreeMap<(usize, usize), Report>,
    documented: &'a BTreeMap<&'static str, usize>,
    verdicts: &'a BTreeMap<(usize, usize), &'static str>,
    /// The reports of the copies of the config's analysis, and the items whose copy of it is not
    /// built or does not compile.
    base_reports: &'a BTreeMap<(usize, usize), Report>,
    base_failed: &'a BTreeSet<(usize, usize)>,
    /// What explains each failure of those: its class, or `NEW` if no listed failure does.
    base_verdicts: &'a BTreeMap<(usize, usize), &'static str>,
    /// The items whose clean copy does not compile, after the result and after the config, and the
    /// verdict of the first: `SHARED` (the clean copy after the config fails too) or `NEW`.
    clean_failed: &'a [BTreeSet<(usize, usize)>; 2],
    clean_verdicts: &'a BTreeMap<(usize, usize), &'static str>,
}

/// What a report says of a copy that was made: the status, and what the macro counted.
fn made(report: &Report) -> bool {
    (report.accepted || report.allowed) && report.rewrite_error.is_none()
}

impl StageCopies<'_> {
    /// Whether the macro made the copy of the item, and it was built in the Referrer's crate.
    fn built(&self, case: usize, item: usize) -> bool {
        self.reports.get(&(case, item)).map_or(false, made)
    }

    /// What became of the copy of the config's analysis of an item.
    fn base(&self, case: usize, item: usize) -> String {
        match (
            self.base_failed.contains(&(case, item)),
            self.base_verdicts.get(&(case, item)),
        ) {
            (false, _) => "the copy of the config's analysis compiles".to_string(),
            (true, Some(&"NEW")) | (true, None) => {
                "the copy of the config's analysis does not compile either, and no listed failure explains it".to_string()
            }
            (true, Some(class)) => format!(
                "the copy of the config's analysis does not compile either, the failure {}",
                class
            ),
        }
    }

    /// What became of the copy of an item.
    fn verdict(&self, case: usize, item: usize) -> String {
        if !self.built(case, item) {
            return "no copy was made".to_string();
        }
        let base = self.base(case, item);
        match self.verdicts.get(&(case, item)) {
            None => format!("compiles ({})", base),
            Some(&"NEW") => format!("DOES NOT COMPILE, a new failure ({})", base),
            Some(&"SHARED") => format!(
                "does not compile, and no listed failure explains it ({})",
                base
            ),
            Some(class) => format!(
                "does not compile, the failure {} of the docs of allow() ({})",
                class, base
            ),
        }
    }

    /// Whether the copy of the item compiles.
    fn compiles(&self, case: usize, item: usize) -> bool {
        self.built(case, item) && !self.verdicts.contains_key(&(case, item))
    }

    /// The lines of the summary about the copies.
    fn summarize(&self) -> String {
        let mut s = String::new();
        let count = |reports: &BTreeMap<(usize, usize), Report>| {
            let (mut ok, mut acc, mut failed, mut errors) = (0, 0, 0, 0);
            let (mut entries, mut positions, mut checked) = (0, 0, 0);
            for report in reports.values() {
                ok += usize::from(report.accepted);
                acc += usize::from(report.allowed);
                failed += usize::from(report.accept_failed);
                errors += usize::from(report.rewrite_error.is_some());
                if made(report) {
                    entries += report.entries;
                    positions += report.positions;
                    checked += report.checked;
                }
            }
            (ok, acc, failed, errors, entries, positions, checked)
        };
        let matters = |reports: &BTreeMap<(usize, usize), Report>| -> usize {
            reports.values().filter(|report| !report.skip).count()
        };
        let (ok, acc, failed, errors, entries, positions, checked) = count(self.reports);
        let built = self
            .reports
            .keys()
            .filter(|(case, item)| self.built(*case, *item))
            .count();
        writeln!(
            s,
            "  allow_stages [{}] copies after the result: {} of {} built in the Referrer's crates ({} with no error left by the rule, {} with the causes of those allowed, {} still rejected, {} not made by the macro); entries: {}, type positions: {}, witnesses checked: {}",
            self.rule, built, matters(self.reports), ok, acc, failed, errors, entries, positions, checked
        )
        .unwrap();
        let (ok, acc, failed, errors, entries, _, _) = count(self.base_reports);
        let base_built = self
            .base_reports
            .iter()
            .filter(|(_, report)| made(report))
            .count();
        writeln!(
            s,
            "  allow_stages [{}] copies after the config: {} of {} built ({} with no error, {} with the causes of the errors allowed, {} still rejected, {} not made by the macro); entries: {}; {} of them do not compile",
            self.rule,
            base_built,
            matters(self.base_reports),
            ok,
            acc,
            failed,
            errors,
            entries,
            self.base_failed
                .iter()
                .filter(|(case, item)| self
                    .base_reports
                    .get(&(*case, *item))
                    .map_or(false, made))
                .count()
        )
        .unwrap();
        // The failures of the copies of the result, by what explains them and by what became of the
        // copy of the config's analysis (when it compiles, the path rule after the analysis is what
        // makes the copy of the result fail).
        let mut rows: BTreeMap<(&str, String), usize> = BTreeMap::new();
        for ((case, item), class) in self.verdicts {
            *rows.entry((*class, self.base(*case, *item))).or_default() += 1;
        }
        writeln!(
            s,
            "  allow_stages [{}] copies after the result that do not compile: {} of {} ({} documented in the docs of allow(): {}; {} not listed and failing the config's copy too (SHARED); {} new)",
            self.rule,
            self.verdicts.len(),
            built,
            self.documented.values().sum::<usize>(),
            self.documented
                .iter()
                .map(|(class, n)| format!("{} {}", n, class))
                .collect::<Vec<_>>()
                .join(", "),
            self.verdicts.values().filter(|v| **v == "SHARED").count(),
            self.verdicts.values().filter(|v| **v == "NEW").count()
        )
        .unwrap();
        for ((class, base), n) in rows {
            writeln!(s, "    {:>3}x {:<6} {}", n, class, base).unwrap();
        }
        // The clean crates copy the items that no error is left in, after the result and after the
        // config (the status `ok`), with none of the Leaker's names.
        let clean_made = |reports: &BTreeMap<(usize, usize), Report>| -> usize {
            reports
                .values()
                .filter(|report| report.accepted && report.rewrite_error.is_none())
                .count()
        };
        writeln!(
            s,
            "  allow_stages [{}] clean copies (the rule leaves no error, none of the Leaker's names): {} built after the result, {} do not compile ({} SHARED with the clean copy after the config, {} new); {} built after the config, {} do not compile",
            self.rule,
            clean_made(self.reports),
            self.clean_failed[0].len(),
            self.clean_verdicts.values().filter(|v| **v == "SHARED").count(),
            self.clean_verdicts.values().filter(|v| **v == "NEW").count(),
            clean_made(self.base_reports),
            self.clean_failed[1].len()
        )
        .unwrap();
        s
    }
}

impl Stages {
    /// `MATTERS\u{1f}PLAIN\u{1f}BEFORE\u{1f}AFTER\u{1f}BEFORE_CAUSES\u{1f}AFTER_CAUSES\u{1f}SAME_ERRORS\u{1f}SAME_REFERRER`
    /// after the id of the item.
    fn parse(case: usize, row: &str) -> Option<Self> {
        let (item, fields) = row.split_once('|')?;
        let mut fields = fields.split('\u{1f}');
        let flag = |text: Option<&str>| text.map(|text| text == "1");
        let matters = flag(fields.next())?;
        let mut count = || fields.next()?.parse().ok();
        let (plain, before, after) = (count()?, count()?, count()?);
        Some(Stages {
            case,
            item: item.parse().ok()?,
            matters,
            plain,
            before,
            after,
            before_causes: fields.next()?.to_string(),
            after_causes: fields.next()?.to_string(),
            same_errors: flag(fields.next())?,
            same_referrer: flag(fields.next())?,
        })
    }

    /// The lines of the summary of a rule: how many items have the same number of errors at both
    /// stages, and each that differs, with its causes and what became of its copy.
    fn summarize(
        rule: &str,
        stages: &[Stages],
        generated: &[Case],
        ws: &Path,
        copies: &StageCopies<'_>,
    ) -> String {
        let mut s = String::new();
        let differ: Vec<&Stages> = stages.iter().filter(|s| s.before != s.after).collect();
        let with_errors = stages.iter().filter(|s| s.plain > 0).count();
        let mattering = stages.iter().filter(|s| s.matters).count();
        writeln!(
            s,
            "  allow_stages [{}] (TYPE_LEAK_PROP_FEATURES): {} items compared, {} where the rule matters, {} of them have errors without it; config and result leave the same number of errors in {}, a different one in {}; errors: {} without the rule, {} with it in the config, {} after it in the result",
            rule,
            stages.len(),
            mattering,
            with_errors,
            stages.len() - differ.len(),
            differ.len(),
            stages.iter().map(|s| s.plain).sum::<usize>(),
            stages.iter().map(|s| s.before).sum::<usize>(),
            stages.iter().map(|s| s.after).sum::<usize>()
        )
        .unwrap();
        if exact_rule(rule) {
            let apart = stages
                .iter()
                .filter(|s| !s.same_errors || !s.same_referrer)
                .count();
            writeln!(
                s,
                "  allow_stages [{}] config and result give identical errors and an equal Referrer in {} items, differ in {}",
                rule,
                stages.len() - apart,
                apart
            )
            .unwrap();
        }
        s += &copies.summarize();
        // Of the items that differ: those where the result leaves more errors than the config and
        // those where it leaves fewer, and how many of their copies compile.
        for (what, wanted) in [("more", true), ("fewer", false)] {
            let of_kind: Vec<&&Stages> = differ
                .iter()
                .filter(|stage| (stage.after > stage.before) == wanted)
                .collect();
            let compiled = of_kind
                .iter()
                .filter(|stage| copies.compiles(stage.case, stage.item))
                .count();
            let base_compiled = of_kind
                .iter()
                .filter(|stage| !copies.base_failed.contains(&(stage.case, stage.item)))
                .count();
            writeln!(
                s,
                "  allow_stages [{}]: the result leaves {} errors than the config in {} items, the copy after the result compiles in {} of them, and the copy after the config in {}",
                rule,
                what,
                of_kind.len(),
                compiled,
                base_compiled
            )
            .unwrap();
        }
        for stage in differ {
            let kind = generated[stage.case]
                .items
                .iter()
                .find(|item| item.id == stage.item)
                .map_or("?", |item| item.kind);
            writeln!(
                s,
                "  allow_stages [{}] DIFFERS: case {} item {} ({}): {} errors without the rule, {} in the config, {} in the result\n    config: {}\n    result: {}\n    copy: {}\n    sources: {}",
                rule,
                stage.case,
                stage.item,
                kind,
                stage.plain,
                stage.before,
                stage.after,
                stage.before_causes,
                stage.after_causes,
                copies.verdict(stage.case, stage.item),
                ws.join(leaker_dir(stage.case)).join("src/lib.rs").display()
            )
            .unwrap();
        }
        s
    }
}

/// What the harness gathers for one rule of `allow_stages`: what the macros counted of each item, and
/// their reports of the copies after the result (`0`) and after the config (`1`).
#[derive(Default)]
struct RuleRun {
    stages: Vec<Stages>,
    reports: [BTreeMap<(usize, usize), Report>; 2],
}

/// The verdicts on the copies of one rule, from the diagnostics of the Referrer's crates
/// (`RuleResult::judge()`).
#[derive(Default)]
struct RuleResult {
    /// The copies after the result that do not compile for a failure the docs of `allow()` list,
    /// by class.
    documented: BTreeMap<&'static str, usize>,
    /// The class of the failure of each item whose copy after the result does not compile: a listed
    /// one, `SHARED` (the copy after the config fails too), or `NEW`.
    verdicts: BTreeMap<(usize, usize), &'static str>,
    /// The items whose copy after the config is not built or does not compile, and what explains it.
    base_failed: BTreeSet<(usize, usize)>,
    base_verdicts: BTreeMap<(usize, usize), &'static str>,
    /// The items whose clean copy does not compile, after the result and after the config, and the
    /// verdict of the first: `SHARED` or `NEW`.
    clean_failed: [BTreeSet<(usize, usize)>; 2],
    clean_verdicts: BTreeMap<(usize, usize), &'static str>,
}

impl RuleResult {
    /// Nothing failed to compile.
    fn is_empty(&self) -> bool {
        self.verdicts.is_empty()
            && self.base_failed.is_empty()
            && self.clean_failed.iter().all(BTreeSet::is_empty)
    }

    /// Judges the diagnostics of the crates of `rule` (taken out of `diags`): the copies that do not
    /// compile are explained (`explain_copies()`), those after the config are the controls, and the
    /// failures that are defects are returned: a failure of the copy after the result that the docs
    /// of `allow()` do not list while the copy after the config compiles, a clean copy that does not
    /// compile while the clean copy after the config does, and a rule that is exact
    /// (`exact_rule()`) whose config and result differ.
    fn judge(
        rule: &'static str,
        run: &RuleRun,
        diags: &mut Vec<Failure>,
        generated: &[Case],
    ) -> (RuleResult, Vec<Failure>) {
        let mut result = RuleResult::default();
        let mut problems = Vec::new();
        let mut take = |side: &str, clean: bool| -> Vec<Failure> {
            let (mine, rest): (Vec<Failure>, Vec<Failure>) =
                std::mem::take(diags).into_iter().partition(|failure| {
                    failure.stage.map_or(false, |k| {
                        k.rule == rule && k.side == side && k.clean == clean
                    })
                });
            *diags = rest;
            mine
        };
        let (cfg_glob, res_glob) = (take("cfg", false), take("res", false));
        let (cfg_clean, res_clean) = (take("cfg", true), take("res", true));

        // The copies that the copies after the result are compared with: those that are not built, or
        // that do not compile, are the items whose failure the rule after the analysis does not cause.
        let mut base_named = Vec::new();
        for failure in cfg_glob {
            match failure.item {
                Some(id) => {
                    result.base_failed.insert((failure.case, id));
                    base_named.push(failure);
                }
                None => problems.push(Failure {
                    class: "STAGES BASE: a diagnostic of the copy of the config's analysis names no item",
                    ..failure
                }),
            }
        }
        for ((case, id), report) in &run.reports[1] {
            if !report.skip
                && (!(report.accepted || report.allowed) || report.rewrite_error.is_some())
            {
                result.base_failed.insert((*case, *id));
            }
        }
        // They are explained like the others, but a failure that is not listed fails nothing: it is a
        // limitation of `allow()` after the config's analysis, which the copies after the result are
        // not about.
        let (_, _, base_verdicts) = explain_copies(
            base_named,
            &run.reports[1],
            generated,
            &STAGE_CLASSES,
            &BTreeSet::new(),
            true,
        );
        result.base_verdicts = base_verdicts;
        let (documented, stage_problems, verdicts) = explain_copies(
            res_glob,
            &run.reports[0],
            generated,
            &STAGE_CLASSES,
            &result.base_failed,
            true,
        );
        result.documented = documented;
        result.verdicts = verdicts;
        problems.extend(stage_problems);

        // The clean crates: a copy that does not compile although the rule left no error is a failure
        // of the rule, unless the clean copy after the config fails too (SHARED).
        for failure in cfg_clean {
            match failure.item {
                Some(id) => {
                    result.clean_failed[1].insert((failure.case, id));
                }
                None => problems.push(Failure {
                    class: "STAGES CLEAN: a diagnostic of the clean copy of the config's analysis names no item",
                    ..failure
                }),
            }
        }
        let mut by_item: BTreeMap<(usize, Option<usize>), Vec<Failure>> = BTreeMap::new();
        for failure in res_clean {
            by_item
                .entry((failure.case, failure.item))
                .or_default()
                .push(failure);
        }
        for ((case, id), group) in by_item {
            let texts = group
                .iter()
                .map(|f| f.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let Some(id) = id else {
                problems.push(Failure {
                    class: "STAGES CLEAN: a diagnostic of the clean copy names no item",
                    text: texts,
                    ..group.into_iter().next().unwrap()
                });
                continue;
            };
            result.clean_failed[0].insert((case, id));
            if result.clean_failed[1].contains(&(case, id)) {
                result.clean_verdicts.insert((case, id), "SHARED");
                continue;
            }
            result.clean_verdicts.insert((case, id), "NEW");
            let config = match run.reports[1].get(&(case, id)) {
                Some(report) if report.accepted && report.rewrite_error.is_none() => {
                    "the clean copy after the config compiles"
                }
                Some(report) if report.skip => "the rule does not matter after the config",
                Some(_) => "no clean copy was made after the config: the rule left errors there",
                None => "the macro reported nothing of the copy after the config",
            };
            problems.push(Failure {
                case,
                item: Some(id),
                class: CLEAN_NEW,
                code: String::new(),
                text: format!("{}\n\n{}", config, texts),
                stage: group[0].stage,
            });
        }

        if exact_rule(rule) {
            for stage in &run.stages {
                if stage.before != stage.after || !stage.same_errors || !stage.same_referrer {
                    problems.push(Failure {
                        case: stage.case,
                        item: Some(stage.item),
                        class: "STAGES DIFFER: the config and the result give other errors or another Referrer, for a rule that is a cause or a place",
                        code: String::new(),
                        text: format!(
                            "rule {}: {} errors in the config, {} in the result; the same errors: {}; an equal Referrer: {}\n    config: {}\n    result: {}",
                            rule,
                            stage.before,
                            stage.after,
                            stage.same_errors,
                            stage.same_referrer,
                            stage.before_causes,
                            stage.after_causes
                        ),
                        stage: None,
                    });
                }
            }
        }
        (result, problems)
    }
}

/// `TYPE_LEAK_PROP_ACCEPT` is set (and is not `0`).
fn accept_mode() -> bool {
    std::env::var("TYPE_LEAK_PROP_ACCEPT").map_or(false, |value| !matches!(value.trim(), "" | "0"))
}

/// A short reason of a rejection, for the statistics.
fn reason_head(detail: &str) -> String {
    // The cycles of a bound that names a leaked type (F2) are counted together.
    if detail.contains("in a bound or where predicate") {
        return "a leaked type in a bound or where predicate that rustc cannot normalize"
            .to_string();
    }
    // An entry in the self type of an impl whose context depends on `Self` (F13) is counted
    // together.
    if detail.contains("is in the self type of the impl") {
        return "a type in the self type of an impl that a bound of the impl makes depend on `Self`"
            .to_string();
    }
    // "use absolute path: 'Vec < X >'" is counted by the start of the type.
    match detail.find('\'') {
        Some(at) => {
            let ty: String = detail[at + 1..].chars().take(24).collect();
            format!("{} '{}", detail[..at].trim(), ty)
        }
        None => detail.chars().take(60).collect(),
    }
}
// ---------------------------------------------------------------------------------------------
// Every rejection is explained.
// ---------------------------------------------------------------------------------------------

/// The traits of the prelude that the generator spells bare in bounds (`gen::absolute_trait()`),
/// the names of the tags `bare:<Name>`.
const BARE_TRAITS: &[&str] = &[
    "Sized",
    "Send",
    "Sync",
    "Unpin",
    "Copy",
    "Clone",
    "Default",
    "Fn",
    "FnMut",
    "FnOnce",
    "Into",
    "From",
    "AsRef",
    "Iterator",
    "PartialEq",
    "PartialOrd",
    "Eq",
    "Ord",
];

/// A reason for which `type-leak` may reject an item: what it is, and the shapes of an item (its
/// tags, see `gen::Item::tags`) of which the item needs one to justify it.
#[derive(Debug, PartialEq)]
struct Family {
    name: &'static str,
    tags: Vec<String>,
}

fn family_of(name: &'static str, tags: &[&str]) -> Option<Family> {
    Some(Family {
        name,
        tags: tags.iter().map(|tag| tag.to_string()).collect(),
    })
}

/// The reason of a rejection among the ones that the generator knows (the allow-list), or `None`.
///
/// A rejection counts as a success of the property, so one must not hide a defect: each error of
/// it is one of these, and the generator has put its shape on the item (`family().tags`). A new
/// reason fails the run with the item and the message, to be judged: a limitation to list, or a
/// defect.
fn family(error: &Rejection) -> Option<Family> {
    let message = error.message.as_str();
    if message.contains("names `Self` in a bound or where predicate") {
        return family_of(
            "a leaked type that names `Self` in a bound or where predicate of its trait",
            &[gen::TAG_BOUND_CYCLE, gen::TAG_SELF_IN_BOUNDS],
        );
    }
    if message.contains("is in a bound or where predicate") {
        return family_of(
            "a leaked type in a bound or where predicate that rustc cannot normalize",
            &[gen::TAG_BOUND_CYCLE],
        );
    }
    if message.contains("is in the self type of the impl") {
        return family_of(
            "a type in the self type of an impl whose context depends on `Self` through a bound",
            &[gen::TAG_SELF_IN_BOUNDS, gen::TAG_BOUND_CYCLE],
        );
    }
    if message.contains("has a generic whose bound mentions `Self`") {
        return family_of(
            "a type that uses a generic whose bound mentions a `Self` that cannot be leaked",
            &[
                gen::TAG_BOUND_CYCLE,
                gen::TAG_SELF_IN_BOUNDS,
                gen::TAG_SELF_DENIED,
            ],
        );
    }
    if message.starts_with("write `<Self as ") {
        return family_of("`Self::A` where `Self` is a type", &[gen::TAG_SELF_ASSOC]);
    }
    if message.contains("leaves the lifetime") {
        return family_of(
            "a lifetime of a method that only a leaked type keeps (E0195)",
            &[gen::TAG_METHOD_LIFETIME],
        );
    }
    // A bare name (`u32`, `T`, `R0`) as an argument of a type that cannot be leaked as a whole:
    // it may be a const, so it is not leaked alone. The note says why the holder cannot be leaked,
    // a reason that is an error of the item too.
    if message.contains("may be a const, and the type that holds it cannot be leaked as a whole") {
        let tags: &[&str] = if error
            .note
            .contains("which the module declares and the copy declares again")
        {
            &[gen::TAG_MODULE_SCOPE]
        } else if error.note.contains("because of `'") {
            &[gen::TAG_METHOD_LIFETIME, gen::TAG_BINDER_NO_ANCHOR]
        } else {
            &[
                gen::TAG_SELF_ASSOC,
                gen::TAG_SELF_DENIED,
                gen::TAG_SELF_IN_BOUNDS,
                gen::TAG_BOUND_CYCLE,
            ]
        };
        return family_of(
            "a bare name that may be a const, in a type that cannot be leaked as a whole",
            tags,
        );
    }
    if !message.starts_with("use absolute path: '") {
        return None;
    }
    // The tokens of the path: a bound with a binder starts with it (`for<'h>Fn(..)`).
    let text: String = error.tokens.replace(char::is_whitespace, "");
    let bound = match text.strip_prefix("for<") {
        Some(rest) => rest.split_once('>').map_or(rest, |(_, rest)| rest),
        None => text.as_str(),
    };
    // A path can be several reasons at once (a bare `Fn` bound with a binder, a type with `Self`
    // and a binder): the shapes that the item has for any of them explain it.
    let mut reasons: Vec<(&'static str, Vec<String>)> = Vec::new();
    for name in BARE_TRAITS {
        for spelled in [name.to_string(), format!("?{}", name)] {
            let is_it = bound.strip_prefix(spelled.as_str()).map_or(false, |rest| {
                !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_')
            });
            if is_it {
                reasons.push((
                    "a bare trait of the prelude, where the Leaker does not allow the prelude",
                    vec![format!("bare:{}", spelled)],
                ));
            }
        }
    }
    // `Rl<'h5>`: a type of the module that names a lifetime of a binder (not the `for<'h5>` of one).
    if text.match_indices("<'h").any(|(at, _)| {
        !text[..at].ends_with("for") && text[at + 3..].starts_with(|c: char| c.is_ascii_digit())
    }) {
        reasons.push((
            "a type that names the lifetime of a binder that no input keeps",
            vec![gen::TAG_BINDER_NO_ANCHOR.to_string()],
        ));
    }
    // A path that names a lifetime of a method (`it cannot be leaked as a whole because of 'x`), and
    // that cannot be split either (a part of it must be leaked on its own, as a primitive type
    // that is not allowed, or a trait object takes the lifetime of the reference around it).
    if error
        .note
        .starts_with("it cannot be leaked as a whole because of `'")
    {
        reasons.push((
            "a path that names a lifetime of a method or a binder and cannot be split",
            vec![
                gen::TAG_METHOD_LIFETIME.to_string(),
                gen::TAG_BINDER_NO_ANCHOR.to_string(),
            ],
        ));
    }
    // `Self::A0`, `Tr<'a, T>::A0`, `Tr<'a>::G<'c>`: an associated type of the trait.
    let assoc = text
        .match_indices("::A")
        .any(|(at, _)| text[at + 3..].starts_with(|c: char| c.is_ascii_digit()))
        || text.contains("::G<");
    if assoc {
        reasons.push((
            "`Self::A` where `Self` is a type",
            vec![gen::TAG_SELF_ASSOC.to_string()],
        ));
    }
    // A type that names `Self`, where `allow_self` is false.
    if text.contains("Self") {
        reasons.push((
            "a type that names `Self`, which the item does not allow",
            vec![
                gen::TAG_SELF_ASSOC.to_string(),
                gen::TAG_SELF_DENIED.to_string(),
            ],
        ));
    }
    // A relative path in a value (`crate::base::bf`, `RLEN`), which the Leaker reads with
    // `check_values()` and copies as written: the note says which value (`values`).
    if error.note.contains("which the copy writes as it is") {
        reasons.push((
            "a relative path in a value, which the copy writes as it is",
            vec![gen::TAG_VALUE_PATH.to_string()],
        ));
    }
    // `RLEN`, `crate::base::BLEN`: a constant, a value, which is not leaked: in the default of a
    // parameter, or in the length of an array of a type that is split into parts because it
    // names the `Self` that the item does not allow.
    if text.ends_with("LEN") {
        reasons.push((
            "a relative constant that is not leaked as a whole",
            vec![
                gen::TAG_CONST_DEFAULT.to_string(),
                gen::TAG_SELF_DENIED.to_string(),
            ],
        ));
    }
    let name = reasons.first()?.0;
    let mut tags: Vec<String> = reasons.into_iter().flat_map(|(_, tags)| tags).collect();
    tags.sort();
    tags.dedup();
    Some(Family { name, tags })
}

/// What is wrong with the way a rejection is explained, one line for each error: it is not one of
/// the reasons that are known, or the item has no shape that gives it.
fn unexplained(tags: &std::collections::BTreeSet<String>, report: &Report) -> Vec<String> {
    if report.rejections.is_empty() {
        return vec!["the report of the rejection has no error".to_string()];
    }
    let mut problems = Vec::new();
    // The shapes of `self_bound_adts` (F41, F42) are accepted: no reason explains a rejection but a
    // bare trait of the prelude, where the Leaker does not allow the prelude.
    let must_accept = tags.contains(gen::TAG_SELF_BOUND_ADT);
    for error in &report.rejections {
        let family = family(error);
        if must_accept
            && !family
                .as_ref()
                .map_or(false, |f| f.name.starts_with("a bare trait"))
        {
            problems.push(format!(
                "SHOULD BE ACCEPTED: a struct, enum or union with a bound that names its `Self` \
                 and a default that uses the generic: {} [tokens: {}]",
                error.message, error.tokens
            ));
            continue;
        }
        match family {
            None => problems.push(format!(
                "NOT A KNOWN REASON: {} [tokens: {}] [note: {}]",
                error.message, error.tokens, error.note
            )),
            Some(family) => {
                if !family.tags.iter().any(|tag| tags.contains(tag)) {
                    problems.push(format!(
                        "NO SHAPE THAT GIVES IT: {} [tokens: {}]: {}, and the item has none of the tags {:?} (its tags are {:?})",
                        error.message, error.tokens, family.name, family.tags, tags
                    ));
                }
            }
        }
    }
    problems
}

// ---------------------------------------------------------------------------------------------
// Every failure of an accepted copy is explained (`TYPE_LEAK_PROP_ACCEPT`).
// ---------------------------------------------------------------------------------------------

/// A failure of the copy of an item accepted by `allow()` that the docs of `LeakerResult::allow()`
/// list ("What accepting does not make compile"): which one, what it is, and the shapes an item
/// needs to have it (`tags`: the item needs one tag of each list).
#[derive(Debug, PartialEq)]
struct AcceptFamily {
    /// The number of the shape in the docs: `(i)`, `(ii)` or `(iii)`.
    class: &'static str,
    name: &'static str,
    tags: Vec<Vec<String>>,
}

fn accept_family(class: &'static str, name: &'static str, tags: &[&[&str]]) -> AcceptFamily {
    AcceptFamily {
        class,
        name,
        tags: tags
            .iter()
            .map(|any| any.iter().map(|tag| tag.to_string()).collect())
            .collect(),
    }
}

/// The failures that explain a diagnostic of the copy of an item accepted by `allow()`: the error
/// code of rustc, and the errors that were allowed (their `Reason`, and the kind and the tokens of
/// their cause). The generator has put the shapes on the item (`tags`).
///
/// - (i) `Self::A`, or `Self` as a type, in a predicate whose context has a bound that names a
///   leaked type (`UnprovableBound`, the cause is the type): E0277. Only the cause is copied as
///   written, and the bound stays leaked. The item has a predicate that names `Self`
///   (`self_in_predicate`) and a bound that names a leaked type (`bound_cycle`, or `relative_std`).
/// - (ii) The same cause with an alias bound (`T: Fn(<T as Iterator>::Item) -> u8` next to a bound
///   that fixes the `Item`): E0277 too, but rustc cannot implement the trait with a generic impl at
///   all, so nothing makes it compile. The item has the alias bound (`alias_bound`).
/// - (iii) `Self` with `allow_self = false` (`RelativePath`, the cause is `Self`) in a function
///   pointer whose elided output takes the lifetime of an input that a path hides:
///   `Visitor::with_self()` replaces `Self` by a type with lifetimes inside the accepted part, and
///   the elision fails (E0106). The item denies `Self` (`self_denied`) and has the function pointer
///   (`hidden_self`).
///
/// Anything else is a defect, or a new limitation to list in the docs of `allow()` and here, with
/// its shape.
fn accept_families(code: &str, allowed: &[Rejection]) -> Vec<AcceptFamily> {
    let cause = |reason: &str, matches: &dyn Fn(&str) -> bool| {
        allowed.iter().any(|error| {
            let tokens: String = error.tokens.replace(char::is_whitespace, "");
            error.reason == reason && error.kind == "Type" && matches(&tokens)
        })
    };
    let self_assoc = cause("UnprovableBound", &|t| t.starts_with("Self::"));
    let self_type = cause("UnprovableBound", &|t| t == "Self");
    let mut families = Vec::new();
    match code {
        "E0277" if self_assoc || self_type => {
            if self_assoc {
                families.push(accept_family(
                    "(ii)",
                    "`Self::A` in a predicate of a trait with an alias bound, which no impl can have",
                    &[&[gen::TAG_ALIAS_BOUND]],
                ));
            }
            // The leaked type of the bound may be a path of `std` that the Leaker does not allow
            // (`relative_std`).
            families.push(accept_family(
                "(i)",
                "`Self::A` or `Self` in a predicate whose context has a bound that names a leaked type",
                &[
                    &[gen::TAG_SELF_IN_PREDICATE],
                    &[gen::TAG_BOUND_CYCLE, gen::TAG_RELATIVE_STD],
                ],
            ));
        }
        "E0106" if cause("RelativePath", &|t| t == "Self") => {
            families.push(accept_family(
                "(iii)",
                "`Self` accepted in a function pointer whose elided output takes the lifetime of a hidden one, and `with_self()` replaces it",
                &[&[gen::TAG_SELF_DENIED], &[gen::TAG_HIDDEN_SELF]],
            ));
        }
        _ => (),
    }
    families
}

/// The errors that were allowed, on one line.
fn describe_allowed(allowed: &[Rejection]) -> String {
    allowed
        .iter()
        .map(|r| format!("{} ({}) [tokens: {}]", r.reason, r.kind, r.tokens))
        .collect::<Vec<_>>()
        .join("; ")
}

/// What is wrong with the way the failures of the copy of an item accepted by `allow()` are
/// explained, one line each, or the class of the first failure if all are: `codes` are the error
/// codes of the diagnostics of rustc for the item. A diagnostic that is not a failure the docs list
/// ("NOT A KNOWN FAILURE"), or that the item has no shape for ("NO SHAPE THAT GIVES IT"), is a
/// defect or a limitation to list.
fn unexplained_accept(
    tags: &std::collections::BTreeSet<String>,
    codes: &[&str],
    report: &Report,
) -> Result<&'static str, Vec<String>> {
    let mut problems = Vec::new();
    let mut class = "";
    for code in codes {
        let families = accept_families(code, &report.rejections);
        if families.is_empty() {
            problems.push(format!(
                "NOT A KNOWN FAILURE: {} on an item accepted by allow() for these errors: {}",
                code,
                describe_allowed(&report.rejections)
            ));
            continue;
        }
        let has_shape = |family: &AcceptFamily| {
            family
                .tags
                .iter()
                .all(|any| any.iter().any(|tag| tags.contains(tag)))
        };
        match families.iter().find(|family| has_shape(family)) {
            Some(family) => class = if class.is_empty() { family.class } else { class },
            None => problems.push(format!(
                "NO SHAPE THAT GIVES IT: {} on an item accepted by allow() for these errors: {}; it is {}, and the item has none of the tags in {:?} (its tags are {:?})",
                code,
                describe_allowed(&report.rejections),
                families
                    .iter()
                    .map(|family| format!("{} {}", family.class, family.name))
                    .collect::<Vec<_>>()
                    .join(" or "),
                families.iter().map(|family| &family.tags).collect::<Vec<_>>(),
                tags
            )),
        }
    }
    if problems.is_empty() {
        Ok(class)
    } else {
        Err(problems)
    }
}

/// The classes of the failures of the copies of one kind (`explain_copies()`).
struct CopyClasses {
    /// A diagnostic that no failure that the docs of `allow()` list explains.
    unknown: &'static str,
    /// A listed failure, on an item that has not the shape.
    no_shape: &'static str,
    /// The diagnostic names no item for which an error was allowed by its cause.
    none: &'static str,
}

const ACCEPT_CLASSES: CopyClasses = CopyClasses {
    unknown: "ACCEPT: NOT A KNOWN FAILURE: the copy of an item accepted by allow() does not compile for a reason that the docs of allow() do not list",
    no_shape: "ACCEPT: NO SHAPE THAT GIVES IT: a failure of an accepted copy that the docs of allow() list, on an item that has not the shape",
    none: "the diagnostic names no item that was accepted by allow()",
};

const STAGE_CLASSES: CopyClasses = CopyClasses {
    unknown: "STAGES: NEW FAILURE: the copy made after the analysis without the rule does not compile, the copy of the config's analysis does, and no failure that the docs of allow() list explains it",
    no_shape: "STAGES: NO SHAPE THAT GIVES IT: a failure of a stage copy that the docs of allow() list, on an item that has not the shape, and the copy of the config's analysis compiles",
    none: "the diagnostic names no item for which the cause of an error that the rule leaves was allowed, and the copy of the config's analysis compiles",
};

/// A clean copy (the rule left no error, the crate has none of the Leaker's names but the allowed
/// ones) that does not compile while the clean copy of the config's analysis does: the promise of the
/// rule after the analysis does not hold, a part kept as written names more than the rule allows.
const CLEAN_NEW: &str = "STAGES CLEAN: NEW FAILURE: the copy made after the analysis without the rule does not compile in the clean crate although the rule left no error, and the clean copy of the config's analysis compiles or was not made";

/// Explains the diagnostics of the copies that a macro made of items for which causes were allowed
/// (`failures` are the diagnostics of the Referrer's crate of those, `reports` what the macros
/// reported of the items): each is one of the failures that the docs of `LeakerResult::allow()` list
/// on an item that has the shape that gives it (`unexplained_accept()`), or a defect that fails the
/// run, unless the item is one of `shared`: its copy of the other analysis does not compile either,
/// so the failure is not about what is compared (verdict `SHARED`). Returns how many items fail by
/// the class of the failure, the failures that are defects, and the verdict of each item that does
/// not compile (its class, `SHARED`, or `NEW`).
#[allow(clippy::type_complexity)]
fn explain_copies(
    copies: Vec<Failure>,
    reports: &BTreeMap<(usize, usize), Report>,
    generated: &[Case],
    classes: &CopyClasses,
    shared: &BTreeSet<(usize, usize)>,
    stage: bool,
) -> (
    BTreeMap<&'static str, usize>,
    Vec<Failure>,
    BTreeMap<(usize, usize), &'static str>,
) {
    let mut documented: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut verdicts = BTreeMap::new();
    let mut by_item: BTreeMap<(usize, Option<usize>), Vec<Failure>> = BTreeMap::new();
    for failure in copies {
        by_item
            .entry((failure.case, failure.item))
            .or_default()
            .push(failure);
    }
    for ((case, id), group) in by_item {
        // The copy of `allow_stages` is explained by the errors of the analysis without the rule, the
        // ones the rule dropped as well as those that remain.
        let report = id.and_then(|id| reports.get(&(case, id))).filter(|report| {
            if stage {
                report.allowed || report.accepted
            } else {
                report.allowed
            }
        });
        let item = id.and_then(|id| generated[case].items.iter().find(|item| item.id == id));
        let verdict = match (report, item) {
            (Some(report), Some(item)) => {
                let mut tags = item.tags.clone();
                if item.kind == "mod" || item.scope == "mod" {
                    tags.insert(gen::TAG_MODULE_SCOPE.to_string());
                }
                let codes: Vec<&str> = group.iter().map(|f| f.code.as_str()).collect();
                if stage {
                    // (The errors that came up when the rule dropped one are in the second list only.)
                    let mut errors = report.plain.clone();
                    errors.extend(report.rejections.iter().cloned());
                    let explained = Report {
                        rejections: errors,
                        ..report.clone()
                    };
                    unexplained_accept(&tags, &codes, &explained)
                } else {
                    unexplained_accept(&tags, &codes, report)
                }
            }
            _ => Err(vec![format!("NOT A KNOWN FAILURE: {}", classes.none)]),
        };
        match verdict {
            Ok(class) => {
                *documented.entry(class).or_default() += 1;
                if let Some(id) = id {
                    verdicts.insert((case, id), class);
                }
            }
            Err(_) if id.map_or(false, |id| shared.contains(&(case, id))) => {
                verdicts.insert((case, id.unwrap()), "SHARED");
            }
            Err(problems) => {
                if let Some(id) = id {
                    verdicts.insert((case, id), "NEW");
                }
                for problem in problems {
                    let class = if problem.starts_with("NO SHAPE") {
                        classes.no_shape
                    } else {
                        classes.unknown
                    };
                    failures.push(Failure {
                        stage: None,
                        case,
                        item: id,
                        class,
                        code: String::new(),
                        text: format!(
                            "{}\n\n{}",
                            problem,
                            group
                                .iter()
                                .map(|f| f.text.as_str())
                                .collect::<Vec<_>>()
                                .join("\n")
                        ),
                    });
                }
            }
        }
    }
    (documented, failures, verdicts)
}

/// The names of the standard prelude (types, traits, functions and variants) and of the primitive
/// types, which `allow_prelude()` and `allow_primitive()` take for the names of the prelude: the
/// Referrer's crate of the stage copies must not shadow any.
const PRELUDE_NAMES: &[&str] = &[
    "Box",
    "String",
    "Vec",
    "Option",
    "Some",
    "None",
    "Result",
    "Ok",
    "Err",
    "Copy",
    "Send",
    "Sized",
    "Sync",
    "Unpin",
    "Drop",
    "Fn",
    "FnMut",
    "FnOnce",
    "Clone",
    "PartialEq",
    "PartialOrd",
    "Eq",
    "Ord",
    "AsRef",
    "AsMut",
    "Into",
    "From",
    "Default",
    "Iterator",
    "IntoIterator",
    "DoubleEndedIterator",
    "ExactSizeIterator",
    "Extend",
    "FromIterator",
    "ToOwned",
    "ToString",
    "TryFrom",
    "TryInto",
    "drop",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "f32",
    "f64",
    "bool",
    "char",
    "str",
];

/// The names of `PRELUDE_NAMES` that the source of a crate declares (`struct Vec`, `type u8 = ..`,
/// `fn drop`, `mod Option`, ...) or imports (`use a::Vec;`, which is no import of the prelude's
/// own): a crate that does so has the name mean something else than the prelude, and the Referrer's
/// crate of the stage copies, whose globs import what the Leaker's modules declare, would too.
fn shadowed_prelude_names(source: &str) -> Vec<&'static str> {
    const DECLARING: &[&str] = &[
        "struct",
        "enum",
        "union",
        "trait",
        "type",
        "fn",
        "mod",
        "const",
        "static",
        "macro_rules",
    ];
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    // The words with where they start and end.
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (at, c) in source.char_indices() {
        match (start, is_ident(c)) {
            (None, true) => start = Some(at),
            (Some(from), false) => {
                words.push((from, at));
                start = None;
            }
            _ => (),
        }
    }
    if let Some(from) = start {
        words.push((from, source.len()));
    }
    let mut found: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    let mut find = |name: &str| {
        if let Some(prelude) = PRELUDE_NAMES.iter().find(|prelude| **prelude == name) {
            found.insert(*prelude);
        }
    };
    // A name that a keyword declares follows it (`struct Vec`, `macro_rules! Vec`), as opposed to the
    // keyword in a type: `fn(u8)`, `*const u8`, `&'static u8`.
    for pair in words.windows(2) {
        let ((from, to), (next, end)) = (pair[0], pair[1]);
        let between = &source[to..next];
        let spaced = between
            .trim_matches(|c: char| c.is_whitespace() || c == '!')
            .is_empty();
        let before = source[..from].chars().next_back();
        let in_type = matches!(before, Some('*') | Some('\''));
        if DECLARING.contains(&&source[from..to]) && spaced && !in_type {
            find(&source[next..end]);
        }
    }
    // What a `use` brings in: the last segment of each path of the statement, as written, or the
    // name after `as`. `use` is a keyword, so a word `use` starts the imports of a statement.
    for statement in source.split(';') {
        let at = statement
            .match_indices("use")
            .filter(|(at, _)| {
                !statement[..*at].chars().next_back().map_or(false, is_ident)
                    && !statement[at + 3..].chars().next().map_or(false, is_ident)
            })
            .map(|(at, _)| at)
            .last();
        let Some(at) = at else {
            continue;
        };
        let imports = &statement[at + "use".len()..];
        for piece in imports.split(|c: char| matches!(c, '{' | '}' | ',')) {
            let piece = piece.trim();
            let name = match piece.split_once(" as ") {
                Some((_, alias)) => alias.trim(),
                None => piece.rsplit("::").next().unwrap_or("").trim(),
            };
            find(name);
        }
    }
    found.into_iter().collect()
}

// ---------------------------------------------------------------------------------------------
// The watchdog: a run that tested nothing must not pass.
// ---------------------------------------------------------------------------------------------

/// The packages of the generated cases.
fn case_packages(cases: &[Case]) -> HashSet<String> {
    cases
        .iter()
        .enumerate()
        .flat_map(|(i, case)| {
            let mut names = vec![leaker_name(i), referrer_name(i), accept_name(i)];
            names.extend(case.stages.iter().map(|stage| stage_name(i, stage)));
            names
        })
        .collect()
}

/// Why a build tells nothing about the cases, or `None`. A failure of the helper crate
/// (`tests/crates/macro`), of `type-leak`, or of cargo itself is not a finding of a case, and a
/// case that no crate of it was built for would count as "rejected" or "nothing to check".
fn build_problem(build: &Build, known: &HashSet<String>) -> Option<String> {
    if !build.success && build.diags.is_empty() {
        return Some(format!(
            "cargo failed without a compile error:\n{}",
            excerpt(&build.stderr, 60)
        ));
    }
    let helper: Vec<&Diag> = build
        .diags
        .iter()
        .filter(|d| !known.contains(&d.package))
        .collect();
    if helper.is_empty() {
        return None;
    }
    let mut message = String::from(
        "a crate that is not a generated case fails to build (the helper crate \
         `tests/crates/macro`, `type-leak`, or a build script): nothing was tested, and this is \
         not a finding of the property.\n",
    );
    for diag in helper.iter().take(3) {
        write!(
            message,
            "\n--- package `{}`:\n{}\n",
            diag.package,
            excerpt(&diag.rendered, 30)
        )
        .unwrap();
    }
    Some(message)
}

/// What the build left of the case `i` when none of its crates reported an error: the Leaker's
/// executable and the Referrer's library (and, with `accept`, the library of the second Referrer's
/// crate of `TYPE_LEAK_PROP_ACCEPT`, and with `stages` that of `allow_stages`). A missing one is
/// reported, not counted as a rejection.
fn missing_artifacts(
    build: &Build,
    i: usize,
    leaker_ok: bool,
    accept: bool,
    stages: &[gen::StageCrate],
) -> Vec<String> {
    let mut missing = Vec::new();
    if !build.executables.contains_key(&leaker_name(i)) {
        missing.push(format!(
            "no executable was built for `{}`, so the macros reported nothing",
            leaker_name(i)
        ));
    }
    if leaker_ok && !build.artifacts.contains(&referrer_name(i)) {
        missing.push(format!(
            "`{}` was not built, and reported no error",
            referrer_name(i)
        ));
    }
    if leaker_ok && accept && !build.artifacts.contains(&accept_name(i)) {
        missing.push(format!(
            "`{}` was not built, and reported no error",
            accept_name(i)
        ));
    }
    for stage in stages {
        let name = stage_name(i, stage);
        if leaker_ok && !build.artifacts.contains(&name) {
            missing.push(format!("`{}` was not built, and reported no error", name));
        }
    }
    missing
}

// ---------------------------------------------------------------------------------------------
// The statistics and the acceptance floor.
// ---------------------------------------------------------------------------------------------

/// What the macros reported over all the cases.
#[derive(Default)]
struct Stats {
    /// The items generated, by kind (a module counts what it holds too).
    generated: BTreeMap<&'static str, usize>,
    accepted: BTreeMap<&'static str, usize>,
    rejected: BTreeMap<&'static str, usize>,
    /// The same by the scope an item was analyzed through (`gen::SCOPES`, by the name of the API).
    scope_accepted: BTreeMap<&'static str, usize>,
    scope_rejected: BTreeMap<&'static str, usize>,
    /// The reason of each rejection, by kind.
    reasons: BTreeMap<(&'static str, String), usize>,
    analyzed: usize,
    entries: usize,
    positions: usize,
    checked: usize,
    /// The Repeater impls of optional entries that were not written, for the items invoked as
    /// trusting.
    skipped: usize,
    /// The items with a bare trait of the prelude that were compared with their twin (the
    /// prelude is allowed), and the rejections for a bare trait (it is not).
    twins: usize,
    bare_rejected: usize,
    /// The items analyzed where the Leaker does not allow the primitive types.
    primitive_off: usize,
    /// The rejections by their reason (`family()`).
    families: BTreeMap<&'static str, usize>,
    /// The traits of `trait_self_defaults` (F39), by shape (`SELF_DEFAULT_SHAPES`).
    self_defaults: BTreeMap<&'static str, SelfDefaultCount>,
    /// `TYPE_LEAK_PROP_ACCEPT`: the items rejected at the first analysis that the second one
    /// accepted (`allow()` of the cause of every error), by kind, with what the macro wrote for
    /// them, and the ones it rejected again. They are in `rejected` too: the floors are those of the
    /// first analysis.
    allowed: BTreeMap<&'static str, usize>,
    allowed_entries: usize,
    allowed_positions: usize,
    allowed_checked: usize,
    accept_failed: usize,
    /// The items accepted by `allow()` whose copy does not compile for a failure that the docs of
    /// `allow()` list (`accept_families()`), by its class (`(i)`, `(ii)`, `(iii)`).
    documented: BTreeMap<&'static str, usize>,
    /// The errors that were allowed, by `Reason` and the kind of the cause.
    allowed_errors: BTreeMap<(String, String), usize>,
}

/// The shapes of `trait_self_defaults` that the statistics count, with what each is: a trait
/// with a bound that names its `Self` and a later parameter with a default that uses it, the
/// control with a bound that does not, and the traits that the Referrer's crate declares again.
const SELF_DEFAULT_SHAPES: &[(&str, &str)] = &[
    (gen::TAG_SELF_DEFAULT, "default needs Self"),
    (gen::TAG_PLAIN_DEFAULT, "control, no Self"),
    (gen::TAG_REDECLARED, "declared again"),
    // The structs, enums and unions of `self_bound_adts` (F41, F42).
    (gen::TAG_SELF_BOUND_ADT, "adt default, Self"),
    (gen::TAG_SELF_BOUND_TYPE, "adt type names Self"),
];

/// The traits of one shape of `trait_self_defaults` that the macro was applied to: how many the
/// Leaker accepted, and the same for those analyzed nested in `enter_mod`.
#[derive(Default)]
struct SelfDefaultCount {
    analyzed: usize,
    accepted: usize,
    nested_analyzed: usize,
    nested_accepted: usize,
}

fn kind_of(name: &str) -> &'static str {
    KINDS.iter().find(|k| **k == name).copied().unwrap_or("?")
}

impl Stats {
    fn collect(cases: &[Case], reports: &BTreeMap<(usize, usize), Report>) -> Stats {
        let mut stats = Stats::default();
        for (case_index, case) in cases.iter().enumerate() {
            for item in &case.items {
                for kind in &item.contents {
                    *stats.generated.entry(kind).or_default() += 1;
                }
                if item.kind == "mod" {
                    *stats.generated.entry("mod").or_default() += 1;
                }
                if let Some(report) = reports.get(&(case_index, item.id)) {
                    stats.analyzed += 1;
                    if !case.config.primitive {
                        stats.primitive_off += 1;
                    }
                    stats.skipped += report.skipped;
                    stats.twins += usize::from(report.twin);
                    for (tag, _) in SELF_DEFAULT_SHAPES {
                        if item.tags.contains(*tag) {
                            let count = stats.self_defaults.entry(tag).or_default();
                            let nested = item.scope == "mod";
                            count.analyzed += 1;
                            count.accepted += usize::from(report.accepted);
                            count.nested_analyzed += usize::from(nested);
                            count.nested_accepted += usize::from(nested && report.accepted);
                        }
                    }
                    for error in &report.rejections {
                        if let Some(family) = family(error) {
                            *stats.families.entry(family.name).or_default() += 1;
                            if family.name.starts_with("a bare trait") {
                                stats.bare_rejected += 1;
                            }
                        }
                    }
                    let kind = kind_of(&report.kind);
                    if let Some((_, api)) = SCOPES.iter().find(|(scope, _)| *scope == item.scope) {
                        let counts = if report.accepted {
                            &mut stats.scope_accepted
                        } else {
                            *stats
                                .reasons
                                .entry((api, reason_head(&report.detail)))
                                .or_default() += 1;
                            &mut stats.scope_rejected
                        };
                        *counts.entry(api).or_default() += 1;
                    }
                    if report.accepted {
                        *stats.accepted.entry(kind).or_default() += 1;
                        stats.positions += report.positions;
                        stats.checked += report.checked;
                        stats.entries += report.entries;
                    } else {
                        *stats.rejected.entry(kind).or_default() += 1;
                        *stats
                            .reasons
                            .entry((kind, reason_head(&report.detail)))
                            .or_default() += 1;
                    }
                    if report.allowed {
                        *stats.allowed.entry(kind).or_default() += 1;
                        stats.allowed_entries += report.entries;
                        stats.allowed_positions += report.positions;
                        stats.allowed_checked += report.checked;
                        for error in &report.rejections {
                            *stats
                                .allowed_errors
                                .entry((error.reason.clone(), error.kind.clone()))
                                .or_default() += 1;
                        }
                    }
                    stats.accept_failed += usize::from(report.accept_failed);
                }
            }
        }
        stats
    }

    /// The items accepted of a kind, or analyzed through a scope (the name of its API).
    fn accepted_of(&self, name: &str) -> usize {
        self.accepted.get(name).copied().unwrap_or(0)
            + self.scope_accepted.get(name).copied().unwrap_or(0)
    }

    fn rejected_of(&self, name: &str) -> usize {
        self.rejected.get(name).copied().unwrap_or(0)
            + self.scope_rejected.get(name).copied().unwrap_or(0)
    }

    fn total_accepted(&self) -> usize {
        self.accepted.values().sum()
    }

    /// The reasons of the rejections of a kind, most frequent first.
    fn reasons_of(&self, kind: &str) -> Vec<(&str, usize)> {
        let mut reasons: Vec<(&str, usize)> = self
            .reasons
            .iter()
            .filter(|((k, _), _)| *k == kind)
            .map(|((_, reason), n)| (reason.as_str(), *n))
            .collect();
        reasons.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        reasons
    }
}

/// The least share of the analyzed items of each kind that the Leaker must accept, in percent.
///
/// A test that counts a rejection as a success passes when the Leaker rejects everything, or when
/// the generator makes shapes that cannot be leaked (before the floors, 93% of the modules and
/// 60% of the traits were rejected, and a macro that failed every `intern_*` still printed `ok`).
/// These floors fail the run then. Measured over the seeds 1 to 16 at 24 and at 60 cases, the
/// accepted share was, per kind, lowest and pooled: struct 90/95, enum 90/94, union 85/92, trait
/// 70/75, inherent impl 79/85, trait impl 59/73, module 74/78, and per scope `enter_mod` 81/88,
/// `enter_generics` 87/94, `enter_trait` 57/79, `enter_impl` 100/100 (the seeds 17 to 32 at 24
/// cases: module down to 65, of 34). A floor is about 10 points below the lowest (more for the
/// rare kinds, whose rate varies most), so that a different seed or a short run does not trip it,
/// but a regression of a kind or the old generator (module 10%, trait 39%, inherent and trait impl
/// 41%) does.
/// `TYPE_LEAK_PROP_MIN_ACCEPT` overrides them.
const FLOORS: &[(&str, f64)] = &[
    ("struct", 80.0),
    ("enum", 80.0),
    ("union", 75.0),
    ("trait", 60.0),
    ("impl-inherent", 70.0),
    ("impl-trait", 50.0),
    ("mod", 55.0),
    // The items analyzed through a scope (`gen::SCOPES`), by the name of its API.
    ("enter_mod", 70.0),
    ("enter_generics", 75.0),
    ("enter_trait", 45.0),
    ("enter_impl", 90.0),
];

/// The fewest analyzed items of a kind for its floor to count (a short run has too few of the
/// rare kinds to say anything about their rate): `TYPE_LEAK_PROP_MIN_SAMPLES`.
const MIN_SAMPLES: usize = 12;

/// The least share of the type positions of the accepted items whose rewritten type was compared
/// with the original (the rest has an elided lifetime, `impl Trait`, or a name the module
/// declares), in percent: `TYPE_LEAK_PROP_MIN_CHECKED`. Without it the oracle could silently
/// check nothing. Measured: 48% to 53% over the seeds 1 to 12 at 60 cases (the accepted modules
/// have many positions that name what the module declares).
const MIN_CHECKED: f64 = 30.0;

#[derive(Debug, PartialEq)]
struct Floors {
    by_kind: BTreeMap<&'static str, f64>,
    min_samples: usize,
    min_checked: f64,
}

impl Floors {
    /// `spec`: the value of `TYPE_LEAK_PROP_MIN_ACCEPT`, a comma-separated list read in order of
    /// `N` (every kind) and `kind=N`: `"0"` turns the floors off, `"30,mod=5"` is 30% but 5% for
    /// a module.
    fn parse(spec: &str, min_samples: usize, min_checked: f64) -> Result<Floors, String> {
        let mut by_kind: BTreeMap<&'static str, f64> = FLOORS.iter().copied().collect();
        for token in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let (kind, value) = match token.split_once('=') {
                Some((kind, value)) => (Some(kind.trim()), value.trim()),
                None => (None, token),
            };
            let value: f64 = value
                .parse()
                .map_err(|_| format!("`{}` is not a percentage", token))?;
            if !(0.0..=100.0).contains(&value) {
                return Err(format!("`{}` is not between 0 and 100", token));
            }
            match kind {
                None => {
                    for floor in by_kind.values_mut() {
                        *floor = value;
                    }
                }
                Some(kind) => match by_kind.keys().find(|k| **k == kind).copied() {
                    Some(kind) => {
                        by_kind.insert(kind, value);
                    }
                    None => {
                        return Err(format!(
                            "`{}`: no kind or scope `{}` (they are {:?})",
                            token,
                            kind,
                            by_kind.keys().collect::<Vec<_>>()
                        ))
                    }
                },
            }
        }
        Ok(Floors {
            by_kind,
            min_samples,
            min_checked,
        })
    }

    fn from_env() -> Floors {
        let spec = std::env::var("TYPE_LEAK_PROP_MIN_ACCEPT").unwrap_or_default();
        Floors::parse(
            &spec,
            env_or("TYPE_LEAK_PROP_MIN_SAMPLES", MIN_SAMPLES),
            env_or("TYPE_LEAK_PROP_MIN_CHECKED", MIN_CHECKED),
        )
        .unwrap_or_else(|e| panic!("TYPE_LEAK_PROP_MIN_ACCEPT: {}", e))
    }

    /// Why the statistics are too poor for the run to count as a test, one line each.
    fn problems(&self, stats: &Stats) -> Vec<String> {
        let mut problems = Vec::new();
        let all_off = self.by_kind.values().all(|f| *f == 0.0);
        if stats.analyzed == 0 {
            problems.push(
                "no macro was applied to any item (`analyzed` is 0): the run tested nothing"
                    .to_string(),
            );
        } else if stats.total_accepted() == 0 {
            problems.push(format!(
                "the Leaker accepted none of the {} items analyzed (a macro that fails every \
                 `intern_*` would pass otherwise: a rejection counts as a success)",
                stats.analyzed
            ));
        }
        if all_off {
            return problems;
        }
        for kind in KINDS.iter().chain(SCOPES.iter().map(|(_, api)| api)) {
            let (a, r) = (stats.accepted_of(kind), stats.rejected_of(kind));
            let floor = self.by_kind.get(kind).copied().unwrap_or(0.0);
            if a + r < self.min_samples || floor == 0.0 {
                continue;
            }
            let rate = 100.0 * a as f64 / (a + r) as f64;
            if rate < floor {
                let reasons: Vec<String> = stats
                    .reasons_of(kind)
                    .iter()
                    .take(3)
                    .map(|(reason, n)| format!("{}x {}", n, reason))
                    .collect();
                problems.push(format!(
                    "{}: {} of {} analyzed items accepted ({:.0}%), below the floor of {:.0}%; \
                     the generator makes shapes that cannot be leaked, or the Leaker rejects \
                     too much. Most frequent rejections: {}",
                    kind,
                    a,
                    a + r,
                    rate,
                    floor,
                    reasons.join("; ")
                ));
            }
        }
        if stats.total_accepted() > 0 && self.min_checked > 0.0 {
            let share = if stats.positions == 0 {
                0.0
            } else {
                100.0 * stats.checked as f64 / stats.positions as f64
            };
            if share < self.min_checked {
                problems.push(format!(
                    "only {} of {} type positions of the accepted items have a witness \
                     ({:.0}%), below {:.0}%: the oracle checks (almost) nothing",
                    stats.checked, stats.positions, share, self.min_checked
                ));
            }
        }
        problems
    }
}

fn summarize(stats: &Stats, floors: &Floors, seed: u64, n: usize, features: Features) -> String {
    let mut s = format!(
        "type-leak property test: seed {}, {} cases, features {:?}\n",
        seed, n, features
    );
    let total: usize = stats.generated.values().sum();
    writeln!(
        s,
        "  generated items: {} (nested items of a module included)",
        total
    )
    .unwrap();
    writeln!(
        s,
        "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6}",
        "kind", "generated", "accepted", "rejected", "accepted%", "floor"
    )
    .unwrap();
    for kind in KINDS
        .iter()
        .chain(["fn", "alias", "const", "static"].iter())
        .chain(SCOPES.iter().map(|(_, api)| api))
    {
        // An item analyzed through a scope is of a kind too: `generated` counts kinds only.
        let g = if SCOPES.iter().any(|(_, api)| api == kind) {
            stats.accepted_of(kind) + stats.rejected_of(kind)
        } else {
            stats.generated.get(kind).copied().unwrap_or(0)
        };
        if g == 0 {
            continue;
        }
        let (a, r) = (stats.accepted_of(kind), stats.rejected_of(kind));
        let rate = if a + r > 0 {
            format!("{:.0}%", 100.0 * a as f64 / (a + r) as f64)
        } else {
            "-".into()
        };
        let floor = match floors.by_kind.get(kind) {
            Some(f) if *f == 0.0 => "off".into(),
            Some(f) if a + r >= floors.min_samples => format!("{:.0}%", f),
            Some(_) => "n/a".into(),
            None => "-".into(),
        };
        writeln!(
            s,
            "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6}",
            kind, g, a, r, rate, floor
        )
        .unwrap();
    }
    writeln!(
        s,
        "  analyzed (macros applied): {}; entries: {}; type positions: {}, witnesses checked: {} ({} skipped: elided lifetimes, or names declared in a module)",
        stats.analyzed,
        stats.entries,
        stats.positions,
        stats.checked,
        stats.positions - stats.checked
    )
    .unwrap();
    writeln!(
        s,
        "  bare traits of the prelude: {} items compared with their twin, {} errors of rejections; optional entries without an impl (trusting items): {}; items without allow_primitive: {}",
        stats.twins, stats.bare_rejected, stats.skipped, stats.primitive_off
    )
    .unwrap();
    for (tag, what) in SELF_DEFAULT_SHAPES {
        if let Some(count) = stats.self_defaults.get(tag) {
            writeln!(
                s,
                "  defaults and `Self` (trait_self_defaults, self_bound_adts) {:<19} {:>4} analyzed, {:>4} accepted; nested in enter_mod: {:>4} analyzed, {:>4} accepted",
                what, count.analyzed, count.accepted, count.nested_analyzed, count.nested_accepted
            )
            .unwrap();
        }
    }
    for (name, n) in &stats.families {
        writeln!(s, "  explained rejection errors: {:>4}x {}", n, name).unwrap();
    }
    let allowed: usize = stats.allowed.values().sum();
    if allowed + stats.accept_failed > 0 {
        let by_kind: Vec<String> = stats
            .allowed
            .iter()
            .map(|(kind, n)| format!("{} {}", n, kind))
            .collect();
        writeln!(
            s,
            "  accept mode (TYPE_LEAK_PROP_ACCEPT): {} items rejected at the first analysis were accepted by allow() of the causes of their errors ({}), {} still rejected (ACCEPT FAILED); entries: {}; type positions: {}, witnesses checked: {}",
            allowed,
            by_kind.join(", "),
            stats.accept_failed,
            stats.allowed_entries,
            stats.allowed_positions,
            stats.allowed_checked
        )
        .unwrap();
        for ((reason, kind), n) in &stats.allowed_errors {
            writeln!(
                s,
                "  allowed errors: {:>4}x {} (cause: {})",
                n, reason, kind
            )
            .unwrap();
        }
        let documented: usize = stats.documented.values().sum();
        let by_class: Vec<String> = stats
            .documented
            .iter()
            .map(|(class, n)| format!("{} {}", n, class))
            .collect();
        writeln!(
            s,
            "  accepted copies that do not compile for a failure the docs of allow() list: {} of {} ({})",
            documented,
            allowed,
            by_class.join(", ")
        )
        .unwrap();
    }
    for kind in KINDS.iter().chain(SCOPES.iter().map(|(_, api)| api)) {
        for (reason, n) in stats.reasons_of(kind).iter().take(3) {
            writeln!(s, "  rejected {:>4}x {:<13} {}", n, kind, reason).unwrap();
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------
// The test.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "slow, builds generated crates; run with `cargo test --test prop_cross_crate -- --ignored`"]
fn random_items_leak_across_crates() {
    let cases: usize = env_or("TYPE_LEAK_PROP_CASES", 24);
    let seed: u64 = env_or("TYPE_LEAK_PROP_SEED", 1);
    let items_per_module: usize = env_or("TYPE_LEAK_PROP_ITEMS", 8);
    let features = Features::parse(&std::env::var("TYPE_LEAK_PROP_FEATURES").unwrap_or_default());
    let floors = Floors::from_env();
    let generated: Vec<Case> = (0..cases)
        .map(|i| generate(case_seed(seed, i), i, items_per_module, features))
        .collect();

    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("prop");
    let target = base.join("target");
    let accept = accept_mode();
    let stages_on = allow_stages();
    let ws = base.join(format!(
        "ws-{}-{}{}{}",
        seed,
        cases,
        if accept { "-accept" } else { "" },
        if stages_on { "-stages" } else { "" }
    ));
    let list: Vec<(usize, &Case)> = generated.iter().enumerate().collect();
    write_workspace(
        &ws,
        &list,
        if accept { Mode::Accept } else { Mode::Real },
        stages_on,
    );
    let build = cargo_build(&ws, &target);

    // The watchdog: the helper crate or cargo failing is not a finding, and not a rejection.
    let known = case_packages(&generated);
    if let Some(problem) = build_problem(&build, &known) {
        panic!(
            "type-leak property test: the build is broken (seed {}, {} cases):\n{}",
            seed, cases, problem
        );
    }

    let mut failures: Vec<Failure> = Vec::new();

    // The errors of each crate. The Referrer's crate is built only if the Leaker's compiles.
    let mut leaker_failed: Vec<usize> = Vec::new();
    let mut unbuilt: Vec<usize> = Vec::new();
    for (i, case) in generated.iter().enumerate() {
        let leaker = leaker_name(i);
        let referrer = referrer_name(i);
        let leaker_errors: Vec<&Diag> =
            build.diags.iter().filter(|d| d.package == leaker).collect();
        if !leaker_errors.is_empty() {
            leaker_failed.push(i);
            continue;
        }
        // The Referrer's crate, and, in the accept mode, the second one (it copies only the items
        // that the Leaker accepted by `allow()`, and so every error in it is about one of them),
        // and with `allow_stages` the one of the copies made after the analysis without a rule.
        let mut referrer_failed = false;
        let mut crates: Vec<BuiltCrate> = vec![
            (
                referrer.clone(),
                referrer_dir(i),
                &case.referrer_lines,
                "the Referrer's crate does not compile",
                None,
            ),
            (
                accept_name(i),
                accept_dir(i),
                &case.accept_lines,
                ACCEPT_COPY,
                None,
            ),
        ];
        if stages_on {
            for stage in &case.stages {
                crates.push((
                    stage_name(i, stage),
                    stage_dir(i, stage),
                    &stage.lines,
                    STAGE_COPY,
                    Some(StageKey::from(stage)),
                ));
            }
        }
        for (package, dir, lines, class, key) in crates {
            let file = format!("{}/src/lib.rs", dir);
            for diag in build.diags.iter().filter(|d| d.package == package) {
                referrer_failed = true;
                // The item whose macro invocation the error is in.
                let line = diag
                    .lines
                    .iter()
                    .find(|(f, _)| *f == file)
                    .map(|(_, line)| *line as usize);
                let item =
                    line.and_then(|line| lines.iter().find(|(l, _)| *l == line).map(|(_, id)| *id));
                // The same error is often reported several times.
                if failures.iter().any(|f| {
                    f.case == i
                        && f.item == item
                        && f.class == class
                        && f.stage == key
                        && f.text.lines().next() == diag.rendered.lines().next()
                }) {
                    continue;
                }
                failures.push(Failure {
                    stage: key,
                    code: diag.code.clone(),
                    case: i,
                    item,
                    class,
                    text: diag.rendered.clone(),
                });
            }
        }
        // Neither crate reported an error: all must have been built.
        if !referrer_failed {
            let missing = missing_artifacts(
                &build,
                i,
                true,
                accept,
                if stages_on { &case.stages } else { &[] },
            );
            if !missing.is_empty() {
                unbuilt.push(i);
                failures.push(Failure {
                    stage: None,
                    code: String::new(),
                    case: i,
                    item: None,
                    class: "WATCHDOG: a crate of the case was not built, and no error says why",
                    text: missing.join("\n"),
                });
            }
        }
    }

    // A Leaker's crate that does not compile: is it the generator or the macro?
    if !leaker_failed.is_empty() {
        let id_ws = base.join(format!("ws-{}-{}-identity", seed, cases));
        let list: Vec<(usize, &Case)> = leaker_failed.iter().map(|&i| (i, &generated[i])).collect();
        write_workspace(&id_ws, &list, Mode::Identity, false);
        let identity = cargo_build(&id_ws, &target);
        if let Some(problem) = build_problem(&identity, &known) {
            panic!(
                "type-leak property test: the build without `type-leak` is broken (seed {}, {} cases):\n{}",
                seed, cases, problem
            );
        }
        // In the accept mode, does it compile when no item is accepted by `allow()`, and with
        // `allow_stages`, when the macros make no copy of the second analysis? If it does, the
        // output of an item that is accepted by it (or of its stage copy) is what does not.
        let ordinary = if accept || stages_on {
            let ordinary_ws = base.join(format!("ws-{}-{}-ordinary", seed, cases));
            write_workspace(&ordinary_ws, &list, Mode::Real, false);
            Some(cargo_build(&ordinary_ws, &target))
        } else {
            None
        };
        for &i in &leaker_failed {
            let name = leaker_name(i);
            let own: Vec<&Diag> = identity
                .diags
                .iter()
                .filter(|d| d.package == name)
                .collect();
            let by_accept = ordinary
                .as_ref()
                .map_or(false, |b| !b.diags.iter().any(|d| d.package == name));
            let (class, diags) = if own.is_empty() {
                (
                    if by_accept && !accept {
                        "STAGES: the Leaker's crate does not compile with the output of the copy made after the analysis without a rule"
                    } else if by_accept {
                        "ACCEPT: the Leaker's crate does not compile with the output of an item accepted by allow()"
                    } else {
                        "the Leaker's crate does not compile with the output of the macro"
                    },
                    build
                        .diags
                        .iter()
                        .filter(|d| d.package == name)
                        .collect::<Vec<_>>(),
                )
            } else {
                (
                    "GENERATOR BUG: the Leaker's crate does not compile by itself",
                    own,
                )
            };
            // The item whose attribute the error is in, if the diagnostic says.
            let file = format!("{}/src/lib.rs", leaker_dir(i));
            let attributes = attr_lines(&generated[i]);
            for diag in diags {
                let item = diag
                    .lines
                    .iter()
                    .filter(|(f, _)| *f == file)
                    .find_map(|(_, line)| {
                        attributes
                            .iter()
                            .find(|(l, _)| *l == *line as usize)
                            .map(|(_, id)| *id)
                    });
                failures.push(Failure {
                    stage: None,
                    code: String::new(),
                    case: i,
                    item,
                    class,
                    text: diag.rendered.clone(),
                });
            }
        }
    }

    // What the macros reported. Every item must have a report: a missing one is a failure, not a
    // rejection.
    let mut reports: BTreeMap<(usize, usize), Report> = BTreeMap::new();
    // `allow_stages`: for each rule what the macro of each item counted, and reported of the copies
    // made after the result (the analysis without the rule, then the rule) and after the config.
    let mut runs: BTreeMap<&'static str, RuleRun> = BTreeMap::new();
    if stages_on {
        for rule in gen::rules() {
            runs.insert(rule, RuleRun::default());
        }
    }
    for (i, case) in generated.iter().enumerate() {
        if leaker_failed.contains(&i) || unbuilt.contains(&i) {
            continue;
        }
        let Some(exe) = build.executables.get(&leaker_name(i)) else {
            continue;
        };
        let output = Command::new(exe).output().unwrap();
        if !output.status.success() {
            failures.push(Failure {
                stage: None,
                code: String::new(),
                case: i,
                item: None,
                class: "WATCHDOG: the binary that prints the reports fails",
                text: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
            continue;
        }
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            // `allow_stages`: what the macro of an item counted for a rule (see `Stages`).
            if let Some(row) = line.strip_prefix("stages|") {
                let parsed = row
                    .split_once('|')
                    .and_then(|(rule, row)| Some((runs.get_mut(rule)?, Stages::parse(i, row)?)));
                match parsed {
                    Some((run, stage)) => run.stages.push(stage),
                    None => failures.push(Failure {
                        stage: None,
                        code: String::new(),
                        case: i,
                        item: None,
                        class: "WATCHDOG: a line of allow_stages cannot be read",
                        text: line.to_string(),
                    }),
                }
                continue;
            }
            // The copy after the result is what the copy after the config is compared with, so that a
            // macro that cannot make the second, or an error that remains, fails nothing.
            if let Some(row) = line.strip_prefix("stage|") {
                let parsed = row.split_once('|').and_then(|(rule, row)| {
                    let (side, row) = row.split_once('|')?;
                    let side = ["res", "cfg"].iter().position(|word| *word == side)?;
                    let (id, report) = parse_report(row)?;
                    Some((runs.get_mut(rule)?, side, id, report))
                });
                match parsed {
                    Some((run, side, id, report)) => {
                        if side == 0 {
                            if let Some(error) = &report.rewrite_error {
                                failures.push(Failure {
                                    stage: None,
                                    code: String::new(),
                                    case: i,
                                    item: Some(id),
                                    class: "STAGES: the macro cannot make the copy made after the analysis without the rule",
                                    text: error.clone(),
                                });
                            }
                            if report.accept_failed {
                                failures.push(Failure {
                                    stage: None,
                                    code: String::new(),
                                    case: i,
                                    item: Some(id),
                                    class: "STAGES FAILED: an error remains when the cause of every error that the rule leaves is allowed",
                                    text: format!(
                                        "the errors that the rule leaves:\n{}\nthe errors that remain:\n{}",
                                        describe(&report.rejections),
                                        describe(&report.remaining)
                                    ),
                                });
                            }
                        }
                        run.reports[side].insert((i, id), report);
                    }
                    None => failures.push(Failure {
                        stage: None,
                        code: String::new(),
                        case: i,
                        item: None,
                        class: "WATCHDOG: a line of the copy of allow_stages cannot be read",
                        text: line.to_string(),
                    }),
                }
                continue;
            }
            match parse_report(line) {
                Some((id, report)) => {
                    if let Some(error) = &report.rewrite_error {
                        failures.push(Failure {
                            stage: None,
                            code: String::new(),
                            case: i,
                            item: Some(id),
                            class: if report.allowed {
                                "ACCEPT: the rewrite of an item accepted by allow() fails"
                            } else {
                                "the rewrite of an accepted item fails"
                            },
                            text: error.clone(),
                        });
                    }
                    if report.accept_failed {
                        failures.push(Failure {
                            stage: None,
                            code: String::new(),
                            case: i,
                            item: Some(id),
                            class: "ACCEPT FAILED: an error remains when the cause of every error is allowed",
                            text: format!(
                                "the errors:\n{}\nthe errors that remain:\n{}",
                                describe(&report.rejections),
                                describe(&report.remaining)
                            ),
                        });
                    }
                    reports.insert((i, id), report);
                }
                None => failures.push(Failure {
                    stage: None,
                    code: String::new(),
                    case: i,
                    item: None,
                    class: "WATCHDOG: a report line cannot be read",
                    text: line.to_string(),
                }),
            }
        }
        let missing: Vec<String> = case
            .items
            .iter()
            .filter(|item| !reports.contains_key(&(i, item.id)))
            .map(|item| item.id.to_string())
            .collect();
        if !missing.is_empty() {
            failures.push(Failure {
                stage: None,
                code: String::new(),
                case: i,
                item: None,
                class: "WATCHDOG: the macros of some items reported nothing",
                text: format!("items {}", missing.join(", ")),
            });
        }
        for (rule, run) in &runs {
            let missing: Vec<String> = case
                .items
                .iter()
                .filter(|item| {
                    !run.reports[0].contains_key(&(i, item.id))
                        || !run.reports[1].contains_key(&(i, item.id))
                        || !run.stages.iter().any(|s| s.case == i && s.item == item.id)
                })
                .map(|item| item.id.to_string())
                .collect();
            if !missing.is_empty() {
                failures.push(Failure {
                    stage: None,
                    code: String::new(),
                    case: i,
                    item: None,
                    class: "WATCHDOG: the macros of some items reported nothing of the copy of allow_stages",
                    text: format!("rule {}: items {}", rule, missing.join(", ")),
                });
            }
        }
    }
    // The names of the prelude mean the prelude in the Referrer's crate of the stage copies: that is
    // what allowing them is a promise of. Nothing generated declares one.
    if stages_on {
        for (i, case) in generated.iter().enumerate() {
            let shadowed = shadowed_prelude_names(&case.leaker);
            if !shadowed.is_empty() {
                failures.push(Failure {
                    stage: None,
                    code: String::new(),
                    case: i,
                    item: None,
                    class: "STAGES: the Leaker's crate declares a name of the prelude, which the stage copies must not shadow",
                    text: format!("shadowed: {}", shadowed.join(", ")),
                });
            }
        }
    }

    // The diagnostics of the copy of an item accepted by `allow()` are failures that the docs of
    // `LeakerResult::allow()` list, or defects: each must be one of them, and the item must have the
    // shape that gives it (`accept_families()`). The documented ones are counted and are no failure.
    // The same for the copies of `allow_stages`, which allowed the causes of the errors that
    // `allow_prelude()` leaves.
    let (copies, failures): (Vec<Failure>, Vec<Failure>) = failures
        .into_iter()
        .partition(|failure| failure.class == ACCEPT_COPY);
    let (mut stage_diags, mut failures): (Vec<Failure>, Vec<Failure>) = failures
        .into_iter()
        .partition(|failure| failure.class == STAGE_COPY);
    let (documented, accept_problems, _) = explain_copies(
        copies,
        &reports,
        &generated,
        &ACCEPT_CLASSES,
        &BTreeSet::new(),
        false,
    );
    failures.extend(accept_problems);
    let mut results: BTreeMap<&'static str, RuleResult> = BTreeMap::new();
    for (&rule, run) in &runs {
        let (result, problems) = RuleResult::judge(rule, run, &mut stage_diags, &generated);
        failures.extend(problems);
        results.insert(rule, result);
    }

    // The failure of an item that was accepted by `allow()` says what was allowed.
    for failure in &mut failures {
        let report = failure
            .item
            .and_then(|id| reports.get(&(failure.case, id)))
            .filter(|report| report.allowed);
        if let Some(report) = report {
            failure.text = format!(
                "{}\n\nthe item was rejected with these errors, and accepted by allow() of their causes:\n{}",
                failure.text,
                describe(&report.rejections)
            );
        }
    }

    // The documented failures of the copies of accepted items make the build fail too.
    if !build.success
        && failures.is_empty()
        && documented.is_empty()
        && results.values().all(RuleResult::is_empty)
    {
        panic!(
            "cargo failed, and no failure of a case says why (seed {}):\n{}",
            seed,
            excerpt(&build.stderr, 60)
        );
    }

    // Every rejection has a reason that is known, and an item that has its shape.
    for ((i, id), report) in &reports {
        if report.accepted {
            continue;
        }
        let item = generated[*i]
            .items
            .iter()
            .find(|item| item.id == *id)
            .unwrap();
        let mut tags = item.tags.clone();
        if item.kind == "mod" || item.scope == "mod" {
            tags.insert(gen::TAG_MODULE_SCOPE.to_string());
        }
        for problem in unexplained(&tags, report) {
            failures.push(Failure {
                stage: None,
                code: String::new(),
                case: *i,
                item: Some(*id),
                class: "UNEXPLAINED REJECTION: type-leak rejects an item for a reason the generator does not explain",
                text: problem,
            });
        }
    }
    if std::env::var_os("TYPE_LEAK_PROP_REJECTIONS").is_some() {
        for ((i, id), report) in &reports {
            if report.accepted {
                continue;
            }
            let item = generated[*i]
                .items
                .iter()
                .find(|item| item.id == *id)
                .unwrap();
            eprintln!(
                "REJECTED{} case {} item {} ({}, scope {:?}, prelude {}, tags {:?}):",
                if report.allowed {
                    " (accepted by allow())"
                } else if report.accept_failed {
                    " (ACCEPT FAILED)"
                } else {
                    ""
                },
                i,
                id,
                item.kind,
                item.scope,
                generated[*i].config.prelude,
                item.tags
            );
            for r in &report.rejections {
                eprintln!(
                    "    {} || {} || {} || {} ({})",
                    r.message, r.tokens, r.note, r.reason, r.kind
                );
            }
        }
    }
    let mut stats = Stats::collect(&generated, &reports);
    stats.documented = documented;
    let mut summary = summarize(&stats, &floors, seed, cases, features);
    for (rule, run) in &runs {
        let result = &results[rule];
        let copies = StageCopies {
            rule,
            reports: &run.reports[0],
            documented: &result.documented,
            verdicts: &result.verdicts,
            base_reports: &run.reports[1],
            base_failed: &result.base_failed,
            base_verdicts: &result.base_verdicts,
            clean_failed: &result.clean_failed,
            clean_verdicts: &result.clean_verdicts,
        };
        summary.push_str(&Stages::summarize(
            rule,
            &run.stages,
            &generated,
            &ws,
            &copies,
        ));
    }
    eprintln!("{}", summary);
    let floor_problems = floors.problems(&stats);

    if !failures.is_empty() || !floor_problems.is_empty() {
        let mut message = format!(
            "type-leak property failed (TYPE_LEAK_PROP_SEED={} TYPE_LEAK_PROP_CASES={} TYPE_LEAK_PROP_ITEMS={} TYPE_LEAK_PROP_FEATURES={:?}):\n{} failure(s), {} acceptance problem(s).\nSources: {}\n\n{}\n",
            seed,
            cases,
            items_per_module,
            std::env::var("TYPE_LEAK_PROP_FEATURES").unwrap_or_default(),
            failures.len(),
            floor_problems.len(),
            ws.display(),
            summary
        );
        if !floor_problems.is_empty() {
            message.push_str(
                "\nACCEPTANCE: the run passes only if the Leaker accepts enough of what is \
                 generated (TYPE_LEAK_PROP_MIN_ACCEPT, TYPE_LEAK_PROP_MIN_SAMPLES and \
                 TYPE_LEAK_PROP_MIN_CHECKED change the floors; see the docs of this test):\n",
            );
            for problem in &floor_problems {
                writeln!(message, "  - {}", problem).unwrap();
            }
        }
        for failure in failures.iter().take(6) {
            let case = &generated[failure.case];
            write!(
                message,
                "\n=== case {} (case seed {}), {}{}\n",
                failure.case,
                case_seed(seed, failure.case),
                failure.class,
                failure.stage.map_or(String::new(), |key| format!(
                    " [allow_stages: rule {}, copy after the {}, crate {}]",
                    key.rule,
                    if key.side == "res" {
                        "result"
                    } else {
                        "config"
                    },
                    if key.clean { "clean" } else { "with globs" }
                ))
            )
            .unwrap();
            writeln!(
                message,
                "crates: {}/{}, {}/{}",
                ws.display(),
                leaker_dir(failure.case),
                ws.display(),
                referrer_dir(failure.case)
            )
            .unwrap();
            writeln!(message, "settings: {:?}", case.config).unwrap();
            if let Some(item) = case.items.iter().find(|i| Some(i.id) == failure.item) {
                writeln!(
                    message,
                    "item {} ({}, in {:?}, allow_self {}, strict {}):\n{}",
                    item.id,
                    item.kind,
                    item.module,
                    item.allow_self,
                    item.strict,
                    excerpt(&item.source, 60)
                )
                .unwrap();
            }
            writeln!(message, "{}", excerpt(&failure.text, 40)).unwrap();
        }
        if !failures.is_empty() {
            message.push_str("\nall failures:\n");
        }
        for failure in &failures {
            let head = failure.text.lines().next().unwrap_or("");
            writeln!(
                message,
                "  case {} item {:?} [{}]{}: {}",
                failure.case,
                failure.item,
                failure.class.chars().take(24).collect::<String>(),
                failure.stage.map_or(String::new(), |key| format!(
                    " ({} {} {})",
                    key.rule,
                    key.side,
                    if key.clean { "clean" } else { "glob" }
                )),
                head.chars().take(150).collect::<String>()
            )
            .unwrap();
        }
        panic!("{}", message);
    }
}

// ---------------------------------------------------------------------------------------------
// Tests of the checks themselves (fast; they run with `cargo test` too).
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod checks {
    use super::*;

    fn build(success: bool, packages: &[&str]) -> Build {
        Build {
            success,
            diags: packages
                .iter()
                .map(|p| Diag {
                    package: p.to_string(),
                    code: String::new(),
                    rendered: "error: boom".into(),
                    lines: Vec::new(),
                })
                .collect(),
            executables: HashMap::new(),
            artifacts: HashSet::new(),
            stderr: "cargo said".into(),
        }
    }

    /// `n` reports of `kind`: `accepted` of them accepted (a case with one item each).
    fn stats_of(kinds: &[(&str, usize, usize)]) -> Stats {
        let mut stats = Stats::default();
        for (kind, accepted, rejected) in kinds {
            let kind = kind_of(kind);
            for _ in 0..*accepted {
                stats.analyzed += 1;
                *stats.accepted.entry(kind).or_default() += 1;
                stats.positions += 4;
                stats.checked += 4;
            }
            for _ in 0..*rejected {
                stats.analyzed += 1;
                *stats.rejected.entry(kind).or_default() += 1;
                *stats.reasons.entry((kind, "no".into())).or_default() += 1;
            }
        }
        stats
    }

    fn defaults() -> Floors {
        Floors::parse("", MIN_SAMPLES, MIN_CHECKED).unwrap()
    }

    #[test]
    fn a_broken_helper_crate_is_not_a_finding_and_not_a_rejection() {
        let cases: Vec<Case> = (0..2)
            .map(|i| generate_with(i, i as usize, 6, Features::parse(""), Flags::default()))
            .collect();
        let known = case_packages(&cases);
        // The macro crate does not compile: no crate of any case is built, and no case reports.
        let problem = build_problem(&build(false, &["type_leak_test_macro"]), &known).unwrap();
        assert!(problem.contains("type_leak_test_macro"), "{}", problem);
        assert!(problem.contains("nothing was tested"), "{}", problem);
        // Cargo itself fails.
        let problem = build_problem(&build(false, &[]), &known).unwrap();
        assert!(problem.contains("without a compile error"), "{}", problem);
        // The errors of the cases are the findings of the test, not of the watchdog.
        assert!(build_problem(&build(false, &["tlp_l_001", "tlp_r_000"]), &known).is_none());
        assert!(build_problem(&build(true, &[]), &known).is_none());
    }

    #[test]
    fn a_case_with_nothing_built_is_reported() {
        let mut b = build(true, &[]);
        assert_eq!(missing_artifacts(&b, 3, true, false, &[]).len(), 2);
        b.executables.insert(leaker_name(3), PathBuf::from("/x"));
        assert_eq!(missing_artifacts(&b, 3, true, false, &[]).len(), 1);
        b.artifacts.insert(referrer_name(3));
        assert!(missing_artifacts(&b, 3, true, false, &[]).is_empty());
        // The second Referrer's crate of the accept mode is built too.
        assert_eq!(missing_artifacts(&b, 3, true, true, &[]).len(), 1);
        b.artifacts.insert(accept_name(3));
        assert!(missing_artifacts(&b, 3, true, true, &[]).is_empty());
        // And those of `allow_stages`: each rule has four, glob and clean after the result and the
        // config.
        let case = generate_with(3, 3, 6, Features::parse(""), Flags::default());
        let stages = &case.stages;
        assert_eq!(stages.len(), 4 * Flags::default().rules().len());
        assert_eq!(
            missing_artifacts(&b, 3, true, true, stages).len(),
            stages.len()
        );
        for stage in stages {
            b.artifacts.insert(stage_name(3, stage));
        }
        assert!(missing_artifacts(&b, 3, true, true, stages).is_empty());
        // The leaker's crate that does not compile builds none of them.
        assert!(missing_artifacts(&b, 3, false, true, stages).is_empty());
    }

    #[test]
    fn the_floors_fail_a_run_that_accepts_nothing() {
        // Every `intern_*` fails: all rejected.
        let all_rejected = stats_of(&[("struct", 0, 40), ("trait", 0, 40), ("mod", 0, 20)]);
        let problems = defaults().problems(&all_rejected);
        assert!(
            problems.iter().any(|p| p.contains("accepted none")),
            "{:?}",
            problems
        );
        assert!(
            problems.iter().any(|p| p.starts_with("struct:")),
            "{:?}",
            problems
        );
        // Nothing analyzed at all.
        let problems = defaults().problems(&Stats::default());
        assert!(
            problems.iter().any(|p| p.contains("tested nothing")),
            "{:?}",
            problems
        );
        // Turned off per kind, the total still has to be positive.
        let off = Floors::parse("0", MIN_SAMPLES, MIN_CHECKED).unwrap();
        assert_eq!(off.problems(&all_rejected).len(), 1);
        assert!(off.problems(&stats_of(&[("struct", 1, 40)])).is_empty());
    }

    #[test]
    fn a_kind_below_its_floor_fails_when_it_has_enough_samples() {
        let rich = stats_of(&[("struct", 90, 10), ("mod", 1, 30)]);
        let problems = defaults().problems(&rich);
        assert_eq!(problems.len(), 1, "{:?}", problems);
        assert!(problems[0].starts_with("mod:"), "{:?}", problems);
        // Too few samples of the kind: not counted.
        let poor = stats_of(&[("struct", 90, 10), ("mod", 0, MIN_SAMPLES - 1)]);
        assert!(defaults().problems(&poor).is_empty());
        // The floor is a minimum, not a bound on the rate.
        let ok = stats_of(&[("struct", 100, 0), ("mod", 4, 6)]);
        assert!(defaults().problems(&ok).is_empty());
    }

    #[test]
    fn the_floors_are_overridden_by_the_environment_format() {
        let floors = Floors::parse("30, mod=5", 7, 55.0).unwrap();
        assert_eq!(floors.by_kind["struct"], 30.0);
        assert_eq!(floors.by_kind["mod"], 5.0);
        assert_eq!((floors.min_samples, floors.min_checked), (7, 55.0));
        let floors = Floors::parse("mod=90", MIN_SAMPLES, MIN_CHECKED).unwrap();
        assert_eq!(floors.by_kind["mod"], 90.0);
        let default_of = |kind: &str| FLOORS.iter().find(|(k, _)| *k == kind).unwrap().1;
        assert_eq!(floors.by_kind["struct"], default_of("struct"));
        assert_eq!(floors.by_kind["trait"], default_of("trait"));
        // A scope's floor too.
        let floors = Floors::parse("enter_impl=1", MIN_SAMPLES, MIN_CHECKED).unwrap();
        assert_eq!(floors.by_kind["enter_impl"], 1.0);
        assert!(Floors::parse("fn=5", 1, 1.0).is_err());
        assert!(Floors::parse("150", 1, 1.0).is_err());
        assert!(Floors::parse("a", 1, 1.0).is_err());
    }

    #[test]
    fn an_oracle_that_checks_nothing_fails() {
        let mut stats = stats_of(&[("struct", 90, 10)]);
        stats.checked = 0;
        let problems = defaults().problems(&stats);
        assert!(
            problems.iter().any(|p| p.contains("witness")),
            "{:?}",
            problems
        );
        let mut stats = stats_of(&[("struct", 90, 10)]);
        stats.checked = stats.positions / 5;
        assert_eq!(defaults().problems(&stats).len(), 1);
        stats.checked = stats.positions * 9 / 10;
        assert!(defaults().problems(&stats).is_empty());
    }

    fn rejection(message: &str, tokens: &str) -> Rejection {
        Rejection {
            message: message.to_string(),
            tokens: tokens.to_string(),
            ..Default::default()
        }
    }

    fn tags(names: &[&str]) -> std::collections::BTreeSet<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn rejections_are_classified_into_the_known_reasons() {
        let tags_of =
            |message: &str, tokens: &str| family(&rejection(message, tokens)).map(|f| f.tags);
        // A bare trait of the prelude: its name is the tag, with the binder and the `?` left out.
        assert_eq!(
            tags_of("use absolute path: 'Clone'", "Clone"),
            Some(vec!["bare:Clone".to_string()])
        );
        assert_eq!(
            tags_of("use absolute path: '? Sized'", "? Sized"),
            Some(vec!["bare:?Sized".to_string()])
        );
        assert_eq!(
            tags_of(
                "use absolute path: 'for < 'h3 > Fn(& 'h3 u8) -> & 'h3 u8'",
                "for < 'h3 > Fn(& 'h3 u8) -> & 'h3 u8"
            ),
            Some(vec!["bare:Fn".to_string()])
        );
        // A path can be several reasons at once: the shapes of any of them explain it.
        let tags = tags_of(
            "use absolute path: 'Into < Self :: A0 >'",
            "Into < Self :: A0 >",
        )
        .unwrap();
        assert!(tags.contains(&"bare:Into".to_string()), "{:?}", tags);
        assert!(
            tags.contains(&gen::TAG_SELF_ASSOC.to_string()),
            "{:?}",
            tags
        );
        // The binder of a bound is not a type that names one.
        let tags = tags_of(
            "use absolute path: 'for < 'h1 > Fn(& 'h1 u8) -> & 'h1 u8'",
            "for < 'h1 > Fn(& 'h1 u8) -> & 'h1 u8",
        )
        .unwrap();
        assert!(
            !tags.contains(&gen::TAG_BINDER_NO_ANCHOR.to_string()),
            "{:?}",
            tags
        );
        // The name of another trait is not the name of the prelude's (`Fn` of `FnMut`, `Fnord`).
        assert_eq!(
            tags_of("use absolute path: 'Fnord'", "Fnord"),
            None,
            "a relative path that is not a trait of the prelude is not explained"
        );
        assert_eq!(
            tags_of("use absolute path: 'FnMut(u8)'", "FnMut(u8)"),
            Some(vec!["bare:FnMut".to_string()])
        );
        // The other reasons.
        assert_eq!(
            tags_of("use absolute path: 'Rl < 'h5 >'", "Rl < 'h5 >"),
            Some(vec![gen::TAG_BINDER_NO_ANCHOR.to_string()])
        );
        assert_eq!(
            tags_of(
                "use absolute path: 'Tr1_1 < 'a > :: A0'",
                "Tr1_1 < 'a > :: A0"
            ),
            Some(vec![gen::TAG_SELF_ASSOC.to_string()])
        );
        assert_eq!(
            tags_of(
                "write `<Self as ::k::Tr>::A0` instead of `Self::A0` in `Wrap<Self::A0>`: the copy cannot tell",
                "Wrap<Self::A0>"
            ),
            Some(vec![gen::TAG_SELF_ASSOC.to_string()])
        );
        assert_eq!(
            tags_of(
                "`Wrap < T >` is in a bound or where predicate, and a bound of its generics names a leaked type",
                "Wrap < T >"
            ),
            Some(vec![gen::TAG_BOUND_CYCLE.to_string()])
        );
        assert_eq!(
            tags_of(
                "use absolute path: 'crate :: base :: BLEN'",
                "crate :: base :: BLEN"
            )
            .map(|tags| tags.contains(&gen::TAG_CONST_DEFAULT.to_string())),
            Some(true)
        );
        // A reason nobody listed.
        assert_eq!(tags_of("`Self` has no meaning here", "Self"), None);
        assert_eq!(tags_of("use absolute path: 'Strange'", "Strange"), None);
    }

    #[test]
    fn a_rejection_needs_a_reason_that_is_known_and_a_shape_that_gives_it() {
        let report = |errors: Vec<Rejection>| Report {
            kind: "trait".into(),
            rejections: errors,
            ..Default::default()
        };
        let bare = report(vec![rejection("use absolute path: 'Default'", "Default")]);
        // The item has the shape: explained.
        assert!(unexplained(&tags(&["bare:Default"]), &bare).is_empty());
        // The item has none (it spells another trait bare, or none), so something else rejected it.
        let problems = unexplained(&tags(&["bare:Clone", "bound_cycle"]), &bare);
        assert_eq!(problems.len(), 1, "{:?}", problems);
        assert!(
            problems[0].starts_with("NO SHAPE THAT GIVES IT"),
            "{:?}",
            problems
        );
        // A bare `?Sized` is explained by its own tag, which the generator does not put on when
        // `gen::BARE_RELAXED_SIZED_ALWAYS_OK` is set.
        let sized = report(vec![rejection("use absolute path: '? Sized'", "? Sized")]);
        assert!(unexplained(&tags(&["bare:Sized"]), &sized).len() == 1);
        assert!(unexplained(&tags(&["bare:?Sized"]), &sized).is_empty());
        // A reason that is not in the list.
        let unknown = report(vec![rejection("`Self` has no meaning here", "Self")]);
        let problems = unexplained(&tags(&["bound_cycle"]), &unknown);
        assert!(
            problems[0].starts_with("NOT A KNOWN REASON"),
            "{:?}",
            problems
        );
        // Every error counts: one that is explained does not explain the other.
        let two = report(vec![
            rejection("use absolute path: 'Default'", "Default"),
            rejection("`Self` has no meaning here", "Self"),
        ]);
        assert_eq!(unexplained(&tags(&["bare:Default"]), &two).len(), 1);
        // A rejection with no error (the report was not read) is not explained either.
        assert_eq!(unexplained(&tags(&[]), &report(Vec::new())).len(), 1);
    }

    #[test]
    fn the_errors_of_a_rejection_are_read_back_from_the_report() {
        let line = "8|trait|rej|use absolute path: 'Clone' use absolute path: 'Default'\u{1e}\
                    use absolute path: 'Clone'\u{1f}Clone\u{1f}\u{1d}\
                    use absolute path: 'Default'\u{1f}Default\u{1f}the note";
        let (id, report) = parse_report(line).unwrap();
        assert_eq!((id, report.accepted), (8, false));
        assert_eq!(report.rejections.len(), 2);
        assert_eq!(report.rejections[1].tokens, "Default");
        assert_eq!(report.rejections[1].note, "the note");
        // An accepted item has no errors, and says what the macro wrote.
        let (_, report) =
            parse_report("3|struct|ok|entries=2 positions=5 checked=4 skipped=1 twin=1\u{1e}")
                .unwrap();
        assert!(report.accepted && report.rejections.is_empty());
        assert_eq!((report.skipped, report.twin), (1, true));
    }

    #[test]
    fn the_generated_cases_say_what_the_macro_needs() {
        let mut bare_sized = 0;
        let mut strict = 0;
        // The traits of `trait_self_defaults` (F39): needing `Self` and nested, and declared again.
        let (mut nested_self_default, mut redeclared, mut plain_default) = (0, 0, 0);
        for seed in 0..12 {
            let case = generate_with(seed, 0, 6, Features::parse(""), Flags::default());
            for item in &case.items {
                let attr = case
                    .leaker
                    .lines()
                    .find(|line| line.contains(&format!("(id = {},", item.id)))
                    .unwrap_or_else(|| panic!("no attribute for item {}", item.id));
                assert!(
                    attr.contains(&format!("strict = {},", item.strict)),
                    "{} {}",
                    attr,
                    item.strict
                );
                assert!(attr.contains(&format!("primitive = {},", case.config.primitive)));
                assert!(attr.contains(&format!(
                    "relaxed_sized = {})",
                    gen::BARE_RELAXED_SIZED_ALWAYS_OK
                )));
                strict += usize::from(item.strict);
                // `redeclare` is for a trait nested in `enter_mod`, and only for that.
                let redeclare = item.tags.contains(gen::TAG_REDECLARED);
                assert_eq!(attr.contains("redeclare = true,"), redeclare, "{}", attr);
                if redeclare {
                    assert!(item.kind == "trait" && item.scope == "mod", "{}", attr);
                    redeclared += 1;
                }
                if item.tags.contains(gen::TAG_SELF_DEFAULT) {
                    assert!(item.kind == "trait", "{}", attr);
                    nested_self_default += usize::from(item.scope == "mod");
                }
                plain_default += usize::from(item.tags.contains(gen::TAG_PLAIN_DEFAULT));
                // A bare trait without the prelude gets its tag: a bare `?Sized` unless it is accepted.
                if item.tags.contains("bare:?Sized") {
                    assert!(!case.config.prelude && !gen::BARE_RELAXED_SIZED_ALWAYS_OK);
                    bare_sized += 1;
                }
                if case.config.prelude {
                    assert!(
                        !item.tags.iter().any(|tag| tag.starts_with("bare:")),
                        "{:?}",
                        item.tags
                    );
                }
            }
        }
        assert!(strict > 0, "no item is strict in 12 cases");
        assert!(
            nested_self_default > 0 && redeclared > 0 && plain_default > 0,
            "the traits of `trait_self_defaults` are not generated in 12 cases: {} nested that \
             need Self, {} declared again, {} controls",
            nested_self_default,
            redeclared,
            plain_default
        );
        let _ = bare_sized;
    }

    /// The structs, enums and unions of `self_bound_adts` (F41, F42) are generated, nested in
    /// `enter_mod`, and the Leaker must accept them: no reason explains their rejection.
    #[test]
    fn the_adts_with_a_bound_that_names_their_self_are_generated() {
        let count = |features: Features, tag: &str| -> (usize, usize) {
            let (mut found, mut types) = (0, 0);
            for seed in 0..12 {
                let case = generate_with(seed, 0, 6, features, Flags::default());
                for item in &case.items {
                    if item.tags.contains(gen::TAG_SELF_BOUND_ADT) {
                        assert!(
                            matches!(item.kind, "struct" | "enum" | "union") && item.scope == "mod",
                            "{} {}",
                            item.kind,
                            item.scope
                        );
                        found += 1;
                        types += usize::from(item.tags.contains(tag));
                        let attr = case
                            .leaker
                            .lines()
                            .find(|line| line.contains(&format!("(id = {},", item.id)));
                        assert_eq!(
                            attr.map_or(false, |a| a.contains("nested_self_entries = true,")),
                            features.nested_self_entries
                        );
                    }
                }
            }
            (found, types)
        };
        let without = Features {
            nested_self_entries: false,
            ..Features::parse("")
        };
        let (found, types) = count(without, gen::TAG_SELF_BOUND_TYPE);
        assert!(
            found > 0,
            "no struct, enum or union of `self_bound_adts` in 12 cases"
        );
        assert_eq!(
            types, 0,
            "the type that names `Self` needs `nested_self_entries`"
        );
        let (found, types) = count(Features::parse(""), gen::TAG_SELF_BOUND_TYPE);
        assert!(
            found > 0 && types > 0,
            "{} items, {} that name `Self`",
            found,
            types
        );
        // Nothing explains a rejection of these.
        let rejected = Report {
            kind: "struct".into(),
            rejections: vec![rejection(
                "`Wrap < T >` has a generic whose bound mentions `Self`, which cannot be leaked",
                "Wrap < T >",
            )],
            ..Default::default()
        };
        let problems = unexplained(
            &tags(&[gen::TAG_SELF_BOUND_ADT, "self_in_bounds"]),
            &rejected,
        );
        assert!(
            problems[0].starts_with("SHOULD BE ACCEPTED"),
            "{:?}",
            problems
        );
        // A bare trait of the prelude is a reason, where the Leaker does not allow the prelude.
        let bare = Report {
            kind: "struct".into(),
            rejections: vec![rejection("use absolute path: 'Clone'", "Clone")],
            ..Default::default()
        };
        assert!(unexplained(&tags(&[gen::TAG_SELF_BOUND_ADT, "bare:Clone"]), &bare).is_empty());
    }

    /// The reports of the accept mode: `acc` is not accepted as it is, and has the errors and what
    /// the macro wrote; `accfail` has the errors that remain too.
    #[test]
    fn the_reports_of_the_accept_mode_are_read_back() {
        let first = "use absolute path: 'Rel'\u{1f}Rel\u{1f}a note\u{1f}RelativePath\u{1f}Type";
        let line = format!(
            "5|trait|acc|entries=3 positions=7 checked=2 skipped=1 twin=0\u{1e}{}",
            first
        );
        let (id, report) = parse_report(&line).unwrap();
        assert_eq!(id, 5);
        assert!(!report.accepted && report.allowed && !report.accept_failed);
        assert_eq!(
            (report.entries, report.positions, report.checked),
            (3, 7, 2)
        );
        assert_eq!(report.rejections.len(), 1);
        assert_eq!(report.rejections[0].tokens, "Rel");
        assert_eq!(report.rejections[0].reason, "RelativePath");
        assert_eq!(report.rejections[0].kind, "Type");
        assert!(report.remaining.is_empty() && report.rewrite_error.is_none());
        // The reason of the rejection is the messages, as for a rejected item.
        assert_eq!(report.detail, "use absolute path: 'Rel'");
        // The rewrite of the copy failed.
        let (_, report) =
            parse_report("5|trait|acc|entries=3 positions=7 rewrite-error=no host\u{1e}").unwrap();
        assert_eq!(
            report.rewrite_error.as_deref(),
            Some("rewrite-error=no host")
        );
        // The second analysis failed.
        let line = format!(
            "6|struct|accfail|use absolute path: 'Rel'\u{1e}{}\u{1c}`Self` has no meaning here\u{1f}Self\u{1f}\u{1f}SelfWithoutMeaning\u{1f}Type",
            first
        );
        let (_, report) = parse_report(&line).unwrap();
        assert!(!report.accepted && !report.allowed && report.accept_failed);
        assert_eq!(report.rejections.len(), 1);
        assert_eq!(report.remaining.len(), 1);
        assert_eq!(report.remaining[0].reason, "SelfWithoutMeaning");
        // A rejection, as before.
        let (_, report) =
            parse_report("7|mod|rej|x\u{1e}use absolute path: 'X'\u{1f}X\u{1f}").unwrap();
        assert!(!report.accepted && !report.allowed && !report.accept_failed);
        assert_eq!(report.rejections[0].reason, "");
    }

    /// The second Referrer's crate of the accept mode invokes each item once, in the module it is
    /// in (the same place of the module tree as in the Leaker's crate), with the same imports.
    #[test]
    fn the_crate_of_the_accepted_items_mirrors_the_modules_of_the_items() {
        let mut nested = 0;
        for seed in 0..12 {
            let case = generate_with(seed, 0, 6, Features::parse(""), Flags::default());
            let text = &case.accept_referrer;
            assert!(
                text.contains("pub mod base { pub use ::tlp_l_000::base::*; }"),
                "{}",
                text
            );
            assert_eq!(case.accept_lines.len(), case.items.len());
            for item in &case.items {
                let call = format!("::__tl_prop_{}!(", item.id);
                let lines: Vec<(usize, &str)> = text
                    .lines()
                    .enumerate()
                    .filter(|(_, line)| line.contains(&call))
                    .collect();
                assert_eq!(lines.len(), 1, "{}", text);
                let (at, line) = lines[0];
                assert!(line.ends_with(", accept);"), "{}", line);
                assert_eq!(line.contains(", strict"), item.strict, "{}", line);
                assert!(case.accept_lines.contains(&(at + 1, item.id)));
                // The path is the module, and the invocation is in that module: the modules above
                // it are open (`pub mod`) and not closed again before it.
                let path = format!(
                    "::tlp_l_000{}",
                    item.module
                        .iter()
                        .map(|m| format!("::{}", m))
                        .collect::<String>()
                );
                assert!(line.contains(&format!("(path = {}", path)), "{}", line);
                let mut open: Vec<&str> = Vec::new();
                for before in text.lines().take(at) {
                    if let Some(name) = before
                        .strip_prefix("pub mod ")
                        .and_then(|r| r.strip_suffix(" {"))
                    {
                        open.push(name);
                    } else if before == "}" {
                        open.pop();
                    }
                }
                // A module analyzed as a whole declares itself at the root.
                let expected: Vec<&str> = item.module.iter().map(String::as_str).collect();
                assert_eq!(open, expected, "{}", line);
                nested += usize::from(item.module.len() > 1);
            }
        }
        assert!(nested > 0, "no item in a nested module in 12 cases");
    }

    /// The crates of `allow_stages` invoke each item once, as the crate of the accepted items does
    /// (the glob ones) or at the root (the clean ones), with the macro of their own, and nothing
    /// generated shadows a name of the prelude, which the copies of `allow_stages` rely on.
    #[test]
    fn the_crates_of_the_stage_copies_mirror_the_accepted_ones_and_the_prelude_is_not_shadowed() {
        for seed in 0..12 {
            let case = generate_with(seed, 0, 6, Features::parse(""), Flags::default());
            assert!(
                shadowed_prelude_names(&case.leaker).is_empty(),
                "{:?}",
                shadowed_prelude_names(&case.leaker)
            );
            for stage in &case.stages {
                let prefix = format!("__tl_st_{}_{}_", stage.rule, stage.side);
                assert_eq!(stage.lines.len(), case.items.len());
                for item in &case.items {
                    let call = format!("::{}{}!(", prefix, item.id);
                    let invoked: Vec<&str> = stage
                        .lib
                        .lines()
                        .filter(|line| line.contains(&call))
                        .collect();
                    assert_eq!(invoked.len(), 1, "{}", stage.lib);
                    assert!(invoked[0].ends_with(", accept);"), "{}", invoked[0]);
                    // The clean crates say so first, in the invocation.
                    assert_eq!(invoked[0].contains("(clean path = "), stage.clean);
                }
                if !stage.clean {
                    // Only the macros differ: the modules and imports are those of the accepted items.
                    let accepted = case.accept_referrer.replace("__tl_prop_", &prefix);
                    assert_eq!(accepted, stage.lib);
                } else {
                    // None of the Leaker's names but the ones the rule allows: no glob import, `base`
                    // where `crate::base` or `super::base` is allowed, and the modules where the paths
                    // are relative to the module.
                    assert!(stage.lib.matches("::*;").count() <= 1, "{}", stage.lib);
                    let base = matches!(stage.rule, "crate" | "path" | "values" | "default");
                    assert_eq!(stage.lib.contains("pub mod base {"), base, "{}", stage.lib);
                    let modules = matches!(stage.rule, "path" | "values" | "default");
                    assert_eq!(stage.lib.contains("pub mod m0 {"), modules, "{}", stage.lib);
                }
            }
        }
    }

    #[test]
    fn a_name_of_the_prelude_that_a_crate_declares_or_imports_is_found() {
        let found = shadowed_prelude_names;
        assert_eq!(
            found("pub struct Vec<T>(T); enum Option {}"),
            ["Option", "Vec"]
        );
        assert_eq!(
            found("pub type u8 = i32; pub trait Clone {} fn drop() {}"),
            ["Clone", "drop", "u8"]
        );
        assert_eq!(
            found("mod Box {} macro_rules! Result { () => {} }"),
            ["Box", "Result"]
        );
        assert_eq!(
            found("use a::b::Vec; pub use c::{Foo, String};"),
            ["String", "Vec"]
        );
        // Importing under another name does not shadow, and importing as the name does; a use of
        // the name is not a declaration.
        assert_eq!(
            found("use a::Vec as MyVec; use a::{Foo, Bar as Option};"),
            ["Option"]
        );
        assert!(
            found("pub struct R0; fn f(x: Vec<R0>, y: Option<u8>) -> Result<(), String> {}")
                .is_empty()
        );
        // The keyword in a type is no declaration: `fn(u8)`, `*const u8`, `&'static u8`.
        assert!(found(
            "type F = fn(u8) -> u8; type P = *const u8; type S = &'static u8; const N: u8 = 3;"
        )
        .iter()
        .all(|name| *name != "u8"));
    }

    fn allowed(reason: &str, tokens: &str) -> Rejection {
        Rejection {
            tokens: tokens.to_string(),
            reason: reason.to_string(),
            kind: "Type".to_string(),
            ..Default::default()
        }
    }

    fn allowed_report(errors: Vec<Rejection>) -> Report {
        Report {
            kind: "trait".into(),
            allowed: true,
            rejections: errors,
            ..Default::default()
        }
    }

    /// The failures of the copy of an item accepted by `allow()` are the ones that the docs of
    /// `LeakerResult::allow()` list, on an item that has the shape.
    #[test]
    fn a_failure_of_an_accepted_copy_needs_a_documented_class_and_a_shape() {
        let unprovable_assoc = allowed_report(vec![allowed("UnprovableBound", "Self :: A0")]);
        let unprovable_self = allowed_report(vec![allowed("UnprovableBound", "Self")]);
        let denied_self = allowed_report(vec![allowed("RelativePath", "Self")]);
        let classify = |tags_of: &[&str], codes: &[&str], report: &Report| {
            unexplained_accept(&tags(tags_of), codes, report)
        };
        // (i): a predicate that names `Self`, and a bound that names a leaked type.
        assert_eq!(
            classify(
                &["bound_cycle", "self_in_predicate"],
                &["E0277"],
                &unprovable_assoc
            ),
            Ok("(i)")
        );
        assert_eq!(
            classify(
                &["bound_cycle", "self_in_predicate"],
                &["E0277"],
                &unprovable_self
            ),
            Ok("(i)")
        );
        // The shape is missing: the item has no such predicate, or no such bound.
        for missing in [&["bound_cycle"][..], &["self_in_predicate"][..]] {
            let problems = classify(missing, &["E0277"], &unprovable_self).unwrap_err();
            assert!(
                problems[0].starts_with("NO SHAPE THAT GIVES IT"),
                "{:?}",
                problems
            );
        }
        // (ii): the alias bound explains `Self::A`, but not `Self`.
        assert_eq!(
            classify(&["alias_bound"], &["E0277", "E0277"], &unprovable_assoc),
            Ok("(ii)")
        );
        assert!(classify(&["alias_bound"], &["E0277"], &unprovable_self).is_err());
        // (iii): `Self` denied, in a function pointer that hides a lifetime.
        assert_eq!(
            classify(&["self_denied", "hidden_self"], &["E0106"], &denied_self),
            Ok("(iii)")
        );
        assert!(
            classify(&["self_denied"], &["E0106"], &denied_self).unwrap_err()[0]
                .starts_with("NO SHAPE THAT GIVES IT")
        );
        // Anything else is not a known failure: the lifetime that E0195 was about, another code, an
        // error that was not allowed, every diagnostic of the item counts.
        let all = &[
            "bound_cycle",
            "self_in_predicate",
            "alias_bound",
            "self_denied",
            "hidden_self",
        ];
        let unconstrained = allowed_report(vec![allowed("UnconstrainedLifetime", "Wrap < 'b >")]);
        for (codes, report) in [
            (&["E0195"][..], &unconstrained),
            (&["E0308"][..], &unprovable_assoc),
            (&["E0277"][..], &denied_self),
            (&["E0277"][..], &allowed_report(Vec::new())),
            (&["E0277", "E0425"][..], &unprovable_assoc),
        ] {
            let problems = classify(all, codes, report).unwrap_err();
            assert!(
                problems
                    .iter()
                    .all(|p| p.starts_with("NOT A KNOWN FAILURE")),
                "{:?}",
                problems
            );
        }
    }

    #[test]
    fn a_predicate_that_names_self_is_found_in_a_where_clause() {
        let found = gen::names_self_in_where;
        assert!(found("fn m<X>() where X: Into<Self::A> { loop {} }"));
        assert!(found("fn m<X>(&self) where X: Fn(&Self) -> u8;"));
        // An array type in the clause has a `;` in it, and the predicate after it counts.
        assert!(found(
            "fn m<U>() where U: From<([u16; 1], u8)>, U: Fn(&Self) -> u8, B<U>: Default { loop {} }"
        ));
        // The predicates on `Self` itself do not, nor does a `Self` outside a where clause.
        assert!(!found("fn m(&self) where Self: Sized, Self: 'x;"));
        assert!(!found(
            "fn m(&self, x: Wrap<Self>) -> Self::A where X: Clone;"
        ));
        assert!(!found(
            "struct S<T>(T) where T: Clone; impl S<u8> { fn m(&self) -> Self { loop {} } }"
        ));
    }

    #[test]
    fn the_shapes_of_the_failures_of_accepted_copies_are_generated() {
        let mut found = std::collections::BTreeSet::new();
        for seed in 0..40 {
            let case = generate_with(seed, 0, 6, Features::parse(""), Flags::default());
            for item in &case.items {
                for tag in [
                    gen::TAG_SELF_IN_PREDICATE,
                    gen::TAG_ALIAS_BOUND,
                    gen::TAG_HIDDEN_SELF,
                ] {
                    if item.tags.contains(tag) {
                        found.insert(tag);
                    }
                }
            }
        }
        assert_eq!(found.len(), 3, "{:?}", found);
    }

    #[test]
    fn the_flags_of_the_generator_are_off_by_default_and_name_their_rules() {
        let default = Flags::default();
        assert_eq!(default.rules(), ["prelude", "crate", "path", "default"]);
        let all = Flags {
            allow_stages: true,
            relative_std: true,
            values: true,
        };
        assert_eq!(
            all.rules(),
            ["prelude", "crate", "path", "std", "values", "default"]
        );
        // Without them the generated crates have no value, no relative path of `std`, no `alloc`.
        for seed in 0..6 {
            let case = generate_with(seed, 0, 6, Features::parse(""), default);
            for text in [&case.leaker, &case.referrer, &case.accept_referrer] {
                assert!(!text.contains("extern crate alloc;"), "{}", text);
            }
            assert!(!case.leaker.contains("bodies = true"));
            assert!(!case.leaker.contains("fn bf()"));
            assert!(!case.leaker.contains("\"std,core,alloc\""));
        }
    }

    #[test]
    fn the_values_and_the_paths_of_std_are_generated_with_their_flags() {
        let flags = Flags {
            allow_stages: true,
            relative_std: true,
            values: true,
        };
        let (mut bodies, mut relative, mut paths, mut alloc) = (0, 0, 0, 0);
        for seed in 0..12 {
            let case = generate_with(seed, 0, 6, Features::parse(""), flags);
            // The Leaker's crate and the Referrer's have `alloc`, and the items have the arguments.
            assert!(case.leaker.contains("extern crate alloc;"));
            assert!(case.referrer.contains("extern crate alloc;"));
            assert!(case.leaker.contains("pub const fn bf() -> usize"));
            assert!(case
                .leaker
                .contains("pub const fn bg<T: ?Sized>() -> usize"));
            bodies += case.leaker.matches("bodies = true").count();
            relative += case.leaker.matches(" core::").count();
            paths += case
                .items
                .iter()
                .filter(|item| item.tags.contains(gen::TAG_VALUE_PATH))
                .count();
            alloc += case.leaker.matches(" alloc::").count();
            // A value that the Leaker reads says so, and an item that has one also is checked as its
            // signatures are.
            for line in case
                .leaker
                .lines()
                .filter(|line| line.contains("values = true"))
            {
                assert!(line.contains("bodies = true"), "{}", line);
            }
            // All the rules, in a crate with globs and a clean one, after the result and the config.
            assert_eq!(case.stages.len(), 4 * flags.rules().len());
        }
        assert!(bodies > 0 && relative > 0 && paths > 0 && alloc > 0);
    }

    /// `relativize_std()` only drops the leading `::` of `std` and `core` (and says `alloc` where
    /// `alloc` has the item), leaves the attribute, a path that hides a lifetime and an item with
    /// bounds that name its `Self` alone, and does nothing in a module analyzed as a whole.
    #[test]
    fn the_paths_of_std_are_relativized_where_they_are_safe() {
        let flags = Flags {
            relative_std: true,
            ..Flags::default()
        };
        let text = "#[@ATTR@(id = 1, allow = \"::std,std\")]\n\
                    pub struct S(::std::vec::Vec<::core::marker::PhantomData<u8>>, <::core::option::Option<u8>>, \
                    fn(::core::str::Chars) -> &u8, ::std::collections::BTreeMap<u8, ::std::sync::Arc<u8>>);\n";
        // The normal form: relative, and `alloc` as `std`.
        let normal = |text: &str| {
            text.replace("::std::", "std::")
                .replace("::core::", "core::")
                .replace("alloc::", "std::")
        };
        let (mut relative, mut alloc) = (0, 0);
        for seed in 0..40 {
            let mut g = gen::Gen::new(seed, "tlp_l_000".into(), Features::parse(""), flags);
            let out = g.relativize_std(text);
            assert_eq!(normal(&out), normal(text), "{}", out);
            assert!(out.starts_with(text.lines().next().unwrap()));
            assert!(out.contains("fn(::core::str::Chars) -> &u8"), "{}", out);
            relative += out.matches("core::").count() - out.matches("::core::").count();
            alloc += out.matches("alloc::").count();
        }
        assert!(relative > 0 && alloc > 0);
        // Not asked for: unchanged.
        let mut g = gen::Gen::new(1, "tlp_l_000".into(), Features::parse(""), Flags::default());
        assert_eq!(g.relativize_std(text), text);
    }

    #[test]
    fn reports_are_read_back() {
        let (id, r) = parse_report("7|struct|ok|entries=2 positions=5 checked=4").unwrap();
        assert_eq!(
            (id, r.accepted, r.entries, r.positions, r.checked),
            (7, true, 2, 5, 4)
        );
        let (id, r) = parse_report("8|mod|rej|use absolute path: 'X'").unwrap();
        assert_eq!((id, r.accepted), (8, false));
        assert!(parse_report("nonsense").is_none());
    }
}

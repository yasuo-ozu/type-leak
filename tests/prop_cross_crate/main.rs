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
//!   without an error saying why (a crate of the case that has a diagnostic does not excuse the
//!   others), a binary that fails or prints a line that is not a report, and an item with no
//!   report (`missing_artifacts()`). None of them is read as "rejected".
//! - The acceptance floor. The Leaker must accept at least a share of the items analyzed of each
//!   kind (struct, enum, union, trait, inherent impl, trait impl, module), of each scope they
//!   are analyzed through, and of each construct they have (a trait with a generic associated
//!   type, a const parameter, an `impl Trait` argument, a default, a `for<>` binder, ...: the tags
//!   `construct:<name>` that the generator puts, `gen::CONSTRUCTS`, counted in a table of the
//!   summary), and the total accepted must not be 0 (`Floors`, `FLOORS`: they come from the rates
//!   measured over several seeds, and the failure says which is below its floor and why its items
//!   were rejected). A kind with fewer than `TYPE_LEAK_PROP_MIN_SAMPLES` items is not checked, and
//!   a construct with fewer than three times that. At least `TYPE_LEAK_PROP_MIN_CHECKED` percent
//!   of the type positions of the accepted items must have been compared with their witness
//!   (turning the floors off with `TYPE_LEAK_PROP_MIN_ACCEPT=0` does not turn this off).
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
//! (`family()`, which decides by the `Reason` of the error, and reads its message, tokens and note
//! only to tell the shapes of one `Reason` apart: a leaked type in a bound or where predicate that
//! rustc cannot normalize, a bare trait of the prelude where the Leaker does not allow the prelude,
//! `Self::A` where `Self` is a type, the lifetime of a binder that no input keeps, a relative
//! constant that cannot be leaked, ...), and the item must have a shape that gives it: the
//! generator tags each item with the shapes it has (`gen::Item::tags`: `bound_cycle`,
//! `self_assoc`, `binder_no_anchor`, `bare:Fn`, ...). A reason outside the list fails the run
//! ("NOT A KNOWN REASON") with the item and the message, and so does a known one on an item that
//! has no tag for it ("NO SHAPE THAT GIVES IT").
//! Either is a defect of `type-leak`, or a limitation to put in the list with its shape.
//! `TYPE_LEAK_PROP_REJECTIONS=1` prints each rejection with the tags of its item.
//!
//! A tag is put on what the item writes, and not on a share of the items that may have the shape
//! (the tag `bound_cycle` was on every item of the share that may have a cycle, and a Leaker that
//! rejected the predicates that an impl proves, `Rel<T>: Default`, as bounds of the environment,
//! rejected 20 items of 60 cases that it explained). `Gen::note_text()` reads the final text of
//! each item (and of each part of a module), `Gen::note_bounds()` among the rest:
//!
//! - an `UnprovableBound` is a cycle between a type in a predicate and a bound of a generic that
//!   names a leaked type (`T: Fn(Rel) -> u8`). The note of the error names that bound; it must be
//!   one that the item wrote: a tag `bound_cycle=<bounded>|<bound>` of the same generic and with
//!   that bound (`Agree::Bound`). A predicate on a type that is no generic is proved by an impl
//!   and is none, unless it names the `Item` of an iterator;
//! - an `UnconstrainedLifetime` needs a leaked path or trait object that holds the lifetime of a
//!   method, behind a reference in its arguments too (`Rel<&'x str>`): `method_lifetime=<type>`,
//!   which the cause is (`Agree::Type`);
//! - `bare:<Name>` is read from the text of the item where the Leaker does not allow the prelude.
//!
//! What a Leaker leaks is read from the text and from the settings of the Leaker's crate
//! (`Gen::names_leaked_type()`: a relative path, `Self`, a path of `std` or of the Leaker's crate
//! that it does not allow, a primitive type if it does not allow them), and not from what the
//! generator meant. A module has the tags of all its items, and each reason needs one tag: the
//! rejection of a module is explained by an item of it that has the shape (for an `UnprovableBound`
//! or an `UnconstrainedLifetime`, by the text that the item wrote).
//!
//! # Accepting the rejections (`TYPE_LEAK_PROP_ACCEPT`)
//!
//! The Leaker copies what conflicts as written and goes on: of a type that cannot be leaked, the
//! path of its head (what is in its arguments is analyzed on its own, see `tests/flat_errors.rs`),
//! and whole the trait bounds, the types in predicates, the references and the holders of a bare
//! name that may be a const. The recovered Referrer (`LeakerResult::into_referrer()`) has the
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
//! A copy of an item accepted by `allow()` may not build, for the failures that `accept_families()`
//! lists (i, ii and iii, described there), and the test explains each failure the way it explains a
//! rejection (`unexplained_accept()`): the diagnostic is one of the listed failures, if its error
//! code (`E0277`, `E0106`), the errors that were allowed (their `Reason` and the kind and tokens of
//! their cause) and the shapes of the item (its tags) say so. The failure is counted and is no
//! failure of the run. Anything else fails the run: "ACCEPT: NOT A KNOWN FAILURE" for a diagnostic
//! that no listed failure explains, and "ACCEPT: NO SHAPE THAT GIVES IT" for a listed one on an item
//! that has not the shape. Either is a defect, or a limitation to add to `accept_families()`, with
//! its shape.
//!
//! The summary counts them by class. The generator avoids none of these shapes: the rejected ones
//! (`gen::Gen::cyclic`) have them.
//!
//! # The copies after the analysis without a rule (`TYPE_LEAK_PROP_FEATURES=allow_stages`)
//!
//! `Rules` given to the Leaker and to `LeakerResult::allow()` are one rule, but the result judges an
//! analysis made without it, so for a path rule it can leave other errors and other entries (see
//! `tests/allow_stages.rs`). With `allow_stages` the macro of each item takes each rule of
//! `gen::rules()` in turn and makes two copies, and the test builds them, to see whether the
//! difference is more than a number:
//!
//! - after the result: the item is analyzed without the rule (the primitive types and the names of
//!   the prelude allowed, but for the rule `prelude`), the result is given the rule, then
//!   `allow(cause)` for the cause of each error that remains until none does (the errors are a flat
//!   list, so once), and its Referrer copies the item (`__tl_st_<rule>_res_<id>!`);
//! - after the config: the item is analyzed with a config that has the rule, the causes of the
//!   errors that remain are allowed, and its Referrer copies the item (`__tl_st_<rule>_cfg_<id>!`).
//!
//! The rules are `prelude` (`Rules::prelude()`, every item), `crate` (`Rules::crate_paths()`), `path`
//! (`Rules::or_if(super::base)`: the module `base` as the modules at depth 1 name it), `std`
//! (`Rules::std_crates()`, with `relative_std`), `values` (`Rules::values()`, with `values`) and
//! `default` (`Rules::defaults()`). A rule is made for the items where it matters: it drops or trims an
//! error of the analysis without it, or the config finds other errors or another Referrer than the
//! result. `values` and `default` are place rules: the config and the result must give the same
//! errors (`STAGES DIFFER` fails the run), but not the same Referrer: the config writes the paths of a
//! value or a default as they are, and records no part for them.
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
//!   modules of the Leaker with `RLEN` and `ILEN` for the paths of values and defaults after the
//!   result, and for the rules `values` and `default` after the config, which writes every relative
//!   path of a value or a default as it is, the `use`s of each module and a glob import of it (the
//!   names of a default are those of the types, so there this crate cannot tell a type kept by
//!   mistake in another place; the copy after the result is still checked in the crate with the
//!   constants only, and fails as `STAGES CLEAN` if it needs more). A relative type that the rule
//!   allowed to drop with an error but that the copy kept as written fails to compile there. (Remove
//!   `pub mod base` from the crate of the rule `crate` and it does.) The copies after the result
//!   keep the trust that the rule gives (`Rules::std_crates()` trusts the optional entries of
//!   `std`): the serialized Referrer does not, so the Referrer's crate gives it again. An item
//!   invoked with `all` keeps it, as the items with a shape that breaks when an optional entry is
//!   replaced are.
//!
//! A copy after the result that does not compile is one of
//!
//! - a failure that `accept_families()` lists (explained by the errors of the analysis without the
//!   rule), counted by class; when the copy after the config compiles, it is the rule after the
//!   analysis that makes the copy fail;
//! - `SHARED`: no listed failure explains it, but the copy after the config fails too, so the rule
//!   after the analysis is not what makes it fail. Not a failure of the run, and printed with what
//!   explains the copy after the config;
//! - `NEW`: only the copy after the result fails, for a reason that `accept_families()` does not
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
//! applies to the real `Sized`): the Leaker accepts it without the prelude, so it has no tag and
//! its twin is compared either way.
//!
//! # What the Referrer trusts
//!
//! An optional entry is a type whose own path is absolute into another crate (`::tlp_l_000::base::B1<u8>`,
//! `::std::rc::Rc<u8>`). A Referrer replaces it through the Repeater unless the rule it gives to
//! `LeakerResult::allow()` trusts every crate the entry names (its type, the bounds of its context).
//! Each item is invoked with a mode (`gen::Trust`, drawn per item, 40/20/20/20 of the modes that can
//! differ from the first in the case): `none` (the library's default, `strict`), `all`
//! (`Rules::external_crates()`), `std` (`Rules::std_crates()`) and `leaker` (a path rule for the
//! Leaker's crate), so entries that name two crates (`::tlp_l_000::base::B1<::std::rc::Rc<u8>>`) are
//! trusted by none of the partial modes. The macro writes the Repeater's impls of the required
//! entries and of the optional ones the mode does not trust, and nothing for the others: a Referrer
//! that replaced a trusted entry fails to compile.
//!
//! Two checks tell a library that trusts too much or too little:
//!
//! - The macro writes out which entries the mode trusts, from the crates each names
//!   (`Trust::trusts()`, `Referrer::external_crates_require()`), and compares it with the entries the
//!   copy replaces, in the Leaker's crate and in the Referrer's (`TRUST DISAGREES`). It tells a
//!   trust that ignores the bounds of the context of an entry, which the crate cannot: the copy has
//!   those bounds as written whatever is trusted.
//! - The decoy crate (`gen::Decoy`, `DECOY_COPY`). An absolute path names the same crate in both
//!   crates, so a Referrer that trusts too much still compiles. In the Referrer's crate
//!   `tlp_x_NNN`, the dependency `tlp_l_NNN` is another crate with the same names in `base`
//!   (`tlp_d_NNN`, another `BLEN`), and the Leaker's crate is `real_NNN`: the macros are invoked as
//!   `real_NNN::__tl_prop_N!(path = ::real_NNN::m0, ..)`, and they name the Repeater and the
//!   witnesses through that path, so a replaced entry is the Leaker's type, and one written as it is
//!   is the decoy's, which the witness does not accept. It is for the items of the modes that do not
//!   trust the Leaker's crate (`none` and `std`) in a case whose Leaker does not allow its crate
//!   (a promise that it is the same crate); not a module (the original is written again for the
//!   crate), nor an item whose copy names the crate otherwise (the header of an impl, a bound).
//!
//! # Other checks of the macro
//!
//! - The helpers agree: `repeater_impl()` declares the generics of the context of its entry
//!   without the bounds its arguments encode, and the parameter of the implementor, `required()`
//!   and `must()` say the same, and `required()` comes first (`HELPERS DISAGREE`).
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
//! - `TYPE_LEAK_PROP_SEED` (default 1): the seed of the generator. A case is generated from the
//!   seed and its number alone, with a fixed hash (`gen::StableHasher`), so a seed gives the same
//!   cases on every Rust release.
//! - `TYPE_LEAK_PROP_CASE`: only this case (a number below `TYPE_LEAK_PROP_CASES`) is built and
//!   checked. It has the same items as in the full run of the seed, and its crates keep their
//!   number (`c017_l`). The failure message prints the command that reproduces a failure.
//! - `TYPE_LEAK_PROP_ITEMS` (default 8, at least 4): the most items in a module. A module has
//!   between 4 and that many.
//! - `TYPE_LEAK_PROP_MIN_ACCEPT`: the acceptance floors, in percent. A number sets every kind and
//!   scope, `name=number` one of them, read in order: `0` turns the floors off (the total accepted
//!   must still be positive, and `TYPE_LEAK_PROP_MIN_CHECKED` still counts), `40,mod=10,enter_impl=70`
//!   is 40% but 10% for modules and 70% for `enter_impl`. The names are the kinds of the table of the summary and the scopes `enter_mod`,
//!   `enter_generics`, `enter_trait` and `enter_impl`.
//! - `TYPE_LEAK_PROP_MIN_SAMPLES` (default 12): the fewest items of a kind for its floor to count
//!   (three times as many for a construct).
//! - `TYPE_LEAK_PROP_MIN_CHECKED` (default 85): the least percentage of type positions compared.
//! - `TYPE_LEAK_PROP_REJECTIONS`: if set (and not `0`), the rejections are printed with the tags of
//!   their items, and with the `Reason` and the kind of the cause of each error.
//! - `TYPE_LEAK_PROP_ACCEPT`: if set (and not `0`), the Leaker accepts the items it rejects by
//!   `LeakerResult::allow(cause)` of the cause of every error, and the test compiles the result: see
//!   "Accepting the rejections".
//! - `TYPE_LEAK_PROP_FEATURES`: a comma-separated list of constructs that `type-leak` is known to
//!   fail on (see `gen::Features`). The generator avoids them by default, so that the test passes
//!   on the current library, and a new failure is a new finding. A name turns a feature on and
//!   `-name` turns it off, also one that is on by default (`-nested_self_entries`); `all` turns
//!   every feature on, and the words are read in order (`all,-strict_anchors`). With one of `inferred_outlives`,
//!   `impl_self_outlives`, or `all`, the test fails with the reproduction of that finding
//!   (`tests/findings.rs` has each one as a library-level test). `strict_anchors` (F18, about the
//!   items whose Referrer replaces optional entries, the modes but `all`) is fixed, and off by
//!   default: the items that have it are invoked with `all` unless it is asked for. `unsized_params`,
//!   `bound_cycles`, `impl_assoc_self`, `method_lifetime_paths`, `module_crate_paths`,
//!   `bound_merge_syntax`, `fn_pointer_self`, `elided_fn_generics`, `impl_self_bounds` (F13),
//!   `impl_pred_leaked` (F14), `self_elision` (F15, F23), `implied_outlives` (F16),
//!   `strict_cycles` (F17, about the items whose Referrer replaces optional entries too), `impl_self_bound_cycle` (F19),
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
//!   `nested_self_entries` for the items nested in `enter_mod`, and those of
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
//!   rule" and builds their copies (the workspace has `-stages` in its name); `relative_std`
//!   writes about a third of the paths of `std` and `core` of the items analyzed one by one without
//!   the leading `::` (some `alloc::`), which the Leaker allows (`allow = "std,core,alloc"`) but for
//!   the rule `std`, and `extern crate alloc;` is in every crate; `values` gives the items values
//!   (`gen::Flags::values`): bodies of functions, methods and default methods, the values of
//!   constants and statics, the defaults of associated constants and of const parameters, with
//!   the types of the Leaker's crate in a `let`, a turbofish and a cast, which the Leaker leaks with
//!   `check_values` and the witnesses check (`Args::bodies`), and in a share of them a relative
//!   path of a value (`crate::base::bf()`, `RLEN`), which it rejects (the tag `value_path`).
//!
//! The test is ignored by default: it is slow, as it builds the generated crates. Run it with
//!
//! ```text
//! cargo test --test prop_cross_crate -- --ignored
//! TYPE_LEAK_PROP_CASES=200 TYPE_LEAK_PROP_SEED=7 cargo test --test prop_cross_crate -- --ignored
//! ```
//!
//! The sources of every case stay in `target/tmp/prop/<workspace>/`, and each is a workspace member
//! that builds on its own (`cargo build` there, with `CARGO_TARGET_DIR` as the test has it). The
//! workspace is `ws-<seed>-<cases>-i<items>`, then `-accept`, `-stages`, `-f<hash of
//! TYPE_LEAK_PROP_FEATURES>` and `-case<n>` as they apply (`workspace_name()`), so that two runs
//! that differ in any of them do not delete each other's sources. Next to it, if a Leaker's crate
//! does not compile:
//!
//! - `<workspace>-identity` has the Leaker's crates of those cases without `type-leak` (a crate
//!   that does not compile there is a bug of the generator);
//! - `<workspace>-ordinary`, with `TYPE_LEAK_PROP_ACCEPT` or `allow_stages`, has them with the
//!   macros of an ordinary run (no accepted item, no stage copy), to tell whether the output of the
//!   accepted items or of the stage copies is what does not compile;
//! - `<workspace>-accept-only`, with both, has them with the accepted items and no stage copy, to
//!   tell which of the two it is.
//!
//! The builds run with `CARGO_INCREMENTAL=0`, with the `cargo` that built the test and the `rustc` of
//! its toolchain (the one next to it, unless `RUSTC` is set), and the `Cargo.lock` of the repository.

mod gen;
mod json;

use gen::{
    generate, generate_with, Case, Features, Flags, StableHasher, CONSTRUCTS, KINDS, MIN_ITEMS,
    SCOPES,
};
use json::Json;
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

/// Whether a setting says yes: anything but nothing or `0`.
fn is_on(value: &str) -> bool {
    !matches!(value.trim(), "" | "0")
}

/// Whether the variable is set to something other than nothing or `0`.
fn env_flag(name: &str) -> bool {
    std::env::var(name).map_or(false, |value| is_on(&value))
}

/// The seed of the generator for a case: a function of the seed and the number of the case alone,
/// so that a case is the same in a run of one case (`TYPE_LEAK_PROP_CASE`) and in a full run.
fn case_seed(seed: u64, index: usize) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.write_u64(seed);
    hasher.write_u64(index as u64);
    hasher.finish()
}

/// What a run was started with: the settings that make its cases, and what its failure message
/// repeats.
#[derive(Clone)]
struct Run {
    seed: u64,
    cases: usize,
    items: usize,
    /// `TYPE_LEAK_PROP_FEATURES` as it is.
    features: String,
    accept: bool,
    stages: bool,
    /// `TYPE_LEAK_PROP_CASE`.
    only: Option<usize>,
}

impl Run {
    /// The directory of the sources (under `target/tmp/prop`): runs that differ in any setting that
    /// changes the sources do not delete each other's.
    fn workspace_name(&self) -> String {
        let mut name = format!("ws-{}-{}-i{}", self.seed, self.cases, self.items);
        if self.accept {
            name += "-accept";
        }
        if self.stages {
            name += "-stages";
        }
        let features = self.features.trim();
        if !features.is_empty() {
            let mut hasher = StableHasher::new();
            hasher.write(features.as_bytes());
            write!(name, "-f{:08x}", hasher.finish() >> 32).unwrap();
        }
        if let Some(case) = self.only {
            write!(name, "-case{}", case).unwrap();
        }
        name
    }

    /// The command that repeats the run, or only `case` of it (the case of a failure).
    fn command(&self, case: Option<usize>) -> String {
        let mut command = format!(
            "TYPE_LEAK_PROP_SEED={} TYPE_LEAK_PROP_CASES={} TYPE_LEAK_PROP_ITEMS={}",
            self.seed, self.cases, self.items
        );
        if !self.features.trim().is_empty() {
            write!(
                command,
                " TYPE_LEAK_PROP_FEATURES=\"{}\"",
                self.features.trim()
            )
            .unwrap();
        }
        if self.accept {
            command += " TYPE_LEAK_PROP_ACCEPT=1";
        }
        if let Some(case) = case.or(self.only) {
            write!(command, " TYPE_LEAK_PROP_CASE={}", case).unwrap();
        }
        command + " cargo test --test prop_cross_crate -- --ignored --nocapture"
    }
}

/// The cases that run: all of them, or the one `TYPE_LEAK_PROP_CASE` names.
fn selected_cases(cases: usize, only: Option<usize>) -> Result<Vec<usize>, String> {
    match only {
        None => Ok((0..cases).collect()),
        Some(case) if case < cases => Ok(vec![case]),
        Some(case) => Err(format!(
            "TYPE_LEAK_PROP_CASE={} is not a case of TYPE_LEAK_PROP_CASES={} (they are 0..{})",
            case, cases, cases
        )),
    }
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

/// The decoy crate of a case (`gen::Decoy`), which `tlp_l_NNN` means in the decoy Referrer's crate.
fn decoy_dir(i: usize) -> String {
    format!("c{:03}_d", i)
}

fn decoy_name(i: usize) -> String {
    format!("tlp_d_{:03}", i)
}

/// The Referrer's crate that has the decoy as `tlp_l_NNN` and the Leaker's as `real_NNN`.
fn decoy_referrer_dir(i: usize) -> String {
    format!("c{:03}_x", i)
}

fn decoy_referrer_name(i: usize) -> String {
    format!("tlp_x_{:03}", i)
}

/// Which crate of `allow_stages` a failure is in: the rule, the side (`res` or `cfg`) and the layout
/// (`glob` or `clean`), see `gen::StageCrate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct StageKey {
    rule: &'static str,
    side: &'static str,
    clean: bool,
}

impl StageKey {
    /// What names the crate of the case `i`.
    fn suffix(self) -> String {
        format!(
            "{}_{}_{}",
            self.rule,
            self.side,
            if self.clean { "clean" } else { "glob" }
        )
    }

    fn dir(self, i: usize) -> String {
        format!("c{:03}_{}", i, self.suffix())
    }

    fn name(self, i: usize) -> String {
        format!("tlp_{}_{:03}", self.suffix(), i)
    }
}

fn stage_dir(i: usize, stage: &gen::StageCrate) -> String {
    StageKey::from(stage).dir(i)
}

fn stage_name(i: usize, stage: &gen::StageCrate) -> String {
    StageKey::from(stage).name(i)
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
            if mode != Mode::Identity && case.decoy.is_some() {
                dirs.push(format!("\"{}\"", decoy_dir(*i)));
                dirs.push(format!("\"{}\"", decoy_referrer_dir(*i)));
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
    // The versions of the repository's lock, if it has one (it is not committed).
    let lock = root().join("Cargo.lock");
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

        if let Some(decoy) = &case.decoy {
            let package = |name: String, dependencies: String| {
                format!(
                    "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
                     [dependencies]\n{}",
                    name, dependencies
                )
            };
            let dir = ws.join(decoy_dir(*i));
            std::fs::create_dir_all(dir.join("src")).unwrap();
            std::fs::write(
                dir.join("Cargo.toml"),
                package(decoy_name(*i), String::new()),
            )
            .unwrap();
            std::fs::write(dir.join("src/lib.rs"), &decoy.lib).unwrap();
            // The decoy is `tlp_l_NNN` there, and the Leaker's crate is `real_NNN`.
            let dir = ws.join(decoy_referrer_dir(*i));
            std::fs::create_dir_all(dir.join("src")).unwrap();
            std::fs::write(
                dir.join("Cargo.toml"),
                package(
                    decoy_referrer_name(*i),
                    format!(
                        "{} = {{ package = \"{}\", path = \"../{}\" }}\n{} = {{ package = \"{}\", path = \"../{}\" }}\n",
                        leaker_name(*i),
                        decoy_name(*i),
                        decoy_dir(*i),
                        gen::real_name(&leaker_name(*i)),
                        leaker_name(*i),
                        leaker_dir(*i)
                    ),
                ),
            )
            .unwrap();
            std::fs::write(dir.join("src/lib.rs"), &decoy.referrer).unwrap();
        }

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
/// `accept_families()` explains if they are the failures it lists.
const ACCEPT_COPY: &str =
    "ACCEPT: the copy of an item accepted by allow() does not compile in the Referrer's crate";

/// The class of the diagnostics of the decoy Referrer's crate (`gen::Decoy`): the copies of the items
/// that do not trust the Leaker's crate, in a crate where `::tlp_l_NNN` is another crate. One that
/// does not compile is a Referrer that wrote an entry as it is although it should have replaced it,
/// or a copy that names the Leaker's crate by a path the macro did not skip.
const DECOY_COPY: &str =
    "DECOY: the copy in the Referrer's crate where the Leaker's crate name is another crate does not compile";

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

/// The `rustc` of the toolchain of the `cargo` that built this test (the one next to it), so that
/// the generated crates are not built by another compiler than the one that runs the test (a
/// nightly default toolchain, with `cargo +stable test`). `None` if `RUSTC` is set: the choice of
/// whoever runs the test.
fn toolchain_rustc() -> Option<PathBuf> {
    if std::env::var_os("RUSTC").is_some() {
        return None;
    }
    let rustc =
        Path::new(env!("CARGO")).with_file_name(format!("rustc{}", std::env::consts::EXE_SUFFIX));
    if rustc.is_file() {
        Some(rustc)
    } else {
        None
    }
}

/// Builds the workspace with the `cargo` and the `rustc` that run the test, without incremental
/// compilation (the sources are built once, and the cache of a generated crate would only grow).
fn cargo_build(ws: &Path, target: &Path) -> Build {
    let mut command = Command::new(env!("CARGO"));
    command
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
        .env("CARGO_INCREMENTAL", "0");
    if let Some(rustc) = toolchain_rustc() {
        command.env("RUSTC", rustc);
    }
    let output = command.output().unwrap();
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
    /// The impl headers compared, and those left out: rustc cannot normalize them
    /// (`prop::header_cycles()`).
    headers: usize,
    header_cycles: usize,
    /// The optional entries whose Repeater impl the macro did not write (the mode of the Referrer
    /// trusts them).
    skipped: usize,
    /// The item has bare traits of the prelude and was compared with its twin.
    twin: bool,
    /// The optional entries by the crates they name: `std` only, the Leaker's crate only, both,
    /// another; and how many of them the Referrer's mode trusts (`prop::optional_entries()`).
    optional: [usize; 4],
    trusted: usize,
    /// The decoy crate has a copy of the item, if its mode lets it (`prop::apply()`).
    decoyed: bool,
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
                Some(("headers", n)) => report.headers = n.parse().unwrap_or(0),
                Some(("header_cycles", n)) => report.header_cycles = n.parse().unwrap_or(0),
                Some(("twin", n)) => report.twin = n == "1",
                Some(("optstd", n)) => report.optional[0] = n.parse().unwrap_or(0),
                Some(("optkr", n)) => report.optional[1] = n.parse().unwrap_or(0),
                Some(("optboth", n)) => report.optional[2] = n.parse().unwrap_or(0),
                Some(("optother", n)) => report.optional[3] = n.parse().unwrap_or(0),
                Some(("trusted", n)) => report.trusted = n.parse().unwrap_or(0),
                Some(("decoyed", n)) => report.decoyed = n == "1",
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
/// are: the config and the result give the same errors, or the run fails (their Referrers differ
/// in the parts).
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

/// The rules whose config and result give the same errors (not always the same Referrer: see
/// `RuleResult::judge()`).
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
                "does not compile, the failure {} of accept_families() ({})",
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
            "  allow_stages [{}] copies after the result that do not compile: {} of {} ({} listed by accept_families(): {}; {} not listed and failing the config's copy too (SHARED); {} new)",
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
    /// The copies after the result that do not compile for a failure `accept_families()` lists,
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
                // The Referrer may differ: the config is asked about each path of a value or of a
                // default, so it writes them as they are and records no part for them, where the
                // result found the conflict and kept the part.
                if stage.before != stage.after || !stage.same_errors {
                    problems.push(Failure {
                        case: stage.case,
                        item: Some(stage.item),
                        class: "STAGES DIFFER: the config and the result give other errors, for a rule that is a place",
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
    env_flag("TYPE_LEAK_PROP_ACCEPT")
}

/// A short reason of a rejection, for the statistics: the family of its first error (the message of
/// an error that no family explains, which fails the run anyway).
fn reason_head(report: &Report) -> String {
    match report.rejections.first() {
        Some(error) => family(error).map_or_else(
            || error.message.chars().take(60).collect(),
            |family| family.name.to_string(),
        ),
        None => report.detail.chars().take(60).collect(),
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
///
/// A tag that ends with `=` stands for the tags that carry a text after it (`gen::TAG_BOUND_AT`): the
/// item needs one of them, and with `agree`, the text must agree with the error.
#[derive(Debug, PartialEq)]
struct Family {
    name: &'static str,
    tags: Vec<String>,
    agree: Option<Agree>,
}

/// How the text of a tag that carries one must agree with the error that it is to explain.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Agree {
    /// `bound_cycle=<bounded>|<bound>`: the error names the bound that makes the cycle in its note
    /// (`the bound `T : A + Fn(R0) -> u8` of its generics`): it is a bound of the same generic and
    /// it has the bound that the item wrote.
    Bound,
    /// `method_lifetime=<type>`: the cause of the error is that type, or a part of it, or has it.
    Type,
}

fn family_of(name: &'static str, tags: &[&str]) -> Option<Family> {
    Some(Family {
        name,
        tags: tags.iter().map(|tag| tag.to_string()).collect(),
        agree: None,
    })
}

fn family_agreeing(name: &'static str, tags: &[&str], agree: Agree) -> Option<Family> {
    Some(Family {
        agree: Some(agree),
        ..family_of(name, tags)?
    })
}

/// The bound that the note of an `UnprovableBound` names (`the bound `T : A + B` of ...`), as the
/// Leaker's tokens print it.
fn bound_of_note(note: &str) -> Option<&str> {
    let at = note.find("the bound `")? + "the bound `".len();
    let rest = &note[at..];
    rest.find('`').map(|end| &rest[..end])
}

impl Agree {
    /// Whether `payload` (the text of a tag, after the `=`) agrees with `error`.
    fn holds(self, error: &Rejection, payload: &str) -> bool {
        match self {
            Agree::Bound => {
                let Some((bounded, bound)) = payload.split_once('|') else {
                    return false;
                };
                let Some(note) = bound_of_note(&error.note) else {
                    return false;
                };
                let note = gen::norm(note);
                note.starts_with(&format!("{}:", bounded)) && note.contains(bound)
            }
            Agree::Type => {
                let cause = gen::norm(&error.tokens);
                !cause.is_empty() && (payload.contains(&cause) || cause.contains(payload))
            }
        }
    }
}

impl Family {
    /// Whether the tags of an item give the family, for this error.
    fn given_by(&self, error: &Rejection, tags: &std::collections::BTreeSet<String>) -> bool {
        self.tags.iter().any(|want| match want.strip_suffix('=') {
            None => tags.contains(want),
            Some(_) => tags
                .iter()
                .filter_map(|tag| tag.strip_prefix(want.as_str()))
                .any(|payload| self.agree.map_or(true, |agree| agree.holds(error, payload))),
        })
    }
}

/// The reason of a rejection among the ones that the generator knows (the allow-list), or `None`.
///
/// A rejection counts as a success of the property, so one must not hide a defect: each error of
/// it is one of these, and the generator has put its shape on the item (`family().tags`). A new
/// reason fails the run with the item and the message, to be judged: a limitation to list, or a
/// defect.
///
/// The `Reason` of the error decides. The tokens and the note tell the shapes of a reason apart
/// (`MaybeConst`, `RelativePath`), and the message is not read. The reasons that the generator makes
/// no item for (`Unreadable`, `SelfWithoutMeaning`, `SelfInDefault`, `UnnameableSelf`) are none of
/// them.
fn family(error: &Rejection) -> Option<Family> {
    match error.reason.as_str() {
        "SelfInTraitPredicate" => family_of(
            "a leaked type that names `Self` in a bound or where predicate of its trait",
            &[gen::TAG_BOUND_CYCLE, gen::TAG_SELF_IN_BOUNDS],
        ),
        // The bound of a generic that makes the cycle is one the item wrote: `Gen::note_bounds()`.
        "UnprovableBound" => family_agreeing(
            "a leaked type in a bound or where predicate that rustc cannot normalize",
            &[gen::TAG_BOUND_AT],
            Agree::Bound,
        ),
        "SelfInImplSelfType" => family_of(
            "a type in the self type of an impl whose context depends on `Self` through a bound",
            &[gen::TAG_SELF_IN_BOUNDS, gen::TAG_BOUND_CYCLE],
        ),
        // `NestedSelfInBounds` is the same fault in a nested struct, enum or union, rejected
        // without `nested_self_entries`.
        "DeniedSelfInBounds" | "NestedSelfInBounds" => family_of(
            "a type that uses a generic whose bound mentions a `Self` that cannot be leaked",
            &[
                gen::TAG_BOUND_CYCLE,
                gen::TAG_SELF_IN_BOUNDS,
                gen::TAG_SELF_DENIED,
            ],
        ),
        "SelfProjection" => family_of("`Self::A` where `Self` is a type", &[gen::TAG_SELF_ASSOC]),
        "UnconstrainedLifetime" => family_agreeing(
            "a lifetime of a method that only a leaked type keeps (E0195)",
            &[gen::TAG_LIFETIME_AT],
            Agree::Type,
        ),
        "MaybeConst" => maybe_const_family(error),
        "RelativePath" => relative_path_family(error),
        _ => None,
    }
}

/// A bare name (`u32`, `T`, `R0`) as an argument of a type that cannot be leaked as a whole: it may
/// be a const, so it is not leaked alone. The note says why the holder cannot be leaked, a reason
/// that is an error of the item too.
fn maybe_const_family(error: &Rejection) -> Option<Family> {
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
    family_of(
        "a bare name that may be a const, in a type that cannot be leaked as a whole",
        tags,
    )
}

/// A relative path (`RelativePath`): which shapes of the item explain it is read from its tokens
/// and its note.
fn relative_path_family(error: &Rejection) -> Option<Family> {
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
    // `check_values` (a const default either way) and copies as written: the note says which
    // value (`values`).
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
    Some(Family {
        name,
        tags,
        agree: None,
    })
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
                if !family.given_by(error, tags) {
                    problems.push(format!(
                        "NO SHAPE THAT GIVES IT: {} [tokens: {}] [note: {}]: {}, and the item has none of the tags {:?}{} (its tags are {:?})",
                        error.message,
                        error.tokens,
                        error.note,
                        family.name,
                        family.tags,
                        match family.agree {
                            Some(Agree::Bound) => " that agrees with the bound of the note",
                            Some(Agree::Type) => " that agrees with the cause",
                            None => "",
                        },
                        tags
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

/// A failure of the copy of an item accepted by `allow()` that `accept_families()` lists: which
/// one, what it is, and the shapes an item needs to have it (`tags`: the item needs one tag of each
/// list).
#[derive(Debug, PartialEq)]
struct AcceptFamily {
    /// The number of the failure in `accept_families()`: `(i)`, `(ii)` or `(iii)`.
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
/// This is the list of the failures: anything else is a defect, or a new limitation to add here,
/// with its shape.
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
/// codes of the diagnostics of rustc for the item. A diagnostic that is not a failure
/// `accept_families()` lists ("NOT A KNOWN FAILURE"), or that the item has no shape for ("NO SHAPE
/// THAT GIVES IT"), is a defect or a limitation to list.
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
    /// A diagnostic that no failure that `accept_families()` lists explains.
    unknown: &'static str,
    /// A listed failure, on an item that has not the shape.
    no_shape: &'static str,
    /// The diagnostic names no item for which an error was allowed by its cause.
    none: &'static str,
}

const ACCEPT_CLASSES: CopyClasses = CopyClasses {
    unknown: "ACCEPT: NOT A KNOWN FAILURE: the copy of an item accepted by allow() does not compile for a reason that accept_families() does not list",
    no_shape: "ACCEPT: NO SHAPE THAT GIVES IT: a failure of an accepted copy that accept_families() lists, on an item that has not the shape",
    none: "the diagnostic names no item that was accepted by allow()",
};

const STAGE_CLASSES: CopyClasses = CopyClasses {
    unknown: "STAGES: NEW FAILURE: the copy made after the analysis without the rule does not compile, the copy of the config's analysis does, and no failure that accept_families() lists explains it",
    no_shape: "STAGES: NO SHAPE THAT GIVES IT: a failure of a stage copy that accept_families() lists, on an item that has not the shape, and the copy of the config's analysis compiles",
    none: "the diagnostic names no item for which the cause of an error that the rule leaves was allowed, and the copy of the config's analysis compiles",
};

/// A clean copy (the rule left no error, the crate has none of the Leaker's names but the allowed
/// ones) that does not compile while the clean copy of the config's analysis does: the promise of the
/// rule after the analysis does not hold, a part kept as written names more than the rule allows.
const CLEAN_NEW: &str = "STAGES CLEAN: NEW FAILURE: the copy made after the analysis without the rule does not compile in the clean crate although the rule left no error, and the clean copy of the config's analysis compiles or was not made";

/// Explains the diagnostics of the copies that a macro made of items for which causes were allowed
/// (`failures` are the diagnostics of the Referrer's crate of those, `reports` what the macros
/// reported of the items): each is one of the failures that `accept_families()` lists on an item
/// that has the shape that gives it (`unexplained_accept()`), or a defect that fails the
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
                        // The crate of the stage copy (`StageKey`), if it is one.
                        stage: group.first().and_then(|failure| failure.stage),
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
/// types, which `Rules::prelude()` and `Rules::primitive()` take for the names of the prelude: the
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
            if case.decoy.is_some() {
                names.extend([decoy_name(i), decoy_referrer_name(i)]);
            }
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

/// What the build left of the case `i` that its diagnostics do not explain: the Leaker's
/// executable and the Referrer's library (and, with `accept`, the library of the second Referrer's
/// crate of `TYPE_LEAK_PROP_ACCEPT`, and with `stages` that of `allow_stages`). A crate with a
/// diagnostic is explained by it, and does not excuse the others. A missing one is reported, not
/// counted as a rejection.
fn missing_artifacts(
    build: &Build,
    i: usize,
    leaker_ok: bool,
    accept: bool,
    stages: &[gen::StageCrate],
) -> Vec<String> {
    let mut missing = Vec::new();
    let explained = |name: &str| build.diags.iter().any(|d| d.package == name);
    if !build.executables.contains_key(&leaker_name(i)) {
        missing.push(format!(
            "no executable was built for `{}`, so the macros reported nothing",
            leaker_name(i)
        ));
    }
    let mut libraries = vec![referrer_name(i)];
    if accept {
        libraries.push(accept_name(i));
    }
    libraries.extend(stages.iter().map(|stage| stage_name(i, stage)));
    for name in libraries {
        if leaker_ok && !explained(&name) && !build.artifacts.contains(&name) {
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
    /// The same by the construct an item has (`gen::CONSTRUCTS`): a trait with a generic associated
    /// type, a const parameter, an `impl Trait` argument, a default, a binder, ... A module is not
    /// counted (one of its dozen items has it).
    construct_accepted: BTreeMap<&'static str, usize>,
    construct_rejected: BTreeMap<&'static str, usize>,
    /// The reason of each rejection, by kind.
    reasons: BTreeMap<(&'static str, String), usize>,
    analyzed: usize,
    entries: usize,
    positions: usize,
    checked: usize,
    /// The impl headers compared, and those left out (`Report::header_cycles`).
    headers: usize,
    header_cycles: usize,
    /// The Repeater impls of optional entries that were not written, for the items whose Referrer
    /// trusts them.
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
    /// The items accepted by `allow()` whose copy does not compile for a failure that
    /// `accept_families()` lists, by its class (`(i)`, `(ii)`, `(iii)`).
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
    /// `cases`: the cases that ran, with their numbers.
    fn collect(cases: &[(usize, &Case)], reports: &BTreeMap<(usize, usize), Report>) -> Stats {
        let mut stats = Stats::default();
        for &(case_index, case) in cases {
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
                    stats.headers += report.headers;
                    stats.header_cycles += report.header_cycles;
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
                    if item.kind != "mod" {
                        for name in CONSTRUCTS {
                            if item
                                .tags
                                .contains(&format!("{}{}", gen::TAG_CONSTRUCT, name))
                            {
                                let counts = if report.accepted {
                                    &mut stats.construct_accepted
                                } else {
                                    &mut stats.construct_rejected
                                };
                                *counts.entry(name).or_default() += 1;
                            }
                        }
                    }
                    if let Some((_, api)) = SCOPES.iter().find(|(scope, _)| *scope == item.scope) {
                        let counts = if report.accepted {
                            &mut stats.scope_accepted
                        } else {
                            *stats.reasons.entry((api, reason_head(report))).or_default() += 1;
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
                            .entry((kind, reason_head(report)))
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

    /// The items accepted of a kind, analyzed through a scope (the name of its API), or with a
    /// construct.
    fn accepted_of(&self, name: &str) -> usize {
        self.accepted.get(name).copied().unwrap_or(0)
            + self.scope_accepted.get(name).copied().unwrap_or(0)
            + self.construct_accepted.get(name).copied().unwrap_or(0)
    }

    fn rejected_of(&self, name: &str) -> usize {
        self.rejected.get(name).copied().unwrap_or(0)
            + self.scope_rejected.get(name).copied().unwrap_or(0)
            + self.construct_rejected.get(name).copied().unwrap_or(0)
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

/// The least share of the analyzed items of each kind, scope and construct that the Leaker must
/// accept, in percent.
///
/// A test that counts a rejection as a success passes when the Leaker rejects everything, or when
/// the generator makes shapes that cannot be leaked (before the floors, 93% of the modules and
/// 60% of the traits were rejected, and a macro that failed every `intern_*` still printed `ok`).
/// These floors fail the run then.
///
/// Measured over the seeds 1 to 16 at 60 cases, the accepted
/// share was, lowest over the seeds / pooled (and the fewest items of a seed):
///
/// - kinds: struct 94.1/96.0 (296), enum 93.5/95.5 (152), union 90.4/95.5 (75), trait 72.7/77.1
///   (351), inherent impl 81.4/86.5 (135), trait impl 63.0/71.2 (63), module 68.9/76.8 (84);
/// - scopes: `enter_mod` 89.9/93.4 (160), `enter_generics` 90.5/95.2 (47), `enter_trait` 67.9/80.3
///   (28), `enter_impl` 100/100 (31);
/// - constructs (`gen::CONSTRUCTS`, the items that have it, not the modules): gat 50.8/57.8 (52),
///   const_param 76.5/79.7 (463), impl_trait_arg 58.9/66.0 (165), param_default 82.2/85.7 (170),
///   binder 73.4/78.7 (254), dyn_object 78.7/82.4 (468), fn_pointer 78.5/80.9 (375), assoc_const
///   69.6/73.5 (225), lifetime_param 80.7/84.4 (619), unsized_param 76.7/80.7 (316), where_clause
///   78.1/80.9 (694).
///
/// A floor is about 10 points below the lowest (more for the kinds with the fewest items, whose rate
/// varies most: the trait impl, the module, `enter_trait`), so that another seed or a short run does
/// not trip it, but a regression of a kind or of a construct does: a Leaker that rejects every
/// trait with a generic associated type, every const parameter, every `impl Trait` argument or every
/// default is at 0% for it, and the old generator (module 10%, trait 39%, inherent and trait impl
/// 41%) was far below. A construct is checked when a run has three times the `MIN_SAMPLES` of it.
/// `TYPE_LEAK_PROP_MIN_ACCEPT` overrides them.
const FLOORS: &[(&str, f64)] = &[
    ("struct", 84.0),
    ("enum", 83.0),
    ("union", 80.0),
    ("trait", 62.0),
    ("impl-inherent", 71.0),
    ("impl-trait", 50.0),
    ("mod", 58.0),
    // The items analyzed through a scope (`gen::SCOPES`), by the name of its API.
    ("enter_mod", 80.0),
    ("enter_generics", 80.0),
    ("enter_trait", 50.0),
    ("enter_impl", 90.0),
    // The items that have a construct (`gen::CONSTRUCTS`).
    ("gat", 40.0),
    ("const_param", 66.0),
    ("impl_trait_arg", 48.0),
    ("param_default", 72.0),
    ("binder", 63.0),
    ("dyn_object", 68.0),
    ("fn_pointer", 68.0),
    ("assoc_const", 59.0),
    ("lifetime_param", 70.0),
    ("unsized_param", 66.0),
    ("where_clause", 68.0),
];

/// The fewest analyzed items of a kind for its floor to count (a short run has too few of the
/// rare kinds to say anything about their rate): `TYPE_LEAK_PROP_MIN_SAMPLES`. A construct needs
/// `CONSTRUCT_SAMPLES` times as many: the items that have one are a share of the kinds, and its
/// rate varies as much.
const MIN_SAMPLES: usize = 12;
const CONSTRUCT_SAMPLES: usize = 3;

/// The least share of the type positions of the accepted items whose rewritten type was compared
/// with the original (the rest has an elided lifetime or `impl Trait`), in percent:
/// `TYPE_LEAK_PROP_MIN_CHECKED`. Without it the oracle could silently check nothing. Measured:
/// 93.5% to 93.8% over the seeds 1, 2, 3 and 7 at 60 cases.
const MIN_CHECKED: f64 = 85.0;

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

    /// The fewest items of `name`, a kind, a scope or a construct, for its floor to count.
    fn samples_of(&self, name: &str) -> usize {
        if CONSTRUCTS.contains(&name) {
            self.min_samples * CONSTRUCT_SAMPLES
        } else {
            self.min_samples
        }
    }

    /// Why the statistics are too poor for the run to count as a test, one line each.
    fn problems(&self, stats: &Stats) -> Vec<String> {
        let mut problems = Vec::new();
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
        // A floor of 0 is off (`TYPE_LEAK_PROP_MIN_ACCEPT=0` turns them all off, not the check of
        // the witnesses below).
        for kind in KINDS
            .iter()
            .chain(SCOPES.iter().map(|(_, api)| api))
            .chain(CONSTRUCTS.iter())
        {
            let (a, r) = (stats.accepted_of(kind), stats.rejected_of(kind));
            let floor = self.by_kind.get(kind).copied().unwrap_or(0.0);
            if a + r < self.samples_of(kind) || floor == 0.0 {
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

/// The modes of the Referrers (`gen::Trust`) over the analyzed items: how many items have each, the
/// optional entries they have by the crates they name, and how many the mode trusts (written as they
/// are in the copy, with no Repeater impl) or replaces. `decoy`: the items copied in the Referrer's
/// crate where the Leaker's crate name is another crate (`gen::Decoy`).
fn trust_summary(cases: &[(usize, &Case)], reports: &BTreeMap<(usize, usize), Report>) -> String {
    let mut s = String::new();
    writeln!(
        s,
        "  optional entries by the mode of the Referrer (trust): items (copied in the decoy crate), entries that name std only / the Leaker's crate only / both / another, and how many are trusted"
    )
    .unwrap();
    for trust in gen::Trust::ALL {
        let (mut items, mut optional, mut trusted, mut decoyed) = (0, [0; 4], 0, 0);
        for &(i, case) in cases {
            for item in case.items.iter().filter(|item| item.trust == trust) {
                if let Some(report) = reports.get(&(i, item.id)) {
                    if report.accepted || report.allowed {
                        items += 1;
                        for (total, n) in optional.iter_mut().zip(report.optional) {
                            *total += n;
                        }
                        trusted += report.trusted;
                        // Only the cases with a decoy crate invoke the item there.
                        decoyed += usize::from(
                            report.decoyed
                                && case.decoy.is_some()
                                && trust != gen::Trust::All
                                && trust != gen::Trust::Leaker,
                        );
                    }
                }
            }
        }
        writeln!(
            s,
            "  trust {:<7} {:>5} items ({:>4}); optional entries {:>5} / {:>5} / {:>5} / {:>5}, trusted {:>5}, replaced {:>5}",
            trust.name(),
            items,
            decoyed,
            optional[0],
            optional[1],
            optional[2],
            optional[3],
            trusted,
            optional.iter().sum::<usize>() - trusted
        )
        .unwrap();
    }
    s
}

fn summarize(
    stats: &Stats,
    floors: &Floors,
    run: &Run,
    features: Features,
    flags: Flags,
) -> String {
    let mut s = format!(
        "type-leak property test: seed {}, {} cases{}, {} items per module at most, features {:?}, flags {:?}\n",
        run.seed,
        run.cases,
        run.only
            .map_or(String::new(), |case| format!(" (only case {})", case)),
        run.items,
        features,
        flags
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
            Some(f) if a + r >= floors.samples_of(kind) => format!("{:.0}%", f),
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
        "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6}",
        "construct", "analyzed", "accepted", "rejected", "accepted%", "floor"
    )
    .unwrap();
    for name in CONSTRUCTS {
        let (a, r) = (stats.accepted_of(name), stats.rejected_of(name));
        let rate = if a + r > 0 {
            format!("{:.0}%", 100.0 * a as f64 / (a + r) as f64)
        } else {
            "-".into()
        };
        let floor = match floors.by_kind.get(name) {
            Some(f) if *f == 0.0 => "off".into(),
            Some(f) if a + r >= floors.samples_of(name) => format!("{:.0}%", f),
            Some(_) => "n/a".into(),
            None => "-".into(),
        };
        writeln!(
            s,
            "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6}",
            name,
            a + r,
            a,
            r,
            rate,
            floor
        )
        .unwrap();
    }
    writeln!(
        s,
        "  analyzed (macros applied): {}; entries: {}; type positions: {}, witnesses checked: {} ({} skipped: elided lifetimes or `impl Trait`); impl headers checked: {} ({} left out: rustc cannot normalize an entry with arguments there)",
        stats.analyzed,
        stats.entries,
        stats.positions,
        stats.checked,
        stats.positions - stats.checked,
        stats.headers,
        stats.header_cycles
    )
    .unwrap();
    writeln!(
        s,
        "  bare traits of the prelude: {} items compared with their twin, {} errors of rejections; optional entries without an impl (trusted): {}; items without allow_primitive: {}",
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
            "  accepted copies that do not compile for a failure accept_families() lists: {} of {} ({})",
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

/// The crates a failure is about: the Leaker's, and the Referrer's crate that has the diagnostic.
fn failure_crates(failure: &Failure, ws: &Path) -> Vec<String> {
    let i = failure.case;
    let referrer = match failure.stage {
        Some(key) => key.dir(i),
        None if failure.class == DECOY_COPY => decoy_referrer_dir(i),
        None if failure.class == ACCEPT_CLASSES.unknown
            || failure.class == ACCEPT_CLASSES.no_shape =>
        {
            accept_dir(i)
        }
        None => referrer_dir(i),
    };
    [leaker_dir(i), referrer]
        .iter()
        .map(|dir| ws.join(dir).display().to_string())
        .collect()
}

const LEAKER_STAGES: &str = "STAGES: the Leaker's crate does not compile with the output of the copy made after the analysis without a rule";
const LEAKER_ACCEPT: &str =
    "ACCEPT: the Leaker's crate does not compile with the output of an item accepted by allow()";
const LEAKER_MACRO: &str = "the Leaker's crate does not compile with the output of the macro";

/// What a Leaker's crate that compiles without `type-leak` but not with the macros fails on.
/// `ordinary_compiles`: it compiles with the macros of an ordinary run (no accepted item, no stage
/// copy), so the extra output of `accept` or of `stages` is what does not compile; with both,
/// `accept_alone_compiles` (the accepted items and no stage copy) tells which.
fn leaker_failure_class(
    ordinary_compiles: bool,
    accept: bool,
    stages: bool,
    accept_alone_compiles: bool,
) -> &'static str {
    if !ordinary_compiles {
        LEAKER_MACRO
    } else if stages && (!accept || accept_alone_compiles) {
        LEAKER_STAGES
    } else {
        LEAKER_ACCEPT
    }
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
    assert!(
        items_per_module >= MIN_ITEMS,
        "TYPE_LEAK_PROP_ITEMS={} is too few: it is the most items in a module, and a module has at least {}",
        items_per_module,
        MIN_ITEMS
    );
    let only: Option<usize> = match std::env::var("TYPE_LEAK_PROP_CASE") {
        Ok(value) if !value.trim().is_empty() => {
            Some(value.trim().parse().unwrap_or_else(|_| {
                panic!("TYPE_LEAK_PROP_CASE is not a case number: {:?}", value)
            }))
        }
        _ => None,
    };
    let features_env = std::env::var("TYPE_LEAK_PROP_FEATURES").unwrap_or_default();
    let features = Features::parse(&features_env);
    let flags = Flags::from_env();
    let floors = Floors::from_env();
    let accept = accept_mode();
    let stages_on = flags.allow_stages;
    let run = Run {
        seed,
        cases,
        items: items_per_module,
        features: features_env,
        accept,
        stages: stages_on,
        only,
    };
    let numbers = selected_cases(cases, only).unwrap_or_else(|error| panic!("{}", error));
    let generated: Vec<Case> = (0..cases)
        .map(|i| generate(case_seed(seed, i), i, items_per_module, features))
        .collect();
    // The cases that run, with their numbers: the crates keep them.
    let selected: Vec<(usize, &Case)> = numbers.iter().map(|&i| (i, &generated[i])).collect();

    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("prop");
    let target = base.join("target");
    let ws = base.join(run.workspace_name());
    write_workspace(
        &ws,
        &selected,
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
    for &(i, case) in &selected {
        let leaker = leaker_name(i);
        let leaker_errors: Vec<&Diag> =
            build.diags.iter().filter(|d| d.package == leaker).collect();
        if !leaker_errors.is_empty() {
            leaker_failed.push(i);
            continue;
        }
        // The Referrer's crate, and, in the accept mode, the second one (it copies only the items
        // that the Leaker accepted by `allow()`, and so every error in it is about one of them),
        // and with `allow_stages` the one of the copies made after the analysis without a rule.
        let mut crates: Vec<BuiltCrate> = vec![
            (
                referrer_name(i),
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
        if let Some(decoy) = &case.decoy {
            crates.push((
                decoy_referrer_name(i),
                decoy_referrer_dir(i),
                &decoy.lines,
                DECOY_COPY,
                None,
            ));
        }
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
        // A crate that reported no error must have been built, whatever the others did.
        let mut missing = missing_artifacts(
            &build,
            i,
            true,
            accept,
            if stages_on { &case.stages } else { &[] },
        );
        if case.decoy.is_some() {
            for name in [decoy_name(i), decoy_referrer_name(i)] {
                if !build.diags.iter().any(|d| d.package == name)
                    && !build.artifacts.contains(&name)
                {
                    missing.push(format!("`{}` was not built, and reported no error", name));
                }
            }
        }
        if !missing.is_empty() {
            unbuilt.push(i);
            failures.push(Failure {
                stage: None,
                code: String::new(),
                case: i,
                item: None,
                class: "WATCHDOG: a crate of the case was not built, and no error says why",
                text: format!(
                    "{}\n\ncargo's stderr:\n{}",
                    missing.join("\n"),
                    excerpt(&build.stderr, 40)
                ),
            });
        }
    }

    // A Leaker's crate that does not compile: is it the generator or the macro?
    if !leaker_failed.is_empty() {
        let id_ws = base.join(format!("{}-identity", run.workspace_name()));
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
            let ordinary_ws = base.join(format!("{}-ordinary", run.workspace_name()));
            write_workspace(&ordinary_ws, &list, Mode::Real, false);
            Some(cargo_build(&ordinary_ws, &target))
        } else {
            None
        };
        // With both, does it compile with the accepted items and no stage copy? If it does, the
        // stage copies are what does not.
        let accept_only = if accept && stages_on {
            let only_ws = base.join(format!("{}-accept-only", run.workspace_name()));
            write_workspace(&only_ws, &list, Mode::Accept, false);
            Some(cargo_build(&only_ws, &target))
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
            let compiles = |built: &Option<Build>| {
                built
                    .as_ref()
                    .map_or(false, |b| !b.diags.iter().any(|d| d.package == name))
            };
            let (class, diags) = if own.is_empty() {
                (
                    leaker_failure_class(
                        compiles(&ordinary),
                        accept,
                        stages_on,
                        compiles(&accept_only),
                    ),
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
    for &(i, case) in &selected {
        if leaker_failed.contains(&i) || unbuilt.contains(&i) {
            continue;
        }
        let Some(exe) = build.executables.get(&leaker_name(i)) else {
            // (`missing_artifacts()` has said so already.)
            failures.push(Failure {
                stage: None,
                code: String::new(),
                case: i,
                item: None,
                class: "WATCHDOG: no executable was built for the Leaker's crate, so the macros reported nothing",
                text: format!("cargo's stderr:\n{}", excerpt(&build.stderr, 40)),
            });
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
                    Some((rule, runs.get_mut(rule)?, side, id, report))
                });
                match parsed {
                    Some((rule, run, side, id, report)) => {
                        if side == 0 {
                            if let Some(error) = &report.rewrite_error {
                                failures.push(Failure {
                                    stage: None,
                                    code: String::new(),
                                    case: i,
                                    item: Some(id),
                                    class: "STAGES: the macro cannot make the copy made after the analysis without the rule",
                                    text: format!("rule {}: {}", rule, error),
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
                                        "rule {}: the errors that the rule leaves:\n{}\nthe errors that remain:\n{}",
                                        rule,
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
        for &(i, case) in &selected {
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

    // The diagnostics of the copy of an item accepted by `allow()` are failures that
    // `accept_families()` lists, or defects: each must be one of them, and the item must have the
    // shape that gives it. The listed ones are counted and are no failure.
    // The same for the copies of `allow_stages`, which allowed the causes of the errors that
    // `Rules::prelude()` leaves.
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
    if env_flag("TYPE_LEAK_PROP_REJECTIONS") {
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
    let mut stats = Stats::collect(&selected, &reports);
    stats.documented = documented;
    let mut summary = summarize(&stats, &floors, &run, features, flags);
    summary.push_str(&trust_summary(&selected, &reports));
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
            "type-leak property failed: {} failure(s), {} acceptance problem(s).\nRepeat: {}\nSources: {}\n\n{}\n",
            failures.len(),
            floor_problems.len(),
            run.command(None),
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
            writeln!(message, "repeat: {}", run.command(Some(failure.case))).unwrap();
            writeln!(
                message,
                "crates: {}",
                failure_crates(failure, &ws).join(", ")
            )
            .unwrap();
            writeln!(message, "settings: {:?}", case.config).unwrap();
            if let Some(item) = case.items.iter().find(|i| Some(i.id) == failure.item) {
                writeln!(
                    message,
                    "item {} ({}, in {:?}, allow_self {}, trust {}):\n{}",
                    item.id,
                    item.kind,
                    item.module,
                    item.allow_self,
                    item.trust.name(),
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
                failure.class,
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
        // And a construct's.
        let floors = Floors::parse("gat=5", MIN_SAMPLES, MIN_CHECKED).unwrap();
        assert_eq!(floors.by_kind["gat"], 5.0);
        assert!(Floors::parse("fn=5", 1, 1.0).is_err());
        assert!(Floors::parse("150", 1, 1.0).is_err());
        assert!(Floors::parse("a", 1, 1.0).is_err());
    }

    #[test]
    fn a_leaker_that_rejects_every_trait_with_a_generic_associated_type_fails_the_run() {
        let mut stats = stats_of(&[("struct", 90, 10), ("trait", 90, 10)]);
        stats.construct_rejected.insert("gat", 60);
        let problems = defaults().problems(&stats);
        assert_eq!(problems.len(), 1, "{:?}", problems);
        assert!(problems[0].starts_with("gat:"), "{:?}", problems);
        // A run with too few of them says nothing about their rate.
        stats
            .construct_rejected
            .insert("gat", MIN_SAMPLES * CONSTRUCT_SAMPLES - 1);
        assert!(defaults().problems(&stats).is_empty());
        // The floor is a minimum: at it, it holds.
        stats.construct_accepted.insert("gat", 40);
        stats.construct_rejected.insert("gat", 60);
        assert!(defaults().problems(&stats).is_empty());
        stats.construct_rejected.insert("gat", 61);
        assert_eq!(defaults().problems(&stats).len(), 1);
        // Every construct has a floor of its own.
        for name in CONSTRUCTS {
            let floors = defaults();
            assert!(
                floors.by_kind.get(name).map_or(false, |f| *f > 0.0),
                "{}",
                name
            );
            let mut stats = stats_of(&[("struct", 90, 10)]);
            stats.construct_rejected.insert(name, 100);
            let problems = floors.problems(&stats);
            assert!(problems.iter().any(|p| p.starts_with(name)), "{}", name);
        }
    }

    #[test]
    fn the_items_with_a_construct_are_counted_apart() {
        let cases: Vec<Case> = (0..4)
            .map(|i| generate_with(case_seed(1, i), i, 8, Features::parse(""), Flags::default()))
            .collect();
        let selected: Vec<(usize, &Case)> = cases.iter().enumerate().collect();
        let mut reports = BTreeMap::new();
        for (i, case) in &selected {
            for item in &case.items {
                reports.insert(
                    (*i, item.id),
                    Report {
                        kind: item.kind.to_string(),
                        accepted: item.id % 3 != 0,
                        ..Default::default()
                    },
                );
            }
        }
        let stats = Stats::collect(&selected, &reports);
        for name in CONSTRUCTS {
            let tag = format!("{}{}", gen::TAG_CONSTRUCT, name);
            let items = selected
                .iter()
                .flat_map(|(_, case)| case.items.iter())
                .filter(|item| item.kind != "mod" && item.tags.contains(&tag));
            let (accepted, rejected) = items.fold((0, 0), |(a, r), item| {
                if item.id % 3 != 0 {
                    (a + 1, r)
                } else {
                    (a, r + 1)
                }
            });
            assert_eq!(stats.accepted_of(name), accepted, "{}", name);
            assert_eq!(stats.rejected_of(name), rejected, "{}", name);
        }
        // They are there in 4 cases: a const parameter, a binder, a where clause, ...
        for name in ["const_param", "binder", "where_clause", "dyn_object"] {
            assert!(
                stats.accepted_of(name) + stats.rejected_of(name) > 0,
                "{}",
                name
            );
        }
        // And the summary has a row for each.
        let run = Run {
            seed: 1,
            cases: 4,
            items: 8,
            features: String::new(),
            accept: false,
            stages: false,
            only: None,
        };
        let summary = summarize(
            &stats,
            &defaults(),
            &run,
            Features::default(),
            Flags::default(),
        );
        for name in CONSTRUCTS {
            assert!(summary.contains(&format!("  {} ", name)), "{}", name);
        }
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

    /// A relative path (`Reason::RelativePath`), as the Leaker reports every one.
    fn rejection(message: &str, tokens: &str) -> Rejection {
        rejection_of("RelativePath", message, tokens)
    }

    fn rejection_of(reason: &str, message: &str, tokens: &str) -> Rejection {
        Rejection {
            message: message.to_string(),
            tokens: tokens.to_string(),
            reason: reason.to_string(),
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
                "use absolute path: 'for<'h3> Fn(&'h3 u8) -> &'h3 u8'",
                "for < 'h3 > Fn(& 'h3 u8) -> & 'h3 u8"
            ),
            Some(vec!["bare:Fn".to_string()])
        );
        // A path can be several reasons at once: the shapes of any of them explain it.
        let tags = tags_of("use absolute path: 'Into<Self::A0>'", "Into < Self :: A0 >").unwrap();
        assert!(tags.contains(&"bare:Into".to_string()), "{:?}", tags);
        assert!(
            tags.contains(&gen::TAG_SELF_ASSOC.to_string()),
            "{:?}",
            tags
        );
        // The binder of a bound is not a type that names one.
        let tags = tags_of(
            "use absolute path: 'for<'h1> Fn(&'h1 u8) -> &'h1 u8'",
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
        // The other shapes of a relative path.
        assert_eq!(
            tags_of("use absolute path: 'Rl<'h5>'", "Rl < 'h5 >"),
            Some(vec![gen::TAG_BINDER_NO_ANCHOR.to_string()])
        );
        assert_eq!(
            tags_of("use absolute path: 'Tr1_1<'a>::A0'", "Tr1_1 < 'a > :: A0"),
            Some(vec![gen::TAG_SELF_ASSOC.to_string()])
        );
        assert_eq!(
            tags_of(
                "use absolute path: 'crate::base::BLEN'",
                "crate :: base :: BLEN"
            )
            .map(|tags| tags.contains(&gen::TAG_CONST_DEFAULT.to_string())),
            Some(true)
        );
        // The other reasons are decided by the `Reason`, whatever the message says.
        let by_reason = |reason: &str, message: &str, tokens: &str| {
            family(&rejection_of(reason, message, tokens)).map(|f| f.tags)
        };
        assert_eq!(
            by_reason(
                "SelfProjection",
                "write `<Self as ::k::Tr>::A0` instead of `Self::A0` in `Wrap<Self::A0>`: the copy cannot tell",
                "Wrap<Self::A0>"
            ),
            Some(vec![gen::TAG_SELF_ASSOC.to_string()])
        );
        assert_eq!(
            by_reason("SelfProjection", "any words", "Wrap<Self::A0>"),
            Some(vec![gen::TAG_SELF_ASSOC.to_string()])
        );
        assert_eq!(
            by_reason(
                "UnprovableBound",
                "`Wrap<T>` is in a bound or where predicate, and a bound of its generics names a leaked type",
                "Wrap < T >"
            ),
            Some(vec![gen::TAG_BOUND_AT.to_string()])
        );
        assert_eq!(
            by_reason("UnprovableBound", "any words", "Wrap < T >"),
            Some(vec![gen::TAG_BOUND_AT.to_string()])
        );
        let same = |a: &str, b: &str| {
            let (a, b) = (
                family(&rejection_of(a, "", "X")),
                family(&rejection_of(b, "", "X")),
            );
            a.is_some() && a == b
        };
        // Both reasons of a generic whose bound names a `Self` that cannot be leaked.
        assert!(same("DeniedSelfInBounds", "NestedSelfInBounds"));
        for reason in [
            "SelfInTraitPredicate",
            "SelfInImplSelfType",
            "DeniedSelfInBounds",
            "UnconstrainedLifetime",
            "MaybeConst",
        ] {
            assert!(
                family(&rejection_of(reason, "", "X")).is_some(),
                "{}",
                reason
            );
        }
        // A message that looks like a known reason does not make another reason one.
        assert_eq!(
            by_reason(
                "SelfWithoutMeaning",
                "`Wrap<T>` is in a bound or where predicate, and a bound of its generics names a leaked type",
                "Wrap < T >"
            ),
            None
        );
        // The reasons that nothing is generated for, and a reason nobody listed.
        for reason in [
            "SelfWithoutMeaning",
            "Unreadable",
            "SelfInDefault",
            "UnnameableSelf",
            "",
            "Strange",
        ] {
            assert_eq!(by_reason(reason, "any words", "Self"), None, "{}", reason);
        }
        assert_eq!(tags_of("use absolute path: 'Strange'", "Strange"), None);
    }

    /// The table of the summary names the family of the first error.
    #[test]
    fn the_reason_of_a_rejection_in_the_table_is_the_family_of_its_first_error() {
        let report = |errors: Vec<Rejection>| Report {
            detail: errors
                .iter()
                .map(|error| error.message.as_str())
                .collect::<Vec<_>>()
                .join(" ; "),
            rejections: errors,
            ..Default::default()
        };
        let bound = report(vec![
            rejection_of(
                "UnprovableBound",
                "`Wrap<T>` is in a bound or where predicate",
                "Wrap < T >",
            ),
            rejection("use absolute path: 'Clone'", "Clone"),
        ]);
        assert_eq!(
            reason_head(&bound),
            "a leaked type in a bound or where predicate that rustc cannot normalize"
        );
        let bare = report(vec![rejection("use absolute path: 'Clone'", "Clone")]);
        assert!(reason_head(&bare).starts_with("a bare trait"));
        // Nothing explains it: the message, cut.
        let unknown = report(vec![rejection_of("Strange", &"x".repeat(100), "X")]);
        assert_eq!(reason_head(&unknown), "x".repeat(60));
        // No error was read.
        let none = Report {
            detail: "rewrite-error=no host".into(),
            ..Default::default()
        };
        assert_eq!(reason_head(&none), "rewrite-error=no host");
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
        // A bare `?Sized` is explained by its own tag, which the generator never puts on: the Leaker
        // accepts it, so a rejection for it would be a defect.
        let sized = report(vec![rejection("use absolute path: '? Sized'", "? Sized")]);
        assert!(unexplained(&tags(&["bare:Sized"]), &sized).len() == 1);
        assert!(unexplained(&tags(&["bare:?Sized"]), &sized).is_empty());
        // A reason that is not in the list.
        let unknown = report(vec![rejection_of(
            "SelfWithoutMeaning",
            "`Self` has no meaning here",
            "Self",
        )]);
        let problems = unexplained(&tags(&["bound_cycle"]), &unknown);
        assert!(
            problems[0].starts_with("NOT A KNOWN REASON"),
            "{:?}",
            problems
        );
        // Every error counts: one that is explained does not explain the other.
        let two = report(vec![
            rejection("use absolute path: 'Default'", "Default"),
            rejection_of("SelfWithoutMeaning", "`Self` has no meaning here", "Self"),
        ]);
        assert_eq!(unexplained(&tags(&["bare:Default"]), &two).len(), 1);
        // A rejection with no error (the report was not read) is not explained either.
        assert_eq!(unexplained(&tags(&[]), &report(Vec::new())).len(), 1);
    }

    /// The settings of a Leaker's crate that allows nothing but the primitive types.
    fn strict_config() -> gen::Config {
        gen::Config {
            krate: "tlp_l_000".to_string(),
            prelude: false,
            allow_std: false,
            allow_crate: false,
            primitive: true,
        }
    }

    /// The tags that the generator puts on an item with this text.
    fn tags_of_text(config: gen::Config, text: &str) -> std::collections::BTreeSet<String> {
        let mut g = gen::Gen::with_config(config);
        g.note_text(text);
        g.tags().clone()
    }

    /// An `UnprovableBound` as the Leaker reports it: the type in the predicate, and in the note
    /// the bound of its generics that makes the cycle.
    fn unprovable(tokens: &str, bound: &str) -> Rejection {
        Rejection {
            message: format!(
                "`{}` is in a bound or where predicate, and a bound of its generics names a leaked type",
                tokens
            ),
            tokens: tokens.to_string(),
            note: format!(
                "the bound `{}` of its generics names a type that is leaked too: the copy writes both as paths",
                bound
            ),
            reason: "UnprovableBound".to_string(),
            kind: "Type".to_string(),
        }
    }

    fn rejected(errors: Vec<Rejection>) -> Report {
        Report {
            kind: "struct".into(),
            rejections: errors,
            ..Default::default()
        }
    }

    /// The share of the items that may have a cycle is no tag: a cycle is explained by a bound that
    /// the item wrote, and that the Leaker names (M10b, a Leaker that took the predicates that an
    /// impl proves for bounds of the environment, rejected 20 items that the old tag explained).
    #[test]
    fn a_cycle_is_explained_by_the_bound_that_the_item_wrote() {
        let item = "pub struct S<T: ?::core::marker::Sized> where T: ::core::ops::Fn(R0) -> u8, \
                    R2<T>: ::core::default::Default, Vec<T>: ::core::fmt::Debug { pub f0: R1<T> }";
        let item_tags = tags_of_text(strict_config(), item);
        assert!(item_tags.contains("bound_cycle"), "{:?}", item_tags);
        let bounds: Vec<&String> = item_tags
            .iter()
            .filter(|tag| tag.starts_with(gen::TAG_BOUND_AT))
            .collect();
        assert_eq!(bounds, vec!["bound_cycle=T|core::ops::Fn(R0)->u8"]);
        // The bound of the generic, named in the note as the Leaker prints it, with the others of the
        // generic joined to it.
        let cycle = unprovable("R1 < T >", "T : ? Sized + :: core :: ops :: Fn(R0) -> u8");
        assert!(unexplained(&item_tags, &rejected(vec![cycle])).is_empty());
        // A predicate that an impl proves is not an environment bound.
        for bound in [
            "R2 < T > : :: core :: default :: Default",
            "Vec < T > : Debug",
        ] {
            let wrong = unprovable("R1 < T >", bound);
            let problems = unexplained(&item_tags, &rejected(vec![wrong]));
            assert_eq!(problems.len(), 1, "{:?}", problems);
            assert!(
                problems[0].starts_with("NO SHAPE THAT GIVES IT"),
                "{:?}",
                problems
            );
        }
        // The bound of another generic, or one that the item did not write.
        for bound in [
            "U : :: core :: ops :: Fn(R0) -> u8",
            "T : :: core :: ops :: Fn(R3) -> u8",
        ] {
            let wrong = unprovable("R1 < T >", bound);
            assert_eq!(
                unexplained(&item_tags, &rejected(vec![wrong])).len(),
                1,
                "{}",
                bound
            );
        }
        // The plain tag is not enough, and a note without a bound explains nothing.
        let mut error = unprovable("R1 < T >", "T : Fn(R0) -> u8");
        assert_eq!(
            unexplained(&tags(&["bound_cycle"]), &rejected(vec![error.clone()])).len(),
            1
        );
        error.note = "no bound named".to_string();
        assert_eq!(unexplained(&item_tags, &rejected(vec![error])).len(), 1);
    }

    /// The item of seed 21 (case 11, item 3): the bound names the types of the module in a binder, in
    /// an item that no share of cyclic items had, and `Self` is what the Leaker names as the cause.
    #[test]
    fn the_cycle_of_a_bound_with_a_binder_is_tagged() {
        let item = "pub trait Tr<T: ?::core::marker::Sized> where T: for<'h3> Fn(String, Rl<'h3>) \
                    -> &'h3 Bl<'static> { fn m(&self); }";
        let tags = tags_of_text(strict_config(), item);
        let bound = "T : for<'h3> Fn(String, Rl<'h3>) -> & 'h3 Bl<'static>";
        let error = unprovable("Self", bound);
        assert!(unexplained(&tags, &rejected(vec![error])).is_empty());
        // Where the Leaker allows nothing, the names of the prelude are paths of the item too.
        let tags = tags_of_text(strict_config(), "pub struct S<T> where T: Fn(u8) -> u8 { }");
        assert!(!tags.contains("bound_cycle"), "{:?}", tags);
        let tags = tags_of_text(
            gen::Config {
                primitive: false,
                ..strict_config()
            },
            "pub struct S<T> where T: ::core::ops::Fn(u8) -> u8 { }",
        );
        assert!(tags.contains("bound_cycle"), "{:?}", tags);
    }

    /// `UnconstrainedLifetime` on `I1<&'x str>` (seed 19, case 17, item 17): the lifetime is in the
    /// arguments of the path, behind a reference.
    #[test]
    fn a_lifetime_that_a_path_holds_in_a_reference_is_tagged() {
        let error = |tokens: &str| Rejection {
            tokens: tokens.to_string(),
            reason: "UnconstrainedLifetime".to_string(),
            kind: "Type".to_string(),
            ..Default::default()
        };
        let tagged = tags(&[
            gen::TAG_METHOD_LIFETIME,
            &format!("{}{}", gen::TAG_LIFETIME_AT, gen::norm("I1<&'x str>")),
        ]);
        assert!(unexplained(&tagged, &rejected(vec![error("I1 < & 'x str >")])).is_empty());
        // The type that holds the lifetime is the one the item wrote.
        assert_eq!(
            unexplained(&tagged, &rejected(vec![error("R1 < &'x u8 >")])).len(),
            1
        );
        // The tag without the type is not enough.
        let plain = tags(&[gen::TAG_METHOD_LIFETIME]);
        assert_eq!(
            unexplained(&plain, &rejected(vec![error("I1 < & 'x str >")])).len(),
            1
        );
    }

    #[test]
    fn the_text_of_an_item_is_read_for_the_types_the_leaker_leaks() {
        let leaked = |config: gen::Config, text: &str, generics: &[&str]| {
            let generics = generics.iter().map(|g| g.to_string()).collect();
            gen::Gen::with_config(config).names_leaked_type(text, &generics)
        };
        let strict = strict_config;
        // A relative path, `Self`, a projection.
        assert!(leaked(strict(), "Into<R0>", &["T"]));
        assert!(leaked(strict(), "Fn(&Self) -> u8", &["T"]));
        assert!(leaked(strict(), "Fn(Self::A0) -> u8", &[]));
        assert!(leaked(
            strict(),
            "Fn(<T as ::core::iter::Iterator>::Item)",
            &["T"]
        ));
        assert!(leaked(strict(), "Fn(T::Item)", &["T"]));
        // The generics are no type, but the type of the module with the name of one is.
        assert!(!leaked(strict(), "Into<T>", &["T"]));
        assert!(leaked(strict(), "AsRef<U>", &["T"]));
        // The settings of the Leaker.
        assert!(!leaked(strict(), "Fn(u8) -> bool", &[]));
        let no_primitive = gen::Config {
            primitive: false,
            ..strict()
        };
        assert!(leaked(no_primitive, "Fn(u8) -> bool", &[]));
        let std = "Into<::std::string::String>";
        assert!(leaked(strict(), std, &[]));
        let allow_std = gen::Config {
            allow_std: true,
            ..strict()
        };
        assert!(!leaked(allow_std, std, &[]));
        let own = "Into<::tlp_l_000::base::B0>";
        assert!(leaked(strict(), own, &[]));
        let allow_crate = gen::Config {
            allow_crate: true,
            ..strict()
        };
        assert!(!leaked(allow_crate, own, &[]));
        assert!(!leaked(strict(), "Into<::core::option::Option<u8>>", &[]));
        // A bare trait is a trait in a bound, a path in a type where the prelude is not allowed.
        assert!(!leaked(strict(), "Fn(u8) -> u8 + Send", &[]));
        let object = "Into<&'static dyn for<'h1> Fn(&'h1 u8) -> &'h1 u8>";
        assert!(leaked(strict(), object, &[]));
        let prelude = gen::Config {
            prelude: true,
            ..strict()
        };
        assert!(!leaked(prelude, object, &[]));
        // An associated type that binds is no path.
        assert!(!leaked(strict(), "Iterator<Item = u8>", &[]));
        assert!(leaked(strict(), "Iterator<Item = R0>", &[]));
    }

    #[test]
    fn the_bare_traits_of_an_item_are_read_from_its_text() {
        let bare = |text: &str| {
            let mut names = gen::bare_traits_in(text);
            names.sort_unstable();
            names
        };
        assert_eq!(bare("T: Fn(u8) -> u8 + Send"), ["Fn", "Send"]);
        assert_eq!(bare("Box<dyn Iterator<Item = u8>>"), ["Iterator"]);
        assert_eq!(bare("T: FnMut(u8), U: Sized"), ["FnMut", "Sized"]);
        // The end of a path, `?Sized`, and the name of something else are no bare trait.
        assert!(bare("T: ::core::default::Default + ::core::ops::Fn(u8)").is_empty());
        assert!(bare("T: ?Sized + ?::core::marker::Sized").is_empty());
        assert!(bare("struct Defaults<T: Cloned> { copy: Copy_ }").is_empty());
        // In the tags only where the Leaker does not allow the prelude.
        let text = "pub struct S<T: Clone> where T: ?Sized {}";
        assert!(tags_of_text(strict_config(), text).contains("bare:Clone"));
        let prelude = gen::Config {
            prelude: true,
            ..strict_config()
        };
        assert!(!tags_of_text(prelude, text)
            .iter()
            .any(|tag| tag.starts_with("bare:")));
    }

    #[test]
    fn the_predicates_and_the_generics_of_an_item_are_read() {
        let item = "pub trait Tr<'a, T: Clone + Fn(R0), const N: usize> where U: Debug, \
                    Vec<T>: Default { fn m<X: AsRef<R1<T>>, Y>(&self, a: &'a X) where \
                    <X as Iterator>::Item: Sized, Y: for<'h> Fn(&'h T) -> &'h Y + Send; }";
        let generics = gen::declared_generics(item);
        assert_eq!(
            generics.iter().map(String::as_str).collect::<Vec<_>>(),
            ["N", "T", "X", "Y"]
        );
        let clauses = gen::bound_clauses(item);
        let has = |bounded: &str, bounds: &str| {
            clauses
                .iter()
                .any(|(a, b)| a == bounded && b.replace(' ', "") == bounds.replace(' ', ""))
        };
        assert!(has("T", "Clone + Fn(R0)"), "{:?}", clauses);
        assert!(has("X", "AsRef<R1<T>>"), "{:?}", clauses);
        assert!(has("U", "Debug"), "{:?}", clauses);
        assert!(has("Vec<T>", "Default"), "{:?}", clauses);
        assert!(has("<X as Iterator>::Item", "Sized"), "{:?}", clauses);
        assert!(
            has("Y", "for<'h> Fn(&'h T) -> &'h Y + Send"),
            "{:?}",
            clauses
        );
        // Only the generics of the item are bounded types of the environment.
        assert_eq!(gen::bounded_generic("T", &generics), Some("T"));
        assert_eq!(gen::bounded_generic("X::Item", &generics), Some("X"));
        assert_eq!(
            gen::bounded_generic("<X as Iterator>::Item", &generics),
            Some("X")
        );
        assert_eq!(gen::bounded_generic("Self", &generics), Some("Self"));
        assert_eq!(gen::bounded_generic("U", &generics), None);
        assert_eq!(gen::bounded_generic("Vec<T>", &generics), None);
        assert_eq!(
            gen::split_top("A + Fn(B + C) -> D<E + F> + G", '+'),
            ["A", "Fn(B + C) -> D<E + F>", "G"]
        );
        // The spellings of the Leaker's tokens and of `relativize_std()` are one.
        assert_eq!(
            gen::norm("<T as ::core::iter::Iterator>::Item"),
            gen::norm("< T as core :: iter :: Iterator > :: Item")
        );
        assert_eq!(gen::norm("::std::vec::Vec"), gen::norm("alloc::vec::Vec"));
    }

    /// The predicates that rustc proves by an impl make no cycle, and the `Item` of an iterator that a
    /// predicate names does.
    #[test]
    fn a_predicate_on_a_type_that_is_no_generic_is_a_bound_only_if_it_names_an_item() {
        let tags = tags_of_text(
            strict_config(),
            "pub struct S<T: Iterator> where R2<T>: Default, \
             ::core::option::Option<<T as ::core::iter::Iterator>::Item>: Default {}",
        );
        let bounds: Vec<&String> = tags
            .iter()
            .filter(|tag| tag.starts_with(gen::TAG_BOUND_AT))
            .collect();
        assert_eq!(bounds.len(), 1, "{:?}", bounds);
        assert!(bounds[0].contains("Option"), "{:?}", bounds);
    }

    #[test]
    fn the_generator_tags_the_lifetime_that_a_path_holds_in_a_reference() {
        let mut found = 0;
        for index in 0..24 {
            let case = generate_with(
                case_seed(19, index),
                index,
                8,
                Features::parse(""),
                Flags::default(),
            );
            for item in &case.items {
                found += item
                    .tags
                    .iter()
                    .filter(|tag| {
                        tag.strip_prefix(gen::TAG_LIFETIME_AT)
                            .map_or(false, |ty| ty.contains("<&'"))
                    })
                    .count();
            }
        }
        assert!(
            found > 0,
            "no item has a path with a reference to a lifetime of a method"
        );
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
    fn the_decoy_crate_is_for_the_items_that_do_not_trust_the_leaker() {
        let mut with_decoy = 0;
        for seed in 0..12 {
            let case = generate_with(seed, 0, 6, Features::parse(""), Flags::default());
            let wanted: Vec<&gen::Item> = case
                .items
                .iter()
                .filter(|item| {
                    item.kind != "mod" && matches!(item.trust, gen::Trust::None | gen::Trust::Std)
                })
                .collect();
            // The Leaker's own rule allowing its crate is a promise that it is the same crate.
            let expected = !case.config.allow_crate && !wanted.is_empty();
            assert_eq!(case.decoy.is_some(), expected, "{:?}", case.config);
            let Some(decoy) = &case.decoy else { continue };
            with_decoy += 1;
            assert_eq!(decoy.lines.len(), wanted.len());
            assert!(
                decoy.lib.contains("pub const BLEN: usize = 3;"),
                "{}",
                decoy.lib
            );
            assert!(!decoy.lib.contains("tlp_l_000"), "{}", decoy.lib);
            for item in wanted {
                let call = format!("real_000::__tl_prop_{}!(", item.id);
                let lines: Vec<(usize, &str)> = decoy
                    .referrer
                    .lines()
                    .enumerate()
                    .filter(|(_, line)| line.contains(&call))
                    .collect();
                assert_eq!(lines.len(), 1, "{}", decoy.referrer);
                let (at, line) = lines[0];
                assert!(line.contains("path = ::real_000"), "{}", line);
                assert!(
                    line.contains(&format!("trust({})", item.trust.name())),
                    "{}",
                    line
                );
                assert!(line.ends_with(", decoy(tlp_l_000));"), "{}", line);
                assert!(decoy.lines.contains(&(at + 1, item.id)));
            }
        }
        assert!(with_decoy > 0, "no case has a decoy in 12 cases");
    }

    #[test]
    fn the_generated_cases_say_what_the_macro_needs() {
        let mut trusts = std::collections::BTreeMap::new();
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
                    attr.contains(&format!(
                        "trust = \"{}\", krate = \"{}\",",
                        item.trust.name(),
                        case.config.krate
                    )),
                    "{} {}",
                    attr,
                    item.trust.name()
                );
                assert!(attr.contains(&format!("primitive = {})", case.config.primitive)));
                *trusts.entry(item.trust.name()).or_insert(0) += 1;
                // A mode that nothing could tell from `none` is not drawn.
                assert!(item.trust != gen::Trust::Std || !case.config.allow_std);
                assert!(item.trust != gen::Trust::Leaker || !case.config.allow_crate);
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
                // A bare `?Sized` is accepted whatever the prelude: it gets no tag.
                assert!(!item.tags.contains("bare:?Sized"), "{:?}", item.tags);
                if case.config.prelude {
                    assert!(
                        !item.tags.iter().any(|tag| tag.starts_with("bare:")),
                        "{:?}",
                        item.tags
                    );
                }
            }
        }
        for trust in gen::Trust::ALL {
            assert!(
                trusts.contains_key(trust.name()),
                "no item has the trust {} in 12 cases: {:?}",
                trust.name(),
                trusts
            );
        }
        assert!(
            nested_self_default > 0 && redeclared > 0 && plain_default > 0,
            "the traits of `trait_self_defaults` are not generated in 12 cases: {} nested that \
             need Self, {} declared again, {} controls",
            nested_self_default,
            redeclared,
            plain_default
        );
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
            rejections: vec![rejection_of(
                "DeniedSelfInBounds",
                "`Wrap<T>` has a generic whose bound mentions `Self`, which cannot be leaked",
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
                assert!(
                    line.contains(&format!(", trust({})", item.trust.name())),
                    "{}",
                    line
                );
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
                    // The mode of the Referrer is the rule's, which the attribute of the copy says,
                    // so the invocation does not.
                    let mut accepted = case.accept_referrer.replace("__tl_prop_", &prefix);
                    for trust in gen::Trust::ALL {
                        accepted = accepted.replace(&format!(", trust({})", trust.name()), "");
                    }
                    assert_eq!(accepted, stage.lib);
                } else {
                    // None of the Leaker's names but the ones the rule allows: no glob import (but for
                    // the modules of the config's copy of a rule that is a place, which wrote every
                    // relative path of a value or a default as it is), `base` where `crate::base`
                    // or `super::base` is allowed, and the modules where the paths are relative to
                    // the module.
                    let globs = matches!(stage.rule, "values" | "default") && stage.side == "cfg";
                    if !globs {
                        assert!(stage.lib.matches("::*;").count() <= 1, "{}", stage.lib);
                    } else {
                        assert!(
                            stage.lib.contains("pub use ::tlp_l_000::m0::*;"),
                            "{}",
                            stage.lib
                        );
                    }
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

    /// The failures of the copy of an item accepted by `allow()` are the ones that
    /// `accept_families()` lists, on an item that has the shape.
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

    /// `TYPE_LEAK_PROP_MIN_ACCEPT=0` turns the floors of the kinds off, not the check that the
    /// witnesses compare the types.
    #[test]
    fn turning_the_floors_off_keeps_the_check_of_the_witnesses() {
        let off = Floors::parse("0", MIN_SAMPLES, MIN_CHECKED).unwrap();
        let mut stats = stats_of(&[("struct", 90, 10), ("mod", 1, 30)]);
        // A module far below its floor is no problem now.
        assert!(off.problems(&stats).is_empty());
        stats.checked = 0;
        let problems = off.problems(&stats);
        assert_eq!(problems.len(), 1, "{:?}", problems);
        assert!(problems[0].contains("witness"), "{:?}", problems);
        // That check has its own variable (`TYPE_LEAK_PROP_MIN_CHECKED`).
        let none = Floors::parse("0", MIN_SAMPLES, 0.0).unwrap();
        assert!(none.problems(&stats).is_empty());
    }

    /// A crate of the case that has a diagnostic is explained by it; the other crates of the case
    /// still have to be built, and the executable of the Leaker too.
    #[test]
    fn a_crate_with_a_diagnostic_does_not_excuse_the_other_crates_of_the_case() {
        let case = generate_with(3, 3, 6, Features::parse(""), Flags::default());
        let stage = &case.stages[0];
        let stages = std::slice::from_ref(stage);
        let mut b = build(false, &[referrer_name(3).as_str()]);
        b.executables.insert(leaker_name(3), PathBuf::from("/x"));
        let missing = missing_artifacts(&b, 3, true, true, stages);
        assert_eq!(missing.len(), 2, "{:?}", missing);
        assert!(missing[0].contains(&accept_name(3)), "{:?}", missing);
        assert!(missing[1].contains(&stage_name(3, stage)), "{:?}", missing);
        // Each crate that has a diagnostic is explained.
        let mut b = build(
            false,
            &[
                referrer_name(3).as_str(),
                accept_name(3).as_str(),
                stage_name(3, stage).as_str(),
            ],
        );
        b.executables.insert(leaker_name(3), PathBuf::from("/x"));
        assert!(missing_artifacts(&b, 3, true, true, stages).is_empty());
        // A Leaker's executable that is missing is reported although the Referrer's crate failed.
        let b = build(false, &[referrer_name(3).as_str()]);
        let missing = missing_artifacts(&b, 3, true, false, &[]);
        assert_eq!(missing.len(), 1, "{:?}", missing);
        assert!(missing[0].contains("no executable"), "{:?}", missing);
    }

    /// A failure of a stage copy keeps its rule, side and crate, and so does the crate it names.
    #[test]
    fn a_failure_of_a_stage_copy_keeps_its_stage_and_names_its_crate() {
        let key = StageKey {
            rule: "prelude",
            side: "res",
            clean: true,
        };
        let failure = |class, stage| Failure {
            case: 0,
            item: None,
            class,
            code: String::new(),
            text: "error".into(),
            stage,
        };
        let (_, problems, _) = explain_copies(
            vec![failure(STAGE_COPY, Some(key))],
            &BTreeMap::new(),
            &[],
            &STAGE_CLASSES,
            &BTreeSet::new(),
            true,
        );
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].stage, Some(key));
        let ws = Path::new("/ws");
        assert_eq!(
            failure_crates(&problems[0], ws),
            ["/ws/c000_l", "/ws/c000_prelude_res_clean"]
        );
        // An accepted copy is in the second Referrer's crate.
        let (_, problems, _) = explain_copies(
            vec![failure(ACCEPT_COPY, None)],
            &BTreeMap::new(),
            &[],
            &ACCEPT_CLASSES,
            &BTreeSet::new(),
            false,
        );
        assert_eq!(problems[0].stage, None);
        assert_eq!(
            failure_crates(&problems[0], ws),
            ["/ws/c000_l", "/ws/c000_a"]
        );
        assert_eq!(
            failure_crates(&failure("", None), ws),
            ["/ws/c000_l", "/ws/c000_r"]
        );
    }

    /// With both modes, the stage copies are not blamed on the accepted items, nor the other way.
    #[test]
    fn a_leaker_crate_that_fails_is_blamed_on_the_mode_that_makes_it_fail() {
        // It does not compile without the extra output either: the macro.
        for (accept, stages) in [(false, false), (true, false), (false, true), (true, true)] {
            assert_eq!(
                leaker_failure_class(false, accept, stages, true),
                LEAKER_MACRO
            );
        }
        assert_eq!(
            leaker_failure_class(true, true, false, false),
            LEAKER_ACCEPT
        );
        assert_eq!(
            leaker_failure_class(true, false, true, false),
            LEAKER_STAGES
        );
        // Both: the accepted items alone compile, so the stage copies do not; else the accepted
        // items do not.
        assert_eq!(leaker_failure_class(true, true, true, true), LEAKER_STAGES);
        assert_eq!(leaker_failure_class(true, true, true, false), LEAKER_ACCEPT);
    }

    #[test]
    fn a_setting_is_off_when_it_is_empty_or_zero() {
        assert!(is_on("1") && is_on("yes") && is_on(" 2 "));
        assert!(!is_on("0") && !is_on(" 0 ") && !is_on("") && !is_on("  "));
    }

    fn run() -> Run {
        Run {
            seed: 1,
            cases: 60,
            items: 8,
            features: String::new(),
            accept: false,
            stages: false,
            only: None,
        }
    }

    /// Two runs that differ in a setting that changes the sources do not share a workspace.
    #[test]
    fn runs_with_other_settings_have_other_workspaces() {
        let base = run();
        assert_eq!(base.workspace_name(), "ws-1-60-i8");
        let name = |change: &dyn Fn(&mut Run)| {
            let mut run = base.clone();
            change(&mut run);
            run.workspace_name()
        };
        let names = [
            name(&|r| r.seed = 2),
            name(&|r| r.cases = 24),
            name(&|r| r.items = 6),
            name(&|r| r.accept = true),
            name(&|r| r.stages = true),
            name(&|r| {
                r.accept = true;
                r.stages = true
            }),
            name(&|r| r.features = "values".into()),
            name(&|r| r.features = "relative_std".into()),
            name(&|r| r.only = Some(3)),
            name(&|r| r.only = Some(4)),
        ];
        let mut all: BTreeSet<&str> = names.iter().map(String::as_str).collect();
        all.insert("ws-1-60-i8");
        assert_eq!(all.len(), names.len() + 1, "{:?}", names);
        assert_eq!(names[3], "ws-1-60-i8-accept");
        assert_eq!(names[5], "ws-1-60-i8-accept-stages");
        assert_eq!(names[8], "ws-1-60-i8-case3");
        // The features, however they are spelled around the words.
        assert!(names[6].starts_with("ws-1-60-i8-f"), "{}", names[6]);
        assert_eq!(names[6], name(&|r| r.features = " values ".into()));
    }

    /// The failure message prints a command that repeats the run, or its case.
    #[test]
    fn the_command_that_repeats_a_run_has_its_settings() {
        let command = run().command(None);
        assert!(
            command.starts_with(
                "TYPE_LEAK_PROP_SEED=1 TYPE_LEAK_PROP_CASES=60 TYPE_LEAK_PROP_ITEMS=8 cargo test"
            ),
            "{}",
            command
        );
        assert!(command.ends_with("--test prop_cross_crate -- --ignored --nocapture"));
        assert!(!command.contains("FEATURES") && !command.contains("ACCEPT"));
        assert!(!command.contains("TYPE_LEAK_PROP_CASE="));
        let mut other = run();
        other.accept = true;
        other.features = " allow_stages,-strict_cycles ".into();
        let command = other.command(Some(17));
        assert!(command.contains("TYPE_LEAK_PROP_FEATURES=\"allow_stages,-strict_cycles\""));
        assert!(command.contains("TYPE_LEAK_PROP_ACCEPT=1"));
        assert!(command.contains("TYPE_LEAK_PROP_CASE=17"));
        // A run of one case repeats that case.
        let mut one = run();
        one.only = Some(5);
        assert!(one.command(None).contains("TYPE_LEAK_PROP_CASE=5 cargo"));
        assert!(one.command(Some(9)).contains("TYPE_LEAK_PROP_CASE=9 cargo"));
    }

    #[test]
    fn one_case_of_a_run_is_selected_by_its_number() {
        assert_eq!(selected_cases(3, None), Ok(vec![0, 1, 2]));
        assert_eq!(selected_cases(60, Some(17)), Ok(vec![17]));
        let error = selected_cases(3, Some(3)).unwrap_err();
        assert!(error.contains("TYPE_LEAK_PROP_CASE=3"), "{}", error);
    }

    /// A case is a function of the seed and its number, so a run of one case has the items of the
    /// case in a full run, on any release of Rust: the hash is fixed (these numbers must not change).
    #[test]
    fn a_case_is_the_same_alone_and_in_a_run_and_the_hash_is_fixed() {
        use std::hash::Hasher;
        let mut hasher = StableHasher::new();
        hasher.write(b"type-leak");
        assert_eq!(hasher.finish(), 0x22da_11fe_e25c_2233);
        assert_eq!(case_seed(1, 0), 0x1468_1df3_0754_59cb);
        assert_eq!(case_seed(1, 1), 0xa09f_6798_6254_8a98);
        assert_ne!(case_seed(1, 7), case_seed(2, 7));
        assert_ne!(case_seed(1, 7), case_seed(1, 8));
        let features = Features::parse("");
        let case = |seed| generate_with(case_seed(seed, 7), 7, 8, features, Flags::default());
        assert_eq!(case(1).leaker, case(1).leaker);
        assert_ne!(case(1).leaker, case(2).leaker);
    }

    #[test]
    fn features_can_be_turned_off_as_well_as_on() {
        let default = Features::parse("");
        assert!(default.nested_self_entries && !default.strict_anchors);
        let changed = Features::parse("-nested_self_entries, strict_anchors");
        assert!(!changed.nested_self_entries && changed.strict_anchors);
        // The others stay as they are.
        assert!(changed.nested_self && !changed.inferred_outlives);
        let all = Features::parse("all");
        assert!(all.strict_anchors && all.inferred_outlives && all.nested_self_entries);
        // In order.
        let almost = Features::parse("all,-strict_anchors");
        assert!(almost.inferred_outlives && !almost.strict_anchors);
        assert!(!Features::parse("-all").nested_self);
        // `allow_stages`, `relative_std` and `values` are flags of the generator, not features.
        assert_eq!(
            format!("{:?}", Features::parse("allow_stages,-values")),
            format!("{:?}", default)
        );
    }

    #[test]
    #[should_panic(expected = "unknown feature `oops`")]
    fn an_unknown_feature_is_refused_off_as_well() {
        Features::parse("-oops");
    }

    #[test]
    #[should_panic(expected = "TYPE_LEAK_PROP_ITEMS is 3")]
    fn a_module_has_at_least_four_items() {
        let features = Features::parse("");
        generate_with(1, 0, MIN_ITEMS, features, Flags::default());
        generate_with(1, 0, MIN_ITEMS - 1, features, Flags::default());
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

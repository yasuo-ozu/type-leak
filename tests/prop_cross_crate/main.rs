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
//!   The floors count the items that have no forced shape (below): the table of the summary
//!   has the share of all the items too (`all%`), and a line says how many items were left out.
//! - The forced shapes. The generator writes some shapes only to reach a `Reason` that random types
//!   rarely do, and the Leaker must reject the item that has one, under the settings of the item
//!   (`FORCED`, which classifies `SHAPE_TAGS`, with the numbers measured): such an item is left out
//!   of the floors, and fails the run if the Leaker accepts it ("SHOULD BE REJECTED") or rejects it
//!   with no error of the shape's reason ("SHOULD BE REJECTED FOR ITS SHAPE": the reason, and the
//!   kind of the cause or its tokens where they tell the shape, `Forced`). A shape that the Leaker
//!   may accept (`nested_self_in_bounds_option`, a macro of the Leaker's crate by an absolute path)
//!   is not forced, and its items count in the floors, and so are those of a shape that only some of
//!   its items force (the tag `gen::FORCED_*` that the text of the item gives, `gen::forced_in()`).
//!   The summary counts the items of each forced shape, so that the check cannot pass with none
//!   (`Stats::forced_check_problems()`: at least one item in `FORCED_SHARE` has one).
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
//! - an `UnprovableBound` whose note says that the path of a bound of a generic may be a type alias
//!   of a projection (`U: Fn(BIt<U>) -> u8` next to `U: Iterator<Item = u8>` or `U: BItem`; the
//!   Leaker cannot tell an alias from a struct, so a struct path is rejected too) needs the generic
//!   and the bound in a tag `hidden_bound=<generic>|<bound>` that the item wrote, with an entry
//!   that keeps the bound (`Self`, or the self type of the impl, in a predicate): `Agree::Hidden`;
//! - an `UnconstrainedLifetime` needs a leaked path or trait object that holds the lifetime of a
//!   method, behind a reference in its arguments too (`Rel<&'x str>`): `method_lifetime=<type>`,
//!   which the cause is (`Agree::Type`);
//! - `bare:<Name>` is read from the text of the item where the Leaker does not allow the prelude;
//! - the shapes that reach the other reasons are tags of what the item writes (`gen::TAG_*`,
//!   `SHAPE_TAGS`): `SelfWithoutMeaning` needs `self_without_meaning` (a field that names `Self` in
//!   an item analyzed in `enter_generics`), `SelfInDefault` `self_in_default` (a trait whose default
//!   names `Self`), `SelfInTraitPredicate` `self_in_trait_predicate` (a type that names `Self` in
//!   the header of a trait or an impl, not as the bounded type), `SelfInImplSelfType`
//!   `impl_header_self` (the generics or the where clause of an impl name `Self`), `Unreadable`
//!   `unreadable_value` (a glob import or a `macro_rules!` in a body, with `values`),
//!   `NestedSelfInBounds` `nested_self_in_bounds` (the shape of `self_bound_adts` without the
//!   option; the Leaker must reject it) or `nested_self_in_bounds_text` (a bound that names `Self`
//!   in an item nested in `enter_mod` without the option), a cause of the kind `Macro` `macro_path`,
//!   a path that holds an array whose length is a macro call `macro_call`, and `Rl<'_>`
//!   `elided_arg`. Items with the option off are rejected if they have the shape, and with it on
//!   no rejection has a tag, so a `NestedSelfInBounds` with the option on fails the run.
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
//!    KNOWN REASON", "NO SHAPE THAT GIVES IT"). If an error remains, the macro panics with the
//!    item and the errors that remain, the Leaker's crate does not build, and the run fails
//!    ("ACCEPT FAILED"): allowing the causes of all the errors must leave none. (The same for the
//!    copies of `allow_stages`.)
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
//! built after the result and after the config, with globs and clean, and the failures by kind. The
//! path rules can leave other errors after the result than in the config (see
//! `tests/allow_stages.rs`); the summary counts the items where they do, and no more.
//!
//! # More of the API than the copies compare
//!
//! - **Other calls of one scope** (`via` in the attribute of an item, `gen::Gen::with_via()`, `Args::via`
//!   of the macro): a struct, an enum or a union by `enter_type`, with its generics given to the view
//!   by parts (`visit_type_param_bound`, `visit_where_clause`, `visit_type`), with `View::check()` on its
//!   types first (a scope for the generics of a method nested in the view), or nested in `enter_mod`
//!   by `View::enter_type`; a trait by `enter_trait` with the parts or the checks; a trait impl by
//!   `enter_impl` with the path of its trait, and an impl by `enter_impl` and the walk of syn. The
//!   analysis must equal that of the plain scope (`VIA DIFFERS`), `check()` must record nothing and
//!   say what the copy does (`CHECK DISAGREES`), and the copy is compared as the others are.
//! - **Use sites with arguments** (`inst(N)`, `prop.rs: Instance`): the item is also used at an
//!   instantiation of its generics (lifetimes `'static` or named by the use site, types `(u8, u16)`,
//!   `Vec<u8>`, `str`, consts `3`, `{ 2 + 1 }`, the defaults at the end left out for
//!   `Referrer::complete_args()`): a trait is implemented for a host with them, each position is
//!   compared with the witness called with them, `S<given>` must be `S<completed>`, and
//!   `Referrer::context_generic_args()` must refuse a list with an argument too many, of another
//!   kind or missing. A parameter that a function pointer or `Fn(..)` with an elided output names in
//!   its inputs is given a type that holds a lifetime (`&'static str`) in a share of the items, and
//!   the copy must name the elided lifetime there (F69, `UseSite::substitute()`, and F70 where the
//!   path through the Repeater that replaces an entry in the inputs names it): the summary counts the
//!   pairs of a parameter and a signature (`inst_elided`), and those with a replaced entry in the
//!   inputs (`inst_elided_entry`).
//! - **Shadowing** (`shadow(N)`, `prop.rs: shadow_check()`): the item is copied inside a module and a
//!   block that declare names its entries start with: the visitor must leave those entries.
//! - **Unique names**: each copy is in a module of its own (`c<id>`), so a relative path that it keeps by
//!   mistake cannot resolve to the copy of another item.
//! - **The carried errors** (`prop.rs: carry()`, `gen::errors_crate()`): the crate `tlp_e_NNN` reads the
//!   result of each item back from tokens (`LeakerResult`, the `@[..]` list), the errors must be the
//!   Leaker's, `allow()` drops none, the first or all, and the rest is reported with
//!   `ErrorItem::emit()`: the crate does not compile, and its diagnostics are matched with the
//!   Leaker's errors (`carry_failures()`). With `TYPE_LEAK_PROP_ACCEPT` a fifth of the items are leaked
//!   with `prop_leak_result` and the crate of the accepted items drops their errors (`allow(all)`).
//!
//! What the macros did is counted (`TLCOUNT` on their stderr, `use_summary()`), and a run where `inst`
//! or `shadow` was asked of many items and compared nothing fails.
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
//!   (`tests/findings.rs` has most of them as a library-level test). `strict_anchors` (F18, about the
//!   items whose Referrer replaces optional entries, the modes but `all`) is fixed, and on by
//!   default (`-strict_anchors`: the items that have it are invoked with `all`). `unsized_params`,
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
//!   `nested_self_entries` for half of the items nested in `enter_mod`, and those of
//!   `self_bound_adts` have a field that names `Self` in a relative type: without the option the
//!   Leaker rejects it with `NestedSelfInBounds`, which `unexplained()` requires, and the option
//!   makes it accept it),
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
//!   path of a value (`crate::base::bf()`, `RLEN`), which it rejects (the tag `value_path`). An item
//!   whose text writes one, the root of a `use`, or a name of the prelude where its attribute says
//!   `prelude = false`, is a forced shape (`forced:value_path`) unless it is analyzed by `enter_impl`,
//!   which reads no body: it is left out of the floors, and fails the run if the Leaker accepts it or
//!   rejects it for another value.
//!
//! The test is ignored by default: it is slow, as it builds the generated crates. Run it with
//!
//! ```text
//! cargo test --test prop_cross_crate -- --ignored
//! TYPE_LEAK_PROP_CASES=200 TYPE_LEAK_PROP_SEED=7 cargo test --test prop_cross_crate -- --ignored
//! ```
//!
//! CI runs it (the job `prop` of `.github/workflows/rust.yml`) with fixed seeds: seed 1 with 60
//! cases, `TYPE_LEAK_PROP_ACCEPT` with seed 3 and 60 cases, and `allow_stages` with seed 1 and 30
//! cases (a stages run takes about three times as long per case), and the tests of
//! `tests/crates/macro`. A failure prints the command that repeats it. A run with fewer cases has
//! fewer kinds with enough samples for their floors (`TYPE_LEAK_PROP_MIN_SAMPLES`).
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

/// The crate that reports the errors the Leaker carries (`gen::Case::errors_referrer`).
fn errors_dir(i: usize) -> String {
    format!("c{:03}_e", i)
}

fn errors_name(i: usize) -> String {
    format!("tlp_e_{:03}", i)
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
            if mode != Mode::Identity {
                dirs.push(format!("\"{}\"", errors_dir(*i)));
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
        let mut leaker = case.leaker.clone();
        if mode == Mode::Accept {
            // A fifth of the items carry their result to the Referrer (`gen::carries_result()`).
            for item in case
                .items
                .iter()
                .filter(|item| gen::carries_result(item.id))
            {
                leaker = leaker.replace(
                    &format!("{}(id = {},", gen::ATTR, item.id),
                    &format!("crate::__macro::prop_leak_result(id = {},", item.id),
                );
            }
        }
        let leaker = leaker.replace(gen::ATTR, attr);
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

        {
            let dir = ws.join(errors_dir(*i));
            std::fs::create_dir_all(dir.join("src")).unwrap();
            std::fs::write(
                dir.join("Cargo.toml"),
                format!(
                    "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
                     [dependencies]\n{} = {{ path = \"../{}\" }}\n",
                    errors_name(*i),
                    leaker_name(*i),
                    leaker_dir(*i)
                ),
            )
            .unwrap();
            std::fs::write(dir.join("src/lib.rs"), &case.errors_referrer).unwrap();
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
    /// What the macros of the Referrer's crates counted of the copies they made at an instantiation
    /// of the generics and inside a module and a block (`TLCOUNT <what> <id> <n>` on stderr).
    counts: Vec<Count>,
}

/// `what` is `inst` (positions compared at an instantiation) or `shadow` (entries that a module and a
/// block hid) of one item, and `n` how many.
struct Count {
    what: String,
    n: usize,
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
        counts: Vec::new(),
    };
    for line in build.stderr.lines() {
        if let Some(count) = line.strip_prefix("TLCOUNT ") {
            let mut words = count.split_whitespace();
            if let (Some(what), Some(_), Some(n)) = (
                words.next(),
                words.next(),
                words.next().and_then(|w| w.parse().ok()),
            ) {
                build.counts.push(Count {
                    what: what.to_string(),
                    n,
                });
            }
        }
    }
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

impl Failure {
    /// A failure with no error code and in no crate of `allow_stages`.
    fn new(case: usize, item: Option<usize>, class: &'static str, text: String) -> Failure {
        Failure {
            case,
            item,
            class,
            code: String::new(),
            text,
            stage: None,
        }
    }
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
    /// The reason of the rejection (the messages of its errors, on one line), or the error of the
    /// rewrite.
    detail: String,
    /// The errors of the rejection, whatever was allowed.
    rejections: Vec<Rejection>,
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
    // The errors of a rejection follow the report. The copy of `allow_stages` has the errors of the
    // analysis without the rule at the end.
    let (line, errors) = line.split_once('\u{1e}').unwrap_or((line, ""));
    let (errors, plain) = errors.split_once('\u{1b}').unwrap_or((errors, ""));
    let mut parts = line.splitn(4, '|');
    let id = parts.next()?.parse().ok()?;
    let kind = parts.next()?.to_string();
    let status = parts.next()?;
    let rest = parts.next().unwrap_or("");
    let mut report = Report {
        kind,
        accepted: status == "ok",
        allowed: status == "acc",
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
/// difference is only counted. The cause and place rules (`values`, `default`) are: the config and
/// the result give the same errors, or the run fails (their Referrers differ in the parts).
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

    /// The lines of the summary about the copies.
    fn summarize(&self) -> String {
        let mut s = String::new();
        let count = |reports: &BTreeMap<(usize, usize), Report>| {
            let (mut ok, mut acc, mut errors) = (0, 0, 0);
            let (mut entries, mut positions, mut checked) = (0, 0, 0);
            for report in reports.values() {
                ok += usize::from(report.accepted);
                acc += usize::from(report.allowed);
                errors += usize::from(report.rewrite_error.is_some());
                if made(report) {
                    entries += report.entries;
                    positions += report.positions;
                    checked += report.checked;
                }
            }
            (ok, acc, errors, entries, positions, checked)
        };
        let matters = |reports: &BTreeMap<(usize, usize), Report>| -> usize {
            reports.values().filter(|report| !report.skip).count()
        };
        let (ok, acc, errors, entries, positions, checked) = count(self.reports);
        let built = self
            .reports
            .keys()
            .filter(|(case, item)| self.built(*case, *item))
            .count();
        writeln!(
            s,
            "  allow_stages [{}] copies after the result: {} of {} built in the Referrer's crates ({} with no error left by the rule, {} with the causes of those allowed, {} not made by the macro); entries: {}, type positions: {}, witnesses checked: {}",
            self.rule, built, matters(self.reports), ok, acc, errors, entries, positions, checked
        )
        .unwrap();
        let (ok, acc, errors, entries, _, _) = count(self.base_reports);
        let base_built = self
            .base_reports
            .iter()
            .filter(|(_, report)| made(report))
            .count();
        writeln!(
            s,
            "  allow_stages [{}] copies after the config: {} of {} built ({} with no error, {} with the causes of the errors allowed, {} not made by the macro); entries: {}; {} of them do not compile",
            self.rule,
            base_built,
            matters(self.base_reports),
            ok,
            acc,
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
    /// stages, and the copies.
    fn summarize(rule: &str, stages: &[Stages], copies: &StageCopies<'_>) -> String {
        let mut s = String::new();
        let differ = stages.iter().filter(|s| s.before != s.after).count();
        let with_errors = stages.iter().filter(|s| s.plain > 0).count();
        let mattering = stages.iter().filter(|s| s.matters).count();
        writeln!(
            s,
            "  allow_stages [{}] (TYPE_LEAK_PROP_FEATURES): {} items compared, {} where the rule matters, {} of them have errors without it; config and result leave the same number of errors in {}, a different one in {}; errors: {} without the rule, {} with it in the config, {} after it in the result",
            rule,
            stages.len(),
            mattering,
            with_errors,
            stages.len() - differ,
            differ,
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
                    problems.push(Failure::new(stage.case, Some(stage.item), "STAGES DIFFER: the config and the result give other errors, for a rule that is a place", format!(
                            "rule {}: {} errors in the config, {} in the result; the same errors: {}; an equal Referrer: {}\n    config: {}\n    result: {}",
                            rule,
                            stage.before,
                            stage.after,
                            stage.same_errors,
                            stage.same_referrer,
                            stage.before_causes,
                            stage.after_causes
                        )));
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
    /// `hidden_bound=<generic>|<bound>`: the note names the same generic and the same bound, as in
    /// "names `U` in the arguments of the bound `Fn(BIt<U>) -> u8`".
    Hidden,
}

/// What the note of an `UnprovableBound` says of a bound that may hide a projection.
const HIDDEN_NOTE: &str = "whose path may be a type alias of a projection";

/// The generic and the bound that the note of an `UnprovableBound` names as hiding a projection
/// ("names `U` in the arguments of the bound `Fn(BIt<U>) -> u8`, whose path may be .."), as the
/// Leaker's tokens print them.
fn hidden_of_note(note: &str) -> Option<(&str, &str)> {
    let marker = "` in the arguments of the bound `";
    let at = note.find(marker)?;
    let generic = note[..at].rsplit('`').next()?;
    let rest = &note[at + marker.len()..];
    Some((generic, &rest[..rest.find('`')?]))
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
            Agree::Hidden => {
                let Some((generic, bound)) = payload.split_once('|') else {
                    return false;
                };
                hidden_of_note(&error.note).map_or(false, |(named, hides)| {
                    named == generic && gen::norm(hides) == bound
                })
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
/// (`MaybeConst`, `RelativePath`), and the message is not read. The reason that the generator makes
/// no item for (`UnnameableSelf`: it needs an item nested in a scope that has a `Self` of its own
/// under generics that bound the outer one, which no valid item is) is none of them.
fn family(error: &Rejection) -> Option<Family> {
    match error.reason.as_str() {
        // The header of the trait writes `Self` in a type of a bound or a where predicate.
        "SelfInTraitPredicate" => family_of(
            "a leaked type that names `Self` in a bound or where predicate of its trait",
            &[gen::TAG_SELF_IN_TRAIT_PRED],
        ),
        // `enter_generics` gives `Self` no meaning, and the item names it.
        "SelfWithoutMeaning" => family_of(
            "`Self` in an item analyzed in enter_generics, which gives it no meaning",
            &[gen::TAG_SELF_NO_MEANING],
        ),
        // The default of a parameter of a trait names `Self`.
        "SelfInDefault" => family_of(
            "`Self` in the default of a parameter of a trait",
            &[gen::TAG_SELF_IN_DEFAULT],
        ),
        // A glob import or a `macro_rules!` in a value that the Leaker reads.
        "Unreadable" => family_of(
            "syntax in a value that the Leaker cannot read (a glob import, a macro_rules!)",
            &[gen::TAG_UNREADABLE],
        ),
        // A bound that names its generic through a path that may be an alias of a projection, next
        // to a bound that fixes an associated type of it: the generic and the bound are ones the
        // item wrote (`gen::hidden_bounds_in()`). The Leaker cannot tell an alias from a struct.
        "UnprovableBound" if error.note.contains(HIDDEN_NOTE) => family_agreeing(
            "a bound of a generic that names it in the arguments of a path that may be an alias of a projection that another bound of the generic fixes",
            &[gen::TAG_HIDDEN_AT],
            Agree::Hidden,
        ),
        // The bound of a generic that makes the cycle is one the item wrote: `Gen::note_bounds()`.
        "UnprovableBound" => family_agreeing(
            "a leaked type in a bound or where predicate that rustc cannot normalize",
            &[gen::TAG_BOUND_AT],
            Agree::Bound,
        ),
        // The generics or the where clause of the impl write `Self` (`impl<T: AsRef<Self>> Rel<T>`).
        "SelfInImplSelfType" => family_of(
            "a type in the self type of an impl whose context depends on `Self` through a bound",
            &[gen::TAG_IMPL_HEADER_SELF],
        ),
        "DeniedSelfInBounds" => family_of(
            "a type that uses a generic whose bound mentions a `Self` that cannot be leaked",
            &[
                gen::TAG_BOUND_CYCLE,
                gen::TAG_SELF_IN_BOUNDS,
                gen::TAG_SELF_DENIED,
            ],
        ),
        // The same fault in a struct, an enum or a union nested in `enter_mod`, analyzed without
        // `nested_self_entries`: the item wrote a type that names `Self` and a generic whose bound
        // does (`gen::TAG_NESTED_SELF_BOUNDS`, put only where the option is off).
        "NestedSelfInBounds" => family_of(
            "a type that names the `Self` of a nested item and uses a generic whose bound names it, without nested_self_entries",
            &[gen::TAG_NESTED_SELF_BOUNDS, gen::TAG_NESTED_SELF_TEXT],
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

/// Whether the note says that the type cannot be leaked whole because of a macro call
/// (`because of `::tlp_l_000::two! ()``): an array length that a macro gives.
fn because_of_macro(note: &str) -> bool {
    let note: String = note.replace(char::is_whitespace, "");
    note.starts_with("itcannotbeleakedasawholebecauseof`") && note.contains("two!(")
}

/// A bare name (`u32`, `T`, `R0`) as an argument of a type that cannot be leaked as a whole: it may
/// be a const, so it is not leaked alone. The note says why the holder cannot be leaked, a reason
/// that is an error of the item too.
fn maybe_const_family(error: &Rejection) -> Option<Family> {
    let tags: &[&str] = if because_of_macro(&error.note) {
        &[gen::TAG_MACRO_CALL]
    } else if error
        .note
        .contains("which the module declares and the copy declares again")
    {
        &[gen::TAG_MODULE_SCOPE]
    } else if error.note.contains("because of `'") {
        &[
            gen::TAG_METHOD_LIFETIME,
            gen::TAG_BINDER_NO_ANCHOR,
            gen::TAG_ELIDED_ARG,
        ]
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
    for (name, _) in gen::PRELUDE_TRAITS {
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
                gen::TAG_ELIDED_ARG.to_string(),
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
    // A type that holds an array whose length is a macro call (`[u8; ::k::two!()]`) cannot be leaked
    // whole, so the relative path that holds it is an error.
    if because_of_macro(&error.note) {
        reasons.push((
            "a type that holds a macro call in an array length, which cannot be leaked whole",
            vec![gen::TAG_MACRO_CALL.to_string()],
        ));
    }
    // The bare name of a macro (`two!()`), which the Referrer's crate cannot resolve: the cause is
    // the macro's path.
    if error.kind == "Macro" {
        reasons.push((
            "a macro called by its bare name, which the copy cannot resolve",
            vec![gen::TAG_MACRO_PATH.to_string()],
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
    // bare trait of the prelude, where the Leaker does not allow the prelude. With a type that names
    // `Self` and the option off (`nested_self_in_bounds`) it must be rejected, by the reason of the
    // shape: `NestedSelfInBounds`.
    let nested_bounds = tags.contains(gen::TAG_NESTED_SELF_BOUNDS);
    let must_accept = tags.contains(gen::TAG_SELF_BOUND_ADT) && !nested_bounds;
    if nested_bounds
        && !report
            .rejections
            .iter()
            .any(|error| error.reason == "NestedSelfInBounds")
        && !report
            .rejections
            .iter()
            .any(|error| family(error).map_or(false, |f| f.name.starts_with("a bare trait")))
    {
        problems.push(
            "SHOULD BE REJECTED WITH NestedSelfInBounds: a struct, enum or union nested in enter_mod \
             without nested_self_entries, with a type that names `Self` and a generic whose bound does"
                .to_string(),
        );
    }
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
                            Some(Agree::Hidden) => {
                                " that agrees with the generic and the bound of the note"
                            }
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
fn prelude_names() -> impl Iterator<Item = &'static str> {
    const OTHERS: &[&str] = &[
        "Box",
        "String",
        "Vec",
        "Option",
        "Some",
        "None",
        "Result",
        "Ok",
        "Err",
        "Drop",
        "AsMut",
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
    ];
    gen::PRELUDE_TRAITS
        .iter()
        .map(|(name, _)| *name)
        .chain(gen::PRIMITIVE_NAMES.iter().copied())
        .chain(OTHERS.iter().copied())
}

/// The names of `prelude_names()` that the source of a crate declares (`struct Vec`, `type u8 = ..`,
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
        if let Some(prelude) = prelude_names().find(|prelude| *prelude == name) {
            found.insert(prelude);
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
            let mut names = vec![
                leaker_name(i),
                referrer_name(i),
                accept_name(i),
                errors_name(i),
            ];
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
        if !explained(&name) && !build.artifacts.contains(&name) {
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
    /// The items accepted by `allow()` whose copy does not compile for a failure that
    /// `accept_families()` lists, by its class (`(i)`, `(ii)`, `(iii)`).
    documented: BTreeMap<&'static str, usize>,
    /// The errors that were allowed, by `Reason` and the kind of the cause.
    allowed_errors: BTreeMap<(String, String), usize>,
    /// Every error of every rejection, by `Reason` and the kind of the cause: which of them the
    /// generator reaches.
    error_kinds: BTreeMap<(String, String), usize>,
    /// The items that have a shape of `SHAPE_TAGS`: how many the Leaker accepted and rejected.
    shapes: BTreeMap<&'static str, (usize, usize)>,
    /// The items rejected that have a shape of `SHAPE_TAGS`, by the kind, the scope (the name of its
    /// API) and the construct they have: what the shape costs the acceptance floors of each.
    shape_rejected: BTreeMap<(&'static str, &'static str), usize>,
    /// The items with a forced shape (`FORCED`), left out of the floors, and what the Leaker did
    /// with each shape (by its tag).
    left_out: usize,
    forced: BTreeMap<&'static str, ForcedCount>,
    /// Accepted and rejected over all the items, the forced ones too, by kind, scope and construct,
    /// for the column `all%` of the tables.
    all: BTreeMap<&'static str, (usize, usize)>,
}

/// What the Leaker did with the items of one forced shape: accepted (a failure), rejected with no
/// error of the shape's reason (a failure), rejected with it.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
struct ForcedCount {
    accepted: usize,
    other: usize,
    rejected: usize,
}

/// The shapes that the generator makes to reach a reason or a construct that the random types rarely
/// do, each with the tag that the item has when it writes it (`gen::TAG_*`), and what it is. The
/// summary counts the items with each, accepted and rejected.
const SHAPE_TAGS: &[(&str, &str)] = &[
    (
        gen::TAG_NESTED_SELF_BOUNDS,
        "nested Self in a bound, option off",
    ),
    (
        gen::TAG_NESTED_SELF_OPTION,
        "nested Self in a bound, option on",
    ),
    (gen::TAG_SELF_NO_MEANING, "Self in enter_generics"),
    (gen::TAG_SELF_IN_DEFAULT, "Self in a trait default"),
    (gen::TAG_SELF_IN_TRAIT_PRED, "Self in a trait predicate"),
    (gen::TAG_IMPL_HEADER_SELF, "Self in an impl header"),
    (gen::TAG_UNREADABLE, "unreadable value"),
    (gen::TAG_MACRO_PATH, "relative macro path"),
    (gen::TAG_ELIDED_ARG, "'_ in a path"),
    (gen::TAG_DISCRIMINANT, "enum discriminant"),
    (gen::TAG_QUALIFIED_PATH, "<T>::X path"),
    (gen::TAG_MACRO_CALL, "macro call"),
    (gen::TAG_WIDE_CONST, "wide const type"),
    (gen::FORCED_VALUE_PATH, "relative path in a value"),
];

/// A shape that the generator writes so that the Leaker must reject the item, under the settings
/// of every item (no rule of the analysis allows what the shape writes): the item is left out of the
/// acceptance floors, and must be rejected with an error that says why (`Expect`).
struct Forced {
    /// The tag of the item that writes it: one of `SHAPE_TAGS`, or a tag `gen::FORCED_*` that the
    /// text of the item gives where the tag of the shape is also on items that the Leaker accepts.
    tag: &'static str,
    /// The tag of `SHAPE_TAGS` that counts the shape.
    shape: &'static str,
    /// The errors of which the rejection must have one.
    expect: &'static [Expect],
    /// Only read when the Leaker reads the bodies of what it analyzes: not through `enter_trait`,
    /// nor in a module.
    in_body: bool,
    /// Only a run with `values` writes the shape, and the summary leaves its line out of a run
    /// without (`unreadable_value` is one too, but its line was always printed, with no item).
    values: bool,
}

/// An error that a rejection must have: its `Reason`, and, where they tell the shape from the item's
/// other errors, the kind of its cause, a text that its tokens (without whitespace) or its note
/// hold, whether the item is analyzed with `allow_self`, and a tag of the item. Where the Leaker has
/// another reason that takes the place of the shape's own, it is another `Expect` with its
/// condition.
#[derive(Clone, Copy)]
struct Expect {
    reason: &'static str,
    cause: Option<&'static str>,
    tokens: Option<&'static str>,
    note: Option<&'static str>,
    allow_self: Option<bool>,
    tag: Option<&'static str>,
}

impl Expect {
    const fn reason(reason: &'static str) -> Expect {
        Expect {
            reason,
            cause: None,
            tokens: None,
            note: None,
            allow_self: None,
            tag: None,
        }
    }

    const fn cause(self, cause: &'static str) -> Expect {
        Expect {
            cause: Some(cause),
            ..self
        }
    }

    const fn tokens(self, tokens: &'static str) -> Expect {
        Expect {
            tokens: Some(tokens),
            ..self
        }
    }

    const fn note(self, note: &'static str) -> Expect {
        Expect {
            note: Some(note),
            ..self
        }
    }

    const fn allow_self(self, allow_self: bool) -> Expect {
        Expect {
            allow_self: Some(allow_self),
            ..self
        }
    }

    const fn tag(self, tag: &'static str) -> Expect {
        Expect {
            tag: Some(tag),
            ..self
        }
    }

    /// Whether `error` of `item` is the one.
    fn holds(&self, item: &gen::Item, error: &Rejection) -> bool {
        error.reason == self.reason
            && self.cause.map_or(true, |cause| error.kind == cause)
            && self.tokens.map_or(true, |tokens| {
                error
                    .tokens
                    .replace(char::is_whitespace, "")
                    .contains(tokens)
            })
            && self.note.map_or(true, |note| error.note.contains(note))
            && self.allow_self.map_or(true, |a| a == item.allow_self)
            && self.tag.map_or(true, |tag| item.tags.contains(tag))
    }

    fn describe(&self) -> String {
        let mut text = format!("the reason {}", self.reason);
        if let Some(cause) = self.cause {
            write!(text, " with a cause of the kind {}", cause).unwrap();
        }
        if let Some(tokens) = self.tokens {
            write!(text, " whose tokens hold `{}`", tokens).unwrap();
        }
        if let Some(note) = self.note {
            write!(text, " whose note holds `{}`", note).unwrap();
        }
        if let Some(allow_self) = self.allow_self {
            write!(text, " (item with allow_self {})", allow_self).unwrap();
        }
        if let Some(tag) = self.tag {
            write!(text, " (item with the tag {})", tag).unwrap();
        }
        text
    }
}

/// The forced shapes. The classification of `SHAPE_TAGS` (measured over the seeds 1 to 16, 60 cases;
/// accepted/rejected of the items with the tag): forced, with no accepted item of the tag,
/// `nested_self_in_bounds` (0/319), `self_without_meaning` (0/211), `self_in_default` (0/56) and
/// `unreadable_value` (with `values`) and `forced:value_path` (with `values`, 0/1420: the shape is
/// the tag that the text of the item gives, `gen::forced_value_path()`, so it is on no item that
/// the Leaker accepts; the plain tag `value_path`, with the discriminants, `Vec::new()` where the
/// prelude is allowed, `enter_impl` and the impls of an item with such a body, is on 186 accepted
/// and 1596 rejected); forced in part, where the tag of the shape is also on items that are
/// accepted and the shape is the tag `gen::FORCED_*` that the text gives:
/// `self_in_trait_predicate` (741/565: a bound `T: AsRef<Self>` has it too), `impl_header_self`
/// (71/177: a self type that no leaked type of names the generic), `macro_path` (22/207: an impl
/// takes the tag of its type, `enter_impl` does not read signatures), `elided_arg` (44/314: a type of
/// another crate is written as it is) and `discriminant` (189/85: the constants of the Leaker's crate
/// by an absolute path are written as they are); not forced, `nested_self_in_bounds_option` (315/11:
/// the Leaker must accept it), `macro_call` (429/312: the Leaker's own macro by an absolute path is
/// written as it is), `qualified_path` (509/174: `<T>::Item` is a generic's) and `wide_const`
/// (6600/1947, no shape of its own that rejects).
const FORCED: &[Forced] = &[
    Forced {
        tag: gen::TAG_NESTED_SELF_BOUNDS,
        shape: gen::TAG_NESTED_SELF_BOUNDS,
        expect: &[Expect::reason("NestedSelfInBounds")],
        in_body: false,
        values: false,
    },
    Forced {
        tag: gen::TAG_SELF_NO_MEANING,
        shape: gen::TAG_SELF_NO_MEANING,
        expect: &[Expect::reason("SelfWithoutMeaning")],
        in_body: false,
        values: false,
    },
    Forced {
        tag: gen::TAG_SELF_IN_DEFAULT,
        shape: gen::TAG_SELF_IN_DEFAULT,
        expect: &[Expect::reason("SelfInDefault")],
        in_body: false,
        values: false,
    },
    // With a bound of the generics that names a leaked type, the predicate is a cycle first.
    Forced {
        tag: gen::FORCED_SELF_IN_TRAIT_PRED,
        shape: gen::TAG_SELF_IN_TRAIT_PRED,
        expect: &[
            Expect::reason("SelfInTraitPredicate"),
            Expect::reason("UnprovableBound")
                .tokens("<Self>")
                .tag(gen::TAG_BOUND_CYCLE),
        ],
        in_body: false,
        values: false,
    },
    // Without `allow_self` the `Self` of the bound cannot be leaked at all.
    Forced {
        tag: gen::FORCED_IMPL_HEADER_SELF,
        shape: gen::TAG_IMPL_HEADER_SELF,
        expect: &[
            Expect::reason("SelfInImplSelfType").allow_self(true),
            Expect::reason("DeniedSelfInBounds").allow_self(false),
        ],
        in_body: false,
        values: false,
    },
    Forced {
        tag: gen::TAG_UNREADABLE,
        shape: gen::TAG_UNREADABLE,
        expect: &[Expect::reason("Unreadable")],
        in_body: true,
        values: false,
    },
    Forced {
        tag: gen::FORCED_MACRO_PATH,
        shape: gen::TAG_MACRO_PATH,
        expect: &[Expect::reason("RelativePath").cause("Macro")],
        in_body: false,
        values: false,
    },
    Forced {
        tag: gen::FORCED_ELIDED_ARG,
        shape: gen::TAG_ELIDED_ARG,
        expect: &[Expect::reason("RelativePath").cause("Type").tokens("<'_>")],
        in_body: false,
        values: false,
    },
    Forced {
        tag: gen::FORCED_RELATIVE_DISCRIMINANT,
        shape: gen::TAG_DISCRIMINANT,
        expect: &[Expect::reason("RelativePath").cause("Expr")],
        in_body: false,
        values: false,
    },
    // A relative path, the root of a `use`, or a name of the prelude where the Leaker does not allow
    // it, in a body or a constant: it reads them with `check_values` (`values = true`), whatever the
    // scope but `enter_impl`, and a module as a whole. The note tells a value from the default of a
    // const parameter and a discriminant, which have the same reason and cause.
    Forced {
        tag: gen::FORCED_VALUE_PATH,
        shape: gen::FORCED_VALUE_PATH,
        expect: &[
            Expect::reason("RelativePath")
                .cause("Expr")
                .note("body of `"),
            Expect::reason("RelativePath")
                .cause("Expr")
                .note("value of `"),
        ],
        in_body: false,
        values: true,
    },
];

/// Whether the items with a shape of `SHAPE_TAGS` are forced: all of them, those that the tag
/// `gen::FORCED_*` of the shape marks, or none.
fn forced_class(shape: &str) -> String {
    let marks: Vec<&str> = FORCED
        .iter()
        .filter(|forced| forced.shape == shape)
        .map(|forced| forced.tag)
        .collect();
    match marks.first() {
        None => "not forced".to_string(),
        Some(tag) if *tag == shape => "forced".to_string(),
        Some(tag) => format!("forced in part ({})", tag),
    }
}

/// What a report says of an item with a forced shape.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Verdict {
    /// The Leaker accepted it.
    Accepted,
    /// It was rejected, but no error is one that the shape needs.
    OtherReason,
    /// It was rejected with one of the errors that the shape needs.
    Rejected,
}

impl Forced {
    fn verdict(&self, item: &gen::Item, report: &Report) -> Verdict {
        if report.accepted {
            Verdict::Accepted
        } else if report
            .rejections
            .iter()
            .any(|error| self.expect.iter().any(|expect| expect.holds(item, error)))
        {
            Verdict::Rejected
        } else {
            Verdict::OtherReason
        }
    }

    /// What the shape is, as `SHAPE_TAGS` says.
    fn what(&self) -> &'static str {
        SHAPE_TAGS
            .iter()
            .find(|(tag, _)| *tag == self.shape)
            .map_or(self.shape, |(_, what)| what)
    }

    /// The errors that the rejection must have one of, for `item`, as a sentence.
    fn wanted(&self, item: &gen::Item) -> String {
        let wanted: Vec<String> = self
            .expect
            .iter()
            .filter(|expect| {
                expect.allow_self.map_or(true, |a| a == item.allow_self)
                    && expect.tag.map_or(true, |tag| item.tags.contains(tag))
            })
            .map(Expect::describe)
            .collect();
        format!("an error with {}", wanted.join(", or with "))
    }
}

/// The forced shapes that an item has, under its own settings. Through `enter_impl` only the self
/// type and the arguments of the trait are analyzed, so nothing of an item of that scope is forced.
fn forced_of(item: &gen::Item) -> Vec<&'static Forced> {
    if item.scope == "impl" {
        return Vec::new();
    }
    FORCED
        .iter()
        .filter(|forced| item.tags.contains(forced.tag))
        .filter(|forced| !(forced.in_body && (item.scope == "trait" || item.kind == "mod")))
        .collect()
}

/// The generator must make a forced shape in at least one item in `FORCED_SHARE` of the analyzed
/// ones, if the run has `FORCED_MIN_ANALYZED` of them or more (measured: about one in seven).
const FORCED_SHARE: usize = 25;
const FORCED_MIN_ANALYZED: usize = 100;

const FORCED_ACCEPTED: &str = "SHOULD BE REJECTED: type-leak accepts an item with a forced shape";
const FORCED_OTHER: &str = "SHOULD BE REJECTED FOR ITS SHAPE: type-leak rejects an item with a forced shape, with no error that the shape needs";

/// What is wrong with the way the Leaker treated the forced shapes of an item: it accepted the item,
/// or rejected it with no error that a shape needs. One entry each, with the class of the failure.
fn forced_problems(item: &gen::Item, report: &Report) -> Vec<(&'static str, String)> {
    forced_of(item)
        .into_iter()
        .filter_map(|shape| {
            let (class, what) = match shape.verdict(item, report) {
                Verdict::Rejected => return None,
                Verdict::Accepted => (FORCED_ACCEPTED, "accepted the item".to_string()),
                Verdict::OtherReason => (
                    FORCED_OTHER,
                    format!("rejected the item with:\n{}", describe(&report.rejections)),
                ),
            };
            Some((
                class,
                format!(
                    "the forced shape `{}` ({}) is in the item, and the Leaker must reject it with {}; it {}",
                    shape.tag,
                    shape.what(),
                    shape.wanted(item),
                    what
                ),
            ))
        })
        .collect()
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
                    for (tag, _) in SHAPE_TAGS {
                        if item.tags.contains(*tag) {
                            let count = stats.shapes.entry(tag).or_default();
                            if report.accepted {
                                count.0 += 1;
                            } else {
                                count.1 += 1;
                                let mut names: Vec<&'static str> = vec![kind_of(&report.kind)];
                                names.extend(
                                    SCOPES
                                        .iter()
                                        .filter(|(scope, _)| *scope == item.scope)
                                        .map(|(_, api)| *api),
                                );
                                if item.kind != "mod" {
                                    names.extend(CONSTRUCTS.iter().filter(|name| {
                                        item.tags.contains(&format!(
                                            "{}{}",
                                            gen::TAG_CONSTRUCT,
                                            name
                                        ))
                                    }));
                                }
                                for name in names {
                                    *stats.shape_rejected.entry((tag, name)).or_default() += 1;
                                }
                            }
                        }
                    }
                    for error in &report.rejections {
                        *stats
                            .error_kinds
                            .entry((error.reason.clone(), error.kind.clone()))
                            .or_default() += 1;
                        if let Some(family) = family(error) {
                            *stats.families.entry(family.name).or_default() += 1;
                            if family.name.starts_with("a bare trait") {
                                stats.bare_rejected += 1;
                            }
                        }
                    }
                    let kind = kind_of(&report.kind);
                    // The items with a forced shape are rejected by construction: the floors do not
                    // count them, and each is checked for its shape (`forced_checks()`).
                    let forced = forced_of(item);
                    let counted = forced.is_empty();
                    stats.left_out += usize::from(!counted);
                    for shape in &forced {
                        let count = stats.forced.entry(shape.tag).or_default();
                        match shape.verdict(item, report) {
                            Verdict::Accepted => count.accepted += 1,
                            Verdict::OtherReason => count.other += 1,
                            Verdict::Rejected => count.rejected += 1,
                        }
                    }
                    let mut names: Vec<&'static str> = vec![kind];
                    names.extend(
                        SCOPES
                            .iter()
                            .filter(|(scope, _)| *scope == item.scope)
                            .map(|(_, api)| *api),
                    );
                    if item.kind != "mod" {
                        names.extend(CONSTRUCTS.iter().filter(|name| {
                            item.tags
                                .contains(&format!("{}{}", gen::TAG_CONSTRUCT, name))
                        }));
                    }
                    for name in names {
                        let all = stats.all.entry(name).or_default();
                        if report.accepted {
                            all.0 += 1;
                        } else {
                            all.1 += 1;
                        }
                    }
                    if counted && item.kind != "mod" {
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
                    if let Some((_, api)) = SCOPES
                        .iter()
                        .find(|(scope, _)| *scope == item.scope)
                        .filter(|_| counted)
                    {
                        let counts = if report.accepted {
                            &mut stats.scope_accepted
                        } else {
                            *stats.reasons.entry((api, reason_head(report))).or_default() += 1;
                            &mut stats.scope_rejected
                        };
                        *counts.entry(api).or_default() += 1;
                    }
                    if report.accepted {
                        if counted {
                            *stats.accepted.entry(kind).or_default() += 1;
                        }
                        stats.positions += report.positions;
                        stats.checked += report.checked;
                        stats.entries += report.entries;
                    } else if counted {
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

    /// The share accepted of all the items of a kind, scope or construct, the forced ones too.
    fn all_rate(&self, name: &str) -> String {
        match self.all.get(name) {
            Some((a, r)) if a + r > 0 => format!("{:.0}%", 100.0 * *a as f64 / (a + r) as f64),
            _ => "-".into(),
        }
    }

    /// The shapes counted (the items that have a forced shape, with each of its shapes: an item can
    /// have several).
    fn forced_seen(&self) -> usize {
        self.forced
            .values()
            .map(|c| c.accepted + c.other + c.rejected)
            .sum()
    }

    /// Why the check of the forced shapes may have checked (almost) nothing, one line each: the
    /// generator must make them in a share of the items (`FORCED_SHARE`), and a run that has too few
    /// items for a share is not asked. The items that the Leaker accepted, or rejected for another
    /// reason, are failures of their own (`forced_problems()`).
    fn forced_check_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.analyzed >= FORCED_MIN_ANALYZED && self.left_out * FORCED_SHARE < self.analyzed {
            problems.push(format!(
                "only {} of {} analyzed items have a forced shape, fewer than one in {}: the check that \
                 the Leaker rejects them for their reason tests (almost) nothing",
                self.left_out, self.analyzed, FORCED_SHARE
            ));
        }
        problems
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
/// accept, in percent. The items that have a forced shape (`FORCED`, which the run checks apart) are
/// left out: 7.2% of the items analyzed (5.5% to 8.7% in a seed). The table of the summary has the
/// share over all the items too (`all%`).
///
/// A test that counts a rejection as a success passes when the Leaker rejects everything, or when
/// the generator makes shapes that cannot be leaked (before the floors, 93% of the modules and
/// 60% of the traits were rejected, and a macro that failed every `intern_*` still printed `ok`).
/// These floors fail the run then.
///
/// Measured over the seeds 1 to 16 at 60 cases (2026-11-02, with the fixes of F65 to F67), the share
/// accepted of the items without a forced shape was, lowest over the seeds / pooled (and the fewest
/// items of a seed); in brackets the lowest before wave 3A, and the lowest over all the items:
///
/// - kinds: struct 94.8/96.1 (293) [94.1; 85.9], enum 90.3/95.7 (145) [93.5; 80.9], union 90.8/96.3
///   (69) [90.4; 84.4], trait 73.4/77.9 (312) [72.7; 66.3], inherent impl 80.1/87.3 (115) [81.4;
///   72.7], trait impl 58.6/69.1 (54) [63.0; 55.4], module 71.6/79.8 (81) [68.9; 71.6];
/// - scopes: `enter_mod` 89.4/93.8 (124) [89.9; 75.0], `enter_generics` 87.8/93.8 (33) [90.5; 61.0],
///   `enter_trait` 70.7/81.6 (27) [67.9; 64.4], `enter_impl` 100/100 (34) [100; 100];
/// - constructs (`gen::CONSTRUCTS`, the items that have it, not the modules): gat 47.5/58.0 (46)
///   [50.8; 43.3], const_param 77.5/80.6 (442) [76.5; 72.0], impl_trait_arg 61.3/67.4 (160) [58.9;
///   54.3], param_default 85.6/88.0 (165) [82.2; 79.8], binder 75.3/79.0 (222) [73.4; 67.0],
///   dyn_object 79.3/83.0 (394) [78.7; 73.3], fn_pointer 77.0/81.6 (327) [78.5; 70.9], assoc_const
///   63.6/72.4 (183) [69.6; 57.4], lifetime_param 81.1/85.0 (597) [80.7; 75.9], unsized_param
///   77.6/81.3 (275) [76.7; 71.6], where_clause 77.6/81.4 (651) [78.1; 70.7].
///
/// A floor is 10 points below the lowest, rounded down (`enter_trait` would be 60 and the module 61,
/// and they stay at 50 and 58: they have the fewest items, whose rate varies most). They are within
/// 3 points of the floors before the shapes, but `assoc_const` (59 -> 53). What the Leaker still
/// rejects there is not forced: bounds that name a leaked type, `Self::A`, the bare traits of the
/// prelude. The lowest of 16 seeds is a noisy number at 50 items a seed (a spread of 6 points):
/// with `strict_anchors` off, the shape of wave 3A that the Leaker rejects in some settings, the
/// pooled rates are the same (trait impl 69.3, assoc_const 73.5) and the lowest are 63.0 and 67.6.
///
/// Eight floors of this table are above what the opt-in run of `values` is measured to reach (struct,
/// enum, union, `enter_generics`, const_param, param_default, unsized_param, module), so that run has
/// `VALUES_FLOORS`, which are the lower of the two, measured the same way.
/// A construct is checked when a run has three times the `MIN_SAMPLES` of it.
/// `TYPE_LEAK_PROP_MIN_ACCEPT` overrides them.
const FLOORS: &[(&str, f64)] = &[
    ("struct", 84.0),
    ("enum", 80.0),
    ("union", 80.0),
    ("trait", 63.0),
    ("impl-inherent", 70.0),
    ("impl-trait", 48.0),
    ("mod", 58.0),
    // The items analyzed through a scope (`gen::SCOPES`), by the name of its API.
    ("enter_mod", 79.0),
    ("enter_generics", 77.0),
    ("enter_trait", 50.0),
    ("enter_impl", 90.0),
    // The items that have a construct (`gen::CONSTRUCTS`).
    ("gat", 37.0),
    ("const_param", 67.0),
    ("impl_trait_arg", 51.0),
    ("param_default", 75.0),
    ("binder", 65.0),
    ("dyn_object", 69.0),
    ("fn_pointer", 67.0),
    ("assoc_const", 53.0),
    ("lifetime_param", 71.0),
    ("unsized_param", 67.0),
    ("where_clause", 67.0),
];

/// The floors of the run with `values` (`TYPE_LEAK_PROP_FEATURES=values`), 10 points below the lowest
/// over the seeds 1 to 16 at 60 cases, rounded down, or the floor of `FLOORS` if that is lower. The
/// items that have a forced shape are left out as in `FLOORS`: 13.7% of the items analyzed (11.7% to
/// 15.2% in a seed), 6.8% of them for `forced:value_path`, a relative path, the root of a `use` or a
/// name of the prelude that the Leaker does not allow, in a body or a constant (all rejected, as the
/// run checks), and the rest for the other shapes, with the glob imports and the `macro_rules!` of
/// `unreadable_value`. Measured on 2026-11-03, the lowest accepted share / pooled (and the fewest
/// items of a seed): struct 90.6/92.5 (305), enum 88.0/93.0 (152), union 87.3/93.5 (66), trait
/// 73.7/76.8 (267), inherent impl 86.3/89.4 (86), trait impl 67.3/73.8 (55), module 63.6/78.3 (68),
/// `enter_mod` 90.8/93.4 (125), `enter_generics` 85.4/91.0 (29), `enter_trait` 68.4/79.2 (23),
/// `enter_impl` 100/100 (29), gat 50.0/59.8 (36), const_param 73.8/77.0 (381), impl_trait_arg
/// 63.1/67.6 (121), param_default 76.9/79.1 (225), binder 75.6/80.1 (230), dyn_object 81.3/83.9
/// (462), fn_pointer 82.3/84.1 (403), assoc_const 67.2/72.1 (162), lifetime_param 82.8/84.2 (534),
/// unsized_param 76.0/81.0 (232), where_clause 79.5/81.2 (552). With the value paths counted, the
/// lowest were trait 64.5, inherent impl 76.1, trait impl 57.8, module 54.4 (its floor was 49), gat
/// 42.0 and assoc_const 56.8. The run with `relative_std` is above `FLOORS` everywhere (the lowest:
/// trait 73.7, gat 50.0, assoc_const 68.1, impl_trait_arg 58.4), and needs none of its own.
const VALUES_FLOORS: &[(&str, f64)] = &[
    ("struct", 80.0),
    ("enum", 77.0),
    ("union", 77.0),
    ("trait", 63.0),
    ("impl-inherent", 70.0),
    ("impl-trait", 48.0),
    ("mod", 53.0),
    ("enter_mod", 79.0),
    ("enter_generics", 75.0),
    ("enter_trait", 50.0),
    ("enter_impl", 90.0),
    ("gat", 37.0),
    ("const_param", 63.0),
    ("impl_trait_arg", 51.0),
    ("param_default", 66.0),
    ("binder", 65.0),
    ("dyn_object", 69.0),
    ("fn_pointer", 67.0),
    ("assoc_const", 53.0),
    ("lifetime_param", 71.0),
    ("unsized_param", 65.0),
    ("where_clause", 67.0),
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
/// 90.6% to 91.1% over the seeds 1 to 16 at 60 cases (93.5% to 93.8% before the types with an
/// elided lifetime and a macro in an array length, which have no witness).
const MIN_CHECKED: f64 = 85.0;

/// The least number of times, over a run with at least 100 items invoked with `inst`, that an
/// instantiation passes an argument that holds a lifetime to a signature that relies on elision
/// (`use_summary()`, `gen::ELIDED_PARAM_PERCENT`): the shape of F69. Without it a generator that
/// stopped making the shape would leave the check vacuous. Measured at 60 cases: 23 to 49 over the
/// seeds 1 to 16.
const MIN_ELIDED_LIFETIME_ARGS: usize = 1;

/// The same for the instantiations that pass the argument to a signature that has an entry in its
/// inputs, which the copy replaced (F70). Measured at 60 cases: 2 to 19 over the seeds 1 to 16.
const MIN_ELIDED_ENTRY_LIFETIME_ARGS: usize = 1;

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
        Floors::parse_for(spec, min_samples, min_checked, false)
    }

    /// [`Floors::parse()`] for a run that has `values` or not: with it the floors are the lower of
    /// `FLOORS` and `VALUES_FLOORS`.
    fn parse_for(
        spec: &str,
        min_samples: usize,
        min_checked: f64,
        values: bool,
    ) -> Result<Floors, String> {
        let mut by_kind: BTreeMap<&'static str, f64> = FLOORS.iter().copied().collect();
        if values {
            for (kind, floor) in VALUES_FLOORS {
                let ours = by_kind.get_mut(kind).expect("a floor of FLOORS");
                *ours = ours.min(*floor);
            }
        }
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
        Floors::parse_for(
            &spec,
            env_or("TYPE_LEAK_PROP_MIN_SAMPLES", MIN_SAMPLES),
            env_or("TYPE_LEAK_PROP_MIN_CHECKED", MIN_CHECKED),
            gen::Flags::from_env().values,
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

/// The errors that the Leaker carries to the crate `tlp_e_NNN` (`gen::Case::errors_referrer`,
/// `prop.rs: carry()`), where the macro reads them back from tokens, checks that they are the Leaker's,
/// lets `LeakerResult::allow()` drop some by the number of its invocation, and reports the others with
/// `ErrorItem::emit()`: the crate does not compile, and its diagnostics are what remains.
///
/// - each diagnostic is one of the Leaker's errors of the case (its first line is the message, or the
///   start of it: the report writes a message on one line);
/// - the macros read as many errors as the Leaker's reports have, and reported as many as the crate has
///   diagnostics.
///
/// The macro fails by itself (`CARRY:`) if the errors differ after the round trip, if `allow()` does
/// not drop what it should, or if `emit()` and `to_compile_error()` differ: it is then a diagnostic
/// that is none of the Leaker's.
fn carry_failures(
    build: &Build,
    cases: &[(usize, &Case)],
    reports: &BTreeMap<(usize, usize), Report>,
    skipped: &[usize],
) -> Vec<Failure> {
    let clean = |text: &str| text.replace(['\n', '|', '\u{1f}', '\u{1d}'], " ");
    let mut failures = Vec::new();
    let mut diagnostics = 0;
    let mut leaker_errors = 0;
    for &(i, _) in cases.iter().filter(|(i, _)| !skipped.contains(i)) {
        let messages: Vec<String> = reports
            .iter()
            .filter(|((case, _), _)| *case == i)
            .flat_map(|(_, report)| report.rejections.iter().map(|e| clean(&e.message)))
            .collect();
        leaker_errors += messages.len();
        for diag in build.diags.iter().filter(|d| d.package == errors_name(i)) {
            diagnostics += 1;
            let first = diag.rendered.lines().next().unwrap_or("");
            let first = clean(first.strip_prefix("error: ").unwrap_or(first));
            if !messages.iter().any(|m| m.starts_with(&first)) {
                failures.push(Failure {
                    stage: None,
                    code: diag.code.clone(),
                    case: i,
                    item: None,
                    class: "CARRY: a diagnostic of the crate of the carried errors is none of the Leaker's errors",
                    text: diag.rendered.clone(),
                });
            }
        }
    }
    let sum = |what: &str| -> usize {
        build
            .counts
            .iter()
            .filter(|c| c.what == what)
            .map(|c| c.n)
            .sum()
    };
    let (read, reported) = (sum("carry_errors"), sum("carry_reported"));
    let mismatch = if read != leaker_errors {
        Some(format!(
            "the macros read {} carried errors, the Leaker's reports have {}",
            read, leaker_errors
        ))
    } else if reported != diagnostics {
        Some(format!(
            "the macros reported {} errors, the crates have {} diagnostics",
            reported, diagnostics
        ))
    } else {
        None
    };
    if let Some(text) = mismatch {
        failures.push(Failure {
            stage: None,
            code: String::new(),
            case: cases.first().map_or(0, |(i, _)| *i),
            item: None,
            class: "CARRY: the errors that the Leaker carried and the ones the crate reported are not those counted",
            text,
        });
    }
    failures
}

/// The copies that the Referrer's macros made at an instantiation of the generics of the item (`inst`)
/// and inside a module and a block that declare names (`shadow`), as they counted them (`Count`): the
/// items invoked with each, the items where the check ran, and what it compared (positions at the
/// arguments, entries hidden by a declaration). The problems are the checks that ran nowhere although
/// many items asked for them: a macro that does nothing would pass.
fn use_summary(cases: &[(usize, &Case)], build: &Build) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut problems = Vec::new();
    for (what, flag, noun) in [
        ("inst", ", inst(", "positions compared at the arguments"),
        (
            "inst_elided",
            ", inst(",
            "lifetime arguments passed to a signature that relies on elision",
        ),
        (
            "inst_elided_entry",
            ", inst(",
            "of those, passed to one with an entry in its inputs that the copy replaced (F70)",
        ),
        ("shadow", ", shadow(", "entries hidden by a declaration"),
    ] {
        let asked: usize = cases
            .iter()
            .map(|(_, case)| case.referrer.matches(flag).count())
            .sum();
        let counts: Vec<&Count> = build.counts.iter().filter(|c| c.what == what).collect();
        let ran = counts.len();
        let total: usize = counts.iter().map(|c| c.n).sum();
        let with: usize = counts.iter().filter(|c| c.n > 0).count();
        writeln!(
            text,
            "  copies with `{}`: {} items invoked, the check ran for {}, with something to compare for {}; {}: {}",
            what, asked, ran, with, noun, total
        )
        .unwrap();
        // The shapes of F69 and F70 (below) depend on what the generator made: a run of one case may
        // have none, and a run of many is held to a least number.
        let made_on_purpose = matches!(what, "inst_elided" | "inst_elided_entry");
        if asked >= 10 && with == 0 && !made_on_purpose {
            problems.push(format!(
                "`{}` was asked of {} items and compared nothing: the macro does not do it",
                what, asked
            ));
        }
        // The shapes of F69 and F70 are made on purpose (`gen::ELIDED_PARAM_PERCENT`,
        // `ELIDED_ENTRY_PERCENT`): over many items, some instantiation must pass a lifetime into such
        // a signature, or the run does not test it.
        let least = match what {
            "inst_elided" => MIN_ELIDED_LIFETIME_ARGS,
            "inst_elided_entry" => MIN_ELIDED_ENTRY_LIFETIME_ARGS,
            _ => 0,
        };
        if least > 0 && asked >= 100 && total < least {
            problems.push(format!(
                "{} items were invoked with `inst` and only {} {} (at least {}): the generator or the macro misses the shape",
                asked, total, noun, least
            ));
        }
    }
    (text, problems)
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
        "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6} {:>6}",
        "kind", "generated", "accepted", "rejected", "accepted%", "all%", "floor"
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
            "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6} {:>6}",
            kind,
            g,
            a,
            r,
            rate,
            stats.all_rate(kind),
            floor
        )
        .unwrap();
    }
    writeln!(
        s,
        "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6} {:>6}",
        "construct", "analyzed", "accepted", "rejected", "accepted%", "all%", "floor"
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
            "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6} {:>6}",
            name,
            a + r,
            a,
            r,
            rate,
            stats.all_rate(name),
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
    for ((reason, kind), n) in &stats.error_kinds {
        writeln!(
            s,
            "  rejection errors by Reason and kind of cause: {:>4}x {} ({})",
            n, reason, kind
        )
        .unwrap();
    }
    for (tag, what) in SHAPE_TAGS {
        if let Some((accepted, rejected)) = stats.shapes.get(tag) {
            writeln!(
                s,
                "  shape {:<34} {:>4} analyzed, {:>4} accepted, {:>4} rejected; {}",
                what,
                accepted + rejected,
                accepted,
                rejected,
                forced_class(tag)
            )
            .unwrap();
            let by: Vec<String> = stats
                .shape_rejected
                .iter()
                .filter(|((t, _), _)| t == tag)
                .map(|((_, name), n)| format!("{} {}", name, n))
                .collect();
            if !by.is_empty() {
                writeln!(s, "  shape rejections {:<26} {}", what, by.join(", ")).unwrap();
            }
        }
    }
    writeln!(
        s,
        "  items left out of the acceptance floors (they have a forced shape): {} of {} analyzed, with {} forced shapes",
        stats.left_out,
        stats.analyzed,
        stats.forced_seen()
    )
    .unwrap();
    for shape in FORCED.iter().filter(|shape| flags.values || !shape.values) {
        let count = stats.forced.get(shape.tag).copied().unwrap_or_default();
        writeln!(
            s,
            "  forced shape {:<34} {:>4} items, {:>4} rejected for its reason, {:>4} rejected for another, {:>4} accepted",
            shape.what(),
            count.accepted + count.other + count.rejected,
            count.rejected,
            count.other,
            count.accepted
        )
        .unwrap();
    }
    let allowed: usize = stats.allowed.values().sum();
    if allowed > 0 {
        let by_kind: Vec<String> = stats
            .allowed
            .iter()
            .map(|(kind, n)| format!("{} {}", n, kind))
            .collect();
        writeln!(
            s,
            "  accept mode (TYPE_LEAK_PROP_ACCEPT): {} items rejected at the first analysis were accepted by allow() of the causes of their errors ({}); entries: {}; type positions: {}, witnesses checked: {}",
            allowed,
            by_kind.join(", "),
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
/// The message of the panic of the macro when an error remains after `allow()` of the cause of every
/// error (`allow_all()` in `tests/crates/macro/prop.rs`).
const ALLOW_LEFT: &str = "ALLOW LEFT ERRORS";
const ACCEPT_FAILED: &str = "ACCEPT FAILED: an error remains when the cause of every error is allowed (the macro panicked, so the Leaker's crate does not build)";

/// The class of a diagnostic of a Leaker's crate that does not compile with the macros: `class`,
/// unless the macro panicked because `allow()` of the cause of every error left an error.
fn leaker_diag_class(class: &'static str, text: &str) -> &'static str {
    if text.contains(ALLOW_LEFT) {
        ACCEPT_FAILED
    } else {
        class
    }
}

/// The text of that diagnostic, led by the message of the panic if it is one of `ALLOW_LEFT`.
fn leaker_diag_text(rendered: &str) -> String {
    match rendered.lines().find(|line| line.contains(ALLOW_LEFT)) {
        Some(line) => format!(
            "{}\n\n{}",
            line.trim().trim_start_matches("= help: message: "),
            rendered
        ),
        None => rendered.to_string(),
    }
}

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
            failures.push(Failure::new(
                i,
                None,
                "WATCHDOG: a crate of the case was not built, and no error says why",
                format!(
                    "{}\n\ncargo's stderr:\n{}",
                    missing.join("\n"),
                    excerpt(&build.stderr, 40)
                ),
            ));
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
                failures.push(Failure::new(
                    i,
                    item,
                    leaker_diag_class(class, &diag.rendered),
                    leaker_diag_text(&diag.rendered),
                ));
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
            failures.push(Failure::new(i, None, "WATCHDOG: no executable was built for the Leaker's crate, so the macros reported nothing", format!("cargo's stderr:\n{}", excerpt(&build.stderr, 40))));
            continue;
        };
        let output = Command::new(exe).output().unwrap();
        if !output.status.success() {
            failures.push(Failure::new(
                i,
                None,
                "WATCHDOG: the binary that prints the reports fails",
                String::from_utf8_lossy(&output.stderr).into_owned(),
            ));
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
                    None => failures.push(Failure::new(
                        i,
                        None,
                        "WATCHDOG: a line of allow_stages cannot be read",
                        line.to_string(),
                    )),
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
                                failures.push(Failure::new(i, Some(id), "STAGES: the macro cannot make the copy made after the analysis without the rule", format!("rule {}: {}", rule, error)));
                            }
                        }
                        run.reports[side].insert((i, id), report);
                    }
                    None => failures.push(Failure::new(
                        i,
                        None,
                        "WATCHDOG: a line of the copy of allow_stages cannot be read",
                        line.to_string(),
                    )),
                }
                continue;
            }
            match parse_report(line) {
                Some((id, report)) => {
                    if let Some(error) = &report.rewrite_error {
                        failures.push(Failure::new(
                            i,
                            Some(id),
                            if report.allowed {
                                "ACCEPT: the rewrite of an item accepted by allow() fails"
                            } else {
                                "the rewrite of an accepted item fails"
                            },
                            error.clone(),
                        ));
                    }
                    reports.insert((i, id), report);
                }
                None => failures.push(Failure::new(
                    i,
                    None,
                    "WATCHDOG: a report line cannot be read",
                    line.to_string(),
                )),
            }
        }
        let missing: Vec<String> = case
            .items
            .iter()
            .filter(|item| !reports.contains_key(&(i, item.id)))
            .map(|item| item.id.to_string())
            .collect();
        if !missing.is_empty() {
            failures.push(Failure::new(
                i,
                None,
                "WATCHDOG: the macros of some items reported nothing",
                format!("items {}", missing.join(", ")),
            ));
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
                failures.push(Failure::new(i, None, "WATCHDOG: the macros of some items reported nothing of the copy of allow_stages", format!("rule {}: items {}", rule, missing.join(", "))));
            }
        }
    }
    // The names of the prelude mean the prelude in the Referrer's crate of the stage copies: that is
    // what allowing them is a promise of. Nothing generated declares one.
    if stages_on {
        for &(i, case) in &selected {
            let shadowed = shadowed_prelude_names(&case.leaker);
            if !shadowed.is_empty() {
                failures.push(Failure::new(i, None, "STAGES: the Leaker's crate declares a name of the prelude, which the stage copies must not shadow", format!("shadowed: {}", shadowed.join(", "))));
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

    // The documented failures of the copies of accepted items make the build fail too, and so do the
    // errors that the crates of the carried errors report (`carry_failures()` checks them).
    let carried_only = build.diags.iter().all(|d| d.package.starts_with("tlp_e_"));
    if !build.success
        && !carried_only
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
        let item = generated[*i]
            .items
            .iter()
            .find(|item| item.id == *id)
            .unwrap();
        // A forced shape is rejected, for its reason.
        for (class, text) in forced_problems(item, report) {
            failures.push(Failure {
                stage: None,
                code: String::new(),
                case: *i,
                item: Some(*id),
                class,
                text,
            });
        }
        if report.accepted {
            continue;
        }
        let mut tags = item.tags.clone();
        if item.kind == "mod" || item.scope == "mod" {
            tags.insert(gen::TAG_MODULE_SCOPE.to_string());
        }
        for problem in unexplained(&tags, report) {
            failures.push(Failure::new(*i, Some(*id), "UNEXPLAINED REJECTION: type-leak rejects an item for a reason the generator does not explain", problem));
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
    let mut skipped: Vec<usize> = leaker_failed.clone();
    skipped.extend(&unbuilt);
    failures.extend(carry_failures(&build, &selected, &reports, &skipped));
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
        summary.push_str(&Stages::summarize(rule, &run.stages, &copies));
    }
    eprintln!("{}", summary);
    let mut floor_problems = floors.problems(&stats);
    floor_problems.extend(stats.forced_check_problems());
    let (use_text, use_problems) = use_summary(&selected, &build);
    let use_text = format!(
        "{}  errors carried to the crate that reports them: {} read, {} reported (the rest dropped by `allow()`)\n",
        use_text,
        build.counts.iter().filter(|c| c.what == "carry_errors").map(|c| c.n).sum::<usize>(),
        build.counts.iter().filter(|c| c.what == "carry_reported").map(|c| c.n).sum::<usize>(),
    );
    eprintln!("{}", use_text);
    floor_problems.extend(use_problems);

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
            counts: Vec::new(),
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
        assert_eq!(missing_artifacts(&b, 3, false, &[]).len(), 2);
        b.executables.insert(leaker_name(3), PathBuf::from("/x"));
        assert_eq!(missing_artifacts(&b, 3, false, &[]).len(), 1);
        b.artifacts.insert(referrer_name(3));
        assert!(missing_artifacts(&b, 3, false, &[]).is_empty());
        // The second Referrer's crate of the accept mode is built too.
        assert_eq!(missing_artifacts(&b, 3, true, &[]).len(), 1);
        b.artifacts.insert(accept_name(3));
        assert!(missing_artifacts(&b, 3, true, &[]).is_empty());
        // And those of `allow_stages`: each rule has four, glob and clean after the result and the
        // config.
        let case = generate_with(3, 3, 6, Features::parse(""), Flags::default());
        let stages = &case.stages;
        assert_eq!(stages.len(), 4 * Flags::default().rules().len());
        assert_eq!(missing_artifacts(&b, 3, true, stages).len(), stages.len());
        for stage in stages {
            b.artifacts.insert(stage_name(3, stage));
        }
        assert!(missing_artifacts(&b, 3, true, stages).is_empty());
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
    fn the_run_with_values_has_floors_that_are_not_above_the_others() {
        let plain = Floors::parse("", MIN_SAMPLES, MIN_CHECKED).unwrap();
        let values = Floors::parse_for("", MIN_SAMPLES, MIN_CHECKED, true).unwrap();
        for (kind, floor) in &plain.by_kind {
            assert!(values.by_kind[kind] <= *floor, "{}", kind);
        }
        // Lower where the run with `values` reaches less. The traits and the impls reach as much now
        // that their value paths are forced and left out.
        assert!(values.by_kind["struct"] < plain.by_kind["struct"]);
        assert!(values.by_kind["param_default"] < plain.by_kind["param_default"]);
        // The same names, and the command line still overrides them.
        let names = |table: &'static [(&'static str, f64)]| {
            table.iter().map(|(k, _)| *k).collect::<Vec<&'static str>>()
        };
        assert_eq!(names(FLOORS), names(VALUES_FLOORS));
        let floors = Floors::parse_for("trait=70", MIN_SAMPLES, MIN_CHECKED, true).unwrap();
        assert_eq!(floors.by_kind["trait"], 70.0);
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
        let floor = FLOORS.iter().find(|(k, _)| *k == "gat").unwrap().1 as usize;
        stats.construct_accepted.insert("gat", floor);
        stats.construct_rejected.insert("gat", 100 - floor);
        assert!(defaults().problems(&stats).is_empty());
        stats.construct_rejected.insert("gat", 100 - floor + 1);
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
            // The items with a forced shape are left out of the floors.
            let items = items.filter(|item| forced_of(item).is_empty());
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
        // `NestedSelfInBounds` is explained by its own shape, `DeniedSelfInBounds` is not.
        assert!(!same("DeniedSelfInBounds", "NestedSelfInBounds"));
        assert_eq!(
            family(&rejection_of("NestedSelfInBounds", "", "Wrap<Self, T>")).map(|f| f.tags),
            Some(vec![
                gen::TAG_NESTED_SELF_BOUNDS.to_string(),
                gen::TAG_NESTED_SELF_TEXT.to_string()
            ])
        );
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
        // A message that looks like a known reason does not make another reason one: the `Reason`
        // decides.
        assert_eq!(
            by_reason(
                "SelfWithoutMeaning",
                "`Wrap<T>` is in a bound or where predicate, and a bound of its generics names a leaked type",
                "Wrap < T >"
            ),
            Some(vec![gen::TAG_SELF_NO_MEANING.to_string()])
        );
        // The reasons that nothing is generated for (`UnnameableSelf` needs an item nested in a scope
        // that has a `Self` of its own and generics that bound it: no valid item is that, only a
        // macro that nests scopes), and a reason nobody listed.
        for reason in ["UnnameableSelf", "", "Strange"] {
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
            "UnnameableSelf",
            "`Self` has no name here",
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
            rejection_of("UnnameableSelf", "`Self` has no name here", "Self"),
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

    /// An `UnprovableBound` as the Leaker reports it for an entry whose context keeps a bound that
    /// names its generic through a path that may be an alias of a projection: the type in the
    /// predicate, and in the note the generic and the bound.
    fn hidden(tokens: &str, generic: &str, bound: &str) -> Rejection {
        Rejection {
            message: format!(
                "`{}` is in a bound or where predicate, and a bound of its generics names `{}` in the arguments of `{}`, which may hide a projection",
                tokens, generic, bound
            ),
            tokens: tokens.to_string(),
            note: format!(
                "`Self` is the self type of the impl, which the copy writes with its leaked types as paths through the Repeater, and the bound `{g} : 'a + Iterator<Item = u8> + {b}` of their generics names `{g}` in the arguments of the bound `{b}`, whose path may be a type alias of a projection that another bound of `{g}` equates, which the Leaker cannot tell from a struct: rustc normalizes it in the impl of the Repeater, but not yet in the environment (E0277)",
                g = generic,
                b = bound
            ),
            reason: "UnprovableBound".to_string(),
            kind: "Type".to_string(),
        }
    }

    /// Seed 22 (case 26, item 14; case 41, item 3): `Self` or `Self::A` in a predicate, and a bound
    /// of a generic with an equation of an associated type that names the generic in the arguments
    /// of a path (`BIt<U>`, `B2<T, B0>`). The Leaker cannot tell an alias from a struct, so it
    /// rejects both. The item must have written the generic and the bound, with the entry.
    #[test]
    fn a_bound_that_names_its_generic_in_a_path_is_explained_by_the_bound_the_item_wrote() {
        let in_impl = "#[@ATTR@(id = 14, scope = \"impl\")]\n\
            impl<'a, T, U: ::core::default::Default + 'a + ::core::iter::Iterator<Item = u8>, V> S12_1<'a, T, U, V> \
            where V: ::core::fmt::Debug, U: Fn(::tlp_l_026::base::BIt<U>) -> u8, crate::base::B1<U>: Default {\n\
            pub fn m14_3(&self) where V: ::core::ops::Fn(&Self) -> u8, R1<V>: ::core::default::Default { loop {} }\n}";
        let bit = "Fn(::tlp_l_026::base::BIt<U>) -> u8";
        let payload = |generic: &str, bound: &str| {
            format!("{}{}|{}", gen::TAG_HIDDEN_AT, generic, gen::norm(bound))
        };
        // The settings of the seed's case: the crate's own paths are allowed.
        let own_crate = |krate: &str| gen::Config {
            krate: krate.to_string(),
            allow_crate: true,
            ..strict_config()
        };
        let item_tags = tags_of_text(own_crate("tlp_l_026"), in_impl);
        let hidden_tags: Vec<&String> = item_tags
            .iter()
            .filter(|tag| tag.starts_with(gen::TAG_HIDDEN_AT))
            .collect();
        assert_eq!(hidden_tags, vec![&payload("U", bit)]);
        let error = hidden("Self", "U", "Fn(::tlp_l_026::base::BIt<U>) -> u8");
        assert!(unexplained(&item_tags, &rejected(vec![error.clone()])).is_empty());
        // The Leaker prints the tokens with spaces.
        let spaced = hidden("Self", "U", "Fn(:: tlp_l_026 :: base :: BIt < U >) -> u8");
        assert!(unexplained(&item_tags, &rejected(vec![spaced])).is_empty());

        // The trait of the other seed: an equation in a where predicate of the trait, the entry
        // `Self::A1` in a predicate of a method.
        let in_trait = "pub trait Tr3_1<'a, 'b, 'c, T: ::core::fmt::Debug + 'b> where T: \
            ::core::ops::Fn(&'b ::tlp_l_041::base::B2<T, ::tlp_l_041::base::B0>) -> u8, T: \
            Iterator<Item = ::core::cell::Cell<(::tlp_l_041::base::B0, usize)>> {\n\
            type A1: ::core::fmt::Debug;\n\
            fn m3_3<'x, 'y: 'x>(&self, a0: &'b Self::A0) where T: ::core::convert::Into<Self::A1>;\n}";
        let b2 = "::core::ops :: Fn(&'b::tlp_l_041::base::B2<T, ::tlp_l_041::base::B0>) -> u8";
        let trait_tags = tags_of_text(own_crate("tlp_l_041"), in_trait);
        assert!(trait_tags.contains(&payload("T", b2)), "{:?}", trait_tags);
        let error = hidden("Self :: A1", "T", b2);
        assert!(unexplained(&trait_tags, &rejected(vec![error])).is_empty());

        // The generic or the bound that the note names is not the one the item wrote.
        for wrong in [
            hidden("Self", "V", bit),
            hidden("Self", "U", "Fn(::tlp_l_026::base::BIt<V>) -> u8"),
            hidden("Self", "U", "Fn(::tlp_l_026::base::B1<U>) -> u8"),
        ] {
            let problems = unexplained(&item_tags, &rejected(vec![wrong]));
            assert_eq!(problems.len(), 1, "{:?}", problems);
            assert!(
                problems[0].starts_with("NO SHAPE THAT GIVES IT"),
                "{:?}",
                problems
            );
        }
        // A tag for a cycle explains no such error, and the plain shapes are not the tag.
        let cycle_tags = tags(&[
            "bound_cycle",
            "self_in_predicate",
            &format!("bound_cycle=U|{}", gen::norm(bit)),
        ]);
        assert_eq!(
            unexplained(&cycle_tags, &rejected(vec![hidden("Self", "U", bit)])).len(),
            1
        );
        // A note that does not say it is a bound that hides a projection is a cycle: the tag does not
        // explain it.
        let mut cycle = hidden("Self", "U", bit);
        cycle.note = cycle.note.replace(HIDDEN_NOTE, "which names a leaked type");
        assert_eq!(unexplained(&item_tags, &rejected(vec![cycle])).len(), 1);
    }

    /// The tag needs what the Leaker needs: an equation of the generic, and an entry that keeps the
    /// bound in a predicate (`Self`, or the self type of the impl).
    #[test]
    fn a_hidden_bound_is_tagged_with_the_equation_and_the_entry_it_needs() {
        let hidden_tags = |text: &str| -> Vec<String> {
            gen::hidden_bounds_in(text)
                .into_iter()
                .map(|(generic, bound)| format!("{}|{}", generic, bound))
                .collect()
        };
        let item = |bounds: &str, predicate: &str| {
            format!(
                "pub trait Tr<U: {}> {{ fn m(&self) where {}; }}",
                bounds, predicate
            )
        };
        let wrote = vec!["U|Fn(BIt<U>)->u8".to_string()];
        // The equation and `Self` in a predicate.
        let both = item(
            "Iterator<Item = u8> + Fn(BIt<U>) -> u8",
            "U: AsRef<Self::A>",
        );
        assert_eq!(hidden_tags(&both), wrote);
        // Without the equation, or without the entry, the Leaker has nothing to reject.
        let none = item("Fn(BIt<U>) -> u8", "U: AsRef<Self::A>");
        assert!(hidden_tags(&none).is_empty());
        let none = item("Iterator<Item = u8> + Fn(BIt<U>) -> u8", "U: Clone");
        assert!(hidden_tags(&none).is_empty());
        // `Self: Sized` is no entry.
        let none = item("Iterator<Item = u8> + Fn(BIt<U>) -> u8", "Self: Sized");
        assert!(hidden_tags(&none).is_empty());
        // A trait of the crate fixes an associated type too (a supertrait may equate one), unless it
        // names the generic in its arguments; a trait of the standard library does not.
        let crate_trait = item("::k::base::BItem + Fn(BIt<U>) -> u8", "U: AsRef<Self::A>");
        assert_eq!(hidden_tags(&crate_trait), wrote);
        for fixes_nothing in ["::k::base::Tr<U>", "Clone", "::core::fmt::Debug", "'a"] {
            let none = item(
                &format!("{} + Fn(BIt<U>) -> u8", fixes_nothing),
                "U: AsRef<Self::A>",
            );
            assert!(hidden_tags(&none).is_empty(), "{}", fixes_nothing);
        }
        // The equation of an enclosing scope: a predicate of the method equates the generic.
        let outer = "pub trait Tr<U: Fn(BIt<U>) -> u8> { fn m(&self) where U: Iterator<Item = u8>, R1<Self>: Default; }";
        assert_eq!(hidden_tags(outer), wrote);
        // The self type of the impl, written in a predicate.
        let impl_self =
            "impl<U: Iterator<Item = u8> + Fn(BIt<U>) -> u8> R1<U> where R1<U>: Default { }";
        assert_eq!(hidden_tags(impl_self), wrote);
        let other =
            "impl<U: Iterator<Item = u8> + Fn(BIt<U>) -> u8> R1<U> where R2<U>: Default { }";
        assert!(hidden_tags(other).is_empty());
        // Another path with the name (`self::R1<U>`, `R10<U>`) is not the self type.
        let other = "impl<U: Iterator<Item = u8> + Fn(BIt<U>) -> u8> R1<U> where self::R1<U>: Default, R10<U>: Default { }";
        assert!(hidden_tags(other).is_empty());
        // Only a bound that names the generic in a path may hide a projection.
        for bound in [
            "Fn(&U) -> u8",
            "Fn((U, u8)) -> u8",
            "Fn(R0) -> u8",
            "From<U>",
            "Fn(::core::option::Option<U>) -> u8",
            "Fn(&dyn Tr<U>) -> u8",
            "Fn(fn(U)) -> u8",
            "Iterator<Item = BIt<U>>",
        ] {
            let none = item(
                &format!("Iterator<Item = u8> + {}", bound),
                "U: AsRef<Self::A>",
            );
            assert!(hidden_tags(&none).is_empty(), "{}", bound);
        }
        for bound in [
            "Fn(<U as Iterator>::Item) -> u8",
            "Fn(U::Item) -> u8",
            "Fn(W<::core::option::Option<U>>) -> u8",
            "for<'h> Fn(&'h W<U>) -> &'h u8",
            "PartialEq<W<U>>",
        ] {
            let found = item(
                &format!("Iterator<Item = u8> + {}", bound),
                "U: AsRef<Self::A>",
            );
            assert_eq!(hidden_tags(&found).len(), 1, "{}", bound);
        }
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

    /// The `>` of an arrow closes no bracket: a predicate on a type that holds a function pointer is a
    /// clause (its tag was missing, and the rejection it gives unexplained).
    #[test]
    fn a_predicate_with_an_arrow_in_brackets_is_a_clause() {
        let text = "pub trait Tr<U: Iterator> where ::core::marker::PhantomData<unsafe extern \"C\" fn() -> U::Item>: ::core::clone::Clone, U: Clone {\n}";
        let clauses = gen::bound_clauses(text);
        assert!(
            clauses.iter().any(|(bounded, bounds)| bounded
                .starts_with("::core::marker::PhantomData<unsafe")
                && bounds == "::core::clone::Clone"),
            "{:?}",
            clauses
        );
    }

    /// Every shape of `SHAPE_TAGS` is generated in a few cases (`unreadable_value` needs `values`), so
    /// the run reaches the reason or the cause it is for, and the text of what it writes is valid
    /// enough for the tag to be read back from it.
    #[test]
    fn the_shapes_of_the_rare_reasons_are_generated() {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for flags in [
            Flags::default(),
            Flags {
                values: true,
                ..Flags::default()
            },
        ] {
            for seed in 0..30 {
                let case = generate_with(seed, 0, 8, Features::parse(""), flags);
                for item in &case.items {
                    seen.extend(item.tags.iter().cloned());
                    // A tag that a text gives is read back from the text.
                    if item.tags.contains(gen::TAG_WIDE_CONST) {
                        assert!(item.source.contains("K") || item.source.contains("const "));
                    }
                }
            }
        }
        for (tag, what) in SHAPE_TAGS {
            assert!(
                seen.contains(*tag),
                "no item has the shape {} ({})",
                tag,
                what
            );
        }
        // And every forced shape is in some item, which the Leaker is then asked to reject.
        for forced in FORCED {
            assert!(
                seen.contains(forced.tag),
                "no item has the forced shape {}",
                forced.tag
            );
        }
    }

    /// A generated item with the tags and settings given, for the checks of the forced shapes.
    fn item_with(
        kind: &'static str,
        scope: &'static str,
        allow_self: bool,
        names: &[&str],
    ) -> gen::Item {
        gen::Item {
            id: 0,
            kind,
            module: Vec::new(),
            trust: gen::Trust::None,
            allow_self,
            scope,
            contents: vec![kind],
            tags: tags(names),
            source: "pub struct S;".to_string(),
        }
    }

    fn report_with(accepted: bool, errors: Vec<Rejection>) -> Report {
        Report {
            kind: "struct".into(),
            accepted,
            rejections: errors,
            ..Default::default()
        }
    }

    fn error_of(reason: &str, kind: &str, tokens: &str) -> Rejection {
        Rejection {
            reason: reason.to_string(),
            kind: kind.to_string(),
            tokens: tokens.to_string(),
            ..Default::default()
        }
    }

    /// The forced shapes that a text gives are read from the text of the item itself, and not from a
    /// shape that the Leaker may accept: a bound `T: AsRef<Self>`, an impl of the trait that has the
    /// predicate, a macro or a type of another crate by an absolute path.
    #[test]
    fn the_forced_shapes_are_read_from_the_text_of_the_item() {
        let forced = gen::forced_in;
        let trait_with = |generics: &str, clause: &str| {
            format!(
                "#[@ATTR@(id = 1)]\npub trait Tr1_1{} {} {{\nfn m(&self);\n}}\n",
                generics, clause
            )
        };
        assert_eq!(
            forced(&trait_with("<T>", "where R1<Self>: Default, T: Clone")),
            [gen::FORCED_SELF_IN_TRAIT_PRED]
        );
        assert_eq!(
            forced(&trait_with(
                "<T>",
                "where T: Clone, super::base::B1<Self>: Default"
            )),
            [gen::FORCED_SELF_IN_TRAIT_PRED]
        );
        for (generics, clause) in [
            ("<T: AsRef<Self>>", ""),
            ("<T>", "where T: AsRef<Self>"),
            ("<T>", "where Self: Sized, T: Clone"),
            ("<T>", "where ::std::vec::Vec<Self>: Default"),
            ("<T>", "where Self::A0: Clone"),
            ("<T>", "where <Self as Tr1_1<T>>::A0: Clone"),
            ("<T>", "where T: Into<R1<Self>>"),
        ] {
            assert_eq!(
                forced(&trait_with(generics, clause)),
                Vec::<&str>::new(),
                "{} {}",
                generics,
                clause
            );
        }
        // The impl of such a trait writes the predicate too, and is no shape of its own.
        assert!(
            forced("impl<T> ::k::m0::Tr1_1<T> for u8 where R1<Self>: Default {\n}\n").is_empty()
        );

        let impl_of = |text: &str| {
            format!(
                "#[@ATTR@(id = 2)]\n{} {{\npub fn m(&self) {{ loop {{}} }}\n}}\n",
                text
            )
        };
        for text in [
            "impl<T: AsRef<Self>> R1<T>",
            "impl<T: Clone + AsRef<Self>, U> self::R1<T, U>",
            "impl<T> super::base::B1<T> where T: AsRef<Self>",
            "impl<'a, T: Clone, U> ::k::m0::Tr<'a, U> for R2<U, T> where U: AsRef<Self>",
        ] {
            assert_eq!(
                forced(&impl_of(text)),
                [gen::FORCED_IMPL_HEADER_SELF],
                "{}",
                text
            );
        }
        for text in [
            "impl<T: Clone> R1<T>",
            "impl<T: AsRef<Self>> ::k::base::B1<T>",
            "impl<E: AsRef<Self>> ::k::m0::Tr for (::core::cell::Cell<bool>, E)",
            "impl<E: AsRef<Self>, T> R1<T>",
            "impl<T: AsRef<Self>> &'static unsafe fn(i32)",
        ] {
            assert!(forced(&impl_of(text)).is_empty(), "{}", text);
        }

        assert_eq!(
            forced("pub struct S1_1 { pub f0: [u8; two!()] }"),
            [gen::FORCED_MACRO_PATH]
        );
        assert!(forced("pub struct S1_1 { pub f0: [u8; ::k::two!()] }").is_empty());
        assert_eq!(forced("fn m(&self, ae: Rl<'_>);"), [gen::FORCED_ELIDED_ARG]);
        assert_eq!(
            forced("fn m(&self, ae: super::base::Bl<'_>);"),
            [gen::FORCED_ELIDED_ARG]
        );
        assert!(forced("fn m(&self, ae: ::k::base::Bl<'_>);").is_empty());
        assert!(forced("fn m(&self, a0: Rl<'a>, ae: ::k::base::Bl<'_>);").is_empty());
        assert!(forced("fn m(&self, a0: Rl<'_>);").is_empty());
        assert_eq!(
            forced("pub enum E1_1 { V0 = (RLEN as isize) + 0, V1 = (3) + 100 }"),
            [gen::FORCED_RELATIVE_DISCRIMINANT]
        );
        assert_eq!(
            forced("pub enum E1_1 { V0 = (super::super::base::BLEN as isize) + 0 }"),
            [gen::FORCED_RELATIVE_DISCRIMINANT]
        );
        assert!(forced(
            "pub enum E1_1 { V0 = (::k::base::BLEN as isize) + 0, V1 = (::core::mem::size_of::<*const R1>() as isize) + 100 }"
        )
        .is_empty());
    }

    /// A path of a value is a forced shape where the attribute of the item makes the Leaker read it
    /// and no rule of the analysis allows it: a relative path or the root of a `use` always, a name of
    /// the prelude where the attribute says `prelude = false`; not an absolute path, not through
    /// `enter_impl` (`scope = "impl"`), not without `values = true`, and no other forced shape.
    #[test]
    fn a_path_in_a_value_is_forced_where_the_leaker_must_reject_it() {
        let forced = gen::forced_in;
        let item = |settings: &str, body: &str| {
            format!(
                "#[@ATTR@(id = 1, allow_self = true, {}, krate = \"k\")]\npub trait Tr1_1 {{\nfn m(&self) {{ {} loop {{}} }}\n}}\n",
                settings, body
            )
        };
        let reads = "prelude = true, scope = \"\", values = true, bodies = true";
        let value = [gen::FORCED_VALUE_PATH];
        // A relative path in a `usize`, in each form that the generator writes.
        for path in [
            "crate::base::BLEN",
            "crate::base::bf()",
            "super::base::BLEN",
            "super::super::base::bf()",
            "RLEN",
            "self::RLEN",
            "ILEN",
            "crate::base::bg::<*const R1<u8>>()",
            "super::base::bg::<*const [R1; 3]>()",
        ] {
            let body = format!("let _d: usize = {} + 1;", path);
            assert_eq!(forced(&item(reads, &body)), value, "{}", path);
            let constant = format!(
                "const K0: usize = ::core::mem::size_of::<*const R1<u8>>() + {};",
                path
            );
            assert_eq!(forced(&constant), value, "{}", constant);
        }
        // The absolute path of the Leaker's crate is written as it is.
        for path in [
            "::k::base::BLEN",
            "::k::base::bf()",
            "::k::base::bg::<*const R1<u8>>()",
        ] {
            let body = format!("let _d: usize = {} + 1;", path);
            assert!(forced(&item(reads, &body)).is_empty(), "{}", path);
            let constant = format!(
                "pub const KV0: usize = ::core::mem::size_of::<*const R1>() + {};",
                path
            );
            assert!(forced(&constant).is_empty(), "{}", constant);
        }
        // The head of the path names the form.
        assert_eq!(
            gen::value_paths_in("let _d: usize = super::base::bf() + 1; let _e = 1;"),
            [gen::ValuePath::Relative("super".to_string())]
        );
        assert_eq!(
            gen::value_paths_in(
                "pub const KV0: usize = ::core::mem::size_of::<*const R1>() + RLEN;"
            ),
            [gen::ValuePath::Relative("RLEN".to_string())]
        );
        assert!(
            gen::value_paths_in("let _b = ::core::mem::size_of::<*const R1>(); let _c = 1u8;")
                .is_empty()
        );
        // A name of the prelude is relative where the Leaker does not allow the prelude.
        let vec = "let _e = Vec::<*const R1>::new();";
        assert_eq!(
            forced(&item(
                &reads.replace("prelude = true", "prelude = false"),
                vec
            )),
            value
        );
        assert!(forced(&item(reads, vec)).is_empty());
        assert!(forced(&format!("pub fn f() {{ {} loop {{}} }}", vec)).is_empty());
        // The root of a `use` is a relative path in a body.
        assert_eq!(
            forced(&item(reads, "use super::base::bf as local_bf;")),
            value
        );
        assert!(forced(&item(reads, "use ::core::mem::size_of;")).is_empty());
        // Not read: through `enter_impl`, and without `values = true`.
        let relative = "let _d: usize = RLEN + 1;";
        assert!(forced(&item(
            &reads.replace("scope = \"\"", "scope = \"impl\""),
            relative
        ))
        .is_empty());
        assert!(forced(&item("prelude = true, scope = \"\"", relative)).is_empty());
        // The part of a module has no attribute: the module's reads it.
        assert_eq!(
            forced("pub fn f0() -> usize { let _d: usize = crate::base::BLEN + 1; loop {} }"),
            value
        );
        assert!(
            forced("pub fn f0() -> usize { let _d: usize = ::k::base::BLEN + 1; loop {} }")
                .is_empty()
        );
        // The other shapes of a value are not this one: a discriminant, a macro, an absolute macro.
        assert_eq!(
            forced("pub enum E1_1 { V0 = (RLEN as isize) + 0, V1 = (::core::mem::size_of::<*const R1>() as isize) + 100 }"),
            [gen::FORCED_RELATIVE_DISCRIMINANT]
        );
        assert_eq!(
            forced(&item(reads, "let _m = two!() + 1;")),
            [gen::FORCED_MACRO_PATH]
        );
        assert!(forced(&item(
            reads,
            "let _m = ::k::two!() + 1; let _a = 1u8 as RA;"
        ))
        .is_empty());
    }

    /// The item with a forced path of a value that the Leaker accepts fails the run, and so does one
    /// that it rejects for another reason, even for a relative path in another value (the default of a
    /// const parameter, a discriminant) or a `RelativePath` of another cause; one rejected for the
    /// path in the body or the constant passes. Nothing is forced through `enter_impl`.
    #[test]
    fn an_accepted_item_with_a_forced_value_path_fails_and_a_rejected_one_passes() {
        let relative = |kind: &str, note: &str| Rejection {
            reason: "RelativePath".to_string(),
            kind: kind.to_string(),
            tokens: "crate :: base :: BLEN".to_string(),
            note: format!("it is in {}, which the copy writes as it is", note),
            ..Default::default()
        };
        for (kind, scope) in [
            ("trait", ""),
            ("trait", "mod"),
            ("trait", "trait"),
            ("impl-trait", ""),
            ("impl-inherent", ""),
            ("mod", ""),
        ] {
            let item = item_with(kind, scope, true, &[gen::FORCED_VALUE_PATH]);
            let problems = forced_problems(&item, &report_with(true, Vec::new()));
            assert_eq!(problems.len(), 1, "{} {}", kind, scope);
            assert_eq!(problems[0].0, FORCED_ACCEPTED);
            assert!(problems[0].1.contains("forced:value_path"));
            assert!(problems[0].1.contains("whose note holds `body of `"));
            for note in [
                "the body of `m`",
                "the default body of `m`",
                "the value of `KV0`",
                "the default value of `K0`",
            ] {
                let right = report_with(false, vec![relative("Expr", note)]);
                assert!(forced_problems(&item, &right).is_empty(), "{}", note);
            }
            for wrong in [
                // The same reason and cause, for another value: the default of a const parameter, the
                // discriminant of an enum.
                relative("Expr", "the default of `N`"),
                relative("Expr", "the discriminant of `V0`"),
                // A path of a type, or a macro, is not a path of a value.
                relative("Type", "the body of `m`"),
                relative("Macro", "the body of `m`"),
                error_of("Unreadable", "Unreadable", "use :: core :: mem :: * ;"),
                error_of("UnprovableBound", "Type", "R1 < T >"),
            ] {
                let report = report_with(false, vec![wrong]);
                let problems = forced_problems(&item, &report);
                assert_eq!(problems.len(), 1);
                assert_eq!(problems[0].0, FORCED_OTHER);
            }
            // One error of the shape among others is enough.
            let report = report_with(
                false,
                vec![
                    relative("Expr", "the default of `N`"),
                    error_of("UnprovableBound", "Type", "R1 < T >"),
                    relative("Expr", "the body of `m`"),
                ],
            );
            assert!(forced_problems(&item, &report).is_empty());
        }
        let through_impl = item_with("impl-trait", "impl", true, &[gen::FORCED_VALUE_PATH]);
        assert!(forced_of(&through_impl).is_empty());
        assert!(forced_problems(&through_impl, &report_with(true, Vec::new())).is_empty());
        // An item whose plain tag `value_path` is the discriminant's, or the stale tag of an impl, has
        // no forced shape: the Leaker may accept it.
        let plain = item_with("impl-trait", "", true, &[gen::TAG_VALUE_PATH]);
        assert!(forced_problems(&plain, &report_with(true, Vec::new())).is_empty());
    }

    /// The forced paths of values are in the summary of a run with `values` only: a run without it
    /// prints the lines it always printed.
    #[test]
    fn the_summary_counts_the_forced_value_paths_with_values_only() {
        assert!(FORCED
            .iter()
            .any(|shape| shape.tag == gen::FORCED_VALUE_PATH && shape.values));
        assert_eq!(forced_class(gen::FORCED_VALUE_PATH), "forced");
        let mut case = generate_with(1, 0, 8, Features::parse(""), Flags::default());
        let mut item = case.items.remove(0);
        item.kind = "trait";
        item.scope = "";
        item.tags.insert(gen::FORCED_VALUE_PATH.to_string());
        let id = item.id;
        case.items = vec![item];
        let selected = [(0, &case)];
        let mut reports = BTreeMap::new();
        reports.insert(
            (0, id),
            Report {
                kind: "trait".to_string(),
                accepted: false,
                rejections: vec![Rejection {
                    reason: "RelativePath".to_string(),
                    kind: "Expr".to_string(),
                    note: "it is in the body of `m`, which the copy writes as it is".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        let stats = Stats::collect(&selected, &reports);
        let run = Run {
            seed: 1,
            cases: 1,
            items: 8,
            features: String::new(),
            accept: false,
            stages: false,
            only: None,
        };
        let without = summarize(
            &stats,
            &defaults(),
            &run,
            Features::default(),
            Flags::default(),
        );
        let with = summarize(
            &stats,
            &defaults(),
            &run,
            Features::default(),
            Flags {
                values: true,
                ..Flags::default()
            },
        );
        // The shape is counted whatever the flags, as the other shapes are; its line of the forced
        // shapes is printed with `values` only.
        assert!(without.contains("shape relative path in a value"));
        assert!(without.contains("1 analyzed,    0 accepted,    1 rejected; forced"));
        assert!(!without.contains("forced shape relative path in a value"));
        assert!(without.contains("forced shape unreadable value"));
        assert!(with.contains("forced shape relative path in a value"));
        assert!(with.contains(
            "1 items,    1 rejected for its reason,    0 rejected for another,    0 accepted"
        ));
    }

    /// The generated items have the forced tags only where their own text gives them, and an impl
    /// takes none from its type or its trait.
    #[test]
    fn the_forced_tags_of_the_generated_items_are_read_from_their_text() {
        let (mut checked, mut paths) = (0, 0);
        for flags in [
            Flags::default(),
            Flags {
                values: true,
                ..Flags::default()
            },
        ] {
            for seed in 0..20 {
                let case = generate_with(seed, 0, 8, Features::parse(""), flags);
                for item in &case.items {
                    let from_text: BTreeSet<String> = gen::forced_in(&item.source)
                        .into_iter()
                        .map(String::from)
                        .collect();
                    let has: BTreeSet<String> = item
                        .tags
                        .iter()
                        .filter(|tag| tag.starts_with(gen::TAG_FORCED))
                        .cloned()
                        .collect();
                    // A module has the tags of its items, which its own text has too.
                    assert_eq!(has, from_text, "{}", item.source);
                    checked += usize::from(!has.is_empty());
                    paths += usize::from(has.contains(gen::FORCED_VALUE_PATH));
                    // Without `values` no value is generated, and an item of `enter_impl` is not read.
                    assert!(
                        flags.values || !has.contains(gen::FORCED_VALUE_PATH),
                        "{}",
                        item.source
                    );
                    if item.scope == "impl" {
                        assert!(!has.contains(gen::FORCED_VALUE_PATH), "{}", item.source);
                    }
                }
            }
        }
        assert!(checked > 5, "{} items have a forced tag", checked);
        assert!(paths > 5, "{} items have a forced path of a value", paths);
    }

    /// The classification of `SHAPE_TAGS` into forced, forced in part and not forced.
    #[test]
    fn the_shape_tags_are_classified() {
        assert_eq!(forced_class(gen::TAG_NESTED_SELF_BOUNDS), "forced");
        assert_eq!(forced_class(gen::TAG_SELF_NO_MEANING), "forced");
        assert_eq!(
            forced_class(gen::TAG_SELF_IN_TRAIT_PRED),
            "forced in part (forced:self_in_trait_predicate)"
        );
        assert_eq!(forced_class(gen::TAG_NESTED_SELF_OPTION), "not forced");
        assert_eq!(forced_class(gen::TAG_MACRO_CALL), "not forced");
        // Every forced shape is counted by a shape of the table.
        for forced in FORCED {
            assert!(
                SHAPE_TAGS.iter().any(|(tag, _)| *tag == forced.shape),
                "{}",
                forced.tag
            );
        }
        // And the tag of a part that the text gives is a `gen::FORCED_*` tag.
        for forced in FORCED {
            assert!(
                forced.tag == forced.shape || forced.tag.starts_with(gen::TAG_FORCED),
                "{}",
                forced.tag
            );
        }
    }

    /// The item with a forced shape that the Leaker accepts fails the run, and so does one that it
    /// rejects with no error that the shape needs; one rejected for the reason of the shape passes.
    #[test]
    fn an_accepted_forced_item_fails_and_a_rejected_one_passes() {
        let nested = item_with("struct", "mod", true, &[gen::TAG_NESTED_SELF_BOUNDS]);
        let accepted = report_with(true, Vec::new());
        let problems = forced_problems(&nested, &accepted);
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].0, FORCED_ACCEPTED);
        assert!(problems[0].1.contains(gen::TAG_NESTED_SELF_BOUNDS));
        assert!(problems[0].1.contains("NestedSelfInBounds"));
        assert!(problems[0].1.contains("accepted"));

        let right = report_with(
            false,
            vec![
                error_of("RelativePath", "Type", "Rl"),
                error_of("NestedSelfInBounds", "Type", "R1 < Self, T >"),
            ],
        );
        assert!(forced_problems(&nested, &right).is_empty());
        let wrong = report_with(false, vec![error_of("RelativePath", "Type", "Rl")]);
        let problems = forced_problems(&nested, &wrong);
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].0, FORCED_OTHER);
        assert!(problems[0].1.contains("RelativePath"), "{}", problems[0].1);
        // An item with no forced shape is not asked.
        let plain = item_with("struct", "mod", true, &[gen::TAG_NESTED_SELF_OPTION]);
        assert!(forced_problems(&plain, &accepted).is_empty());
        assert!(forced_problems(&plain, &wrong).is_empty());
    }

    /// What decides whether an error is the one that a shape needs: the cause, the tokens, the
    /// `allow_self` of the item, its tags; and the scopes where nothing is forced.
    #[test]
    fn the_error_that_a_forced_shape_needs_is_told_by_its_cause_tokens_and_the_item() {
        // `Rl<'_>`: a relative path of the kind `Type` that holds the elided lifetime.
        let elided = item_with("trait", "", true, &[gen::FORCED_ELIDED_ARG]);
        let ok = report_with(false, vec![error_of("RelativePath", "Type", "Rl < '_ >")]);
        assert!(forced_problems(&elided, &ok).is_empty());
        for other in [
            error_of("RelativePath", "Type", "Rl < 'x >"),
            error_of("RelativePath", "Trait", "Rl < '_ >"),
            error_of("MaybeConst", "Type", "Rl < '_ >"),
        ] {
            let report = report_with(false, vec![other]);
            assert_eq!(forced_problems(&elided, &report).len(), 1);
        }
        // A bare macro: the cause is a macro; a constant: an expression.
        let two = item_with("struct", "", true, &[gen::FORCED_MACRO_PATH]);
        let report = report_with(false, vec![error_of("RelativePath", "Macro", "two !")]);
        assert!(forced_problems(&two, &report).is_empty());
        let report = report_with(false, vec![error_of("RelativePath", "Expr", "RLEN")]);
        assert_eq!(forced_problems(&two, &report).len(), 1);
        // Without `allow_self` the `Self` of a bound cannot be leaked: another reason says it.
        let denied = report_with(
            false,
            vec![error_of("DeniedSelfInBounds", "Type", "S1_1 < T >")],
        );
        let header = report_with(
            false,
            vec![error_of("SelfInImplSelfType", "Type", "S1_1 < T >")],
        );
        let allowing = item_with("impl-inherent", "", true, &[gen::FORCED_IMPL_HEADER_SELF]);
        let denying = item_with("impl-inherent", "", false, &[gen::FORCED_IMPL_HEADER_SELF]);
        assert!(forced_problems(&allowing, &header).is_empty());
        assert_eq!(forced_problems(&allowing, &denied).len(), 1);
        assert!(forced_problems(&denying, &denied).is_empty());
        assert_eq!(forced_problems(&denying, &header).len(), 1);
        // A predicate on a type that names `Self` is a cycle first if a bound of the generics names a
        // leaked type; the error is still the one about that type.
        let predicate = report_with(
            false,
            vec![error_of(
                "UnprovableBound",
                "Type",
                "super :: base :: B1 < Self >",
            )],
        );
        let plain = item_with("trait", "", true, &[gen::FORCED_SELF_IN_TRAIT_PRED]);
        let cyclic = item_with(
            "trait",
            "",
            true,
            &[gen::FORCED_SELF_IN_TRAIT_PRED, gen::TAG_BOUND_CYCLE],
        );
        assert_eq!(forced_problems(&plain, &predicate).len(), 1);
        assert!(forced_problems(&cyclic, &predicate).is_empty());
        let elsewhere = report_with(
            false,
            vec![error_of("UnprovableBound", "Type", "Self :: A0")],
        );
        assert_eq!(forced_problems(&cyclic, &elsewhere).len(), 1);
        // `enter_impl` visits the self type and the trait's arguments only: nothing is forced, even if
        // the item is accepted; `enter_trait` and a module do not read bodies.
        let through_impl = item_with("impl-trait", "impl", true, &[gen::FORCED_MACRO_PATH]);
        assert!(forced_of(&through_impl).is_empty());
        assert!(forced_problems(&through_impl, &report_with(true, Vec::new())).is_empty());
        let body = |kind, scope| item_with(kind, scope, true, &[gen::TAG_UNREADABLE]);
        assert_eq!(forced_of(&body("struct", "")).len(), 1);
        assert_eq!(forced_of(&body("trait", "")).len(), 1);
        assert!(forced_of(&body("trait", "trait")).is_empty());
        assert!(forced_of(&body("mod", "")).is_empty());
    }

    /// The floors count the items with no forced shape; the table of all the items counts them too,
    /// and the summary says how many were left out.
    #[test]
    fn the_floors_leave_out_the_items_with_a_forced_shape() {
        let mut cases: Vec<Case> = (0..3)
            .map(|i| generate_with(case_seed(1, i), i, 8, Features::parse(""), Flags::default()))
            .collect();
        // The items have no forced shape but the one that the structs get, an elided lifetime.
        let mut forced_items = 0;
        for case in &mut cases {
            for item in &mut case.items {
                for forced in FORCED {
                    item.tags.remove(forced.tag);
                }
                if item.kind == "struct" {
                    item.tags.insert(gen::FORCED_ELIDED_ARG.to_string());
                    item.tags.insert(gen::TAG_ELIDED_ARG.to_string());
                    item.scope = "";
                    forced_items += 1;
                }
            }
        }
        assert!(forced_items > 5);
        let selected: Vec<(usize, &Case)> = cases.iter().enumerate().collect();
        let report_for = |item: &gen::Item, accepted: bool| Report {
            kind: item.kind.to_string(),
            accepted,
            positions: 4,
            checked: 4,
            rejections: if accepted {
                Vec::new()
            } else {
                vec![error_of("RelativePath", "Type", "Rl < '_ >")]
            },
            ..Default::default()
        };
        // The Leaker accepts everything but the forced items.
        let mut reports = BTreeMap::new();
        for (i, case) in &selected {
            for item in &case.items {
                reports.insert((*i, item.id), report_for(item, item.kind != "struct"));
            }
        }
        let stats = Stats::collect(&selected, &reports);
        assert_eq!(stats.left_out, forced_items);
        assert_eq!(stats.rejected_of("struct"), 0);
        assert_eq!(stats.accepted_of("struct"), 0);
        assert_eq!(stats.total_accepted(), stats.analyzed - forced_items);
        let (a, r) = stats.all["struct"];
        assert_eq!((a, r), (0, forced_items));
        assert_eq!(stats.all_rate("struct"), "0%");
        assert_eq!(stats.all_rate("enum"), "100%");
        let count = stats.forced[gen::FORCED_ELIDED_ARG];
        assert_eq!(
            count,
            ForcedCount {
                accepted: 0,
                other: 0,
                rejected: forced_items
            }
        );
        assert!(defaults().problems(&stats).is_empty());
        // A rejected struct that has no forced shape counts in the floors as before.
        let mut reports = BTreeMap::new();
        for (i, case) in &selected {
            for item in &case.items {
                reports.insert((*i, item.id), report_for(item, item.kind != "struct"));
            }
        }
        let mut plain = Vec::new();
        for case in &mut cases {
            let item = case.items.iter_mut().find(|i| i.kind == "struct").unwrap();
            item.tags.remove(gen::FORCED_ELIDED_ARG);
            plain.push(item.id);
        }
        let selected: Vec<(usize, &Case)> = cases.iter().enumerate().collect();
        let stats = Stats::collect(&selected, &reports);
        assert_eq!(stats.left_out, forced_items - plain.len());
        assert_eq!(stats.rejected_of("struct"), plain.len());
        // The summary says how many were left out, and counts each forced shape.
        let run = Run {
            seed: 1,
            cases: 3,
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
        assert!(summary.contains(&format!(
            "items left out of the acceptance floors (they have a forced shape): {} of {}",
            stats.left_out, stats.analyzed
        )));
        assert!(summary.contains("forced shape '_ in a path"));
        assert!(summary.contains("all%"));
        assert!(summary.contains("; forced in part (forced:elided_arg)"));
    }

    /// An accepted forced item is counted as a failure, not as an accepted item, and a run with too
    /// few forced shapes, which checks nothing, fails.
    #[test]
    fn the_check_of_the_forced_shapes_cannot_pass_with_none() {
        let mut stats = Stats {
            analyzed: 300,
            ..Default::default()
        };
        assert_eq!(stats.forced_check_problems().len(), 1);
        stats.left_out = 300 / FORCED_SHARE;
        assert!(stats.forced_check_problems().is_empty());
        stats.left_out -= 1;
        assert_eq!(stats.forced_check_problems().len(), 1);
        // A short run is not asked.
        let short = Stats {
            analyzed: FORCED_MIN_ANALYZED - 1,
            ..Default::default()
        };
        assert!(short.forced_check_problems().is_empty());
    }

    /// The structs, enums and unions of `self_bound_adts` (F41, F42) are generated, nested in
    /// `enter_mod`, and the Leaker must accept them: no reason explains their rejection.
    #[test]
    fn the_adts_with_a_bound_that_names_their_self_are_generated() {
        let count = |features: Features, tag: &str| -> (usize, usize) {
            let (mut found, mut types) = (0, 0);
            let (mut option_on, mut option_off) = (0, 0);
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
                        // Half of the items have the option, if the feature is on, and the tag of
                        // the type that names `Self` says which half.
                        let option =
                            attr.map_or(false, |a| a.contains("nested_self_entries = true,"));
                        assert!(features.nested_self_entries || !option);
                        assert!(!(option && item.tags.contains(gen::TAG_NESTED_SELF_BOUNDS)));
                        assert!(!(!option && item.tags.contains(gen::TAG_NESTED_SELF_OPTION)));
                        option_on += usize::from(option);
                        option_off += usize::from(!option);
                    }
                }
            }
            if features.nested_self_entries {
                assert!(
                    option_on > 0 && option_off > 0,
                    "{} {}",
                    option_on,
                    option_off
                );
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

    /// The ordinary Referrer's crate has each copy in a module of its own, and the crate of the carried
    /// errors has one macro of each item; the flags of the invocation are those of the item.
    #[test]
    fn each_copy_has_a_module_of_its_own_and_each_item_a_macro_of_carried_errors() {
        let (mut inst, mut shadow) = (0, 0);
        for seed in 0..12 {
            let case = generate_with(seed, 0, 6, Features::parse(""), Flags::default());
            for item in &case.items {
                let call = format!("tlp_l_000::__tl_prop_{}!(", item.id);
                let lines: Vec<&str> = case
                    .referrer
                    .lines()
                    .filter(|line| line.contains(&call))
                    .collect();
                assert_eq!(lines.len(), 1, "{}", case.referrer);
                assert!(
                    lines[0].starts_with(&format!("pub mod c{} {{ ", item.id))
                        && lines[0].ends_with(" }"),
                    "{}",
                    lines[0]
                );
                // A module has none of the flags: its copy is one module.
                let flags = lines[0].contains(", inst(") || lines[0].contains(", shadow(");
                assert!(!(flags && item.kind == "mod"), "{}", lines[0]);
                inst += usize::from(lines[0].contains(", inst("));
                shadow += usize::from(lines[0].contains(", shadow("));
                let carry = format!("tlp_l_000::__tl_carry_{}!(carry(", item.id);
                let carried = case
                    .errors_referrer
                    .lines()
                    .filter(|line| line.contains(&carry))
                    .count();
                assert_eq!(carried, 1, "{}", case.errors_referrer);
            }
        }
        assert!(inst > 0 && shadow > 0, "{} {}", inst, shadow);
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
                    // (The accepted items that the Leaker carries drop their errors: `allow(all)`.)
                    let mut accepted = case
                        .accept_referrer
                        .replace("__tl_prop_", &prefix)
                        .replace(", allow(all)", "");
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

    /// The macro crate has its own table (a superset, in another crate): it must spell each trait
    /// of the generator's as the generator does, or the twin of a bare trait is not the item.
    #[test]
    fn the_prelude_traits_of_the_generator_are_those_of_the_macro_crate() {
        let text = std::fs::read_to_string(root().join("tests/crates/macro/bare.rs")).unwrap();
        for (name, absolute) in gen::PRELUDE_TRAITS {
            let entry = format!("(\"{}\", \"{}\")", name, absolute);
            assert!(
                text.contains(&entry),
                "tests/crates/macro/bare.rs has no {}",
                entry
            );
        }
        // Every one is a name of the prelude, and the table has no name twice.
        let names: Vec<&str> = prelude_names().collect();
        let mut seen = BTreeSet::new();
        for (name, _) in gen::PRELUDE_TRAITS {
            assert!(names.contains(name), "{} is not a prelude name", name);
            assert!(seen.insert(*name), "{} is twice in the table", name);
            assert!(gen::is_prelude_trait(name));
        }
        assert!(!gen::is_prelude_trait("Vec"));
        assert!(gen::PRIMITIVE_NAMES.iter().all(|name| names.contains(name)));
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
        let missing = missing_artifacts(&b, 3, true, stages);
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
        assert!(missing_artifacts(&b, 3, true, stages).is_empty());
        // A Leaker's executable that is missing is reported although the Referrer's crate failed.
        let b = build(false, &[referrer_name(3).as_str()]);
        let missing = missing_artifacts(&b, 3, false, &[]);
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

    /// A macro that panics because `allow()` of the cause of every error left one is told from the
    /// rest, whatever the mode.
    #[test]
    fn a_macro_that_panics_for_an_error_that_remains_after_allowing_every_cause_is_named() {
        let panicked = format!(
            "error: proc macro panicked\n = help: message: {}: item 5: 1 error(s) remain",
            ALLOW_LEFT
        );
        assert_eq!(leaker_diag_class(LEAKER_MACRO, &panicked), ACCEPT_FAILED);
        assert_eq!(leaker_diag_class(LEAKER_ACCEPT, &panicked), ACCEPT_FAILED);
        assert_eq!(leaker_diag_class(LEAKER_STAGES, &panicked), ACCEPT_FAILED);
        assert_eq!(
            leaker_diag_class(
                LEAKER_ACCEPT,
                "error: proc macro panicked\n = help: message: x"
            ),
            LEAKER_ACCEPT
        );
        assert!(ACCEPT_FAILED.starts_with("ACCEPT FAILED"));
        // The failure leads with the message of the panic.
        let text = leaker_diag_text(&panicked);
        assert!(text.starts_with("ALLOW LEFT ERRORS: item 5"), "{}", text);
        assert_eq!(leaker_diag_text("error: x"), "error: x");
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
        assert!(default.nested_self_entries && default.strict_anchors);
        let changed = Features::parse("-nested_self_entries, -strict_anchors");
        assert!(!changed.nested_self_entries && !changed.strict_anchors);
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

    /// What a macro reports of an item: the status (`ok` accepted, `rej` rejected, `acc` rejected
    /// and accepted by `allow()` of the cause of every error), what it wrote for an item it copied,
    /// and the errors of the analysis.
    #[test]
    fn reports_are_read_back() {
        let (id, r) = parse_report("7|struct|ok|entries=2 positions=5 checked=4").unwrap();
        assert_eq!(
            (id, r.accepted, r.entries, r.positions, r.checked),
            (7, true, 2, 5, 4)
        );
        // An accepted item has no errors, and says what the macro wrote.
        let (_, r) =
            parse_report("3|struct|ok|entries=2 positions=5 checked=4 skipped=1 twin=1\u{1e}")
                .unwrap();
        assert!(r.accepted && r.rejections.is_empty());
        assert_eq!((r.skipped, r.twin), (1, true));
        let (id, r) = parse_report("8|mod|rej|use absolute path: 'X'").unwrap();
        assert_eq!((id, r.accepted), (8, false));
        // The errors of a rejection follow the report.
        let line = "8|trait|rej|use absolute path: 'Clone' use absolute path: 'Default'\u{1e}\
                    use absolute path: 'Clone'\u{1f}Clone\u{1f}\u{1d}\
                    use absolute path: 'Default'\u{1f}Default\u{1f}the note";
        let (id, r) = parse_report(line).unwrap();
        assert_eq!((id, r.accepted), (8, false));
        assert_eq!(r.rejections.len(), 2);
        assert_eq!(r.rejections[1].tokens, "Default");
        assert_eq!(r.rejections[1].note, "the note");
        let (_, r) = parse_report("7|mod|rej|x\u{1e}use absolute path: 'X'\u{1f}X\u{1f}").unwrap();
        assert!(!r.accepted && !r.allowed);
        assert_eq!(r.rejections[0].reason, "");
        // `acc` is not accepted as it is, and has the errors and what the macro wrote.
        let line = format!(
            "5|trait|acc|entries=3 positions=7 checked=2 skipped=1 twin=0\u{1e}{}",
            "use absolute path: 'Rel'\u{1f}Rel\u{1f}a note\u{1f}RelativePath\u{1f}Type"
        );
        let (id, r) = parse_report(&line).unwrap();
        assert_eq!(id, 5);
        assert!(!r.accepted && r.allowed);
        assert_eq!((r.entries, r.positions, r.checked), (3, 7, 2));
        assert_eq!(r.rejections.len(), 1);
        assert_eq!(r.rejections[0].tokens, "Rel");
        assert_eq!(r.rejections[0].reason, "RelativePath");
        assert_eq!(r.rejections[0].kind, "Type");
        assert!(r.rewrite_error.is_none());
        // The reason of the rejection is the messages, as for a rejected item.
        assert_eq!(r.detail, "use absolute path: 'Rel'");
        // The rewrite of the copy failed.
        let (_, r) =
            parse_report("5|trait|acc|entries=3 positions=7 rewrite-error=no host\u{1e}").unwrap();
        assert_eq!(r.rewrite_error.as_deref(), Some("rewrite-error=no host"));
        assert!(parse_report("nonsense").is_none());
    }
}

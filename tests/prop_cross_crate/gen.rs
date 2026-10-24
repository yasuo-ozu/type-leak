//! A seeded generator of random but valid Rust: the Leaker's crate and the Referrer's crate.
//!
//! The Leaker's crate has a module `base` of types and traits (`B0`, `B1<T>`, `BL<'a, T>`, `BC<T,
//! N>`, `BB<B>`, `Bl<'a>`, `BT0`, `BT1<T>`, `BT2`, `BLEN`), then
//!
//! - modules `m0..` whose items are analyzed one by one (`intern_struct`, `intern_enum`,
//!   `intern_union`, `intern_trait`, `intern_impl`), some with a nested module, and
//! - modules `mm0..` analyzed as a whole (`intern_mod`), some with a nested module, and with free
//!   functions and type aliases besides.
//!
//! Each module declares the same types as `base` under other names (`R0`, `R1<T>`, ...), imports
//! some of those of `base` with `use` (`I1`, from `super::base` or `crate::base`, or `J1`, from the
//! Leaker's own crate by its name) and names them by path (`super::base::B1`, `crate::base::B1`,
//! `::tlp_l_000::base::B1`, `self::R1`), so the types are relative in several ways.
//!
//! Types: primitives, `String`, tuples, arrays (with literal, const generic and constant lengths),
//! `Option`, `Vec`, `Box`, `Rc`, `Arc`, `Cell`, `Result`, maps and `PhantomData` (by absolute path
//! or, if the Leaker allows the prelude, bare), references (with `'static` or named lifetimes),
//! raw pointers, function pointers (also `for<'h>`, `unsafe` and `extern`), `dyn` objects (of
//! `Debug`, `Any`, `Fn*`, `Iterator`, and the traits of the module, with bounds), the types of the
//! module with lifetime, type and const arguments, generic parameters (with `?Sized` and `T::Item`
//! if they are iterators), `Self` where it is valid, and `Self::A` in traits.
//!
//! Items: structs (named, tuple and unit), enums, unions; traits (generics with bounds, `T: 'a`
//! and where clauses, supertraits, associated types and consts, methods with every receiver,
//! generic methods, `where Self: Sized`); inherent impls (of the structs and enums, and of the
//! types of the module) and impls of the traits of the module.
//!
//! Generics: lifetimes, types (`?Sized`, bounds, outlives bounds, iterators), and const parameters
//! of `usize`, `bool`, `i32`, `u8` and `char` (`RI<V>`, `RU<V>`, `RH<V>` take the last three). A
//! third of the structs, enums, unions and traits give their trailing parameters that nothing
//! constrains a default (`T = Vec<u8>`, `U = BTreeMap<T, u8>`, `const N: usize = 3`, `{ RLEN }`),
//! which the Leaker analyzes with the parameters before it. A constant that is a single
//! identifier is also written bare as an argument (`R2C<T, RLEN>`: it parses as a type that may be
//! a const), except in the paths a module analyzed as a whole declares.
//!
//! Defaults of traits (`Features::trait_self_defaults`, F39): a fifth of the traits analyzed one by
//! one are made for their defaults (`Gen::gen_self_default_trait()`): `trait Tr<T: AsRef<Self>, U =
//! Rel<T>>`, a parameter with a bound that names the trait's own `Self` (written in the parameter or
//! in a where predicate) and trailing parameters whose defaults name it. The control has a bound
//! that names no `Self` (`T: Clone`). A trait of that shape is nested in `enter_mod` in 45% of the
//! cases, where the Visitor writes the default through the host (`with_nested_self()`); then
//! 60% are declared again in the Referrer's crate (`redeclare = true` in the attribute), with a body
//! that names no entry. The others are analyzed by `intern_trait` or `enter_trait`, where the
//! default is looked up without `Self`. The traits carry the tags `shape:self_default`,
//! `shape:plain_default` and `shape:redeclared`, which the statistics count and no rejection needs.
//!
//! Structs, enums and unions with a generic whose bound names their own `Self` and a default that
//! uses it (`Features::self_bound_adts`, F41): `struct S<T: Clone + AsRef<Self>, U = Rel<T>>`, nested
//! in `enter_mod` (`Gen::gen_self_bound_adt()`). With `Features::nested_self_entries` (F42) the
//! attribute asks for `LeakerConfig::nested_self_entries()` (all the items nested in `enter_mod`),
//! and some have a field that names `Self` in a relative type (`Rel2<Self, T>`), which is for any
//! type that the bound holds for.
//!
//! Scopes: the macro analyzes some items through `enter_mod`, `enter_generics`, `enter_trait` or
//! `enter_impl` instead of the `intern_*` of their kind (`Item::scope`, `SCOPES`).
//!
//! Spellings: the traits of the prelude in bounds (`Sized`, `?Sized`, `Send`, `Fn`, `Into`,
//! `AsRef`, ...) are bare or in full, and a type parameter has a marker trait or a comparison as a
//! bound (`Gen::trait_path()`, `Gen::bare_percent`).
//!
//! Names: every module declares types with the names of generics (`T`, `U`, `X`, `SHADOW_NAMES`), a
//! relative type next to a generic of the same name in a sibling scope; a const parameter takes the
//! name of a type parameter that a sibling scope declares (`V`, `Y`); a type parameter's bound
//! names another parameter (`T: Into<V>`); a path with the lifetime of a binder (`Rl<'h>`) is in
//! `for<'h> Fn(..)` bounds, function pointers and trait objects, with an input that keeps the
//! lifetime or without one.
//!
//! Settings: the Leaker does not allow the primitive types in a fifth of the cases
//! (`Config::primitive`), and each item says whether the Referrer invokes it as strict.
//!
//! Not generated: type macros (the test macro's witness cannot return a type that a macro
//! expands to), const expressions with types in them, and unsized or `dyn` implementors.
//!
//! The grammar is built to be valid: every type is generated with what the generics in scope
//! promise (sizedness, bounds, outlives), so the Leaker's crate compiles by itself. If it does not,
//! the harness builds it again without `type-leak` and reports a bug of the generator.
//!
//! What `type-leak` is known to fail on is avoided unless a feature asks for it ([`Features`]):
//! each is a finding of the test, with a reduced reproduction in the report of the test. The
//! generator also avoids the constructs that cannot be copied to another crate whatever
//! `type-leak` does, such as an inherent impl on a type of another module in the copy of a module.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeSet;
use std::hash::Hasher;

// ---------------------------------------------------------------------------------------------
// Configuration and results.
// ---------------------------------------------------------------------------------------------

/// Whether a bare `?Sized` is accepted whatever the configuration says about the names of the
/// prelude.
///
/// `?` applies to the real `Sized` only, so `T: ?Sized` is no more context-dependent than
/// `T: ?::core::marker::Sized`. `type-leak` at fb11f41 does not know it: it takes the bare `Sized`
/// of `?Sized` for a relative path like any other bare name of the prelude, and rejects it unless
/// `LeakerConfig::allow_prelude()` is on. The Leaker accepts a bare `?Sized` always (`?` applies
/// to the real `Sized` only), so it is `true`; it is the only place that says so. It reaches
/// three places: the generator (a bare `?Sized` is an expected rejection without the prelude only
/// if it is `false`: the tag `bare:?Sized`), the macro (`relaxed_sized` argument of `prop_leak`:
/// the item with a bare `?Sized` must then equal its twin with the full path, prelude or not) and
/// the harness (`main.rs`: a "use absolute path: '? Sized'" is justified only by that tag).
pub const BARE_RELAXED_SIZED_ALWAYS_OK: bool = true;

/// The kinds the statistics count.
pub const KINDS: &[&str] = &[
    "struct",
    "enum",
    "union",
    "trait",
    "impl-inherent",
    "impl-trait",
    "mod",
];

/// Constructs that `type-leak` is known to fail on (see the findings in the report of the test):
/// the generator avoids them unless they are asked for, so the test passes where nothing new is
/// found. The feature of a finding that is fixed is on by default (see `Features::parse()`).
#[derive(Clone, Copy, Default, Debug)]
pub struct Features {
    /// Type parameters with `?Sized`. On by default: the encoders wrap them in a pointer (`*const
    /// T`), since a tuple is well-formed only if its elements but the last are `Sized`.
    pub unsized_params: bool,
    /// A bound or where predicate that names a leaked type using a generic (`T: Into<Wrap<T>>`,
    /// `T: Iterator<Item = Wrap<'a>>`). It was a finding (the copy is cyclic, or the type is not
    /// found), fixed by rewriting or rejecting, so it is generated by default, in a share of the
    /// items (`Gen::cyclic`).
    pub bound_cycles: bool,
    /// A field of a struct whose outlives bounds (`T: 'a`) only rustc infers (RFC 2093).
    pub inferred_outlives: bool,
    /// `Self::A`, an associated type of the trait, in an impl of the trait: in a leaked type, the
    /// Repeater's impl cannot write it, so type-leak rejects it (on by default, F6).
    pub impl_assoc_self: bool,
    /// `dyn Fn(&Rel<T>) -> &u8`, with the elided lifetime of the input given to the output, where
    /// the leaked type uses generics: the path that replaces it names their lifetimes, so the
    /// elision is ambiguous (E0106). The Leaker leaks the whole holder (`dyn Fn(..) -> &u8`, a
    /// function pointer) or rejects the item, so this is on by default (see `Features::parse()`).
    pub elided_fn_generics: bool,
    /// A bound `T: Fn(A) -> &'static dyn X` next to other bounds: type-leak moves the where
    /// predicates to the parameter, and prints `T: Fn(A) -> (&'static dyn X) + Y` (F8, fixed: on by
    /// default). Off, the generator parenthesizes the return type itself.
    pub bound_merge_syntax: bool,
    /// `Self: 'y` in a method of an impl or a struct: the copy lacks `T: 'y` of `Self = S<T>` (E0309).
    pub impl_self_outlives: bool,
    /// An impl for a function pointer (`impl Tr for &fn()`) with `where Self: 'y`: type-leak prints
    /// the predicate `(&fn()): 'y`, parenthesized because rustc reads `&fn(): 'y` as the start of
    /// a return type (F9, fixed: on by default).
    pub fn_pointer_self: bool,
    /// A leaked path (`Wrap<'x>`) that names a lifetime of a method of a trait which another
    /// argument does not keep, with the lifetime in the return type: E0195 in the impl of the
    /// Referrer, so type-leak rejects it (on by default, F7). A free function and a method of an
    /// inherent impl have no trait signature to differ from, so they have such paths whether it
    /// is on or not (F22).
    pub method_lifetime_paths: bool,
    /// A path relative to the crate (`crate::base::B1<R0>`) in a type of a module analyzed as a
    /// whole that mentions an item of the module: the entry is keyed by the module (on by default,
    /// F4), so that the Referrer rewrites it in the copy of the module.
    pub module_crate_paths: bool,
    /// A bound of the generics of an impl that names `Self` (`impl<T: AsRef<Option<Self>>> W<T>`):
    /// the self type is an entry that depends on `Self`, which is the self type itself, so the
    /// copy cannot write it (F13, fixed: the Leaker rejects it, and the rejections are counted, so
    /// it is on by default).
    pub impl_self_bounds: bool,
    /// An impl of a trait whose where clause names a leaked type (`impl<T> Tr<T> for S where T:
    /// Fn(Rel)`), and a method of it with a predicate that names `Self::A`
    /// (`where X: Into<Self::A>`): rustc normalizes the projection in the environment of the
    /// method, and the impl it needs has `T: Fn(Rel)`, which the copy writes as `T:
    /// Fn(<() as Repeater<..>>::Type)`, not normalized yet (E0277, E0271). The context of the
    /// leaked type has no generics, so the check for a bound that names a leaked type misses it
    /// (F14, fixed: on by default, the Leaker rejects it). Off, the impls of the traits that have
    /// both are not generated.
    pub impl_pred_leaked: bool,
    /// `fn(&A) -> &R` or `Fn(&A) -> &R` (the elided lifetime of the input is the output's) where
    /// `A` names `Self`, and the copy writes `Self` as a type that has lifetimes (the self type of
    /// an impl, an ADT with lifetime parameters), or `Self::A` as `<Self as Tr<'a, ..>>::A` with
    /// the lifetime parameters of the trait: the input then has more lifetimes, and rustc cannot
    /// tell which one the output has (E0106, F15). The oracle of the test macro does not compare a
    /// position that has it, and the Referrer names the lifetime that elision gives (`for<'e>`,
    /// F23), or the Leaker leaks the whole signature or rejects the item where it cannot (a path
    /// hides the lifetime, F28), so this is on by default (see `Features::parse()`). Off, such a
    /// sugar is not generated.
    pub self_elision: bool,
    /// An outlives bound that the original has only by the well-formedness of its signature
    /// (`Self::A: 'x` from `&'x mut Self::A`, with `Self: 'x` from `&'x self`), which the
    /// Repeater's impl lacks (E0309, F16: fixed, on by default). Off, `Self::A` is not put under a
    /// lifetime. `Self` under a path or a trait object with a lifetime always needs `Self: 'x`
    /// written, and so does a where predicate: they are the limitation F5 and the test macro's
    /// witness (no receiver).
    pub implied_outlives: bool,
    /// A strict Referrer (`allow_ambiguous_crate_name(false)`) replaces the optional entries too.
    /// An optional entry in a where predicate that names a lifetime (`T: Fn(&'a ::ext::W<T,
    /// ::ext::B0>)`) has the predicate in its context, so it is replaced by a path that needs
    /// itself (E0277, F17): `not_optional` of the Leaker misses the type that holds another
    /// optional type. Fixed (on by default): the Leaker copies it as written. Off, the item that
    /// has it is not invoked as strict.
    pub strict_cycles: bool,
    /// A strict Referrer replaces an optional entry that holds a lifetime of a method (`fn
    /// m<'x>(&self, a: ::ext::W<&'x u8>) -> Rel<'x>`): the copy has `'x` in no argument that
    /// constrains it, so it is early-bound, where it is late-bound in the trait (E0195, F18, fixed:
    /// the Leaker copies the type as written). Off (the default), the item that has it is not
    /// invoked as strict.
    pub strict_anchors: bool,
    /// In an impl whose self type is leaked, a method with a bound of a generic that names `Self`
    /// (`X: Fn(&Self) -> u8`) and a leaked type in another predicate that uses the generic
    /// (`Rel<X>: Default`): the entry's Repeater impl has the bound with the Leaker's self type,
    /// and the copy has it with `Self` = the unnormalized projection that replaces the self type
    /// (E0277, E0271, F19, fixed: on by default, the Leaker rejects it). Off, the other predicates
    /// that use the generic are dropped, and a trait whose where clause has such a bound
    /// (`impl_self_bounds`) is not implemented where a method predicate uses the generic.
    pub impl_self_bound_cycle: bool,
    /// The same in a trait with a bound that names `Self::A` (`X: Fn(&Self::A0) -> u8`): the
    /// impl of the Repeater for the host has the bound with the host's type for `A0`, and the copy
    /// has `<Host as Tr>::A0`, which rustc does not normalize in the environment (E0271, F20,
    /// fixed: on by default, the Leaker rejects it). Off, the other predicates that use the
    /// generic are dropped.
    pub trait_assoc_bound_cycle: bool,
    /// A bound of an iterator that names its `Item` (`T: Fn(<T as Iterator>::Item)`), and a
    /// predicate that equates the `Item` to a leaked type (`T: Iterator<Item = Rel<T>>`): the
    /// entry has the first bound in its context, and rustc must normalize `T::Item`, which the
    /// predicate defines by the entry: a cycle (E0275, F21, fixed: on by default, the Leaker
    /// rejects it). Off, the predicate with `Item = ..` is dropped where the bound is.
    pub iterator_item_cycle: bool,
    /// A bound of an iterator whose `Item` a supertrait fixes (`T: ::krate::base::BItem`, `trait
    /// BItem: Iterator<Item = u8>`) that names the `Item` (`T: Fn(<T as Iterator>::Item) -> u8`),
    /// and a predicate with an entry that uses the iterator (`Rel<T>: Default`): the Leaker does not
    /// read the supertraits, so the copy would be E0277, like F21 (F25, fixed: on by default, the
    /// Leaker takes the `Item` of a generic with the bound of a trait that is not one of the
    /// standard library for one that may be fixed, and rejects it).
    pub supertrait_item: bool,
    /// The same with the `Item` fixed by a bound (`T: Iterator<Item = u8>`) and the bound naming it
    /// through a type alias of the Leaker's crate (`T: Fn(::krate::base::BIt<T>) -> u8`), which the
    /// Leaker takes for a struct: the Repeater's impl of the entry carried the bound, and rustc
    /// could not prove it in the environment of the copy (F26, fixed: on by default, the context
    /// of an entry leaves out a bound of a generic that names the generic, if another bound of it
    /// fixes an associated type). A trait with such a bound has no method predicate that names
    /// `Self::A`: rustc cannot prove the impl of the trait there, whatever `type-leak` does.
    pub alias_item: bool,
    /// An argument-position `impl Trait` whose bound names an entry that uses a generic with a bound
    /// that names a leaked type (`X: Fn(Rel0) -> u8`, `a: impl Into<Rel1<X>>`): the Leaker treats it
    /// as the bound of a generic, as F2 does, and rejects it (F27, fixed: on by default). The
    /// oracle of the test macro does not compare a position that has `impl Trait`.
    pub impl_trait_args: bool,
    /// A path that hides a lifetime in the inputs of an elided `fn(..) -> &R` or `Fn(..) -> &R`
    /// (`fn(::core::str::Chars, A) -> &R`), next to a type that adds lifetimes: E0106 in the copy
    /// like F12, as the Leaker did not see the lifetime (F28, fixed: on by default). An elided
    /// output with no lifetime in the inputs has the hidden one, so the Leaker leaks the whole
    /// signature or rejects the item.
    pub hidden_lifetimes: bool,
    /// `Rel<&'a T::Item>` in a method of a trait, where `'a` is a lifetime of the trait and `T` an
    /// iterator that does not outlive it: the signature implies `T::Item: 'a`, which the Leaker
    /// records as the where predicate `T::Item: 'a`, and not as `T: 'a`, which the copy cannot
    /// prove (E0309, like F16; F30: fixed, on by default).
    pub projection_outlives: bool,
    /// A where predicate of an item that names the `Item` of an iterator without being a bound of
    /// it (`Option<<T as Iterator>::Item>: Default`), and a predicate of a method that equates the
    /// `Item` to a type that holds a leaked type (`T: Iterator<Item = (Rel<T>, u8)>`): the entry has
    /// the predicate in its context, and rustc proves it while it normalizes the entry, with the
    /// `Item` that the entry defines: a cycle (E0275, F33, fixed: on by default, the Leaker rejects
    /// it). Off, the equation is dropped where an enclosing predicate names the `Item`.
    pub item_predicate_cycle: bool,
    /// `Rel<&'a P>` where `P` is an associated type path of `Self` that is not the bare `Self::A0`
    /// of a receiver's lifetime: `<Self as Tr<'a, T>>::A0`, `Self::A0` with a lifetime of the trait,
    /// or `Self::G<'b>` of a generic associated type. The signature implies `P: 'a` and nothing of
    /// `Self` or of the arguments of the trait, which the copy for a host type has to prove from
    /// the predicate that the Leaker records (E0491, E0477, E0309, F34: fixed, on by default).
    pub self_projection_outlives: bool,
    /// The generic associated type of `self_projection_outlives` has the where clause `where Self:
    /// 'g, Self: 'c` (rustc requires the second one for a method that names the lifetime of the
    /// trait), and the impls and the host keep it: the signature that names `Self::G<'c>` implies
    /// `Self: 'c` through it, which the impl of the Repeater of the entry needs (E0309 on `__S`
    /// without it, F35: fixed, on by default).
    pub gat_where_clause: bool,
    /// `Self` in a struct, an enum, a union or a trait analyzed as an item nested in `enter_mod`
    /// (`View::visit_item()`, like the items of `intern_mod()`): the entry that names the item's
    /// own `Self` is recorded, and the Visitor finds it in the nested item and writes it with
    /// `Visitor::with_nested_self()` (F36: fixed, on by default). Before the fix the Visitor never
    /// found it, so the copy kept the leaked type (`Rel<Self>`), which the Referrer's crate cannot
    /// resolve. Off, such an item names no `Self`. F38 (fixed, on by default): an absolute type
    /// that names it is no entry, so the strict items copy it as written.
    pub nested_self: bool,
    /// A relative type with several segments whose last segment is the name of a generic in scope
    /// (`self::U` next to a generic `U`). The Visitor used to take the path for the generic, so the
    /// Referrer's crate kept `self::U` (E0425); fixed (the context of a type is computed from the
    /// first segment of a relative path only), on by default.
    pub path_named_like_generic: bool,
    /// A module analyzed as a whole that declares a type with the name of a generic in scope
    /// (`mod m { struct X; fn f<X>(a: crate::base::B1<X>) {} }`). The Visitor used to hide the
    /// entry `crate::base::B1<X>` because the module declares `X`, so the Referrer's crate kept
    /// `crate::base` (E0433); fixed (a generic of the entry hides the declared item), on by default.
    pub declared_named_like_generic: bool,
    /// A bare identifier that may be a const as an argument of a path of another crate, next to an
    /// argument that is leaked (`::l::BC<Rel<T>, ILEN>`, `::l::W<[Self], ILEN>` with `allow_self =
    /// false`, `::l::BC<super::RC<..>, ILEN>` in a module that declares `RC`): the Leaker used to
    /// leak the identifier alone as a type (`type Type = ILEN`, E0573); fixed (it is rejected), on
    /// by default. Off, the generator braces the constant (`{ ILEN }`) in such a path.
    pub bare_const_in_absolute_path: bool,
    /// A trait whose type parameters have defaults that use a generic whose bound names the
    /// trait's own `Self` (`trait Tr<T: AsRef<Self>, U = Rel<T>>`; F39, fixed: on by default), and
    /// the control, where the bound names no `Self` (`T: Clone`). The entry of such a default is
    /// for the implementors of the trait, which the Visitor writes through the host
    /// (`with_nested_self()`) for a trait nested in `enter_mod`, and finds without `Self` for a
    /// trait analyzed as the item itself (`intern_trait`, `enter_trait`). A trait nested in
    /// `enter_mod` is often declared again in the Referrer's crate (`redeclare`), whose
    /// declaration cannot pass the defaulted parameter to its own default (E0128). Before the fix
    /// the Visitor left the default as written, which the Referrer's crate cannot resolve. Off,
    /// no trait has the shape (the defaults of the other traits are as before).
    pub trait_self_defaults: bool,
    /// A struct, an enum or a union nested in `enter_mod` with a generic whose bound names its own
    /// `Self` (`T: Clone + AsRef<Self>`, also in a where predicate) and parameters whose defaults
    /// use it (`U = Rel<T>`). The copy cannot write `Self` in a default (E0735), so the entry in the
    /// default has no `Self` and none of the bounds that name it; the Leaker used to reject the
    /// item (F41: fixed, on by default). A field of the same type is an entry for any type, which the
    /// copy passes its own type for (F37).
    pub self_bound_adts: bool,
    /// The Leaker is configured with `LeakerConfig::nested_self_entries()` for the items nested in
    /// `enter_mod` (`nested_self_entries = true` in the attribute), and the structs, enums and unions
    /// of `self_bound_adts` have a field that names their `Self` in a relative type (`Rel2<Self, T>`
    /// with `T: AsRef<Self>`), which the Leaker rejects without the option (F42: on by default). The
    /// option is an opt-in of the Leaker, with risks that the generator's types do not have (a bound
    /// on the parameter of the leaked type that receives `Self`; rustc before 1.70 and the last
    /// field). It is on for every item nested in `enter_mod`, so the items that name `Self` in a
    /// field and have a bound that does are built, not rejected.
    pub nested_self_entries: bool,
}

/// `TYPE_LEAK_PROP_FEATURES` has `allow_stages`: the macro compares the errors that remain with
/// `LeakerConfig::allow_prelude()` and with `LeakerResult::allow_prelude()` for every item
/// (`stages = true` in the attribute, which only asks it to report both counts), and the harness
/// prints how many items differ. Off by default, so that the tables do not change. It is no field
/// of `Features`, whose `Debug` is in the tables, and not part of `all`.
pub fn allow_stages() -> bool {
    std::env::var("TYPE_LEAK_PROP_FEATURES").map_or(false, |list| {
        list.split(',').any(|name| name.trim() == "allow_stages")
    })
}

impl Features {
    /// The features of the findings that are fixed are on by default, and `list` adds the known
    /// failures.
    pub fn parse(list: &str) -> Self {
        // The findings that are fixed are generated by default.
        let mut features = Features {
            unsized_params: true,
            impl_assoc_self: true,
            method_lifetime_paths: true,
            module_crate_paths: true,
            bound_merge_syntax: true,
            fn_pointer_self: true,
            elided_fn_generics: true,
            bound_cycles: true,
            implied_outlives: true,
            self_elision: true,
            impl_self_bounds: true,
            impl_pred_leaked: true,
            strict_cycles: true,
            impl_self_bound_cycle: true,
            trait_assoc_bound_cycle: true,
            iterator_item_cycle: true,
            hidden_lifetimes: true,
            supertrait_item: true,
            alias_item: true,
            impl_trait_args: true,
            projection_outlives: true,
            item_predicate_cycle: true,
            self_projection_outlives: true,
            gat_where_clause: true,
            nested_self: true,
            path_named_like_generic: true,
            declared_named_like_generic: true,
            bare_const_in_absolute_path: true,
            trait_self_defaults: true,
            self_bound_adts: true,
            nested_self_entries: true,
            ..Features::default()
        };
        for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            match name {
                "all" => {
                    features.unsized_params = true;
                    features.bound_cycles = true;
                    features.inferred_outlives = true;
                    features.impl_assoc_self = true;
                    features.bound_merge_syntax = true;
                    features.elided_fn_generics = true;
                    features.impl_self_outlives = true;
                    features.fn_pointer_self = true;
                    features.method_lifetime_paths = true;
                    features.module_crate_paths = true;
                    features.impl_self_bounds = true;
                    features.impl_pred_leaked = true;
                    features.self_elision = true;
                    features.implied_outlives = true;
                    features.strict_cycles = true;
                    features.strict_anchors = true;
                    features.impl_self_bound_cycle = true;
                    features.trait_assoc_bound_cycle = true;
                    features.iterator_item_cycle = true;
                    features.supertrait_item = true;
                    features.alias_item = true;
                    features.impl_trait_args = true;
                    features.hidden_lifetimes = true;
                    features.projection_outlives = true;
                    features.item_predicate_cycle = true;
                    features.self_projection_outlives = true;
                    features.gat_where_clause = true;
                    features.nested_self = true;
                    features.path_named_like_generic = true;
                    features.declared_named_like_generic = true;
                    features.bare_const_in_absolute_path = true;
                    features.trait_self_defaults = true;
                    features.self_bound_adts = true;
                    features.nested_self_entries = true;
                }
                "inferred_outlives" => features.inferred_outlives = true,
                "impl_assoc_self" => features.impl_assoc_self = true,
                "bound_merge_syntax" => features.bound_merge_syntax = true,
                "elided_fn_generics" => features.elided_fn_generics = true,
                "impl_self_outlives" => features.impl_self_outlives = true,
                "fn_pointer_self" => features.fn_pointer_self = true,
                "method_lifetime_paths" => features.method_lifetime_paths = true,
                "module_crate_paths" => features.module_crate_paths = true,
                "impl_self_bounds" => features.impl_self_bounds = true,
                "impl_pred_leaked" => features.impl_pred_leaked = true,
                "self_elision" => features.self_elision = true,
                "implied_outlives" => features.implied_outlives = true,
                "strict_cycles" => features.strict_cycles = true,
                "strict_anchors" => features.strict_anchors = true,
                "impl_self_bound_cycle" => features.impl_self_bound_cycle = true,
                "trait_assoc_bound_cycle" => features.trait_assoc_bound_cycle = true,
                "iterator_item_cycle" => features.iterator_item_cycle = true,
                "supertrait_item" => features.supertrait_item = true,
                "alias_item" => features.alias_item = true,
                "impl_trait_args" => features.impl_trait_args = true,
                "hidden_lifetimes" => features.hidden_lifetimes = true,
                "projection_outlives" => features.projection_outlives = true,
                "item_predicate_cycle" => features.item_predicate_cycle = true,
                "self_projection_outlives" => features.self_projection_outlives = true,
                "gat_where_clause" => features.gat_where_clause = true,
                "nested_self" => features.nested_self = true,
                "path_named_like_generic" => features.path_named_like_generic = true,
                "declared_named_like_generic" => features.declared_named_like_generic = true,
                "bare_const_in_absolute_path" => features.bare_const_in_absolute_path = true,
                "trait_self_defaults" => features.trait_self_defaults = true,
                "self_bound_adts" => features.self_bound_adts = true,
                "nested_self_entries" => features.nested_self_entries = true,
                // Read by `allow_stages()`: it is no feature of the generator.
                "allow_stages" => (),
                "unsized_params" => features.unsized_params = true,
                "bound_cycles" => features.bound_cycles = true,
                other => panic!("unknown feature `{}`", other),
            }
        }
        features
    }
}

/// Settings of a generated crate.
#[derive(Clone, Debug)]
pub struct Config {
    /// The name of the Leaker's crate.
    pub krate: String,
    /// The Leaker allows the names of the standard prelude, and the generator uses them.
    pub prelude: bool,
    /// The Leaker allows `::std`.
    pub allow_std: bool,
    /// The Leaker allows its own crate.
    pub allow_crate: bool,
    /// The Leaker allows the primitive types (`LeakerConfig::allow_primitive()`). Without it, `u8`
    /// is a relative path like any other: each is an entry of the item.
    pub primitive: bool,
}

impl Config {
    fn allow_arg(&self) -> String {
        let mut allow = Vec::new();
        if self.allow_std {
            allow.push("::std".to_string());
        }
        if self.allow_crate {
            allow.push(format!("::{}", self.krate));
        }
        allow.join(",")
    }
}

/// The tags: the shapes of an item that make `type-leak` reject it, which the generator puts on the
/// items that have one (`Item::tags`). The harness fails a rejection that no tag of the item
/// justifies (`main.rs`, `family()`).
///
/// - `bound_cycle`: a bound or where predicate that may name a leaked type that uses a generic (the
///   item is one of the `CYCLIC_*` shares), or a bound of a generic that names `Self`.
/// - `self_assoc`: `Self::A` or `<Self as Tr>::A` is in the text of the item.
/// - `binder_no_anchor`: a bound, function pointer or trait object `for<'h> Fn(A, Rl<'h>) -> &'h R`,
///   where no input keeps `'h` (the bound is always rejected, the others where they are split).
/// - `const_default`: a relative constant (`{ RLEN }`) is the default of a const parameter.
/// - `method_lifetime`: a lifetime of a method is named by a path (not by a reference).
/// - `self_in_bounds`: a predicate names `Self` (`Self: 'x`, `Self: Sized`).
/// - `self_denied`: the item is analyzed with `allow_self = false` and names `Self`.
/// - `bare:<Name>`: the prelude trait `Name` is spelled bare (`bare:?Sized` for `?Sized`), and the
///   Leaker does not allow the names of the prelude (a bare `?Sized` only if
///   `BARE_RELAXED_SIZED_ALWAYS_OK` is `false`).
pub const TAG_BOUND_CYCLE: &str = "bound_cycle";
pub const TAG_SELF_ASSOC: &str = "self_assoc";
pub const TAG_BINDER_NO_ANCHOR: &str = "binder_no_anchor";
pub const TAG_CONST_DEFAULT: &str = "const_default";
pub const TAG_METHOD_LIFETIME: &str = "method_lifetime";
pub const TAG_SELF_IN_BOUNDS: &str = "self_in_bounds";
pub const TAG_SELF_DENIED: &str = "self_denied";
/// Not put by the generator: the harness gives it to a module analyzed whole and an item analyzed
/// in `enter_mod` (`main.rs`), where a path of another crate can hold a name that the module
/// declares beside a bare name (`::l::B2<I0, R0>`), which the Leaker rejects.
pub const TAG_MODULE_SCOPE: &str = "module_scope";
/// The shapes of `trait_self_defaults` (F39), which explain no rejection (`family()` lists none of
/// them): the statistics count the traits that have them. `shape:self_default`: a parameter has a
/// bound that names the trait's `Self` and a later one has a default; `shape:plain_default`: the
/// control, the same but the bound names no `Self`; `shape:redeclared`: the Referrer's crate
/// declares the trait again (`redeclare`).
pub const TAG_SELF_DEFAULT: &str = "shape:self_default";
pub const TAG_PLAIN_DEFAULT: &str = "shape:plain_default";
pub const TAG_REDECLARED: &str = "shape:redeclared";
/// The shapes of `self_bound_adts` (F41, F42), which explain no rejection, and the Leaker must accept
/// (`main.rs`, `unexplained()`): `shape:self_bound_adt`:
/// a struct, enum or union nested in `enter_mod` whose bound names its `Self` and whose default uses
/// the generic; `shape:self_bound_type`: a field also names `Self` in a relative type, which needs
/// `nested_self_entries`.
pub const TAG_SELF_BOUND_ADT: &str = "shape:self_bound_adt";
pub const TAG_SELF_BOUND_TYPE: &str = "shape:self_bound_type";
/// The shapes that the items accepted by `LeakerResult::allow()` fail on (`TYPE_LEAK_PROP_ACCEPT`,
/// `main.rs`, `accept_families()`; the docs of `LeakerResult::allow()` list them). They explain no
/// rejection:
///
/// - `self_in_predicate`: a where predicate of the item names `Self` or `Self::A`
///   (`where X: Into<Self::A>`, `where Y: Fn(&Self) -> u8`), which `bound_cycle` (a bound of the
///   generics that names a leaked type) makes unprovable;
/// - `alias_bound`: a bound of an iterator names its `Item` (`T: Fn(<T as Iterator>::Item) -> u8`,
///   also through the alias `BIt<T>`) next to a bound or a supertrait that fixes it, which no impl of
///   a trait can have next to a method predicate that names `Self::A`;
/// - `hidden_self`: a function pointer or `Fn(..)` whose output has an elided lifetime and whose
///   inputs name `Self` and a path that hides a lifetime (`fn(Chars, Self) -> &u8`).
pub const TAG_SELF_IN_PREDICATE: &str = "self_in_predicate";
pub const TAG_ALIAS_BOUND: &str = "alias_bound";
pub const TAG_HIDDEN_SELF: &str = "hidden_self";

/// An item with a macro on it, which the Referrer invokes.
#[derive(Clone)]
pub struct Item {
    pub id: usize,
    pub kind: &'static str,
    /// The module that contains it, from the root.
    pub module: Vec<String>,
    pub strict: bool,
    pub allow_self: bool,
    /// How the macro analyzes it: `""` by the `intern_*` method of its kind, or through a scope
    /// (`mod`: `enter_mod`, `generics`: `enter_generics`, `trait`: `enter_trait`, `impl`:
    /// `enter_impl`, see `SCOPES`).
    pub scope: &'static str,
    /// The kinds of the items in it (for a module), and the item itself otherwise.
    pub contents: Vec<&'static str>,
    /// The shapes in it that justify a rejection of `type-leak` (`bound_cycle`, `bare:Sized`, ...),
    /// of the module and of all the items in it for a module. A rejection that none of them
    /// explains fails the run: see `main.rs`.
    pub tags: BTreeSet<String>,
    /// Its source, with the attribute.
    pub source: String,
}

/// A generated case.
pub struct Case {
    pub config: Config,
    /// The Leaker's `lib.rs`, with `@ATTR@` where the macro is: `@ATTR@` is replaced.
    pub leaker: String,
    pub referrer: String,
    pub items: Vec<Item>,
    /// The line of the Referrer's `lib.rs` where each item is invoked.
    pub referrer_lines: Vec<(usize, usize)>,
    /// The `lib.rs` of the second Referrer's crate of `TYPE_LEAK_PROP_ACCEPT`, which copies the
    /// items that the Leaker accepts only by `LeakerResult::allow()` (see [`accept_referrer()`]).
    pub accept_referrer: String,
    /// The line of `accept_referrer` where each item is invoked.
    pub accept_lines: Vec<(usize, usize)>,
}

pub const ATTR: &str = "@ATTR@";

/// The scopes the macro analyzes an item through, and the name of the API of each (the statistics
/// count them apart, and the acceptance floor applies to each).
pub const SCOPES: &[(&str, &str)] = &[
    ("mod", "enter_mod"),
    ("generics", "enter_generics"),
    ("trait", "enter_trait"),
    ("impl", "enter_impl"),
];

/// The share (percent) of the items, analyzed one by one, whose predicates may name a leaked type
/// that uses a generic (`bound_cycles`). The Leaker rejects them, with one reason, "a leaked type
/// in a bound or where predicate that rustc cannot normalize": when every item had them, a
/// trait or an impl was rejected 60% of the time, and a module (which is rejected with any of its
/// dozen items) 93% (seeds 1 to 8, 60 cases).
const CYCLIC_ITEM_PERCENT: u32 = 20;
/// The share of the modules analyzed as a whole that have such items, and of the items of those.
const CYCLIC_MODULE_PERCENT: u32 = 15;
/// The share of the structs, enums, unions and traits that have parameters with defaults, if one
/// can.
const DEFAULT_PERCENT: u32 = 35;
/// The share of the cases whose Leaker does not allow the primitive types.
const PRIMITIVE_OFF_PERCENT: u32 = 20;
/// The share of the traits and of the inherent impls that have sibling methods whose generics are
/// named alike (`Gen::clash_methods()`).
const CLASH_PERCENT: u32 = 20;
const CYCLIC_ITEM_IN_MODULE_PERCENT: u32 = 30;
/// The share of the traits analyzed one by one that have the shape of `trait_self_defaults`: a
/// parameter with a bound and later ones with defaults that use it.
const SELF_DEFAULT_PERCENT: u32 = 20;
/// Of those, the share whose bound names `Self` (the rest is the control: `T: Clone`, or no bound).
const SELF_DEFAULT_NAMES_SELF_PERCENT: u32 = 70;
/// ... the share nested in `enter_mod` (the Visitor writes the default through the host), and of
/// those the share that the Referrer's crate declares again (`redeclare`).
const SELF_DEFAULT_NESTED_PERCENT: u32 = 45;
const SELF_DEFAULT_REDECLARE_PERCENT: u32 = 60;
/// The share of the structs, enums and unions analyzed one by one that have the shape of
/// `self_bound_adts`.
const SELF_BOUND_ADT_PERCENT: u32 = 10;

// ---------------------------------------------------------------------------------------------
// The model: bounds, types, the generics in scope.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Bounds {
    clone: bool,
    debug: bool,
    default: bool,
}

impl Bounds {
    fn is_none(self) -> bool {
        !self.clone && !self.debug && !self.default
    }

    fn contains(self, other: Bounds) -> bool {
        (!other.clone || self.clone)
            && (!other.debug || self.debug)
            && (!other.default || self.default)
    }

    fn only_clone() -> Self {
        Bounds {
            clone: true,
            ..Default::default()
        }
    }

    fn only_debug() -> Self {
        Bounds {
            debug: true,
            ..Default::default()
        }
    }

    fn only_default() -> Self {
        Bounds {
            default: true,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
struct TyParam {
    name: String,
    sized: bool,
    bounds: Bounds,
    /// The text of the bounds as written (`::core::clone::Clone`, `'a`, ...).
    bound_texts: Vec<String>,
    /// The lifetimes it outlives, as declared.
    outlives: Vec<String>,
    /// It is an `Iterator`: `T::Item` is a type.
    iter: bool,
    /// An iterator whose `Item` is fixed to `u8` by a supertrait (1, `supertrait_item`) or by a
    /// bound (2, `alias_item`): no predicate equates it to another type.
    fixed_item: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum SelfKind {
    /// `Self` has no meaning, or must not be named.
    #[default]
    No,
    Sized,
    /// A trait method: `Self` may be unsized, but the method can add `where Self: Sized`.
    MaybeUnsized,
    /// A trait: `Self` may be unsized and nothing can require otherwise.
    Unsized,
}

#[derive(Clone, Default, Debug)]
struct Scope {
    lifetimes: Vec<String>,
    /// `(a, b)` is `a: b`.
    outlives: Vec<(String, String)>,
    tys: Vec<TyParam>,
    consts: Vec<(String, &'static str)>,
    self_kind: SelfKind,
    self_outlives: Vec<String>,
    /// `Self::A` of a trait, with the bounds of `A`.
    assoc: Vec<(String, Bounds)>,
    /// The trait with its arguments (`Tr<'a, T>`), for `<Self as Tr<'a, T>>::A`.
    trait_ref: String,
    /// `Self` is the struct, enum or union that is declared: it must not contain itself.
    adt_self: bool,
    /// The outlives bounds of the type parameters in a field are inferred by rustc.
    infer: bool,
    /// The lifetimes of a method: only references may use them, unless `method_paths`.
    ref_only: Vec<String>,
    /// The method may name its lifetimes in leaked paths and trait objects: the feature
    /// `method_lifetime_paths`, or a method that no trait declares (a free function, a method of
    /// an inherent impl), where the copy compiles with the lifetime early-bound (F22).
    method_paths: bool,
    /// `(parameter, "iter" | "fn" | "into")`: a parameter has one `Iterator`, one `Fn` and one
    /// `Into` bound at most, or the bounds conflict (two `Into` bounds that differ by a lifetime
    /// only are ambiguous to rustc: E0283).
    bounded: Vec<(String, &'static str)>,
    /// `Self` has lifetimes where a copy writes it out: the self type of an impl that names one,
    /// an ADT with lifetime parameters (`self_elision`).
    self_lts: bool,
    /// `Self::A` has lifetimes where a copy writes it out: the trait has lifetime parameters
    /// (`<Self as Tr<'a>>::A`, `self_elision`).
    assoc_lts: bool,
    /// The types are in a where predicate: rustc proves their well-formedness, with the bounds
    /// that the signature implies as the only help (`implied_outlives`).
    pred: bool,
    /// The lifetimes `'x` of a `Self: 'x` that a where clause writes.
    self_written: Vec<String>,
    /// The parameters whose `Item` a predicate of an enclosing item names (`T: Fn(T::Item)`), and
    /// those whose `Item` it equates to a type (`T: Iterator<Item = Rel>`) (`iterator_item_cycle`).
    names_item: Vec<String>,
    binds_item: Vec<String>,
    /// Names of generics that are not in `tys` and `consts` yet, but already mean the generic where
    /// a type is named: those after a parameter in the default of its (`struct S<A = U, U = u8>`).
    reserved: Vec<String>,
    /// The types are those of a method (its parameters, its result, its where clause).
    method: bool,
}

impl Scope {
    /// Whether `name`, the name of a type that a module declares, cannot be used where the scope
    /// is: it means a generic of the scope, or it is a name of the generics of a method (`X`) and
    /// the type is not in a method. The witness and the check of a position have one function
    /// whose generics are those of the item and of the method together: a type of the item that
    /// had the name of a generic of one of its methods would mean the generic there.
    fn shadowed(&self, name: &str) -> bool {
        is_shadow_name(name)
            && (self.tys.iter().any(|t| t.name == name)
                || self.consts.iter().any(|(n, _)| n == name)
                || self.reserved.iter().any(|n| n == name)
                || (name == METHOD_SHADOW_NAME && !self.method))
    }
}

/// What a type to generate must be.
#[derive(Clone, Copy)]
struct Con {
    sized: bool,
    bounds: Bounds,
    /// No generics, no lifetime other than `'static`, no `Self`.
    closed: bool,
    /// The type is not behind a pointer: `Self` would contain itself.
    direct: bool,
    /// No type parameter and no `Self`, and no lifetime but `'static`.
    no_params: bool,
    /// No lifetime but `'static`.
    no_lifetimes: bool,
}

impl Con {
    fn sized() -> Self {
        Con {
            sized: true,
            bounds: Bounds::default(),
            closed: false,
            direct: true,
            no_params: false,
            no_lifetimes: false,
        }
    }

    fn unsized_ok() -> Self {
        Con {
            sized: false,
            ..Self::sized()
        }
    }

    fn indirect(self) -> Self {
        Con {
            direct: false,
            ..self
        }
    }
}

/// A generated type with what it mentions.
#[derive(Clone, Debug, Default)]
struct Ty {
    text: String,
    tys: BTreeSet<String>,
    lts: BTreeSet<String>,
    consts: BTreeSet<String>,
    uses_self: bool,
    /// It needs `Self: Sized`.
    self_sized: bool,
    /// Only forms for which rustc infers the outlives bounds of a field.
    infer_ok: bool,
    /// It names an associated type of a parameter (`T::Item`), which a bound may equate to
    /// another type: `T: 'a` does not tell it outlives `'a`.
    projection: bool,
    /// It names `Self` or `Self::A`, and a copy writes that with lifetimes (`self_elision`).
    adds_lts: bool,
}

impl Ty {
    fn new(text: impl Into<String>) -> Self {
        Ty {
            text: text.into(),
            infer_ok: true,
            ..Default::default()
        }
    }

    fn with(text: String, parts: &[&Ty]) -> Self {
        let mut ty = Ty::new(text);
        for part in parts {
            ty.tys.extend(part.tys.iter().cloned());
            ty.lts.extend(part.lts.iter().cloned());
            ty.consts.extend(part.consts.iter().cloned());
            ty.uses_self |= part.uses_self;
            ty.self_sized |= part.self_sized;
            ty.infer_ok &= part.infer_ok;
            ty.projection |= part.projection;
            ty.adds_lts |= part.adds_lts;
        }
        ty
    }

    fn no_infer(mut self) -> Self {
        self.infer_ok = false;
        self
    }
}

/// Whether `ty` outlives `lt`. `written`: `Self: 'x` must be written in a where clause to count (a
/// path or a trait object with a lifetime records no bound, F5, and a where predicate has no
/// receiver to imply it).
fn outlives(sc: &Scope, ty: &Ty, lt: &str, infer: bool, written: bool) -> bool {
    for p in &ty.tys {
        let declared = sc
            .tys
            .iter()
            .find(|t| &t.name == p)
            .map_or(false, |t| t.outlives.iter().any(|l| l == lt));
        if !declared && !(infer && ty.infer_ok && lt != "'static") {
            return false;
        }
    }
    for l in &ty.lts {
        if l != lt && l != "'static" && !sc.outlives.iter().any(|(a, b)| a == l && b == lt) {
            return false;
        }
    }
    let self_ok = if written {
        sc.self_written.iter().any(|l| l == lt)
    } else {
        sc.self_outlives.iter().any(|l| l == lt)
    };
    !ty.projection && (!ty.uses_self || self_ok)
}

// ---------------------------------------------------------------------------------------------
// The module environment: the names of types available in a module.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Unit,
    T1,
    T2,
    /// `Name<'a, T: ?Sized + 'a>`
    Lt,
    /// `Name<'a>`
    LOnly,
    /// `Name<T: ?Sized, const N: usize>`
    TC,
    /// `Name<const B: bool>`
    CB,
    /// `Name<const V: i32>`, `Name<const V: u8>` or `Name<const V: char>` (the type of the const
    /// parameter): the Repeater needs a slot of that type for it.
    CV(&'static str),
}

const SHAPES: &[(&str, Shape)] = &[
    ("0", Shape::Unit),
    ("1", Shape::T1),
    ("2", Shape::T2),
    ("L", Shape::Lt),
    ("C", Shape::TC),
    ("B", Shape::CB),
    ("l", Shape::LOnly),
    ("I", Shape::CV("i32")),
    ("U", Shape::CV("u8")),
    ("H", Shape::CV("char")),
];

#[derive(Clone, Debug)]
struct Head {
    name: String,
    shape: Shape,
}

/// A trait for `dyn`: no generics, one type argument, or an associated type `Out`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TShape {
    T0,
    T1,
    Out,
}

#[derive(Clone, Debug)]
struct THead {
    name: String,
    shape: TShape,
}

#[derive(Clone, Debug, Default)]
struct ModEnv {
    heads: Vec<Head>,
    theads: Vec<THead>,
    /// Paths of constants of type `usize`.
    consts: Vec<String>,
}

/// The types every module declares (`prefix` is `R`) and the `base` module has (`B`). `sized` is
/// how `?Sized` is spelled: `?::core::marker::Sized`, or `?Sized`.
fn types_template(p: &str, sized: &str) -> String {
    let mut s = String::new();
    let phantom = "::core::marker::PhantomData";
    s += &format!("#[derive(Debug, Clone, Copy, Default, PartialEq)] pub struct {p}0;\n");
    s += &format!("pub struct {p}1<T: ?::core::marker::Sized>({phantom}<*const T>);\n");
    s += &format!("pub struct {p}2<T: ?::core::marker::Sized, U: ?::core::marker::Sized>({phantom}<(*const T, *const U)>);\n");
    s += &format!("pub struct {p}L<'a, T: ?::core::marker::Sized + 'a>({phantom}<&'a T>);\n");
    s += &format!(
        "pub struct {p}C<T: ?::core::marker::Sized, const N: usize>({phantom}<*const T>);\n"
    );
    s += &format!("pub struct {p}B<const B: bool>;\n");
    s += &format!("pub struct {p}l<'a>({phantom}<&'a ()>);\n");
    for (name, ty) in [("I", "i32"), ("U", "u8"), ("H", "char")] {
        s += &format!("pub struct {p}{name}<const V: {ty}>;\n");
    }
    for (name, generics, args) in [
        ("1", "T: ?::core::marker::Sized", "T"),
        (
            "2",
            "T: ?::core::marker::Sized, U: ?::core::marker::Sized",
            "T, U",
        ),
        ("L", "'a, T: ?::core::marker::Sized + 'a", "'a, T"),
        ("C", "T: ?::core::marker::Sized, const N: usize", "T, N"),
        ("B", "const B: bool", "B"),
        ("l", "'a", "'a"),
        ("I", "const V: i32", "V"),
        ("U", "const V: u8", "V"),
        ("H", "const V: char", "V"),
    ] {
        let ty = format!("{p}{name}<{args}>");
        s += &format!(
            "impl<{generics}> ::core::clone::Clone for {ty} {{ fn clone(&self) -> {ty} {{ loop {{}} }} }}\n\
             impl<{generics}> ::core::marker::Copy for {ty} {{}}\n\
             impl<{generics}> ::core::fmt::Debug for {ty} {{ fn fmt(&self, _: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {{ loop {{}} }} }}\n\
             impl<{generics}> ::core::default::Default for {ty} {{ fn default() -> {ty} {{ loop {{}} }} }}\n"
        );
    }
    s += &format!("pub trait {p}T0 {{}}\n");
    s += &format!("pub trait {p}T1<T: ?::core::marker::Sized> {{}}\n");
    s += &format!("pub trait {p}T2 {{ type Out; }}\n");
    s += &format!("pub const {p}LEN: usize = 2;\n");
    s.replace("?::core::marker::Sized", sized)
}

/// Whether `name`, a type of the module environment, is one that a module declares (`R0`, `self::R1`,
/// `U`, ...) and not one it imports or names by a path of another module.
fn is_declared(name: &str) -> bool {
    name.starts_with('R')
        || name.starts_with("self::R")
        || name.starts_with("super::R")
        || is_shadow_name(name)
        || name.strip_prefix("self::").map_or(false, is_shadow_name)
}

/// The types that every module declares with the name of a generic parameter (`T`, `U`: the
/// parameters of an item, `X`: of a method). Where a generic of that name is in scope, the name is
/// the generic's, and the type cannot be named (`shadowed()`); where none is, it is a relative
/// type, which must not be taken for the generic of a sibling scope.
const SHADOW_NAMES: &[&str] = &["T", "U", "X"];

/// The one of `SHADOW_NAMES` that a method declares as a generic: its type is named only in methods.
const METHOD_SHADOW_NAME: &str = "X";

fn is_shadow_name(name: &str) -> bool {
    SHADOW_NAMES.contains(&name)
}

/// The full path of a trait of the prelude that the generator spells bare in bounds.
fn absolute_trait(name: &str) -> &'static str {
    match name {
        "Sized" => "::core::marker::Sized",
        "Send" => "::core::marker::Send",
        "Sync" => "::core::marker::Sync",
        "Unpin" => "::core::marker::Unpin",
        "Copy" => "::core::marker::Copy",
        "Clone" => "::core::clone::Clone",
        "Default" => "::core::default::Default",
        "Fn" => "::core::ops::Fn",
        "FnMut" => "::core::ops::FnMut",
        "FnOnce" => "::core::ops::FnOnce",
        "Into" => "::core::convert::Into",
        "From" => "::core::convert::From",
        "AsRef" => "::core::convert::AsRef",
        "Iterator" => "::core::iter::Iterator",
        "PartialEq" => "::core::cmp::PartialEq",
        "PartialOrd" => "::core::cmp::PartialOrd",
        "Eq" => "::core::cmp::Eq",
        "Ord" => "::core::cmp::Ord",
        other => panic!("generator: `{}` is not a prelude trait it writes", other),
    }
}

/// The marker traits (and comparisons) that a type parameter may have as a bound, each of which a
/// type with no more than its own bounds satisfies once the impls copy the generics. `Copy` needs
/// a `Sized` parameter.
const MARKER_BOUNDS: &[&str] = &[
    "Send",
    "Sync",
    "Unpin",
    "Copy",
    "PartialEq",
    "Eq",
    "PartialOrd",
    "Ord",
];

// ---------------------------------------------------------------------------------------------
// Generics.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Item,
    Method,
}

/// The generics of an item or a method.
#[derive(Clone, Debug, Default)]
struct Skeleton {
    lifetimes: Vec<String>,
    outlives: Vec<(String, String)>,
    tys: Vec<TyParam>,
    consts: Vec<(String, &'static str)>,
    preds: Vec<String>,
    bounded: Vec<(String, &'static str)>,
    /// A where predicate names a lifetime and an external type (`Risk::cycle`).
    risk_cycle: bool,
    /// The tags of the struct, enum, union or trait that has these generics (`Item::tags`): an
    /// impl of it has its generics, predicates and methods, so its shapes.
    tags: BTreeSet<String>,
    /// The defaults of the trailing type and const parameters, by name (`T = Vec<u8>`, `const N:
    /// usize = 3`). Only the declaration of a struct, enum, union or trait writes them
    /// (`params_decl()`): an impl or a function cannot.
    defaults: Vec<(String, String)>,
}

/// What makes an item break under a strict Referrer, which replaces the optional entries
/// (`strict_cycles`, `strict_anchors`): the item is not invoked as strict then, unless a feature
/// asks for it.
#[derive(Clone, Copy, Default, Debug)]
struct Risk {
    /// A where predicate names a lifetime and an absolute type of a crate that the Leaker does not
    /// allow.
    cycle: bool,
    /// A method names a lifetime of its own inside such a type.
    anchor: bool,
}

impl Skeleton {
    /// `<'a, 'b: 'a, T: ?Sized + Clone + 'a, const N: usize>`
    fn params(&self) -> String {
        self.params_with(false)
    }

    /// `params()` with the defaults of the declaration of a type or a trait.
    fn params_decl(&self) -> String {
        self.params_with(true)
    }

    fn params_with(&self, defaults: bool) -> String {
        let default_of = |name: &str| -> String {
            match self.defaults.iter().find(|(n, _)| n == name) {
                Some((_, value)) if defaults => format!(" = {}", value),
                _ => String::new(),
            }
        };
        let mut parts = Vec::new();
        for l in &self.lifetimes {
            let bounds: Vec<&str> = self
                .outlives
                .iter()
                .filter(|(a, _)| a == l)
                .map(|(_, b)| b.as_str())
                .collect();
            if bounds.is_empty() {
                parts.push(l.clone());
            } else {
                parts.push(format!("{}: {}", l, bounds.join(" + ")));
            }
        }
        for t in &self.tys {
            if t.bound_texts.is_empty() {
                parts.push(format!("{}{}", t.name, default_of(&t.name)));
            } else {
                parts.push(format!(
                    "{}: {}{}",
                    t.name,
                    t.bound_texts.join(" + "),
                    default_of(&t.name)
                ));
            }
        }
        for (n, ty) in &self.consts {
            parts.push(format!("const {}: {}{}", n, ty, default_of(n)));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!("<{}>", parts.join(", "))
        }
    }

    /// `<'a, T, N>`
    fn args(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        parts.extend(self.lifetimes.iter().cloned());
        parts.extend(self.tys.iter().map(|t| t.name.clone()));
        parts.extend(self.consts.iter().map(|(n, _)| n.clone()));
        if parts.is_empty() {
            String::new()
        } else {
            format!("<{}>", parts.join(", "))
        }
    }

    fn where_clause(&self) -> String {
        if self.preds.is_empty() {
            String::new()
        } else {
            format!(" where {}", self.preds.join(", "))
        }
    }

    /// The scope with these generics added.
    fn scope(&self, outer: &Scope) -> Scope {
        let mut sc = outer.clone();
        sc.lifetimes.extend(self.lifetimes.iter().cloned());
        sc.outlives.extend(self.outlives.iter().cloned());
        sc.tys.extend(self.tys.iter().cloned());
        sc.consts.extend(self.consts.iter().cloned());
        sc.bounded.extend(self.bounded.iter().cloned());
        for t in &self.tys {
            if self.preds.iter().any(|p| names_item(p, &t.name)) {
                sc.names_item.push(t.name.clone());
            }
            if self.preds.iter().any(|p| binds_item(p, &t.name)) {
                sc.binds_item.push(t.name.clone());
            }
        }
        sc
    }
}

// ---------------------------------------------------------------------------------------------
// The generator.
// ---------------------------------------------------------------------------------------------

/// A trait declared in a module, which an impl can implement.
#[derive(Clone)]
struct TraitInfo {
    name: String,
    sk: Skeleton,
    supers: Bounds,
    assoc: Vec<(String, Bounds)>,
    /// The trait has a marker supertrait (`Send`): an impl would need a self type that has it.
    marker_super: bool,
    /// The trait declares `type G<'g>;`.
    gat: bool,
    /// ... with the where clause `where Self: 'g, Self: 'c` (`gat_where_clause`).
    gat_where: bool,
    /// `const C0: ::core::marker::PhantomData<..>` or `const K0: usize`, with its value in an impl.
    consts: Vec<(String, String)>,
    /// The signatures of the methods, with their where clauses.
    methods: Vec<String>,
    implemented: bool,
    /// A method or a constant names `Self::A`.
    uses_assoc_self: bool,
    /// A method has `where Self: 'x`, or a receiver `&'x self` and names `Self`.
    uses_self_outlives: bool,
    /// A where predicate of a method names `Self::A` (`impl_pred_leaked`).
    self_assoc_in_pred: bool,
    /// The where clause of the trait names a type that may be leaked (`impl_pred_leaked`).
    preds_leakable: bool,
    /// An elided `fn(&Self) -> &R` or `Fn(..)` is in it (`self_elision`).
    elided_self: bool,
    risk: Risk,
}

pub struct Gen {
    /// The state of the random numbers: each one is the hash of the seed and of what was drawn
    /// before (`finish()` does not reset a `DefaultHasher`).
    rng: DefaultHasher,
    cfg: Config,
    features: Features,
    env: ModEnv,
    no_self: bool,
    /// A module analyzed as a whole is being generated: its copy declares its own items.
    module_mode: bool,
    /// Only absolute paths are written (no relative name, nor one of the prelude).
    abs_only: bool,
    /// No `T::Item`: it does not constrain `T` in the self type of an impl.
    no_assoc_paths: bool,
    /// The probability of `Self` where it is valid, in percent.
    self_prob: u32,
    next_id: usize,
    /// A counter for the names of lifetimes bound by `for<..>`.
    next_binder: usize,
    /// The id of the item being generated, which its names carry.
    cur_id: usize,
    /// A counter for the names of the declarations in an item or a module.
    next_name: usize,
    traits: Vec<TraitInfo>,
    /// Structs, enums and unions declared in the module, with the generics they declare.
    adts: Vec<(String, Skeleton)>,
    items: Vec<Item>,
    /// The kinds of the items in the module analyzed as a whole.
    contents: Vec<&'static str>,
    /// An elided `fn(&A) -> &R` or `Fn(&A) -> &R` with `Self` in `A` was generated since it was
    /// cleared (`self_elision`).
    elided_self: bool,
    /// What the item being generated has that breaks under a strict Referrer.
    risk: Risk,
    /// The item being generated may have the shapes that the Leaker rejects: a predicate that
    /// names a leaked type that uses a generic (`bound_cycles`), a bound of a generic that names
    /// `Self` with a predicate that uses the generic (`impl_self_bound_cycle`,
    /// `trait_assoc_bound_cycle`), an `impl Trait` argument (`impl_trait_args`), an iterator whose
    /// `Item` a predicate equates to a leaked type (`iterator_item_cycle`, `item_predicate_cycle`,
    /// `supertrait_item`). It is a share of the items, which keeps the shapes generated without
    /// making most items (and nearly every module) rejected.
    cyclic: bool,
    /// The module analyzed as a whole that is being generated has items with such predicates.
    cyclic_module: bool,
    /// The scope of the item that was generated last (`Item::scope`).
    last_scope: &'static str,
    /// The tags of the item (or of the module analyzed as a whole) being generated.
    tags: BTreeSet<String>,
    /// The share (percent) of the prelude traits in the bounds of the item being generated that
    /// are written bare (`Sized`, `Fn(..)`, `Into<..>`, ...) instead of in full. It is the item's:
    /// with the prelude off the bare names are rejected, so few items have any (`bare_trait()`).
    bare_percent: u32,
    /// The item being generated is invoked as strict if it can be (`strict_ok()`).
    want_strict: bool,
    /// Whether the attribute of the item generated last says `strict`.
    last_strict: bool,
    /// The self types of the impls of the traits with associated types in the crate: `Self::A0`
    /// is ambiguous (E0223) for a type that two such traits are implemented for, even if they are
    /// in different modules.
    assoc_self_types: Vec<String>,
    /// A default of a parameter is being generated.
    in_default: bool,
    /// The attribute of the item being generated says `redeclare = true`: the Referrer's crate
    /// declares the trait again (`trait_self_defaults`).
    redeclare: bool,
    /// The `use`s of each module whose items are analyzed one by one, by its path (the imports that
    /// the names `I1`, `J1`, `ILEN` come from), for `accept_referrer()`. They draw no random number.
    mod_uses: Vec<(Vec<String>, String)>,
}

const PRIMS: &[&str] = &[
    "u8", "u16", "u32", "i32", "i64", "usize", "bool", "char", "f64", "()",
];

impl Gen {
    pub fn new(seed: u64, krate: String, features: Features) -> Self {
        let mut rng = DefaultHasher::new();
        rng.write_u64(seed);
        let cfg = Config {
            krate,
            prelude: false,
            allow_std: false,
            allow_crate: false,
            primitive: true,
        };
        let mut g = Gen {
            rng,
            cfg,
            features,
            env: ModEnv::default(),
            no_self: false,
            module_mode: false,
            abs_only: false,
            no_assoc_paths: false,
            self_prob: 30,
            next_id: 0,
            next_binder: 0,
            cur_id: 0,
            next_name: 0,
            traits: Vec::new(),
            adts: Vec::new(),
            items: Vec::new(),
            contents: Vec::new(),
            elided_self: false,
            risk: Risk::default(),
            cyclic: false,
            cyclic_module: false,
            last_scope: "",
            tags: BTreeSet::new(),
            bare_percent: 0,
            want_strict: false,
            last_strict: false,
            assoc_self_types: Vec::new(),
            in_default: false,
            redeclare: false,
            mod_uses: Vec::new(),
        };
        g.cfg.prelude = g.chance(50);
        g.cfg.allow_std = g.chance(35);
        g.cfg.allow_crate = g.chance(30);
        g.cfg.primitive = !g.chance(PRIMITIVE_OFF_PERCENT);
        g
    }

    /// The next random number: the hash of the seed and of what was drawn before.
    fn next(&mut self) -> u64 {
        self.rng.write_u64(0);
        self.rng.finish()
    }

    /// The predicates of the item may name a leaked type that uses a generic.
    fn bound_cycles(&self) -> bool {
        self.cyclic && self.features.bound_cycles
    }

    /// A number in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// `true` with the probability `percent`%.
    fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent as usize
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    fn binder(&mut self) -> String {
        self.next_binder += 1;
        format!("'h{}", self.next_binder)
    }

    /// A type of the module that names the lifetime `h` of a binder (`Rl<'h>`), sometimes.
    fn binder_path(&mut self, h: &str) -> Option<String> {
        let env = self.env.clone();
        self.binder_path_in(&env, h)
    }

    /// The same with the types of `env`. The relative ones: an absolute path is not what is tested.
    fn binder_path_in(&mut self, env: &ModEnv, h: &str) -> Option<String> {
        let heads: Vec<String> = env
            .heads
            .iter()
            .filter(|head| head.shape == Shape::LOnly && !head.name.starts_with("::"))
            .filter(|head| !(self.module_mode && is_declared(&head.name)))
            .map(|head| head.name.clone())
            .collect();
        if heads.is_empty() || !self.chance(40) {
            return None;
        }
        Some(format!("{}<{}>", self.pick(&heads), h))
    }

    fn name(&mut self, prefix: &str) -> String {
        self.next_name += 1;
        format!("{}{}_{}", prefix, self.cur_id, self.next_name)
    }

    // -----------------------------------------------------------------------------------------
    // Names.
    // -----------------------------------------------------------------------------------------

    fn bare(&mut self) -> bool {
        let p = if self.cfg.prelude { 35 } else { 4 };
        let chosen = self.chance(p);
        // The names of the prelude are not leaked if the Leaker allows them.
        chosen && (!self.abs_only || self.cfg.prelude)
    }

    /// The share of bare prelude traits in the bounds of a new item (see `Gen::bare_percent`).
    ///
    /// With the prelude allowed a bare name is as good as the full path, and the macro checks that
    /// (`bare.rs`): many items have some. Without it the item is rejected, so few do.
    fn item_bare_percent(&mut self) -> u32 {
        if self.cfg.prelude {
            if self.chance(55) {
                50
            } else {
                0
            }
        } else if self.chance(8) {
            40
        } else {
            0
        }
    }

    /// The same for a module analyzed as a whole, where one bare name rejects the module.
    fn module_bare_percent(&mut self) -> u32 {
        if self.cfg.prelude {
            50
        } else if self.chance(10) {
            35
        } else {
            0
        }
    }

    /// A trait of the prelude in a bound, a where clause, a supertrait or a trait object, bare
    /// (`Into`) or in full (`::core::convert::Into`). Bare without the prelude is a relative path
    /// that the Leaker rejects, and the tag `bare:<name>` says so.
    fn trait_path(&mut self, name: &'static str) -> String {
        if self.bare_percent > 0 && self.chance(self.bare_percent) {
            if !self.cfg.prelude {
                self.tags.insert(format!("bare:{}", name));
            }
            name.to_string()
        } else {
            absolute_trait(name).to_string()
        }
    }

    /// `?Sized`, bare or in full (`?::core::marker::Sized`). A bare one is a relative path without
    /// the prelude, unless `BARE_RELAXED_SIZED_ALWAYS_OK`.
    fn maybe_sized(&mut self) -> String {
        if self.bare_percent > 0 && self.chance(self.bare_percent) {
            if !self.cfg.prelude && !BARE_RELAXED_SIZED_ALWAYS_OK {
                self.tags.insert("bare:?Sized".to_string());
            }
            "?Sized".to_string()
        } else {
            "?::core::marker::Sized".to_string()
        }
    }

    /// `Fn`, mostly, or `FnMut` or `FnOnce`.
    fn fn_family(&mut self) -> &'static str {
        match self.below(10) {
            0 => "FnMut",
            1 => "FnOnce",
            _ => "Fn",
        }
    }

    fn clone_path(&mut self) -> String {
        self.trait_path("Clone")
    }

    fn default_path(&mut self) -> String {
        self.trait_path("Default")
    }

    fn bound_texts(&mut self, b: Bounds) -> Vec<String> {
        let mut texts = Vec::new();
        if b.clone {
            texts.push(self.clone_path());
        }
        if b.debug {
            texts.push("::core::fmt::Debug".to_string());
        }
        if b.default {
            texts.push(self.default_path());
        }
        texts
    }

    // -----------------------------------------------------------------------------------------
    // Types.
    // -----------------------------------------------------------------------------------------

    fn leaf(&mut self, sc: &Scope, c: Con) -> Ty {
        for _ in 0..8 {
            let pick = self.below(100);
            let ty = match pick {
                0..=29 => Some(Ty::new(*self.pick(PRIMS))),
                30..=34 => {
                    let bare = self.bare();
                    if !bare && self.abs_only && !self.cfg.allow_std {
                        None
                    } else {
                        Some(Ty::new(if bare {
                            "String"
                        } else {
                            "::std::string::String"
                        }))
                    }
                }
                35..=54 => self.head_unit(sc),
                55..=79 => self.param(sc, c),
                80..=99 => self.self_ty(sc, c),
                _ => None,
            };
            if let Some(ty) = ty {
                return ty;
            }
        }
        Ty::new("u8")
    }

    fn head_unit(&mut self, sc: &Scope) -> Option<Ty> {
        let units: Vec<String> = self
            .env
            .heads
            .iter()
            .filter(|h| h.shape == Shape::Unit && !sc.shadowed(&h.name))
            .map(|h| h.name.clone())
            .collect();
        if units.is_empty() {
            return None;
        }
        Some(Ty::new(self.pick(&units).clone()))
    }

    fn param(&mut self, sc: &Scope, c: Con) -> Option<Ty> {
        if c.closed || c.no_params || sc.tys.is_empty() {
            return None;
        }
        let candidates: Vec<&TyParam> = sc
            .tys
            .iter()
            .filter(|p| (p.sized || !c.sized) && p.bounds.contains(c.bounds))
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let p = (*self.pick(&candidates)).clone();
        // `T::Item` and `<T as Iterator>::Item` are associated type paths.
        if p.iter && c.bounds.is_none() && !self.no_assoc_paths && self.chance(40) {
            let text = if self.chance(50) {
                format!("{}::Item", p.name)
            } else {
                let iterator = self.trait_path("Iterator");
                format!("<{} as {}>::Item", p.name, iterator)
            };
            let mut ty = Ty::new(text).no_infer();
            ty.tys.insert(p.name);
            ty.projection = true;
            return Some(ty);
        }
        let mut ty = Ty::new(p.name.clone());
        ty.tys.insert(p.name);
        Some(ty)
    }

    fn self_ty(&mut self, sc: &Scope, c: Con) -> Option<Ty> {
        if self.no_self || c.closed || c.no_params || !c.bounds.is_none() {
            return None;
        }
        if sc.assoc.is_empty() && sc.self_kind == SelfKind::No {
            return None;
        }
        if !self.chance(self.self_prob) {
            return None;
        }
        // `Self::A`
        if !sc.assoc.is_empty() && self.chance(40) {
            let (name, _) = self.pick(&sc.assoc).clone();
            let mut ty = Ty::new(format!("Self::{}", name)).no_infer();
            ty.uses_self = true;
            ty.adds_lts = sc.assoc_lts || sc.self_lts;
            // `Self::A: 'x` needs `Self: 'x` and the parameters of the trait to outlive `'x` too.
            // A signature implies it for its types, but not a where predicate: and the Repeater's
            // impl has neither (`implied_outlives`).
            ty.projection = sc.pred || !self.features.implied_outlives;
            return Some(ty);
        }
        let mut ty = Ty::new("Self");
        ty.uses_self = true;
        ty.adds_lts = sc.self_lts;
        match sc.self_kind {
            SelfKind::No => return None,
            SelfKind::Sized => {
                if sc.adt_self && c.direct {
                    return None;
                }
            }
            SelfKind::MaybeUnsized => ty.self_sized = c.sized,
            SelfKind::Unsized => {
                if c.sized {
                    return None;
                }
            }
        }
        Some(ty)
    }

    fn ty(&mut self, sc: &Scope, c: Con, depth: u32) -> Ty {
        if depth >= 3 || self.chance(30) {
            return self.leaf(sc, c);
        }
        for _ in 0..6 {
            let pick = self.below(100);
            let ty = match pick {
                0..=10 => self.tuple(sc, c, depth),
                11..=17 => self.array(sc, c, depth),
                18..=34 => self.std_wrap(sc, c, depth),
                35..=40 => self.std_two(sc, c, depth),
                41..=54 => self.head_app(sc, c, depth),
                55..=64 => self.reference(sc, c, depth),
                65..=68 => self.pointer(sc, c, depth),
                69..=74 => self.fn_ptr(sc, c, depth),
                75..=84 => self.dyn_obj(sc, c, depth),
                85..=89 if !c.sized => self.slice_or_str(sc, c, depth),
                _ => None,
            };
            if let Some(ty) = ty {
                return ty;
            }
        }
        self.leaf(sc, c)
    }

    fn tuple(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        let n = self.range(2, 3);
        let elem = Con { sized: true, ..c };
        let parts: Vec<Ty> = (0..n).map(|_| self.ty(sc, elem, depth + 1)).collect();
        let text = format!(
            "({})",
            parts
                .iter()
                .map(|t| t.text.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Some(Ty::with(text, &parts.iter().collect::<Vec<_>>()))
    }

    fn array(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        let elem = Con { sized: true, ..c };
        let inner = self.ty(sc, elem, depth + 1);
        let len = self.array_len(sc, c.bounds.default || c.no_params || c.closed);
        let text = format!("[{}; {}]", inner.text, len);
        let mut ty = Ty::with(text, &[&inner]);
        if sc.consts.iter().any(|(n, _)| *n == len) {
            ty.consts.insert(len);
        }
        Some(ty)
    }

    fn array_len(&mut self, sc: &Scope, literal_only: bool) -> String {
        let usizes: Vec<&String> = sc
            .consts
            .iter()
            .filter(|(_, ty)| *ty == "usize")
            .map(|(n, _)| n)
            .collect();
        if !literal_only && !usizes.is_empty() && self.chance(45) {
            return (*self.pick(&usizes)).clone();
        }
        if !self.env.consts.is_empty() && self.chance(20) {
            let consts = self.env.consts.clone();
            let constant = self.pick(&consts).clone();
            if self.in_default && !constant.starts_with("::") {
                self.tags.insert(TAG_CONST_DEFAULT.to_string());
            }
            return constant;
        }
        self.pick(&["2", "3", "0", "1"]).to_string()
    }

    fn std_wrap(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        // (names, argument may be unsized, behind a pointer, ok with the bounds)
        let which = self.below(7);
        let (name, unsized_ok, indirect): (String, bool, bool) = match which {
            0 => (
                if self.abs_only {
                    if self.cfg.prelude && self.chance(30) {
                        "Option".to_string()
                    } else {
                        "::core::option::Option".to_string()
                    }
                } else {
                    self.pick(&["::core::option::Option", "::std::option::Option", "Option"])
                        .to_string()
                },
                false,
                false,
            ),
            1 => (
                if self.bare() {
                    "Vec".into()
                } else {
                    "::std::vec::Vec".into()
                },
                false,
                true,
            ),
            2 => (
                if self.bare() {
                    "Box".into()
                } else {
                    "::std::boxed::Box".into()
                },
                true,
                true,
            ),
            3 => ("::std::rc::Rc".into(), true, true),
            4 => ("::std::sync::Arc".into(), true, true),
            5 => {
                if !c.bounds.clone && !c.bounds.debug {
                    ("::core::cell::Cell".into(), false, false)
                } else {
                    ("::std::vec::Vec".into(), false, true)
                }
            }
            _ => ("::core::marker::PhantomData".into(), true, true),
        };
        if name == "Option" && !self.cfg.prelude && !self.chance(15) {
            return None;
        }
        // A path of `::std` is an optional entry unless the Leaker allows the crate.
        if self.abs_only && name.starts_with("::std") && !self.cfg.allow_std {
            return None;
        }
        let allow_unsized = unsized_ok && c.bounds.is_none();
        let inner_con = Con {
            sized: !allow_unsized,
            direct: c.direct && !indirect,
            ..c
        };
        let inner_con = if name.ends_with("PhantomData") {
            Con {
                bounds: Bounds::default(),
                ..inner_con
            }
        } else {
            inner_con
        };
        let inner = self.ty(sc, inner_con, depth + 1);
        Some(Ty::with(format!("{}<{}>", name, inner.text), &[&inner]))
    }

    fn std_two(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        let (name, ok) = match self.below(3) {
            0 => (
                if self.bare() {
                    "Result"
                } else {
                    "::core::result::Result"
                },
                !c.bounds.default,
            ),
            1 => ("::std::collections::BTreeMap", true),
            _ => ("::std::collections::HashMap", true),
        };
        if !ok || (name == "Result" && !self.cfg.prelude && !self.chance(20)) {
            return None;
        }
        if self.abs_only && name.starts_with("::std") && !self.cfg.allow_std {
            return None;
        }
        let inner = Con { sized: true, ..c };
        let direct = name.ends_with("Result");
        let inner = Con {
            direct: c.direct && direct,
            ..inner
        };
        let a = self.ty(sc, inner, depth + 1);
        let b = self.ty(sc, inner, depth + 1);
        Some(Ty::with(
            format!("{}<{}, {}>", name, a.text, b.text),
            &[&a, &b],
        ))
    }

    /// A lifetime to use in `sc`, or `None` if the type must not use one.
    ///
    /// `for_ref`: the lifetime of a reference. The lifetimes of a method of a trait are used by
    /// references only: a leaked path or trait object that names one gives a method that does not
    /// match the trait in the Referrer's crate (E0195, `method_lifetime_paths`). Those of a free
    /// function and of a method of an inherent impl may be used anywhere (F22).
    fn lifetime(&mut self, sc: &Scope, c: Con, for_ref: bool) -> String {
        let pool: Vec<&String> = sc
            .lifetimes
            .iter()
            .filter(|l| for_ref || sc.method_paths || !sc.ref_only.contains(*l))
            .collect();
        if c.closed || c.no_params || c.no_lifetimes || pool.is_empty() || self.chance(15) {
            "'static".to_string()
        } else {
            let lifetime = (*self.pick(&pool)).clone();
            if !for_ref && sc.ref_only.contains(&lifetime) {
                self.tags.insert(TAG_METHOD_LIFETIME.to_string());
            }
            lifetime
        }
    }

    /// A type of the standard library that hides a lifetime in its path (`hidden_lifetimes`).
    fn hidden_path(&mut self) -> &'static str {
        // Without the primitive types allowed, `u8` is a relative path: `Iter<u8>` is leaked as
        // an entry, and its hidden lifetime is the one that "is not seen" (the docs of the Leaker
        // say to write `Iter<'_, u8>`).
        if !self.cfg.primitive {
            return "::core::str::Chars";
        }
        self.pick(&["::core::str::Chars", "::core::slice::Iter<u8>"])
    }

    fn head_app(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if self.env.heads.is_empty() {
            return None;
        }
        let head = self.pick(&self.env.heads.clone()).clone();
        if sc.shadowed(&head.name) {
            return None;
        }
        let inner = Con {
            sized: false,
            bounds: Bounds::default(),
            direct: false,
            ..c
        };
        match head.shape {
            Shape::Unit => Some(Ty::new(head.name)),
            Shape::T1 => {
                let a = self.ty(sc, inner, depth + 1);
                Some(Ty::with(format!("{}<{}>", head.name, a.text), &[&a]))
            }
            Shape::T2 => {
                let a = self.ty(sc, inner, depth + 1);
                let b = self.ty(sc, inner, depth + 1);
                Some(Ty::with(
                    format!("{}<{}, {}>", head.name, a.text, b.text),
                    &[&a, &b],
                ))
            }
            Shape::Lt => {
                let lt = self.lifetime(sc, c, false);
                let a = self.ty(sc, inner, depth + 1);
                // The bound `T: 'a` of the definition must be written, as the `Leaker` docs say: a
                // path with a lifetime records no bound, so `Self: 'x` that only the receiver
                // implies is not enough (F5).
                if !outlives(sc, &a, &lt, false, true) {
                    return None;
                }
                let mut ty =
                    Ty::with(format!("{}<{}, {}>", head.name, lt, a.text), &[&a]).no_infer();
                if lt != "'static" {
                    ty.lts.insert(lt);
                }
                Some(ty)
            }
            Shape::LOnly => {
                let lt = self.lifetime(sc, c, false);
                let mut ty = Ty::new(format!("{}<{}>", head.name, lt));
                if lt != "'static" {
                    ty.lts.insert(lt);
                }
                Some(ty)
            }
            Shape::TC => {
                let a = self.ty(sc, inner, depth + 1);
                let len = self.array_len(sc, c.no_params || c.closed);
                let arg = self.const_arg(&head.name, &len, &sc.consts);
                let mut ty = Ty::with(format!("{}<{}, {}>", head.name, a.text, arg), &[&a]);
                if sc.consts.iter().any(|(n, _)| *n == len) {
                    ty.consts.insert(len);
                }
                Some(ty)
            }
            Shape::CB => {
                let bools: Vec<&String> = sc
                    .consts
                    .iter()
                    .filter(|(_, ty)| *ty == "bool")
                    .map(|(n, _)| n)
                    .collect();
                let value = if !bools.is_empty() && !c.no_params && !c.closed && self.chance(60) {
                    (*self.pick(&bools)).clone()
                } else {
                    self.pick(&["true", "false"]).to_string()
                };
                let mut ty = Ty::new(format!("{}<{}>", head.name, value));
                if sc.consts.iter().any(|(n, _)| *n == value) {
                    ty.consts.insert(value);
                }
                Some(ty)
            }
            Shape::CV(const_ty) => {
                let params: Vec<&String> = sc
                    .consts
                    .iter()
                    .filter(|(_, ty)| *ty == const_ty)
                    .map(|(n, _)| n)
                    .collect();
                let value = if !params.is_empty() && !c.no_params && !c.closed && self.chance(60) {
                    (*self.pick(&params)).clone()
                } else {
                    self.const_literal(const_ty)
                };
                let mut ty = Ty::new(format!("{}<{}>", head.name, value));
                if sc.consts.iter().any(|(n, _)| *n == value) {
                    ty.consts.insert(value);
                }
                Some(ty)
            }
        }
    }

    fn reference(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if !c.bounds.is_none() {
            return None;
        }
        let lt = self.lifetime(sc, c, true);
        let inner_con = Con {
            sized: false,
            direct: false,
            ..c
        };
        let a = self.ty(sc, inner_con, depth + 1);
        // A where predicate has no help of the receiver: `&'x Self` there needs `Self: 'x` written
        // (the witness of the Leaker and the check of the Referrer have no receiver).
        let written = sc.pred;
        if !outlives(sc, &a, &lt, sc.infer, written) {
            return None;
        }
        let m = if self.chance(25) { "mut " } else { "" };
        let mut ty = Ty::with(format!("&{} {}{}", lt, m, paren(&a.text)), &[&a]);
        if lt != "'static" {
            ty.lts.insert(lt);
        }
        Some(ty)
    }

    fn pointer(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if !c.bounds.is_none() {
            return None;
        }
        let inner_con = Con {
            sized: false,
            direct: false,
            ..c
        };
        let a = self.ty(sc, inner_con, depth + 1);
        let m = if self.chance(50) { "const" } else { "mut" };
        Some(Ty::with(format!("*{} {}", m, paren(&a.text)), &[&a]))
    }

    fn fn_ptr(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if !c.bounds.is_none() {
            return None;
        }
        let arg = Con {
            sized: true,
            direct: false,
            ..c
        };
        let prefix = *self.pick(&["", "", "unsafe ", "extern \"C\" ", "unsafe extern \"C\" "]);
        // `elided_fn_generics` asks for `fn(&A) -> &R` too: the elided lifetime of the input is the
        // output's, so the Leaker leaks the function pointer as a whole where `A` has lifetimes.
        if self.features.elided_fn_generics && self.chance(25) {
            // `hidden_lifetimes` asks for the lifetime of the input to be hidden by a path: `fn(Chars,
            // A) -> &R`, which `A` must be `Sized` for.
            let hidden = self.features.hidden_lifetimes && self.chance(50);
            let a = self.ty(
                sc,
                if hidden {
                    Con::sized().with_scope(c)
                } else {
                    Con::unsized_ok().with_scope(c)
                },
                depth + 1,
            );
            let r = self.ty(sc, Con::unsized_ok().with_scope(c), depth + 1);
            // The copy writes `Self` with lifetimes if it has any (`self_elision`), so `Self` is
            // only named in the scopes that are not a trait method (a struct, an enum, a union, an
            // impl) with that feature.
            if !a.text.contains(['\'', '&'])
                && !(a.uses_self
                    && sc.self_kind != SelfKind::MaybeUnsized
                    && !self.features.self_elision)
                && (self.features.self_elision || !a.adds_lts)
            {
                self.elided_self |= a.uses_self;
                if hidden && a.uses_self {
                    self.tags.insert(TAG_HIDDEN_SELF.to_string());
                }
                let input = if hidden {
                    format!("{}, {}", self.hidden_path(), a.text)
                } else {
                    format!("&{}", paren(&a.text))
                };
                let text = format!("{}fn({}) -> &{}", prefix, input, paren(&r.text));
                return Some(Ty::with(text, &[&a, &r]).no_infer());
            }
        }
        let n = self.range(0, 2);
        let mut parts: Vec<Ty> = Vec::new();
        let (text, hr) = if self.chance(40) {
            // `for<'h> fn(&'h A, B) -> &'h R`: an input keeps `'h`.
            let h = self.binder();
            // A type of the module with the lifetime of the binder: `Rl<'h>`. Some have no input
            // that keeps the lifetime (`for<'h> fn(A, Rl<'h>) -> &'h R`): the Leaker leaks the whole
            // function pointer then.
            let rl = self.binder_path(&h);
            let unanchored = rl.is_some() && self.chance(30);
            if unanchored {
                // Leaked as a whole it is fine, but where the Leaker must split it (it names a
                // lifetime of the method too) it is the error of a binder that no input keeps.
                self.tags.insert(TAG_BINDER_NO_ANCHOR.to_string());
            }
            let a = self.ty(
                sc,
                if unanchored {
                    Con::sized().with_scope(c)
                } else {
                    Con::unsized_ok().with_scope(c)
                },
                depth + 1,
            );
            let mut args = vec![if unanchored {
                a.text.clone()
            } else {
                format!("&{} {}", h, paren(&a.text))
            }];
            parts.push(a);
            args.extend(rl);
            for _ in 0..n {
                let t = self.ty(sc, arg, depth + 1);
                args.push(t.text.clone());
                parts.push(t);
            }
            let ret = if self.chance(60) {
                let r = self.ty(sc, Con::unsized_ok().with_scope(c), depth + 1);
                let text = format!(" -> &{} {}", h, paren(&r.text));
                parts.push(r);
                text
            } else if self.chance(50) {
                let r = self.ty(sc, arg, depth + 1);
                let text = format!(" -> {}", r.text);
                parts.push(r);
                text
            } else {
                String::new()
            };
            (
                format!("for<{}> {}fn({}){}", h, prefix, args.join(", "), ret),
                true,
            )
        } else {
            let mut args = Vec::new();
            for _ in 0..n {
                let t = self.ty(sc, arg, depth + 1);
                args.push(t.text.clone());
                parts.push(t);
            }
            let ret = if self.chance(50) {
                let r = self.ty(sc, arg, depth + 1);
                let text = format!(" -> {}", r.text);
                parts.push(r);
                text
            } else {
                String::new()
            };
            (format!("{}fn({}){}", prefix, args.join(", "), ret), false)
        };
        let _ = hr;
        Some(Ty::with(text, &parts.iter().collect::<Vec<_>>()).no_infer())
    }

    fn slice_or_str(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if self.chance(30) {
            return Some(Ty::new("str"));
        }
        let elem = Con {
            sized: true,
            direct: false,
            ..c
        };
        let a = self.ty(sc, elem, depth + 1);
        Some(Ty::with(format!("[{}]", a.text), &[&a]))
    }

    /// `dyn Trait`, with arguments that do not depend on the generics (`open` allows them).
    fn dyn_bare(&mut self, sc: &Scope, depth: u32, open: bool) -> Ty {
        let closed = Con {
            sized: true,
            bounds: Bounds::default(),
            closed: !open,
            direct: false,
            no_params: false,
            no_lifetimes: false,
        };
        let mut parts: Vec<Ty> = Vec::new();
        // `elided_fn_generics` asks for the elided form, with generics in the input.
        let pick = if self.features.elided_fn_generics && open && self.chance(60) {
            8
        } else {
            self.below(9)
        };
        let text = match pick {
            0 => "dyn ::core::fmt::Debug".to_string(),
            1 => "dyn ::core::any::Any".to_string(),
            2 | 3 if !self.env.theads.is_empty() => {
                let ts: Vec<THead> = self.env.theads.clone();
                let t = self.pick(&ts).clone();
                match t.shape {
                    TShape::T0 => format!("dyn {}", t.name),
                    TShape::T1 => {
                        let a = self.ty(sc, closed, depth + 1);
                        let text = format!("dyn {}<{}>", t.name, a.text);
                        parts.push(a);
                        text
                    }
                    TShape::Out => {
                        let a = self.ty(sc, closed, depth + 1);
                        let text = format!("dyn {}<Out = {}>", t.name, a.text);
                        parts.push(a);
                        text
                    }
                }
            }
            2 | 3 => "dyn ::core::fmt::Debug".to_string(),
            4 | 5 => {
                let f = *self.pick(&["Fn", "FnMut", "FnOnce"]);
                let path = self.trait_path(f);
                let a = self.ty(sc, closed, depth + 1);
                let r = self.ty(sc, closed, depth + 1);
                let text = format!("dyn {}({}) -> {}", path, a.text, ret_paren(&r.text));
                parts.push(a);
                parts.push(r);
                text
            }
            6 => {
                let path = self.trait_path("Iterator");
                let a = self.ty(sc, closed, depth + 1);
                let text = format!("dyn {}<Item = {}>", path, a.text);
                parts.push(a);
                text
            }
            7 => {
                // A higher-ranked bound, which the input anchors.
                let h = self.binder();
                let a = self.ty(sc, closed, depth + 1);
                let r = self.ty(sc, closed, depth + 1);
                let fn_path = self.trait_path("Fn");
                // The type of the module that names the binder's lifetime (`Rl<'h>`) next to the
                // input that keeps it, or, with none, the whole object is leaked.
                let rl = self.binder_path(&h);
                let input = match &rl {
                    Some(_) if self.chance(30) => {
                        self.tags.insert(TAG_BINDER_NO_ANCHOR.to_string());
                        a.text.clone()
                    }
                    _ => format!("&{h} {}", paren(&a.text)),
                };
                let text = format!(
                    "dyn for<{h}> {fn_path}({}{}) -> &{h} {}",
                    input,
                    rl.map_or(String::new(), |rl| format!(", {}", rl)),
                    ret_paren(&paren(&r.text))
                );
                parts.push(a);
                parts.push(r);
                text
            }
            _ => {
                // Elided lifetimes in the sugar of `Fn`: one in the input, so it is the output's.
                // `hidden_lifetimes` asks for it to be hidden by a path.
                let hidden = self.features.hidden_lifetimes && self.chance(50);
                let a = self.ty(sc, closed, depth + 1);
                let r = self.ty(sc, closed, depth + 1);
                if a.text.contains(['\'', '&'])
                    || (a.uses_self
                        && sc.self_kind != SelfKind::MaybeUnsized
                        && !self.features.self_elision)
                    || (!self.features.self_elision && a.adds_lts)
                {
                    return self.dyn_bare(sc, depth, open);
                }
                // The generics of a leaked type are lifetimes and types in the path that replaces
                // it, which stop the input from being the only lifetime to elide
                // (`elided_fn_generics`).
                if open && !self.features.elided_fn_generics {
                    return self.dyn_bare(sc, depth, false);
                }
                self.elided_self |= a.uses_self;
                if hidden && a.uses_self {
                    self.tags.insert(TAG_HIDDEN_SELF.to_string());
                }
                let input = if hidden {
                    format!("{}, {}", self.hidden_path(), a.text)
                } else {
                    format!("&{}", paren(&a.text))
                };
                let fn_path = self.trait_path("Fn");
                let text = format!(
                    "dyn {}({}) -> &{}",
                    fn_path,
                    input,
                    ret_paren(&paren(&r.text))
                );
                parts.push(a);
                parts.push(r);
                text
            }
        };
        let text = if self.chance(12) && !text.contains("Any") {
            let send = self.trait_path("Send");
            format!("{} + {}", text, send)
        } else {
            text
        };
        Ty::with(text, &parts.iter().collect::<Vec<_>>()).no_infer()
    }

    fn dyn_obj(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if !c.bounds.is_none() {
            return None;
        }
        let open = !c.closed && !c.no_params && !c.no_lifetimes && self.chance(30);
        let obj = self.dyn_bare(sc, depth, open);
        // The bound of the object: `'static` by default for a box, the lifetime of a reference.
        let lt_bound = if open || self.chance(35) {
            Some(self.lifetime(sc, c, false))
        } else {
            None
        };
        if let Some(lt) = &lt_bound {
            // `Any: 'static`, so only `dyn Any + 'static` is well formed (E0478).
            if !outlives(sc, &obj, lt, false, true) || (obj.text.contains("Any") && lt != "'static")
            {
                return None;
            }
        } else if open && (!obj.tys.is_empty() || !obj.lts.is_empty() || obj.uses_self) {
            // Without a bound the object must be `'static`.
            return None;
        }
        let with_bound = |text: &str, lt: &str| format!("{} + {}", text, lt);
        let mut ty;
        // `Box` and `Rc` are paths of `::std`.
        let std_ok = !self.abs_only || self.cfg.allow_std;
        match if std_ok { self.below(4) } else { 2 } {
            0 | 1 => {
                let text = match &lt_bound {
                    Some(lt) => with_bound(&obj.text, lt),
                    None => obj.text.clone(),
                };
                ty = Ty::with(format!("::std::boxed::Box<{}>", text), &[&obj]);
            }
            2 => {
                // `&'a dyn Tr`: the bound of the object is `'a`.
                let lt = match &lt_bound {
                    Some(lt) => lt.clone(),
                    None => self.lifetime(sc, c, true),
                };
                if !outlives(sc, &obj, &lt, false, false) {
                    return None;
                }
                // The object takes the lifetime of the reference: a path that names it too.
                if sc.ref_only.contains(&lt) {
                    self.tags.insert(TAG_METHOD_LIFETIME.to_string());
                }
                let text = match &lt_bound {
                    Some(b) => format!("&{} ({})", lt, with_bound(&obj.text, b)),
                    None => format!("&{} {}", lt, paren(&obj.text)),
                };
                ty = Ty::with(text, &[&obj]);
                if lt != "'static" {
                    ty.lts.insert(lt);
                }
            }
            _ => {
                let text = match &lt_bound {
                    Some(lt) => with_bound(&obj.text, lt),
                    None => obj.text.clone(),
                };
                ty = Ty::with(format!("::std::rc::Rc<{}>", text), &[&obj]);
            }
        }
        if let Some(lt) = lt_bound {
            if lt != "'static" {
                ty.lts.insert(lt);
            }
        }
        Some(ty.no_infer())
    }

    // -----------------------------------------------------------------------------------------
    // Generics.
    // -----------------------------------------------------------------------------------------

    fn skeleton(&mut self, level: Level, outer: &Scope, want_bounds: Bounds) -> Skeleton {
        let (lts, tys, consts): (&[&str], &[&str], &[&str]) = match level {
            Level::Item => (&["'a", "'b", "'c"], &["T", "U", "V"], &["N", "K"]),
            Level::Method => (&["'x", "'y"], &["X", "Y"], &["M"]),
        };
        let mut sk = Skeleton::default();
        let n_lt = match self.below(100) {
            0..=44 => 0,
            45..=79 => 1,
            _ => 2,
        }
        .min(lts.len());
        let n_ty = match self.below(100) {
            0..=24 => 0,
            25..=64 => 1,
            65..=92 => 2,
            _ => 3,
        }
        .min(tys.len());
        let n_const = if self.chance(30) { 1 } else { 0 };
        for l in lts.iter().take(n_lt) {
            sk.lifetimes.push(l.to_string());
        }
        if sk.lifetimes.len() >= 2 && self.chance(30) {
            sk.outlives
                .push((sk.lifetimes[1].clone(), sk.lifetimes[0].clone()));
        }
        if level == Level::Method
            && !sk.lifetimes.is_empty()
            && !outer.lifetimes.is_empty()
            && self.chance(15)
        {
            let outer_lt = self.pick(&outer.lifetimes).clone();
            sk.outlives.push((sk.lifetimes[0].clone(), outer_lt));
        }
        let all_lts: Vec<String> = outer
            .lifetimes
            .iter()
            .chain(sk.lifetimes.iter())
            .cloned()
            .collect();
        for name in tys.iter().take(n_ty) {
            let sized = !self.features.unsized_params || self.chance(82);
            let mut bounds = Bounds {
                clone: self.chance(20) && sized,
                debug: self.chance(20),
                default: self.chance(15) && sized,
            };
            if !sized {
                bounds.clone = false;
                bounds.default = false;
            }
            if level == Level::Item && !want_bounds.is_none() {
                bounds.clone |= want_bounds.clone && sized;
                bounds.debug |= want_bounds.debug;
                bounds.default |= want_bounds.default && sized;
            }
            let mut bound_texts = Vec::new();
            if !sized {
                bound_texts.push(self.maybe_sized());
            } else if self.chance(7) {
                bound_texts.push(self.trait_path("Sized"));
            }
            bound_texts.extend(self.bound_texts(bounds));
            // A marker trait or a comparison (`Send`, `Copy`, `Ord`, ...), bare or in full.
            if self.chance(14) {
                let marker = *self.pick(MARKER_BOUNDS);
                if sized || marker != "Copy" {
                    bound_texts.push(self.trait_path(marker));
                }
            }
            let mut outlives_list = Vec::new();
            for l in &all_lts {
                if self.chance(45) {
                    outlives_list.push(l.clone());
                    bound_texts.push(l.clone());
                }
            }
            if self.chance(5) {
                outlives_list.push("'static".to_string());
                bound_texts.push("'static".to_string());
            }
            // `iterator_item_cycle` asks for iterators.
            let iter = sized
                && self.chance(if self.features.iterator_item_cycle {
                    40
                } else {
                    10
                });
            let mut fixed_item = 0;
            if iter {
                let iterator = self.trait_path("Iterator");
                // `supertrait_item` and `alias_item` ask for an `Item` that is fixed.
                if self.features.supertrait_item && self.chance(50) {
                    bound_texts.push(format!("::{}::base::BItem", self.cfg.krate));
                    fixed_item = 1;
                } else if self.features.alias_item && self.chance(50) {
                    bound_texts.push("::core::iter::Iterator<Item = u8>".to_string());
                    fixed_item = 2;
                } else {
                    bound_texts.push(iterator);
                }
            }
            sk.tys.push(TyParam {
                name: name.to_string(),
                sized,
                bounds,
                bound_texts,
                outlives: outlives_list,
                iter,
                fixed_item,
            });
        }
        // A const parameter may have the name of a type parameter that a sibling scope declares
        // (`V` for an item, `Y` for a method), when no type parameter of this scope has it: the
        // two kinds are different parameters, as the scopes are siblings. (Not a name that a
        // module declares as a type, `SHADOW_NAMES`: a bare `T` in the arguments of a path would
        // be that type, not the const.)
        let alt = if n_ty < tys.len() {
            tys.last().copied()
        } else {
            None
        };
        for name in consts.iter().take(n_const) {
            let name = match alt {
                Some(alt) if self.chance(35) => alt,
                _ => name,
            };
            let ty = match self.below(100) {
                0..=59 => "usize",
                60..=74 => "bool",
                75..=84 => "i32",
                85..=91 => "u8",
                _ => "char",
            };
            sk.consts.push((name.to_string(), ty));
        }
        // A bound that names another parameter (`T: Into<V>`, `V` declared after or before it): a
        // leaked type that uses `T` needs `V` in its generics too (`inline_into`).
        if sk.tys.len() >= 2 && self.chance(20) {
            let n = sk.tys.len();
            let a = self.below(n);
            let b = (a + 1 + self.below(n - 1)) % n;
            if sk.tys[a].sized && sk.tys[b].sized {
                let into = self.trait_path("Into");
                let target = sk.tys[b].name.clone();
                sk.tys[a].bound_texts.push(format!("{}<{}>", into, target));
                let name = sk.tys[a].name.clone();
                sk.bounded.push((name, "into"));
            }
        }
        sk
    }

    /// Where predicates on the generics of `sc`, those of an item or of a method (`level`).
    fn preds(&mut self, level: Level, sc: &Scope, sk: &mut Skeleton, max: usize) {
        // `Self` is not known to be `Sized` while the predicates are checked.
        let mut sc = sc.clone();
        if sc.self_kind == SelfKind::MaybeUnsized {
            sc.self_kind = SelfKind::Unsized;
        }
        sc.pred = true;
        let sc = &sc;
        // A bound that names a leaked type that uses a generic is cyclic once rewritten, if the
        // generic has a bound itself, and a where predicate that names a lifetime is part of the
        // context of the types that use the lifetime (`bound_cycles`).
        let sized = Con {
            no_params: !self.bound_cycles(),
            ..Con::sized()
        };
        let unsized_ok = Con {
            no_params: !self.bound_cycles(),
            ..Con::unsized_ok()
        };
        let mut bounded = sc.bounded.clone();
        // The Repeater of a type with a bound that names an item of the module would be
        // for the Leaker's item, and the copy of the module has its own.
        let saved_env = self.env.clone();
        // Rewritten, a bound that names a leaked type is a projection, which rustc has to
        // normalize in the environment of the item while the other types of the item need the
        // original bounds to be normalized themselves: the types of a bound are absolute
        // (`bound_cycles`).
        let saved_abs = self.abs_only;
        if !self.bound_cycles() {
            self.abs_only = true;
            let crate_ok = self.cfg.allow_crate;
            self.env
                .heads
                .retain(|h| crate_ok && h.name.starts_with("::"));
            self.env
                .theads
                .retain(|h| crate_ok && h.name.starts_with("::"));
            self.env.consts.retain(|c| crate_ok && c.starts_with("::"));
        }
        if self.module_mode {
            let declared = is_declared;
            self.env.heads.retain(|h| !declared(&h.name));
            self.env.theads.retain(|h| !declared(&h.name));
            self.env.consts.retain(|c| !declared(c));
        }
        let n = self.below(max + 1);
        for _ in 0..n {
            let sized_params: Vec<TyParam> = sc.tys.iter().filter(|p| p.sized).cloned().collect();
            // `impl_pred_leaked` asks for a predicate of a method that names `Self::A`.
            if self.features.impl_pred_leaked
                && !self.no_self
                && !sc.assoc.is_empty()
                && !sized_params.is_empty()
                && self.chance(30)
            {
                let p = self.pick(&sized_params).clone();
                let (a, _) = self.pick(&sc.assoc).clone();
                let into = self.trait_path("Into");
                sk.preds.push(format!("{}: {}<Self::{}>", p.name, into, a));
                self.tags.insert(TAG_BOUND_CYCLE.to_string());
                continue;
            }
            let pred = match self.below(11) {
                // `strict_cycles` asks for an optional entry in a predicate that names a
                // lifetime: `T: Fn(&'a ::krate::base::B2<T, ::krate::base::B0>) -> u8`.
                8 if self.features.strict_cycles => {
                    let mut candidates: Vec<(TyParam, String)> = Vec::new();
                    for p in &sized_params {
                        for l in p.outlives.iter().filter(|l| *l != "'static") {
                            candidates.push((p.clone(), l.clone()));
                        }
                    }
                    if candidates.is_empty() {
                        continue;
                    }
                    let (p, lt) = self.pick(&candidates).clone();
                    if bounded.contains(&(p.name.clone(), "fn")) {
                        continue;
                    }
                    bounded.push((p.name.clone(), "fn"));
                    let krate = self.cfg.krate.clone();
                    let fn_path = self.trait_path("Fn");
                    format!(
                        "{}: {}(&{} ::{}::base::B2<{}, ::{}::base::B0>) -> u8",
                        p.name, fn_path, lt, krate, p.name, krate
                    )
                }
                // A marker trait or a comparison on a parameter, bare or in full.
                9 if !sc.tys.is_empty() => {
                    let p = self.pick(&sc.tys).clone();
                    let marker = *self.pick(MARKER_BOUNDS);
                    if marker == "Copy" && !p.sized {
                        continue;
                    }
                    format!("{}: {}", p.name, self.trait_path(marker))
                }
                // `T: From<A>`: a parameter has one, as it has one `Into`.
                10 if !sized_params.is_empty() => {
                    let p = self.pick(&sized_params).clone();
                    if bounded.contains(&(p.name.clone(), "from")) {
                        continue;
                    }
                    bounded.push((p.name.clone(), "from"));
                    let a = self.ty(sc, sized, 1);
                    format!("{}: {}<{}>", p.name, self.trait_path("From"), a.text)
                }
                0 if !sc.tys.is_empty() => {
                    let p = self.pick(&sc.tys).clone();
                    let b = if p.sized {
                        *self.pick(&[
                            Bounds::only_clone(),
                            Bounds::only_debug(),
                            Bounds::only_default(),
                        ])
                    } else {
                        Bounds::only_debug()
                    };
                    format!("{}: {}", p.name, self.bound_texts(b).join(" + "))
                }
                1 if !sized_params.is_empty() => {
                    let p = self.pick(&sized_params).clone();
                    if bounded.contains(&(p.name.clone(), "into")) {
                        continue;
                    }
                    bounded.push((p.name.clone(), "into"));
                    let a = self.ty(sc, sized, 1);
                    format!("{}: {}<{}>", p.name, self.trait_path("Into"), a.text)
                }
                2 if !sc.tys.is_empty() => {
                    let p = self.pick(&sc.tys).clone();
                    let a = self.ty(sc, unsized_ok, 1);
                    format!("{}: {}<{}>", p.name, self.trait_path("AsRef"), a.text)
                }
                3 if !sized_params.is_empty() => {
                    let p = self.pick(&sized_params).clone();
                    // The `Item` of a supertrait or of a bound is not equated to another type.
                    if bounded.contains(&(p.name.clone(), "iter")) || p.fixed_item != 0 {
                        continue;
                    }
                    bounded.push((p.name.clone(), "iter"));
                    // The item does not name the associated types of the iterators (`X::Item`):
                    // `X: Iterator<Item = W<X::Item>>` is a cycle for rustc itself.
                    // Nor `Self`: a type that holds `X::Item` by value would hold itself (E0391,
                    // a cycle in its layout).
                    let outer = std::mem::replace(&mut self.no_assoc_paths, true);
                    let outer_self = std::mem::replace(&mut self.no_self, true);
                    let a = self.ty(sc, sized, 1);
                    self.no_self = outer_self;
                    self.no_assoc_paths = outer;
                    format!(
                        "{}: {}<Item = {}>",
                        p.name,
                        self.trait_path("Iterator"),
                        a.text
                    )
                }
                4 if !sized_params.is_empty() => {
                    let p = self.pick(&sized_params).clone();
                    if bounded.contains(&(p.name.clone(), "fn")) {
                        continue;
                    }
                    bounded.push((p.name.clone(), "fn"));
                    let a = self.ty(sc, sized, 1);
                    let r = self.ty(sc, sized, 1);
                    let r = if self.features.bound_merge_syntax {
                        r.text
                    } else {
                        ret_paren(&r.text)
                    };
                    let family = self.fn_family();
                    format!(
                        "{}: {}({}) -> {}",
                        p.name,
                        self.trait_path(family),
                        a.text,
                        r
                    )
                }
                5 if !sized_params.is_empty() => {
                    let h = self.binder();
                    let p = self.pick(&sized_params).clone();
                    if bounded.contains(&(p.name.clone(), "fn")) {
                        continue;
                    }
                    bounded.push((p.name.clone(), "fn"));
                    let a = self.ty(sc, sized, 1);
                    let r = self.ty(sc, sized, 1);
                    let r = paren(&r.text);
                    let r = if self.features.bound_merge_syntax {
                        r
                    } else {
                        ret_paren(&r)
                    };
                    let family = self.fn_family();
                    let fn_path = self.trait_path(family);
                    // `Rl<'h>`: a type of the module that names the binder's lifetime. With an
                    // input that keeps it (`&'h A`) the bound is fine; without one it is the
                    // binder lifetime that only a leaked type would keep, which the Leaker
                    // rejects (E0581). The `Rl` is a relative type of the module, so both are
                    // generated from the types the module has (`saved_env`, `preds()` filters the
                    // environment of the bound).
                    let rl = self.binder_path_in(&saved_env, &h);
                    match (rl, self.below(100)) {
                        (Some(rl), 0..=24) => {
                            self.tags.insert(TAG_BINDER_NO_ANCHOR.to_string());
                            format!(
                                "{}: for<{h}> {}({}, {}) -> &{h} {}",
                                p.name, fn_path, a.text, rl, r
                            )
                        }
                        (Some(rl), 25..=49) => format!(
                            "{}: for<{h}> {}(&{h} {}, {}) -> &{h} {}",
                            p.name,
                            fn_path,
                            paren(&a.text),
                            rl,
                            r
                        ),
                        _ => format!(
                            "{}: for<{h}> {}(&{h} {}) -> &{h} {}",
                            p.name,
                            fn_path,
                            paren(&a.text),
                            r
                        ),
                    }
                }
                6 => {
                    // A type with bounds it has: `Vec<T>: Debug`.
                    let b = *self.pick(&[
                        Bounds::only_clone(),
                        Bounds::only_debug(),
                        Bounds::only_default(),
                    ]);
                    let a = self.ty(
                        sc,
                        Con {
                            bounds: b,
                            no_lifetimes: !self.bound_cycles(),
                            ..Con::sized()
                        },
                        1,
                    );
                    format!("{}: {}", a.text, self.bound_texts(b).join(" + "))
                }
                _ => continue,
            };
            if names_lifetime(&pred) && self.names_external(&pred) {
                self.risk.cycle = true;
            }
            sk.preds.push(pred);
        }
        if self.features.iterator_item_cycle {
            // They ask for both on an iterator: `T: Fn(<T as Iterator>::Item) -> u8` and `T:
            // Iterator<Item = (Rel<T>, u8)>`.
            let iters: Vec<TyParam> = sc
                .tys
                .iter()
                .filter(|p| p.iter && p.sized && p.fixed_item == 0)
                .cloned()
                .collect();
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::T1)
                .map(|head| head.name.clone())
                .collect();
            if !iters.is_empty() && !heads.is_empty() && self.cyclic && self.chance(60) {
                let x = self.pick(&iters).name.clone();
                if !bounded.contains(&(x.clone(), "fn")) && !bounded.contains(&(x.clone(), "iter"))
                {
                    bounded.push((x.clone(), "fn"));
                    bounded.push((x.clone(), "iter"));
                    self.tags.insert(TAG_ALIAS_BOUND.to_string());
                    let head = self.pick(&heads).clone();
                    let (fn_path, iterator) = (self.trait_path("Fn"), self.trait_path("Iterator"));
                    sk.preds.push(format!(
                        "{}: {}(<{} as {}>::Item) -> u8",
                        x, fn_path, x, iterator
                    ));
                    sk.preds
                        .push(format!("{}: {}<Item = ({}<{}>, u8)>", x, iterator, head, x));
                }
            }
        } else {
            // A bound of an iterator that names its `Item`, and a predicate that equates the
            // `Item` to a type: the second goes (`iterator_item_cycle`).
            for t in sc.tys.iter().filter(|t| t.iter) {
                let x = &t.name;
                let names_outer = sc.names_item.contains(x);
                let binds_outer = sc.binds_item.contains(x);
                let names_here = sk.preds.iter().any(|p| names_item(p, x));
                let binds_here = sk.preds.iter().any(|p| binds_item(p, x));
                if (names_outer || names_here) && (binds_outer || binds_here) {
                    sk.preds.retain(|p| !binds_item(p, x));
                    if binds_outer {
                        sk.preds.retain(|p| !names_item(p, x));
                    }
                }
            }
        }
        // `item_predicate_cycle` asks for a predicate of an item that names the `Item` of an
        // iterator (`Option<<T as Iterator>::Item>: Default`), and a predicate of a method that
        // equates it to a type that holds an entry (`T: Iterator<Item = (Rel<T>, u8)>`). Both
        // are decided by the hash of the state, which draws nothing: the other choices are
        // the same as without it.
        if self.features.item_predicate_cycle {
            let iters: Vec<TyParam> = sc
                .tys
                .iter()
                .filter(|p| p.iter && p.sized && p.fixed_item == 0)
                .cloned()
                .collect();
            let mut state = self.rng.clone();
            state.write(b"item_predicate_cycle");
            state.write(&[level as u8]);
            let draw = state.finish();
            if level == Level::Item && !iters.is_empty() && draw % 100 < 40 {
                let x = &iters[(draw / 100) as usize % iters.len()].name;
                let iterator = self.trait_path("Iterator");
                let item = format!("<{} as {}>::Item", x, iterator);
                let default = self.trait_path("Default");
                let pred = match (draw / 1000) % 4 {
                    0 => format!("::core::option::Option<{}>: {}", item, default),
                    1 => format!("::core::marker::PhantomData<{}>: {}", item, default),
                    2 => format!("{}: {}", item, self.trait_path("Sized")),
                    _ => format!("::core::option::Option<({}, u8)>: {}", item, default),
                };
                self.tags.insert(TAG_BOUND_CYCLE.to_string());
                sk.preds.push(pred);
            } else if level == Level::Method && self.cyclic && draw % 100 < 70 {
                let heads: Vec<String> = self
                    .env
                    .heads
                    .iter()
                    .filter(|head| head.shape == Shape::T1)
                    .map(|head| head.name.clone())
                    .collect();
                let candidates: Vec<&TyParam> = iters
                    .iter()
                    .filter(|p| {
                        sc.names_item.contains(&p.name)
                            && !bounded.contains(&(p.name.clone(), "iter"))
                    })
                    .collect();
                if !candidates.is_empty() && !heads.is_empty() {
                    let x = candidates[(draw / 100) as usize % candidates.len()]
                        .name
                        .clone();
                    let head = &heads[(draw / 1000) as usize % heads.len()];
                    bounded.push((x.clone(), "iter"));
                    let iterator = self.trait_path("Iterator");
                    sk.preds
                        .push(format!("{}: {}<Item = ({}<{}>, u8)>", x, iterator, head, x));
                }
            }
        } else {
            // The `Item` that a predicate of an enclosing item names is not equated to a type.
            for t in sc.tys.iter().filter(|t| t.iter) {
                if sc.names_item.contains(&t.name) {
                    sk.preds.retain(|p| !binds_item(p, &t.name));
                }
            }
        }
        // `supertrait_item` and `alias_item` ask for a bound of an iterator that names its `Item`,
        // which a supertrait or a bound fixes, and an entry that uses the iterator in a predicate:
        // `X: Fn(<X as Iterator>::Item) -> u8` (or through the alias `BIt<X>`), `Rel<X>: Default`.
        if self.features.supertrait_item || self.features.alias_item {
            let fixed: Vec<TyParam> = sc
                .tys
                .iter()
                .filter(|p| p.fixed_item != 0 && p.sized)
                .cloned()
                .collect();
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::T1)
                .map(|head| head.name.clone())
                .collect();
            // A predicate that names `Self::A` next to the alias bound is no impl (see
            // `method_sig()`).
            let names_assoc = sk.preds.iter().any(|p| p.contains("Self::"));
            if !fixed.is_empty() && !heads.is_empty() && self.cyclic && self.chance(60) {
                let x = self.pick(&fixed).clone();
                if !(x.fixed_item == 2 && names_assoc) && !bounded.contains(&(x.name.clone(), "fn"))
                {
                    bounded.push((x.name.clone(), "fn"));
                    self.tags.insert(TAG_ALIAS_BOUND.to_string());
                    let head = self.pick(&heads).clone();
                    let iterator = self.trait_path("Iterator");
                    let item = if x.fixed_item == 2 {
                        format!("::{}::base::BIt<{}>", self.cfg.krate, x.name)
                    } else {
                        format!("<{} as {}>::Item", x.name, iterator)
                    };
                    let (fn_path, default) = (self.trait_path("Fn"), self.trait_path("Default"));
                    sk.preds
                        .push(format!("{}: {}({}) -> u8", x.name, fn_path, item));
                    sk.preds.push(format!("{}<{}>: {}", head, x.name, default));
                }
            }
        }
        sk.bounded = bounded;
        self.env = saved_env;
        self.abs_only = saved_abs;
        // A predicate that names `Self` in a bound (`T: AsRef<Option<Self>>`).
        if sk.preds.iter().any(|p| names_self_in_bound(p)) {
            self.tags.insert(TAG_SELF_IN_BOUNDS.to_string());
        }
        // Where the Leaker does not allow the primitive types, `u8` in a bound is a leaked type, and
        // the types of the predicates that use the generics it bounds are in a cycle with it.
        if !self.cfg.primitive && sk.preds.iter().any(|p| names_primitive(p)) {
            self.tags.insert(TAG_BOUND_CYCLE.to_string());
        }
    }

    /// Gives defaults to the trailing parameters of the generics of a struct, enum, union or
    /// trait that nothing constrains (`DEFAULT_PERCENT` of the items).
    ///
    /// Rustc checks a default against the bounds and predicates of its parameter, so only a
    /// parameter with no bound but `?Sized` and outlives bounds, named by no predicate, has one.
    /// The parameters with a default are trailing. A default is analyzed with the parameters before
    /// it, so it names those and no `Self`. A const parameter's default is a literal, or a constant
    /// (`{ RLEN }`), or an earlier const parameter.
    fn defaults(&mut self, sk: &mut Skeleton) {
        if !self.chance(DEFAULT_PERCENT) {
            return;
        }
        // The parameters in the order of the declaration: types, then consts.
        let mut order: Vec<(bool, usize)> = (0..sk.tys.len()).map(|i| (false, i)).collect();
        order.extend((0..sk.consts.len()).map(|j| (true, j)));
        let free = |sk: &Skeleton, (is_const, i): (bool, usize)| -> bool {
            let name = if is_const {
                &sk.consts[i].0
            } else {
                &sk.tys[i].name
            };
            let constrained = !is_const && {
                let t = &sk.tys[i];
                !t.bounds.is_none()
                    || t.iter
                    || t.fixed_item != 0
                    || sk.bounded.iter().any(|(n, _)| n == name)
                    || !t
                        .bound_texts
                        .iter()
                        .all(|b| b.starts_with('\'') || b.starts_with("?"))
            };
            !constrained
                && !sk.preds.iter().any(|p| has_word(p, name))
                && !sk
                    .tys
                    .iter()
                    .any(|t| t.bound_texts.iter().any(|b| has_word(b, name)))
        };
        let mut start = order.len();
        while start > 0 && free(sk, order[start - 1]) {
            start -= 1;
        }
        let candidates = order.len() - start;
        if candidates == 0 {
            return;
        }
        let first = order.len() - self.range(1, candidates);
        let saved = std::mem::replace(&mut self.no_assoc_paths, true);
        self.in_default = true;
        for &(is_const, i) in &order[first..] {
            if is_const {
                let (name, ty) = sk.consts[i].clone();
                let earlier: Vec<String> = sk.consts[..i]
                    .iter()
                    .filter(|(_, t)| *t == ty)
                    .map(|(n, _)| n.clone())
                    .collect();
                let value = if !earlier.is_empty() && self.chance(30) {
                    self.pick(&earlier).clone()
                } else if ty == "usize" && !self.env.consts.is_empty() && self.chance(25) {
                    let consts = self.env.consts.clone();
                    let constant = self.pick(&consts).clone();
                    // A value in a const expression cannot be leaked: a relative one is rejected.
                    if !constant.starts_with("::") {
                        self.tags.insert(TAG_CONST_DEFAULT.to_string());
                    }
                    format!("{{ {} }}", constant)
                } else {
                    self.const_literal(ty)
                };
                sk.defaults.push((name, value));
            } else {
                let before = Skeleton {
                    lifetimes: sk.lifetimes.clone(),
                    outlives: sk.outlives.clone(),
                    tys: sk.tys[..i].to_vec(),
                    ..Default::default()
                };
                let mut sc = before.scope(&Scope::default());
                sc.reserved = sk.tys[i..]
                    .iter()
                    .map(|t| t.name.clone())
                    .chain(sk.consts.iter().map(|(n, _)| n.clone()))
                    .collect();
                let declared = sk.tys[i].outlives.clone();
                let mut text = None;
                for attempt in 0..4 {
                    let con = Con {
                        closed: attempt >= 2,
                        no_lifetimes: true,
                        ..Con::sized()
                    };
                    let ty = self.ty(&sc, con, 1);
                    // It outlives what the parameter is declared to outlive.
                    let ok = declared.iter().all(|lt| {
                        if lt == "'static" {
                            ty.tys.is_empty() && ty.lts.is_empty() && !ty.uses_self
                        } else {
                            outlives(&sc, &ty, lt, false, false)
                        }
                    });
                    if ok && !ty.uses_self {
                        text = Some(ty.text);
                        break;
                    }
                }
                sk.defaults
                    .push((sk.tys[i].name.clone(), text.unwrap_or_else(|| "u8".into())));
            }
        }
        self.in_default = false;
        self.no_assoc_paths = saved;
    }

    /// A const argument of a path: a constant that is a single identifier (`RLEN`) may be written
    /// bare (`R2C<T, RLEN>`, which parses as a type that may be a const: the Leaker leaks the whole
    /// path), or in braces like a path of several segments (`{ self::RLEN }`).
    ///
    /// In a module analyzed as a whole, a path that the module declares (or imports) holds the
    /// identifier as a type, which is the limitation that the `Leaker` docs name ("Bare identifiers
    /// and consts"): only a path of `base` is written with it.
    fn const_arg(&mut self, head: &str, len: &str, params: &[(String, &'static str)]) -> String {
        let braced = braced_path(len, params);
        let bare_ok = (!self.module_mode || head.contains("base::"))
            && (!head.starts_with("::") || self.features.bare_const_in_absolute_path);
        if braced != len && !len.contains("::") && bare_ok && self.chance(65) {
            len.to_string()
        } else {
            braced
        }
    }

    /// A literal of a primitive type, as a const argument or default.
    fn const_literal(&mut self, ty: &str) -> String {
        let values: &[&str] = match ty {
            "bool" => &["true", "false"],
            "i32" => &["0", "7", "-1"],
            "u8" => &["0", "255"],
            "char" => &["'a'", "'z'", "'\\n'"],
            _ => &["2", "3", "0", "1"],
        };
        self.pick(values).to_string()
    }

    // -----------------------------------------------------------------------------------------
    // Methods.
    // -----------------------------------------------------------------------------------------

    /// An associated type path of `Self` that a method of a trait can name: `Self::A0` and, with
    /// `self_projection_outlives`, `<Self as Tr<..>>::A0`.
    fn self_projection(&mut self, sc: &Scope) -> Option<String> {
        let more = self.features.self_projection_outlives && !sc.trait_ref.is_empty();
        let mut forms: Vec<String> = Vec::new();
        for (name, _) in &sc.assoc {
            forms.push(format!("Self::{}", name));
            if more {
                forms.push(format!("<Self as {}>::{}", sc.trait_ref, name));
            }
        }
        if forms.is_empty() {
            None
        } else {
            Some(self.pick(&forms).clone())
        }
    }

    /// A parameter or the result of a method. The lifetimes of the method are written in a
    /// reference at the top of the type only, which the copy keeps: a lifetime that is in a leaked
    /// type only is early-bound in the copy, and late-bound in the trait (E0195, see
    /// `method_lifetime_paths`).
    fn method_ty(&mut self, sc: &Scope) -> Ty {
        // `projection_outlives` asks for `Rel<&'a T::Item>`, where `'a` is a lifetime of the item,
        // `T` an iterator that does not outlive it, and the signature implies `T::Item: 'a`.
        if self.features.projection_outlives && !sc.pred && self.chance(30) {
            let item_lifetimes: Vec<String> = sc
                .lifetimes
                .iter()
                .filter(|l| !sc.ref_only.contains(*l))
                .cloned()
                .collect();
            let iters: Vec<TyParam> = sc
                .tys
                .iter()
                .filter(|p| p.iter && p.sized)
                .cloned()
                .collect();
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::T1 && !head.name.starts_with("::"))
                .map(|head| head.name.clone())
                .collect();
            if !item_lifetimes.is_empty() && !iters.is_empty() && !heads.is_empty() {
                let lt = self.pick(&item_lifetimes).clone();
                let candidates: Vec<TyParam> = iters
                    .into_iter()
                    .filter(|p| !p.outlives.contains(&lt))
                    .collect();
                if !candidates.is_empty() {
                    let p = self.pick(&candidates).clone();
                    let head = self.pick(&heads).clone();
                    let mut ty = Ty::new(format!("{}<&{} {}::Item>", head, lt, p.name)).no_infer();
                    ty.tys.insert(p.name);
                    ty.lts.insert(lt);
                    ty.projection = true;
                    return ty;
                }
            }
        }
        // `self_projection_outlives` asks for `Rel<&'a P>` or `&'a P`, where `'a` is a lifetime of
        // the trait and `P` an associated type path of `Self` (`Self::A0`, `<Self as Tr<'a,
        // T>>::A0`): the signature implies `P: 'a`, and not `Self: 'a` or the arguments of the
        // trait.
        let item_lifetimes: Vec<String> = sc
            .lifetimes
            .iter()
            .filter(|l| !sc.ref_only.contains(*l))
            .cloned()
            .collect();
        if self.features.self_projection_outlives
            && !self.no_self
            && !sc.pred
            && !sc.assoc.is_empty()
            && !item_lifetimes.is_empty()
            && self.chance(30)
        {
            let lt = self.pick(&item_lifetimes).clone();
            if let Some(path) = self.self_projection(sc) {
                let heads: Vec<String> = self
                    .env
                    .heads
                    .iter()
                    .filter(|head| head.shape == Shape::T1 && !head.name.starts_with("::"))
                    .map(|head| head.name.clone())
                    .collect();
                let text = if !heads.is_empty() && self.chance(50) {
                    format!("{}<&{} {}>", self.pick(&heads), lt, path)
                } else {
                    format!("&{} {}", lt, path)
                };
                let mut ty = Ty::new(text).no_infer();
                ty.uses_self = true;
                ty.projection = true;
                ty.adds_lts = sc.assoc_lts || sc.self_lts;
                ty.lts.insert(lt);
                return ty;
            }
        }
        // `implied_outlives` asks for `&'x mut Self::A` where only the signature implies
        // `Self::A: 'x`.
        if self.features.implied_outlives
            && !self.no_self
            && !sc.assoc.is_empty()
            && !sc.self_outlives.is_empty()
            && self.chance(35)
        {
            let lt = self.pick(&sc.self_outlives).clone();
            let path = self.self_projection(sc).unwrap();
            let m = if self.chance(30) { "mut " } else { "" };
            let mut ty = Ty::new(format!("&{} {}{}", lt, m, path)).no_infer();
            ty.uses_self = true;
            ty.lts.insert(lt);
            return ty;
        }
        if sc.method_paths || sc.ref_only.is_empty() {
            return self.ty(sc, Con::sized(), 0);
        }
        let mut inner = sc.clone();
        inner.lifetimes.retain(|l| !sc.ref_only.contains(l));
        if self.chance(30) {
            let lt = self.pick(&sc.ref_only).clone();
            let a = self.ty(&inner, Con::unsized_ok().indirect(), 1);
            if outlives(sc, &a, &lt, false, false) {
                let mut ty = Ty::with(format!("&{} {}", lt, paren(&a.text)), &[&a]);
                ty.lts.insert(lt);
                return ty;
            }
        }
        self.ty(&inner, Con::sized(), 0)
    }

    /// Methods whose generics are named alike, which are sibling scopes: `fn m<Y>(&self, a0: R1<Y>)`
    /// next to `fn n<const Y: usize>(&self, a0: RC<u8, { Y }>)` (a type and a const parameter of
    /// the same name), and `fn m<X>(&self, a0: R1<X>)` next to `fn n(&self, a0: R1<X>)`, where the
    /// module declares a type `X` (a type named like a generic of a sibling).
    fn clash_methods(&mut self) -> Vec<String> {
        let heads = |g: &Gen, shape: Shape| -> Vec<String> {
            g.env
                .heads
                .iter()
                .filter(|h| h.shape == shape && !is_shadow_name(&h.name))
                .map(|h| h.name.clone())
                .collect()
        };
        let (t1, tc) = (heads(self, Shape::T1), heads(self, Shape::TC));
        let mut methods = Vec::new();
        if !t1.is_empty() && !tc.is_empty() {
            let (a, b) = (self.pick(&t1).clone(), self.pick(&tc).clone());
            let (n1, n2) = (self.name("m"), self.name("m"));
            methods.push(format!("fn {}<Y>(&self, a0: {}<Y>)", n1, a));
            methods.push(format!(
                "fn {}<const Y: usize>(&self, a0: {}<u8, {{ Y }}>)",
                n2, b
            ));
        }
        let declares_x = self.env.heads.iter().any(|h| h.name == METHOD_SHADOW_NAME);
        if !t1.is_empty() && declares_x {
            let a = self.pick(&t1).clone();
            let (n1, n2) = (self.name("m"), self.name("m"));
            methods.push(format!("fn {}<X>(&self, a0: {}<X>)", n1, a));
            methods.push(format!("fn {}(&self, a0: {}<X>)", n2, a));
        }
        methods
    }

    /// The signature of a method with its where clause. `in_trait` is a method of a trait.
    fn method_sig(&mut self, outer: &Scope, name: &str, in_trait: bool) -> String {
        let sk = self.skeleton(Level::Method, outer, Bounds::default());
        let mut sc = sk.scope(outer);
        sc.method = true;
        sc.ref_only.extend(sk.lifetimes.iter().cloned());
        sc.method_paths = self.features.method_lifetime_paths || !in_trait;
        // The receiver.
        let has_self = outer.self_kind != SelfKind::No;
        let typed_self_ok = !self.no_self && has_self && !outer.adt_self;
        let by_value_ok = !self.no_self && has_self;
        let mut self_sized = false;
        // `implied_outlives` asks for the lifetime of the receiver, which implies `Self: 'x`.
        let pick = if self.features.implied_outlives
            && in_trait
            && !sk.lifetimes.is_empty()
            && self.chance(40)
        {
            55
        } else {
            self.below(100)
        };
        let receiver = match pick {
            0..=9 => None,
            _ if !has_self => None,
            10..=39 => Some("&self".to_string()),
            40..=54 => Some("&mut self".to_string()),
            // The lifetime of the receiver implies `Self: 'x`, which a copy of a check of an
            // impl lacks (`impl_self_outlives`): a bound that names `Self` could need it.
            55..=64
                if !sk.lifetimes.is_empty() && (in_trait || self.features.impl_self_outlives) =>
            {
                let l = sk.lifetimes[0].clone();
                sc.self_outlives.push(l.clone());
                Some(format!("&{} self", l))
            }
            65..=76 if by_value_ok => {
                self_sized = true;
                Some("self".to_string())
            }
            77..=88 if typed_self_ok => Some("self: ::std::boxed::Box<Self>".to_string()),
            89..=94 if typed_self_ok => Some("self: ::std::rc::Rc<Self>".to_string()),
            _ => Some("&self".to_string()),
        };
        let mut pre_preds: Vec<String> = Vec::new();
        if !sk.lifetimes.is_empty()
            && receiver.is_some()
            && by_value_ok
            && (in_trait || self.features.impl_self_outlives)
            && self.chance(12)
        {
            let l = sk.lifetimes[sk.lifetimes.len() - 1].clone();
            pre_preds.push(format!("Self: {}", l));
            self.tags.insert(TAG_SELF_IN_BOUNDS.to_string());
            sc.self_written.push(l.clone());
            sc.self_outlives.push(l);
        }
        let mut params = Vec::new();
        if let Some(r) = &receiver {
            params.push(r.clone());
        }
        let n = match self.below(100) {
            0..=24 => 0,
            25..=59 => 1,
            60..=84 => 2,
            _ => 3,
        };
        for i in 0..n {
            let t = self.method_ty(&sc);
            self_sized |= t.self_sized;
            params.push(format!("a{}: {}", i, t.text));
        }
        let mut ret = String::new();
        if self.chance(60) {
            let t = self.method_ty(&sc);
            self_sized |= t.self_sized;
            ret = format!(" -> {}", t.text);
        }
        // `strict_anchors` asks for an external type that holds a lifetime of the method, and a
        // return type that names it: a strict Referrer replaces the one, so nothing constrains it.
        if self.features.strict_anchors && !sk.lifetimes.is_empty() && self.chance(25) {
            let l = sk.lifetimes[0].clone();
            let holder = if !self.cfg.allow_crate {
                Some(format!("::{}::base::B1<&{} u8>", self.cfg.krate, l))
            } else if !self.cfg.allow_std {
                Some(format!("::std::boxed::Box<&{} u8>", l))
            } else {
                None
            };
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::LOnly)
                .map(|head| head.name.clone())
                .collect();
            if let (Some(holder), false) = (holder, heads.is_empty()) {
                params.push(format!("a{}: {}", n, holder));
                ret = format!(" -> {}<{}>", self.pick(&heads), l);
            }
        }
        // A free function or a method of an inherent impl has no trait signature to differ from,
        // so a leaked path that is the only argument to name a lifetime of the method, with the
        // lifetime in the return type, is accepted and its copy compiles, with the lifetime
        // early-bound (F22). It is made often, as the random types rarely make it. In a trait it
        // is E0195 (`method_lifetime_paths`). It is decided by the hash of the state and of the
        // name, which draws nothing: the other choices are the same as without it.
        if !in_trait && !sk.lifetimes.is_empty() {
            let mut state = self.rng.clone();
            state.write(name.as_bytes());
            let draw = state.finish();
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::LOnly)
                .map(|head| head.name.clone())
                .collect();
            if draw % 5 == 0 && !heads.is_empty() {
                let l = sk.lifetimes[0].clone();
                let head = &heads[(draw / 5) as usize % heads.len()];
                params.push(format!("af: {}<{}>", head, l));
                ret = format!(" -> &{} u8", l);
            }
        }
        let mut sk = sk;
        sk.preds = pre_preds;
        self.preds(Level::Method, &sc, &mut sk, 2);
        let generic_names: Vec<String> = sc.tys.iter().map(|t| t.name.clone()).collect();
        // A bound of the trait that names the `Item` of an iterator through an alias
        // (`alias_item`, `T: Fn(BIt<T>) -> u8`) is not normalized in the environment of a method
        // whose predicate names `Self::A`: rustc proves the impl of the trait there, so no
        // generic impl of the trait exists (E0277 with no `type-leak` at all: `trait Tr<U:
        // Iterator<Item = u8>> where U: Fn(WIt<U>) -> u8 { type A: Clone; fn m(&self) where
        // Self::A: Clone; }` and `impl<U: Iterator<Item = u8>> Tr<U> for Host where U:
        // Fn(WIt<U>) -> u8`; the same with `X: Into<Self::A>`, and not with `Self::A` in a
        // position, nor with a struct instead of the alias), and the Leaker cannot see through
        // the alias to reject the trait. So a method of such a trait has no predicate that names
        // `Self::A`.
        let alias_bound = sc
            .tys
            .iter()
            .any(|t| t.fixed_item == 2 && sc.names_item.contains(&t.name))
            || sk.preds.iter().any(|p| p.contains("::base::BIt<"));
        if alias_bound {
            sk.preds.retain(|p| !p.contains("Self::"));
        }
        let (allowed, can_name) = if in_trait {
            (
                self.features.trait_assoc_bound_cycle,
                !sc.assoc.is_empty() && !self.no_self && !alias_bound,
            )
        } else {
            (
                self.features.impl_self_bound_cycle,
                has_self && !self.no_self && sc.self_kind == SelfKind::Sized,
            )
        };
        if !allowed {
            // A bound of a generic that names `Self`, and a predicate that uses the generic:
            // the entry of the second has the bound in its context (`impl_self_bound_cycle`,
            // `trait_assoc_bound_cycle`).
            drop_self_bound_cycles(&mut sk.preds, &generic_names);
        } else if can_name && self.cyclic && self.chance(50) {
            // They ask for both: `X: Fn(&Self) -> u8` or `X: Fn(&Self::A) -> u8`, and `Rel<X>:
            // Default`.
            // On a parameter that has no `Fn` bound yet: two of them are ambiguous when `Self::A`
            // is a type that both can take.
            let sized: Vec<&TyParam> = sc
                .tys
                .iter()
                .filter(|p| p.sized && !sk.bounded.contains(&(p.name.clone(), "fn")))
                .collect();
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::T1)
                .map(|head| head.name.clone())
                .collect();
            if !sized.is_empty() && !heads.is_empty() {
                let x = self.pick(&sized).name.clone();
                let named = if in_trait {
                    format!("Self::{}", self.pick(&sc.assoc).0)
                } else {
                    "Self".to_string()
                };
                let head = self.pick(&heads).clone();
                sk.bounded.push((x.clone(), "fn"));
                let fn_path = self.trait_path("Fn");
                sk.preds
                    .push(format!("{}: {}(&{}) -> u8", x, fn_path, named));
                let default = self.trait_path("Default");
                sk.preds.push(format!("{}<{}>: {}", head, x, default));
            }
        }
        // `impl_trait_args` asks for an argument `impl Into<Rel<X>>` where `X` has a bound that
        // names a leaked type, `X: Fn(Rel0) -> u8`: the bound of the `impl Trait` is a bound of the
        // function, and the entry in it is in a cycle with the bound of `X`.
        if self.features.impl_trait_args && self.cyclic && self.chance(60) {
            let sized: Vec<String> = sk
                .tys
                .iter()
                .filter(|p| p.sized && !sk.bounded.contains(&(p.name.clone(), "fn")))
                .map(|p| p.name.clone())
                .collect();
            // In a module analyzed as a whole, a bound of an entry that names an item of the module
            // is the limitation F10, as in `preds()`.
            let module_mode = self.module_mode;
            let declared = is_declared;
            let relative = |shape: Shape| -> Vec<String> {
                self.env
                    .heads
                    .iter()
                    .filter(|head| head.shape == shape && !head.name.starts_with("::"))
                    .filter(|head| !(module_mode && declared(&head.name)))
                    .filter(|head| !is_shadow_name(&head.name))
                    .map(|head| head.name.clone())
                    .collect()
            };
            let units = relative(Shape::Unit);
            let heads = relative(Shape::T1);
            if !sized.is_empty() && !units.is_empty() && !heads.is_empty() {
                let x = self.pick(&sized).clone();
                let unit = self.pick(&units).clone();
                let head = self.pick(&heads).clone();
                sk.bounded.push((x.clone(), "fn"));
                let fn_path = self.trait_path("Fn");
                sk.preds.push(format!("{}: {}({}) -> u8", x, fn_path, unit));
                let into = self.trait_path("Into");
                params.push(format!("ai: impl {}<{}<{}>>", into, head, x));
            }
        }
        if self_sized && outer.self_kind == SelfKind::MaybeUnsized {
            let sized = self.trait_path("Sized");
            sk.preds.push(format!("Self: {}", sized));
            self.tags.insert(TAG_SELF_IN_BOUNDS.to_string());
        }
        let sig = format!(
            "fn {}{}({}){}{}",
            name,
            sk.params(),
            params.join(", "),
            ret,
            sk.where_clause()
        );
        if self.external_holds_lifetime(&sig, &sk.lifetimes) {
            self.risk.anchor = true;
        }
        sig
    }

    /// Whether `text` names a type of a crate that the Leaker does not allow by an absolute path:
    /// an optional entry.
    fn names_external(&self, text: &str) -> bool {
        (!self.cfg.allow_std && text.contains("::std::"))
            || (!self.cfg.allow_crate && text.contains(&format!("::{}::", self.cfg.krate)))
    }

    /// Whether the arguments of an external absolute path in `text` name one of `lifetimes`, or an
    /// array whose length is an external constant (it is leaked as a whole).
    fn external_holds_lifetime(&self, text: &str, lifetimes: &[String]) -> bool {
        if !self.cfg.allow_crate {
            let length = format!("; ::{}::base::BLEN]", self.cfg.krate);
            let mut from = 0;
            while let Some(at) = text[from..].find(&length) {
                let end = from + at;
                from = end + length.len();
                // The `[` that opens the array.
                let mut depth = 0;
                let mut start = 0;
                for (i, c) in text[..end].char_indices().rev() {
                    match c {
                        ']' => depth += 1,
                        '[' if depth == 0 => {
                            start = i;
                            break;
                        }
                        '[' => depth -= 1,
                        _ => (),
                    }
                }
                if lifetimes
                    .iter()
                    .any(|l| text[start..end].contains(l.as_str()))
                {
                    return true;
                }
            }
        }
        let mut prefixes = Vec::new();
        if !self.cfg.allow_std {
            prefixes.push("::std::".to_string());
        }
        if !self.cfg.allow_crate {
            prefixes.push(format!("::{}::", self.cfg.krate));
        }
        for prefix in prefixes {
            let mut from = 0;
            while let Some(at) = text[from..].find(&prefix) {
                let start = from + at;
                from = start + prefix.len();
                // The type that starts with the path: its arguments, and the bounds of a trait
                // object (`dyn ::ext::Tr + 'y`), up to the end of the enclosing element.
                let mut depth = 0;
                let mut end = text.len();
                let mut previous = ' ';
                for (i, c) in text[start..].char_indices() {
                    match c {
                        '<' | '(' | '[' => depth += 1,
                        '>' if previous == '-' => (),
                        '>' | ')' | ']' if depth == 0 => {
                            end = start + i;
                            break;
                        }
                        '>' | ')' | ']' => depth -= 1,
                        ',' | ';' | '{' if depth == 0 => {
                            end = start + i;
                            break;
                        }
                        _ => (),
                    }
                    previous = c;
                }
                if lifetimes
                    .iter()
                    .any(|l| text[start..end].contains(l.as_str()))
                {
                    return true;
                }
            }
        }
        false
    }

    /// Whether the item that is generated may be invoked as strict.
    fn strict_ok(&self) -> bool {
        (self.features.strict_cycles || !self.risk.cycle)
            && (self.features.strict_anchors || !self.risk.anchor)
    }
}

impl Con {
    /// Keeps the restrictions of `c` on the generics.
    fn with_scope(self, c: Con) -> Con {
        Con {
            closed: c.closed,
            no_params: c.no_params,
            no_lifetimes: c.no_lifetimes,
            direct: false,
            ..self
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Items.
// ---------------------------------------------------------------------------------------------

/// The generics of `sk` that the type `ty` uses.
fn prune(sk: &Skeleton, ty: &Ty) -> Skeleton {
    let mut out = Skeleton {
        lifetimes: sk
            .lifetimes
            .iter()
            .filter(|l| ty.lts.contains(*l))
            .cloned()
            .collect(),
        ..Default::default()
    };
    out.outlives = sk
        .outlives
        .iter()
        .filter(|(a, b)| out.lifetimes.contains(a) && out.lifetimes.contains(b))
        .cloned()
        .collect();
    for t in sk.tys.iter().filter(|t| ty.tys.contains(&t.name)) {
        let mut t = t.clone();
        let kept: Vec<String> = t
            .outlives
            .iter()
            .filter(|l| *l == "'static" || out.lifetimes.contains(l))
            .cloned()
            .collect();
        let dropped: Vec<String> = t
            .outlives
            .iter()
            .filter(|l| !kept.contains(l))
            .cloned()
            .collect();
        t.bound_texts.retain(|b| !dropped.contains(b));
        t.outlives = kept;
        out.tys.push(t);
    }
    out.consts = sk
        .consts
        .iter()
        .filter(|(n, _)| ty.consts.contains(n))
        .cloned()
        .collect();
    // A bound that names a parameter that goes (`T: Into<V>`) goes too: the parameter is not
    // constrained by the self type of an impl (E0207) and not used by an alias (E0091).
    let dropped: Vec<String> = sk
        .tys
        .iter()
        .map(|t| t.name.clone())
        .filter(|name| !out.tys.iter().any(|t| &t.name == name))
        .collect();
    for t in &mut out.tys {
        t.bound_texts
            .retain(|b| !dropped.iter().any(|name| has_word(b, name)));
    }
    out
}

/// The shape of a trait of `trait_self_defaults` (F39): `Gen::self_default_shape()`.
#[derive(Clone, Copy)]
struct SelfDefault {
    /// A parameter has a bound that names the trait's `Self`; else the control.
    names_self: bool,
    /// The trait is analyzed nested in `enter_mod`.
    nested: bool,
    /// The Referrer's crate declares the trait again (`redeclare`): only for a nested trait.
    redeclare: bool,
}

/// The shape of a struct, an enum or a union of `self_bound_adts` (F41, F42):
/// `Gen::self_bound_adt_shape()`.
#[derive(Clone, Copy)]
struct SelfBoundAdt {
    /// The bound is a where predicate; else it is written in the parameter.
    in_where: bool,
    /// A field names `Self` in a relative type (`Rel2<Self, T>`): `nested_self_entries`.
    names_self_in_type: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdtKind {
    Struct,
    Enum,
    Union,
}

impl Gen {
    fn new_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.cur_id = id;
        self.next_name = 0;
        id
    }

    fn attr(&self, id: usize, allow_self: bool, scope: &str, strict: bool) -> String {
        format!(
            "#[{}(id = {}, allow_self = {}, prelude = {}, scope = \"{}\", allow = \"{}\", \
             {}{}{}strict = {}, primitive = {}, relaxed_sized = {})]\n",
            ATTR,
            id,
            allow_self,
            self.cfg.prelude,
            scope,
            self.cfg.allow_arg(),
            if self.redeclare {
                "redeclare = true, "
            } else {
                ""
            },
            if scope == "mod" && self.features.nested_self_entries {
                "nested_self_entries = true, "
            } else {
                ""
            },
            if allow_stages() {
                "stages = true, "
            } else {
                ""
            },
            strict,
            self.cfg.primitive,
            BARE_RELAXED_SIZED_ALWAYS_OK
        )
    }

    /// The attribute of an item that is analyzed through `scope` (`""` for none). It says whether
    /// the Referrer invokes the item as strict: the macro then writes the Repeater's impls of all
    /// the entries, and of the required ones only otherwise.
    fn attr_with(&mut self, ann: Option<(usize, bool)>, scope: &'static str) -> String {
        self.last_scope = scope;
        let strict = self.want_strict && self.strict_ok();
        self.last_strict = strict;
        ann.map_or(String::new(), |(id, a)| self.attr(id, a, scope, strict))
    }

    /// The names of the types available in the module `path`, and the `use`s that import some.
    fn build_env(&mut self, path: &[String]) -> (ModEnv, String) {
        let up = "super::".repeat(path.len());
        let krate = self.cfg.krate.clone();
        let mut env = ModEnv::default();
        let mut uses = String::new();
        let mut names: Vec<(String, String, Option<Shape>, Option<TShape>)> = Vec::new();
        for (suffix, shape) in SHAPES {
            names.push((suffix.to_string(), suffix.to_string(), Some(*shape), None));
        }
        for (suffix, shape) in [("T0", TShape::T0), ("T1", TShape::T1), ("T2", TShape::Out)] {
            names.push((suffix.to_string(), suffix.to_string(), None, Some(shape)));
        }
        for (suffix, _, shape, tshape) in &names {
            let mut paths: Vec<String> = Vec::new();
            // Declared in the module.
            paths.push(format!("R{}", suffix));
            if self.chance(30) {
                paths.push(format!("self::R{}", suffix));
            }
            if path.len() > 1 && self.chance(30) {
                paths.push(format!("super::R{}", suffix));
            }
            // Declared in `base`, by path or imported.
            paths.push(format!("{}base::B{}", up, suffix));
            paths.push(format!("crate::base::B{}", suffix));
            paths.push(format!("::{}::base::B{}", krate, suffix));
            if self.chance(45) {
                let from = if self.chance(70) {
                    format!("{}base", up)
                } else {
                    "crate::base".to_string()
                };
                uses += &format!("use {}::B{} as I{};\n", from, suffix, suffix);
                paths.push(format!("I{}", suffix));
            }
            if self.chance(15) {
                uses += &format!("use ::{}::base::B{} as J{};\n", krate, suffix, suffix);
                paths.push(format!("J{}", suffix));
            }
            let skip_crate = self.module_mode && !self.features.module_crate_paths;
            for name in paths {
                if skip_crate && name.starts_with("crate::") {
                    continue;
                }
                if let Some(shape) = shape {
                    env.heads.push(Head {
                        name,
                        shape: *shape,
                    });
                } else if let Some(shape) = tshape {
                    env.theads.push(THead {
                        name,
                        shape: *shape,
                    });
                }
            }
        }
        // The types named like the generics of an item or of a method, which the module declares.
        if !self.module_mode || self.features.declared_named_like_generic {
            for name in SHADOW_NAMES {
                env.heads.push(Head {
                    name: name.to_string(),
                    shape: Shape::Unit,
                });
                if self.features.path_named_like_generic && self.chance(25) {
                    env.heads.push(Head {
                        name: format!("self::{}", name),
                        shape: Shape::Unit,
                    });
                }
            }
        }
        env.consts.push("RLEN".to_string());
        env.consts.push("self::RLEN".to_string());
        // In a module analyzed as a whole, a constant of another module in a type that names an
        // item of the module makes an entry that the module keys by itself (`module_crate_paths`,
        // on by default).
        let relative_consts = !self.module_mode || self.features.module_crate_paths;
        if relative_consts {
            env.consts.push(format!("{}base::BLEN", up));
            env.consts.push("crate::base::BLEN".to_string());
        }
        env.consts.push(format!("::{}::base::BLEN", krate));
        if self.chance(30) {
            uses += &format!("use {}base::BLEN as ILEN;\n", up);
            if relative_consts {
                env.consts.push("ILEN".to_string());
            }
        }
        (env, uses)
    }

    fn adt_field_count(&mut self) -> usize {
        match self.below(100) {
            0..=14 => 0,
            15..=49 => 1,
            50..=79 => 2,
            _ => 3,
        }
    }

    fn gen_adt(&mut self, kind: AdtKind, ann: Option<(usize, bool)>) -> String {
        if let Some(shape) = self.self_bound_adt_shape(ann) {
            return self.gen_self_bound_adt(kind, ann, shape);
        }
        let (prefix, kw) = match kind {
            AdtKind::Struct => ("S", "struct"),
            AdtKind::Enum => ("E", "enum"),
            AdtKind::Union => ("U", "union"),
        };
        let name = self.name(prefix);
        let allow_self = ann.map_or(true, |(_, a)| a);
        // A fifth of the items are analyzed through a scope: nested in `enter_mod` (it has a
        // `Self` of its own, and the generics are local), or in `enter_generics`, which has no
        // `Self` (so the fields do not name it).
        let scope: &'static str = match (ann.is_some(), self.below(100)) {
            (true, 0..=9) => "mod",
            (true, 10..=19) => "generics",
            _ => "",
        };
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        if scope == "generics" || (scope == "mod" && !self.features.nested_self) {
            self.self_prob = 0;
        }
        let base = Scope {
            self_kind: SelfKind::Sized,
            adt_self: true,
            infer: self.features.inferred_outlives && self.chance(35),
            ..Default::default()
        };
        let mut sk = self.skeleton(Level::Item, &Scope::default(), Bounds::default());
        let mut sc = sk.scope(&base);
        sc.self_lts = !sk.lifetimes.is_empty();

        let mut used_tys = BTreeSet::new();
        let mut used_lts = BTreeSet::new();
        let mut field = |g: &mut Gen, wrap: bool| -> String {
            let t = g.ty(&sc, Con::sized(), 0);
            used_tys.extend(t.tys.iter().cloned());
            used_lts.extend(t.lts.iter().cloned());
            if wrap {
                format!("::core::mem::ManuallyDrop<{}>", t.text)
            } else {
                t.text
            }
        };
        // The fields of each variant (one for a struct or a union).
        let n_variants = if kind == AdtKind::Enum {
            self.range(1, 3)
        } else {
            1
        };
        let mut variants: Vec<Vec<String>> = Vec::new();
        for _ in 0..n_variants {
            let n = if kind == AdtKind::Union {
                self.range(1, 3)
            } else {
                self.adt_field_count()
            };
            let fields: Vec<String> = (0..n)
                .map(|_| field(self, kind == AdtKind::Union))
                .collect();
            variants.push(fields);
        }
        // Every type and lifetime parameter must be used.
        let mut phantoms: Vec<String> = Vec::new();
        for l in &sk.lifetimes {
            if !used_lts.contains(l) {
                phantoms.push(format!("::core::marker::PhantomData<&{} ()>", l));
            }
        }
        for t in &sk.tys {
            if !used_tys.contains(&t.name) {
                phantoms.push(format!("::core::marker::PhantomData<*const {}>", t.name));
            }
        }
        self.preds(Level::Item, &sc, &mut sk, 2);
        self.self_prob = saved_prob;
        self.defaults(&mut sk);

        let attr = self.attr_with(ann, scope);
        let params = sk.params_decl();
        let wc = sk.where_clause();
        let text = match kind {
            AdtKind::Union => {
                let mut fields = variants.remove(0);
                for p in phantoms {
                    fields.push(format!("::core::mem::ManuallyDrop<{}>", p));
                }
                let body: Vec<String> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("pub f{}: {}", i, t))
                    .collect();
                format!(
                    "pub union {}{}{} {{ {} }}",
                    name,
                    params,
                    wc,
                    body.join(", ")
                )
            }
            AdtKind::Struct => {
                let mut fields = variants.remove(0);
                fields.extend(phantoms);
                let style = self.below(100);
                if fields.is_empty() {
                    format!("pub struct {}{}{};", name, params, wc)
                } else if style < 55 {
                    let body: Vec<String> = fields
                        .iter()
                        .enumerate()
                        .map(|(i, t)| format!("pub f{}: {}", i, t))
                        .collect();
                    format!(
                        "pub struct {}{}{} {{ {} }}",
                        name,
                        params,
                        wc,
                        body.join(", ")
                    )
                } else {
                    let body: Vec<String> = fields.iter().map(|t| format!("pub {}", t)).collect();
                    format!("pub struct {}{}({}){};", name, params, body.join(", "), wc)
                }
            }
            AdtKind::Enum => {
                let mut texts: Vec<String> = Vec::new();
                for (i, fields) in variants.iter().enumerate() {
                    let text = if fields.is_empty() {
                        format!("V{}", i)
                    } else if self.chance(50) {
                        format!("V{}({})", i, fields.join(", "))
                    } else {
                        let body: Vec<String> = fields
                            .iter()
                            .enumerate()
                            .map(|(j, t)| format!("f{}: {}", j, t))
                            .collect();
                        format!("V{} {{ {} }}", i, body.join(", "))
                    };
                    texts.push(text);
                }
                if !phantoms.is_empty() {
                    texts.push(format!("Phantom({})", phantoms.join(", ")));
                }
                format!(
                    "pub enum {}{}{} {{ {} }}",
                    name,
                    params,
                    wc,
                    texts.join(", ")
                )
            }
        };
        let _ = kw;
        sk.risk_cycle = self.risk.cycle;
        sk.tags = self.tags.clone();
        self.adts.push((name, sk));
        format!("{}{}\n", attr, text)
    }

    /// Whether the struct, enum or union about to be generated has the shape of `self_bound_adts` (F41,
    /// F42), and which. Decided by the hash of the state, which draws nothing, as
    /// `Gen::self_default_shape()` does. Only the items that the macro is applied to have it, and
    /// they are nested in `enter_mod` (`nested_self`), which gives them a `Self` of their own.
    fn self_bound_adt_shape(&self, ann: Option<(usize, bool)>) -> Option<SelfBoundAdt> {
        ann?;
        if !self.features.self_bound_adts || !self.features.nested_self || self.module_mode {
            return None;
        }
        let mut state = self.rng.clone();
        state.write(b"self_bound_adts");
        let draw = state.finish();
        if draw % 100 >= u64::from(SELF_BOUND_ADT_PERCENT) {
            return None;
        }
        let percent = |n: u64, limit: u32| (draw / n) % 100 < u64::from(limit);
        Some(SelfBoundAdt {
            in_where: percent(100, 40),
            names_self_in_type: self.features.nested_self_entries && percent(10_000, 60),
        })
    }

    /// A struct, an enum or a union nested in `enter_mod` whose first parameter has a bound that
    /// names its own `Self` (`T: Clone + AsRef<Self>`, or `AsRef<(Self, ())>`, `AsRef<Option<Self>>`;
    /// in the parameter or in a where predicate), and whose trailing parameters have defaults that
    /// use it (`U = Rel<T>`, `W = Rel2<T, U>`): `self_bound_adts`, F41. A field has the type of the
    /// default too (an entry for any type, F37), and, with `nested_self_entries`, one names `Self`
    /// in a relative type (`Rel2<Self, T>`, F42). The generics have no lifetime or const parameter,
    /// and the item has no impl: the unit tests of `tests/findings.rs` cover them.
    fn gen_self_bound_adt(
        &mut self,
        kind: AdtKind,
        ann: Option<(usize, bool)>,
        shape: SelfBoundAdt,
    ) -> String {
        let (prefix, kw) = match kind {
            AdtKind::Struct => ("S", "struct"),
            AdtKind::Enum => ("E", "enum"),
            AdtKind::Union => ("U", "union"),
        };
        let name = self.name(prefix);
        // The bound of the first parameter, and a bound of its own that names no `Self`.
        let as_ref = self.trait_path("AsRef");
        let bound = match self.below(3) {
            0 => format!("{}<Self>", as_ref),
            1 => format!("{}<(Self, ())>", as_ref),
            _ => format!("{}<::core::option::Option<Self>>", as_ref),
        };
        let bounds = if self.chance(40) {
            *self.pick(&[
                Bounds::only_clone(),
                Bounds::only_debug(),
                Bounds::only_default(),
            ])
        } else {
            Bounds::default()
        };
        let mut bound_texts = self.bound_texts(bounds);
        let mut sk = Skeleton::default();
        if shape.in_where {
            sk.preds.push(format!("T: {}", bound));
        } else {
            bound_texts.push(bound);
        }
        let param = |name: &str, bounds: Bounds, bound_texts: Vec<String>| TyParam {
            name: name.to_string(),
            sized: true,
            bounds,
            bound_texts,
            outlives: Vec::new(),
            iter: false,
            fixed_item: 0,
        };
        sk.tys.push(param("T", bounds, bound_texts));
        let n_default = self.range(1, 2);
        let defaulted = ["U", "W"];
        for tyname in defaulted.iter().take(n_default) {
            sk.tys.push(param(tyname, Bounds::default(), Vec::new()));
        }
        // The types of the module with one and two type arguments, that are not shadowed by `T`, `U`
        // or `W`, and are written by a path that the Leaker leaks (not `::krate::base::B1`).
        let heads = |g: &Gen, shape: Shape| -> Vec<String> {
            g.env
                .heads
                .iter()
                .filter(|h| h.shape == shape && !h.name.starts_with("::"))
                .filter(|h| !is_shadow_name(&h.name))
                .map(|h| h.name.clone())
                .collect()
        };
        let (t1, t2) = (heads(self, Shape::T1), heads(self, Shape::T2));
        if t1.is_empty() || t2.is_empty() {
            // The module declares them all: unreachable, but a type with no entry is harmless.
            return self.gen_self_bound_adt_fallback(kind, ann);
        }
        // The first default always names `T`, which has the bound; the second names the first.
        let first = format!("{}<T>", self.pick(&t1));
        sk.defaults.push(("U".to_string(), first.clone()));
        if n_default == 2 {
            let second = if self.chance(50) {
                format!("{}<U>", self.pick(&t1))
            } else {
                format!("{}<T, U>", self.pick(&t2))
            };
            sk.defaults.push(("W".to_string(), second));
        }
        // The fields: every parameter, a field of the type of the default (an entry for any type),
        // the type that names `Self` (F42), and one that is copied as written.
        let mut fields: Vec<String> = sk.tys.iter().map(|t| t.name.clone()).collect();
        if self.chance(60) {
            fields.push(first);
        }
        if shape.names_self_in_type {
            let ty = if self.chance(50) {
                format!("{}<Self>", self.pick(&t1))
            } else {
                format!("{}<Self, T>", self.pick(&t2))
            };
            fields.push(if self.chance(40) {
                format!("::core::option::Option<{}>", ty)
            } else {
                ty
            });
            self.tags.insert(TAG_SELF_BOUND_TYPE.to_string());
        }
        if self.chance(30) {
            fields.push("::core::marker::PhantomData<*const Self>".to_string());
        }
        // No tag explains a rejection but the bare traits of the prelude (`bare:Clone`, where the
        // Leaker does not allow the prelude): the Leaker must accept the item (`main.rs`,
        // `unexplained()`).
        self.tags.insert(TAG_SELF_BOUND_ADT.to_string());
        let attr = self.attr_with(ann, "mod");
        let (params, wc) = (sk.params_decl(), sk.where_clause());
        let text = match kind {
            AdtKind::Union => {
                let body: Vec<String> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("pub f{}: ::core::mem::ManuallyDrop<{}>", i, t))
                    .collect();
                format!(
                    "pub union {}{}{} {{ {} }}",
                    name,
                    params,
                    wc,
                    body.join(", ")
                )
            }
            AdtKind::Struct if self.chance(50) => {
                let body: Vec<String> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("pub f{}: {}", i, t))
                    .collect();
                format!(
                    "pub struct {}{}{} {{ {} }}",
                    name,
                    params,
                    wc,
                    body.join(", ")
                )
            }
            AdtKind::Struct => {
                let body: Vec<String> = fields.iter().map(|t| format!("pub {}", t)).collect();
                format!("pub struct {}{}({}){};", name, params, body.join(", "), wc)
            }
            AdtKind::Enum => {
                let variants: Vec<String> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("V{}({})", i, t))
                    .collect();
                format!(
                    "pub enum {}{}{} {{ {} }}",
                    name,
                    params,
                    wc,
                    variants.join(", ")
                )
            }
        };
        let _ = kw;
        format!("{}{}\n", attr, text)
    }

    /// The item of `gen_self_bound_adt()` in a module that declares no type with one and two
    /// parameters: a plain unit struct.
    fn gen_self_bound_adt_fallback(
        &mut self,
        _kind: AdtKind,
        ann: Option<(usize, bool)>,
    ) -> String {
        let name = self.name("S");
        let attr = self.attr_with(ann, "mod");
        format!("{}pub struct {};\n", attr, name)
    }

    /// Whether the trait about to be generated has the shape of `trait_self_defaults` (F39), and
    /// which. Decided by the hash of the state, which draws nothing: the other choices are the same
    /// as without the feature, and a trait that does not have the shape is made as before.
    ///
    /// Only the items that the macro is applied to have it (a module analyzed as a whole names no
    /// `Self`), and a parameter bound that names `Self` needs an item that may: a nested trait has
    /// a `Self` of its own whatever `allow_self` says, the others are rejected with `allow_self =
    /// false`, so those are the control.
    fn self_default_shape(&self, ann: Option<(usize, bool)>) -> Option<SelfDefault> {
        let (_, allow_self) = ann?;
        if !self.features.trait_self_defaults || self.module_mode {
            return None;
        }
        let mut state = self.rng.clone();
        state.write(b"trait_self_defaults");
        let draw = state.finish();
        if draw % 100 >= u64::from(SELF_DEFAULT_PERCENT) {
            return None;
        }
        let percent = |n: u64, limit: u32| (draw / n) % 100 < u64::from(limit);
        let nested = self.features.nested_self && percent(100, SELF_DEFAULT_NESTED_PERCENT);
        Some(SelfDefault {
            names_self: (allow_self || nested) && percent(10_000, SELF_DEFAULT_NAMES_SELF_PERCENT),
            nested,
            redeclare: nested && percent(1_000_000, SELF_DEFAULT_REDECLARE_PERCENT),
        })
    }

    /// A trait whose parameter defaults use a parameter with a bound (`trait Tr<T: AsRef<Self>, U
    /// = Rel<T>>`), and the control, where the bound names no `Self` (`trait Tr<T: Clone, U =
    /// Rel<T>>`): `trait_self_defaults`, F39.
    ///
    /// The leading parameters (one or two) have no default, the first has the bound, and one or two
    /// trailing parameters have a default that names the leading ones (`Rel<T>`, `Rel2<T, U>`, a
    /// closed type, or any type the generator makes), the first of them always `Head<T>` if the
    /// bound names `Self`. The bound is written in the parameter or in a where predicate.
    ///
    /// The body is a few methods, or, for a trait that the Referrer's crate declares again
    /// (`redeclare`), one method that names the parameters only: the declaration has `T:
    /// AsRef<Self>`, and the type of an entry that names `T` is written through the host, which has
    /// the impl of the Repeater for `T: AsRef<Host>` (E0277 in the declaration). The defaults of
    /// that trait name the leading parameters only, because the declaration passes `()` for a
    /// defaulted one (E0128) and the copy of the default must be the original. The trait is not
    /// implemented by the other items of the module.
    fn gen_self_default_trait(&mut self, ann: Option<(usize, bool)>, shape: SelfDefault) -> String {
        let name = self.name("Tr");
        let allow_self = ann.map_or(true, |(_, a)| a);
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        let saved_no_self = self.no_self;
        if shape.redeclare {
            self.no_self = true;
        }
        let names = ["T", "U", "V"];
        let n_lead = if self.chance(30) { 2 } else { 1 };
        let n_default = self.range(1, 3 - n_lead);
        let param = |name: &str, bounds: Bounds, bound_texts: Vec<String>| TyParam {
            name: name.to_string(),
            sized: true,
            bounds,
            bound_texts,
            outlives: Vec::new(),
            iter: false,
            fixed_item: 0,
        };
        let mut sk = Skeleton::default();
        for (i, tyname) in names.iter().take(n_lead).enumerate() {
            if i == 0 && shape.names_self {
                let bound = format!("{}<Self>", self.trait_path("AsRef"));
                if self.chance(35) {
                    sk.preds.push(format!("{}: {}", tyname, bound));
                    sk.tys.push(param(tyname, Bounds::default(), Vec::new()));
                } else {
                    sk.tys.push(param(tyname, Bounds::default(), vec![bound]));
                }
            } else if self.chance(50) {
                let bounds = *self.pick(&[
                    Bounds::only_clone(),
                    Bounds::only_debug(),
                    Bounds::only_default(),
                ]);
                let texts = self.bound_texts(bounds);
                sk.tys.push(param(tyname, bounds, texts));
            } else {
                sk.tys.push(param(tyname, Bounds::default(), Vec::new()));
            }
        }
        for tyname in names.iter().skip(n_lead).take(n_default) {
            sk.tys.push(param(tyname, Bounds::default(), Vec::new()));
        }
        // The defaults.
        let saved_assoc = std::mem::replace(&mut self.no_assoc_paths, true);
        self.in_default = true;
        for i in n_lead..sk.tys.len() {
            // A default names the parameters before it, but a trait that is declared again passes
            // `()` for the defaulted ones, so only the leading ones.
            let usable = if shape.redeclare { n_lead } else { i };
            let params: Vec<String> = sk.tys[..usable].iter().map(|t| t.name.clone()).collect();
            let any = self.chance(15);
            let heads = |g: &Gen, shape: Shape| -> Vec<String> {
                g.env
                    .heads
                    .iter()
                    .filter(|h| h.shape == shape && (any || !h.name.starts_with("::")))
                    .filter(|h| !is_shadow_name(&h.name))
                    .map(|h| h.name.clone())
                    .collect()
            };
            let (t0, t1, t2) = (
                heads(self, Shape::Unit),
                heads(self, Shape::T1),
                heads(self, Shape::T2),
            );
            let first = shape.names_self && i == n_lead;
            let form = if first { 0 } else { self.below(100) };
            let mut text = match form {
                0..=59 if !t1.is_empty() => {
                    // The parameter with the bound, mostly.
                    let used = if first || self.chance(70) {
                        params[0].clone()
                    } else {
                        self.pick(&params).clone()
                    };
                    format!("{}<{}>", self.pick(&t1), used)
                }
                60..=79 if !t2.is_empty() => {
                    let (a, b) = (self.pick(&params).clone(), self.pick(&params).clone());
                    format!("{}<{}, {}>", self.pick(&t2), a, b)
                }
                80..=89 if !t0.is_empty() => self.pick(&t0).clone(),
                _ => String::new(),
            };
            if text.is_empty() {
                // Any type the generator makes, with the parameters before the default.
                let before = Skeleton {
                    tys: sk.tys[..usable].to_vec(),
                    ..Default::default()
                };
                let mut sc = before.scope(&Scope::default());
                sc.reserved = sk.tys[usable..].iter().map(|t| t.name.clone()).collect();
                text = "u8".to_string();
                for attempt in 0..3 {
                    let con = Con {
                        closed: attempt >= 1,
                        no_lifetimes: true,
                        ..Con::sized()
                    };
                    let ty = self.ty(&sc, con, 1);
                    if !ty.uses_self {
                        text = ty.text;
                        break;
                    }
                }
            }
            sk.defaults.push((sk.tys[i].name.clone(), text));
        }
        self.in_default = false;
        self.no_assoc_paths = saved_assoc;

        let base = Scope {
            self_kind: SelfKind::Unsized,
            trait_ref: format!("{}{}", name, sk.args()),
            ..Default::default()
        };
        let sc = sk.scope(&base);
        let mut methods: Vec<String> = Vec::new();
        let by_value = |g: &mut Gen, sk: &Skeleton| -> String {
            let params: Vec<String> = sk
                .tys
                .iter()
                .enumerate()
                .map(|(i, t)| format!("a{}: {}", i, t.name))
                .collect();
            let m = g.name("m");
            format!("fn {}(&self, {})", m, params.join(", "))
        };
        if shape.redeclare {
            methods.push(by_value(self, &sk));
        } else {
            let method_scope = Scope {
                self_kind: SelfKind::MaybeUnsized,
                ..sc
            };
            if self.chance(50) {
                methods.push(by_value(self, &sk));
            }
            for _ in 0..self.range(0, 2) {
                let m = self.name("m");
                methods.push(self.method_sig(&method_scope, &m, true));
            }
        }
        self.self_prob = saved_prob;
        self.no_self = saved_no_self;

        self.tags.insert(
            if shape.names_self {
                TAG_SELF_DEFAULT
            } else {
                TAG_PLAIN_DEFAULT
            }
            .to_string(),
        );
        if shape.redeclare {
            self.tags.insert(TAG_REDECLARED.to_string());
        }
        // Nested in `enter_mod` (it has a `Self` of its own), or analyzed as the item itself: by
        // `intern_trait`, mostly, and by `enter_trait`.
        let scope: &'static str = if shape.nested {
            "mod"
        } else if self.chance(25) {
            "trait"
        } else {
            ""
        };
        self.redeclare = shape.redeclare;
        let attr = self.attr_with(ann, scope);
        self.redeclare = false;
        let body: Vec<String> = methods.iter().map(|m| format!("{};", m)).collect();
        format!(
            "{}pub trait {}{}{} {{\n{}\n}}\n",
            attr,
            name,
            sk.params_decl(),
            sk.where_clause(),
            body.join("\n")
        )
    }

    fn gen_trait(&mut self, ann: Option<(usize, bool)>) -> String {
        if let Some(shape) = self.self_default_shape(ann) {
            return self.gen_self_default_trait(ann, shape);
        }
        let name = self.name("Tr");
        let allow_self = ann.map_or(true, |(_, a)| a);
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        // A tenth of the traits are analyzed nested in `enter_mod`, and name no `Self`
        // (`nested_self`).
        let in_mod = ann.is_some() && self.chance(10);
        let saved_no_self = self.no_self;
        if in_mod && !self.features.nested_self {
            self.no_self = true;
        }
        let mut sk = self.skeleton(Level::Item, &Scope::default(), Bounds::default());
        self.elided_self = false;
        // Supertraits.
        let supers = Bounds {
            clone: self.chance(15),
            debug: self.chance(25),
            default: self.chance(10),
        };
        let mut super_texts = self.bound_texts(supers);
        let explicit_sized = self.chance(8);
        if explicit_sized {
            super_texts.push(self.trait_path("Sized"));
        }
        // A marker supertrait (`Send`, `Sync`, `Unpin`): the Referrer's host type has them, but an
        // impl of the trait in the Leaker's crate would need the self type to have them too.
        let marker_super = self.chance(12);
        if marker_super {
            let marker = *self.pick(&["Send", "Sync", "Unpin"]);
            super_texts.push(self.trait_path(marker));
        }
        let self_sized = supers.clone || supers.default || explicit_sized;
        // Associated types.
        let mut assoc: Vec<(String, Bounds)> = Vec::new();
        let mut assoc_texts: Vec<String> = Vec::new();
        for i in 0..self.range(0, 2) {
            let b = Bounds {
                clone: self.chance(30),
                debug: self.chance(30),
                default: self.chance(20),
            };
            let mut texts = self.bound_texts(b);
            if self.chance(20) {
                texts.push(format!("{}<u32>", self.trait_path("Into")));
                if !self.cfg.primitive {
                    self.tags.insert(TAG_BOUND_CYCLE.to_string());
                }
            }
            let n = format!("A{}", i);
            assoc_texts.push(if texts.is_empty() {
                format!("type {};", n)
            } else {
                format!("type {}: {};", n, texts.join(" + "))
            });
            assoc.push((n, b));
        }
        // `self_projection_outlives` asks for a generic associated type `type G<'g>;`, and a method
        // that names it with a lifetime of the trait that nothing else names (`'c`, which no
        // skeleton declares, and is declared after the other methods are made): rustc requires
        // bounds on the associated type for each outlives relation that the signature or the trait
        // has with the lifetime it is given ("missing required bound"), and the reference around
        // it has another lifetime.
        let gat = self.features.self_projection_outlives
            && !sk.lifetimes.is_empty()
            && !self.no_self
            && self.chance(30);
        let gat_where = gat && self.features.gat_where_clause && self.chance(60);
        let mut with_gat = sk.clone();
        if gat {
            with_gat.lifetimes.push("'c".to_string());
        }
        let base = Scope {
            self_kind: if self_sized {
                SelfKind::Sized
            } else {
                SelfKind::Unsized
            },
            assoc: assoc.clone(),
            trait_ref: format!("{}{}", name, with_gat.args()),
            assoc_lts: !sk.lifetimes.is_empty(),
            ..Default::default()
        };
        let sc = sk.scope(&base);
        // A where clause of the trait that names `Self::A` is cyclic once the trait is implemented.
        let mut pred_scope = sc.clone();
        pred_scope.assoc.clear();
        self.preds(
            Level::Item,
            &pred_scope,
            &mut sk,
            if self.features.impl_pred_leaked { 2 } else { 1 },
        );
        self.defaults(&mut sk);
        let sc = sk.scope(&base);
        // Constants.
        let mut consts: Vec<(String, String)> = Vec::new();
        let mut const_texts: Vec<String> = Vec::new();
        for i in 0..self.range(0, 2) {
            if self.chance(70) {
                let t = self.ty(&sc, Con::unsized_ok().indirect(), 1);
                let decl = format!("const C{}: ::core::marker::PhantomData<{}>", i, t.text);
                consts.push((decl.clone(), "::core::marker::PhantomData".to_string()));
                const_texts.push(format!("{};", decl));
            } else {
                let (ty, value) = *self.pick(&[("usize", "0"), ("bool", "false"), ("u8", "1")]);
                let decl = format!("const K{}: {}", i, ty);
                consts.push((decl.clone(), value.to_string()));
                const_texts.push(format!("{};", decl));
            }
        }
        // Methods.
        let method_scope = Scope {
            self_kind: if self_sized {
                SelfKind::Sized
            } else {
                SelfKind::MaybeUnsized
            },
            ..sc.clone()
        };
        let mut methods: Vec<String> = Vec::new();
        for _ in 0..self.range(0, 4) {
            let m = self.name("m");
            methods.push(self.method_sig(&method_scope, &m, true));
        }
        if self.chance(CLASH_PERCENT) {
            methods.extend(self.clash_methods());
        }
        if gat {
            let outer = self.pick(&sk.lifetimes).clone();
            sk.lifetimes.push("'c".to_string());
            // `gat_where_clause` asks for the lifetime of the method (`fn m<'x>(&'x self, a:
            // Rel<&'x Self::G<'x>>)`, F35) in a leaked type, which implies `Self: 'x` through the
            // where clause of `G`. The other form names the lifetime of the trait, and rustc then
            // requires `Self: 'c` on `G` too.
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::T1 && !head.name.starts_with("::"))
                .map(|head| head.name.clone())
                .collect();
            let method_form = gat_where && !heads.is_empty() && self.chance(70);
            let lt = if method_form { "'x" } else { "'c" };
            let path = if self.chance(50) {
                format!("Self::G<{}>", lt)
            } else {
                format!("<Self as {}{}>::G<{}>", name, sk.args(), lt)
            };
            let m = self.name("m");
            if method_form {
                let head = self.pick(&heads).clone();
                methods.push(format!("fn {}<'x>(&'x self, a: {}<&'x {}>)", m, head, path));
            } else {
                methods.push(format!("fn {}(&self, a: &{} {})", m, outer, path));
            }
            assoc_texts.push(if gat_where {
                "type G<'g> where Self: 'g, Self: 'c;".to_string()
            } else {
                "type G<'g>;".to_string()
            });
        }
        self.self_prob = saved_prob;
        self.no_self = saved_no_self;

        // Another tenth are analyzed by `enter_trait`, which does not know the generic associated
        // types of the trait.
        let scope: &'static str = match (ann.is_some(), in_mod, self.below(100)) {
            (true, true, _) => "mod",
            (true, false, 0..=11) if !gat => "trait",
            _ => "",
        };
        let attr = self.attr_with(ann, scope);
        let supers_text = if super_texts.is_empty() {
            String::new()
        } else {
            format!(": {}", super_texts.join(" + "))
        };
        let body: Vec<String> = assoc_texts
            .iter()
            .cloned()
            .chain(const_texts.iter().cloned())
            .chain(methods.iter().map(|m| format!("{};", m)))
            .collect();
        let text = format!(
            "{}pub trait {}{}{}{} {{\n{}\n}}\n",
            attr,
            name,
            sk.params_decl(),
            supers_text,
            sk.where_clause(),
            body.join("\n")
        );
        let self_assoc_in_pred = methods.iter().any(|m| {
            m.split_once(" where ")
                .map_or(false, |(_, w)| w.contains("Self::"))
        });
        let preds_leakable = {
            let params: Vec<String> = sk.tys.iter().map(|t| t.name.clone()).collect();
            sk.preds.iter().any(|p| names_leakable(p, &params))
        };
        sk.tags = self.tags.clone();
        self.traits.push(TraitInfo {
            name,
            sk,
            supers,
            assoc,
            gat,
            gat_where,
            consts,
            marker_super,
            uses_assoc_self: body.iter().any(|item| item.contains("Self::")),
            self_assoc_in_pred,
            preds_leakable,
            elided_self: self.elided_self,
            risk: self.risk,
            uses_self_outlives: body.iter().any(|item| {
                item.contains("Self: '")
                    || ((item.contains("(&'x self") || item.contains("(&'y self"))
                        && item.contains("Self"))
            }),
            methods,
            implemented: false,
        });
        text
    }

    /// An impl of a trait of this module that is not implemented yet.
    fn gen_impl_trait(
        &mut self,
        ann: Option<(usize, bool)>,
        path: &[String],
        whole: bool,
    ) -> Option<String> {
        let candidates: Vec<usize> = (0..self.traits.len())
            .filter(|&i| !self.traits[i].implemented)
            .filter(|&i| !self.traits[i].marker_super)
            .filter(|&i| !self.traits[i].uses_assoc_self || self.features.impl_assoc_self)
            // `Self: 'x` (or the lifetime of a receiver) holds of the type of an impl only if it
            // is the function pointer that `fn_pointer_self` gives it, which has no parameter: a
            // check has no receiver to imply it (`impl_self_outlives`).
            .filter(|&i| {
                !self.traits[i].uses_self_outlives
                    || self.features.impl_self_outlives
                    || (self.features.fn_pointer_self
                        && !self.traits[i].supers.default
                        && !(self.traits[i].elided_self && !self.features.self_elision))
            })
            // The impl has the predicates of the trait.
            .filter(|&i| {
                self.features.impl_self_bounds
                    || !self.traits[i]
                        .sk
                        .preds
                        .iter()
                        .any(|p| names_self_in_bound(p))
            })
            // A generic whose bound names `Self`, which another predicate or a where clause of a
            // method uses in a leaked type, is F19 in an impl whose self type is leaked
            // (`impl_self_bound_cycle`).
            .filter(|&i| {
                let tr = &self.traits[i];
                let names: Vec<String> = tr.sk.tys.iter().map(|t| t.name.clone()).collect();
                self.features.impl_self_bound_cycle
                    || !self_bound_used(&tr.sk.preds, &tr.methods, &names)
            })
            // The impl has the where clause of the trait, and the methods with their predicates:
            // a predicate that names `Self::A` makes rustc prove the impl in its environment,
            // where the bound of the impl that names a leaked type is not normalized yet
            // (`impl_pred_leaked`).
            .filter(|&i| {
                self.features.impl_pred_leaked
                    || !(self.traits[i].self_assoc_in_pred && self.traits[i].preds_leakable)
            })
            // The same without a leaked type: a bound of the trait that names the `Item` of an
            // iterator that a predicate equates to a type is not normalized in the environment
            // either, so rustc itself cannot prove the impl there (E0277, with no `type-leak`).
            .filter(|&i| {
                !(self.traits[i].self_assoc_in_pred && item_bound_in_trait(&self.traits[i]))
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let idx = *self.pick(&candidates);
        self.traits[idx].implemented = true;
        let tr = self.traits[idx].clone();
        self.tags.extend(tr.sk.tags.iter().cloned());
        // The impl has the where clause and the methods of the trait.
        self.risk.cycle |= tr.risk.cycle;
        self.risk.anchor |= tr.risk.anchor;
        let allow_self = ann.map_or(true, |(_, a)| a);
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        // The impl declares the generics of the trait, and maybe one more.
        let mut sk = tr.sk.clone();
        let mut extra: Option<String> = None;
        let function_pointer_self = tr.uses_self_outlives && !self.features.impl_self_outlives;
        if self.chance(30) && !function_pointer_self {
            let mut bounds = tr.supers;
            bounds.clone |= self.chance(20);
            let mut texts = self.bound_texts(bounds);
            // Besides `Sized`, which a type parameter has by default.
            texts.retain(|t| !t.is_empty());
            sk.tys.push(TyParam {
                name: "E".to_string(),
                sized: true,
                bounds,
                bound_texts: texts,
                outlives: Vec::new(),
                iter: false,
                fixed_item: 0,
            });
            extra = Some("E".to_string());
        }
        let sc = sk.scope(&Scope {
            self_kind: SelfKind::Sized,
            assoc: tr.assoc.clone(),
            ..Default::default()
        });
        // The self type: it cannot name `Self`.
        let saved_no_self = std::mem::replace(&mut self.no_self, true);
        let mut self_ty = self.ty(
            &sc,
            Con {
                bounds: tr.supers,
                ..Con::sized()
            },
            0,
        );
        // Not a blanket impl, which would overlap with the impl for the host in the Referrer.
        // `T::Item` is as general.
        let top_projection = |text: &str| {
            sc.tys.iter().any(|p| {
                text == p.name
                    || text.starts_with(&format!("{}::", p.name))
                    || text.starts_with(&format!("<{} as ", p.name))
            }) && !text.contains(['(', ',', '<', '['])
                || text.starts_with('<')
        };
        if top_projection(&self_ty.text) {
            self_ty = Ty::new("u8");
        }
        if !self.features.fn_pointer_self && self_ty.text.contains("fn(") {
            self_ty = Ty::new("u8");
        }
        // A self type with a lifetime adds it to the inputs of `fn(&Self) -> &R`, where the copy
        // (and the witness) writes it out (`self_elision`).
        let lifetime_free = tr.elided_self && !self.features.self_elision;
        // `fn_pointer_self` asks for a function pointer to implement the trait.
        if self.features.fn_pointer_self && !tr.supers.default && extra.is_none() && !lifetime_free
        {
            self_ty = Ty::new("&'static unsafe fn(i32)");
        }
        if lifetime_free {
            // An entry of the self type with a bound `T: 'a` has `'a` in its arguments too.
            self_ty = Ty::new("u8");
        }
        self.no_self = saved_no_self;
        // An impl of a trait with associated types has a self type that no other one in the crate
        // has (`[u8; 9]`, `[u8; 10]`, ...), or `Self::A0` is ambiguous.
        let mut self_base = self_ty.text.clone();
        if !tr.assoc.is_empty() {
            let text = match &extra {
                Some(e) => format!("({}, {})", self_base, e),
                None => self_base.clone(),
            };
            if self.assoc_self_types.contains(&text) {
                self_base = format!("[u8; {}]", 9 + self.assoc_self_types.len());
            }
            let text = match &extra {
                Some(e) => format!("({}, {})", self_base, e),
                None => self_base.clone(),
            };
            self.assoc_self_types.push(text);
        }
        let self_text = match &extra {
            Some(e) => format!("({}, {})", self_base, e),
            None => self_base,
        };
        self.self_prob = saved_prob;

        let tr_path = if whole {
            format!("{}{}", tr.name, tr.sk.args())
        } else {
            let mut p = format!("::{}", self.cfg.krate);
            for m in path {
                p += &format!("::{}", m);
            }
            format!("{}::{}{}", p, tr.name, tr.sk.args())
        };
        let mut body: Vec<String> = Vec::new();
        for (n, _) in &tr.assoc {
            body.push(format!("type {} = u8;", n));
        }
        if tr.gat {
            body.push(if tr.gat_where {
                "type G<'g> = u8 where Self: 'g, Self: 'c;".to_string()
            } else {
                "type G<'g> = u8;".to_string()
            });
        }
        for (decl, value) in &tr.consts {
            body.push(format!("{} = {};", decl, value));
        }
        for m in &tr.methods {
            body.push(format!("{} {{ loop {{}} }}", m));
        }
        let scope = self.impl_scope(
            ann.is_some(),
            &format!("{}{}", sk.params(), sk.where_clause()),
        );
        let attr = self.attr_with(ann, scope);
        Some(format!(
            "{}impl{} {} for {}{} {{\n{}\n}}\n",
            attr,
            sk.params(),
            tr_path,
            self_text,
            sk.where_clause(),
            body.join("\n")
        ))
    }

    /// Whether an impl with these generics is analyzed through `enter_impl` (a third of those that
    /// can be), which visits the self type and the arguments of the trait only: the generics are
    /// not analyzed, so they must not name a type that the Leaker leaks, nor `Self`.
    fn impl_scope(&mut self, annotated: bool, generics: &str) -> &'static str {
        let relative = self
            .env
            .heads
            .iter()
            .map(|h| &h.name)
            .chain(self.env.theads.iter().map(|h| &h.name))
            .chain(self.env.consts.iter())
            .filter(|name| !name.starts_with("::"))
            // (`T` is a word of any generics: whether it is the type or a parameter is not told.)
            .any(|name| {
                if is_shadow_name(name) {
                    has_word(generics, name)
                } else {
                    generics.contains(name.as_str())
                }
            });
        if annotated && !relative && !generics.contains("Self") && self.chance(35) {
            "impl"
        } else {
            ""
        }
    }

    fn gen_impl_inherent(&mut self, ann: Option<(usize, bool)>) -> String {
        let allow_self = ann.map_or(true, |(_, a)| a);
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        // A copy of a module cannot have an inherent impl on a type of another module, which
        // the Referrer replaces by a projection.
        let own_only = self.no_self;
        // The impl has the predicates of the ADT.
        let adts: Vec<(String, Skeleton)> = self
            .adts
            .iter()
            .filter(|(_, sk)| {
                self.features.impl_self_bounds || !sk.preds.iter().any(|p| names_self_in_bound(p))
            })
            .cloned()
            .collect();
        let (sk, self_text) = if !adts.is_empty() && self.chance(55) {
            // An ADT of this module, with the generics it declares.
            let (name, sk) = self.pick(&adts).clone();
            self.tags.extend(sk.tags.iter().cloned());
            self.risk.cycle |= sk.risk_cycle;
            let text = format!("{}{}", name, sk.args());
            (sk, text)
        } else {
            // A type of the module or of `base`, with generics all used. In the copy of a module, a
            // parameter that only a leaked type uses is not constrained by the self type (E0207).
            let sk = if own_only {
                Skeleton::default()
            } else {
                self.skeleton(Level::Item, &Scope::default(), Bounds::default())
            };
            let sc = sk.scope(&Scope {
                self_kind: SelfKind::Sized,
                ..Default::default()
            });
            let saved_no_self = std::mem::replace(&mut self.no_self, true);
            self.no_assoc_paths = true;
            let mut ty = None;
            let saved_heads = self.env.heads.clone();
            let saved_consts = self.env.consts.clone();
            if own_only {
                self.env
                    .heads
                    .retain(|h| h.name.starts_with('R') || is_shadow_name(&h.name));
                // A constant of another module (`crate::base::BLEN`, `ILEN`) makes the whole self
                // type an entry, which the copy replaces by a projection: an inherent impl on it
                // is E0118 (a limitation, F11).
                self.env
                    .consts
                    .retain(|c| c.starts_with("::") || c == "RLEN" || c == "self::RLEN");
            }
            for _ in 0..20 {
                if let Some(t) = self.head_app(&sc, Con::sized(), 0) {
                    ty = Some(t);
                    break;
                }
            }
            self.env.heads = saved_heads;
            self.env.consts = saved_consts;
            self.no_assoc_paths = false;
            self.no_self = saved_no_self;
            let ty = ty.unwrap_or_else(|| Ty::new("R0"));
            (prune(&sk, &ty), ty.text)
        };
        let sc = sk.scope(&Scope {
            self_kind: SelfKind::Sized,
            self_lts: self_text.contains('\'') || !sk.lifetimes.is_empty(),
            ..Default::default()
        });
        let mut body: Vec<String> = Vec::new();
        for _ in 0..self.range(1, 3) {
            let m = self.name("m");
            let mut sig = self.method_sig(&sc, &m, false);
            // The copy of a check has no receiver, so `Self: 'x` would not be implied.
            if !self.features.impl_self_outlives && sig.contains("Self") {
                // Only the receiver: `(&'x self::R0, ..)` is a tuple with a type of the module.
                sig = sig
                    .replace("(&'x self,", "(&self,")
                    .replace("(&'x self)", "(&self)")
                    .replace("(&'y self,", "(&self,")
                    .replace("(&'y self)", "(&self)");
            }
            body.push(format!("pub {} {{ loop {{}} }}", sig));
        }
        if self.chance(CLASH_PERCENT) {
            for method in self.clash_methods() {
                body.push(format!("pub {} {{ loop {{}} }}", method));
            }
        }
        if self.chance(35) {
            let t = self.ty(&sc, Con::unsized_ok().indirect(), 1);
            let n = self.name("C");
            body.push(format!(
                "pub const {}: ::core::marker::PhantomData<{}> = ::core::marker::PhantomData;",
                n, t.text
            ));
        }
        self.self_prob = saved_prob;
        let scope = self.impl_scope(
            ann.is_some(),
            &format!("{}{}", sk.params(), sk.where_clause()),
        );
        let attr = self.attr_with(ann, scope);
        format!(
            "{}impl{} {}{} {{\n{}\n}}\n",
            attr,
            sk.params(),
            self_text,
            sk.where_clause(),
            body.join("\n")
        )
    }

    /// A free function of a module analyzed as a whole.
    fn gen_fn(&mut self) -> String {
        let name = self.name("f");
        let sig = self.method_sig(&Scope::default(), &name, false);
        format!("pub {} {{ loop {{}} }}\n", sig)
    }

    /// A type alias of a module analyzed as a whole.
    fn gen_alias(&mut self) -> String {
        let name = self.name("Al");
        let sk = self.skeleton(Level::Item, &Scope::default(), Bounds::default());
        let sc = sk.scope(&Scope::default());
        let ty = self.ty(&sc, Con::sized(), 0);
        let sk = prune(&sk, &ty);
        format!("pub type {}{} = {};\n", name, sk.params(), ty.text)
    }

    /// One item of the module `path`. `whole`: the module is analyzed as a whole.
    fn gen_one(&mut self, path: &[String], whole: bool) -> String {
        let (ann, allow_self) = if whole {
            (None, true)
        } else {
            let id = self.new_id();
            let allow_self = self.chance(70);
            (Some((id, allow_self)), allow_self)
        };
        let ann_id = ann.map(|(id, _)| id);
        self.last_scope = "";
        if ann.is_some() {
            self.risk = Risk::default();
            self.tags.clear();
            self.want_strict = self.chance(25);
            self.bare_percent = self.item_bare_percent();
        }
        // A quarter of the items may have predicates that name a leaked type with generics, and
        // those of a module analyzed as a whole only in some modules: a module is rejected with
        // its first such item (`bound_cycles`, the Leaker rejects them).
        let cyclic_percent = match (self.module_mode, self.cyclic_module) {
            (false, _) => CYCLIC_ITEM_PERCENT,
            (true, true) => CYCLIC_ITEM_IN_MODULE_PERCENT,
            (true, false) => 0,
        };
        self.cyclic = self.chance(cyclic_percent);
        if self.cyclic && self.features.bound_cycles {
            self.tags.insert(TAG_BOUND_CYCLE.to_string());
        }
        let pick = self.below(100);
        let (kind, text): (&'static str, String) = match pick {
            0..=21 => ("struct", self.gen_adt(AdtKind::Struct, ann)),
            22..=36 => ("enum", self.gen_adt(AdtKind::Enum, ann)),
            37..=43 => ("union", self.gen_adt(AdtKind::Union, ann)),
            44..=63 => ("trait", self.gen_trait(ann)),
            64..=75 => ("impl-inherent", self.gen_impl_inherent(ann)),
            76..=93 => match self.gen_impl_trait(ann, path, whole) {
                Some(text) => ("impl-trait", text),
                None => ("trait", self.gen_trait(ann)),
            },
            94..=96 if whole => {
                let text = self.gen_fn();
                self.contents.push("fn");
                return text;
            }
            97..=99 if whole => {
                let text = self.gen_alias();
                self.contents.push("alias");
                return text;
            }
            _ => ("struct", self.gen_adt(AdtKind::Struct, ann)),
        };
        if text.contains("Self::") || text.contains("<Self as ") {
            self.tags.insert(TAG_SELF_ASSOC.to_string());
        }
        if !allow_self && has_word(&text, "Self") {
            self.tags.insert(TAG_SELF_DENIED.to_string());
        }
        if names_self_in_where(&text) {
            self.tags.insert(TAG_SELF_IN_PREDICATE.to_string());
        }
        if let Some(id) = ann_id {
            self.items.push(Item {
                id,
                kind,
                module: path.to_vec(),
                strict: self.last_strict,
                allow_self,
                scope: self.last_scope,
                contents: vec![kind],
                tags: self.tags.clone(),
                source: text.clone(),
            });
        } else {
            self.contents.push(kind);
        }
        text
    }

    /// The body of the module `path`: the `use`s, the types it declares, items and modules.
    fn gen_module_body(
        &mut self,
        path: &[String],
        whole: bool,
        items: usize,
        nested: usize,
    ) -> String {
        let (env, uses) = self.build_env(path);
        if !whole {
            self.mod_uses.push((path.to_vec(), uses.clone()));
        }
        let saved_env = std::mem::replace(&mut self.env, env);
        let saved_traits = std::mem::take(&mut self.traits);
        let saved_adts = std::mem::take(&mut self.adts);
        let mut text = String::new();
        text += &uses;
        // A module analyzed as a whole has the `?Sized` of its types in its bounds, bare or not.
        let sized = if whole && self.bare_percent > 0 && self.chance(70) {
            if !self.cfg.prelude && !BARE_RELAXED_SIZED_ALWAYS_OK {
                self.tags.insert("bare:?Sized".to_string());
            }
            "?Sized"
        } else {
            "?::core::marker::Sized"
        };
        text += &types_template("R", sized);
        if !whole || self.features.declared_named_like_generic {
            for name in SHADOW_NAMES {
                text += &format!(
                    "#[derive(Debug, Clone, Copy, Default, PartialEq)] pub struct {};\n",
                    name
                );
            }
        }
        for _ in 0..items {
            text += &self.gen_one(path, whole);
        }
        for i in 0..nested {
            let name = format!("n{}", i);
            let mut sub = path.to_vec();
            sub.push(name.clone());
            let n = self.range(2, items.max(3));
            let body = self.gen_module_body(&sub, whole, n, 0);
            text += &format!("pub mod {} {{\n{}\n}}\n", name, body);
        }
        self.env = saved_env;
        self.traits = saved_traits;
        self.adts = saved_adts;
        text
    }
}

/// Generates the case `index` of the seed.
pub fn generate(seed: u64, index: usize, items_per_module: usize, features: Features) -> Case {
    let krate = format!("tlp_l_{:03}", index);
    let mut g = Gen::new(seed, krate.clone(), features);
    let mut lib = String::new();
    lib += "#![allow(warnings)]\n";
    lib += &format!("extern crate self as {};\n", krate);
    lib += "#[doc(hidden)]\npub use type_leak_test_macro as __macro;\n";
    lib += "type_leak_test_macro::define_repeater!(pub Repeater);\n";
    let mut base = types_template("B", "?::core::marker::Sized");
    // The supertrait that fixes the `Item` of an iterator, and an alias of its projection
    // (`supertrait_item`, `alias_item`).
    if g.features.supertrait_item {
        base += "pub trait BItem: ::core::iter::Iterator<Item = u8> {}\n";
    }
    if g.features.alias_item {
        base += "pub type BIt<T> = <T as ::core::iter::Iterator>::Item;\n";
    }
    lib += &format!("pub mod base {{\n{}\n}}\n", base);

    // Modules whose items are analyzed one by one.
    let n_item_modules = g.range(2, 3);
    for i in 0..n_item_modules {
        let name = format!("m{}", i);
        let n = g.range(4, items_per_module);
        let nested = usize::from(g.chance(50));
        let body = g.gen_module_body(std::slice::from_ref(&name), false, n, nested);
        lib += &format!("pub mod {} {{\n{}\n}}\n", name, body);
    }
    // Modules analyzed as a whole.
    let n_whole = g.range(1, 2);
    for i in 0..n_whole {
        let name = format!("mm{}", i);
        let id = g.new_id();
        g.no_self = true;
        g.module_mode = true;
        g.cyclic_module = g.chance(CYCLIC_MODULE_PERCENT);
        g.contents.clear();
        g.risk = Risk::default();
        g.tags.clear();
        g.want_strict = g.chance(25);
        g.bare_percent = g.module_bare_percent();
        let n = g.range(4, items_per_module);
        let nested = usize::from(g.chance(60));
        let body = g.gen_module_body(std::slice::from_ref(&name), true, n, nested);
        g.no_self = false;
        g.module_mode = false;
        g.cyclic_module = false;
        let strict = g.want_strict && g.strict_ok();
        let attr = g.attr(id, true, "", strict);
        let module = format!("{}pub mod {} {{\n{}\n}}\n", attr, name, body);
        lib += &module;
        let contents = std::mem::take(&mut g.contents);
        let tags = std::mem::take(&mut g.tags);
        g.items.push(Item {
            id,
            kind: "mod",
            module: Vec::new(),
            strict,
            allow_self: true,
            scope: "",
            contents,
            tags,
            source: module,
        });
    }

    // The Referrer's crate invokes the macro of each item.
    let mut referrer = String::from("#![allow(warnings)]\n");
    let mut referrer_lines = Vec::new();
    let mut items = g.items.clone();
    items.sort_by_key(|item| item.id);
    for item in &items {
        let mut path = format!("::{}", krate);
        for m in &item.module {
            path += &format!("::{}", m);
        }
        let line = referrer.lines().count() + 1;
        referrer += &format!(
            "{}::__tl_prop_{}!(path = {}{});\n",
            krate,
            item.id,
            path,
            if item.strict { ", strict" } else { "" }
        );
        referrer_lines.push((line, item.id));
    }
    let (accept_referrer, accept_lines) = accept_referrer(&krate, &g.mod_uses, &items);
    Case {
        config: g.cfg.clone(),
        leaker: lib,
        referrer,
        items,
        referrer_lines,
        accept_referrer,
        accept_lines,
    }
}

/// The `lib.rs` of the second Referrer's crate of `TYPE_LEAK_PROP_ACCEPT` (the crate `tlp_a_NNN`),
/// which copies the items that the Leaker accepts only by `LeakerResult::allow(cause)` of the
/// causes of its errors, and the line where each item is invoked.
///
/// The copy keeps the part that conflicts as written, whole: the paths in it are those the item has
/// in the Leaker's crate (`R0`, `self::R1`, `super::R1`, `super::base::B1`, `crate::base::B1`, an
/// import `I1` or `J1`), and it is up to the user of `allow()` that they resolve to the same items
/// in the crate of the copy. This crate makes them: the module of the Leaker's crate that the item
/// is in is declared again with the same `use`s and a glob import of the Leaker's module, in the same
/// place of the module tree (so that `super::` and `self::` mean the same), and the root has `base`,
/// the glob import of the Leaker's `base` (so that `crate::base::B1` means the same). The items are
/// copied in the module they are invoked in, as they are in the Leaker's crate (a module analyzed
/// as a whole declares itself at the root).
///
/// It is a crate of its own, not a module of the other Referrer's crate, because `crate::base` is
/// the root of the crate that has the copy: `base` at the root of the other crate would let an
/// ordinary item that keeps a relative path by mistake resolve it, and that crate must not have one.
fn accept_referrer(
    krate: &str,
    mod_uses: &[(Vec<String>, String)],
    items: &[Item],
) -> (String, Vec<(usize, usize)>) {
    fn invocation(krate: &str, item: &Item, path: &[String]) -> String {
        let mut module = format!("::{}", krate);
        for m in path {
            module += &format!("::{}", m);
        }
        format!(
            "::{}::__tl_prop_{}!(path = {}{}, accept);\n",
            krate,
            item.id,
            module,
            if item.strict { ", strict" } else { "" }
        )
    }
    fn module(
        krate: &str,
        path: &[String],
        mod_uses: &[(Vec<String>, String)],
        items: &[Item],
        out: &mut String,
    ) {
        if let Some((_, uses)) = mod_uses.iter().find(|(p, _)| p == path) {
            *out += uses;
        }
        *out += &format!("pub use ::{}::{}::*;\n", krate, path.join("::"));
        for item in items.iter().filter(|i| i.kind != "mod" && i.module == path) {
            *out += &invocation(krate, item, path);
        }
        for (sub, _) in mod_uses
            .iter()
            .filter(|(p, _)| p.len() == path.len() + 1 && p.starts_with(path))
        {
            *out += &format!("pub mod {} {{\n", sub.last().unwrap());
            module(krate, sub, mod_uses, items, out);
            *out += "}\n";
        }
    }
    let mut text = String::from("#![allow(warnings)]\n");
    text += &format!("pub mod base {{ pub use ::{}::base::*; }}\n", krate);
    for (path, _) in mod_uses.iter().filter(|(p, _)| p.len() == 1) {
        text += &format!("pub mod {} {{\n", path[0]);
        module(krate, path, mod_uses, items, &mut text);
        text += "}\n";
    }
    // A module analyzed as a whole declares itself, at the root.
    for item in items.iter().filter(|i| i.kind == "mod") {
        text += &invocation(krate, item, &[]);
    }
    let lines = items
        .iter()
        .filter_map(|item| {
            let call = format!("::__tl_prop_{}!(", item.id);
            let at = text.lines().position(|line| line.contains(&call))?;
            Some((at + 1, item.id))
        })
        .collect();
    (text, lines)
}

/// Whether the where predicate `pred` may name a type that the Leaker leaks: a name that is not
/// from `::core`, a primitive or a generic parameter (`params`). A relative path is, and an
/// absolute path of another crate is an optional entry that a strict Referrer replaces.
fn names_leakable(pred: &str, params: &[String]) -> bool {
    const KNOWN: &[&str] = &[
        "core",
        "ops",
        "Fn",
        "FnMut",
        "FnOnce",
        "fmt",
        "Debug",
        "convert",
        "Into",
        "AsRef",
        "iter",
        "Iterator",
        "Item",
        "clone",
        "Clone",
        "default",
        "Default",
        "marker",
        "Sized",
        "Send",
        "Sync",
        "Unpin",
        "Copy",
        "PartialEq",
        "PartialOrd",
        "Eq",
        "Ord",
        "From",
        "cmp",
        "PhantomData",
        "option",
        "Option",
        "result",
        "Result",
        "any",
        "Any",
        "Self",
        "for",
        "dyn",
        "fn",
        "unsafe",
        "extern",
        "mut",
        "const",
        "str",
        "C",
        "static",
        "true",
        "false",
    ];
    let chars: Vec<char> = pred.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if !(chars[i].is_alphanumeric() || chars[i] == '_') {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
            i += 1;
        }
        let word: String = chars[start..i].iter().collect();
        // A lifetime, a number, a name of the item's own, an associated type (`A0`).
        let lifetime = start > 0 && chars[start - 1] == '\'';
        let number = word.chars().next().unwrap().is_ascii_digit();
        let assoc = word.len() == 2 && word.starts_with('A');
        if !(lifetime
            || number
            || assoc
            || KNOWN.contains(&word.as_str())
            || PRIMS.contains(&word.as_str())
            || params.contains(&word))
        {
            return true;
        }
    }
    false
}

/// Whether the predicate `pred` names the `Item` of the parameter `x` (`x::Item`).
fn names_item(pred: &str, x: &str) -> bool {
    pred.contains(&format!("{}::Item", x))
        || pred.contains(&format!("<{} as ::core::iter::Iterator>::Item", x))
        || pred.contains(&format!("<{} as Iterator>::Item", x))
        || pred.contains(&format!("::base::BIt<{}>", x))
}

/// Whether the predicate `pred` equates the `Item` of the parameter `x` to a type.
fn binds_item(pred: &str, x: &str) -> bool {
    pred.starts_with(&format!("{}: ::core::iter::Iterator<Item = ", x))
        || pred.starts_with(&format!("{}: Iterator<Item = ", x))
}

/// Whether the where clause of `tr` names the `Item` of an iterator that a predicate of the trait or
/// of a method equates to a type, or that a supertrait or a bound fixes (`fixed_item`).
fn item_bound_in_trait(tr: &TraitInfo) -> bool {
    tr.sk.tys.iter().filter(|t| t.iter).any(|t| {
        let x = &t.name;
        tr.sk.preds.iter().any(|p| names_item(p, x))
            && (t.fixed_item != 0
                || tr.sk.preds.iter().any(|p| binds_item(p, x))
                || tr.methods.iter().any(|m| {
                    m.contains(&format!("{}: ::core::iter::Iterator<Item = ", x))
                        || m.contains(&format!("{}: Iterator<Item = ", x))
                }))
    })
}

/// Whether `text` names a primitive type.
fn names_primitive(text: &str) -> bool {
    [
        "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize",
        "f32", "f64", "bool", "char", "str",
    ]
    .iter()
    .any(|name| has_word(text, name))
}

/// Whether `text` has `name` as a word (an identifier).
fn has_word(text: &str, name: &str) -> bool {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(|word| word == name)
}

/// Whether a predicate of a where clause of the text (from `where` up to the `{` or the `;` that
/// ends the item) names `Self` or `Self::A`, other than the ones on `Self` itself (`Self: Sized`,
/// `Self: 'x`).
pub fn names_self_in_where(text: &str) -> bool {
    text.match_indices("where ").any(|(at, _)| {
        let clause = where_clause(&text[at + "where ".len()..]);
        split_predicates(clause)
            .iter()
            .any(|pred| !pred.trim_start().starts_with("Self:") && has_word(pred, "Self"))
    })
}

/// The where clause at the start of `text`: up to the first `{` or `;` that is not in brackets (an
/// array type has a `;` in it).
fn where_clause(text: &str) -> &str {
    let mut depth = 0i32;
    let mut previous = ' ';
    for (at, c) in text.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' => depth -= 1,
            '>' if previous != '-' => depth -= 1,
            '{' | ';' if depth <= 0 => return &text[..at],
            _ => (),
        }
        previous = c;
    }
    text
}

/// The predicates of a where clause: the text split at the commas that are not in brackets.
fn split_predicates(clause: &str) -> Vec<&str> {
    let (mut depth, mut start) = (0i32, 0);
    let mut out = Vec::new();
    let mut previous = ' ';
    for (at, c) in clause.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' => depth -= 1,
            '>' if previous != '-' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&clause[start..at]);
                start = at + 1;
            }
            _ => (),
        }
        previous = c;
    }
    out.push(&clause[start..]);
    out
}

/// Drops the predicates that use a generic whose bound names `Self` (`X: Fn(&Self) -> u8`): the
/// bound itself stays.
fn drop_self_bound_cycles(preds: &mut Vec<String>, params: &[String]) {
    let bound_of = |pred: &str| -> Option<String> {
        let (bounded, bounds) = pred.split_once(": ")?;
        (params.iter().any(|p| p == bounded) && has_word(bounds, "Self"))
            .then(|| bounded.to_string())
    };
    let cyclic: Vec<String> = preds.iter().filter_map(|p| bound_of(p)).collect();
    if cyclic.is_empty() {
        return;
    }
    preds.retain(|p| bound_of(p).is_some() || !cyclic.iter().any(|x| has_word(p, x)));
}

/// Whether a generic of `params` is in a predicate of `preds` that names `Self`
/// (`X: Fn(&Self) -> u8`, `Vec<X>: Into<Self::A>`), and in another predicate or in the where clause
/// of a method (`Rel<X>: Default`). The entry of the second has the first in its context.
fn self_bound_used(preds: &[String], methods: &[String], params: &[String]) -> bool {
    params
        .iter()
        .filter(|x| {
            preds
                .iter()
                .any(|p| names_self_in_bound(p) && has_word(p, x))
        })
        .any(|x| {
            preds
                .iter()
                .any(|p| !names_self_in_bound(p) && has_word(p, x))
                || methods
                    .iter()
                    .any(|m| m.find(" where ").map_or(false, |at| has_word(&m[at..], x)))
        })
}

/// Whether `text` names a lifetime other than `'static`.
fn names_lifetime(text: &str) -> bool {
    text.match_indices('\'').any(|(at, _)| {
        let name: String = text[at + 1..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        !name.is_empty() && name != "static"
    })
}

/// Whether the predicate `pred` names `Self` in a bound or in a bounded type that is not `Self`.
fn names_self_in_bound(pred: &str) -> bool {
    pred.contains("Self") && !pred.starts_with("Self:")
}

/// Parenthesizes a trait object with several bounds behind a reference or a pointer.
fn paren(text: &str) -> String {
    if text.starts_with("dyn ") && text.contains(" + ") {
        format!("({})", text)
    } else {
        text.to_string()
    }
}

/// A constant argument of a path: a path of several segments needs braces.
fn braced_path(len: &str, params: &[(String, &'static str)]) -> String {
    let is_param = params.iter().any(|(n, _)| n == len);
    let is_literal = len.chars().all(|c| c.is_ascii_digit());
    if is_param || is_literal {
        len.to_string()
    } else {
        format!("{{ {} }}", len)
    }
}

/// A return type of the sugar of `Fn` that ends with a trait object, which a bound after it would
/// join.
fn ret_paren(text: &str) -> String {
    if text.contains("dyn ") {
        format!("({})", text)
    } else {
        text.to_string()
    }
}

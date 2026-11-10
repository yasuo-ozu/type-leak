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
//! attribute asks for `nested_self_entries` (all the items nested in `enter_mod`),
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
//! (`Config::primitive`), and each item says which optional entries the Referrer trusts (`Trust`).
//!
//! Shapes made for the reasons and the causes that random types rarely reach (each a small share of
//! the items, decided by the hash of the state, which draws nothing, and put as a tag on what the item
//! writes: `TAG_NESTED_SELF_BOUNDS` and the tags after it, `main.rs` `SHAPE_TAGS` counts them):
//! `Self` in a struct, an enum or a union analyzed in `enter_generics` (`SelfWithoutMeaning`), in the
//! default of a parameter of a trait (`SelfInDefault`), in a where predicate of a trait on a type of the
//! module (`SelfInTraitPredicate`) and in the generics of an impl whose self type uses them
//! (`SelfInImplSelfType`); half of the items nested in `enter_mod` without `nested_self_entries`
//! (`NestedSelfInBounds`); with `values`, a glob import or a `macro_rules!` in a body (`Unreadable`) and
//! a relative `use`; a macro that expands to an array length (`two!()`, by its bare name: a cause of the
//! kind `Macro`, or by an absolute path), macro calls in values; `Rl<'_>` as an argument; an enum with
//! discriminants that are constants, types and macros; `<T>::Item`; and const parameters of the types
//! that are not `usize`, `bool`, `i32`, `u8` and `char`. Not reachable: `UnnameableSelf` (an item
//! nested in a scope that has a `Self` of its own, under generics that bound the outer one: no valid
//! item is that, only a macro that nests scopes) and a cause of the kind `Lifetime` (a lifetime that no
//! scope declares). The paths of kind `Unknown` (the root of a `use`, `<T>::X`) are in the `paths` of the
//! errors, which the report of the macro does not carry.
//!
//! The shapes that the Leaker must reject under every setting are forced (`FORCED_*` and some of the
//! tags above, `main.rs` `FORCED`): an item that has one is left out of the acceptance floors, and
//! fails the run unless the Leaker rejects it with an error of the shape's reason.
//!
//! Shapes that are accepted, made for the use sites (`inst`): one more field of a struct, an enum or a
//! union that is a function pointer or trait object whose output has an elided lifetime and whose input
//! names a type parameter (`ELIDED_PARAM_PERCENT`, F69), or is the optional entry `::std::vec::Vec<T>`
//! (`ELIDED_ENTRY_PERCENT`, F70). The test macro gives such a parameter a type that holds a lifetime in
//! most of the instantiations, and the copy must name the elided lifetime, written by the use site
//! (F69) or by the path through the Repeater that replaces the entry (F70).
//!
//! Not generated: type macros (the test macro's witness cannot return a type that a macro
//! expands to), and unsized or `dyn` implementors. Const expressions with types in them, and the
//! values of items, only with `Flags::values`; paths of `std` without the leading `::`, only with
//! `Flags::relative_std` (see [`Flags`]).
//!
//! The grammar is built to be valid: every type is generated with what the generics in scope
//! promise (sizedness, bounds, outlives), so the Leaker's crate compiles by itself. If it does not,
//! the harness builds it again without `type-leak` and reports a bug of the generator.
//!
//! What `type-leak` is known to fail on is avoided unless a feature asks for it ([`Features`]):
//! each is a finding of the test, with a reduced reproduction in the report of the test. The
//! generator also avoids the constructs that cannot be copied to another crate whatever
//! `type-leak` does, such as an inherent impl on a type of another module in the copy of a module.

use std::collections::BTreeSet;
use std::hash::Hasher;

// ---------------------------------------------------------------------------------------------
// Configuration and results.
// ---------------------------------------------------------------------------------------------

/// A hash that is the same on every release and platform: FNV-1a, finished by splitmix64.
/// `std`'s `DefaultHasher` is not stable across Rust releases, so a seed would generate other
/// cases after an update.
#[derive(Clone)]
pub struct StableHasher(u64);

impl StableHasher {
    pub fn new() -> Self {
        StableHasher(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }

    // The default is native-endian.
    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn finish(&self) -> u64 {
        let mut z = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// The fewest items a module can have (`TYPE_LEAK_PROP_ITEMS`): the generator draws between 4 and
/// that many.
pub const MIN_ITEMS: usize = 4;

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
    /// An optional entry in a where predicate that names a lifetime (`T: Fn(&'a ::ext::W<T,
    /// ::ext::B0>)`) has the predicate in its context, so a Referrer that replaces it (one that
    /// does not trust its crate: `Trust::None`, and the partial modes) uses a path that needs
    /// itself (E0277, F17): `not_optional` of the Leaker misses the type that holds another
    /// optional type. Fixed (on by default): the Leaker copies it as written. Off, the item that
    /// has it is invoked with `Trust::All`.
    pub strict_cycles: bool,
    /// A Referrer that replaces an optional entry that holds a lifetime of a method (`fn
    /// m<'x>(&self, a: ::ext::W<&'x u8>) -> Rel<'x>`): the copy has `'x` in no argument that
    /// constrains it, so it is early-bound, where it is late-bound in the trait (E0195, F18, fixed:
    /// the Leaker copies the type as written, on by default). Off, the item that has it is invoked
    /// with `Trust::All`.
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
    /// The Leaker is configured with `nested_self_entries` for the items nested in
    /// `enter_mod` (`nested_self_entries = true` in the attribute), and the structs, enums and unions
    /// of `self_bound_adts` have a field that names their `Self` in a relative type (`Rel2<Self, T>`
    /// with `T: AsRef<Self>`), which the Leaker rejects without the option (F42: on by default). The
    /// option is an opt-in of the Leaker, with risks that the generator's types do not have (a bound
    /// on the parameter of the leaked type that receives `Self`; rustc before 1.70 and the last
    /// field). It is on for half of the items nested in `enter_mod` (`Gen::nested_entries`), so the
    /// items that name `Self` in a field and have a bound that does are built there, and rejected
    /// (`NestedSelfInBounds`, the tag `nested_self_in_bounds`) in the other half.
    pub nested_self_entries: bool,
}

/// The constructs of the generator that `TYPE_LEAK_PROP_FEATURES` asks for besides `Features`, whose
/// `Debug` is in the tables: they are off by default, so the tables do not change.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Flags {
    /// `allow_stages`: for each rule of [`Flags::rules()`] the macro compares the errors that remain
    /// with the rule given to `Allow` and to `LeakerResult` for every item and makes the copy
    /// of each (`stages = true` in the attribute, which only asks it to report both counts and to
    /// write the copies), and the harness prints how many items differ and builds the copies
    /// (`main.rs`: "The copies after the analysis without a rule").
    pub allow_stages: bool,
    /// `relative_std`: the generated items name `std`, `core` and `alloc` without the leading `::` in
    /// about a third of their paths (`Gen::relativize_std()`), the Leaker allows them (`allow =
    /// "std,core,alloc"`), and the rule `std` of `allow_stages` has errors to drop.
    pub relative_std: bool,
    /// `values`: the generated items have values (`Gen::method_body()`, `Gen::value_const()`): the
    /// bodies of functions, methods and default methods, the values of constants and statics, the
    /// defaults of associated constants. They have the types of the Leaker's crate in type positions
    /// (a `let`, a turbofish, a cast), which the Leaker leaks with `check_values`, and some have a
    /// relative path of a value (`crate::base::bf()`), which it rejects. There are more defaults of
    /// const parameters that name a relative constant. The rules `values` and `default` of
    /// `allow_stages` have errors to drop.
    pub values: bool,
}

impl Flags {
    pub fn from_env() -> Self {
        let list = std::env::var("TYPE_LEAK_PROP_FEATURES").unwrap_or_default();
        let has = |name: &str| list.split(',').any(|word| word.trim() == name);
        Flags {
            allow_stages: has("allow_stages"),
            relative_std: has("relative_std"),
            values: has("values"),
        }
    }

    /// The rules `allow_stages` compares, by the names of the macro (`prop.rs`, `Rule::name()`).
    /// `std` and `values` are in only if the generator makes what they are about.
    pub fn rules(self) -> Vec<&'static str> {
        let mut rules = vec!["prelude", "crate", "path"];
        if self.relative_std {
            rules.push("std");
        }
        if self.values {
            rules.push("values");
        }
        rules.push("default");
        rules
    }
}

/// The rules of the environment (see [`Flags::rules()`]).
pub fn rules() -> Vec<&'static str> {
    Flags::from_env().rules()
}

/// Lists the fields of [`Features`] by name, for `Features::flag()`.
macro_rules! feature_fields {
    ($($name:ident),* $(,)?) => {
        impl Features {
            /// Every feature.
            const NAMES: &'static [&'static str] = &[$(stringify!($name)),*];

            /// The field of a feature, by its name.
            fn flag(&mut self, name: &str) -> Option<&mut bool> {
                match name {
                    $(stringify!($name) => Some(&mut self.$name),)*
                    _ => None,
                }
            }
        }
    };
}

feature_fields!(
    unsized_params,
    bound_cycles,
    inferred_outlives,
    impl_assoc_self,
    elided_fn_generics,
    bound_merge_syntax,
    impl_self_outlives,
    fn_pointer_self,
    method_lifetime_paths,
    module_crate_paths,
    impl_self_bounds,
    impl_pred_leaked,
    self_elision,
    implied_outlives,
    strict_cycles,
    strict_anchors,
    impl_self_bound_cycle,
    trait_assoc_bound_cycle,
    iterator_item_cycle,
    supertrait_item,
    alias_item,
    impl_trait_args,
    hidden_lifetimes,
    projection_outlives,
    item_predicate_cycle,
    self_projection_outlives,
    gat_where_clause,
    nested_self,
    path_named_like_generic,
    declared_named_like_generic,
    bare_const_in_absolute_path,
    trait_self_defaults,
    self_bound_adts,
    nested_self_entries,
);

impl Features {
    /// The features of the findings that are fixed are on by default.
    ///
    /// `list` (a comma-separated list, `TYPE_LEAK_PROP_FEATURES`) changes them: `name` turns a
    /// feature on (a known failure, or a default feature again), `-name` turns it off (also one that
    /// is on by default: `-nested_self_entries`), and `all` turns every feature on. They are read in
    /// order, so `all,-strict_anchors` is all but one. `allow_stages`, `relative_std` and `values`
    /// are no features (see [`Flags`]), and are skipped here.
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
            strict_anchors: true,
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
        for word in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            let (name, on) = match word.strip_prefix('-') {
                Some(name) => (name, false),
                None => (word, true),
            };
            match name {
                "all" => {
                    for name in Features::NAMES {
                        *features.flag(name).unwrap() = on;
                    }
                }
                // Read by `Flags::from_env()`: they are no features of the generator.
                "allow_stages" | "relative_std" | "values" => (),
                other => match features.flag(other) {
                    Some(flag) => *flag = on,
                    None => panic!("unknown feature `{}`", other),
                },
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
    /// The Leaker allows the primitive types (`Rules::primitive()`). Without it, `u8`
    /// is a relative path like any other: each is an entry of the item.
    pub primitive: bool,
}

impl Config {
    fn allow_arg(&self, flags: Flags) -> String {
        let mut allow = Vec::new();
        if self.allow_std {
            allow.push("::std".to_string());
        }
        if self.allow_crate {
            allow.push(format!("::{}", self.krate));
        }
        if flags.relative_std {
            allow.extend(["std", "core", "alloc"].map(String::from));
        }
        allow.join(",")
    }
}

/// The tags: the shapes of an item that make `type-leak` reject it, which the generator puts on the
/// items that have one (`Item::tags`). The harness fails a rejection that no tag of the item
/// justifies (`main.rs`, `family()`).
///
/// A tag is put on what the item writes, never on a share of the items that may. Some carry the
/// text they are about (`<tag>=<text>`, see `norm()`), which the rejection must match
/// (`TAG_BOUND_AT`, `TAG_LIFETIME_AT`).
///
/// - `bound_cycle`: a bound of a generic (`T: Fn(Rel) -> u8`, in the generics or a where
///   predicate) names a type that the Leaker leaks, or `Self` (`Gen::note_bounds()`); a predicate
///   of a method or an associated type bound that names a leaked type or `Self::A`.
/// - `hidden_bound=<generic>|<bound>`: a bound of a generic that has a bound fixing an associated
///   type names it in the arguments of a path (`U: Fn(BIt<U>) -> u8`), which may be a type alias
///   of the projection, and the item has an entry in a predicate that keeps the bound
///   (`hidden_bounds_in()`).
/// - `self_assoc`: `Self::A` or `<Self as Tr>::A` is in the text of the item.
/// - `binder_no_anchor`: a bound, function pointer or trait object `for<'h> Fn(A, Rl<'h>) -> &'h R`,
///   where no input keeps `'h` (the bound is always rejected, the others where they are split).
/// - `const_default`: a relative constant (`{ RLEN }`) is the default of a const parameter.
/// - `method_lifetime`: a leaked path or trait object names a lifetime of a method, in its
///   arguments too (`Rel<&'x str>`): `Gen::note_method_lifetime()`.
/// - `self_in_bounds`: a predicate names `Self` (`Self: 'x`, `Self: Sized`).
/// - `self_denied`: the item is analyzed with `allow_self = false` and names `Self`.
/// - `bare:<Name>`: the prelude trait `Name` is spelled bare, and the Leaker does not allow the names
///   of the prelude. A bare `?Sized` has none: `?` applies to the real `Sized` only, so the Leaker
///   accepts it whatever the prelude.
pub const TAG_BOUND_CYCLE: &str = "bound_cycle";
/// `bound_cycle=<bounded>|<bound>`: the bound that `Gen::note_bounds()` found, as `norm()` writes
/// it: `T|Fn(R0)->u8`. The Leaker names the bound that makes the cycle in its note.
pub const TAG_BOUND_AT: &str = "bound_cycle=";
/// `hidden_bound=<generic>|<bound>`: a bound of the generic (`norm()`) names it in the arguments of a
/// path that may be a type alias of a projection (`U: Fn(BIt<U>) -> u8`), another bound of the
/// generic fixes an associated type (`U: Iterator<Item = u8>`, or a trait of the crate that may
/// have a supertrait that does, `U: ::krate::base::BItem`), and the item has an entry that keeps
/// the bound in a predicate: `Self` or the self type of the impl in a bound or where predicate
/// (`hidden_bounds_in()`). The Leaker names the generic and the bound in its note.
pub const TAG_HIDDEN_AT: &str = "hidden_bound=";
/// `method_lifetime=<type>`: the leaked type that names the lifetime of a method (`norm()`).
pub const TAG_LIFETIME_AT: &str = "method_lifetime=";
/// `construct:<name>`: the item has the construct (`CONSTRUCTS`), for the statistics. They explain
/// nothing.
pub const TAG_CONSTRUCT: &str = "construct:";
/// The constructs the statistics count, each with the acceptance floor of its own: what the item
/// text has (`constructs_of()`), or what the generator wrote (`param_default`, `lifetime_param`).
pub const CONSTRUCTS: &[&str] = &[
    "gat",
    "const_param",
    "impl_trait_arg",
    "param_default",
    "binder",
    "dyn_object",
    "fn_pointer",
    "assoc_const",
    "lifetime_param",
    "unsized_param",
    "where_clause",
];
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
/// `value_path`: a body, a constant or a default has a relative path of a value (`crate::base::bf()`,
/// `RLEN`), which `Options::check_values()` rejects (`values`); a discriminant too. It explains the
/// rejection (`family()`), but is no forced shape: `forced:value_path` is, from the text.
pub const TAG_VALUE_PATH: &str = "value_path";
/// `relative_std`: a path of `std`, `core` or `alloc` is written without the leading `::` in the item.
/// The Leaker leaks it unless it allows them (the rule `std` of `allow_stages` does not), so in a
/// bound it is a leaked type: the shape `bound_cycle` has for the failures of `accept_families()`.
pub const TAG_RELATIVE_STD: &str = "relative_std";
/// The shapes that the generator writes to reach a `Reason` or a construct that the random types rarely
/// do (`main.rs`, `SHAPE_TAGS`). Each is put on what the item writes, and the ones that explain a
/// rejection are the only tag that does (`family()`):
///
/// - `nested_self_in_bounds`: a struct, an enum or a union analyzed nested in `enter_mod` without
///   `nested_self_entries`, a field of which names `Self` in a relative type next to a generic whose
///   bound names it (`Rel2<Self, T>` with `T: AsRef<Self>`): `Reason::NestedSelfInBounds`. With the
///   option the same item is accepted (`nested_self_in_bounds_option`, no rejection has it).
/// - `self_without_meaning`: a struct, an enum or a union analyzed in `enter_generics`, which gives
///   `Self` no meaning, names it in a field: `Reason::SelfWithoutMeaning`.
/// - `self_in_default`: the default of a parameter of a trait names `Self` (`trait Tr<T =
///   Rel<Self>>`): `Reason::SelfInDefault`.
/// - `self_in_trait_predicate`: a where predicate of a trait is on a leaked type that names `Self`
///   (`where Rel<Self>: Default`): `Reason::SelfInTraitPredicate`.
/// - `impl_header_self`: the generics or the where clause of an impl bound a generic by `Self`
///   (`impl<T: AsRef<Self>> Rel<T>`) and the self type uses it: `Reason::SelfInImplSelfType`.
/// - `unreadable_value`: with `values`, a body has syntax that the Leaker cannot read (a glob
///   import, a `macro_rules!`): `Reason::Unreadable`.
/// - `macro_path`: a relative macro in a type or a value (`two!()` in an array length): the
///   cause is a `Cause::Macro`.
/// - `elided_arg`: `Rel<'_>`, a relative type with the elided lifetime in its arguments.
/// - `discriminant`: an enum has discriminants, which the Leaker reads as values whether it reads
///   values or not.
/// - `qualified_path`: `<T>::X`, a qualified path with no trait (a cause of the kind `Unknown`).
/// - `macro_call`: a macro call in a type or a value of an item.
/// - `wide_const`: a const parameter of a type other than `usize`, `bool`, `i32`, `u8` and `char`.
pub const TAG_NESTED_SELF_BOUNDS: &str = "nested_self_in_bounds";
pub const TAG_NESTED_SELF_OPTION: &str = "nested_self_in_bounds_option";
/// `nested_self_in_bounds_text`: an item nested in `enter_mod` without `nested_self_entries` whose
/// generics or where clause have a bound that names `Self`, a shape that may reject it with
/// `Reason::NestedSelfInBounds` if a type in it names `Self` too. Read from the text, so it also
/// explains the items that are not of `self_bound_adts` (the Leaker may accept them).
pub const TAG_NESTED_SELF_TEXT: &str = "nested_self_in_bounds_text";
pub const TAG_SELF_NO_MEANING: &str = "self_without_meaning";
pub const TAG_SELF_IN_DEFAULT: &str = "self_in_default";
pub const TAG_SELF_IN_TRAIT_PRED: &str = "self_in_trait_predicate";
pub const TAG_IMPL_HEADER_SELF: &str = "impl_header_self";
pub const TAG_UNREADABLE: &str = "unreadable_value";
pub const TAG_MACRO_PATH: &str = "macro_path";
pub const TAG_ELIDED_ARG: &str = "elided_arg";
pub const TAG_DISCRIMINANT: &str = "discriminant";
pub const TAG_QUALIFIED_PATH: &str = "qualified_path";
pub const TAG_MACRO_CALL: &str = "macro_call";
pub const TAG_WIDE_CONST: &str = "wide_const";
/// The shapes the Leaker must reject, under the settings of every item (`main.rs`, `FORCED`, which
/// says which `Reason` rejects each). Of the tags above `nested_self_in_bounds`,
/// `self_without_meaning`, `self_in_default` and `unreadable_value` are such shapes. The others are
/// read from what the text of the item itself writes (`forced_in()`, in `Gen::note_text()`: an impl
/// does not take them from its type or its trait, a rejected candidate of the generator leaves none),
/// because the tag above of the same shape is also on items that the Leaker accepts:
///
/// - `forced:self_in_trait_predicate`: a where predicate of a trait is on a type of a path with no
///   leading `::` that names `Self` in its arguments (`where Rel<Self>: Default`); the tag above is
///   also on a bound `T: AsRef<Self>`, and on the impls of such traits.
/// - `forced:impl_header_self`: the self type of an impl is a path with no leading `::` whose
///   arguments name a generic that a bound of the impl bounds by `Self` (`impl<T: AsRef<Self>>
///   Rel<T>`); the tag above is also on a self type that no leaked type of it names the generic in.
/// - `forced:macro_path`: a macro is called by its bare name in the item (`[u8; two!()]`); an impl
///   does not write the macro of its type.
/// - `forced:elided_arg`: `ae: Rel<'_>` in a method, `Rel` a path with no leading `::` (a type of
///   another crate is written as it is).
/// - `forced:relative_discriminant`: a discriminant that is a constant of the module or of its
///   parent (`RLEN`, `super::base::BLEN`), not one of the Leaker's crate by an absolute path.
/// - `forced:value_path`: with `values`, a body or a constant writes a path of a value that the
///   Leaker's rules do not allow: a relative one (`RLEN`, `crate::base::bf()`), the root of a `use`
///   (`use super::base::bf as local_bf;`), or a name of the prelude where the attribute says
///   `prelude = false` (`Vec::<*const R1>::new()`). Read from the text and its attribute
///   (`value_paths_in()`): not where the attribute says `scope = "impl"`, which reads no body. The
///   tag `value_path` is also on a discriminant, and on the impls of an item with such a body.
pub const TAG_FORCED: &str = "forced:";
pub const FORCED_SELF_IN_TRAIT_PRED: &str = "forced:self_in_trait_predicate";
pub const FORCED_IMPL_HEADER_SELF: &str = "forced:impl_header_self";
pub const FORCED_MACRO_PATH: &str = "forced:macro_path";
pub const FORCED_ELIDED_ARG: &str = "forced:elided_arg";
pub const FORCED_RELATIVE_DISCRIMINANT: &str = "forced:relative_discriminant";
pub const FORCED_VALUE_PATH: &str = "forced:value_path";
/// The shapes that the items accepted by `LeakerResult::allow()` fail on (`TYPE_LEAK_PROP_ACCEPT`;
/// `accept_families()` in `main.rs` lists the failures). They explain no rejection:
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

/// Which optional entries (a type of another crate by an absolute path) the Referrer of an item
/// trusts: the rule it gives to `LeakerResult::allow()`. The rest are replaced through the Repeater.
/// The trusted crates are promised to mean the same in the Referrer's crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trust {
    /// Nothing (the library's default, `strict`).
    None,
    /// Every crate (`Rules::external_crates()`).
    All,
    /// `std`, `alloc` and `core` (`Rules::std_crates()`).
    Std,
    /// The Leaker's crate only (a path rule).
    Leaker,
}

impl Trust {
    pub const ALL: [Trust; 4] = [Trust::None, Trust::All, Trust::Std, Trust::Leaker];

    pub fn name(self) -> &'static str {
        match self {
            Trust::None => "none",
            Trust::All => "all",
            Trust::Std => "std",
            Trust::Leaker => "leaker",
        }
    }

    /// The flag of the Referrer's invocation.
    fn flag(self) -> String {
        format!(", trust({})", self.name())
    }
}

/// An item with a macro on it, which the Referrer invokes.
#[derive(Clone)]
pub struct Item {
    pub id: usize,
    pub kind: &'static str,
    /// The module that contains it, from the root.
    pub module: Vec<String>,
    /// Which optional entries the Referrer trusts.
    pub trust: Trust,
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
    /// The `lib.rs` of the crate that reports the errors the Leaker carries (`tlp_e_NNN`,
    /// `errors_crate()`): it does not compile when an item has errors that the invocation does not
    /// allow, and each is a diagnostic that the harness matches with the Leaker's.
    pub errors_referrer: String,
    /// The line of `accept_referrer` where each item is invoked.
    pub accept_lines: Vec<(usize, usize)>,
    /// The Referrer's crates of `TYPE_LEAK_PROP_FEATURES=allow_stages` (`stage_crates()`), which the
    /// harness writes with that feature only.
    pub stages: Vec<StageCrate>,
    /// The decoy crates of the case, if it has them (`decoy_crates()`).
    pub decoy: Option<Decoy>,
}

/// A Referrer's crate whose dependency `tlp_l_NNN` is another crate than the Leaker's, with the
/// same names in `base` and other definitions (`Decoy::lib`), so that an optional entry that the
/// Referrer trusts although it should replace it resolves to a type of the wrong crate, and the
/// witness fails. The Leaker's crate is the dependency `real_NNN` of it.
pub struct Decoy {
    /// The `lib.rs` of the decoy crate (`tlp_d_NNN`): `base` of the Leaker, defined again.
    pub lib: String,
    /// The `lib.rs` of the Referrer's crate (`tlp_x_NNN`), which invokes the macros of the items
    /// that do not trust the Leaker's crate.
    pub referrer: String,
    /// The line of `referrer` where each item is invoked.
    pub lines: Vec<(usize, usize)>,
}

/// A Referrer's crate of `TYPE_LEAK_PROP_FEATURES=allow_stages`: it copies each item as the macro of
/// the Leaker analyzed it without the rule and gave its result the rule and the causes of the errors
/// that remain (`side` `res`), or as it analyzed it with a config that has the rule (`cfg`).
pub struct StageCrate {
    /// The rule, as `rules()` names it.
    pub rule: &'static str,
    /// `res` or `cfg`.
    pub side: &'static str,
    /// The crate has none of the Leaker's names but the ones the rule allows, and copies the items
    /// that the rule leaves no error in (`stage_crates()`). Else it is built like the crate of the
    /// accepted items (`accept_referrer()`).
    pub clean: bool,
    /// Its `lib.rs`.
    pub lib: String,
    /// The line of `lib` where each item is invoked.
    pub lines: Vec<(usize, usize)>,
}

pub const ATTR: &str = "@ATTR@";

/// The macro of the Leaker's crate that an array length names (`gen_discriminant_enum()`,
/// `Gen::array()`, the decoy's crate has it too).
const TWO_MACRO: &str = "#[macro_export]\nmacro_rules! two { () => { 2 }; }\n";

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
/// The share of the structs, enums and unions with a type parameter that have one more field: a
/// function pointer or trait object that names a type parameter in the input that gives its elided
/// lifetime to the output ([`elided_param_signature()`]). A use site that gives the parameter a type
/// with a lifetime must name that lifetime (F69), and the random types rarely write such a
/// signature around a parameter that a use site can choose the type of.
const ELIDED_PARAM_PERCENT: u32 = 30;
/// The share of the structs, enums and unions with a `Sized` type parameter that has only marker
/// bounds (so that a use site can choose its type) that have one more field: a function pointer or
/// trait object whose elided output takes the lifetime of an input that is the optional entry
/// `::std::vec::Vec<T>` ([`elided_entry_signature()`]). Such an entry is replaced by a strict
/// Referrer in the cases where the Leaker does not allow `::std`, which is a share of the items, and
/// only then does the path through the Repeater name the lifetimes of the use site (F70): so it is
/// made often.
const ELIDED_ENTRY_PERCENT: u32 = 80;

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
    /// ... of a method of a trait, which an impl must match (E0195).
    in_trait_sig: bool,
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
    ("Ku16", Shape::CV("u16")),
    ("Ku32", Shape::CV("u32")),
    ("Ku64", Shape::CV("u64")),
    ("Ku128", Shape::CV("u128")),
    ("Ki8", Shape::CV("i8")),
    ("Ki16", Shape::CV("i16")),
    ("Ki64", Shape::CV("i64")),
    ("Ki128", Shape::CV("i128")),
    ("Kisize", Shape::CV("isize")),
];

/// The types of const parameters besides `usize`, `bool`, `i32`, `u8` and `char`: a type of the
/// module has a parameter of each (`RKu16<const V: u16>`), and an item may declare one
/// (`wide_const`). The Repeater has a slot for each type of const parameter, and the `u128` and
/// `i128` ones need the wide one.
const WIDE_CONSTS: &[(&str, &str)] = &[
    ("Ku16", "u16"),
    ("Ku32", "u32"),
    ("Ku64", "u64"),
    ("Ku128", "u128"),
    ("Ki8", "i8"),
    ("Ki16", "i16"),
    ("Ki64", "i64"),
    ("Ki128", "i128"),
    ("Kisize", "isize"),
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
    /// How the module names the root of the crate's modules: `super::` for each level of the path
    /// (`super::super::` in `m0::n0`), for `base` (`values`).
    up: String,
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
    for (name, ty) in [("I", "i32"), ("U", "u8"), ("H", "char")]
        .iter()
        .chain(WIDE_CONSTS)
    {
        s += &format!("pub struct {p}{name}<const V: {ty}>;\n");
    }
    let wide: Vec<(String, String, String)> = WIDE_CONSTS
        .iter()
        .map(|(name, ty)| {
            (
                name.to_string(),
                format!("const V: {}", ty),
                "V".to_string(),
            )
        })
        .collect();
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
    ]
    .iter()
    .map(|(n, g, a)| (n.to_string(), g.to_string(), a.to_string()))
    .chain(wide)
    {
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

/// The traits of the prelude that the generator spells bare in bounds, with their full paths: the
/// names of the tags `bare:<Name>`. `tests/crates/macro/bare.rs` has these and more.
pub const PRELUDE_TRAITS: &[(&str, &str)] = &[
    ("Sized", "::core::marker::Sized"),
    ("Send", "::core::marker::Send"),
    ("Sync", "::core::marker::Sync"),
    ("Unpin", "::core::marker::Unpin"),
    ("Copy", "::core::marker::Copy"),
    ("Clone", "::core::clone::Clone"),
    ("Default", "::core::default::Default"),
    ("Fn", "::core::ops::Fn"),
    ("FnMut", "::core::ops::FnMut"),
    ("FnOnce", "::core::ops::FnOnce"),
    ("Into", "::core::convert::Into"),
    ("From", "::core::convert::From"),
    ("AsRef", "::core::convert::AsRef"),
    ("Iterator", "::core::iter::Iterator"),
    ("PartialEq", "::core::cmp::PartialEq"),
    ("PartialOrd", "::core::cmp::PartialOrd"),
    ("Eq", "::core::cmp::Eq"),
    ("Ord", "::core::cmp::Ord"),
];

/// The full path of a trait of `PRELUDE_TRAITS`.
fn absolute_trait(name: &str) -> &'static str {
    PRELUDE_TRAITS
        .iter()
        .find(|(bare, _)| *bare == name)
        .map(|(_, absolute)| *absolute)
        .unwrap_or_else(|| panic!("generator: `{}` is not a prelude trait it writes", name))
}

/// Whether `name` is a trait of `PRELUDE_TRAITS`.
pub fn is_prelude_trait(name: &str) -> bool {
    PRELUDE_TRAITS.iter().any(|(bare, _)| *bare == name)
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

/// What makes an item break under a Referrer that replaces optional entries
/// (`strict_cycles`, `strict_anchors`): the item is invoked with `Trust::All` then, unless a feature
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
    flags: Flags,
    /// The state of the random numbers: each one is the hash of the seed and of what was drawn
    /// before (`finish()` does not reset a `StableHasher`).
    rng: StableHasher,
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
    /// The mode the item being generated is invoked with if it can be (`item_trust()`).
    want_trust: Trust,
    /// The mode in the attribute of the item generated last.
    last_trust: Trust,
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
    /// The item (or module) being generated has a value (`Flags::values`): the attribute says
    /// `values = true`.
    values_used: bool,
    /// The item being generated is analyzed with `nested_self_entries` where it is nested in
    /// `enter_mod` (`nested_self_entries = true` in the attribute): the feature, and half of the items.
    nested_entries: bool,
    /// The item being generated has an argument `impl Into<Rel<X>>` already (the plain one of
    /// `impl_trait_args`): one is enough. Two of them in methods whose generics are bounded by the same
    /// trait spelled bare in one and in full in the other are one entry where the traits are in
    /// full and two where they are bare (the twin of the bare spellings differs, which is no defect).
    plain_ai: bool,
}

const PRIMS: &[&str] = &[
    "u8", "u16", "u32", "i32", "i64", "usize", "bool", "char", "f64", "()",
];

impl Gen {
    /// A generator with the settings `cfg`, for the checks of what the tags read in a text
    /// (`note_text()`, `names_leaked_type()`).
    pub fn with_config(cfg: Config) -> Self {
        let mut g = Gen::new(1, cfg.krate.clone(), Features::default(), Flags::default());
        g.cfg = cfg;
        g
    }

    /// The tags put so far (`Item::tags`).
    pub fn tags(&self) -> &BTreeSet<String> {
        &self.tags
    }

    pub fn new(seed: u64, krate: String, features: Features, flags: Flags) -> Self {
        let mut rng = StableHasher::new();
        rng.write_u64(seed);
        let cfg = Config {
            krate,
            prelude: false,
            allow_std: false,
            allow_crate: false,
            primitive: true,
        };
        let mut g = Gen {
            flags,
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
            want_trust: Trust::All,
            last_trust: Trust::All,
            assoc_self_types: Vec::new(),
            in_default: false,
            redeclare: false,
            mod_uses: Vec::new(),
            values_used: false,
            nested_entries: false,
            plain_ai: false,
        };
        g.cfg.prelude = g.chance(50);
        g.cfg.allow_std = g.chance(35);
        g.cfg.allow_crate = g.chance(30);
        g.cfg.primitive = !g.chance(PRIMITIVE_OFF_PERCENT);
        g
    }

    /// The next random number: the hash of the seed and of what was drawn before.
    /// A number from the state and `tag`, which draws nothing: the other choices of the generator
    /// are the same with or without what it decides.
    fn hash_draw(&self, tag: &[u8]) -> u64 {
        let mut state = self.rng.clone();
        state.write(tag);
        state.finish()
    }

    /// `limit` percent of the time, decided by `hash_draw()`.
    fn hash_percent(&self, tag: &[u8], limit: u32) -> bool {
        self.hash_draw(tag) % 100 < u64::from(limit)
    }

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
    /// that the Leaker rejects, and the tag `bare:<name>` says so (`bare_traits_in()`, from the
    /// text of the item).
    fn trait_path(&mut self, name: &'static str) -> String {
        if self.bare_percent > 0 && self.chance(self.bare_percent) {
            name.to_string()
        } else {
            absolute_trait(name).to_string()
        }
    }

    /// `?Sized`, bare or in full (`?::core::marker::Sized`). A bare one is accepted whatever the
    /// prelude: `?` applies to the real `Sized` only, so it has no tag.
    fn maybe_sized(&mut self) -> String {
        if self.bare_percent > 0 && self.chance(self.bare_percent) {
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
                // A qualified path with no trait, which rustc resolves through the bounds of the
                // parameter (a type in a signature or a field: the checks of the predicates read
                // `T::Item`).
                if !sc.pred && self.hash_percent(b"qualified_item", 35) {
                    self.tags.insert(TAG_QUALIFIED_PATH.to_string());
                    format!("<{}>::Item", p.name)
                } else {
                    format!("{}::Item", p.name)
                }
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
            // A type that holds a leaked type that holds a lifetime of the method (a tuple, an array, a
            // path of `std`, a pointer, a function pointer) is the holder of a culprit: the Leaker may
            // report the types in it on the way to the lifetime, so the tag names the holder too (a
            // reference keeps its lifetime, and stops the way).
            let holds = matches!(pick, 0..=40 | 65..=74 | 85..=89);
            let ty = match pick {
                0..=10 => self.tuple(sc, c, depth),
                11..=17 => self.array(sc, c, depth),
                18..=34 => self.std_wrap(sc, c, depth),
                35..=40 => self.std_two(sc, c, depth),
                41..=54 => {
                    let ty = self.head_app(sc, c, depth);
                    self.note_method_lifetime(sc, ty.as_ref());
                    ty
                }
                55..=64 => self.reference(sc, c, depth),
                65..=68 => self.pointer(sc, c, depth),
                69..=74 => self.fn_ptr(sc, c, depth),
                75..=84 => {
                    let ty = self.dyn_obj(sc, c, depth);
                    self.note_method_lifetime(sc, ty.as_ref());
                    ty
                }
                85..=89 if !c.sized => self.slice_or_str(sc, c, depth),
                _ => None,
            };
            if let Some(ty) = ty {
                if holds {
                    self.note_method_lifetime(sc, Some(&ty));
                }
                return ty;
            }
        }
        self.leaf(sc, c)
    }

    /// Tags `method_lifetime=<type>` for a leaked path or trait object that holds a lifetime of the
    /// method, in its arguments too (`Rel<&'x str>`, `&'x dyn Tr`).
    fn note_method_lifetime(&mut self, sc: &Scope, ty: Option<&Ty>) {
        let Some(ty) = ty else { return };
        let generics: BTreeSet<String> = sc
            .tys
            .iter()
            .map(|t| t.name.clone())
            .chain(sc.consts.iter().map(|(n, _)| n.clone()))
            .collect();
        if ty.lts.iter().any(|l| sc.ref_only.contains(l))
            && self.names_leaked_type(&ty.text, &generics)
        {
            self.tags.insert(TAG_METHOD_LIFETIME.to_string());
            self.tags
                .insert(format!("{}{}", TAG_LIFETIME_AT, norm(&ty.text)));
        }
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
        // A macro call that expands to the length: the Leaker's own, by an absolute path (written
        // as it is) or the bare name of the macro, which the copy cannot resolve (`Cause::Macro`).
        // Only as the whole type of a position: a holder around it that the Leaker leaks is an
        // error (the array cannot be leaked whole, and the holder's path is relative).
        if depth == 0 && self.hash_percent(b"macro_len", 12) {
            let text = if self.module_mode || self.hash_percent(b"macro_len which", 50) {
                format!("::{}::two!()", self.cfg.krate)
            } else {
                self.tags.insert(TAG_MACRO_PATH.to_string());
                "two!()".to_string()
            };
            self.tags.insert(TAG_MACRO_CALL.to_string());
            return Some(Ty::with(format!("[{}; {}]", inner.text, text), &[&inner]));
        }
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
            (*self.pick(&pool)).clone()
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
            let mut ty = match self.below(100) {
                0..=59 => "usize",
                60..=74 => "bool",
                75..=84 => "i32",
                85..=91 => "u8",
                _ => "char",
            };
            // A tenth of them have a type that the others never have (`wide_const`), decided by the
            // hash of the state: the other choices are the same.
            let draw = self.hash_draw(format!("wide_const {}", name).as_bytes());
            if draw % 100 < 10 {
                ty = WIDE_CONSTS[(draw / 100) as usize % WIDE_CONSTS.len()].1;
            }
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
        if !sk.lifetimes.is_empty() {
            self.tags.insert(format!("{}lifetime_param", TAG_CONSTRUCT));
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
    ///
    /// `self_default`: the default may name `Self` (`trait Tr<T = Rel<Self>>`), which a struct, an
    /// enum or a union cannot write (E0735): `Reason::SelfInDefault`. Not for a trait nested in
    /// `enter_mod`: the Leaker accepts it with `nested_self_entries` (the copy passes the host),
    /// and the test macro cannot compare such a default (its witness for a default that names
    /// `Self` is made for the bounds that do, F39).
    fn defaults(&mut self, sk: &mut Skeleton, self_default: bool) {
        // `values` asks for more defaults, and more of the constants among them, which are errors
        // (not in a module analyzed as a whole, which one such error rejects).
        let more = self.flags.values && !self.module_mode;
        if !self.chance(if more { 60 } else { DEFAULT_PERCENT }) {
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
                } else if ty == "usize"
                    && !self.env.consts.is_empty()
                    && self.chance(if more { 60 } else { 25 })
                {
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
                // Not with `allow_stages`: the rule `default` drops the error and the copy keeps the
                // relative path of the default (`R1<Self>`), which the clean crate of the copy after
                // the result, with the constants only, cannot resolve.
                if self_default
                    && !self.flags.allow_stages
                    && !self.no_self
                    && declared.is_empty()
                    && self.hash_percent(b"self_in_default", 40)
                {
                    let heads: Vec<String> = self
                        .env
                        .heads
                        .iter()
                        .filter(|head| head.shape == Shape::T1 && !head.name.starts_with("::"))
                        .filter(|head| !is_shadow_name(&head.name))
                        .map(|head| head.name.clone())
                        .collect();
                    if !heads.is_empty() {
                        let at = self.hash_draw(b"self_in_default head") as usize % heads.len();
                        text = Some(format!("{}<Self>", heads[at]));
                        self.tags.insert(TAG_SELF_IN_DEFAULT.to_string());
                    }
                }
                sk.defaults
                    .push((sk.tys[i].name.clone(), text.unwrap_or_else(|| "u8".into())));
            }
        }
        self.in_default = false;
        self.no_assoc_paths = saved;
        if !sk.defaults.is_empty() {
            self.tags.insert(format!("{}param_default", TAG_CONSTRUCT));
        }
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
            "u16" => &["0", "65535"],
            "u32" => &["0", "4294967295"],
            "u64" => &["0", "18446744073709551615"],
            "u128" => &["0", "340282366920938463463374607431768211455"],
            "i8" => &["0", "-128", "127"],
            "i16" => &["0", "-32768"],
            "i64" => &["0", "-1", "9223372036854775807"],
            "i128" => &["0", "-170141183460469231731687303715884105728"],
            "isize" => &["0", "-1"],
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
        sc.in_trait_sig = in_trait;
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
        // A type of the module with a lifetime, with the elided one (`Rl<'_>`), as an argument: a
        // relative path cannot be leaked whole with it, and one of another crate is written as it is.
        if !self.module_mode && self.hash_percent(format!("elided_arg {}", name).as_bytes(), 2) {
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::LOnly)
                .map(|head| head.name.clone())
                .collect();
            if !heads.is_empty() {
                let at = self.hash_draw(b"elided_arg head") as usize % heads.len();
                params.push(format!("ae: {}<'_>", heads[at]));
                self.tags.insert(TAG_ELIDED_ARG.to_string());
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
        } else if self.features.impl_trait_args && !self.cyclic && self.cfg.primitive {
            // The same argument where no bound names a leaked type, which the Leaker accepts
            // (`ai: impl Into<Rel<X>>`; the predicates of an item that is not cyclic name types
            // that are not leaked, and the primitive types are allowed). Decided by the hash of
            // the state and of the name, which draws nothing: the other choices are the same as
            // without it.
            let mut state = self.rng.clone();
            state.write(name.as_bytes());
            state.write(b"impl_trait_plain");
            let draw = state.finish();
            let module_mode = self.module_mode;
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::T1 && !head.name.starts_with("::"))
                .filter(|head| !(module_mode && is_declared(&head.name)))
                .filter(|head| !is_shadow_name(&head.name))
                .map(|head| head.name.clone())
                .collect();
            if draw % 100 < 30 && !heads.is_empty() && !self.plain_ai {
                self.plain_ai = true;
                let head = &heads[(draw / 100) as usize % heads.len()];
                let sized: Vec<&TyParam> = sk.tys.iter().filter(|p| p.sized).collect();
                let arg = if !sized.is_empty() && (draw / 10_000) % 2 == 0 {
                    sized[(draw / 20_000) as usize % sized.len()].name.clone()
                } else {
                    "u8".to_string()
                };
                params.push(format!(
                    "ai: impl {}<{}<{}>>",
                    absolute_trait("Into"),
                    head,
                    arg
                ));
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

    /// Whether `text`, a bound or a type, names a type that the Leaker leaks, read from its paths
    /// and the settings (`Config`): a relative path, `Self`, a projection (`T::Item`), a path of
    /// `std` or of the Leaker's crate that it does not allow, a primitive type if it does not allow
    /// them, a bare type of the prelude, or a bare trait in a trait object, if it does not allow the
    /// prelude. `generics`: those in scope; `T` is a type of the module where none is.
    pub fn names_leaked_type(&self, text: &str, generics: &BTreeSet<String>) -> bool {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let is_id = |c: char| c.is_alphanumeric() || c == '_';
        // The depth in brackets, and the depth of the trait object (`dyn A + B`) being read: a bare
        // trait of the prelude in it is a path in a type, which is leaked where the prelude is not
        // allowed (as a bound `T: Fn(..)` it is a trait, another reason).
        let mut depth = 0i32;
        let mut dyn_at: Option<i32> = None;
        let mut i = 0;
        while i < n {
            let c = chars[i];
            if c == '"' {
                i += 1;
                while i < n && chars[i] != '"' {
                    i += 1;
                }
                i += 1;
                continue;
            }
            if c == '\'' || c.is_ascii_digit() {
                i += 1;
                while i < n && is_id(chars[i]) {
                    i += 1;
                }
                continue;
            }
            let leading = c == ':' && i + 1 < n && chars[i + 1] == ':';
            if !(leading || is_id(c)) {
                match c {
                    '(' | '[' | '<' => depth += 1,
                    ')' | ']' => depth -= 1,
                    '>' if i == 0 || chars[i - 1] != '-' => depth -= 1,
                    ',' if dyn_at == Some(depth) => dyn_at = None,
                    _ => (),
                }
                if dyn_at.map_or(false, |at| depth < at) {
                    dyn_at = None;
                }
                i += 1;
                continue;
            }
            // `<T as Tr>::Item`: the `::` that follows a `>` is a projection.
            let projection = leading
                && chars[..i]
                    .iter()
                    .rev()
                    .find(|c| !c.is_whitespace())
                    .map_or(false, |c| *c == '>');
            let mut segments: Vec<String> = Vec::new();
            if leading {
                i += 2;
            }
            loop {
                let start = i;
                while i < n && is_id(chars[i]) {
                    i += 1;
                }
                if start == i {
                    break;
                }
                segments.push(chars[start..i].iter().collect());
                if i + 2 < n && chars[i] == ':' && chars[i + 1] == ':' && is_id(chars[i + 2]) {
                    i += 2;
                } else {
                    break;
                }
            }
            if projection {
                return true;
            }
            let Some(head) = segments.first().map(String::as_str) else {
                continue;
            };
            // `Item = u8` and `Out = R0` name an associated type, which binds.
            let rest: String = chars[i..].iter().take(3).collect();
            if segments.len() == 1 && rest.trim_start().starts_with('=') {
                continue;
            }
            let leaked = if leading {
                match head {
                    "core" | "alloc" => false,
                    "std" => !self.cfg.allow_std,
                    krate if krate == self.cfg.krate => !self.cfg.allow_crate,
                    _ => true,
                }
            } else if segments.len() == 1 {
                match head {
                    "dyn" | "impl" => {
                        dyn_at = Some(depth);
                        false
                    }
                    "as" | "for" | "fn" | "unsafe" | "extern" | "mut" | "const" | "where"
                    | "true" | "false" => false,
                    "Self" => true,
                    _ if generics.contains(head) => false,
                    _ if PRIMITIVE_NAMES.contains(&head) => !self.cfg.primitive,
                    _ if is_prelude_trait(head) => dyn_at == Some(depth) && !self.cfg.prelude,
                    "String" | "Vec" | "Box" | "Option" | "Result" => !self.cfg.prelude,
                    _ => true,
                }
            } else {
                true
            };
            if leaked {
                return true;
            }
        }
        false
    }

    /// Tags the bounds that the item `text` writes on a generic or `Self`, in its generics and where
    /// predicates, that name a leaked type (`names_leaked_type()`): `bound_cycle=<bounded>|<bound>`
    /// for each bound of `T: A + B`, and `bound_cycle`. A predicate on another type (`Rel<T>:
    /// Default`) is proved by an impl and is none, unless it names the `Item` of an iterator.
    fn note_bounds(&mut self, text: &str) {
        let mut found = BTreeSet::new();
        let generics = declared_generics(text);
        for (bounded, bounds) in bound_clauses(text) {
            let generic = bounded_generic(&bounded, &generics).is_some();
            let item = bounded.contains("::Item") || bounds.contains("::Item");
            if !generic && !item {
                continue;
            }
            for bound in split_top(&bounds, '+') {
                if item || self.names_leaked_type(bound, &generics) {
                    found.insert(format!(
                        "{}{}|{}",
                        TAG_BOUND_AT,
                        norm(&bounded),
                        norm(bound)
                    ));
                }
            }
        }
        if !found.is_empty() {
            self.tags.insert(TAG_BOUND_CYCLE.to_string());
            self.tags.extend(found);
        }
    }

    /// The tags of the struct, enum, union or trait that an impl is for, but its constructs.
    fn extend_tags(&mut self, tags: &BTreeSet<String>) {
        // The shapes that only the declaration has (a default, a `Self` in `enter_generics`) are
        // not the impl's.
        const DECLARATION: &[&str] = &[
            TAG_SELF_IN_DEFAULT,
            TAG_SELF_NO_MEANING,
            TAG_NESTED_SELF_BOUNDS,
            TAG_NESTED_SELF_OPTION,
            TAG_DISCRIMINANT,
            TAG_UNREADABLE,
        ];
        self.tags.extend(
            tags.iter()
                .filter(|tag| !tag.starts_with(TAG_CONSTRUCT))
                .filter(|tag| !tag.starts_with(TAG_FORCED))
                .filter(|tag| !DECLARATION.contains(&tag.as_str()))
                .cloned(),
        );
    }

    /// The tags that the final text of an item, or of a part of a module, gives.
    pub fn note_text(&mut self, text: &str) {
        if !self.cfg.prelude {
            for name in bare_traits_in(text) {
                self.tags.insert(format!("bare:{}", name));
            }
        }
        self.note_bounds(text);
        self.tags.extend(
            hidden_bounds_in(text)
                .into_iter()
                .map(|(generic, bound)| format!("{}{}|{}", TAG_HIDDEN_AT, generic, bound)),
        );
        for name in constructs_of(text) {
            self.tags.insert(format!("{}{}", TAG_CONSTRUCT, name));
        }
        // What the text writes, whatever drew it: the generics or the where clause of an impl that
        // name `Self`, a const parameter of a type besides the usual ones, a macro by its bare name.
        if impl_header_names_self(text) {
            self.tags.insert(TAG_IMPL_HEADER_SELF.to_string());
        }
        if header_names_self_in_type(text) {
            self.tags.insert(TAG_SELF_IN_TRAIT_PRED.to_string());
        }
        if wide_const_in(text) {
            self.tags.insert(TAG_WIDE_CONST.to_string());
        }
        if text
            .match_indices("two!(")
            .any(|(at, _)| !text[..at].ends_with("::"))
        {
            self.tags.insert(TAG_MACRO_PATH.to_string());
        }
        if text.contains("two!(") || text.contains("::core::concat!(") {
            self.tags.insert(TAG_MACRO_CALL.to_string());
        }
        self.tags
            .extend(forced_in(text).into_iter().map(String::from));
        // A glob import or a `macro_rules!` in a body, which the Leaker reads with `values`.
        if self.flags.values
            && (text.contains("use ::core::mem::*;")
                || text.contains("use core::mem::*;")
                || text.contains("macro_rules! mm {"))
        {
            self.tags.insert(TAG_UNREADABLE.to_string());
        }
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

    /// Whether the item that is generated may be invoked with a Referrer that replaces optional
    /// entries.
    fn strict_ok(&self) -> bool {
        (self.features.strict_cycles || !self.risk.cycle)
            && (self.features.strict_anchors || !self.risk.anchor)
    }

    /// The mode the item that is generated is invoked with: the one drawn, or every crate if it
    /// has a shape that breaks when optional entries are replaced.
    fn item_trust(&self) -> Trust {
        if self.strict_ok() {
            self.want_trust
        } else {
            Trust::All
        }
    }

    /// Draws the mode of an item (`Trust`): `none` (the library's default) in two fifths, the others
    /// in a fifth each. A mode that cannot differ from `none` here is not drawn: when the Leaker's
    /// own rule allows `::std` no optional entry names only `std`, and when it allows its own crate
    /// none names only that.
    fn draw_trust(&mut self) -> Trust {
        let weights = [
            (Trust::None, 40),
            (Trust::All, 20),
            (Trust::Std, if self.cfg.allow_std { 0 } else { 20 }),
            (Trust::Leaker, if self.cfg.allow_crate { 0 } else { 20 }),
        ];
        let total: u32 = weights.iter().map(|(_, weight)| weight).sum();
        let mut roll = self.below(total as usize) as u32;
        for (trust, weight) in weights {
            if roll < weight {
                return trust;
            }
            roll -= weight;
        }
        unreachable!()
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

    fn attr(&self, id: usize, allow_self: bool, scope: &str, trust: Trust) -> String {
        format!(
            "#[{}(id = {}, allow_self = {}, prelude = {}, scope = \"{}\", allow = \"{}\", \
             {}{}{}{}trust = \"{}\", krate = \"{}\", primitive = {})]\n",
            ATTR,
            id,
            allow_self,
            self.cfg.prelude,
            scope,
            self.cfg.allow_arg(self.flags),
            if self.values_used {
                "values = true, bodies = true, "
            } else {
                ""
            },
            if self.redeclare {
                "redeclare = true, "
            } else {
                ""
            },
            if scope == "mod" && self.nested_entries {
                "nested_self_entries = true, "
            } else {
                ""
            },
            if self.flags.allow_stages {
                "stages = true, "
            } else {
                ""
            },
            trust.name(),
            self.cfg.krate,
            self.cfg.primitive
        )
    }

    /// The attribute of an item that is analyzed through `scope` (`""` for none). It says which
    /// optional entries the Referrer of the item trusts: the macro then writes the Repeater's impls
    /// of the others and of the required entries.
    fn attr_with(&mut self, ann: Option<(usize, bool)>, scope: &'static str) -> String {
        self.last_scope = scope;
        let trust = self.item_trust();
        self.last_trust = trust;
        ann.map_or(String::new(), |(id, a)| self.attr(id, a, scope, trust))
    }

    /// The analysis of an item through other calls of the same API (`via` in the attribute,
    /// `Via` in `tests/crates/macro/prop.rs`), for a share of the items of each kind and scope:
    ///
    /// - `type`: a struct, an enum or a union by `enter_type` and the walk of syn;
    /// - `parts`: the same, or through `enter_generics`, or a trait by `enter_trait`, but the
    ///   bounds of the generics, the where clause and the types are given to the `View` one by one
    ///   (`visit_type_param_bound`, `visit_where_clause`, `visit_type`);
    /// - `check`: `View::check()` on the types first, in a scope nested in the view for the
    ///   generics of a method; it must record nothing, and say what the copy does;
    /// - `nested`: a struct, an enum or a union nested in `enter_mod` by `View::enter_type`;
    /// - `some`: the types of a trait impl in `enter_impl` with the path of the trait;
    /// - `visit`: an impl by `enter_impl` and the walk of syn, as `intern_impl` does it.
    ///
    /// The copy and the checks are those of the scope. The draw is by the hash of the state and
    /// of the number of the item, and it draws nothing, so the items are the same as without it.
    /// The first line of `text` is the attribute.
    fn with_via(&mut self, kind: &'static str, ann_id: Option<usize>, text: String) -> String {
        let id = match ann_id {
            Some(id) => id,
            None => return text,
        };
        let mut state = self.rng.clone();
        state.write(b"via");
        state.write_u64(id as u64);
        let draw = (state.finish() % 100) as u32;
        let scope = self.last_scope;
        let adt = matches!(kind, "struct" | "enum" | "union");
        // `Self` in the header of an impl (a bound) is for `intern_impl`, which cuts the entries.
        let header_self = || {
            let at = text.find("\nimpl").map_or(0, |at| at + 1);
            let end = text[at..].find('{').map_or(text.len(), |end| at + end);
            has_word(&text[at..end], "Self")
        };
        let via = match (kind, scope) {
            _ if adt && scope.is_empty() => match draw {
                0..=9 => "type",
                10..=19 => "parts",
                20..=29 => "check",
                _ => "",
            },
            _ if adt && scope == "generics" => match draw {
                0..=24 => "parts",
                25..=49 => "check",
                _ => "",
            },
            // Not the shapes that name the own `Self` of a nested item in a bound (`self_bound_adts`).
            _ if adt
                && scope == "mod"
                && !self.tags.contains(TAG_SELF_BOUND_ADT)
                && !self.tags.contains(TAG_SELF_BOUND_TYPE) =>
            {
                match draw {
                    0..=29 => "nested",
                    _ => "",
                }
            }
            ("trait", "trait") => match draw {
                0..=34 => "parts",
                35..=69 => "check",
                _ => "",
            },
            ("impl-trait", "impl") => match draw {
                0..=39 => "some",
                _ => "",
            },
            ("impl-trait" | "impl-inherent", "") if !header_self() => match draw {
                0..=19 => "visit",
                _ => "",
            },
            _ => "",
        };
        if via.is_empty() {
            return text;
        }
        let from = format!("scope = \"{}\", ", scope);
        match text.find(&from) {
            Some(at) if text.starts_with("#[") && at < text.find('\n').unwrap_or(0) => {
                let end = at + from.len();
                format!("{}via = \"{}\", {}", &text[..end], via, &text[end..])
            }
            _ => text,
        }
    }

    /// The names of the types available in the module `path`, and the `use`s that import some.
    fn build_env(&mut self, path: &[String]) -> (ModEnv, String) {
        let up = "super::".repeat(path.len());
        let krate = self.cfg.krate.clone();
        let mut env = ModEnv {
            up: up.clone(),
            ..ModEnv::default()
        };
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

    /// An enum of unit variants with discriminants (`V0 = ..`), which the Leaker reads as values
    /// whether it reads values or not: the types in them are entries (`size_of::<Rel>()`), a
    /// relative constant or macro is an error, and the copy writes the rest as it is.
    fn gen_discriminant_enum(&mut self, ann: Option<(usize, bool)>) -> String {
        let name = self.name("E");
        let krate = self.cfg.krate.clone();
        let up = self.env.up.clone();
        let mut variants: Vec<String> = Vec::new();
        for i in 0..self.range(2, 4) {
            let mut forms: Vec<String> = vec![
                format!("{}", i + 3),
                format!("::{}::base::BLEN as isize", krate),
                format!("::{}::two!() as isize", krate),
            ];
            let ty = self.value_type(&Scope::default());
            forms.push(format!("::core::mem::size_of::<*const {}>() as isize", ty));
            // A relative constant or macro is an error, and it rejects a module analyzed as a
            // whole.
            if !self.module_mode
                && self.hash_percent(format!("relative_variant {}", i).as_bytes(), 30)
            {
                forms.push("RLEN as isize".to_string());
                forms.push(format!("{}base::BLEN as isize", up));
                forms.push("two!() as isize".to_string());
            }
            let form = self.pick(&forms).clone();
            if form.starts_with("RLEN") || form.contains("base::BLEN") && !form.starts_with("::") {
                self.tags.insert(TAG_VALUE_PATH.to_string());
            }
            if form.starts_with("two!") {
                self.tags.insert(TAG_MACRO_PATH.to_string());
                self.tags.insert(TAG_VALUE_PATH.to_string());
            }
            // The values differ: the constants are small, so are the others `+ 100 * i`.
            variants.push(format!("V{} = ({}) + {}", i, form, 100 * i));
        }
        self.tags.insert(TAG_DISCRIMINANT.to_string());
        let attr = self.attr_with(ann, "");
        format!("{}pub enum {} {{ {} }}\n", attr, name, variants.join(", "))
    }

    fn gen_adt(&mut self, kind: AdtKind, ann: Option<(usize, bool)>) -> String {
        if let Some(shape) = self.self_bound_adt_shape(ann) {
            return self.gen_self_bound_adt(kind, ann, shape);
        }
        if kind == AdtKind::Enum && ann.is_some() && self.hash_percent(b"discriminant_enum", 12) {
            return self.gen_discriminant_enum(ann);
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
        let names_self_anyway =
            scope == "generics" && self.hash_percent(b"self_without_meaning", 25);
        if names_self_anyway {
            // `enter_generics` gives `Self` no meaning, so a field that names it is an error
            // (`Reason::SelfWithoutMeaning`); the rest name none.
        } else if scope == "generics" || (scope == "mod" && !self.features.nested_self) {
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
        // A function pointer or trait object that names a type parameter in the input that gives
        // its elided lifetime to the output (`ELIDED_PARAM_PERCENT`), decided by the hash of the
        // state, which draws nothing. An enum's variant with no field is written without a draw
        // for its style, so it is left as it is: the other choices of the generator are the same
        // with or without the field.
        let elided_tag = format!("elided_param {}", name);
        if !sk.tys.is_empty() && self.hash_percent(elided_tag.as_bytes(), ELIDED_PARAM_PERCENT) {
            let draw = self.hash_draw(format!("{} shape", elided_tag).as_bytes());
            let param = &sk.tys[(draw / 100) as usize % sk.tys.len()];
            let text = elided_param_signature(&param.name, param.sized, draw / 10_000);
            if let Some(fields) = variants
                .iter_mut()
                .find(|fields| kind != AdtKind::Enum || !fields.is_empty())
            {
                used_tys.insert(param.name.clone());
                fields.push(if kind == AdtKind::Union {
                    format!("::core::mem::ManuallyDrop<{}>", text)
                } else {
                    text
                });
            }
        }
        // The same with the optional entry `::std::vec::Vec<T>` in the inputs, for a parameter that
        // a use site can choose the type of (`ELIDED_ENTRY_PERCENT`, F70). A strict Referrer
        // replaces the entry, so the path through the Repeater names the lifetimes of the site.
        let entry_tag = format!("elided_entry {}", name);
        let free: Vec<&TyParam> = sk
            .tys
            .iter()
            .filter(|t| t.sized && t.bound_texts.iter().all(|text| marker_bound(text)))
            .collect();
        if !free.is_empty() && self.hash_percent(entry_tag.as_bytes(), ELIDED_ENTRY_PERCENT) {
            let draw = self.hash_draw(format!("{} shape", entry_tag).as_bytes());
            let param = free[(draw / 100) as usize % free.len()];
            let text = elided_entry_signature(&param.name, draw / 10_000);
            if let Some(fields) = variants
                .iter_mut()
                .find(|fields| kind != AdtKind::Enum || !fields.is_empty())
            {
                used_tys.insert(param.name.clone());
                fields.push(if kind == AdtKind::Union {
                    format!("::core::mem::ManuallyDrop<{}>", text)
                } else {
                    text
                });
            }
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
        // The `Self` that `enter_generics` gives no meaning, named for certain.
        if names_self_anyway {
            phantoms.push("::core::marker::PhantomData<*const Self>".to_string());
        }
        self.preds(Level::Item, &sc, &mut sk, 2);
        self.self_prob = saved_prob;
        self.defaults(&mut sk, false);

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
        if scope == "generics" && has_word(&text, "Self") {
            self.tags.insert(TAG_SELF_NO_MEANING.to_string());
        }
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
            // Without the option the Leaker rejects it: the shape of the rejection is what the item
            // wrote, a type that names `Self` and a generic whose bound does.
            self.tags.insert(
                if self.nested_entries {
                    TAG_NESTED_SELF_OPTION
                } else {
                    TAG_NESTED_SELF_BOUNDS
                }
                .to_string(),
            );
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
        if !sk.defaults.is_empty() {
            self.tags.insert(format!("{}param_default", TAG_CONSTRUCT));
        }

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
        self.defaults(&mut sk, !in_mod);
        // A where predicate on a type of the module that names `Self` (`where Rel<Self>: Default`),
        // which the copy writes through the Repeater, whose impl is for the implementors of the
        // trait (`Reason::SelfInTraitPredicate`).
        if !self.no_self
            && (allow_self || in_mod)
            && self.hash_percent(b"self_in_trait_predicate", 8)
        {
            let heads: Vec<String> = self
                .env
                .heads
                .iter()
                .filter(|head| head.shape == Shape::T1 && !head.name.starts_with("::"))
                .filter(|head| !is_shadow_name(&head.name))
                .map(|head| head.name.clone())
                .collect();
            if !heads.is_empty() {
                let at = self.hash_draw(b"self_in_trait_predicate head") as usize % heads.len();
                let default = self.trait_path("Default");
                sk.preds.push(format!("{}<Self>: {}", heads[at], default));
                self.tags.insert(TAG_SELF_IN_TRAIT_PRED.to_string());
            }
        }
        let sc = sk.scope(&base);
        // Constants.
        let mut consts: Vec<(String, String)> = Vec::new();
        let mut const_texts: Vec<String> = Vec::new();
        for i in 0..self.range(0, 2) {
            if self.chance(70) {
                let t = self.ty(&sc, Con::unsized_ok().indirect(), 1);
                let decl = format!("const C{}: ::core::marker::PhantomData<{}>", i, t.text);
                consts.push((decl.clone(), "::core::marker::PhantomData".to_string()));
                // `values`: the default names the type again (a turbofish).
                if self.flags.values && self.chance(40) {
                    self.values_used = true;
                    const_texts.push(format!(
                        "{} = ::core::marker::PhantomData::<{}>;",
                        decl, t.text
                    ));
                } else {
                    const_texts.push(format!("{};", decl));
                }
            } else {
                let (ty, value) = *self.pick(&[("usize", "0"), ("bool", "false"), ("u8", "1")]);
                let decl = format!("const K{}: {}", i, ty);
                consts.push((decl.clone(), value.to_string()));
                // `values`: the default of a constant of type `usize` has types and paths.
                if self.flags.values && ty == "usize" && self.chance(60) {
                    let default = self.value_const(&sc);
                    const_texts.push(format!("{} = {};", decl, default));
                } else {
                    const_texts.push(format!("{};", decl));
                }
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
        // `values`: some methods have a default body.
        let mut method_texts: Vec<String> = Vec::new();
        for m in &methods {
            method_texts.push(match self.default_body(&method_scope) {
                Some(value) => format!("{} {}", m, value),
                None => format!("{};", m),
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
            .chain(method_texts)
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
        self.extend_tags(&tr.sk.tags);
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
        self.bound_by_self(&mut sk, &self_text);

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
            let value = self.method_body(&sc, 35);
            body.push(format!("{} {}", m, value));
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

    /// Bounds a generic of an impl by `Self` (`impl<T: AsRef<Self>> Rel<T>`), in its generics or its
    /// where clause, if the self type uses it: the self type is an entry whose context depends on
    /// `Self`, which is the self type itself (`Reason::SelfInImplSelfType`). Decided by the hash of
    /// the state, which draws nothing.
    fn bound_by_self(&mut self, sk: &mut Skeleton, self_text: &str) {
        if !self.features.impl_self_bounds
            || self.no_self
            || !self.hash_percent(b"impl_header_self", 40)
        {
            return;
        }
        let at = sk.tys.iter().position(|t| {
            t.sized
                && !t.iter
                && t.fixed_item == 0
                && !sk.bounded.iter().any(|(name, _)| name == &t.name)
                && !t.bound_texts.iter().any(|b| b.contains("AsRef"))
                && has_word(self_text, &t.name)
        });
        let Some(at) = at else { return };
        let as_ref = self.trait_path("AsRef");
        if self.hash_percent(b"impl_header_self where", 50) {
            let name = sk.tys[at].name.clone();
            sk.preds.push(format!("{}: {}<Self>", name, as_ref));
        } else {
            sk.tys[at].bound_texts.push(format!("{}<Self>", as_ref));
        }
        self.tags.insert(TAG_IMPL_HEADER_SELF.to_string());
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
            self.extend_tags(&sk.tags);
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
        let mut sk = sk;
        self.bound_by_self(&mut sk, &self_text);
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
            let value = self.method_body(&sc, 45);
            body.push(format!("pub {} {}", sig, value));
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
        let body = self.method_body(&Scope::default(), 70);
        format!("pub {} {}\n", sig, body)
    }

    // -----------------------------------------------------------------------------------------
    // Values (`Flags::values`).
    // -----------------------------------------------------------------------------------------

    /// A type for a type position of a value (a `let`, a turbofish): any type of the module that the
    /// generics of `sc` allow, with no `Self`.
    fn value_type(&mut self, sc: &Scope) -> String {
        let saved = (self.no_self, self.no_assoc_paths);
        self.no_self = true;
        self.no_assoc_paths = true;
        // The generics of an item whose predicates may name leaked types (`bound_cycles`) are not
        // used: the type would be in a cycle with them, as one of a signature is.
        let con = Con {
            no_params: self.bound_cycles(),
            ..Con::unsized_ok().indirect()
        };
        let ty = self.ty(sc, con, 1);
        self.no_self = saved.0;
        self.no_assoc_paths = saved.1;
        ty.text
    }

    /// An expression of type `usize` that a constant can have: a constant or a `const fn` of `base`,
    /// by an absolute path or, if `relative`, a path that the Leaker rejects in a value
    /// (`crate::base::BLEN`, `super::base::bf()`, `RLEN`). The path of a constant that the module
    /// of a module analyzed as a whole declares is no error there.
    fn value_usize(&mut self, sc: &Scope, relative: bool) -> String {
        let krate = self.cfg.krate.clone();
        let mut paths: Vec<String> = vec![
            format!("::{}::base::BLEN", krate),
            format!("::{}::base::bf()", krate),
        ];
        if relative {
            let up = self.env.up.clone();
            let mut relatives = vec![
                "crate::base::BLEN".to_string(),
                "crate::base::bf()".to_string(),
            ];
            relatives.push(format!("{}base::BLEN", up));
            relatives.push(format!("{}base::bf()", up));
            // The constants of the module and its import are not in a copy of an item. The generic
            // `const fn` of `base` has a type in its turbofish, which the path of the value holds.
            if !self.module_mode {
                relatives.push("RLEN".to_string());
                relatives.push("self::RLEN".to_string());
                if self.env.consts.iter().any(|c| c == "ILEN") {
                    relatives.push("ILEN".to_string());
                }
                let ty = self.value_type(sc);
                relatives.push(format!("crate::base::bg::<*const {}>()", ty));
                relatives.push(format!("{}base::bg::<*const {}>()", up, ty));
            }
            paths = relatives;
        }
        let path = self.pick(&paths).clone();
        if relative {
            self.tags.insert(TAG_VALUE_PATH.to_string());
        }
        path
    }

    /// The type of a cast in a value: an alias of `u8`, a relative one (`RA`, `crate::base::BA`) or an
    /// absolute one.
    fn value_cast_target(&mut self) -> String {
        let krate = self.cfg.krate.clone();
        let up = self.env.up.clone();
        let targets = [
            "RA".to_string(),
            "self::RA".to_string(),
            "crate::base::BA".to_string(),
            format!("{}base::BA", up),
            format!("::{}::base::BA", krate),
            "u8".to_string(),
        ];
        self.pick(&targets).clone()
    }

    /// The statements of a value, which end in `loop {}` when it is a body: a `PhantomData` with a
    /// type of the module in its annotation (the type of a `let`), `size_of` of a pointer to one (a
    /// turbofish), a cast to an alias of `u8`, a `usize` that may be a relative path, and a `Vec`.
    fn value_statements(&mut self, sc: &Scope) -> String {
        self.values_used = true;
        let mut stmts: Vec<String> = Vec::new();
        let ty = self.value_type(sc);
        stmts.push(format!(
            "let _a: ::core::marker::PhantomData<*const {}> = ::core::marker::PhantomData;",
            ty
        ));
        if self.chance(55) {
            let ty = self.value_type(sc);
            stmts.push(format!("let _b = ::core::mem::size_of::<*const {}>();", ty));
        }
        if self.chance(40) {
            let target = self.value_cast_target();
            stmts.push(format!("let _c = 1u8 as {};", target));
        }
        if self.chance(40) {
            let relative = self.chance(if self.module_mode { 4 } else { 20 });
            let path = self.value_usize(sc, relative);
            stmts.push(format!("let _d: usize = {} + 1;", path));
        }
        // A path of the prelude with a type in its turbofish: relative if the Leaker does not allow
        // the prelude (not in a module analyzed as a whole, which one such error rejects).
        if !self.module_mode && self.chance(15) {
            let ty = self.value_type(sc);
            stmts.push(format!("let _e = Vec::<*const {}>::new();", ty));
            if !self.cfg.prelude {
                self.tags.insert(TAG_VALUE_PATH.to_string());
            }
        }
        // Syntax the Leaker cannot read in a value: a glob import and a `macro_rules!`
        // (`Reason::Unreadable`). One of them rejects a module analyzed as a whole, so not there.
        if !self.module_mode && self.hash_percent(b"unreadable", 3) {
            stmts.push(if self.hash_percent(b"unreadable which", 50) {
                "use ::core::mem::*;".to_string()
            } else {
                "macro_rules! mm { () => {}; }".to_string()
            });
        }
        // The root of a `use` is a relative path (`Cause::Unknown` as a path of the error).
        if !self.module_mode && self.hash_percent(b"relative_use", 3) {
            let up = self.env.up.clone();
            stmts.push(format!("use {}base::bf as local_bf;", up));
            self.tags.insert(TAG_VALUE_PATH.to_string());
        }
        // A macro call: of the standard library, or the Leaker's own by an absolute path.
        if self.hash_percent(b"value_macro", 25) {
            let krate = self.cfg.krate.clone();
            stmts.push(match self.hash_draw(b"value_macro which") % 3 {
                0 => "let _m = ::core::concat!(\"a\", \"b\");".to_string(),
                1 => format!("let _m = ::{}::two!() + 1;", krate),
                _ => "let _m = ::core::matches!(1u8, 1 | 2);".to_string(),
            });
            self.tags.insert(TAG_MACRO_CALL.to_string());
        }
        stmts.join(" ")
    }

    /// The body of a function or a method with values, in `percent` of the cases when `Flags::values`
    /// asks for them: `{ .. loop {} }`. Else `{ loop {} }`, and no random number is drawn.
    fn method_body(&mut self, sc: &Scope, percent: u32) -> String {
        if !self.flags.values || !self.chance(percent) {
            return "{ loop {} }".to_string();
        }
        format!("{{ {} loop {{}} }}", self.value_statements(sc))
    }

    /// The default body of a method of a trait with values (`{ .. loop {} }`), or `None` (the method
    /// is only declared).
    fn default_body(&mut self, sc: &Scope) -> Option<String> {
        if !self.flags.values || !self.chance(35) {
            return None;
        }
        Some(format!("{{ {} loop {{}} }}", self.value_statements(sc)))
    }

    /// The value of a constant of type `usize`: `size_of` of a pointer to a type of the module,
    /// plus a `usize` that may be a relative path.
    fn value_const(&mut self, sc: &Scope) -> String {
        self.values_used = true;
        let ty = self.value_type(sc);
        let relative = self.chance(if self.module_mode { 4 } else { 20 });
        let path = self.value_usize(sc, relative);
        format!("::core::mem::size_of::<*const {}>() + {}", ty, path)
    }

    /// A constant or a static of a module analyzed as a whole, with a value (`Flags::values`).
    fn gen_value_item(&mut self) -> String {
        let name = self.name("KV");
        let value = self.value_const(&Scope::default());
        if self.chance(60) {
            self.contents.push("const");
            format!("pub const {}: usize = {};\n", name, value)
        } else {
            self.contents.push("static");
            format!("pub static {}: usize = {};\n", name, value)
        }
    }

    /// `relative_std`: writes the paths that start with `::std::`, `::core::` in a third of their
    /// places without the leading `::`, and some of `::std::` as `alloc::` where `alloc` has the
    /// item, in the items analyzed one by one. The first line of `text` is its attribute, left as it
    /// is.
    pub fn relativize_std(&mut self, text: &str) -> String {
        // Not in a module analyzed as a whole: a relative path with an argument that the module
        // declares (`std::boxed::Box<R0>`) is an entry that names the Leaker's `R0`, where the copy
        // declares its own. The generator avoids that for every relative path (`is_declared()`).
        if !self.flags.relative_std || self.module_mode {
            return text.to_string();
        }
        let (head, tail) = match text.find('\n') {
            Some(at) if text.starts_with("#[") => text.split_at(at),
            _ => ("", text),
        };
        let mut out = String::from(head);
        let mut rest = tail;
        while let Some(at) = rest.find("::") {
            let (before, here) = rest.split_at(at);
            out += before;
            let start_ok = before
                .chars()
                .next_back()
                .map_or(true, |c| c.is_whitespace() || "(<,&*?+[{=;!".contains(c));
            let crate_name = ["std", "core"]
                .iter()
                .find(|name| here[2..].starts_with(&format!("{}::", name)));
            // `Chars` and `Iter<u8>` hide a lifetime (`hidden_path()`), which the Leaker knows only of
            // an absolute path: as a relative one it is a type like any other, leaked alone, and the
            // docs of the Leaker say to write `Iter<'_, u8>` (F28).
            let hides_lifetime = crate_name.map_or(false, |name| {
                let after = &here[2 + name.len()..];
                after.starts_with("::str::Chars") || after.starts_with("::slice::Iter")
            });
            match crate_name {
                Some(name) if start_ok && !hides_lifetime && self.chance(35) => {
                    let after = &here[2 + name.len()..];
                    let in_alloc = [
                        "::vec::",
                        "::boxed::",
                        "::string::",
                        "::rc::",
                        "::sync::Arc",
                    ]
                    .iter()
                    .any(|module| after.starts_with(module))
                        || after.starts_with("::collections::BTreeMap");
                    let name = if *name == "std" && in_alloc && self.chance(40) {
                        "alloc"
                    } else {
                        name
                    };
                    out += name;
                    rest = after;
                }
                _ => {
                    out += "::";
                    rest = &here[2..];
                }
            }
        }
        out += rest;
        if out != text {
            self.tags.insert(TAG_RELATIVE_STD.to_string());
        }
        out
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
            self.values_used = false;
            self.plain_ai = false;
            self.want_trust = self.draw_trust();
            self.bare_percent = self.item_bare_percent();
            // Half of the items nested in `enter_mod` are analyzed with `nested_self_entries`, the
            // other half without, where a type that names `Self` next to a generic whose bound
            // does is an error (`Reason::NestedSelfInBounds`).
            self.nested_entries =
                self.features.nested_self_entries && self.hash_percent(b"nested_entries", 50);
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
                let text = self.relativize_std(&text);
                self.note_text(&text);
                return text;
            }
            97..=99 if whole => {
                let text = self.gen_alias();
                self.contents.push("alias");
                let text = self.relativize_std(&text);
                self.note_text(&text);
                return text;
            }
            _ => ("struct", self.gen_adt(AdtKind::Struct, ann)),
        };
        let text = self.relativize_std(&text);
        let text = self.with_via(kind, ann_id, text);
        self.note_text(&text);
        if text.contains("Self::") || text.contains("<Self as ") {
            self.tags.insert(TAG_SELF_ASSOC.to_string());
        }
        if !allow_self && has_word(&text, "Self") {
            self.tags.insert(TAG_SELF_DENIED.to_string());
        }
        if names_self_in_where(&text) {
            self.tags.insert(TAG_SELF_IN_PREDICATE.to_string());
        }
        if self.last_scope == "mod"
            && ann.is_some()
            && !self.nested_entries
            && bound_clauses(&text).iter().any(|(bounded, bounds)| {
                has_word(bounds, "Self") || (has_word(bounded, "Self") && bounded.trim() != "Self")
            })
        {
            self.tags.insert(TAG_NESTED_SELF_TEXT.to_string());
        }
        if let Some(id) = ann_id {
            self.items.push(Item {
                id,
                kind,
                module: path.to_vec(),
                trust: self.last_trust,
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
            "?Sized"
        } else {
            "?::core::marker::Sized"
        };
        text += &types_template("R", sized);
        if self.flags.values {
            // The target of a cast in a value (`value_cast_target()`).
            text += "pub type RA = u8;\n";
        }
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
        // `values`: constants and statics with values, in a module analyzed as a whole.
        if whole && self.flags.values {
            for _ in 0..2 {
                if self.chance(60) {
                    let item = self.gen_value_item();
                    // The constant is no item of `gen_one()`, whose text `note_text()` reads.
                    self.tags
                        .extend(forced_in(&item).into_iter().map(String::from));
                    text += &self.relativize_std(&item);
                }
            }
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
    generate_with(seed, index, items_per_module, features, Flags::from_env())
}

/// [`generate()`] with the flags given.
pub fn generate_with(
    seed: u64,
    index: usize,
    items_per_module: usize,
    features: Features,
    flags: Flags,
) -> Case {
    assert!(
        items_per_module >= MIN_ITEMS,
        "TYPE_LEAK_PROP_ITEMS is {}: it is the most items in a module, and a module has at least {}",
        items_per_module,
        MIN_ITEMS
    );
    let krate = format!("tlp_l_{:03}", index);
    let mut g = Gen::new(seed, krate.clone(), features, flags);
    let mut lib = String::new();
    lib += &crate_header(flags);
    lib += &format!("extern crate self as {};\n", krate);
    lib += "#[doc(hidden)]\npub use type_leak_test_macro as __macro;\n";
    lib += "type_leak_test_macro::define_repeater!(pub Repeater);\n";
    // A macro that expands to a length, which an array may write by an absolute path (`::krate::two!()`:
    // a macro of another crate, written as it is) or bare (`two!()`: a macro that the Referrer's
    // crate cannot name, an error of the cause `Cause::Macro`).
    lib += TWO_MACRO;
    let mut base = types_template("B", "?::core::marker::Sized");
    // The cast target and the function of a value, which the paths of `values` name.
    if flags.values {
        base += "pub type BA = u8;\npub const fn bf() -> usize { 1 }\n";
        base += "pub const fn bg<T: ?Sized>() -> usize { ::core::mem::size_of::<*const T>() }\n";
    }
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
        g.values_used = false;
        g.plain_ai = false;
        g.want_trust = g.draw_trust();
        g.bare_percent = g.module_bare_percent();
        let n = g.range(4, items_per_module);
        let nested = usize::from(g.chance(60));
        let body = g.gen_module_body(std::slice::from_ref(&name), true, n, nested);
        g.no_self = false;
        g.module_mode = false;
        g.cyclic_module = false;
        let trust = g.item_trust();
        let attr = g.attr(id, true, "", trust);
        let module = format!("{}pub mod {} {{\n{}\n}}\n", attr, name, body);
        lib += &module;
        let contents = std::mem::take(&mut g.contents);
        let tags = std::mem::take(&mut g.tags);
        g.items.push(Item {
            id,
            kind: "mod",
            module: Vec::new(),
            trust,
            allow_self: true,
            scope: "",
            contents,
            tags,
            source: module,
        });
    }

    // The Referrer's crate invokes the macro of each item.
    let mut referrer = crate_header(flags);
    let mut referrer_lines = Vec::new();
    let mut items = g.items.clone();
    items.sort_by_key(|item| item.id);
    for item in &items {
        let mut path = format!("::{}", krate);
        for m in &item.module {
            path += &format!("::{}", m);
        }
        let line = referrer.lines().count() + 1;
        // Each copy is in a module of its own, so that a relative path that it keeps by mistake
        // cannot resolve to the copy of another item (the copies of the structs, the enums and the
        // unions have the names they have in the Leaker's crate), and the names of one item do not
        // clash with those of another. The Referrer's crate has no other name of the Leaker's.
        referrer += &format!(
            "pub mod c{} {{ {}::__tl_prop_{}!(path = {}{}{}); }}\n",
            item.id,
            krate,
            item.id,
            path,
            item.trust.flag(),
            use_flags(&g.rng, item)
        );
        referrer_lines.push((line, item.id));
    }
    let (accept_referrer, accept_lines) =
        accept_referrer(&krate, &g.mod_uses, &items, "__tl_prop_", flags);
    let stages = stage_crates(&krate, &g.mod_uses, &items, flags);
    let decoy = decoy_crates(&g.cfg, &items, &base, flags);
    let errors_referrer = errors_crate(&krate, &g.rng, &items, flags);
    Case {
        config: g.cfg.clone(),
        leaker: lib,
        referrer,
        items,
        referrer_lines,
        accept_referrer,
        errors_referrer,
        accept_lines,
        stages,
        decoy,
    }
}

/// The items that `TYPE_LEAK_PROP_ACCEPT` leaks with `prop_leak_result` instead of `prop_leak_accept`
/// (a fifth of them): the Leaker does not decide, the macro of the Referrer carries the result with
/// its errors, and its invocation in the crate of the accepted items says `allow(all)`.
pub fn carries_result(id: usize) -> bool {
    id % 5 == 0
}

/// The share of the items (percent) whose copy is also used at an instantiation of its generics
/// (`inst`, `prop.rs: Instance`), and that are copied again inside a module and a block that declare
/// the names the item leaks (`shadow`, `prop.rs: shadow_check()`).
const INST_PERCENT: u64 = 40;
const SHADOW_PERCENT: u64 = 30;

/// The flags of the invocation of `item` in the ordinary Referrer's crate that ask the macro to do
/// more with the copy: `, inst(N)` and `, shadow(N)`, where `N` seeds the macro's choices. They
/// come from the hash of the final state of the generator and the number of the item, which draws
/// nothing, so the items of a seed do not change. A module has none.
fn use_flags(rng: &StableHasher, item: &Item) -> String {
    if item.kind == "mod" {
        return String::new();
    }
    let mut flags = String::new();
    for (word, percent) in [("inst", INST_PERCENT), ("shadow", SHADOW_PERCENT)] {
        let mut state = rng.clone();
        state.write(word.as_bytes());
        state.write_u64(item.id as u64);
        let draw = state.finish();
        if draw % 100 < percent {
            flags += &format!(", {}({})", word, draw >> 8);
        }
    }
    flags
}

/// The `lib.rs` of the crate of the errors that the Leaker carries (`Case::errors_referrer`): each
/// item has the macro `__tl_carry_<id>!` (which has no error to report if the Leaker accepted the
/// item), invoked with a number that chooses what `LeakerResult::allow()` drops of them (`carry(N)`,
/// `prop.rs: carry()`: nothing, the first error, every error).
fn errors_crate(krate: &str, rng: &StableHasher, items: &[Item], flags: Flags) -> String {
    let mut lib = crate_header(flags);
    for item in items {
        let mut state = rng.clone();
        state.write(b"carry");
        state.write_u64(item.id as u64);
        lib += &format!(
            "pub mod c{} {{ {}::__tl_carry_{}!(carry({})); }}\n",
            item.id,
            krate,
            item.id,
            state.finish() >> 8
        );
    }
    lib
}

/// The name of the Leaker's crate in the Referrer's crate of the decoy (`Decoy`).
pub fn real_name(krate: &str) -> String {
    format!("real_{}", krate.trim_start_matches("tlp_l_"))
}

/// The decoy crates of a case (`Decoy`), if it has any item for them: the Leaker's own rule must not
/// allow its crate (it is then promised to be the same crate), and the item's Referrer must not
/// trust it (`Trust::None` and `Trust::Std`). A module is not copied there (`prop::apply()`), nor an
/// item whose copy names the Leaker's crate whatever the Referrer trusts, as the header of an impl of
/// a trait of the module does: the macro expands to nothing for it.
///
/// What it tells that the ordinary Referrer's crate cannot: an absolute path always names the same
/// crate there, so a Referrer that trusts an entry although the crate it names is not trusted still
/// compiles. Here `::tlp_l_NNN::base::B1` is the decoy's type, and the copy that writes it as it is
/// does not equal the original.
fn decoy_crates(cfg: &Config, items: &[Item], base: &str, flags: Flags) -> Option<Decoy> {
    if cfg.allow_crate {
        return None;
    }
    let krate = &cfg.krate;
    let real = real_name(krate);
    let mut referrer = crate_header(flags);
    let mut lines = Vec::new();
    for item in items
        .iter()
        .filter(|item| item.kind != "mod" && matches!(item.trust, Trust::None | Trust::Std))
    {
        let mut path = format!("::{}", real);
        for m in &item.module {
            path += &format!("::{}", m);
        }
        lines.push((referrer.lines().count() + 1, item.id));
        referrer += &format!(
            "{}::__tl_prop_{}!(path = {}{}, decoy({}));\n",
            real,
            item.id,
            path,
            item.trust.flag(),
            krate
        );
    }
    if lines.is_empty() {
        return None;
    }
    // The definitions differ: another length, and every type is another type anyway.
    let base = base.replace("pub const BLEN: usize = 2;", "pub const BLEN: usize = 3;");
    let lib = format!(
        "{}{}pub mod base {{\n{}\n}}\n",
        crate_header(flags),
        TWO_MACRO,
        base
    );
    Some(Decoy {
        lib,
        referrer,
        lines,
    })
}

/// The start of every generated crate: no warnings, and `alloc` (`relative_std`), which the items
/// that the Leaker allows (`std`, `core`, `alloc`) name.
fn crate_header(flags: Flags) -> String {
    let mut header = String::from("#![allow(warnings)]\n");
    if flags.relative_std {
        header += "extern crate alloc;\n";
    }
    header
}

/// The Referrer's crates of `TYPE_LEAK_PROP_FEATURES=allow_stages`: for each rule (`rules()`) and
/// each side the crate built like the crate of the accepted items (`accept_referrer()`), and the
/// clean one (`clean_referrer()`).
fn stage_crates(
    krate: &str,
    mod_uses: &[(Vec<String>, String)],
    items: &[Item],
    flags: Flags,
) -> Vec<StageCrate> {
    let mut crates = Vec::new();
    for rule in flags.rules() {
        for side in ["res", "cfg"] {
            let prefix = format!("__tl_st_{}_{}_", rule, side);
            let (lib, lines) = accept_referrer(krate, mod_uses, items, &prefix, flags);
            crates.push(StageCrate {
                rule,
                side,
                clean: false,
                lib,
                lines,
            });
            let (lib, lines) = clean_referrer(krate, mod_uses, items, &prefix, rule, side, flags);
            crates.push(StageCrate {
                rule,
                side,
                clean: true,
                lib,
                lines,
            });
        }
    }
    crates
}

/// The `lib.rs` of a clean Referrer's crate of `allow_stages` (the crate `tlp_<rule>_<side>_clean_NNN`),
/// and the line where each item is invoked: it copies the items that the rule left no error in, and
/// has none of the Leaker's names but the ones that the rule allows, so a type that a copy keeps as
/// written by mistake does not resolve. It is laid out like the ordinary Referrer's crate: the
/// copies are at the root, and an item of a module is invoked with the path of the module only for
/// its witnesses, unless the paths that the rule allows are relative to the module (below).
///
/// What the rule allows is a promise that the names resolve, and the crate keeps it:
///
/// - `prelude`: nothing, the prelude is the prelude (`shadowed_prelude_names()` of the harness);
/// - `std`: `extern crate alloc;` (`crate_header()`), the other two are in every crate;
/// - `crate`: the module `base` at the root, which re-exports the Leaker's;
/// - `path` (`super::base`): the same, and each item invoked in a module of its own, empty, in the
///   same place of the module tree as in the Leaker's crate, so that `super::base` is the root's
///   (the copy of a module analyzed as a whole declares itself, at the root);
/// - `values`, `default`: every relative path of a value or of a default is written as it is, so
///   every name the module of the item has must resolve. The crate has the modules of `path`, and
///   after the config (`side` `cfg`), which writes them all as they are, each module has its `use`s
///   and a glob import of the Leaker's module, as the crate of the accepted items has. The names of
///   a default or a value are those of the types too, so this crate cannot tell a type kept by
///   mistake in another place; after the result (`res`), which leaves a name as it is only where
///   it kept an error as written, each module has the constants `RLEN` and `ILEN` only. (With only
///   those in both, half of the copies after the config did not compile, and a copy after the
///   result that did not was excused as shared with it.)
fn clean_referrer(
    krate: &str,
    mod_uses: &[(Vec<String>, String)],
    items: &[Item],
    macro_prefix: &str,
    rule: &str,
    side: &str,
    flags: Flags,
) -> (String, Vec<(usize, usize)>) {
    fn invocation(krate: &str, item: &Item, macro_prefix: &str) -> String {
        let mut module = format!("::{}", krate);
        for m in &item.module {
            module += &format!("::{}", m);
        }
        format!(
            "::{}::{}{}!(clean path = {}, accept);\n",
            krate, macro_prefix, item.id, module
        )
    }
    fn module(
        krate: &str,
        path: &[String],
        mod_uses: &[(Vec<String>, String)],
        items: &[Item],
        macro_prefix: &str,
        names: Names,
        out: &mut String,
    ) {
        let uses = mod_uses
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, uses)| uses);
        match names {
            Names::Constants => {
                *out += "pub const RLEN: usize = 2;\n";
                for line in uses.iter().flat_map(|uses| uses.lines()) {
                    if line.ends_with(" as ILEN;") {
                        *out += line;
                        *out += "\n";
                    }
                }
            }
            Names::All => {
                *out += uses.map_or("", |uses| uses.as_str());
                *out += &format!("pub use ::{}::{}::*;\n", krate, path.join("::"));
            }
            Names::None => (),
        }
        for item in items.iter().filter(|i| i.kind != "mod" && i.module == path) {
            *out += &invocation(krate, item, macro_prefix);
        }
        for (sub, _) in mod_uses
            .iter()
            .filter(|(p, _)| p.len() == path.len() + 1 && p.starts_with(path))
        {
            *out += &format!("pub mod {} {{\n", sub.last().unwrap());
            module(krate, sub, mod_uses, items, macro_prefix, names, out);
            *out += "}\n";
        }
    }
    /// The names of the Leaker's modules that a module of the crate has.
    #[derive(Clone, Copy)]
    enum Names {
        None,
        /// `RLEN` and the import of `ILEN`.
        Constants,
        /// The `use`s of the module and a glob import of it.
        All,
    }
    let mut text = crate_header(flags);
    let skeleton = matches!(rule, "path" | "values" | "default");
    let constants = matches!(rule, "values" | "default");
    let names = match (constants, side) {
        (false, _) => Names::None,
        (true, "cfg") => Names::All,
        (true, _) => Names::Constants,
    };
    match rule {
        "crate" | "path" | "values" | "default" => {
            text += &format!("pub mod base {{ pub use ::{}::base::*; }}\n", krate);
        }
        _ => (),
    }
    if skeleton {
        for (path, _) in mod_uses.iter().filter(|(p, _)| p.len() == 1) {
            text += &format!("pub mod {} {{\n", path[0]);
            module(krate, path, mod_uses, items, macro_prefix, names, &mut text);
            text += "}\n";
        }
        for item in items.iter().filter(|i| i.kind == "mod") {
            text += &invocation(krate, item, macro_prefix);
        }
    } else {
        for item in items {
            text += &invocation(krate, item, macro_prefix);
        }
    }
    let lines = items
        .iter()
        .filter_map(|item| {
            let call = format!("::{}{}!(", macro_prefix, item.id);
            let at = text.lines().position(|line| line.contains(&call))?;
            Some((at + 1, item.id))
        })
        .collect();
    (text, lines)
}

/// The `lib.rs` of the second Referrer's crate of `TYPE_LEAK_PROP_ACCEPT` (the crate `tlp_a_NNN`),
/// which copies the items that the Leaker accepts only by `LeakerResult::allow(cause)` of the
/// causes of its errors, and the line where each item is invoked.
///
/// The copy keeps what conflicts as written: of a type that cannot be leaked, the path of its head
/// (its arguments are analyzed on their own, see `tests/flat_errors.rs`), and whole the trait
/// bounds, the types in predicates, the references and the holders of a bare name that may be a
/// const. The paths in it are those the item has in the Leaker's crate (`R0`, `self::R1`,
/// `super::R1`, `super::base::B1`, `crate::base::B1`, an import `I1` or `J1`), and it is up to the
/// user of `allow()` that they resolve to the same items in the crate of the copy. This crate makes them: the module of the Leaker's crate that the item
/// is in is declared again with the same `use`s and a glob import of the Leaker's module, in the same
/// place of the module tree (so that `super::` and `self::` mean the same), and the root has `base`,
/// the glob import of the Leaker's `base` (so that `crate::base::B1` means the same). The items are
/// copied in the module they are invoked in, as they are in the Leaker's crate (a module analyzed
/// as a whole declares itself at the root).
///
/// It is a crate of its own, not a module of the other Referrer's crate, because `crate::base` is
/// the root of the crate that has the copy: `base` at the root of the other crate would let an
/// ordinary item that keeps a relative path by mistake resolve it, and that crate must not have one.
///
/// `macro_prefix` is the start of the name of the macro of each item (`__tl_prop_`), and the crates of
/// `TYPE_LEAK_PROP_FEATURES=allow_stages` invoke `__tl_st_<rule>_<side>_` instead. The crate declares
/// no name of the standard prelude, so that they mean what they mean in the prelude: the globs import
/// what the Leaker's modules declare, and `shadowed_prelude_names()` of the harness checks that none of
/// it has such a name.
fn accept_referrer(
    krate: &str,
    mod_uses: &[(Vec<String>, String)],
    items: &[Item],
    macro_prefix: &str,
    flags: Flags,
) -> (String, Vec<(usize, usize)>) {
    fn invocation(krate: &str, item: &Item, path: &[String], macro_prefix: &str) -> String {
        let mut module = format!("::{}", krate);
        for m in path {
            module += &format!("::{}", m);
        }
        format!(
            "::{}::{}{}!(path = {}{}{}, accept);\n",
            krate,
            macro_prefix,
            item.id,
            module,
            // The copies of `allow_stages` have the trust of their rule in their attribute.
            if macro_prefix == "__tl_prop_" {
                item.trust.flag()
            } else {
                String::new()
            },
            // The Leaker carries the errors of these items, and the Referrer drops them.
            if macro_prefix == "__tl_prop_" && carries_result(item.id) {
                ", allow(all)"
            } else {
                ""
            }
        )
    }
    fn module(
        krate: &str,
        path: &[String],
        mod_uses: &[(Vec<String>, String)],
        items: &[Item],
        macro_prefix: &str,
        out: &mut String,
    ) {
        if let Some((_, uses)) = mod_uses.iter().find(|(p, _)| p == path) {
            *out += uses;
        }
        *out += &format!("pub use ::{}::{}::*;\n", krate, path.join("::"));
        for item in items.iter().filter(|i| i.kind != "mod" && i.module == path) {
            *out += &invocation(krate, item, path, macro_prefix);
        }
        for (sub, _) in mod_uses
            .iter()
            .filter(|(p, _)| p.len() == path.len() + 1 && p.starts_with(path))
        {
            *out += &format!("pub mod {} {{\n", sub.last().unwrap());
            module(krate, sub, mod_uses, items, macro_prefix, out);
            *out += "}\n";
        }
    }
    let mut text = crate_header(flags);
    // The macros of the Leaker's crate by their bare names (`two!()`): the copy of what an error
    // allowed names the macro as the Leaker's crate does.
    text += &format!("#[macro_use]\nextern crate {};\n", krate);
    text += &format!("pub mod base {{ pub use ::{}::base::*; }}\n", krate);
    for (path, _) in mod_uses.iter().filter(|(p, _)| p.len() == 1) {
        text += &format!("pub mod {} {{\n", path[0]);
        module(krate, path, mod_uses, items, macro_prefix, &mut text);
        text += "}\n";
    }
    // A module analyzed as a whole declares itself, at the root.
    for item in items.iter().filter(|i| i.kind == "mod") {
        text += &invocation(krate, item, &[], macro_prefix);
    }
    let lines = items
        .iter()
        .filter_map(|item| {
            let call = format!("::{}{}!(", macro_prefix, item.id);
            let at = text.lines().position(|line| line.contains(&call))?;
            Some((at + 1, item.id))
        })
        .collect();
    (text, lines)
}

/// The primitive types, by name.
pub const PRIMITIVE_NAMES: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32",
    "f64", "bool", "char", "str",
];

/// The traits of the prelude that the text names bare (`Fn(..)`, `T: Default`, `dyn Iterator`): a
/// path of its own, not the end of one (`::core::default::Default`), and not `?Sized`, which the
/// Leaker accepts bare whatever the prelude.
pub fn bare_traits_in(text: &str) -> Vec<&'static str> {
    let is_id = |c: char| c.is_alphanumeric() || c == '_';
    PRELUDE_TRAITS
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| {
            text.match_indices(name).any(|(at, _)| {
                let before = text[..at].chars().next_back();
                let before_that = text[..at].chars().rev().nth(1);
                let after = text[at + name.len()..].chars().next();
                !before.map_or(false, |c| {
                    is_id(c) || c == '?' || c == ':' && before_that == Some(':')
                }) && !after.map_or(false, is_id)
            })
        })
        .collect()
}

/// The names of the generics that the item `text` declares (`struct S<'a, T: A, const N: usize>`,
/// `impl<T> Tr for S<T>`, `fn m<X>(..)`). The modules declare types with some of these names
/// (`SHADOW_NAMES`), which mean the generic where one is declared.
pub fn declared_generics(text: &str) -> BTreeSet<String> {
    let is_id = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = BTreeSet::new();
    for keyword in ["struct ", "enum ", "union ", "trait ", "fn ", "impl"] {
        for (at, _) in text.match_indices(keyword) {
            if text[..at].chars().next_back().map_or(false, is_id) {
                continue;
            }
            let mut rest = &text[at + keyword.len()..];
            if keyword != "impl" {
                rest = rest.trim_start_matches(is_id);
            }
            let Some(rest) = rest.strip_prefix('<') else {
                continue;
            };
            // Up to the `>` that closes the list.
            let mut depth = 0i32;
            let mut previous = ' ';
            let mut end = rest.len();
            for (k, c) in rest.char_indices() {
                match c {
                    '(' | '[' | '<' => depth += 1,
                    ')' | ']' => depth -= 1,
                    '>' if previous != '-' => depth -= 1,
                    _ => (),
                }
                if depth < 0 {
                    end = k;
                    break;
                }
                previous = c;
            }
            for param in split_top(&rest[..end], ',') {
                let param = param.strip_prefix("const ").unwrap_or(param);
                if param.starts_with('\'') {
                    continue;
                }
                let name: String = param.chars().take_while(|c| is_id(*c)).collect();
                if !name.is_empty() {
                    out.insert(name);
                }
            }
        }
    }
    out
}

/// A text as the Leaker prints its tokens, so that the two can be compared: no whitespace, and the
/// paths of `std`, `core` and `alloc` without the `::` that `Gen::relativize_std()` may take off
/// (`alloc` is `std`).
pub fn norm(text: &str) -> String {
    let mut text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    for name in ["std", "core", "alloc"] {
        text = text.replace(&format!("::{}::", name), &format!("{}::", name));
    }
    text.replace("alloc::", "std::")
}

/// The text split at the `sep` that is not in brackets (`T: A + Fn(B + C)` at `+`).
pub fn split_top(text: &str, sep: char) -> Vec<&str> {
    let (mut depth, mut start) = (0i32, 0);
    let mut out = Vec::new();
    let mut previous = ' ';
    for (at, c) in text.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' => depth -= 1,
            '>' if previous != '-' => depth -= 1,
            c if c == sep && depth == 0 => {
                out.push(text[start..at].trim());
                start = at + c.len_utf8();
            }
            _ => (),
        }
        previous = c;
    }
    out.push(text[start..].trim());
    out
}

/// The generic that the bounded type of a predicate is, or `Self` (`T`, `X::Item`, `<X as
/// Iterator>::Item`), if it is one of `generics`: the Leaker's environment proves the bound of
/// those, and an impl proves the bound of any other type (`Vec<T>: Clone`).
pub fn bounded_generic<'a>(bounded: &'a str, generics: &BTreeSet<String>) -> Option<&'a str> {
    let bounded = bounded.trim();
    let bounded = bounded.strip_prefix('<').unwrap_or(bounded);
    let end = bounded
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(bounded.len());
    let (head, rest) = bounded.split_at(end);
    if !(generics.contains(head) || head == "Self") {
        return None;
    }
    if rest.is_empty() || rest.starts_with("::") || rest.trim_start().starts_with("as ") {
        Some(head)
    } else {
        None
    }
}

/// The predicates of an item as `(bounded type, bounds)`: those of its where clauses (also of the
/// methods), and the bounds of the type parameters in the generics (`<T: A, U: B>`, `fn m<X:
/// C>`).
pub fn bound_clauses(text: &str) -> Vec<(String, String)> {
    // The bounded type and the bounds apart: at the first `:` that is not a `::` and not in brackets.
    fn colon(pred: &str) -> Option<usize> {
        let chars: Vec<(usize, char)> = pred.char_indices().collect();
        let mut depth = 0i32;
        for (k, &(at, c)) in chars.iter().enumerate() {
            // The `>` of an arrow (`fn() -> U::Item`) closes nothing.
            let arrow = k > 0 && chars[k - 1].1 == '-';
            match c {
                '(' | '[' | '<' => depth += 1,
                '>' if arrow => (),
                ')' | ']' | '>' => depth -= 1,
                ':' if depth == 0 => {
                    let before = k.checked_sub(1).map(|j| chars[j].1);
                    let after = chars.get(k + 1).map(|&(_, c)| c);
                    if before != Some(':') && after != Some(':') {
                        return Some(at);
                    }
                }
                _ => (),
            }
        }
        None
    }
    let mut out = Vec::new();
    for (at, _) in text.match_indices("where ") {
        let clause = where_clause(&text[at + "where ".len()..]);
        for pred in split_predicates(clause) {
            if let Some(at) = colon(pred) {
                out.push((
                    pred[..at].trim().to_string(),
                    pred[at + 1..].trim().to_string(),
                ));
            }
        }
    }
    // `<T: A + B, U: C>`: a type parameter after a `<` or a `,`, up to the end of its bounds.
    let generics = declared_generics(text);
    let chars: Vec<char> = text.chars().collect();
    for (i, w) in chars.windows(3).enumerate() {
        if !(w[1] == ':' && w[2] == ' ' && generics.contains(&w[0].to_string())) {
            continue;
        }
        let word_start = i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '_');
        let opener = chars[..i]
            .iter()
            .rev()
            .find(|c| !c.is_whitespace())
            .map_or(false, |c| *c == '<' || *c == ',');
        if !(word_start && opener) {
            continue;
        }
        let mut depth = 0i32;
        let mut end = chars.len();
        let mut previous = ' ';
        for (k, &c) in chars.iter().enumerate().skip(i + 3) {
            match c {
                '(' | '[' | '<' => depth += 1,
                ')' | ']' => depth -= 1,
                '>' if previous != '-' => depth -= 1,
                _ => (),
            }
            if depth < 0 || (depth == 0 && matches!(c, ',' | '{' | ';' | '=')) {
                end = k;
                break;
            }
            previous = c;
        }
        let bounds: String = chars[i + 3..end].iter().collect();
        out.push((w[0].to_string(), bounds.trim().to_string()));
    }
    out
}

/// Whether the header of the trait or the impl in `text` (its generics, supertraits and where
/// clause) names `Self` in a type (`where Rel<Self>: Default`, `T: AsRef<Rel<Self>>`), and not only
/// as the bounded type of a predicate (`Self: Sized`, `Self: 'x`).
fn header_names_self_in_type(text: &str) -> bool {
    text.lines()
        .map(str::trim_start)
        .find(|line| {
            ["pub trait ", "trait ", "impl<", "impl "]
                .iter()
                .any(|start| line.starts_with(start))
        })
        .map_or(false, |line| {
            let header = line.replace("Self: ", "").replace("Self:", "");
            has_word(&header, "Self")
        })
}

/// Whether the header of the impl in `text` (its generics and its where clause) names `Self`.
fn impl_header_names_self(text: &str) -> bool {
    text.lines()
        .map(str::trim_start)
        .find(|line| line.starts_with("impl<") || line.starts_with("impl "))
        .map_or(false, |line| has_word(line, "Self"))
}

/// The shapes of `FORCED_*` that the text of one item writes (what the Leaker must reject, `main.rs`
/// `FORCED`). Only the item's own text is read, so an impl has none of the type or the trait it is
/// for.
pub fn forced_in(text: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    if predicate_on_relative_type_with_self(text) {
        out.push(FORCED_SELF_IN_TRAIT_PRED);
    }
    if impl_self_type_names_bounded_generic(text) {
        out.push(FORCED_IMPL_HEADER_SELF);
    }
    if text
        .match_indices("two!(")
        .any(|(at, _)| !text[..at].ends_with("::"))
    {
        out.push(FORCED_MACRO_PATH);
    }
    if relative_elided_arg(text) {
        out.push(FORCED_ELIDED_ARG);
    }
    if relative_discriminant(text) {
        out.push(FORCED_RELATIVE_DISCRIMINANT);
    }
    if forced_value_path(text) {
        out.push(FORCED_VALUE_PATH);
    }
    out
}

/// Whether a body or a constant of `text` writes a path of a value that the Leaker must reject
/// under the settings in the attribute of the item (its first line, if it has one; the part of a
/// module has none, and the attribute of the module is read when the module is):
///
/// - the Leaker reads the values (`values = true`), and not through `enter_impl` (`scope = "impl"`),
///   which analyzes the self type and the trait's arguments only;
/// - no rule of the analysis allows what is written: not `crate::..`, `super::..`, `self::..` and the
///   constants of the module (the rules allow `::std`, the Leaker's crate by `::`, the primitive
///   types, and the names of the prelude where the attribute says `prelude = true`).
fn forced_value_path(text: &str) -> bool {
    let attribute = text.lines().next().filter(|line| line.starts_with("#["));
    let says = |setting: &str| attribute.map_or(false, |line| line.contains(setting));
    if attribute.is_some() && (!says("values = true") || says("scope = \"impl\"")) {
        return false;
    }
    value_paths_in(text).iter().any(|path| match path {
        ValuePath::Prelude => says("prelude = false"),
        ValuePath::Relative(_) | ValuePath::UseRoot => true,
    })
}

/// The arguments of `ty` if it is a path with no leading `::` applied to arguments, the last segment
/// of which is a name that starts with a capital (`R1<T>`, `self::R1<T, U>`, `super::base::B1<T>`):
/// a type that the Leaker leaks, unless it is `Self::A<..>` or a generic.
fn relative_path_arguments(ty: &str) -> Option<&str> {
    let ty = ty.trim();
    let end = ty
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
        .unwrap_or(ty.len());
    let (path, rest) = ty.split_at(end);
    if path.starts_with("::") || !rest.starts_with('<') || !ty.ends_with('>') {
        return None;
    }
    let mut segments = path.split("::");
    let (first, last) = (segments.next()?, path.rsplit("::").next()?);
    if first == "Self" || !last.starts_with(|c: char| c.is_ascii_uppercase()) {
        return None;
    }
    Some(&rest[1..rest.len() - 1])
}

/// Whether the where clause of the trait in `text` has a predicate on a type of a relative path that
/// names `Self` in its arguments (`where Rel<Self>: Default`).
fn predicate_on_relative_type_with_self(text: &str) -> bool {
    let header = text
        .lines()
        .map(str::trim_start)
        .find(|line| line.starts_with("pub trait ") || line.starts_with("trait "));
    let Some((_, clause)) = header.and_then(|line| line.split_once(" where ")) else {
        return false;
    };
    split_predicates(where_clause(clause)).iter().any(|pred| {
        pred.split_once(": ")
            .and_then(|(bounded, _)| relative_path_arguments(bounded))
            .map_or(false, |args| has_word(args, "Self"))
    })
}

/// Whether the self type of the impl in `text` is a path of the kind of `relative_path_arguments()`
/// that names a generic of the impl, which a bound of the generics or the where clause bounds by
/// `Self` (`impl<T: AsRef<Self>> Rel<T>`).
fn impl_self_type_names_bounded_generic(text: &str) -> bool {
    let Some(line) = text
        .lines()
        .map(str::trim_start)
        .find(|line| line.starts_with("impl<") || line.starts_with("impl "))
    else {
        return false;
    };
    let Some(args) = impl_self_type(text)
        .as_deref()
        .and_then(relative_path_arguments)
        .map(str::to_string)
    else {
        return false;
    };
    let mut bounded: Vec<String> = Vec::new();
    let mut take = |bounded_ty: &str, bounds: &str| {
        let name = bounded_ty.trim();
        if has_word(bounds, "Self")
            && !name.is_empty()
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            bounded.push(name.to_string());
        }
    };
    if let Some(rest) = line["impl".len()..].strip_prefix('<') {
        let (mut depth, mut previous, mut end) = (1i32, ' ', rest.len());
        for (at, c) in rest.char_indices() {
            match c {
                '(' | '[' | '<' => depth += 1,
                ')' | ']' => depth -= 1,
                '>' if previous != '-' => depth -= 1,
                _ => (),
            }
            if depth == 0 {
                end = at;
                break;
            }
            previous = c;
        }
        for param in split_top(&rest[..end], ',') {
            if let Some((name, bounds)) = param.split_once(':') {
                take(name, bounds);
            }
        }
    }
    if let Some((_, clause)) = line.split_once(" where ") {
        for pred in split_predicates(where_clause(clause)) {
            if let Some((name, bounds)) = pred.split_once(':') {
                take(name, bounds);
            }
        }
    }
    bounded.iter().any(|name| has_word(&args, name))
}

/// Whether a method of `text` takes `ae: Rel<'_>`, `Rel` a path with no leading `::`
/// (`Gen::method_sig()`).
fn relative_elided_arg(text: &str) -> bool {
    text.match_indices("<'_>").any(|(at, _)| {
        let before = &text[..at];
        let start = before
            .char_indices()
            .rev()
            .find(|(_, c)| !(c.is_alphanumeric() || *c == '_' || *c == ':'))
            .map_or(0, |(i, c)| i + c.len_utf8());
        let path = &before[start..];
        before[..start].ends_with("ae: ") && !path.is_empty() && !path.starts_with("::")
    })
}

/// Whether an enum of `text` has a discriminant that is a constant of its module or of an ancestor,
/// by a relative path (`V0 = (RLEN as isize) + 0`, `V1 = (super::base::BLEN as isize) + 100`).
fn relative_discriminant(text: &str) -> bool {
    text.match_indices(" as isize)").any(|(at, _)| {
        let Some(open) = text[..at].rfind("= (") else {
            return false;
        };
        let form = &text[open + "= (".len()..at];
        form == "RLEN" || (form.ends_with("base::BLEN") && !form.starts_with("::"))
    })
}

/// A path of a value that the text of an item writes and the copy writes as it is
/// (`Gen::value_usize()`, `Gen::value_statements()`), read from the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValuePath {
    /// A relative path in an expression of type `usize` (`crate::base::BLEN`, `super::base::bf()`,
    /// `RLEN`, `self::RLEN`, `ILEN`), with its first segment (`crate`, `super`, `self`, `RLEN`, ..).
    Relative(String),
    /// `Vec::<*const T>::new()`: a path of the prelude, relative where the Leaker does not allow
    /// the prelude.
    Prelude,
    /// `use super::base::bf as local_bf;`: the root of a `use` in a body.
    UseRoot,
}

/// The paths of values that `text` writes by a relative path or a name of the prelude, not the
/// absolute ones (`::krate::base::BLEN`). Only the values of `Gen::value_statements()` and
/// `Gen::value_const()`: not the discriminants (`relative_discriminant()`).
pub fn value_paths_in(text: &str) -> Vec<ValuePath> {
    // Where the `usize` expression of a body (`let _d: usize = P + 1;`) and of a constant
    // (`size_of::<*const T>() + P`) starts.
    let mut starts: Vec<&str> = Vec::new();
    for (at, found) in text.match_indices("let _d: usize = ") {
        starts.push(&text[at + found.len()..]);
    }
    for (at, found) in text.match_indices("size_of::<*const ") {
        let rest = &text[at + found.len()..];
        if let Some(call) = rest.find(">()") {
            if let Some(path) = rest[call + ">()".len()..].strip_prefix(" + ") {
                starts.push(path);
            }
        }
    }
    let mut out: Vec<ValuePath> = starts
        .into_iter()
        .filter(|start| !start.starts_with("::"))
        .map(|start| {
            let end = start
                .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
                .unwrap_or(start.len());
            // The first segment names the form (`crate`, `super`, `self`, `RLEN`).
            let head = start[..end].split("::").next().unwrap_or_default();
            ValuePath::Relative(head.to_string())
        })
        .collect();
    if text.contains("let _e = Vec::<*const ") {
        out.push(ValuePath::Prelude);
    }
    if text.contains(" as local_bf;") {
        out.push(ValuePath::UseRoot);
    }
    out
}

/// Whether the text of a bound is one that every type of the use site's choice has (`FRIENDLY` in
/// `prop.rs`, `?Sized`, a lifetime).
fn marker_bound(text: &str) -> bool {
    let text = text.trim();
    text.starts_with(['\'', '?'])
        || [
            "Sized",
            "Clone",
            "Copy",
            "Debug",
            "Default",
            "Send",
            "Sync",
            "Unpin",
            "PartialEq",
            "Eq",
            "PartialOrd",
            "Ord",
        ]
        .contains(&text.rsplit("::").next().unwrap_or(text))
}

/// A type whose output has an elided lifetime that the one lifetime of its inputs gives, and whose
/// inputs name the type parameter `param` (`sized`: it is `Sized`, which an array element needs):
/// a function pointer, one with a binder that names the lifetime of the input (the output takes it),
/// a trait object, and a function pointer whose input holds the parameter in a type of `::core`.
/// `draw` chooses.
fn elided_param_signature(param: &str, sized: bool, draw: u64) -> String {
    match draw % if sized { 4 } else { 3 } {
        0 => format!("fn(&{}) -> &u8", param),
        1 => format!("for<'hp> fn(&'hp {}) -> &u8", param),
        2 => format!("::std::boxed::Box<dyn ::core::ops::Fn(&{}) -> &u8>", param),
        _ => format!("fn(&::core::cell::Cell<[{}; 1]>) -> &u8", param),
    }
}

/// A type whose output has an elided lifetime that the one lifetime of its inputs gives, and whose
/// input is the optional entry `::std::vec::Vec<T>` (`Vec` needs `T: Sized`): a function pointer, and
/// the `Fn(..)` of a trait object that no box or reference holds (it would be the entry itself).
/// `draw` chooses.
fn elided_entry_signature(param: &str, draw: u64) -> String {
    if draw % 2 == 0 {
        format!("fn(&::std::vec::Vec<{}>) -> &u8", param)
    } else {
        format!(
            "::core::marker::PhantomData<dyn ::core::ops::Fn(&::std::vec::Vec<{}>) -> &u8>",
            param
        )
    }
}

/// Whether `text` has a const parameter of a type besides `usize`, `bool`, `i32`, `u8` and `char`
/// (`const V: u16`), or a type of the module with one (`RKu16<..>`).
fn wide_const_in(text: &str) -> bool {
    let named = text
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(|word| {
            word.len() > 3
                && matches!(word.as_bytes()[0], b'R' | b'B' | b'I' | b'J')
                && WIDE_CONSTS.iter().any(|(name, _)| &word[1..] == *name)
        });
    named
        || text.match_indices("const ").any(|(at, _)| {
            let rest = &text[at + "const ".len()..];
            let Some(colon) = rest.find(": ") else {
                return false;
            };
            let ty: String = rest[colon + 2..]
                .chars()
                .take_while(|c| c.is_alphanumeric())
                .collect();
            WIDE_CONSTS.iter().any(|(_, wide)| *wide == ty)
        })
}

/// The constructs (`CONSTRUCTS`) that the text of an item has. `param_default` and `lifetime_param`
/// are not in the text of the item as such, and are put where the generator writes them.
pub fn constructs_of(text: &str) -> Vec<&'static str> {
    let has = |needle: &str| text.contains(needle);
    let mut out = Vec::new();
    if has("type G<'g>") {
        out.push("gat");
    }
    if ["N", "K", "M", "V", "Y"]
        .iter()
        .any(|name| has(&format!("const {}: ", name)))
    {
        out.push("const_param");
    }
    if has(": impl ") {
        out.push("impl_trait_arg");
    }
    if has("for<'h") {
        out.push("binder");
    }
    if has("dyn ") {
        out.push("dyn_object");
    }
    if has("fn(") {
        out.push("fn_pointer");
    }
    if has("const C0: ") || has("const K0: ") {
        out.push("assoc_const");
    }
    if has("?::core::marker::Sized") || has("?Sized") {
        out.push("unsized_param");
    }
    if has(" where ") {
        out.push("where_clause");
    }
    out
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

/// The bounds that make the Leaker reject an entry in a predicate because it cannot tell a path from
/// an alias of a projection: `(generic, bound)` for each bound of a generic of the text (`norm()`
/// writes the bound) that names the generic in the arguments of a path (`hides_generic()`), if a
/// bound of the generic fixes an associated type (`fixes_assoc()`, in any scope of the text: a
/// where predicate of a method equates the generic of its trait too), and the text has an entry
/// that keeps the bound: a bound or where predicate that names `Self` or the self type of the impl.
pub fn hidden_bounds_in(text: &str) -> BTreeSet<(String, String)> {
    let generics = declared_generics(text);
    let clauses = bound_clauses(text);
    let plain = |bounded: &str| generics.contains(bounded.trim());
    let fixed: BTreeSet<&str> = clauses
        .iter()
        .filter(|(bounded, bounds)| {
            plain(bounded)
                && split_top(bounds, '+')
                    .into_iter()
                    .any(|bound| fixes_assoc(bound, bounded.trim()))
        })
        .map(|(bounded, _)| bounded.trim())
        .collect();
    let self_type = impl_self_type(text).map(|ty| norm(&ty));
    let keeps = clauses.iter().any(|(bounded, bounds)| {
        (bounded.trim() != "Self" && (has_word(bounded, "Self") || has_word(bounds, "Self")))
            || self_type.as_ref().map_or(false, |ty| {
                names_type(&norm(bounded), ty) || names_type(&norm(bounds), ty)
            })
    });
    let mut found = BTreeSet::new();
    if !keeps {
        return found;
    }
    for (bounded, bounds) in &clauses {
        let name = bounded.trim();
        if !plain(bounded) || !fixed.contains(name) {
            continue;
        }
        for bound in split_top(bounds, '+') {
            if !equates_assoc(bound) && hides_generic(bound, name) {
                found.insert((name.to_string(), norm(bound)));
            }
        }
    }
    found
}

/// Whether the bound `bound` of the generic `name` fixes an associated type of it, as the Leaker
/// reads it: an equation (`Iterator<Item = u8>`), or a trait that is not one of the standard
/// library and does not name `name` in its arguments (`::krate::base::BItem`), which may have a
/// supertrait that equates one.
fn fixes_assoc(bound: &str, name: &str) -> bool {
    if equates_assoc(bound) {
        return true;
    }
    let bound = without_binder(bound);
    let bound = bound.strip_prefix('?').unwrap_or(bound).trim_start();
    if bound.is_empty() || bound.starts_with('\'') {
        return false;
    }
    let open = bound.find(['<', '(']).unwrap_or(bound.len());
    let path = bound[..open].trim();
    let std = match path.strip_prefix("::") {
        Some(rest) => matches!(rest.split("::").next(), Some("core" | "std" | "alloc")),
        None => is_prelude_trait(path),
    };
    !std && !has_word(&bound[open..], name)
}

/// Whether the trait bound `bound` equates an associated type (`Iterator<Item = u8>`, `Tr<'a, A =
/// X>`), which a `Fn(..) -> R` does not.
fn equates_assoc(bound: &str) -> bool {
    let bound = without_binder(bound);
    let Some(open) = bound.find(['<', '(']) else {
        return false;
    };
    if !bound[open..].starts_with('<') {
        return false;
    }
    // The arguments, up to the `>` that closes them.
    let (mut depth, mut previous) = (0i32, ' ');
    let mut end = bound.len();
    for (at, c) in bound[open..].char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' => depth -= 1,
            '>' if previous != '-' => depth -= 1,
            _ => (),
        }
        if depth == 0 {
            end = open + at;
            break;
        }
        previous = c;
    }
    split_top(&bound[open + 1..end], ',')
        .into_iter()
        .any(|arg| {
            let name: String = arg
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let rest = arg[name.len()..].trim_start();
            !name.is_empty() && rest.starts_with('=') && !rest.starts_with("==")
        })
}

/// The bound without the binder `for<'h>` before it.
fn without_binder(bound: &str) -> &str {
    let bound = bound.trim();
    let Some(rest) = bound.strip_prefix("for") else {
        return bound;
    };
    let rest = rest.trim_start();
    if !rest.starts_with('<') {
        return bound;
    }
    match rest.find('>') {
        Some(end) => rest[end + 1..].trim_start(),
        None => bound,
    }
}

/// Whether the arguments of the path of the bound `bound` (`Fn(X<T>) -> u8`, `PartialEq<W<T>>`) name
/// `name` in a path that may be a type alias of a projection of it, as the Leaker reads a bound: not
/// `name` itself (`From<T>`), a type that syntax builds (`&T`, `(T,)`), nor an absolute path of the
/// standard library (`::core::option::Option<T>`), but a path that has it in its arguments or
/// starts with it (`X<T>`, `T::Item`, `<T as Iterator>::Item`), and a path around one of those.
fn hides_generic(bound: &str, name: &str) -> bool {
    let bound = without_binder(bound);
    let chars: Vec<char> = bound.chars().collect();
    let is_id = |c: char| c.is_alphanumeric() || c == '_';
    let Some(start) = chars.iter().position(|c| matches!(c, '<' | '(')) else {
        return false;
    };
    // The open brackets after the path of the bound, each with whether it belongs to a path or a
    // qualified type that may be an alias.
    let mut frames: Vec<bool> = Vec::new();
    let mut i = start;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '<' | '(' | '[' => frames.push(i != start && opens_path(&chars[..i], c)),
            ')' | ']' => {
                frames.pop();
            }
            '>' if chars[i - 1] != '-' => {
                frames.pop();
            }
            _ if is_id(c)
                && !chars[i - 1].is_alphanumeric()
                && !matches!(chars[i - 1], '_' | '\'') =>
            {
                let word: String = chars[i..].iter().take_while(|c| is_id(**c)).collect();
                let len = word.chars().count();
                let before: String = chars[..i]
                    .iter()
                    .rev()
                    .filter(|c| !c.is_whitespace())
                    .take(2)
                    .collect();
                let after: String = chars[i + len..]
                    .iter()
                    .filter(|c| !c.is_whitespace())
                    .take(2)
                    .collect();
                // A later segment of a path (`base::T`) is not the generic.
                if word == name && before != "::" && (after == "::" || frames.contains(&true)) {
                    return true;
                }
                i += len;
                continue;
            }
            _ => (),
        }
        i += 1;
    }
    false
}

/// Whether the bracket `open`, after `prefix`, belongs to a path (`B2<`, `Fn(`) or a qualified type
/// (`<T as Tr>`) that may be a type alias: not a tuple, an array, a function pointer, the path of a
/// bound (`dyn Tr<T>`), or an absolute path of the standard library.
fn opens_path(prefix: &[char], open: char) -> bool {
    let is_id = |c: char| c.is_alphanumeric() || c == '_';
    if open == '[' {
        return false;
    }
    let trimmed: String = prefix.iter().collect::<String>().trim_end().to_string();
    if !trimmed.chars().next_back().map_or(false, is_id) {
        // `<T as Tr>::Item` has a qualified self type, `(A, B)` is a tuple.
        return open == '<';
    }
    let head: String = trimmed
        .chars()
        .rev()
        .take_while(|c| is_id(*c) || *c == ':')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let before_head = trimmed[..trimmed.len() - head.len()].trim_end();
    let bound_path = before_head.ends_with('+')
        || ["dyn", "impl"].iter().any(|keyword| {
            before_head.ends_with(keyword)
                && !before_head[..before_head.len() - keyword.len()]
                    .chars()
                    .next_back()
                    .map_or(false, is_id)
        });
    let absolute_std = ["::core::", "::std::", "::alloc::"]
        .iter()
        .any(|prefix| head.starts_with(prefix));
    head != "fn" && !bound_path && !absolute_std
}

/// The self type of an impl, written out (`impl<T> Tr for S<T> where ..`: `S<T>`), or `None` if the
/// text has no impl.
fn impl_self_type(text: &str) -> Option<String> {
    let line = text
        .lines()
        .map(str::trim_start)
        .find(|line| line.starts_with("impl<") || line.starts_with("impl "))?;
    let mut rest = &line["impl".len()..];
    if rest.starts_with('<') {
        let (mut depth, mut previous) = (0i32, ' ');
        for (at, c) in rest.char_indices() {
            match c {
                '(' | '[' | '<' => depth += 1,
                ')' | ']' => depth -= 1,
                '>' if previous != '-' => depth -= 1,
                _ => (),
            }
            if depth == 0 {
                rest = &rest[at + 1..];
                break;
            }
            previous = c;
        }
    }
    let head = where_clause(rest.trim_start());
    let head = head.split(" where ").next().unwrap_or(head).trim();
    // `Tr<T> for S<T>`: the part after the ` for ` that is not in brackets.
    let (mut depth, mut previous) = (0i32, ' ');
    for (at, c) in head.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' => depth -= 1,
            '>' if previous != '-' => depth -= 1,
            _ => (),
        }
        if depth == 0 && head[at..].starts_with(" for ") {
            return Some(head[at + " for ".len()..].trim().to_string());
        }
        previous = c;
    }
    (!head.is_empty()).then(|| head.to_string())
}

/// Whether `text` (`norm()`) has the type `ty` in it as a whole: not the end of a longer path
/// (`self::S<T>`) or the start of a longer name.
fn names_type(text: &str, ty: &str) -> bool {
    let is_id = |c: char| c.is_alphanumeric() || c == '_';
    text.match_indices(ty).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + ty.len()..].chars().next();
        !before.map_or(false, |c| is_id(c) || c == ':')
            && (ty.ends_with('>') || !after.map_or(false, is_id))
    })
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

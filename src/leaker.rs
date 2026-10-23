//! The [`Leaker`], which finds the types depending on the type context.

use crate::analysis::{drop_optional, names_self, Analysis, PartItem};
use crate::analyze::{declared_names, AnalyzeVisitor, Finished, ModuleNames, ScopeSpec, Scopes};
use crate::config::LeakerConfig;
use crate::context::{Context, ContextId, Contexts, Gat, SelfRequirement};
#[cfg(doc)]
use crate::error::Reason;
use crate::error::{Cause, ErrorItem};
use crate::graph::{is_whole_reference, NodeKey, TypeGraph};
use crate::referrer::Referrer;
use crate::util::{ident_name, std_crate_paths, AllowedPaths};
use crate::view::View;
use proc_macro2::{Span, TokenStream};
use safegraph::algo::max_flow::edmonds_karp_with_flows;
use safegraph::graph::Graph;
use safegraph::BTreeGraph;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Display;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::result::Result;
use syn::parse::{Parse, ParseStream};
use syn::visit::Visit;
use syn::*;
use template_quote::{quote, ToTokens};

/// Finds the types of an item that depend on where they are written, and collects them in a
/// [`Referrer`].
///
/// Analyze a whole item with an `intern_*` method, or parts of one in a scope with an `enter_*`
/// method, then call [`Leaker::result()`]. Each returns the top-level [`ContextId`] of what it
/// analyzed (keep it for [`Referrer::visitor()`]). See the [quick start](crate#quick-start).
///
/// ```
/// # use {syn::*, type_leak::*};
/// let item: ItemTrait = parse_quote! {
///     trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self) -> Rel; }
/// };
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_trait(&item, true);
/// let referrer = leaker.result().finish().unwrap();
/// // `Wrap<T>` and `Rel` depend on where they are written, so each needs a Repeater impl.
/// assert_eq!(referrer.required().count(), 2);
/// assert_eq!(referrer.context_generics(top_level).params.len(), 1);
/// ```
///
/// # Errors and recovery
///
/// An error does not stop the analysis: the Leaker copies the part that cannot be interned as
/// written, whole, with nothing in it leaked (`Rel` in `MyType<'_, Rel>` stays as written). The
/// errors are kept, each once, in the order found (those of nested scopes in place), and read
/// through [`Leaker::result()`]. If a closure panics, its scope rolls back and records nothing. The
/// [`Referrer`] does not depend on what is allowed afterwards ([`LeakerResult`]).
///
/// ```
/// # use {syn::*, type_leak::*};
/// let item: ItemStruct = parse_quote!(struct S { a: Rel, b: ::p::W<'_, dyn RelTr> });
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&item, true);
/// let mut result = leaker.result();
/// // `W<'_, ..>` cannot leak `dyn RelTr`.
/// assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));
/// // `Rel` is an entry as usual, and `dyn RelTr` is copied as written.
/// let referrer = result.clone().into_referrer();
/// assert_eq!(referrer.iter().count(), 1);
/// // Allowing the error leaves the Referrer as it is, and `finish()` returns it.
/// result.allow(Cause::Trait(parse_quote!(RelTr)));
/// assert_eq!(result.finish().unwrap(), referrer);
/// ```
///
/// # Limitations
///
/// The Leaker reads only the item you give it, not the definitions of the paths it names, so it
/// reports these as errors (the error's `help` says what to write instead;
/// [`LeakerResult::allow()`] accepts one):
///
/// - `MyType<'_>`, `impl Trait`, and a relative trait or value in a bound: write an absolute path.
///   (`T: ?Sized` is left as written: rustc allows `?` only on the language's `Sized`. A bound
///   `Sized`, as in `Self: Sized`, is a relative trait like `Clone`:
///   [`allow_std_prelude()`](LeakerConfig::allow_std_prelude).)
/// - A path whose definition needs `T: 'a` ([Outlives bounds](#outlives-bounds)).
/// - An output lifetime that only a leaked type would keep ([Binder](#binder-lifetimes),
///   [Elided](#elided-lifetimes) and [Method](#method-lifetimes) lifetimes).
/// - A `dyn Trait` without a lifetime bound ([Trait objects](#trait-objects)).
/// - A leaked type in a bound or where predicate ([Bounds](#bounds-that-name-leaked-types)).
/// - `Self::A` where `Self` is a type, and bounds that name `Self`
///   ([`allow_self`](Leaker::intern_struct)).
/// - A bare identifier that may be a const ([Bare identifiers](#bare-identifiers-and-consts)).
/// - An absolute path of another crate, trusted only if both crates have it
///   ([`allowed_paths`](Leaker::allowed_paths)).
/// - A shadowed primitive name (`type u8 = i32;`): emit [`Leaker::assertions()`] to catch it.
/// - A value: the default of an associated const, the value of a const or static, and the body of a
///   function are not analyzed, so a relative path in them (`const N: usize = inner::N;`, or a
///   type: `size_of::<inner::Rel>()`) breaks the copy and the Leaker does not tell.
///   [`LeakerConfig::check_values()`] reads them: it rejects a path that is not a type, and leaks a
///   type in a type position (a cast, a `let`, a closure, a turbofish, a qualified path), which the
///   [`Visitor`](crate::Visitor) replaces. An enum discriminant and an array length are always analyzed.
///
/// # What the Leaker checks
///
/// Each check rejects code whose copy rustc would reject. A copy that still fails to compile shows
/// the check in its rustc error:
///
/// | `rustc` error in the copy | Check |
/// | --- | --- |
/// | E0309 | [Outlives bounds](#outlives-bounds) |
/// | E0581, E0582 | [Binder lifetimes](#binder-lifetimes) |
/// | E0106 | [Elided lifetimes](#elided-lifetimes) |
/// | E0478 | [Trait objects](#trait-objects) |
/// | E0195 | [Method lifetimes](#method-lifetimes) |
/// | E0277, E0271, E0275, E0284 in a bound | [Bounds](#bounds-that-name-leaked-types) |
/// | E0220, E0223 on `Self::A` | [`allow_self`](Leaker::intern_struct) |
/// | a constant read as a type | [Bare identifiers and consts](#bare-identifiers-and-consts) |
///
/// ## Outlives bounds
///
/// An entry under `&'a` needs `T: 'a` in its Repeater impl, and the Leaker records only the bounds
/// you write. So write them, and give [`Referrer::visitor()`] the whole item, not a type cut out of
/// another. For `Self: 'a` and associated types, see [`Referrer::context_self_outlives()`] and
/// [`Leaker::intern_trait()`].
///
/// ```ignore
/// fn f<'a, T: 'a>(x: Rel<'a, T>);  // `T: 'a` is needed if `struct Rel<'a, T>(&'a T)`
/// ```
///
/// ## Binder lifetimes
///
/// If the output of a `for<'x> ..` signature names `'x`, an input must keep `'x` as written, which
/// a leaked type does not. So the whole function pointer or trait object is leaked, which is an
/// error in a bound:
///
/// ```ignore
/// F: for<'x> Fn(Rel<'x>) -> &'x u8          // error
/// F: for<'x> Fn(&'x u8, Rel<'x>) -> &'x u8  // good: `&'x u8` keeps `'x`
/// ```
///
/// ## Elided lifetimes
///
/// An elided output lifetime comes from the only input lifetime, and a leaked type adds more. So
/// the whole function pointer or trait object is leaked, which is an error in a bound, in `impl
/// Fn(..)`, under an elided reference, and in a method without a reference receiver. A hidden
/// lifetime (`Ref<u8>`) is not seen: write `Ref<'_, u8>`. rustc counts the lifetimes of each input
/// apart, so a leaked type adds one even if another input has it.
///
/// ```ignore
/// fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>);  // good: the trait object is leaked whole
/// fn f(&self, x: &dyn Fn(&Rel<T>) -> &u8);      // error: elided reference
/// fn g<'x>(x: &'x Rel<T>) -> &'x u8;            // good: the lifetimes are named
/// fn g(x: &'a u8, y: Rel<T>) -> &u8;            // error: `Rel<T>` names `'a` too (`T: 'a`)
/// ```
///
/// ## Trait objects
///
/// A `dyn Trait` without a lifetime bound takes its bound from where it is written, which the
/// Leaker cannot see, so the reference or path around it is leaked instead. Writing the bound (`&'a
/// (dyn RelTrait + 'a)`) also works, except for a trait with a bound of its own.
///
/// ```ignore
/// fn f(x: &dyn RelTrait);          // error: elided lifetime
/// fn f<'a>(x: &'a dyn RelTrait);   // good: the whole reference is leaked
/// ```
///
/// ## Method lifetimes
///
/// A method lifetime `'b` that only a leaked type mentions is late-bound in a trait but early-bound
/// in the copy, if the return type names `'b`. This affects methods of traits and trait impls:
///
/// ```ignore
/// fn f<'b>(&self, x: Wrap<'b>) -> &'b u8;       // error
/// fn f<'b>(&'b self, x: Wrap<'b>) -> &'b u8;    // good: `&'b self` keeps `'b`
/// ```
///
/// An absolute type of another crate that is the only one to keep `'b` (`fn f<'b>(&self, x:
/// ::ext::V<'b>) -> &'b u8`) is no optional entry: every Referrer copies it as written, so its
/// crate must resolve there.
///
/// ## Bounds that name leaked types
///
/// A leaked type in a bound or where predicate fails if its Repeater impl needs a bound that the
/// environment of the item has only in its rewritten form; a type in a field or a signature is
/// fine. The same holds for a bound that names `Self`, `Self::A` or an associated type that rustc
/// normalizes (such as the `Item` of an `Iterator<Item = ..>` bound in scope), and for a projection
/// of a generic bounded by a trait outside the standard library: supertraits are not read, so it
/// counts even where it would compile. (Not so where the bound is of the generic itself and
/// another bound of it fixes an associated type: the bound is left out, see below.)
///
/// ```ignore
/// struct S<T: Fn(Rel)>(Wrap<T>) where Wrap<T>: Sized;  // error: `Wrap<T>` in a predicate
/// struct S<T: Fn(Rel)>(Wrap<T>);                       // good
/// ```
///
/// A bound of a generic `T` that names `T` itself in the arguments of its path (`T: Fn(Wrap<T>)`,
/// `T: PartialEq<<T as Iterator>::Item>`, also as a where predicate of the scope that declares `T`)
/// is left out of the context of the entries that use `T`, so the Repeater's impl of the entry does
/// not carry it, if another bound of `T` fixes an associated type: an equation (`T: Iterator<Item
/// = u8>`, `T: Tr<A = X>`), or a bound of a trait that is not one of the standard library, whose
/// supertrait may equate one (`T: ::p::WIter`). It matters because rustc proves the bounds of the
/// impl whenever the copy names the entry in a predicate, in an environment in which it has not
/// normalized the bounds yet, and cannot normalize `T: Fn(<T as Iterator>::Item)` with the
/// equation first (E0277 "expected an `Fn(u8)` closure, found `T`"): an alias that hides the
/// projection (`type It<T> = <T as Iterator>::Item;`) looks like a struct to the Leaker, so the
/// bound is left out whatever it names. The bounds that are equations (even if they name `T`:
/// `T: Iterator<Item = Wrap<T>>`), the bounds of other generics (`U: Fn(T::Item)`: the Leaker still
/// rejects it where it is written as a projection, and does not see an alias), a bound of a
/// predicate of an inner scope (a method's where clause) and the other bounds of `T` stay. So do
/// all the bounds in the context of an entry that names `Self` (the self type of the item,
/// `Self::A` of its trait), and of one that writes out the self type of the item (`impl<T:
/// Iterator<Item = u8> + Fn(T::Item)> S<T>` with the entry `S<T>`): the Leaker knows that such a
/// type is well-formed only with the bounds that its item declares. An entry that depends on `Self`
/// only through a bound or an outlives predicate (`Self: 'x`) keeps whole the bounds of the
/// generics that the description of `Self` names (`impl<T: ..> S<T>`, `Tr<T>`), and no others.
///
/// The leaked type must not need a bound that is left out: with `struct Rel<T: Fn(Alias<T>)>(T);`
/// and `T: Iterator<Item = u8> + Fn(Alias<T>)`, the Repeater's impl of `Rel<T>` has only `T:
/// Iterator<Item = u8>`, and rustc rejects it in the crate of the Leaker (E0277). The Leaker cannot
/// see the declaration of the type, so it does not tell. Write the type without the bound, or the
/// bound without the generic (`T: Fn(u8)`).
///
/// ## Bare identifiers and consts
///
/// A bare identifier argument (`MyType<MY_CONST>`) may be a type or a const, and the Leaker cannot
/// tell which. It leaks the type that holds it as a whole, which is right for both. If it cannot,
/// leaking the identifier alone is right only for a type: a const read as a type fails in the copy
/// (rustc: E0573 in the Repeater's impl, E0747 in the copy). The Leaker does so only where there
/// is no other way, and the copy fails if it is a const:
///
/// - an elided lifetime or reference in the holder (`MyType<'_, MY_CONST>`, `&dyn Tr<MY_CONST>`),
/// - `impl Trait` in the holder,
/// - the path of the holder, if it is a name that a module of [`intern_mod()`](Leaker::intern_mod)
///   declares (`Local<MY_CONST>`: the copy declares `Local` again, and the module's own type takes
///   a type, as a rule),
/// - a trait bound, a supertrait or the trait of an impl (`T: Tr<MY_CONST>`), which has no type
///   to leak.
///
/// Any other cause is an error, with the identifier as its cause: `Self` where it is denied
/// ([`allow_self`](Leaker::intern_struct)) or `Self::A` of a type, the default of a parameter that
/// names `Self`, a bound of a generic that names `Self`, a method lifetime that the copy would leave
/// unconstrained ([Method lifetimes](#method-lifetimes)), an expression that cannot be leaked, and
/// a name that a module declares beside the identifier in the path of another crate
/// (`Other<Local, MY_CONST>`: the copy declares `Local` again, so the path is not leaked as a
/// whole, and the Leaker knows nothing of the parameters of `Other`).
/// Write the identifier so that it is not leaked: an absolute path, which a const needs in braces,
/// a name in [`allowed_paths`](Leaker::allowed_paths), or, in a module, a name that the module
/// declares (a declaration, or `use` with an absolute path), which is copied as written.
///
/// ```ignore
/// fn f(x: ::path::MyType<MY_CONST>);                  // good: the path is leaked
/// fn f(x: ::path::MyType<'_, MY_CONST>);              // error: treated as a type
/// fn f(x: ::path::MyType<{ ::my_crate::MY_CONST }>);  // good: not leaked
/// struct S(::path::MyType<[Self], MY_CONST>);         // error if `Self` is denied
/// struct S(::path::MyType<[Self], { ::my_crate::MY_CONST }>);  // good
/// // In a module that declares `Local`:
/// struct S(Local<MY_CONST>);                          // the identifier is leaked as a type
/// struct S(::path::MyType<Local, MY_CONST>);          // error: `MY_CONST` may be a const
/// ```
#[derive(Default)]
pub struct Leaker {
    /// Paths that are left as written instead of being leaked, such as `::serde`.
    ///
    /// A path matches if it starts with one of these, and both have a leading `::` or neither:
    /// `parse_quote!(::serde)` allows `::serde::..` and not `serde::..`. Its generic arguments are
    /// still analyzed. You promise that an allowed crate is the same on both sides.
    ///
    /// Another absolute path, such as `::serde::Wrap<(Rel,)>`, may name a different crate in the
    /// copy, which the Leaker cannot tell. So it is leaked as an *optional entry*
    /// ([`Referrer::must()`] is `false`), and the Referrer chooses whether a copy replaces it
    /// ([`Referrer::allow_ambiguous_crate_name()`]). `::core::..` and `$crate::..` are never
    /// optional, and `::std` and `::alloc` are like any other crate unless allowed.
    /// [`Referrer::external_crates_require()`] lists the crates an entry names.
    ///
    /// Not so for an absolute type that names the `Self` of a struct, an enum or a union nested in
    /// the analyzed syntax (`::serde::Wrap<Self>` in [`View::visit_item()`]): it is copied as
    /// written whatever the Referrer chooses, so that its `Self` is the copy's own type (see
    /// [`Visitor::with_nested_self()`](crate::Visitor::with_nested_self)).
    pub allowed_paths: Vec<Path>,
    /// The types that depend on the context (the required entries).
    types: TypeGraph,
    /// The types the Referrer may replace (the optional entries).
    optional_types: TypeGraph,
    /// The result of [`Leaker::reduce_roots()`]: required, then optional. `None` if outdated.
    reachable_types: Option<(Vec<NodeKey>, Vec<NodeKey>)>,
    contexts: Contexts,
    /// Numbered on their own, so that optional entries do not change the ids of required ones.
    optional_contexts: Contexts,
    /// The top-level contexts of the scopes (also of those with errors), which the [`Referrer`]
    /// keeps without an entry too.
    top_levels: BTreeSet<ContextId>,
    assumed_primitives: Vec<Ident>,
    /// Whether the values of an item are read ([`LeakerConfig::check_values()`]).
    check_values: bool,
    /// Whether an entry that names the `Self` of a nested struct, enum or union together with a
    /// bound that does is accepted ([`LeakerConfig::nested_self_entries()`]).
    nested_self_entries: bool,
    /// The parts copied as written.
    parts: Vec<PartItem>,
    /// The errors of the outermost calls, in the order found.
    errors: Vec<ErrorItem>,
}

impl Leaker {
    /// Creates a [`Leaker`] that allows no path, so every relative path is leaked.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`Leaker`] that leaves the paths allowed by `config` as written.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allowed_paths: config.allowed_paths,
            check_values: config.check_values,
            nested_self_entries: config.nested_self_entries,
            ..Self::default()
        }
    }

    /// Returns a compile-time check that primitive type names are not shadowed.
    ///
    /// The Leaker copies primitive names as written (`u8` in `const N: u8`, or a name allowed by
    /// [`LeakerConfig::allow_primitive()`]), which is wrong if an item shadows the name (`type u8 =
    /// i32;`). Emit the result in the Leaker's type context: it fails to compile if a name is
    /// shadowed, and is empty if none was copied.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S<const N: u8> { a: Rel }), true);
    /// assert!(!leaker.assertions().is_empty()); // it checks that `u8` is the primitive type
    /// ```
    pub fn assertions(&self) -> TokenStream {
        if self.assumed_primitives.is_empty() {
            return TokenStream::new();
        }
        quote! {
            #[allow(dead_code, non_snake_case)]
            const _: () = {
                #(for ident in &self.assumed_primitives) {
                    fn #{Ident::new(
                        &format!("__type_leak_assert_{}_is_primitive", ident),
                        Span::call_site(),
                    )}(x: &#ident) -> &::core::primitive::#ident { x }
                }
            };
        }
    }

    /// Analyzes a struct and returns its top-level context.
    ///
    /// The generics of the item are the generics of the analysis, and `Self` is the struct
    /// ([`SelfRequirement::Type`]). Every `intern_*` method works alike: the errors are recorded
    /// ([recovery](Leaker#errors-and-recovery)).
    ///
    /// `allow_self` says what happens to a type that names `Self`, in every method that takes it.
    /// With it, the type is one entry, written for the struct (for the implementors of a trait:
    /// [`intern_trait()`](Leaker::intern_trait)). But `Self::A` where `Self` is a type (struct,
    /// enum, union or impl) is never leaked whole (E0220, E0223). That is an error where it must be
    /// (`Rel<Self::A>`), as is a bound of the generics of an entry that names it: write `<Self as
    /// ::path::Trait>::A`. A type that needs no leaking is copied as written (`Vec<Self::A>`).
    ///
    /// Without it, a type that names `Self` is split into parts, which is an error if it cannot be
    /// (`Rel<Self>`, `Self::Item`), or if the bounds of the generics it uses name `Self`. Use it if
    /// the Repeater impls are for a marker type, which cannot name `Self`:
    /// [`Visitor::with_self()`](crate::Visitor::with_self) then replaces `Self` in the copies.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: Wrap<T>, b: Rel<Self> });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&item, true);
    /// // `Rel<Self>` is an entry, written for `S<T>`.
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
    /// // Without `allow_self`, `Rel<Self>` is split into parts, and `Rel` cannot be leaked alone.
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&item, false);
    /// assert!(leaker.result().finish().is_err());
    /// ```
    pub fn intern_struct(&mut self, item: &ItemStruct, allow_self: bool) -> ContextId {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_struct(view, item)
        })
        .0
    }

    /// Analyzes an enum and returns its top-level context, like
    /// [`intern_struct()`](Leaker::intern_struct).
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemEnum = parse_quote!(enum E<T> { A(Wrap<T>), B { next: Rel2<Self> } });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_enum(&item, true);
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
    /// ```
    pub fn intern_enum(&mut self, item: &ItemEnum, allow_self: bool) -> ContextId {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_enum(view, item)
        })
        .0
    }

    /// Analyzes a union and returns its top-level context, like
    /// [`intern_struct()`](Leaker::intern_struct).
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemUnion = parse_quote!(union U<T: ::core::marker::Copy> { a: Wrap<T>, b: Rel });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_union(&item, true);
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
    /// ```
    pub fn intern_union(&mut self, item: &ItemUnion, allow_self: bool) -> ContextId {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_union(view, item)
        })
        .0
    }

    /// Analyzes an impl and returns its top-level context.
    ///
    /// `Self` is the self type ([`SelfRequirement::Type`]), and `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct). Function bodies and const values are not
    /// analyzed unless [`LeakerConfig::check_values()`] is on, and the trait path must not depend
    /// on the context.
    ///
    /// An entry in the self type must not depend on `Self`, because the copy writes the self type
    /// for `Self`. It does if a bound of the impl names `Self` and the entry uses that generic
    /// (`impl<T> S<T> where T: AsRef<Option<Self>>` has the entry `S<T>`). That is an error, here
    /// and in an impl that [`View::visit_item()`] or [`intern_mod()`](Leaker::intern_mod) visits
    /// outside of another item: write the self type with an absolute path (`::my_crate::S<T>`), or
    /// the bounds without `Self`. Not affected: an optional entry (copied as written),
    /// [`enter_impl()`](Leaker::enter_impl), and a bound of an outer scope or of a method.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemImpl = parse_quote! {
    ///     impl<T> ::path::MyTrait<Rel<T>> for ::path::Host<T> {
    ///         fn f(&self) -> Rel2<Self> { Skipped::new() } // the body is skipped
    ///     }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_impl(&item, true);
    /// // `Rel<T>` and `Rel2<Self>` (`::path::Host<T>` names another crate: it is optional).
    /// assert_eq!(leaker.result().finish().unwrap().required().count(), 2);
    /// ```
    pub fn intern_impl(&mut self, item: &ItemImpl, allow_self: bool) -> ContextId {
        // The scope checks the path of the trait, and the walk analyzes its arguments.
        let trait_path = item.trait_.as_ref().map(|(_, path, _)| path);
        self.enter_impl_of(
            &item.generics,
            &item.self_ty,
            trait_path,
            allow_self,
            |view| view.walk_item_impl(item),
        )
        .0
    }

    /// Analyzes a trait and returns its top-level context.
    ///
    /// `Self` is the implementor ([`SelfRequirement::Bound`]), so an entry that names it is written
    /// for the implementors of the trait. `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct).
    ///
    /// Unlike [`enter_trait()`](Leaker::enter_trait), it knows the generic associated types (GATs)
    /// of the trait. A projection that names a GAT (`Self::G<'x>`, `<Self as Tr<..>>::G<'x>`)
    /// records the outlives bounds of the GAT (`where Self: 'b`, `T: 'b`, `'a: 'b`, `Self::A: 'b`)
    /// with `'b` replaced by `'x`, as if the method wrote them. The copy has to prove them, as for
    /// a bound you write: in an impl for another type, add `where T: 'x` or `where 'a: 'x` to the
    /// method, and `where Self: 'x` unless the receiver is `&'x self`. The GAT of a supertrait or
    /// of the trait of a generic (`T::G<'x>`) is not seen: write the bound on the method.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemTrait = parse_quote!(trait Tr { type Out; fn g(&self) -> Wrap<Self::Out>; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
    /// // `Wrap<Self::Out>` names `Self`, so its impl is for the implementors of the trait.
    /// let context = referrer.context(0);
    /// assert!(matches!(referrer.context_self(context), Some(SelfRequirement::Bound(_))));
    /// ```
    pub fn intern_trait(&mut self, item: &ItemTrait, allow_self: bool) -> ContextId {
        // The scope knows the generic associated types of the trait: what they require is part of
        // the context of an entry that names them.
        let gats = Gat::of_items(&item.items);
        self.enter_trait_of(&item.ident, &item.generics, gats, allow_self, |view| {
            syn::visit::visit_item_trait(view, item)
        })
        .0
    }

    /// Analyzes a module and returns its [`ContextId`].
    ///
    /// Every item in it is a nested item, so its generics are local, and `Self` has no meaning in
    /// it. Function bodies and const values are not analyzed unless [`LeakerConfig::check_values()`]
    /// is on. The names declared in the module are copied with it, not leaked (the rules of
    /// `declared` in [`enter_mod()`](Leaker::enter_mod)).
    /// [`Referrer::module_path()`] gives the path of a nested module, and
    /// [`Referrer::context_module()`] the module of an entry. Two calls return two modules: rewrite
    /// the copy of each with its own id ([`Referrer::visitor()`]).
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemMod = parse_quote! {
    ///     mod m {
    ///         struct Local;
    ///         struct S { a: Local, b: Outside }
    ///         mod inner { struct T(Outside2); }
    ///     }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_mod(&item);
    /// let referrer = leaker.result().finish().unwrap();
    /// assert_eq!(referrer.iter().count(), 2); // `Local` is declared in `m`: it is not leaked
    /// // The Repeater impl of `Outside2` is written in `inner`.
    /// let module = referrer.context_module(referrer.context(1)).unwrap();
    /// assert_eq!(referrer.module_path(module).unwrap()[0], "inner");
    /// ```
    pub fn intern_mod(&mut self, item: &ItemMod) -> ContextId {
        let names = match &item.content {
            Some((_, items)) => declared_names(items),
            None => ModuleNames::default(),
        };
        let spec = ScopeSpec::Module {
            names,
            path: Some(Vec::new()),
            hint: None,
        };
        self.enter_spec(spec, |view| syn::visit::visit_item_mod(view, item))
            .0
    }

    /// Analyzes part of a module: runs `f` on a [`View`] inside the module, and returns the
    /// [`ContextId`] of the module and the result of `f`.
    ///
    /// The items visited are nested items, so their generics are local, and `Self` has no meaning
    /// directly in `f`. Module ids come from a counter, so equal calls give equal ids; `hint` is
    /// only for debugging ([`Referrer::module_hint()`]).
    ///
    /// `declared` lists the names declared in the module, which are copied with it, not leaked.
    /// Give them first, because a relative path is decided when it is visited. A name counts as
    /// declared if it is declared directly (unless a generic parameter shadows it), named by
    /// `self::` or `super::` without leaving the analyzed module, or imported with an absolute
    /// path. `use super::X` and `use crate::X` are leaked.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemStruct = parse_quote!(struct S { a: Local, b: Outside });
    /// let mut leaker = Leaker::new();
    /// let declared = [item.ident.clone(), parse_quote!(Local)]; // the module declares them
    /// leaker
    ///     .enter_mod("my_module", declared, |view| view.visit_item(&Item::Struct(item.clone())));
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // only `Outside`
    /// ```
    pub fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_mod(self, hint, declared, f)
    }

    /// Analyzes types that use the given generics: runs `f` on a [`View`] in which `generics` are
    /// declared, and returns the top-level [`ContextId`] and the result of `f`.
    ///
    /// The generics themselves are not analyzed: visit their bounds in `f` with
    /// [`View::visit_type_param_bound()`] and [`View::visit_where_clause()`]. `Self` is not
    /// declared, so naming it is an error ([`Reason::SelfWithoutMeaning`]). Scopes with
    /// equal contexts return equal ids ([`ContextId`]). The errors of the scope and of the scopes
    /// in it are recorded once, in the order found ([recovery](Leaker#errors-and-recovery)).
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let mut leaker = Leaker::new();
    /// leaker.enter_generics(&parse_quote!(<T>), |view| {
    ///     view.visit_type(&parse_quote!(Rel<T>));
    ///     // A relative trait cannot be leaked: an error of the inner scope, the outer one goes on.
    ///     view.enter_generics(&parse_quote!(<U: Relative>), |view| {
    ///         view.visit_type_param_bound(&parse_quote!(Relative));
    ///     });
    /// });
    /// let result = leaker.result();
    /// assert_eq!(result.errors().len(), 1);
    /// assert_eq!(result.into_referrer().iter().count(), 1);
    /// ```
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_generics(self, generics, f)
    }

    /// Analyzes parts of a struct, an enum or a union: runs `f` on a [`View`] inside the type
    /// `ident` with `generics`.
    ///
    /// `Self` is `ident` with the parameters as arguments ([`SelfRequirement::Type`]), and
    /// `allow_self` is as for [`intern_struct()`](Leaker::intern_struct). Visit the fields in `f`.
    /// Otherwise it is as [`enter_generics()`](Leaker::enter_generics), but `generics` may name
    /// `Self`.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: Wrap<T>, b: Skipped });
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .enter_type(&item.ident, &item.generics, true, |view| {
    ///         view.visit_type(&item.fields.iter().next().unwrap().ty); // only the first field
    ///     });
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Wrap<T>`
    /// ```
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_type(self, ident, generics, allow_self, f)
    }

    /// Analyzes parts of an impl: runs `f` on a [`View`] inside an impl with `generics` for
    /// `self_ty`.
    ///
    /// `Self` is `self_ty` ([`SelfRequirement::Type`]), and `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct). Visit the fragments of the impl in `f`.
    /// Otherwise it is as [`enter_generics()`](Leaker::enter_generics).
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// // Analyze only the arguments of the trait of an impl.
    /// let item: ItemImpl = parse_quote!(impl<T> ::path::MyTrait<Rel<T>> for MyType<T, Rel2> {});
    /// let (_, trait_path, _) = item.trait_.as_ref().unwrap();
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .enter_impl(&item.generics, &item.self_ty, true, |view| {
    ///         for segment in &trait_path.segments {
    ///             view.visit_path_arguments(&segment.arguments);
    ///         }
    ///     });
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Rel<T>`, not `Rel2`
    /// ```
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_impl(self, generics, self_ty, allow_self, f)
    }

    /// Analyzes parts of a trait: runs `f` on a [`View`] inside the trait `ident` with `generics`.
    ///
    /// `Self` is the implementor ([`SelfRequirement::Bound`]), and `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct). Otherwise it is as
    /// [`enter_generics()`](Leaker::enter_generics), but the items of the trait are not known, so a
    /// projection of one of its generic associated types gets no outlives bounds: use
    /// [`intern_trait()`](Leaker::intern_trait), or write the bound.
    ///
    /// ```
    /// # use {syn::{visit::Visit, *}, type_leak::*};
    /// // Analyze one method of a trait: the signature, with the generics of the trait.
    /// let item: ItemTrait = parse_quote! {
    ///     trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self); }
    /// };
    /// let TraitItem::Fn(method) = &item.items[0] else { unreachable!() };
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .enter_trait(&item.ident, &item.generics, true, |view| {
    ///         view.visit_signature(&method.sig)
    ///     });
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Wrap<T>`
    /// ```
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_trait(self, ident, generics, allow_self, f)
    }

    /// Ends the analysis and returns the [`Referrer`] with the errors found.
    ///
    /// The Referrer is deterministic: required entries come first, then optional ones
    /// ([`Referrer::must()`]), each sorted by context (in order of first appearance) and tokens. So
    /// optional entries never change the indices of required ones. It calls
    /// [`Leaker::reduce_roots()`] if needed.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemTrait = parse_quote! {
    ///     trait Tr { fn f(&self) -> ::foo::X; fn g(&self) -> Rel; }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
    /// // `Rel` is required. `::foo::X` is optional: it names another crate.
    /// assert_eq!(referrer.iter().count(), 2);
    /// assert!(referrer.must(0) && !referrer.must(1));
    /// ```
    pub fn result(mut self) -> LeakerResult {
        let errors = std::mem::take(&mut self.errors);
        LeakerResult {
            referrer: self.referrer(),
            errors,
        }
    }

    fn referrer(mut self) -> Referrer {
        self.reduce_roots();
        let (required, optional) = self.reachable_types.take().unwrap_or_default();
        let contexts = self.contexts;
        let context_of = |id: ContextId| contexts.get(id).clone();
        let mut context_map: BTreeMap<ContextId, Context> = self
            .top_levels
            .iter()
            .map(|&id| (id, context_of(id)))
            .collect();
        // The parts are looked up by their contexts too.
        for item in &self.parts {
            let anchor = item.within.iter().map(|(_, id)| *id);
            for id in item.context.into_iter().chain(anchor) {
                context_map.entry(id).or_insert_with(|| context_of(id));
            }
        }
        let mut entries = Vec::new();
        let keys = required.into_iter().map(|key| (key, true));
        for (key, must) in keys.chain(optional.into_iter().map(|key| (key, false))) {
            let context = context_map
                .entry(key.context)
                .or_insert_with(|| context_of(key.context));
            let crates = external_crates(&key.ty.ty, context);
            entries.push((key.context, key.ty.ty, must, crates));
        }
        // Keep the modules the contexts depend on, and their parents, for nested lookups.
        let mut modules = BTreeMap::new();
        let mut pending: Vec<ContextId> = Vec::new();
        for (id, context) in &context_map {
            if context.is_module_scope() {
                pending.push(*id);
            }
            pending.extend(context.module());
        }
        while let Some(module) = pending.pop() {
            if let Some(info) = contexts.module(module) {
                if modules.insert(module, info.clone()).is_none() {
                    context_map
                        .entry(module)
                        .or_insert_with(|| context_of(module));
                    pending.extend(info.parent);
                }
            }
        }
        Referrer::new(entries, context_map, modules, self.parts)
    }

    /// Chooses the fewest entries from which every leaked type can be built.
    ///
    /// [`Leaker::result()`] calls it, so you rarely need to call it yourself. `Rel` alone covers
    /// `Vec<(Rel,)>` and `Box<(Rel,)>`. A reference with a lifetime around an entry (`&'a Rel`), a
    /// path with a lifetime argument or a bare identifier argument (`Wrap<SIZE>`), and a type with
    /// a type in an expression (`[u8; size_of::<Big>()]`) are entries as a whole, never replaced by
    /// their parts (so is a trait object: [Trait objects](crate::Leaker#trait-objects)).
    pub fn reduce_roots(&mut self) {
        if self.reachable_types.is_some() {
            return;
        }
        let required = reduce(&self.types, &HashSet::new());
        // The required entries are cut for free in the graph of the optional ones.
        let blocked: HashSet<NodeKey> = required
            .iter()
            .filter_map(|key| {
                let context = self
                    .optional_contexts
                    .find(self.contexts.get(key.context))?;
                Some(NodeKey {
                    context,
                    ty: key.ty.clone(),
                })
            })
            .collect();
        let mut optional: Vec<NodeKey> = reduce(&self.optional_types, &blocked)
            .into_iter()
            .map(|key| NodeKey {
                context: self
                    .contexts
                    .intern(self.optional_contexts.get(key.context).clone()),
                ty: key.ty,
            })
            .collect();
        optional.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
        self.reachable_types = Some((required, optional));
    }
}

/// What a [`Leaker`] found: the [`Referrer`], and the errors that remain.
///
/// [`Leaker::result()`] returns it with every error of the analysis
/// ([recovery](Leaker#errors-and-recovery)). The `allow*` methods drop the errors you accept. They
/// never change the Referrer, which has every rejected part copied as written either way. Take the
/// Referrer with [`finish()`](LeakerResult::finish) if no error remains, or with
/// [`into_referrer()`](LeakerResult::into_referrer) whatever remains.
///
/// [`ToTokens`] and [`Parse`] pass it to another macro, which can then decide what to allow.
#[derive(Clone, Debug)]
pub struct LeakerResult {
    referrer: Referrer,
    errors: Vec<ErrorItem>,
}

impl LeakerResult {
    /// Drops the remaining errors whose cause equals `cause` (the same kind and tokens, spans
    /// ignored).
    ///
    /// You promise that the part can be written as it is in the copy; if not, the copy fails to
    /// build, in the crate of the copy. The cause may be smaller than the part that is copied: in
    /// `[u8; { let c = |a: u8| a; 3 }]` the error is about `u8`, and the array is copied whole. A
    /// part that the signature of a method made ([`Reason::UnconstrainedLifetime`], an elided
    /// output) is copied as written in the arguments of a signature with the same fault only.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemImpl = parse_quote! {
    ///     impl ::p::S {
    ///         fn f(_: ::p::W<'_, dyn RelTr>) {}  // an elided lifetime, and `RelTr` is relative
    ///         fn g(_: ::p::X<dyn RelTr>) {}      // fine: `dyn RelTr` is leaked
    ///     }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_impl(&item, true);
    /// let mut result = leaker.result();
    /// // The error is about the trait, which is smaller than the type `W<'_, dyn RelTr>` it is in.
    /// assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));
    ///
    /// // `f` is copied as written either way, and only `g` leaks `dyn RelTr`.
    /// result.allow(Cause::Trait(parse_quote!(RelTr)));
    /// assert_eq!(result.finish().unwrap().required().count(), 1);
    /// ```
    ///
    /// Allowing a cause does not make these compile:
    ///
    /// - `Self::A`, or `Self` as a type, in a bound or where predicate when a bound in its context
    ///   names a leaked type ([`Reason::UnprovableBound`]): E0277, until the type of the bound is
    ///   written with an absolute path or allowed with
    ///   [`allowed_paths`](LeakerConfig::allowed_paths). Only the cause is copied as written.
    /// - A trait with a generic whose bound names an associated type that another bound fixes (`T:
    ///   Iterator<Item = (R0, u8)> + Fn(<T as Iterator>::Item) -> u8`), and a method predicate that
    ///   names `Self::A`: rustc cannot implement it with a generic impl at all (E0277). See [Bounds
    ///   that name leaked types](Leaker#bounds-that-name-leaked-types).
    /// - [`Visitor::with_self()`](crate::Visitor::with_self) replaces `Self` inside a part copied as
    ///   written too, which can change the elision of a lifetime (E0106): with `allow_self =
    ///   false`, `struct S<'a>(PhantomData<&'a ()>, fn(::core::slice::Iter<u8>, *mut Self) ->
    ///   &u32)` becomes a function pointer with two input lifetimes.
    pub fn allow(&mut self, cause: Cause) -> &mut Self {
        self.errors.retain(|error| error.cause != cause);
        self
    }

    /// Drops the remaining errors whose cause is a path (a type, a trait, a macro or a value) that
    /// starts with `std`, `core` or `alloc`, such as `core::fmt::Debug`.
    ///
    /// A cause with no path (a reference, a `dyn`, an array, a tuple, a lifetime) is never dropped.
    /// The promise is that of [`allow()`](LeakerResult::allow).
    pub fn allow_std_crates(&mut self) -> &mut Self {
        self.allow_paths(&std_crate_paths())
    }

    /// Drops the errors whose cause is a path that starts with one of `paths`, as
    /// [`LeakerConfig::allowed_paths`] does.
    fn allow_paths(&mut self, paths: &[Path]) -> &mut Self {
        let allowed = AllowedPaths::new(paths);
        self.errors.retain(|error| {
            let path = error.cause.as_path();
            !path.map_or(false, |path| allowed.is_allowed_path(path))
        });
        self
    }

    /// Returns the errors that remain, in the order found.
    pub fn errors(&self) -> &[ErrorItem] {
        &self.errors
    }

    /// Returns the Referrer, whatever errors remain: a part that conflicts is copied as written.
    pub fn into_referrer(self) -> Referrer {
        self.referrer
    }

    /// Returns the Referrer if no error remains, else the errors that remain.
    pub fn finish(self) -> Result<Referrer, Vec<ErrorItem>> {
        if self.errors.is_empty() {
            Ok(self.referrer)
        } else {
            Err(self.errors)
        }
    }
}

/// Writes the [`LeakerResult`] as tokens: the [`Referrer`] as it writes itself, then one brace group
/// for each remaining error ([`ErrorItem`]), in the order of [`errors()`](LeakerResult::errors).
///
/// A Leaker's macro can emit the result and let the Referrer's macro decide what to allow and report
/// what remains.
///
/// ```
/// # use {syn::*, type_leak::*};
/// # use template_quote::ToTokens;
/// let item: ItemTrait = parse_quote!(trait Tr: core::fmt::Debug { fn c(&self) -> Rel; });
/// let mut leaker = Leaker::new();
/// leaker.intern_trait(&item, true);
/// let tokens = leaker.result().to_token_stream();
///
/// let mut result: LeakerResult = parse2(tokens).unwrap();
/// assert_eq!(result.errors().len(), 1);
/// assert!(result.allow_std_crates().errors().is_empty());
/// ```
impl ToTokens for LeakerResult {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.referrer.to_tokens(tokens);
        for error in &self.errors {
            error.to_tokens(tokens);
        }
    }
}

/// Reads a [`LeakerResult`] written by its [`ToTokens`] implementation: a [`Referrer`], then the
/// brace groups that follow it, one [`ErrorItem`] each.
///
/// The tokens after the last error are not consumed, but a brace group is read as an error. Fails
/// as the [`Referrer`] does (a missing or different format tag), and on an error that does not
/// parse.
impl Parse for LeakerResult {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let referrer = input.parse()?;
        let mut errors = Vec::new();
        while input.peek(token::Brace) {
            errors.push(input.parse()?);
        }
        Ok(LeakerResult { referrer, errors })
    }
}

impl Scopes for Leaker {
    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        let root = spec.declared_generics();
        let mut visitor = AnalyzeVisitor::new(AllowedPaths::new(&self.allowed_paths), &root)
            .reading_values(self.check_values)
            .with_nested_self_entries(self.nested_self_entries);
        let mark = self.contexts.len();
        // The visitor owns the table of contexts while it analyzes, so the swap is the only step
        // before `catch_unwind`: no allocation, no user code. Whatever happens next (a result, or a
        // panic of `f`), the table is swapped back before anything else, and an unfinished
        // analysis is rolled back to `mark`.
        visitor.swap_contexts(&mut self.contexts);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            visitor.start(&spec);
            let result = f(&mut View::new(&mut visitor));
            let (finished, errors) = visitor.finish(&root);
            (finished, errors, result)
        }));
        visitor.swap_contexts(&mut self.contexts);
        let (finished, errors, result) = match outcome {
            Ok(done) => done,
            Err(payload) => {
                self.contexts.truncate(mark);
                resume_unwind(payload)
            }
        };
        let Finished {
            mut effects,
            top_levels,
            module,
            not_optional,
        } = finished;
        drop_optional(&mut effects, &not_optional.into_iter().collect());
        let mut analysis = Analysis::default();
        analysis.apply(&mut self.contexts, &effects, None);
        // Optional entries never fail the call: a type that cannot be leaked is copied as written.
        analysis.apply_optional(&mut self.optional_contexts, &effects, None);
        for ident in analysis.primitives {
            if !self.assumed_primitives.contains(&ident) {
                self.assumed_primitives.push(ident);
            }
        }
        for part in analysis.parts {
            if !self.parts.contains(&part) {
                self.parts.push(part);
            }
        }
        if !(analysis.types.is_empty() && analysis.optional.is_empty()) {
            self.reachable_types = None;
        }
        self.types.merge(analysis.types);
        self.optional_types.merge(analysis.optional);
        // Shares the context of an entry that uses exactly these generics.
        // `Self` is part of the top-level context only if an entry names it, or has a bound
        // rewritten for it (see `Context::of()`).
        let scope_self = spec
            .declared_self()
            .filter(|declared| names_self(&effects, declared));
        let top_level = module.unwrap_or_else(|| {
            let context = Context::new(root, scope_self).with_gats(spec.declared_gats());
            self.contexts.intern(context)
        });
        self.top_levels.insert(top_level);
        self.top_levels.extend(top_levels);
        // Nothing is recorded if the closure panics: the outcome above resumes the unwind.
        self.errors.extend(errors);
        (top_level, result)
    }
}

/// Min vertex cut from the roots to the must-intern types, by max flow (Edmonds-Karp, node-split).
/// Ties go to the cut closest to the sink, so inner types win.
fn reduce(types: &TypeGraph, blocked: &HashSet<NodeKey>) -> Vec<NodeKey> {
    let mut keys: Vec<&NodeKey> = types.keys().filter(|key| !blocked.contains(*key)).collect();
    keys.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
    let index: HashMap<&NodeKey, usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let n = keys.len();
    let children: Vec<Vec<usize>> = keys
        .iter()
        .map(|key| {
            let children = types.children(key).into_iter();
            children.filter_map(|c| index.get(&c).copied()).collect()
        })
        .collect();
    let flags = |set: &HashSet<NodeKey>| -> Vec<bool> {
        keys.iter().map(|key| set.contains(*key)).collect()
    };
    let (red, root) = (flags(&types.must_intern), flags(&types.roots));
    // A type that must be an entry cannot be forbidden to be one.
    let forbidden: Vec<bool> = flags(&types.forbidden)
        .into_iter()
        .enumerate()
        .map(|(i, forbidden)| forbidden && !red[i] && !needs_parent(keys[i]))
        .collect();
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, children) in children.iter().enumerate() {
        for &child in children {
            parents[child].push(i);
        }
    }
    let mut has_red = red.clone();
    spread(&mut has_red, &parents);

    // A root that must be interned is in every minimum cut, so it is cut without a flow.
    let cut: Vec<bool> = (0..n).map(|i| red[i] && root[i]).collect();
    let mut kept: Vec<NodeKey> = (0..n)
        .filter(|&i| cut[i])
        .map(|i| keys[i].clone())
        .collect();

    // The edges of the network. A parent that its components cannot replace goes to the sink.
    let mut next: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut to_sink = vec![false; n];
    for i in (0..n).filter(|&i| has_red[i] && !cut[i]) {
        to_sink[i] = red[i];
        let whole = needs_parent(keys[i]);
        for &child in children[i].iter().filter(|&&child| has_red[child]) {
            if whole {
                // The parent is interned as a whole, like a must-intern type.
                to_sink[i] = true;
            } else if !cut[child] {
                next[i].push(child);
            }
        }
    }
    let mut reached: Vec<bool> = (0..n).map(|i| root[i] && has_red[i] && !red[i]).collect();
    spread(&mut reached, &next);

    // Components are independent. Solving each alone keeps the quadratic max flow small.
    let mut component: Vec<usize> = (0..n).collect();
    for i in (0..n).filter(|&i| reached[i]) {
        for &child in &next[i] {
            let (a, b) = (find(&mut component, i), find(&mut component, child));
            component[a.max(b)] = a.min(b);
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut local = vec![0; n];
    for i in (0..n).filter(|&i| reached[i]) {
        let members = groups.entry(find(&mut component, i)).or_default();
        local[i] = members.len();
        members.push(i);
    }
    for members in groups.values() {
        // The nodes: `v_in` is `2 * v`, `v_out` is `2 * v + 1`.
        let m = members.len();
        let (source, sink) = (2 * m, 2 * m + 1);
        // Larger than all node capacities together, so never the bottleneck.
        let infinite = m + 1;
        let mut edges = Vec::new();
        for &i in members {
            let (v_in, v_out) = (2 * local[i], 2 * local[i] + 1);
            // The capacity of a type is what cutting it costs: one entry, or none if it cannot be
            // one.
            edges.push((v_in, v_out, if forbidden[i] { infinite } else { 1 }));
            for &child in &next[i] {
                edges.push((v_out, 2 * local[child], infinite));
            }
            if root[i] {
                edges.push((source, v_in, infinite));
            }
            if to_sink[i] {
                edges.push((v_out, sink, infinite));
            }
        }
        let reaches_sink = sink_side(2 * m + 2, &edges, source, sink);
        // The cut closest to the sink: `v_in` cannot reach it, `v_out` can.
        kept.extend(
            members
                .iter()
                .filter(|&&i| {
                    let v_in = 2 * local[i];
                    reaches_sink[v_in + 1] && !reaches_sink[v_in]
                })
                .map(|&i| keys[i].clone()),
        );
    }

    kept.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
    kept
}

/// Marks every node reachable from a marked one.
fn spread(marked: &mut [bool], edges: &[Vec<usize>]) {
    let mut stack: Vec<usize> = (0..marked.len()).filter(|&i| marked[i]).collect();
    while let Some(i) = stack.pop() {
        for &j in &edges[i] {
            if !marked[j] {
                marked[j] = true;
                stack.push(j);
            }
        }
    }
}

/// See [`is_whole_reference()`].
fn needs_parent(parent: &NodeKey) -> bool {
    is_whole_reference(&parent.ty.ty)
}

/// Runs max flow from `source` to `sink` and returns the nodes that can still reach the sink.
fn sink_side(
    nodes: usize,
    edges: &[(usize, usize, usize)],
    source: usize,
    sink: usize,
) -> Vec<bool> {
    let mut network = BTreeGraph::<usize, usize>::default();
    for node in 0..nodes {
        network.insert_node(node).unwrap();
    }
    for (eix, &(from, to, _)) in edges.iter().enumerate() {
        network.insert_edge(eix, [from, to]).unwrap();
    }
    let (_, flows) = edmonds_karp_with_flows(&network, source, sink, |eix| edges[*eix].2);

    // Residual edges: forward with spare capacity, backward to cancel flow.
    let mut into: Vec<Vec<usize>> = vec![Vec::new(); nodes];
    for (eix, &(from, to, capacity)) in edges.iter().enumerate() {
        let flow = flows.get(&eix).copied().unwrap_or(0);
        if flow < capacity {
            into[to].push(from);
        }
        if flow > 0 {
            into[from].push(to);
        }
    }
    let mut reaches_sink = vec![false; nodes];
    reaches_sink[sink] = true;
    spread(&mut reaches_sink, &into);
    reaches_sink
}

/// Union-find: returns the representative of the set of `i`.
fn find(component: &mut [usize], mut i: usize) -> usize {
    while component[i] != i {
        component[i] = component[component[i]];
        i = component[i];
    }
    i
}

/// Returns the sorted names of the external crates (not `core`) that `ty` and `context` name as
/// `::name`.
fn external_crates(ty: &Type, context: &Context) -> Vec<String> {
    use proc_macro2::{Spacing, TokenTree};
    #[derive(Default)]
    struct Finder(BTreeSet<String>);
    impl Finder {
        fn add(&mut self, ident: &Ident) {
            let name = ident_name(ident);
            if name != "core" {
                self.0.insert(name);
            }
        }
        /// Scans macro tokens for a leading `::` followed by a name.
        fn scan(&mut self, tokens: TokenStream) {
            let mut prev: Option<TokenTree> = None;
            let mut iter = tokens.into_iter().peekable();
            while let Some(tt) = iter.next() {
                match &tt {
                    TokenTree::Group(group) => self.scan(group.stream()),
                    TokenTree::Punct(p) if p.as_char() == ':' && p.spacing() == Spacing::Joint => {
                        let after_name = matches!(&prev, Some(TokenTree::Ident(_)))
                            || matches!(&prev, Some(TokenTree::Punct(p)) if p.as_char() == '>');
                        let second =
                            matches!(iter.peek(), Some(TokenTree::Punct(p)) if p.as_char() == ':');
                        if !after_name && second {
                            iter.next();
                            if let Some(TokenTree::Ident(ident)) = iter.peek() {
                                self.add(ident);
                            }
                        }
                    }
                    _ => (),
                }
                prev = Some(tt);
            }
        }
    }
    impl<'ast> Visit<'ast> for Finder {
        fn visit_path(&mut self, i: &'ast Path) {
            if let (Some(_), Some(seg)) = (&i.leading_colon, i.segments.first()) {
                self.add(&seg.ident);
            }
            syn::visit::visit_path(self, i)
        }
        fn visit_macro(&mut self, i: &'ast Macro) {
            self.scan(i.tokens.clone());
            syn::visit::visit_macro(self, i)
        }
    }
    let mut finder = Finder::default();
    finder.visit_type(ty);
    finder.visit_generics(context.generics());
    match context.self_ty() {
        Some(SelfRequirement::Bound(path)) => finder.visit_path(path),
        Some(SelfRequirement::Type(ty)) => finder.visit_type(ty),
        _ => (),
    }
    finder.0.into_iter().collect()
}

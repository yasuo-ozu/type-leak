//! The [`Leaker`], which finds the types depending on the type context.

use crate::allow::Allow;
use crate::analysis::{drop_optional, names_self, Analysis, PartItem};
use crate::analyze::{declared_names, AnalyzeVisitor, Finished, ModuleNames, ScopeSpec, Scopes};
use crate::config::LeakerConfig;
use crate::context::{Context, ContextId, Contexts, Gat, SelfRequirement};
#[cfg(doc)]
use crate::error::Reason;
use crate::error::{Cause, ErrorItem};
use crate::graph::{is_whole_reference, NodeKey, TypeGraph};
use crate::referrer::Referrer;
use crate::util::{ident_name, AllowedPaths};
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
/// Analyze a whole item with an `intern_*` method, or parts of one with an `enter_*` method, then
/// call [`Leaker::result()`]. Keep the returned [`ContextId`] for [`Referrer::visitor()`]. See the
/// [quick start](crate#quick-start).
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
/// An error does not stop the analysis: the part that cannot be interned is copied as written,
/// whole, with nothing in it leaked. The errors are kept once each, in the order found (those of
/// nested scopes in place). If a closure panics, its scope rolls back and records nothing.
///
/// What you allow drops errors, on [`LeakerConfig`] (when the outermost call ends) and on
/// [`LeakerResult`] (after the analysis), and never changes the [`Referrer`]. A path rule on the
/// config also makes the path context-free (no entry); on the result the entries stay.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let item: ItemStruct = parse_quote!(struct S { a: Rel, b: ::p::W<'_, dyn RelTr> });
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&item, true);
/// let mut result = leaker.result();
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
/// The Leaker reads only the item you give it, not the definitions of the paths it names, and
/// rejects what would not compile in the copy. The `help` of an error says what to write instead,
/// and [`LeakerResult::allow()`] accepts one. The sections below name the rustc errors. Without a
/// section:
///
/// - `MyType<'_>`, `impl Trait`, and a relative trait or value in a bound: write an absolute path.
///   `T: ?Sized` is left as written; `Self: Sized` is a relative trait like `Clone`
///   ([`allow_prelude()`](LeakerConfig::allow_prelude) accepts both).
/// - An absolute path of another crate is copied as written, which assumes that both crates
///   resolve it alike ([`LeakerResult::allow_ambiguous_crate_name()`]).
/// - A shadowed primitive name (`type u8 = i32;`): emit [`Leaker::assertions()`].
/// - Values and bodies are not analyzed unless [`LeakerConfig::check_values()`] is on: it rejects a
///   path that is not a type and leaks the types in type positions. Enum discriminants and array
///   lengths are always analyzed.
///
/// ## Outlives bounds
///
/// An entry under `&'a` needs `T: 'a` in its Repeater impl (E0309), and the Leaker records only the
/// bounds you write: write them (`fn f<'a, T: 'a>(x: Rel<'a, T>)`), and give
/// [`Referrer::visitor()`] the whole item. For `Self: 'a` and associated types, see
/// [`Referrer::context_self_outlives()`] and [`Leaker::intern_trait()`].
///
/// ## Binder lifetimes
///
/// If the output of a `for<'x> ..` signature names `'x`, an input must keep `'x` as written, which
/// a leaked type does not (E0581, E0582). So the whole function pointer or trait object is leaked,
/// which is an error in a bound. Keep `'x` in an input: `for<'x> Fn(&'x u8, Rel<'x>) -> &'x u8` is
/// good, `for<'x> Fn(Rel<'x>) -> &'x u8` is an error.
///
/// ## Elided lifetimes
///
/// An elided output lifetime comes from the only input lifetime, and a leaked type adds more
/// (E0106), even if another input has one: rustc counts each input apart. So the whole function
/// pointer or trait object is leaked, which is an error in a bound, in `impl Fn(..)`, under an
/// elided reference, and in a method without a reference receiver. A hidden lifetime (`Ref<u8>`) is
/// not seen: write `Ref<'_, u8>`. Name the lifetimes (`fn g<'x>(x: &'x Rel<T>) -> &'x u8`) or put
/// the trait object in a `Box` (`Box<dyn Fn(&Rel<T>) -> &u8>` is leaked whole, `&dyn Fn(&Rel<T>) ->
/// &u8` is an error).
///
/// ## Trait objects
///
/// A `dyn Trait` without a lifetime bound takes its bound from where it is written, which the
/// Leaker cannot see (E0478). So the reference or path around it is leaked instead, which is an
/// error under an elided lifetime (`&dyn Tr`). Write `&'a dyn Tr` (the whole reference is leaked)
/// or `&'a (dyn Tr + 'a)`, which fails for a trait with a bound of its own.
///
/// ## Method lifetimes
///
/// A method lifetime `'b` that only a leaked type mentions is late-bound in a trait but early-bound
/// in the copy (E0195), if the return type names `'b`. This affects methods of traits and trait
/// impls. Keep `'b` in an input that is not leaked: `fn f<'b>(&'b self, x: Wrap<'b>) -> &'b u8` is
/// good, `fn f<'b>(&self, x: Wrap<'b>) -> &'b u8` is an error. An absolute type of another crate
/// that alone keeps `'b` (`::ext::V<'b>`) is no optional entry: every Referrer copies it as
/// written, so its crate must resolve there.
///
/// ## Bounds that name leaked types
///
/// A leaked type in a bound or where predicate fails (E0277, E0271, E0275, E0284) if its Repeater
/// impl needs a bound that the environment has only in its rewritten form; a type in a field or a
/// signature is fine. It also fails if the bound names `Self`, `Self::A` or an associated type that
/// rustc normalizes, or projects a generic bounded by a trait outside the standard library
/// (supertraits are not read). `struct S<T: Fn(Rel)>(Wrap<T>) where Wrap<T>: Sized;` is an error;
/// without the predicate it is good.
///
/// If another bound of `T` fixes an associated type (`T: Iterator<Item = u8>`), a bound of `T` that
/// names `T` (`T: Fn(Wrap<T>)`) is left out of the context of the entries that use `T` (not of one
/// that names `Self`), because rustc cannot normalize it when it proves the impl. A trait that
/// also has a method predicate naming `Self::A` cannot be implemented generically by rustc at all
/// (E0277). If the leaked type needs the bound that is left out, rustc
/// rejects the Repeater impl (E0277) and the Leaker does not see it: write the type without the
/// bound, or the bound without the generic (`T: Fn(u8)`).
///
/// ## Bare identifiers and consts
///
/// A bare identifier argument (`MyType<MY_CONST>`) may be a type or a const. The Leaker leaks the
/// type that holds it as a whole, which is right for both. Where it cannot, it leaks the identifier
/// alone, which fails for a const (E0573 in the Repeater's impl, E0747 in the copy): in a holder
/// with an elided lifetime or reference or `impl Trait`, in a holder path that a module of
/// [`intern_mod()`](Leaker::intern_mod) declares, and in a trait bound, a supertrait or the trait
/// of an impl. Any other cause is an error with the identifier as its cause (`Self` where denied, a
/// default or a bound that names `Self`, an unconstrained [method lifetime](#method-lifetimes), an
/// expression that cannot be leaked, a name that a module declares beside the identifier in the
/// path of another crate). Write the identifier so that it is not leaked: an absolute path (a const
/// needs braces), a name in [`allow_path()`](LeakerConfig::allow_path), or a name that the module
/// declares.
///
/// ```ignore
/// fn f(x: ::path::MyType<MY_CONST>);                  // good: the path is leaked
/// fn f(x: ::path::MyType<'_, MY_CONST>);              // error: treated as a type
/// fn f(x: ::path::MyType<{ ::my_crate::MY_CONST }>);  // good: not leaked
/// ```
#[derive(Default)]
pub struct Leaker {
    /// What the configuration allows: the paths that are left as written instead of being leaked,
    /// and the errors that are not recorded.
    allow: Allow,
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
    /// Creates a [`Leaker`] that allows nothing, so every relative path is leaked.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`Leaker`] that leaves the paths allowed by `config` as written, and records
    /// only the errors that it does not accept.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allow: config.allow,
            check_values: config.check_values,
            nested_self_entries: config.nested_self_entries,
            ..Self::default()
        }
    }

    /// Returns a compile-time check that the primitive names copied as written (`u8` in `const N:
    /// u8`, or by [`LeakerConfig::allow_primitive()`]) are not shadowed.
    ///
    /// Emit it in the Leaker's crate: it fails to compile if an item shadows such a name (`type
    /// u8 = i32;`), and is empty if none was copied.
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
    /// `Self` is the struct ([`SelfRequirement::Type`]). `allow_self` says what happens to a type
    /// that names `Self`, in every method that takes it.
    ///
    /// `true`: the type is one entry, written for the struct (for a trait: its implementors). But
    /// `Self::A` where `Self` is a type is never leaked whole (E0220, E0223): that is an error
    /// where it must be (`Rel<Self::A>`), as is a bound of the generics of an entry that names it.
    /// Write `<Self as ::path::Trait>::A`. A type that needs no leaking is copied as written
    /// (`Vec<Self::A>`).
    ///
    /// `false`: the type is split into parts, which is an error if it cannot be (`Rel<Self>`,
    /// `Self::Item`), or if the bounds of the generics it uses name `Self`. Use it if the Repeater
    /// impls are for a marker type: [`Visitor::with_self()`](crate::Visitor::with_self) then
    /// replaces `Self` in the copies.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: Wrap<T>, b: Rel<Self> });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&item, true);
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
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

    /// Analyzes an enum like [`intern_struct()`](Leaker::intern_struct).
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

    /// Analyzes a union like [`intern_struct()`](Leaker::intern_struct).
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
    /// [`intern_struct()`](Leaker::intern_struct). Bodies and const values are not analyzed unless
    /// [`LeakerConfig::check_values()`] is on. The trait path must not depend on the context.
    ///
    /// An entry in the self type must not depend on `Self` through a bound of the impl (`impl<T>
    /// S<T> where T: AsRef<Option<Self>>` has the entry `S<T>`), because the copy writes the self
    /// type for `Self`. That is an error, here and in an impl that [`View::visit_item()`] or
    /// [`intern_mod()`](Leaker::intern_mod) visits outside of another item: write the self type
    /// with an absolute path (`::my_crate::S<T>`), or the bound without `Self`. Not affected: an
    /// optional entry, [`enter_impl()`](Leaker::enter_impl), and a bound of an outer scope or of a
    /// method.
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
    /// `Self` is the implementor ([`SelfRequirement::Bound`]), and `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct).
    ///
    /// Unlike [`enter_trait()`](Leaker::enter_trait), it knows the generic associated types (GATs)
    /// of the trait. A projection of a GAT (`Self::G<'x>`) records the outlives bounds of the GAT
    /// (`where Self: 'b`, `T: 'b`, `'a: 'b`) with `'b` replaced by `'x`, which the copy must prove:
    /// in an impl for another type, add `where T: 'x` or `where Self: 'x` (unless the receiver is
    /// `&'x self`) to the method. The GATs of supertraits or of `T::G<'x>` are not seen: write the
    /// bound on the method.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemTrait = parse_quote!(trait Tr { type Out; fn g(&self) -> Wrap<Self::Out>; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
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
    /// Every item in it is a nested item: its generics are local, and `Self` has no meaning. Bodies
    /// and const values are not analyzed unless [`LeakerConfig::check_values()`] is on. The names
    /// declared in the module are copied with it, not leaked (see `declared` in
    /// [`enter_mod()`](Leaker::enter_mod)). Two calls give two modules: rewrite the copy of each
    /// with its own id ([`Referrer::visitor()`]). [`Referrer::module_path()`] gives the path of a
    /// nested module, and [`Referrer::context_module()`] the module of an entry.
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
    /// Module ids come from a counter, so equal calls give equal ids; `hint` is only for debugging
    /// ([`Referrer::module_hint()`]).
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
    /// [`View::visit_type_param_bound()`] and [`View::visit_where_clause()`]. Naming `Self` is an
    /// error ([`Reason::SelfWithoutMeaning`]). Scopes with equal contexts return equal ids.
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
    /// Like [`enter_generics()`](Leaker::enter_generics), but `Self` is the type
    /// ([`SelfRequirement::Type`]) and `generics` may name it. `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct).
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
    /// Like [`enter_type()`](Leaker::enter_type), with `self_ty` as `Self`.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
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
    /// Like [`enter_type()`](Leaker::enter_type), but `Self` is the implementor
    /// ([`SelfRequirement::Bound`]). The GATs of the trait are not known, so a projection of one
    /// gets no outlives bounds: use [`intern_trait()`](Leaker::intern_trait), or write the bound.
    ///
    /// ```
    /// # use {syn::{visit::Visit, *}, type_leak::*};
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
    /// Required entries come first, then optional ones ([`Referrer::must()`]), each sorted by
    /// context (in order of first appearance) and tokens. So optional entries never change the
    /// indices of required ones. It calls [`Leaker::reduce_roots()`] if needed.
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
    /// [`Leaker::result()`] calls it. `&'a Rel`, a path with a lifetime or bare-identifier argument
    /// (`Wrap<SIZE>`), a type with a type in an expression (`[u8; size_of::<Big>()]`) and a trait
    /// object ([Trait objects](crate::Leaker#trait-objects)) are entries as a whole, never replaced
    /// by their parts.
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
/// ([recovery](Leaker#errors-and-recovery)). The `allow*` methods drop the errors you accept, by
/// the rules of the same methods of [`LeakerConfig`], and never change the Referrer (every rejected
/// part is copied as written either way). You promise that the part compiles as written, see
/// [`allow()`](LeakerResult::allow). A path rule drops an error only if every relative path of it
/// ([`ErrorItem::paths`]) is allowed. The [nested errors](ErrorItem::nested) of a dropped error
/// come up in its place and are filtered too: allow until `errors()` is empty.
///
/// A path rule here only approximates the config's, because it judges an analysis made without the
/// rule: it can leave more errors, rarely fewer, and the entries that the config would leave as
/// written stay entries, so a copy may fail to build. Give path rules to the config when you can.
///
/// Take the Referrer with [`finish()`](LeakerResult::finish) if no error remains, or with
/// [`into_referrer()`](LeakerResult::into_referrer) whatever remains. [`ToTokens`] and [`Parse`]
/// pass it to another macro, which can then decide what to allow.
#[derive(Clone, Debug)]
pub struct LeakerResult {
    referrer: Referrer,
    errors: Vec<ErrorItem>,
}

impl LeakerResult {
    /// Drops the errors that the rules in `rules` accept.
    fn drop_accepted(&mut self, rules: impl FnOnce(&mut Allow)) -> &mut Self {
        let mut allow = Allow::default();
        rules(&mut allow);
        allow.filter(&mut self.errors);
        self
    }

    /// Drops the remaining errors whose cause equals `cause` (the same kind and tokens, spans
    /// ignored).
    ///
    /// You promise that the part can be written as it is in the copy; if not, the copy fails to
    /// build. The cause may be smaller than the part that is copied: in `[u8; { let c = |a: u8| a;
    /// 3 }]` the error is about `u8`, and the array is copied whole. A part that a method signature
    /// made (an elided output, [`Reason::UnconstrainedLifetime`]) is copied as written in a
    /// signature with the same fault only.
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
    /// - `Self::A`, or `Self` as a type, in a bound whose context names a leaked type
    ///   ([`Reason::UnprovableBound`]): E0277, until that type is written with an absolute path or
    ///   allowed with [`allow_path()`](LeakerConfig::allow_path).
    /// - A trait with a bound left out of a context and a method predicate that names `Self::A`:
    ///   rustc cannot implement it generically (E0277). See [Bounds that name leaked
    ///   types](Leaker#bounds-that-name-leaked-types).
    /// - [`Visitor::with_self()`](crate::Visitor::with_self) replaces `Self` in a part copied as
    ///   written too (with `allow_self = false`), which can change the elision of a lifetime
    ///   (E0106).
    pub fn allow(&mut self, cause: Cause) -> &mut Self {
        self.drop_accepted(|allow| allow.cause(cause))
    }

    /// Drops the errors whose relative paths all start with `path`, such as `my_mod::Tr`.
    ///
    /// An absolute path is never an error, so `::serde` changes nothing here.
    pub fn allow_path(&mut self, path: Path) -> &mut Self {
        self.drop_accepted(|allow| allow.path(path))
    }

    /// Drops the errors whose relative paths are all primitive type names.
    pub fn allow_primitive(&mut self) -> &mut Self {
        self.drop_accepted(Allow::primitive)
    }

    /// Drops the errors whose relative paths are all names of the standard prelude (`Clone`, `Vec`).
    pub fn allow_prelude(&mut self) -> &mut Self {
        self.drop_accepted(Allow::prelude)
    }

    /// Drops the errors whose relative paths all start with `std`, `core` or `alloc`, such as
    /// `core::fmt::Debug`.
    pub fn allow_std_crates(&mut self) -> &mut Self {
        self.drop_accepted(Allow::std_crates)
    }

    /// Drops the errors whose relative paths all start with `crate`.
    pub fn allow_crate(&mut self) -> &mut Self {
        self.drop_accepted(Allow::crate_root)
    }

    /// Drops the errors found in a value (see [`ErrorItem::in_value`]).
    pub fn allow_values(&mut self) -> &mut Self {
        self.drop_accepted(Allow::values)
    }

    /// Drops the errors found in a default (see [`ErrorItem::in_default`]).
    pub fn allow_default(&mut self) -> &mut Self {
        self.drop_accepted(Allow::defaults)
    }

    /// Chooses whether the optional entries (absolute paths of other crates) are left as written
    /// (`true`, the default) or replaced as a whole through the Repeater (`false`, a strict
    /// Referrer).
    ///
    /// Required entries are always replaced. Crates are matched by name only, so with `true` a
    /// crate of the same name with another version or origin is taken for the same one: use `false`
    /// if the Referrer's crate may have other dependencies than the Leaker's. With `true`, the
    /// types inside an optional entry are still replaced (`Rel` in `::serde::Wrap<(Rel,)>`).
    ///
    /// An absolute type that names the `Self` of a nested struct, enum or union
    /// (`::serde::Wrap<Self>`) is never optional
    /// ([`Visitor::with_nested_self()`](crate::Visitor::with_nested_self)).
    ///
    /// The flag is not serialized, so the Leaker's side writes the impls of all the entries
    /// ([`Referrer::iter()`]), the optional ones too. A serialized [`Referrer`] parses as a
    /// `LeakerResult` with no errors (if no brace group follows it), so a macro that receives one
    /// chooses with this method.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_struct(&parse_quote!(struct S { a: ::foo::X }), true);
    /// let mut result = leaker.result();
    /// let ty: Type = parse_quote!(::foo::X);
    /// let replaced: Type = parse_quote!(Replaced);
    /// let referrer = result.clone().finish().unwrap();
    /// assert_eq!(referrer.expand(context, ty.clone(), |_, _| replaced.clone()), ty);
    /// let referrer = result.allow_ambiguous_crate_name(false).clone().finish().unwrap();
    /// assert_eq!(referrer.expand(context, ty.clone(), |_, _| replaced.clone()), replaced);
    /// ```
    pub fn allow_ambiguous_crate_name(&mut self, allow: bool) -> &mut Self {
        self.referrer.allow_ambiguous_crate_name = allow;
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

/// Writes the [`LeakerResult`] as tokens: the [`Referrer`], then one brace group for each remaining
/// error ([`ErrorItem`]), in the order of [`errors()`](LeakerResult::errors).
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

/// Reads a [`LeakerResult`] written by its [`ToTokens`] implementation: a [`Referrer`], then one
/// [`ErrorItem`] for each brace group that follows.
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
        let mut visitor = AnalyzeVisitor::new(AllowedPaths::new(self.allow.paths()), &root)
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
        // Nothing is recorded if the closure panics: the outcome above resumes the unwind. The errors
        // that the configuration accepts are dropped here, as `LeakerResult` drops them later.
        let allow = &self.allow;
        let mut errors = errors;
        allow.filter(&mut errors);
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

//! The [`Leaker`], which finds the types depending on the type context.

use crate::analysis::{drop_optional, names_self, Analysis};
use crate::analyze::{
    declared_names, AnalyzeVisitor, Finished, ModuleNames, Res, ScopeSpec, Scopes,
};
use crate::config::LeakerConfig;
use crate::context::{Context, ContextId, Contexts, Gat, SelfRequirement};
#[cfg(doc)]
use crate::error::NotInternableError;
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
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// Finds the types of an item that depend on where they are written, and collects them in a
/// [`Referrer`].
///
/// This is the first step of copying an item into another crate. Analyze a whole item with an
/// `intern_*` method, or only some fragments of it in a scope (`enter_*`). Then call
/// [`Leaker::finish()`] to get the [`Referrer`]. Each analysis method returns the [`ContextId`] of
/// what it analyzed (its top-level context), or fails with a [`NotInternableError`] that says
/// which paths must be absolute. See the [quick start](crate#quick-start).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let item: ItemTrait = parse_quote! {
///     trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self) -> Rel; }
/// };
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_trait(&item, true).unwrap();
/// let referrer = leaker.finish();
/// // `Wrap<T>` and `Rel` depend on where they are written, so each needs a Repeater impl.
/// assert_eq!(referrer.required().count(), 2);
/// // Keep `top_level` to rewrite a copy of the trait with `Referrer::visitor()`.
/// assert_eq!(referrer.context_generics(top_level).params.len(), 1);
/// ```
#[derive(Default)]
pub struct Leaker {
    /// Paths that are left as written instead of being leaked, such as `::serde`.
    ///
    /// A path matches if it starts with one of these, and both have a leading `::` or neither.
    /// The generic arguments of a matching path are still analyzed. You promise that an allowed
    /// absolute path names the same crate on both sides ([External
    /// crates](crate::rules#external-crates)). [`LeakerConfig`] has ready-made lists.
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
    /// The top-level contexts of successful scopes, which the [`Referrer`] keeps without an entry
    /// too.
    top_levels: BTreeSet<ContextId>,
    assumed_primitives: Vec<Ident>,
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
            ..Self::default()
        }
    }

    /// Returns a compile-time check that primitive type names are not shadowed.
    ///
    /// The Leaker copies primitive names as they are, such as `u8` in `const N: u8` or a name
    /// allowed by [`LeakerConfig::allow_primitive()`]. That is wrong if an item shadows the name
    /// (`type u8 = i32;`). Emit the result in the Leaker's type context: it fails to compile
    /// ("mismatched types") if a name is shadowed, and is empty if no primitive name was copied.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S<const N: u8> { a: Rel }), true).unwrap();
    /// // `u8` is copied as written, so `assertions()` checks that `u8` is the primitive type.
    /// // Write the tokens next to the item, in the crate that defines it.
    /// let assertions = leaker.assertions();
    /// assert!(assertions.to_string().contains("__type_leak_assert_u8_is_primitive"));
    ///
    /// // No primitive name was copied here: there is nothing to check.
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S { a: Rel }), true).unwrap();
    /// assert!(leaker.assertions().is_empty());
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
    /// ([`SelfRequirement::Type`]). With `allow_self`, a type that names `Self` is one entry,
    /// written for the struct. Without it, such a type is split into parts, and that is an error
    /// where it cannot be ([Self rules](crate::rules#self-rules)). This is [`Leaker::enter_type()`] with
    /// the item visited.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: Wrap<T>, b: Rel<Self> });
    ///
    /// // `Rel<Self>` is an entry, which the Repeater impl writes for `S<T>`.
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_struct(&item, true).unwrap();
    /// let referrer = leaker.finish();
    /// assert_eq!(referrer.iter().count(), 2);
    /// assert!(matches!(referrer.context_self(top_level), Some(SelfRequirement::Type(_))));
    ///
    /// // Without `allow_self`, `Rel<Self>` is split into parts, and `Rel` cannot be leaked alone.
    /// let err = Leaker::new().intern_struct(&item, false).unwrap_err();
    /// assert_eq!(err.to_string(), "use absolute path: 'Rel < Self >'");
    /// ```
    pub fn intern_struct(&mut self, item: &ItemStruct, allow_self: bool) -> Res<ContextId> {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_struct(view, item)
        })
        .map(|(context, ())| context)
    }

    /// Analyzes an enum and returns its top-level context.
    ///
    /// It is [`Leaker::intern_struct()`] for an enum, with the same `Self` and `allow_self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemEnum = parse_quote!(enum E<T> { A(Wrap<T>), B { next: Option<Rel2<Self>> } });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_enum(&item, true).unwrap();
    /// // `Option` is a path of the prelude, which is not allowed here: it is leaked too.
    /// assert_eq!(leaker.finish().iter().count(), 2);
    /// ```
    pub fn intern_enum(&mut self, item: &ItemEnum, allow_self: bool) -> Res<ContextId> {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_enum(view, item)
        })
        .map(|(context, ())| context)
    }

    /// Analyzes a union and returns its top-level context.
    ///
    /// It is [`Leaker::intern_struct()`] for a union, with the same `Self` and `allow_self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemUnion = parse_quote!(union U<T: ::core::marker::Copy> { a: Wrap<T>, b: Rel });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_union(&item, true).unwrap();
    /// assert_eq!(leaker.finish().iter().count(), 2);
    ///
    /// // A relative trait in a bound cannot be leaked: write it in full, or allow it
    /// // (`LeakerConfig::allow_std_prelude()`).
    /// let item: ItemUnion = parse_quote!(union U<T: Copy> { a: Wrap<T> });
    /// assert!(Leaker::new().intern_union(&item, true).is_err());
    /// ```
    pub fn intern_union(&mut self, item: &ItemUnion, allow_self: bool) -> Res<ContextId> {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_union(view, item)
        })
        .map(|(context, ())| context)
    }

    /// Analyzes an impl and returns its top-level context.
    ///
    /// The generics of the item are the generics of the analysis, and `Self` is the self type
    /// ([`SelfRequirement::Type`]). It analyzes the generics, the trait and its arguments, the self
    /// type, and the signatures, associated types and consts. It skips function bodies and const
    /// values.
    ///
    /// The trait path must not depend on the context. An entry in the self type must not depend on
    /// `Self` through a bound of the generics ([Self rules](crate::rules#self-rules)). `allow_self` is as
    /// for [`Leaker::intern_struct()`]. This is [`Leaker::enter_impl()`] with the item visited.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemImpl = parse_quote! {
    ///     impl<T> ::path::MyTrait<Rel<T>> for ::path::Host<T> {
    ///         type Out = Wrap<T>;
    ///         fn f(&self) -> Rel2 { Skipped::new() } // the body is not analyzed
    ///         fn g(&self) -> Rel3<Self> { todo!() }
    ///     }
    /// };
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_impl(&item, true).unwrap();
    /// let referrer = leaker.finish();
    /// // `Rel<T>`, `Wrap<T>`, `Rel2` and `Rel3<Self>` (`::path::Host<T>` names another crate: it is
    /// // optional).
    /// assert_eq!(referrer.required().count(), 4);
    /// // `Rel3<Self>` names `Self`, which is the self type of the impl, as written.
    /// match referrer.context_self(top_level) {
    ///     Some(SelfRequirement::Type(ty)) => assert_eq!(*ty, parse_quote!(::path::Host<T>)),
    ///     _ => unreachable!(),
    /// }
    /// ```
    pub fn intern_impl(&mut self, item: &ItemImpl, allow_self: bool) -> Res<ContextId> {
        // The scope checks the path of the trait, and the walk analyzes its arguments.
        let trait_path = item.trait_.as_ref().map(|(_, path, _)| path);
        self.enter_impl_of(
            &item.generics,
            &item.self_ty,
            trait_path,
            allow_self,
            |view| view.walk_item_impl(item),
        )
        .map(|(context, ())| context)
    }

    /// Analyzes a trait and returns its top-level context.
    ///
    /// The generics of the item are the generics of the analysis, and `Self` is the implementor
    /// ([`SelfRequirement::Bound`]). An entry that names `Self` is written for the implementors of
    /// the trait. `allow_self` is as for [`Leaker::intern_struct()`].
    ///
    /// This is [`Leaker::enter_trait()`] with the item visited, and with its generic associated
    /// types known: a projection that names one that has outlives bounds
    /// (`type G<'b> where Self: 'b;`) requires them ([Outlives bounds](crate::rules#outlives-bounds)).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote! {
    ///     trait Tr<T> {
    ///         type Out;
    ///         fn f(&self, t: T) -> Wrap<T>;
    ///         fn g(&self) -> Wrap2<Self::Out>;
    ///     }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&item, true).unwrap();
    /// let referrer = leaker.finish();
    /// // `Wrap2<Self::Out>` names `Self`, so its impl is for the implementors of the trait.
    /// let ix = referrer.iter().position(|ty| *ty == parse_quote!(Wrap2<Self::Out>)).unwrap();
    /// let context = referrer.context(ix);
    /// assert!(matches!(referrer.context_self(context), Some(SelfRequirement::Bound(_))));
    ///
    /// // A relative trait in a bound cannot be leaked: the error says to write it in full.
    /// let item: ItemTrait = parse_quote!(trait Tr: Clone {});
    /// let err = Leaker::new().intern_trait(&item, true).unwrap_err();
    /// assert_eq!(err.to_string(), "use absolute path: 'Clone'");
    /// ```
    pub fn intern_trait(&mut self, item: &ItemTrait, allow_self: bool) -> Res<ContextId> {
        // The scope knows the generic associated types of the trait: what they require is part of
        // the context of an entry that names them.
        let gats = Gat::of_items(&item.items);
        self.enter_trait_of(&item.ident, &item.generics, gats, allow_self, |view| {
            syn::visit::visit_item_trait(view, item)
        })
        .map(|(context, ())| context)
    }

    /// Analyzes a module and returns its [`ContextId`]. A module has no generics, and `Self` has no
    /// meaning in it.
    ///
    /// Every item in it is a nested item, so its generics are local. Function bodies and const
    /// values are not analyzed, and a `mod m;` has nothing to analyze. The names declared in the
    /// module are copied with it, so they are not leaked. Nested modules get ids as in
    /// [`Leaker::enter_mod()`], and their paths are recorded ([`Referrer::module_path()`]). See
    /// [Modules](crate::rules#modules).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemMod = parse_quote! {
    ///     mod m {
    ///         struct Local;
    ///         struct S { a: Local, b: Outside }
    ///         mod inner { struct T { c: Outside2 } }
    ///     }
    /// };
    /// let mut leaker = Leaker::new();
    /// let module = leaker.intern_mod(&item).unwrap();
    /// let referrer = leaker.finish();
    /// // `Local` is declared in the module, so it is copied with it. `Outside` and `Outside2` are
    /// // not: their relative paths mean something only in the module they are written in.
    /// assert_eq!(referrer.iter().count(), 2);
    /// let path = |id| -> Vec<String> {
    ///     referrer.module_path(id).unwrap().iter().map(|ident| ident.to_string()).collect()
    /// };
    /// // The path of a module is relative to the one analyzed: `[]` for `m`, `[inner]` for the other.
    /// assert!(path(module).is_empty());
    /// let modules: Vec<_> = (0..2).map(|ix| referrer.context_module(referrer.context(ix)).unwrap()).collect();
    /// assert_eq!(path(modules[0]), Vec::<String>::new());
    /// assert_eq!(path(modules[1]), ["inner"]);
    /// // The Repeater impl of an entry is written in its module (`RepeaterImpl::module`).
    /// ```
    pub fn intern_mod(&mut self, item: &ItemMod) -> Res<ContextId> {
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
            .map(|(context, ())| context)
    }

    /// Analyzes part of a module: runs `f` on a [`View`] inside the module.
    ///
    /// Returns the [`ContextId`] of the module, which has no generics, and the result of `f`.
    /// `declared` lists the names declared in the module (give them first): they are copied with
    /// the module, so they are not leaked. `hint` is only for debugging. Module ids come from a
    /// counter, so they are unique, and equal calls give equal ids. `Self` has no meaning directly
    /// in `f`. Errors and rollback are as in [`Leaker::enter_generics()`]. See
    /// [Modules](crate::rules#modules).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S { a: Local, b: Outside });
    /// let mut leaker = Leaker::new();
    /// // The module declares `S` and `Local`; `f` visits the item as one of its items.
    /// let (module, ()) = leaker
    ///     .enter_mod("my_module", [item.ident.clone(), parse_quote!(Local)], |view| {
    ///         view.visit_item(&Item::Struct(item.clone()));
    ///     })
    ///     .unwrap();
    /// let referrer = leaker.finish();
    /// // Only `Outside` is an entry. The hint is for debugging.
    /// assert_eq!(referrer.iter().count(), 1);
    /// assert_eq!(referrer.module_hint(module), Some("my_module"));
    /// ```
    pub fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_mod(self, hint, declared, f)
    }

    /// Analyzes types that use the given generics: runs `f` on a [`View`] in which `generics` are
    /// declared.
    ///
    /// Returns the top-level [`ContextId`] of `generics` and the result of `f`. The generics
    /// themselves are not analyzed: visit their bounds in `f` if you copy them. `Self` is not
    /// declared here, so naming it in `f` or in the generics is an error
    /// ([`NotInternableError::SelfWithoutMeaning`]).
    ///
    /// The scope is all-or-nothing. If the syntax visited in `f` has an error, the scope forgets
    /// everything it found and returns its errors with their spans, and an outer scope can go on.
    /// A panic in `f` rolls the scope back too. See [Scopes](crate::rules#scopes).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .enter_generics(&parse_quote!(<T>), |view| {
    ///         view.visit_type(&parse_quote!(Rel<T>));
    ///         // A relative trait cannot be leaked, so the inner scope fails and the outer one goes on.
    ///         let failed = view.enter_generics(&parse_quote!(<U: Relative>), |view| {
    ///             view.visit_type_param_bound(&parse_quote!(Relative));
    ///         });
    ///         assert!(failed.is_err());
    ///     })
    ///     .unwrap();
    /// assert_eq!(leaker.finish().iter().count(), 1);
    /// ```
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_generics(self, generics, f)
    }

    /// Analyzes parts of a struct, an enum or a union: runs `f` on a [`View`] inside the type
    /// `ident` with `generics`.
    ///
    /// Returns the top-level context and the result of `f`. The fields are not analyzed: visit them
    /// in `f`. `Self` is `ident` with the parameters as arguments ([`SelfRequirement::Type`]).
    /// `allow_self` is as for [`Leaker::intern_struct()`]. Context, errors and rollback are as in
    /// [`Leaker::enter_generics()`], but `generics` may name `Self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// // Analyze two of the fields of a struct.
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: Wrap<T>, b: Rel<Self>, c: Skipped });
    /// let mut leaker = Leaker::new();
    /// let (top_level, ()) = leaker
    ///     .enter_type(&item.ident, &item.generics, true, |view| {
    ///         for field in item.fields.iter().take(2) {
    ///             view.visit_type(&field.ty);
    ///         }
    ///     })
    ///     .unwrap();
    /// let referrer = leaker.finish();
    /// assert_eq!(referrer.iter().count(), 2); // not `Skipped`
    /// assert!(matches!(referrer.context_self(top_level), Some(SelfRequirement::Type(_))));
    /// ```
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_type(self, ident, generics, allow_self, f)
    }

    /// Analyzes parts of an impl: runs `f` on a [`View`] inside an impl with `generics` for
    /// `self_ty`.
    ///
    /// Returns the top-level context and the result of `f`. Visit the fragments of the impl in `f`.
    /// `Self` is `self_ty` ([`SelfRequirement::Type`]). `allow_self` is as for
    /// [`Leaker::intern_struct()`]. Context, errors and rollback are as in
    /// [`Leaker::enter_generics()`]. See [Scopes](crate::rules#scopes).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// // Analyze only the arguments of the trait of an impl.
    /// let item: ItemImpl = parse_quote! {
    ///     impl<T: ::path::Bound> ::path::MyTrait<Rel<T>> for MyType<T, Rel2> {}
    /// };
    /// let (_, trait_path, _) = item.trait_.as_ref().unwrap();
    /// let mut leaker = Leaker::new();
    /// let (_top_level, ()) = leaker
    ///     .enter_impl(&item.generics, &item.self_ty, true, |view| {
    ///         for segment in &trait_path.segments {
    ///             view.visit_path_arguments(&segment.arguments);
    ///         }
    ///     })
    ///     .unwrap();
    /// // Only `Rel<T>` was visited. `Rel2` is in the self type.
    /// assert_eq!(leaker.finish().iter().count(), 1);
    /// ```
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_impl(self, generics, self_ty, allow_self, f)
    }

    /// Analyzes parts of a trait: runs `f` on a [`View`] inside the trait `ident` with `generics`.
    ///
    /// Returns the top-level context and the result of `f`. `Self` is the implementor, built from
    /// `ident` and the parameters ([`SelfRequirement::Bound`]). `allow_self` is as for
    /// [`Leaker::intern_struct()`]. Context, errors and rollback are as in
    /// [`Leaker::enter_generics()`].
    ///
    /// The items of the trait are not known here. So a projection that names a generic associated
    /// type of the trait gets none of its outlives bounds: use [`Leaker::intern_trait()`], or
    /// write the bound.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// use syn::visit::Visit;
    /// // Analyze one method of a trait: the signature, with the generics of the trait.
    /// let item: ItemTrait = parse_quote! {
    ///     trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self) -> Skipped; }
    /// };
    /// let mut leaker = Leaker::new();
    /// let (top_level, ()) = leaker
    ///     .enter_trait(&item.ident, &item.generics, true, |view| {
    ///         if let TraitItem::Fn(method) = &item.items[0] {
    ///             view.visit_signature(&method.sig);
    ///         }
    ///     })
    ///     .unwrap();
    /// let referrer = leaker.finish();
    /// assert_eq!(referrer.iter().count(), 1); // `Wrap<T>`
    /// assert!(matches!(referrer.context_self(top_level), None | Some(SelfRequirement::Bound(_))));
    /// ```
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_trait(self, ident, generics, allow_self, f)
    }

    /// Ends the analysis and returns the [`Referrer`] with the entries found.
    ///
    /// The result is deterministic. Required entries come first, then optional ones
    /// ([`Referrer::must()`]), each sorted by context (in order of first appearance) and tokens. So
    /// optional entries never change the indices of required ones, and contexts used only by
    /// optional entries are numbered last, in the order of the entries. The Referrer keeps the ids
    /// the scopes returned: the top-level contexts, the contexts of the entries, and their modules
    /// with their parents. It calls [`Leaker::reduce_roots()`] if needed.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> ::foo::X; fn g(&self) -> Rel; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&item, true).unwrap();
    /// let referrer = leaker.finish();
    /// // `Rel` is required: every copy replaces it. `::foo::X` is optional: it names another
    /// // crate, which a copy may have or not (`Referrer::allow_ambiguous_crate_name()`).
    /// // The required entries come first.
    /// let entries: Vec<&Type> = referrer.iter().collect();
    /// assert_eq!(entries, [&parse_quote!(Rel), &parse_quote!(::foo::X)]);
    /// assert!(referrer.must(0) && !referrer.must(1));
    /// ```
    pub fn finish(mut self) -> Referrer {
        self.reduce_roots();
        let (required, optional) = self.reachable_types.take().unwrap_or_default();
        let contexts = self.contexts;
        let context_of = |id: ContextId| contexts.get(id).clone();
        let mut context_map: BTreeMap<ContextId, Context> = self
            .top_levels
            .iter()
            .map(|&id| (id, context_of(id)))
            .collect();
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
        Referrer::new(entries, context_map, modules)
    }

    /// Chooses the fewest entries from which every leaked type can be built.
    ///
    /// [`Leaker::finish()`] calls it, so you rarely need to call it yourself. Contexts used only by
    /// optional entries are numbered, after the others, when this runs. See [Which types become
    /// entries](crate::rules#which-types-become-entries).
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

impl Scopes for Leaker {
    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let root = spec.declared_generics();
        let mut visitor = AnalyzeVisitor::new(AllowedPaths::new(&self.allowed_paths), &root);
        let mark = self.contexts.len();
        // The visitor owns the table of contexts while it analyzes, so the swap is the only step
        // before `catch_unwind`: no allocation, no user code. Whatever happens next (a result, an
        // error, or a panic of `f`), the table is swapped back before anything else, and an
        // unfinished analysis is rolled back to `mark`.
        visitor.swap_contexts(&mut self.contexts);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            visitor.start(&spec);
            let result = f(&mut View::new(&mut visitor));
            visitor.finish(&root).map(|finished| (finished, result))
        }));
        visitor.swap_contexts(&mut self.contexts);
        let (finished, result) = match outcome {
            Ok(Ok(done)) => done,
            Ok(Err(error)) => {
                self.contexts.truncate(mark);
                return Err(error);
            }
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
        Ok((top_level, result))
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

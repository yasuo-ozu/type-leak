//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::encode::{complete_generics_args, EncodeError};
use crate::graph::TypeKey;
use crate::requirements::{Requirements, Scope};
use crate::util::{mentions_names, token_names};
use proc_macro2::TokenStream;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};

/// Holds the list of types that need to be encoded, as produced by
/// [`Leaker::finish()`](crate::Leaker::finish()).
///
/// `Referrer` provides methods to iterate over leak types and to expand or transform
/// types using a custom mapping function.
///
/// Each type is an entry, identified by its context (see [`Referrer::context()`]), its tokens, and
/// the local generics it depends on (see [`Referrer::local_generics()`]): the same tokens with
/// different local generics (e.g. `Wrap<B>` where `B` is a `bool` in one method and a `u8` in
/// another) are separate entries, and the one a copy refers to is found by resolving the generics
/// in scope where it appears.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    pub(crate) leak_types: Vec<Type>,
    pub(crate) requirements: Vec<Requirements>,
    pub(crate) contexts: Vec<usize>,
    /// The root generics of each context, including contexts without entries (see
    /// [`Referrer::context_generics()`]).
    pub(crate) context_generics: Vec<Generics>,
    /// The candidates (and their indices) of a type in a context, keyed by [`TypeKey`] (by tokens)
    /// as in the Leaker, so that e.g. a type wrapped in an invisible group is found as well.
    pub(crate) map: HashMap<(usize, TypeKey), Vec<(Requirements, usize)>>,
}

impl Referrer {
    /// Builds a Referrer from its entries (the context, the type and its requirements), in the
    /// order of their indices, and the root generics of the contexts. There is at least one
    /// context, and one for each context of the entries.
    pub(crate) fn new(
        entries: impl IntoIterator<Item = (usize, Type, Requirements)>,
        mut context_generics: Vec<Generics>,
    ) -> Self {
        let mut referrer = Self {
            leak_types: Vec::new(),
            requirements: Vec::new(),
            contexts: Vec::new(),
            context_generics: Vec::new(),
            map: HashMap::new(),
        };
        for (ix, (context, ty, requirements)) in entries.into_iter().enumerate() {
            referrer
                .map
                .entry((context, TypeKey::new(&ty)))
                .or_default()
                .push((requirements.clone(), ix));
            referrer.leak_types.push(ty);
            referrer.requirements.push(requirements);
            referrer.contexts.push(context);
        }
        let n_contexts = referrer
            .contexts
            .iter()
            .max()
            .map_or(1, |context| context + 1);
        if context_generics.len() < n_contexts {
            context_generics.resize(n_contexts, Generics::default());
        }
        referrer.context_generics = context_generics;
        referrer
    }

    /// Returns the index of the type `ty` in `context`, if it is leaked with the requirements it
    /// has with the generics in `scopes`.
    fn find(&self, context: usize, ty: &Type, scopes: &[Scope]) -> Option<usize> {
        let candidates = self.map.get(&(context, TypeKey::new(ty)))?;
        let requirements = Requirements::of(ty, scopes);
        candidates
            .iter()
            .find(|(candidate, _)| candidate == &requirements)
            .map(|(_, ix)| *ix)
    }

    /// Returns `true` if there are no types to be encoded.
    pub fn is_empty(&self) -> bool {
        self.leak_types.is_empty()
    }

    /// Returns an iterator over the types that need to be encoded.
    ///
    /// ```ignore
    /// let args = encode_generics_params_to_ty(&generics.params);
    /// referrer.iter().enumerate().map(|(ix, ty)| quote!{
    ///     impl #crate::TypeRef<#ix, #args> for #marker_ty {
    ///         type Type = #ty;
    ///     }
    /// }).collect::<TokenStream>();
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.leak_types.iter()
    }

    /// Returns the local generics which the `ix`-th type depends on, i.e. generics of a trait
    /// method or a GAT (or of a nested item, or a `for<..>` binder) which are not generics of the
    /// Leaker. It is empty for most types.
    ///
    /// They are in a canonical form, which is the same for declarations with the same meaning: the
    /// parameters are sorted by name (lifetimes first), and the bounds are sorted and moved from
    /// the where clause to the parameter they bound, if any. Defaults are removed.
    ///
    /// A type such as `Wrap<B>` in `fn f<const B: bool>(&self) -> Wrap<B>` (where `Wrap` depends on
    /// the type context) cannot be decomposed, so it is interned together with `B`. The Repeater's
    /// implementation must then declare these generics and constrain them in its trait arguments,
    /// e.g. with [`encode_generics_params_to_ty()`](crate::encode_generics_params_to_ty()). The
    /// Referrer declares the same generics with the same names (as it copies the method's
    /// signature), so it can encode them in the same way. Note that lifetimes must be declared
    /// before the other generics when merging them with the Leaker's generics.
    ///
    /// The bounds and where predicates of the local generics are kept as they are, so they may
    /// mention `Self` (e.g. `fn f<T: PartialEq<Self>>(&self)`), which refers to the implementor of
    /// the Leaker trait. The Repeater's implementation should therefore be for the implementor
    /// (e.g. `impl<.., S: ?Sized> Repeater<..> for S where S: MyTrait<..>`), where `Self` has the
    /// same meaning; in an implementation for another type (e.g. a marker type), `Self` would
    /// refer to that type instead.
    ///
    ///
    /// ```ignore
    /// // In the Leaker (`MyTrait` is the Leaker trait with `generics`)
    /// for (ix, (ty, locals)) in referrer.iter_with_local_generics().enumerate() {
    ///     let args = encode_generics_params_to_ty(&generics.params);
    ///     let local_args = encode_generics_params_to_ty(&locals.params);
    ///     let local_where = locals.where_clause.iter().flat_map(|wc| wc.predicates.iter());
    ///     quote! {
    ///         impl<#generics_params, #locals_params, S: ?Sized>
    ///             Repeater<(#args, #local_args), #ID, #ix> for S
    ///         where
    ///             S: MyTrait<#generics_args>,
    ///             #(#local_where,)*
    ///         {
    ///             type Type = #ty;
    ///         }
    ///     }
    /// }
    ///
    /// // In the Referrer (in the implementation of `MyTrait<#trait_args>`)
    /// let args = encode_generics_args_to_ty_with(&trait_args, Some(&generics.params), |expr, _| {
    ///     parse_quote!([(); #expr])
    /// })?;
    /// let local_args = encode_generics_params_to_ty(&referrer.local_generics(ix).params);
    /// parse_quote!(<Self as Repeater<(#args, #local_args), #ID, #ix>>::Type)
    /// ```
    pub fn local_generics(&self, ix: usize) -> &Generics {
        self.requirements[ix].generics()
    }

    /// Returns the context of the `ix`-th type: the number of the
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()) call it was found by (counting the
    /// successful calls from 0), or its constructor ([`Leaker::from_struct()`](crate::Leaker::from_struct())
    /// etc.), which is the call 0. A type found in several contexts has an entry for each, since
    /// the generics of the Leaker may differ.
    pub fn context(&self, ix: usize) -> usize {
        self.contexts[ix]
    }

    /// Returns the number of contexts (at least 1), including those without entries (see
    /// [`Referrer::context()`]).
    pub fn n_contexts(&self) -> usize {
        self.context_generics.len()
    }

    /// Returns the root generics of `context`: the generics of the Leaker which the types of the
    /// context can refer to, in a form to be declared by the Repeater's implementation (with the
    /// defaults removed). The Repeater's implementation of an entry in `context` must declare
    /// them (and the entry's [`Referrer::local_generics()`]) and encode exactly their parameters
    /// in its trait arguments.
    ///
    /// For the context of an [`intern_with()`](crate::Leaker::intern_with()) call, they are the
    /// `generics` passed to it. A type parameter with a default has a context of its own (see
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with())), whose root generics are the
    /// parameters before it, and the where predicates which do not mention the others. The
    /// Repeater's implementation of an entry in such a context must not bound `Self` by the
    /// Leaker trait (as in `impl<.., S: ?Sized> Repeater<..> for S where S: MyTrait<..>`), since
    /// the arguments of the later parameters are unknown there: the entry is used while computing
    /// them (see [`Referrer::complete_args()`]).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> U; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    /// // The context 0 is the trait, and 1 is the default of `U`.
    /// assert_eq!(referrer.n_contexts(), 2);
    /// let generics = referrer.context_generics(0);
    /// assert_eq!(generics.params.len(), 2);
    /// let generics = referrer.context_generics(1);
    /// assert_eq!(generics.params.len(), 1);
    /// ```
    pub fn context_generics(&self, context: usize) -> &Generics {
        &self.context_generics[context]
    }

    /// Returns `true` if a leaked type, or the local generics (the bounds and the where
    /// predicates) of an entry, refers to `Self` (e.g. `Self`, `Self::Item`, `Rel<Self>`,
    /// `&Self`, or `<Self as ::p::Tr>::X`, but not `::p::Self`).
    ///
    /// Then the Repeater must be implemented for the implementor of the Leaker trait, where
    /// `Self` has the same meaning, and not for a separate marker type (see
    /// [`Referrer::iter_with_self()`] otherwise). No leaked type mentions `Self` with
    /// [`Leaker::self_ty_can_be_interned`](crate::Leaker::self_ty_can_be_interned) unset (it is
    /// an error), but the bounds of local generics are not checked, so they still may.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> Rel<Self>; });
    /// assert!(Leaker::from_trait(&tr).unwrap().finish().has_self());
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> Rel; });
    /// assert!(!Leaker::from_trait(&tr).unwrap().finish().has_self());
    /// ```
    pub fn has_self(&self) -> bool {
        struct Finder(bool);
        impl<'ast> Visit<'ast> for Finder {
            fn visit_path(&mut self, i: &'ast Path) {
                if i.leading_colon.is_none()
                    && i.segments.first().map_or(false, |s| s.ident == "Self")
                {
                    self.0 = true;
                }
                syn::visit::visit_path(self, i)
            }
            fn visit_macro(&mut self, i: &'ast Macro) {
                // The tokens of a macro are not parsed, so they are scanned for the name.
                let mut names = HashSet::new();
                token_names(i.tokens.clone(), &mut names);
                if names.contains("Self") {
                    self.0 = true;
                }
                syn::visit::visit_macro(self, i)
            }
        }
        let mut finder = Finder(false);
        for ty in &self.leak_types {
            finder.visit_type(ty);
        }
        for requirements in &self.requirements {
            finder.visit_generics(requirements.generics());
        }
        finder.0
    }

    /// Like [`Referrer::iter_with_local_generics()`], with `Self` replaced by `self_ty` in the
    /// types and the local generics (see [`Referrer::has_self()`]), for a Repeater implemented
    /// for another type than the implementor of the Leaker trait (e.g. a marker type generic
    /// over `__S`), where `Self` would refer to that type instead.
    ///
    /// `Self` alone becomes `self_ty`, `Self::A::B` becomes `<self_ty>::A::B`, and the self type
    /// of a qualified path (e.g. `<Self as ::p::Tr>::X`) is replaced as well. In the tokens of a
    /// macro, the identifier `Self` is replaced by the tokens of `self_ty`. Only the copies
    /// returned are rewritten: [`Referrer::into_visitor()`] and [`Referrer::expand()`] still find
    /// the types in the code of the Leaker's trait, which is written with `Self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> Rel<Self::Item>; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    /// assert!(referrer.has_self());
    /// let marker: Type = parse_quote!(__S);
    /// for (ix, (ty, _locals)) in referrer.iter_with_self(&marker).enumerate() {
    ///     // In the Leaker's crate: `impl<__S> Repeater<.., #ix> for Marker<__S> { type Type = #ty; }`
    ///     # let _ = ix;
    ///     let expected: Type = parse_quote!(Rel<<__S>::Item>);
    ///     assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
    /// }
    /// ```
    pub fn iter_with_self<'a>(
        &'a self,
        self_ty: &'a Type,
    ) -> impl Iterator<Item = (Type, Generics)> + 'a {
        use syn::fold::Fold;
        self.iter_with_local_generics().map(move |(ty, generics)| {
            let mut folder = ReplaceSelf { self_ty };
            (
                folder.fold_type(ty.clone()),
                folder.fold_generics(generics.clone()),
            )
        })
    }

    /// Completes the generic arguments `args` of the trait of the Leaker (whose parameters are
    /// `params`) in `context` (the context of the call of
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()), usually 0): returns one argument
    /// for each parameter, in the order of `params`, so the encoders (such as
    /// [`RepeaterSlots::encode_generics_args_to_ty()`](crate::RepeaterSlots::encode_generics_args_to_ty()))
    /// and the substitution of the parameters need not fill the defaults.
    ///
    /// The arguments are matched with `params` like
    /// [`encode_generics_args_to_ty_with()`](crate::encode_generics_args_to_ty_with()) does:
    /// lifetime arguments come first, a bare identifier corresponding to a const parameter (e.g.
    /// `N` in `MyTrait<N>`) is treated as a const argument, associated item bindings and
    /// constraints are skipped, and [`EncodeError`] is returned if there are too many arguments,
    /// or an argument without a default is missing. An omitted const argument takes the default
    /// as it is written, with the preceding parameters substituted.
    ///
    /// An omitted type argument takes the default rewritten in the context of the default (see
    /// [`Referrer::context_generics()`]): a default is written in the Leaker's type context
    /// (e.g. `U = RelType<T>`), where its relative types resolve, so they are replaced by
    /// `type_fn`, like [`Referrer::expand()`] does. It receives the leaked type, its index, and
    /// the complete arguments of the parameters before the defaulted one, which are the
    /// arguments of the context's root generics (to be encoded like those of the Repeater's
    /// implementation, e.g. with `RepeaterSlots::encode_generics_args_to_ty()` and
    /// `context_generics()`), and returns the projection to the leaked type. The result of
    /// `type_fn` is not substituted again, so a projection may contain the names of the
    /// parameters.
    ///
    /// The default of the `n`-th defaulted type parameter belongs to the context `context + 1 +
    /// n` (see [`Leaker::intern_with()`](crate::Leaker::intern_with())), so `params` must be the
    /// generics passed to it. [`EncodeError::MissingArgument`] is returned for an omitted
    /// argument if there is no such context.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// // In the Leaker's crate
    /// let tr: ItemTrait = parse_quote!(trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> U; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    ///
    /// // In the Referrer's crate (for `impl Tr<u8> for ..`, omitting `U`)
    /// let args: AngleBracketedGenericArguments = parse_quote!(<u8>);
    /// let completed = referrer
    ///     .complete_args(0, &args.args, &tr.generics.params, |_, ix, context_args| {
    ///         let context = referrer.context(ix);
    ///         let params = &referrer.context_generics(context).params;
    ///         let encoded =
    ///             encode_generics_args_to_ty_with(context_args, Some(params), |_, _| unreachable!())
    ///                 .unwrap();
    ///         parse_quote!(<Self as ::leaker::Repeater<#encoded, #ix>>::Type)
    ///     })
    ///     .unwrap();
    /// let expected: Vec<GenericArgument> =
    ///     vec![parse_quote!(u8), parse_quote!(<Self as ::leaker::Repeater<(u8,), 0usize>>::Type)];
    /// assert_eq!(quote!(#(#completed),*).to_string(), quote!(#(#expected),*).to_string());
    /// ```
    pub fn complete_args<'a>(
        &self,
        context: usize,
        args: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
        mut type_fn: impl FnMut(Type, usize, &[GenericArgument]) -> Type,
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        use syn::fold::Fold;
        complete_generics_args(
            args,
            params,
            |index, param, default, substitute, completed| {
                // The context of the default of the `n`-th defaulted type parameter.
                let n = params
                    .iter()
                    .take(index)
                    .filter(|p| {
                        matches!(
                            p,
                            GenericParam::Type(TypeParam {
                                default: Some(_),
                                ..
                            })
                        )
                    })
                    .count();
                let context = context + 1 + n;
                if self
                    .context_generics
                    .get(context)
                    .map_or(true, |generics| generics.params.len() != index)
                {
                    return Err(EncodeError::MissingArgument(Box::new(GenericParam::Type(
                        param.clone(),
                    ))));
                }
                // Replace the leaked types with placeholders, substitute the preceding parameters,
                // and then replace the placeholders, so that the projections are not substituted.
                let mut leaked = Vec::new();
                let ty = self.expand_in(context, default.clone(), |ty, ix| {
                    leaked.push((ty, ix));
                    let placeholder = Ident::new(
                        &format!("{}{}", PLACEHOLDER, leaked.len() - 1),
                        proc_macro2::Span::call_site(),
                    );
                    parse_quote!(#placeholder)
                });
                let ty = substitute.fold_type(ty);
                Ok(Resolve {
                    leaked,
                    completed,
                    type_fn: &mut type_fn,
                }
                .fold_type(ty))
            },
        )
    }

    /// Returns an iterator over the types that need to be encoded, together with their local
    /// generics. See [`Referrer::local_generics()`].
    pub fn iter_with_local_generics(&self) -> impl Iterator<Item = (&Type, &Generics)> {
        self.leak_types
            .iter()
            .zip(self.requirements.iter().map(Requirements::generics))
    }

    /// Converts this `Referrer` into a [`Visitor`] that can be used to transform AST nodes of the
    /// context 0.
    ///
    /// The `type_fn` callback is called for each type that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type.
    /// See [`Referrer::expand()`] for details.
    pub fn into_visitor<F: FnMut(Type, usize) -> Type>(self, type_fn: F) -> Visitor<F> {
        self.into_visitor_in(0, type_fn)
    }

    /// Like [`Referrer::into_visitor()`], for AST nodes of `context` (see
    /// [`Referrer::context()`]): only the types found in it are replaced.
    pub fn into_visitor_in<F: FnMut(Type, usize) -> Type>(
        self,
        context: usize,
        type_fn: F,
    ) -> Visitor<F> {
        Visitor {
            referrer: self,
            type_fn: Rc::new(RefCell::new(type_fn)),
            context,
            scopes: Vec::new(),
            outermost_as_leakers: false,
            self_ty: None,
            item_depth: 0,
        }
    }

    /// Expands a type by replacing any interned types using the provided mapping function.
    ///
    /// The `type_fn` callback is called for each subtype that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type. The type is
    /// expanded as in the context 0, with no generics in scope other than those of the Leaker.
    ///
    /// ```ignore
    /// let args = encode_generics_params_to_ty(&generics.params);
    /// referrer.expand(ty, |_, ix| {
    ///     parse_quote!(<#marker_ty as #crate::TypeRef<#ix, #args>>::Type)
    /// })
    /// ```
    pub fn expand(&self, ty: Type, type_fn: impl FnMut(Type, usize) -> Type) -> Type {
        self.expand_in(0, ty, type_fn)
    }

    /// Like [`Referrer::expand()`], for a type of `context` (see [`Referrer::context()`]).
    pub fn expand_in(
        &self,
        context: usize,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        self.expand_with(context, &mut Vec::new(), None, ty, type_fn)
    }

    /// Expands `ty` as in `context`, where the generics of `scopes` are in scope. `Self` is
    /// replaced by `self_ty`, if any, in the leaked types passed to `type_fn` and in the rest of
    /// `ty` (not in the types returned by `type_fn`).
    fn expand_with(
        &self,
        context: usize,
        scopes: &mut Vec<Scope>,
        self_ty: Option<&Type>,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        use syn::fold::Fold;
        struct Folder<'a, F> {
            referrer: &'a Referrer,
            context: usize,
            scopes: &'a mut Vec<Scope>,
            replace_self: Option<ReplaceSelf<'a>>,
            type_fn: F,
        }
        impl<'a, F> Folder<'a, F> {
            /// Pushes the scope of the lifetimes of a `for<..>` binder, as the Leaker does.
            fn push_binder(&mut self, lifetimes: Option<&BoundLifetimes>) -> bool {
                lifetimes
                    .map(|lifetimes| self.scopes.push(Scope::binder(lifetimes)))
                    .is_some()
            }
        }
        impl<'a, F: FnMut(Type, usize) -> Type> Fold for Folder<'a, F> {
            fn fold_type(&mut self, ty: Type) -> Type {
                match (
                    self.referrer.find(self.context, &ty, self.scopes),
                    self.replace_self,
                ) {
                    (Some(ix), None) => (self.type_fn)(ty, ix),
                    (Some(ix), Some(mut replace)) => (self.type_fn)(replace.fold_type(ty), ix),
                    (None, None) => syn::fold::fold_type(self, ty),
                    (None, Some(replace)) => replace.fold_type_with(self, ty),
                }
            }
            fn fold_expr(&mut self, expr: Expr) -> Expr {
                match self.replace_self {
                    Some(replace) => replace.fold_expr_with(self, expr),
                    None => syn::fold::fold_expr(self, expr),
                }
            }
            fn fold_macro(&mut self, i: Macro) -> Macro {
                match self.replace_self {
                    Some(replace) => replace.fold_macro_with(i),
                    None => syn::fold::fold_macro(self, i),
                }
            }
            fn fold_type_bare_fn(&mut self, i: TypeBareFn) -> TypeBareFn {
                let pushed = self.push_binder(i.lifetimes.as_ref());
                let i = syn::fold::fold_type_bare_fn(self, i);
                if pushed {
                    self.scopes.pop();
                }
                i
            }
            fn fold_trait_bound(&mut self, i: TraitBound) -> TraitBound {
                let pushed = self.push_binder(i.lifetimes.as_ref());
                let i = syn::fold::fold_trait_bound(self, i);
                if pushed {
                    self.scopes.pop();
                }
                i
            }
        }
        Folder {
            referrer: self,
            context,
            scopes,
            replace_self: self_ty.map(|self_ty| ReplaceSelf { self_ty }),
            type_fn,
        }
        .fold_type(ty)
    }
}

/// Replaces `Self` with a type, see [`Referrer::iter_with_self()`].
#[derive(Clone, Copy)]
struct ReplaceSelf<'a> {
    self_ty: &'a Type,
}

impl<'a> ReplaceSelf<'a> {
    /// Replaces the identifiers `Self` in `tokens` with the tokens of the type.
    fn replace_tokens(&self, tokens: TokenStream) -> TokenStream {
        use proc_macro2::{Group, TokenTree};
        tokens
            .into_iter()
            .map(|tt| match tt {
                TokenTree::Ident(ident) if ident == "Self" => {
                    let self_ty = self.self_ty;
                    quote!(#self_ty)
                }
                TokenTree::Group(group) => {
                    let mut new =
                        Group::new(group.delimiter(), self.replace_tokens(group.stream()));
                    new.set_span(group.span());
                    TokenTree::Group(new).into()
                }
                tt => TokenStream::from(tt),
            })
            .collect()
    }

    /// Returns the path without `Self` if `path` is `Self::A::B` (with no leading colon and no
    /// arguments on `Self`), or `None` if it is `Self` alone or does not start with `Self`.
    fn strip_self(path: &Path) -> Option<Option<Path>> {
        let first = path.segments.first()?;
        if path.leading_colon.is_some() || first.ident != "Self" || !first.arguments.is_none() {
            return None;
        }
        if path.segments.len() == 1 {
            return Some(None);
        }
        // The leading colon is the `::` after the qualified self type (see `qself()`).
        Some(Some(Path {
            leading_colon: Some(Default::default()),
            segments: path.segments.iter().skip(1).cloned().collect(),
        }))
    }

    fn qself(&self) -> QSelf {
        QSelf {
            lt_token: Default::default(),
            ty: Box::new(self.self_ty.clone()),
            position: 0,
            as_token: None,
            gt_token: Default::default(),
        }
    }

    /// Folds `ty` with `folder` (which folds the components), replacing `Self`.
    fn fold_type_with(&self, folder: &mut impl syn::fold::Fold, ty: Type) -> Type {
        match ty {
            Type::Path(TypePath { qself: None, path }) => match Self::strip_self(&path) {
                Some(None) => self.self_ty.clone(),
                Some(Some(rest)) => Type::Path(TypePath {
                    qself: Some(self.qself()),
                    path: folder.fold_path(rest),
                }),
                None => syn::fold::fold_type(folder, Type::Path(TypePath { qself: None, path })),
            },
            Type::Verbatim(tokens) => Type::Verbatim(self.replace_tokens(tokens)),
            ty => syn::fold::fold_type(folder, ty),
        }
    }

    /// Folds `expr` with `folder`, replacing `Self::A`. `Self` alone is a value, which cannot be
    /// replaced with a type.
    fn fold_expr_with(&self, folder: &mut impl syn::fold::Fold, expr: Expr) -> Expr {
        match expr {
            Expr::Path(ExprPath {
                attrs,
                qself: None,
                path,
            }) => match Self::strip_self(&path) {
                Some(Some(rest)) => Expr::Path(ExprPath {
                    attrs,
                    qself: Some(self.qself()),
                    path: folder.fold_path(rest),
                }),
                _ => syn::fold::fold_expr(
                    folder,
                    Expr::Path(ExprPath {
                        attrs,
                        qself: None,
                        path,
                    }),
                ),
            },
            expr => syn::fold::fold_expr(folder, expr),
        }
    }

    fn fold_macro_with(&self, mut i: Macro) -> Macro {
        i.tokens = self.replace_tokens(i.tokens);
        i
    }
}

impl<'a> syn::fold::Fold for ReplaceSelf<'a> {
    fn fold_type(&mut self, ty: Type) -> Type {
        let this = *self;
        this.fold_type_with(self, ty)
    }

    fn fold_expr(&mut self, expr: Expr) -> Expr {
        let this = *self;
        this.fold_expr_with(self, expr)
    }

    fn fold_macro(&mut self, i: Macro) -> Macro {
        self.fold_macro_with(i)
    }
}

/// The prefix of the names of the placeholders of the leaked types in [`Referrer::complete_args()`].
const PLACEHOLDER: &str = "__TypeLeakPlaceholder";

/// Replaces the placeholders with the result of `type_fn`.
struct Resolve<'a, F> {
    leaked: Vec<(Type, usize)>,
    completed: &'a [GenericArgument],
    type_fn: &'a mut F,
}

impl<'a, F: FnMut(Type, usize, &[GenericArgument]) -> Type> syn::fold::Fold for Resolve<'a, F> {
    fn fold_type(&mut self, ty: Type) -> Type {
        if let Type::Path(TypePath { qself: None, path }) = &ty {
            let number = path.get_ident().and_then(|ident| {
                ident
                    .to_string()
                    .strip_prefix(PLACEHOLDER)?
                    .parse::<usize>()
                    .ok()
            });
            if let Some((leaked, ix)) = number.and_then(|number| self.leaked.get(number).cloned()) {
                return (self.type_fn)(leaked, ix, self.completed);
            }
        }
        syn::fold::fold_type(self, ty)
    }
}

/// Rewrites AST nodes with the leaked types of a [`Referrer`], see [`Referrer::into_visitor()`].
///
/// It keeps track of the generics declared at the visited site (those of items, methods, impls and
/// `for<..>` binders), including those of the outermost item visited by default: a type is
/// rewritten only if it was leaked with the same local generics as it depends on here (see
/// [`Referrer::local_generics()`]). A type mentioning generics which are not leaked with it (e.g.
/// `T` in the `impl<T>` of the visited item) is not rewritten, since it refers to them rather than
/// to the Leaker's items. See [`Visitor::with_outermost_generics_as_leakers()`] to treat the
/// generics of the outermost item as those of the Leaker instead.
#[derive(Debug)]
pub struct Visitor<F> {
    referrer: Referrer,
    type_fn: Rc<RefCell<F>>,
    context: usize,
    /// The generics in scope, the innermost last.
    scopes: Vec<Scope>,
    /// See [`Visitor::with_outermost_generics_as_leakers()`].
    outermost_as_leakers: bool,
    /// See [`Visitor::with_self()`].
    self_ty: Option<Type>,
    /// The number of items enclosing the syntax being visited.
    item_depth: usize,
}

impl<F> Clone for Visitor<F> {
    fn clone(&self) -> Self {
        Self {
            referrer: self.referrer.clone(),
            type_fn: self.type_fn.clone(),
            context: self.context,
            scopes: self.scopes.clone(),
            outermost_as_leakers: self.outermost_as_leakers,
            self_ty: self.self_ty.clone(),
            item_depth: self.item_depth,
        }
    }
}

impl<F: FnMut(Type, usize) -> Type> Visitor<F> {
    /// Treats the generics of the outermost item visited as those of the Leaker, like
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()) does for the item it visits: they
    /// are not kept track of as local generics, so a type depending on them only (e.g. `Wrap<T>`
    /// in `impl<T> MyTrait<T> for W { fn f(&self) -> Wrap<T>; }`, visited as a whole, where `T` is
    /// a generic of the Leaker's trait) is rewritten if it was leaked.
    ///
    /// The outermost item is a struct, enum, union, trait, type alias, function, impl or trait
    /// alias (or the signature, if only a signature is visited). The generics of the items,
    /// methods, associated types and consts and `for<..>` binders in it are still local, and a
    /// module (which has no generics) only makes the items in it nested.
    ///
    /// By default, the generics of the outermost item are local too, so such a type is not
    /// rewritten.
    pub fn with_outermost_generics_as_leakers(mut self) -> Self {
        self.outermost_as_leakers = true;
        self
    }

    /// Replaces `Self` by `self_ty` in the rewritten copies, for a Repeater implemented for
    /// another type than the implementor of the Leaker trait (see [`Referrer::has_self()`] and
    /// [`Referrer::iter_with_self()`], which replaces `Self` in the leaked types in the same
    /// way).
    ///
    /// The leaked types are still found by their original tokens (e.g. `Rel<Self>` in the copy
    /// matches the leaked `Rel<Self>`), and the type passed to `type_fn` is the leaked type with
    /// `Self` replaced, so it agrees with the types of the Repeater's implementations. `Self` in
    /// the rest of the types of the copy is replaced as well (e.g. `::p::W<Self>` or
    /// `Self::Out`, also in bounds, where clauses and signatures), but not in the types returned
    /// by `type_fn`. Receivers (e.g. `&self`) and expressions other than those in types are not
    /// rewritten.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self, x: Rel<Self>) -> Self::Out; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    /// let mut sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>) -> Self::Out);
    /// referrer
    ///     .into_visitor(|_, ix| parse_quote!(<Marker<__S> as Repeater<#ix>>::Type))
    ///     .with_self(parse_quote!(__S))
    ///     .visit_signature_mut(&mut sig);
    /// let expected: Signature = parse_quote!(
    ///     fn f(&self, x: <Marker<__S> as Repeater<0usize>>::Type) -> <__S>::Out
    /// );
    /// assert_eq!(quote!(#sig).to_string(), quote!(#expected).to_string());
    /// ```
    pub fn with_self(mut self, self_ty: Type) -> Self {
        self.self_ty = Some(self_ty);
        self
    }

    /// Stops rewriting the leaked types which mention `idents` (items or imports declared in a
    /// block or a module).
    fn shadow_idents(&mut self, idents: Vec<Ident>) {
        self.referrer
            .map
            .retain(|(_, key), _| !mentions_names(&key.ty, idents.clone(), Vec::new()));
    }

    /// Shadows the items declared in `items`.
    fn shadow_items<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>) {
        fn use_names(tree: &UseTree, names: &mut Vec<Ident>) {
            match tree {
                UseTree::Path(UsePath { tree, .. }) => use_names(tree, names),
                UseTree::Name(UseName { ident }) if ident != "self" => names.push(ident.clone()),
                UseTree::Rename(UseRename { rename, .. }) => names.push(rename.clone()),
                UseTree::Group(UseGroup { items, .. }) => {
                    for tree in items {
                        use_names(tree, names)
                    }
                }
                _ => (),
            }
        }
        let mut names = Vec::new();
        for item in items {
            match item {
                Item::Struct(ItemStruct { ident, .. })
                | Item::Enum(ItemEnum { ident, .. })
                | Item::Union(ItemUnion { ident, .. })
                | Item::Trait(ItemTrait { ident, .. })
                | Item::TraitAlias(ItemTraitAlias { ident, .. })
                | Item::Type(ItemType { ident, .. })
                | Item::Const(ItemConst { ident, .. })
                | Item::Static(ItemStatic { ident, .. })
                | Item::Mod(ItemMod { ident, .. }) => names.push(ident.clone()),
                Item::Use(ItemUse { tree, .. }) => use_names(tree, &mut names),
                _ => (),
            }
        }
        if !names.is_empty() {
            self.shadow_idents(names);
        }
    }

    /// Returns a visitor with the generics of an item (or of a method, an associated type or a
    /// const if not `item`) in scope, and visits them. Those of the outermost item are not in
    /// scope if [`Visitor::with_outermost_generics_as_leakers()`] is set.
    fn with_generics_as(&mut self, generics: &mut Generics, item: bool) -> Self {
        let mut visitor = self.clone();
        visitor.item_depth += 1;
        // All the parameters are in scope of each other's bounds.
        if !(item && self.outermost_as_leakers && self.item_depth == 0) {
            visitor.scopes.push(Scope::new(generics));
        }
        for gp in generics.params.iter_mut() {
            visitor.visit_generic_param_mut(gp);
        }
        if let Some(where_clause) = &mut generics.where_clause {
            visitor.visit_where_clause_mut(where_clause);
        }
        visitor
    }

    fn with_item_generics(&mut self, generics: &mut Generics) -> Self {
        self.with_generics_as(generics, true)
    }

    fn with_generics(&mut self, generics: &mut Generics) -> Self {
        self.with_generics_as(generics, false)
    }

    fn with_signature_as(&mut self, sig: &mut Signature, item: bool) -> Self {
        let mut visitor = self.with_generics_as(&mut sig.generics, item);
        for input in sig.inputs.iter_mut() {
            visitor.visit_fn_arg_mut(input);
        }
        visitor.visit_return_type_mut(&mut sig.output);
        visitor
    }
}

impl<F: FnMut(Type, usize) -> Type> VisitMut for Visitor<F> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        *i = self.referrer.expand_with(
            self.context,
            &mut self.scopes,
            self.self_ty.as_ref(),
            i.clone(),
            &mut *self.type_fn.borrow_mut(),
        );
    }
    fn visit_receiver_mut(&mut self, i: &mut Receiver) {
        if self.self_ty.is_none() {
            return syn::visit_mut::visit_receiver_mut(self, i);
        }
        // The type of a receiver (`self: Box<Self>`, or `Self` in `&self`) is not rewritten.
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
    }
    fn visit_predicate_type_mut(&mut self, i: &mut PredicateType) {
        let binder = i.lifetimes.as_ref().map(Scope::binder);
        let pushed = binder.is_some();
        self.scopes.extend(binder);
        syn::visit_mut::visit_predicate_type_mut(self, i);
        if pushed {
            self.scopes.pop();
        }
    }
    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        let binder = i.lifetimes.as_ref().map(Scope::binder);
        let pushed = binder.is_some();
        self.scopes.extend(binder);
        syn::visit_mut::visit_trait_bound_mut(self, i);
        if pushed {
            self.scopes.pop();
        }
    }
    fn visit_const_param_mut(&mut self, i: &mut ConstParam) {
        // The type of a const parameter cannot be a projection, so it is kept as is.
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        if let Some(expr) = &mut i.default {
            self.visit_expr_mut(expr);
        }
    }
    fn visit_item_struct_mut(&mut self, i: &mut ItemStruct) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        visitor.visit_fields_mut(&mut i.fields);
    }
    fn visit_item_enum_mut(&mut self, i: &mut ItemEnum) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        for variant in i.variants.iter_mut() {
            visitor.visit_variant_mut(variant);
        }
    }
    fn visit_item_trait_mut(&mut self, i: &mut ItemTrait) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        for supertrait in i.supertraits.iter_mut() {
            visitor.visit_type_param_bound_mut(supertrait);
        }
        for item in i.items.iter_mut() {
            visitor.visit_trait_item_mut(item);
        }
    }
    fn visit_item_union_mut(&mut self, i: &mut ItemUnion) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        visitor.visit_fields_named_mut(&mut i.fields);
    }
    fn visit_item_type_mut(&mut self, i: &mut ItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
    }
    fn visit_signature_mut(&mut self, i: &mut Signature) {
        self.with_signature_as(i, true);
    }
    fn visit_item_impl_mut(&mut self, i: &mut ItemImpl) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        if let Some((_, path, _)) = &mut i.trait_ {
            visitor.visit_path_mut(path);
        }
        visitor.visit_type_mut(&mut i.self_ty);
        for item in i.items.iter_mut() {
            visitor.visit_impl_item_mut(item);
        }
    }
    fn visit_impl_item_fn_mut(&mut self, i: &mut ImplItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature_as(&mut i.sig, false);
        visitor.visit_block_mut(&mut i.block);
    }
    fn visit_impl_item_type_mut(&mut self, i: &mut ImplItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
    }
    fn visit_item_mod_mut(&mut self, i: &mut ItemMod) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        if let Some((_, items)) = &mut i.content {
            let mut visitor = self.clone();
            // The items in the module are nested items.
            visitor.item_depth += 1;
            visitor.shadow_items(items.iter());
            for item in items.iter_mut() {
                visitor.visit_item_mut(item);
            }
        }
    }
    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature_as(&mut i.sig, true);
        visitor.visit_block_mut(i.block.as_mut());
    }
    fn visit_trait_item_fn_mut(&mut self, i: &mut TraitItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature_as(&mut i.sig, false);
        if let Some(block) = &mut i.default {
            visitor.visit_block_mut(block);
        }
    }
    fn visit_trait_item_type_mut(&mut self, i: &mut TraitItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        for bound in i.bounds.iter_mut() {
            visitor.visit_type_param_bound_mut(bound);
        }
        if let Some((_, ty)) = &mut i.default {
            visitor.visit_type_mut(ty);
        }
    }
    fn visit_trait_item_const_mut(&mut self, i: &mut TraitItemConst) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
        if let Some((_, expr)) = &mut i.default {
            visitor.visit_expr_mut(expr);
        }
    }
    fn visit_block_mut(&mut self, i: &mut Block) {
        let mut visitor = self.clone();
        visitor.shadow_items(i.stmts.iter().filter_map(|stmt| match stmt {
            Stmt::Item(item) => Some(item),
            _ => None,
        }));
        for stmt in i.stmts.iter_mut() {
            visitor.visit_stmt_mut(stmt);
        }
    }
}

/// The version of this crate, emitted with a serialized [`Referrer`] to detect a Referrer passed
/// between incompatible versions of `type-leak`.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Parses a [`Referrer`] serialized with its [`ToTokens`] implementation: the version of
/// `type-leak` as a string literal, (the number of contexts if it is more than one,) and a group
/// for each context (see [`Referrer::context()`]): its root generics if they are not empty (the
/// generics and the where clause, followed by `;`, see [`Referrer::context_generics()`]), and the
/// leaked types in parentheses. The tokens after it are not consumed, so a Referrer can be
/// followed by other tokens. It is an error if the version differs from that of this crate.
impl Parse for Referrer {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let version: LitStr = input.parse()?;
        if version.value() != VERSION {
            return Err(Error::new(
                version.span(),
                format!(
                    "the Referrer was emitted by type-leak {}, but is parsed by type-leak {}",
                    version.value(),
                    VERSION
                ),
            ));
        }
        let n_contexts = if input.peek(LitInt) {
            input.parse::<LitInt>()?.base10_parse::<usize>()?
        } else {
            1
        };
        let mut entries = Vec::new();
        let mut context_generics = Vec::new();
        for context in 0..n_contexts {
            // The root generics, if any, are followed by `;`.
            let mut generics = Generics::default();
            if !input.peek(token::Paren) {
                generics = input.parse()?;
                generics.where_clause = input.parse()?;
                input.parse::<Token![;]>()?;
            }
            context_generics.push(generics);
            let content: syn::parse::ParseBuffer<'_>;
            parenthesized!(content in input);
            // Each entry is either `Type` or `{ <local generics> where ..; Type }`.
            let group = Punctuated::<(Generics, Type), Token![,]>::parse_terminated_with(
                &content,
                |input| {
                    if input.peek(token::Brace) {
                        let content: syn::parse::ParseBuffer<'_>;
                        braced!(content in input);
                        let mut generics: Generics = content.parse()?;
                        generics.where_clause = content.parse()?;
                        content.parse::<Token![;]>()?;
                        Ok((generics, content.parse()?))
                    } else {
                        Ok((Generics::default(), input.parse()?))
                    }
                },
            )?;
            entries.extend(
                group
                    .into_iter()
                    .map(|(generics, ty)| (context, ty, Requirements::new(generics))),
            );
        }
        Ok(Self::new(entries, context_generics))
    }
}

/// Serializes the [`Referrer`] (with the version of `type-leak`), to be parsed back with its
/// [`Parse`] implementation, e.g. in another macro invocation. There is a group for each context
/// (at least one), preceded by the number of the groups if it is more than one (e.g. `"0.8.0" 2
/// (MyType) <T> ; (Wrap<T>)`). A group is the leaked types in parentheses, preceded by the root
/// generics of the context (see [`Referrer::context_generics()`]) and `;` if they are not empty
/// (e.g. `<T> where T: Clone ; (Wrap<T>)`).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::{quote, ToTokens};
/// let item: ItemStruct = parse_quote!(struct S { a: MyType });
/// let referrer = Leaker::from_struct(&item).unwrap().finish();
/// let tokens = referrer.to_token_stream();
/// assert_eq!(
///     tokens.to_string(),
///     format!("\"{}\" (MyType)", env!("CARGO_PKG_VERSION"))
/// );
/// assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
/// ```
impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(quote!(#VERSION));
        let n_contexts = self.n_contexts();
        if n_contexts > 1 {
            // The number of the groups, so that the Referrer can be followed by other tokens.
            tokens.extend(quote!(#{proc_macro2::Literal::usize_unsuffixed(n_contexts)}));
        }
        for context in 0..n_contexts {
            let entries: Vec<_> = self
                .leak_types
                .iter()
                .zip(&self.requirements)
                .zip(&self.contexts)
                .filter(|(_, c)| **c == context)
                .map(|(entry, _)| entry)
                .collect();
            let generics = &self.context_generics[context];
            if !generics.params.is_empty() || generics.where_clause.is_some() {
                tokens.extend(quote!(#generics #{&generics.where_clause} ;));
            }
            tokens.extend(quote! {
                (
                    #(for (item, requirements) in entries), {
                        #(if requirements.is_empty()) {
                            #item
                        } #(else) {
                            { #{requirements.generics()} #{&requirements.generics().where_clause}; #item }
                        }
                    }
                )
            });
        }
    }
}

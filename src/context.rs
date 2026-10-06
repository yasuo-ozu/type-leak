//! The generics (and the meaning of `Self`) which a leaked type depends on (see
//! [`Referrer::generics()`](crate::Referrer::generics())).

use crate::util::{
    generic_param_mentions_self, generic_param_name, ident_name, token_names, type_mentions_self,
    unwrap_type_group, where_predicate_mentions_self,
};
use proc_macro2::{Literal, TokenStream};
use std::collections::{BTreeMap, HashMap, HashSet};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::{quote, ToTokens};

/// The generics declared at one level: those of the Leaker (the outermost level), or those inside
/// it, e.g. declared by a trait method, a GAT, a nested item or a `for<..>` binder.
#[derive(Clone, Debug)]
pub(crate) struct Scope {
    params: Vec<GenericParam>,
    predicates: Vec<WherePredicate>,
    /// The scope is that of the lifetimes of a `for<..>` binder (see [`Scope::binder()`]).
    binder: bool,
}

impl Scope {
    /// Returns the scope of `generics`, without the defaults (which are not part of a
    /// declaration an implementation can copy, and are not dependencies of the type).
    pub(crate) fn new(generics: &Generics) -> Self {
        Self {
            params: generics.params.iter().cloned().map(strip_default).collect(),
            predicates: generics
                .where_clause
                .iter()
                .flat_map(|wc| wc.predicates.iter().cloned())
                .collect(),
            binder: false,
        }
    }

    /// The scope of the lifetimes of a `for<..>` binder.
    pub(crate) fn binder(lifetimes: &BoundLifetimes) -> Self {
        Self {
            params: lifetimes.lifetimes.iter().cloned().collect(),
            predicates: Vec::new(),
            binder: true,
        }
    }

    pub(crate) fn params(&self) -> impl Iterator<Item = &GenericParam> {
        self.params.iter()
    }

    /// Returns `true` if the scope declares the parameter named `name` (e.g. `T` or `'a`).
    fn declares(&self, name: &str) -> bool {
        self.params
            .iter()
            .any(|param| generic_param_name(param) == name)
    }
}

/// What `Self` means in a leaked type (see [`Referrer::context_self()`]).
///
/// A leaked type mentioning `Self` is the same type only where `Self` has the same meaning, so it
/// is part of the context of the type. The Repeater's implementation must be written for
/// the type described, e.g. `impl<..> Repeater<..> for S` where `S` is the implementor of the
/// trait.
///
/// [`Referrer::context_self()`]: crate::Referrer::context_self()
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SelfRequirement {
    /// `Self` is the implementor of the trait: `Self: path<args>` (e.g. `Self: MyTrait<T, U>`).
    Bound(Path),
    /// `Self` is this type (a struct, enum, union, or the self type of an impl), e.g.
    /// `MyStruct<T>`.
    Type(Type),
    /// `Self` of an [`intern_with()`](crate::Leaker::intern_with()) call without an enclosing item
    /// (identified by the number of the call).
    Call(usize),
}

impl SelfRequirement {
    /// Returns the `Self` of the trait `ident` with the parameters `generics` as arguments, e.g.
    /// `MyTrait<'a, T, N>` for `trait MyTrait<'a, T, const N: usize>`.
    pub(crate) fn of_trait(ident: &Ident, generics: &Generics) -> Self {
        let args = generics_as_args(generics);
        Self::Bound(parse2(quote!(#ident #args)).unwrap())
    }

    /// Returns the `Self` of the struct, enum or union `ident` with the parameters `generics` as
    /// arguments, e.g. `MyStruct<T>`.
    pub(crate) fn of_item(ident: &Ident, generics: &Generics) -> Self {
        let args = generics_as_args(generics);
        Self::Type(parse2(quote!(#ident #args)).unwrap())
    }

    /// Returns the names in the description (e.g. `T` in `MyStruct<T>`).
    fn names(&self, names: &mut HashSet<String>) {
        match self {
            Self::Bound(path) => token_names(quote!(#path), names),
            Self::Type(ty) => token_names(quote!(#ty), names),
            Self::Call(_) => (),
        }
    }

    /// Returns the tokens by which the descriptions are compared.
    fn render(&self) -> String {
        match self {
            Self::Bound(path) => format!(" Self: {}", quote!(#path)),
            Self::Type(ty) => format!(" Self = {}", quote!(#ty)),
            Self::Call(n) => format!(" Self = #{}", n),
        }
    }
}

impl PartialEq for SelfRequirement {
    fn eq(&self, other: &Self) -> bool {
        self.render() == other.render()
    }
}

impl Eq for SelfRequirement {}

impl std::hash::Hash for SelfRequirement {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.render().hash(state)
    }
}

/// Returns the parameters of `generics` as the arguments of a path: `<'a, T, N>` (or nothing if
/// there are no parameters).
fn generics_as_args(generics: &Generics) -> TokenStream {
    if generics.params.is_empty() {
        return TokenStream::new();
    }
    let args = generics.params.iter().map(|param| match param {
        GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => quote!(#lifetime),
        GenericParam::Type(TypeParam { ident, .. })
        | GenericParam::Const(ConstParam { ident, .. }) => quote!(#ident),
    });
    quote!(<#(#args),*>)
}

/// Removes the default of a parameter.
fn strip_default(mut param: GenericParam) -> GenericParam {
    match &mut param {
        GenericParam::Type(param) => {
            param.eq_token = None;
            param.default = None;
        }
        GenericParam::Const(param) => {
            param.eq_token = None;
            param.default = None;
        }
        GenericParam::Lifetime(_) => (),
    }
    param
}

/// The context of a leaked type: the generics (with their bounds and the where predicates
/// mentioning them) which a type depends on, and what `Self` means if it mentions it, in a
/// canonical form: two declarations with the same meaning up to the order of the parameters, the
/// bounds and the predicates, and to whether a bound is written inline or in the where clause
/// (e.g. `<T: Debug + Clone, U>` and `<U, T> where T: Clone + Debug`), are equal.
///
/// If the type depends on the module where it is written (it has a relative path, which resolves
/// differently in another module) in an analysis of a module, the path of that module is part of
/// the context too (see
/// [`Referrer::context_module()`](crate::Referrer::context_module())).
///
/// Types with the same tokens and equal contexts are the same leaked type, whichever call of
/// [`Leaker::intern_with()`](crate::Leaker::intern_with()) found them.
#[derive(Clone, Debug)]
pub(crate) struct Context {
    generics: Generics,
    self_ty: Option<SelfRequirement>,
    /// The lifetimes `'a` (parameters of `generics`) such that the types of the context imply
    /// `Self: 'a`, e.g. by `&'a Self::Item` (see [`implied_outlives()`]), or
    /// that a predicate written in the item says so (`where Self: 'a`, see
    /// [`written_self_outlives()`]), sorted and without duplicates. Empty if `self_ty` is `None`,
    /// as `Self` has no meaning then. They are not among the bounds of `generics`, which `Self` is
    /// not a parameter of, nor among its where predicates.
    self_outlives: Vec<Lifetime>,
    /// The path of the module where the type is written, relative to the analyzed module (empty
    /// for the analyzed module itself). `None` for a type which does not depend on its module, or
    /// if no module is analyzed.
    module: Option<Vec<Ident>>,
}

impl Context {
    /// Returns the canonical form of `generics`, with `self_ty`.
    pub(crate) fn new(generics: Generics, self_ty: Option<SelfRequirement>) -> Self {
        Self {
            generics: canonicalize(generics),
            self_ty,
            self_outlives: Vec::new(),
            module: None,
        }
    }

    /// Returns the context with `Self: 'a` for each of `lifetimes` (see [`Context`]), if `Self` has
    /// a meaning in it.
    pub(crate) fn with_self_outlives(mut self, lifetimes: Vec<Lifetime>) -> Self {
        let mut lifetimes: Vec<(String, Lifetime)> = lifetimes
            .into_iter()
            .map(|lifetime| (lifetime.to_string(), lifetime))
            .collect();
        lifetimes.sort_by(|a, b| a.0.cmp(&b.0));
        lifetimes.dedup_by(|a, b| a.0 == b.0);
        self.self_outlives = if self.self_ty.is_some() {
            lifetimes
                .into_iter()
                .map(|(_, lifetime)| lifetime)
                .collect()
        } else {
            Vec::new()
        };
        self
    }

    /// Returns the context with the module `module` (see [`Context`]).
    pub(crate) fn with_module(mut self, module: Option<Vec<Ident>>) -> Self {
        self.module = module;
        self
    }

    /// Returns the tokens of the canonical form, by which contexts are compared.
    fn render(&self) -> String {
        let generics = &self.generics;
        let where_clause = &generics.where_clause;
        let mut rendered = quote!(#generics #where_clause).to_string();
        if let Some(self_ty) = &self.self_ty {
            rendered.push_str(&self_ty.render());
        }
        for (i, lifetime) in self.self_outlives.iter().enumerate() {
            rendered.push_str(if i == 0 { " Self: " } else { " + " });
            rendered.push_str(&lifetime.to_string());
        }
        if let Some(module) = &self.module {
            rendered.push_str(" mod ");
            if module.is_empty() {
                rendered.push_str("self");
            }
            for (i, ident) in module.iter().enumerate() {
                if i > 0 {
                    rendered.push_str("::");
                }
                rendered.push_str(&ident.to_string());
            }
        }
        rendered
    }

    /// Returns the generics declared in `scopes` (the outermost, i.e. those of the Leaker, first,
    /// and the innermost last) which `ty` depends on, and `self_req` if it mentions `Self`. A
    /// name refers to its innermost declaration visible from where it is used: from the type for
    /// the names in it, and from the declaring scope for the names in the bounds of a parameter.
    /// The names in `self_req` (e.g. the parameters of the trait) are used by the type as well.
    /// Names which are not declared in `scopes` (e.g. paths) are ignored.
    ///
    /// If a bound or a predicate of a generic the type depends on mentions `Self`, the type
    /// depends on the meaning of `Self` too.
    ///
    /// The outlives bounds which the type implies on the generics (e.g. `T: 'a` for `&'a T`, see
    /// [`implied_outlives()`]; not one for `Rel<'a, T>`, whose definition is not seen) are added to
    /// the bounds of the parameters, as if they were written: the Repeater's implementation encodes
    /// them in its trait arguments, so that they are implied wherever the type is copied (and does
    /// not declare them, see [`Referrer::generics()`](crate::Referrer::generics())).
    ///
    /// Identifiers are matched by name, so a parameter may be included even if the name refers to
    /// something else (e.g. `::path::N`). This is harmless, because the Leaker and the Referrer
    /// include the same generics.
    ///
    /// `assumed` are the generics which no scope declares, but which are known to be generics of
    /// the Leaker, as the Referrer assumes the generics it does not see declared (see
    /// [`Referrer::into_visitor()`](crate::Referrer::into_visitor())): the outlives bounds
    /// implied by the type on the declared generics (e.g. `T: 'a` of `&'a T` where only `T` is
    /// declared) are those of the Leaker's. The Leaker passes none: a lifetime which no scope
    /// declares is a relative one, not a generic.
    ///
    /// `enclosing` are the lifetimes which the types around `ty` make it outlive (see
    /// [`Enclosing`]): `ty: 'a` implies the same bounds as the types inside `ty` do (see
    /// [`implied_outlives()`]), for each generic lifetime `'a` of them, so a type leaked alone
    /// (e.g. `Rel2<T>` of `&'a Vec<Rel2<T>>`) has the bounds which the type it is a component of
    /// implies on it. `'a` is among the generics of the context then, even if the type
    /// does not mention it.
    pub(crate) fn of(
        ty: &Type,
        scopes: &[Scope],
        self_req: Option<&SelfRequirement>,
        assumed: &Assumed,
        enclosing: &Enclosing,
    ) -> Self {
        let mut names = HashSet::new();
        token_names(quote!(#ty), &mut names);
        let mut mentions_self = type_mentions_self(ty);
        let mut self_names = HashSet::new();
        if let Some(self_req) = self_req {
            self_req.names(&mut self_names);
        }
        if mentions_self {
            names.extend(self_names.iter().cloned());
        }
        // The names to resolve, each with the innermost scope it is visible from.
        let mut work: Vec<(usize, String)> = names
            .into_iter()
            .filter_map(|name| Some((scopes.len().checked_sub(1)?, name)))
            .collect();
        // The lifetimes around the type which it is bounded by (see `Enclosing`). One which a scope
        // declares is a generic of the context (with its own bounds, as any other) if the type
        // mentions a generic to bound by it: it is not named by the type otherwise.
        let ancestors = enclosing.generic(scopes, assumed);
        let mentions = Mentions::of(ty, scopes, assumed, !ancestors.is_empty());
        for (lifetime, depth) in &ancestors {
            // A type parameter, which a lifetime bounds (a const parameter is not bounded).
            let is_type_param = |name: &String| {
                scopes.iter().any(|scope| {
                    scope.params().any(|param| {
                        matches!(param, GenericParam::Type(param) if &ident_name(&param.ident) == name)
                    })
                })
            };
            let bounds_something = (self_req.is_some()
                && mentions.types.iter().any(|name| name == "Self"))
                || mentions
                    .types
                    .iter()
                    .any(|name| is_type_param(name) || assumed.types.contains(name))
                || mentions.lifetimes.iter().any(|other| other != lifetime);
            if let (true, Some(from)) = (bounds_something, depth.checked_sub(1)) {
                work.push((from, lifetime.to_string()));
            }
        }
        let ancestors: Vec<Lifetime> = ancestors
            .into_iter()
            .map(|(lifetime, _)| lifetime)
            .collect();
        let mut params = BTreeMap::new();
        let mut predicates = BTreeMap::new();
        // Whether the predicates bounding `Self` are included (once `Self` is mentioned).
        let mut swept_self = false;
        loop {
            let (from, name) = match work.pop() {
                Some(item) => item,
                None if mentions_self && !swept_self => {
                    // A predicate bounding `Self` (e.g. `where Self: Sized`) is part of what the
                    // type depends on, as the implementation must declare it too.
                    swept_self = true;
                    for (ix, scope) in scopes.iter().enumerate() {
                        for (k, predicate) in scope.predicates.iter().enumerate() {
                            let bounds_self = match predicate {
                                WherePredicate::Type(predicate) => {
                                    type_mentions_self(&predicate.bounded_ty)
                                }
                                _ => false,
                            };
                            if bounds_self && !predicates.contains_key(&(ix, k)) {
                                let mut names = HashSet::new();
                                token_names(quote!(#predicate), &mut names);
                                predicates.insert((ix, k), predicate.clone());
                                work.extend(names.into_iter().map(|name| (ix, name)));
                            }
                        }
                    }
                    continue;
                }
                None => break,
            };
            let found = (0..=from).rev().find_map(|ix| {
                scopes[ix]
                    .params
                    .iter()
                    .position(|param| generic_param_name(param) == name)
                    .map(|pos| (ix, pos))
            });
            let (ix, pos) = match found {
                Some(found) => found,
                None => continue,
            };
            if params.contains_key(&(ix, pos)) {
                continue;
            }
            let param = &scopes[ix].params[pos];
            params.insert((ix, pos), param.clone());
            let mut names = HashSet::new();
            token_names(quote!(#param), &mut names);
            work.extend(names.into_iter().map(|name| (ix, name)));
            let mut found_self = generic_param_mentions_self(param);
            // The predicates mentioning the parameter, and the names in them.
            for (k, predicate) in scopes[ix].predicates.iter().enumerate() {
                if predicates.contains_key(&(ix, k)) {
                    continue;
                }
                let mut names = HashSet::new();
                token_names(quote!(#predicate), &mut names);
                if names.contains(&name) {
                    predicates.insert((ix, k), predicate.clone());
                    work.extend(names.into_iter().map(|name| (ix, name)));
                    found_self |= where_predicate_mentions_self(predicate);
                }
            }
            if found_self && !mentions_self {
                mentions_self = true;
                work.extend(
                    self_names
                        .iter()
                        .map(|name| (scopes.len() - 1, name.clone())),
                );
            }
        }
        let mut params: Vec<GenericParam> = params.into_values().collect();
        let mut predicates: Vec<WherePredicate> = predicates.into_values().collect();
        let mut self_outlives = Vec::new();
        // A bound `Self: 'a` which is written (`where Self: 'a`, `where Self: Sized + 'a`) is the
        // same as one which the type implies: it is a part of the context as `self_outlives`, so it
        // is encoded as `&'a Self` and never declared, and a context is the same whether it is
        // written or implied (see `written_self_outlives()`).
        if self_req.is_some() {
            self_outlives.extend(written_self_outlives(&mut predicates, &params));
        }
        for (bounded, lifetime) in
            implied_outlives(ty, scopes, assumed, &params, self_req.is_some(), &ancestors)
        {
            if bounded == Bounded::SelfTy {
                self_outlives.push(lifetime);
                continue;
            }
            for param in params.iter_mut() {
                match (param, &bounded) {
                    (GenericParam::Type(param), Bounded::Type(name))
                        if &ident_name(&param.ident) == name =>
                    {
                        param
                            .bounds
                            .push(TypeParamBound::Lifetime(lifetime.clone()))
                    }
                    (GenericParam::Lifetime(param), Bounded::Lifetime(name))
                        if &param.lifetime == name =>
                    {
                        param.bounds.push(lifetime.clone())
                    }
                    _ => (),
                }
            }
        }
        let mut generics = Generics {
            params: params.into_iter().collect(),
            ..Default::default()
        };
        if !predicates.is_empty() {
            generics.where_clause = Some(WhereClause {
                where_token: Default::default(),
                predicates: predicates.into_iter().collect(),
            });
        }
        Self::new(generics, self_req.filter(|_| mentions_self).cloned())
            .with_self_outlives(self_outlives)
    }

    /// Returns the context of a type which depends on no generics and not on `Self`.
    pub(crate) fn none() -> Self {
        Self::new(Generics::default(), None)
    }

    /// Returns `true` if the type depends on no generics and not on `Self` (whichever module it
    /// is written in).
    pub(crate) fn is_empty(&self) -> bool {
        self.generics.params.is_empty()
            && self.generics.where_clause.is_none()
            && self.self_ty.is_none()
    }

    pub(crate) fn generics(&self) -> &Generics {
        &self.generics
    }

    pub(crate) fn self_ty(&self) -> Option<&SelfRequirement> {
        self.self_ty.as_ref()
    }

    pub(crate) fn self_outlives(&self) -> &[Lifetime] {
        &self.self_outlives
    }

    pub(crate) fn module(&self) -> Option<&[Ident]> {
        self.module.as_deref()
    }

    /// Returns the context without the module.
    pub(crate) fn without_module(&self) -> Self {
        self.clone().with_module(None)
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.render() == other.render()
    }
}

impl Eq for Context {}

impl PartialOrd for Context {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Context {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.render().cmp(&other.render())
    }
}

impl std::hash::Hash for Context {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.render().hash(state)
    }
}

/// The identifier of a context: a distinct combination of the generics a type uses, what `Self`
/// means in it, and its module, see [`Referrer::context()`](crate::Referrer::context()). It
/// identifies the generics of a call of
/// [`Leaker::intern_with()`](crate::Leaker::intern_with()) too, as its top-level context.
///
/// The entries of a [`Referrer`](crate::Referrer) whose Repeater impls are written the same way
/// (they declare the same generics, are for the same `Self` and in the same module) have the same
/// id, also if they were found by different calls of
/// [`Leaker::intern_with()`](crate::Leaker::intern_with()). An id is assigned when the
/// context first appears in the [`Leaker`](crate::Leaker), counting from 0, and it is never
/// renumbered: it is the same in the value returned by
/// [`Leaker::intern_with()`](crate::Leaker::intern_with()) and in the
/// [`Referrer`](crate::Referrer) (see [`Referrer::contexts()`](crate::Referrer::contexts())),
/// where the ids of the contexts which have no entry left (and are not top-level) do not appear
/// (so they may have gaps).
///
/// It is written as an integer literal (e.g. `3`) by [`ToTokens`], and parsed from one (with the
/// suffix `usize` or without any) by [`Parse`].
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::ToTokens;
/// let (_, id) = Leaker::from_struct(&parse_quote!(struct S { a: Rel })).unwrap();
/// assert_eq!(id.to_string(), "0");
/// assert_eq!(id.to_token_stream().to_string(), "0");
/// assert_eq!(parse_str::<ContextId>("0usize").unwrap(), id);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextId(usize);

impl ContextId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the number of the id, counting from 0 in the order in which the contexts first
    /// appear in the [`Leaker`](crate::Leaker).
    pub fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for ContextId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl ToTokens for ContextId {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::usize_unsuffixed(self.0).to_tokens(tokens)
    }
}

impl Parse for ContextId {
    fn parse(input: ParseStream) -> Result<Self> {
        let lit: LitInt = input.parse()?;
        if !matches!(lit.suffix(), "" | "usize") {
            return Err(Error::new(
                lit.span(),
                "a context id is an integer, with no suffix or `usize`",
            ));
        }
        Ok(Self(lit.base10_parse()?))
    }
}

/// The distinct [`Context`]s of the leaked types (see
/// [`Referrer::context()`](crate::Referrer::context())), numbered from 0 in the order of their
/// first appearance. The number of a context is never changed by the calls which follow, as the
/// contexts are only added (but for those of a failed call, see [`Contexts::truncate()`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct Contexts {
    ids: HashMap<Context, ContextId>,
    list: Vec<Context>,
}

impl Contexts {
    /// Returns the id of `context`, which is added if it is new.
    pub(crate) fn intern(&mut self, context: Context) -> ContextId {
        if let Some(id) = self.ids.get(&context) {
            return *id;
        }
        let id = ContextId(self.list.len());
        self.ids.insert(context.clone(), id);
        self.list.push(context);
        id
    }

    /// Returns the id of `context` if it is there.
    pub(crate) fn find(&self, context: &Context) -> Option<ContextId> {
        self.ids.get(context).copied()
    }

    pub(crate) fn len(&self) -> usize {
        self.list.len()
    }

    pub(crate) fn get(&self, id: ContextId) -> &Context {
        &self.list[id.0]
    }

    /// Removes the contexts numbered `len` or more (added after `len()` returned `len`).
    pub(crate) fn truncate(&mut self, len: usize) {
        for context in self.list.drain(len..) {
            self.ids.remove(&context);
        }
    }
}

/// Sorts and deduplicates `bounds` by their tokens.
fn sort_bounds<T: template_quote::ToTokens, P: Default>(
    bounds: Punctuated<T, P>,
) -> Punctuated<T, P> {
    let mut bounds: Vec<(String, T)> = bounds
        .into_iter()
        .map(|bound| (quote!(#bound).to_string(), bound))
        .collect();
    bounds.sort_by(|a, b| a.0.cmp(&b.0));
    bounds.dedup_by(|a, b| a.0 == b.0);
    bounds.into_iter().map(|(_, bound)| bound).collect()
}

/// Moves the lifetime bounds on `Self` out of `predicates`. For each written predicate `Self: 'a
/// + ..` (not under a `for<..>` binder), the bounds by a lifetime parameter of `params` are
/// removed from it and returned, and the predicate is removed if nothing else is left of it
/// (`Self: Sized + 'a` is left as `Self: Sized`). The others stay, as ordinary bounds: a bound by
/// `'static` or by a lifetime which is not a parameter of the context, and the trait bounds.
fn written_self_outlives(
    predicates: &mut Vec<WherePredicate>,
    params: &[GenericParam],
) -> Vec<Lifetime> {
    let is_param = |lifetime: &Lifetime| {
        params.iter().any(
            |param| matches!(param, GenericParam::Lifetime(param) if &param.lifetime == lifetime),
        )
    };
    let mut moved = Vec::new();
    let mut remaining = Vec::with_capacity(predicates.len());
    for predicate in std::mem::take(predicates) {
        match predicate {
            WherePredicate::Type(mut predicate)
                if predicate.lifetimes.is_none()
                    && matches!(unwrap_type_group(&predicate.bounded_ty),
                        Type::Path(TypePath { qself: None, path }) if path.is_ident("Self")) =>
            {
                let mut kept: Punctuated<TypeParamBound, Token![+]> = Punctuated::new();
                for bound in std::mem::take(&mut predicate.bounds) {
                    match bound {
                        TypeParamBound::Lifetime(lifetime) if is_param(&lifetime) => {
                            moved.push(lifetime)
                        }
                        bound => kept.push(bound),
                    }
                }
                if !kept.is_empty() {
                    predicate.bounds = kept;
                    remaining.push(WherePredicate::Type(predicate));
                }
            }
            predicate => remaining.push(predicate),
        }
    }
    *predicates = remaining;
    moved
}

/// The generics of the Leaker which no scope declares, but which are known to be generics: the
/// Referrer assumes the generics it does not see declared to be those of the Leaker (see
/// [`Context::of()`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct Assumed {
    /// The lifetime parameters.
    pub(crate) lifetimes: Vec<Lifetime>,
    /// The names of the type parameters.
    pub(crate) types: Vec<String>,
}

/// The lifetimes which a type must outlive because of the types around it (it is a component of
/// them, in the sense of [`Enclosing::components_of()`]), each with the number of scopes in scope
/// where it is written, which tells the parameter it names (see [`is_generic_lifetime()`]). The
/// type `X` of `&'a X` is one, and so is `Rel2<T>` of `&'a Vec<Rel2<T>>`: `&'a Vec<X>` is
/// well-formed only if `X: 'a`, which neither the implementation of the Repeater nor the copy of
/// a field (rustc infers the bound of the original, but not through a projection) has unless the
/// context of `X` carries it. The lifetime arguments of a path (`Ref<'a, Rel2<T>>`) are not
/// recorded: they are not assumed to bound its type arguments, whose definition is not seen (see
/// [`implied_outlives()`]).
///
/// Whether a path `T::A<X>` is a projection depends on `T` being a type parameter, which the
/// Referrer may not see declared (it assumes the generics it does not see declared to be those of
/// the Leaker, see [`Assumed`]). So it is recorded as such, and what the type outlives is resolved
/// when the context is computed, with the scopes and the generics assumed then (see
/// [`Enclosing::generic()`]): the Leaker knows the generics in scope, and the Referrer knows them
/// per entry it compares the context with.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Enclosing(Vec<Link>);

/// A link of [`Enclosing`], in the order of the types around the one being looked up (the outermost
/// first).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Link {
    /// The types inside must outlive the lifetime of a reference, written where `depth` scopes
    /// were in scope.
    Lifetime { lifetime: Lifetime, depth: usize },
    /// The path `head::A<..>` (written where `depth` scopes were in scope), which is a projection
    /// if `head` is a type parameter. A projection does not pass on what is required of it to the
    /// types inside it (see [`Enclosing::components_of()`]), so everything before it is dropped.
    Projection { head: String, depth: usize },
}

impl Enclosing {
    /// Returns what the types which are components of `ty` must outlive, when `self` is what `ty`
    /// itself must (`depth` is the number of scopes in scope where `ty` is written). The
    /// requirement `ty: 'a` is that of its components, as rustc derives it (see
    /// [`implied_outlives()`] for what it makes of them):
    ///
    /// - `&'a X` (also `&'a mut X`) requires `X: 'a`, so `X` must outlive `'a` too. It is the only
    ///   type which adds a lifetime to what its components must outlive: the lifetime arguments of
    ///   a path (`R<'a, X>`) do not, as `R<'a, X>` is well-formed without `X: 'a` unless the
    ///   definition of `R` says so, which cannot be seen, and a bound is not assumed (see
    ///   [`implied_outlives()`]).
    /// - Everything else which is made of types passes what `ty` must outlive on to them: the
    ///   elements of a tuple, an array or a slice, the type in parentheses or a group or behind a
    ///   raw pointer, the type arguments of a path (also one without a lifetime argument, like
    ///   `Vec<X>`), the inputs and the output of a function pointer, and the type arguments and
    ///   the associated type bindings of a trait object (`dyn Tr<X> + 'b: 'a` requires `X: 'a`
    ///   as well as `'b: 'a`, whatever the bound `X: 'b` says).
    /// - A projection does not: it outlives `'a` if its components do (but also by a bound
    ///   written for it), so `X: 'a` is not required of `X`. That is the projection with a
    ///   qualified self (`<X as Tr<Y>>::A<Z>`), and a path starting with a type parameter or
    ///   `Self` (`T::A<X>`, `Self::A<X>`), whose arguments are not those of a type either (a GAT
    ///   bound is not implied). A path starting with a type parameter is told by the generics in
    ///   scope, so it is recorded as such and decided by [`Enclosing::generic()`]. An `impl Trait`
    ///   is not made of types in this sense, and neither is the length of an array.
    ///
    /// A `for<..>` binder does not change them (`&'a for<'x> fn(&'x X)` requires `X: 'a`), but its
    /// own lifetimes are not generics, so they are never recorded (see
    /// [`is_generic_lifetime()`]).
    pub(crate) fn components_of(&self, ty: &Type, depth: usize) -> Self {
        let mut next = self.clone();
        match ty {
            Type::Reference(TypeReference {
                lifetime: Some(lifetime),
                ..
            }) => next.push(lifetime, depth),
            Type::Path(TypePath { qself: None, path }) => {
                let head = projection_head(path);
                if head.map_or(false, |head| head == "Self") {
                    // `Self::A<X>` is always a projection.
                    return Self::default();
                }
                if let Some(head) = head {
                    next.0.push(Link::Projection {
                        head: ident_name(head),
                        depth,
                    });
                }
            }
            Type::Path(_) | Type::ImplTrait(_) => return Self::default(),
            _ => (),
        }
        next
    }

    fn push(&mut self, lifetime: &Lifetime, depth: usize) {
        let link = Link::Lifetime {
            lifetime: lifetime.clone(),
            depth,
        };
        if lifetime.ident != "static" && lifetime.ident != "_" && !self.0.contains(&link) {
            self.0.push(link);
        }
    }

    /// Returns the lifetimes which the types must outlive, each with the number of scopes where it
    /// was written, once the paths which are projections (the first segment of which is a type
    /// parameter, see [`is_type_parameter()`]) have dropped what was before them.
    fn active(&self, scopes: &[Scope], assumed: &Assumed) -> Vec<(Lifetime, usize)> {
        let mut active: Vec<(Lifetime, usize)> = Vec::new();
        for link in &self.0 {
            match link {
                Link::Projection { head, depth } => {
                    if is_type_parameter(head, visible(scopes, *depth), assumed) {
                        active.clear();
                    }
                }
                Link::Lifetime { lifetime, depth } => {
                    let entry = (lifetime.clone(), *depth);
                    if !active.contains(&entry) {
                        active.push(entry);
                    }
                }
            }
        }
        active
    }

    /// Returns the lifetimes which are generic (see [`is_generic_lifetime()`]), each with the
    /// number of scopes it was written in, in the order they enclose the type, from the outermost.
    /// `assumed` tells which names are type parameters if no scope declares them (see
    /// [`Enclosing::undeclared_heads()`]).
    pub(crate) fn generic(&self, scopes: &[Scope], assumed: &Assumed) -> Vec<(Lifetime, usize)> {
        let mut generic: Vec<(Lifetime, usize)> = Vec::new();
        for (lifetime, depth) in self.active(scopes, assumed) {
            // What the lifetime names is what the scopes declared where it is written say.
            if is_generic_lifetime(visible(scopes, depth), assumed, &[], &lifetime)
                && !generic.iter().any(|(other, _)| *other == lifetime)
            {
                generic.push((lifetime, depth));
            }
        }
        generic
    }

    /// Returns the lifetimes (see [`Enclosing::generic()`], also those which are not generic).
    pub(crate) fn lifetimes(&self, scopes: &[Scope], assumed: &Assumed) -> Vec<Lifetime> {
        self.active(scopes, assumed)
            .into_iter()
            .map(|(lifetime, _)| lifetime)
            .collect()
    }

    /// Returns the first segments of the paths (`head::A<..>`) around the type which no scope
    /// declares, and which are not among the types assumed: the Referrer cannot tell whether they
    /// are type parameters of the Leaker (which the entry does not mention, so it does not know
    /// them either) or something else (a module, an enum, ..). Sorted and without duplicates.
    pub(crate) fn undeclared_heads(&self, scopes: &[Scope], assumed: &Assumed) -> Vec<String> {
        let mut heads: Vec<String> = Vec::new();
        for link in &self.0 {
            let (head, depth) = match link {
                Link::Projection { head, depth } => (head, *depth),
                Link::Lifetime { .. } => continue,
            };
            let declared = visible(scopes, depth)
                .iter()
                .any(|scope| scope.declares(head));
            if !declared && !assumed.types.contains(head) && !heads.contains(head) {
                heads.push(head.clone());
            }
        }
        heads.sort();
        heads
    }
}

/// Returns the scopes in scope where `depth` of them were declared.
fn visible(scopes: &[Scope], depth: usize) -> &[Scope] {
    &scopes[..depth.min(scopes.len())]
}

/// Returns the first segment of `path` if it may start a projection (`T::A<X>`, `Self::A<X>`): the
/// path is relative, has an associated item after it, and the segment has no arguments. Whether it
/// is a type parameter (or `Self`) is not told by the path.
fn projection_head(path: &Path) -> Option<&Ident> {
    if path.leading_colon.is_some() || path.segments.len() < 2 {
        return None;
    }
    path.segments
        .first()
        .filter(|seg| seg.arguments.is_none())
        .map(|seg| &seg.ident)
}

/// Returns `true` if `name` is a type parameter in `scopes`: declared by the innermost scope
/// which declares it, or, if no scope does, one of the generics `assumed` to be the Leaker's.
fn is_type_parameter(name: &str, scopes: &[Scope], assumed: &Assumed) -> bool {
    match scopes.iter().rev().find(|scope| scope.declares(name)) {
        Some(scope) => scope.params().any(
            |param| matches!(param, GenericParam::Type(param) if ident_name(&param.ident) == name),
        ),
        None => assumed.types.iter().any(|assumed| assumed == name),
    }
}

/// A generic which an outlives bound implied by a type is on (see [`implied_outlives()`]).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Bounded {
    /// The type parameter with the name.
    Type(String),
    /// The lifetime parameter.
    Lifetime(Lifetime),
    /// `Self`, which is not a parameter, but is encoded as if it was (see
    /// [`Context::self_outlives`]).
    SelfTy,
}

/// Returns `true` if `lifetime` is a lifetime parameter of the context of a type: not `'static`,
/// `'_` or a lifetime of a `for<..>` binder (one of `binders` inside the type, or a scope of
/// `scopes`), which the Repeater's implementation cannot declare. A lifetime which no scope
/// declares is a generic only if it is one of `assumed` (see [`Context::of()`]).
fn is_generic_lifetime(
    scopes: &[Scope],
    assumed: &Assumed,
    binders: &[Lifetime],
    lifetime: &Lifetime,
) -> bool {
    lifetime.ident != "static"
        && lifetime.ident != "_"
        && !binders.contains(lifetime)
        && scopes
            .iter()
            .rev()
            .find(|scope| scope.declares(&lifetime.to_string()))
            .map_or_else(
                || assumed.lifetimes.contains(lifetime),
                |scope| !scope.binder,
            )
}

/// The generics mentioned in a type, which `X: 'a` bounds by `'a` (see [`implied_outlives()`]).
struct Mentions<'a> {
    scopes: &'a [Scope],
    assumed: &'a Assumed,
    binders: Vec<Lifetime>,
    /// The first segments of the paths (the names which may be type parameters, and `Self`).
    types: Vec<String>,
    /// The generic lifetimes.
    lifetimes: Vec<Lifetime>,
}

impl<'a> Mentions<'a> {
    /// The generics mentioned in `ty`, if `wanted` (it may be skipped otherwise).
    fn of(ty: &Type, scopes: &'a [Scope], assumed: &'a Assumed, wanted: bool) -> Self {
        let mut mentions = Self::new(scopes, assumed, Vec::new());
        if wanted {
            mentions.visit_type(ty);
        }
        mentions
    }

    fn new(scopes: &'a [Scope], assumed: &'a Assumed, binders: Vec<Lifetime>) -> Self {
        Self {
            scopes,
            assumed,
            binders,
            types: Vec::new(),
            lifetimes: Vec::new(),
        }
    }
}

impl<'a, 'ast> Visit<'ast> for Mentions<'a> {
    fn visit_path(&mut self, i: &'ast Path) {
        if i.leading_colon.is_none() {
            self.types
                .extend(i.segments.first().map(|seg| ident_name(&seg.ident)));
        }
        syn::visit::visit_path(self, i)
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        if is_generic_lifetime(self.scopes, self.assumed, &self.binders, i) {
            self.lifetimes.push(i.clone());
        }
    }
    fn visit_type_impl_trait(&mut self, i: &'ast TypeImplTrait) {
        // Not made of the types in its bounds (it cannot be leaked, see `Enclosing`): only its
        // lifetime bounds count.
        for bound in &i.bounds {
            if let TypeParamBound::Lifetime(lifetime) = bound {
                self.visit_lifetime(lifetime);
            }
        }
    }
    fn visit_type_bare_fn(&mut self, i: &'ast TypeBareFn) {
        let outer = self.binders.len();
        self.binders.extend(binder_lifetimes(i.lifetimes.as_ref()));
        syn::visit::visit_type_bare_fn(self, i);
        self.binders.truncate(outer);
    }
    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        let outer = self.binders.len();
        self.binders.extend(binder_lifetimes(i.lifetimes.as_ref()));
        syn::visit::visit_trait_bound(self, i);
        self.binders.truncate(outer);
    }
    // The types in an expression are not components.
    fn visit_expr(&mut self, _: &'ast Expr) {}
    fn visit_macro(&mut self, _: &'ast Macro) {}
}

/// Returns the outlives bounds which `ty` implies on the generics in `params`, each with the
/// lifetime it is bounded by (`T: 'a` or `'b: 'a`): the type is well-formed only if they hold, so
/// they are added to the bounds of the generics, which the encoding of the generics carries to
/// make them implied wherever the type is written (see
/// [`encode_generics_params_to_ty()`](crate::encode_generics_params_to_ty())).
///
/// - A reference `&'a X` requires `X: 'a`.
/// - `ancestors` (the generic lifetimes among the [`Enclosing`] ones) require `ty: 'a`, which is
///   `X: 'a` for `ty` as `X` (a type leaked alone is a component of the type around it).
///
/// A path with a lifetime argument `R<'a, .., X, ..>` is not assumed to require `X: 'a` (it does
/// if the definition of `R` says so, e.g. `struct R<'a, T>(&'a T)`, but that cannot be seen, and
/// where it does not, as with `struct R<'a, T>(&'a (), T)`, an assumed bound would make the copy
/// stricter than the original): only the bounds which hold whatever the definitions say are
/// recorded, so `T: 'a` of such a type must be written. A lifetime which `ty` must outlive passes
/// through a path to its arguments, though (`&'a R<X>` requires `X: 'a`, see
/// [`Enclosing::components_of()`]).
///
/// `X: 'a` is `T: 'a` for each type parameter `T` mentioned in `X`, and `'b: 'a` for each
/// lifetime parameter `'b` mentioned in `X` other than `'a`, and `Self: 'a` if `X` mentions `Self`
/// (`Self`, `Self::Item`, `<Self as Tr>::Item`), when `has_self` (it has a meaning, so it is
/// encoded: `Self::Item: 'a` follows from `Self: 'a`). Everything in `X` is looked into, as rustc
/// does (a trait object `dyn Tr<T> + 'b` outlives `'a` if `'b` and its type arguments do), but
/// the lifetimes of a `for<..>` binder are not mentioned, and `'a` must be a lifetime parameter
/// (see [`is_generic_lifetime()`]).
fn implied_outlives(
    ty: &Type,
    scopes: &[Scope],
    assumed: &Assumed,
    params: &[GenericParam],
    has_self: bool,
    ancestors: &[Lifetime],
) -> Vec<(Bounded, Lifetime)> {
    /// The types containing a lifetime-carrying construct.
    struct Finder<'a> {
        scopes: &'a [Scope],
        assumed: &'a Assumed,
        params: &'a [GenericParam],
        has_self: bool,
        binders: Vec<Lifetime>,
        found: Vec<(Bounded, Lifetime)>,
    }
    impl<'a> Finder<'a> {
        /// Records the bounds implied by `X: 'a` where `X` is `referent`.
        fn require(&mut self, lifetime: &Lifetime, referent: &Type) {
            if !is_generic_lifetime(self.scopes, self.assumed, &self.binders, lifetime) {
                return;
            }
            self.require_generic(lifetime, referent)
        }

        /// Like [`Finder::require()`], for a lifetime which is known to be generic.
        fn require_generic(&mut self, lifetime: &Lifetime, referent: &Type) {
            let mut mentions = Mentions::new(self.scopes, self.assumed, self.binders.clone());
            mentions.visit_type(referent);
            if self.has_self && mentions.types.iter().any(|name| name == "Self") {
                self.found.push((Bounded::SelfTy, lifetime.clone()));
            }
            for name in mentions.types {
                let is_param = self.params.iter().any(|param| {
                    matches!(param, GenericParam::Type(param) if ident_name(&param.ident) == name)
                });
                if is_param {
                    self.found.push((Bounded::Type(name), lifetime.clone()));
                }
            }
            for other in mentions.lifetimes {
                let is_param = self.params.iter().any(
                    |param| matches!(param, GenericParam::Lifetime(param) if param.lifetime == other),
                );
                if is_param && &other != lifetime {
                    self.found
                        .push((Bounded::Lifetime(other), lifetime.clone()));
                }
            }
        }
    }
    impl<'a, 'ast> Visit<'ast> for Finder<'a> {
        fn visit_type_reference(&mut self, i: &'ast TypeReference) {
            if let Some(lifetime) = &i.lifetime {
                self.require(lifetime, &i.elem);
            }
            syn::visit::visit_type_reference(self, i)
        }
        fn visit_type_bare_fn(&mut self, i: &'ast TypeBareFn) {
            let outer = self.binders.len();
            self.binders.extend(binder_lifetimes(i.lifetimes.as_ref()));
            syn::visit::visit_type_bare_fn(self, i);
            self.binders.truncate(outer);
        }
        fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
            let outer = self.binders.len();
            self.binders.extend(binder_lifetimes(i.lifetimes.as_ref()));
            syn::visit::visit_trait_bound(self, i);
            self.binders.truncate(outer);
        }
        fn visit_expr(&mut self, _: &'ast Expr) {}
        fn visit_macro(&mut self, _: &'ast Macro) {}
    }
    let mut finder = Finder {
        scopes,
        assumed,
        params,
        has_self,
        binders: Vec::new(),
        found: Vec::new(),
    };
    finder.visit_type(ty);
    for lifetime in ancestors {
        finder.require_generic(lifetime, ty);
    }
    finder.found
}

/// Returns `true` if the bounds which `ty` implies (see [`implied_outlives()`], with the
/// lifetimes of `enclosing`) on the generics of `entry` which `scopes` do not declare, are bounds
/// of them in `entry`. The Referrer assumes those to be generics of the Leaker (see
/// [`Context::of()`], whose generics are only the declared ones), so it cannot compare their
/// declarations, but they must have at least the bounds the type implies where it is found: the
/// entry of a type found alone (`Rel2<T>`) differs from that of the same type found under a
/// reference (`&'a Rel2<T>`), where `T: 'a` as well.
pub(crate) fn assumed_bounds_hold(
    ty: &Type,
    scopes: &[Scope],
    assumed: &Assumed,
    enclosing: &Enclosing,
    entry: &Context,
) -> bool {
    let undeclared: Vec<GenericParam> = entry
        .generics()
        .params
        .iter()
        .filter(|param| {
            let name = generic_param_name(param);
            !scopes.iter().any(|scope| scope.declares(&name))
        })
        .cloned()
        .collect();
    if undeclared.is_empty() {
        return true;
    }
    let ancestors: Vec<Lifetime> = enclosing
        .generic(scopes, assumed)
        .into_iter()
        .map(|(lifetime, _)| lifetime)
        .collect();
    implied_outlives(ty, scopes, assumed, &undeclared, false, &ancestors)
        .iter()
        .all(|(bounded, lifetime)| {
            undeclared.iter().any(|param| match (param, bounded) {
                (GenericParam::Type(param), Bounded::Type(name))
                    if &ident_name(&param.ident) == name =>
                {
                    param.bounds.iter().any(
                        |bound| matches!(bound, TypeParamBound::Lifetime(other) if other == lifetime),
                    )
                }
                (GenericParam::Lifetime(param), Bounded::Lifetime(name))
                    if &param.lifetime == name =>
                {
                    param.bounds.iter().any(|other| other == lifetime)
                }
                _ => false,
            })
        })
}

/// Returns the lifetimes declared by a `for<..>` binder.
pub(crate) fn binder_lifetimes(
    lifetimes: Option<&BoundLifetimes>,
) -> impl Iterator<Item = Lifetime> + '_ {
    lifetimes
        .into_iter()
        .flat_map(|lifetimes| lifetimes.lifetimes.iter())
        .filter_map(|param| match param {
            GenericParam::Lifetime(param) => Some(param.lifetime.clone()),
            _ => None,
        })
}

/// Returns the canonical form of `generics` (see [`Context`]): the bounds of a single
/// parameter in the where clause are moved to the parameter, the bounds are sorted and
/// deduplicated, the parameters are sorted by kind (lifetimes, types, consts) and name, the other predicates are
/// sorted, and defaults and attributes are removed (an implementation cannot have them).
fn canonicalize(generics: Generics) -> Generics {
    let mut params: Vec<GenericParam> = generics.params.into_iter().collect();
    let mut predicates = Vec::new();
    for predicate in generics
        .where_clause
        .into_iter()
        .flat_map(|wc| wc.predicates)
    {
        // The parameter which the predicate bounds alone, if any.
        let target = match &predicate {
            WherePredicate::Type(PredicateType {
                lifetimes: None,
                bounded_ty: Type::Path(TypePath { qself: None, path }),
                ..
            }) => path.get_ident().and_then(|ident| {
                params.iter().position(|param| {
                    matches!(param, GenericParam::Type(param) if ident_name(&param.ident) == ident_name(ident))
                })
            }),
            WherePredicate::Lifetime(PredicateLifetime { lifetime, .. }) => {
                params.iter().position(|param| {
                    matches!(param, GenericParam::Lifetime(param) if &param.lifetime == lifetime)
                })
            }
            _ => None,
        };
        match (target, predicate) {
            (Some(pos), WherePredicate::Type(predicate)) => {
                if let GenericParam::Type(param) = &mut params[pos] {
                    param.bounds.extend(predicate.bounds);
                }
            }
            (Some(pos), WherePredicate::Lifetime(predicate)) => {
                if let GenericParam::Lifetime(param) = &mut params[pos] {
                    param.bounds.extend(predicate.bounds);
                }
            }
            (_, predicate) => predicates.push(predicate),
        }
    }
    for param in params.iter_mut() {
        match param {
            GenericParam::Type(param) => {
                param.attrs.clear();
                param.bounds = sort_bounds(std::mem::take(&mut param.bounds));
                param.colon_token = (!param.bounds.is_empty()).then(Default::default);
                param.eq_token = None;
                param.default = None;
            }
            GenericParam::Lifetime(param) => {
                param.attrs.clear();
                param.bounds = sort_bounds(std::mem::take(&mut param.bounds));
                param.colon_token = (!param.bounds.is_empty()).then(Default::default);
            }
            GenericParam::Const(param) => {
                param.attrs.clear();
                param.eq_token = None;
                param.default = None;
            }
        }
    }
    // Lifetimes, then types, then consts (the order valid on every version of Rust).
    params.sort_by_cached_key(|param| {
        (
            match param {
                GenericParam::Lifetime(_) => 0,
                GenericParam::Type(_) => 1,
                GenericParam::Const(_) => 2,
            },
            generic_param_name(param),
        )
    });
    let mut predicates: Vec<(String, WherePredicate)> = predicates
        .into_iter()
        .map(|mut predicate| {
            if let WherePredicate::Type(predicate) = &mut predicate {
                predicate.bounds = sort_bounds(std::mem::take(&mut predicate.bounds));
            }
            (quote!(#predicate).to_string(), predicate)
        })
        .collect();
    predicates.sort_by(|a, b| a.0.cmp(&b.0));
    predicates.dedup_by(|a, b| a.0 == b.0);
    let has_params = !params.is_empty();
    Generics {
        lt_token: has_params.then(Default::default),
        params: params.into_iter().collect(),
        gt_token: has_params.then(Default::default),
        where_clause: (!predicates.is_empty()).then(|| WhereClause {
            where_token: Default::default(),
            predicates: predicates.into_iter().map(|(_, p)| p).collect(),
        }),
    }
}

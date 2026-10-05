//! The generics (and the meaning of `Self`) which a leaked type depends on (see
//! [`Referrer::generics()`](crate::Referrer::generics())).

use crate::util::{
    generic_param_mentions_self, generic_param_name, ident_name, token_names, type_mentions_self,
    where_predicate_mentions_self,
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
        }
    }

    /// The scope of the lifetimes of a `for<..>` binder.
    pub(crate) fn binder(lifetimes: &BoundLifetimes) -> Self {
        Self {
            params: lifetimes.lifetimes.iter().cloned().collect(),
            predicates: Vec::new(),
        }
    }

    pub(crate) fn params(&self) -> impl Iterator<Item = &GenericParam> {
        self.params.iter()
    }
}

/// What `Self` means in a leaked type (see [`Referrer::context_self()`]).
///
/// A leaked type mentioning `Self` is the same type only where `Self` has the same meaning, so it
/// is part of the requirements of the type. The Repeater's implementation must be written for
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

/// The generics (with their bounds and the where predicates mentioning them) which a type depends
/// on, and what `Self` means if it mentions it, in a canonical form: two declarations with the
/// same meaning up to the order of the parameters, the bounds and the predicates, and to whether
/// a bound is written inline or in the where clause (e.g. `<T: Debug + Clone, U>` and `<U, T>
/// where T: Clone + Debug`), are equal.
///
/// If the type depends on the module where it is written (it has a relative path, which resolves
/// differently in another module) in an analysis of a module, the path of that module is part of
/// the requirements too (see
/// [`Referrer::context_module()`](crate::Referrer::context_module())).
///
/// Types with the same tokens and equal requirements are the same leaked type, whichever call of
/// [`Leaker::intern_with()`](crate::Leaker::intern_with()) found them.
#[derive(Clone, Debug)]
pub(crate) struct Requirements {
    generics: Generics,
    self_ty: Option<SelfRequirement>,
    /// The path of the module where the type is written, relative to the analyzed module (empty
    /// for the analyzed module itself). `None` for a type which does not depend on its module, or
    /// if no module is analyzed.
    module: Option<Vec<Ident>>,
}

impl Requirements {
    /// Returns the canonical form of `generics`, with `self_ty`.
    pub(crate) fn new(generics: Generics, self_ty: Option<SelfRequirement>) -> Self {
        Self {
            generics: canonicalize(generics),
            self_ty,
            module: None,
        }
    }

    /// Returns the requirements with the module `module` (see [`Requirements`]).
    pub(crate) fn with_module(mut self, module: Option<Vec<Ident>>) -> Self {
        self.module = module;
        self
    }

    /// Returns the tokens of the canonical form, by which requirements are compared.
    fn render(&self) -> String {
        let generics = &self.generics;
        let where_clause = &generics.where_clause;
        let mut rendered = quote!(#generics #where_clause).to_string();
        if let Some(self_ty) = &self.self_ty {
            rendered.push_str(&self_ty.render());
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
    /// Identifiers are matched by name, so a parameter may be included even if the name refers to
    /// something else (e.g. `::path::N`). This is harmless, because the Leaker and the Referrer
    /// include the same generics.
    pub(crate) fn of(ty: &Type, scopes: &[Scope], self_req: Option<&SelfRequirement>) -> Self {
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
        let params: Vec<GenericParam> = params.into_values().collect();
        let mut predicates: Vec<WherePredicate> = predicates.into_values().collect();
        predicates.extend(implied_outlives(ty, &params));
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
    }

    /// Returns the requirements of a type which depends on no generics and not on `Self`.
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

    pub(crate) fn module(&self) -> Option<&[Ident]> {
        self.module.as_deref()
    }

    /// Returns the requirements without the module.
    pub(crate) fn without_module(&self) -> Self {
        self.clone().with_module(None)
    }
}

impl PartialEq for Requirements {
    fn eq(&self, other: &Self) -> bool {
        self.render() == other.render()
    }
}

impl Eq for Requirements {}

impl PartialOrd for Requirements {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Requirements {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.render().cmp(&other.render())
    }
}

impl std::hash::Hash for Requirements {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.render().hash(state)
    }
}

/// The identifier of a context: a distinct set of requirements of the leaked types (the generics
/// a type uses, what `Self` means in it, and its module), see
/// [`Referrer::context()`](crate::Referrer::context()). It identifies the generics of a call of
/// [`Leaker::intern_with()`](crate::Leaker::intern_with()) too, as its top-level context.
///
/// The entries of a [`Referrer`](crate::Referrer) whose Repeater impls are written the same way
/// (they declare the same generics, are for the same `Self` and in the same module) have the same
/// id, also if they were found by different calls of
/// [`Leaker::intern_with()`](crate::Leaker::intern_with()). An id is assigned when the
/// requirements first appear in the [`Leaker`](crate::Leaker), counting from 0, and it is never
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

    /// Returns the number of the id, counting from 0 in the order in which the requirements first
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

/// The distinct [`Requirements`] of the leaked types, which are the contexts of the types (see
/// [`Referrer::context()`](crate::Referrer::context())), numbered from 0 in the order of their
/// first appearance. The number of a context is never changed by the calls which follow, as the
/// contexts are only added (but for those of a failed call, see [`Contexts::truncate()`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct Contexts {
    ids: HashMap<Requirements, ContextId>,
    list: Vec<Requirements>,
}

impl Contexts {
    /// Returns the id of `requirements`, which is added if it is new.
    pub(crate) fn intern(&mut self, requirements: Requirements) -> ContextId {
        if let Some(id) = self.ids.get(&requirements) {
            return *id;
        }
        let id = ContextId(self.list.len());
        self.ids.insert(requirements.clone(), id);
        self.list.push(requirements);
        id
    }

    pub(crate) fn len(&self) -> usize {
        self.list.len()
    }

    pub(crate) fn get(&self, id: ContextId) -> &Requirements {
        &self.list[id.0]
    }

    /// Removes the contexts numbered `len` or more (added after `len()` returned `len`).
    pub(crate) fn truncate(&mut self, len: usize) {
        for requirements in self.list.drain(len..) {
            self.ids.remove(&requirements);
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

/// Returns the predicates `X: 'a` which `ty` implies by containing `&'a X` (or `&'a mut X`) where
/// `'a` is a lifetime parameter in `params` and `X` mentions a type parameter in `params`: the
/// implementation must declare them, as the type is well-formed only if they hold.
fn implied_outlives(ty: &Type, params: &[GenericParam]) -> Vec<WherePredicate> {
    struct Finder<'a> {
        params: &'a [GenericParam],
        found: Vec<WherePredicate>,
    }
    impl<'a, 'ast> Visit<'ast> for Finder<'a> {
        fn visit_type_reference(&mut self, i: &'ast TypeReference) {
            if let Some(lifetime) = &i.lifetime {
                let is_param = self.params.iter().any(
                    |param| matches!(param, GenericParam::Lifetime(param) if &param.lifetime == lifetime),
                );
                let mut names = HashSet::new();
                let elem = &i.elem;
                token_names(quote!(#elem), &mut names);
                let mentions_type = self.params.iter().any(|param| {
                    matches!(param, GenericParam::Type(_))
                        && names.contains(&generic_param_name(param))
                });
                if is_param && mentions_type {
                    self.found.push(parse_quote!(#elem: #lifetime));
                }
            }
            syn::visit::visit_type_reference(self, i)
        }
    }
    let mut finder = Finder {
        params,
        found: Vec::new(),
    };
    finder.visit_type(ty);
    finder.found
}

/// Returns the canonical form of `generics` (see [`Requirements`]): the bounds of a single
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

//! The generics which a leaked type depends on, other than those of the Leaker itself (see
//! [`Referrer::local_generics()`](crate::Referrer::local_generics())).

use crate::util::{generic_param_name, token_names};
use std::collections::{BTreeMap, HashSet};
use syn::punctuated::Punctuated;
use syn::*;
use template_quote::quote;

/// The generics declared at one level inside an item, e.g. by a trait method, a GAT, a nested
/// item or a `for<..>` binder.
#[derive(Clone, Debug)]
pub(crate) struct Scope {
    params: Vec<GenericParam>,
    predicates: Vec<WherePredicate>,
}

impl Scope {
    pub(crate) fn new(generics: &Generics) -> Self {
        Self {
            params: generics.params.iter().cloned().collect(),
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

/// The generics (with their bounds and the where predicates mentioning them) which a type depends
/// on, in a canonical form: two declarations with the same meaning up to the order of the
/// parameters, the bounds and the predicates, and to whether a bound is written inline or in the
/// where clause (e.g. `<T: Debug + Clone, U>` and `<U, T> where T: Clone + Debug`), are equal.
#[derive(Clone, Debug)]
pub(crate) struct Requirements {
    generics: Generics,
    /// The tokens of the canonical form, by which requirements are compared.
    key: String,
}

impl Requirements {
    /// Returns the canonical form of `generics`.
    pub(crate) fn new(generics: Generics) -> Self {
        let generics = canonicalize(generics);
        let where_clause = &generics.where_clause;
        let key = quote!(#generics #where_clause).to_string();
        Self { generics, key }
    }

    /// Returns the generics declared in `scopes` (the innermost last) which `ty` depends on. A
    /// name refers to its innermost declaration visible from where it is used: from the type for
    /// the names in it, and from the declaring scope for the names in the bounds of a parameter.
    /// Names which are not declared in `scopes` (e.g. the generics of the Leaker, or paths) are
    /// ignored.
    ///
    /// Identifiers are matched by name, so a parameter may be included even if the name refers to
    /// something else (e.g. `::path::N`). This is harmless, because the Leaker and the Referrer
    /// include the same generics.
    pub(crate) fn of(ty: &Type, scopes: &[Scope]) -> Self {
        if scopes.is_empty() {
            return Self::none();
        }
        let mut names = HashSet::new();
        token_names(quote!(#ty), &mut names);
        // The names to resolve, each with the innermost scope it is visible from.
        let mut work: Vec<(usize, String)> = names
            .into_iter()
            .map(|name| (scopes.len() - 1, name))
            .collect();
        let mut params = BTreeMap::new();
        let mut predicates = BTreeMap::new();
        while let Some((from, name)) = work.pop() {
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
                }
            }
        }
        let mut generics = Generics {
            params: params.into_values().collect(),
            ..Default::default()
        };
        if !predicates.is_empty() {
            generics.where_clause = Some(WhereClause {
                where_token: Default::default(),
                predicates: predicates.into_values().collect(),
            });
        }
        Self::new(generics)
    }

    /// Returns the requirements of a type which depends on no generics of `scopes`.
    pub(crate) fn none() -> Self {
        Self::new(Generics::default())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.generics.params.is_empty() && self.generics.where_clause.is_none()
    }

    pub(crate) fn generics(&self) -> &Generics {
        &self.generics
    }

    pub(crate) fn key(&self) -> &str {
        &self.key
    }
}

impl PartialEq for Requirements {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
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
        self.key.cmp(&other.key)
    }
}

impl std::hash::Hash for Requirements {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state)
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

/// Returns the canonical form of `generics` (see [`Requirements`]): the bounds of a single
/// parameter in the where clause are moved to the parameter, the bounds are sorted and
/// deduplicated, the parameters are sorted by name (lifetimes first), the other predicates are
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
                    matches!(param, GenericParam::Type(param) if &param.ident == ident)
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
    params.sort_by_cached_key(|param| {
        (
            !matches!(param, GenericParam::Lifetime(_)),
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

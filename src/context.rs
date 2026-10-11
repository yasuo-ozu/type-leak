//! The context of a leaked type: its generics, what `Self` means, and its module.

use crate::util::{
    canonical_bounds, generic_param_mentions_self, generic_param_name, ident_name, self_in_param,
    self_in_predicate, token_names, type_mentions_self, unwrap_type_group,
    where_predicate_mentions_self, ReplaceSelf, SelfMentions,
};
use proc_macro2::{Literal, TokenStream};
use std::collections::{BTreeMap, HashMap, HashSet};
use syn::fold::Fold;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::{quote, ToTokens};

/// The generics declared at one level: those of the Leaker, or local ones.
#[derive(Clone, Debug)]
pub(crate) struct Scope {
    params: Vec<GenericParam>,
    predicates: Vec<WherePredicate>,
    binder: bool,
    /// What `Self` meant where the generics were declared, which their bounds and where predicates
    /// are written for. `None` where it has no meaning. See [`Context::of()`].
    self_ty: Option<SelfRequirement>,
    /// The generic associated types of the trait that these generics belong to, if the Leaker can
    /// see them (see [`Gat`]).
    gats: Vec<Gat>,
}

impl Scope {
    /// `self_ty` is what `Self` means where `generics` are declared. Defaults are dropped: an impl
    /// cannot copy them and the type does not depend on them.
    pub(crate) fn new(generics: &Generics, self_ty: Option<SelfRequirement>) -> Self {
        Self {
            params: generics.params.iter().cloned().map(strip_default).collect(),
            predicates: generics
                .where_clause
                .iter()
                .flat_map(|wc| wc.predicates.iter().cloned())
                .collect(),
            binder: false,
            self_ty,
            gats: Vec::new(),
        }
    }

    pub(crate) fn binder(lifetimes: &BoundLifetimes) -> Self {
        Self {
            params: lifetimes.lifetimes.iter().cloned().collect(),
            predicates: Vec::new(),
            binder: true,
            self_ty: None,
            gats: Vec::new(),
        }
    }

    /// Sets the generic associated types of the trait of the scope (see [`Gat`]).
    pub(crate) fn with_gats(mut self, gats: Vec<Gat>) -> Self {
        self.gats = gats;
        self
    }

    /// Like [`Scope::with_gats()`], for a scope that exists before they are known.
    pub(crate) fn set_gats(&mut self, gats: Vec<Gat>) {
        self.gats = gats;
    }

    fn gat(&self, name: &str) -> Option<&Gat> {
        self.gats.iter().find(|gat| ident_name(&gat.ident) == name)
    }

    /// The generic associated types of the trait of the scope (see [`Gat`]).
    pub(crate) fn gats(&self) -> &[Gat] {
        &self.gats
    }

    /// Sets what `Self` meant where the generics were declared, for a scope that exists before it
    /// is known.
    pub(crate) fn set_self(&mut self, self_ty: Option<SelfRequirement>) {
        self.self_ty = self_ty;
    }

    /// Like [`Scope::set_self()`], returning what `Self` meant before.
    pub(crate) fn replace_self(
        &mut self,
        self_ty: Option<SelfRequirement>,
    ) -> Option<SelfRequirement> {
        std::mem::replace(&mut self.self_ty, self_ty)
    }

    /// Returns `true` if the generics are the same, whatever `Self` meant where they were declared.
    pub(crate) fn same_generics(&self, other: &Scope) -> bool {
        self.params == other.params
            && self.predicates == other.predicates
            && self.binder == other.binder
    }

    pub(crate) fn params(&self) -> impl Iterator<Item = &GenericParam> {
        self.params.iter()
    }

    pub(crate) fn predicates(&self) -> impl Iterator<Item = &WherePredicate> {
        self.predicates.iter()
    }

    /// The generics of the scope, as declared.
    pub(crate) fn generics(&self) -> Generics {
        Generics {
            params: self.params.iter().cloned().collect(),
            where_clause: (!self.predicates.is_empty()).then(|| WhereClause {
                where_token: Default::default(),
                predicates: self.predicates.iter().cloned().collect(),
            }),
            ..Default::default()
        }
    }

    pub(crate) fn self_ty(&self) -> Option<&SelfRequirement> {
        self.self_ty.as_ref()
    }

    pub(crate) fn declares(&self, name: &str) -> bool {
        self.params
            .iter()
            .any(|param| generic_param_name(param) == name)
    }
}

/// What `Self` means in a context (`Referrer::context_self()`). The Repeater impl of an entry that
/// names `Self` is written for the type described here. See [Self rules](crate::rules#self-rules).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// let in_struct = leaker
///     .intern_struct(&parse_quote!(struct S<T> { a: Rel<Self> }), true)
///     .unwrap();
/// let in_trait = leaker
///     .intern_trait(&parse_quote!(trait Tr<T> { fn f(&self) -> Rel<Self>; }), true)
///     .unwrap();
/// let referrer = leaker.finish();
/// // In a struct, `Self` is the struct with its own generics as arguments.
/// match referrer.context_self(in_struct) {
///     Some(SelfRequirement::Type(ty)) => assert_eq!(*ty, parse_quote!(S<T>)),
///     _ => unreachable!(),
/// }
/// // In a trait, `Self` is any type that implements the trait.
/// match referrer.context_self(in_trait) {
///     Some(SelfRequirement::Bound(path)) => assert_eq!(*path, parse_quote!(Tr<T>)),
///     _ => unreachable!(),
/// }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SelfRequirement {
    /// `Self` is any implementor of the trait, e.g. `Self: MyTrait<T, U>`.
    Bound(Path),
    /// `Self` is this type: a struct, enum, union or the self type of an impl, e.g. `MyStruct<T>`.
    Type(Type),
}

impl SelfRequirement {
    pub(crate) fn of_trait(ident: &Ident, generics: &Generics) -> Self {
        let args = generics_as_args(generics);
        Self::Bound(parse2(quote!(#ident #args)).unwrap())
    }

    pub(crate) fn of_item(ident: &Ident, generics: &Generics) -> Self {
        let args = generics_as_args(generics);
        Self::Type(parse2(quote!(#ident #args)).unwrap())
    }

    fn names(&self, names: &mut HashSet<String>) {
        match self {
            Self::Bound(path) => token_names(quote!(#path), names),
            Self::Type(ty) => token_names(quote!(#ty), names),
        }
    }

    fn render(&self) -> String {
        match self {
            Self::Bound(path) => format!(" Self: {}", quote!(#path)),
            Self::Type(ty) => format!(" Self = {}", quote!(#ty)),
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

/// An associated type of a trait that has outlives bounds in its where clause or on its
/// parameters (`type G<'b> where Self: 'b;`, `type G<'b, U: 'b>;`), as the Leaker reads it from the
/// trait it analyzes.
///
/// `Self::G<'x>` is well-formed only if `Self: 'x`, which the signature of the original implies and
/// the impl of the Repeater, which has no signature, does not have: the entry requires it (see
/// [`implied_outlives()`]). Only the outlives relations are kept: any other predicate must be
/// written by the method that names the type, as rustc requires. The parameters have no bounds.
#[derive(Clone, Debug)]
pub(crate) struct Gat {
    ident: Ident,
    /// The parameters of the type, and a where clause of its outlives relations.
    generics: Generics,
}

impl Gat {
    /// The associated types of `items` that have outlives relations.
    pub(crate) fn of_items(items: &[TraitItem]) -> Vec<Self> {
        items
            .iter()
            .filter_map(|item| match item {
                TraitItem::Type(item) => Self::new(&item.ident, &item.generics),
                _ => None,
            })
            .collect()
    }

    /// Returns the outlives relations of `generics` (the bounds of the parameters and the where
    /// clause) as the where clause of an associated type, or `None` if it has none.
    pub(crate) fn new(ident: &Ident, generics: &Generics) -> Option<Self> {
        let lifetimes = |bounds: &Punctuated<TypeParamBound, Token![+]>| -> Vec<TypeParamBound> {
            bounds
                .iter()
                .filter(|bound| matches!(bound, TypeParamBound::Lifetime(_)))
                .cloned()
                .collect()
        };
        let mut params = Punctuated::new();
        let mut predicates: Vec<WherePredicate> = Vec::new();
        for param in &generics.params {
            match param {
                GenericParam::Lifetime(param) => {
                    let lifetime = &param.lifetime;
                    let bounds = &param.bounds;
                    if !bounds.is_empty() {
                        predicates.push(parse_quote!(#lifetime: #bounds));
                    }
                    params.push(GenericParam::Lifetime(LifetimeParam::new(lifetime.clone())));
                }
                GenericParam::Type(param) => {
                    let (name, bounds) = (&param.ident, lifetimes(&param.bounds));
                    if !bounds.is_empty() {
                        predicates.push(parse_quote!(#name: #(#bounds)+*));
                    }
                    params.push(GenericParam::Type(TypeParam::from(name.clone())));
                }
                GenericParam::Const(param) => {
                    let (name, ty) = (&param.ident, &param.ty);
                    params.push(parse_quote!(const #name: #ty));
                }
            }
        }
        for predicate in generics
            .where_clause
            .iter()
            .flat_map(|where_clause| where_clause.predicates.iter())
        {
            match predicate {
                // A bound under `for<..>` is not a relation of the context.
                WherePredicate::Type(PredicateType {
                    lifetimes: None,
                    bounded_ty,
                    bounds,
                    ..
                }) => {
                    let bounds = lifetimes(bounds);
                    if !bounds.is_empty() {
                        predicates.push(parse_quote!(#bounded_ty: #(#bounds)+*));
                    }
                }
                WherePredicate::Lifetime(predicate) if !predicate.bounds.is_empty() => {
                    predicates.push(WherePredicate::Lifetime(predicate.clone()))
                }
                _ => (),
            }
        }
        if predicates.is_empty() {
            return None;
        }
        Some(Self {
            ident: ident.clone(),
            generics: Generics {
                params,
                where_clause: Some(WhereClause {
                    where_token: Default::default(),
                    predicates: predicates.into_iter().collect(),
                }),
                ..Default::default()
            },
        })
    }

    /// Reads `type G<'b> where Self: 'b ;` (see [`ToTokens`]).
    pub(crate) fn parse(input: ParseStream) -> Result<Self> {
        let span = input.span();
        input.parse::<Token![type]>()?;
        let ident: Ident = input.parse()?;
        let mut generics: Generics = input.parse()?;
        generics.where_clause = input.parse()?;
        input.parse::<Token![;]>()?;
        Self::new(&ident, &generics).ok_or_else(|| {
            Error::new(
                span,
                "an associated type needs an outlives relation in its where clause",
            )
        })
    }

    fn render(&self) -> String {
        let (ident, generics) = (&self.ident, &self.generics);
        let where_clause = &generics.where_clause;
        quote!(type #ident #generics #where_clause ;).to_string()
    }

    /// Returns the outlives relations that `G<..>` requires, as where predicates of the context:
    /// those of the where clause of the type, with its parameters substituted by `args`, the
    /// arguments of the projection, and the `trait_params` of the trait by `trait_args`, the
    /// arguments of the trait in the path (the parameters are kept if there are none). There are
    /// none if the arguments are not those of the parameters (a lifetime that is elided in the
    /// path): nothing can be written for them.
    fn requirements(
        &self,
        args: &[GenericArgument],
        trait_params: &[GenericParam],
        trait_args: &[GenericArgument],
    ) -> Vec<WherePredicate> {
        let mut substitution = Substitution::default();
        let params: Vec<&GenericParam> = self.generics.params.iter().collect();
        if !substitution.add(&params, args) {
            return Vec::new();
        }
        if !trait_args.is_empty() {
            let trait_params: Vec<&GenericParam> = trait_params.iter().collect();
            if !substitution.add(&trait_params, trait_args) {
                return Vec::new();
            }
        }
        self.generics
            .where_clause
            .iter()
            .flat_map(|where_clause| where_clause.predicates.iter())
            .map(|predicate| substitution.fold_where_predicate(predicate.clone()))
            .collect()
    }
}

impl ToTokens for Gat {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let (ident, generics) = (&self.ident, &self.generics);
        tokens.extend(quote!(type #ident #generics #{&generics.where_clause} ;));
    }
}

/// The arguments that replace the parameters of a generic associated type and of its trait.
#[derive(Default)]
struct Substitution {
    lifetimes: HashMap<String, Lifetime>,
    types: HashMap<String, Type>,
}

impl Substitution {
    /// Returns `false` if `args` are not the arguments of `params`.
    fn add(&mut self, params: &[&GenericParam], args: &[GenericArgument]) -> bool {
        if params.len() != args.len() {
            return false;
        }
        for (param, arg) in params.iter().zip(args) {
            match (param, arg) {
                (GenericParam::Lifetime(param), GenericArgument::Lifetime(arg)) => {
                    self.lifetimes
                        .insert(param.lifetime.to_string(), arg.clone());
                }
                (GenericParam::Type(param), GenericArgument::Type(arg)) => {
                    self.types.insert(ident_name(&param.ident), arg.clone());
                }
                // A const argument is no part of an outlives relation.
                (GenericParam::Const(_), _) => (),
                _ => return false,
            }
        }
        true
    }
}

impl Fold for Substitution {
    fn fold_lifetime(&mut self, lifetime: Lifetime) -> Lifetime {
        self.lifetimes
            .get(&lifetime.to_string())
            .cloned()
            .unwrap_or(lifetime)
    }

    fn fold_type(&mut self, ty: Type) -> Type {
        if let Type::Path(TypePath { qself: None, path }) = &ty {
            let first = path.segments.first();
            if let Some(arg) = first
                .filter(|seg| path.leading_colon.is_none() && seg.arguments.is_none())
                .and_then(|seg| self.types.get(&ident_name(&seg.ident)))
            {
                if path.segments.len() == 1 {
                    return arg.clone();
                }
                // `U::Item` is that of the argument, if it is a path.
                return match arg {
                    Type::Path(TypePath {
                        qself: None,
                        path: arg_path,
                    }) => {
                        let mut segments = arg_path.segments.clone();
                        segments.extend(path.segments.iter().skip(1).cloned());
                        Type::Path(TypePath {
                            qself: None,
                            path: Path {
                                leading_colon: arg_path.leading_colon,
                                segments,
                            },
                        })
                    }
                    _ => ty,
                };
            }
        }
        syn::fold::fold_type(self, ty)
    }
}

/// Returns the outlives relations that the generic associated type named by `projection` requires
/// (see [`Gat`]), if it is one of the trait that `self_req` is the implementor of: `Self::G<..>`,
/// or `<Self as Tr<..>>::G<..>` where `Tr` is that trait. The Leaker sees it if it analyzed the
/// trait with its items. The type of a supertrait, and that of a generic (`T::G<'x>`), are not
/// seen: nothing is recorded for them.
fn gat_requirements(
    projection: &TypePath,
    scopes: &[Scope],
    self_req: Option<&SelfRequirement>,
) -> Vec<WherePredicate> {
    let trait_path = match self_req {
        Some(SelfRequirement::Bound(path)) => path,
        _ => return Vec::new(),
    };
    if scopes.iter().all(|scope| scope.gats().is_empty()) {
        return Vec::new();
    }
    let segments: Vec<&PathSegment> = projection.path.segments.iter().collect();
    let (gat, trait_args) = match &projection.qself {
        // `Self::G<..>`
        None => match segments[..] {
            [first, gat]
                if projection.path.leading_colon.is_none()
                    && first.ident == "Self"
                    && first.arguments.is_none() =>
            {
                (gat, None)
            }
            _ => return Vec::new(),
        },
        // `<Self as Tr<..>>::G<..>`
        Some(qself) => {
            let is_self = matches!(unwrap_type_group(&qself.ty),
                Type::Path(TypePath { qself: None, path }) if path.is_ident("Self"));
            if !is_self || qself.position == 0 || segments.len() != qself.position + 1 {
                return Vec::new();
            }
            let tr = segments[qself.position - 1];
            if trait_path.segments.last().map(|seg| &seg.ident) != Some(&tr.ident) {
                return Vec::new();
            }
            (segments[qself.position], Some(&tr.arguments))
        }
    };
    let angle = |arguments: &PathArguments| -> Option<Vec<GenericArgument>> {
        match arguments {
            PathArguments::None => Some(Vec::new()),
            PathArguments::AngleBracketed(args) => Some(args.args.iter().cloned().collect()),
            PathArguments::Parenthesized(_) => None,
        }
    };
    let (args, trait_args) = match (angle(&gat.arguments), trait_args.map(angle)) {
        (Some(args), None) => (args, Vec::new()),
        (Some(args), Some(Some(trait_args))) => (args, trait_args),
        _ => return Vec::new(),
    };
    let name = ident_name(&gat.ident);
    let scope = scopes
        .iter()
        .rev()
        .find(|scope| scope.self_ty() == self_req && scope.gat(&name).is_some());
    match scope.and_then(|scope| Some((scope, scope.gat(&name)?))) {
        Some((scope, gat)) => {
            let trait_params: Vec<GenericParam> = scope.params().cloned().collect();
            // A trait with no arguments written takes its parameters as they are.
            if !trait_args.is_empty() && trait_args.len() != trait_params.len() {
                return Vec::new();
            }
            gat.requirements(&args, &trait_params, &trait_args)
        }
        None => Vec::new(),
    }
}

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

/// The context of a leaked type: its generics, what `Self` means, and its module.
/// The generics are in a canonical form, so equal declarations compare equal.
#[derive(Clone, Debug)]
pub(crate) struct Context {
    generics: Generics,
    self_ty: Option<SelfRequirement>,
    /// Lifetimes `'a` with `Self: 'a`, implied or written. Sorted, and empty without `self_ty`.
    /// They are not in `generics`, because `Self` is not a parameter.
    self_outlives: Vec<Lifetime>,
    /// The generic associated types of the trait, for the top-level context of a trait scope whose
    /// entries depend on them (see [`Gat`]). Sorted by name, and empty in the context of an entry.
    gats: Vec<Gat>,
    /// The module scope, if the type depends on its module.
    module: Option<ContextId>,
    /// Set for the context of a module scope: it makes each one distinct.
    scope: Option<usize>,
    /// The `Self` of the scopes whose bounds were rewritten for another `Self` (see
    /// [`Context::of()`]). Not part of the identity: it only tells the scope that declares one of
    /// them that this context depends on it, so its top-level context records it.
    substituted: Vec<SelfRequirement>,
}

/// A bound that mentions the `Self` of an outer scope, which cannot be written where `Self` is
/// something else (see [`Context::of()`]).
#[derive(Clone, Debug)]
pub(crate) struct UnnameableSelf {
    /// The mention of `Self` that cannot be written.
    pub(crate) cause: TokenStream,
    /// Tells what is wrong, as the headline of the error: the problem is the bound, not the path
    /// of the type that uses it.
    pub(crate) headline: String,
    /// Tells why, and where `Self` is something else.
    pub(crate) note: String,
}

impl Context {
    pub(crate) fn new(generics: Generics, self_ty: Option<SelfRequirement>) -> Self {
        Self {
            generics: canonicalize(generics),
            self_ty,
            self_outlives: Vec::new(),
            gats: Vec::new(),
            module: None,
            scope: None,
            substituted: Vec::new(),
        }
    }

    /// `number` is the index in [`Contexts`], so it is the same in the Referrer.
    pub(crate) fn module_scope(number: usize) -> Self {
        let mut context = Self::none();
        context.scope = Some(number);
        context
    }

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

    /// Returns `true` if a bound of this context was rewritten for another `Self`, the one that
    /// `declared` is.
    pub(crate) fn substitutes(&self, declared: &SelfRequirement) -> bool {
        self.substituted.contains(declared)
    }

    /// The generic associated types of the trait that this top-level context is the scope of. They
    /// are part of its identity, and the Referrer's scope has them (see [`Gat`]). A context where
    /// `Self` has no meaning has none.
    pub(crate) fn with_gats(mut self, mut gats: Vec<Gat>) -> Self {
        gats.sort_by_key(|gat| ident_name(&gat.ident));
        self.gats = if self.self_ty.is_some() {
            gats
        } else {
            Vec::new()
        };
        self
    }

    pub(crate) fn with_module(mut self, module: Option<ContextId>) -> Self {
        self.module = module;
        self
    }

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
        for gat in &self.gats {
            rendered.push(' ');
            rendered.push_str(&gat.render());
        }
        if let Some(module) = &self.module {
            rendered.push_str(&format!(" mod #{}", module.index()));
        }
        if let Some(scope) = &self.scope {
            rendered.push_str(&format!(" scope #{}", scope));
        }
        rendered
    }

    /// Returns the context of `ty` in `scopes` (outermost first). Names match by name only, which
    /// is harmless: the Referrer computes the same. `enclosing` lifetimes become generics even if
    /// `ty` does not name them.
    ///
    /// `self_req` is what `Self` means where `ty` is written. The bounds and where predicates of a
    /// generic are written for the `Self` of the scope that declares it. If that is not `self_req`
    /// (a nested impl, struct or trait has a `Self` of its own), the bounds that mention `Self` are
    /// rewritten for the one that declared them: `T: Tr<Self>` of `struct S<T: Tr<Self>>` is
    /// `T: Tr<S<T>>` where `Self` is something else, and the context does not depend on `Self`.
    /// That is impossible if the `Self` of the scope is an implementor of a trait, or if the
    /// bound names an associated item of `Self` (`Self::X`): it fails with [`UnnameableSelf`]. The
    /// Referrer has the same scopes with the same `Self`, so it computes the same context.
    pub(crate) fn of(
        ty: &Type,
        scopes: &[Scope],
        self_req: Option<&SelfRequirement>,
        enclosing: &Enclosing,
    ) -> std::result::Result<Self, UnnameableSelf> {
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
        let mut work: Vec<(usize, String)> = names
            .into_iter()
            .filter_map(|name| Some((scopes.len().checked_sub(1)?, name)))
            .collect();
        // An enclosing lifetime is a generic only if the type has a generic to bound by it.
        let ancestors = enclosing.generic(scopes);
        let self_bound = matches!(self_req, Some(SelfRequirement::Bound(_)));
        let mentions = Mentions::of(ty, scopes, self_bound, !ancestors.is_empty());
        for (lifetime, depth) in &ancestors {
            let is_type_param = |name: &String| {
                scopes.iter().any(|scope| {
                    scope.params().any(|param| {
                        matches!(param, GenericParam::Type(param) if &ident_name(&param.ident) == name)
                    })
                })
            };
            let bounds_something = (self_req.is_some()
                && mentions.types.iter().any(|name| name == "Self"))
                || mentions.types.iter().any(is_type_param)
                || !mentions.projections.is_empty()
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
        let mut swept_self = false;
        // The `Self` of the scope `ix`, if it is not the one where `ty` is written.
        let foreign = |ix: usize| {
            scopes[ix]
                .self_ty()
                .filter(|outer| Some(*outer) != self_req)
        };
        let mut substituted: Vec<SelfRequirement> = Vec::new();
        loop {
            let (from, name) = match work.pop() {
                Some(item) => item,
                None if mentions_self && !swept_self => {
                    // `where Self: Sized` belongs to the context: the impl must declare it too.
                    swept_self = true;
                    for (ix, scope) in scopes.iter().enumerate() {
                        // What bounds another `Self` is no business of this one.
                        if foreign(ix).is_some() {
                            continue;
                        }
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
            // The bounds are written for the `Self` of the scope, which may not be this one.
            let outer = foreign(ix);
            let mut param = scopes[ix].params[pos].clone();
            if let Some(outer) = outer.filter(|_| generic_param_mentions_self(&param)) {
                param = for_self_of(outer, &param)?;
                if !substituted.contains(outer) {
                    substituted.push(outer.clone());
                }
            }
            let mut names = HashSet::new();
            token_names(quote!(#param), &mut names);
            work.extend(names.into_iter().map(|name| (ix, name)));
            let mut found_self = generic_param_mentions_self(&param);
            params.insert((ix, pos), param);
            for (k, predicate) in scopes[ix].predicates.iter().enumerate() {
                if predicates.contains_key(&(ix, k)) {
                    continue;
                }
                let mut names = HashSet::new();
                token_names(quote!(#predicate), &mut names);
                // A predicate on a generic of the scope is a bound of it, as `canonicalize()` makes
                // it: it is in the context of what uses that generic, not of what its arguments
                // name. So `T: Iterator<Item = Rel<'a>>` is no more in the context of `Rel<'a>`
                // than the inline bound is, which the Referrer sees. Any other predicate belongs to
                // each generic it names.
                let belongs = match bounded_generic(predicate) {
                    Some(bounded) if scopes[ix].declares(&bounded) => bounded == name,
                    _ => names.contains(&name),
                };
                if belongs {
                    let mut predicate = predicate.clone();
                    if let Some(outer) = outer.filter(|_| where_predicate_mentions_self(&predicate))
                    {
                        predicate = for_self_of(outer, &predicate)?;
                        if !substituted.contains(outer) {
                            substituted.push(outer.clone());
                        }
                        names.clear();
                        token_names(quote!(#predicate), &mut names);
                    }
                    work.extend(names.into_iter().map(|name| (ix, name)));
                    found_self |= where_predicate_mentions_self(&predicate);
                    predicates.insert((ix, k), predicate);
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
        // A written `Self: 'a` moves into `self_outlives`, so written and implied give one context.
        if self_req.is_some() {
            self_outlives.extend(written_self_outlives(&mut predicates, &params));
        }
        let env = Binding::of(&params, &predicates);
        for (bounded, lifetime) in implied_outlives(ty, scopes, &params, &env, self_req, &ancestors)
        {
            match bounded {
                Bounded::SelfTy => {
                    self_outlives.push(lifetime);
                    continue;
                }
                Bounded::Projection(projection) => {
                    predicates.push(parse_quote!(#projection: #lifetime));
                    continue;
                }
                _ => (),
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
        let mut context = Self::new(generics, self_req.filter(|_| mentions_self).cloned())
            .with_self_outlives(self_outlives);
        context.substituted = substituted;
        Ok(context)
    }

    pub(crate) fn none() -> Self {
        Self::new(Generics::default(), None)
    }

    /// No generics and no `Self`, whichever the module.
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

    pub(crate) fn gats(&self) -> &[Gat] {
        &self.gats
    }

    pub(crate) fn module(&self) -> Option<ContextId> {
        self.module
    }

    pub(crate) fn is_module_scope(&self) -> bool {
        self.scope.is_some()
    }

    pub(crate) fn without_module(&self) -> Self {
        self.clone().with_module(None)
    }

    /// Returns `true` if `entry` is this context plus where predicates that name none of the
    /// generics (bounds on `Self` alone, such as `Self: Sized`). A copy that writes `Self`
    /// otherwise, as a type parameter, cannot repeat them.
    pub(crate) fn is_entry_with_extra_self_predicates(&self, entry: &Context) -> bool {
        let show = |tokens: &dyn ToTokens| tokens.to_token_stream().to_string();
        if self.self_ty != entry.self_ty
            || self.self_outlives != entry.self_outlives
            || self.module != entry.module
            || self.scope != entry.scope
        {
            return false;
        }
        let (params, entry_params) = (&self.generics.params, &entry.generics.params);
        if params.len() != entry_params.len()
            || params
                .iter()
                .zip(entry_params)
                .any(|(a, b)| show(a) != show(b))
        {
            return false;
        }
        let predicates = |context: &Context| -> Vec<(String, WherePredicate)> {
            context
                .generics
                .where_clause
                .iter()
                .flat_map(|where_clause| where_clause.predicates.iter())
                .map(|predicate| (show(predicate), predicate.clone()))
                .collect()
        };
        let (mine, theirs) = (predicates(self), predicates(entry));
        let names: HashSet<String> = entry_params.iter().map(generic_param_name).collect();
        mine.iter()
            .all(|(text, _)| theirs.iter().any(|(other, _)| other == text))
            && theirs.iter().all(|(text, predicate)| {
                let mut used = HashSet::new();
                token_names(predicate.to_token_stream(), &mut used);
                mine.iter().any(|(other, _)| other == text)
                    || !used.iter().any(|name| names.contains(name))
            })
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.render() == other.render()
    }
}

impl Eq for Context {}

impl std::hash::Hash for Context {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.render().hash(state)
    }
}

/// Identifies a [context](crate::rules#contexts): what gives a leaked type its meaning, which is
/// the generics it uses, what `Self` is, and the module it is written in.
///
/// Every analysis method of the [`Leaker`](crate::Leaker) returns the id of the top-level context
/// of what it analyzed (keep it: [`Referrer::visitor()`](crate::Referrer::visitor) needs it), and
/// [`Referrer::context()`](crate::Referrer::context) returns the id of the context of an entry.
/// Ids count from 0 in order of first appearance and are never renumbered, so the Referrer may
/// skip some. Entries with the same generics, `Self` and module share an id. `ToTokens` writes it
/// as an integer literal, and `Parse` reads one with or without `usize`.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let item: ItemStruct = parse_quote!(struct S<T> { a: Wrap<T>, b: Rel });
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_struct(&item, true).unwrap();
/// let referrer = leaker.finish();
/// // `Wrap<T>` uses the generics of the struct, as the top-level context has them.
/// // `Rel` uses no generic, so its context is another one.
/// let wrap = referrer.iter().position(|ty| *ty == parse_quote!(Wrap<T>)).unwrap();
/// let rel = referrer.iter().position(|ty| *ty == parse_quote!(Rel)).unwrap();
/// assert_eq!(referrer.context(wrap), top_level);
/// assert_ne!(referrer.context(rel), top_level);
///
/// // It is written as an integer literal, so it travels with the Referrer in a macro call.
/// # use template_quote::ToTokens;
/// let id: ContextId = parse_str("3usize").unwrap();
/// assert_eq!(id.index(), 3);
/// assert_eq!(id.to_token_stream().to_string(), "3");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextId(usize);

impl ContextId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the number of the id, counted from 0 in order of first appearance.
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

/// What is known of a module scope. Its [`ContextId`] is the id of the module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModuleInfo {
    /// `None` for a module that `enter_mod()` enters itself (see `Referrer::module_path()`).
    pub(crate) path: Option<Vec<Ident>>,
    pub(crate) hint: Option<String>,
    pub(crate) parent: Option<ContextId>,
}

/// The distinct [`Context`]s, numbered in order of first appearance. They are never renumbered,
/// except by [`Contexts::truncate()`] for a failed scope.
#[derive(Clone, Debug, Default)]
pub(crate) struct Contexts {
    ids: HashMap<Context, ContextId>,
    list: Vec<Context>,
    modules: BTreeMap<ContextId, ModuleInfo>,
}

impl Contexts {
    pub(crate) fn intern(&mut self, context: Context) -> ContextId {
        if let Some(id) = self.ids.get(&context) {
            return *id;
        }
        let id = ContextId(self.list.len());
        self.ids.insert(context.clone(), id);
        self.list.push(context);
        id
    }

    /// The module number is the context index, so ids are the same for the same calls.
    pub(crate) fn intern_module(&mut self, info: ModuleInfo) -> ContextId {
        let id = self.intern(Context::module_scope(self.list.len()));
        self.modules.insert(id, info);
        id
    }

    pub(crate) fn module(&self, id: ContextId) -> Option<&ModuleInfo> {
        self.modules.get(&id)
    }

    pub(crate) fn find(&self, context: &Context) -> Option<ContextId> {
        self.ids.get(context).copied()
    }

    pub(crate) fn len(&self) -> usize {
        self.list.len()
    }

    pub(crate) fn get(&self, id: ContextId) -> &Context {
        &self.list[id.0]
    }

    /// Removes the contexts numbered `len` or more, and their modules (for a failed scope).
    pub(crate) fn truncate(&mut self, len: usize) {
        for context in self.list.drain(len..) {
            self.ids.remove(&context);
        }
        self.modules.split_off(&ContextId(len));
    }
}

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

/// Moves `Self: 'a` bounds, by a lifetime parameter of `params`, out of `predicates`.
/// `Self: Sized + 'a` leaves `Self: Sized`. Other lifetimes and `for<..>` predicates stay.
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

/// The name of the generic that `predicate` bounds, if it is one that `canonicalize()` moves to the
/// generic: `T: Tr` and `'a: 'b`, but not `Vec<T>: Tr`, `for<'x> T: Tr<'x>` or `Self: Tr`.
fn bounded_generic(predicate: &WherePredicate) -> Option<String> {
    match predicate {
        WherePredicate::Type(PredicateType {
            lifetimes: None,
            bounded_ty: Type::Path(TypePath { qself: None, path }),
            ..
        }) => path.get_ident().map(ident_name),
        WherePredicate::Lifetime(PredicateLifetime { lifetime, .. }) => Some(lifetime.to_string()),
        _ => None,
    }
}

/// A generic parameter or a where predicate, written for the `Self` of the scope that declares it.
trait WrittenForSelf: Clone {
    fn self_mentions(&self) -> SelfMentions;
    /// What it is, for the note of an error.
    fn what(&self) -> String;
    fn rewrite(self, folder: &mut ReplaceSelf<'_>) -> Self;
}

impl WrittenForSelf for GenericParam {
    fn self_mentions(&self) -> SelfMentions {
        self_in_param(self)
    }
    fn what(&self) -> String {
        format!("the bound of `{}`", generic_param_name(self))
    }
    fn rewrite(self, folder: &mut ReplaceSelf<'_>) -> Self {
        folder.fold_generic_param(self)
    }
}

impl WrittenForSelf for WherePredicate {
    fn self_mentions(&self) -> SelfMentions {
        self_in_predicate(self)
    }
    fn what(&self) -> String {
        format!("the where predicate `{}`", quote!(#self))
    }
    fn rewrite(self, folder: &mut ReplaceSelf<'_>) -> Self {
        folder.fold_where_predicate(self)
    }
}

/// Rewrites `node`, which is written for `outer` as `Self`, for any other `Self`: `Self` becomes
/// the type that `outer` is. It fails if `outer` is the implementor of a trait, or if `node` names
/// an associated item of `Self`, which have no name where `Self` is something else (`S::X` of a
/// type is ambiguous).
fn for_self_of<T: WrittenForSelf>(
    outer: &SelfRequirement,
    node: &T,
) -> std::result::Result<T, UnnameableSelf> {
    let mentions = node.self_mentions();
    match outer {
        SelfRequirement::Bound(path) => Err(UnnameableSelf {
            cause: mentions.first.unwrap_or_else(|| quote!(Self)),
            headline: format!("{} mentions a `Self` which has no name here", node.what()),
            note: format!(
                "{} mentions the `Self` of the trait `{}`, which has no name here: `Self` is \
                 something else in the item around",
                node.what(),
                quote!(#path)
            ),
        }),
        SelfRequirement::Type(ty) => match mentions.associated {
            Some(cause) => Err(UnnameableSelf {
                headline: format!(
                    "{} mentions `{}`, which has no name here",
                    node.what(),
                    cause
                ),
                note: format!(
                    "{} mentions `{}`, an associated item of the `Self` of `{}`, which has no \
                     name here: `Self` is something else in the item around",
                    node.what(),
                    cause,
                    quote!(#ty)
                ),
                cause,
            }),
            None => Ok(node.clone().rewrite(&mut ReplaceSelf { self_ty: ty })),
        },
    }
}

/// The lifetimes a type must outlive because of the types around it, as in `&'a Vec<Rel2<T>>`
/// (see [Outlives bounds](crate::rules#outlives-bounds)). `T::A<X>` is decided late, by the scopes in
/// scope where it is written: see [`Enclosing::generic()`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Enclosing(Vec<Link>);

/// A step around the type, outermost first.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Link {
    /// The types inside outlive this reference lifetime, written with `depth` scopes in scope.
    Lifetime { lifetime: Lifetime, depth: usize },
    /// `head::A<..>`: an associated type path if `head` is a type parameter. Drops earlier links.
    Projection { head: String, depth: usize },
}

impl Enclosing {
    /// Returns what the components of `ty` must outlive, as rustc derives it. `&'a X` adds `'a`;
    /// tuples, arrays, pointers, fn types, trait object and path arguments pass it on. Associated
    /// type paths, `impl Trait`, array lengths and path lifetime arguments do not.
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
                    // `Self::A<X>` is always an associated type path.
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

    /// The lifetimes still required after associated type paths drop what came before them.
    fn active(&self, scopes: &[Scope]) -> Vec<(Lifetime, usize)> {
        let mut active: Vec<(Lifetime, usize)> = Vec::new();
        for link in &self.0 {
            match link {
                Link::Projection { head, depth } => {
                    if is_type_parameter(head, visible(scopes, *depth)) {
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

    /// The generic lifetimes among them, outermost first, each with its depth.
    pub(crate) fn generic(&self, scopes: &[Scope]) -> Vec<(Lifetime, usize)> {
        let mut generic: Vec<(Lifetime, usize)> = Vec::new();
        for (lifetime, depth) in self.active(scopes) {
            // The scopes in scope where it is written decide what it names.
            if is_generic_lifetime(visible(scopes, depth), &[], &lifetime)
                && !generic.iter().any(|(other, _)| *other == lifetime)
            {
                generic.push((lifetime, depth));
            }
        }
        generic
    }
}

fn visible(scopes: &[Scope], depth: usize) -> &[Scope] {
    &scopes[..depth.min(scopes.len())]
}

/// The head of `path` if it may start an associated type path (`T::A<X>`): relative, with an item
/// after it and no arguments. The path does not tell if the head is a type parameter.
fn projection_head(path: &Path) -> Option<&Ident> {
    if path.leading_colon.is_some() || path.segments.len() < 2 {
        return None;
    }
    path.segments
        .first()
        .filter(|seg| seg.arguments.is_none())
        .map(|seg| &seg.ident)
}

/// `name` is a type parameter of the innermost scope that declares it.
fn is_type_parameter(name: &str, scopes: &[Scope]) -> bool {
    scopes
        .iter()
        .rev()
        .find(|scope| scope.declares(name))
        .map_or(false, |scope| {
            scope.params().any(
                |param| matches!(param, GenericParam::Type(param) if ident_name(&param.ident) == name),
            )
        })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Bounded {
    Type(String),
    Lifetime(Lifetime),
    /// `Self` is not a parameter, but it is encoded as if it were (see [`Context::self_outlives`]).
    SelfTy,
    /// `T::A` or `<T as Tr>::A`, an associated type of a generic, or `Self::A` and `<Self as
    /// Tr>::A`, one of the implementor of a trait (see [`is_generic_projection()`]). It stays a where
    /// predicate of the context (see [`implied_outlives()`]).
    Projection(TypePath),
}

/// `lifetime` is a lifetime parameter of the context. `'static`, `'_` and `for<..>` lifetimes are
/// not, as the impl cannot declare them. An undeclared one is not either.
fn is_generic_lifetime(scopes: &[Scope], binders: &[Lifetime], lifetime: &Lifetime) -> bool {
    lifetime.ident != "static"
        && lifetime.ident != "_"
        && !binders.contains(lifetime)
        && scopes
            .iter()
            .rev()
            .find(|scope| scope.declares(&lifetime.to_string()))
            .map_or(false, |scope| !scope.binder)
}

/// The generics a type mentions, which `X: 'a` bounds (see [`implied_outlives()`]).
struct Mentions<'a> {
    scopes: &'a [Scope],
    /// `Self` is the implementor of a trait, so `Self::A` is an associated type of a generic too
    /// (see [`is_generic_projection()`]).
    self_bound: bool,
    /// The associated types that the bounds of the context equate (see [`Binding`]).
    env: &'a [Binding],
    /// How many bindings were substituted on the way here.
    depth: usize,
    binders: Vec<Lifetime>,
    /// The first segments of paths: names that may be type parameters, and `Self`. Not those of
    /// `projections`.
    types: Vec<String>,
    lifetimes: Vec<Lifetime>,
    /// The associated type paths of generics (see [`is_generic_projection()`]), which are a unit: what
    /// they name is not mentioned.
    projections: Vec<TypePath>,
}

impl<'a> Mentions<'a> {
    fn of(ty: &Type, scopes: &'a [Scope], self_bound: bool, wanted: bool) -> Self {
        let mut mentions = Self::new(scopes, self_bound, &[], Vec::new());
        if wanted {
            mentions.visit_type(ty);
        }
        mentions
    }

    fn new(
        scopes: &'a [Scope],
        self_bound: bool,
        env: &'a [Binding],
        binders: Vec<Lifetime>,
    ) -> Self {
        Self {
            scopes,
            self_bound,
            env,
            depth: 0,
            binders,
            types: Vec::new(),
            lifetimes: Vec::new(),
            projections: Vec::new(),
        }
    }
}

impl<'a, 'ast> Visit<'ast> for Mentions<'a> {
    fn visit_type_path(&mut self, i: &'ast TypePath) {
        if is_generic_projection(i, self.scopes, self.self_bound) {
            match Binding::find(self.env, i) {
                // The bounds equate it: rustc normalizes the path, in the environment of the impl
                // and in the signature, so what outlives is the type it equals. A binding is not
                // cyclic, but the depth is limited all the same.
                Some(value) if self.depth < 16 => {
                    let mut equal =
                        Mentions::new(self.scopes, self.self_bound, self.env, self.binders.clone());
                    equal.depth = self.depth + 1;
                    equal.visit_type(&value);
                    self.types.extend(equal.types);
                    self.lifetimes.extend(equal.lifetimes);
                    self.projections.extend(equal.projections);
                    return;
                }
                Some(_) => (),
                None if is_writable_projection(i, self.scopes, &self.binders) => {
                    self.projections.push(i.clone());
                    return;
                }
                None => (),
            }
        }
        syn::visit::visit_type_path(self, i)
    }
    fn visit_path(&mut self, i: &'ast Path) {
        if i.leading_colon.is_none() {
            self.types
                .extend(i.segments.first().map(|seg| ident_name(&seg.ident)));
        }
        syn::visit::visit_path(self, i)
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        if is_generic_lifetime(self.scopes, &self.binders, i) {
            self.lifetimes.push(i.clone());
        }
    }
    fn visit_type_impl_trait(&mut self, i: &'ast TypeImplTrait) {
        // Only the lifetime bounds count: `impl Trait` is not a component (see `Enclosing`).
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

/// Returns `true` if `ty` is an associated type path of a generic, which `&'a ty` makes outlive
/// `'a` as a whole: `T::A` and `<T as Tr<..>>::A` with `T` a type parameter in `scopes`, or such a
/// path in place of `T` (`<T::A as Tr2>::B`). `Self` counts as a type parameter where it is the
/// implementor of a trait (`self_bound`): `Self::A`, `<Self as Tr<..>>::A` and `Self::A<'b>` are
/// the same. An associated type of a type that is not a type parameter (`<Vec<T> as
/// IntoIterator>::Item`) is not one: rustc normalizes it, so what it becomes is what outlives `'a`.
fn is_generic_projection(ty: &TypePath, scopes: &[Scope], self_bound: bool) -> bool {
    // The type parameter, `Self` of a trait, or the projection that the associated type belongs to.
    fn is_base(name: &str, scopes: &[Scope], self_bound: bool) -> bool {
        (self_bound && name == "Self") || is_type_parameter(name, scopes)
    }
    fn base(ty: &Type, scopes: &[Scope], self_bound: bool) -> bool {
        match unwrap_type_group(ty) {
            Type::Path(TypePath { qself: None, path }) if path.segments.len() == 1 => {
                let seg = &path.segments[0];
                path.leading_colon.is_none()
                    && seg.arguments.is_none()
                    && is_base(&ident_name(&seg.ident), scopes, self_bound)
            }
            Type::Path(path) => is_generic_projection(path, scopes, self_bound),
            _ => false,
        }
    }
    match &ty.qself {
        // `T::A`
        None => {
            let mut segments = ty.path.segments.iter();
            match (
                &ty.path.leading_colon,
                segments.next(),
                segments.next(),
                segments.next(),
            ) {
                (None, Some(first), Some(_), None) => {
                    first.arguments.is_none()
                        && is_base(&ident_name(&first.ident), scopes, self_bound)
                }
                _ => false,
            }
        }
        // `<T as Tr>::A`: one segment follows the trait. `<T>::A` is not one: rustc reads it as
        // generics where a where predicate starts with it (`where <T>::A: 'a`).
        Some(qself) => {
            qself.position > 0
                && ty.path.segments.len() == qself.position + 1
                && base(&qself.ty, scopes, self_bound)
        }
    }
}

/// Returns `true` if the associated type path `ty` (see [`is_generic_projection()`]) can be the
/// bounded type of a where predicate of the context: it names no lifetime other than `'static` and
/// the lifetime parameters of the context (not a binder's, `'_` or an elided one in a reference),
/// and no `impl Trait`, `_`, type macro, function pointer or `Fn(..)` sugar. Otherwise its parts
/// are mentioned as before.
fn is_writable_projection(ty: &TypePath, scopes: &[Scope], binders: &[Lifetime]) -> bool {
    struct Writable<'a> {
        scopes: &'a [Scope],
        binders: &'a [Lifetime],
        writable: bool,
    }
    impl<'a, 'ast> Visit<'ast> for Writable<'a> {
        fn visit_lifetime(&mut self, i: &'ast Lifetime) {
            if i.ident != "static" && !is_generic_lifetime(self.scopes, self.binders, i) {
                self.writable = false;
            }
        }
        fn visit_type_reference(&mut self, i: &'ast TypeReference) {
            self.writable &= i.lifetime.is_some();
            syn::visit::visit_type_reference(self, i)
        }
        fn visit_type_impl_trait(&mut self, _: &'ast TypeImplTrait) {
            self.writable = false;
        }
        fn visit_type_infer(&mut self, _: &'ast TypeInfer) {
            self.writable = false;
        }
        fn visit_type_macro(&mut self, _: &'ast TypeMacro) {
            self.writable = false;
        }
        fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {
            // Its own binder and elision: not probed.
            self.writable = false;
        }
        fn visit_parenthesized_generic_arguments(
            &mut self,
            _: &'ast ParenthesizedGenericArguments,
        ) {
            self.writable = false;
        }
    }
    let mut writable = Writable {
        scopes,
        binders,
        writable: true,
    };
    writable.visit_type_path(ty);
    writable.writable
}

/// An associated type that a bound of the context equates: `T: Iterator<Item = X>`, or `F: Fn(A)
/// -> B`, which equates `Output`. rustc normalizes the path (`T::Item`) to the type in the
/// environment of an impl that declares the bound, and in the signature of the original, so what
/// `&'a T::Item` requires is that of `X`, whatever the predicate `T::Item: 'a` says: the
/// environment of the impl has it as `X: 'a`, and it is no help if `X` is not a parameter.
#[derive(Clone, Debug)]
struct Binding {
    /// The bounded type as written, the part before `::` of the path: `T`, or `T :: A`.
    bounded: String,
    /// The last segment of the trait of the bound, unless it is the sugar of `Fn`.
    of_trait: Option<String>,
    name: String,
    value: Type,
}

impl Binding {
    /// The bindings of the bounds of `params` and of `predicates`.
    fn of(params: &[GenericParam], predicates: &[WherePredicate]) -> Vec<Self> {
        let mut found = Vec::new();
        let mut collect = |bounded: String, bounds: &Punctuated<TypeParamBound, Token![+]>| {
            for bound in bounds {
                let seg = match bound {
                    TypeParamBound::Trait(bound) => match bound.path.segments.last() {
                        Some(seg) => seg,
                        None => continue,
                    },
                    _ => continue,
                };
                match &seg.arguments {
                    PathArguments::AngleBracketed(args) => {
                        for arg in &args.args {
                            if let GenericArgument::AssocType(assoc) = arg {
                                if assoc.generics.is_none() {
                                    found.push(Self {
                                        bounded: bounded.clone(),
                                        of_trait: Some(ident_name(&seg.ident)),
                                        name: ident_name(&assoc.ident),
                                        value: assoc.ty.clone(),
                                    });
                                }
                            }
                        }
                    }
                    PathArguments::Parenthesized(sugar) => found.push(Self {
                        bounded: bounded.clone(),
                        of_trait: None,
                        name: "Output".to_string(),
                        value: match &sugar.output {
                            ReturnType::Type(_, ty) => (**ty).clone(),
                            ReturnType::Default => parse_quote!(()),
                        },
                    }),
                    PathArguments::None => (),
                }
            }
        };
        for param in params {
            if let GenericParam::Type(param) = param {
                collect(ident_name(&param.ident), &param.bounds);
            }
        }
        for predicate in predicates {
            if let WherePredicate::Type(predicate) = predicate {
                let ty = unwrap_type_group(&predicate.bounded_ty);
                collect(quote!(#ty).to_string(), &predicate.bounds);
            }
        }
        found
    }

    /// The type that `projection` equals if a binding says so (`projection` has the shape that
    /// [`is_generic_projection()`] tells).
    fn find(env: &[Self], projection: &TypePath) -> Option<Type> {
        let (bounded, of_trait, name) = match &projection.qself {
            None => {
                let mut segments = projection.path.segments.iter();
                let first = segments.next()?;
                (
                    ident_name(&first.ident),
                    None,
                    ident_name(&segments.next()?.ident),
                )
            }
            Some(qself) => {
                let ty = unwrap_type_group(&qself.ty);
                let segments: Vec<&PathSegment> = projection.path.segments.iter().collect();
                (
                    quote!(#ty).to_string(),
                    segments
                        .get(qself.position - 1)
                        .map(|seg| ident_name(&seg.ident)),
                    ident_name(&segments.get(qself.position)?.ident),
                )
            }
        };
        env.iter()
            .find(|binding| {
                binding.bounded == bounded
                    && binding.name == name
                    && match (&binding.of_trait, &of_trait) {
                        (Some(bound), Some(written)) => bound == written,
                        _ => true,
                    }
            })
            .map(|binding| binding.value.clone())
    }
}

/// Returns the outlives bounds `ty` implies on `params`: `&'a X` requires `X: 'a`, as does each of
/// `ancestors` for `ty`. A path with a lifetime argument adds none, as a guessed bound could make
/// the copy stricter than the original. See [Outlives bounds](crate::rules#outlives-bounds).
///
/// An associated type path of a generic (`T::A`, `<T as Tr<'b>>::A`: see [`is_generic_projection()`])
/// requires `T::A: 'a`, which the original gets from the well-formedness of its signature, and the
/// Repeater's impl has no signature. It is returned as itself and not as the bounds of its
/// components (`T` and the arguments of the trait): the copy, where only the projection is known to
/// outlive `'a`, could not prove those. The signature of a trait gives `T::A: 'a` and not `T: 'a`,
/// so an impl for another type, whose signature has the encoded `T: 'a`, could not prove it
/// (E0309).
///
/// The same holds for the associated types of `Self` of a trait (`self_req` is its bound):
/// `Self::A`, `<Self as Tr<'b, U>>::A` and `Self::A<'b>` are returned as themselves, and bring no
/// `Self: 'a` (it would be stricter than the original, as `T: 'a` is: the copy for a generic host
/// `Gen<H>` needs `&'a Gen<H>`, where the original needs `H::Item: 'a`), nor `'b: 'a` or `U: 'a`
/// for the arguments of the trait. `Self` mentioned elsewhere in the type does require `Self: 'a`.
///
/// If a bound of the context equates the associated type (`T: Iterator<Item = X>`, `where Self:
/// Iterator<Item = X>`, see [`Binding`]), rustc normalizes the path in the signature and in the
/// environment of the impl, where the predicate would be `X: 'a`, which is not decomposed into the
/// bounds of the parts of `X`: the parts of `X` are mentioned instead. A supertrait that equates it
/// (`trait Tr: Iterator<Item = X>`) is not seen, since the Referrer's copy has none.
///
/// A projection that names a generic associated type of the trait (`Self::G<'x>`, `<Self as
/// Tr>::G<'x>`: see [`Gat`]) also requires what its where clause says (`type G<'b> where Self: 'b`
/// gives `Self: 'x`), with the arguments of the projection for its parameters, wherever it is
/// written: the well-formedness of the signature gives the original those bounds, and the impl
/// of the Repeater has no signature.
fn implied_outlives(
    ty: &Type,
    scopes: &[Scope],
    params: &[GenericParam],
    env: &[Binding],
    self_req: Option<&SelfRequirement>,
    ancestors: &[Lifetime],
) -> Vec<(Bounded, Lifetime)> {
    struct Finder<'a> {
        scopes: &'a [Scope],
        self_bound: bool,
        env: &'a [Binding],
        params: &'a [GenericParam],
        self_req: Option<&'a SelfRequirement>,
        binders: Vec<Lifetime>,
        found: Vec<(Bounded, Lifetime)>,
    }
    impl<'a> Finder<'a> {
        fn require(&mut self, lifetime: &Lifetime, referent: &Type) {
            if !is_generic_lifetime(self.scopes, &self.binders, lifetime) {
                return;
            }
            self.require_generic(lifetime, referent)
        }

        /// A projection that names a generic associated type of the trait requires what its where
        /// clause says, wherever it is written (see [`Gat`]).
        fn require_gat(&mut self, projection: &TypePath) {
            for predicate in gat_requirements(projection, self.scopes, self.self_req) {
                match predicate {
                    WherePredicate::Type(predicate) => {
                        for bound in &predicate.bounds {
                            if let TypeParamBound::Lifetime(lifetime) = bound {
                                self.require(lifetime, &predicate.bounded_ty);
                            }
                        }
                    }
                    WherePredicate::Lifetime(predicate) => {
                        // `'a: 'x` is what `&'x &'a ()` requires.
                        let outlived = &predicate.lifetime;
                        let referent: Type = parse_quote!(&#outlived ());
                        for lifetime in &predicate.bounds {
                            self.require(lifetime, &referent);
                        }
                    }
                    _ => (),
                }
            }
        }

        fn require_generic(&mut self, lifetime: &Lifetime, referent: &Type) {
            let mut mentions =
                Mentions::new(self.scopes, self.self_bound, self.env, self.binders.clone());
            mentions.visit_type(referent);
            if self.self_req.is_some() && mentions.types.iter().any(|name| name == "Self") {
                self.found.push((Bounded::SelfTy, lifetime.clone()));
            }
            self.found.extend(
                mentions
                    .projections
                    .into_iter()
                    .map(|projection| (Bounded::Projection(projection), lifetime.clone())),
            );
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
        fn visit_type_path(&mut self, i: &'ast TypePath) {
            self.require_gat(i);
            syn::visit::visit_type_path(self, i)
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
        self_bound: matches!(self_req, Some(SelfRequirement::Bound(_))),
        env,
        params,
        self_req,
        binders: Vec::new(),
        found: Vec::new(),
    };
    finder.visit_type(ty);
    for lifetime in ancestors {
        finder.require_generic(lifetime, ty);
    }
    finder.found
}

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

/// Returns the canonical form of `generics`: predicates on one parameter move to it, everything is
/// sorted, and defaults and attributes (which an impl cannot have) are removed.
fn canonicalize(generics: Generics) -> Generics {
    let mut params: Vec<GenericParam> = generics.params.into_iter().collect();
    let mut predicates = Vec::new();
    for predicate in generics
        .where_clause
        .into_iter()
        .flat_map(|wc| wc.predicates)
    {
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
                param.bounds = canonical_bounds(std::mem::take(&mut param.bounds));
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
                predicate.bounds = canonical_bounds(std::mem::take(&mut predicate.bounds));
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

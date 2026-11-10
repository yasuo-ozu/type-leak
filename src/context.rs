//! The context of a leaked type: its generics and what `Self` means.

use crate::error::Cause;
use crate::unstable::{bound_text, hiding_bounds, trait_bounds};
use crate::util::{
    binder_lifetimes, bound_mentions_self, bounded_ident, canonical_bounds, first_self_in_generics,
    generic_param_mentions_self, generic_param_name, has_lifetime, ident_name, is_bare_self,
    is_lifetime_param, is_named, is_self_projection, is_type_param_named, lifetime_name, names_in,
    plain_self_head, projection, projection_head, push_once, same_lifetime, same_name,
    self_in_param, self_in_predicate, sorted_by_tokens, tidy, token_names, tokens_text,
    type_contains, type_mentions_self, type_names, unwrap_type_group, where_clause,
    where_predicate_mentions_self, without_default, without_self_bounds, ReplaceSelf, SelfMentions,
    Unraw,
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
    /// The parameters in the order they were declared, if `params` are in another (those of a
    /// context are by kind and name).
    declared_params: Option<Vec<GenericParam>>,
    predicates: Vec<WherePredicate>,
    is_binder: bool,
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
            params: generics
                .params
                .iter()
                .cloned()
                .map(without_default)
                .collect(),
            declared_params: None,
            predicates: generics
                .where_clause
                .iter()
                .flat_map(|wc| wc.predicates.iter().cloned())
                .collect(),
            is_binder: false,
            self_ty,
            gats: Vec::new(),
        }
    }

    /// The generics and what `Self` means in the scope of `context`.
    pub(crate) fn of_context(context: &Context) -> Self {
        let mut scope =
            Self::new(&context.generics, context.self_ty.clone()).with_gats(context.gats.clone());
        scope.declared_params = context
            .declared
            .as_ref()
            .map(|declared| declared.params.iter().cloned().collect());
        scope
    }

    pub(crate) fn binder(lifetimes: &BoundLifetimes) -> Self {
        Self {
            params: lifetimes.lifetimes.iter().cloned().collect(),
            declared_params: None,
            predicates: Vec::new(),
            is_binder: true,
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

    /// Drops the bounds and where predicates that mention `Self`, and `Self` itself: what the
    /// default of a parameter of a struct, an enum or a union has, whose entries do not depend on
    /// the bounds that its copy proves for its own type.
    pub(crate) fn without_self(&mut self) {
        for param in &mut self.params {
            if let GenericParam::Type(param) = param {
                param.bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| !bound_mentions_self(bound))
                    .cloned()
                    .collect();
            }
        }
        self.predicates = self
            .predicates
            .iter()
            .filter_map(without_self_bounds)
            .collect();
        self.self_ty = None;
    }

    /// Returns `true` if the generics are the same, whatever `Self` meant where they were declared.
    pub(crate) fn same_generics(&self, other: &Scope) -> bool {
        let params = |scope: &Scope| -> Vec<GenericParam> {
            let unraw = |param: &GenericParam| Unraw.fold_generic_param(param.clone());
            scope.params.iter().map(unraw).collect()
        };
        let predicates = |scope: &Scope| -> Vec<WherePredicate> {
            let unraw = |predicate: &WherePredicate| Unraw.fold_where_predicate(predicate.clone());
            scope.predicates.iter().map(unraw).collect()
        };
        params(self) == params(other)
            && predicates(self) == predicates(other)
            && self.is_binder == other.is_binder
    }

    pub(crate) fn params(&self) -> impl Iterator<Item = &GenericParam> {
        self.params.iter()
    }

    /// The parameters in the order they were declared.
    fn declared_params(&self) -> &[GenericParam] {
        self.declared_params.as_deref().unwrap_or(&self.params)
    }

    pub(crate) fn predicates(&self) -> impl Iterator<Item = &WherePredicate> {
        self.predicates.iter()
    }

    /// The generics of the scope, as declared.
    pub(crate) fn generics(&self) -> Generics {
        Generics {
            params: self.params.iter().cloned().collect(),
            where_clause: where_clause(self.predicates.iter().cloned()),
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

/// What `Self` means where an entry was found (`Context::self_ty`).
///
/// If an entry needs `Self`, its Repeater impl must be for the type that `Self` means. Where
/// there is no `Self`, there is no requirement, and the impl may be for any type.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S<T>(Rel<Self, T>);));
/// let referrer = leaker.result().finish().unwrap();
/// // `Self` of the entry `Rel<Self, T>` is the struct.
/// let (_, _, context) = referrer.iter().next().unwrap();
/// assert_eq!(context.self_ty, Some(SelfRequirement::Type(parse_quote!(S<T>))));
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SelfRequirement {
    /// `Self` is any type that implements the trait (`Self: MyTrait<T, U>`).
    ///
    /// The impl is generic: `impl<.., __S: ?Sized + MyTrait<T, U>> .. for __S`. `__S` takes the
    /// place of `Self`, so `Self::Item` becomes `<__S>::Item`. The copy writes
    /// `<Self as Repeater<..>>::Type`, where `Self` implements the trait. That type must define
    /// the associated types that the entry uses (`type Item = u8;`).
    ///
    /// ```ignore
    /// // In crate a: the entry `Rel<Self::A>` of a trait.
    /// pub trait Tr { type A; fn f(&self) -> Rel<Self::A>; }
    /// impl<__S: ?Sized + Tr> Repeater<(), KEY> for __S { type Type = Rel<<__S>::A>; }
    ///
    /// // In crate b: `Self` implements `a::Tr`.
    /// fn f(&self) -> <Self as ::a::Repeater<(), KEY>>::Type;
    /// ```
    Bound(Path),
    /// `Self` is one type: a struct, an enum, a union, or the self type of an impl
    /// (`MyStruct<T>`).
    ///
    /// In a struct, enum or union nested in the analyzed code, only an entry that must be
    /// replaced has this requirement (`Rel2<Self>`). An absolute type that names `Self`
    /// (`::path::Wrap<Self>`) has none: the copy keeps it as written, with its own `Self`.
    ///
    /// The type is written as `Self` means it in the item's module, without the module path:
    ///
    /// | Item | `Type(..)` |
    /// |---|---|
    /// | `struct S<'a, T: Clone, const N: usize = 3> { .. }` | `S<'a, T, N>` |
    /// | `enum E<T> { .. }` | `E<T>` |
    /// | `struct N<T> { .. }` (nested) | `N<T>` |
    /// | `impl<T> MyType<T, u8> { .. }` | `MyType<T, u8>`, as written |
    /// | `impl<T> ::a::W<T> { .. }` | `::a::W<T>`, as written |
    ///
    /// The impl is for this type. A copy in another crate cannot write `Self` for it, because
    /// there `Self` is the copy's own type. So the copy writes the item's type with its module path
    /// in front ([`Visitor::with_self()`](crate::Visitor::with_self)). A proc macro cannot find
    /// that path by itself: it must come from the macro's input. For a nested item, give it to
    /// [`Visitor::with_nested_self()`](crate::Visitor::with_nested_self), which gets this
    /// requirement; without it the entry is left as written. If the
    /// self type of an impl is a relative path, rewrite it like any other type
    /// ([`Referrer::visitor()`](crate::Referrer::visitor)).
    ///
    /// ```ignore
    /// // In crate a: the entry `Rel<Self>` of a struct.
    /// pub struct S { a: Rel<Self> }
    /// impl Repeater<(), KEY> for S { type Type = Rel<Self>; }
    ///
    /// // In crate b: `Self` would be the copy's own type, so write `a`'s `S` by its path.
    /// pub struct S { a: <::a::S as ::a::Repeater<(), KEY>>::Type }
    /// ```
    Type(Type),
    /// `Self` is any type that the bounds of the generics hold for. The entry itself does not
    /// name `Self`: `Rel<T>` with `T: AsRef<Self>`.
    ///
    /// This happens for an entry of a struct, enum or union nested in the analyzed code
    /// ([`View::visit_item()`](crate::View::visit_item)). The copy has the same bound on its
    /// own type, so it writes `<Self as Repeater<..>>::Type` with its own `Self`
    /// ([`Referrer::visitor()`](crate::Referrer::visitor) does it). The entry may name
    /// `Self` too (`Rel2<Self, T>`), with the error
    /// [`Reason::NestedSelfInBounds`](crate::Reason::NestedSelfInBounds), which you accept with
    /// [`LeakerResult::allow()`](crate::LeakerResult::allow).
    ///
    /// The impl is generic over `__S`, which takes the place of `Self` in the bounds and has no
    /// bound of its own. `__S` is `?Sized` unless a bound or the type may need `Sized`. A marker
    /// type cannot be used: the call of
    /// [`repeater_impls()`](crate::LeakerResult::repeater_impls) for
    /// [`ImplSelf::Marker`](crate::ImplSelf::Marker) fails with
    /// [`EncodeError::NeedsSelf`](crate::EncodeError::NeedsSelf).
    ///
    /// ```ignore
    /// // In crate a: the entry `Rel<T>`, with `Self` in the bound of `T`.
    /// pub mod m { pub struct N<T: AsRef<Self>>(Rel<T>); }
    /// impl<T: AsRef<__S>, __S: ?Sized> Repeater<(T,), KEY> for __S { type Type = Rel<T>; }
    ///
    /// // In crate b: the copy's own `Self`.
    /// pub mod m { pub struct N<T: AsRef<Self>>(<Self as ::a::Repeater<(T,), KEY>>::Type); }
    /// ```
    ///
    /// Before Rust 1.70, rustc fails with E0275 when the last field of a struct is the path
    /// through an impl that needs `Sized` (on 1.56, also for an enum). Put that field first, add
    /// `where Self: Sized` to the item, or use a bound that does not need `Sized`.
    Any,
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
            Self::Any => (),
        }
    }

    fn render(&self) -> String {
        match self {
            Self::Bound(path) => format!(" Self: {}", tokens_text(path)),
            Self::Type(ty) => format!(" Self = {}", tokens_text(ty)),
            Self::Any => " Self any".to_string(),
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

/// A generic associated type of a trait with its outlives relations (`type G<'b, U> where U: 'b,
/// Self: 'b;`): `Self::G<'x>` needs `Self: 'x`, which the trait's signature implies and the impl
/// of the Repeater must require. Only the top-level context of a trait has them.
#[derive(Clone, Debug)]
pub(crate) struct Gat {
    /// The name of the type (`G`).
    ident: Ident,
    /// The parameters, without bounds, and a where clause of the outlives relations.
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
    fn new(ident: &Ident, generics: &Generics) -> Option<Self> {
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
                where_clause: where_clause(predicates),
                ..Default::default()
            },
        })
    }

    /// Reads what [`ToTokens`] writes (the grammar is above `FORMAT_TAG_PREFIX` in referrer.rs).
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

    /// The where clause of `G<..>` with its parameters replaced by `args`, and the trait's
    /// `trait_params` by `trait_args` unless that is empty. Empty if the arguments do not fit the
    /// parameters (a lifetime elided in the path).
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
                        .insert(lifetime_name(&param.lifetime), arg.clone());
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
            .get(&lifetime_name(&lifetime))
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
                let arg = arg.clone();
                // `U::Item` for a name (`T`, which may be a type parameter) is `T::Item`. For any
                // other type it is `<arg>::Item`, whose parts are what the requirement mentions:
                // `Arg::Item` would not resolve (`T::A::Item` is no projection of `T`).
                return match &arg {
                    Type::Path(TypePath {
                        qself: None,
                        path: arg_path,
                    }) if arg_path.leading_colon.is_none()
                        && arg_path.segments.len() == 1
                        && arg_path.segments[0].arguments.is_none() =>
                    {
                        let rest = syn::fold::fold_path(
                            self,
                            Path {
                                leading_colon: None,
                                segments: path.segments.iter().skip(1).cloned().collect(),
                            },
                        );
                        let mut segments = arg_path.segments.clone();
                        segments.extend(rest.segments);
                        Type::Path(TypePath {
                            qself: None,
                            path: Path {
                                leading_colon: None,
                                segments,
                            },
                        })
                    }
                    _ => Type::Path(projection(arg, path, |rest| {
                        syn::fold::fold_path(self, rest)
                    })),
                };
            }
        }
        syn::fold::fold_type(self, ty)
    }
}

/// The outlives relations that `projection` requires if it is `Self::G<..>` or `<Self as
/// Tr<..>>::G<..>` for a [`Gat`] of the trait that `self_req` bounds. Those of a supertrait or of
/// a generic (`T::G<'x>`) are not seen.
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
            [_, gat] if plain_self_head(&projection.path).is_some() => (gat, None),
            _ => return Vec::new(),
        },
        // `<Self as Tr<..>>::G<..>`
        Some(qself) => {
            if !is_bare_self(&qself.ty)
                || qself.position == 0
                || segments.len() != qself.position + 1
            {
                return Vec::new();
            }
            let tr = segments[qself.position - 1];
            let same_trait = trait_path
                .segments
                .last()
                .map_or(false, |seg| same_name(&seg.ident, &tr.ident));
            if !same_trait {
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
            let trait_params = scope.declared_params();
            // A trait with no arguments written takes its parameters as they are.
            if !trait_args.is_empty() && trait_args.len() != trait_params.len() {
                return Vec::new();
            }
            gat.requirements(&args, trait_params, &trait_args)
        }
        None => Vec::new(),
    }
}

fn generics_as_args(generics: &Generics) -> TokenStream {
    if generics.params.is_empty() {
        return TokenStream::new();
    }
    let args: Vec<GenericArgument> = generics.params.iter().map(param_as_arg).collect();
    quote!(<#(#args),*>)
}

/// The argument that passes `param` on by its own name.
fn param_as_arg(param: &GenericParam) -> GenericArgument {
    match param {
        GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
            GenericArgument::Lifetime(lifetime.clone())
        }
        GenericParam::Type(TypeParam { ident, .. }) => GenericArgument::Type(parse_quote!(#ident)),
        GenericParam::Const(ConstParam { ident, .. }) => {
            GenericArgument::Const(parse_quote!(#ident))
        }
    }
}

/// What a leaked type needs from the place where it is written: its generics and what `Self`
/// means.
///
/// A leaked type (an entry) may use the generics of its item and `Self`. The Repeater impl of the
/// entry must have the same. Entries that need the same share one context, and a [`ContextId`]
/// names it. The Leaker also makes a context for each item or scope that it analyzes (the
/// top-level context).
///
/// [`Referrer::iter()`](crate::Referrer::iter) gives the context of each entry. The fields are
/// public to be read. You cannot build a `Context`, and more fields may come. Two contexts are
/// equal if all their fields are equal. The top-level context of a scope also holds the generics
/// as written (not a public field), so it is never the context of an entry. A Referrer that is
/// written as tokens and read back keeps every field.
///
/// ```
/// # use syn::*;
/// # use type_leak::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S<T>(Rel<T>);));
/// let referrer = leaker.result().finish().unwrap();
///
/// // The context of the entry `Rel<T>` uses the generic `T`, but no `Self`.
/// let (_, _, context) = referrer.iter().next().unwrap();
/// assert_eq!(context.generics, parse_quote!(<T>));
/// assert!(context.self_ty.is_none());
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Context {
    /// The generics that the entries of this context use.
    ///
    /// They are in a canonical form, so equal declarations are equal. Lifetimes come first, then
    /// types, then consts, each sorted by name. The bounds of a generic are sorted. A where
    /// predicate on one generic (`T: Default`) moves to the generic. Any other predicate stays in
    /// the where clause (`Vec<T>: Clone`). Defaults are dropped.
    ///
    /// ```ignore
    /// // In crate a: the entries `Rel<T>` and `&'a Rel2<'a, U, N>`.
    /// pub struct S<'a, T: Clone = u8, U = (), const N: usize = 3>
    /// where T: Default, Vec<T>: Clone, U: 'a
    /// { a: Rel<T>, b: &'a Rel2<'a, U, N> }
    ///
    /// // `generics` of the context of `Rel<T>`:
    /// <T: Clone + Default> where Vec<T>: Clone
    /// // `generics` of the context of `&'a Rel2<'a, U, N>`:
    /// <'a, U: 'a, const N: usize>
    /// ```
    ///
    /// A Referrer read back from tokens keeps it.
    pub generics: Generics,
    /// What `Self` means for the entries of this context, or `None` if they do not depend on it.
    ///
    /// See [`SelfRequirement`] for the three kinds.
    ///
    /// ```ignore
    /// // In crate a: the entries `Rel<Self>`, `Rel2<Self>` and `Rel3<T>`.
    /// pub struct S { a: Rel<Self> }
    /// pub trait Tr { fn f(&self) -> Rel2<Self>; }
    /// pub struct N<T: AsRef<Self>>(Rel3<T>); // nested: `View::visit_item()`
    ///
    /// // `self_ty` of the context of `Rel<Self>`:
    /// Some(SelfRequirement::Type(S))
    /// // `self_ty` of the context of `Rel2<Self>`:
    /// Some(SelfRequirement::Bound(Tr))
    /// // `self_ty` of the context of `Rel3<T>`, whose generic has a bound that names `Self`:
    /// Some(SelfRequirement::Any)
    /// // `self_ty` of the context of any entry that does not need `Self`:
    /// None
    /// ```
    ///
    /// A Referrer read back from tokens keeps it.
    pub self_ty: Option<SelfRequirement>,
    /// The lifetimes `'a` for which the entries need `Self: 'a`, sorted.
    ///
    /// The entry may imply it (`&'a Self`) or the trait may write it. It is empty if `self_ty` is
    /// `None`. The lifetimes are not in `generics`, because `Self` is not a parameter. See
    /// [Outlives bounds](crate::Leaker#outlives-bounds).
    ///
    /// ```ignore
    /// // In crate a: the entry `Rel<&'a Self>` implies `Self: 'a`.
    /// pub trait Tr { fn f<'a>(&'a self) -> Rel<&'a Self>; }
    /// // `self_outlives` of the context of `Rel<&'a Self>`:
    /// ['a]
    ///
    /// // In crate a: the trait writes `Self: 'a`.
    /// pub trait Tr<'a> where Self: 'a { fn f(&self) -> Rel<'a, Self>; }
    /// // `self_outlives` of the context of `Rel<'a, Self>`:
    /// ['a]
    /// ```
    ///
    /// A Referrer read back from tokens keeps it.
    pub self_outlives: Vec<Lifetime>,
    /// The generic associated types of the trait, sorted by name: only in the top-level context of
    /// a trait that has a `self_ty` (see [`Gat`]). Kept in tokens.
    pub(crate) gats: Vec<Gat>,
    /// The generics of the item as written (order, defaults, where clause), only in the top-level
    /// context of a scope. Part of the identity: equal `generics` declared otherwise are two
    /// contexts.
    pub(crate) declared: Option<Generics>,
    /// The `Self` of the outer scopes whose bounds this context depends on (`S<T>` for the entry
    /// `Wrap<S<T>>` of `struct S<T: Fn(T::Item) + Iterator<Item = u8>>`), so that the scope that
    /// declares it records it in its `self_ty`. Not part of the identity, not kept in tokens.
    substituted: Vec<SelfRequirement>,
}

/// A bound that mentions the `Self` of an outer scope, which cannot be written where `Self` is
/// something else (see [`Context::of()`]).
#[derive(Clone, Debug)]
pub(crate) struct UnnameableSelf {
    /// The mention of `Self` that cannot be written.
    pub(crate) cause: Box<Cause>,
    /// Tells what is wrong, as the headline of the error: the problem is the bound, not the path
    /// of the type that uses it.
    pub(crate) headline: String,
    /// Tells why, and where `Self` is something else.
    pub(crate) note: String,
    /// Tells what to write instead of the bound, as the start of a sentence.
    pub(crate) help: String,
}

impl Context {
    pub(crate) fn new(generics: Generics, self_ty: Option<SelfRequirement>) -> Self {
        Self {
            generics: canonicalize(generics),
            self_ty,
            self_outlives: Vec::new(),
            gats: Vec::new(),
            declared: None,
            substituted: Vec::new(),
        }
    }

    /// Returns `true` if an entry of this context needs what `Self` stands for: the context has a
    /// self type, or its generics name `Self`, even only in `Self: 'a`. The Referrer,
    /// `Projection::of()` and the impls of `repeater_impls()` all decide it here.
    pub(crate) fn needs_self(&self) -> bool {
        self.self_ty.is_some() || first_self_in_generics(&self.generics).is_some()
    }

    pub(crate) fn with_self_outlives(mut self, lifetimes: Vec<Lifetime>) -> Self {
        self.self_outlives = if self.self_ty.is_some() {
            sorted_by_tokens(lifetimes)
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

    /// Sets the generics of the scope as written: its top-level context.
    pub(crate) fn with_declared(mut self, declared: Generics) -> Self {
        self.declared = Some(declared);
        self
    }

    fn render(&self) -> String {
        let generics = &self.generics;
        let where_clause = &generics.where_clause;
        let mut rendered = tokens_text(quote!(#generics #where_clause));
        if let Some(self_ty) = &self.self_ty {
            rendered.push_str(&self_ty.render());
        }
        for (i, lifetime) in self.self_outlives.iter().enumerate() {
            rendered.push_str(if i == 0 { " Self: " } else { " + " });
            rendered.push_str(&lifetime_name(lifetime));
        }
        for gat in &self.gats {
            rendered.push(' ');
            rendered.push_str(&tokens_text(gat));
        }
        if let Some(declared) = &self.declared {
            let where_clause = &declared.where_clause;
            let declared = tokens_text(quote!(#declared #where_clause));
            rendered.push_str(&format!(" declared {}", declared));
        }
        rendered
    }

    /// Returns the context of `ty` in `scopes` (outermost first), where `Self` is `self_req`. A
    /// bound written for another `Self` is rewritten (see [`for_self_of()`]). With `nested_type`,
    /// `self_req` is a nested struct, enum or union, and `ty` may be for any type.
    pub(crate) fn of(
        ty: &Type,
        scopes: &[Scope],
        self_req: Option<&SelfRequirement>,
        nested_type: bool,
        enclosing: &Enclosing,
    ) -> std::result::Result<Self, UnnameableSelf> {
        Self::compute(ty, scopes, self_req, nested_type, false, enclosing)
    }

    /// Like [`Context::of()`], but with `nested_type` a `ty` that names `Self` is for any type too
    /// ([`Reason::NestedSelfInBounds`](crate::Reason::NestedSelfInBounds)): `Self` is then the
    /// implementor of the impl.
    fn of_any(
        ty: &Type,
        scopes: &[Scope],
        self_req: Option<&SelfRequirement>,
        nested_type: bool,
        enclosing: &Enclosing,
    ) -> std::result::Result<Self, UnnameableSelf> {
        Self::compute(ty, scopes, self_req, nested_type, true, enclosing)
    }

    /// [`Context::of_any()`] if `any`, else [`Context::of()`].
    pub(crate) fn of_as(
        ty: &Type,
        scopes: &[Scope],
        self_req: Option<&SelfRequirement>,
        nested_type: bool,
        any: bool,
        enclosing: &Enclosing,
    ) -> std::result::Result<Self, UnnameableSelf> {
        if any {
            Self::of_any(ty, scopes, self_req, nested_type, enclosing)
        } else {
            Self::of(ty, scopes, self_req, nested_type, enclosing)
        }
    }

    /// Computes the context without the bounds that [`LeftOut`] leaves out, except where `ty`
    /// depends on `Self` or writes out the self type of its scope.
    fn compute(
        ty: &Type,
        scopes: &[Scope],
        self_req: Option<&SelfRequirement>,
        nested_type: bool,
        any: bool,
        enclosing: &Enclosing,
    ) -> std::result::Result<Self, UnnameableSelf> {
        let context = Self::compute_with(
            ty,
            scopes,
            self_req,
            nested_type,
            any,
            enclosing,
            Whole::Nothing,
        )?;
        // The self type of an impl, a struct, an enum or a union, written out (`impl<T: Fn(T::Item)>
        // S<T> { .. }` with the entry `S<T>`), is declared with the bounds of the item.
        let names_self_type = matches!(self_req, Some(SelfRequirement::Type(self_ty))
            if type_contains(ty, self_ty));
        let depends_on_self = matches!(
            context.self_ty,
            Some(SelfRequirement::Type(_)) | Some(SelfRequirement::Bound(_))
        );
        if !(names_self_type || depends_on_self) {
            return Ok(context);
        }
        // The self type, or an associated type of its trait, is well-formed only with the bounds
        // that the item declares (E0277 in the Leaker's crate otherwise). A type that depends on
        // `Self` only through a bound or `Self: 'x` keeps those of the generics `Self` names.
        let keep = if names_self_type || type_mentions_self(ty) {
            Whole::Everything
        } else {
            let mut names = HashSet::new();
            if let Some(self_req) = self_req {
                self_req.names(&mut names);
            }
            Whole::Named(names)
        };
        let mut whole =
            Self::compute_with(ty, scopes, self_req, nested_type, any, enclosing, keep)?;
        // The Referrer knows the self type of the scope only if the top-level context records it:
        // it does if the context depends on it (`substituted`, see `names_self()`), which it does
        // when keeping the bounds makes a difference.
        if names_self_type && whole != context {
            if let Some(self_req) = self_req {
                if !whole.substituted.contains(self_req) {
                    whole.substituted.push(self_req.clone());
                }
            }
        }
        Ok(whole)
    }

    /// Like `compute()`; `whole` says which generics keep the bounds that [`LeftOut`] leaves out.
    fn compute_with(
        ty: &Type,
        scopes: &[Scope],
        self_req: Option<&SelfRequirement>,
        nested_type: bool,
        any: bool,
        enclosing: &Enclosing,
        whole: Whole,
    ) -> std::result::Result<Self, UnnameableSelf> {
        let mut names = HashSet::new();
        type_names(ty, &mut names);
        let mut mentions_self = type_mentions_self(ty);
        let free_self = nested_type
            && (any || !mentions_self)
            && matches!(self_req, Some(SelfRequirement::Type(_)));
        let mut self_names = HashSet::new();
        if let Some(self_req) = self_req {
            self_req.names(&mut self_names);
        }
        if mentions_self && !free_self {
            names.extend(self_names.iter().cloned());
        }
        // `Self` and `Self::A` are not replaced where they are written: the copy resolves them by
        // what declares `Self`, which the bounds of a method do not change.
        let inner = !(is_bare_self(ty) || is_self_projection(ty));
        let mut gather = Gather {
            scopes,
            self_req,
            work: Vec::new(),
            params: BTreeMap::new(),
            predicates: BTreeMap::new(),
            substituted: Vec::new(),
            left_out: LeftOut::new(whole, inner),
        };
        if let Some(from) = scopes.len().checked_sub(1) {
            push_names(&mut gather.work, scopes, from, names);
        }
        // An enclosing lifetime is a generic only if the type has a generic to bound by it.
        let ancestors = enclosing.generic(scopes);
        if !ancestors.is_empty() {
            let self_bound = matches!(self_req, Some(SelfRequirement::Bound(_)));
            let mentions = Mentions::of(ty, scopes, self_bound);
            let is_type_param = |name: &String| {
                scopes
                    .iter()
                    .any(|scope| scope.params().any(|param| is_type_param_named(param, name)))
            };
            let bounds_generic = (self_req.is_some()
                && mentions.types.iter().any(|name| name == "Self"))
                || mentions.types.iter().any(is_type_param)
                || !mentions.projections.is_empty();
            for (lifetime, depth) in &ancestors {
                let bounds_something = bounds_generic
                    || mentions
                        .lifetimes
                        .iter()
                        .any(|other| !same_lifetime(other, lifetime));
                if let (true, Some(from)) = (bounds_something, depth.checked_sub(1)) {
                    gather.work.push((from, lifetime_name(lifetime)));
                }
            }
        }
        let ancestors: Vec<Lifetime> = ancestors
            .into_iter()
            .map(|(lifetime, _)| lifetime)
            .collect();
        let mut swept_self = false;
        loop {
            let (from, name) = match gather.work.pop() {
                Some(item) => item,
                None if mentions_self && !swept_self => {
                    swept_self = true;
                    gather.sweep_self();
                    continue;
                }
                None => break,
            };
            let (ix, pos) = match declared_at(scopes, from, &name) {
                Some(found) => found,
                None => continue,
            };
            if gather.params.contains_key(&(ix, pos)) {
                continue;
            }
            if gather.take(ix, pos, &name)? && !mentions_self {
                mentions_self = true;
                // The impl for any type has no use for the generics of the type that `Self` is.
                if !free_self {
                    push_names(
                        &mut gather.work,
                        scopes,
                        scopes.len() - 1,
                        self_names.iter().cloned(),
                    );
                }
            }
        }
        let mut params: Vec<GenericParam> = gather.params.into_values().collect();
        let mut predicates: Vec<WherePredicate> = gather.predicates.into_values().collect();
        let mut self_outlives = Vec::new();
        // A written `Self: 'a` moves into `self_outlives`, so written and implied give one context.
        if self_req.is_some() {
            self_outlives.extend(written_self_outlives(&mut predicates, &params));
        }
        add_implied_outlives(
            ty,
            scopes,
            self_req,
            &ancestors,
            &mut params,
            &mut predicates,
            &mut self_outlives,
        );
        let generics = Generics {
            params: params.into_iter().collect(),
            where_clause: where_clause(predicates),
            ..Default::default()
        };
        let self_ty = match self_req.filter(|_| mentions_self) {
            Some(_) if free_self => Some(SelfRequirement::Any),
            self_req => self_req.cloned(),
        };
        let mut context = Self::new(generics, self_ty).with_self_outlives(self_outlives);
        context.substituted = gather.substituted;
        Ok(context)
    }

    pub(crate) fn none() -> Self {
        Self::new(Generics::default(), None)
    }

    /// No generics and no `Self`.
    pub(crate) fn is_empty(&self) -> bool {
        self.generics.params.is_empty()
            && self.generics.where_clause.is_none()
            && self.self_ty.is_none()
    }

    /// Returns `true` if `entry` is this context plus where predicates that name none of the
    /// generics (bounds on `Self` alone, such as `Self: Sized`). A copy that writes `Self`
    /// otherwise, as a type parameter, cannot repeat them.
    pub(crate) fn is_entry_with_extra_self_predicates(&self, entry: &Context) -> bool {
        let show = |tokens: &dyn ToTokens| tokens_text(tokens.to_token_stream());
        let outlives = |context: &Context| -> Vec<String> {
            context.self_outlives.iter().map(lifetime_name).collect()
        };
        if self.self_ty != entry.self_ty || outlives(self) != outlives(entry) {
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
                let used = names_in(predicate);
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

/// Identifies a context: what gives a leaked type its meaning, which is the generics it uses and
/// what `Self` is.
///
/// Every analysis method of the [`Leaker`](crate::Leaker) returns the id of the context of what it
/// analyzed (keep it: [`Referrer::visitor()`](crate::Referrer::visitor) takes the generics and
/// `Self` of the copy from it, and [`Referrer::complete_args()`](crate::Referrer::complete_args)
/// the generics as declared). Ids count from 0 in order of first appearance, and the Referrer may
/// skip some. `ToTokens` writes it as an integer literal,
/// and `Parse` reads one with or without `usize`.
///
/// The generics of a context are the ones its entries use: those the type names, those that their
/// bounds name and, if it names `Self`, those of the description of `Self`. A where predicate on
/// one generic is a bound of it, and one on another type (`Vec<T>: Clone`) belongs to each generic
/// it names. A bound of a generic that names the generic itself (`T: Fn(Alias<T>)`) is left out if
/// another bound of it fixes an associated type ([Bounds that name leaked
/// types](crate::Leaker#bounds-that-name-leaked-types)). The generics are matched with those of the
/// item by name ([`substitute_args()`](crate::substitute_args)).
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

/// The distinct [`Context`]s, numbered in order of first appearance. They are never renumbered,
/// except by [`Contexts::truncate()`] for a failed scope.
#[derive(Clone, Debug, Default)]
pub(crate) struct Contexts {
    ids: HashMap<Context, ContextId>,
    list: Vec<Context>,
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

    /// The id of `context`, if it is interned.
    pub(crate) fn find(&self, context: &Context) -> Option<ContextId> {
        self.ids.get(context).copied()
    }

    pub(crate) fn len(&self) -> usize {
        self.list.len()
    }

    pub(crate) fn get(&self, id: ContextId) -> &Context {
        &self.list[id.0]
    }

    /// Removes the contexts numbered `len` or more (for a failed scope).
    pub(crate) fn truncate(&mut self, len: usize) {
        for context in self.list.drain(len..) {
            self.ids.remove(&context);
        }
    }
}

/// Moves `Self: 'a` bounds, by a lifetime parameter of `params`, out of `predicates`.
/// `Self: Sized + 'a` leaves `Self: Sized`. Other lifetimes and `for<..>` predicates stay.
fn written_self_outlives(
    predicates: &mut Vec<WherePredicate>,
    params: &[GenericParam],
) -> Vec<Lifetime> {
    let is_param = |lifetime: &Lifetime| {
        params
            .iter()
            .any(|param| is_lifetime_param(param, lifetime))
    };
    let mut moved = Vec::new();
    let mut remaining = Vec::with_capacity(predicates.len());
    for predicate in std::mem::take(predicates) {
        match predicate {
            WherePredicate::Type(mut predicate)
                if predicate.lifetimes.is_none() && is_bare_self(&predicate.bounded_ty) =>
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
/// generic: `T: Tr`, `(T): Tr` and `'a: 'b`, but not `Vec<T>: Tr`, `for<'x> T: Tr<'x>` or `Self: Tr`.
fn bounded_generic(predicate: &WherePredicate) -> Option<String> {
    match predicate {
        WherePredicate::Type(PredicateType {
            lifetimes: None,
            bounded_ty,
            ..
        }) => bounded_ident(bounded_ty),
        WherePredicate::Lifetime(PredicateLifetime { lifetime, .. }) => {
            Some(lifetime_name(lifetime))
        }
        _ => None,
    }
}

/// Returns `true` if `predicate` of `scopes[at]` bounds a generic in scope there or an associated
/// type of one (`T`, `T::Item`, `<T as Tr>::Item`, `'a`), which only the environment proves.
fn bounds_generic(scopes: &[Scope], at: usize, predicate: &WherePredicate) -> bool {
    let head = match predicate {
        WherePredicate::Type(predicate) => match unwrap_type_group(&predicate.bounded_ty) {
            Type::Path(TypePath {
                qself: Some(qself), ..
            }) => bounded_ident(&qself.ty),
            Type::Path(TypePath { qself: None, path }) if path.leading_colon.is_none() => path
                .segments
                .first()
                .filter(|seg| seg.arguments.is_none())
                .map(|seg| ident_name(&seg.ident)),
            _ => None,
        },
        WherePredicate::Lifetime(predicate) => Some(lifetime_name(&predicate.lifetime)),
        _ => None,
    };
    head.map_or(false, |head| declared_at(scopes, at, &head).is_some())
}

/// The generics and where predicates that `Context::compute_with()` gathers from the scopes.
struct Gather<'a> {
    scopes: &'a [Scope],
    /// What `Self` means where the type is written.
    self_req: Option<&'a SelfRequirement>,
    /// The names still to take, each with the scope where it is written (see [`push_names()`]).
    work: Vec<(usize, String)>,
    /// The generics taken, by scope and position.
    params: BTreeMap<(usize, usize), GenericParam>,
    /// The where predicates taken, by scope and position.
    predicates: BTreeMap<(usize, usize), WherePredicate>,
    /// The outer `Self`s that a bound was rewritten for, in the order found.
    substituted: Vec<SelfRequirement>,
    left_out: LeftOut,
}

impl<'a> Gather<'a> {
    /// The `Self` of the scope `ix`, if it is not the one where the type is written.
    fn foreign(&self, ix: usize) -> Option<&'a SelfRequirement> {
        let self_req = self.self_req;
        self.scopes[ix]
            .self_ty()
            .filter(|outer| Some(*outer) != self_req)
    }

    /// Takes the where predicates on `Self` (`where Self: Sized`): the impl must declare them too.
    fn sweep_self(&mut self) {
        let scopes = self.scopes;
        for (ix, scope) in scopes.iter().enumerate() {
            // What bounds another `Self` is no business of this one.
            if self.foreign(ix).is_some() {
                continue;
            }
            for (k, predicate) in scope.predicates.iter().enumerate() {
                let bounds_self = match predicate {
                    WherePredicate::Type(predicate) => type_mentions_self(&predicate.bounded_ty),
                    _ => false,
                };
                if bounds_self && !self.predicates.contains_key(&(ix, k)) {
                    let names = names_in(predicate);
                    self.predicates.insert((ix, k), predicate.clone());
                    push_names(&mut self.work, scopes, ix, names);
                }
            }
        }
    }

    /// Takes the generic `name`, the parameter `pos` of `scopes[ix]`, and the where predicates
    /// that belong to it, of that scope and of the scopes inside it, and puts the names they write
    /// on `work`. Returns `true` if they mention `Self`.
    fn take(
        &mut self,
        ix: usize,
        pos: usize,
        name: &str,
    ) -> std::result::Result<bool, UnnameableSelf> {
        let scopes = self.scopes;
        // The bounds are written for the `Self` of the scope, which may not be this one.
        let outer = self.foreign(ix);
        let mut param = scopes[ix].params[pos].clone();
        if let GenericParam::Type(param) = &mut param {
            let left_out = self.left_out.of(scopes, ix, name);
            if !left_out.is_empty() {
                param.bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| !left_out.contains(&bound_text(bound)))
                    .cloned()
                    .collect();
            }
        }
        if let Some(outer) = outer.filter(|_| generic_param_mentions_self(&param)) {
            param = self.substitute(outer, &param)?;
        }
        let names = names_in(&param);
        push_names(&mut self.work, scopes, ix, names);
        let mut found_self = generic_param_mentions_self(&param);
        self.params.insert((ix, pos), param);
        for at in ix..self.left_out.reach(scopes, ix) {
            // A scope inside (a method) sees the generic, unless it declares one of the name.
            if declared_at(scopes, at, name) != Some((ix, pos)) {
                continue;
            }
            // The predicates are written for the `Self` of their own scope.
            let outer = self.foreign(at);
            for (k, predicate) in scopes[at].predicates.iter().enumerate() {
                // A scope inside adds its bounds of the generics around it (`T: Clone`, `T::Item:
                // Clone`), which the type may need. Its predicates on other types (`Rel<T>:
                // Default`) are conditions of its own item, which the copy keeps.
                if self.predicates.contains_key(&(at, k))
                    || (at > ix && !bounds_generic(scopes, at, predicate))
                {
                    continue;
                }
                // The bounds of a generic that the context leaves out are not its predicates.
                let predicate = &match self.left_out.without(scopes, at, predicate) {
                    Some(predicate) => predicate,
                    None => continue,
                };
                let mut names = names_in(predicate);
                // A predicate on a generic is a bound of it, as `canonicalize()` makes it: it is
                // in the context of what uses that generic, not of what its arguments name. So
                // `T: Iterator<Item = Rel<'a>>` is no more in the context of `Rel<'a>` than the
                // inline bound is, which the Referrer sees. Any other predicate belongs to each
                // generic it names.
                let bounded = bounded_generic(predicate)
                    .and_then(|bounded| declared_at(scopes, at, &bounded));
                let belongs = match bounded {
                    Some(bounded) => bounded == (ix, pos),
                    None => names.contains(name),
                };
                if belongs {
                    let mut predicate = predicate.clone();
                    if let Some(outer) = outer.filter(|_| where_predicate_mentions_self(&predicate))
                    {
                        predicate = self.substitute(outer, &predicate)?;
                        names = names_in(&predicate);
                    }
                    push_names(&mut self.work, scopes, at, names);
                    found_self |= where_predicate_mentions_self(&predicate);
                    self.predicates.insert((at, k), predicate);
                }
            }
        }
        Ok(found_self)
    }

    /// Rewrites `node` of a scope whose `Self` is `outer` (see [`for_self_of()`]), and records
    /// `outer` in `substituted`.
    fn substitute<T: WrittenForSelf>(
        &mut self,
        outer: &SelfRequirement,
        node: &T,
    ) -> std::result::Result<T, UnnameableSelf> {
        let node = for_self_of(outer, node)?;
        if !self.substituted.contains(outer) {
            self.substituted.push(outer.clone());
        }
        Ok(node)
    }
}

/// The bounds that a context leaves out (see [`left_out_bounds()`]), by scope and type parameter,
/// except for the generics that [`Whole`] keeps.
struct LeftOut {
    whole: Whole,
    /// The scopes inside the one that declares a generic bound it too.
    inner: bool,
    cache: HashMap<(usize, String), HashSet<String>>,
}

/// The generics of a context that keep every bound.
enum Whole {
    Nothing,
    /// The generics with these names.
    Named(HashSet<String>),
    Everything,
}

impl Whole {
    fn keeps(&self, name: &str) -> bool {
        match self {
            Self::Nothing => false,
            Self::Named(names) => names.contains(name),
            Self::Everything => true,
        }
    }
}

impl LeftOut {
    fn new(whole: Whole, inner: bool) -> Self {
        Self {
            whole,
            inner,
            cache: HashMap::new(),
        }
    }

    /// The end of the scopes whose predicates bound a generic of `scopes[ix]`.
    fn reach(&self, scopes: &[Scope], ix: usize) -> usize {
        if self.inner {
            scopes.len()
        } else {
            ix + 1
        }
    }

    /// The texts of the bounds of the type parameter `name` of `scopes[ix]` that the context
    /// leaves out.
    fn of(&mut self, scopes: &[Scope], ix: usize, name: &str) -> &HashSet<String> {
        let kept = self.whole.keeps(name);
        let reach = self.reach(scopes, ix);
        self.cache.entry((ix, name.to_string())).or_insert_with(|| {
            if kept {
                HashSet::new()
            } else {
                left_out_bounds(&scopes[..reach], ix, name)
            }
        })
    }

    /// `predicate` of `scopes[at]` without the bounds of the generic it bounds that the context
    /// leaves out: `None` if nothing is left.
    fn without(
        &mut self,
        scopes: &[Scope],
        at: usize,
        predicate: &WherePredicate,
    ) -> Option<WherePredicate> {
        let (bounded, predicate_type) = match predicate {
            WherePredicate::Type(predicate_type) => (
                bounded_ident(&predicate_type.bounded_ty).and_then(|name| {
                    declared_at(scopes, at, &name).map(|(declared, _)| (declared, name))
                }),
                predicate_type,
            ),
            _ => return Some(predicate.clone()),
        };
        let left_out = match bounded {
            Some((declared, name)) => self.of(scopes, declared, &name),
            None => return Some(predicate.clone()),
        };
        if left_out.is_empty() {
            return Some(predicate.clone());
        }
        let bounds: Punctuated<TypeParamBound, Token![+]> = predicate_type
            .bounds
            .iter()
            .filter(|bound| !left_out.contains(&bound_text(bound)))
            .cloned()
            .collect();
        if bounds.is_empty() {
            return None;
        }
        Some(WherePredicate::Type(PredicateType {
            bounds,
            ..predicate_type.clone()
        }))
    }
}

/// The texts of the bounds of the type parameter `name` of `scopes[ix]` that its context leaves
/// out (see [`hiding_bounds()`]), with the bounds that the scopes inside write for it. The leaked
/// type may need one (`struct Rel<T: Fn(X<T>)>`): then rustc rejects the Repeater's impl (E0277).
fn left_out_bounds(scopes: &[Scope], ix: usize, name: &str) -> HashSet<String> {
    let mut bounds: Vec<&TypeParamBound> = Vec::new();
    for param in scopes[ix].params() {
        match param {
            GenericParam::Type(param) if ident_name(&param.ident) == name => {
                bounds.extend(param.bounds.iter())
            }
            _ => (),
        }
    }
    let sees =
        |at: &usize| declared_at(scopes, *at, name).map(|(declared, _)| declared) == Some(ix);
    for predicate in (ix..scopes.len())
        .filter(sees)
        .flat_map(|at| scopes[at].predicates())
    {
        match predicate {
            WherePredicate::Type(predicate)
                if bounded_ident(&predicate.bounded_ty).as_deref() == Some(name) =>
            {
                bounds.extend(predicate.bounds.iter())
            }
            _ => (),
        }
    }
    // rustc proves the impl's bounds where the copy names the entry in a predicate, before it
    // normalizes them: `T: Fn(<T as Iterator>::Item)` is no `T: Fn(u8)` there (E0277), and the
    // impl does not need such a bound to name the entry.
    hiding_bounds(&bounds, name, false)
        .into_iter()
        .map(bound_text)
        .collect()
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
        format!("the where predicate `{}`", tidy(&quote!(#self)))
    }
    fn rewrite(self, folder: &mut ReplaceSelf<'_>) -> Self {
        folder.fold_where_predicate(self)
    }
}

/// The scope and the position of the generic `name` that is in scope in `scopes[from]`: the
/// innermost one that declares it.
fn declared_at(scopes: &[Scope], from: usize, name: &str) -> Option<(usize, usize)> {
    (0..=from).rev().find_map(|ix| {
        scopes[ix]
            .params
            .iter()
            .position(|param| generic_param_name(param) == name)
            .map(|pos| (ix, pos))
    })
}

/// Puts the `names` written in `scopes[from]` on `work`, to be taken in the order they are
/// declared (outer scopes first, what is no generic first).
fn push_names(
    work: &mut Vec<(usize, String)>,
    scopes: &[Scope],
    from: usize,
    names: impl IntoIterator<Item = String>,
) {
    let mut names: Vec<(Option<(usize, usize)>, String)> = names
        .into_iter()
        .map(|name| (declared_at(scopes, from, &name), name))
        .collect();
    // Not the order of the set: the walk reports the first failure it finds and lists the outer
    // `Self`s that it rewrites in the order found.
    names.sort();
    work.extend(names.into_iter().rev().map(|(_, name)| (from, name)));
}

/// Rewrites `node`, written where `Self` is `outer`, with `Self` replaced by the type `outer` is.
/// Fails if `outer` is the implementor of a trait or `node` names an associated item of `Self`
/// (`S::X` of a type is ambiguous).
fn for_self_of<T: WrittenForSelf>(
    outer: &SelfRequirement,
    node: &T,
) -> std::result::Result<T, UnnameableSelf> {
    let mentions = node.self_mentions();
    let note = match outer {
        // No scope declares an `Any` `Self`: it is the context of an entry.
        SelfRequirement::Any => format!(
            "{} mentions a `Self` of no type, which has no name here",
            node.what()
        ),
        SelfRequirement::Bound(path) => format!(
            "{} mentions the `Self` of the trait `{}`, which has no name here: `Self` is \
             something else in the item around",
            node.what(),
            tidy(&quote!(#path))
        ),
        SelfRequirement::Type(ty) => {
            return match mentions.associated {
                Some(cause) => Err(UnnameableSelf {
                    headline: format!(
                        "{} mentions `{}`, which has no name here",
                        node.what(),
                        tidy(&cause.shown())
                    ),
                    note: format!(
                        "{} mentions `{}`, an associated item of the `Self` of `{}`, which has \
                         no name here: `Self` is something else in the item around",
                        node.what(),
                        tidy(&cause.shown()),
                        tidy(&quote!(#ty))
                    ),
                    help: format!("write {} without `{}`", node.what(), tidy(&cause.shown())),
                    cause: Box::new(cause),
                }),
                None => Ok(node.clone().rewrite(&mut ReplaceSelf { self_ty: ty })),
            }
        }
    };
    Err(UnnameableSelf {
        cause: Box::new(mentions.first.unwrap_or_else(Cause::self_type)),
        headline: format!("{} mentions a `Self` which has no name here", node.what()),
        note,
        help: format!("write {} without `Self`", node.what()),
    })
}

/// The lifetimes a type must outlive because of the types around it, as in `&'a Vec<Rel2<T>>` (see
/// [Outlives bounds](crate::Leaker#outlives-bounds)). `T::A<X>` is decided late, by the scopes in
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
        if !is_named(&lifetime.ident, "static") && lifetime.ident != "_" {
            push_once(&mut self.0, link);
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
                    let known = active
                        .iter()
                        .any(|(other, at)| at == depth && same_lifetime(other, lifetime));
                    if !known {
                        active.push((lifetime.clone(), *depth));
                    }
                }
            }
        }
        active
    }

    /// The generic lifetimes among them, outermost first, each with its depth.
    fn generic(&self, scopes: &[Scope]) -> Vec<(Lifetime, usize)> {
        let mut generic: Vec<(Lifetime, usize)> = Vec::new();
        for (lifetime, depth) in self.active(scopes) {
            // The scopes in scope where it is written decide what it names.
            if is_generic_lifetime(visible(scopes, depth), &[], &lifetime)
                && !generic
                    .iter()
                    .any(|(other, _)| same_lifetime(other, &lifetime))
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

/// `name` is a type parameter of the innermost scope that declares it.
fn is_type_parameter(name: &str, scopes: &[Scope]) -> bool {
    scopes
        .iter()
        .rev()
        .find(|scope| scope.declares(name))
        .map_or(false, |scope| {
            scope.params().any(|param| is_type_param_named(param, name))
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
    !is_named(&lifetime.ident, "static")
        && lifetime.ident != "_"
        && !has_lifetime(binders, lifetime)
        && scopes
            .iter()
            .rev()
            .find(|scope| scope.declares(&lifetime_name(lifetime)))
            .map_or(false, |scope| !scope.is_binder)
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
    fn of(ty: &Type, scopes: &'a [Scope], self_bound: bool) -> Self {
        let mut mentions = Self::new(scopes, self_bound, &[], Vec::new());
        mentions.visit_type(ty);
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
                // Not the bounds of its parts: the copy knows only that the projection outlives
                // `'a` (E0309). So `Self::A` of a trait brings no `Self: 'a`, which would be
                // stricter than the original.
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

/// Finds the outlives bounds that a type implies (see [`implied_outlives()`]).
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
            let is_param = self
                .params
                .iter()
                .any(|param| is_type_param_named(param, &name));
            if is_param {
                self.found.push((Bounded::Type(name), lifetime.clone()));
            }
        }
        for other in mentions.lifetimes {
            let is_param = self
                .params
                .iter()
                .any(|param| is_lifetime_param(param, &other));
            if is_param && !same_lifetime(&other, lifetime) {
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
        // A lifetime argument implies nothing: a guessed bound could make the copy stricter.
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

/// Returns `true` if `ty` is an associated type of a generic (`T::A`, `<T as Tr>::A`, `<T::A as
/// Tr2>::B`, and `Self::A` with `self_bound`), which `&'a ty` makes outlive `'a` as a whole. Not
/// `<Vec<T> as IntoIterator>::Item`, which rustc normalizes.
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

/// Returns `true` if the projection `ty` can be the bounded type of a where predicate of the
/// context: its lifetimes are `'static` or the context's (none elided), and it has no `impl
/// Trait`, `_`, type macro, function pointer or `Fn(..)` sugar.
fn is_writable_projection(ty: &TypePath, scopes: &[Scope], binders: &[Lifetime]) -> bool {
    struct Writable<'a> {
        scopes: &'a [Scope],
        binders: &'a [Lifetime],
        writable: bool,
    }
    impl<'a, 'ast> Visit<'ast> for Writable<'a> {
        fn visit_lifetime(&mut self, i: &'ast Lifetime) {
            if !is_named(&i.ident, "static") && !is_generic_lifetime(self.scopes, self.binders, i) {
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

/// An associated type that a bound of the context equates (`T: Iterator<Item = X>`, or `Output`
/// of `F: Fn(A) -> B`). rustc normalizes `T::Item` to `X`, so `&'a T::Item` requires what `X`
/// requires.
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
        for (bounded, bound) in trait_bounds(params.iter(), predicates.iter()) {
            let seg = match bound.path.segments.last() {
                Some(seg) => seg,
                None => continue,
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
                    bounded,
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
                    tokens_text(ty),
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

/// Adds the outlives bounds that `ty` implies (see [`implied_outlives()`]) to `params`, the where
/// predicates `predicates` and `self_outlives`.
fn add_implied_outlives(
    ty: &Type,
    scopes: &[Scope],
    self_req: Option<&SelfRequirement>,
    ancestors: &[Lifetime],
    params: &mut [GenericParam],
    predicates: &mut Vec<WherePredicate>,
    self_outlives: &mut Vec<Lifetime>,
) {
    let env = Binding::of(params, predicates);
    for (bounded, lifetime) in implied_outlives(ty, scopes, params, &env, self_req, ancestors) {
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
                    if same_lifetime(&param.lifetime, name) =>
                {
                    param.bounds.push(lifetime.clone())
                }
                _ => (),
            }
        }
    }
}

/// Returns the outlives bounds `ty` implies on `params`: `&'a X` requires `X: 'a`, as does each of
/// `ancestors` for `ty`, and a projection of a generic is one unit (`T::A: 'a`). See [Outlives
/// bounds](crate::Leaker#outlives-bounds).
fn implied_outlives(
    ty: &Type,
    scopes: &[Scope],
    params: &[GenericParam],
    env: &[Binding],
    self_req: Option<&SelfRequirement>,
    ancestors: &[Lifetime],
) -> Vec<(Bounded, Lifetime)> {
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
        let target = bounded_generic(&predicate).and_then(|name| {
            params.iter().position(|param| {
                !matches!(param, GenericParam::Const(_)) && generic_param_name(param) == name
            })
        });
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
                param.bounds = sorted_by_tokens(std::mem::take(&mut param.bounds))
                    .into_iter()
                    .collect();
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
    let predicates = sorted_by_tokens(predicates.into_iter().map(|mut predicate| {
        if let WherePredicate::Type(predicate) = &mut predicate {
            predicate.bounds = canonical_bounds(std::mem::take(&mut predicate.bounds));
        }
        predicate
    }));
    let has_params = !params.is_empty();
    Generics {
        lt_token: has_params.then(Default::default),
        params: params.into_iter().collect(),
        gt_token: has_params.then(Default::default),
        where_clause: where_clause(predicates),
    }
}

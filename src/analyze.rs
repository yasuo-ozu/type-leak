//! The analysis of items and types by the [`Leaker`].
//!
//! [`AnalyzeVisitor`] visits each type once, in post-order: the causes found in a type (see
//! [`CheckResult`]) decide whether it is interned as a whole, decomposed into its components, or
//! reported as an error, and the effects of its components (the types to intern, the errors and
//! the assumed primitives) are kept or discarded accordingly.

use crate::encode::is_const_param_ty;
use crate::graph::{TypeGraph, TypeKey};
use crate::leaker::Leaker;
use crate::util::{
    generic_param_name, is_primitive_name, looks_like_const, mentions_generics, token_names,
    unwrap_type_group,
};
use proc_macro2::TokenStream;
use std::collections::HashSet;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// The result of [`Leaker::check()`]: the causes which make a type need to be interned, and those
/// which make it impossible to intern.
///
/// | `must` | `impossible` | the type |
/// |---|---|---|
/// | empty | empty | does not depend on the type context, and can be copied as it is |
/// | non-empty, [`is_own`](CheckResult::is_own) | empty | must be interned (e.g. `MyType`) |
/// | non-empty, not `is_own` | empty | must be interned, or its components (e.g. `::path::Wrap<MyType>`) |
/// | empty | non-empty | cannot be interned, but does not need to be (e.g. `impl ::path::Trait`) |
/// | non-empty | non-empty | must be interned but cannot be: a conflict (e.g. `MyType<'_>`, or `::path::Wrap<MyType, '_>`, whose components can be interned instead unless `is_own`) |
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CheckResult {
    /// The causes which depend on the type context, so they must be interned (e.g. `A` and `B` in
    /// `::path::Wrap<A, B>`). The outermost ones come first (in the order of appearance among
    /// those at the same depth).
    pub must: Vec<TokenStream>,
    /// The causes which cannot be interned (e.g. an elided lifetime or `impl Trait`), in the
    /// order of appearance.
    pub impossible: Vec<TokenStream>,
    /// `true` if the first of `must` is the type's own path (e.g. `MyType` in `MyType<'_>`), so
    /// the type must be interned as a whole: it cannot be decomposed, i.e. copied while interning
    /// only its components. `false` if `must` is empty, or all of `must` are in components (e.g.
    /// `MyType` in `::path::Wrap<MyType, '_>`), which can be interned alone.
    pub is_own: bool,
}

impl CheckResult {
    /// Builds the result from the causes, each with the depth of the type it belongs to (`0` for
    /// the type itself).
    fn from_causes(mut must: Vec<(usize, TokenStream)>, impossible: Vec<TokenStream>) -> Self {
        // The outermost causes first (in the order of appearance among the same depth).
        must.sort_by_key(|(depth, _)| *depth);
        CheckResult {
            is_own: must.first().map_or(false, |(depth, _)| *depth == 0),
            must: must.into_iter().map(|(_, m)| m).collect(),
            impossible,
        }
    }
}

/// Returns the local generics which `ty` depends on, including those used in the bounds and
/// where predicates of them, in the order of declaration.
///
/// Identifiers are matched by name, so a local generic may be included even if the name refers to
/// something else (e.g. `::path::N`). This is harmless, because the Leaker and the Referrer
/// include the same generics.
fn required_local_generics<'a>(
    ty: &Type,
    locals: impl Iterator<Item = &'a GenericParam> + Clone,
    predicates: &[WherePredicate],
) -> Generics {
    let mut names = HashSet::new();
    token_names(quote!(#ty), &mut names);
    let mut included_params = vec![false; locals.clone().count()];
    let mut included_predicates = vec![false; predicates.len()];
    loop {
        let mut changed = false;
        for (included, param) in included_params.iter_mut().zip(locals.clone()) {
            if !*included && names.contains(&generic_param_name(param)) {
                *included = true;
                token_names(quote!(#param), &mut names);
                changed = true;
            }
        }
        for (included, predicate) in included_predicates.iter_mut().zip(predicates) {
            if *included {
                continue;
            }
            let mut predicate_names = HashSet::new();
            token_names(quote!(#predicate), &mut predicate_names);
            let mentions_included = locals
                .clone()
                .zip(&included_params)
                .any(|(param, inc)| *inc && predicate_names.contains(&generic_param_name(param)));
            if mentions_included {
                *included = true;
                names.extend(predicate_names);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let params: Punctuated<GenericParam, Token![,]> = locals
        .zip(&included_params)
        .filter(|(_, inc)| **inc)
        .map(|(param, _)| {
            let mut param = param.clone();
            match &mut param {
                GenericParam::Type(tp) => {
                    tp.eq_token = None;
                    tp.default = None;
                }
                GenericParam::Const(cp) => {
                    cp.eq_token = None;
                    cp.default = None;
                }
                GenericParam::Lifetime(_) => (),
            }
            param
        })
        .collect();
    let predicates: Punctuated<WherePredicate, Token![,]> = predicates
        .iter()
        .zip(&included_predicates)
        .filter(|(_, inc)| **inc)
        .map(|(predicate, _)| predicate.clone())
        .collect();
    Generics {
        lt_token: Some(Default::default()),
        gt_token: Some(Default::default()),
        where_clause: if predicates.is_empty() {
            None
        } else {
            Some(WhereClause {
                where_token: Default::default(),
                predicates,
            })
        },
        params,
    }
}

/// A type to be added to the [`TypeGraph`], with its components.
struct PendingType {
    ty: Type,
    is_must_intern: bool,
    local_generics: Generics,
    /// The type looks like a const (see [`looks_like_const()`]), which is an error unless it is a
    /// component of another type.
    looks_like_const: bool,
    /// The effects of its components, applied if the type is added.
    components: Vec<Effect>,
}

/// An effect of the analysis, applied (in order) when the analysis is finished.
enum Effect {
    /// A type to add (not as a component of another type).
    Type(Box<PendingType>),
    /// A relative path which cannot be interned.
    Error(CheckResult),
    /// The name of a primitive type copied as it is.
    Primitive(Ident),
}

/// What is found in a type (or outside any type).
#[derive(Default)]
struct Frame {
    /// The causes of the [`CheckResult`] of the type, with the depth of the type they belong to.
    must: Vec<(usize, TokenStream)>,
    impossible: Vec<TokenStream>,
    /// The effects, which are applied if the type is decomposed.
    effects: Vec<Effect>,
}

/// The result of [`AnalyzeVisitor`].
pub(crate) struct Analysis {
    pub(crate) types: TypeGraph,
    /// The relative paths which cannot be interned.
    pub(crate) errors: Vec<CheckResult>,
    /// The first type used with different local generics in different places.
    pub(crate) conflicting_local_generics: Option<TokenStream>,
    /// The names of the primitive types copied as they are (see [`Leaker::assertions()`]).
    pub(crate) primitives: Vec<Ident>,
}

/// Analyzes items (see [`Leaker::intern_with()`]) and types (see [`Leaker::check()`]).
pub(crate) struct AnalyzeVisitor<'a> {
    leaker: &'a Leaker,
    /// The generics in scope: those of the Leaker, then local ones (e.g. of trait methods, GATs
    /// or `for<..>` binders).
    generics: Generics,
    /// The number of leading `generics.params` which belong to the Leaker itself. The rest are
    /// local generics, which cannot be referred to from the Repeater's implementation.
    n_outer_generics: usize,
    /// Where predicates of the local generics.
    local_predicates: Vec<WherePredicate>,
    /// What is found in the type being visited (or outside any type).
    frame: Frame,
    /// The number of types enclosing the syntax being visited.
    type_depth: usize,
    /// The result of the last type visited.
    last_result: Option<CheckResult>,
}

impl<'a> AnalyzeVisitor<'a> {
    pub(crate) fn new(leaker: &'a Leaker, generics: &Generics) -> Self {
        Self {
            leaker,
            generics: generics.clone(),
            n_outer_generics: generics.params.len(),
            local_predicates: Vec::new(),
            frame: Frame::default(),
            type_depth: 0,
            last_result: None,
        }
    }

    /// Analyzes `ty` alone (see [`Leaker::check()`]).
    pub(crate) fn check(leaker: &Leaker, ty: &Type, generics: &Generics) -> CheckResult {
        let mut visitor = AnalyzeVisitor::new(leaker, generics);
        visitor.visit_type(ty);
        visitor.last_result.unwrap()
    }

    /// Finishes the analysis, adding the types found to a graph.
    pub(crate) fn finish(self) -> Analysis {
        let mut analysis = Analysis {
            types: TypeGraph::default(),
            errors: Vec::new(),
            conflicting_local_generics: None,
            primitives: Vec::new(),
        };
        analysis.apply(self.frame.effects, None);
        analysis
    }

    /// Decides how to handle the type `ty`, given what is found in it (`inner`).
    fn decide(&mut self, ty: &Type, inner: Frame) {
        let Frame {
            must,
            impossible,
            effects,
        } = inner;
        // The causes also belong to the enclosing type, as those of a component.
        self.frame
            .must
            .extend(must.iter().map(|(depth, m)| (depth + 1, m.clone())));
        self.frame.impossible.extend(impossible.iter().cloned());
        let result = CheckResult::from_causes(must, impossible);

        let locals = self.generics.params.iter().skip(self.n_outer_generics);
        if !result.impossible.is_empty() {
            if result.is_own {
                // The type must be interned (because of its own path, the first of `must`, which
                // depends on the type context) but cannot be (e.g. `MyType<'_>`), and cannot be
                // decomposed either. The path should be made absolute.
                self.frame.effects.push(Effect::Error(result.clone()));
            } else if !result.must.is_empty() {
                // The type cannot be interned as a whole (e.g. `&MyType` with an elided lifetime,
                // or `impl Iterator<Item = MyType>`), but its components can.
                self.frame.effects.extend(effects);
            }
            // Otherwise, the type cannot be interned, but does not need to be (e.g.
            // `impl ::path::Trait`).
        } else if mentions_generics(ty, locals.clone()) {
            if result.is_own {
                // The type depends on both the type context and local generics, and cannot be
                // decomposed further. It is interned together with the local generics it needs,
                // which the Repeater's implementation must declare (see
                // [`Referrer::local_generics()`](crate::Referrer::local_generics())).
                self.frame.effects.push(Effect::Type(Box::new(PendingType {
                    ty: ty.clone(),
                    is_must_intern: true,
                    local_generics: required_local_generics(ty, locals, &self.local_predicates),
                    looks_like_const: false,
                    components: Vec::new(),
                })));
            } else {
                // The type itself cannot be interned because the Repeater's implementation
                // cannot refer to the local generics, but its components can.
                self.frame.effects.extend(effects);
            }
        } else if result.is_own {
            // The type must be interned as a whole.
            self.frame.effects.push(Effect::Type(Box::new(PendingType {
                ty: ty.clone(),
                is_must_intern: true,
                local_generics: Generics::default(),
                looks_like_const: looks_like_const(ty),
                components: Vec::new(),
            })));
        } else {
            // The type may be interned as a whole, or its components.
            self.frame.effects.push(Effect::Type(Box::new(PendingType {
                ty: ty.clone(),
                is_must_intern: false,
                local_generics: Generics::default(),
                looks_like_const: false,
                components: effects,
            })));
        }
        self.last_result = Some(result);
    }

    /// Records that the relative path `must` cannot be interned (because of `impossible`, e.g. a
    /// trait bound or a value containing it), so it should be absolute.
    fn relative_path(&mut self, must: TokenStream, impossible: TokenStream) {
        self.frame.effects.push(Effect::Error(CheckResult {
            must: vec![must],
            impossible: vec![impossible],
            is_own: true,
        }));
    }

    fn is_generic_lifetime(&self, lifetime: &Lifetime) -> bool {
        self.generics.params.iter().any(
            |param| matches!(param, GenericParam::Lifetime(param) if &param.lifetime == lifetime),
        )
    }

    /// Returns `true` if `path` starts with a generic parameter (e.g. `T` or `T::Item`).
    fn starts_with_generic(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && path.segments.first().map_or(false, |seg| {
                seg.arguments.is_none()
                    && self.generics.params.iter().any(|param| match param {
                        GenericParam::Type(TypeParam { ident, .. })
                        | GenericParam::Const(ConstParam { ident, .. }) => ident == &seg.ident,
                        GenericParam::Lifetime(_) => false,
                    })
            })
    }

    /// Runs `f` with `generics` (of a trait method or a GAT) added as local generics.
    fn with_local_generics(&mut self, generics: &Generics, f: impl FnOnce(&mut Self)) {
        let n_params = self.generics.params.len();
        let n_predicates = self.local_predicates.len();
        self.generics.params.extend(generics.params.iter().cloned());
        if let Some(wc) = &generics.where_clause {
            self.local_predicates.extend(wc.predicates.iter().cloned());
        }
        f(self);
        while self.generics.params.len() > n_params {
            self.generics.params.pop();
        }
        self.local_predicates.truncate(n_predicates);
    }

    /// Runs `f` with the lifetimes of a `for<..>` binder (e.g. `'a` in `for<'a> Fn(&'a T)`) in
    /// scope. They are treated as local generics, which the Repeater's implementation cannot
    /// refer to.
    fn with_bound_lifetimes(
        &mut self,
        lifetimes: Option<&BoundLifetimes>,
        f: impl FnOnce(&mut Self),
    ) {
        let len = self.generics.params.len();
        if let Some(lifetimes) = lifetimes {
            self.generics
                .params
                .extend(lifetimes.lifetimes.iter().cloned());
        }
        f(self);
        while self.generics.params.len() > len {
            self.generics.params.pop();
        }
    }
}

impl Analysis {
    /// Applies `effects` in order, adding the types as components of `parent` (if any).
    fn apply(&mut self, effects: Vec<Effect>, parent: Option<&TypeKey>) {
        for effect in effects {
            match effect {
                Effect::Type(pending) => {
                    let ty = &pending.ty;
                    if parent.is_none() && pending.looks_like_const {
                        // A bare identifier such as `SIZE` in `T: MyTrait<SIZE>` is parsed as a
                        // type, but is probably a const (by the naming convention), which cannot be
                        // leaked on its own (like `T: MyTrait<{ SIZE }>`).
                        self.errors.push(CheckResult {
                            must: vec![quote!(#ty)],
                            impossible: vec![quote!(#ty)],
                            is_own: true,
                        });
                        continue;
                    }
                    match self
                        .types
                        .add(ty, parent, pending.is_must_intern, pending.local_generics)
                    {
                        Ok(key) => self.apply(pending.components, Some(&key)),
                        Err(()) => {
                            self.conflicting_local_generics
                                .get_or_insert_with(|| quote!(#ty));
                        }
                    }
                }
                Effect::Error(error) => self.errors.push(error),
                Effect::Primitive(ident) => self.primitives.push(ident),
            }
        }
    }
}

impl<'a, 'ast> Visit<'ast> for AnalyzeVisitor<'a> {
    fn visit_trait_item_fn(&mut self, i: &TraitItemFn) {
        let mut i = i.clone();
        i.default = None;
        i.semi_token = Some(Default::default());
        self.with_local_generics(&i.sig.generics, |visitor| {
            syn::visit::visit_trait_item_fn(visitor, &i)
        });
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_type(visitor, i)
        });
    }

    fn visit_trait_item_const(&mut self, i: &TraitItemConst) {
        // Like the bodies of methods, defaults of associated consts are not copied.
        let mut i = i.clone();
        i.default = None;
        syn::visit::visit_trait_item_const(self, &i)
    }

    fn visit_type(&mut self, i: &Type) {
        let outer = std::mem::take(&mut self.frame);
        self.type_depth += 1;
        match i {
            Type::Macro(_) => {
                // The tokens of a type macro cannot be analyzed, so it is interned as a whole (it
                // is expanded in the Leaker's context).
                self.frame.must.push((0, quote!(#i)));
            }
            Type::ImplTrait(_)
            | Type::Verbatim(_)
            | Type::Reference(TypeReference { lifetime: None, .. }) => {
                self.frame.impossible.push(quote!(#i));
                syn::visit::visit_type(self, i);
            }
            _ => syn::visit::visit_type(self, i),
        }
        self.type_depth -= 1;
        let inner = std::mem::replace(&mut self.frame, outer);
        self.decide(i, inner);
    }

    fn visit_type_bare_fn(&mut self, i: &TypeBareFn) {
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_type_bare_fn(v, i)
        })
    }

    fn visit_type_trait_object(&mut self, i: &TypeTraitObject) {
        // All the trait bounds are checked (e.g. both `A` and `B` in `dyn A + B`).
        for bound in &i.bounds {
            match bound {
                TypeParamBound::Trait(bound) => self.visit_trait_bound(bound),
                TypeParamBound::Verbatim(_) => self.frame.impossible.push(quote!(#bound)),
                _ => (),
            }
        }
    }

    fn visit_predicate_type(&mut self, i: &PredicateType) {
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_predicate_type(v, i)
        })
    }

    fn visit_trait_bound(&mut self, i: &TraitBound) {
        // A trait bound with a relative path cannot be interned (in a type, e.g. `dyn MyTrait`,
        // the path is a cause of the type instead).
        if self.type_depth == 0 && !self.leaker.is_context_free_path(&i.path) {
            self.relative_path(quote!(#i), quote!(#i));
            return;
        }
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_trait_bound(v, i)
        })
    }

    fn visit_const_param(&mut self, i: &ConstParam) {
        // The type of a const parameter cannot be a projection (such as
        // `<MyLeakerType as Repeater<0>>::Type`), so it is never interned. It must be a primitive
        // type, or a context-free path to a type alias of it (e.g. `::my_crate::Idx`).
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        match unwrap_type_group(&i.ty) {
            ty @ Type::Path(TypePath { qself: None, path }) if is_const_param_ty(ty) => {
                // A bare primitive name (e.g. `u8`, not `::core::primitive::u8`) is copied as it
                // is, assuming that it is not shadowed.
                if let Some(ident) = path.get_ident() {
                    self.frame.effects.push(Effect::Primitive(ident.clone()));
                }
            }
            Type::Path(TypePath { qself: None, path })
                if self.leaker.is_context_free_path(path) => {}
            _ => {
                // The type of a const parameter cannot be interned.
                self.relative_path(quote!(#{&i.ty}), quote!(#i));
                return;
            }
        }
        if let Some(expr) = &i.default {
            self.visit_expr(expr);
        }
    }

    fn visit_path(&mut self, i: &Path) {
        let starts_with_generic = self.starts_with_generic(i);
        if self.leaker.is_context_free_path(i) || starts_with_generic {
            // A primitive type allowed by `allowed_paths` (e.g. by
            // `LeakerConfig::allow_primitive()`) is copied as it is, assuming that its name is not
            // shadowed. The name alone cannot prove it (e.g. `type u16 = i32;` in the Leaker's
            // module), so it is recorded for `Leaker::assertions()`, which fails to compile there
            // if the name is shadowed.
            if !starts_with_generic && i.leading_colon.is_none() {
                if let Some(seg) = i.segments.first() {
                    if is_primitive_name(&seg.ident) {
                        self.frame
                            .effects
                            .push(Effect::Primitive(seg.ident.clone()));
                    }
                }
            }
        } else if self.type_depth > 0 {
            let is_self = i.leading_colon.is_none()
                && matches!(i.segments.first(), Some(seg) if seg.ident == "Self" && seg.arguments.is_none());
            if !is_self {
                // The path depends on the type context.
                self.frame.must.push((0, quote!(#i)));
            } else if !self.leaker.self_ty_can_be_interned {
                self.frame.impossible.push(quote!(#i));
            }
        }
        // The generic arguments are checked in any case (e.g. `RelType` in
        // `::core::option::Option<RelType>`).
        syn::visit::visit_path(self, i)
    }

    fn visit_qself(&mut self, i: &QSelf) {
        // `<T>::Ty` cannot be interned.
        if self.type_depth > 0 && i.as_token.is_none() {
            let ty = &i.ty;
            self.frame.impossible.push(quote!(#ty));
        }
        syn::visit::visit_qself(self, i)
    }

    fn visit_lifetime(&mut self, i: &Lifetime) {
        if self.type_depth == 0 {
            return;
        }
        if i.ident == "_" {
            self.frame.impossible.push(quote!(#i));
        } else if i.ident != "static" && !self.is_generic_lifetime(i) {
            // The lifetime depends on the type context.
            self.frame.must.push((0, quote!(#i)));
        }
    }

    fn visit_expr(&mut self, i: &Expr) {
        if self.type_depth > 0 {
            if let Expr::Closure(_) | Expr::Assign(_) | Expr::Verbatim(_) | Expr::Macro(_) = i {
                self.frame.impossible.push(quote!(#i));
            }
        }
        syn::visit::visit_expr(self, i)
    }

    fn visit_expr_path(&mut self, i: &ExprPath) {
        if self.type_depth > 0 {
            // In a type (e.g. `[u8; MY_LEN]`), the path is a cause of the type.
            return syn::visit::visit_expr_path(self, i);
        }
        // Values starting with a generic parameter in scope (e.g. `N` or `T::N` where
        // `const N: usize` or `T` is a generic parameter) or `Self` do not depend on the type
        // context.
        let is_self = i.qself.is_none()
            && i.path.leading_colon.is_none()
            && matches!(i.path.segments.first(), Some(seg) if seg.ident == "Self" && seg.arguments.is_none());
        if (i.qself.is_none() && self.starts_with_generic(&i.path))
            || (is_self && self.leaker.self_ty_can_be_interned)
            || self.leaker.is_context_free_path(&i.path)
        {
            syn::visit::visit_expr_path(self, i)
        } else {
            // A value with a relative path, which cannot be interned
            self.relative_path(quote!(#i), quote!(#i));
        }
    }
}

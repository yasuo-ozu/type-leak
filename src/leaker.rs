//! The [`Leaker`], which finds the types depending on the type context.

use crate::check::CheckResult;
use crate::config::LeakerConfig;
use crate::encode::is_const_param_ty;
use crate::error::NotInternableError;
use crate::graph::{descendants, reaches, TypeGraph, TypeKey};
use crate::referrer::Referrer;
use crate::util::{
    generic_param_name, is_primitive_name, looks_like_const, mentions_generics, path_starts_with,
    token_names, unwrap_type_group,
};
use proc_macro2::{Span, TokenStream};
use std::collections::{HashMap, HashSet};
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// The entry point of this crate.
pub struct Leaker {
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
    /// The types found by [`Leaker::intern_with()`] and their composition.
    types: TypeGraph,
    /// The nodes of the types to be interned, computed from `types` by [`Leaker::reduce_roots()`]
    /// (`None` if not computed yet, or outdated).
    reachable_types: Option<Vec<TypeKey>>,
    /// Names of primitive types which were assumed to be the primitive types (not shadowed), as
    /// they are copied as they are (see [`Leaker::assertions()`]).
    assumed_primitives: Vec<Ident>,
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

struct AnalyzeVisitor<'a> {
    leaker: &'a mut Leaker,
    generics: Generics,
    /// The number of leading `generics.params` which belong to the Leaker itself. The rest are
    /// local generics (e.g. of trait methods or GATs), which cannot be referred to from the
    /// Repeater's implementation.
    n_outer_generics: usize,
    /// Where predicates of the local generics.
    local_predicates: Vec<WherePredicate>,
    /// The type whose components are being visited.
    parent: Option<TypeKey>,
    /// The relative paths which cannot be interned.
    errors: Vec<CheckResult>,
    /// The first type used with different local generics in different places.
    conflicting_local_generics: Option<TokenStream>,
    /// The types found by this visitor, merged into the Leaker if successful.
    types: TypeGraph,
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

    fn visit_type(&mut self, i: &Type) {
        let locals = self.generics.params.iter().skip(self.n_outer_generics);
        let mentions_local = mentions_generics(i, locals.clone());
        let result = self.leaker.check(i, &self.generics);
        if !result.impossible.is_empty() {
            if result.is_own {
                // The type must be interned (because of its own path, the first of `must`, which
                // depends on the type context) but cannot be (e.g. `MyType<'_>`), and cannot be
                // decomposed either. The path should be made absolute.
                self.errors.push(result);
            } else if !result.must.is_empty() {
                // The type cannot be interned as a whole (e.g. `&MyType` with an elided lifetime,
                // or `impl Iterator<Item = MyType>`), but its components can.
                syn::visit::visit_type(self, i);
            }
            // Otherwise, the type cannot be interned, but does not need to be (e.g.
            // `impl ::path::Trait`).
            return;
        }
        let is_must_intern = result.is_own;
        if mentions_local {
            if is_must_intern {
                // The type depends on both the type context and local generics, and cannot be
                // decomposed further. It is interned together with the local generics it needs,
                // which the Repeater's implementation must declare (see
                // [`Referrer::local_generics()`]).
                let local_generics = required_local_generics(i, locals, &self.local_predicates);
                if self
                    .types
                    .add(i, self.parent.as_ref(), true, local_generics)
                    .is_err()
                {
                    self.conflict_local_generics(i);
                }
            } else {
                // The type itself cannot be interned because the Repeater's implementation
                // cannot refer to the local generics, but its components can.
                syn::visit::visit_type(self, i);
            }
            return;
        }

        if self.parent.is_none() && is_must_intern && looks_like_const(i) {
            // A bare identifier such as `SIZE` in `T: MyTrait<SIZE>` is parsed as a type, but is
            // probably a const (by the naming convention), which cannot be leaked on its own (like
            // `T: MyTrait<{ SIZE }>`).
            self.relative_path(quote!(#i), quote!(#i));
            return;
        }

        let ix = match self
            .types
            .add(i, self.parent.as_ref(), is_must_intern, Generics::default())
        {
            Ok(ix) => ix,
            Err(()) => {
                self.conflict_local_generics(i);
                return;
            }
        };

        if !is_must_intern {
            // Visit the components
            let parent = self.parent.replace(ix);
            syn::visit::visit_type(self, i);
            self.parent = parent;
        }
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
                    self.leaker.assume_primitive(ident);
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

    fn visit_trait_bound(&mut self, i: &TraitBound) {
        // Trait bounds with a non-absolute path
        if !self.leaker.is_context_free_path(&i.path) {
            // A trait bound cannot be interned.
            self.relative_path(quote!(#i), quote!(#i));
            return;
        }
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_trait_bound(v, i)
        })
    }

    fn visit_type_bare_fn(&mut self, i: &TypeBareFn) {
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_type_bare_fn(v, i)
        })
    }

    fn visit_predicate_type(&mut self, i: &PredicateType) {
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_predicate_type(v, i)
        })
    }

    fn visit_path(&mut self, i: &Path) {
        // A primitive type allowed by `allowed_paths` (e.g. by `LeakerConfig::allow_primitive()`)
        // is copied as it is, assuming that its name is not shadowed. The name alone cannot
        // prove it (e.g. `type u16 = i32;` in the Leaker's module), so it is recorded for
        // `Leaker::assertions()`, which fails to compile there if the name is shadowed.
        if i.leading_colon.is_none() && self.leaker.is_context_free_path(i) {
            if let Some(seg) = i.segments.first() {
                let is_generic = self.generics.params.iter().any(|gp| match gp {
                    GenericParam::Type(TypeParam { ident, .. })
                    | GenericParam::Const(ConstParam { ident, .. }) => ident == &seg.ident,
                    _ => false,
                });
                if !is_generic && is_primitive_name(&seg.ident) {
                    self.leaker.assume_primitive(&seg.ident);
                }
            }
        }
        syn::visit::visit_path(self, i)
    }

    fn visit_trait_item_const(&mut self, i: &TraitItemConst) {
        // Like the bodies of methods, defaults of associated consts are not copied.
        let mut i = i.clone();
        i.default = None;
        syn::visit::visit_trait_item_const(self, &i)
    }

    fn visit_expr_path(&mut self, i: &ExprPath) {
        // Values starting with a generic parameter in scope (e.g. `N` or `T::N` where
        // `const N: usize` or `T` is a generic parameter) or `Self` do not depend on the type
        // context.
        let starts_with_generic = i.qself.is_none()
            && i.path.leading_colon.is_none()
            && i.path.segments.first().map_or(false, |seg| {
                seg.arguments.is_none()
                    && ((seg.ident == "Self" && self.leaker.self_ty_can_be_interned)
                        || self.generics.params.iter().any(|gp| match gp {
                            GenericParam::Type(TypeParam { ident, .. })
                            | GenericParam::Const(ConstParam { ident, .. }) => ident == &seg.ident,
                            _ => false,
                        }))
            });
        if starts_with_generic || self.leaker.is_context_free_path(&i.path) {
            syn::visit::visit_expr_path(self, i)
        } else {
            // A value with a relative path, which cannot be interned
            self.relative_path(quote!(#i), quote!(#i));
        }
    }
}

impl AnalyzeVisitor<'_> {
    /// Records that the relative path `must` cannot be interned (because of `impossible`, e.g. a
    /// trait bound or a value containing it), so it should be absolute.
    fn relative_path(&mut self, must: TokenStream, impossible: TokenStream) {
        self.errors.push(CheckResult {
            must: vec![must],
            impossible: vec![impossible],
            is_own: true,
        });
    }

    /// Records that `ty` is used with different local generics in different places.
    fn conflict_local_generics(&mut self, ty: &Type) {
        self.conflicting_local_generics
            .get_or_insert_with(|| quote!(#ty));
    }

    /// Runs `f` with a visitor where `generics` (of a trait method or a GAT) are added as local
    /// generics, sharing the error and the types found with this visitor.
    fn with_local_generics(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut AnalyzeVisitor<'_>),
    ) {
        let mut visitor = AnalyzeVisitor {
            leaker: self.leaker,
            generics: self.generics.clone(),
            n_outer_generics: self.n_outer_generics,
            local_predicates: self.local_predicates.clone(),
            parent: self.parent.clone(),
            errors: std::mem::take(&mut self.errors),
            conflicting_local_generics: self.conflicting_local_generics.take(),
            types: std::mem::take(&mut self.types),
        };
        visitor
            .generics
            .params
            .extend(generics.params.iter().cloned());
        if let Some(wc) = &generics.where_clause {
            visitor
                .local_predicates
                .extend(wc.predicates.iter().cloned());
        }
        f(&mut visitor);
        self.errors = visitor.errors;
        self.conflicting_local_generics = visitor.conflicting_local_generics;
        self.types = visitor.types;
    }

    /// Runs `f` with the lifetimes of a `for<..>` binder (e.g. `'a` in `for<'a> Fn(&'a T)`) in
    /// scope. They are treated as local generics, which the Repeater's implementation cannot
    /// refer to.
    fn with_bound_lifetimes<R>(
        &mut self,
        lifetimes: Option<&BoundLifetimes>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let len = self.generics.params.len();
        if let Some(lifetimes) = lifetimes {
            self.generics
                .params
                .extend(lifetimes.lifetimes.iter().cloned());
        }
        let result = f(self);
        while self.generics.params.len() > len {
            self.generics.params.pop();
        }
        result
    }
}

impl Default for Leaker {
    fn default() -> Self {
        Self::new()
    }
}

impl Leaker {
    /// Initialize with [`ItemStruct`]
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let test_struct: ItemStruct = parse_quote!(
    ///     pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
    ///         field1: MyType1,
    ///         field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
    ///         field3: &'a (T1, T2),
    ///     }
    /// );
    /// let leaker = Leaker::from_struct(&test_struct).unwrap();
    /// ```
    pub fn from_struct(input: &ItemStruct) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_struct(input);
        })?;
        Ok(leaker)
    }

    /// Initialize with [`ItemEnum`]
    pub fn from_enum(input: &ItemEnum) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_enum(input);
        })?;
        Ok(leaker)
    }

    /// Initialize with [`ItemUnion`]
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let u: ItemUnion = parse_quote! {
    ///     pub union MyUnion<T: ::core::marker::Copy> {
    ///         a: T,
    ///         b: MyType,
    ///     }
    /// };
    /// let referrer = Leaker::from_union(&u).unwrap().finish();
    /// assert_eq!(referrer.iter().collect::<Vec<_>>(), [&parse_quote!(MyType)]);
    /// ```
    pub fn from_union(input: &ItemUnion) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_union(input);
        })?;
        Ok(leaker)
    }

    /// Build a [`Leaker`] from a given trait.
    ///
    /// Unlike an enum or struct, a trait Leaker requires an alternate struct — an absolute path
    /// to a struct declared in the same crate as the leaker trait and visible from
    /// [`Referrer`]s' context. That struct serves as the `impl` target of the Repeater
    /// instead of the Leaker's path.
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let s: ItemTrait = parse_quote!{
    ///     pub trait MyTrait<T, U> {
    ///         fn func(self, t: T) -> U;
    ///     }
    /// };
    /// let alternate: ItemStruct = parse_quote!{
    ///     pub struct MyAlternate;
    /// };
    /// let _ = Leaker::from_trait(&s);
    /// ```
    pub fn from_trait(input: &ItemTrait) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_trait(input);
        })?;
        Ok(leaker)
    }

    /// Initialize empty [`Leaker`] with no generics.
    pub fn new() -> Self {
        Self {
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
            types: TypeGraph::default(),
            reachable_types: None,
            assumed_primitives: Vec::new(),
        }
    }

    /// Initialize [`Leaker`] from configuration.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allowed_paths: config.allowed_paths,
            self_ty_can_be_interned: config.self_ty_can_be_interned,
            types: TypeGraph::default(),
            reachable_types: None,
            assumed_primitives: Vec::new(),
        }
    }

    fn assume_primitive(&mut self, ident: &Ident) {
        if !self.assumed_primitives.iter().any(|p| p == ident) {
            self.assumed_primitives.push(ident.clone());
        }
    }

    /// Items to emit in the Leaker's type context, which check the assumptions made by the
    /// Leaker at compile time.
    ///
    /// The names of primitive types (e.g. `u8` in `const N: u8`, or in a type when allowed by
    /// [`LeakerConfig::allow_primitive()`]) are copied as they are, assuming that they are not
    /// shadowed in the Leaker's type context (e.g. by `type u8 = i32;`), which cannot be known
    /// without name resolution. These items fail to compile (with "mismatched types") if a name
    /// is shadowed, instead of the copies silently referring to another type.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait MyTrait<const N: u8> {});
    /// let leaker = Leaker::from_trait(&item).unwrap();
    /// assert!(leaker
    ///     .assertions()
    ///     .to_string()
    ///     .contains("__type_leak_assert_u8_is_primitive"));
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

    /// Returns `true` if `path` does not depend on the type context, i.e. it is absolute, starts
    /// with `$crate`, or starts with one of [`Leaker::allowed_paths`].
    pub(crate) fn is_context_free_path(&self, path: &Path) -> bool {
        path.leading_colon.is_some()
            || matches!(
                path.segments.iter().next(),
                Some(PathSegment {
                    ident,
                    arguments: PathArguments::None,
                }) if ident == "$crate"
            )
            || self
                .allowed_paths
                .iter()
                .any(|allowed| path_starts_with(path, allowed))
    }

    /// Intern AST elements using a visitor callback.
    ///
    /// The types found are added to those of the previous calls; nothing is added if an error is
    /// returned.
    pub fn intern_with<'ast, F>(
        &mut self,
        generics: &Generics,
        f: F,
    ) -> std::result::Result<&mut Self, NotInternableError>
    where
        F: FnOnce(&mut dyn syn::visit::Visit<'ast>),
    {
        let mut visitor = AnalyzeVisitor {
            leaker: self,
            parent: None,
            errors: Vec::new(),
            conflicting_local_generics: None,
            generics: generics.clone(),
            n_outer_generics: generics.params.len(),
            local_predicates: Vec::new(),
            types: TypeGraph::default(),
        };
        f(&mut visitor);
        if !visitor.errors.is_empty() {
            return Err(NotInternableError::RelativePath(visitor.errors));
        }
        if let Some(ty) = visitor.conflicting_local_generics {
            return Err(NotInternableError::ConflictingLocalGenerics(ty));
        }
        let types = visitor.types;
        // The Referrer finds leaked types by the type alone, so the same type must depend on the
        // same local generics wherever it appears (e.g. not `Wrap<B>` both in
        // `fn f<const B: bool>()` and in `fn g()` where `B` is a type).
        if let Some(key) = self.types.find_conflict(&types) {
            let ty = &key.ty;
            return Err(NotInternableError::ConflictingLocalGenerics(quote!(#ty)));
        }
        if !types.is_empty() {
            self.types.merge(types);
            self.reachable_types = None;
        }
        Ok(self)
    }

    /// Checks whether the given `ty` must be interned, and whether it can be. See [`CheckResult`].
    pub fn check(&self, ty: &Type, generics: &Generics) -> CheckResult {
        crate::check::check(self, ty, generics)
    }

    /// Finish building the [`Leaker`] and convert it into [`Referrer`].
    ///
    /// The types are reduced by [`Leaker::reduce_roots()`] (called here if needed), and sorted by
    /// their tokens, so the result (including the index of each type) is deterministic.
    pub fn finish(mut self) -> Referrer {
        self.reduce_roots();
        let keys = self.reachable_types.take().unwrap_or_default();
        let leak_types: Vec<Type> = keys.iter().map(|key| key.ty.clone()).collect();
        let map = leak_types
            .iter()
            .enumerate()
            .map(|(n, ty)| (TypeKey::new(ty), n))
            .collect();
        let local_generics = keys
            .iter()
            .map(|key| self.types.local_generics[key].clone())
            .collect();
        Referrer {
            leak_types,
            map,
            local_generics,
        }
    }

    /// Reduce nodes to decrease the cost of the [`Leaker`]'s implementation.
    ///
    /// # Algorithm
    ///
    /// Consider the following [`Leaker`] input:
    ///
    /// ```ignore
    /// pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
    ///     field1: MyType1,
    ///     field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
    ///     field3: &'a (T1, T2),
    /// }
    /// ```
    ///
    /// When initialized with [`Leaker::from_struct()`], the [`Leaker`] analyzes the AST and
    /// constructs a DAG representing all internable types and their dependency relations:
    ///
    /// ![before_graph](https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/resources/before_graph.svg)
    ///
    /// **Red** nodes are those whose own path depends on the type context
    /// ([`CheckResult::is_own`] by [`Leaker::check()`]), meaning the type literally depends on the
    /// type context and must be interned.
    ///
    /// This algorithm reduces the node set while ensuring that every root type (annotated with ★)
    /// can still be expressed using the remaining **red** nodes.
    ///
    /// There are two degrees of freedom in this reduction:
    ///
    /// - Intern all **red** nodes and ignore the rest (since non-red nodes either do not need to
    ///   be interned or can be reconstructed from red nodes).
    /// - Skip interning individual **red** nodes and instead intern their common ancestors when
    ///   doing so is more efficient.
    ///
    /// The final result looks like:
    ///
    /// ![after_graph](https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/resources/after_graph.svg)
    pub fn reduce_roots(&mut self) {
        if self.reachable_types.is_none() {
            self.reachable_types = Some(self.build_graph_and_reduce());
        }
    }

    /// Returns the types to be interned, sorted by their tokens.
    fn build_graph_and_reduce(&self) -> Vec<TypeKey> {
        let types = &self.types;
        let children: HashMap<&TypeKey, Vec<TypeKey>> =
            types.keys().map(|key| (key, types.children(key))).collect();

        // 1. Keep the roots which must be interned, or contain types which must be interned.
        let mut kept: HashSet<&TypeKey> = types
            .roots
            .iter()
            .filter(|key| reaches(key, &children, |key| types.must_intern.contains(key)))
            .collect();
        // 2. Keep the types which must be interned and are not components of kept types.
        let components: HashSet<TypeKey> = kept
            .iter()
            .flat_map(|key| descendants(key, &children))
            .collect();
        kept.extend(
            types
                .must_intern
                .iter()
                .filter(|key| !components.contains(*key)),
        );

        // Sort by the tokens, for deterministic output.
        let mut kept: Vec<TypeKey> = kept.into_iter().cloned().collect();
        kept.sort_by_cached_key(|key| key.to_string());
        kept
    }
}

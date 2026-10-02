//! The analysis of items and types by the [`Leaker`].
//!
//! [`AnalyzeVisitor`] visits each type once, in post-order: the causes found in a type (see
//! [`CheckResult`]) decide whether it is interned as a whole, decomposed into its components, or
//! reported as an error, and the effects of its components (the types to intern, the errors and
//! the assumed primitives) are kept or discarded accordingly.

use crate::encode::is_const_param_ty;
use crate::graph::{NodeKey, TypeGraph};
use crate::leaker::Leaker;
use crate::requirements::{Requirements, Scope};
use crate::util::{is_primitive_name, looks_like_const, unwrap_type_group};
use proc_macro2::TokenStream;
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

/// A type to be added to the [`TypeGraph`], with its components.
struct PendingType {
    ty: Type,
    is_must_intern: bool,
    requirements: Requirements,
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
    /// The names of the primitive types copied as they are (see [`Leaker::assertions()`]).
    pub(crate) primitives: Vec<Ident>,
}

/// Analyzes items (see [`Leaker::intern_with()`]) and types (see [`Leaker::check()`]).
pub(crate) struct AnalyzeVisitor<'a> {
    leaker: &'a Leaker,
    /// The context of the types found (see [`Referrer::context()`](crate::Referrer::context())).
    context: usize,
    /// The generics of the Leaker, which the Repeater's implementation can refer to.
    root: Generics,
    /// The generics declared inside the Leaker's item (e.g. by trait methods, GATs, nested items
    /// or `for<..>` binders), the innermost last. The Repeater's implementation cannot refer to
    /// them, so a type depending on them is interned with its [`Requirements`].
    scopes: Vec<Scope>,
    /// What is found in the type being visited (or outside any type).
    frame: Frame,
    /// The number of types enclosing the syntax being visited.
    type_depth: usize,
    /// The number of items enclosing the syntax being visited. The generics of the outermost item
    /// are the `root`.
    item_depth: usize,
    /// The result of the last type visited.
    last_result: Option<CheckResult>,
}

impl<'a> AnalyzeVisitor<'a> {
    pub(crate) fn new(leaker: &'a Leaker, context: usize, generics: &Generics) -> Self {
        Self {
            leaker,
            context,
            root: generics.clone(),
            scopes: Vec::new(),
            frame: Frame::default(),
            type_depth: 0,
            item_depth: 0,
            last_result: None,
        }
    }

    /// Analyzes `ty` alone (see [`Leaker::check()`]).
    pub(crate) fn check(leaker: &Leaker, ty: &Type, generics: &Generics) -> CheckResult {
        let mut visitor = AnalyzeVisitor::new(leaker, 0, generics);
        visitor.visit_type(ty);
        visitor.last_result.unwrap()
    }

    /// Finishes the analysis, adding the types found to a graph.
    pub(crate) fn finish(self) -> Analysis {
        let mut analysis = Analysis {
            types: TypeGraph::default(),
            errors: Vec::new(),
            primitives: Vec::new(),
        };
        analysis.apply(self.context, self.frame.effects, None);
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

        let requirements = Requirements::of(ty, &self.scopes);
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
        } else if !requirements.is_empty() {
            if result.is_own {
                // The type depends on both the type context and local generics, and cannot be
                // decomposed further. It is interned together with the local generics it needs,
                // which the Repeater's implementation must declare (see
                // [`Referrer::local_generics()`](crate::Referrer::local_generics())).
                self.frame.effects.push(Effect::Type(Box::new(PendingType {
                    ty: ty.clone(),
                    is_must_intern: true,
                    requirements,
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
                requirements: Requirements::none(),
                looks_like_const: looks_like_const(ty),
                components: Vec::new(),
            })));
        } else {
            // The type may be interned as a whole, or its components.
            self.frame.effects.push(Effect::Type(Box::new(PendingType {
                ty: ty.clone(),
                is_must_intern: false,
                requirements: Requirements::none(),
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

    /// Returns the generic parameters in scope, those of the Leaker and the local ones.
    fn params(&self) -> impl Iterator<Item = &GenericParam> {
        self.root
            .params
            .iter()
            .chain(self.scopes.iter().flat_map(|scope| scope.params()))
    }

    fn is_generic_lifetime(&self, lifetime: &Lifetime) -> bool {
        self.params().any(
            |param| matches!(param, GenericParam::Lifetime(param) if &param.lifetime == lifetime),
        )
    }

    /// Returns `true` if `path` starts with a generic parameter (e.g. `T` or `T::Item`).
    fn starts_with_generic(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && path.segments.first().map_or(false, |seg| {
                seg.arguments.is_none()
                    && self.params().any(|param| match param {
                        GenericParam::Type(TypeParam { ident, .. })
                        | GenericParam::Const(ConstParam { ident, .. }) => ident == &seg.ident,
                        GenericParam::Lifetime(_) => false,
                    })
            })
    }

    /// Runs `f` with `scope` pushed.
    fn with_scope(&mut self, scope: Scope, f: impl FnOnce(&mut Self)) {
        self.scopes.push(scope);
        f(self);
        self.scopes.pop();
    }

    /// Runs `f` with the generics of a trait or impl item (a method, a type or a const) in scope,
    /// as local generics.
    fn with_local_generics(&mut self, generics: &Generics, f: impl FnOnce(&mut Self)) {
        self.with_scope(Scope::new(generics), f)
    }

    /// Runs `f` with the generics of an item. They are those of the Leaker for the outermost
    /// item, and local generics for a nested one.
    fn with_item_generics(&mut self, generics: &Generics, f: impl FnOnce(&mut Self)) {
        let nested = self.item_depth > 0;
        self.item_depth += 1;
        if nested {
            self.with_local_generics(generics, f);
        } else {
            f(self);
        }
        self.item_depth -= 1;
    }

    /// Runs `f` with the lifetimes of a `for<..>` binder (e.g. `'a` in `for<'a> Fn(&'a T)`) in
    /// scope. They are treated as local generics, which the Repeater's implementation cannot
    /// refer to.
    fn with_bound_lifetimes(
        &mut self,
        lifetimes: Option<&BoundLifetimes>,
        f: impl FnOnce(&mut Self),
    ) {
        match lifetimes {
            Some(lifetimes) => self.with_scope(Scope::binder(lifetimes), f),
            None => f(self),
        }
    }
}

impl Analysis {
    /// Applies `effects` in order, adding the types in `context` as components of `parent` (if
    /// any).
    fn apply(&mut self, context: usize, effects: Vec<Effect>, parent: Option<&NodeKey>) {
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
                    let key = self.types.add(
                        context,
                        ty,
                        pending.requirements,
                        parent,
                        pending.is_must_intern,
                    );
                    self.apply(context, pending.components, Some(&key));
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
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_const(visitor, &i)
        });
    }

    fn visit_impl_item_fn(&mut self, i: &ImplItemFn) {
        self.with_local_generics(&i.sig.generics, |visitor| {
            syn::visit::visit_impl_item_fn(visitor, i)
        });
    }

    fn visit_impl_item_type(&mut self, i: &ImplItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_impl_item_type(visitor, i)
        });
    }

    fn visit_impl_item_const(&mut self, i: &ImplItemConst) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_impl_item_const(visitor, i)
        });
    }

    fn visit_item_struct(&mut self, i: &ItemStruct) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_struct(visitor, i)
        });
    }

    fn visit_item_enum(&mut self, i: &ItemEnum) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_enum(visitor, i)
        });
    }

    fn visit_item_union(&mut self, i: &ItemUnion) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_union(visitor, i)
        });
    }

    fn visit_item_trait(&mut self, i: &ItemTrait) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_trait(visitor, i)
        });
    }

    fn visit_item_type(&mut self, i: &ItemType) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_type(visitor, i)
        });
    }

    fn visit_item_fn(&mut self, i: &ItemFn) {
        self.with_item_generics(&i.sig.generics, |visitor| {
            syn::visit::visit_item_fn(visitor, i)
        });
    }

    fn visit_item_impl(&mut self, i: &ItemImpl) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_impl(visitor, i)
        });
    }

    fn visit_item_trait_alias(&mut self, i: &ItemTraitAlias) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_trait_alias(visitor, i)
        });
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

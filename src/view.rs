//! The [`View`], which a scope of the [`Leaker`](crate::Leaker) gives to its closure.

use crate::allow::Allow;
use crate::analysis::CheckResult;
use crate::analyze::{AnalyzeVisitor, ScopeSpec, Scopes};
use crate::context::ContextId;
use std::fmt::Display;
use syn::*;

/// The handle that a scope of the [`Leaker`](crate::Leaker) gives to its closure. Pass it the
/// syntax to analyze.
///
/// The `visit_*` methods analyze one kind of syntax each, and [`View::check()`] looks at a type
/// without recording anything. The `enter_*` methods open a nested scope: its generics are local,
/// and the generics around stay in scope. `Self` is what the scope declares: `enter_generics()`
/// keeps the one around, so naming it is an error only where none is (at the top, or inside
/// `enter_mod()`). A nested scope returns the id and the result, but no errors: the outermost call
/// records them ([`Leaker::result()`](crate::Leaker::result)).
///
/// `View` also implements [`Visit`](syn::visit::Visit): `syn::visit::visit_item_struct(view,
/// &item)` does what [`Leaker::intern_struct()`](crate::Leaker::intern_struct) does. The walk
/// declares nothing, so walk the item the scope is for. For an impl, `enter_impl()` with
/// `syn::visit::visit_item_impl(view, &item)` does what
/// [`Leaker::intern_impl()`](crate::Leaker::intern_impl) does, except that it does not cut the
/// entries in the self type if a bound of the impl names `Self` (`intern_impl()` rejects or copies
/// them as written): use `intern_impl()` for such an impl. `visit_signature` analyzes the
/// signature in a scope of its own for its generics, and reports a method lifetime that the copy
/// would leave unconstrained (E0195, [Method lifetimes](crate::Leaker#method-lifetimes), not in an
/// inherent impl) and an elided output lifetime that it could not resolve (E0106, [Elided
/// lifetimes](crate::Leaker#elided-lifetimes)).
/// `visit_path` at the root analyzes only the arguments of the path: to check the root too, use
/// [`View::visit_type_param_bound()`].
///
/// `A` is the [`Allow`] of the Leaker: a helper takes a `View` as `fn f<A: Allow>(view: &mut
/// View<'_, '_, A>)`.
///
/// A function that is itself the analyzed item has no local generics, so do not visit its
/// signature. Enter a scope that declares its generics, and visit its types and where clause:
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let sig: Signature = parse_quote!(fn f<T>(x: Rel<T>) where T: ::p::Bound<Rel2>);
/// let mut leaker = Leaker::new();
/// leaker.enter_generics(&sig.generics, |view| {
///     view.visit_type(&parse_quote!(Rel<T>)); // the type of `x`
///     view.visit_where_clause(sig.generics.where_clause.as_ref().unwrap());
/// });
/// assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
/// ```
pub struct View<'v, 'a, A = ()> {
    visitor: &'v mut AnalyzeVisitor<'a, A>,
}

impl<'v, 'a, A: Allow> View<'v, 'a, A> {
    pub(crate) fn new(visitor: &'v mut AnalyzeVisitor<'a, A>) -> Self {
        Self { visitor }
    }

    /// Checks whether `ty` must become an entry, and whether it can ([`CheckResult`]), without
    /// recording anything.
    ///
    /// It finds what [`View::visit_type()`] finds in `ty` here, and leaves no entry, context,
    /// error or module in the Leaker.
    pub fn check(&mut self, ty: &Type) -> CheckResult {
        self.visitor.check(ty)
    }

    /// Opens a module scope inside this one: [`Leaker::enter_mod()`](crate::Leaker::enter_mod).
    pub fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_mod(self, hint, declared, f)
    }

    /// Opens a scope that declares more generics:
    /// [`Leaker::enter_generics()`](crate::Leaker::enter_generics).
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_generics(self, generics, f)
    }

    /// Opens a scope for the parts of a struct, an enum or a union:
    /// [`Leaker::enter_type()`](crate::Leaker::enter_type).
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_type(self, ident, generics, f)
    }

    /// Opens a scope for the parts of an impl: [`Leaker::enter_impl()`](crate::Leaker::enter_impl).
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        trait_: Option<&Path>,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_impl(self, generics, self_ty, trait_, f)
    }

    /// Opens a scope for the parts of a trait:
    /// [`Leaker::enter_trait()`](crate::Leaker::enter_trait).
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_trait(self, ident, generics, f)
    }
}

impl<A: Allow> Scopes for View<'_, '_, A> {
    type Rule = A;

    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        self.visitor.enter(spec, f)
    }
}

// The `visit_*` methods of the [`View`]: its own, and the [`Visit`] implementation, which forwards
// the methods the analysis overrides.

/// Forwards the `visit_*` methods `$method` of the nodes `$ty` to the analysis.
macro_rules! forward {
    ($($method:ident: $ty:ty,)*) => {
        $(
            fn $method(&mut self, i: &'ast $ty) {
                syn::visit::Visit::$method(&mut *self.visitor, i)
            }
        )*
    };
}

impl<A: Allow> View<'_, '_, A> {
    /// Walks the parts of an impl of the scope: the walk of `syn::visit::visit_item_impl()`, with
    /// the self type checked.
    pub(crate) fn walk_item_impl(&mut self, item: &ItemImpl) {
        self.visitor.walk_item_impl(item)
    }

    /// Analyzes a nested item (a struct, trait, impl, module, function, ..) like an item of a
    /// module ([`Leaker::intern_mod()`](crate::Leaker::intern_mod)).
    ///
    /// Its generics are local and the generics around stay in scope; to make the generics of an
    /// item the root, use an `intern_*` method. A struct, enum, union, trait or impl declares its
    /// own `Self` ([`Visitor::with_nested_self()`](crate::Visitor::with_nested_self) finds its
    /// entries); the other items keep the one around. The copy must nest the same scopes as the Leaker did.
    ///
    /// In a nested struct, enum or union, `Self` is the copy's own type. A type that names it is
    /// copied as written if its path is absolute (`::path::Wrap<Self>`), and leaked if it is
    /// relative (`Wrap<Self>`). A type that uses a generic bounded by `Self`
    /// (`struct S<T: AsRef<Self>>(Rel<T>);`) is for any type that the bound holds for
    /// ([`SelfRequirement::Any`](crate::SelfRequirement)), and is an error if it names `Self` too
    /// (`Rel2<Self, T>`): drop the `Self`, write the type with an absolute path, or use an
    /// `intern_*` method.
    ///
    /// A bound is rewritten for the `Self` of the nested item that uses its generic. That is an
    /// error if `Self` is the implementor of a trait, or if the bound names an associated item of
    /// `Self` (`Self::X`): see [`CheckResult::headline`](crate::CheckResult::headline).
    pub fn visit_item(&mut self, item: &Item) {
        syn::visit::Visit::visit_item(&mut *self.visitor, item)
    }

    /// Analyzes a type, with the generics, `Self` and module of the scope.
    pub fn visit_type(&mut self, ty: &Type) {
        syn::visit::Visit::visit_type(&mut *self.visitor, ty)
    }

    /// Analyzes the arguments of a path whose name is resolved elsewhere (`<Rel>` of `Wrap<Rel>`),
    /// not the path itself.
    pub fn visit_path_arguments(&mut self, arguments: &PathArguments) {
        syn::visit::Visit::visit_path_arguments(&mut *self.visitor, arguments)
    }

    /// Analyzes a where clause, such as the one of the generics of the scope.
    pub fn visit_where_clause(&mut self, where_clause: &WhereClause) {
        syn::visit::Visit::visit_where_clause(&mut *self.visitor, where_clause)
    }

    /// Analyzes a bound, such as a bound of a generic parameter of the scope.
    ///
    /// The path of a trait bound must not depend on the context, nor start with `Self` where it has
    /// no meaning: that is an error.
    pub fn visit_type_param_bound(&mut self, bound: &TypeParamBound) {
        syn::visit::Visit::visit_type_param_bound(&mut *self.visitor, bound)
    }
}

// Forwards only the methods the analysis overrides, so walking the view walks the analysis. The
// default walk of the other methods reaches the forwarded ones through the view. The test below
// keeps this list in sync with the overrides of the analysis.
impl<'ast, A: Allow> syn::visit::Visit<'ast> for View<'_, '_, A> {
    forward! {
        visit_angle_bracketed_generic_arguments: AngleBracketedGenericArguments,
        visit_const_param: ConstParam,
        visit_expr: Expr,
        visit_expr_path: ExprPath,
        visit_fn_arg: FnArg,
        visit_foreign_item_fn: ForeignItemFn,
        visit_impl_item_const: ImplItemConst,
        visit_impl_item_fn: ImplItemFn,
        visit_impl_item_type: ImplItemType,
        visit_item_const: ItemConst,
        visit_item_enum: ItemEnum,
        visit_item_fn: ItemFn,
        visit_item_impl: ItemImpl,
        visit_item_mod: ItemMod,
        visit_item_static: ItemStatic,
        visit_item_struct: ItemStruct,
        visit_item_trait: ItemTrait,
        visit_item_trait_alias: ItemTraitAlias,
        visit_item_type: ItemType,
        visit_item_union: ItemUnion,
        visit_lifetime: Lifetime,
        visit_macro: Macro,
        visit_parenthesized_generic_arguments: ParenthesizedGenericArguments,
        visit_path: Path,
        visit_predicate_type: PredicateType,
        visit_qself: QSelf,
        visit_receiver: Receiver,
        visit_signature: Signature,
        visit_trait_bound: TraitBound,
        visit_trait_item_const: TraitItemConst,
        visit_trait_item_fn: TraitItemFn,
        visit_trait_item_type: TraitItemType,
        visit_type: Type,
        visit_type_bare_fn: TypeBareFn,
        visit_type_param: TypeParam,
        visit_type_param_bound: TypeParamBound,
        visit_type_path: TypePath,
        visit_type_trait_object: TypeTraitObject,
        visit_variant: Variant,
    }
}

#[cfg(test)]
mod tests {
    use proc_macro2::{TokenStream, TokenTree};
    use std::collections::BTreeSet;
    use syn::*;

    /// The `Visit` methods of the type `ty` in `source`: written out, or named in the `forward!`
    /// list.
    fn visit_methods(source: &str, ty: &str) -> BTreeSet<String> {
        fn idents(tokens: TokenStream, out: &mut BTreeSet<String>) {
            for token in tokens {
                match token {
                    TokenTree::Ident(ident) if ident.to_string().starts_with("visit_") => {
                        out.insert(ident.to_string());
                    }
                    TokenTree::Group(group) => idents(group.stream(), out),
                    _ => (),
                }
            }
        }
        let file = parse_file(source).unwrap();
        let mut out = BTreeSet::new();
        for item in file.items {
            let item = match item {
                Item::Impl(item) => item,
                _ => continue,
            };
            let is_visit = matches!(&item.trait_, Some((_, path, _))
                if path.segments.last().map_or(false, |seg| seg.ident == "Visit"));
            let is_ty = matches!(&*item.self_ty, Type::Path(path)
                if path.path.segments.last().map_or(false, |seg| seg.ident == ty));
            if !is_visit || !is_ty {
                continue;
            }
            for item in item.items {
                match item {
                    ImplItem::Fn(method) => {
                        out.insert(method.sig.ident.to_string());
                    }
                    ImplItem::Macro(mac) => idents(mac.mac.tokens, &mut out),
                    _ => (),
                }
            }
        }
        out
    }

    /// The view must forward exactly the methods the analysis overrides: a missing one would bypass
    /// the analysis silently, and an extra one is a stale entry.
    #[test]
    fn the_view_forwards_exactly_the_methods_the_analysis_overrides() {
        let overridden = visit_methods(include_str!("analyze.rs"), "AnalyzeVisitor");
        let forwarded = visit_methods(include_str!("view.rs"), "View");
        assert!(!overridden.is_empty(), "the analysis was not found");
        assert!(
            !forwarded.is_empty(),
            "the forwarded methods were not found"
        );
        let missing: Vec<_> = overridden.difference(&forwarded).collect();
        let stale: Vec<_> = forwarded.difference(&overridden).collect();
        assert!(
            missing.is_empty(),
            "overridden by the analysis but not forwarded by the view: {:?}",
            missing
        );
        assert!(
            stale.is_empty(),
            "forwarded by the view but not overridden by the analysis: {:?}",
            stale
        );
    }
}

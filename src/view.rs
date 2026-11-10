//! The [`View`], which a scope of the [`Leaker`](crate::Leaker) gives to its closure.

use crate::analysis::CheckResult;
use crate::analyze::{AnalyzeVisitor, ScopeSpec, Scopes};
use crate::context::ContextId;
use crate::result::Allow;

use syn::*;

/// The handle that a scope of the [`Leaker`](crate::Leaker) gives to its closure: give it the
/// syntax to analyze.
///
/// The `visit_*` methods analyze one kind of syntax each, and [`View::check()`] tests a type
/// without recording anything. The `enter_*` methods open a nested scope: its generics are local,
/// and the generics around stay in scope. `Self` is what the scope declares (`enter_generics()`
/// keeps the `Self` around). A nested scope returns its context and the result of the closure,
/// but no errors: the outermost call records them.
///
/// `A` is the [`Allow`] rule of the Leaker. A helper takes a `View` as
/// `fn f<A: Allow>(view: &mut View<'_, '_, A>)`.
///
/// `View` implements [`Visit`](syn::visit::Visit) too. In the scope of
/// [`Leaker::enter_type()`](crate::Leaker::enter_type), `syn::visit::visit_item_struct(view,
/// &item)` does what [`Leaker::intern_struct()`](crate::Leaker::intern_struct) does. The walk
/// declares nothing, so walk only the item that the scope is for. For an impl, use
/// [`Leaker::intern_impl()`](crate::Leaker::intern_impl): `visit_item_impl` in `enter_impl()` is
/// the same, but does not report the entries in the self type when a bound of the impl names
/// `Self`. Also:
///
/// - `visit_signature` analyzes the signature in its own scope for its generics. It reports a
///   method lifetime that the copy leaves unconstrained (E0195, not in an inherent impl; see
///   [Method lifetimes](crate::Leaker#method-lifetimes)), and an elided output lifetime that the
///   copy cannot resolve (E0106; see [Elided lifetimes](crate::Leaker#elided-lifetimes)).
/// - `visit_path` outside a type analyzes only the arguments of the path. To check the path too,
///   use [`View::visit_type_param_bound()`].
///
/// For a function that is the analyzed item, do not visit its signature: its generics would be
/// local to it. Enter a scope that declares its generics, and visit its types and where clause:
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

    /// Tells whether `ty` must be leaked, and whether it can be ([`CheckResult`]).
    ///
    /// It finds what [`View::visit_type()`] would find in `ty` here, but leaves no entry, context
    /// or error in the Leaker.
    pub fn check(&mut self, ty: &Type) -> CheckResult {
        self.visitor.check(ty)
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

    /// Analyzes a nested item (a struct, a trait, an impl, a function, ..).
    ///
    /// Its generics are local, and the generics around stay in scope. To make the generics of an
    /// item the root, use an `intern_*` method instead. A struct, enum, union, trait or impl
    /// declares its own `Self`, and [`Referrer::visitor()`](crate::Referrer::visitor) finds its
    /// entries; other items keep the `Self` around. The copy must nest the same scopes.
    ///
    /// In a nested struct, enum or union, `Self` is the copy's own type:
    ///
    /// - A type that names `Self` is leaked if its path is relative (`Wrap<Self>`), and kept as
    ///   written if it is absolute (`::path::Wrap<Self>`).
    /// - A type that uses a generic whose bound names `Self` (`Rel<T>` in
    ///   `struct S<T: AsRef<Self>>(Rel<T>);`) is an entry for any type that the bound holds for
    ///   ([`SelfRequirement::Any`](crate::SelfRequirement::Any)). If it names `Self` too, see
    ///   [`Reason::NestedSelfInBounds`](crate::Reason::NestedSelfInBounds).
    ///
    /// If the nested item uses a generic of the scope whose bound names the outer `Self`, the bound
    /// is written with the outer self type. This is an error
    /// ([`Reason::UnnameableSelf`](crate::Reason::UnnameableSelf)) if the outer `Self` is the
    /// implementor of a trait, or if the bound names `Self::X`.
    ///
    /// A `mod` is an error ([`Reason::Unreadable`](crate::Reason::Unreadable)), and nothing in it
    /// is visited: analyze each item of the module with its own call.
    pub fn visit_item(&mut self, item: &Item) {
        syn::visit::Visit::visit_item(&mut *self.visitor, item)
    }

    /// Analyzes a type, with the generics and `Self` of the scope.
    pub fn visit_type(&mut self, ty: &Type) {
        syn::visit::Visit::visit_type(&mut *self.visitor, ty)
    }

    /// Analyzes the arguments of a path (`<Rel>` in `Wrap<Rel>`), not the path itself. Use it for a
    /// path that your macro resolves itself.
    pub fn visit_path_arguments(&mut self, arguments: &PathArguments) {
        syn::visit::Visit::visit_path_arguments(&mut *self.visitor, arguments)
    }

    /// Analyzes a where clause, such as the one of the generics of the scope.
    pub fn visit_where_clause(&mut self, where_clause: &WhereClause) {
        syn::visit::Visit::visit_where_clause(&mut *self.visitor, where_clause)
    }

    /// Analyzes a bound, such as a bound of a generic parameter of the scope.
    ///
    /// A relative trait path is an error, and so is a path that starts with `Self` where `Self`
    /// has no meaning.
    pub fn visit_type_param_bound(&mut self, bound: &TypeParamBound) {
        syn::visit::Visit::visit_type_param_bound(&mut *self.visitor, bound)
    }
}

// Forwards only the methods the analysis overrides, so walking the view walks the analysis. The
// default walk of the other methods reaches the forwarded ones through the view. A test in
// tests/leaker/scopes.rs keeps this list in sync with the overrides of the analysis.
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

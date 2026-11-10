//! The [`View`], which a scope of the [`Leaker`](crate::Leaker) gives to its closure.

use crate::allow::Allow;
use crate::analysis::CheckResult;
use crate::analyze::{AnalyzeVisitor, ScopeSpec, Scopes};
use crate::context::ContextId;
use std::fmt::Display;
use syn::*;

mod forward;

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
/// View<'_, A>)`.
///
/// A function that is itself the analyzed item has no local generics, so do not visit its
/// signature. Enter a scope that declares its generics, and visit its types and where clause:
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let sig: Signature = parse_quote!(fn f<T>(x: Rel<T>) -> Rel2<T> where T: ::p::Bound<Rel3>);
/// let mut leaker = Leaker::new();
/// leaker
///     .enter_generics(&sig.generics, |view| {
///         for input in &sig.inputs {
///             if let FnArg::Typed(arg) = input {
///                 view.visit_type(&arg.ty);
///             }
///         }
///         if let ReturnType::Type(_, ty) = &sig.output {
///             view.visit_type(ty);
///         }
///         if let Some(where_clause) = &sig.generics.where_clause {
///             view.visit_where_clause(where_clause);
///         }
///     });
/// assert_eq!(leaker.result().finish().unwrap().iter().count(), 3);
/// ```
pub struct View<'v, A = ()> {
    visitor: &'v mut AnalyzeVisitor<A>,
}

impl<'v, A: Allow> View<'v, A> {
    pub(crate) fn new(visitor: &'v mut AnalyzeVisitor<A>) -> Self {
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
        f: impl FnOnce(&mut View<'_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_mod(self, hint, declared, f)
    }

    /// Opens a scope that declares more generics:
    /// [`Leaker::enter_generics()`](crate::Leaker::enter_generics).
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_generics(self, generics, f)
    }

    /// Opens a scope for the parts of a struct, an enum or a union:
    /// [`Leaker::enter_type()`](crate::Leaker::enter_type).
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_type(self, ident, generics, allow_self, f)
    }

    /// Opens a scope for the parts of an impl: [`Leaker::enter_impl()`](crate::Leaker::enter_impl).
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        trait_: Option<&Path>,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_impl(self, generics, self_ty, trait_, allow_self, f)
    }

    /// Opens a scope for the parts of a trait:
    /// [`Leaker::enter_trait()`](crate::Leaker::enter_trait).
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_trait(self, ident, generics, allow_self, f)
    }
}

impl<A: Allow> Scopes for View<'_, A> {
    type Rule = A;

    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_, A>) -> R,
    ) -> (ContextId, R) {
        self.visitor.enter(spec, f)
    }
}

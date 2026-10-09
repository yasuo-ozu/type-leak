//! The [`View`], which a scope of the [`Leaker`](crate::Leaker) gives to its closure.

use crate::analysis::CheckResult;
use crate::analyze::{AnalyzeVisitor, Res, ScopeSpec, Scopes};
use crate::context::ContextId;
use std::fmt::Display;
use syn::*;

mod forward;

/// What a scope gives to its closure: the syntax in the scope is analyzed through it.
///
/// It has `visit_*` methods, `check()` to inspect a type without analyzing it, and the `enter_*`
/// methods for nested scopes ([Scopes](crate#scopes)).
/// A scope sees what its nested scopes find as if it was written in it.
///
/// `View` implements [`Visit`](syn::visit::Visit), so `syn::visit::visit_item_struct(view, &item)`
/// analyzes a struct as [`Leaker::intern_struct()`](crate::Leaker::intern_struct) does. The walk
/// declares nothing, so walk the item the scope is for. Two methods of the trait mean more than the
/// walk, which the `syn::visit` functions call:
///
/// - `Visit::visit_signature(view, &sig)` is a method's signature, with local generics. It
///   declares the generics of `sig` in a scope of their own, and reports an error for a method
///   lifetime that the copy would leave unconstrained (E0195, see [Method
///   lifetimes](crate#method-lifetimes)), and for an elided output lifetime that the copy could
///   not resolve (E0106, see [Elided lifetimes](crate#elided-lifetimes)). Walking a trait or an
///   impl reaches it for each method.
/// - `Visit::visit_path(view, &path)` at the root, outside a type, analyzes only the arguments of
///   the path and does not check the path itself, which is resolved elsewhere. To analyze a trait
///   bound with its root checked, use [`View::visit_type_param_bound()`]: the path of the trait
///   must not depend on the context.
///
/// A function that is itself the analyzed item has no local generics, so do not visit its
/// signature. Enter a scope that declares its generics, and visit the types of its inputs and
/// output, and its where clause:
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
///     })
///     .unwrap();
/// assert_eq!(leaker.finish().iter().count(), 3);
/// ```
pub struct View<'v> {
    visitor: &'v mut AnalyzeVisitor,
}

impl<'v> View<'v> {
    pub(crate) fn new(visitor: &'v mut AnalyzeVisitor) -> Self {
        Self { visitor }
    }

    /// Checks whether `ty` must become an entry, and whether it can ([`CheckResult`]), without
    /// recording anything.
    ///
    /// It finds what [`View::visit_type()`] finds in `ty` here: with the generics of all the scopes
    /// around, `Self` and its `allow_self`, and the module with the names it declares (they are not
    /// causes). An absolute path of an external crate is not a cause, so `::foo::W<(Rel,)>` has
    /// only `Rel`, and `optional`. `Self` where it has no meaning is an impossible cause. The
    /// Leaker is as if `check()` was not called: no entry, context, error or module remains. It
    /// takes `&mut self` because it works on the analysis and puts it back.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let generics: Generics = parse_quote!(<T>);
    /// let mut leaker = Leaker::new();
    /// let (_, result) = leaker
    ///     .enter_generics(&generics, |view| {
    ///         view.check(&parse_quote!(::foo::W<T, (Rel,)>))
    ///     })
    ///     .unwrap();
    /// assert_eq!(result.must.len(), 1); // `Rel`
    /// assert!(!result.is_own && result.optional);
    /// assert_eq!(leaker.finish().iter().count(), 0);
    /// ```
    pub fn check(&mut self, ty: &Type) -> CheckResult {
        self.visitor.check(ty)
    }

    /// Like [`Leaker::enter_mod()`](crate::Leaker::enter_mod), nested in the module around if any.
    pub fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_mod(self, hint, declared, f)
    }

    /// Like [`Leaker::enter_generics()`](crate::Leaker::enter_generics), with local generics.
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_generics(self, generics, f)
    }

    /// Like [`Leaker::enter_type()`](crate::Leaker::enter_type).
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_type(self, ident, generics, allow_self, f)
    }

    /// Like [`Leaker::enter_impl()`](crate::Leaker::enter_impl).
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_impl(self, generics, self_ty, allow_self, f)
    }

    /// Like [`Leaker::enter_trait()`](crate::Leaker::enter_trait).
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_trait(self, ident, generics, allow_self, f)
    }
}

impl Scopes for View<'_> {
    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        self.visitor.enter(spec, f)
    }
}

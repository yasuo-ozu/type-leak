//! The [`View`], which a scope of the [`Leaker`](crate::Leaker) gives to its closure.

use crate::analyze::{Host, Res, ScopeSpec, Scopes};
use crate::context::ContextId;
use std::fmt::Display;
use syn::visit::Visit;
use syn::*;

mod forward;

/// What a scope gives to its closure: the syntax in the scope is analyzed through it.
///
/// It has `visit_*` methods, and the `enter_*` methods for nested scopes ([Scopes](crate#scopes)).
/// A scope sees what its nested scopes find as if it was written in it. `View` implements
/// [`Visit`], so `syn::visit::visit_item_struct(view, &item)` analyzes a struct as
/// [`Leaker::intern_struct()`](crate::Leaker::intern_struct) does. The walk declares nothing, so
/// walk the item the scope is for. [`View::visit_path()`] and [`View::visit_signature()`] differ
/// from the [`Visit`] methods of the same name, which the `syn::visit` functions call.
pub struct View<'v> {
    host: &'v mut dyn Host,
}

impl<'v> View<'v> {
    pub(crate) fn new(host: &'v mut dyn Host) -> Self {
        Self { host }
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

    /// Analyzes a nested item, such as a struct, a trait, an impl, a module or a function.
    ///
    /// It is analyzed like an item of a module
    /// ([`Leaker::intern_mod()`](crate::Leaker::intern_mod)). Its generics are local, and the
    /// generics around stay in scope. Use an `intern_*` method to analyze an item with its own
    /// generics as the Leaker's. A struct, enum, union, trait or impl declares its own `Self`,
    /// which may be interned whatever `allow_self` the scope around has.
    pub fn visit_item(&mut self, item: &Item) {
        Visit::visit_item(&mut *self.host, item)
    }

    /// Checks `path` as the root of a trait bound without analyzing it.
    pub(crate) fn check_bound_path(&mut self, path: &Path) {
        self.host.host_check_bound_path(path)
    }

    /// Analyzes the signature of a function or method. Its generics are local, and the body is not
    /// analyzed.
    ///
    /// It reports an error for a method lifetime that the copy would leave unconstrained (E0195).
    /// See [Method lifetimes](crate#method-lifetimes). If the innermost scope declares exactly
    /// `sig.generics`, they are not declared again, and the lifetimes are not checked. Walking a
    /// function with `syn::visit` declares and checks as well.
    pub fn visit_signature(&mut self, sig: &Signature) {
        self.host.host_visit_signature(sig)
    }

    /// Analyzes a type, with the generics in scope.
    pub fn visit_type(&mut self, ty: &Type) {
        Visit::visit_type(&mut *self.host, ty)
    }

    /// Analyzes a path as the root of a trait bound.
    ///
    /// The path is not leaked, but the types in its arguments are (`Rel` in `::path::Tr<Rel>`). The
    /// path must not depend on the context: a relative path, or `Self::X` where `Self` has no
    /// meaning, is an error. The arguments are analyzed even then.
    pub fn visit_path(&mut self, path: &Path) {
        self.host.host_check_bound_path(path);
        Visit::visit_path(&mut *self.host, path)
    }

    /// Analyzes the arguments of a path whose name is resolved elsewhere (`<Rel>` of `Wrap<Rel>`).
    pub fn visit_path_arguments(&mut self, arguments: &PathArguments) {
        Visit::visit_path_arguments(&mut *self.host, arguments)
    }

    /// Analyzes a where clause, like the one of the generics of the scope.
    pub fn visit_where_clause(&mut self, where_clause: &WhereClause) {
        Visit::visit_where_clause(&mut *self.host, where_clause)
    }

    /// Analyzes a bound, like a bound of a generic parameter of the scope.
    ///
    /// The path of a trait bound must not depend on the context, nor start with `Self` where it has
    /// no meaning. It is an error otherwise.
    pub fn visit_type_param_bound(&mut self, bound: &TypeParamBound) {
        Visit::visit_type_param_bound(&mut *self.host, bound)
    }
}

impl Scopes for View<'_> {
    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let mut f = Some(f);
        let mut out = None;
        let context = self.host.host_enter(spec, &mut |view| {
            if let Some(f) = f.take() {
                out = Some(f(view));
            }
        })?;
        match out {
            Some(out) => Ok((context, out)),
            None => unreachable!("a scope runs its closure"),
        }
    }
}

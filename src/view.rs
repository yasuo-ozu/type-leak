//! The [`View`], which a scope of the [`Leaker`](crate::Leaker) gives to its closure.

use crate::analysis::CheckResult;
use crate::analyze::{AnalyzeVisitor, Res, ScopeSpec, Scopes};
use crate::context::ContextId;
use std::fmt::Display;
use syn::*;

mod forward;

/// The handle that a scope of the [`Leaker`](crate::Leaker) gives to its closure. Pass it the
/// syntax to analyze.
///
/// Analyze a fragment with [`View::visit_type()`], [`View::visit_path_arguments()`],
/// [`View::visit_where_clause()`], [`View::visit_type_param_bound()`] or [`View::visit_item()`].
/// [`View::check()`] looks at a type without recording anything. The `enter_*` methods open a
/// nested scope: its generics are local generics, and the generics around stay in scope. A scope
/// sees what its nested scopes find as if it was written in it. `Self` is what the scope declares.
/// It has no meaning in `enter_generics()` and `enter_mod()`, so naming it there is an error, and
/// an item visited with `visit_item()` has a `Self` of its own. The ids, errors and rollback of a
/// scope are those of [`Leaker::enter_generics()`](crate::Leaker::enter_generics).
///
/// `View` also implements [`Visit`](syn::visit::Visit), so a walk with the `syn::visit` functions
/// analyzes an item: `syn::visit::visit_item_struct(view, &item)` does what
/// [`Leaker::intern_struct()`](crate::Leaker::intern_struct) does. The walk declares nothing, so
/// walk the item the scope is for. The default of a type parameter that it visits is analyzed with
/// the earlier parameters as the generics, and the default of a const parameter with the syntax
/// that visits it. Two methods of the trait do more than the walk:
///
/// - `Visit::visit_signature(view, &sig)` analyzes a method's signature. The generics of `sig` are
///   local generics, declared in a scope of their own. It also reports a method lifetime that the
///   copy would leave unconstrained (E0195, see [Method lifetimes](crate::Leaker#method-lifetimes))
///   and an elided output lifetime that the copy could not resolve (E0106, see [Elided
///   lifetimes](crate::Leaker#elided-lifetimes)). Walking a trait or an impl reaches it for each
///   method. The E0195 check is skipped for the methods of an inherent impl that the walk is in,
///   because no trait declares their signature. A signature visited alone may be a trait's, so it
///   is checked.
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
///
/// Each `visit_*` method analyzes one kind of syntax, with the generics, `Self` and module of the
/// scope:
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut generics: Generics = parse_quote!(<T: ::p::Bound<Rel1>>);
/// generics.where_clause = Some(parse_quote!(where T: ::p::Other<Rel2>));
/// let mut leaker = Leaker::new();
/// leaker
///     .enter_generics(&generics, |view| {
///         view.visit_type(&parse_quote!(Wrap<T>)); // a type
///         let arguments = PathArguments::AngleBracketed(parse_quote!(<Rel3>));
///         view.visit_path_arguments(&arguments); // the arguments of a path resolved elsewhere
///         if let Some(GenericParam::Type(param)) = generics.params.first() {
///             for bound in &param.bounds {
///                 view.visit_type_param_bound(bound); // a bound
///             }
///         }
///         view.visit_where_clause(generics.where_clause.as_ref().unwrap()); // a where clause
///     })
///     .unwrap();
/// // `Wrap<T>`, `Rel3`, `Rel1` and `Rel2`.
/// assert_eq!(leaker.finish().iter().count(), 4);
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
    /// Leaker is as if `check()` was not called: no entry, context, error or module remains. (It
    /// takes `&mut self` only because it borrows the analysis and puts it back.)
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

    /// Opens a module scope inside this one, like [`Leaker::enter_mod()`](crate::Leaker::enter_mod),
    /// nested in the module around if any.
    pub fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_mod(self, hint, declared, f)
    }

    /// Opens a scope that declares more generics, like
    /// [`Leaker::enter_generics()`](crate::Leaker::enter_generics). The generics around stay in
    /// scope, and these are local.
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_generics(self, generics, f)
    }

    /// Opens a scope for the parts of a struct, an enum or a union inside this one, like
    /// [`Leaker::enter_type()`](crate::Leaker::enter_type). The generics around stay in scope.
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_type(self, ident, generics, allow_self, f)
    }

    /// Opens a scope for the parts of an impl inside this one, like
    /// [`Leaker::enter_impl()`](crate::Leaker::enter_impl). The generics around stay in scope.
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_impl(self, generics, self_ty, allow_self, f)
    }

    /// Opens a scope for the parts of a trait inside this one, like
    /// [`Leaker::enter_trait()`](crate::Leaker::enter_trait). The generics around stay in scope.
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

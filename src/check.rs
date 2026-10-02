//! The analysis of a single type by [`Leaker::check()`].

use crate::leaker::Leaker;
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

/// Analyzes `ty` (see [`Leaker::check()`]).
pub(crate) fn check(leaker: &Leaker, ty: &Type, generics: &Generics) -> CheckResult {
    let mut visitor = CheckVisitor {
        leaker,
        generics,
        bound_lifetimes: Vec::new(),
        depth: 0,
        must: Vec::new(),
        impossible: Vec::new(),
    };
    visitor.visit_type(ty);
    // The outermost causes first (in the order of appearance among the same depth).
    let mut must = visitor.must;
    must.sort_by_key(|(depth, _)| *depth);
    CheckResult {
        is_own: must.first().map_or(false, |(depth, _)| *depth == 0),
        must: must.into_iter().map(|(_, m)| m).collect(),
        impossible: visitor.impossible,
    }
}

/// Collects the causes of [`CheckResult`] in a type.
struct CheckVisitor<'a> {
    leaker: &'a Leaker,
    /// The generics in scope (of the Leaker and local ones).
    generics: &'a Generics,
    /// The lifetimes of the `for<..>` binders in scope.
    bound_lifetimes: Vec<Lifetime>,
    /// The depth of the type being visited: `0` for the checked type itself, `1` for its
    /// components, and so on.
    depth: isize,
    /// The causes which depend on the type context, with the depth of the type they belong to.
    must: Vec<(isize, TokenStream)>,
    /// The causes which cannot be interned.
    impossible: Vec<TokenStream>,
}

impl CheckVisitor<'_> {
    /// Runs `f` for a component of the type being visited.
    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }

    /// Runs `f` with the lifetimes of the `for<..>` binder `lifetimes` in scope.
    fn with_bound_lifetimes(
        &mut self,
        lifetimes: Option<&BoundLifetimes>,
        f: impl FnOnce(&mut Self),
    ) {
        let len = self.bound_lifetimes.len();
        self.bound_lifetimes.extend(
            lifetimes
                .into_iter()
                .flat_map(|bound| bound.lifetimes.iter())
                .filter_map(|param| match param {
                    GenericParam::Lifetime(param) => Some(param.lifetime.clone()),
                    _ => None,
                }),
        );
        f(self);
        self.bound_lifetimes.truncate(len);
    }

    /// Records a path or a lifetime (of the type enclosing it) which depends on the type context.
    fn must_intern(&mut self, tokens: TokenStream) {
        self.must.push((self.depth - 1, tokens));
    }

    fn is_generic_lifetime(&self, lifetime: &Lifetime) -> bool {
        self.bound_lifetimes.contains(lifetime)
            || self.generics.params.iter().any(|param| {
                matches!(param, GenericParam::Lifetime(param) if &param.lifetime == lifetime)
            })
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
}

impl<'ast> Visit<'ast> for CheckVisitor<'_> {
    fn visit_type(&mut self, i: &Type) {
        match i {
            Type::BareFn(TypeBareFn {
                lifetimes,
                inputs,
                output,
                ..
            }) => {
                self.nested(|v| {
                    v.with_bound_lifetimes(lifetimes.as_ref(), |v| {
                        for input in inputs {
                            v.visit_type(&input.ty);
                        }
                        if let ReturnType::Type(_, output) = output {
                            v.visit_type(output);
                        }
                    })
                });
                return;
            }
            Type::TraitObject(TypeTraitObject { bounds, .. }) => {
                // All the bounds are checked (e.g. both `A` and `B` in `dyn A + B`).
                for bound in bounds {
                    match bound {
                        TypeParamBound::Trait(TraitBound {
                            lifetimes, path, ..
                        }) => self.nested(|v| {
                            v.with_bound_lifetimes(lifetimes.as_ref(), |v| v.visit_path(path))
                        }),
                        TypeParamBound::Verbatim(_) => self.impossible.push(quote!(#bound)),
                        _ => (),
                    }
                }
                return;
            }
            Type::Macro(_) => {
                // The tokens of a type macro cannot be analyzed, so it is interned as a whole (it
                // is expanded in the Leaker's context).
                self.must.push((self.depth, quote!(#i)));
                return;
            }
            Type::ImplTrait(_)
            | Type::Verbatim(_)
            | Type::Reference(TypeReference { lifetime: None, .. }) => {
                self.impossible.push(quote!(#i));
            }
            _ => (),
        }
        self.nested(|v| syn::visit::visit_type(v, i));
    }

    fn visit_qself(&mut self, i: &QSelf) {
        if i.as_token.is_none() {
            let ty = &i.ty;
            self.impossible.push(quote!(#ty));
        }
        syn::visit::visit_qself(self, i)
    }

    fn visit_lifetime(&mut self, i: &Lifetime) {
        if i.ident == "_" {
            self.impossible.push(quote!(#i));
        } else if i.ident != "static" && !self.is_generic_lifetime(i) {
            self.must_intern(quote!(#i));
        }
    }

    fn visit_expr(&mut self, i: &Expr) {
        if let Expr::Closure(_) | Expr::Assign(_) | Expr::Verbatim(_) | Expr::Macro(_) = i {
            self.impossible.push(quote!(#i));
        }
        syn::visit::visit_expr(self, i)
    }

    fn visit_path(&mut self, i: &Path) {
        // The generic arguments are checked in any case (e.g. `RelType` in
        // `::core::option::Option<RelType>`).
        let is_self = i.leading_colon.is_none()
            && matches!(i.segments.first(), Some(seg) if seg.ident == "Self" && seg.arguments.is_none());
        if self.leaker.is_context_free_path(i) || self.starts_with_generic(i) {
        } else if is_self {
            if !self.leaker.self_ty_can_be_interned {
                self.impossible.push(quote!(#i));
            }
        } else {
            self.must_intern(quote!(#i));
        }
        syn::visit::visit_path(self, i)
    }
}

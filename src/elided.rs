//! The copy of a signature whose elided lifetime the type written for `Self` would break.
//!
//! The output of a function pointer or of `Fn(..) -> R` may have an elided lifetime, which
//! elision takes from the only lifetime of the inputs. `Self` names none, so `fn(&Self) -> &u8` is
//! valid in `struct S<'a>`, but the copy writes a type for `Self` (`Visitor::with_self()`):
//! `fn(&::p::S<'a>) -> &u8` has two lifetimes in its inputs, and rustc cannot tell which one the
//! output has (E0106). The Visitor names the lifetime that elision gives instead, with a fresh
//! `for<'e>` binder: `for<'e> fn(&'e ::p::S<'a>) -> &'e u8`. The Leaker does not count such a
//! signature as broken (`AnalyzeVisitor::self_elision_broken()`), so the two agree on which
//! signatures this handles: those whose inputs show their one lifetime. See
//! [Elided lifetimes](crate::rules#elided-lifetimes).

use crate::analyze::{ElidedOutput, InputLifetimes};
use crate::context::{binder_lifetimes, Scope};
use proc_macro2::Span;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;

/// Names the elided lifetime of the signature of the function pointer `i`, if `self_ty` written
/// for the `Self` of its inputs adds lifetimes to them. `scopes` are the generics in scope, whose
/// lifetimes the binder must not shadow.
pub(crate) fn name_elided_in_bare_fn(self_ty: &Type, scopes: &[Scope], i: &mut TypeBareFn) {
    name_elided(
        self_ty,
        scopes,
        &mut i.lifetimes,
        i.inputs.iter_mut().map(|arg| &mut arg.ty).collect(),
        &mut i.output,
    )
}

/// Like [`name_elided_in_bare_fn()`], for the signature of `Fn(..) -> R` (the last segment of the
/// path of `i` has parenthesized arguments), which `i` binds.
pub(crate) fn name_elided_in_bound(self_ty: &Type, scopes: &[Scope], i: &mut TraitBound) {
    if let Some(PathArguments::Parenthesized(args)) =
        i.path.segments.last_mut().map(|seg| &mut seg.arguments)
    {
        name_elided(
            self_ty,
            scopes,
            &mut i.lifetimes,
            args.inputs.iter_mut().collect(),
            &mut args.output,
        )
    }
}

fn name_elided(
    self_ty: &Type,
    scopes: &[Scope],
    binder: &mut Option<BoundLifetimes>,
    mut inputs: Vec<&mut Type>,
    output: &mut ReturnType,
) {
    let output = match output {
        ReturnType::Type(_, output) => output,
        ReturnType::Default => return,
    };
    let mut elided = ElidedOutput(false);
    elided.visit_type(output);
    if !elided.0 {
        return;
    }
    // The inputs show their one lifetime. With none, a path hides it (`Ref<u8>`): it cannot be
    // named, and the Leaker leaks such a signature whole or rejects it. With more, the original is
    // not valid.
    let mut lifetimes = InputLifetimes::default();
    for input in &inputs {
        lifetimes.visit_type(input);
    }
    let kept = match (lifetimes.elided, lifetimes.named.len()) {
        (1, 0) => None,
        (0, 1) => lifetimes.named.pop(),
        _ => return,
    };
    let mut selves = MentionsSelf(false);
    for input in &inputs {
        selves.visit_type(input);
    }
    if !selves.0 {
        return;
    }
    // `Self` adds what `self_ty` names, but the lifetime that the inputs keep.
    let mut added = InputLifetimes::default();
    added.visit_type(self_ty);
    if added.elided == 0
        && added
            .named
            .iter()
            .all(|lifetime| Some(lifetime) == kept.as_ref())
    {
        return;
    }
    let name = match kept {
        // The output takes the lifetime that the inputs name.
        Some(kept) => kept,
        None => {
            let name = fresh(scopes, binder.as_ref());
            let param = GenericParam::Lifetime(LifetimeParam::new(name.clone()));
            match binder {
                Some(binder) => binder.lifetimes.push(param),
                None => {
                    *binder = Some(BoundLifetimes {
                        for_token: Default::default(),
                        lt_token: Default::default(),
                        lifetimes: std::iter::once(param).collect(),
                        gt_token: Default::default(),
                    })
                }
            }
            for input in &mut inputs {
                Name(&name).visit_type_mut(input);
            }
            name
        }
    };
    Name(&name).visit_type_mut(output);
}

/// Returns a lifetime that nothing in `scopes` or `binder` declares.
fn fresh(scopes: &[Scope], binder: Option<&BoundLifetimes>) -> Lifetime {
    (0..)
        .map(|n| Lifetime::new(&format!("'__tl_e{}", n), Span::call_site()))
        .find(|name| {
            !scopes.iter().any(|scope| scope.declares(&name.to_string()))
                && !binder_lifetimes(binder).any(|declared| declared == *name)
        })
        .expect("an unbounded range")
}

/// `true` if a type names `Self` (`Self`, `Self::A`) where it is a lifetime of the signature
/// around that it adds to. A function pointer or `Fn(..)` inside has its own inputs.
struct MentionsSelf(bool);

impl<'ast> Visit<'ast> for MentionsSelf {
    fn visit_type_path(&mut self, i: &'ast TypePath) {
        // As `ReplaceSelf` replaces it.
        if let (None, None, Some(first)) =
            (&i.qself, &i.path.leading_colon, i.path.segments.first())
        {
            self.0 |= first.ident == "Self" && first.arguments.is_none();
        }
        syn::visit::visit_type_path(self, i)
    }
    fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {}
    fn visit_parenthesized_generic_arguments(&mut self, _: &'ast ParenthesizedGenericArguments) {}
}

/// Replaces the elided lifetimes that elision counts for the signature around (`&` with none,
/// `'_`) with a name. A function pointer or `Fn(..)` inside has its own.
struct Name<'a>(&'a Lifetime);

impl<'a> VisitMut for Name<'a> {
    fn visit_type_reference_mut(&mut self, i: &mut TypeReference) {
        if i.lifetime.is_none() {
            i.lifetime = Some(self.0.clone());
        }
        syn::visit_mut::visit_type_reference_mut(self, i)
    }
    fn visit_lifetime_mut(&mut self, i: &mut Lifetime) {
        if i.ident == "_" {
            *i = self.0.clone();
        }
    }
    fn visit_type_bare_fn_mut(&mut self, _: &mut TypeBareFn) {}
    fn visit_parenthesized_generic_arguments_mut(&mut self, _: &mut ParenthesizedGenericArguments) {
    }
}

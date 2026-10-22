use proc_macro2::TokenStream;
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::{CheckResult, ContextId, ErrorItem, Leaker, View};

/// Converts the required types of a `Referrer` (see `Referrer::must()`) to a `BTreeSet<String>`
/// for order-independent comparison. The optional ones, which only the paths of external crates
/// (`::std::vec::Vec<..>`) cause, are not among them.
#[allow(dead_code)]
pub fn finish_as_strings(leaker: Leaker) -> BTreeSet<String> {
    let referrer = leaker.finish();
    referrer
        .iter()
        .enumerate()
        .filter(|(ix, _)| referrer.must(*ix))
        .map(|(_, t)| quote!(#t).to_string())
        .collect()
}

/// Interns the items `items` of a trait `Tr` with the generics `generics` (e.g. `fn f(&self) ->
/// Rel<Self>;`), where `Self` may be interned if `allow_self`.
#[allow(dead_code)]
pub fn intern_fns(
    leaker: &mut Leaker,
    generics: &Generics,
    items: TokenStream,
    allow_self: bool,
) -> std::result::Result<ContextId, Vec<ErrorItem>> {
    let where_clause = &generics.where_clause;
    let item: ItemTrait = parse_quote!(trait Tr #generics #where_clause { #items });
    leaker.intern_trait(&item, allow_self)
}

/// Analyzes the types of the function `sig`, which is itself the analyzed item, in a scope that
/// declares its generics: its inputs, its output and its where clause (see `View`). The function
/// has no receiver.
#[allow(dead_code)]
pub fn visit_function(view: &mut View<'_>, sig: &Signature) {
    for input in &sig.inputs {
        if let FnArg::Typed(arg) = input {
            view.visit_type(&arg.ty);
        }
    }
    if let ReturnType::Type(_, ty) = &sig.output {
        view.visit_type(ty);
    }
    if let Some(where_clause) = &sig.generics.where_clause {
        view.visit_where_clause(where_clause);
    }
}

/// Enters the generics of the function `sig` and analyzes it there ([`visit_function()`]).
#[allow(dead_code)]
pub fn enter_function(
    leaker: &mut Leaker,
    sig: &Signature,
) -> std::result::Result<(ContextId, ()), Vec<ErrorItem>> {
    leaker.enter_generics(&sig.generics, |view| visit_function(view, sig))
}

/// Checks `ty` in a scope that declares the generics `generics` and nothing else (see
/// `View::check()`). `Self` has no meaning there.
#[allow(dead_code)]
pub fn check_in(leaker: &mut Leaker, ty: &Type, generics: &Generics) -> CheckResult {
    leaker
        .enter_generics(generics, |view| view.check(ty))
        .unwrap()
        .1
}

use proc_macro2::TokenStream;
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::{ContextId, Leaker, NotInternableError};

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
) -> std::result::Result<ContextId, NotInternableError> {
    let where_clause = &generics.where_clause;
    let item: ItemTrait = parse_quote!(trait Tr #generics #where_clause { #items });
    leaker.intern_trait(&item, allow_self)
}

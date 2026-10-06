use std::collections::BTreeSet;
use template_quote::quote;
use type_leak::Leaker;

/// Converts the required types of a `Referrer` (see `Referrer::must()`) to a `BTreeSet<String>`
/// for order-independent comparison. The optional ones, which only the paths of external crates
/// (`::std::vec::Vec<..>`) cause, are not among them.
pub fn finish_as_strings(leaker: Leaker) -> BTreeSet<String> {
    let referrer = leaker.finish();
    referrer
        .iter()
        .enumerate()
        .filter(|(ix, _)| referrer.must(*ix))
        .map(|(_, t)| quote!(#t).to_string())
        .collect()
}

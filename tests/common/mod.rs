use std::collections::BTreeSet;
use template_quote::quote;
use type_leak::Leaker;

/// Converts a `Referrer`'s types to a `BTreeSet<String>` for order-independent comparison.
pub fn finish_as_strings(leaker: Leaker) -> BTreeSet<String> {
    leaker
        .finish()
        .iter()
        .map(|t| quote!(#t).to_string())
        .collect()
}

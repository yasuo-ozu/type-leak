//! Helpers for inspecting paths and types.

use crate::error::ABSOLUTE_PATHS;
use proc_macro2::TokenStream;
use std::collections::HashSet;
use syn::ext::IdentExt;
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// Returns `true` if the segments of `path` start with those of `prefix` (e.g. `Option<T>` starts
/// with `Option`, and `std::vec::Vec<T>` starts with `std::vec`), comparing only the identifiers:
/// generic arguments of `path` are allowed.
pub(crate) fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    prefix.leading_colon.is_some() == path.leading_colon.is_some()
        && prefix.segments.len() <= path.segments.len()
        && prefix
            .segments
            .iter()
            .zip(path.segments.iter())
            .all(|(prefix_seg, seg)| prefix_seg.ident == seg.ident)
}

/// Returns `true` if `ident` is the name of a primitive type (e.g. `u8` or `str`).
pub(crate) fn is_primitive_name(ident: &Ident) -> bool {
    ABSOLUTE_PATHS
        .iter()
        .any(|(name, absolute)| ident == name && absolute.starts_with("::core::primitive::"))
}

/// Returns `true` if `ty` refers to any of `idents` (as the first segment of a relative path) or
/// `lifetimes`.
pub(crate) fn mentions_names(ty: &Type, idents: Vec<Ident>, lifetimes: Vec<Lifetime>) -> bool {
    struct Visitor {
        idents: Vec<Ident>,
        lifetimes: Vec<Lifetime>,
        found: bool,
    }
    impl<'ast> Visit<'ast> for Visitor {
        fn visit_path(&mut self, i: &'ast Path) {
            if i.leading_colon.is_none()
                && i.segments
                    .first()
                    .map_or(false, |seg| self.idents.contains(&seg.ident))
            {
                self.found = true;
            }
            syn::visit::visit_path(self, i)
        }
        fn visit_lifetime(&mut self, i: &'ast Lifetime) {
            if self.lifetimes.contains(i) {
                self.found = true;
            }
        }
        fn visit_macro(&mut self, i: &'ast Macro) {
            // The tokens of a macro are not parsed, so they are matched by name.
            let mut names = HashSet::new();
            token_names(i.tokens.clone(), &mut names);
            if self
                .idents
                .iter()
                .any(|ident| names.contains(&ident.to_string()))
                || self
                    .lifetimes
                    .iter()
                    .any(|lt| names.contains(&lt.to_string()))
            {
                self.found = true;
            }
            syn::visit::visit_macro(self, i)
        }
    }
    let mut visitor = Visitor {
        idents,
        lifetimes,
        found: false,
    };
    if visitor.idents.is_empty() && visitor.lifetimes.is_empty() {
        return false;
    }
    visitor.visit_type(ty);
    visitor.found
}

/// Returns the name of `ident` without `r#` (e.g. `T` for `r#T`), by which names are compared.
pub(crate) fn ident_name(ident: &Ident) -> String {
    ident.unraw().to_string()
}

/// Collects the names of identifiers and lifetimes (e.g. `N` and `'a`) in `tokens`, without `r#`.
pub(crate) fn token_names(tokens: TokenStream, names: &mut HashSet<String>) {
    let mut after_quote = false;
    for tt in tokens {
        match &tt {
            proc_macro2::TokenTree::Ident(ident) if after_quote => {
                names.insert(format!("'{}", ident));
            }
            proc_macro2::TokenTree::Ident(ident) => {
                names.insert(ident_name(ident));
            }
            proc_macro2::TokenTree::Group(group) => token_names(group.stream(), names),
            _ => (),
        }
        after_quote = false;
        if let proc_macro2::TokenTree::Punct(punct) = &tt {
            after_quote = punct.as_char() == '\'';
        }
    }
}

/// Returns the name of a generic parameter, e.g. `N` or `'a`.
pub(crate) fn generic_param_name(param: &GenericParam) -> String {
    match param {
        GenericParam::Type(TypeParam { ident, .. })
        | GenericParam::Const(ConstParam { ident, .. }) => ident_name(ident),
        GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => lifetime.to_string(),
    }
}

/// Returns the type inside invisible groups (e.g. from `$t:ty` in `macro_rules!`) and
/// parentheses.
pub(crate) fn unwrap_type_group(ty: &Type) -> &Type {
    match ty {
        Type::Group(TypeGroup { elem, .. }) | Type::Paren(TypeParen { elem, .. }) => {
            unwrap_type_group(elem)
        }
        _ => ty,
    }
}

/// Finds the first mention of `Self` (a path starting with it, e.g. `Self`, `Self::X` or the self
/// type of `<Self as ..>`, or the name in the tokens of a macro, but not `::p::Self`).
#[derive(Default)]
struct SelfFinder {
    found: Option<TokenStream>,
}

impl SelfFinder {
    /// Scans the tokens, which are not parsed (e.g. of a macro), for the name.
    fn scan(&mut self, tokens: TokenStream) {
        let mut names = HashSet::new();
        token_names(tokens, &mut names);
        if names.contains("Self") {
            self.found.get_or_insert_with(|| quote!(Self));
        }
    }
}

impl<'ast> Visit<'ast> for SelfFinder {
    fn visit_path(&mut self, i: &'ast Path) {
        if i.leading_colon.is_none() && i.segments.first().map_or(false, |s| s.ident == "Self") {
            self.found.get_or_insert_with(|| quote!(#i));
        }
        syn::visit::visit_path(self, i)
    }
    fn visit_macro(&mut self, i: &'ast Macro) {
        self.scan(i.tokens.clone());
        syn::visit::visit_macro(self, i)
    }
    fn visit_type(&mut self, i: &'ast Type) {
        if let Type::Verbatim(tokens) = i {
            self.scan(tokens.clone());
        }
        syn::visit::visit_type(self, i)
    }
}

/// Returns `true` if `ty` mentions `Self`.
pub(crate) fn type_mentions_self(ty: &Type) -> bool {
    let mut finder = SelfFinder::default();
    finder.visit_type(ty);
    finder.found.is_some()
}

/// Returns `true` if the bounds (or the default) of `param` mention `Self`.
pub(crate) fn generic_param_mentions_self(param: &GenericParam) -> bool {
    let mut finder = SelfFinder::default();
    finder.visit_generic_param(param);
    finder.found.is_some()
}

/// Returns `true` if `predicate` mentions `Self`.
pub(crate) fn where_predicate_mentions_self(predicate: &WherePredicate) -> bool {
    let mut finder = SelfFinder::default();
    finder.visit_where_predicate(predicate);
    finder.found.is_some()
}

/// Returns the first mention of `Self` in `generics` (the bounds and the where predicates).
pub(crate) fn first_self_in_generics(generics: &Generics) -> Option<TokenStream> {
    let mut finder = SelfFinder::default();
    finder.visit_generics(generics);
    finder.found
}

/// Returns the first mention of `Self` in `ty`.
pub(crate) fn first_self_in_type(ty: &Type) -> Option<TokenStream> {
    let mut finder = SelfFinder::default();
    finder.visit_type(ty);
    finder.found
}

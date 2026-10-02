//! Helpers for inspecting paths and types.

use crate::error::ABSOLUTE_PATHS;
use proc_macro2::TokenStream;
use std::collections::HashSet;
use syn::visit::Visit;
use syn::*;

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

/// Returns `true` if `ty` is a single identifier named like a const by the naming convention
/// (`SCREAMING_SNAKE_CASE` with at least two characters, e.g. `SIZE`), which syn parses as a type
/// when used as a generic argument.
pub(crate) fn looks_like_const(ty: &Type) -> bool {
    match ty {
        Type::Path(TypePath { qself: None, path }) => path.get_ident().map_or(false, |ident| {
            let name = ident.to_string();
            name.len() >= 2
                && name.chars().any(|c| c.is_ascii_uppercase())
                && name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        }),
        _ => false,
    }
}

/// Returns `true` if `ty` refers to any of `generics`.
pub(crate) fn mentions_generics<'a>(
    ty: &Type,
    generics: impl IntoIterator<Item = &'a GenericParam>,
) -> bool {
    let mut idents = Vec::new();
    let mut lifetimes = Vec::new();
    for gp in generics {
        match gp {
            GenericParam::Type(TypeParam { ident, .. })
            | GenericParam::Const(ConstParam { ident, .. }) => idents.push(ident.clone()),
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                lifetimes.push(lifetime.clone())
            }
        }
    }
    mentions_names(ty, idents, lifetimes)
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

/// Collects the names of identifiers and lifetimes (e.g. `N` and `'a`) in `tokens`.
pub(crate) fn token_names(tokens: TokenStream, names: &mut HashSet<String>) {
    let mut after_quote = false;
    for tt in tokens {
        match &tt {
            proc_macro2::TokenTree::Ident(ident) if after_quote => {
                names.insert(format!("'{}", ident));
            }
            proc_macro2::TokenTree::Ident(ident) => {
                names.insert(ident.to_string());
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
        | GenericParam::Const(ConstParam { ident, .. }) => ident.to_string(),
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

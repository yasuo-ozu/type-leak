//! Helpers for inspecting paths and types.

use crate::error::ABSOLUTE_PATHS;
use proc_macro2::TokenStream;
use std::collections::HashSet;
use std::rc::Rc;
use syn::ext::IdentExt;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::{quote, ToTokens};

/// Returns `true` if `path` starts with `prefix`, comparing identifiers. Both have a leading `::`
/// or neither.
pub(crate) fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    prefix.leading_colon.is_some() == path.leading_colon.is_some()
        && prefix.segments.len() <= path.segments.len()
        && prefix
            .segments
            .iter()
            .zip(path.segments.iter())
            .all(|(prefix_seg, seg)| prefix_seg.ident == seg.ident)
}

/// The paths a Leaker treats as context-free: a copy of its `allowed_paths` taken when an analysis
/// starts, shared by the throwaway visitors of that analysis.
#[derive(Clone)]
pub(crate) struct AllowedPaths(Rc<[Path]>);

impl AllowedPaths {
    pub(crate) fn new(paths: &[Path]) -> Self {
        Self(Rc::from(paths))
    }

    /// Returns `true` if `path` is absolute, starts with `$crate`, or starts with an allowed path.
    /// A type with an absolute path may still be an optional entry (see `is_soft_path()`).
    pub(crate) fn is_context_free_path(&self, path: &Path) -> bool {
        path.leading_colon.is_some()
            || matches!(
                path.segments.iter().next(),
                Some(PathSegment {
                    ident,
                    arguments: PathArguments::None,
                }) if ident == "$crate"
            )
            || self.is_allowed_path(path)
    }

    fn is_allowed_path(&self, path: &Path) -> bool {
        self.0.iter().any(|allowed| path_starts_with(path, allowed))
    }

    /// Returns `true` for an absolute path to an external crate (not `::core`, not allowed). The
    /// Referrer may replace such a type, so nothing in it is anchored (see `scan_anchors()`).
    pub(crate) fn is_soft_path(&self, path: &Path) -> bool {
        path.leading_colon.is_some()
            && !matches!(path.segments.first(), Some(seg) if seg.ident == "core")
            && !self.is_allowed_path(path)
    }
}

pub(crate) fn is_primitive_name(ident: &Ident) -> bool {
    ABSOLUTE_PATHS
        .iter()
        .any(|(name, absolute)| ident == name && absolute.starts_with("::core::primitive::"))
}

/// Returns `true` if `ty` refers to `idents` (as the first segment of a relative path) or
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

/// Returns the name of `ident` without `r#`, by which names are compared.
pub(crate) fn ident_name(ident: &Ident) -> String {
    ident.unraw().to_string()
}

/// Collects the names of the identifiers and lifetimes in `tokens`, without `r#`.
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

pub(crate) fn generic_param_name(param: &GenericParam) -> String {
    match param {
        GenericParam::Type(TypeParam { ident, .. })
        | GenericParam::Const(ConstParam { ident, .. }) => ident_name(ident),
        GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => lifetime.to_string(),
    }
}

/// Returns the type inside invisible groups (from `$t:ty` in `macro_rules!`) and parentheses.
pub(crate) fn unwrap_type_group(ty: &Type) -> &Type {
    match ty {
        Type::Group(TypeGroup { elem, .. }) | Type::Paren(TypeParen { elem, .. }) => {
            unwrap_type_group(elem)
        }
        _ => ty,
    }
}

pub(crate) fn has_lifetime_bound(object: &TypeTraitObject) -> bool {
    object
        .bounds
        .iter()
        .any(|bound| matches!(bound, TypeParamBound::Lifetime(_)))
}

/// Returns the trait object that `ty` is, also inside parentheses and groups.
pub(crate) fn trait_object_of(ty: &Type) -> Option<&TypeTraitObject> {
    match unwrap_type_group(ty) {
        Type::TraitObject(object) => Some(object),
        _ => None,
    }
}

/// Finds the mentions of `Self`: a path starting with it (`Self`, `Self::X`, the self type of
/// `<Self as ..>`), or the name in the tokens of a macro. `::p::Self` is not one.
#[derive(Default)]
struct SelfFinder {
    found: Vec<TokenStream>,
    /// The paths of `found` with an item after `Self` (`Self::X`).
    associated: Vec<TokenStream>,
}

impl SelfFinder {
    /// Scans tokens that are not parsed, such as those of a macro, for the name.
    fn scan(&mut self, tokens: TokenStream) {
        let mut names = HashSet::new();
        token_names(tokens, &mut names);
        if names.contains("Self") {
            self.found.push(quote!(Self));
        }
    }
}

impl<'ast> Visit<'ast> for SelfFinder {
    fn visit_path(&mut self, i: &'ast Path) {
        if i.leading_colon.is_none() && i.segments.first().map_or(false, |s| s.ident == "Self") {
            self.found.push(quote!(#i));
            if i.segments.len() > 1 {
                self.associated.push(quote!(#i));
            }
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

pub(crate) fn type_mentions_self(ty: &Type) -> bool {
    let mut finder = SelfFinder::default();
    finder.visit_type(ty);
    !finder.found.is_empty()
}

pub(crate) fn generic_param_mentions_self(param: &GenericParam) -> bool {
    let mut finder = SelfFinder::default();
    finder.visit_generic_param(param);
    !finder.found.is_empty()
}

pub(crate) fn where_predicate_mentions_self(predicate: &WherePredicate) -> bool {
    let mut finder = SelfFinder::default();
    finder.visit_where_predicate(predicate);
    !finder.found.is_empty()
}

/// The mentions of `Self` in a generic parameter or a where predicate.
pub(crate) struct SelfMentions {
    pub(crate) first: Option<TokenStream>,
    /// The first `Self::X`, an associated item of `Self`.
    pub(crate) associated: Option<TokenStream>,
}

impl From<SelfFinder> for SelfMentions {
    fn from(finder: SelfFinder) -> Self {
        Self {
            first: finder.found.into_iter().next(),
            associated: finder.associated.into_iter().next(),
        }
    }
}

pub(crate) fn self_in_param(param: &GenericParam) -> SelfMentions {
    let mut finder = SelfFinder::default();
    finder.visit_generic_param(param);
    finder.into()
}

pub(crate) fn self_in_predicate(predicate: &WherePredicate) -> SelfMentions {
    let mut finder = SelfFinder::default();
    finder.visit_where_predicate(predicate);
    finder.into()
}

pub(crate) fn first_self_in_generics(generics: &Generics) -> Option<TokenStream> {
    selves_in_generics(generics).into_iter().next()
}

/// Returns the mentions of `Self` in the bounds, defaults and where predicates of `generics`.
pub(crate) fn selves_in_generics(generics: &Generics) -> Vec<TokenStream> {
    let mut finder = SelfFinder::default();
    finder.visit_generics(generics);
    finder.found
}

pub(crate) fn first_self_in_type(ty: &Type) -> Option<TokenStream> {
    let mut finder = SelfFinder::default();
    finder.visit_type(ty);
    finder.found.into_iter().next()
}

/// Returns the first `n` parameters without defaults, and the where predicates that do not name the
/// others.
pub(crate) fn without_defaults(generics: &Generics, n: usize) -> Generics {
    let params: Punctuated<GenericParam, Token![,]> = generics
        .params
        .iter()
        .take(n)
        .cloned()
        .map(|mut param| {
            match &mut param {
                GenericParam::Type(tp) => {
                    tp.eq_token = None;
                    tp.default = None;
                }
                GenericParam::Const(cp) => {
                    cp.eq_token = None;
                    cp.default = None;
                }
                GenericParam::Lifetime(_) => (),
            }
            param
        })
        .collect();
    let later: HashSet<String> = generics
        .params
        .iter()
        .skip(n)
        .map(generic_param_name)
        .collect();
    let predicates: Punctuated<WherePredicate, Token![,]> = generics
        .where_clause
        .iter()
        .flat_map(|wc| wc.predicates.iter())
        .filter(|predicate| {
            let mut names = HashSet::new();
            token_names(quote!(#predicate), &mut names);
            names.is_disjoint(&later)
        })
        .cloned()
        .collect();
    let has_params = !params.is_empty();
    Generics {
        lt_token: if has_params {
            Some(Default::default())
        } else {
            None
        },
        params,
        gt_token: if has_params {
            Some(Default::default())
        } else {
            None
        },
        where_clause: if predicates.is_empty() {
            None
        } else {
            Some(WhereClause {
                where_token: Default::default(),
                predicates,
            })
        },
    }
}

/// Replaces `Self` by a type, in the types, expressions and macros it folds.
#[derive(Clone, Copy)]
pub(crate) struct ReplaceSelf<'a> {
    pub(crate) self_ty: &'a Type,
}

impl<'a> ReplaceSelf<'a> {
    fn replace_tokens(&self, tokens: TokenStream) -> TokenStream {
        use proc_macro2::{Group, TokenTree};
        tokens
            .into_iter()
            .map(|tt| match tt {
                TokenTree::Ident(ident) if ident == "Self" => self.self_ty.to_token_stream(),
                TokenTree::Group(group) => {
                    let mut new =
                        Group::new(group.delimiter(), self.replace_tokens(group.stream()));
                    new.set_span(group.span());
                    TokenTree::Group(new).into()
                }
                tt => tt.into(),
            })
            .collect()
    }

    // `Some(Some(rest))` for `Self::A::B`, `Some(None)` for `Self` alone.
    fn strip_self(path: &Path) -> Option<Option<Path>> {
        let first = path.segments.first()?;
        if path.leading_colon.is_some() || first.ident != "Self" || !first.arguments.is_none() {
            return None;
        }
        // The `::` after the qualified self type (see `qself()`).
        Some((path.segments.len() > 1).then(|| Path {
            leading_colon: Some(Default::default()),
            segments: path.segments.iter().skip(1).cloned().collect(),
        }))
    }

    fn qself(&self) -> QSelf {
        QSelf {
            lt_token: Default::default(),
            ty: Box::new(self.self_ty.clone()),
            position: 0,
            as_token: None,
            gt_token: Default::default(),
        }
    }

    pub(crate) fn fold_type_with(&self, folder: &mut impl syn::fold::Fold, ty: Type) -> Type {
        let stripped = match &ty {
            Type::Path(TypePath { qself: None, path }) => Self::strip_self(path),
            _ => None,
        };
        match (ty, stripped) {
            (_, Some(None)) => self.self_ty.clone(),
            (_, Some(Some(rest))) => Type::Path(TypePath {
                qself: Some(self.qself()),
                path: folder.fold_path(rest),
            }),
            (Type::Verbatim(tokens), _) => Type::Verbatim(self.replace_tokens(tokens)),
            (ty, _) => syn::fold::fold_type(folder, ty),
        }
    }

    /// `Self` alone in an expression is a value, so only `Self::A` is replaced.
    pub(crate) fn fold_expr_with(&self, folder: &mut impl syn::fold::Fold, expr: Expr) -> Expr {
        let stripped = match &expr {
            Expr::Path(ExprPath {
                qself: None, path, ..
            }) => Self::strip_self(path),
            _ => None,
        };
        match (expr, stripped) {
            (Expr::Path(path), Some(Some(rest))) => Expr::Path(ExprPath {
                qself: Some(self.qself()),
                path: folder.fold_path(rest),
                ..path
            }),
            (expr, _) => syn::fold::fold_expr(folder, expr),
        }
    }
}

impl<'a> syn::fold::Fold for ReplaceSelf<'a> {
    fn fold_type(&mut self, ty: Type) -> Type {
        let this = *self;
        this.fold_type_with(self, ty)
    }

    fn fold_expr(&mut self, expr: Expr) -> Expr {
        let this = *self;
        this.fold_expr_with(self, expr)
    }

    fn fold_macro(&mut self, i: Macro) -> Macro {
        Macro {
            tokens: self.replace_tokens(i.tokens),
            ..i
        }
    }
}

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

/// Collects the names by which `ty` can refer to a generic of the scopes it is written in: the
/// first segment of each relative path, the lifetimes, and the names in the tokens that are not
/// parsed (a macro, syntax that `syn` leaves verbatim).
///
/// A later segment (`U` in `inner::U`), the name of an associated type binding (`Item` in
/// `Iterator<Item = u8>`) and the name of a function pointer argument name an item, not a
/// generic. The Leaker finds the generics a type mentions the same way (`AnalyzeVisitor`
/// records a mention at `starts_with_generic()`), so the context of the type it chooses is the
/// one that the Referrer computes by this function: a segment named like a generic must not add
/// the generic to the context (`inner::U` in `fn f<U>(&self) -> inner::U`).
pub(crate) fn type_names(ty: &Type, names: &mut HashSet<String>) {
    struct Names<'a>(&'a mut HashSet<String>);
    impl Names<'_> {
        fn tokens(&mut self, tokens: &TokenStream) {
            token_names(tokens.clone(), self.0);
        }
    }
    impl<'ast> Visit<'ast> for Names<'_> {
        fn visit_path(&mut self, i: &'ast Path) {
            if i.leading_colon.is_none() {
                if let Some(seg) = i.segments.first() {
                    self.0.insert(ident_name(&seg.ident));
                }
            }
            syn::visit::visit_path(self, i)
        }
        fn visit_lifetime(&mut self, i: &'ast Lifetime) {
            self.0.insert(i.to_string());
        }
        fn visit_macro(&mut self, i: &'ast Macro) {
            self.tokens(&i.tokens);
            syn::visit::visit_macro(self, i)
        }
        fn visit_type(&mut self, i: &'ast Type) {
            match i {
                Type::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_type(self, i),
            }
        }
        fn visit_expr(&mut self, i: &'ast Expr) {
            match i {
                Expr::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_expr(self, i),
            }
        }
        fn visit_pat(&mut self, i: &'ast Pat) {
            match i {
                Pat::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_pat(self, i),
            }
        }
        fn visit_type_param_bound(&mut self, i: &'ast TypeParamBound) {
            match i {
                TypeParamBound::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_type_param_bound(self, i),
            }
        }
        fn visit_item(&mut self, i: &'ast Item) {
            match i {
                Item::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_item(self, i),
            }
        }
    }
    Names(names).visit_type(ty)
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

/// Wraps `ty` in parentheses.
fn parenthesize(ty: &mut Type) {
    let inner = std::mem::replace(ty, Type::Verbatim(TokenStream::new()));
    *ty = Type::Paren(TypeParen {
        paren_token: Default::default(),
        elem: Box::new(inner),
    });
}

/// Returns the bounds of the trait object (or `impl Trait`) that `ty` ends in, through the pointee
/// of a reference or a pointer and the output of a function pointer. A parenthesized type ends the
/// search: nothing in it can be joined by what follows it.
///
/// A `+` after such a type joins the bounds: `Fn() -> &dyn A + B` bounds the trait object by `B`,
/// and rustc rejects it (E0225, "ambiguous `+` in a type").
fn tail_object(ty: &Type) -> Option<&Punctuated<TypeParamBound, Token![+]>> {
    match ty {
        Type::Reference(TypeReference { elem, .. }) | Type::Ptr(TypePtr { elem, .. }) => {
            tail_object(elem)
        }
        Type::BareFn(TypeBareFn {
            output: ReturnType::Type(_, output),
            ..
        }) => tail_object(output),
        Type::TraitObject(TypeTraitObject { bounds, .. })
        | Type::ImplTrait(TypeImplTrait { bounds, .. }) => Some(bounds),
        _ => None,
    }
}

/// The output type of a bound that is the sugar of `Fn`: `R` in `Fn(A) -> R`.
fn bound_output_mut(bound: &mut TypeParamBound) -> Option<&mut Type> {
    match bound {
        // A parenthesized bound `(Fn() -> R)` is closed already.
        TypeParamBound::Trait(TraitBound {
            paren_token: None,
            path,
            ..
        }) => match &mut path.segments.last_mut()?.arguments {
            PathArguments::Parenthesized(ParenthesizedGenericArguments {
                output: ReturnType::Type(_, output),
                ..
            }) => Some(output),
            _ => None,
        },
        _ => None,
    }
}

/// Returns the bounds in the form that rustc reads as they mean, whichever way they were written.
///
/// The output of the sugar of `Fn` (`Fn(A) -> R`) is parsed without a `+`, so a bare trait object
/// there cannot be followed by a bound (`Fn() -> &dyn A + B`), nor have several bounds
/// (`Fn() -> dyn A + B`): they need parentheses. Merging the bounds of a parameter with its where
/// predicates, and sorting them, decides what follows a bound, so this adds the parentheses that
/// the order needs, and removes those that it does not, to make the result independent of how the
/// bounds were written before. The bounds are sorted without the parentheses, so that it is
/// idempotent.
pub(crate) fn canonical_bounds(
    bounds: Punctuated<TypeParamBound, Token![+]>,
) -> Punctuated<TypeParamBound, Token![+]> {
    let mut bounds: Vec<(String, TypeParamBound)> = bounds
        .into_iter()
        .map(|mut bound| {
            // Parentheses that a trait object needs, or that it does not.
            if let Some(output) = bound_output_mut(&mut bound) {
                while let Type::Paren(TypeParen { elem, .. }) = output {
                    if tail_object(elem).is_none() {
                        break;
                    }
                    *output = std::mem::replace(&mut **elem, Type::Verbatim(TokenStream::new()));
                }
            }
            (quote!(#bound).to_string(), bound)
        })
        .collect();
    bounds.sort_by(|a, b| a.0.cmp(&b.0));
    bounds.dedup_by(|a, b| a.0 == b.0);
    let count = bounds.len();
    bounds
        .into_iter()
        .enumerate()
        .map(|(ix, (_, mut bound))| {
            let followed = ix + 1 < count;
            if let Some(output) = bound_output_mut(&mut bound) {
                if tail_object(output).map_or(false, |object| followed || object.len() > 1) {
                    parenthesize(output);
                }
            }
            bound
        })
        .collect()
}

/// Parenthesizes the pointee of a reference or a pointer if it is a trait object with several
/// bounds: `&(dyn A + B)`. A type that replaces `Self` can be one.
pub(crate) fn parenthesize_pointee(mut ty: Type) -> Type {
    if let Type::Reference(TypeReference { elem, .. }) | Type::Ptr(TypePtr { elem, .. }) = &mut ty {
        let several = match &**elem {
            Type::TraitObject(TypeTraitObject { bounds, .. })
            | Type::ImplTrait(TypeImplTrait { bounds, .. }) => bounds.len() > 1,
            _ => false,
        };
        if several {
            parenthesize(elem);
        }
    }
    ty
}

/// Parenthesizes the bounded type of a where predicate that rustc cannot parse as it is.
///
/// A function pointer without an output that is reached through a reference or a pointer
/// (`&fn(A): 'a`) takes the `:` for the start of its output (rustc: "return types are denoted
/// using `->`"). A type that replaces `Self` can be one: `where Self: 'a`.
pub(crate) fn parenthesize_bounded(ty: &mut Type) {
    let mut under_pointer = false;
    let mut current = &*ty;
    let ambiguous = loop {
        match current {
            Type::Reference(TypeReference { elem, .. }) | Type::Ptr(TypePtr { elem, .. }) => {
                under_pointer = true;
                current = elem;
            }
            Type::BareFn(TypeBareFn {
                output: ReturnType::Type(_, output),
                ..
            }) => current = output,
            Type::BareFn(_) => break under_pointer,
            _ => break false,
        }
    };
    if ambiguous {
        parenthesize(ty);
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

/// Prints tokens of a path or a type as one writes them (`<Self as Tr>::A`, `Wrap<Rel<T>>`), not with
/// the spaces that a token stream has. For the messages that suggest what to write. Other
/// tokens may lose a space (`a < b` is `a<b`).
pub(crate) fn tidy(tokens: &TokenStream) -> String {
    tidy_str(&tokens.to_string())
}

/// Like [`tidy()`] for the text of tokens.
pub(crate) fn tidy_str(text: &str) -> String {
    let mut text = text.to_string();
    for (from, to) in [
        (" :: ", "::"),
        (":: ", "::"),
        (" ::", "::"),
        (" <", "<"),
        ("< ", "<"),
        (" >", ">"),
        ("& ", "&"),
        (" ,", ","),
    ] {
        text = text.replace(from, to);
    }
    text
}

/// The path without the segments that name a place in the crate (`crate`, `self`, `super`), for
/// the example of the absolute path in a message. At least one segment stays.
pub(crate) fn without_crate_prefix(path: &Path) -> Path {
    let skip = path
        .segments
        .iter()
        .take_while(|seg| seg.ident == "crate" || seg.ident == "self" || seg.ident == "super")
        .count()
        .min(path.segments.len().saturating_sub(1));
    Path {
        leading_colon: None,
        segments: path.segments.iter().skip(skip).cloned().collect(),
    }
}

/// Joins the ways to fix an error: `a`, `a, or b`, `a, b, or c`.
pub(crate) fn or_list(options: Vec<String>) -> String {
    match options.as_slice() {
        [] => String::new(),
        [only] => only.clone(),
        [init @ .., last] => format!("{}, or {}", init.join(", "), last),
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

pub(crate) fn bound_mentions_self(bound: &TypeParamBound) -> bool {
    let mut finder = SelfFinder::default();
    finder.visit_type_param_bound(bound);
    !finder.found.is_empty()
}

/// Returns `predicate` without the bounds that mention `Self`, or `None` if nothing is left: a
/// predicate on a type that mentions `Self` (`Self: Sized`, `Vec<Self>: Clone`) goes whole.
pub(crate) fn without_self_bounds(predicate: &WherePredicate) -> Option<WherePredicate> {
    match predicate {
        WherePredicate::Type(predicate) => {
            if type_mentions_self(&predicate.bounded_ty) {
                return None;
            }
            let bounds: Punctuated<TypeParamBound, Token![+]> = predicate
                .bounds
                .iter()
                .filter(|bound| !bound_mentions_self(bound))
                .cloned()
                .collect();
            (!bounds.is_empty()).then(|| {
                WherePredicate::Type(PredicateType {
                    bounds,
                    ..predicate.clone()
                })
            })
        }
        other => Some(other.clone()).filter(|predicate| !where_predicate_mentions_self(predicate)),
    }
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

/// Returns the first `Self::X` in the bounds, defaults and where predicates of `generics`.
pub(crate) fn first_associated_self_in_generics(generics: &Generics) -> Option<TokenStream> {
    let mut finder = SelfFinder::default();
    finder.visit_generics(generics);
    finder.associated.into_iter().next()
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
            (ty, _) => parenthesize_pointee(syn::fold::fold_type(folder, ty)),
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

    fn fold_predicate_type(&mut self, i: PredicateType) -> PredicateType {
        let mut i = syn::fold::fold_predicate_type(self, i);
        parenthesize_bounded(&mut i.bounded_ty);
        i
    }

    fn fold_macro(&mut self, i: Macro) -> Macro {
        Macro {
            tokens: self.replace_tokens(i.tokens),
            ..i
        }
    }
}

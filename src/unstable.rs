//! The bounds that the environment of the copy cannot prove for an impl: see
//! [`AnalyzeVisitor::env_cycle()`](crate::analyze::AnalyzeVisitor).
//!
//! rustc normalizes a path through the Repeater (and `Self::A`, which the impl of a trait defines)
//! in a predicate of an item while it builds the environment of the item. It selects an impl, and
//! needs the bounds of that impl, normalized, in the environment, which has the bounds of the item
//! as the copy writes them, not normalized yet. They match only if the copy writes a bound as the
//! impl has it, and rustc does not normalize anything in it. A bound is *unstable* if it names
//!
//! - a leaked type, which the copy writes as a path through the Repeater;
//! - `Self`, if the copy writes the self type of an impl that names a leaked type;
//! - `Self::A`, which the impl of the trait for the host defines;
//! - an associated type that rustc normalizes: `T::Item` if a predicate in scope defines it
//!   (`T: Iterator<Item = ..>`), and `<u8 as Tr>::A`, which an impl defines. `T::Item` of a
//!   generic whose bounds are all traits of the standard library, and none defines it, stays as
//!   it is. A bound of another trait may define it by a supertrait (`trait Tr:
//!   Iterator<Item = u8>`), which the Leaker cannot read, so `T::Item` of a generic with such a
//!   bound counts too.
//!
//! The context of a leaked type leaves out some of these bounds before they are looked at: a bound
//! of a generic `T` that names `T` in the arguments of its path, if another bound of `T` fixes an
//! associated type (`T: Iterator<Item = u8> + Fn(T::Item)`, also behind an alias that hides the
//! projection), is not a bound of the Repeater's impl (see the context module,
//! `left_out_bounds()`), so it is not unstable. The `T::Item` that a bound of another generic names
//! (`U: Fn(T::Item)`), and one that an equation of a method defines, still are.
//!
//! The bounds that the context keeps whole (an entry that names `Self`, or writes out the self type
//! of its scope) are not left out, so a bound of `T` that names `T` in the arguments of its path
//! (`T: Iterator<Item = u8> + Fn(X<T>)`) stays, and `X` may be a type alias of the projection that
//! the equation fixes, which the Leaker cannot tell from a struct: such a bound is unstable too, if
//! an equation of `T` is in scope (`hidden_projection()`).
//!
//! A where predicate that is not a bound of a generic (`Vec<T::Item>: Default`, `u8:
//! Into<T::Item>`) is proved by an impl, so it names no unstable bound as far as a leaked type,
//! `Self` or an associated type of an impl goes; an optional type in it does, as the Leaker cannot
//! tell that an impl proves it (`::std::vec::Vec<T>: Clone` holds only because the predicate says
//! so, F71), and is no entry. It is not so for the `T::Item` that a predicate in scope defines (an
//! enclosing item or a method equates it: `T: Iterator<Item = Rel<T>>`): rustc normalizes it with
//! that predicate, which names the type whose impl needs the predicate, so the impl overflows
//! (E0275) while it proves its own predicate.

use crate::context::{binds_assoc, bound_text, bounded_ident, hiding_bounds};
use crate::util::{ident_name, tidy_str, token_names, unwrap_type_group};
use proc_macro2::TokenStream;
use std::collections::HashSet;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// A bound of the generics of a context that the environment cannot prove.
pub(crate) struct Unstable {
    /// The bound: a parameter with its bounds, or a where predicate.
    pub(crate) bound: TokenStream,
    pub(crate) why: Instability,
}

/// Why the environment cannot prove a bound.
pub(crate) enum Instability {
    /// It names a leaked type. The errors for it are written where the cycle is found (see
    /// [`AnalyzeVisitor::env_cycle()`](crate::analyze::AnalyzeVisitor)).
    Leaked,
    /// It names `Self`, `Self::A` or an associated type that rustc normalizes.
    Named(Why),
}

/// What an unstable bound that names no leaked type names.
pub(crate) enum Why {
    /// It names `Self`, the self type of an impl, and the copy writes that as a path through the
    /// Repeater.
    SelfType,
    /// It names `Self::A`, shown as written.
    SelfAssoc(String),
    /// It names a projection that rustc normalizes, shown as written.
    Projection(String),
    /// A bound of the generic `generic`, shown as written, names `generic` in the arguments of
    /// its path (`Fn(X<T>)`), and an equation of `generic` is in scope: `X` may be a type alias
    /// of the projection that it fixes.
    Hidden { generic: String, bound: String },
}

impl Why {
    /// Tells why the environment cannot prove the bound, as the end of a sentence about it.
    pub(crate) fn explain(&self) -> String {
        match self {
            Why::SelfType => "names `Self`, which the copy writes as the self type of the \
                              impl, a path through the Repeater that rustc has not normalized yet in \
                              the environment, where it must prove the bound (E0277, E0271)"
                .to_string(),
            Why::SelfAssoc(path) => format!(
                "names `{}`, which rustc normalizes to the associated type of the host in the \
                 impl of the Repeater, but not yet in the environment (E0277, E0271)",
                tidy_str(path)
            ),
            Why::Projection(path) => format!(
                "names `{}`, which rustc normalizes in the impl of the Repeater (an impl, a \
                 predicate of the item that equates it, or a supertrait of a bound of its \
                 generic, defines it), but not yet in the environment (E0277, E0271, E0275)",
                tidy_str(path)
            ),
            Why::Hidden { generic, bound } => format!(
                "names `{}` in the arguments of the bound `{}`, whose path may be a type alias of \
                 a projection that another bound of `{}` equates, which the Leaker cannot tell \
                 from a struct: rustc normalizes it in the impl of the Repeater, but not yet in \
                 the environment (E0277)",
                generic,
                tidy_str(bound),
                generic
            ),
        }
    }

    /// How to write the bound, for the help of the error.
    pub(crate) fn advice(&self) -> String {
        match self {
            Why::SelfType => "write the bound without `Self`".to_string(),
            Why::SelfAssoc(path) | Why::Projection(path) => {
                format!("write the bound without `{}`", tidy_str(path))
            }
            Why::Hidden { generic, bound } => format!(
                "write the bound `{}` without a path that names `{}`",
                tidy_str(bound),
                generic
            ),
        }
    }

    /// What the bound names, for the headline of the error.
    pub(crate) fn names(&self) -> String {
        match self {
            Why::SelfType => "`Self`, which is a leaked type".to_string(),
            Why::SelfAssoc(path) | Why::Projection(path) => format!("`{}`", tidy_str(path)),
            Why::Hidden { generic, bound } => format!(
                "`{}` in the arguments of `{}`, which may hide a projection",
                generic,
                tidy_str(bound)
            ),
        }
    }
}

/// Returns the first bound of a type parameter of `generics` (the generics of a context) that may
/// hide a projection of the parameter: one that names it in the arguments of its path, if another
/// bound of it fixes an associated type, or `equated` has it (see [`equated_generics()`]). See
/// `hiding_bounds()`, the rule that the context follows to leave such a bound out.
pub(crate) fn hidden_projection(
    generics: &Generics,
    equated: &HashSet<String>,
) -> Option<Unstable> {
    for param in &generics.params {
        let param = match param {
            GenericParam::Type(param) => param,
            _ => continue,
        };
        let name = ident_name(&param.ident);
        let predicates: Vec<&PredicateType> = generics
            .where_clause
            .iter()
            .flat_map(|where_clause| &where_clause.predicates)
            .filter_map(|predicate| match predicate {
                WherePredicate::Type(predicate)
                    if bounded_ident(&predicate.bounded_ty).as_deref() == Some(&name) =>
                {
                    Some(predicate)
                }
                _ => None,
            })
            .collect();
        let bounds: Vec<&TypeParamBound> = param
            .bounds
            .iter()
            .chain(predicates.iter().flat_map(|predicate| &predicate.bounds))
            .collect();
        let hiding = hiding_bounds(&bounds, &name, equated.contains(&name))
            .into_iter()
            .find(|bound| may_hide(bound, &name));
        let hiding = match hiding {
            Some(hiding) => hiding,
            None => continue,
        };
        let text = bound_text(hiding);
        let has = |bounds: &Punctuated<TypeParamBound, Token![+]>| {
            bounds.iter().any(|bound| bound_text(bound) == text)
        };
        let bound = match predicates.iter().find(|predicate| has(&predicate.bounds)) {
            Some(predicate) if !has(&param.bounds) => quote!(#predicate),
            _ => quote!(#param),
        };
        return Some(Unstable {
            bound,
            why: Instability::Named(Why::Hidden {
                generic: name,
                bound: text,
            }),
        });
    }
    None
}

/// Returns `true` if the path of `bound` has a path in its arguments that names `name` and may be
/// a type alias of a projection of it: not `name` itself (`From<T>`), a type that syntax builds
/// (`&T`, `(T,)`), nor an absolute path of the standard library (`::core::option::Option<T>`),
/// which are no aliases of a projection of `T`. A path inside them still counts.
fn may_hide(bound: &TypeParamBound, name: &str) -> bool {
    struct Hides<'a> {
        name: &'a str,
        found: bool,
    }
    impl<'ast, 'a> Visit<'ast> for Hides<'a> {
        fn visit_type_path(&mut self, i: &'ast TypePath) {
            let mut names = HashSet::new();
            token_names(quote!(#i), &mut names);
            let bare = i.qself.is_none() && i.path.is_ident(self.name);
            let std = i.qself.is_none()
                && i.path.leading_colon.is_some()
                && i.path.segments.first().map_or(false, |seg| {
                    matches!(ident_name(&seg.ident).as_str(), "core" | "std" | "alloc")
                });
            if names.contains(self.name) && !bare && !std {
                self.found = true;
            }
            syn::visit::visit_type_path(self, i)
        }
    }
    let mut hides = Hides { name, found: false };
    if let TypeParamBound::Trait(bound) = bound {
        for seg in &bound.path.segments {
            hides.visit_path_arguments(&seg.arguments);
        }
    }
    hides.found
}

/// The names of the generics that an equation fixes an associated type of (`T: Iterator<Item =
/// u8>`), in the bounds of `params` and in `predicates`.
pub(crate) fn equated_generics<'a>(
    params: impl Iterator<Item = &'a GenericParam>,
    predicates: impl Iterator<Item = &'a WherePredicate>,
) -> HashSet<String> {
    let mut names = HashSet::new();
    for param in params {
        if let GenericParam::Type(param) = param {
            if param.bounds.iter().any(binds_assoc) {
                names.insert(ident_name(&param.ident));
            }
        }
    }
    for predicate in predicates {
        if let WherePredicate::Type(predicate) = predicate {
            if predicate.bounds.iter().any(binds_assoc) {
                names.extend(bounded_ident(&predicate.bounded_ty));
            }
        }
    }
    names
}

/// Finds the first associated type in a node that rustc normalizes (see the module).
pub(crate) struct Projections<'a, F> {
    /// Whether a name is a type parameter in scope.
    is_param: F,
    /// The `(bounded type, name)` of the equations `T: Tr<Name = ..>` in scope, and
    /// `(bounded type, ANY_NAME)` for a bound that may define any (see [`bindings()`]).
    bindings: &'a [(String, String)],
    /// Only the associated types that an equation in scope defines count: not `Self::A`, and not
    /// the projection of a type that an impl defines (see [`Projections::equated()`]).
    equated_only: bool,
    pub(crate) found: Option<Why>,
}

impl<'a, F: Fn(&str) -> bool> Projections<'a, F> {
    pub(crate) fn new(is_param: F, bindings: &'a [(String, String)]) -> Self {
        Self {
            is_param,
            bindings,
            equated_only: false,
            found: None,
        }
    }

    /// Like `new()`, for a type that rustc proves by an impl (the bounded type of a where
    /// predicate, and the types of its bounds, if it is not a bound of a generic): the impl
    /// normalizes `Self::A` and the projection of a type that an impl defines to what it needs,
    /// and the projection of a generic only by the equation that a predicate in scope gives.
    pub(crate) fn equated(is_param: F, bindings: &'a [(String, String)]) -> Self {
        Self {
            equated_only: true,
            ..Self::new(is_param, bindings)
        }
    }

    fn classify(&self, i: &TypePath) -> Option<Why> {
        let shown = || quote!(#i).to_string();
        let (root, name) = match &i.qself {
            None => {
                let segments = &i.path.segments;
                let first = segments.first()?;
                if i.path.leading_colon.is_some()
                    || segments.len() < 2
                    || !first.arguments.is_none()
                {
                    return None;
                }
                let name = ident_name(&segments[1].ident);
                if first.ident == "Self" {
                    return (!self.equated_only).then(|| Why::SelfAssoc(shown()));
                }
                if !(self.is_param)(&ident_name(&first.ident)) {
                    return None;
                }
                (ident_name(&first.ident), name)
            }
            Some(qself) => {
                let name = ident_name(&i.path.segments.iter().nth(qself.position)?.ident);
                let ty = unwrap_type_group(&qself.ty);
                if is_bare_self(ty) {
                    return (!self.equated_only).then(|| Why::SelfAssoc(shown()));
                }
                // The associated type of a type that impls define, not of a generic.
                if !self.rooted_in_param(ty) {
                    return (!self.equated_only).then(|| Why::Projection(shown()));
                }
                (quote!(#ty).to_string(), name)
            }
        };
        self.bindings
            .iter()
            .any(|(bounded, bound)| *bounded == root && (*bound == name || *bound == ANY_NAME))
            .then(|| Why::Projection(shown()))
    }

    /// Returns `true` if `ty` is a generic, or an associated type path that starts with one.
    fn rooted_in_param(&self, ty: &Type) -> bool {
        match unwrap_type_group(ty) {
            Type::Path(TypePath { qself: None, path }) => {
                path.leading_colon.is_none()
                    && path
                        .segments
                        .first()
                        .map_or(false, |seg| (self.is_param)(&ident_name(&seg.ident)))
            }
            Type::Path(TypePath {
                qself: Some(qself), ..
            }) => self.rooted_in_param(&qself.ty),
            _ => false,
        }
    }
}

impl<'ast, 'a, F: Fn(&str) -> bool> Visit<'ast> for Projections<'a, F> {
    fn visit_type_path(&mut self, i: &'ast TypePath) {
        if self.found.is_none() {
            self.found = self.classify(i);
        }
        syn::visit::visit_type_path(self, i)
    }
}

fn is_bare_self(ty: &Type) -> bool {
    matches!(ty, Type::Path(TypePath { qself: None, path }) if path.is_ident("Self"))
}

/// The name of a bound that may define any associated type of what it bounds: the bound of a trait
/// that the Leaker cannot read, whose supertrait may equate an associated type of another trait
/// (`trait Tr: Iterator<Item = u8>`). It is not an identifier, so it is no name.
const ANY_NAME: &str = "*";

/// Returns the `(bounded type, name)` of the equations of associated types in the bounds of
/// `params` and in `predicates`: `T: Iterator<Item = ..>` is `("T", "Item")`, and `F: Fn(A) -> B` is
/// `("F", "Output")`. A bound of a trait that is not one of the standard library (`T: ::p::Tr`) is
/// `("T", ANY_NAME)`.
pub(crate) fn bindings<'a>(
    params: impl Iterator<Item = &'a GenericParam>,
    predicates: impl Iterator<Item = &'a WherePredicate>,
) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for param in params {
        if let GenericParam::Type(param) = param {
            let bounded = ident_name(&param.ident);
            for bound in &param.bounds {
                found.extend(
                    names_of(bound)
                        .into_iter()
                        .map(|name| (bounded.clone(), name)),
                );
            }
        }
    }
    for predicate in predicates {
        if let WherePredicate::Type(predicate) = predicate {
            let ty = unwrap_type_group(&predicate.bounded_ty);
            let bounded = quote!(#ty).to_string();
            for bound in &predicate.bounds {
                found.extend(
                    names_of(bound)
                        .into_iter()
                        .map(|name| (bounded.clone(), name)),
                );
            }
        }
    }
    found
}

/// The associated types that a trait bound equates, and `ANY_NAME` if a supertrait of the trait may
/// equate any.
fn names_of(bound: &TypeParamBound) -> Vec<String> {
    let (path, seg) = match bound {
        TypeParamBound::Trait(bound) => match bound.path.segments.last() {
            Some(seg) => (&bound.path, seg),
            None => return Vec::new(),
        },
        _ => return Vec::new(),
    };
    let mut names: Vec<String> = match &seg.arguments {
        PathArguments::AngleBracketed(args) => args
            .args
            .iter()
            .filter_map(|arg| match arg {
                GenericArgument::AssocType(assoc) => Some(ident_name(&assoc.ident)),
                _ => None,
            })
            .collect(),
        PathArguments::Parenthesized(_) => vec!["Output".to_string()],
        PathArguments::None => Vec::new(),
    };
    if !is_std_trait(path) {
        names.push(ANY_NAME.to_string());
    }
    names
}

/// The traits of the standard library that a bare name means (a bound `T: Iterator`): none of them
/// has a supertrait that equates an associated type. A trait of the analyzed code with the name of
/// one is taken for it.
const STD_TRAITS: &[&str] = &[
    "Sized",
    "Send",
    "Sync",
    "Unpin",
    "Copy",
    "Clone",
    "Drop",
    "Eq",
    "PartialEq",
    "Ord",
    "PartialOrd",
    "Default",
    "Debug",
    "Display",
    "Hash",
    "AsRef",
    "AsMut",
    "Into",
    "From",
    "TryFrom",
    "TryInto",
    "ToString",
    "ToOwned",
    "Borrow",
    "BorrowMut",
    "Fn",
    "FnMut",
    "FnOnce",
    "Iterator",
    "IntoIterator",
    "DoubleEndedIterator",
    "ExactSizeIterator",
    "FusedIterator",
    "Extend",
    "FromIterator",
    "Deref",
    "DerefMut",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Rem",
    "Neg",
    "Not",
    "Index",
    "IndexMut",
    "FromStr",
    "Future",
    "IntoFuture",
    "Any",
];

/// Returns `true` if `path` is a trait of the standard library: an absolute path in `::core`,
/// `::std` or `::alloc`, or a bare name of `STD_TRAITS`. Another path may be a trait of the analyzed
/// code, or of a crate that the Leaker cannot read.
pub(crate) fn is_std_trait(path: &Path) -> bool {
    let first = match path.segments.first() {
        Some(first) => ident_name(&first.ident),
        None => return false,
    };
    if path.leading_colon.is_some() {
        return matches!(first.as_str(), "core" | "std" | "alloc");
    }
    path.segments.len() == 1 && STD_TRAITS.contains(&first.as_str())
}

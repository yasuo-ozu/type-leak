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

use crate::util::{ident_name, unwrap_type_group};
use proc_macro2::TokenStream;
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// A bound of the generics of a context that the environment cannot prove.
pub(crate) struct Unstable {
    /// The bound: a parameter with its bounds, or a where predicate.
    pub(crate) bound: TokenStream,
    pub(crate) why: Why,
}

impl Unstable {
    /// Tells why the environment cannot prove the bound, as the end of a sentence about it.
    pub(crate) fn explain(&self) -> String {
        match &self.why {
            Why::Leaked => "names a leaked type, which the copy writes as a path through the \
                            Repeater, and rustc cannot normalize it in the environment (E0277, E0275)"
                .to_string(),
            Why::SelfType => "names `Self`, which the copy writes as the self type of the \
                              impl, a path through the Repeater that rustc has not normalized yet in \
                              the environment, where it must prove the bound (E0277, E0271)"
                .to_string(),
            Why::SelfAssoc(path) => format!(
                "names `{}`, which rustc normalizes to the associated type of the host in the \
                 impl of the Repeater, but not yet in the environment (E0277, E0271)",
                path
            ),
            Why::Projection(path) => format!(
                "names `{}`, which rustc normalizes in the impl of the Repeater (an impl, a \
                 predicate of the item that equates it, or a supertrait of a bound of its \
                 generic, defines it), but not yet in the environment (E0277, E0271, E0275)",
                path
            ),
        }
    }
}

pub(crate) enum Why {
    /// It names a leaked type.
    Leaked,
    /// It names `Self`, the self type of an impl, and the copy writes that as a path through the
    /// Repeater.
    SelfType,
    /// It names `Self::A`, shown as written.
    SelfAssoc(String),
    /// It names a projection that rustc normalizes, shown as written.
    Projection(String),
}

impl Why {
    /// How to write the bound, for the note of the error.
    pub(crate) fn advice(&self) -> String {
        match self {
            Why::Leaked => "Write the leaked type of the bound with an absolute path".to_string(),
            Why::SelfType => "Write the bound without `Self`".to_string(),
            Why::SelfAssoc(path) | Why::Projection(path) => {
                format!("Write the bound without `{}`", path)
            }
        }
    }

    /// What the bound names, for the headline of the error.
    pub(crate) fn names(&self) -> String {
        match self {
            Why::Leaked => "a leaked type".to_string(),
            Why::SelfType => "`Self`, which is a leaked type".to_string(),
            Why::SelfAssoc(path) | Why::Projection(path) => format!("`{}`", path),
        }
    }
}

/// Finds the first associated type in a node that rustc normalizes (see the module).
pub(crate) struct Projections<'a, F> {
    /// Whether a name is a type parameter in scope.
    is_param: F,
    /// The `(bounded type, name)` of the equations `T: Tr<Name = ..>` in scope, and
    /// `(bounded type, ANY_NAME)` for a bound that may define any (see [`bindings()`]).
    bindings: &'a [(String, String)],
    pub(crate) found: Option<Why>,
}

impl<'a, F: Fn(&str) -> bool> Projections<'a, F> {
    pub(crate) fn new(is_param: F, bindings: &'a [(String, String)]) -> Self {
        Self {
            is_param,
            bindings,
            found: None,
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
                    return Some(Why::SelfAssoc(shown()));
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
                    return Some(Why::SelfAssoc(shown()));
                }
                // The associated type of a type that impls define, not of a generic.
                if !self.rooted_in_param(ty) {
                    return Some(Why::Projection(shown()));
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
fn is_std_trait(path: &Path) -> bool {
    let first = match path.segments.first() {
        Some(first) => ident_name(&first.ident),
        None => return false,
    };
    if path.leading_colon.is_some() {
        return matches!(first.as_str(), "core" | "std" | "alloc");
    }
    path.segments.len() == 1 && STD_TRAITS.contains(&first.as_str())
}

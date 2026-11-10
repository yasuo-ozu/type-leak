//! The bounds that the environment of the copy cannot prove for an impl (see
//! [`AnalyzeVisitor::env_cycle()`](crate::analyze::AnalyzeVisitor)).
//!
//! rustc normalizes a path through the Repeater (or `Self::A`) in a predicate of an item while it
//! builds the item's environment: the impl it selects needs its bounds there, but the environment
//! has them only as the copy writes them, not normalized yet. A bound is *unstable* if it names a
//! leaked type, `Self` that the copy writes as a leaked self type, `Self::A`, or an associated type
//! that rustc normalizes (`<u8 as Tr>::A`, or `T::Item` that a predicate in scope or a bound of a
//! trait outside the standard library may define). The bounds that the context leaves out are not
//! looked at; one that it keeps whole may hide a projection (`hidden_projection()`). A where
//! predicate that is no bound of a generic is proved by an impl: only an optional type in it, or a
//! `T::Item` that a predicate in scope equates (E0275), makes it unstable.

use crate::util::{
    bounded_ident, ident_name, is_bare_self, names_in, tidy_str, token_names, unwrap_type_group,
};
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

/// Returns the first bound of a type parameter of the context's `generics` that may hide a
/// projection of it (see [`hiding_bounds()`]; `equated` is from [`equated_generics()`]).
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

/// Returns `true` if the arguments of `bound` have a path that names `name` and may alias a
/// projection of it: not `name` itself (`From<T>`), `&T`, `(T,)` or a path of the standard library
/// (`::core::option::Option<T>`), though a path inside them counts.
fn may_hide(bound: &TypeParamBound, name: &str) -> bool {
    struct Hides<'a> {
        name: &'a str,
        found: bool,
    }
    impl<'ast, 'a> Visit<'ast> for Hides<'a> {
        fn visit_type_path(&mut self, i: &'ast TypePath) {
            let names = names_in(i);
            let bare = i.qself.is_none() && i.path.is_ident(self.name);
            let std = i.qself.is_none() && is_std_root(&i.path);
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
    /// the projection of a type that an impl defines (see [`Projections::equated_only()`]).
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

    /// Like `new()`, for a where predicate that is no bound of a generic, which an impl proves:
    /// only a projection of a generic that an equation in scope defines counts.
    pub(crate) fn equated_only(is_param: F, bindings: &'a [(String, String)]) -> Self {
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

/// The projection in `predicate` that rustc normalizes for the impl that needs it (see
/// [`Projections`]), where `on_environment` says that the environment must prove it.
pub(crate) fn predicate_projection(
    predicate: &WherePredicate,
    on_environment: bool,
    is_param: &impl Fn(&str) -> bool,
    bindings: &[(String, String)],
) -> Option<Why> {
    // What the predicate bounds is not a bound of it: only the bounds are normalized for the impl
    // that needs them (`Self::A: Clone` is proved by an impl).
    let mut finder = Projections::new(is_param, bindings);
    if let (true, WherePredicate::Type(predicate)) = (on_environment, predicate) {
        for bound in &predicate.bounds {
            finder.visit_type_param_bound(bound);
        }
    }
    if finder.found.is_some() {
        return finder.found;
    }
    // An impl proves the predicate, and normalizes the `Item` that a predicate in scope equates,
    // wherever it is named: the bounded type, and the bounds of a predicate that is not on a
    // generic.
    let mut finder = Projections::equated_only(is_param, bindings);
    finder.visit_where_predicate(predicate);
    finder.found
}

/// The name of a bound that may define any associated type of what it bounds: the bound of a trait
/// that the Leaker cannot read, whose supertrait may equate an associated type of another trait
/// (`trait Tr: Iterator<Item = u8>`). It is not an identifier, so it is no name.
const ANY_NAME: &str = "*";

/// The `(bounded type, name)` of each equation in the bounds of `params` and `predicates`:
/// `("T", "Item")` for `T: Iterator<Item = ..>`, `("F", "Output")` for `F: Fn(A) -> B`, and
/// `("T", ANY_NAME)` for `T: ::p::Tr`, a trait outside the standard library.
pub(crate) fn bindings<'a>(
    params: impl Iterator<Item = &'a GenericParam>,
    predicates: impl Iterator<Item = &'a WherePredicate>,
) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for (bounded, bound) in trait_bounds(params, predicates) {
        found.extend(
            names_of(bound)
                .into_iter()
                .map(|name| (bounded.clone(), name)),
        );
    }
    found
}

/// The trait bounds of the type parameters `params` and of the where predicates `predicates`, in
/// order, each with what it bounds: the name of the parameter, or the bounded type as written.
pub(crate) fn trait_bounds<'a>(
    params: impl Iterator<Item = &'a GenericParam>,
    predicates: impl Iterator<Item = &'a WherePredicate>,
) -> impl Iterator<Item = (String, &'a TraitBound)> {
    let params = params.filter_map(|param| match param {
        GenericParam::Type(param) => Some((ident_name(&param.ident), &param.bounds)),
        _ => None,
    });
    let predicates = predicates.filter_map(|predicate| match predicate {
        WherePredicate::Type(predicate) => {
            let ty = unwrap_type_group(&predicate.bounded_ty);
            Some((quote!(#ty).to_string(), &predicate.bounds))
        }
        _ => None,
    });
    params.chain(predicates).flat_map(|(bounded, bounds)| {
        bounds.iter().filter_map(move |bound| match bound {
            TypeParamBound::Trait(bound) => Some((bounded.clone(), bound)),
            _ => None,
        })
    })
}

/// The associated types that a trait bound equates, and `ANY_NAME` if a supertrait of the trait may
/// equate any.
fn names_of(bound: &TraitBound) -> Vec<String> {
    let seg = match bound.path.segments.last() {
        Some(seg) => seg,
        None => return Vec::new(),
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
    if !is_std_trait(&bound.path) {
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
    if path.leading_colon.is_some() {
        return is_std_root(path);
    }
    path.segments.len() == 1 && STD_TRAITS.contains(&ident_name(&path.segments[0].ident).as_str())
}

/// Returns `true` if `path` is an absolute path in `::core`, `::std` or `::alloc`.
fn is_std_root(path: &Path) -> bool {
    path.leading_colon.is_some()
        && path.segments.first().map_or(false, |seg| {
            matches!(ident_name(&seg.ident).as_str(), "core" | "std" | "alloc")
        })
}

pub(crate) fn bound_text(bound: &TypeParamBound) -> String {
    quote!(#bound).to_string()
}

/// Returns `true` if `bound` is a trait bound with an equation of an associated type
/// (`Iterator<Item = u8>`).
fn binds_assoc(bound: &TypeParamBound) -> bool {
    match bound {
        TypeParamBound::Trait(bound) => matches!(
            bound.path.segments.last().map(|seg| &seg.arguments),
            Some(PathArguments::AngleBracketed(args))
                if args.args.iter().any(|arg| matches!(arg, GenericArgument::AssocType(_)))
        ),
        _ => false,
    }
}

/// Returns `true` if `bound` is a trait bound with the name `name` in the arguments of its path
/// (`Fn(Wrap<T>)`, `Tr<<T as Iterator>::Item>`, `PartialEq<Wrap<T>>`).
fn arguments_mention(bound: &TypeParamBound, name: &str) -> bool {
    match bound {
        TypeParamBound::Trait(bound) => {
            let mut names = HashSet::new();
            for seg in &bound.path.segments {
                let arguments = &seg.arguments;
                token_names(quote!(#arguments), &mut names);
            }
            names.contains(name)
        }
        _ => false,
    }
}

/// Returns `true` if the trait of `bound` is not one of the standard library, so a supertrait of
/// it may equate an associated type that the Leaker cannot read (`trait Tr: Iterator<Item = u8>`).
fn may_define_assoc(bound: &TypeParamBound) -> bool {
    match bound {
        TypeParamBound::Trait(bound) => !is_std_trait(&bound.path),
        _ => false,
    }
}

/// Returns those of `bounds` (all of the type parameter `name`) that are no equation and name
/// `name` in their path's arguments (`T: Fn(X<T>)`), if another fixes an associated type (an
/// equation, or a non-std trait that does not name `name`) or `equated` says one is in scope.
pub(crate) fn hiding_bounds<'b>(
    bounds: &[&'b TypeParamBound],
    name: &str,
    equated: bool,
) -> Vec<&'b TypeParamBound> {
    let fixes = equated
        || bounds.iter().any(|bound| {
            binds_assoc(bound) || (may_define_assoc(bound) && !arguments_mention(bound, name))
        });
    if !fixes {
        return Vec::new();
    }
    bounds
        .iter()
        .copied()
        .filter(|bound| !binds_assoc(bound) && arguments_mention(bound, name))
        .collect()
}

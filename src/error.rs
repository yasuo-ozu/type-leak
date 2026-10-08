//! The error of the analysis by the [`Leaker`](crate::Leaker).

use crate::analysis::CheckResult;
use proc_macro2::{Span, TokenStream};
use syn::spanned::Spanned;
use template_quote::ToTokens;

/// The error returned when an item cannot be analyzed by the [`Leaker`](crate::Leaker).
///
/// It has one error per offending path or type. Each has notes: a fix if possible (e.g. "use
/// `::std::vec::Vec` instead of `Vec`"), the [`CheckResult::impossible`] causes that stop it from
/// being leaked as a whole, and the other paths in it that depend on the context (the rest of
/// [`CheckResult::must`]), which are reported once the first is fixed.
/// [`syn::Error`] and [`ToTokens`] render the notes as `note`s.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum NotInternableError {
    /// Paths that depend on the context but cannot be leaked, so they must be absolute.
    ///
    /// Examples: `MyType` in `MyType<'_>`, a trait in a bound, a value in a const expression, and
    /// `Rel` in `F: for<'x> Fn(Rel<'x>) -> &'x u8` ([Binder lifetimes](crate#binder-lifetimes)).
    /// Also a type with a generic whose bound names the `Self` of a trait (or `Self::X`), where
    /// `Self` is something else, so the bound cannot be written ([Self rules](crate#self-rules)).
    /// That is not about the path of the type, so its headline is "the bound of `T` mentions a
    /// `Self` which has no name here" ([`CheckResult::headline`]), with a note on why, at the
    /// type. Each conflict is reported at the first of its [`CheckResult::must`], with
    /// [`CheckResult::is_own`] `true`. For a trait bound, a value or the type of a const
    /// parameter, [`CheckResult::impossible`] is that bound, value or parameter. Never empty.
    RelativePath(Vec<CheckResult>),
    /// `Self` where no scope gives it a meaning. Never empty.
    ///
    /// That is directly in a module, or in generics that no struct, enum, union, impl or trait
    /// encloses (`Leaker::enter_mod()`, `Leaker::enter_generics()`). It is also an error in the
    /// bounds, where predicates and defaults of those generics. See
    /// [Self rules](crate#self-rules).
    /// Each item is the tokens of a mention (`Self`, `Self::Item`, or a type macro that names it).
    /// Its span is where the error is reported, with the message "`Self` has no meaning here".
    /// If a scope finds both kinds of errors, only [`NotInternableError::RelativePath`] is
    /// returned. This kind is found after those are fixed.
    SelfWithoutMeaning(Vec<TokenStream>),
}

struct Report {
    tokens: TokenStream,
    message: String,
    notes: Vec<(Span, String)>,
}

pub(crate) const SELF_NOTE: &str =
    "`Self` has no meaning here: it is declared only by a struct, an enum, a \
                         union, an impl or a trait, not by a module or generics";

impl NotInternableError {
    fn reports(&self) -> Vec<Report> {
        let conflicts = match self {
            NotInternableError::RelativePath(conflicts) => conflicts,
            NotInternableError::SelfWithoutMeaning(selves) => {
                return selves
                    .iter()
                    .map(|tokens| Report {
                        tokens: tokens.clone(),
                        message: "`Self` has no meaning here".to_string(),
                        notes: vec![(tokens.span(), SELF_NOTE.to_string())],
                    })
                    .collect();
            }
        };
        conflicts
            .iter()
            .filter_map(|conflict| {
                let (tokens, others) = conflict.must.split_first()?;
                let mut notes: Vec<_> = match &conflict.note {
                    // The conflict is not about the path.
                    Some(note) => Some((tokens.span(), note.clone())),
                    None => absolute_path_note(tokens),
                }
                .into_iter()
                .collect();
                // A cause that is the path itself (e.g. a trait bound) needs no note.
                for cause in conflict
                    .impossible
                    .iter()
                    .filter(|cause| cause.to_string() != tokens.to_string())
                {
                    notes.push((
                        cause.span(),
                        format!("it cannot be leaked as a whole because of `{}`", cause),
                    ));
                    if is_default_lifetime_object(cause) {
                        notes.push((
                            cause.span(),
                            "the lifetime of the trait object defaults to that of the enclosing \
                             reference or path; name it (e.g. `&'a dyn Trait` or `dyn Trait + 'a`)"
                                .to_string(),
                        ));
                    }
                }
                notes.extend(others.iter().map(|other| {
                    (
                        other.span(),
                        format!("`{}` also depends on the type context", other),
                    )
                }));
                Some(Report {
                    tokens: tokens.clone(),
                    message: conflict
                        .headline
                        .clone()
                        .unwrap_or_else(|| format!("use absolute path: '{}'", tokens)),
                    notes,
                })
            })
            .collect()
    }

    /// Returns the span of the first error.
    pub fn span(&self) -> Span {
        self.reports()
            .first()
            .map_or_else(Span::call_site, |report| report.tokens.span())
    }

    /// Returns the note of the first error, which suggests how to fix it.
    ///
    /// For `RelativePath` it is the absolute path to use instead of the first segment, if that is a
    /// prelude item or a primitive type. For `SelfWithoutMeaning` it tells which items declare
    /// `Self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait MyTrait: Clone {});
    /// let err = Leaker::new().intern_trait(&item, true).err().unwrap();
    /// assert_eq!(err.to_string(), "use absolute path: 'Clone'");
    /// assert_eq!(
    ///     err.note().as_deref(),
    ///     Some("use `::core::clone::Clone` instead of `Clone`")
    /// );
    /// ```
    pub fn note(&self) -> Option<String> {
        let report = self.reports().into_iter().next()?;
        report.notes.into_iter().next().map(|(_, note)| note)
    }
}

/// A trait object without a lifetime bound is a cause only under an elided lifetime, or where the
/// path cannot be leaked (see [Trait objects](crate#trait-objects)).
fn is_default_lifetime_object(cause: &TokenStream) -> bool {
    match syn::parse2::<syn::Type>(cause.clone()) {
        Ok(syn::Type::TraitObject(object)) => !object
            .bounds
            .iter()
            .any(|bound| matches!(bound, syn::TypeParamBound::Lifetime(_))),
        _ => false,
    }
}

/// Parses the offending `tokens`, a path or a trait bound (e.g. `for<'a> Fn(&'a u8)`).
fn offending_path(tokens: &TokenStream) -> Option<syn::Path> {
    syn::parse2::<syn::Path>(tokens.clone()).ok().or_else(|| {
        match syn::parse2(tokens.clone()).ok()? {
            syn::TypeParamBound::Trait(bound) => Some(bound.path),
            _ => None,
        }
    })
}

fn absolute_path_note(tokens: &TokenStream) -> Option<(Span, String)> {
    let path = offending_path(tokens)?;
    if path.leading_colon.is_some() {
        return None;
    }
    let ident = &path.segments.iter().next()?.ident;
    let name = ident.to_string();
    ABSOLUTE_PATHS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(name, absolute)| {
            (
                ident.span(),
                format!("use `{}` instead of `{}`", absolute, name),
            )
        })
}

/// The messages of the errors, one per line.
impl std::fmt::Display for NotInternableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (n, report) in self.reports().iter().enumerate() {
            if n > 0 {
                writeln!(f)?;
            }
            f.write_str(&report.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for NotInternableError {}

/// Converts into a [`syn::Error`], with one error per offending path combined by
/// [`syn::Error::combine()`].
///
/// Stable Rust has no sub-diagnostics for proc macros, so each note is appended to the message.
impl From<NotInternableError> for syn::Error {
    fn from(err: NotInternableError) -> Self {
        err.to_syn_error()
    }
}

impl NotInternableError {
    fn to_syn_error(&self) -> syn::Error {
        self.reports()
            .into_iter()
            .map(|report| {
                let mut message = report.message;
                if !report.notes.is_empty() {
                    message.push('\n');
                    for (_, note) in report.notes {
                        message.push_str(&format!("\n  = note: {}", note));
                    }
                    message.push('\n');
                }
                syn::Error::new(report.tokens.span(), message)
            })
            .reduce(|mut combined, error| {
                combined.combine(error);
                combined
            })
            .unwrap_or_else(|| syn::Error::new(Span::call_site(), "not internable"))
    }
}

/// Emits the errors as `compile_error!` invocations, with the notes rendered as for [`syn::Error`].
/// A proc macro can report them with `Err(e) => e.into_token_stream().into()`.
///
/// On nightly, when `proc_macro::Diagnostic` is available inside a proc macro, the errors are
/// emitted as diagnostics instead, with notes at their own spans (e.g. the first segment of the
/// offending path), and no tokens are produced.
impl ToTokens for NotInternableError {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        // Only compiled on nightly, so the MSRV does not apply.
        #[cfg(type_leak_diagnostic)]
        #[allow(clippy::incompatible_msrv)]
        if proc_macro::is_available() {
            self.emit_diagnostics();
            return;
        }
        tokens.extend(self.to_syn_error().to_compile_error());
    }
}

#[cfg(type_leak_diagnostic)]
impl NotInternableError {
    fn emit_diagnostics(&self) {
        /// Converts a span into that of the compiler (only valid inside a proc macro).
        fn unwrap_span(span: Span) -> proc_macro::Span {
            let ident = proc_macro2::TokenTree::Ident(proc_macro2::Ident::new("x", span));
            proc_macro::TokenStream::from(TokenStream::from(ident))
                .into_iter()
                .next()
                .unwrap()
                .span()
        }
        for report in self.reports() {
            let mut diagnostic = proc_macro::Diagnostic::spanned(
                unwrap_span(report.tokens.span()),
                proc_macro::Level::Error,
                report.message,
            );
            for (span, note) in report.notes {
                diagnostic = diagnostic.span_note(unwrap_span(span), note);
            }
            diagnostic.emit();
        }
    }
}

/// Absolute paths of the standard prelude items and the primitive types.
pub(crate) const ABSOLUTE_PATHS: &[(&str, &str)] = &[
    ("Box", "::std::boxed::Box"),
    ("String", "::std::string::String"),
    ("Vec", "::std::vec::Vec"),
    ("Option", "::core::option::Option"),
    ("Result", "::core::result::Result"),
    ("Copy", "::core::marker::Copy"),
    ("Send", "::core::marker::Send"),
    ("Sized", "::core::marker::Sized"),
    ("Sync", "::core::marker::Sync"),
    ("Unpin", "::core::marker::Unpin"),
    ("Drop", "::core::ops::Drop"),
    ("Fn", "::core::ops::Fn"),
    ("FnMut", "::core::ops::FnMut"),
    ("FnOnce", "::core::ops::FnOnce"),
    ("Clone", "::core::clone::Clone"),
    ("PartialEq", "::core::cmp::PartialEq"),
    ("PartialOrd", "::core::cmp::PartialOrd"),
    ("Eq", "::core::cmp::Eq"),
    ("Ord", "::core::cmp::Ord"),
    ("AsRef", "::core::convert::AsRef"),
    ("AsMut", "::core::convert::AsMut"),
    ("Into", "::core::convert::Into"),
    ("From", "::core::convert::From"),
    ("Default", "::core::default::Default"),
    ("Iterator", "::core::iter::Iterator"),
    ("IntoIterator", "::core::iter::IntoIterator"),
    ("Extend", "::core::iter::Extend"),
    ("FromIterator", "::core::iter::FromIterator"),
    ("ToOwned", "::std::borrow::ToOwned"),
    ("ToString", "::std::string::ToString"),
    ("drop", "::core::mem::drop"),
    // In the prelude of the 2021 edition.
    ("TryFrom", "::core::convert::TryFrom"),
    ("TryInto", "::core::convert::TryInto"),
    ("Some", "::core::option::Option::Some"),
    ("None", "::core::option::Option::None"),
    ("Ok", "::core::result::Result::Ok"),
    ("Err", "::core::result::Result::Err"),
    ("bool", "::core::primitive::bool"),
    ("char", "::core::primitive::char"),
    ("str", "::core::primitive::str"),
    ("u8", "::core::primitive::u8"),
    ("u16", "::core::primitive::u16"),
    ("u32", "::core::primitive::u32"),
    ("u64", "::core::primitive::u64"),
    ("u128", "::core::primitive::u128"),
    ("usize", "::core::primitive::usize"),
    ("i8", "::core::primitive::i8"),
    ("i16", "::core::primitive::i16"),
    ("i32", "::core::primitive::i32"),
    ("i64", "::core::primitive::i64"),
    ("i128", "::core::primitive::i128"),
    ("isize", "::core::primitive::isize"),
    ("f32", "::core::primitive::f32"),
    ("f64", "::core::primitive::f64"),
];

//! The error of the analysis by the [`Leaker`](crate::Leaker).

use crate::check::CheckResult;
use proc_macro2::{Span, TokenStream};
use syn::spanned::Spanned;
use template_quote::ToTokens;

/// Error indicating that an item cannot be analyzed by the [`Leaker`](crate::Leaker).
///
/// It is reported as one error per offending path (or type), each with notes:
///
/// - how to fix it if possible, e.g. "use `::std::vec::Vec` instead of `Vec`" when the offending
///   path starts with an item of the standard prelude or a primitive type,
/// - why it cannot be leaked as a whole (the [`CheckResult::impossible`] causes, e.g. `'_`), and
/// - the other paths inside it which depend on the type context (the rest of
///   [`CheckResult::must`]), which are reported after it is fixed.
///
/// Converting the error into [`syn::Error`] (or into tokens with [`ToTokens`]) renders the notes
/// as `note`s of the diagnostics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum NotInternableError {
    /// Paths which depend on the type context, but cannot be leaked (e.g. `MyType` in
    /// `MyType<'_>`, a trait in a bound, or a value in a const expression), so they should be
    /// absolute. Each conflict is reported at the first of its [`CheckResult::must`] (the
    /// offending path), with [`CheckResult::is_own`] being `true`. For a trait bound, a value
    /// or the type of a const parameter, [`CheckResult::impossible`] is the bound, the value or
    /// the const parameter. Never empty.
    RelativePath(Vec<CheckResult>),
    /// The type is used with different method-local generics (or with and without them) in
    /// different places, so a Referrer cannot tell which one a copy refers to. (The Referrer finds
    /// leaked types by the type alone; see
    /// [`Referrer::local_generics()`](crate::Referrer::local_generics()).)
    ConflictingLocalGenerics(TokenStream),
}

/// One error to report: the offending tokens, the message and the notes.
struct Report {
    tokens: TokenStream,
    message: String,
    /// The notes, with the spans they refer to.
    notes: Vec<(Span, String)>,
}

impl NotInternableError {
    /// Returns the errors to report, one per offending path (or type).
    fn reports(&self) -> Vec<Report> {
        match self {
            NotInternableError::RelativePath(conflicts) => conflicts
                .iter()
                .filter_map(|conflict| {
                    let (tokens, others) = conflict.must.split_first()?;
                    let mut notes: Vec<_> = absolute_path_note(tokens).into_iter().collect();
                    // Why it cannot be leaked as a whole (unless it is the path itself, e.g. a
                    // trait bound).
                    notes.extend(
                        conflict
                            .impossible
                            .iter()
                            .filter(|cause| cause.to_string() != tokens.to_string())
                            .map(|cause| {
                                (
                                    cause.span(),
                                    format!(
                                        "it cannot be leaked as a whole because of `{}`",
                                        cause
                                    ),
                                )
                            }),
                    );
                    // The other paths inside it, which are found after this one is fixed.
                    notes.extend(others.iter().map(|other| {
                        (
                            other.span(),
                            format!("`{}` also depends on the type context", other),
                        )
                    }));
                    Some(Report {
                        tokens: tokens.clone(),
                        message: format!("use absolute path: '{}'", tokens),
                        notes,
                    })
                })
                .collect(),
            NotInternableError::ConflictingLocalGenerics(tokens) => vec![Report {
                tokens: tokens.clone(),
                message: format!(
                    "'{}' is used with different method-local generics, which cannot be leaked",
                    tokens
                ),
                notes: vec![(
                    tokens.span(),
                    "use an absolute path for it, or the same method-local generics wherever it \
                     is used"
                        .to_string(),
                )],
            }],
        }
    }

    /// Returns the span of the first error.
    pub fn span(&self) -> Span {
        self.reports()
            .first()
            .map_or_else(Span::call_site, |report| report.tokens.span())
    }

    /// Returns the note of the first error, suggesting how to fix it: for
    /// [`NotInternableError::RelativePath`], the absolute path to use instead of the first
    /// segment of the offending path, if it is an item of the standard prelude or a primitive
    /// type.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait MyTrait: Clone {});
    /// let err = Leaker::from_trait(&item).err().unwrap();
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

/// Returns a note suggesting the absolute path to use instead of the first segment of `tokens`
/// (a path or a trait bound, e.g. `for<'a> Fn(&'a u8)`), with the span of the segment.
fn absolute_path_note(tokens: &TokenStream) -> Option<(Span, String)> {
    let path = syn::parse2::<syn::Path>(tokens.clone()).ok().or_else(|| {
        match syn::parse2(tokens.clone()).ok()? {
            syn::TypeParamBound::Trait(bound) => Some(bound.path),
            _ => None,
        }
    })?;
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

/// Converts into a [`syn::Error`] with one error per offending path (combined with
/// [`syn::Error::combine()`]), each with its notes rendered as `note`s of the diagnostic.
///
/// Since stable Rust has no API for sub-diagnostics of proc macros, the note is appended to the
/// message (as `proc-macro-error` does on stable), which rustc renders as:
///
/// ```text
/// error: use absolute path: 'Clone'
///
///          = note: use `::core::clone::Clone` instead of `Clone`
/// ```
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let item: ItemTrait = parse_quote!(trait MyTrait: Clone {});
/// let err: syn::Error = Leaker::from_trait(&item).err().unwrap().into();
/// assert_eq!(
///     err.to_string(),
///     "use absolute path: 'Clone'\n\n  = note: use `::core::clone::Clone` instead of `Clone`\n"
/// );
/// ```
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

/// Emits the errors as `compile_error!` invocations, with the notes rendered as in the conversion
/// into [`syn::Error`], so a proc macro can report them with e.g.
/// `Err(e) => e.into_token_stream().into()`.
///
/// On nightly Rust (when `proc_macro::Diagnostic` is available), inside a proc macro, the errors
/// are instead emitted directly as diagnostics, whose notes have their own spans (e.g. the first
/// segment of the offending path), and no tokens are produced:
///
/// ```text
/// error: use absolute path: 'Clone'
///  --> src/lib.rs:2:20
///   |
/// 2 | pub trait MyTrait: Clone {}
///   |                    ^^^^^
///   |
/// note: use `::core::clone::Clone` instead of `Clone`
///  --> src/lib.rs:2:20
///   |
/// 2 | pub trait MyTrait: Clone {}
///   |                    ^^^^^
/// ```
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::ToTokens;
/// let item: ItemTrait = parse_quote!(trait MyTrait: Clone {});
/// let err = Leaker::from_trait(&item).err().unwrap();
/// assert_eq!(
///     err.to_token_stream().to_string(),
///     syn::Error::from(err).to_compile_error().to_string()
/// );
/// ```
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
    /// Emits the errors as diagnostics, with the notes at their own spans.
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

/// Items of the standard prelude and primitive types, with their absolute paths.
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
    // Primitive types.
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

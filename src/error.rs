//! The error of the analysis by the [`Leaker`](crate::Leaker).

use crate::analysis::CheckResult;
use crate::util::{or_list, without_crate_prefix};
use proc_macro2::{Span, TokenStream};
use syn::spanned::Spanned;
use template_quote::ToTokens;

/// The error returned when an item cannot be analyzed by the [`Leaker`](crate::Leaker).
///
/// It lists every offending path or type. [`NotInternableError::items()`] returns one
/// [`ErrorItem`] for each, with its span, a message, notes and a help, so a macro can report them
/// all. The message says `use absolute path: '..'` where the fault is that a path is relative. It
/// is the headline of the cause where the fault is something else, such as `Self`, a bound rule or
/// a default that names `Self` ([`CheckResult::headline`]). `Display` writes the messages, one per
/// line.
///
/// Each error has a help and notes, as rustc's errors do. The help says what to write instead,
/// with the tokens of the error (e.g. "use `::std::vec::Vec` instead of `Vec`"). The notes say
/// why: the causes that stop the type from being leaked as a whole
/// ([`CheckResult::impossible`]), and the other paths in it that depend on the context (the rest
/// of [`CheckResult::must`]), which would be reported once the first is fixed. [`syn::Error`] and
/// [`ToTokens`] render the notes as `note`s, then the help as `help`.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let item: ItemTrait = parse_quote!(trait Tr: Clone { fn f(&self, a: MyType<'_>); });
/// let err = Leaker::new().intern_trait(&item, true).unwrap_err();
/// // One error for each offending path, in order.
/// assert_eq!(err.items().count(), 2);
/// // As one `syn::Error` for a proc macro, with the notes and the help appended to the messages.
/// let error: syn::Error = err.into();
/// assert_eq!(error.into_iter().count(), 2);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum NotInternableError {
    /// Paths or types that depend on the context but cannot be leaked, so the code must change:
    /// usually, the path must be written in full.
    ///
    /// Never empty. Each [`CheckResult`] is one conflict. It is reported at the first of its
    /// [`CheckResult::must`], with [`CheckResult::is_own`] `true`. For a trait bound, a value or
    /// the type of a const parameter, [`CheckResult::impossible`] is that bound, value or
    /// parameter. The cases are:
    ///
    /// - A relative path that cannot be leaked, with the headline `use absolute path: '..'`.
    ///   Examples: `MyType` in `MyType<'_>`, a trait in a bound, a value in a const expression,
    ///   `Rel` in `F: for<'x> Fn(Rel<'x>) -> &'x u8` ([Binder
    ///   lifetimes](crate::Leaker#binder-lifetimes)), and `Rel<T>` in `F: Fn(&Rel<T>) -> &u8` where
    ///   `Rel<T>` has lifetimes among its generics ([Elided
    ///   lifetimes](crate::Leaker#elided-lifetimes)).
    /// - A relative path in a value, with [`LeakerConfig::check_values()`](crate::LeakerConfig::check_values): the default of an
    ///   associated const, the value of a const or a static, the body of a function. It has the
    ///   headline `use absolute path: '..'`, and the note says which value it is in ("it is in the
    ///   default value of `N`, which the copy writes as it is"). A value is copied as written, so
    ///   nothing can be leaked there. What the Leaker cannot read (a macro whose arguments are not
    ///   expressions, a `macro_rules!`, a glob import) has the headline "cannot analyze .. in the
    ///   body of ..".
    /// - A type with a generic whose bound names the `Self` of a trait (or `Self::X`), where
    ///   `Self` is something else, so the bound cannot be written
    ///   ([`View::visit_item()`](crate::View::visit_item)). The fault is that bound, not the path
    ///   of the type. The headline is "the bound of `T` mentions a `Self` which has no name here",
    ///   with a note on why, at the type.
    /// - A type leaked as a whole that names `Self::A` where `Self` is a type (an impl, a struct,
    ///   an enum or a union), which the Repeater's impl cannot write. The headline says what to
    ///   write: "write `<Self as Tr>::A` instead of `Self::A` in `Wrap<Self::A>`: the copy cannot
    ///   tell which trait `A` comes from", with the trait of the impl as written (`SomeTrait` for a
    ///   struct, an enum, a union or an inherent impl, which have none). The help gives the path
    ///   in full (`<Self as ::my_crate::Tr>::A` if the trait is relative).
    /// - A type leaked as a whole that names a lifetime of a method which the copy would leave
    ///   unconstrained (`Wrap<'x>` in `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8`), for a method of a
    ///   trait or of an impl of a trait, not for a free function or a method of an inherent impl
    ///   ([Method lifetimes](crate::Leaker#method-lifetimes)). The fault is the lifetime, which an
    ///   absolute path is only one way to fix. The headline is "`Wrap<'x>` leaves the lifetime
    ///   `'x` of the method unconstrained in the copy (E0195)", with a note on E0195.
    /// - A type in the default of a parameter that names `Self`. The headline is "`Wrap<Self>` is
    ///   in the default of a parameter, and names `Self`".
    /// - A type that uses a generic whose bound mentions a `Self` that cannot be leaked. The
    ///   headline is "`Wrap<T>` has a generic whose bound mentions `Self`, which cannot be
    ///   leaked". That is the `Self` that `allow_self = false` denies. It is also the `Self` of a
    ///   struct, an enum or a union that [`View::visit_item()`](crate::View::visit_item) visits,
    ///   for a type that names it too (`Wrap<Self, T>` with `T: AsRef<Self>`): the Repeater's impl
    ///   of the type is for the Leaker's own type, and the copy declares a type of its own, which
    ///   has the bound for itself. A type that does not name it (`Wrap<T>`) is fine: its impl is
    ///   for any type that the bound holds for ([`SelfRequirement::Any`](crate::SelfRequirement)).
    ///   Write the type without `Self`, or analyze the item with its own `intern_*` call.
    /// - A type in a bound or where predicate that the copy cannot write through the Repeater,
    ///   because rustc cannot prove a bound of its generics that names a leaked type
    ///   ([Bounds that name leaked types](crate::Leaker#bounds-that-name-leaked-types)). The
    ///   headline is "`Wrap<T>` is in a bound or where predicate, and a bound of its generics names
    ///   a leaked type", or "`Rel<Self>` names `Self` in a bound or where predicate of its trait".
    /// - A type in the self type of an impl whose context depends on `Self` through a bound of
    ///   the impl ([`Leaker::intern_impl()`](crate::Leaker::intern_impl)). The headline is "`S<T>`
    ///   is in the self type of the impl, and a bound of its generics names `Self`".
    RelativePath(Vec<CheckResult>),
    /// `Self` where no scope gives it a meaning. Never empty.
    ///
    /// That is directly in a module, or in generics that no struct, enum, union, impl or trait
    /// encloses (`Leaker::enter_mod()`, `Leaker::enter_generics()`). It is also an error in the
    /// bounds, where predicates and defaults of those generics. See
    /// [`Leaker::enter_generics()`](crate::Leaker::enter_generics).
    ///
    /// Each item is the tokens of a mention (`Self`, `Self::Item`, or a type macro that names
    /// it). Its span is where the error is reported, with the message "`Self` has no meaning
    /// here" and a help that says to write the type, or to enter a scope that declares `Self`. If
    /// a scope finds both kinds of errors, only [`NotInternableError::RelativePath`] is
    /// returned. This kind is found after those are fixed.
    SelfWithoutMeaning(Vec<TokenStream>),
}

/// One error of a [`NotInternableError`]: an offending path, type or mention of `Self`.
///
/// Get them with [`NotInternableError::items()`] to report each one, for example as a
/// separate diagnostic of a proc macro, with its own span, notes and help. They are laid out as
/// rustc lays out its own errors: the `message` says what is wrong, the `notes` say why, and the
/// `help` says what to write instead, with the tokens of the error. Every error has a help.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let item: ItemTrait = parse_quote!(trait Tr: Clone { fn f(&self, a: MyType<'_>); });
/// let err = Leaker::new().intern_trait(&item, true).err().unwrap();
/// let items: Vec<_> = err.items().collect();
/// assert_eq!(items.len(), 2);
/// assert_eq!(items[0].message, "use absolute path: 'Clone'");
/// assert_eq!(items[0].help_text(), Some("use `::core::clone::Clone` instead of `Clone`"));
/// assert_eq!(items[1].message, "use absolute path: 'MyType < '_ >'");
/// assert_eq!(items[1].tokens.to_string(), "MyType < '_ >");
/// assert_eq!(items[1].note(), Some("it cannot be leaked as a whole because of `'_`"));
/// assert_eq!(
///     items[1].help_text(),
///     Some("write `MyType` with an absolute path (e.g. `::my_crate::MyType`), or allow it with \
///           `LeakerConfig::allowed_paths`")
/// );
/// // The `Display` of the error is the messages, one per line.
/// assert_eq!(err.to_string(), format!("{}\n{}", items[0].message, items[1].message));
/// ```
///
/// # Reporting the error from a proc macro
///
/// [`syn::Error`] (`err.into()`) and [`ToTokens`] (`err.into_token_stream()`) report every item,
/// with `= note:` lines and a `= help:` line after the message. On nightly, [`ToTokens`] emits a
/// `Diagnostic` instead, with the notes and the help at their own spans.
///
/// A macro that uses [`proc-macro-error`](https://docs.rs/proc-macro-error) reports each item
/// itself, with the notes as `note`s and the help as the `help` (the `hint` of an older version)
/// of the error:
///
/// ```text
/// for item in err.items() {
///     let help = item.help_text().unwrap_or_default();
///     match item.note() {
///         Some(note) => emit_error!(item.span, "{}", item.message; note = "{}", note; help = "{}", help),
///         None => emit_error!(item.span, "{}", item.message; help = "{}", help),
///     }
/// }
/// abort_if_dirty();
/// ```
///
/// or, to keep the span of every note and of the help (the first segment of a path to write in
/// full, for example), with its `Diagnostic`:
///
/// ```text
/// for item in err.items() {
///     let mut diagnostic = Diagnostic::spanned(item.span, Level::Error, item.message);
///     for (span, note) in item.notes {
///         diagnostic = diagnostic.span_note(span, note);
///     }
///     if let Some((span, help)) = item.help {
///         diagnostic = diagnostic.span_help(span, help);
///     }
///     diagnostic.emit();
/// }
/// ```
///
/// With `syn::Error` alone, the same layout is a message with the lines appended:
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let item: ItemTrait = parse_quote!(trait Tr: Clone {});
/// let err = Leaker::new().intern_trait(&item, true).unwrap_err();
/// let item = err.items().next().unwrap();
/// let mut message = item.message.clone();
/// for (_span, note) in &item.notes {
///     message += &format!("\n  = note: {}", note);
/// }
/// if let Some((_span, help)) = &item.help {
///     message += &format!("\n  = help: {}", help);
/// }
/// let error = syn::Error::new(item.span, message);
/// assert!(error.to_string().ends_with("= help: use `::core::clone::Clone` instead of `Clone`"));
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ErrorItem {
    /// The offending path, type or mention of `Self`, as written.
    pub tokens: TokenStream,
    /// Where the error is reported: the span of `tokens`.
    pub span: Span,
    /// What is wrong: the headline of the cause where the fault is not that the path is relative
    /// ([`CheckResult::headline`]; "`Self` has no meaning here"), and `use absolute path: '..'`
    /// where it is.
    pub message: String,
    /// Why, each with the span it is about, the most useful first: what the headline says in more
    /// words, the causes that stop the path from being leaked as a whole, and the other paths that
    /// depend on the type context.
    pub notes: Vec<(Span, String)>,
    /// What to write instead, with the span it is about, as the tokens of the error say it (for
    /// example the absolute path of a prelude name: "use `::std::vec::Vec` instead of `Vec`").
    /// Every error has one.
    pub help: Option<(Span, String)>,
}

impl ErrorItem {
    /// Returns the first note, which says why, if there is one.
    pub fn note(&self) -> Option<&str> {
        self.notes.first().map(|(_, note)| note.as_str())
    }

    /// Returns the text of the help, which says what to write instead.
    pub fn help_text(&self) -> Option<&str> {
        self.help.as_ref().map(|(_, help)| help.as_str())
    }
}

pub(crate) const SELF_NOTE: &str =
    "`Self` has no meaning here: it is declared only by a struct, an enum, a \
                         union, an impl or a trait, not by a module or generics";

pub(crate) const SELF_HELP: &str =
    "write the type instead of `Self`, or enter a scope that declares `Self` around it \
     (`Leaker::enter_type`, `enter_impl` or `enter_trait`)";

impl NotInternableError {
    fn reports(&self) -> Vec<ErrorItem> {
        let conflicts = match self {
            NotInternableError::RelativePath(conflicts) => conflicts,
            NotInternableError::SelfWithoutMeaning(selves) => {
                return selves
                    .iter()
                    .map(|tokens| ErrorItem {
                        tokens: tokens.clone(),
                        span: tokens.span(),
                        message: "`Self` has no meaning here".to_string(),
                        notes: vec![(tokens.span(), SELF_NOTE.to_string())],
                        help: Some((tokens.span(), SELF_HELP.to_string())),
                    })
                    .collect();
            }
        };
        conflicts
            .iter()
            .filter_map(|conflict| {
                let (tokens, others) = conflict.must.split_first()?;
                // The conflict is not about the path if it has a note.
                let mut notes: Vec<_> = conflict
                    .note
                    .iter()
                    .map(|note| (tokens.span(), note.clone()))
                    .collect();
                let mut default_object = false;
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
                        default_object = true;
                        notes.push((
                            cause.span(),
                            "the lifetime of the trait object defaults to that of the enclosing \
                             reference or path"
                                .to_string(),
                        ));
                    }
                }
                let help = match &conflict.help {
                    Some(help) => (tokens.span(), help.clone()),
                    None => absolute_path_help(
                        tokens,
                        default_object,
                        conflict.impossible.iter().any(is_self_path),
                    ),
                };
                notes.extend(others.iter().map(|other| {
                    (
                        other.span(),
                        format!("`{}` also depends on the type context", other),
                    )
                }));
                Some(ErrorItem {
                    tokens: tokens.clone(),
                    span: tokens.span(),
                    message: conflict
                        .headline
                        .clone()
                        .unwrap_or_else(|| format!("use absolute path: '{}'", tokens)),
                    notes,
                    help: Some(help),
                })
            })
            .collect()
    }

    /// Returns the errors, one per offending path, type or mention of `Self`, in order.
    ///
    /// Each has its span, a message, notes and a help, so a proc macro can report every one
    /// instead of only the first ([`NotInternableError::span()`], [`NotInternableError::note()`],
    /// [`NotInternableError::help()`]). The message is `use absolute path: '..'` where the fault
    /// is that a path is relative, and the headline of the cause where it is something else.
    /// `Display` writes the messages, one per line. Never empty.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait Tr: Clone { fn f(&self, a: MyType<'_>); });
    /// let err = Leaker::new().intern_trait(&item, true).unwrap_err();
    /// for item in err.items() {
    ///     // Report `item.message` at `item.span`, then each note at its own span.
    ///     let _ = (&item.message, item.span);
    ///     for (_span, _note) in &item.notes {}
    ///     if let Some((_span, _help)) = &item.help {}
    /// }
    /// let messages: Vec<String> = err.items().map(|item| item.message).collect();
    /// assert_eq!(messages, ["use absolute path: 'Clone'", "use absolute path: 'MyType < '_ >'"]);
    /// ```
    pub fn items(&self) -> impl Iterator<Item = ErrorItem> {
        self.reports().into_iter()
    }

    /// Returns the span of the first error.
    pub fn span(&self) -> Span {
        self.reports()
            .first()
            .map_or_else(Span::call_site, |report| report.span)
    }

    /// Returns the first note of the first error, which says why.
    ///
    /// For `SelfWithoutMeaning` it tells which items declare `Self`. A conflict about the path
    /// itself has none unless a cause stops it from being leaked as a whole.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S { a: MyType<'_> });
    /// let err = Leaker::new().intern_struct(&item, true).err().unwrap();
    /// assert_eq!(err.to_string(), "use absolute path: 'MyType < '_ >'");
    /// assert_eq!(
    ///     err.note().as_deref(),
    ///     Some("it cannot be leaked as a whole because of `'_`")
    /// );
    /// ```
    pub fn note(&self) -> Option<String> {
        let report = self.reports().into_iter().next()?;
        report.notes.into_iter().next().map(|(_, note)| note)
    }

    /// Returns the help of the first error, which says what to write instead.
    ///
    /// For `RelativePath` it is the absolute path to use instead of the first segment, if that is a
    /// prelude item or a primitive type, and else how to write the path. For
    /// `SelfWithoutMeaning` it tells which scopes declare `Self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait MyTrait: Clone {});
    /// let err = Leaker::new().intern_trait(&item, true).err().unwrap();
    /// assert_eq!(err.to_string(), "use absolute path: 'Clone'");
    /// assert_eq!(
    ///     err.help().as_deref(),
    ///     Some("use `::core::clone::Clone` instead of `Clone`")
    /// );
    /// ```
    pub fn help(&self) -> Option<String> {
        let report = self.reports().into_iter().next()?;
        report.help.map(|(_, help)| help)
    }
}

/// A trait object without a lifetime bound is a cause only under an elided lifetime, or where the
/// path cannot be leaked (see [Trait objects](crate::Leaker#trait-objects)).
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

/// The absolute path of a prelude item or a primitive type that `path` starts with.
fn prelude_help(path: &syn::Path) -> Option<(Span, String)> {
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

/// What to write for the path `tokens` that depends on the context, when the conflict has nothing
/// more to say: the absolute path of a prelude item, else how to write the path in full.
/// `default_object`: a trait object takes its lifetime from the reference or path around it.
/// `denied_self`: a `Self` that is denied (`allow_self = false`) is among the causes.
fn absolute_path_help(
    tokens: &TokenStream,
    default_object: bool,
    denied_self: bool,
) -> (Span, String) {
    let path = offending_path(tokens);
    // The help is about the name of the path: its first segment.
    let span = path
        .as_ref()
        .and_then(|path| path.segments.first())
        .map_or_else(|| tokens.span(), |seg| seg.ident.span());
    let mut options = Vec::new();
    match path.as_ref().and_then(prelude_help) {
        Some((_, help)) => options.push(help),
        None => match &path {
            Some(path)
                if path
                    .segments
                    .first()
                    .map_or(false, |seg| seg.ident == "Self") =>
            {
                options.push("write the type instead of `Self`".to_string());
            }
            Some(path) if path.leading_colon.is_none() => {
                let (name, example) = path_names(path);
                options.push(format!(
                    "write `{}` with an absolute path (e.g. `{}`)",
                    name, example
                ));
                options.push("allow it with `LeakerConfig::allowed_paths`".to_string());
            }
            Some(path) => {
                let (name, _) = path_names(path);
                options.push(format!(
                    "write `{}` where it is not leaked, or allow it with \
                     `LeakerConfig::allowed_paths`",
                    name
                ));
            }
            None => options.push(format!(
                "write the paths in `{}` as absolute paths",
                crate::util::tidy(tokens)
            )),
        },
    }
    if denied_self {
        options.push("analyze the item with `allow_self = true`".to_string());
    }
    if default_object {
        options.push(
            "name the lifetime of the trait object (e.g. `&'a dyn Trait` or `dyn Trait + 'a`)"
                .to_string(),
        );
    }
    (span, or_list(options))
}

/// The path as it is written, without its arguments, and an example of its absolute form.
fn path_names(path: &syn::Path) -> (String, String) {
    let names = |path: &syn::Path| {
        path.segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::")
    };
    let name = format!(
        "{}{}",
        if path.leading_colon.is_some() {
            "::"
        } else {
            ""
        },
        names(path)
    );
    // `crate::m::X` and `super::m::X` are `m::X` in the crate.
    let example = format!("::my_crate::{}", names(&without_crate_prefix(path)));
    (name, example)
}

/// `true` if `cause` is `Self` or a path that starts with it.
fn is_self_path(cause: &TokenStream) -> bool {
    matches!(cause.clone().into_iter().next(), Some(proc_macro2::TokenTree::Ident(ident)) if ident == "Self")
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
                if !report.notes.is_empty() || report.help.is_some() {
                    message.push('\n');
                    for (_, note) in report.notes {
                        message.push_str(&format!("\n  = note: {}", note));
                    }
                    if let Some((_, help)) = report.help {
                        message.push_str(&format!("\n  = help: {}", help));
                    }
                    message.push('\n');
                }
                syn::Error::new(report.span, message)
            })
            .reduce(|mut combined, error| {
                combined.combine(error);
                combined
            })
            .unwrap_or_else(|| syn::Error::new(Span::call_site(), "not internable"))
    }
}

/// Emits the errors as `compile_error!` invocations, with the notes and the help rendered as for
/// [`syn::Error`]. A proc macro can report them with `Err(e) => e.into_token_stream().into()`.
///
/// On nightly, when `proc_macro::Diagnostic` is available inside a proc macro, the errors are
/// emitted as diagnostics instead, with notes and help at their own spans (e.g. the first segment
/// of the offending path), and no tokens are produced.
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
                unwrap_span(report.span),
                proc_macro::Level::Error,
                report.message,
            );
            for (span, note) in report.notes {
                diagnostic = diagnostic.span_note(unwrap_span(span), note);
            }
            if let Some((span, help)) = report.help {
                diagnostic = diagnostic.span_help(unwrap_span(span), help);
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

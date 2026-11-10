//! The errors of the analysis by the [`Leaker`](crate::Leaker): an [`ErrorItem`] for each part of
//! an item that cannot be interned ([recovery](crate::Leaker#errors-and-recovery)).

use crate::allow::Place;
use crate::analysis::{CheckResult, Part, SelfMention};
use crate::context::Context;
use crate::util::{
    has_lifetime_bound, is_std_crate_path, or_list, path_starts_with, path_type, tidy,
    without_arguments, without_crate_prefix, KNOWN_NAMES,
};
use proc_macro2::{Delimiter, Literal, Span, TokenStream, TokenTree};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    braced, bracketed, parenthesized, token, Expr, ExprPath, Ident, Lifetime, LitInt, LitStr, Path,
    Token, Type,
};
use template_quote::{quote, ToTokens};

/// The part of an item that an [`ErrorItem`] is about, as written.
///
/// It is usually the whole syntax that cannot be accepted (`MyType<'_, Rel>`, not `MyType`). Causes
/// are equal if they have the same kind and tokens (spans and spacing ignored), so a cause written
/// by hand (`Cause::Trait(parse_quote!(RelTr))`) can be compared with `==` in an
/// [`Allow`](crate::Allow) rule.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Cause {
    /// A type that must be leaked whole and cannot (`MyType<'_, Rel>`), the type of a const
    /// parameter, `Self` where it has no meaning, or a type macro.
    Type(Type),
    /// The trait of a bound, a supertrait, a where predicate, an impl, a `dyn` or `impl` type, or a
    /// qualified path in a value (`<u8 as Rel>::f()`).
    Trait(Path),
    /// An expression: a path in a const expression (an array length, a discriminant) or, with
    /// [`Leaker::check_values`](crate::Leaker::check_values), in a value; also an expression that
    /// cannot be leaked (a closure in an array length). A path is an [`Expr::Path`], as written.
    Expr(Expr),
    /// The path of a macro call, written with its `!` (`two!` in `[u8; two!()]`). A macro that is a
    /// type is a [`Cause::Type`].
    Macro(Path),
    /// A lifetime: an elided one (`'_`), one of a method, or one that no scope declares.
    Lifetime(Lifetime),
    /// A path whose kind is not known: the root of a `use`, a path in the tokens of a macro.
    Unknown(Path),
    /// Syntax that the Leaker cannot read (`macro_rules!`, `use inner::*;`) or that no node of the
    /// syntax holds: a bound with a binder (`for<'a> Fn(&'a u8)`) or in parentheses.
    Unreadable(TokenStream),
}

impl Cause {
    /// The name of the kind in the serialized form of an [`ErrorItem`] and of a part of the
    /// Referrer.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Cause::Type(_) => "type",
            Cause::Trait(_) => "trait",
            Cause::Expr(_) => "expr",
            Cause::Macro(_) => "macro",
            Cause::Lifetime(_) => "lifetime",
            Cause::Unknown(_) => "unknown",
            Cause::Unreadable(_) => "unreadable",
        }
    }

    /// The cause that is the path of a type (`Rel`, `Self::Item`).
    pub(crate) fn type_path(path: &Path) -> Self {
        Cause::Type(path_type(path))
    }

    /// The cause that is a path in an expression.
    pub(crate) fn expr_path(path: &Path) -> Self {
        Cause::Expr(Expr::Path(ExprPath {
            attrs: Vec::new(),
            qself: None,
            path: path.clone(),
        }))
    }

    /// The cause that is the path expression `path`.
    pub(crate) fn path_expr(path: &ExprPath) -> Self {
        Cause::Expr(Expr::Path(path.clone()))
    }

    /// The cause that is `Self`, as the mention of a type macro leaves it.
    pub(crate) fn self_type() -> Self {
        Cause::Type(syn::parse_quote!(Self))
    }

    /// The cause as messages show it: as written, but a macro by its path (`two`, not `two !`).
    pub(crate) fn shown(&self) -> TokenStream {
        match self {
            Cause::Macro(path) => path.to_token_stream(),
            other => other.to_token_stream(),
        }
    }

    /// Returns the path of a [`Cause::Type`] that is a path without a qualified self type, of a
    /// [`Cause::Trait`], of a [`Cause::Macro`], of a [`Cause::Unknown`] and of a [`Cause::Expr`]
    /// that is a path, as written (with its generic arguments, if it has any).
    pub fn path(&self) -> Option<&Path> {
        match self {
            Cause::Type(Type::Path(syn::TypePath { qself: None, path }))
            | Cause::Trait(path)
            | Cause::Macro(path)
            | Cause::Unknown(path)
            | Cause::Expr(Expr::Path(ExprPath {
                qself: None, path, ..
            })) => Some(path),
            _ => None,
        }
    }

    /// Returns `true` if [`path()`](Cause::path) starts with `prefix`, segment by segment by
    /// identifier. Both must have a leading `::` or neither; generic arguments are ignored.
    pub fn starts_with(&self, prefix: &Path) -> bool {
        self.path()
            .map_or(false, |path| path_starts_with(path, prefix))
    }

    /// The cause of the same kind with the generic arguments of its path removed (`Rel`, not
    /// `Rel<u8>`): what the rule is asked and what [`ErrorItem::paths`] holds. A cause that is not
    /// a path stays as it is.
    pub(crate) fn bare(&self) -> Self {
        match self {
            Cause::Type(Type::Path(syn::TypePath { qself: None, path })) => {
                Cause::type_path(&without_arguments(path))
            }
            Cause::Trait(path) => Cause::Trait(without_arguments(path)),
            Cause::Macro(path) => Cause::Macro(without_arguments(path)),
            Cause::Unknown(path) => Cause::Unknown(without_arguments(path)),
            Cause::Expr(Expr::Path(ExprPath {
                qself: None, path, ..
            })) => Cause::expr_path(&without_arguments(path)),
            other => other.clone(),
        }
    }

    /// Reads the cause of the kind `kind` from all of `input`.
    pub(crate) fn parse(kind: &Ident, input: ParseStream) -> syn::Result<Self> {
        let cause = match kind.to_string().as_str() {
            "type" => Cause::Type(input.parse()?),
            "trait" => {
                // The sugar of `Fn(..) -> ..`, which `Path` leaves to its users.
                let mut path: Path = input.parse()?;
                let bare = path
                    .segments
                    .last()
                    .map_or(false, |seg| seg.arguments.is_empty());
                if bare && input.peek(token::Paren) {
                    let arguments = input.parse()?;
                    if let Some(seg) = path.segments.last_mut() {
                        seg.arguments = syn::PathArguments::Parenthesized(arguments);
                    }
                }
                Cause::Trait(path)
            }
            "expr" => {
                // A path is an expression path, also one that is not that of an expression (`Rel<u8>`
                // in a pattern), which is a path as a type has it. Else it is any expression.
                let tokens: TokenStream = input.parse()?;
                Cause::Expr(match syn::parse2::<ExprPath>(tokens.clone()) {
                    Ok(path) => Expr::Path(path),
                    Err(_) => match syn::parse2::<Path>(tokens.clone()) {
                        Ok(path) => Expr::Path(ExprPath {
                            attrs: Vec::new(),
                            qself: None,
                            path,
                        }),
                        Err(_) => syn::parse2(tokens)?,
                    },
                })
            }
            "macro" => {
                let path = input.parse()?;
                input.parse::<Token![!]>()?;
                Cause::Macro(path)
            }
            "lifetime" => Cause::Lifetime(input.parse()?),
            "unknown" => Cause::Unknown(input.parse()?),
            "unreadable" => Cause::Unreadable(input.parse()?),
            other => {
                return Err(syn::Error::new(
                    kind.span(),
                    format!("unknown kind `{}`", other),
                ))
            }
        };
        if input.is_empty() {
            Ok(cause)
        } else {
            Err(input.error("unexpected token in the cause"))
        }
    }
}

/// Writes the syntax as it was written, with its spans (a [`Cause::Macro`] with its `!`).
impl ToTokens for Cause {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Cause::Type(ty) => ty.to_tokens(tokens),
            Cause::Trait(path) => path.to_tokens(tokens),
            // A path that a type holds (`Rel<u8>` in a pattern) is written as it is, whereas an
            // expression path would write its arguments with a turbofish (`Rel::<u8>`).
            Cause::Expr(Expr::Path(path)) if path.qself.is_none() => {
                for attr in &path.attrs {
                    attr.to_tokens(tokens);
                }
                path.path.to_tokens(tokens);
            }
            Cause::Expr(expr) => expr.to_tokens(tokens),
            Cause::Macro(path) => {
                path.to_tokens(tokens);
                let span = path
                    .segments
                    .last()
                    .map_or_else(Span::call_site, |seg| seg.ident.span());
                token::Not(span).to_tokens(tokens);
            }
            Cause::Lifetime(lifetime) => lifetime.to_tokens(tokens),
            Cause::Unknown(path) => path.to_tokens(tokens),
            Cause::Unreadable(unreadable) => unreadable.to_tokens(tokens),
        }
    }
}

/// A token of a cause without its span and its spacing: what two causes are compared by.
#[derive(PartialEq, Eq, Hash)]
enum Token {
    /// The start of a group, with its opening delimiter.
    Open(char),
    /// The end of the group.
    Close,
    Ident(String),
    Punct(char),
    Literal(String),
}

impl Token {
    /// Appends the tokens of `stream`. A group without delimiters (what a macro leaves around a
    /// fragment) is no token: its content is.
    fn collect(stream: TokenStream, tokens: &mut Vec<Token>) {
        for tree in stream {
            match tree {
                TokenTree::Group(group) => {
                    let open = match group.delimiter() {
                        Delimiter::Parenthesis => '(',
                        Delimiter::Brace => '{',
                        Delimiter::Bracket => '[',
                        Delimiter::None => {
                            Token::collect(group.stream(), tokens);
                            continue;
                        }
                    };
                    tokens.push(Token::Open(open));
                    Token::collect(group.stream(), tokens);
                    tokens.push(Token::Close);
                }
                TokenTree::Ident(ident) => tokens.push(Token::Ident(ident.to_string())),
                TokenTree::Punct(punct) => tokens.push(Token::Punct(punct.as_char())),
                TokenTree::Literal(literal) => tokens.push(Token::Literal(literal.to_string())),
            }
        }
    }
}

impl Cause {
    /// The tokens of the cause, apart from their spans and their spacing.
    fn canonical(&self) -> Vec<Token> {
        let mut tokens = Vec::new();
        Token::collect(self.to_token_stream(), &mut tokens);
        tokens
    }
}

impl PartialEq for Cause {
    fn eq(&self, other: &Self) -> bool {
        self.kind() == other.kind() && self.canonical() == other.canonical()
    }
}

impl Eq for Cause {}

impl Hash for Cause {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind().hash(state);
        self.canonical().hash(state);
    }
}

/// Why a part of an item cannot be interned: the rule that it breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Reason {
    /// A path or type that depends on the context but cannot be leaked, so it must be written in
    /// full (`MyType` in `MyType<'_>`, a trait in a bound), or a relative path in a value. A macro
    /// can accept a path with an [`Allow`](crate::Allow) rule.
    RelativePath,
    /// `Self` where no scope gives it a meaning: in a module, or in generics that no struct, enum,
    /// union, impl or trait encloses. A macro can enter such a scope
    /// ([`Leaker::enter_type()`](crate::Leaker::enter_type), `enter_impl()`, `enter_trait()`).
    /// The error has no [`paths`](ErrorItem::paths), so a rule that allows its cause (`Self`,
    /// `Self::Item`) accepts it ([`Allow`](crate::Allow)).
    SelfWithoutMeaning,
    /// Syntax that the Leaker cannot read, with `check_values`: a macro whose arguments are not
    /// expressions, a `macro_rules!`, a glob import, syntax that `syn` leaves unparsed.
    Unreadable,
    /// A bare identifier among the arguments of a path, which may be a const, in a type that cannot
    /// be leaked whole ([Bare identifiers and consts](crate::Leaker#bare-identifiers-and-consts)).
    MaybeConst,
    /// A leaked type that names a method lifetime the copy would leave unconstrained (`Wrap<'x>` in
    /// `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8`, E0195). The fault is the lifetime, not the path.
    UnconstrainedLifetime,
    /// A leaked type that names `Self::A` where `Self` is a type: write `<Self as Tr>::A`.
    SelfProjection,
    /// A type with a generic whose bound names the `Self` of a trait, where `Self` is something
    /// else. The fault is that bound.
    UnnameableSelf,
    /// A type in the default of a parameter that names `Self`: the cause is the type of the
    /// default, and the note is at the `Self`. In a nested struct, enum or union it is an error
    /// only if a bound of the generics names `Self` too (E0735).
    SelfInDefault,
    /// A risk, not a part copied as written: the type is an entry all the same. It names the `Self`
    /// of a nested struct, enum or union and uses a generic whose bound names it too
    /// (`Wrap<Self, T>` in `struct S<T: AsRef<Self>>(Wrap<Self, T>);`). The entry is for any type
    /// that the bound holds for ([`SelfRequirement::Any`](crate::SelfRequirement::Any)), and the
    /// copy passes its own type.
    ///
    /// The error has no [`paths`](ErrorItem::paths).
    /// [`LeakerResult::ignore()`](crate::LeakerResult::ignore) removes the entry, so the copy writes
    /// the type as it is, and none of the following matters.
    /// [`LeakerResult::allow()`](crate::LeakerResult::allow) drops the error if the rule allows its
    /// cause, and keeps the entry, and then you promise two things. The declaration of the leaked
    /// type has no trait bound on the parameter that receives `Self` (`struct Wrap<A: Clone, B>`),
    /// else the impl fails in the Leaker's crate (E0277): write the type with an absolute path or
    /// give the item its own `intern_*` call. And rustc before 1.70 fails with E0275 if the copy's
    /// field is the last field of a struct: put it first, or add `where Self: Sized` to the item. A
    /// default of a parameter is not changed (E0735), and a bound that names a leaked type that
    /// names `Self` is still an error.
    NestedSelfInBounds,
    /// A type in a bound or where predicate that the copy cannot write through the Repeater
    /// ([Bounds that name leaked types](crate::Leaker#bounds-that-name-leaked-types)).
    UnprovableBound,
    /// A type that names `Self` in a supertrait bound of its trait, or in a where predicate on a
    /// type that is neither `Self` nor a generic (E0275): `Rel<Self>` in `trait Tr:
    /// ::p::Tr<Rel<Self>>`. In a bound of `Self` or of a generic (`where Self: ::p::Tr<Rel<Self>>`,
    /// `T: ::p::Tr<Rel<Self>>`) the same type is [`UnprovableBound`](Reason::UnprovableBound).
    SelfInTraitPredicate,
    /// A type in the self type of an impl whose context depends on `Self` through a bound of the
    /// impl.
    SelfInImplSelfType,
}

impl Reason {
    /// Every reason.
    const ALL: [Reason; 12] = [
        Reason::RelativePath,
        Reason::SelfWithoutMeaning,
        Reason::Unreadable,
        Reason::MaybeConst,
        Reason::UnconstrainedLifetime,
        Reason::SelfProjection,
        Reason::UnnameableSelf,
        Reason::SelfInDefault,
        Reason::NestedSelfInBounds,
        Reason::UnprovableBound,
        Reason::SelfInTraitPredicate,
        Reason::SelfInImplSelfType,
    ];

    /// The word that names the reason in the serialized form of an [`ErrorItem`].
    fn name(self) -> &'static str {
        match self {
            Reason::RelativePath => "RelativePath",
            Reason::SelfWithoutMeaning => "SelfWithoutMeaning",
            Reason::Unreadable => "Unreadable",
            Reason::MaybeConst => "MaybeConst",
            Reason::UnconstrainedLifetime => "UnconstrainedLifetime",
            Reason::SelfProjection => "SelfProjection",
            Reason::UnnameableSelf => "UnnameableSelf",
            Reason::SelfInDefault => "SelfInDefault",
            Reason::NestedSelfInBounds => "NestedSelfInBounds",
            Reason::UnprovableBound => "UnprovableBound",
            Reason::SelfInTraitPredicate => "SelfInTraitPredicate",
            Reason::SelfInImplSelfType => "SelfInImplSelfType",
        }
    }

    /// Reads the reason that `ident` names.
    fn parse(ident: &Ident) -> syn::Result<Self> {
        let name = ident.to_string();
        Self::ALL
            .iter()
            .copied()
            .find(|reason| reason.name() == name)
            .ok_or_else(|| syn::Error::new(ident.span(), format!("unknown reason `{}`", name)))
    }
}

/// One error of the analysis: a part of an item that cannot be interned.
///
/// [`LeakerResult::errors()`](crate::LeakerResult::errors) lists them in the order found. The
/// `message` says what is wrong, the `notes` why, and the `help` what to write instead, each with
/// the tokens it is about, so it can be reported at its own span.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_trait(&parse_quote!(trait Tr { fn f(&self, a: MyType<'_>); }));
/// let result = leaker.result();
/// let item = &result.errors()[0];
/// assert_eq!(item.message, "use absolute path: 'MyType<'_>'");
/// assert_eq!(item.reason, Reason::RelativePath);
/// assert!(matches!(item.cause, Cause::Type(_))); // the whole `MyType<'_>`
/// assert_eq!(item.notes[0].1, "it cannot be leaked as a whole because of `'_`");
/// let help = "write `MyType` with an absolute path (e.g. `::my_crate::MyType`)";
/// assert_eq!(item.help_text(), Some(help));
/// // `Display` is the message.
/// assert_eq!(item.to_string(), item.message);
/// ```
///
/// Report an item from a macro with [`emit()`](ErrorItem::emit);
/// [`to_compile_error()`](ErrorItem::to_compile_error) is the same without the nightly diagnostic,
/// and `syn::Error::from(item)` is a [`syn::Error`] at the span of the cause. [`ToTokens`] and
/// [`Parse`] serialize an item as one brace group, to pass it to another macro of the same
/// type-leak version.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ErrorItem {
    /// The part of the item that cannot be interned, as written, and where the error is reported
    /// ([`ErrorItem::span()`]). Compare it with `==` in an [`Allow`](crate::Allow) rule: a rule
    /// accepts an error that has [`paths`](ErrorItem::paths) by them, and one that has none by this.
    pub cause: Cause,
    /// Why it cannot be interned.
    pub reason: Reason,
    /// What is wrong: `use absolute path: '..'` for a relative path, else the reason's headline.
    pub message: String,
    /// Why, each with the tokens it is about, the most useful first: the note of the reason, then
    /// the causes that stop the type from being leaked whole or depend on the context. A relative
    /// path that would be leaked once the cause is fixed is no note (`Local` in `Base<Local>`): it
    /// is in [`paths`](ErrorItem::paths).
    pub notes: Vec<(TokenStream, String)>,
    /// What to write instead, with the tokens it is about. The analysis gives every error one; it
    /// is `None` only in an item that [`Parse`] read without.
    pub help: Option<(TokenStream, String)>,
    /// Where it was found: in a value that [`Leaker::check_values`](crate::Leaker::check_values)
    /// reads, in a default, both or neither. It is the place that an [`Allow`](crate::Allow) rule
    /// is asked at.
    ///
    /// ```ignore
    /// // In crate a, with the values read.
    /// trait Tr<U = Rel<'_>> {          // `Rel<'_>`: in_default
    ///     fn f(&self) { rel::f(); }    // `rel::f`: in_value and in_default
    /// }
    /// impl ::a::Tr for X {
    ///     fn f(&self) { rel::f(); }    // `rel::f`: in_value
    /// }
    /// struct S(Rel<'_>);               // `Rel<'_>`: neither
    /// ```
    pub place: Place,
    /// The relative paths that the copy writes as they are, each as the cause of its kind (a
    /// [`Cause::Type`], [`Cause::Trait`], [`Cause::Expr`], [`Cause::Macro`] or [`Cause::Unknown`]
    /// that is a path) and without its arguments: `MyType` in `MyType<'_, Rel>`. What is inside
    /// the arguments is analyzed on its own, so it has its own errors and entries. There are
    /// several where one cause holds several paths (`dyn A + B`). A rule drops the error if it
    /// accepts all of them.
    pub paths: Vec<Cause>,
    /// What the error is about in the Referrer, for
    /// [`LeakerResult::ignore()`](crate::LeakerResult::ignore).
    pub(crate) link: Link,
}

/// What an error is about, so that [`LeakerResult::ignore()`](crate::LeakerResult::ignore) can
/// find it in the Referrer. The analysis gives the data it has (`raw`); the Leaker resolves it to
/// indices when it makes the result. Serialized after the paths of the error, each word only if it
/// is set: `id N`, `part N`, `entry N`, `inner [N, ..]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Link {
    /// The number that the errors held in `inner` are known by: set only for an error that some
    /// other error holds. It counts the errors of the Leaker in the order found, also those that a
    /// rule dropped, so it stays the same when errors are dropped.
    pub(crate) id: Option<usize>,
    /// The index in the parts of the Referrer of the part that the copy writes as written for
    /// this error.
    pub(crate) part: Option<usize>,
    /// The index of the entry that the error is about (a risk, not a part).
    pub(crate) entry: Option<usize>,
    /// The errors revealed inside its part, by `id`.
    pub(crate) inner: Vec<usize>,
    /// What the analysis knows, until the Leaker makes the result.
    pub(crate) raw: Raw,
}

/// What the analysis knows of what an error is about.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Raw {
    /// The part that the copy writes as written.
    pub(crate) part: Option<Part>,
    /// The entry that the error is about, as its context and type.
    pub(crate) entry: Option<(Context, Type)>,
}

impl Link {
    /// Counts `by` more errors before these: for an error recorded after others.
    pub(crate) fn shift(&mut self, by: usize) {
        self.id = self.id.map(|id| id + by);
        for id in &mut self.inner {
            *id += by;
        }
    }
}

/// Makes each error hold only the errors inside its part that are in `errors`: after errors were
/// dropped, the others do not hold them.
pub(crate) fn forget_dropped(errors: &mut [ErrorItem]) {
    let present: HashSet<usize> = errors.iter().filter_map(|error| error.link.id).collect();
    for error in errors {
        error.link.inner.retain(|id| present.contains(id));
    }
}

impl ErrorItem {
    /// Returns the span of the cause, where the error is reported.
    pub fn span(&self) -> Span {
        self.cause.to_token_stream().span()
    }

    /// Returns the text of the help, which says what to write instead.
    pub fn help_text(&self) -> Option<&str> {
        self.help.as_ref().map(|(_, help)| help.as_str())
    }

    /// Returns a `compile_error!` that reports the error, with the notes and the help in its
    /// message. Pure: use it to build or compare tokens; a macro reports with [`emit()`](ErrorItem::emit).
    #[must_use]
    pub fn to_compile_error(&self) -> TokenStream {
        self.to_syn_error().to_compile_error()
    }

    /// Reports the error from a proc macro, and returns the tokens to give back. On nightly it
    /// emits a `Diagnostic` with the notes and the help at their own spans and returns no tokens;
    /// elsewhere it returns [`to_compile_error()`](ErrorItem::to_compile_error).
    #[must_use]
    pub fn emit(&self) -> TokenStream {
        // Only compiled on nightly, so the MSRV does not apply.
        #[cfg(type_leak_diagnostic)]
        #[allow(clippy::incompatible_msrv)]
        if proc_macro::is_available() {
            self.emit_diagnostic();
            return TokenStream::new();
        }
        self.to_compile_error()
    }

    fn to_syn_error(&self) -> syn::Error {
        let mut message = self.message.clone();
        if !self.notes.is_empty() || self.help.is_some() {
            message.push('\n');
            for (_, note) in &self.notes {
                message.push_str(&format!("\n  = note: {}", note));
            }
            if let Some((_, help)) = &self.help {
                message.push_str(&format!("\n  = help: {}", help));
            }
            message.push('\n');
        }
        syn::Error::new(self.span(), message)
    }
}

#[cfg(type_leak_diagnostic)]
impl ErrorItem {
    fn emit_diagnostic(&self) {
        /// Converts a span into that of the compiler (only valid inside a proc macro).
        fn unwrap_span(span: Span) -> proc_macro::Span {
            let ident = proc_macro2::TokenTree::Ident(proc_macro2::Ident::new("x", span));
            proc_macro::TokenStream::from(TokenStream::from(ident))
                .into_iter()
                .next()
                .unwrap()
                .span()
        }
        let mut diagnostic = proc_macro::Diagnostic::spanned(
            unwrap_span(self.span()),
            proc_macro::Level::Error,
            self.message.clone(),
        );
        for (at, note) in &self.notes {
            diagnostic = diagnostic.span_note(unwrap_span(at.span()), note.clone());
        }
        if let Some((at, help)) = &self.help {
            diagnostic = diagnostic.span_help(unwrap_span(at.span()), help.clone());
        }
        diagnostic.emit();
    }
}

/// Two errors are equal if every field is: the tokens by their kinds and text, apart from spans
/// and spacing, and what the error is about in the Referrer.
impl PartialEq for ErrorItem {
    fn eq(&self, other: &Self) -> bool {
        let same_at = |a: &(TokenStream, String), b: &(TokenStream, String)| {
            a.1 == b.1 && same_tokens(&a.0, &b.0)
        };
        self.cause == other.cause
            && self.reason == other.reason
            && self.message == other.message
            && self.notes.len() == other.notes.len()
            && self
                .notes
                .iter()
                .zip(&other.notes)
                .all(|(a, b)| same_at(a, b))
            && match (&self.help, &other.help) {
                (Some(a), Some(b)) => same_at(a, b),
                (None, None) => true,
                _ => false,
            }
            && self.place == other.place
            && self.paths == other.paths
            && self.link == other.link
    }
}

impl Eq for ErrorItem {}

/// Returns `true` if the streams have the same tokens, apart from spans and spacing.
fn same_tokens(a: &TokenStream, b: &TokenStream) -> bool {
    let (mut left, mut right) = (Vec::new(), Vec::new());
    Token::collect(a.clone(), &mut left);
    Token::collect(b.clone(), &mut right);
    left == right
}

/// The message of the error.
impl std::fmt::Display for ErrorItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ErrorItem {}

/// Converts into a [`syn::Error`] at the span of the cause, with the notes and the help appended.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_trait(&parse_quote!(trait Tr { fn f(&self, a: MyType<'_>); }));
/// let error: Error = leaker.result().errors()[0].clone().into();
/// assert_eq!(
///     error.to_string(),
///     concat!(
///         "use absolute path: 'MyType<'_>'\n\n",
///         "  = note: it cannot be leaked as a whole because of `'_`\n",
///         "  = help: write `MyType` with an absolute path (e.g. `::my_crate::MyType`)\n",
///     )
/// );
/// ```
impl From<ErrorItem> for syn::Error {
    fn from(item: ErrorItem) -> Self {
        item.to_syn_error()
    }
}

/// Reads an [`ErrorItem`] written by its [`ToTokens`]; the tokens after it are not consumed. Fails
/// on an unknown kind or reason, and on a cause that is not the whole content of its parentheses.
impl Parse for ErrorItem {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        /// `( TOKENS ) "text"`.
        fn parse_at(input: ParseStream) -> syn::Result<(TokenStream, String)> {
            let tokens;
            parenthesized!(tokens in input);
            let tokens: TokenStream = tokens.parse()?;
            let text: LitStr = input.parse()?;
            Ok((tokens, text.value()))
        }
        let item;
        braced!(item in input);
        let kind = item.call(Ident::parse_any)?;
        let cause;
        parenthesized!(cause in item);
        let cause = Cause::parse(&kind, &cause)?;
        let reason = Reason::parse(&item.parse()?)?;
        let message = item.parse::<LitStr>()?.value();
        let list;
        bracketed!(list in item);
        let mut notes = Vec::new();
        while !list.is_empty() {
            notes.push(parse_at(&list)?);
            if !list.is_empty() {
                list.parse::<Token![,]>()?;
            }
        }
        let help = if item.peek(token::Paren) {
            Some(parse_at(&item)?)
        } else {
            None
        };
        let (mut place, mut paths) = (Place::default(), Vec::new());
        let mut link = Link::default();
        // The words of the link may be written once.
        let mut written: Vec<String> = Vec::new();
        while !item.is_empty() {
            let word = item.call(Ident::parse_any)?;
            let name = word.to_string();
            if matches!(name.as_str(), "id" | "part" | "entry" | "inner") {
                if written.contains(&name) {
                    return Err(syn::Error::new(
                        word.span(),
                        format!("`{}` is written twice", name),
                    ));
                }
                written.push(name.clone());
            }
            match name.as_str() {
                "id" => link.id = Some(item.parse::<LitInt>()?.base10_parse()?),
                "part" => link.part = Some(item.parse::<LitInt>()?.base10_parse()?),
                "entry" => link.entry = Some(item.parse::<LitInt>()?.base10_parse()?),
                "inner" => {
                    let list;
                    bracketed!(list in item);
                    let ids = Punctuated::<LitInt, Token![,]>::parse_terminated(&list)?;
                    for id in ids {
                        link.inner.push(id.base10_parse()?);
                    }
                }
                "value" => place.in_value = true,
                "default" => place.in_default = true,
                "paths" => {
                    let list;
                    bracketed!(list in item);
                    while !list.is_empty() {
                        let kind = list.call(Ident::parse_any)?;
                        let cause;
                        parenthesized!(cause in list);
                        let span = cause.span();
                        let entry = Cause::parse(&kind, &cause)?;
                        let bare = entry.path().map_or(false, |path| {
                            path.segments.iter().all(|seg| seg.arguments.is_none())
                        });
                        if !bare {
                            return Err(syn::Error::new(span, "expected a path without arguments"));
                        }
                        paths.push(entry);
                        if !list.is_empty() {
                            list.parse::<Token![,]>()?;
                        }
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        word.span(),
                        format!("unknown field `{}`", word),
                    ))
                }
            }
        }
        Ok(ErrorItem {
            cause,
            reason,
            message,
            notes,
            help,
            place,
            paths,
            link,
        })
    }
}

/// Writes the [`ErrorItem`] as one brace group, to be read by its [`Parse`]: a serialization, not a
/// report (see [`emit()`](ErrorItem::emit)).
impl ToTokens for ErrorItem {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let kind = Ident::new(self.cause.kind(), Span::call_site());
        let reason = Ident::new(self.reason.name(), Span::call_site());
        let cause = &self.cause;
        let message = &self.message;
        let notes: Vec<TokenStream> = self
            .notes
            .iter()
            .map(|(at, note)| quote!((#at) #note))
            .collect();
        let help = self.help.as_ref().map(|(at, help)| quote!((#at) #help));
        let value = self
            .place
            .in_value
            .then(|| Ident::new("value", Span::call_site()));
        let default = self
            .place
            .in_default
            .then(|| Ident::new("default", Span::call_site()));
        let paths = (!self.paths.is_empty()).then(|| {
            let paths: Vec<TokenStream> = self
                .paths
                .iter()
                .map(|path| {
                    let kind = Ident::new(path.kind(), Span::call_site());
                    quote!(#kind (#path))
                })
                .collect();
            quote!(paths [#(for path in &paths), { #path }])
        });
        let link = &self.link;
        let number = |n: usize| Literal::usize_unsuffixed(n);
        let id = link.id.map(|id| {
            let id = number(id);
            quote!(id #id)
        });
        let part = link.part.map(|part| {
            let part = number(part);
            quote!(part #part)
        });
        let entry = link.entry.map(|entry| {
            let entry = number(entry);
            quote!(entry #entry)
        });
        let inner = (!link.inner.is_empty()).then(|| {
            let ids: Vec<Literal> = link.inner.iter().map(|&id| number(id)).collect();
            quote!(inner [#(for id in &ids), { #id }])
        });
        tokens.extend(quote!({
            #kind (#cause) #reason #message [#(for note in &notes), { #note }] #help #value #default #paths
            #id #part #entry #inner
        }));
    }
}

pub(crate) const SELF_NOTE: &str =
    "`Self` has no meaning here: it is declared only by a struct, an enum, a \
                         union, an impl or a trait, not by a module or generics";

pub(crate) const SELF_HELP: &str = "write the type instead of `Self`";

/// The error of a mention of `Self` that has no meaning where it is written.
pub(crate) fn self_without_meaning(mention: &SelfMention) -> ErrorItem {
    let tokens = mention.cause.to_token_stream();
    ErrorItem {
        cause: mention.cause.clone(),
        reason: Reason::SelfWithoutMeaning,
        message: "`Self` has no meaning here".to_string(),
        notes: vec![(tokens.clone(), SELF_NOTE.to_string())],
        help: Some((tokens, SELF_HELP.to_string())),
        place: mention.place,
        paths: Vec::new(),
        link: Link::default(),
    }
}

/// The error of a conflict, reported at the first of its causes that depend on the context. A
/// conflict without any is no error.
pub(crate) fn report(conflict: &CheckResult) -> Option<ErrorItem> {
    let (first, others) = conflict.must.split_first()?;
    let tokens = first.shown();
    // The default that names `Self` is reported whole, with the note at the `Self` in it.
    let cause = conflict.cause.clone().unwrap_or_else(|| first.clone());
    // The conflict is not about the path if it has a note.
    let mut notes: Vec<_> = conflict
        .note
        .iter()
        .map(|note| (tokens.clone(), note.clone()))
        .collect();
    let mut default_object = false;
    // A cause that is the path itself (e.g. a trait bound) needs no note.
    for cause in conflict.impossible.iter().filter(|cause| *cause != first) {
        let shown = cause.shown();
        notes.push((
            shown.clone(),
            format!(
                "it cannot be leaked as a whole because of `{}`",
                tidy(&shown)
            ),
        ));
        if is_default_lifetime_object(cause) {
            default_object = true;
            notes.push((
                shown,
                "the lifetime of the trait object defaults to that of the enclosing \
                 reference or path"
                    .to_string(),
            ));
        }
    }
    let help = match &conflict.help {
        Some(help) => (tokens.clone(), help.clone()),
        None => absolute_path_help(first, default_object),
    };
    notes.extend(others.iter().map(|other| {
        let shown = other.shown();
        (
            shown.clone(),
            format!("`{}` also depends on the type context", tidy(&shown)),
        )
    }));
    Some(ErrorItem {
        cause,
        reason: conflict.reason,
        message: conflict
            .headline
            .clone()
            .unwrap_or_else(|| format!("use absolute path: '{}'", tidy(&tokens))),
        notes,
        help: Some(help),
        place: conflict.place,
        paths: conflict.paths.clone(),
        link: Link {
            raw: Raw {
                part: conflict.part.clone(),
                entry: conflict.origin.entry.clone(),
            },
            ..Link::default()
        },
    })
}

/// A trait object without a lifetime bound is a cause only under an elided lifetime, or where the
/// path cannot be leaked (see [Trait objects](crate::Leaker#trait-objects)).
fn is_default_lifetime_object(cause: &Cause) -> bool {
    match cause {
        Cause::Type(syn::Type::TraitObject(object)) => !has_lifetime_bound(object),
        _ => false,
    }
}

/// Returns the offending path of `cause`: the path of a type, a trait, an expression or a macro, or
/// the trait of a bound that a path does not hold (`for<'a> Fn(&'a u8)`, `(Clone)`).
fn offending_path(cause: &Cause) -> Option<syn::Path> {
    match cause {
        Cause::Type(syn::Type::Paren(paren)) => offending_path(&Cause::Type((*paren.elem).clone())),
        Cause::Type(syn::Type::Group(group)) => offending_path(&Cause::Type((*group.elem).clone())),
        Cause::Unreadable(tokens) => syn::parse2::<syn::Path>(tokens.clone()).ok().or_else(|| {
            match syn::parse2(tokens.clone()).ok()? {
                syn::TypeParamBound::Trait(bound) => Some(bound.path),
                _ => None,
            }
        }),
        other => other.path().cloned(),
    }
}

/// The help for the absolute path of a prelude item or a primitive type that `path` starts with.
fn prelude_help(path: &syn::Path) -> Option<String> {
    if path.leading_colon.is_some() {
        return None;
    }
    let name = path.segments.iter().next()?.ident.to_string();
    KNOWN_NAMES
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(name, absolute, _)| format!("use `{}` instead of `{}`", absolute, name))
}

/// The help for the path of `cause`: the absolute path of a prelude item, else how to write the
/// path in full, at its first segment. `default_object`: a trait object takes its lifetime from the
/// path around it.
fn absolute_path_help(cause: &Cause, default_object: bool) -> (TokenStream, String) {
    let tokens = cause.shown();
    let path = offending_path(cause);
    // The help is about the name of the path: its first segment.
    let at = path
        .as_ref()
        .and_then(|path| path.segments.first())
        .map_or_else(|| tokens.clone(), |seg| seg.ident.to_token_stream());
    let mut options = Vec::new();
    match path.as_ref().and_then(prelude_help) {
        Some(help) => options.push(help),
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
            }
            Some(path) => {
                let (name, _) = path_names(path);
                options.push(format!("write `{}` where it is not leaked", name));
            }
            None => options.push(format!(
                "write the paths in `{}` as absolute paths",
                tidy(&tokens)
            )),
        },
    }
    if default_object {
        options.push(
            "name the lifetime of the trait object (e.g. `&'a dyn Trait` or `dyn Trait + 'a`)"
                .to_string(),
        );
    }
    (at, or_list(options))
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
    // `core::fmt::Debug` is `::core::fmt::Debug`; `crate::m::X` and `super::m::X` are `m::X` in the
    // crate.
    let example =
        if path.leading_colon.is_none() && path.segments.len() > 1 && is_std_crate_path(path) {
            format!("::{}", names(path))
        } else {
            format!("::my_crate::{}", names(&without_crate_prefix(path)))
        };
    (name, example)
}

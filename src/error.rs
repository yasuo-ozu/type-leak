//! The errors of the analysis by the [`Leaker`](crate::Leaker).
//!
//! An analysis that cannot intern an item returns a `Vec<`[`ErrorItem`]`>`: one for each part of
//! the item that cannot be interned, with the part as written ([`Cause`]), why ([`Reason`]), a
//! message, notes and a help.

use crate::analysis::CheckResult;
use crate::util::{or_list, without_crate_prefix};
use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use std::hash::{Hash, Hasher};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    braced, bracketed, parenthesized, token, Expr, ExprPath, Ident, Lifetime, LitStr, Path, Token,
    Type,
};
use template_quote::{quote, ToTokens};

/// The part of an item that an [`ErrorItem`] is about, as written.
///
/// It is the part that the error is reported at, written back with [`ToTokens`], with the spans
/// it was parsed with. It is the first of the causes that depend on the context
/// ([`CheckResult::must`]), and its kind is that of the syntax which the Leaker found it in. Usually
/// it is the whole syntax that cannot be accepted, not a part of it: the entire type that has to be
/// leaked whole and cannot (`MyType<'_, Rel>` in `fn f(_: MyType<'_, Rel>)`, not the path
/// `MyType`), the whole trait path of a bound (`AsRef<Self>`), the whole path of a value
/// (`inner::N`). Where the error is reported at something smaller than the type that cannot be
/// leaked, that is the cause, and the others are notes (`u8` in `[u8; { let c = |a: u8| a; 3 }]`,
/// which the closure stops from being leaked, with `a` in a note).
///
/// So the trait of a `dyn` or `impl` type is a [`Cause::Trait`], a path in an expression inside a
/// type is a [`Cause::Expr`], a macro call there is a [`Cause::Macro`], and a lifetime is a
/// [`Cause::Lifetime`]. What is none of them is a [`Cause::Unreadable`].
///
/// Two causes are equal if they are of the same kind and are the same syntax, whatever their spans
/// and the spacing of their punctuation: the tokens are compared, a group by its delimiter, an
/// identifier by its name, a punctuation by its character and a literal by its text, so `: :` equals
/// `::` and `* ;` equals `*;`, and a group without delimiters (what a macro leaves around a
/// fragment) is no token. So a cause that is written by hand (`Cause::Trait(parse_quote!(RelTr))`)
/// is equal to the one an error has, and [`LeakerConfig::allow()`](crate::LeakerConfig::allow)
/// takes it to accept the conflict that the error reports.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::ToTokens;
/// let item: ItemTrait = parse_quote!(trait Tr { fn f(&self, a: MyType<'_, Rel>); });
/// let items = Leaker::new().intern_trait(&item, true).unwrap_err();
/// match &items[0].cause {
///     Cause::Type(ty) => assert_eq!(ty.to_token_stream().to_string(), "MyType < '_ , Rel >"),
///     other => panic!("unexpected cause {:?}", other),
/// }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Cause {
    /// A type that must be leaked whole and cannot (`MyType<'_, Rel>`, `Wrap<'x>`,
    /// `Rel<Self::A>`), the type of a const parameter (`const N: Rel`), `Self` or `Self::Item`
    /// where it has no meaning in a type, a type macro (`m!(Self)`).
    ///
    /// Where the error is reported at a type inside the type that cannot be leaked, the cause is
    /// that type: `Rel` in `&'b dyn ::path::Tr<Rel>`, `ILEN` in `::path::W<[Self], ILEN>` (where
    /// `Self` is denied), `Big` in the turbofish of `[u8; { |_: ::path::U| ::path::f::<Big>() }]`,
    /// `Rel<T>` in the input of `dyn Fn(&Rel<T>) -> &u8`.
    Type(Type),
    /// The trait of a bound, a supertrait, a where predicate, an impl or an `enter_trait` scope,
    /// of the `dyn` or `impl` type that holds it (`RelTrait` in `dyn RelTrait` or `impl
    /// RelTrait`), or of a qualified path in a value (`<u8 as Rel>::f()`).
    Trait(Path),
    /// An expression: a path in a const expression (an array length, `{ N }`, a discriminant, the
    /// default of a const parameter) or, with
    /// [`LeakerConfig::check_values()`](crate::LeakerConfig::check_values), in a value (a call, a
    /// struct literal, a variant, a pattern, the root of a `use`, a name in the arguments of a
    /// macro, `Self` in a value). It is an [`Expr::Path`](syn::Expr::Path), written as it was (no
    /// turbofish is added to its arguments).
    ///
    /// An expression that cannot be leaked is one too: the closure of `[u8; { let c = |a: u8| a;
    /// 3 }]`, an assignment, a macro call with its arguments (`two!()` in `[u8; two!()]`). A path
    /// inside it is its own cause (`a`, the [`Expr::Path`](syn::Expr::Path) in the closure).
    Expr(Expr),
    /// The path of a macro call in an expression, in a type or in a value (`foo!`, `two` in
    /// `[u8; two!()]`), written with its `!`. A macro that is a type is a [`Cause::Type`].
    Macro(Path),
    /// A lifetime: an elided one (`'_`), one of a method, one that no scope declares (`'a`).
    Lifetime(Lifetime),
    /// Syntax that the Leaker cannot read (`macro_rules!`, `use inner::*;`, a macro whose
    /// arguments are not expressions, syntax that `syn` leaves unparsed).
    ///
    /// It is also what an error is reported at when no node of the syntax holds it: a bound with a
    /// binder (`for<'a> Fn(&'a u8)`) or in parentheses (`(Clone)`), the path of a qualified type
    /// (`Tr<u8>::X` in `<Self as Tr<u8>>::X`).
    Unreadable(TokenStream),
}

impl Cause {
    /// The word that names the kind in the serialized form of an [`ErrorItem`] and of an
    /// [`Allowed`](crate::Allowed).
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Cause::Type(_) => "type",
            Cause::Trait(_) => "trait",
            Cause::Expr(_) => "expr",
            Cause::Macro(_) => "macro",
            Cause::Lifetime(_) => "lifetime",
            Cause::Unreadable(_) => "unreadable",
        }
    }

    /// The cause that is the path of a type (`Rel`, `Self::Item`).
    pub(crate) fn type_path(path: &Path) -> Self {
        Cause::Type(Type::Path(syn::TypePath {
            qself: None,
            path: path.clone(),
        }))
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

    /// The cause as the messages, the notes and the help show it: as the cause is written, but for
    /// a macro, which they show by its path (`two`, not `two !`).
    pub(crate) fn shown(&self) -> TokenStream {
        match self {
            Cause::Macro(path) => path.to_token_stream(),
            other => other.to_token_stream(),
        }
    }

    /// The path that is the cause, if it is one: the path of a type, a trait, an expression or a
    /// macro, without a `qself`.
    pub(crate) fn as_path(&self) -> Option<&Path> {
        match self {
            Cause::Type(Type::Path(syn::TypePath { qself: None, path }))
            | Cause::Trait(path)
            | Cause::Macro(path)
            | Cause::Expr(Expr::Path(ExprPath {
                qself: None, path, ..
            })) => Some(path),
            _ => None,
        }
    }

    /// Returns `true` if the cause is `Self` or a path that starts with it (`Self::A`): not
    /// `<Self as Tr>::A`, whose `Self` is a type of its own.
    pub(crate) fn is_self_path(&self) -> bool {
        match self {
            Cause::Macro(_) => false,
            other => other.as_path().map_or(false, |path| {
                path.leading_colon.is_none()
                    && path
                        .segments
                        .first()
                        .map_or(false, |seg| seg.ident == "Self")
            }),
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

/// Writes the syntax as it was written, with its spans. A [`Cause::Macro`] has its `!`.
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
            Cause::Macro(path) => tokens.extend(quote!(#path !)),
            Cause::Lifetime(lifetime) => lifetime.to_tokens(tokens),
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

/// Two causes are equal if they are of the same kind and have the same tokens, whatever their
/// spans and the spacing of their punctuation.
impl PartialEq for Cause {
    fn eq(&self, other: &Self) -> bool {
        self.kind() == other.kind() && self.canonical() == other.canonical()
    }
}

impl Eq for Cause {}

/// Hashes the kind and the tokens as `eq()` compares them, so equal causes hash alike.
impl Hash for Cause {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind().hash(state);
        self.canonical().hash(state);
    }
}

/// Why a part of an item cannot be interned: the rule that it breaks.
///
/// Every [`ErrorItem`] has one. The message of the item is the headline that the reason names, and
/// `use absolute path: '..'` for [`Reason::RelativePath`].
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let item: ItemTrait = parse_quote!(trait Tr { fn f(&self, a: Wrap<'_, Self>); });
/// let items = Leaker::new().intern_trait(&item, false).unwrap_err();
/// assert_eq!(items[0].reason, Reason::RelativePath);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Reason {
    /// A path or type that depends on the context but cannot be leaked, so the code must change:
    /// usually, the path must be written in full. The message is `use absolute path: '..'`.
    ///
    /// Examples: `MyType` in `MyType<'_>`, a trait in a bound, a value in a const expression, `Rel`
    /// in `F: for<'x> Fn(Rel<'x>) -> &'x u8` ([Binder
    /// lifetimes](crate::Leaker#binder-lifetimes)), and `Rel<T>` in `F: Fn(&Rel<T>) -> &u8` where
    /// `Rel<T>` has lifetimes among its generics ([Elided
    /// lifetimes](crate::Leaker#elided-lifetimes)). With
    /// [`LeakerConfig::check_values()`](crate::LeakerConfig::check_values) it is also a relative
    /// path in a value: the default of an associated const, the value of a const or a static, the
    /// body of a function. The note says which value it is in ("it is in the default value of `N`,
    /// which the copy writes as it is"). A value is copied as written, so nothing but a type in a
    /// type position can be leaked there (a cast, a `let`, a closure, a turbofish, a qualified
    /// path: a relative one there is an entry, not an error).
    RelativePath,
    /// `Self` where no scope gives it a meaning, with the message "`Self` has no meaning here".
    ///
    /// That is directly in a module, or in generics that no struct, enum, union, impl or trait
    /// encloses (`Leaker::enter_mod()`, `Leaker::enter_generics()`). It is also an error in the
    /// bounds, where predicates and defaults of those generics. The cause is the mention of `Self`
    /// (`Self`, `Self::Item`, or a type macro that names it), and the help says to write the type,
    /// or to enter a scope that declares `Self`. See
    /// [`Leaker::enter_generics()`](crate::Leaker::enter_generics).
    SelfWithoutMeaning,
    /// Syntax that the Leaker cannot read, with
    /// [`LeakerConfig::check_values()`](crate::LeakerConfig::check_values), with the message
    /// "cannot analyze `..` in the body of ..": a macro whose arguments are not expressions, a
    /// `macro_rules!`, a glob import, syntax that `syn` leaves unparsed.
    Unreadable,
    /// A bare identifier among the arguments of a path, which may be a const, in a type that cannot
    /// be leaked whole. The message is "`N` may be a const, and the type that holds it cannot be
    /// leaked as a whole". See [Bare identifiers and
    /// consts](crate::Leaker#bare-identifiers-and-consts).
    MaybeConst,
    /// A type leaked as a whole that names a lifetime of a method which the copy would leave
    /// unconstrained (`Wrap<'x>` in `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8`), for a method of a
    /// trait or of an impl of a trait, not for a free function or a method of an inherent impl
    /// ([Method lifetimes](crate::Leaker#method-lifetimes)). The fault is the lifetime, which an
    /// absolute path is only one way to fix. The message is "`Wrap<'x>` leaves the lifetime `'x`
    /// of the method unconstrained in the copy (E0195)", with a note on E0195.
    UnconstrainedLifetime,
    /// A type leaked as a whole that names `Self::A` where `Self` is a type (an impl, a struct, an
    /// enum or a union), which the Repeater's impl cannot write. The message says what to write:
    /// "write `<Self as Tr>::A` instead of `Self::A` in `Wrap<Self::A>`: the copy cannot tell
    /// which trait `A` comes from", with the trait of the impl as written (`SomeTrait` for a
    /// struct, an enum, a union or an inherent impl, which have none). The help gives the path in
    /// full (`<Self as ::my_crate::Tr>::A` if the trait is relative).
    SelfProjection,
    /// A type with a generic whose bound names the `Self` of a trait (or `Self::X`), where `Self`
    /// is something else, so the bound cannot be written
    /// ([`View::visit_item()`](crate::View::visit_item)). The fault is that bound, not the path of
    /// the type. The message is "the bound of `T` mentions a `Self` which has no name here" (or
    /// "mentions `Self::X`, which has no name here"), with a note on why, at the type.
    UnnameableSelf,
    /// A type in the default of a parameter that names `Self`, which the default cannot do. The
    /// message is "`Wrap<Self>` is in the default of a parameter, and names `Self`". The cause is
    /// the type of the default, and the note is at the `Self` in it.
    ///
    /// It is also the default of a parameter of a struct, an enum or a union nested in the
    /// analyzed syntax, when a bound of the generics names `Self` too: the type in it that does not
    /// name `Self` is fine whatever the bounds of the generics say (the bounds that name `Self` are
    /// left out of its context), and one that names it is an error, the message being "`Wrap<Self,
    /// T>` is in the default of a parameter, and it and a bound of its generics name `Self`"
    /// (E0735).
    SelfInDefault,
    /// A type that uses a generic whose bound mentions a `Self` that cannot be leaked, because
    /// `allow_self` is `false`. The message is "`Wrap<T>` has a generic whose bound mentions
    /// `Self`, which cannot be leaked".
    DeniedSelfInBounds,
    /// A type that uses a generic whose bound mentions the `Self` of a struct, an enum or a union
    /// that [`View::visit_item()`](crate::View::visit_item) visits, and that names it too
    /// (`Wrap<Self, T>` with `T: AsRef<Self>`), with the message of
    /// [`Reason::DeniedSelfInBounds`]. The Repeater's impl of the type is for the Leaker's own
    /// type, and the copy declares a type of its own, which has the bound for itself. A type that does
    /// not name it (`Wrap<T>`) is fine: its impl is for any type that the bound holds for
    /// ([`SelfRequirement::Any`](crate::SelfRequirement)). So is an absolute type that names it
    /// (`::path::Wrap<Self, T>`): it is copied as written, and its `Self` is the copy's own. Write
    /// the type without `Self` or with an absolute path, analyze the item with its own `intern_*`
    /// call, or accept the entry for any type with
    /// [`LeakerConfig::nested_self_entries()`](crate::LeakerConfig::nested_self_entries).
    NestedSelfInBounds,
    /// A type in a bound or where predicate that the copy cannot write through the Repeater,
    /// because rustc cannot prove a bound of its generics that names a leaked type, `Self`,
    /// `Self::A` or an associated type that rustc normalizes ([Bounds that name leaked
    /// types](crate::Leaker#bounds-that-name-leaked-types)). The message is "`Wrap<T>` is in a
    /// bound or where predicate, and a bound of its generics names a leaked type".
    UnprovableBound,
    /// A type that names `Self` in a bound or where predicate of its trait, with the message
    /// "`Rel<Self>` names `Self` in a bound or where predicate of its trait" (E0275).
    SelfInTraitPredicate,
    /// A type in the self type of an impl whose context depends on `Self` through a bound of the
    /// impl ([`Leaker::intern_impl()`](crate::Leaker::intern_impl)). The message is "`S<T>` is in
    /// the self type of the impl, and a bound of its generics names `Self`".
    SelfInImplSelfType,
}

impl Reason {
    /// Every reason.
    const ALL: [Reason; 13] = [
        Reason::RelativePath,
        Reason::SelfWithoutMeaning,
        Reason::Unreadable,
        Reason::MaybeConst,
        Reason::UnconstrainedLifetime,
        Reason::SelfProjection,
        Reason::UnnameableSelf,
        Reason::SelfInDefault,
        Reason::DeniedSelfInBounds,
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
            Reason::DeniedSelfInBounds => "DeniedSelfInBounds",
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
/// A failing analysis ([`Leaker::intern_struct()`](crate::Leaker::intern_struct) and the other
/// `intern_*` methods, the `enter_*` methods) returns a `Vec<ErrorItem>`, one for each offending
/// part, in order, so a macro can report them all, each as a separate diagnostic with its own span,
/// notes and help. They are laid out as rustc lays out its own errors: the `message` says what is
/// wrong, the `notes` say why, and the `help` says what to write instead, with the tokens of the
/// error. Every error has a help. The message says `use absolute path: '..'` where the fault is
/// that a path is relative. It is the headline of the cause where the fault is something else, such
/// as `Self`, a bound rule or a default that names `Self` (see [`Reason`]).
///
/// The `cause` is the part as written that the error is reported at (see [`Cause`]): usually the
/// whole type, trait path or value that cannot be accepted. The notes say why: the causes that stop
/// the type from being leaked as a whole, and the other paths in it that depend on the context,
/// which would be reported once the first is fixed. Each note and the help carry the tokens they
/// are about, so they can be reported at their own spans.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::ToTokens;
/// let item: ItemTrait = parse_quote!(trait Tr: Clone { fn f(&self, a: MyType<'_>); });
/// let items = Leaker::new().intern_trait(&item, true).unwrap_err();
/// assert_eq!(items.len(), 2);
/// assert_eq!(items[0].message, "use absolute path: 'Clone'");
/// assert_eq!(items[0].reason, Reason::RelativePath);
/// assert_eq!(items[0].help_text(), Some("use `::core::clone::Clone` instead of `Clone`"));
/// assert_eq!(items[1].message, "use absolute path: 'MyType < '_ >'");
/// assert_eq!(items[1].cause.to_token_stream().to_string(), "MyType < '_ >");
/// assert_eq!(items[1].note(), Some("it cannot be leaked as a whole because of `'_`"));
/// assert_eq!(
///     items[1].help_text(),
///     Some("write `MyType` with an absolute path (e.g. `::my_crate::MyType`), or allow it with \
///           `LeakerConfig::allowed_paths`")
/// );
/// // `Display` is the message.
/// assert_eq!(items[1].to_string(), items[1].message);
/// ```
///
/// # Reporting the errors from a proc macro
///
/// [`ErrorItem::to_compile_error()`] reports an item, and so does [`to_syn_error()`] for all of
/// them, as `compile_error!` invocations with `= note:` lines and a `= help:` line after the
/// message ([`syn::Error`] is what they build, also with `err.into()` for one item). On nightly,
/// [`ErrorItem::to_compile_error()`] emits a `Diagnostic` instead, with the notes and the help at
/// their own spans. Do not emit an item with its [`ToTokens`], which is the
/// [serialization](#serialization).
///
/// A macro that uses [`proc-macro-error`](https://docs.rs/proc-macro-error) reports each item
/// itself, with the notes as `note`s and the help as the `help` (the `hint` of an older version)
/// of the error:
///
/// ```text
/// for item in &items {
///     let help = item.help_text().unwrap_or_default();
///     match item.note() {
///         Some(note) => emit_error!(item.span(), "{}", item.message; note = "{}", note; help = "{}", help),
///         None => emit_error!(item.span(), "{}", item.message; help = "{}", help),
///     }
/// }
/// abort_if_dirty();
/// ```
///
/// or, to keep the span of every note and of the help (the first segment of a path to write in
/// full, for example), with its `Diagnostic`:
///
/// ```text
/// for item in items {
///     let mut diagnostic = Diagnostic::spanned(item.span(), Level::Error, item.message);
///     for (at, note) in item.notes {
///         diagnostic = diagnostic.span_note(at.span(), note);
///     }
///     if let Some((at, help)) = item.help {
///         diagnostic = diagnostic.span_help(at.span(), help);
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
/// let items = Leaker::new().intern_trait(&item, true).unwrap_err();
/// let item = &items[0];
/// let mut message = item.message.clone();
/// for (_at, note) in &item.notes {
///     message += &format!("\n  = note: {}", note);
/// }
/// if let Some((_at, help)) = &item.help {
///     message += &format!("\n  = help: {}", help);
/// }
/// let error = syn::Error::new(item.span(), message);
/// assert!(error.to_string().ends_with("= help: use `::core::clone::Clone` instead of `Clone`"));
/// ```
///
/// # Serialization
///
/// [`ToTokens`] and [`Parse`] write an item as tokens and read it back, so a macro can pass its
/// errors on to another macro and report them there. The tokens of the cause, the notes and the
/// help are written as they are, so they keep their spans, and only the string literals get new
/// ones. There is no format tag: both macros must use the same version of type-leak. The grammar is
/// one brace group for each item, so items follow one another and are read in a loop
/// (`while !input.is_empty() { items.push(input.parse()?) }`):
///
/// ```text
/// { KIND ( CAUSE ) REASON "message" [ ( TOKENS ) "note" , .. ] ( TOKENS ) "help" }
/// KIND   := type | trait | expr | macro | lifetime | unreadable
/// REASON := the name of the variant of Reason (RelativePath, ..)
/// ```
///
/// - The help is absent if the item has none. The list of notes may be empty (`[]`), and a trailing
///   comma is accepted.
/// - `CAUSE` is a type for `type`, a path for `trait` (with arguments, and the sugar of `Fn(..) ->
///   ..`), an expression for `expr` (a path as a type has it, `Rel<u8>`, is the path of an
///   expression too), a path followed by `!` for `macro`, a lifetime for `lifetime`, and any
///   tokens for `unreadable`. It must be the whole content of its parentheses.
/// - An unknown kind or reason is a parse error that names it.
///
/// ```
/// # use type_leak::*;
/// # use syn::{parse::Parser, *};
/// # use template_quote::ToTokens;
/// let item: ItemTrait = parse_quote!(trait Tr: Clone { fn f(&self, a: MyType<'_>); });
/// let items = Leaker::new().intern_trait(&item, true).unwrap_err();
/// // The errors of one macro, as tokens, and back in the other.
/// let tokens: proc_macro2::TokenStream =
///     items.iter().map(|item| item.to_token_stream()).collect();
/// let parsed = (|input: parse::ParseStream| -> Result<Vec<ErrorItem>> {
///     let mut items = Vec::new();
///     while !input.is_empty() {
///         items.push(input.parse()?);
///     }
///     Ok(items)
/// })
/// .parse2(tokens)
/// .unwrap();
/// assert_eq!(parsed.len(), 2);
/// assert_eq!(parsed[1].message, items[1].message);
/// assert_eq!(parsed[1].reason, items[1].reason);
/// let notes = |item: &ErrorItem| -> Vec<(String, String)> {
///     item.notes.iter().map(|(at, note)| (at.to_string(), note.clone())).collect()
/// };
/// assert_eq!(notes(&parsed[1]), notes(&items[1]));
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ErrorItem {
    /// The part of the item that cannot be interned, as written: usually the whole type, trait
    /// path or value, else the part of it that depends on the context. The error is reported at it
    /// ([`ErrorItem::span()`]). Passing it to [`LeakerConfig::allow()`](crate::LeakerConfig::allow)
    /// accepts the conflict, so that the analysis does not report it.
    pub cause: Cause,
    /// Why it cannot be interned.
    pub reason: Reason,
    /// What is wrong: the headline of the reason where the fault is not that the path is relative
    /// ([`CheckResult::headline`]; "`Self` has no meaning here"), and `use absolute path: '..'`
    /// where it is.
    pub message: String,
    /// Why, each with the tokens it is about, the most useful first: what the headline says in more
    /// words, the causes that stop the path from being leaked as a whole, and the other paths that
    /// depend on the type context.
    pub notes: Vec<(TokenStream, String)>,
    /// What to write instead, with the tokens it is about, as the tokens of the error say it (for
    /// example the absolute path of a prelude name: "use `::std::vec::Vec` instead of `Vec`").
    /// Every error has one.
    pub help: Option<(TokenStream, String)>,
}

impl ErrorItem {
    /// Returns the span of the cause, where the error is reported.
    pub fn span(&self) -> Span {
        self.cause.to_token_stream().span()
    }

    /// Returns the first note, which says why, if there is one.
    ///
    /// A conflict about the path itself has none unless a cause stops it from being leaked as a
    /// whole. For [`Reason::SelfWithoutMeaning`] it tells which items declare `Self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S { a: MyType<'_> });
    /// let items = Leaker::new().intern_struct(&item, true).unwrap_err();
    /// assert_eq!(items[0].message, "use absolute path: 'MyType < '_ >'");
    /// assert_eq!(items[0].note(), Some("it cannot be leaked as a whole because of `'_`"));
    /// ```
    pub fn note(&self) -> Option<&str> {
        self.notes.first().map(|(_, note)| note.as_str())
    }

    /// Returns the text of the help, which says what to write instead.
    ///
    /// For [`Reason::RelativePath`] it is the absolute path to use instead of the first segment, if
    /// that is a prelude item or a primitive type, and else how to write the path. For
    /// [`Reason::SelfWithoutMeaning`] it tells which scopes declare `Self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait MyTrait: Clone {});
    /// let items = Leaker::new().intern_trait(&item, true).unwrap_err();
    /// assert_eq!(items[0].message, "use absolute path: 'Clone'");
    /// assert_eq!(
    ///     items[0].help_text(),
    ///     Some("use `::core::clone::Clone` instead of `Clone`")
    /// );
    /// ```
    pub fn help_text(&self) -> Option<&str> {
        self.help.as_ref().map(|(_, help)| help.as_str())
    }

    /// Reports the error from a proc macro: tokens that emit it as a `compile_error!`, with the
    /// notes and the help rendered as for [`syn::Error`] (`From<ErrorItem>`). A macro can return
    /// `item.to_compile_error()` among its output.
    ///
    /// On nightly, when `proc_macro::Diagnostic` is available inside a proc macro, the error is
    /// emitted as a diagnostic instead, with the notes and the help at their own spans (e.g. the
    /// first segment of the offending path), and no tokens are produced.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait Tr: Clone {});
    /// let items = Leaker::new().intern_trait(&item, true).unwrap_err();
    /// let tokens = items[0].to_compile_error().to_string();
    /// assert!(tokens.contains("compile_error"));
    /// assert!(tokens.contains("use absolute path: 'Clone'"));
    /// ```
    pub fn to_compile_error(&self) -> TokenStream {
        // Only compiled on nightly, so the MSRV does not apply.
        #[cfg(type_leak_diagnostic)]
        #[allow(clippy::incompatible_msrv)]
        if proc_macro::is_available() {
            self.emit_diagnostic();
            return TokenStream::new();
        }
        self.to_syn_error().to_compile_error()
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

/// Converts the errors into one [`syn::Error`], with one error per item combined by
/// [`syn::Error::combine()`]. An empty slice gives an error at the call site.
///
/// Stable Rust has no sub-diagnostics for proc macros, so each note is appended to the message
/// (`= note: ..`), then the help (`= help: ..`).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let item: ItemTrait = parse_quote!(trait Tr: Clone { fn f(&self, a: MyType<'_>); });
/// let items = Leaker::new().intern_trait(&item, true).unwrap_err();
/// let error = to_syn_error(&items);
/// assert_eq!(error.into_iter().count(), 2);
/// ```
pub fn to_syn_error(items: &[ErrorItem]) -> syn::Error {
    items
        .iter()
        .map(ErrorItem::to_syn_error)
        .reduce(|mut combined, error| {
            combined.combine(error);
            combined
        })
        .unwrap_or_else(|| syn::Error::new(Span::call_site(), "not internable"))
}

/// The message of the error.
impl std::fmt::Display for ErrorItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ErrorItem {}

/// Converts into a [`syn::Error`] at the span of the cause, with the notes and the help appended to
/// the message. [`to_syn_error()`] combines several.
impl From<ErrorItem> for syn::Error {
    fn from(item: ErrorItem) -> Self {
        item.to_syn_error()
    }
}

/// Reads an [`ErrorItem`] written by its [`ToTokens`] implementation. The tokens after it are not
/// consumed. Fails on an unknown kind or reason, naming it, and on a cause that is not the whole
/// content of its parentheses.
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
        let help = if item.is_empty() {
            None
        } else {
            Some(parse_at(&item)?)
        };
        if !item.is_empty() {
            return Err(item.error("unexpected token after the help"));
        }
        Ok(ErrorItem {
            cause,
            reason,
            message,
            notes,
            help,
        })
    }
}

/// Writes the [`ErrorItem`] as tokens, to be read back by its [`Parse`] implementation. This is a
/// serialization, not a report: use [`ErrorItem::to_compile_error()`] to report the error.
///
/// Treat the tokens as opaque (see [`ErrorItem`] for the grammar).
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
        tokens.extend(quote!({
            #kind (#cause) #reason #message [#(for note in &notes), { #note }] #help
        }));
    }
}

pub(crate) const SELF_NOTE: &str =
    "`Self` has no meaning here: it is declared only by a struct, an enum, a \
                         union, an impl or a trait, not by a module or generics";

pub(crate) const SELF_HELP: &str =
    "write the type instead of `Self`, or enter a scope that declares `Self` around it \
     (`Leaker::enter_type`, `enter_impl` or `enter_trait`)";

/// The error of a mention of `Self` that has no meaning where it is written.
pub(crate) fn self_without_meaning(cause: &Cause) -> ErrorItem {
    let tokens = cause.to_token_stream();
    ErrorItem {
        cause: cause.clone(),
        reason: Reason::SelfWithoutMeaning,
        message: "`Self` has no meaning here".to_string(),
        notes: vec![(tokens.clone(), SELF_NOTE.to_string())],
        help: Some((tokens, SELF_HELP.to_string())),
    }
}

/// The error of a conflict, reported at the first of its causes that depend on the context, which is
/// the cause of the error. A conflict without any is no error.
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
            format!("it cannot be leaked as a whole because of `{}`", shown),
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
        None => absolute_path_help(
            first,
            default_object,
            conflict.impossible.iter().any(Cause::is_self_path),
        ),
    };
    notes.extend(others.iter().map(|other| {
        let shown = other.shown();
        (
            shown.clone(),
            format!("`{}` also depends on the type context", shown),
        )
    }));
    Some(ErrorItem {
        cause,
        reason: conflict.reason,
        message: conflict
            .headline
            .clone()
            .unwrap_or_else(|| format!("use absolute path: '{}'", tokens)),
        notes,
        help: Some(help),
    })
}

/// A trait object without a lifetime bound is a cause only under an elided lifetime, or where the
/// path cannot be leaked (see [Trait objects](crate::Leaker#trait-objects)).
fn is_default_lifetime_object(cause: &Cause) -> bool {
    match cause {
        Cause::Type(syn::Type::TraitObject(object)) => !object
            .bounds
            .iter()
            .any(|bound| matches!(bound, syn::TypeParamBound::Lifetime(_))),
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
        other => other.as_path().cloned(),
    }
}

/// The help for the absolute path of a prelude item or a primitive type that `path` starts with.
fn prelude_help(path: &syn::Path) -> Option<String> {
    if path.leading_colon.is_some() {
        return None;
    }
    let name = path.segments.iter().next()?.ident.to_string();
    ABSOLUTE_PATHS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(name, absolute)| format!("use `{}` instead of `{}`", absolute, name))
}

/// What to write for the path `tokens` that depends on the context, when the conflict has nothing
/// more to say: the absolute path of a prelude item, else how to write the path in full, at the
/// name of the path (its first segment). `default_object`: a trait object takes its lifetime from
/// the reference or path around it. `denied_self`: a `Self` that is denied (`allow_self = false`)
/// is among the causes.
fn absolute_path_help(
    cause: &Cause,
    default_object: bool,
    denied_self: bool,
) -> (TokenStream, String) {
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
                crate::util::tidy(&tokens)
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
    // `crate::m::X` and `super::m::X` are `m::X` in the crate.
    let example = format!("::my_crate::{}", names(&without_crate_prefix(path)));
    (name, example)
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

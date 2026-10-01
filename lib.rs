#![doc = include_str!("./README.md")]

use proc_macro2::{Span, TokenStream};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};

pub use syn;

/// Configuration for building a [`Leaker`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LeakerConfig {
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
}

impl Default for LeakerConfig {
    fn default() -> Self {
        Self {
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
        }
    }
}

impl LeakerConfig {
    pub fn new() -> Self {
        Default::default()
    }

    /// Allow primitive types from `::core::primitive`.
    pub fn allow_primitive(&mut self) -> &mut Self {
        self.allowed_paths.extend([
            parse_quote!(bool),
            parse_quote!(char),
            parse_quote!(str),
            parse_quote!(u8),
            parse_quote!(u16),
            parse_quote!(u32),
            parse_quote!(u64),
            parse_quote!(u128),
            parse_quote!(usize),
            parse_quote!(i8),
            parse_quote!(i16),
            parse_quote!(i32),
            parse_quote!(i64),
            parse_quote!(i128),
            parse_quote!(isize),
            parse_quote!(f32),
            parse_quote!(f64),
        ]);
        self
    }

    /// Allow items in the standard prelude (`std::prelude::v1`).
    pub fn allow_std_prelude(&mut self) -> &mut Self {
        self.allowed_paths.extend([
            parse_quote!(Box),
            parse_quote!(String),
            parse_quote!(Vec),
            parse_quote!(Option),
            parse_quote!(Result),
            parse_quote!(Copy),
            parse_quote!(Send),
            parse_quote!(Sized),
            parse_quote!(Sync),
            parse_quote!(Unpin),
            parse_quote!(Drop),
            parse_quote!(Fn),
            parse_quote!(FnMut),
            parse_quote!(FnOnce),
            parse_quote!(Clone),
            parse_quote!(PartialEq),
            parse_quote!(PartialOrd),
            parse_quote!(Eq),
            parse_quote!(Ord),
            parse_quote!(AsRef),
            parse_quote!(AsMut),
            parse_quote!(Into),
            parse_quote!(From),
            parse_quote!(Default),
            parse_quote!(Iterator),
            parse_quote!(IntoIterator),
            parse_quote!(Extend),
            parse_quote!(FromIterator),
            parse_quote!(ToOwned),
            parse_quote!(ToString),
            parse_quote!(drop),
        ]);
        self
    }

    /// Allow paths starting with `crate`.
    pub fn allow_crate(&mut self) -> &mut Self {
        self.allowed_paths.push(parse_quote!(crate));
        self
    }
}

/// The entry point of this crate.
pub struct Leaker {
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
    /// All the operations collected by [`Leaker::intern_with()`].
    operations: Vec<GraphOperation>,
    /// The types to be interned, computed from `operations` by [`Leaker::reduce_roots()`] (`None`
    /// if not computed yet, or outdated).
    reachable_types: Option<Vec<Type>>,
    local_generics: HashMap<Type, Generics>,
    /// Types interned without local generics, to detect conflicts with `local_generics`.
    plain_types: HashSet<Type>,
    /// Names of primitive types which were assumed to be the primitive types (not shadowed), as
    /// they are copied as they are (see [`Leaker::assertions()`]).
    assumed_primitives: Vec<Ident>,
}

/// Error indicating that a type is not internable: the offending tokens and the reason.
///
/// [`NotInternableError::note()`] suggests how to fix it, e.g. "use `::std::vec::Vec` instead of
/// `Vec`" when the offending path starts with an item of the standard prelude or a primitive type.
/// Converting the error into [`syn::Error`] renders it as a `note` of the diagnostic.
#[derive(Debug, Clone)]
pub struct NotInternableError(pub TokenStream, pub NotInternableReason);

/// The reason of a [`NotInternableError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NotInternableReason {
    /// The path depends on the type context, but cannot be leaked (e.g. a trait in a bound, or a
    /// value in a const expression), so it should be absolute.
    RelativePath,
    /// The type is used with different method-local generics (or with and without them) in
    /// different places, so a Referrer cannot tell which one a copy refers to. (The Referrer
    /// finds leaked types by the type alone; see [`Referrer::local_generics()`].)
    ConflictingLocalGenerics,
}

impl std::fmt::Display for NotInternableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.1 {
            NotInternableReason::RelativePath => write!(f, "use absolute path: '{}'", self.0),
            NotInternableReason::ConflictingLocalGenerics => write!(
                f,
                "'{}' is used with different method-local generics, which cannot be leaked",
                self.0
            ),
        }
    }
}

impl std::error::Error for NotInternableError {}

impl NotInternableError {
    pub fn span(&self) -> Span {
        self.0.span()
    }

    /// Returns a note suggesting how to fix the error: for [`NotInternableReason::RelativePath`],
    /// the absolute path to use instead of the first segment of the offending path, if it is an
    /// item of the standard prelude or a primitive type.
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
        if self.1 == NotInternableReason::ConflictingLocalGenerics {
            return Some(
                "use an absolute path for it, or the same method-local generics wherever it is \
                 used"
                    .to_string(),
            );
        }
        let name = first_path_ident(self.0.clone())?.to_string();
        ABSOLUTE_PATHS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(name, absolute)| format!("use `{}` instead of `{}`", absolute, name))
    }
}

/// Converts into a [`syn::Error`] at [`NotInternableError::span()`], with the
/// [`NotInternableError::note()`] rendered as a `note` of the diagnostic.
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
        let message = match err.note() {
            Some(note) => format!("{}\n\n  = note: {}\n", err, note),
            None => err.to_string(),
        };
        syn::Error::new(err.span(), message)
    }
}

/// Items of the standard prelude and primitive types, with their absolute paths.
const ABSOLUTE_PATHS: &[(&str, &str)] = &[
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

/// Returns the first segment of the first relative path in `tokens`, skipping keywords, sigils
/// and lifetimes before it (e.g. `Iterator` in `impl for<'a> Iterator<Item = &'a u8>`).
fn first_path_ident(tokens: TokenStream) -> Option<proc_macro2::Ident> {
    let mut prev_punct: Option<char> = None;
    for tt in tokens {
        match tt {
            proc_macro2::TokenTree::Ident(ident) => {
                let s = ident.to_string();
                match (prev_punct, s.as_str()) {
                    // A lifetime (`'a`), or a segment after `::` (e.g. an absolute path).
                    (Some('\''), _) | (Some(':'), _) => {
                        if prev_punct == Some(':') {
                            return None;
                        }
                    }
                    (_, "impl" | "dyn" | "mut" | "for" | "const" | "unsafe" | "extern") => (),
                    _ => return Some(ident),
                }
                prev_punct = None;
            }
            proc_macro2::TokenTree::Punct(punct) => prev_punct = Some(punct.as_char()),
            proc_macro2::TokenTree::Group(_) | proc_macro2::TokenTree::Literal(_) => {
                prev_punct = None
            }
        }
    }
    None
}

/// Encode [`GenericArgument`]s to a type.
///
/// A const argument `N` is encoded as `[(); N]`, so it must be of type `usize`. To encode generics
/// of any kind, use [`RepeaterSlots`]. Associated item bindings and constraints (e.g. `Item = u8`)
/// are skipped.
///
/// Without the corresponding [`GenericParam`]s, the arguments are encoded as they are, so omitted
/// arguments are not filled with the parameters' defaults, and a bare identifier such as `N` in
/// `MyTrait<N>` is encoded as a type even if it refers to a const (syn parses it as a type
/// argument). So the result may not match [`encode_generics_params_to_ty()`]; use
/// [`encode_generics_args_to_ty_with()`] with the parameters instead.
#[deprecated(
    since = "0.8.0",
    note = "cannot handle bare const arguments and omitted defaults without the parameters; use \
            `encode_generics_args_to_ty_with()` with the parameters"
)]
pub fn encode_generics_args_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericArgument>) -> Type {
    encode_generics_args_to_ty_with(iter, None, |expr, _| parse_quote!([(); #expr]))
        .expect("encoding without parameters never fails")
}

/// Encode [`GenericArgument`]s to a type, using `const_fn` to encode const arguments.
///
/// `const_fn` receives each const argument together with its corresponding [`ConstParam`] (if
/// known), and must return the same type as the `const_fn` passed to
/// [`encode_generics_params_to_ty_with()`] for the same value. Associated item bindings and
/// constraints (e.g. `Item = u8`) are skipped.
///
/// If `params` (the [`GenericParam`]s corresponding to `iter`) is given, the result is the same
/// type as [`encode_generics_params_to_ty_with()`] with `params` substituted by `iter`, as
/// [`RepeaterSlots::encode_generics_args_to_ty()`] does: arguments are matched with parameters by
/// position, a bare identifier corresponding to a const parameter (e.g. `N` in `MyTrait<N>`) is
/// treated as a const argument, omitted arguments take the parameters' defaults, and
/// [`EncodeError`] is returned if there are too many arguments or a missing argument without a
/// default. Without `params`, the arguments are encoded as they are, and this never fails.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::quote;
/// let generics: Generics = parse_quote!(<T, const N: usize, const M: usize = 3>);
/// let args: AngleBracketedGenericArguments = parse_quote!(<u8, N>);
/// let ty = encode_generics_args_to_ty_with(&args.args, Some(&generics.params), |expr, _| {
///     parse_quote!([(); #expr])
/// })
/// .unwrap();
/// let expected: Type = parse_quote!((u8, [(); N], [(); 3]));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
/// ```
pub fn encode_generics_args_to_ty_with<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: Option<&'b Punctuated<GenericParam, Token![,]>>,
    mut const_fn: impl FnMut(&Expr, Option<&'b ConstParam>) -> Type,
) -> std::result::Result<Type, EncodeError> {
    if let Some(params) = params {
        return encode_generics_args_with_params(iter, params, |value, cp| {
            Ok(const_fn(value, Some(cp)))
        });
    }
    let elems = iter
        .into_iter()
        .filter_map(|arg| -> Option<Type> {
            Some(match arg {
                GenericArgument::Lifetime(lifetime) => parse_quote!(& #lifetime ()),
                GenericArgument::Const(expr) => const_fn(&normalize_const_arg(expr), None),
                GenericArgument::Type(ty) => ty.clone(),
                _ => return None,
            })
        })
        .collect();
    Ok(Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: with_trailing_comma(elems),
    }))
}

/// Encodes `iter` matched with `params`, filling omitted arguments with the defaults (see
/// [`RepeaterSlots::encode_generics_args_to_ty()`]). `const_fn` encodes a (normalized) const
/// value of the given parameter.
fn encode_generics_args_with_params<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &'b Punctuated<GenericParam, Token![,]>,
    mut const_fn: impl FnMut(&Expr, &'b ConstParam) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    use syn::fold::Fold;
    let mut lifetime_args = Vec::new();
    let mut other_args = Vec::new();
    for arg in iter {
        match arg {
            GenericArgument::Lifetime(_) => lifetime_args.push(arg),
            GenericArgument::Type(_) | GenericArgument::Const(_) => other_args.push(arg),
            _ => (),
        }
    }
    let mut lifetime_args = lifetime_args.into_iter();
    let mut other_args = other_args.into_iter();
    // Substitution of the preceding parameters, used for defaults.
    let mut substitute = SubstituteParams::default();
    let mut elems = Punctuated::<Type, Token![,]>::new();
    for param in params {
        match param {
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => match lifetime_args.next() {
                Some(GenericArgument::Lifetime(arg)) => {
                    substitute.lifetimes.insert(lifetime.clone(), arg.clone());
                    elems.push(parse_quote!(& #arg ()));
                }
                _ => return Err(EncodeError::MissingArgument(Box::new(param.clone()))),
            },
            GenericParam::Type(tp) => {
                let ty: Type = match other_args.next() {
                    Some(GenericArgument::Type(ty)) => ty.clone(),
                    Some(arg) => {
                        return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone())))
                    }
                    None => match &tp.default {
                        Some(default) => substitute.fold_type(default.clone()),
                        None => return Err(EncodeError::MissingArgument(Box::new(param.clone()))),
                    },
                };
                substitute.types.insert(tp.ident.clone(), ty.clone());
                elems.push(ty);
            }
            GenericParam::Const(cp) => {
                let value: Expr = match other_args.next() {
                    Some(GenericArgument::Const(expr)) => expr.clone(),
                    Some(GenericArgument::Type(Type::Path(TypePath { qself: None, path })))
                        if path.get_ident().is_some() =>
                    {
                        parse_quote!(#path)
                    }
                    Some(arg) => {
                        return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone())))
                    }
                    None => match &cp.default {
                        Some(default) => substitute.fold_expr(default.clone()),
                        None => return Err(EncodeError::MissingArgument(Box::new(param.clone()))),
                    },
                };
                substitute.consts.insert(cp.ident.clone(), value.clone());
                elems.push(const_fn(&normalize_const_arg(&value), cp)?);
            }
        }
    }
    if let Some(arg) = lifetime_args.next().or_else(|| other_args.next()) {
        return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone())));
    }
    Ok(Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: with_trailing_comma(elems),
    }))
}

/// Primitive types which can be the type of const parameters on stable Rust.
const CONST_PARAM_PRIMITIVES: &[&str] = &[
    "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128",
    "isize",
];

/// Returns `true` if `ty` is a primitive type which can be the type of const parameters (e.g.
/// `bool` or `::core::primitive::bool`), and thus does not depend on the type context.
fn is_const_param_ty(ty: &Type) -> bool {
    const_param_primitive(ty).is_some()
}

/// Returns the name of the primitive type if `ty` is a primitive type which can be the type of
/// const parameters (e.g. `"bool"` for `bool` or `::core::primitive::bool`).
fn const_param_primitive(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Path(TypePath { qself: None, path }) => {
            if path.segments.iter().any(|seg| !seg.arguments.is_none()) {
                return None;
            }
            let segs: Vec<_> = path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            let name = match (path.leading_colon.is_some(), segs.as_slice()) {
                (false, [s]) => s,
                (_, [c, p, s]) if (c == "core" || c == "std") && p == "primitive" => s,
                _ => return None,
            };
            CONST_PARAM_PRIMITIVES
                .iter()
                .find(|prim| **prim == name.as_str())
                .copied()
        }
        Type::Group(TypeGroup { elem, .. }) | Type::Paren(TypeParen { elem, .. }) => {
            const_param_primitive(elem)
        }
        _ => None,
    }
}

/// Returns `true` if the segments of `path` start with those of `prefix` (e.g. `Option<T>` starts
/// with `Option`, and `std::vec::Vec<T>` starts with `std::vec`), comparing only the identifiers:
/// generic arguments of `path` are allowed.
fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    prefix.leading_colon.is_some() == path.leading_colon.is_some()
        && prefix.segments.len() <= path.segments.len()
        && prefix
            .segments
            .iter()
            .zip(path.segments.iter())
            .all(|(prefix_seg, seg)| prefix_seg.ident == seg.ident)
}

/// Returns `true` if `ident` is the name of a primitive type (e.g. `u8` or `str`).
fn is_primitive_name(ident: &Ident) -> bool {
    ABSOLUTE_PATHS
        .iter()
        .any(|(name, absolute)| ident == name && absolute.starts_with("::core::primitive::"))
}

/// Returns `true` if `ty` is a single identifier named like a const by the naming convention
/// (`SCREAMING_SNAKE_CASE` with at least two characters, e.g. `SIZE`), which syn parses as a type
/// when used as a generic argument.
fn looks_like_const(ty: &Type) -> bool {
    match ty {
        Type::Path(TypePath { qself: None, path }) => path.get_ident().map_or(false, |ident| {
            let name = ident.to_string();
            name.len() >= 2
                && name.chars().any(|c| c.is_ascii_uppercase())
                && name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        }),
        _ => false,
    }
}

/// Returns `true` if `ty` refers to any of `generics`.
fn mentions_generics<'a>(ty: &Type, generics: impl IntoIterator<Item = &'a GenericParam>) -> bool {
    let mut idents = Vec::new();
    let mut lifetimes = Vec::new();
    for gp in generics {
        match gp {
            GenericParam::Type(TypeParam { ident, .. })
            | GenericParam::Const(ConstParam { ident, .. }) => idents.push(ident.clone()),
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                lifetimes.push(lifetime.clone())
            }
        }
    }
    mentions_names(ty, idents, lifetimes)
}

/// Returns `true` if `ty` refers to any of `idents` (as the first segment of a relative path) or
/// `lifetimes`.
fn mentions_names(ty: &Type, idents: Vec<Ident>, lifetimes: Vec<Lifetime>) -> bool {
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

/// Collects the names of identifiers and lifetimes (e.g. `N` and `'a`) in `tokens`.
fn token_names(tokens: TokenStream, names: &mut HashSet<String>) {
    let mut after_quote = false;
    for tt in tokens {
        match &tt {
            proc_macro2::TokenTree::Ident(ident) if after_quote => {
                names.insert(format!("'{}", ident));
            }
            proc_macro2::TokenTree::Ident(ident) => {
                names.insert(ident.to_string());
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

/// Returns the name of a generic parameter, e.g. `N` or `'a`.
fn generic_param_name(param: &GenericParam) -> String {
    match param {
        GenericParam::Type(TypeParam { ident, .. })
        | GenericParam::Const(ConstParam { ident, .. }) => ident.to_string(),
        GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => lifetime.to_string(),
    }
}

/// Returns the local generics which `ty` depends on, including those used in the bounds and
/// where predicates of them, in the order of declaration.
///
/// Identifiers are matched by name, so a local generic may be included even if the name refers to
/// something else (e.g. `::path::N`). This is harmless, because the Leaker and the Referrer
/// include the same generics.
fn required_local_generics(
    ty: &Type,
    locals: &[&GenericParam],
    predicates: &[WherePredicate],
) -> Generics {
    let mut names = HashSet::new();
    token_names(quote!(#ty), &mut names);
    let mut included_params = vec![false; locals.len()];
    let mut included_predicates = vec![false; predicates.len()];
    loop {
        let mut changed = false;
        for (included, param) in included_params.iter_mut().zip(locals) {
            if !*included && names.contains(&generic_param_name(param)) {
                *included = true;
                token_names(quote!(#param), &mut names);
                changed = true;
            }
        }
        for (included, predicate) in included_predicates.iter_mut().zip(predicates) {
            if *included {
                continue;
            }
            let mut predicate_names = HashSet::new();
            token_names(quote!(#predicate), &mut predicate_names);
            let mentions_included = locals
                .iter()
                .zip(&included_params)
                .any(|(param, inc)| *inc && predicate_names.contains(&generic_param_name(param)));
            if mentions_included {
                *included = true;
                names.extend(predicate_names);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let params: Punctuated<GenericParam, Token![,]> = locals
        .iter()
        .zip(&included_params)
        .filter(|(_, inc)| **inc)
        .map(|(param, _)| {
            let mut param = (*param).clone();
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
    let predicates: Punctuated<WherePredicate, Token![,]> = predicates
        .iter()
        .zip(&included_predicates)
        .filter(|(_, inc)| **inc)
        .map(|(predicate, _)| predicate.clone())
        .collect();
    Generics {
        lt_token: Some(Default::default()),
        gt_token: Some(Default::default()),
        where_clause: if predicates.is_empty() {
            None
        } else {
            Some(WhereClause {
                where_token: Default::default(),
                predicates,
            })
        },
        params,
    }
}

/// Normalizes a const argument: redundant braces are removed from a literal or a single
/// identifier (e.g. `{ N }` -> `N`), and any other expression is braced (e.g. `i32::MIN` ->
/// `{ i32::MIN }`), which is required where it could be parsed as a type.
fn normalize_const_arg(expr: &Expr) -> Expr {
    let inner = unwrap_block(expr);
    let is_simple = match inner {
        Expr::Lit(_) => true,
        Expr::Path(ExprPath {
            qself: None, path, ..
        }) => path.get_ident().is_some(),
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => matches!(expr.as_ref(), Expr::Lit(_)),
        _ => false,
    };
    if is_simple {
        inner.clone()
    } else {
        parse_quote!({ #inner })
    }
}

/// Removes redundant braces around a const argument, e.g. `{ N }` -> `N`.
fn unwrap_block(expr: &Expr) -> &Expr {
    match expr {
        Expr::Block(ExprBlock {
            attrs,
            label: None,
            block,
        }) if attrs.is_empty() && block.stmts.len() == 1 => match &block.stmts[0] {
            Stmt::Expr(inner, None) => unwrap_block(inner),
            _ => expr,
        },
        _ => expr,
    }
}

/// Encode [`GenericParam`]s to a type.
///
/// A const parameter `N` is encoded as `[(); N]`, which constrains `N` when used in the trait
/// arguments of an `impl`. Because array lengths are `usize`, this only works for
/// `const N: usize`. To encode generics of any kind, use [`RepeaterSlots`], or
/// [`encode_generics_params_to_ty_with()`] with user-defined types.
pub fn encode_generics_params_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericParam>) -> Type {
    encode_generics_params_to_ty_with(iter, |ConstParam { ident, .. }| parse_quote!([(); #ident]))
}

/// Encode [`GenericParam`]s to a type, using `const_fn` to encode const parameters.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::quote;
/// let generics: Generics = parse_quote!(<'a, T, const B: bool>);
/// // `::my_crate::ConstBool` is defined as `pub struct ConstBool<const B: bool>;`
/// let ty = encode_generics_params_to_ty_with(&generics.params, |ConstParam { ident, .. }| {
///     parse_quote!(::my_crate::ConstBool<#ident>)
/// });
/// assert_eq!(
///     quote!(#ty).to_string(),
///     quote!((&'a (), T, ::my_crate::ConstBool<B>)).to_string()
/// );
/// ```
pub fn encode_generics_params_to_ty_with<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    mut const_fn: impl FnMut(&'a ConstParam) -> Type,
) -> Type {
    Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: iter
            .into_iter()
            .map(|param| -> Type {
                match param {
                    GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                        parse_quote!(& #lifetime ())
                    }
                    GenericParam::Const(cp) => const_fn(cp),
                    GenericParam::Type(TypeParam { ident, .. }) => parse_quote!(#ident),
                }
            })
            .collect(),
    })
}

/// Types of const parameters which have a slot in the Repeater, in the order of the slots. `usize`
/// is encoded as `[(); N]`, and `u64` uses the Repeater's `ID` parameter.
const REPEATER_SLOTS: &[(&str, &str, &str)] = &[
    ("bool", "BOOL", "false"),
    ("char", "CHAR", "'\\0'"),
    ("u8", "U8", "0"),
    ("u16", "U16", "0"),
    ("u32", "U32", "0"),
    ("u128", "U128", "0"),
    ("i8", "I8", "0"),
    ("i16", "I16", "0"),
    ("i32", "I32", "0"),
    ("i64", "I64", "0"),
    ("i128", "I128", "0"),
    ("isize", "ISIZE", "0"),
];

/// Definition of a Repeater which supports const generics of any primitive type.
///
/// Encoding a const generic parameter into a type requires a type with a const parameter of the
/// same type, and `core` has no stable type with a const parameter of a type other than `usize`.
/// So the Repeater itself provides one: besides the leaked type's key `T`, the `ID` and the
/// index `NTH`, it takes a defaulted const "slot" parameter per primitive type. A const value is
/// then encoded as a pointer to a trait object of the Repeater with that slot filled (see
/// [`RepeaterSlots`]):
///
/// ```ignore
/// pub trait Repeater<
///     T: ?Sized, const ID: u64, const NTH: usize,
///     const BOOL: bool = false, const CHAR: char = '\0',
///     const U8: u8 = 0, const U16: u16 = 0, const U32: u32 = 0, const U128: u128 = 0,
///     const I8: i8 = 0, const I16: i16 = 0, const I32: i32 = 0, const I64: i64 = 0,
///     const I128: i128 = 0, const ISIZE: isize = 0,
/// > {
///     type Type: ?Sized;
/// }
/// ```
///
/// The slots have defaults, so a Repeater used as `Repeater<T, ID, NTH>` works as before. A
/// Repeater without the slots also works with [`RepeaterSlots`] as long as all const parameters are
/// `usize` or `u64`. The user crate defines it once, e.g. by emitting this definition from a
/// macro.
///
/// The generated definition requires Rust 1.59 (defaults of const parameters) and the 2018 edition
/// (paths starting with `::core`).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let def = repeater_definition(&parse_quote!(pub), &parse_quote!(Repeater));
/// assert_eq!(def.generics.params.len(), 3 + 12);
/// ```
pub fn repeater_definition(vis: &Visibility, ident: &Ident) -> ItemTrait {
    let slots = REPEATER_SLOTS
        .iter()
        .map(|(ty, name, default)| -> GenericParam {
            let ty = Ident::new(ty, Span::call_site());
            let name = Ident::new(name, Span::call_site());
            let default: Expr = parse_str(default).unwrap();
            parse_quote!(const #name: ::core::primitive::#ty = #default)
        });
    parse_quote! {
        #vis trait #ident<
            T: ?::core::marker::Sized,
            const ID: ::core::primitive::u64,
            const NTH: ::core::primitive::usize,
            #(#slots),*
        > {
            type Type: ?::core::marker::Sized;
        }
    }
}

/// Error returned by [`RepeaterSlots`] when generics cannot be encoded into a type.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum EncodeError {
    /// The type of a const parameter is not a primitive type (e.g. a type alias), so its slot
    /// cannot be chosen.
    UnsupportedConstType(Box<Type>),
    /// A generic argument which does not correspond to any generic parameter (i.e. too many
    /// arguments).
    UnexpectedArgument(Box<GenericArgument>),
    /// A generic parameter without a corresponding argument nor a default.
    MissingArgument(Box<GenericParam>),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::UnsupportedConstType(ty) => {
                write!(f, "unsupported type of const parameter: '{}'", quote!(#ty))
            }
            EncodeError::UnexpectedArgument(arg) => {
                write!(f, "unexpected generic argument: '{}'", quote!(#arg))
            }
            EncodeError::MissingArgument(param) => {
                write!(f, "missing generic argument for '{}'", quote!(#param))
            }
        }
    }
}

impl std::error::Error for EncodeError {}

impl EncodeError {
    pub fn span(&self) -> Span {
        match self {
            EncodeError::UnsupportedConstType(ty) => ty.span(),
            EncodeError::UnexpectedArgument(arg) => arg.span(),
            EncodeError::MissingArgument(param) => param.span(),
        }
    }
}

impl From<EncodeError> for syn::Error {
    fn from(err: EncodeError) -> Self {
        syn::Error::new(err.span(), err)
    }
}

/// Encodes generics into types using the const slots of the Repeater defined by
/// [`repeater_definition()`].
///
/// Like [`encode_generics_params_to_ty()`], generics are encoded into a tuple type: a lifetime
/// `'a` as `&'a ()`, a type `T` as `T` and a `const N: usize` as `[(); N]`. A const parameter of
/// another primitive type is encoded as a trait object of the Repeater with the corresponding
/// slot filled, e.g. `*const dyn path::Repeater<(), 0, 0, B, Type = ()>` for `const B: bool`, and
/// `const X: u64` uses the `ID` parameter.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::quote;
/// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
///
/// // In the Leaker
/// let generics: Generics = parse_quote!(<T, const N: usize, const B: bool, const C: char = 'z'>);
/// let ty = slots.encode_generics_params_to_ty(&generics.params).unwrap();
/// let expected: Type = parse_quote!((
///     T,
///     [(); N],
///     *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
///     *const dyn crate::Repeater<(), 0, 0, false, C, Type = ()>
/// ));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
///
/// // In the Referrer (for `MyTrait<u8, 3, true>`; the defaulted `C` is omitted)
/// let args: AngleBracketedGenericArguments = parse_quote!(<u8, 3, true>);
/// let ty = slots
///     .encode_generics_args_to_ty(&args.args, &generics.params)
///     .unwrap();
/// let expected: Type = parse_quote!((
///     u8,
///     [(); 3],
///     *const dyn crate::Repeater<(), 0, 0, true, Type = ()>,
///     *const dyn crate::Repeater<(), 0, 0, false, 'z', Type = ()>
/// ));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RepeaterSlots {
    /// Path to the Repeater.
    pub repeater: Path,
}

impl RepeaterSlots {
    /// Specifies the path to the Repeater defined by [`repeater_definition()`].
    pub fn new(repeater: Path) -> Self {
        Self { repeater }
    }

    /// Encodes a const value `value` of type `ty` into a type.
    ///
    /// Returns [`EncodeError::UnsupportedConstType`] if `ty` is not a primitive type.
    pub fn encode_const(&self, value: &Expr, ty: &Type) -> std::result::Result<Type, EncodeError> {
        let prim = const_param_primitive(ty)
            .ok_or_else(|| EncodeError::UnsupportedConstType(Box::new(ty.clone())))?;
        let value = normalize_const_arg(value);
        let repeater = &self.repeater;
        if prim == "usize" {
            return Ok(parse_quote!([(); #value]));
        }
        if prim == "u64" {
            return Ok(parse_quote!(*const dyn #repeater<(), #value, 0, Type = ()>));
        }
        let position = REPEATER_SLOTS
            .iter()
            .position(|(ty, _, _)| *ty == prim)
            .unwrap();
        let defaults = REPEATER_SLOTS[..position]
            .iter()
            .map(|(_, _, default)| -> Expr { parse_str(default).unwrap() });
        Ok(parse_quote!(*const dyn #repeater<(), 0, 0, #(#defaults,)* #value, Type = ()>))
    }

    /// Encodes [`GenericParam`]s into a type, like [`encode_generics_params_to_ty()`].
    ///
    /// Returns [`EncodeError::UnsupportedConstType`] if the type of a const parameter is not a
    /// primitive type.
    pub fn encode_generics_params_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
    ) -> std::result::Result<Type, EncodeError> {
        let elems = iter
            .into_iter()
            .map(|param| -> std::result::Result<Type, EncodeError> {
                Ok(match param {
                    GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                        parse_quote!(& #lifetime ())
                    }
                    GenericParam::Type(TypeParam { ident, .. }) => parse_quote!(#ident),
                    GenericParam::Const(ConstParam { ident, ty, .. }) => {
                        self.encode_const(&parse_quote!(#ident), ty)?
                    }
                })
            })
            .collect::<std::result::Result<Punctuated<Type, Token![,]>, _>>()?;
        Ok(Type::Tuple(TypeTuple {
            paren_token: Default::default(),
            elems: with_trailing_comma(elems),
        }))
    }

    /// Encodes [`GenericArgument`]s into a type. `params` are the [`GenericParam`]s of the Leaker
    /// corresponding to `iter`, which are used to find the types of const arguments.
    ///
    /// The result is the same type as [`RepeaterSlots::encode_generics_params_to_ty()`] with
    /// `params` substituted by `iter`:
    ///
    /// - Arguments are matched with parameters by position (lifetime arguments with lifetime
    ///   parameters, and the others with type and const parameters). Every lifetime parameter
    ///   needs an argument (lifetimes cannot be elided in the trait of an `impl`).
    /// - A bare identifier corresponding to a const parameter (e.g. `N` in `MyTrait<N>`, parsed as
    ///   a type argument) is treated as a const argument.
    /// - Omitted type and const arguments take the parameters' defaults, in which the preceding
    ///   parameters are substituted by their arguments. The defaults are used as they are written,
    ///   so they should not depend on the Leaker's type context (e.g. `= MY_CONST` or
    ///   `= MyType<N>` would not resolve in the Referrer's context); otherwise the arguments
    ///   should be given explicitly.
    /// - Associated item bindings and constraints (e.g. `Item = u8`) are skipped.
    ///
    /// Returns [`EncodeError`] if the type of a const parameter is not a primitive type, if there
    /// are too many arguments, or if an argument without a default is missing.
    pub fn encode_generics_args_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
    ) -> std::result::Result<Type, EncodeError> {
        encode_generics_args_with_params(iter, params, |value, cp| self.encode_const(value, &cp.ty))
    }
}

/// Adds a trailing comma to a single-element tuple, so that `(T,)` is not parsed as `(T)`.
fn with_trailing_comma(mut elems: Punctuated<Type, Token![,]>) -> Punctuated<Type, Token![,]> {
    if elems.len() == 1 && !elems.trailing_punct() {
        elems.push_punct(Default::default());
    }
    elems
}

/// Substitutes generic parameters (single-segment paths and lifetimes) with arguments.
#[derive(Default)]
struct SubstituteParams {
    types: HashMap<Ident, Type>,
    consts: HashMap<Ident, Expr>,
    lifetimes: HashMap<Lifetime, Lifetime>,
}

impl syn::fold::Fold for SubstituteParams {
    fn fold_type(&mut self, ty: Type) -> Type {
        if let Type::Path(TypePath { qself: None, path }) = &ty {
            if let Some(ident) = path.get_ident() {
                if let Some(arg) = self.types.get(ident) {
                    return arg.clone();
                }
            }
        }
        syn::fold::fold_type(self, ty)
    }
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        if let Expr::Path(ExprPath {
            qself: None, path, ..
        }) = &expr
        {
            if let Some(arg) = path.get_ident().and_then(|ident| self.consts.get(ident)) {
                return parse_quote!({ #arg });
            }
        }
        syn::fold::fold_expr(self, expr)
    }
    fn fold_generic_argument(&mut self, arg: GenericArgument) -> GenericArgument {
        // A bare const parameter is parsed as a type argument, e.g. `N` in `Foo<N>`.
        if let GenericArgument::Type(Type::Path(TypePath { qself: None, path })) = &arg {
            if let Some(value) = path.get_ident().and_then(|ident| self.consts.get(ident)) {
                return GenericArgument::Const(normalize_const_arg(value));
            }
        }
        syn::fold::fold_generic_argument(self, arg)
    }
    fn fold_lifetime(&mut self, lifetime: Lifetime) -> Lifetime {
        self.lifetimes.get(&lifetime).cloned().unwrap_or(lifetime)
    }
}

/// Represents the result of [`Leaker::check()`], holding the cause in its tuple field.
#[derive(Clone, Debug)]
pub enum CheckResult {
    /// The type must be interned because it directly depends on the type context.
    MustIntern(TokenStream),
    /// The type must be interned because one of the types it is composed of must be interned.
    MustInternOrInherit(TokenStream),
    /// The type must not be interned because `type_leak` cannot intern it (e.g. `impl Trait`,
    /// `_`, trait bounds with a relative trait path, ...)
    MustNotIntern(TokenStream),
    /// Other.
    Neutral,
}

#[derive(Debug, Clone)]
struct GraphOperation {
    node_type: Type,
    parent: Option<Type>,
    is_root: bool,
    is_must_intern: bool,
    /// Local generics (e.g. of trait methods or GATs) which `node_type` depends on.
    local_generics: Generics,
}

struct AnalyzeVisitor<'a> {
    leaker: &'a mut Leaker,
    generics: Generics,
    /// The number of leading `generics.params` which belong to the Leaker itself. The rest are
    /// local generics (e.g. of trait methods or GATs), which cannot be referred to from the
    /// Repeater's implementation.
    n_outer_generics: usize,
    /// Where predicates of the local generics.
    local_predicates: Vec<WherePredicate>,
    parent: Option<Type>,
    error: Option<TokenStream>,
    operations: Vec<GraphOperation>,
}

impl<'a, 'ast> Visit<'ast> for AnalyzeVisitor<'a> {
    fn visit_trait_item_fn(&mut self, i: &TraitItemFn) {
        let mut visitor = AnalyzeVisitor {
            leaker: self.leaker,
            generics: self.generics.clone(),
            n_outer_generics: self.n_outer_generics,
            local_predicates: self.local_predicates.clone(),
            parent: self.parent.clone(),
            error: self.error.clone(),
            operations: Vec::new(),
        };
        for g in &i.sig.generics.params {
            visitor.generics.params.push(g.clone());
        }
        if let Some(wc) = &i.sig.generics.where_clause {
            visitor
                .local_predicates
                .extend(wc.predicates.iter().cloned());
        }
        let mut i = i.clone();
        i.default = None;
        i.semi_token = Some(Default::default());
        syn::visit::visit_trait_item_fn(&mut visitor, &i);
        if visitor.error.is_some() {
            self.error = visitor.error;
        }
        self.operations.extend(visitor.operations);
    }

    fn visit_receiver(&mut self, i: &Receiver) {
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        if let Some((_, Some(lt))) = &i.reference {
            self.visit_lifetime(lt);
        }
        if i.colon_token.is_some() {
            self.visit_type(&i.ty);
        }
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        let mut visitor = AnalyzeVisitor {
            leaker: self.leaker,
            generics: self.generics.clone(),
            n_outer_generics: self.n_outer_generics,
            local_predicates: self.local_predicates.clone(),
            parent: self.parent.clone(),
            error: self.error.clone(),
            operations: Vec::new(),
        };
        for g in &i.generics.params {
            visitor.generics.params.push(g.clone());
        }
        if let Some(wc) = &i.generics.where_clause {
            visitor
                .local_predicates
                .extend(wc.predicates.iter().cloned());
        }
        syn::visit::visit_trait_item_type(&mut visitor, i);
        self.error = visitor.error;
        self.operations.extend(visitor.operations);
    }

    fn visit_type(&mut self, i: &Type) {
        let local_generics =
            &self.generics.params.iter().collect::<Vec<_>>()[self.n_outer_generics..];
        let mentions_local = mentions_generics(i, local_generics.iter().copied());
        match self.leaker.check(i, &self.generics) {
            Err((s, _)) if self.has_relative_own_path(i) => {
                // The type must be interned (because of `s`, which depends on the type context)
                // but cannot be (e.g. `MyType<'_>`), and cannot be decomposed either. Report `s`,
                // which should be made absolute.
                self.error = Some(s);
            }
            Err(_) => {
                // The type cannot be interned as a whole (e.g. `&MyType` with an elided lifetime,
                // or `impl Iterator<Item = MyType>`), but its components can.
                syn::visit::visit_type(self, i);
            }
            Ok(CheckResult::MustNotIntern(_)) => (),
            Ok(CheckResult::MustIntern(_)) if mentions_local => {
                // The type depends on both the type context and local generics, and cannot be
                // decomposed further. It is interned together with the local generics it needs,
                // which the Repeater's implementation must declare (see
                // [`Referrer::local_generics()`]).
                let local_generics = required_local_generics(
                    i,
                    &self.generics.params.iter().collect::<Vec<_>>()[self.n_outer_generics..],
                    &self.local_predicates,
                );
                self.operations.push(GraphOperation {
                    node_type: i.clone(),
                    parent: self.parent.clone(),
                    is_root: self.parent.is_none(),
                    is_must_intern: true,
                    local_generics,
                });
            }
            Ok(_) if mentions_local => {
                // The type itself cannot be interned because the Repeater's implementation
                // cannot refer to the local generics, but its components can.
                syn::visit::visit_type(self, i);
            }
            o => {
                // Collect the operation instead of directly modifying the graph
                let is_must_intern = matches!(o, Ok(CheckResult::MustIntern(_)));
                let is_root = self.parent.is_none();

                if is_root && is_must_intern && looks_like_const(i) {
                    // A bare identifier such as `SIZE` in `T: MyTrait<SIZE>` is parsed as a type,
                    // but is probably a const (by the naming convention), which cannot be leaked
                    // on its own (like `T: MyTrait<{ SIZE }>`).
                    self.error = Some(quote!(#i));
                    return;
                }

                self.operations.push(GraphOperation {
                    node_type: i.clone(),
                    parent: self.parent.clone(),
                    is_root,
                    is_must_intern,
                    local_generics: Generics::default(),
                });

                if !is_must_intern {
                    // Perform recursive call
                    let parent = self.parent.clone();
                    self.parent = Some(i.clone());
                    syn::visit::visit_type(self, i);
                    self.parent = parent;
                }
            }
        }
    }

    fn visit_const_param(&mut self, i: &ConstParam) {
        // The type of a const parameter cannot be a projection (such as
        // `<MyLeakerType as Repeater<0>>::Type`), so it is never interned. It must be a primitive
        // type, or a context-free path to a type alias of it (e.g. `::my_crate::Idx`).
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        let is_context_free_path = matches!(
            &i.ty,
            Type::Path(TypePath { qself: None, path }) if self.leaker.is_context_free_path(path)
        );
        if let Type::Path(TypePath { qself: None, path }) = &i.ty {
            if let Some(ident) = path.get_ident() {
                if is_const_param_ty(&i.ty) {
                    self.leaker.assume_primitive(ident);
                }
            }
        }
        if !is_const_param_ty(&i.ty) && !is_context_free_path {
            self.error = Some(quote!(#{&i.ty}));
            return;
        }
        if let Some(expr) = &i.default {
            self.visit_expr(expr);
        }
    }

    fn visit_trait_bound(&mut self, i: &TraitBound) {
        // Trait bounds with a non-absolute path
        if !self.leaker.is_context_free_path(&i.path) {
            self.error = Some(quote!(#i));
            return;
        }
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_trait_bound(v, i)
        })
    }

    fn visit_type_bare_fn(&mut self, i: &TypeBareFn) {
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_type_bare_fn(v, i)
        })
    }

    fn visit_predicate_type(&mut self, i: &PredicateType) {
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_predicate_type(v, i)
        })
    }

    fn visit_path(&mut self, i: &Path) {
        // A primitive type allowed by `allowed_paths` (e.g. by `LeakerConfig::allow_primitive()`)
        // is copied as it is, assuming that its name is not shadowed.
        if i.leading_colon.is_none() && self.leaker.is_context_free_path(i) {
            if let Some(seg) = i.segments.first() {
                let is_generic = self.generics.params.iter().any(|gp| match gp {
                    GenericParam::Type(TypeParam { ident, .. })
                    | GenericParam::Const(ConstParam { ident, .. }) => ident == &seg.ident,
                    _ => false,
                });
                if !is_generic && is_primitive_name(&seg.ident) {
                    self.leaker.assume_primitive(&seg.ident);
                }
            }
        }
        syn::visit::visit_path(self, i)
    }

    fn visit_trait_item_const(&mut self, i: &TraitItemConst) {
        // Like the bodies of methods, defaults of associated consts are not copied.
        let mut i = i.clone();
        i.default = None;
        syn::visit::visit_trait_item_const(self, &i)
    }

    fn visit_expr_path(&mut self, i: &ExprPath) {
        // Values starting with a generic parameter in scope (e.g. `N` or `T::N` where
        // `const N: usize` or `T` is a generic parameter) or `Self` do not depend on the type
        // context.
        let starts_with_generic = i.qself.is_none()
            && i.path.leading_colon.is_none()
            && i.path.segments.first().map_or(false, |seg| {
                seg.arguments.is_none()
                    && ((seg.ident == "Self" && self.leaker.self_ty_can_be_interned)
                        || self.generics.params.iter().any(|gp| match gp {
                            GenericParam::Type(TypeParam { ident, .. })
                            | GenericParam::Const(ConstParam { ident, .. }) => ident == &seg.ident,
                            _ => false,
                        }))
            });
        if starts_with_generic || self.leaker.is_context_free_path(&i.path) {
            syn::visit::visit_expr_path(self, i)
        } else {
            // Value or trait bounds with a relative path
            self.error = Some(quote!(#i));
        }
    }
}

impl AnalyzeVisitor<'_> {
    /// Runs `f` with the lifetimes of a `for<..>` binder (e.g. `'a` in `for<'a> Fn(&'a T)`) in
    /// scope. They are treated as local generics, which the Repeater's implementation cannot
    /// refer to.
    fn with_bound_lifetimes<R>(
        &mut self,
        lifetimes: Option<&BoundLifetimes>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let len = self.generics.params.len();
        if let Some(lifetimes) = lifetimes {
            self.generics
                .params
                .extend(lifetimes.lifetimes.iter().cloned());
        }
        let result = f(self);
        while self.generics.params.len() > len {
            self.generics.params.pop();
        }
        result
    }

    /// Returns `true` if `ty` is a path which depends on the type context itself (as opposed to
    /// its components), so it cannot be decomposed.
    fn has_relative_own_path(&self, ty: &Type) -> bool {
        match ty {
            Type::Path(TypePath { qself: Some(_), .. }) => true,
            Type::Path(TypePath { qself: None, path }) => {
                let starts_with_generic = path.leading_colon.is_none()
                    && path.segments.first().map_or(false, |seg| {
                        seg.ident == "Self"
                            || self.generics.params.iter().any(|gp| match gp {
                                GenericParam::Type(TypeParam { ident, .. })
                                | GenericParam::Const(ConstParam { ident, .. }) => {
                                    ident == &seg.ident
                                }
                                _ => false,
                            })
                    });
                !starts_with_generic && !self.leaker.is_context_free_path(path)
            }
            Type::Group(TypeGroup { elem, .. }) | Type::Paren(TypeParen { elem, .. }) => {
                self.has_relative_own_path(elem)
            }
            Type::Macro(_) | Type::Verbatim(_) => true,
            _ => false,
        }
    }
}

impl Default for Leaker {
    fn default() -> Self {
        Self::new()
    }
}

impl Leaker {
    /// Initialize with [`ItemStruct`]
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let test_struct: ItemStruct = parse_quote!(
    ///     pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
    ///         field1: MyType1,
    ///         field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
    ///         field3: &'a (T1, T2),
    ///     }
    /// );
    /// let leaker = Leaker::from_struct(&test_struct).unwrap();
    /// ```
    pub fn from_struct(input: &ItemStruct) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_struct(input);
        })?;
        Ok(leaker)
    }

    /// Initialize with [`ItemEnum`]
    pub fn from_enum(input: &ItemEnum) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_enum(input);
        })?;
        Ok(leaker)
    }

    /// Build a [`Leaker`] from a given trait.
    ///
    /// Unlike an enum or struct, a trait Leaker requires an alternate struct — an absolute path
    /// to a struct declared in the same crate as the leaker trait and visible from
    /// [`Referrer`]s' context. That struct serves as the `impl` target of the Repeater
    /// instead of the Leaker's path.
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let s: ItemTrait = parse_quote!{
    ///     pub trait MyTrait<T, U> {
    ///         fn func(self, t: T) -> U;
    ///     }
    /// };
    /// let alternate: ItemStruct = parse_quote!{
    ///     pub struct MyAlternate;
    /// };
    /// let _ = Leaker::from_trait(&s);
    /// ```
    pub fn from_trait(input: &ItemTrait) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_trait(input);
        })?;
        Ok(leaker)
    }

    /// Initialize empty [`Leaker`] with no generics.
    pub fn new() -> Self {
        Self {
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
            operations: Vec::new(),
            reachable_types: None,
            local_generics: HashMap::new(),
            plain_types: HashSet::new(),
            assumed_primitives: Vec::new(),
        }
    }

    /// Initialize [`Leaker`] from configuration.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allowed_paths: config.allowed_paths,
            self_ty_can_be_interned: config.self_ty_can_be_interned,
            operations: Vec::new(),
            reachable_types: None,
            local_generics: HashMap::new(),
            plain_types: HashSet::new(),
            assumed_primitives: Vec::new(),
        }
    }

    fn assume_primitive(&mut self, ident: &Ident) {
        if !self.assumed_primitives.iter().any(|p| p == ident) {
            self.assumed_primitives.push(ident.clone());
        }
    }

    /// Items to emit in the Leaker's type context, which check the assumptions made by the
    /// Leaker at compile time.
    ///
    /// The names of primitive types (e.g. `u8` in `const N: u8`, or in a type when allowed by
    /// [`LeakerConfig::allow_primitive()`]) are copied as they are, assuming that they are not
    /// shadowed in the Leaker's type context (e.g. by `type u8 = i32;`), which cannot be known
    /// without name resolution. These items fail to compile (with "mismatched types") if a name
    /// is shadowed, instead of the copies silently referring to another type.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait MyTrait<const N: u8> {});
    /// let leaker = Leaker::from_trait(&item).unwrap();
    /// assert!(leaker
    ///     .assertions()
    ///     .to_string()
    ///     .contains("__type_leak_assert_u8_is_primitive"));
    /// ```
    pub fn assertions(&self) -> TokenStream {
        if self.assumed_primitives.is_empty() {
            return TokenStream::new();
        }
        let fns = self.assumed_primitives.iter().map(|ident| {
            let fn_ident = Ident::new(
                &format!("__type_leak_assert_{}_is_primitive", ident),
                Span::call_site(),
            );
            quote! {
                fn #fn_ident(x: &#ident) -> &::core::primitive::#ident { x }
            }
        });
        quote! {
            #[allow(dead_code, non_snake_case)]
            const _: () = {
                #(#fns)*
            };
        }
    }

    /// Returns `true` if `path` does not depend on the type context, i.e. it is absolute, starts
    /// with `$crate`, or starts with one of [`Leaker::allowed_paths`].
    fn is_context_free_path(&self, path: &Path) -> bool {
        path.leading_colon.is_some()
            || matches!(
                path.segments.iter().next(),
                Some(PathSegment {
                    ident,
                    arguments: PathArguments::None,
                }) if ident == "$crate"
            )
            || self
                .allowed_paths
                .iter()
                .any(|allowed| path_starts_with(path, allowed))
    }

    /// Adds collected operations. The reduction is recomputed over all operations (including those
    /// of previous calls of [`Leaker::intern_with()`]).
    fn apply_operations(&mut self, operations: Vec<GraphOperation>) {
        if operations.is_empty() {
            return;
        }
        self.operations.extend(operations);
        self.reachable_types = None;
    }

    /// Intern AST elements using a visitor callback.
    pub fn intern_with<'ast, F>(
        &mut self,
        generics: &Generics,
        f: F,
    ) -> std::result::Result<&mut Self, NotInternableError>
    where
        F: FnOnce(&mut dyn syn::visit::Visit<'ast>),
    {
        let mut visitor = AnalyzeVisitor {
            leaker: self,
            parent: None,
            error: None,
            generics: generics.clone(),
            n_outer_generics: generics.params.len(),
            local_predicates: Vec::new(),
            operations: Vec::new(),
        };
        f(&mut visitor);
        if let Some(e) = visitor.error {
            Err(NotInternableError(e, NotInternableReason::RelativePath))
        } else {
            let operations = visitor.operations;
            // The Referrer finds leaked types by the type alone, so the same type must depend on
            // the same local generics wherever it appears.
            for operation in &operations {
                let ty = &operation.node_type;
                let conflicts = if operation.local_generics.params.is_empty() {
                    // e.g. `Wrap<B>` in `fn f(&self) -> Wrap<B>` (where `B` is a type) and in
                    // `fn g<const B: bool>(&self) -> Wrap<B>`.
                    self.local_generics.contains_key(ty)
                } else {
                    // e.g. `Wrap<B>` in `fn f<const B: bool>` and `fn g<const B: u8>`.
                    self.plain_types.contains(ty)
                        || matches!(
                            self.local_generics.get(ty),
                            Some(existing) if existing != &operation.local_generics
                        )
                };
                if conflicts {
                    return Err(NotInternableError(
                        quote!(#ty),
                        NotInternableReason::ConflictingLocalGenerics,
                    ));
                }
                if operation.local_generics.params.is_empty() {
                    self.plain_types.insert(ty.clone());
                } else {
                    self.local_generics
                        .insert(ty.clone(), operation.local_generics.clone());
                }
            }
            self.apply_operations(operations);
            Ok(self)
        }
    }

    /// Check whether the given `ty` is internable. Returns `Err` when there is a contradiction
    /// (the type must both be interned and must not be interned).
    ///
    ///
    /// See [`CheckResult`].
    pub fn check(
        &self,
        ty: &Type,
        generics: &Generics,
    ) -> std::result::Result<CheckResult, (TokenStream, TokenStream)> {
        #[derive(Clone)]
        struct Visitor {
            generic_lifetimes: Vec<Lifetime>,
            generic_idents: Vec<Ident>,
            impossible: Option<TokenStream>,
            must: Option<(isize, TokenStream)>,
            self_ty_can_be_interned: bool,
            allowed_paths: Vec<Path>,
        }

        const _: () = {
            impl<'a> Visit<'a> for Visitor {
                fn visit_type(&mut self, i: &Type) {
                    match i {
                        Type::BareFn(TypeBareFn {
                            lifetimes,
                            inputs,
                            output,
                            ..
                        }) => {
                            let mut visitor = self.clone();
                            visitor.must = None;
                            visitor.generic_lifetimes.extend(
                                lifetimes
                                    .as_ref()
                                    .map(|ls| {
                                        ls.lifetimes.iter().map(|gp| {
                                            if let GenericParam::Lifetime(lt) = gp {
                                                lt.lifetime.clone()
                                            } else {
                                                panic!()
                                            }
                                        })
                                    })
                                    .into_iter()
                                    .flatten(),
                            );
                            for input in inputs {
                                visitor.visit_type(&input.ty);
                            }
                            if let ReturnType::Type(_, output) = output {
                                visitor.visit_type(output.as_ref());
                            }
                            if self.impossible.is_none() {
                                self.impossible = visitor.impossible;
                            }
                            match (self.must.clone(), visitor.must) {
                                (Some((s_l, _)), Some((v_l, v_m))) if v_l + 1 < s_l => {
                                    self.must = Some((v_l + 1, v_m))
                                }
                                (None, Some((v_l, v_m))) => self.must = Some((v_l + 1, v_m)),
                                _ => (),
                            }
                            return;
                        }
                        Type::TraitObject(TypeTraitObject { bounds, .. }) => {
                            for bound in bounds {
                                match bound {
                                    TypeParamBound::Trait(TraitBound {
                                        lifetimes, path, ..
                                    }) => {
                                        let mut visitor = self.clone();
                                        visitor.must = None;
                                        visitor.generic_lifetimes.extend(
                                            lifetimes
                                                .as_ref()
                                                .map(|ls| {
                                                    ls.lifetimes.iter().map(|gp| {
                                                        if let GenericParam::Lifetime(lt) = gp {
                                                            lt.lifetime.clone()
                                                        } else {
                                                            panic!()
                                                        }
                                                    })
                                                })
                                                .into_iter()
                                                .flatten(),
                                        );
                                        visitor.visit_path(path);
                                        if self.impossible.is_none() {
                                            self.impossible = visitor.impossible;
                                        }
                                        match (self.must.clone(), visitor.must) {
                                            (Some((s_l, _)), Some((v_l, v_m))) if v_l + 1 < s_l => {
                                                self.must = Some((v_l + 1, v_m))
                                            }
                                            (None, Some((v_l, v_m))) => {
                                                self.must = Some((v_l + 1, v_m))
                                            }
                                            _ => (),
                                        }
                                    }
                                    TypeParamBound::Verbatim(_) => {
                                        self.impossible = Some(quote!(#bound));
                                        return;
                                    }
                                    _ => (),
                                }
                            }
                            // All the bounds are checked (e.g. both `A` and `B` in `dyn A + B`).
                            return;
                        }
                        Type::Macro(_) => {
                            // The tokens of a type macro cannot be analyzed, so it is interned as
                            // a whole (it is expanded in the Leaker's context).
                            match &self.must {
                                Some((level, _)) if *level <= 0 => (),
                                _ => self.must = Some((0, quote!(#i))),
                            }
                            return;
                        }
                        Type::ImplTrait(_) | Type::Verbatim(_) => {
                            self.impossible = Some(quote!(#i));
                        }
                        Type::Reference(TypeReference { lifetime: None, .. }) => {
                            self.impossible = Some(quote!(#i));
                        }
                        _ => (),
                    }
                    let mut visitor = self.clone();
                    visitor.must = None;
                    syn::visit::visit_type(&mut visitor, i);
                    if visitor.impossible.is_some() {
                        self.impossible = visitor.impossible;
                    }
                    match (self.must.clone(), visitor.must) {
                        (Some((s_l, _)), Some((v_l, v_m))) if v_l + 1 < s_l => {
                            self.must = Some((v_l + 1, v_m))
                        }
                        (None, Some((v_l, v_m))) => self.must = Some((v_l + 1, v_m)),
                        _ => (),
                    }
                }
                fn visit_qself(&mut self, i: &QSelf) {
                    if i.as_token.is_none() {
                        let ty = &i.ty;
                        self.impossible = Some(quote!(#ty));
                    }
                    syn::visit::visit_qself(self, i)
                }
                fn visit_lifetime(&mut self, i: &Lifetime) {
                    match i.to_string().as_str() {
                        "'static" => (),
                        "'_" => self.impossible = Some(quote!(#i)),
                        _ if self.generic_lifetimes.iter().any(|lt| lt == i) => {}
                        _ => {
                            self.must = Some((-1, quote!(#i)));
                        }
                    }
                    syn::visit::visit_lifetime(self, i)
                }
                fn visit_expr(&mut self, i: &Expr) {
                    match i {
                        Expr::Closure(_) | Expr::Assign(_) | Expr::Verbatim(_) | Expr::Macro(_) => {
                            self.impossible = Some(quote!(#i));
                        }
                        _ => (),
                    }
                    syn::visit::visit_expr(self, i)
                }
                fn visit_path(&mut self, i: &Path) {
                    if self
                        .allowed_paths
                        .iter()
                        .any(|allowed| path_starts_with(i, allowed))
                    {
                        // The path itself does not depend on the type context, but its generic
                        // arguments may (e.g. `RelType` in `Option<RelType>`).
                        return syn::visit::visit_path(self, i);
                    }
                    if matches!(i.segments.iter().next(), Some(PathSegment { ident, arguments }) if ident == "Self" && arguments.is_none())
                        && i.leading_colon.is_none()
                    {
                        if !self.self_ty_can_be_interned {
                            self.impossible = Some(quote!(#i));
                        }
                    } else {
                        if i.leading_colon.is_none()
                            && matches!(
                                i.segments.iter().next(),
                                Some(PathSegment {
                                    ident,
                                    arguments: PathArguments::None,
                                }) if ident == "$crate"
                            )
                        {
                            // Same as allowed paths: the generic arguments are still checked.
                            return syn::visit::visit_path(self, i);
                        }
                        match (i.leading_colon, i.get_ident()) {
                            // i is a generic parameter (single-segment, e.g. `T`)
                            (None, Some(ident)) if self.generic_idents.contains(ident) => {}
                            // i is an associated type of a generic param (e.g. `S::Item`)
                            (None, _)
                                if i.segments.first().map_or(false, |seg| {
                                    matches!(&seg.arguments, PathArguments::None)
                                        && self.generic_idents.contains(&seg.ident)
                                }) => {}
                            // relative path, not a generic parameter
                            (None, _) => {
                                self.must = Some((-1, quote!(#i)));
                            }
                            // absolute path
                            (Some(_), _) => (),
                        }
                    }
                    syn::visit::visit_path(self, i)
                }
            }
        };
        let mut visitor = Visitor {
            generic_lifetimes: generics
                .params
                .iter()
                .filter_map(|gp| {
                    if let GenericParam::Lifetime(lt) = gp {
                        Some(lt.lifetime.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            generic_idents: generics
                .params
                .iter()
                .filter_map(|gp| match gp {
                    GenericParam::Type(TypeParam { ident, .. })
                    | GenericParam::Const(ConstParam { ident, .. }) => Some(ident.clone()),
                    _ => None,
                })
                .collect(),
            impossible: None,
            must: None,
            self_ty_can_be_interned: self.self_ty_can_be_interned,
            allowed_paths: self.allowed_paths.clone(),
        };
        visitor.visit_type(ty);
        match (visitor.must, visitor.impossible) {
            (None, None) => Ok(CheckResult::Neutral),
            (Some((0, span)), None) => Ok(CheckResult::MustIntern(span)),
            (Some((_, span)), None) => Ok(CheckResult::MustInternOrInherit(span)),
            (None, Some(span)) => Ok(CheckResult::MustNotIntern(span)),
            (Some((_, span0)), Some(span1)) => Err((span0, span1)),
        }
    }

    /// Finish building the [`Leaker`] and convert it into [`Referrer`].
    ///
    /// The types are reduced by [`Leaker::reduce_roots()`] (called here if needed), and ordered
    /// by their first appearance, so the result (including the index of each type) is
    /// deterministic.
    pub fn finish(mut self) -> Referrer {
        self.reduce_roots();
        let local_generics_map = self.local_generics;
        let leak_types: Vec<_> = self.reachable_types.unwrap_or_default();
        let map = leak_types
            .iter()
            .enumerate()
            .map(|(n, ty)| (ty.clone(), n))
            .collect();
        let local_generics = leak_types
            .iter()
            .map(|ty| local_generics_map.get(ty).cloned().unwrap_or_default())
            .collect();
        Referrer {
            leak_types,
            map,
            local_generics,
        }
    }

    /// Reduce nodes to decrease the cost of the [`Leaker`]'s implementation.
    ///
    /// # Algorithm
    ///
    /// Consider the following [`Leaker`] input:
    ///
    /// ```ignore
    /// pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
    ///     field1: MyType1,
    ///     field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
    ///     field3: &'a (T1, T2),
    /// }
    /// ```
    ///
    /// When initialized with [`Leaker::from_struct()`], the [`Leaker`] analyzes the AST and
    /// constructs a DAG representing all internable types and their dependency relations:
    ///
    /// ![before_graph](https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/resources/before_graph.svg)
    ///
    /// **Red** nodes are flagged as [`CheckResult::MustIntern`] by [`Leaker::check()`], meaning
    /// the type literally depends on the type context and must be interned.
    ///
    /// This algorithm reduces the node set while ensuring that every root type (annotated with ★)
    /// can still be expressed using the remaining **red** nodes.
    ///
    /// There are two degrees of freedom in this reduction:
    ///
    /// - Intern all **red** nodes and ignore the rest (since non-red nodes either do not need to
    ///   be interned or can be reconstructed from red nodes).
    /// - Skip interning individual **red** nodes and instead intern their common ancestors when
    ///   doing so is more efficient.
    ///
    /// The final result looks like:
    ///
    /// ![after_graph](https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/resources/after_graph.svg)
    pub fn reduce_roots(&mut self) {
        if self.reachable_types.is_none() {
            self.reachable_types = Some(self.build_graph_and_reduce(&self.operations));
        }
    }

    /// Returns the types to be interned, in the order of their first appearance in `operations`.
    fn build_graph_and_reduce(&self, operations: &[GraphOperation]) -> Vec<Type> {
        // Build parent-child relationships from operations
        let mut children: HashMap<Type, Vec<Type>> = HashMap::new();
        let mut parents: HashMap<Type, Vec<Type>> = HashMap::new();
        let mut root_types = HashSet::new();
        let mut must_intern_types = HashSet::new();

        for operation in operations {
            if operation.is_root {
                root_types.insert(operation.node_type.clone());
            }
            if operation.is_must_intern {
                must_intern_types.insert(operation.node_type.clone());
            }

            if let Some(parent_type) = &operation.parent {
                children
                    .entry(parent_type.clone())
                    .or_default()
                    .push(operation.node_type.clone());
                parents
                    .entry(operation.node_type.clone())
                    .or_default()
                    .push(parent_type.clone());
            }
        }

        // Implement proper reduction algorithm:
        // 1. Keep all must-intern root types
        // 2. Keep root types that transitively contain must-intern types
        // 3. Do NOT keep non-root must-intern types that are only components of kept root types

        let mut kept_types = HashSet::new();

        // Step 1: Keep must-intern types that are also roots
        for root_type in &root_types {
            if must_intern_types.contains(root_type) {
                kept_types.insert(root_type.clone());
            }
        }

        // Step 2: Keep root types that contain must-intern descendants
        for root_type in &root_types {
            if !kept_types.contains(root_type)
                && self.contains_must_intern_descendant(root_type, &children, &must_intern_types)
            {
                kept_types.insert(root_type.clone());
            }
        }

        // Step 3: Only add must-intern types that are NOT components of any kept type
        for must_intern_type in &must_intern_types {
            if !self.is_component_of_any_kept_type(must_intern_type, &kept_types, &children) {
                kept_types.insert(must_intern_type.clone());
            }
        }

        // Order by the first appearance, for deterministic output.
        let mut seen = HashSet::new();
        operations
            .iter()
            .map(|operation| &operation.node_type)
            .filter(|ty| kept_types.contains(*ty) && seen.insert((*ty).clone()))
            .cloned()
            .collect()
    }

    fn contains_must_intern_descendant(
        &self,
        root: &Type,
        children: &HashMap<Type, Vec<Type>>,
        must_intern_types: &HashSet<Type>,
    ) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![root.clone()];

        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if must_intern_types.contains(&current) {
                return true;
            }

            if let Some(child_list) = children.get(&current) {
                for child in child_list {
                    if !visited.contains(child) {
                        stack.push(child.clone());
                    }
                }
            }
        }

        false
    }

    fn is_component_of_any_kept_type(
        &self,
        target_type: &Type,
        kept_types: &HashSet<Type>,
        children: &HashMap<Type, Vec<Type>>,
    ) -> bool {
        for kept_type in kept_types {
            if self.is_descendant_of(target_type, kept_type, children) {
                return true;
            }
        }
        false
    }

    fn is_descendant_of(
        &self,
        target: &Type,
        ancestor: &Type,
        children: &HashMap<Type, Vec<Type>>,
    ) -> bool {
        if target == ancestor {
            return false; // Not a descendant of itself
        }

        let mut visited = HashSet::new();
        let mut stack = vec![ancestor.clone()];

        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if let Some(child_list) = children.get(&current) {
                for child in child_list {
                    if child == target {
                        return true;
                    }
                    if !visited.contains(child) {
                        stack.push(child.clone());
                    }
                }
            }
        }

        false
    }
}

/// Holds the list of types that need to be encoded, as produced by [`Leaker::finish()`].
///
/// `Referrer` provides methods to iterate over leak types and to expand or transform
/// types using a custom mapping function.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    leak_types: Vec<Type>,
    map: HashMap<Type, usize>,
    local_generics: Vec<Generics>,
}

impl Referrer {
    /// Returns `true` if there are no types to be encoded.
    pub fn is_empty(&self) -> bool {
        self.leak_types.is_empty()
    }

    /// Returns an iterator over the types that need to be encoded.
    ///
    /// ```ignore
    /// let args = encode_generics_params_to_ty(&generics.params);
    /// referrer.iter().enumerate().map(|(ix, ty)| quote!{
    ///     impl #crate::TypeRef<#ix, #args> for #marker_ty {
    ///         type Type = #ty;
    ///     }
    /// }).collect::<TokenStream>();
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.leak_types.iter()
    }

    /// Returns the local generics which the `ix`-th type depends on, i.e. generics of a trait
    /// method or a GAT which are not generics of the Leaker. It is empty for most types.
    ///
    /// A type such as `Wrap<B>` in `fn f<const B: bool>(&self) -> Wrap<B>` (where `Wrap` depends
    /// on the type context) cannot be decomposed, so it is interned together with `B`. The
    /// Repeater's implementation must then declare these generics and constrain them in its trait
    /// arguments, e.g. with [`encode_generics_params_to_ty()`]. The Referrer declares the same
    /// generics with the same names (as it copies the method's signature), so it can encode them
    /// in the same way. Note that lifetimes must be declared before the other generics when
    /// merging them with the Leaker's generics.
    ///
    /// The bounds and where predicates of the local generics are kept as they are, so they may
    /// mention `Self` (e.g. `fn f<T: PartialEq<Self>>(&self)`), which refers to the implementor of
    /// the Leaker trait. The Repeater's implementation should therefore be for the implementor
    /// (e.g. `impl<.., S: ?Sized> Repeater<..> for S where S: MyTrait<..>`), where `Self` has the
    /// same meaning; in an implementation for another type (e.g. a marker type), `Self` would
    /// refer to that type instead.
    ///
    ///
    /// ```ignore
    /// // In the Leaker (`MyTrait` is the Leaker trait with `generics`)
    /// for (ix, (ty, locals)) in referrer.iter_with_local_generics().enumerate() {
    ///     let args = encode_generics_params_to_ty(&generics.params);
    ///     let local_args = encode_generics_params_to_ty(&locals.params);
    ///     let local_where = locals.where_clause.iter().flat_map(|wc| wc.predicates.iter());
    ///     quote! {
    ///         impl<#generics_params, #locals_params, S: ?Sized>
    ///             Repeater<(#args, #local_args), #ID, #ix> for S
    ///         where
    ///             S: MyTrait<#generics_args>,
    ///             #(#local_where,)*
    ///         {
    ///             type Type = #ty;
    ///         }
    ///     }
    /// }
    ///
    /// // In the Referrer (in the implementation of `MyTrait<#trait_args>`)
    /// let args = encode_generics_args_to_ty_with(&trait_args, Some(&generics.params), |expr, _| {
    ///     parse_quote!([(); #expr])
    /// })?;
    /// let local_args = encode_generics_params_to_ty(&referrer.local_generics(ix).params);
    /// parse_quote!(<Self as Repeater<(#args, #local_args), #ID, #ix>>::Type)
    /// ```
    pub fn local_generics(&self, ix: usize) -> &Generics {
        &self.local_generics[ix]
    }

    /// Returns an iterator over the types that need to be encoded, together with their local
    /// generics. See [`Referrer::local_generics()`].
    pub fn iter_with_local_generics(&self) -> impl Iterator<Item = (&Type, &Generics)> {
        self.leak_types.iter().zip(self.local_generics.iter())
    }

    /// Converts this `Referrer` into a [`Visitor`] that can be used to transform AST nodes.
    ///
    /// The `type_fn` callback is called for each type that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type.
    /// See [`Referrer::expand()`] for details.
    pub fn into_visitor<F: FnMut(Type, usize) -> Type>(self, type_fn: F) -> Visitor<F> {
        Visitor(self, Rc::new(RefCell::new(type_fn)))
    }

    /// Expands a type by replacing any interned types using the provided mapping function.
    ///
    /// The `type_fn` callback is called for each subtype that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type.
    ///
    /// ```ignore
    /// let args = encode_generics_params_to_ty(&generics.params);
    /// referrer.expand(ty, |_, ix| {
    ///     parse_quote!(<#marker_ty as #crate::TypeRef<#ix, #args>>::Type)
    /// })
    /// ```
    pub fn expand(&self, ty: Type, type_fn: impl FnMut(Type, usize) -> Type) -> Type {
        use syn::fold::Fold;
        struct Folder<'a, F>(&'a Referrer, F);
        impl<'a, F: FnMut(Type, usize) -> Type> Fold for Folder<'a, F> {
            fn fold_type(&mut self, ty: Type) -> Type {
                if let Some(idx) = self.0.map.get(&ty) {
                    self.1(ty, *idx)
                } else {
                    syn::fold::fold_type(self, ty)
                }
            }
        }
        let mut folder = Folder(self, type_fn);
        folder.fold_type(ty)
    }
}

#[derive(Debug)]
pub struct Visitor<F>(Referrer, Rc<RefCell<F>>);

impl<F> Clone for Visitor<F> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<F: FnMut(Type, usize) -> Type> Visitor<F> {
    /// Stops rewriting the leaked types which mention the generic parameters declared here (they
    /// refer to these parameters rather than to the Leaker's items), unless the leaked type was
    /// interned together with local generics of the same name (see
    /// [`Referrer::local_generics()`]).
    fn shadow_generics(&mut self, generics: &Generics) {
        let referrer = &self.0;
        let shadowed: Vec<Type> = referrer
            .map
            .iter()
            .filter(|(ty, ix)| {
                let locals = &referrer.local_generics[**ix];
                let params = generics.params.iter().filter(|gp| {
                    !locals
                        .params
                        .iter()
                        .any(|local| generic_param_name(local) == generic_param_name(gp))
                });
                mentions_generics(ty, params)
            })
            .map(|(ty, _)| ty.clone())
            .collect();
        for ty in shadowed {
            self.0.map.remove(&ty);
        }
    }

    /// Stops rewriting the leaked types which mention `idents` (items or imports declared in a
    /// block or a module).
    fn shadow_idents(&mut self, idents: Vec<Ident>) {
        let shadowed: Vec<Type> = self
            .0
            .map
            .keys()
            .filter(|ty| mentions_names(ty, idents.clone(), Vec::new()))
            .cloned()
            .collect();
        for ty in shadowed {
            self.0.map.remove(&ty);
        }
    }

    /// Shadows the items declared in `items`.
    fn shadow_items<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>) {
        fn use_names(tree: &UseTree, names: &mut Vec<Ident>) {
            match tree {
                UseTree::Path(UsePath { tree, .. }) => use_names(tree, names),
                UseTree::Name(UseName { ident }) if ident != "self" => names.push(ident.clone()),
                UseTree::Rename(UseRename { rename, .. }) => names.push(rename.clone()),
                UseTree::Group(UseGroup { items, .. }) => {
                    for tree in items {
                        use_names(tree, names)
                    }
                }
                _ => (),
            }
        }
        let mut names = Vec::new();
        for item in items {
            match item {
                Item::Struct(ItemStruct { ident, .. })
                | Item::Enum(ItemEnum { ident, .. })
                | Item::Union(ItemUnion { ident, .. })
                | Item::Trait(ItemTrait { ident, .. })
                | Item::TraitAlias(ItemTraitAlias { ident, .. })
                | Item::Type(ItemType { ident, .. })
                | Item::Const(ItemConst { ident, .. })
                | Item::Static(ItemStatic { ident, .. })
                | Item::Mod(ItemMod { ident, .. }) => names.push(ident.clone()),
                Item::Use(ItemUse { tree, .. }) => use_names(tree, &mut names),
                _ => (),
            }
        }
        if !names.is_empty() {
            self.shadow_idents(names);
        }
    }

    fn with_generics(&mut self, generics: &mut Generics) -> Self {
        let mut visitor = self.clone();
        // All the parameters are in scope of each other's bounds.
        visitor.shadow_generics(generics);
        for gp in generics.params.iter_mut() {
            visitor.visit_generic_param_mut(gp);
        }
        if let Some(where_clause) = &mut generics.where_clause {
            visitor.visit_where_clause_mut(where_clause);
        }
        visitor
    }

    fn with_signature(&mut self, sig: &mut Signature) -> Self {
        let mut visitor = self.with_generics(&mut sig.generics);
        for input in sig.inputs.iter_mut() {
            visitor.visit_fn_arg_mut(input);
        }
        visitor.visit_return_type_mut(&mut sig.output);
        visitor
    }
}

impl<F: FnMut(Type, usize) -> Type> VisitMut for Visitor<F> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        *i = self.0.expand(i.clone(), &mut *self.1.borrow_mut());
    }
    fn visit_const_param_mut(&mut self, i: &mut ConstParam) {
        // The type of a const parameter cannot be a projection, so it is kept as is.
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        if let Some(expr) = &mut i.default {
            self.visit_expr_mut(expr);
        }
    }
    fn visit_item_struct_mut(&mut self, i: &mut ItemStruct) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_fields_mut(&mut i.fields);
    }
    fn visit_item_enum_mut(&mut self, i: &mut ItemEnum) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        for variant in i.variants.iter_mut() {
            visitor.visit_variant_mut(variant);
        }
    }
    fn visit_item_trait_mut(&mut self, i: &mut ItemTrait) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        for supertrait in i.supertraits.iter_mut() {
            visitor.visit_type_param_bound_mut(supertrait);
        }
        for item in i.items.iter_mut() {
            visitor.visit_trait_item_mut(item);
        }
    }
    fn visit_item_union_mut(&mut self, i: &mut ItemUnion) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_fields_named_mut(&mut i.fields);
    }
    fn visit_item_type_mut(&mut self, i: &mut ItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
    }
    fn visit_signature_mut(&mut self, i: &mut Signature) {
        self.with_signature(i);
    }
    fn visit_item_impl_mut(&mut self, i: &mut ItemImpl) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        if let Some((_, path, _)) = &mut i.trait_ {
            visitor.visit_path_mut(path);
        }
        visitor.visit_type_mut(&mut i.self_ty);
        for item in i.items.iter_mut() {
            visitor.visit_impl_item_mut(item);
        }
    }
    fn visit_impl_item_fn_mut(&mut self, i: &mut ImplItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature(&mut i.sig);
        visitor.visit_block_mut(&mut i.block);
    }
    fn visit_impl_item_type_mut(&mut self, i: &mut ImplItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
    }
    fn visit_item_mod_mut(&mut self, i: &mut ItemMod) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        if let Some((_, items)) = &mut i.content {
            let mut visitor = self.clone();
            visitor.shadow_items(items.iter());
            for item in items.iter_mut() {
                visitor.visit_item_mut(item);
            }
        }
    }
    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature(&mut i.sig);
        visitor.visit_block_mut(i.block.as_mut());
    }
    fn visit_trait_item_fn_mut(&mut self, i: &mut TraitItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature(&mut i.sig);
        if let Some(block) = &mut i.default {
            visitor.visit_block_mut(block);
        }
    }
    fn visit_trait_item_type_mut(&mut self, i: &mut TraitItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        for bound in i.bounds.iter_mut() {
            visitor.visit_type_param_bound_mut(bound);
        }
        if let Some((_, ty)) = &mut i.default {
            visitor.visit_type_mut(ty);
        }
    }
    fn visit_trait_item_const_mut(&mut self, i: &mut TraitItemConst) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
        if let Some((_, expr)) = &mut i.default {
            visitor.visit_expr_mut(expr);
        }
    }
    fn visit_block_mut(&mut self, i: &mut Block) {
        let mut visitor = self.clone();
        visitor.shadow_items(i.stmts.iter().filter_map(|stmt| match stmt {
            Stmt::Item(item) => Some(item),
            _ => None,
        }));
        for stmt in i.stmts.iter_mut() {
            visitor.visit_stmt_mut(stmt);
        }
    }
}

impl Parse for Referrer {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let content: syn::parse::ParseBuffer<'_>;
        parenthesized!(content in input);
        // Each entry is either `Type` or `{ <local generics> where ..; Type }`.
        let entries =
            Punctuated::<(Generics, Type), Token![,]>::parse_terminated_with(&content, |input| {
                if input.peek(token::Brace) {
                    let content: syn::parse::ParseBuffer<'_>;
                    braced!(content in input);
                    let mut generics: Generics = content.parse()?;
                    generics.where_clause = content.parse()?;
                    content.parse::<Token![;]>()?;
                    Ok((generics, content.parse()?))
                } else {
                    Ok((Generics::default(), input.parse()?))
                }
            })?;
        let (local_generics, leak_types): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
        let map = leak_types
            .iter()
            .enumerate()
            .map(|(n, ty)| (ty.clone(), n))
            .collect();
        Ok(Self {
            map,
            leak_types,
            local_generics,
        })
    }
}

impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        tokens.extend(quote! {
            (
                #(for (item, generics) in self.leak_types.iter().zip(&self.local_generics)), {
                    #(if generics.params.is_empty()) {
                        #item
                    } #(else) {
                        { #generics #{&generics.where_clause}; #item }
                    }
                }
            )
        })
    }
}

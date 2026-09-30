#![doc = include_str!("./README.md")]

use proc_macro2::{Span, TokenStream};
use safegraph::graph::edge::Endpoints;
use safegraph::graph::Graph;
use safegraph::VecGraph;
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
    graph: VecGraph<Type, ()>,
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
    reachable_types: HashSet<Type>,
    pending_operations: Option<Vec<GraphOperation>>,
    local_generics: HashMap<Type, Generics>,
}

/// Error indicating that a type is not internable.
#[derive(Debug, Clone)]
pub struct NotInternableError(pub TokenStream);

impl std::fmt::Display for NotInternableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("use absolute path: '{}'", &self.0))
    }
}

impl std::error::Error for NotInternableError {}

impl NotInternableError {
    pub fn span(&self) -> Span {
        self.0.span()
    }
}

/// Encode [`GenericArgument`]s to a type.
///
/// A const argument `N` is encoded as `[(); N]`, so it must be of type `usize`. Use
/// [`encode_generics_args_to_ty_with()`] to encode const arguments of other types.
///
/// A bare identifier such as `N` in `MyTrait<N>` is parsed as a type argument even if it refers to
/// a const. Use [`encode_generics_args_to_ty_with()`] with the corresponding [`GenericParam`]s to
/// encode such arguments correctly.
///
/// # Panics
///
/// Panics if an argument cannot be encoded (see [`EncodeError::UnsupportedArgument`]).
pub fn encode_generics_args_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericArgument>) -> Type {
    encode_generics_args_to_ty_with(iter, None, |expr, _| parse_quote!([(); #expr]))
}

/// Encode [`GenericArgument`]s to a type, using `const_fn` to encode const arguments.
///
/// If `params` is given, its type and const parameters are matched against the type and const
/// arguments of `iter` by position. A type argument that
/// corresponds to a [`ConstParam`] and consists only of a single identifier (e.g. `N` in
/// `MyTrait<N>`) is then treated as a const argument. `const_fn` receives each const argument
/// together with its corresponding [`ConstParam`] (if known), and must return the same type as the
/// `const_fn` passed to [`encode_generics_params_to_ty_with()`] for the same value.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::quote;
/// let generics: Generics = parse_quote!(<T, const N: usize>);
/// let args: AngleBracketedGenericArguments = parse_quote!(<u8, N>);
/// let ty = encode_generics_args_to_ty_with(&args.args, Some(&generics.params), |expr, _| {
///     parse_quote!([(); #expr])
/// });
/// assert_eq!(quote!(#ty).to_string(), quote!((u8, [(); N])).to_string());
/// ```
///
/// # Panics
///
/// Panics if an argument cannot be encoded (see [`EncodeError::UnsupportedArgument`]).
pub fn encode_generics_args_to_ty_with<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: Option<&'b Punctuated<GenericParam, Token![,]>>,
    mut const_fn: impl FnMut(&Expr, Option<&'b ConstParam>) -> Type,
) -> Type {
    try_encode_generics_args_to_ty_with(iter, params, |expr, cp| Ok(const_fn(expr, cp)))
        .unwrap_or_else(|e| panic!("{}", e))
}

/// Fallible version of [`encode_generics_args_to_ty_with()`].
fn try_encode_generics_args_to_ty_with<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: Option<&'b Punctuated<GenericParam, Token![,]>>,
    mut const_fn: impl FnMut(&Expr, Option<&'b ConstParam>) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    // Lifetime arguments may be omitted, so type and const arguments are matched only against type
    // and const parameters.
    let mut params = params.map(|params| {
        params
            .iter()
            .filter(|param| !matches!(param, GenericParam::Lifetime(_)))
    });
    Ok(Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: iter
            .into_iter()
            .map(|arg| -> std::result::Result<Type, EncodeError> {
                let param = if matches!(arg, GenericArgument::Lifetime(_)) {
                    None
                } else {
                    params.as_mut().and_then(|params| params.next())
                };
                let const_param = if let Some(GenericParam::Const(cp)) = param {
                    Some(cp)
                } else {
                    None
                };
                Ok(match (arg, const_param) {
                    (GenericArgument::Lifetime(lifetime), _) => {
                        parse_quote!(& #lifetime ())
                    }
                    (GenericArgument::Const(expr), _) => const_fn(unwrap_block(expr), const_param)?,
                    (
                        GenericArgument::Type(Type::Path(TypePath { qself: None, path })),
                        Some(cp),
                    ) if path.get_ident().is_some() => const_fn(
                        &Expr::Path(ExprPath {
                            attrs: Vec::new(),
                            qself: None,
                            path: path.clone(),
                        }),
                        Some(cp),
                    )?,
                    (GenericArgument::Type(ty), _) => ty.clone(),
                    (arg, _) => {
                        return Err(EncodeError::UnsupportedArgument(Box::new(arg.clone())))
                    }
                })
            })
            .collect::<std::result::Result<_, _>>()?,
    }))
}

/// Primitive types which can be the type of const parameters on stable Rust, paired with the name
/// of the marker type used by [`ConstMarkers`].
const CONST_PARAM_PRIMITIVES: &[(&str, &str)] = &[
    ("bool", "ConstBool"),
    ("char", "ConstChar"),
    ("u8", "ConstU8"),
    ("u16", "ConstU16"),
    ("u32", "ConstU32"),
    ("u64", "ConstU64"),
    ("u128", "ConstU128"),
    ("usize", "ConstUsize"),
    ("i8", "ConstI8"),
    ("i16", "ConstI16"),
    ("i32", "ConstI32"),
    ("i64", "ConstI64"),
    ("i128", "ConstI128"),
    ("isize", "ConstIsize"),
];

/// Returns the name of the primitive type if `ty` is a valid type of const parameters which does
/// not depend on the type context.
fn const_param_primitive(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Path(TypePath { qself: None, path }) => {
            let segs: Vec<_> = path
                .segments
                .iter()
                .map(|seg| {
                    if seg.arguments.is_none() {
                        Some(seg.ident.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Option<_>>()?;
            let name = match (path.leading_colon.is_some(), segs.as_slice()) {
                (false, [s]) => s,
                (_, [c, p, s]) if (c == "core" || c == "std") && p == "primitive" => s,
                _ => return None,
            };
            CONST_PARAM_PRIMITIVES
                .iter()
                .find(|(prim, _)| prim == name)
                .map(|(prim, _)| *prim)
        }
        Type::Group(TypeGroup { elem, .. }) | Type::Paren(TypeParen { elem, .. }) => {
            const_param_primitive(elem)
        }
        _ => None,
    }
}

/// Returns `true` if `ty` is a valid type of const parameters which does not depend on the type
/// context.
fn is_const_param_ty(ty: &Type) -> bool {
    const_param_primitive(ty).is_some()
}

/// Returns `true` if `ty` refers to any of `generics`.
fn mentions_generics<'a>(ty: &Type, generics: impl IntoIterator<Item = &'a GenericParam>) -> bool {
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
    }
    let mut visitor = Visitor {
        idents: Vec::new(),
        lifetimes: Vec::new(),
        found: false,
    };
    for gp in generics {
        match gp {
            GenericParam::Type(TypeParam { ident, .. })
            | GenericParam::Const(ConstParam { ident, .. }) => visitor.idents.push(ident.clone()),
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                visitor.lifetimes.push(lifetime.clone())
            }
        }
    }
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
/// `const N: usize`. Use [`encode_generics_params_to_ty_with()`] to encode const parameters of
/// other types (e.g. with a marker type `MyConstBool<const B: bool>`).
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
    match try_encode_generics_params_to_ty_with(iter, |cp| {
        Ok::<_, std::convert::Infallible>(const_fn(cp))
    }) {
        Ok(ty) => ty,
        Err(e) => match e {},
    }
}

/// Fallible version of [`encode_generics_params_to_ty_with()`].
fn try_encode_generics_params_to_ty_with<'a, E>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    mut const_fn: impl FnMut(&'a ConstParam) -> std::result::Result<Type, E>,
) -> std::result::Result<Type, E> {
    Ok(Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: iter
            .into_iter()
            .map(|param| -> std::result::Result<Type, E> {
                Ok(match param {
                    GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                        parse_quote!(& #lifetime ())
                    }
                    GenericParam::Const(cp) => const_fn(cp)?,
                    GenericParam::Type(TypeParam { ident, .. }) => parse_quote!(#ident),
                })
            })
            .collect::<std::result::Result<_, _>>()?,
    }))
}

/// Error returned by [`ConstMarkers`] when generics cannot be encoded into a type.
#[derive(Clone, Debug)]
pub enum EncodeError {
    /// The type of a const parameter is not a primitive type (e.g. a type alias such as
    /// `::my_crate::Idx`), so the marker type cannot be chosen.
    UnsupportedConstType(Box<Type>),
    /// A generic argument which does not correspond to a generic parameter, such as an associated
    /// type binding (`Item = u8`).
    UnsupportedArgument(Box<GenericArgument>),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::UnsupportedConstType(ty) => write!(
                f,
                "unsupported type of const parameter: '{}' (use a primitive type)",
                quote!(#ty)
            ),
            EncodeError::UnsupportedArgument(arg) => {
                write!(f, "unsupported generic argument: '{}'", quote!(#arg))
            }
        }
    }
}

impl std::error::Error for EncodeError {}

impl EncodeError {
    pub fn span(&self) -> Span {
        match self {
            EncodeError::UnsupportedConstType(ty) => ty.span(),
            EncodeError::UnsupportedArgument(arg) => arg.span(),
        }
    }
}

/// Marker types to encode const generics of any type into types.
///
/// [`encode_generics_params_to_ty()`] encodes a const parameter `N` as `[(); N]`, which only works
/// for `const N: usize`, because Rust (without `generic_const_exprs`) does not allow operations
/// such as `N as usize` on generic parameters, and there is no type in `core` with a const
/// parameter of other types. `ConstMarkers` instead encodes a const parameter as a marker type,
/// such as `path::to::markers::ConstBool<B>` for `const B: bool`.
///
/// The marker types must be defined with [`ConstMarkers::definitions()`] at a module which is
/// accessible from both the Leaker and the Referrer (like the Repeater), and that module's path is
/// given to [`ConstMarkers::new()`].
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::quote;
/// let markers = ConstMarkers::new(parse_quote!(::my_crate::markers));
///
/// // In the Leaker
/// let generics: Generics = parse_quote!(<T, const B: bool, const C: char, const N: usize>);
/// let ty = markers.encode_generics_params_to_ty(&generics.params).unwrap();
/// let expected: Type = parse_quote!((
///     T,
///     ::my_crate::markers::ConstBool<B>,
///     ::my_crate::markers::ConstChar<C>,
///     [(); N]
/// ));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
///
/// // In the Referrer
/// let args: AngleBracketedGenericArguments = parse_quote!(<u8, B, 'x', { 1 + 2 }>);
/// let ty = markers.encode_generics_args_to_ty(&args.args, &generics.params).unwrap();
/// let expected: Type = parse_quote!((
///     u8,
///     ::my_crate::markers::ConstBool<B>,
///     ::my_crate::markers::ConstChar<'x'>,
///     [(); 1 + 2]
/// ));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstMarkers {
    /// Path to the module containing the marker types.
    pub path: Path,
}

impl ConstMarkers {
    /// Specifies the path to the module containing the definitions made by
    /// [`ConstMarkers::definitions()`].
    pub fn new(path: Path) -> Self {
        Self { path }
    }

    /// Definitions of the marker types, such as `pub struct ConstBool<const V: bool>;`.
    ///
    /// ```
    /// # use type_leak::*;
    /// let defs = ConstMarkers::definitions();
    /// assert!(defs.to_string().contains("ConstBool"));
    /// ```
    pub fn definitions() -> TokenStream {
        CONST_PARAM_PRIMITIVES
            .iter()
            .map(|(prim, marker)| {
                let prim = Ident::new(prim, Span::call_site());
                let marker = Ident::new(marker, Span::call_site());
                quote! {
                    #[doc(hidden)]
                    pub struct #marker<const V: ::core::primitive::#prim>;
                }
            })
            .collect()
    }

    /// Encodes a const value `value` of type `ty` into a type. A `usize` value is encoded as
    /// `[(); value]`.
    ///
    /// Returns [`EncodeError::UnsupportedConstType`] if `ty` is not a primitive type.
    pub fn encode_const(&self, value: &Expr, ty: &Type) -> std::result::Result<Type, EncodeError> {
        let prim = const_param_primitive(ty)
            .ok_or_else(|| EncodeError::UnsupportedConstType(Box::new(ty.clone())))?;
        let value = unwrap_block(value);
        if prim == "usize" {
            return Ok(parse_quote!([(); #value]));
        }
        let marker = CONST_PARAM_PRIMITIVES
            .iter()
            .find(|(p, _)| *p == prim)
            .map(|(_, marker)| Ident::new(marker, Span::call_site()))
            .unwrap();
        let path = &self.path;
        Ok(match value {
            Expr::Lit(_) | Expr::Path(_) => parse_quote!(#path::#marker<#value>),
            _ => parse_quote!(#path::#marker<{ #value }>),
        })
    }

    /// Encodes [`GenericParam`]s into a type, like [`encode_generics_params_to_ty()`].
    ///
    /// Returns [`EncodeError::UnsupportedConstType`] if the type of a const parameter is not a
    /// primitive type.
    pub fn encode_generics_params_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
    ) -> std::result::Result<Type, EncodeError> {
        try_encode_generics_params_to_ty_with(iter, |ConstParam { ident, ty, .. }| {
            self.encode_const(&parse_quote!(#ident), ty)
        })
    }

    /// Encodes [`GenericArgument`]s into a type, like [`encode_generics_args_to_ty()`].
    /// `params` are the [`GenericParam`]s of the Leaker corresponding to `iter`, which are used
    /// to find the types of const arguments.
    ///
    /// The result is the same type as [`ConstMarkers::encode_generics_params_to_ty()`] with
    /// `params` substituted by `iter`.
    ///
    /// Returns [`EncodeError::UnsupportedConstType`] if the type of a const parameter is not a
    /// primitive type, and [`EncodeError::UnsupportedArgument`] if an argument cannot be encoded
    /// (e.g. an associated type binding).
    pub fn encode_generics_args_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
    ) -> std::result::Result<Type, EncodeError> {
        try_encode_generics_args_to_ty_with(iter, Some(params), |expr, cp| match cp {
            Some(ConstParam { ty, .. }) => self.encode_const(expr, ty),
            None => Ok(parse_quote!([(); #expr])),
        })
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
            Err((_, s)) => {
                // Emit error and terminate searching
                self.error = Some(s);
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
        syn::visit::visit_trait_bound(self, i)
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
            graph: VecGraph::default(),
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
            reachable_types: HashSet::new(),
            pending_operations: None,
            local_generics: HashMap::new(),
        }
    }

    /// Initialize [`Leaker`] from configuration.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            graph: VecGraph::default(),
            allowed_paths: config.allowed_paths,
            self_ty_can_be_interned: config.self_ty_can_be_interned,
            reachable_types: HashSet::new(),
            pending_operations: None,
            local_generics: HashMap::new(),
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
                }) if ident.to_string() == "$crate"
            )
            || self.allowed_paths.iter().any(|allowed| {
                allowed.leading_colon == path.leading_colon
                    && allowed.segments.len() <= path.segments.len()
                    && allowed.segments.iter().zip(path.segments.iter()).all(
                        |(allowed_seg, seg)| {
                            allowed_seg.arguments == seg.arguments
                                && allowed_seg.ident.to_string() == seg.ident.to_string()
                        },
                    )
            })
    }

    /// Apply collected graph operations using `scope_mut()`.
    fn apply_operations(&mut self, operations: Vec<GraphOperation>) {
        if operations.is_empty() {
            return;
        }

        // Store operations for proper graph reduction in reduce_roots()
        self.pending_operations
            .get_or_insert_with(Vec::new)
            .extend(operations);
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
            Err(NotInternableError(e))
        } else {
            let operations = visitor.operations;
            for operation in &operations {
                if operation.local_generics.params.is_empty() {
                    continue;
                }
                match self.local_generics.get(&operation.node_type) {
                    Some(existing) if existing != &operation.local_generics => {
                        // The same type depends on different local generics (e.g. `Wrap<B>` in
                        // `fn f<const B: bool>` and `fn g<const B: u8>`).
                        let ty = &operation.node_type;
                        return Err(NotInternableError(quote!(#ty)));
                    }
                    Some(_) => (),
                    None => {
                        self.local_generics.insert(
                            operation.node_type.clone(),
                            operation.local_generics.clone(),
                        );
                    }
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
        use syn::visit::Visit;
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
            use syn::visit::Visit;
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
                                        return;
                                    }
                                    TypeParamBound::Verbatim(_) => {
                                        self.impossible = Some(quote!(#bound));
                                        return;
                                    }
                                    _ => (),
                                }
                            }
                        }
                        Type::ImplTrait(_) | Type::Macro(_) | Type::Verbatim(_) => {
                            self.impossible = Some(quote!(#i));
                        }
                        Type::Reference(TypeReference { lifetime, .. }) => {
                            if lifetime.is_none() {
                                self.impossible = Some(quote!(#i));
                            }
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
                    if self.allowed_paths.iter().any(|allowed| {
                        allowed.leading_colon == i.leading_colon
                            && allowed.segments.len() <= i.segments.len()
                            && allowed.segments.iter().zip(i.segments.iter()).all(
                                |(allowed_seg, seg)| {
                                    allowed_seg.arguments == seg.arguments
                                        && allowed_seg.ident.to_string() == seg.ident.to_string()
                                },
                            )
                    }) {
                        return;
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
                                }) if ident.to_string() == "$crate"
                            )
                        {
                            return;
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
    pub fn finish(self) -> Referrer {
        // Use the reachable types computed by apply_operations
        let local_generics_map = self.local_generics;
        let leak_types: Vec<_> = self.reachable_types.into_iter().collect();
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
    /// ![before_graph](https://raw.github.com/yasuo-ozu/type-leak/master/resources/before_graph.svg)
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
    /// ![after_graph](https://raw.github.com/yasuo-ozu/type-leak/master/resources/after_graph.svg)
    pub fn reduce_roots(&mut self) {
        if let Some(operations) = self.pending_operations.take() {
            self.build_graph_and_reduce(operations);
        }
        self.reduce_unreachable_nodes();
        self.reduce_obvious_nodes();
    }

    fn build_graph_and_reduce(&mut self, operations: Vec<GraphOperation>) {
        // Build parent-child relationships from operations
        let mut children: HashMap<Type, Vec<Type>> = HashMap::new();
        let mut parents: HashMap<Type, Vec<Type>> = HashMap::new();
        let mut root_types = HashSet::new();
        let mut must_intern_types = HashSet::new();

        for operation in &operations {
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

        self.reachable_types = kept_types;
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

    fn reduce_obvious_nodes(&mut self) {
        // Remove intermediate nodes that have exactly one parent and one child
        // This step eliminates unnecessary intermediate types
    }

    fn reduce_unreachable_nodes(&mut self) {
        if let Some(operations) = self.pending_operations.take() {
            self.apply_vecgraph_reduction(operations);
        }
    }

    fn apply_vecgraph_reduction(&mut self, operations: Vec<GraphOperation>) {
        let mut root_types = HashSet::new();
        let mut must_intern_types = HashSet::new();

        // Collect graph operations
        for operation in &operations {
            if operation.is_root {
                root_types.insert(operation.node_type.clone());
            }
            if operation.is_must_intern {
                must_intern_types.insert(operation.node_type.clone());
            }
        }

        // Build the graph using scope_mut
        self.graph.scope_mut(|mut ctx| {
            let mut type_to_node = HashMap::new();

            // Add all nodes first
            for operation in &operations {
                if !type_to_node.contains_key(&operation.node_type) {
                    let node_tag = ctx.insert_node(operation.node_type.clone()).unwrap();
                    type_to_node.insert(operation.node_type.clone(), node_tag);
                }

                if let Some(parent_type) = &operation.parent {
                    if !type_to_node.contains_key(parent_type) {
                        let parent_node_tag = ctx.insert_node(parent_type.clone()).unwrap();
                        type_to_node.insert(parent_type.clone(), parent_node_tag);
                    }
                }
            }

            // Add edges between nodes
            for operation in &operations {
                if let Some(parent_type) = &operation.parent {
                    let parent_node = type_to_node[parent_type];
                    let child_node = type_to_node[&operation.node_type];
                    let ep =
                        <_ as Endpoints>::try_from_node_indices([parent_node, child_node]).unwrap();
                    ctx.insert_edge((), ep).unwrap();
                }
            }
        });

        // Perform analysis using the same logic as build_graph_and_reduce
        self.reachable_types =
            self.analyze_reachable_types(&root_types, &must_intern_types, &operations);
    }

    fn analyze_reachable_types(
        &self,
        root_types: &HashSet<Type>,
        must_intern_types: &HashSet<Type>,
        operations: &[GraphOperation],
    ) -> HashSet<Type> {
        // Implement the same logic as build_graph_and_reduce
        let mut final_types = HashSet::new();

        // Keep must-intern root types
        for root_type in root_types {
            if must_intern_types.contains(root_type) {
                final_types.insert(root_type.clone());
            }
        }

        // Keep root types that contain must-intern descendants
        for root_type in root_types {
            if !final_types.contains(root_type)
                && self.contains_must_intern_in_tree(root_type, operations, must_intern_types)
            {
                final_types.insert(root_type.clone());
            }
        }

        // Only keep must-intern types that aren't subcomponents of kept root types
        for must_intern_type in must_intern_types {
            if !self.is_subcomponent_of_kept_root(must_intern_type, &final_types, operations) {
                final_types.insert(must_intern_type.clone());
            }
        }

        final_types
    }

    fn contains_must_intern_in_tree(
        &self,
        root: &Type,
        operations: &[GraphOperation],
        must_intern_types: &HashSet<Type>,
    ) -> bool {
        let mut stack = vec![root.clone()];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if must_intern_types.contains(&current) {
                return true;
            }

            for operation in operations {
                if let Some(parent_type) = &operation.parent {
                    if parent_type == &current && !visited.contains(&operation.node_type) {
                        stack.push(operation.node_type.clone());
                    }
                }
            }
        }

        false
    }

    fn is_subcomponent_of_kept_root(
        &self,
        target: &Type,
        kept_roots: &HashSet<Type>,
        operations: &[GraphOperation],
    ) -> bool {
        for root in kept_roots {
            if self.contains_must_intern_in_tree(root, operations, &HashSet::from([target.clone()]))
                && root != target
            {
                return true;
            }
        }
        false
    }

    // TODO: Reimplement reduce_unreachable_nodes with VecGraph APIs
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
    /// merging them with the Leaker's generics:
    ///
    /// ```ignore
    /// // In the Leaker
    /// for (ix, (ty, locals)) in referrer.iter_with_local_generics().enumerate() {
    ///     let args = encode_generics_params_to_ty(&generics.params);
    ///     let local_args = encode_generics_params_to_ty(&locals.params);
    ///     let (_, _, local_where) = locals.split_for_impl();
    ///     quote! {
    ///         impl<#generics_params, #locals_params> Repeater<(#args, #local_args), #ix>
    ///             for #marker_ty #local_where
    ///         {
    ///             type Type = #ty;
    ///         }
    ///     }
    /// }
    ///
    /// // In the Referrer
    /// let local_args = encode_generics_params_to_ty(&referrer.local_generics(ix).params);
    /// parse_quote!(<#marker_ty as Repeater<(#args, #local_args), #ix>>::Type)
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
    fn with_generics(&mut self, generics: &mut Generics) -> Self {
        let mut visitor = self.clone();
        for gp in generics.params.iter_mut() {
            if let GenericParam::Type(TypeParam { ident, .. }) = gp {
                visitor.0.map.remove(&parse_quote!(#ident));
            }
            visitor.visit_generic_param_mut(gp);
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
        for stmt in &i.stmts {
            match stmt {
                Stmt::Item(Item::Struct(ItemStruct { ident, .. }))
                | Stmt::Item(Item::Enum(ItemEnum { ident, .. }))
                | Stmt::Item(Item::Union(ItemUnion { ident, .. }))
                | Stmt::Item(Item::Trait(ItemTrait { ident, .. }))
                | Stmt::Item(Item::Type(ItemType { ident, .. })) => {
                    visitor.0.map.remove(&parse_quote!(#ident));
                }
                _ => (),
            }
        }
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

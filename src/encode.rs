//! Encoding generic parameters and arguments into types.

use crate::util::unwrap_type_group;
use proc_macro2::Span;
use std::collections::HashMap;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::*;
use template_quote::quote;

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
///
/// [`RepeaterSlots::encode_generics_args_to_ty()`]: crate::RepeaterSlots::encode_generics_args_to_ty
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
/// [`RepeaterSlots::encode_generics_args_to_ty()`]). `const_fn` encodes a (normalized) const value
/// of the given parameter.
///
/// [`RepeaterSlots::encode_generics_args_to_ty()`]: crate::RepeaterSlots::encode_generics_args_to_ty
pub(crate) fn encode_generics_args_with_params<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &'b Punctuated<GenericParam, Token![,]>,
    mut const_fn: impl FnMut(&Expr, &'b ConstParam) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    use syn::fold::Fold;
    let args = complete_generics_args(iter, params, |_, _, default, substitute, _| {
        Ok(substitute.fold_type(default.clone()))
    })?;
    let mut elems = Punctuated::<Type, Token![,]>::new();
    for (param, arg) in params.iter().zip(args) {
        elems.push(match (param, arg) {
            (_, GenericArgument::Lifetime(arg)) => parse_quote!(& #arg ()),
            (_, GenericArgument::Type(ty)) => ty,
            (GenericParam::Const(cp), GenericArgument::Const(value)) => const_fn(&value, cp)?,
            // `complete_generics_args()` returns an argument of the kind of each parameter.
            _ => unreachable!(),
        });
    }
    Ok(Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: with_trailing_comma(elems),
    }))
}

/// Matches `iter` with `params` and returns the complete argument list, one for each parameter
/// in order: arguments are matched by position (lifetime arguments with lifetime parameters, and
/// the others with type and const parameters), a bare identifier corresponding to a const
/// parameter is a const argument (normalized, see [`normalize_const_arg()`]), and an omitted
/// const argument takes the default, in which the preceding parameters are substituted.
///
/// An omitted type argument takes `default_type(index, param, default, substitute, args)`, where
/// `substitute` substitutes the preceding parameters by their arguments `args`.
pub(crate) fn complete_generics_args<'a>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &Punctuated<GenericParam, Token![,]>,
    mut default_type: impl FnMut(
        usize,
        &TypeParam,
        &Type,
        &mut SubstituteParams,
        &[GenericArgument],
    ) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
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
    let mut completed = Vec::<GenericArgument>::new();
    for (index, param) in params.iter().enumerate() {
        match param {
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => match lifetime_args.next() {
                Some(GenericArgument::Lifetime(arg)) => {
                    substitute.lifetimes.insert(lifetime.clone(), arg.clone());
                    completed.push(GenericArgument::Lifetime(arg.clone()));
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
                        Some(default) => {
                            default_type(index, tp, default, &mut substitute, &completed)?
                        }
                        None => return Err(EncodeError::MissingArgument(Box::new(param.clone()))),
                    },
                };
                substitute.types.insert(tp.ident.clone(), ty.clone());
                completed.push(GenericArgument::Type(ty));
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
                completed.push(GenericArgument::Const(normalize_const_arg(&value)));
            }
        }
    }
    if let Some(arg) = lifetime_args.next().or_else(|| other_args.next()) {
        return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone())));
    }
    Ok(completed)
}

/// Primitive types which can be the type of const parameters on stable Rust.
const CONST_PARAM_PRIMITIVES: &[&str] = &[
    "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128",
    "isize",
];

/// Returns `true` if `ty` is a primitive type which can be the type of const parameters (e.g.
/// `bool` or `::core::primitive::bool`), and thus does not depend on the type context.
pub(crate) fn is_const_param_ty(ty: &Type) -> bool {
    const_param_primitive(ty).is_some()
}

/// Returns the name of the primitive type if `ty` is a primitive type which can be the type of
/// const parameters (e.g. `"bool"` for `bool` or `::core::primitive::bool`).
pub(crate) fn const_param_primitive(ty: &Type) -> Option<&'static str> {
    match unwrap_type_group(ty) {
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
        _ => None,
    }
}

/// Normalizes a const argument: redundant braces are removed from a literal or a single
/// identifier (e.g. `{ N }` -> `N`), and any other expression is braced (e.g. `i32::MIN` ->
/// `{ i32::MIN }`), which is required where it could be parsed as a type.
pub(crate) fn normalize_const_arg(expr: &Expr) -> Expr {
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
/// arguments of an `impl`. Because array lengths are `usize`, this only works for `const N: usize`.
/// To encode generics of any kind, use [`RepeaterSlots`](crate::RepeaterSlots), or
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

/// Error returned by [`RepeaterSlots`](crate::RepeaterSlots) when generics cannot be encoded into a
/// type.
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

/// Adds a trailing comma to a single-element tuple, so that `(T,)` is not parsed as `(T)`.
pub(crate) fn with_trailing_comma(
    mut elems: Punctuated<Type, Token![,]>,
) -> Punctuated<Type, Token![,]> {
    if elems.len() == 1 && !elems.trailing_punct() {
        elems.push_punct(Default::default());
    }
    elems
}

/// Substitutes generic parameters (single-segment paths and lifetimes) with arguments.
#[derive(Default)]
pub(crate) struct SubstituteParams {
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

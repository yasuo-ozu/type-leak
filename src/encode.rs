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
/// default. The lifetime bounds of `params` are encoded after the parameters, with the arguments
/// substituted: `T: 'a` with the arguments `'static` and `u8` is encoded as `&'static u8`, and `'b:
/// 'a` as `&'a &'b ()` (see [`encode_generics_params_to_ty_with()`]). The argument of a type with
/// such a bound must not be a trait object without a lifetime bound (e.g. `dyn Tr`), because its
/// lifetime bound would default to the lifetime of the reference (`&'a dyn Tr` is `&'a (dyn Tr +
/// 'a)`), which is not the one in the other elements of the type: [`EncodeError`] is returned,
/// write the bound (e.g. `dyn Tr + 'static`).
///
/// **Without `params`**, the arguments are encoded as they are, and this never fails, but only
/// what can be told without the parameters: the lifetime bounds are not encoded, as the
/// arguments have no bound to tell which of them is bounded by which. So the result differs from
/// the encoding of the generics of a leaked type ([`Referrer::generics()`]) if they have a
/// lifetime bound, which the generics of a type implying one always have (e.g. `T: 'a` for
/// `Rel<'a, T>`): always give `params` for the generics of a leaked type.
///
/// The `impl` of the Repeater declares the generics without the encoded bounds (see
/// [`Referrer::impl_generics()`]).
///
/// `Self: 'a` (see [`Referrer::context_self_outlives()`]) is not encoded, which needs the type
/// `Self` is: use [`RepeaterSlots::encode_generics_args_to_ty_with_self()`], or append `&'a Self`.
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
///
/// // A lifetime bound is encoded with the arguments substituted.
/// let generics: Generics = parse_quote!(<'a, T: 'a>);
/// let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8>);
/// let ty = encode_generics_args_to_ty_with(&args.args, Some(&generics.params), |_, _| {
///     unreachable!()
/// })
/// .unwrap();
/// let expected: Type = parse_quote!((&'static (), u8, &'static u8));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
/// ```
///
/// [`RepeaterSlots::encode_generics_args_to_ty()`]: crate::RepeaterSlots::encode_generics_args_to_ty
/// [`RepeaterSlots::encode_generics_args_to_ty_with_self()`]: crate::RepeaterSlots::encode_generics_args_to_ty_with_self
/// [`Referrer::generics()`]: crate::Referrer::generics
/// [`Referrer::impl_generics()`]: crate::Referrer::impl_generics
/// [`Referrer::context_self_outlives()`]: crate::Referrer::context_self_outlives
pub fn encode_generics_args_to_ty_with<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: Option<&'b Punctuated<GenericParam, Token![,]>>,
    mut const_fn: impl FnMut(&Expr, Option<&'b ConstParam>) -> Type,
) -> std::result::Result<Type, EncodeError> {
    if let Some(params) = params {
        return encode_generics_args_with_params(iter, params, None, |value, cp| {
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
/// of the given parameter. `self_outlives` is `Self` and the lifetimes `'a` (parameters of
/// `params`) such that `Self: 'a`, which are encoded after the lifetime bounds (see
/// [`encode_self_outlives()`]).
///
/// [`RepeaterSlots::encode_generics_args_to_ty()`]: crate::RepeaterSlots::encode_generics_args_to_ty
pub(crate) fn encode_generics_args_with_params<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &'b Punctuated<GenericParam, Token![,]>,
    self_outlives: Option<(&Type, &[Lifetime])>,
    mut const_fn: impl FnMut(&Expr, &'b ConstParam) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    use syn::fold::Fold;
    let args = complete_generics_args(iter, params, |_, _, default, substitute, _| {
        Ok(substitute.fold_type(default.clone()))
    })?;
    let mut elems = Punctuated::<Type, Token![,]>::new();
    for (param, arg) in params.iter().zip(&args) {
        elems.push(match (param, arg) {
            (_, GenericArgument::Lifetime(arg)) => parse_quote!(& #arg ()),
            (_, GenericArgument::Type(ty)) => ty.clone(),
            (GenericParam::Const(cp), GenericArgument::Const(value)) => const_fn(value, cp)?,
            // `complete_generics_args()` returns an argument of the kind of each parameter.
            _ => unreachable!(),
        });
    }
    let params: Vec<&GenericParam> = params.iter().collect();
    elems.extend(encode_outlives_bounds(&params, &args)?);
    if let Some((self_ty, lifetimes)) = self_outlives {
        elems.extend(encode_self_outlives(&params, &args, self_ty, lifetimes)?);
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
///
/// The lifetime bounds of the parameters are encoded too (see
/// [`encode_generics_params_to_ty_with()`]), so the `impl` declares the generics without them:
/// see [`Referrer::impl_generics()`].
///
/// [`Referrer::impl_generics()`]: crate::Referrer::impl_generics
pub fn encode_generics_params_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericParam>) -> Type {
    encode_generics_params_to_ty_with(iter, |ConstParam { ident, .. }| parse_quote!([(); #ident]))
}

/// Encode [`GenericParam`]s to a type, using `const_fn` to encode const parameters.
///
/// A lifetime `'a` is encoded as `&'a ()` and a type `T` as `T`. The lifetime bounds of the
/// parameters are encoded after them, in the order of the parameters and of their bounds: a bound
/// `T: 'a` as `&'a T`, and `'b: 'a` as `&'a &'b ()`. The type of an `impl` header containing them
/// implies the bound (`&'a T` is well-formed only if `T: 'a`), so the bound is implied wherever
/// the same encoding is written, e.g. in a signature copied by the Referrer, which does not know
/// the bound otherwise: it is not implied by `(&'a (), T)`.
///
/// This is how the outlives bounds implied by a leaked type are carried (e.g. `T: 'a` of
/// `Rel<'a, T>` where `struct Rel<'a, T>(&'a T)`, see [`Referrer::generics()`]): without them the
/// implementation `impl<'a, T> Repeater<(&'a (), T), ..> for S { type Type = Rel<'a, T>; }` is
/// rejected with E0309, and so is the copy `<S as Repeater<(&'a (), T), ..>>::Type` of a
/// signature. Only the bounds by a lifetime which is a parameter of `iter` are encoded: a bound by
/// `'static` (or another lifetime) adds nothing a signature does not know, and cannot be written
/// by a type.
///
/// The `impl` must not declare the bounds it encodes (a declared bound is an obligation wherever
/// the implementation is selected, which may not hold: E0309 at the copy of a signature using a
/// type such as `struct Loose<'a, T>(&'a (), T)`, which does not imply `T: 'a`): declare
/// [`Referrer::impl_generics()`] (or [`without_encoded_bounds()`]) instead of the generics.
///
/// `Self: 'a` (see [`Referrer::context_self_outlives()`]) is not encoded, which needs the type
/// `Self` is: use [`RepeaterSlots::encode_generics_params_to_ty_with_self()`], or append `&'a Self`.
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
///
/// // The bounds `T: 'a` and `'b: 'a`.
/// let generics: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + Clone>);
/// let ty = encode_generics_params_to_ty(&generics.params);
/// assert_eq!(
///     quote!(#ty).to_string(),
///     quote!((&'a (), &'b (), T, &'a &'b (), &'a T)).to_string()
/// );
/// ```
///
/// [`Referrer::generics()`]: crate::Referrer::generics
/// [`Referrer::impl_generics()`]: crate::Referrer::impl_generics
/// [`Referrer::context_self_outlives()`]: crate::Referrer::context_self_outlives
/// [`RepeaterSlots::encode_generics_params_to_ty_with_self()`]: crate::RepeaterSlots::encode_generics_params_to_ty_with_self
pub fn encode_generics_params_to_ty_with<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    mut const_fn: impl FnMut(&'a ConstParam) -> Type,
) -> Type {
    let params: Vec<&'a GenericParam> = iter.into_iter().collect();
    let mut elems: Punctuated<Type, Token![,]> = params
        .iter()
        .map(|param| -> Type {
            match param {
                GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                    parse_quote!(& #lifetime ())
                }
                GenericParam::Const(cp) => const_fn(cp),
                GenericParam::Type(TypeParam { ident, .. }) => parse_quote!(#ident),
            }
        })
        .collect();
    // The arguments are the parameters themselves, which cannot fail.
    elems.extend(
        encode_outlives_bounds(&params, &params_as_args(&params))
            .expect("the parameters are not trait objects"),
    );
    Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: with_trailing_comma(elems),
    })
}

/// Returns the parameters as the arguments of a path with the same names (a const as an
/// identifier).
pub(crate) fn params_as_args(params: &[&GenericParam]) -> Vec<GenericArgument> {
    params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                GenericArgument::Lifetime(lifetime.clone())
            }
            GenericParam::Type(TypeParam { ident, .. }) => {
                GenericArgument::Type(parse_quote!(#ident))
            }
            GenericParam::Const(ConstParam { ident, .. }) => {
                GenericArgument::Const(parse_quote!(#ident))
            }
        })
        .collect()
}

/// Returns the lifetime bounds written on `param` (`T: 'a` of a type parameter, `'b: 'a` of a
/// lifetime parameter), in order. A where predicate is not one.
fn lifetime_bounds(param: &GenericParam) -> Vec<&Lifetime> {
    match param {
        GenericParam::Lifetime(param) => param.bounds.iter().collect(),
        GenericParam::Type(param) => param
            .bounds
            .iter()
            .filter_map(|bound| match bound {
                TypeParamBound::Lifetime(lifetime) => Some(lifetime),
                _ => None,
            })
            .collect(),
        GenericParam::Const(_) => Vec::new(),
    }
}

/// Returns the position in `params` of the lifetime parameter `bound`, if a lifetime bound `bound`
/// of one of `params` is encoded (see [`encode_outlives_bounds()`]): it is the lifetime of a
/// lifetime parameter of `params`. A bound by another lifetime (`'static`, or a lifetime which
/// the type context provides) is not.
///
/// This is the only rule deciding which bounds are encoded: the encoders use it to find the
/// argument of the lifetime, and [`without_encoded_bounds()`] to leave the others.
fn encoded_bound_position(params: &[&GenericParam], bound: &Lifetime) -> Option<usize> {
    params.iter().position(
        |param| matches!(param, GenericParam::Lifetime(param) if &param.lifetime == bound),
    )
}

/// Returns `generics` without the lifetime bounds that [`encode_generics_params_to_ty()`] (and the
/// other encoders, see [`encode_generics_params_to_ty_with()`]) encode in the trait arguments: a
/// bound `'a` on a type parameter (`T: 'a`) or on a lifetime parameter (`'b: 'a`), where `'a` is a
/// lifetime parameter of `generics`. Everything else is kept: the other bounds (`T: 'static`, a
/// bound by a lifetime which is not a parameter, trait bounds), the const parameters, and the where
/// clause (which the encoders do not look into).
///
/// This is the form to declare as the generics of a Repeater's `impl`, whose trait arguments
/// encode the bounds (see [`Referrer::impl_generics()`]): the header implies them, and a declared
/// bound would be an obligation wherever the implementation is selected, which may not hold.
/// Use this function for generics which are not those of a context, such as those returned by
/// [`Referrer::iter_with_self()`].
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::quote;
/// let generics: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + Clone, U: 'static, V: 'c>);
/// let declared = without_encoded_bounds(&generics);
/// let expected: Generics = parse_quote!(<'a, 'b, T: Clone, U: 'static, V: 'c>);
/// assert_eq!(quote!(#declared).to_string(), quote!(#expected).to_string());
/// ```
///
/// [`Referrer::impl_generics()`]: crate::Referrer::impl_generics
/// [`Referrer::iter_with_self()`]: crate::Referrer::iter_with_self
pub fn without_encoded_bounds(generics: &Generics) -> Generics {
    let params: Vec<&GenericParam> = generics.params.iter().collect();
    let mut declared = generics.clone();
    for param in declared.params.iter_mut() {
        match param {
            GenericParam::Lifetime(param) => {
                param.bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| encoded_bound_position(&params, bound).is_none())
                    .cloned()
                    .collect();
                if param.bounds.is_empty() {
                    param.colon_token = None;
                }
            }
            GenericParam::Type(param) => {
                param.bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| {
                        !matches!(bound, TypeParamBound::Lifetime(lifetime)
                            if encoded_bound_position(&params, lifetime).is_some())
                    })
                    .cloned()
                    .collect();
                if param.bounds.is_empty() {
                    param.colon_token = None;
                }
            }
            GenericParam::Const(_) => (),
        }
    }
    declared
}

/// Returns the elements which encode the lifetime bounds of `params` (in the order of the
/// parameters and of their bounds), after the elements encoding the parameters, with `args` as the
/// arguments of the parameters (one for each, see [`params_as_args()`]): `&'a T` for the bound `T:
/// 'a` of a type parameter, and `&'a &'b ()` for `'b: 'a` of a lifetime parameter, where `T`, `'b`
/// and `'a` are the arguments. A bound is encoded if the lifetime is a lifetime parameter of
/// `params` (see [`without_encoded_bounds()`], which leaves the others).
///
/// [`EncodeError::UnboundedTraitObject`] is returned if the argument of a type with a bound is a
/// trait object without a lifetime bound.
pub(crate) fn encode_outlives_bounds(
    params: &[&GenericParam],
    args: &[GenericArgument],
) -> std::result::Result<Vec<Type>, EncodeError> {
    let mut elems = Vec::new();
    for (param, arg) in params.iter().zip(args) {
        for bound in lifetime_bounds(param) {
            // The argument of the lifetime parameter `bound`, if it is one of `params`.
            let outliving =
                encoded_bound_position(params, bound).and_then(|pos| match args.get(pos) {
                    Some(GenericArgument::Lifetime(outliving)) => Some(outliving),
                    _ => None,
                });
            let outliving = match outliving {
                Some(outliving) => outliving,
                None => continue,
            };
            let referent: Type = match arg {
                GenericArgument::Lifetime(lifetime) => parse_quote!(& #lifetime ()),
                GenericArgument::Type(ty) => referent_of_reference(ty)?,
                _ => continue,
            };
            elems.push(Type::Reference(TypeReference {
                and_token: Default::default(),
                lifetime: Some(outliving.clone()),
                mutability: None,
                elem: Box::new(referent),
            }));
        }
    }
    Ok(elems)
}

/// Returns the elements which encode `Self: 'a` for each of the `lifetimes`, after those encoding
/// the lifetime bounds of `params` (see [`encode_outlives_bounds()`]), with `args` as the arguments
/// of the parameters (one for each, see [`params_as_args()`]): `&'a Self` where `Self` is
/// `self_ty` and `'a` is the argument of the lifetime parameter of `params` with the same name.
/// Like a lifetime bound, it implies the bound wherever the encoding is written, which an
/// implementation for `Self` or for the type `__S` cannot declare (a declared `__S: 'a` is an
/// obligation at every use, and fails in the signature of a free function or the field of a
/// struct, where nothing proves it, whereas `&'a __S` implies it). A lifetime which is not a
/// parameter of `params` is skipped, like a bound by one.
///
/// [`EncodeError::UnboundedTraitObject`] is returned if `self_ty` is a trait object without a
/// lifetime bound and there is a lifetime.
pub(crate) fn encode_self_outlives(
    params: &[&GenericParam],
    args: &[GenericArgument],
    self_ty: &Type,
    lifetimes: &[Lifetime],
) -> std::result::Result<Vec<Type>, EncodeError> {
    let mut elems = Vec::new();
    for lifetime in lifetimes {
        let outliving =
            encoded_bound_position(params, lifetime).and_then(|pos| match args.get(pos) {
                Some(GenericArgument::Lifetime(outliving)) => Some(outliving),
                _ => None,
            });
        if let Some(outliving) = outliving {
            elems.push(Type::Reference(TypeReference {
                and_token: Default::default(),
                lifetime: Some(outliving.clone()),
                mutability: None,
                elem: Box::new(referent_of_reference(self_ty)?),
            }));
        }
    }
    Ok(elems)
}

/// Returns `ty` as the referent of a reference: a trait object is parenthesized (`&'a (dyn Tr +
/// 'b)`), and must have a lifetime bound, because it defaults to the lifetime of the reference
/// otherwise.
fn referent_of_reference(ty: &Type) -> std::result::Result<Type, EncodeError> {
    match unwrap_type_group(ty) {
        Type::TraitObject(object) => {
            if !object
                .bounds
                .iter()
                .any(|bound| matches!(bound, TypeParamBound::Lifetime(_)))
            {
                return Err(EncodeError::UnboundedTraitObject(Box::new(ty.clone())));
            }
            Ok(Type::Paren(TypeParen {
                paren_token: Default::default(),
                elem: Box::new(Type::TraitObject(object.clone())),
            }))
        }
        _ => Ok(ty.clone()),
    }
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
    /// The argument of a type parameter with a lifetime bound (e.g. `T: 'a`, which is encoded as
    /// `&'a T`) is a trait object without a lifetime bound, e.g. `dyn Tr`. The bound defaults to
    /// `'a` there, but to another lifetime (often `'static`) in the other elements of the
    /// encoding, so the encoding would not be that of the generics. Write the bound
    /// (e.g. `dyn Tr + 'static`).
    UnboundedTraitObject(Box<Type>),
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
            EncodeError::UnboundedTraitObject(ty) => write!(
                f,
                "the trait object '{}' has no lifetime bound, which defaults to different lifetimes \
                 in the encoding of a type with a lifetime bound; write it (e.g. `+ 'static`)",
                quote!(#ty)
            ),
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
            EncodeError::UnboundedTraitObject(ty) => ty.span(),
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

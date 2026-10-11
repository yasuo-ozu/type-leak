//! Encoding generic parameters and arguments into types.

use crate::util::{parenthesize_pointee, unwrap_type_group};
use proc_macro2::Span;
use std::collections::HashMap;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::*;
use template_quote::quote;

/// Encodes the arguments that a copy gives to the generics of an entry, into the first argument of
/// the Repeater. It is the counterpart of [`encode_generics_params_to_ty()`] for `params`, with
/// `const N: usize` as `[(); N]`.
///
/// It is [`encode_generics_args_to_ty_with()`] without the closure, for generics that have no const
/// parameter or only `usize` ones (which [`encode_generics_params_to_ty()`] encodes alike). For other
/// const types use [`RepeaterSlots`](crate::RepeaterSlots). The arguments are completed and matched as
/// in [`encode_generics_args_to_ty_with()`], and the errors are its errors.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let generics: Generics = parse_quote!(<'a, T: 'a, const N: usize>);
/// let args: Vec<GenericArgument> = vec![
///     parse_quote!('static),
///     parse_quote!(u8),
///     parse_quote!(3),
/// ];
/// let ty = encode_generics_args_to_ty(&args, &generics.params).unwrap();
/// // `encode_generics_params_to_ty()` gives `(&'a (), T, [(); N], &'a T)` for the parameters.
/// assert_eq!(ty, parse_quote!((&'static (), u8, [(); 3], &'static u8)));
/// ```
pub fn encode_generics_args_to_ty<'a>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &Punctuated<GenericParam, Token![,]>,
) -> std::result::Result<Type, EncodeError> {
    encode_generics_args_with_params(iter, params, None, |value, _| Ok(array_length(value)))
}

/// `[(); value]`, as `encode_generics_params_to_ty()` encodes a const parameter.
pub(crate) fn array_length(value: &Expr) -> Type {
    let value = normalize_const_arg(value);
    parse_quote!([(); #value])
}

/// Like [`array_length()`], but only for a parameter of type `usize`, the type of an array length:
/// the encoding of the helpers that have no closure for the consts.
pub(crate) fn usize_array_length(
    value: &Expr,
    param: &ConstParam,
) -> std::result::Result<Type, EncodeError> {
    match const_param_primitive(&param.ty) {
        Some("usize") => Ok(array_length(value)),
        _ => Err(EncodeError::UnsupportedConstType(Box::new(
            param.ty.clone(),
        ))),
    }
}

/// Encodes the arguments that a copy gives to the generics of an entry, with `const_fn` encoding
/// the const arguments. It gives the type that [`encode_generics_params_to_ty_with()`] gives for
/// `params`.
///
/// `params` are the generics the Leaker encoded: `Referrer::generics()` of an entry with the
/// arguments of `Referrer::generic_args()`, or `Referrer::context_generics()` of the top-level
/// context. `const_fn` encodes each const argument as it did for the parameters.
/// Arguments match by position, lifetimes first. A bare identifier for a const parameter is a
/// const argument, and bindings such as `Item = u8` are skipped. Omitted arguments take the
/// defaults of `params`, so complete the arguments of a context with `Referrer::complete_args()`.
/// Bounds are encoded with the arguments substituted: `T: 'a` with `'static` and `u8` is
/// `&'static u8`. `Self: 'a` is not: use [`encode_generics_args_to_ty_with_self()`].
/// The argument of a parameter with a `?Sized` bound is wrapped, as the parameter is:
/// `T: ?Sized` with `str` is `*const str` (see [`encode_generics_params_to_ty_with()`]).
///
/// Fails with [`EncodeError`] for too many or missing arguments, and with
/// [`EncodeError::UnboundedTraitObject`] for a trait object argument without a lifetime bound.
pub fn encode_generics_args_to_ty_with<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &'b Punctuated<GenericParam, Token![,]>,
    mut const_fn: impl FnMut(&Expr, &'b ConstParam) -> Type,
) -> std::result::Result<Type, EncodeError> {
    encode_generics_args_with_params(iter, params, None, |value, cp| Ok(const_fn(value, cp)))
}

/// Like [`encode_generics_args_to_ty_with()`], but also encodes `Self: 'a`.
///
/// Each lifetime `'a` of `self_outlives` adds `&'b Self` after the bounds, where `'b` is the
/// argument of `'a`. Pass `Referrer::context_self_outlives()`. `self_ty` is `Self` in the impl of
/// the Leaker's trait. A lifetime that is not a parameter of `params` is skipped.
/// It yields the type of [`encode_generics_params_to_ty_with_self()`]. The arguments are those of
/// `RepeaterSlots::encode_generics_args_to_ty_with_self()`, then `const_fn`. Fails like
/// [`encode_generics_args_to_ty_with()`], and with [`EncodeError::UnboundedTraitObject`] if
/// `self_ty` is a trait object without a lifetime bound.
pub fn encode_generics_args_to_ty_with_self<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &'b Punctuated<GenericParam, Token![,]>,
    self_ty: &Type,
    self_outlives: &[Lifetime],
    mut const_fn: impl FnMut(&Expr, &'b ConstParam) -> Type,
) -> std::result::Result<Type, EncodeError> {
    encode_generics_args_with_params(iter, params, Some((self_ty, self_outlives)), |value, cp| {
        Ok(const_fn(value, cp))
    })
}

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
            (GenericParam::Type(tp), GenericArgument::Type(ty)) if is_maybe_unsized(tp) => {
                unsized_element(ty)
            }
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

/// Completes `iter` to one argument per parameter. `default_type` makes an omitted type argument.
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

/// A primitive that can type a const parameter, so it is context-free.
pub(crate) fn is_const_param_ty(ty: &Type) -> bool {
    const_param_primitive(ty).is_some()
}

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

/// Braces everything but a literal or an identifier, or it could parse as a type.
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

/// Encodes the generics of an entry into the type that is the first argument of its Repeater impl,
/// with `const N: usize` as `[(); N]`.
///
/// The result is a tuple type. `[(); N]` constrains `N` in the trait arguments of an impl. Array lengths are `usize`, so other
/// const types need `RepeaterSlots`, or [`encode_generics_params_to_ty_with()`] with your own
/// types. That function describes the encoding. `Self: 'a` is encoded by
/// [`encode_generics_params_to_ty_with_self()`]. The Referrer encodes the arguments with
/// [`encode_generics_args_to_ty_with()`], which needs the same parameters.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let generics: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + Clone, const N: usize>);
/// // The bounds follow the parameters.
/// let ty = encode_generics_params_to_ty(&generics.params);
/// assert_eq!(ty, parse_quote!((&'a (), &'b (), T, [(); N], &'a &'b (), &'a T)));
///
/// // A type parameter that may be unsized is a pointer to it, which is always `Sized`.
/// let generics: Generics = parse_quote!(<T: ?Sized, U>);
/// let ty = encode_generics_params_to_ty(&generics.params);
/// assert_eq!(ty, parse_quote!((*const T, U)));
/// ```
pub fn encode_generics_params_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericParam>) -> Type {
    encode_generics_params_to_ty_with(iter, |ConstParam { ident, .. }| parse_quote!([(); #ident]))
}

/// Encodes generic parameters into a tuple type, using `const_fn` for const parameters.
///
/// A lifetime `'a` becomes `&'a ()` and a type `T` becomes `T`. A type with a `?Sized` bound
/// becomes `*const T` instead: all the elements of a tuple but the last must be `Sized`, and a
/// raw pointer is, whatever it points to. The impl must still declare the `?Sized` bound, which
/// [`without_encoded_bounds()`] keeps. The bound must be on the parameter, because a where clause
/// is not seen. The generics of a context have it there: `Referrer::generics()` moves a predicate
/// `T: ?Sized` to the parameter. The lifetime bounds follow, in the order of the parameters and
/// their bounds: `T: 'a` becomes `&'a T`, and `'b: 'a` becomes `&'a &'b ()`. Only a bound by a
/// lifetime parameter of `iter` is encoded.
/// A reference implies its bound wherever it is written. So the impl must not declare the bounds
/// it encodes. Declare `Referrer::impl_generics()` or [`without_encoded_bounds()`] instead (see
/// [Outlives bounds](crate::rules#outlives-bounds)). `Self: 'a` is not encoded: use
/// [`encode_generics_params_to_ty_with_self()`].
pub fn encode_generics_params_to_ty_with<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    mut const_fn: impl FnMut(&'a ConstParam) -> Type,
) -> Type {
    encode_generics_params_with(iter, None, |cp| Ok(const_fn(cp)))
        .expect("the parameters are not trait objects")
}

/// Like [`encode_generics_params_to_ty_with()`], but also encodes `Self: 'a`.
///
/// Each lifetime `'a` of `self_outlives` adds `&'a Self` after the bounds, with `Self` being
/// `self_ty`. Pass `Referrer::context_self_outlives()` of the context of `iter`. A lifetime that
/// is not a parameter of `iter` is skipped.
/// `self_ty` is the type of the impl: `__S` for any implementor (see `Referrer::iter_with_self()`),
/// or `Self` for an impl for a type. The impl must not declare `__S: 'a`, because the encoding
/// implies it.
/// The arguments are those of `RepeaterSlots::encode_generics_params_to_ty_with_self()`, then
/// `const_fn`. The Referrer encodes the arguments with [`encode_generics_args_to_ty_with_self()`].
/// Fails with [`EncodeError::UnboundedTraitObject`] if `self_ty` is a trait object without a
/// lifetime bound.
pub fn encode_generics_params_to_ty_with_self<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    self_ty: &Type,
    self_outlives: &[Lifetime],
    mut const_fn: impl FnMut(&'a ConstParam) -> Type,
) -> std::result::Result<Type, EncodeError> {
    encode_generics_params_with(iter, Some((self_ty, self_outlives)), |cp| Ok(const_fn(cp)))
}

/// The Leaker's encoding, which `encode_generics_args_with_params()` must reproduce.
pub(crate) fn encode_generics_params_with<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    self_outlives: Option<(&Type, &[Lifetime])>,
    mut const_fn: impl FnMut(&'a ConstParam) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    let params: Vec<&'a GenericParam> = iter.into_iter().collect();
    let mut elems = params
        .iter()
        .map(|param| -> std::result::Result<Type, EncodeError> {
            Ok(match param {
                GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                    parse_quote!(& #lifetime ())
                }
                GenericParam::Const(cp) => const_fn(cp)?,
                GenericParam::Type(tp) if is_maybe_unsized(tp) => {
                    let ident = &tp.ident;
                    unsized_element(&parse_quote!(#ident))
                }
                GenericParam::Type(TypeParam { ident, .. }) => parse_quote!(#ident),
            })
        })
        .collect::<std::result::Result<Punctuated<Type, Token![,]>, _>>()?;
    let args = params_as_args(&params);
    elems.extend(encode_outlives_bounds(&params, &args)?);
    if let Some((self_ty, lifetimes)) = self_outlives {
        elems.extend(encode_self_outlives(&params, &args, self_ty, lifetimes)?);
    }
    Ok(Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: with_trailing_comma(elems),
    }))
}

pub(crate) fn params_as_args(params: &[&GenericParam]) -> Vec<GenericArgument> {
    params.iter().map(|param| param_as_arg(param)).collect()
}

/// The argument that passes `param` on by its own name.
pub(crate) fn param_as_arg(param: &GenericParam) -> GenericArgument {
    match param {
        GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
            GenericArgument::Lifetime(lifetime.clone())
        }
        GenericParam::Type(TypeParam { ident, .. }) => GenericArgument::Type(parse_quote!(#ident)),
        GenericParam::Const(ConstParam { ident, .. }) => {
            GenericArgument::Const(parse_quote!(#ident))
        }
    }
}

/// The lifetime bounds written on `param`, in order. A where predicate is not one.
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

/// The only rule that decides which bounds are encoded. [`without_encoded_bounds()`] keeps the
/// others.
fn encoded_bound_position(params: &[&GenericParam], bound: &Lifetime) -> Option<usize> {
    params.iter().position(
        |param| matches!(param, GenericParam::Lifetime(param) if &param.lifetime == bound),
    )
}

/// Returns the generics to declare in a Repeater impl: `generics` without the lifetime bounds that
/// the encoders encode.
///
/// Removed: a bound by a lifetime parameter of `generics`, on a type parameter (`T: 'a`) or a
/// lifetime parameter (`'b: 'a`). Kept: other bounds (`T: 'static`, a bound by a lifetime that is
/// not a parameter, trait bounds), const parameters and the where clause.
/// Declare the result as the generics of a Repeater impl. `Referrer::impl_generics()` does this
/// for a context. Use this function for other generics, such as those of
/// `Referrer::iter_with_self()`. `where Self: 'a` is not removed. A context has none, because the
/// Leaker moves it into `Referrer::context_self_outlives()` for the `*_with_self()` encoders.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let generics: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + Clone, U: 'static, V: 'c>);
/// let declared = without_encoded_bounds(&generics);
/// assert_eq!(declared, parse_quote!(<'a, 'b, T: Clone, U: 'static, V: 'c>));
/// ```
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

/// The elements encoding the lifetime bounds of `params`, after those of the parameters.
pub(crate) fn encode_outlives_bounds(
    params: &[&GenericParam],
    args: &[GenericArgument],
) -> std::result::Result<Vec<Type>, EncodeError> {
    let mut elems = Vec::new();
    for (param, arg) in params.iter().zip(args) {
        for bound in lifetime_bounds(param) {
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

/// `&'a __S` implies `__S: 'a` wherever it is written. A declared `__S: 'a` would fail where
/// nothing proves it, such as in a free function signature.
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

/// Whether `param` may be unsized: it has a `?Sized` bound, as the generics of a context do (the
/// Leaker moves a where predicate `T: ?Sized` to the parameter).
pub(crate) fn is_maybe_unsized(param: &TypeParam) -> bool {
    param.bounds.iter().any(|bound| {
        matches!(
            bound,
            TypeParamBound::Trait(TraitBound {
                modifier: TraitBoundModifier::Maybe(_),
                ..
            })
        )
    })
}

/// The element of the trait arguments for the type `ty` of a `?Sized` parameter: `*const ty`.
///
/// A tuple is well-formed only if the elements but the last are `Sized`, and a raw pointer is
/// `Sized` whatever it points to. A trait object with several bounds is parenthesized, since
/// `*const dyn A + B` does not parse, and one with a single bound is not, as the parentheses are
/// needless there (rustc's `unused_parens` reports `*const (dyn A)`): the rule is that of
/// [`parenthesize_pointee()`]. Parentheses and groups around a trait object are dropped, so that
/// only the one that is needed remains. The Leaker (the parameter, `T`) and the Referrer (its
/// argument) both come here, so the parentheses are the same wherever it is a trait object.
fn unsized_element(ty: &Type) -> Type {
    let pointee = match unwrap_type_group(ty) {
        object @ Type::TraitObject(_) => object.clone(),
        _ => ty.clone(),
    };
    parenthesize_pointee(Type::Ptr(TypePtr {
        star_token: Default::default(),
        const_token: Some(Default::default()),
        mutability: None,
        elem: Box::new(pointee),
    }))
}

/// Returns `ty` as the referent of a reference. A trait object is parenthesized, and needs a
/// lifetime bound, or it would default to the lifetime of the reference.
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

/// The error of the encoders: the free functions and `RepeaterSlots`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum EncodeError {
    /// The type of a const parameter is not a primitive type (e.g. an alias).
    UnsupportedConstType(Box<Type>),
    /// There are more arguments than parameters.
    UnexpectedArgument(Box<GenericArgument>),
    /// A parameter that has neither an argument nor a default.
    MissingArgument(Box<GenericParam>),
    /// The argument of a type parameter with a lifetime bound is a trait object without one
    /// (`dyn Tr`).
    ///
    /// Its bound would default to `'a` in `&'a T` and to another lifetime elsewhere. Write it, e.g.
    /// `dyn Tr + 'static`.
    UnboundedTraitObject(Box<Type>),
    /// The type of an entry names `Self`, but the Repeater impl was asked for a marker type
    /// ([`ImplSelf::Marker`](crate::ImplSelf::Marker)).
    ///
    /// Such an impl is for what `Self` stands for: any implementor of the trait, or the type. Use
    /// [`ImplSelf::Param`](crate::ImplSelf::Param). An entry for which
    /// [`Referrer::is_plain()`](crate::Referrer::is_plain) holds never gives this error.
    NeedsSelf(Box<Type>),
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
            EncodeError::NeedsSelf(ty) => write!(
                f,
                "the type '{}' names `Self`, so its Repeater impl cannot be for a marker type; use \
                 `ImplSelf::Param`",
                quote!(#ty)
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

impl EncodeError {
    /// Returns the span of the offending parameter, argument or type.
    pub fn span(&self) -> Span {
        match self {
            EncodeError::UnsupportedConstType(ty) => ty.span(),
            EncodeError::UnexpectedArgument(arg) => arg.span(),
            EncodeError::MissingArgument(param) => param.span(),
            EncodeError::UnboundedTraitObject(ty) => ty.span(),
            EncodeError::NeedsSelf(ty) => ty.span(),
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

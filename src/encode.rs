//! Encoding generic parameters and arguments into types.

use crate::elided::{fold_bare_fn, fold_bound, FreshLifetimes, Site};
use crate::util::{
    parenthesize_bounded, parenthesize_pointee, projection, projection_head, unwrap_type_group,
    ReplaceSelf,
};
use proc_macro2::{Span, TokenStream, TokenTree};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter::FromIterator;
use syn::fold::Fold;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;

/// Encodes the const argument `value` of a `usize` parameter as the array type `[(); value]`.
///
/// It is the `const_fn` of [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls)
/// and [`repeater_projection()`](crate::repeater_projection) if every const parameter is `usize`.
/// Any other type is an [`EncodeError::UnsupportedConstType`], which fails the call of
/// `repeater_impls()`: use [`slot_const()`](crate::slot_const).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let n = array_const(&parse_quote!(3), &parse_quote!(const N: usize));
/// assert_eq!(n.unwrap(), parse_quote!([(); 3]));
/// // `bool` has no array length.
/// let b = array_const(&parse_quote!(true), &parse_quote!(const B: bool));
/// assert!(matches!(b, Err(EncodeError::UnsupportedConstType(_))));
/// ```
pub fn array_const(value: &Expr, param: &ConstParam) -> std::result::Result<Type, EncodeError> {
    match const_param_primitive(&param.ty) {
        Some("usize") => {
            let value = normalize_const_arg(value);
            Ok(parse_quote!([(); #value]))
        }
        _ => Err(EncodeError::UnsupportedConstType(Box::new(
            param.ty.clone(),
        ))),
    }
}

/// Returns one argument for each of `params`, in their order: the arguments `given`, and for a
/// parameter with none its default.
///
/// It completes the arguments of an impl of a trait. `params` are the generics of the trait after
/// [`Referrer::visitor()`](crate::Referrer::visitor) has written its defaults: a leaked type in a
/// default is already replaced, by a path through the Repeater that names the parameters before
/// the default. The default of a parameter that has no argument is taken with each of the
/// parameters before it replaced by its argument, as [`substitute_args()`] does. That copy of the
/// trait is only a step (rustc does not accept `Self` in a default): the completed arguments go
/// into the impl.
///
/// An associated type `T::Item` of an earlier parameter with the argument `arg` is `<arg>::Item` if
/// `arg` is one identifier (rustc resolves `<V>::Item` for a type parameter `V`), else `<arg as
/// Trait>::Item` with the one trait bound that `T` declares (`T: Iterator<Item = u8>` gives
/// `Iterator`: the bindings go, and the earlier parameters are replaced in its arguments). A where
/// clause is not known, and a trait that does not declare the type itself (a supertrait does) is
/// refused by rustc.
///
/// `given` match `params` by position and kind, lifetimes first: a binding such as `Item = u8` is
/// skipped, and a bare identifier is a const argument (syn parses it as a type). Fails with
/// [`EncodeError::UnexpectedArgument`] for an argument of the wrong kind or too many, with
/// [`EncodeError::MissingArgument`] for a parameter that has neither an argument nor a default,
/// and with [`EncodeError::AmbiguousAssociatedType`] for an associated type of an argument that is
/// not an identifier, if the parameter has no trait bound or several.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let generics: Generics = parse_quote!(<'a, T, U = Vec<T>>);
///
/// // `U` takes its default, with `T` as `u8`.
/// let given: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
/// let expected: Vec<GenericArgument> =
///     vec![parse_quote!('static), parse_quote!(u8), parse_quote!(Vec<u8>)];
/// assert_eq!(complete_params(&generics.params, &given).unwrap(), expected);
///
/// // `T` has no default.
/// let given: Vec<GenericArgument> = vec![parse_quote!('static)];
/// let err = complete_params(&generics.params, &given).unwrap_err();
/// assert!(matches!(err, EncodeError::MissingArgument(_)));
///
/// // The default names an associated type of `T`: the trait of its bound is written.
/// let generics: Generics = parse_quote!(<T: Iterator<Item = u8>, U = Option<T::Item>>);
/// let given: Vec<GenericArgument> = vec![parse_quote!(std::vec::IntoIter<u8>)];
/// let completed = complete_params(&generics.params, &given).unwrap();
/// let expected: GenericArgument = parse_quote!(Option<<std::vec::IntoIter<u8> as Iterator>::Item>);
/// assert_eq!(completed[1], expected);
///
/// // Without a bound it cannot be written.
/// let generics: Generics = parse_quote!(<T, U = Option<T::Item>>);
/// let err = complete_params(&generics.params, &given).unwrap_err();
/// assert!(matches!(err, EncodeError::AmbiguousAssociatedType(_)));
/// ```
pub fn complete_params<'p, 'a>(
    params: impl IntoIterator<Item = &'p GenericParam>,
    given: impl IntoIterator<Item = &'a GenericArgument>,
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
    let params: Vec<&GenericParam> = params.into_iter().collect();
    let mut substitute = ParamFold::default();
    for param in &params {
        substitute.names.reserve(param);
    }
    let mut lifetime_args = Vec::new();
    let mut other_args = Vec::new();
    for arg in given {
        substitute.names.reserve(arg);
        match arg {
            GenericArgument::Lifetime(_) => lifetime_args.push(arg),
            GenericArgument::Type(_) | GenericArgument::Const(_) => other_args.push(arg),
            _ => (),
        }
    }
    let mut lifetime_args = lifetime_args.into_iter();
    let mut other_args = other_args.into_iter();
    let mut completed = Vec::<GenericArgument>::new();
    for param in params {
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
                            let ty = substitute.fold_type(default.clone());
                            if let Some(error) = substitute.error.take() {
                                return Err(error);
                            }
                            ty
                        }
                        None => return Err(EncodeError::MissingArgument(Box::new(param.clone()))),
                    },
                };
                substitute.types.insert(tp.ident.clone(), ty.clone());
                if let Some(bound) = single_bound(tp) {
                    substitute.bounds.insert(tp.ident.clone(), bound);
                }
                completed.push(GenericArgument::Type(ty));
            }
            GenericParam::Const(cp) => {
                let value: Expr = match other_args.next() {
                    Some(arg) => match const_argument(arg) {
                        Some(value) => value,
                        None => return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone()))),
                    },
                    None => match &cp.default {
                        Some(default) => {
                            let value = substitute.fold_expr(default.clone());
                            if let Some(error) = substitute.error.take() {
                                return Err(error);
                            }
                            value
                        }
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

/// The value that `arg` gives to a const parameter: a const argument, or a bare identifier, which
/// is parsed as a type.
fn const_argument(arg: &GenericArgument) -> Option<Expr> {
    match arg {
        GenericArgument::Const(expr) => Some(expr.clone()),
        GenericArgument::Type(Type::Path(TypePath { qself: None, path }))
            if path.get_ident().is_some() =>
        {
            Some(parse_quote!(#path))
        }
        _ => None,
    }
}

/// The name that `ty` is, if it is one identifier (a type parameter, as far as the syntax tells).
fn type_name(ty: &Type) -> Option<&Ident> {
    match unwrap_type_group(ty) {
        Type::Path(TypePath { qself: None, path }) => path.get_ident(),
        _ => None,
    }
}

/// `<value>`, the start of a path `<value>::A`, at `span` (that of the `T` of `T::A`, so that an
/// error points at the path).
fn qself_of(value: Type, span: Span) -> QSelf {
    QSelf {
        lt_token: Token![<](span),
        ty: Box::new(value),
        position: 0,
        as_token: None,
        gt_token: Token![>](span),
    }
}

/// `<value as bound>::A..` for the trait path `bound` and the segments `rest` (`A..`, with the
/// leading `::` that [`parameter_projection()`] writes).
fn through_bound(value: Type, bound: Path, mut rest: Path, span: Span) -> TypePath {
    rest.leading_colon = None;
    let position = bound.segments.len();
    let mut segments = bound.segments;
    segments.extend(rest.segments);
    TypePath {
        qself: Some(QSelf {
            position,
            as_token: Some(Token![as](span)),
            ..qself_of(value, span)
        }),
        path: Path {
            leading_colon: bound.leading_colon,
            segments,
        },
    }
}

/// The name `T` and the rest of the path if `ty` is an associated type of it, `T::A..` or
/// `<T>::A..` (`T` one identifier, with no arguments). The rest has a leading `::`, as
/// `projection()` writes it.
fn parameter_projection(ty: &TypePath) -> Option<(&Ident, Path)> {
    let rest = |segments: Punctuated<PathSegment, Token![::]>| Path {
        leading_colon: Some(Default::default()),
        segments,
    };
    match &ty.qself {
        None => {
            let name = projection_head(&ty.path)?;
            let segments = ty.path.segments.iter().skip(1).cloned().collect();
            Some((name, rest(segments)))
        }
        Some(QSelf {
            ty: head,
            position: 0,
            as_token: None,
            ..
        }) => Some((type_name(head)?, rest(ty.path.segments.clone()))),
        _ => None,
    }
}

/// The one trait bound of the parameter, as a trait path without associated type bindings
/// (`Iterator` for `T: Iterator<Item = u8>`): the trait that `<value as Trait>::A` names. `None` if
/// the parameter has none, or has several, or the bound has a `for<'a>` binder. A lifetime bound
/// and `?Sized` are not counted; a where clause is not known.
fn single_bound(param: &TypeParam) -> Option<Path> {
    let mut found = None;
    for bound in &param.bounds {
        match bound {
            TypeParamBound::Lifetime(_) => (),
            TypeParamBound::Trait(TraitBound {
                modifier: TraitBoundModifier::Maybe(_),
                ..
            }) => (),
            TypeParamBound::Trait(bound) if found.is_none() && bound.lifetimes.is_none() => {
                found = Some(bound.path.clone());
            }
            _ => return None,
        }
    }
    let mut path = found?;
    if let Some(last) = path.segments.last_mut() {
        match &mut last.arguments {
            PathArguments::AngleBracketed(args) => {
                args.args = std::mem::take(&mut args.args)
                    .into_iter()
                    .filter(|arg| {
                        !matches!(
                            arg,
                            GenericArgument::AssocType(_)
                                | GenericArgument::AssocConst(_)
                                | GenericArgument::Constraint(_)
                        )
                    })
                    .collect();
                if args.args.is_empty() {
                    last.arguments = PathArguments::None;
                }
            }
            PathArguments::Parenthesized(args) => args.output = ReturnType::Default,
            PathArguments::None => (),
        }
    }
    Some(path)
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

/// The first argument of the Repeater for an entry: a tuple type made from the generics `iter` of
/// the entry, each by its own name. It is `RepeaterImpl::args` for the generics an impl declares and
/// what `repeater_projection()` writes for the copy. The tuple has an element for each parameter,
/// then the encoded lifetime bounds:
///
/// - a lifetime `'a` is `&'a ()`, and a type `T` is `T` (`*const T` with a `?Sized` bound, since a
///   tuple needs `Sized` elements but the last);
/// - a const is encoded by `const_fn(param)`;
/// - each bound of a generic on a lifetime is `&'a T` for `T: 'a` and `&'a &'b ()` for `'b: 'a`, and
///   each lifetime `'a` of `self_outlives` (`Context::self_outlives`) adds `&'a Self`, with `Self`
///   the type `self_ty` that stands for it.
///
/// Fails with `UnboundedTraitObject` if `self_ty` is a trait object without the lifetime bound that
/// `Self: 'a` needs, and with the error of `const_fn`.
pub(crate) fn encode_entry_args<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    (self_ty, lifetimes): (&Type, &[Lifetime]),
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
                GenericParam::Type(tp) if is_maybe_unsized(tp) => unsized_element(&tp.ident),
                GenericParam::Type(TypeParam { ident, .. }) => parse_quote!(#ident),
            })
        })
        .collect::<std::result::Result<Punctuated<Type, Token![,]>, _>>()?;
    let args: Vec<GenericArgument> = params_as_args(params.iter().copied());
    elems.extend(encode_outlives_bounds(&params, &args)?);
    elems.extend(encode_self_outlives(&params, &args, self_ty, lifetimes)?);
    Ok(Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: with_trailing_comma(elems),
    }))
}

// The arguments that pass `params` on, each by its own name: `'a` is `'a`, `T` is `T`, and
// `const N: usize` is the `Const` argument `N` (syn parses a bare `N` as a type).
pub(crate) fn params_as_args<I, C>(params: I) -> C
where
    I: IntoIterator,
    I::Item: Borrow<GenericParam>,
    C: FromIterator<GenericArgument>,
{
    params
        .into_iter()
        .map(|param| param_as_arg(param.borrow()))
        .collect()
}

/// The argument that passes `param` on by its own name.
fn param_as_arg(param: &GenericParam) -> GenericArgument {
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

/// The only rule that decides which bounds are encoded. `without_encoded_bounds()` keeps the
/// others.
fn encoded_bound_position(params: &[&GenericParam], bound: &Lifetime) -> Option<usize> {
    params.iter().position(
        |param| matches!(param, GenericParam::Lifetime(param) if &param.lifetime == bound),
    )
}

/// Returns `generics` without the lifetime bounds that the encoding of the generics encodes: a bound
/// by a lifetime parameter of `generics`, on a type parameter (`T: 'a`) or a lifetime parameter
/// (`'b: 'a`). Other bounds (`T: 'static`, trait bounds), const parameters and the where clause
/// stay. These are the generics a Repeater impl declares.
pub(crate) fn without_encoded_bounds(generics: &Generics) -> Generics {
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

/// The element of the trait arguments for the `?Sized` parameter `ident`: `*const ident`.
///
/// A tuple is well-formed only if the elements but the last are `Sized`, and a raw pointer is
/// `Sized` whatever it points to.
fn unsized_element(ident: &Ident) -> Type {
    Type::Ptr(TypePtr {
        star_token: Default::default(),
        const_token: Some(Default::default()),
        mutability: None,
        elem: Box::new(parse_quote!(#ident)),
    })
}

/// Returns `ty` as the referent of a reference. A trait object is parenthesized, and needs a
/// lifetime bound, or it would default to the lifetime of the reference. Only the type of `Self`
/// that the caller of `repeater_projection()` gives can be one: the generics are named by their
/// own names.
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

/// Why the generics of an entry cannot be encoded: the error of the call of
/// `LeakerResult::repeater_impls()`, of [`repeater_projection()`](crate::repeater_projection), of
/// [`complete_params()`], of [`substitute_args()`] and of [`Substitute::finish()`].
///
/// It converts into a `syn::Error` at the offending tokens.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// // `array_const` encodes `const N: usize` only.
/// let err = array_const(&parse_quote!(true), &parse_quote!(const B: bool)).unwrap_err();
/// assert!(matches!(&err, EncodeError::UnsupportedConstType(ty) if **ty == parse_quote!(bool)));
/// let error: syn::Error = err.into();
/// assert_eq!(error.to_string(), "unsupported type of const parameter: 'bool'");
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum EncodeError {
    /// The type of a const parameter is not a primitive type (e.g. an alias).
    UnsupportedConstType(Box<Type>),
    /// An argument that does not fit: there are more arguments than parameters, one is not of the
    /// kind of its parameter (a lifetime for a type), or a value that [`substitute_args()`] or
    /// [`Substitute`] was given for a name is not of a kind that fits the place of the name.
    UnexpectedArgument(Box<GenericArgument>),
    /// A parameter that has neither an argument nor a default.
    MissingArgument(Box<GenericParam>),
    /// `Self: 'a` is encoded for a `Self` that is a trait object without a lifetime bound (`dyn
    /// Tr`). Write it, e.g. `dyn Tr + 'static`.
    UnboundedTraitObject(Box<Type>),
    /// [`ImplSelf::Marker`](crate::ImplSelf::Marker) was given and an entry names `Self`, or its
    /// context's generics do. Use [`ImplSelf::Param`](crate::ImplSelf::Param).
    NeedsSelf(Box<Type>),
    /// The associated type of a type that is not a type parameter, as written by `T::Item` with
    /// `T` given as `IntoIter<u8>`: `<IntoIter<u8>>::Item`, which rustc takes only for a type
    /// parameter (E0223). It needs its trait, `<IntoIter<u8> as Iterator>::Item`.
    /// [`complete_params()`] writes it from the one trait bound of the parameter (`T: Iterator`),
    /// and fails if there is not one. [`substitute_args()`] and [`Substitute`] know no bounds:
    /// write the trait in the copy (`<T as Iterator>::Item`), and `T` is replaced in it. The type
    /// is the path as it would be written.
    AmbiguousAssociatedType(Box<Type>),
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
                "the type '{}' needs `Self` (it names it, or its generics do), so its Repeater impl \
                 cannot be for a marker type; use `ImplSelf::Param`",
                quote!(#ty)
            ),
            EncodeError::AmbiguousAssociatedType(ty) => write!(
                f,
                "the associated type '{}' needs its trait: rustc resolves `<Type>::Item` only for a \
                 type parameter; write `<Type as Trait>::Item`",
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
            EncodeError::AmbiguousAssociatedType(ty) => ty.span(),
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

/// `value` in braces, once: a block that holds it alone (`{ M }`) is taken off first, as rustc
/// rejects `{ { M } }` for a generic `M`.
fn braced_const(value: &Expr) -> Expr {
    let value = unwrap_block(value);
    parse_quote!({ #value })
}

/// The fold of [`complete_params()`]: it writes the default of a parameter with the
/// arguments of the parameters before it.
#[derive(Default)]
pub(crate) struct ParamFold {
    /// The argument of each type parameter completed so far.
    types: HashMap<Ident, Type>,
    /// The one trait bound of each of those that has exactly one, without its associated type
    /// bindings (`Iterator` for `T: Iterator<Item = u8>`).
    bounds: HashMap<Ident, Path>,
    /// The first associated type that could not be written.
    error: Option<EncodeError>,
    /// The value of each const parameter completed so far.
    consts: HashMap<Ident, Expr>,
    /// The argument of each lifetime parameter completed so far.
    lifetimes: HashMap<Lifetime, Lifetime>,
    /// The lifetimes that the copy of a signature binds, if the arguments add lifetimes to its
    /// inputs.
    names: FreshLifetimes,
}

impl ParamFold {
    /// The argument that `ty` is the parameter of.
    fn type_argument(&self, ty: &Type) -> Option<Type> {
        match ty {
            Type::Path(TypePath { qself: None, path }) => {
                self.types.get(path.get_ident()?).cloned()
            }
            _ => None,
        }
    }

    /// The value that `expr` is the const parameter of, in braces. `expr` may be braced already
    /// (`{ N }`), and so may the value (`{ M }`): see [`braced_const()`].
    fn const_value(&self, expr: &Expr) -> Option<Expr> {
        match unwrap_block(expr) {
            Expr::Path(ExprPath {
                qself: None, path, ..
            }) => Some(braced_const(self.consts.get(path.get_ident()?)?)),
            _ => None,
        }
    }

    /// The argument for a bare const parameter, which is parsed as a type argument (`N` in
    /// `Foo<N>`).
    fn const_argument_of(&self, arg: &GenericArgument) -> Option<GenericArgument> {
        match arg {
            GenericArgument::Type(Type::Path(TypePath { qself: None, path })) => {
                let value = self.consts.get(path.get_ident()?)?;
                Some(GenericArgument::Const(normalize_const_arg(value)))
            }
            _ => None,
        }
    }

    fn lifetime(&self, lifetime: Lifetime) -> Lifetime {
        self.lifetimes.get(&lifetime).cloned().unwrap_or(lifetime)
    }

    /// `T::A..` for the earlier parameter `name`, with `value` as its argument and `rest` the path
    /// after it: `<value>::A` if `value` is an identifier (a type parameter, which rustc resolves),
    /// else `<value as Trait>::A` with the one bound of the parameter. `original` stays if there is
    /// none, and the error is kept.
    fn projected(&mut self, name: &Ident, value: Type, rest: Path, original: Type) -> Type {
        let rest = syn::fold::fold_path(self, rest);
        if type_name(&value).is_some() {
            return Type::Path(TypePath {
                qself: Some(qself_of(value, name.span())),
                path: rest,
            });
        }
        match self.bounds.get(name).cloned() {
            Some(bound) => {
                // The arguments of the bound may name the earlier parameters.
                let bound = syn::fold::fold_path(self, bound);
                Type::Path(through_bound(value, bound, rest, name.span()))
            }
            None => {
                let ambiguous = TypePath {
                    qself: Some(qself_of(value, name.span())),
                    path: rest,
                };
                self.error
                    .get_or_insert(EncodeError::AmbiguousAssociatedType(Box::new(Type::Path(
                        ambiguous,
                    ))));
                original
            }
        }
    }
}

impl syn::fold::Fold for ParamFold {
    fn fold_type(&mut self, ty: Type) -> Type {
        if let Some(arg) = self.type_argument(&ty) {
            return arg;
        }
        // `T::A`, for the type `T` of an earlier parameter.
        if let Type::Path(path) = &ty {
            if let Some((name, rest)) = parameter_projection(path) {
                let name = name.clone();
                if let Some(value) = self.types.get(&name).cloned() {
                    return self.projected(&name, value, rest, ty);
                }
            }
        }
        syn::fold::fold_type(self, ty)
    }
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        match self.const_value(&expr) {
            Some(value) => value,
            None => syn::fold::fold_expr(self, expr),
        }
    }
    fn fold_generic_argument(&mut self, arg: GenericArgument) -> GenericArgument {
        match self.const_argument_of(&arg) {
            Some(arg) => arg,
            None => syn::fold::fold_generic_argument(self, arg),
        }
    }
    fn fold_lifetime(&mut self, lifetime: Lifetime) -> Lifetime {
        self.lifetime(lifetime)
    }
    // The elided lifetime of a signature is named where an argument adds lifetimes to its inputs.
    fn fold_type_bare_fn(&mut self, i: TypeBareFn) -> TypeBareFn {
        fold_bare_fn(self, i)
    }
    fn fold_trait_bound(&mut self, i: TraitBound) -> TraitBound {
        fold_bound(self, i)
    }
}

impl Site for ParamFold {
    fn names(&mut self) -> &mut FreshLifetimes {
        &mut self.names
    }
}

/// Replaces the names in `args` by what `f` gives for them: the arguments of a use site.
///
/// `f` is asked for a name and answers the argument that takes its place, or `None` to leave the
/// name as written (a generic the copy declares itself). It is asked once for each name, in the
/// order they come in, before anything is replaced; a value is not substituted again. It is
/// asked for:
///
/// - a type `T`, and `T::A` or `<T>::A` (`<value>::A`, for a `Type`);
/// - a const `N`, in an expression (`[u8; N]`, `{ N }`) or as an argument;
/// - a lifetime `'a`, as `a`; `'static` and `'_` are never asked;
/// - `Self`, in a type and in `Self::A` of an expression (`Self` alone in an expression is a
///   value), and in the tokens of a macro.
///
/// A value must fit the place of its name, or the call fails with
/// [`EncodeError::UnexpectedArgument`] at the value and `args` stay as they are: a `Type` or a
/// `Const` where the name is a whole argument (`T` in `Vec<T>`); a `Type` where it is part of a
/// type (`&T`, `T::A`) or `Self` in a macro; a `Const`, or a `Type` that is one identifier, in an
/// expression; a `Lifetime` for a lifetime.
///
/// `T::A` in a type is `<value>::A`, which rustc resolves only if `value` is a type parameter
/// (one identifier). For any other value the call fails with
/// [`EncodeError::AmbiguousAssociatedType`] and `args` stay as they are: it knows no bounds, so
/// write the trait in the copy (`<T as Iterator>::A`), where `T` is replaced as any name.
/// ([`complete_params()`] knows the bounds of the parameters, and writes it.) `Self::A` in an
/// expression is `<value>::A` for any value.
///
/// [`Substitute`] does the same for a whole syntax node (a signature, an impl item), lifetimes
/// outside the types included.
///
/// A const is braced once (`[u8; { N }]` with `M` is `[u8; { M }]`), a trait object with several
/// bounds that a reference or pointer points to gets parentheses, and a function pointer or
/// `Fn(..)` whose output takes an elided lifetime from the one lifetime of its inputs keeps it
/// where a value adds lifetimes to them: its elided lifetime is named by a `for<'__tl_s0>` binder
/// (`fn(&U) -> &u8` with `&'static str` is `for<'__tl_s0> fn(&'__tl_s0 &'static str) ->
/// &'__tl_s0 u8`). A lifetime that a path hides cannot be named.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// // `impl Tr<u8> for Host`, for `trait Tr<T> { fn f(&self) -> Option<(T, Self)>; }`.
/// let mut args: Vec<GenericArgument> = vec![parse_quote!(Option<(T, Self)>)];
/// substitute_args(args.iter_mut(), |name: &Ident| match name.to_string().as_str() {
///     "T" => Some(parse_quote!(u8)),
///     "Self" => Some(parse_quote!(Host)),
///     _ => None,
/// })
/// .unwrap();
/// assert_eq!(args, [parse_quote!(Option<(u8, Host)>)]);
///
/// // A lifetime for a type is an error, and nothing is replaced.
/// let mut args: Vec<GenericArgument> = vec![parse_quote!(&T)];
/// let err = substitute_args(args.iter_mut(), |_: &Ident| Some(parse_quote!('a))).unwrap_err();
/// assert!(matches!(err, EncodeError::UnexpectedArgument(_)));
/// assert_eq!(args, [parse_quote!(&T)]);
///
/// // `T::Item` for `T` as `IntoIter<u8>` needs its trait, which the copy writes.
/// let into_iter = |_: &Ident| Some(parse_quote!(IntoIter<u8>));
/// let mut args: Vec<GenericArgument> = vec![parse_quote!(Option<T::Item>)];
/// let err = substitute_args(args.iter_mut(), into_iter).unwrap_err();
/// assert!(matches!(err, EncodeError::AmbiguousAssociatedType(_)));
/// let mut args: Vec<GenericArgument> = vec![parse_quote!(Option<<T as Iterator>::Item>)];
/// substitute_args(args.iter_mut(), into_iter).unwrap();
/// assert_eq!(args, [parse_quote!(Option<<IntoIter<u8> as Iterator>::Item>)]);
/// ```
pub fn substitute_args<'a>(
    args: impl IntoIterator<Item = &'a mut GenericArgument>,
    f: impl FnMut(&Ident) -> Option<GenericArgument>,
) -> std::result::Result<(), EncodeError> {
    let mut answers = Answers::new(f);
    substitute_in(&mut answers, args.into_iter().collect())
}

/// The substitution of [`substitute_args()`] with the answers that `answers` has and asks for.
fn substitute_in<F: FnMut(&Ident) -> Option<GenericArgument>>(
    answers: &mut Answers<F>,
    mut args: Vec<&mut GenericArgument>,
) -> std::result::Result<(), EncodeError> {
    // The first pass asks `f` for the names; what it writes is dropped.
    let mut asking = SiteFold {
        answers: &mut *answers,
        names: FreshLifetimes::default(),
        naming: false,
        error: None,
    };
    for arg in &args {
        asking.fold_generic_argument((**arg).clone());
    }
    // The second writes the arguments, with no lifetime of theirs or of a value taken for a name.
    let mut names = FreshLifetimes::default();
    for arg in &args {
        names.reserve(&**arg);
    }
    for value in answers.known.values().flatten() {
        names.reserve(value);
    }
    let mut writing = SiteFold {
        answers,
        names,
        naming: true,
        error: None,
    };
    let written: Vec<GenericArgument> = args
        .iter()
        .map(|arg| writing.fold_generic_argument((**arg).clone()))
        .collect();
    if let Some(error) = writing.error {
        return Err(error);
    }
    for (arg, written) in args.iter_mut().zip(written) {
        **arg = written;
    }
    Ok(())
}

/// Writes the arguments of a use site into a whole syntax node: [`substitute_args()`] as a
/// [`VisitMut`].
///
/// `f` is the closure of [`substitute_args()`]: it answers the argument that takes the place of a
/// name (`T`, `N`, `'a` as `a`, `Self`), or `None` to keep the name. It is asked once for each name
/// over the whole visit.
///
/// The visitor writes each type it meets as one unit, as `substitute_args()` does (a const braced
/// once, a trait object behind a pointer in parentheses, `T::A` by the rules there, the elided
/// lifetime of a function pointer named where it needs a name, `Self` in a macro), and does not
/// visit inside it again. It writes every lifetime that is not in a type too: the receiver of
/// `fn keep(&'a self)`, a bound `T: 'a`, `'a: 'b` or `Self: 'a`, and the lifetime argument of a
/// bound (`T: Tr<'a>`). So do the generic arguments, the expressions (a const parameter's
/// default) and the macros outside a type. A bound `F: Fn(&U) -> &u8` outside a type is written
/// like the type `dyn Fn(&U) -> &u8`.
///
/// Visit what you copy, the items of an impl or a signature: the header of the impl declares the
/// names that the site gives. The visitor knows no scope: a name that the node declares itself, a
/// method's own `<'b, V>`, is asked like any other, so answer `None` for it to keep it.
///
/// A name that does not fit its place, or an associated type that needs its trait
/// ([`EncodeError::AmbiguousAssociatedType`]), leaves the type or lifetime where it happened as it
/// is and the visit goes on, so the node may be partly written: [`Substitute::finish()`] returns
/// the first error, and the node must not be used if it is one.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use syn::visit_mut::VisitMut;
/// // `impl<'a> Tr<'a, u8> for Host` for `trait Tr<'a, K: 'a> { fn keep(&'a self, k: Option<K>) ->
/// // &'a K where Self: 'a; }`: the lifetime outside a type is written too.
/// let mut sig: Signature =
///     parse_quote!(fn keep(&'a self, k: Option<K>) -> &'a K where Self: 'a);
/// let mut site = Substitute::new(|name: &Ident| match name.to_string().as_str() {
///     "a" => Some(parse_quote!('static)),
///     "K" => Some(parse_quote!(u8)),
///     _ => None,
/// });
/// site.visit_signature_mut(&mut sig);
/// site.finish().unwrap();
/// assert_eq!(
///     sig,
///     parse_quote!(fn keep(&'static self, k: Option<u8>) -> &'static u8 where Self: 'static)
/// );
/// ```
pub struct Substitute<F> {
    answers: Answers<F>,
    /// The first error.
    error: Option<EncodeError>,
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> Substitute<F> {
    /// Creates the visitor. `f` answers for a name, as in [`substitute_args()`].
    pub fn new(f: F) -> Self {
        Substitute {
            answers: Answers::new(f),
            error: None,
        }
    }

    /// Returns the first error of the visit, if there was one.
    pub fn finish(self) -> std::result::Result<(), EncodeError> {
        match self.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Substitutes the argument `arg` as a unit: it is left as it is on an error, which is kept.
    fn write(&mut self, arg: &mut GenericArgument) {
        if let Err(error) = substitute_in(&mut self.answers, vec![arg]) {
            self.error.get_or_insert(error);
        }
    }

    fn misfit(&mut self, value: GenericArgument) {
        self.error
            .get_or_insert(EncodeError::UnexpectedArgument(Box::new(value)));
    }
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> VisitMut for Substitute<F> {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        let mut arg = GenericArgument::Type(ty.clone());
        self.write(&mut arg);
        match arg {
            GenericArgument::Type(written) => *ty = written,
            // A const for a name that stands alone as a type.
            other => self.misfit(other),
        }
    }

    // A bare const parameter is parsed as a type argument (`N` in `Foo<N>`), and becomes a const.
    fn visit_generic_argument_mut(&mut self, arg: &mut GenericArgument) {
        self.write(arg)
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        let mut arg = GenericArgument::Const(expr.clone());
        self.write(&mut arg);
        match arg {
            GenericArgument::Const(written) => *expr = written,
            // An identifier for a const name is a type.
            GenericArgument::Type(Type::Path(TypePath { qself, path })) => {
                *expr = Expr::Path(ExprPath {
                    attrs: Vec::new(),
                    qself,
                    path,
                })
            }
            other => self.misfit(other),
        }
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        let mut arg = GenericArgument::Lifetime(lifetime.clone());
        self.write(&mut arg);
        match arg {
            GenericArgument::Lifetime(written) => *lifetime = written,
            other => self.misfit(other),
        }
    }

    // `&'a self` is `Self` by its form: only its lifetime is written. The type that syn keeps for
    // it is not printed, unless it is not `&Self` any more: an answer for `Self` would make it
    // `&'a self: &Host`.
    fn visit_receiver_mut(&mut self, receiver: &mut Receiver) {
        if receiver.colon_token.is_some() {
            return syn::visit_mut::visit_receiver_mut(self, receiver);
        }
        for attr in &mut receiver.attrs {
            self.visit_attribute_mut(attr);
        }
        if let (Some((_, Some(lifetime))), Type::Reference(ty)) =
            (&mut receiver.reference, &mut *receiver.ty)
        {
            self.visit_lifetime_mut(lifetime);
            ty.lifetime = Some(lifetime.clone());
        }
    }

    // The bound `Fn(&U) -> &u8` outside a type has an elided lifetime to name, as the type
    // `dyn Fn(&U) -> &u8` has.
    fn visit_trait_bound_mut(&mut self, bound: &mut TraitBound) {
        let object = Type::TraitObject(TypeTraitObject {
            dyn_token: None,
            bounds: std::iter::once(TypeParamBound::Trait(bound.clone())).collect(),
        });
        let mut arg = GenericArgument::Type(object);
        self.write(&mut arg);
        if let GenericArgument::Type(Type::TraitObject(TypeTraitObject { bounds, .. })) = arg {
            if let Some(TypeParamBound::Trait(written)) = bounds.into_iter().next() {
                *bound = written;
            }
        }
    }

    fn visit_macro_mut(&mut self, mac: &mut Macro) {
        let mut arg = GenericArgument::Const(Expr::Macro(ExprMacro {
            attrs: Vec::new(),
            mac: mac.clone(),
        }));
        self.write(&mut arg);
        if let GenericArgument::Const(Expr::Macro(ExprMacro { mac: written, .. })) = arg {
            *mac = written;
        }
    }

    // What replaced the bounded type (`Self`) may need parentheses there.
    fn visit_predicate_type_mut(&mut self, predicate: &mut PredicateType) {
        syn::visit_mut::visit_predicate_type_mut(self, predicate);
        parenthesize_bounded(&mut predicate.bounded_ty);
    }
}

/// What `f` answered, for each name once.
struct Answers<F> {
    f: F,
    known: HashMap<Ident, Option<GenericArgument>>,
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> Answers<F> {
    fn new(f: F) -> Self {
        Answers {
            f,
            known: HashMap::new(),
        }
    }

    fn of(&mut self, name: &Ident) -> Option<GenericArgument> {
        if let Some(answer) = self.known.get(name) {
            return answer.clone();
        }
        let answer = (self.f)(name);
        self.known.insert(name.clone(), answer.clone());
        answer
    }
}

/// The fold of [`substitute_args()`].
struct SiteFold<'f, F> {
    answers: &'f mut Answers<F>,
    names: FreshLifetimes,
    /// Whether it names the elided lifetime of a signature. The first pass writes nothing.
    naming: bool,
    /// The first value that did not fit its place, or associated type that could not be written.
    error: Option<EncodeError>,
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> SiteFold<'_, F> {
    fn misfit(&mut self, value: GenericArgument) {
        self.error
            .get_or_insert(EncodeError::UnexpectedArgument(Box::new(value)));
    }

    /// `T::A..` for the name `T` with `value` and `rest` the path after it: `<value>::A`, which
    /// rustc takes only if `value` is an identifier (a type parameter). For any other type the
    /// trait is needed, and the bounds of `T` are not known here: `original` stays, and the error
    /// is kept.
    fn projected(&mut self, name: &Ident, value: Type, rest: Path, original: Type) -> Type {
        let projected = TypePath {
            qself: Some(qself_of(value.clone(), name.span())),
            path: self.fold_path(rest),
        };
        if type_name(&value).is_some() {
            return Type::Path(projected);
        }
        self.error
            .get_or_insert(EncodeError::AmbiguousAssociatedType(Box::new(Type::Path(
                projected,
            ))));
        original
    }

    /// The type for `name`, if there is an answer: only a type fits.
    fn type_for(&mut self, name: &Ident) -> Option<Type> {
        match self.answers.of(name)? {
            GenericArgument::Type(ty) => Some(ty),
            value => {
                self.misfit(value);
                None
            }
        }
    }

    /// The const for `name`, if there is an answer, written as the expression `{ value }`: a const,
    /// or a type that is one identifier (a bare const argument is parsed as a type).
    fn braced_const_for(&mut self, name: &Ident) -> Option<Expr> {
        match self.answers.of(name)? {
            GenericArgument::Const(value) => Some(braced_const(&value)),
            GenericArgument::Type(Type::Path(TypePath { qself: None, path }))
                if path.get_ident().is_some() =>
            {
                Some(parse_quote!({ #path }))
            }
            value => {
                self.misfit(value);
                None
            }
        }
    }

    /// The tokens of a macro, or of a type `syn` does not parse, with `Self` replaced.
    fn tokens(&mut self, tokens: TokenStream) -> TokenStream {
        if !mentions_self(&tokens) {
            return tokens;
        }
        match self.type_for(&Ident::new("Self", Span::call_site())) {
            Some(self_ty) => ReplaceSelf { self_ty: &self_ty }.replace_tokens(tokens),
            None => tokens,
        }
    }
}

/// The name that an argument is as a whole: `T` or `N`.
fn argument_name(arg: &GenericArgument) -> Option<&Ident> {
    match arg {
        GenericArgument::Type(Type::Path(TypePath { qself: None, path })) => path.get_ident(),
        GenericArgument::Const(Expr::Path(ExprPath {
            attrs,
            qself: None,
            path,
        })) if attrs.is_empty() => path.get_ident(),
        _ => None,
    }
}

/// The name that an expression is, alone or in braces. `Self` is a value there.
fn expression_name(expr: &Expr) -> Option<&Ident> {
    match unwrap_block(expr) {
        Expr::Path(ExprPath {
            qself: None, path, ..
        }) => path.get_ident().filter(|ident| *ident != "Self"),
        _ => None,
    }
}

fn mentions_self(tokens: &TokenStream) -> bool {
    tokens.clone().into_iter().any(|tree| match tree {
        TokenTree::Ident(ident) => ident == "Self",
        TokenTree::Group(group) => mentions_self(&group.stream()),
        _ => false,
    })
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> syn::fold::Fold for SiteFold<'_, F> {
    fn fold_generic_argument(&mut self, arg: GenericArgument) -> GenericArgument {
        let name = match argument_name(&arg) {
            Some(name) => name,
            None => return syn::fold::fold_generic_argument(self, arg),
        };
        match self.answers.of(name) {
            Some(GenericArgument::Type(ty)) => GenericArgument::Type(ty),
            Some(GenericArgument::Const(value)) => {
                GenericArgument::Const(normalize_const_arg(&value))
            }
            Some(value) => {
                self.misfit(value);
                arg
            }
            None => arg,
        }
    }
    fn fold_type(&mut self, ty: Type) -> Type {
        if let Type::Path(path_ty) = &ty {
            if let (None, Some(name)) = (&path_ty.qself, path_ty.path.get_ident()) {
                return self.type_for(name).unwrap_or(ty);
            }
            if let Some((name, rest)) = parameter_projection(path_ty) {
                let name = name.clone();
                if let Some(value) = self.type_for(&name) {
                    return self.projected(&name, value, rest, ty);
                }
            }
        }
        match ty {
            Type::Verbatim(tokens) => Type::Verbatim(self.tokens(tokens)),
            // A value may be a trait object with several bounds, which needs parentheses behind a
            // reference or a pointer.
            ty => parenthesize_pointee(syn::fold::fold_type(self, ty)),
        }
    }
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        if let Some(value) = expression_name(&expr).and_then(|name| self.braced_const_for(name)) {
            return value;
        }
        // `Self::A`: only `Self` is a type in an expression.
        if let Expr::Path(ExprPath {
            qself: None, path, ..
        }) = &expr
        {
            if let Some(name) = projection_head(path).filter(|name| *name == "Self") {
                if let Some(value) = self.type_for(name) {
                    let TypePath { qself, path } =
                        projection(value, path, |rest| self.fold_path(rest));
                    return Expr::Path(ExprPath {
                        attrs: Vec::new(),
                        qself,
                        path,
                    });
                }
            }
        }
        syn::fold::fold_expr(self, expr)
    }
    fn fold_lifetime(&mut self, lifetime: Lifetime) -> Lifetime {
        if lifetime.ident == "static" || lifetime.ident == "_" {
            return lifetime;
        }
        match self.answers.of(&lifetime.ident) {
            Some(GenericArgument::Lifetime(value)) => value,
            Some(value) => {
                self.misfit(value);
                lifetime
            }
            None => lifetime,
        }
    }
    fn fold_macro(&mut self, mac: Macro) -> Macro {
        Macro {
            tokens: self.tokens(mac.tokens),
            ..mac
        }
    }
    // The elided lifetime of a signature is named where a value adds lifetimes to its inputs.
    fn fold_type_bare_fn(&mut self, i: TypeBareFn) -> TypeBareFn {
        if self.naming {
            fold_bare_fn(self, i)
        } else {
            syn::fold::fold_type_bare_fn(self, i)
        }
    }
    fn fold_trait_bound(&mut self, i: TraitBound) -> TraitBound {
        if self.naming {
            fold_bound(self, i)
        } else {
            syn::fold::fold_trait_bound(self, i)
        }
    }
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> Site for SiteFold<'_, F> {
    fn names(&mut self) -> &mut FreshLifetimes {
        &mut self.names
    }
}

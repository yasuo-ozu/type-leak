//! Encoding generic parameters and arguments into types.

use crate::elided::{fold_bare_fn, fold_bound, Names, Site};
use crate::util::{parenthesize_pointee, unwrap_type_group, ReplaceSelf};
use proc_macro2::{Span, TokenStream, TokenTree};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::iter::FromIterator;
use syn::fold::Fold;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
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

// The first argument of the Repeater for an entry: a tuple type made from the arguments `iter` of
// the entry's generics `params`, one for each by position and kind. It is `RepeaterImpl::args` for
// the arguments `params_as_args()` writes. The tuple has an element for each parameter, then the
// encoded lifetime bounds:
//
// - a lifetime `'a` is `&'a ()`, and a type is itself (`*const T` for a `?Sized` parameter);
// - a const is encoded by `const_fn(value, param)`;
// - each bound of a generic on a lifetime (`T: 'a`) is `&'a T`, and so is `Self: 'a`, from
//   `self_outlives` (`Context::self_outlives`) with the type that stands for `Self` (`None`: no
//   such bounds).
//
// Fails with `MissingArgument` if `iter` has too few arguments, with `UnboundedTraitObject` if an
// argument is a trait object without the lifetime bound the entry needs, and with the error of
// `const_fn`.
pub(crate) fn encode_generics_args_with_params<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &'b Punctuated<GenericParam, Token![,]>,
    self_outlives: Option<(&Type, &[Lifetime])>,
    mut const_fn: impl FnMut(&Expr, &'b ConstParam) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    let args = complete_params(params, iter)?;
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
/// `given` match `params` by position and kind, lifetimes first: a binding such as `Item = u8` is
/// skipped, and a bare identifier is a const argument (syn parses it as a type). Fails with
/// [`EncodeError::UnexpectedArgument`] for an argument of the wrong kind or too many, and with
/// [`EncodeError::MissingArgument`] for a parameter that has neither an argument nor a default.
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
/// ```
pub fn complete_params<'p, 'a>(
    params: impl IntoIterator<Item = &'p GenericParam>,
    given: impl IntoIterator<Item = &'a GenericArgument>,
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
    let params: Vec<&GenericParam> = params.into_iter().collect();
    let mut substitute = SubstituteParams::default();
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
                        Some(default) => substitute.fold_type(default.clone()),
                        None => return Err(EncodeError::MissingArgument(Box::new(param.clone()))),
                    },
                };
                substitute.types.insert(tp.ident.clone(), ty.clone());
                completed.push(GenericArgument::Type(ty));
            }
            GenericParam::Const(cp) => {
                let value: Expr = match other_args.next() {
                    Some(arg) => match const_argument(arg) {
                        Some(value) => value,
                        None => return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone()))),
                    },
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

/// The Leaker's encoding, which `encode_generics_args_with_params()` must reproduce.
///
/// A lifetime `'a` is `&'a ()`, a type `T` is `T` (`*const T` with a `?Sized` bound, since a tuple
/// needs `Sized` elements but the last), and a const is the type of `const_fn`. The lifetime bounds
/// follow: `T: 'a` is `&'a T` and `'b: 'a` is `&'a &'b ()`. Each lifetime `'a` of the `self_outlives`
/// adds `&'a Self`, with `Self` the type given.
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
    let args: Vec<GenericArgument> = params_as_args(params.iter().copied());
    elems.extend(encode_outlives_bounds(&params, &args)?);
    if let Some((self_ty, lifetimes)) = self_outlives {
        elems.extend(encode_self_outlives(&params, &args, self_ty, lifetimes)?);
    }
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

/// Why the generics of an entry cannot be encoded: the error of the call of
/// `LeakerResult::repeater_impls()`, of [`repeater_projection()`](crate::repeater_projection), of
/// [`complete_params()`] and of [`substitute_args()`].
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
    /// kind of its parameter (a lifetime for a type), or a value that [`substitute_args()`] was
    /// given for a name is not of a kind that fits the place of the name.
    UnexpectedArgument(Box<GenericArgument>),
    /// A parameter that has neither an argument nor a default.
    MissingArgument(Box<GenericParam>),
    /// `Self: 'a` is encoded for a `Self` that is a trait object without a lifetime bound (`dyn
    /// Tr`). Write it, e.g. `dyn Tr + 'static`.
    UnboundedTraitObject(Box<Type>),
    /// [`ImplSelf::Marker`](crate::ImplSelf::Marker) was given and an entry names `Self`, or its
    /// context's generics do. Use [`ImplSelf::Param`](crate::ImplSelf::Param).
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
                "the type '{}' needs `Self` (it names it, or its generics do), so its Repeater impl \
                 cannot be for a marker type; use `ImplSelf::Param`",
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
    /// The lifetimes that the copy of a signature binds, if the arguments add lifetimes to its
    /// inputs.
    names: Names,
}

impl SubstituteParams {
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
    /// (`{ N }`): the value is braced once, as rustc rejects `{ { M } }` for a generic `M`.
    fn const_value(&self, expr: &Expr) -> Option<Expr> {
        match unwrap_block(expr) {
            Expr::Path(ExprPath {
                qself: None, path, ..
            }) => {
                let value = self.consts.get(path.get_ident()?)?;
                Some(parse_quote!({ #value }))
            }
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
}

impl syn::fold::Fold for SubstituteParams {
    fn fold_type(&mut self, ty: Type) -> Type {
        match self.type_argument(&ty) {
            Some(arg) => arg,
            None => syn::fold::fold_type(self, ty),
        }
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

impl Site for SubstituteParams {
    fn names(&mut self) -> &mut Names {
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
/// - a type `T`, and `T::A` (`<value>::A`, for a `Type`);
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
/// ```
pub fn substitute_args<'a>(
    args: impl IntoIterator<Item = &'a mut GenericArgument>,
    f: impl FnMut(&Ident) -> Option<GenericArgument>,
) -> std::result::Result<(), EncodeError> {
    let mut args: Vec<&'a mut GenericArgument> = args.into_iter().collect();
    let mut answers = Answers {
        f,
        known: HashMap::new(),
    };
    // The first pass asks `f` for the names; what it writes is dropped.
    let mut asking = Substituter {
        answers: &mut answers,
        names: Names::default(),
        naming: false,
        misfit: None,
    };
    for arg in &args {
        asking.fold_generic_argument((**arg).clone());
    }
    // The second writes the arguments, with no lifetime of theirs or of a value taken for a name.
    let mut names = Names::default();
    for arg in &args {
        names.reserve(&**arg);
    }
    for value in answers.known.values().flatten() {
        names.reserve(value);
    }
    let mut writing = Substituter {
        answers: &mut answers,
        names,
        naming: true,
        misfit: None,
    };
    let written: Vec<GenericArgument> = args
        .iter()
        .map(|arg| writing.fold_generic_argument((**arg).clone()))
        .collect();
    if let Some(value) = writing.misfit {
        return Err(EncodeError::UnexpectedArgument(Box::new(value)));
    }
    for (arg, written) in args.iter_mut().zip(written) {
        **arg = written;
    }
    Ok(())
}

/// What `f` answered, for each name once.
struct Answers<F> {
    f: F,
    known: HashMap<Ident, Option<GenericArgument>>,
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> Answers<F> {
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
struct Substituter<'f, F> {
    answers: &'f mut Answers<F>,
    names: Names,
    /// Whether it names the elided lifetime of a signature. The first pass writes nothing.
    naming: bool,
    /// The first value that did not fit its place.
    misfit: Option<GenericArgument>,
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> Substituter<'_, F> {
    fn misfit(&mut self, value: GenericArgument) {
        if self.misfit.is_none() {
            self.misfit = Some(value);
        }
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
            GenericArgument::Const(value) => {
                let value = unwrap_block(&value);
                Some(parse_quote!({ #value }))
            }
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

    /// `<value>::A` for the path `T::A` and the type `value` for `T`.
    fn projection(&mut self, value: Type, path: &Path) -> TypePath {
        let rest = Path {
            leading_colon: Some(Default::default()),
            segments: path.segments.iter().skip(1).cloned().collect(),
        };
        TypePath {
            qself: Some(QSelf {
                lt_token: Default::default(),
                ty: Box::new(value),
                position: 0,
                as_token: None,
                gt_token: Default::default(),
            }),
            path: self.fold_path(rest),
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

/// The name that a path stands for, as the first segment of `T::A`.
fn projected_name(path: &Path) -> Option<&Ident> {
    let first = path.segments.first()?;
    if path.leading_colon.is_some() || path.segments.len() < 2 || !first.arguments.is_none() {
        return None;
    }
    Some(&first.ident)
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

impl<F: FnMut(&Ident) -> Option<GenericArgument>> syn::fold::Fold for Substituter<'_, F> {
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
        if let Type::Path(TypePath { qself: None, path }) = &ty {
            if let Some(name) = path.get_ident() {
                return self.type_for(name).unwrap_or(ty);
            }
            if let Some(value) = projected_name(path).and_then(|name| self.type_for(name)) {
                return Type::Path(self.projection(value, path));
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
            if let Some(name) = projected_name(path).filter(|name| *name == "Self") {
                if let Some(value) = self.type_for(name) {
                    let TypePath { qself, path } = self.projection(value, path);
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

impl<F: FnMut(&Ident) -> Option<GenericArgument>> Site for Substituter<'_, F> {
    fn names(&mut self) -> &mut Names {
        &mut self.names
    }
}
#[cfg(test)]
mod tests;

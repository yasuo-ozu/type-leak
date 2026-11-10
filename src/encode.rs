//! Encoding generic parameters and arguments into types.

use crate::elided::{fold_bare_fn, fold_bound, FreshLifetimes, Site};
use crate::repeater::encode_slot_const;
use crate::util::{
    parenthesize_bounded, parenthesize_leading_qualified_path, parenthesize_pointee, projection,
    projection_head, unwrap_type_group, ReplaceSelf, Sorted,
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

/// Completes the arguments `given` for an impl of a scope whose generics as declared are
/// `generics`, with the leaked types of their defaults already written: one argument for each
/// parameter, in its order, as a [`Substitute`].
///
/// `given` match the parameters by kind: the lifetimes go to the lifetime parameters, the types and
/// consts to the others, whatever the order between the kinds. A binding such as `Item = u8` is
/// skipped, and a bare identifier is a const argument (syn parses it as a type). The default of a
/// parameter that has no argument is taken with each parameter before it replaced by its argument.
///
/// `T::A` in a default is `<arg>::A` if the argument `arg` is one identifier, else `<arg as
/// Trait>::A` with the one trait bound of `T` (inline or in the where clause, without its
/// associated type bindings, its arguments written with the earlier parameters replaced).
///
/// Fails with [`EncodeError::MissingArgument`] for a parameter that has neither an argument of its
/// kind nor a default, with [`EncodeError::UnexpectedArgument`] for an argument left over, a const
/// argument for a type parameter, or a type that is not one identifier for a const parameter, and
/// with [`EncodeError::AmbiguousAssociatedType`] for an associated type of an argument that is not an
/// identifier if `T` has no trait bound or several.
pub(crate) fn complete_params<'a>(
    generics: &Generics,
    given: impl IntoIterator<Item = &'a GenericArgument>,
) -> std::result::Result<Substitute, EncodeError> {
    let params: Vec<&GenericParam> = generics.params.iter().collect();
    let mut substitute = ParamFold::new(generics);
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
    for param in &params {
        match param {
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => match lifetime_args.next() {
                Some(GenericArgument::Lifetime(arg)) => {
                    substitute.lifetimes.insert(lifetime.clone(), arg.clone());
                    completed.push(GenericArgument::Lifetime(arg.clone()));
                }
                _ => return Err(EncodeError::MissingArgument(Box::new((*param).clone()))),
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
                        None => {
                            return Err(EncodeError::MissingArgument(Box::new((*param).clone())))
                        }
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
                        Some(default) => {
                            let value = substitute.fold_expr(default.clone());
                            if let Some(error) = substitute.error.take() {
                                return Err(error);
                            }
                            value
                        }
                        None => {
                            return Err(EncodeError::MissingArgument(Box::new((*param).clone())))
                        }
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
    let (mut lifetimes, mut others) = (HashMap::new(), HashMap::new());
    for (param, arg) in params.into_iter().zip(completed) {
        match param {
            GenericParam::Lifetime(param) => lifetimes.insert(param.lifetime.ident.clone(), arg),
            GenericParam::Type(param) => others.insert(param.ident.clone(), arg),
            GenericParam::Const(param) => others.insert(param.ident.clone(), arg),
        };
    }
    Ok(Substitute {
        lifetimes: Sorted(lifetimes),
        others: Sorted(others),
        bounds: Sorted(substitute.bounds),
    })
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

/// The one trait bound of the parameter, inline or in a where predicate on it, as a trait path
/// without associated type bindings (`Iterator` for `T: Iterator<Item = u8>`): the trait that
/// `<value as Trait>::A` names. `None` if the parameter has none, or has several, or the bound has a
/// `for<'a>` binder. A lifetime bound and `?Sized` are not counted.
fn single_bound(param: &TypeParam, where_clause: Option<&WhereClause>) -> Option<Path> {
    let predicates = where_clause
        .into_iter()
        .flat_map(|where_clause| where_clause.predicates.iter())
        .filter_map(|predicate| match predicate {
            WherePredicate::Type(PredicateType {
                lifetimes: None,
                bounded_ty,
                bounds,
                ..
            }) if type_name(bounded_ty) == Some(&param.ident) => Some(bounds.iter()),
            _ => None,
        })
        .flatten();
    let mut found = None;
    for bound in param.bounds.iter().chain(predicates) {
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
/// the entry, each by its own name. It is the argument of `RepeaterImpl::path` for the generics an
/// impl declares and what `Projection::of()` writes for the copy. The tuple has an element for
/// each parameter, then the encoded lifetime bounds:
///
/// - a lifetime `'a` is `&'a ()`, and a type `T` is `T` (`*const T` with a `?Sized` bound, since a
///   tuple needs `Sized` elements but the last);
/// - a const is encoded into the slots of the Repeater (`encode_slot_const()`);
/// - each bound of a generic on a lifetime is `&'a T` for `T: 'a` and `&'a &'b ()` for `'b: 'a`, and
///   each lifetime `'a` of `self_outlives` (`Context::self_outlives`) adds `&'a Self`, with `Self`
///   the type `self_ty` that stands for it.
///
/// Fails with `UnboundedTraitObject` if `self_ty` is a trait object without the lifetime bound that
/// `Self: 'a` needs, and with `UnsupportedConstType` for a const of a type that has no slot.
/// `repeater` is the path of the Repeater.
pub(crate) fn encode_entry_args<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    (self_ty, lifetimes): (&Type, &[Lifetime]),
    repeater: &Path,
) -> std::result::Result<Type, EncodeError> {
    let params: Vec<&'a GenericParam> = iter.into_iter().collect();
    let mut elems = params
        .iter()
        .map(|param| -> std::result::Result<Type, EncodeError> {
            Ok(match param {
                GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                    parse_quote!(& #lifetime ())
                }
                GenericParam::Const(ConstParam { ident, ty, .. }) => {
                    encode_slot_const(repeater, ident, ty)?
                }
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
/// that the caller of `Projection::of()` gives can be one: the generics are named by their
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

/// The error of the functions that encode the generics of an entry or write the arguments of a
/// trait: `LeakerResult::repeater_impls()`, [`Projection::of()`](crate::Projection::of),
/// [`Referrer::complete_args()`](crate::Referrer::complete_args) and [`substitute_args()`].
///
/// It converts into a `syn::Error` at the offending tokens.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S<const N: ::my_crate::Alias>(Wrap<{ N }>);));
/// let referrer = leaker.result().finish().unwrap();
/// let (entry, _, context) = referrer.iter().next().unwrap();
///
/// // A const slot cannot be told for the type `Alias`, which may be an alias of any primitive type.
/// let projection = Projection::new(parse_quote!(Repeater), 1, ImplSelf::Param(parse_quote!(__S)));
/// let err = projection.of(&parse_quote!(Self), context, entry).unwrap_err();
/// assert!(matches!(&err, EncodeError::UnsupportedConstType(ty) if **ty == parse_quote!(::my_crate::Alias)));
/// let error: syn::Error = err.into();
/// assert_eq!(error.to_string(), "unsupported type of const parameter: ':: my_crate :: Alias'");
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum EncodeError {
    /// The type of a const parameter is not a primitive type (e.g. an alias).
    UnsupportedConstType(Box<Type>),
    /// An argument that does not fit: there are more arguments than parameters, a const for a type
    /// parameter, or a value that [`substitute_args()`] was given for a name is not of a kind that
    /// fits the place of the name.
    UnexpectedArgument(Box<GenericArgument>),
    /// A parameter that has neither an argument nor a default.
    MissingArgument(Box<GenericParam>),
    /// `Self: 'a` is encoded for a `Self` that is a trait object without a lifetime bound (`dyn
    /// Tr`). Write it, e.g. `dyn Tr + 'static`.
    UnboundedTraitObject(Box<Type>),
    /// [`ImplSelf::Marker`](crate::ImplSelf::Marker) was given and an entry names `Self`, or its
    /// context's generics do. Use [`ImplSelf::Param`](crate::ImplSelf::Param).
    ///
    /// The type is the leaked type of the entry, or, from [`Projection::of()`](crate::Projection::of),
    /// the type that it would write.
    NeedsSelf(Box<Type>),
    /// The associated type of a type that is not a type parameter, as written by `T::Item` with
    /// `T` given as `IntoIter<u8>`: `<IntoIter<u8>>::Item`, which rustc takes only for a type
    /// parameter (E0223). It needs its trait, `<IntoIter<u8> as Iterator>::Item`.
    /// [`Referrer::complete_args()`](crate::Referrer::complete_args) writes it in a default from
    /// the one trait bound of the parameter (`T: Iterator`), and fails if there is not one.
    /// [`substitute_args()`] knows no bounds: write the trait in the copy (`<T as Iterator>::Item`),
    /// and `T` is replaced in it. ([`Substitute`] writes it through the bound, and leaves a failure
    /// to rustc.) The type is the path as it would be written.
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
pub(crate) struct ParamFold {
    /// The argument of each type parameter completed so far.
    types: HashMap<Ident, Type>,
    /// The one trait bound of each type parameter that has exactly one, without its associated type
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
    /// The fold for the parameters of `generics`, none of them completed yet.
    fn new(generics: &Generics) -> Self {
        let mut names = FreshLifetimes::default();
        names.reserve(generics);
        names.reserve(&generics.where_clause);
        let bounds = generics
            .params
            .iter()
            .filter_map(|param| match param {
                GenericParam::Type(param) => {
                    let bound = single_bound(param, generics.where_clause.as_ref())?;
                    Some((param.ident.clone(), bound))
                }
                _ => None,
            })
            .collect();
        ParamFold {
            types: HashMap::new(),
            bounds,
            error: None,
            consts: HashMap::new(),
            lifetimes: HashMap::new(),
            names,
        }
    }

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
        // An argument may be a trait object with several bounds, which needs parentheses behind a
        // reference or a pointer.
        parenthesize_pointee(syn::fold::fold_type(self, ty))
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
/// A type that is one identifier is asked whatever it is (`u8` and `String` too), and so is the
/// first segment of a longer path (`std` in `std::vec::Vec`, `crate`): answer `None` for what is
/// not a parameter. The names that the library writes (the `'__tl_s<n>` binder below) are not
/// asked.
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
/// ([`Substitute`] knows the bounds of the parameters, and writes it.) `Self::A` in an expression
/// is `<value>::A` for any value.
///
/// [`Substitute`] does the same for the items of an impl, lifetimes outside the types included.
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
    mut f: impl FnMut(&Ident) -> Option<GenericArgument>,
) -> std::result::Result<(), EncodeError> {
    let mut answers = Answers::new(|name: &Ident, _: Kind| f(name), false);
    substitute_in(&mut answers, args.into_iter().collect(), None)
}

/// The substitution of [`substitute_args()`] with the answers that `answers` has and asks for.
/// With `bounds` it is that of a [`Substitute`]: it writes `T::A` through them, and never fails.
fn substitute_in<F: FnMut(&Ident, Kind) -> Option<GenericArgument>>(
    answers: &mut Answers<F>,
    mut args: Vec<&mut GenericArgument>,
    bounds: Option<&HashMap<Ident, Path>>,
) -> std::result::Result<(), EncodeError> {
    // The first pass asks for the names; what it writes is dropped.
    let mut asking = SiteFold {
        answers: &mut *answers,
        names: FreshLifetimes::default(),
        naming: false,
        bounds,
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
        bounds,
        error: None,
    };
    let written: Vec<GenericArgument> = args
        .iter()
        .map(|arg| writing.fold_generic_argument((**arg).clone()))
        .collect();
    // A `Substitute` leaves what does not fit as it is: rustc reports it.
    if let Some(error) = writing.error.filter(|_| bounds.is_none()) {
        return Err(error);
    }
    for (arg, written) in args.iter_mut().zip(written) {
        **arg = written;
    }
    Ok(())
}

/// Writes the arguments of an impl into the items it copies: [`substitute_args()`] as a
/// [`VisitMut`].
///
/// [`Referrer::complete_args()`](crate::Referrer::complete_args) makes it. Visit an impl or a
/// trait to write its items, never its header (generics, trait path, self type, supertraits), which
/// declares the generics itself; an item nested in a body is left too, it has a scope of its own.
/// A signature or a type may be visited alone.
///
/// Each type is written as one unit, as `substitute_args()` does, and so is each lifetime outside a
/// type (`&'a self`, `T: 'a`, `where Self: 'a`). A parameter is told by its name and kind, so
/// `'T` and `T` are two. `Self` and the names that are no parameter stay.
///
/// It cannot fail: what does not fit stays as written. `T::A` for an argument that is not one
/// identifier is `<arg as Trait>::A` with the one trait bound of `T`, else `<arg>::A`, which rustc
/// reports (E0223).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use syn::visit_mut::VisitMut;
/// let item: ItemTrait = parse_quote! {
///     trait Tr<'a, K: 'a> { fn keep(&'a self, k: Option<K>) -> &'a K where Self: 'a; }
/// };
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_trait(&item);
/// let referrer = leaker.result().finish().unwrap();
///
/// // `impl Tr<'static, u8> for Host`: the lifetime outside a type is written too.
/// let given: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
/// let projection = Projection::new(parse_quote!(Repeater), 1, ImplSelf::Param(parse_quote!(__S)));
/// let mut substitute = referrer
///     .complete_args(top_level, &given, &projection)
///     .unwrap();
/// let mut imp: ItemImpl = parse_quote! {
///     impl Tr<'static, u8> for Host {
///         fn keep(&'a self, k: Option<K>) -> &'a K where Self: 'a { todo!() }
///     }
/// };
/// substitute.visit_item_impl_mut(&mut imp);
/// let expected: ItemImpl = parse_quote! {
///     impl Tr<'static, u8> for Host {
///         fn keep(&'static self, k: Option<u8>) -> &'static u8 where Self: 'static { todo!() }
///     }
/// };
/// assert_eq!(imp, expected);
/// ```
#[derive(Clone, Debug)]
pub struct Substitute {
    /// For `impl Tr<'static, u8> for W` of `trait Tr<'a, K>`: `{ a: 'static }`.
    lifetimes: Sorted<HashMap<Ident, GenericArgument>>,
    /// For the same impl: `{ K: u8 }`. The consts are here too.
    others: Sorted<HashMap<Ident, GenericArgument>>,
    /// For `trait Tr<T: Iterator<Item = u8>>`: `{ T: Iterator }`.
    bounds: Sorted<HashMap<Ident, Path>>,
}

impl Substitute {
    /// Substitutes the argument `arg` as a unit.
    fn write(&self, arg: &mut GenericArgument) {
        let mut answers = Answers::new(
            |name: &Ident, kind: Kind| {
                let found = match kind {
                    Kind::Lifetime => &self.lifetimes,
                    Kind::TypeOrConst => &self.others,
                };
                found.get(name).cloned()
            },
            true,
        );
        // It leaves what does not fit as it is, and never fails.
        let _ = substitute_in(&mut answers, vec![arg], Some(&self.bounds));
    }
}

impl VisitMut for Substitute {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        let mut arg = GenericArgument::Type(ty.clone());
        self.write(&mut arg);
        // A const for a name that stands alone as a type does not fit: it stays.
        if let GenericArgument::Type(written) = arg {
            *ty = written;
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
            _ => (),
        }
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        let mut arg = GenericArgument::Lifetime(lifetime.clone());
        self.write(&mut arg);
        if let GenericArgument::Lifetime(written) = arg {
            *lifetime = written;
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

    // What replaced the bounded type of the first predicate (`T::Item`) may need parentheses.
    fn visit_where_clause_mut(&mut self, clause: &mut WhereClause) {
        syn::visit_mut::visit_where_clause_mut(self, clause);
        parenthesize_leading_qualified_path(clause);
    }

    // The header declares the generics itself: only the items are written.
    fn visit_item_impl_mut(&mut self, imp: &mut ItemImpl) {
        for item in &mut imp.items {
            self.visit_impl_item_mut(item);
        }
    }

    fn visit_item_trait_mut(&mut self, item: &mut ItemTrait) {
        for item in &mut item.items {
            self.visit_trait_item_mut(item);
        }
    }

    // An item in a body has its own scope: the generics of the impl are not its (E0401).
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        if !matches!(stmt, Stmt::Item(_)) {
            syn::visit_mut::visit_stmt_mut(self, stmt);
        }
    }
}

/// What a name is: a lifetime has a name of its own, apart from a type or a const.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Lifetime,
    TypeOrConst,
}

/// What `answer` said, for each name once.
struct Answers<F> {
    answer: F,
    /// Whether a lifetime and a type of one name are two questions, each asked once.
    by_kind: bool,
    known: HashMap<(Kind, Ident), Option<GenericArgument>>,
}

impl<F: FnMut(&Ident, Kind) -> Option<GenericArgument>> Answers<F> {
    fn new(answer: F, by_kind: bool) -> Self {
        Answers {
            answer,
            by_kind,
            known: HashMap::new(),
        }
    }

    fn of(&mut self, name: &Ident, kind: Kind) -> Option<GenericArgument> {
        let key = (
            if self.by_kind {
                kind
            } else {
                Kind::TypeOrConst
            },
            name.clone(),
        );
        if let Some(answer) = self.known.get(&key) {
            return answer.clone();
        }
        let answer = (self.answer)(name, kind);
        self.known.insert(key, answer.clone());
        answer
    }
}

/// The fold of [`substitute_args()`].
struct SiteFold<'f, F> {
    answers: &'f mut Answers<F>,
    names: FreshLifetimes,
    /// Whether it names the elided lifetime of a signature. The first pass writes nothing.
    naming: bool,
    /// The one trait bound of each type parameter, for the fold of a [`Substitute`]: `T::A` is
    /// written through it, and never refused.
    bounds: Option<&'f HashMap<Ident, Path>>,
    /// The first value that did not fit its place, or associated type that could not be written.
    error: Option<EncodeError>,
}

impl<F: FnMut(&Ident, Kind) -> Option<GenericArgument>> SiteFold<'_, F> {
    fn misfit(&mut self, value: GenericArgument) {
        self.error
            .get_or_insert(EncodeError::UnexpectedArgument(Box::new(value)));
    }

    /// `T::A..` for the name `T` with `value` and `rest` the path after it: `<value>::A`, which
    /// rustc takes only if `value` is an identifier (a type parameter). For any other type the
    /// trait is needed: a `Substitute` writes it from the bound of `T` (and `<value>::A` if there is
    /// none), else the bounds of `T` are not known: `original` stays, and the error is kept.
    fn projected(&mut self, name: &Ident, value: Type, rest: Path, original: Type) -> Type {
        let rest = self.fold_path(rest);
        let span = name.span();
        if type_name(&value).is_none() {
            if let Some(bounds) = self.bounds {
                if let Some(bound) = bounds.get(name).cloned() {
                    let bound = self.fold_path(bound);
                    return Type::Path(through_bound(value, bound, rest, span));
                }
            } else {
                let ambiguous = TypePath {
                    qself: Some(qself_of(value, span)),
                    path: rest,
                };
                self.error
                    .get_or_insert(EncodeError::AmbiguousAssociatedType(Box::new(Type::Path(
                        ambiguous,
                    ))));
                return original;
            }
        }
        Type::Path(TypePath {
            qself: Some(qself_of(value, span)),
            path: rest,
        })
    }

    /// The type for `name`, if there is an answer: only a type fits.
    fn type_for(&mut self, name: &Ident) -> Option<Type> {
        match self.answers.of(name, Kind::TypeOrConst)? {
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
        match self.answers.of(name, Kind::TypeOrConst)? {
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

impl<F: FnMut(&Ident, Kind) -> Option<GenericArgument>> syn::fold::Fold for SiteFold<'_, F> {
    fn fold_generic_argument(&mut self, arg: GenericArgument) -> GenericArgument {
        let name = match argument_name(&arg) {
            Some(name) => name,
            None => return syn::fold::fold_generic_argument(self, arg),
        };
        match self.answers.of(name, Kind::TypeOrConst) {
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
        if lifetime.ident == "static" || lifetime.ident == "_" || self.names.made(&lifetime) {
            return lifetime;
        }
        match self.answers.of(&lifetime.ident, Kind::Lifetime) {
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

impl<F: FnMut(&Ident, Kind) -> Option<GenericArgument>> Site for SiteFold<'_, F> {
    fn names(&mut self) -> &mut FreshLifetimes {
        &mut self.names
    }
}

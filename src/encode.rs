//! Encoding generic parameters and arguments into types.

use crate::util::{parenthesize_pointee, unwrap_type_group, ReplaceSelf};
use proc_macro2::Span;
use std::collections::HashMap;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::*;
use template_quote::quote;

/// Encodes the const argument `value` of a `usize` parameter as the array type `[(); value]`.
///
/// It is the `const_fn` of [`Referrer::repeater_impl()`](crate::Referrer::repeater_impl) and
/// [`Referrer::entry_args()`](crate::Referrer::entry_args) if every const parameter is `usize`.
/// Any other type is an [`EncodeError::UnsupportedConstType`]: use
/// [`RepeaterSlots`](crate::RepeaterSlots).
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

/// Encodes the arguments that a copy gives to the generics of an entry: the counterpart of
/// `encode_generics_params_with()` for `params`.
///
/// The arguments are completed by `complete_generics_args()`, and `const_fn` encodes each const.
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

/// Checks that `args` are arguments for the first of `params`, by position: not more than there
/// are parameters, each of the kind of its parameter. A parameter with no argument is not an
/// error here: the arguments may be those before a default.
pub(crate) fn check_root_args(
    params: &Punctuated<GenericParam, Token![,]>,
    args: &[GenericArgument],
) -> std::result::Result<(), EncodeError> {
    for (param, arg) in params.iter().zip(args) {
        let fits = match (param, arg) {
            (GenericParam::Lifetime(_), GenericArgument::Lifetime(_))
            | (GenericParam::Type(_), GenericArgument::Type(_)) => true,
            (GenericParam::Const(_), arg) => const_argument(arg).is_some(),
            _ => false,
        };
        if !fits {
            return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone())));
        }
    }
    match args.get(params.len()) {
        Some(arg) => Err(EncodeError::UnexpectedArgument(Box::new(arg.clone()))),
        None => Ok(()),
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

/// Why the generics of an entry cannot be encoded: the error of `Referrer::repeater_impl()`,
/// `Referrer::entry_args()` and `Referrer::complete_args()`.
///
/// It converts into a `syn::Error` at the offending tokens.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let tr: ItemTrait = parse_quote!(trait Tr<const B: bool> { fn f(&self) -> Wrap<B>; });
/// let mut leaker = Leaker::new();
/// leaker.intern_trait(&tr, true);
/// let referrer = leaker.result().finish().unwrap();
///
/// // `array_const` encodes `const N: usize` only.
/// let target = ImplSelf::Param(&parse_quote!(__S));
/// let err = referrer.repeater_impl(0, target, array_const).unwrap_err();
/// assert!(matches!(&err, EncodeError::UnsupportedConstType(ty) if **ty == parse_quote!(bool)));
/// let error: syn::Error = err.into();
/// assert_eq!(error.to_string(), "unsupported type of const parameter: 'bool'");
/// ```
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
    /// (`dyn Tr`). Write it, e.g. `dyn Tr + 'static`.
    UnboundedTraitObject(Box<Type>),
    /// An entry that [needs `Self`](crate::Referrer::needs_self) was asked for
    /// [`ImplSelf::Marker`](crate::ImplSelf::Marker). Use
    /// [`ImplSelf::Param`](crate::ImplSelf::Param).
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
}

impl SubstituteParams {
    /// The substitution of `params` by `args`, which pair up by position. An argument of another
    /// kind than its parameter, and a parameter with no argument, are left out.
    pub(crate) fn of(
        params: &Punctuated<GenericParam, Token![,]>,
        args: &[GenericArgument],
    ) -> Self {
        let mut substitute = Self::default();
        for (param, arg) in params.iter().zip(args) {
            match (param, arg) {
                (
                    GenericParam::Lifetime(LifetimeParam { lifetime, .. }),
                    GenericArgument::Lifetime(arg),
                ) => {
                    substitute.lifetimes.insert(lifetime.clone(), arg.clone());
                }
                (GenericParam::Type(TypeParam { ident, .. }), GenericArgument::Type(ty)) => {
                    substitute.types.insert(ident.clone(), ty.clone());
                }
                (GenericParam::Const(ConstParam { ident, .. }), arg) => {
                    if let Some(value) = const_argument(arg) {
                        // `fold_expr()` writes the braces of a value that needs them.
                        let value = unwrap_block(&value).clone();
                        substitute.consts.insert(ident.clone(), value);
                    }
                }
                _ => (),
            }
        }
        substitute
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

    /// The value that `expr` is the const parameter of, in braces.
    fn const_value(&self, expr: &Expr) -> Option<Expr> {
        match expr {
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
}

/// Writes a type at a use site: the substitution of `params`, and `Self` replaced by `self_ty`, in
/// one pass, so that what replaces a name is not substituted again.
pub(crate) struct SubstituteSite<'a> {
    pub(crate) params: SubstituteParams,
    pub(crate) self_ty: Option<&'a Type>,
}

impl SubstituteSite<'_> {
    /// `T::A` for a type parameter `T` that has an argument is `<arg>::A`.
    fn projection_of_parameter(&mut self, ty: &Type) -> Option<Type> {
        let path = match ty {
            Type::Path(TypePath { qself: None, path }) => path,
            _ => return None,
        };
        let first = path.segments.first()?;
        if path.leading_colon.is_some() || path.segments.len() < 2 || !first.arguments.is_none() {
            return None;
        }
        let arg = self.params.types.get(&first.ident)?.clone();
        let rest = Path {
            leading_colon: Some(Default::default()),
            segments: path.segments.iter().skip(1).cloned().collect(),
        };
        Some(Type::Path(TypePath {
            qself: Some(QSelf {
                lt_token: Default::default(),
                ty: Box::new(arg),
                position: 0,
                as_token: None,
                gt_token: Default::default(),
            }),
            path: syn::fold::Fold::fold_path(self, rest),
        }))
    }
}

impl syn::fold::Fold for SubstituteSite<'_> {
    fn fold_type(&mut self, ty: Type) -> Type {
        if let Some(arg) = self.params.type_argument(&ty) {
            return arg;
        }
        if let Some(projection) = self.projection_of_parameter(&ty) {
            return projection;
        }
        match self.self_ty {
            Some(self_ty) => ReplaceSelf { self_ty }.fold_type_with(self, ty),
            // An argument may be a trait object with several bounds, which needs parentheses
            // behind a reference or a pointer.
            None => parenthesize_pointee(syn::fold::fold_type(self, ty)),
        }
    }
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        if let Some(value) = self.params.const_value(&expr) {
            return value;
        }
        match self.self_ty {
            Some(self_ty) => ReplaceSelf { self_ty }.fold_expr_with(self, expr),
            None => syn::fold::fold_expr(self, expr),
        }
    }
    fn fold_generic_argument(&mut self, arg: GenericArgument) -> GenericArgument {
        match self.params.const_argument_of(&arg) {
            Some(arg) => arg,
            None => syn::fold::fold_generic_argument(self, arg),
        }
    }
    fn fold_lifetime(&mut self, lifetime: Lifetime) -> Lifetime {
        self.params.lifetime(lifetime)
    }
    fn fold_macro(&mut self, mac: Macro) -> Macro {
        match self.self_ty {
            Some(self_ty) => ReplaceSelf { self_ty }.fold_macro(mac),
            None => mac,
        }
    }
}

#[cfg(test)]
mod tests;

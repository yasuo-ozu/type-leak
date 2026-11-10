//! Encoding generic parameters and arguments into types, and the arguments of a use site.

use crate::elided::{fold_bare_fn, fold_bound, FreshLifetimes, Site};
use crate::repeater::encode_slot_const;
use crate::util::{
    after_head, has_lifetime_bound, is_named, lifetime_name, parenthesize_bounded,
    parenthesize_leading_qualified_path, parenthesize_pointee, projection, projection_head,
    qself_of, same_lifetime, same_name, unwrap_type_group, ReplaceSelf, Sorted,
};
use proc_macro2::{Span, TokenStream, TokenTree};
use std::collections::HashMap;
use syn::ext::IdentExt;
use syn::fold::Fold;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;

/// An error of writing a path through the Repeater, or the arguments of an impl.
///
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls),
/// [`Projection::of()`](crate::Projection::of),
/// [`Referrer::complete_args()`](crate::Referrer::complete_args),
/// [`Visitor::with_self()`](crate::Visitor::with_self) and [`substitute_args()`] return it.
/// `syn::Error::from()` puts it at the tokens that it holds.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S<const N: ::my_crate::Alias>(Wrap<{ N }>);));
/// let referrer = leaker.result().finish().unwrap();
/// let (entry, _, context) = referrer.iter().next().unwrap();
///
/// // `Alias` may be an alias of any primitive type, so its const slot cannot be known.
/// let projection = Projection::new(parse_quote!(Repeater), 1, ImplSelf::Param(parse_quote!(__S)));
/// let err = projection.of(&parse_quote!(Self), context, entry).unwrap_err();
/// let alias: Type = parse_quote!(::my_crate::Alias);
/// assert!(matches!(&err, EncodeError::UnsupportedConstType(ty) if **ty == alias));
/// let error: syn::Error = err.into();
/// assert_eq!(error.to_string(), "unsupported type of const parameter: ':: my_crate :: Alias'");
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum EncodeError {
    /// The type of a const parameter is not a primitive type (`const N: MyAlias`). A type alias
    /// is refused too: its const slot cannot be known.
    UnsupportedConstType(Box<Type>),
    /// An argument does not fit: one too many, a const for a type parameter, or a value of
    /// [`substitute_args()`] that does not fit the place of its name.
    UnexpectedArgument(Box<GenericArgument>),
    /// A parameter has no argument and no default.
    MissingArgument(Box<GenericParam>),
    /// `Self` is a trait object with no lifetime bound (`dyn Tr`), and an entry needs `Self: 'a`.
    /// Write a bound: `dyn Tr + 'static`.
    UnboundedTraitObject(Box<Type>),
    /// [`ImplSelf::Marker`](crate::ImplSelf::Marker) is used, but an entry needs `Self` (the entry
    /// or its context names `Self`). Use [`ImplSelf::Param`](crate::ImplSelf::Param) or
    /// [`ImplSelf::Bucket`](crate::ImplSelf::Bucket).
    ///
    /// The type is the leaked type of the entry. From [`Projection::of()`](crate::Projection::of),
    /// it is the type that `of()` would write.
    NeedsSelf(Box<Type>),
    /// `T::A`, where the argument of `T` is not one identifier (`T` is `IntoIter<u8>`).
    ///
    /// rustc reads `<IntoIter<u8>>::Item` only for a type parameter (E0223). It needs the trait:
    /// `<IntoIter<u8> as Iterator>::Item`.
    /// [`Referrer::complete_args()`](crate::Referrer::complete_args) writes that in a default from
    /// the one trait bound of `T`, and fails without one. [`substitute_args()`] knows no bounds,
    /// so it fails: write `<T as Iterator>::Item` in the copy instead. The type is the path that
    /// would be written.
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
                "the type '{}' needs `Self` (it names it, or its context does), so its Repeater \
                 impl cannot be for a marker type; use `ImplSelf::Param`",
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
    /// Returns the span of the parameter, argument or type that the error holds.
    pub fn span(&self) -> Span {
        match self {
            EncodeError::UnsupportedConstType(ty)
            | EncodeError::UnboundedTraitObject(ty)
            | EncodeError::NeedsSelf(ty)
            | EncodeError::AmbiguousAssociatedType(ty) => ty.span(),
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

/// The first argument of the Repeater for an entry: a tuple type made from the generics `iter` of
/// the entry, each by its own name. It is the argument of `RepeaterImpl::path` for the generics an
/// impl declares and what `Projection::of()` writes for the copy. The tuple has an element for
/// each parameter, then the encoded lifetime bounds:
///
/// - a lifetime `'a` is `&'a ()`, and a type `T` is `T` (`*const T` with a `?Sized` bound, since a
///   tuple needs `Sized` elements but the last);
/// - a const is encoded into the slots of the Repeater (`encode_slot_const()`);
/// - each bound of a generic on a lifetime is `&'a T` for `T: 'a` and `&'a &'b ()` for `'b: 'a`,
///   and each lifetime `'a` of `lifetimes` (`Context::self_outlives`) adds `&'a Self`, with `Self`
///   the type `self_ty` that stands for it.
///
/// Fails with `UnboundedTraitObject` if `self_ty` is a trait object without the lifetime bound that
/// `Self: 'a` needs, and with `UnsupportedConstType` for a const of a type that has no slot.
/// `repeater` is the path of the Repeater.
pub(crate) fn encode_entry_args<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    self_ty: &Type,
    lifetimes: &[Lifetime],
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
    elems.extend(encode_outlives_bounds(&params));
    elems.extend(encode_self_outlives(&params, self_ty, lifetimes)?);
    Ok(Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: with_trailing_comma(elems),
    }))
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

/// The lifetime parameter of `params` that `bound` names. The only rule that decides which bounds
/// are encoded. `without_encoded_bounds()` keeps the others.
fn encoded_outliving<'p>(params: &[&'p GenericParam], bound: &Lifetime) -> Option<&'p Lifetime> {
    params.iter().find_map(|param| match param {
        GenericParam::Lifetime(param) if same_lifetime(&param.lifetime, bound) => {
            Some(&param.lifetime)
        }
        _ => None,
    })
}

/// The generics a Repeater impl declares: `generics` without the bounds by one of its lifetime
/// parameters (`T: 'a`, `'b: 'a`), which the encoding has. Other bounds (`T: 'static`, trait
/// bounds) and the where clause stay.
pub(crate) fn without_encoded_bounds(generics: &Generics) -> Generics {
    let params: Vec<&GenericParam> = generics.params.iter().collect();
    let mut declared = generics.clone();
    for param in declared.params.iter_mut() {
        match param {
            GenericParam::Lifetime(param) => {
                param.bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| encoded_outliving(&params, bound).is_none())
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
                            if encoded_outliving(&params, lifetime).is_some())
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
fn encode_outlives_bounds(params: &[&GenericParam]) -> Vec<Type> {
    let mut elems = Vec::new();
    for param in params {
        let referent: Type = match param {
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => parse_quote!(& #lifetime ()),
            GenericParam::Type(TypeParam { ident, .. }) => parse_quote!(#ident),
            GenericParam::Const(_) => continue,
        };
        for bound in lifetime_bounds(param) {
            if let Some(outliving) = encoded_outliving(params, bound) {
                elems.push(reference(outliving, referent.clone()));
            }
        }
    }
    elems
}

/// The elements `&'a self_ty` for each lifetime `'a` of `lifetimes` that `params` declare:
/// `&'a __S` implies `__S: 'a` wherever it is written, where a declared `__S: 'a` would fail if
/// nothing proves it, such as in a free function signature.
pub(crate) fn encode_self_outlives(
    params: &[&GenericParam],
    self_ty: &Type,
    lifetimes: &[Lifetime],
) -> std::result::Result<Vec<Type>, EncodeError> {
    let mut elems = Vec::new();
    for lifetime in lifetimes {
        if let Some(outliving) = encoded_outliving(params, lifetime) {
            elems.push(reference(outliving, referent_of_reference(self_ty)?));
        }
    }
    Ok(elems)
}

/// The reference type `&'lifetime referent`.
fn reference(lifetime: &Lifetime, referent: Type) -> Type {
    Type::Reference(TypeReference {
        and_token: Default::default(),
        lifetime: Some(lifetime.clone()),
        mutability: None,
        elem: Box::new(referent),
    })
}

/// Whether `param` may be unsized: it has a `?Sized` bound, as the generics of a context do (the
/// Leaker moves a where predicate `T: ?Sized` to the parameter).
fn is_maybe_unsized(param: &TypeParam) -> bool {
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

/// The element for the `?Sized` parameter `ident`: `*const ident`, which is `Sized` whatever it
/// points to, as a tuple needs for every element but the last.
fn unsized_element(ident: &Ident) -> Type {
    Type::Ptr(TypePtr {
        star_token: Default::default(),
        const_token: Some(Default::default()),
        mutability: None,
        elem: Box::new(parse_quote!(#ident)),
    })
}

/// Returns `ty` as the referent of a reference: a trait object is parenthesized, and needs a
/// lifetime bound, or it would take that of the reference. Only the `Self` that the caller of
/// `Projection::of()` gives can be one.
fn referent_of_reference(ty: &Type) -> std::result::Result<Type, EncodeError> {
    match unwrap_type_group(ty) {
        Type::TraitObject(object) => {
            if !has_lifetime_bound(object) {
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

/// Adds a trailing comma to a single-element tuple, so that `(T,)` is not parsed as `(T)`.
fn with_trailing_comma(mut elems: Punctuated<Type, Token![,]>) -> Punctuated<Type, Token![,]> {
    if elems.len() == 1 && !elems.trailing_punct() {
        elems.push_punct(Default::default());
    }
    elems
}

/// Completes the arguments `given` for the generics as declared `generics`, whose leaked defaults
/// are written already, as a [`Substitute`]: see
/// [`Referrer::complete_args()`](crate::Referrer::complete_args) for the rules and the errors.
pub(crate) fn complete_params<'a>(
    generics: &Generics,
    given: impl IntoIterator<Item = &'a GenericArgument>,
) -> std::result::Result<Substitute, EncodeError> {
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
    let (mut lifetimes, mut others) = (HashMap::new(), HashMap::new());
    for param in &generics.params {
        let missing = || EncodeError::MissingArgument(Box::new(param.clone()));
        match param {
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => match lifetime_args.next() {
                Some(GenericArgument::Lifetime(arg)) => {
                    substitute
                        .lifetimes
                        .insert(lifetime_name(lifetime), arg.clone());
                    let arg = GenericArgument::Lifetime(arg.clone());
                    lifetimes.insert(lifetime.ident.unraw(), arg);
                }
                _ => return Err(missing()),
            },
            GenericParam::Type(tp) => {
                let ty: Type = match (other_args.next(), &tp.default) {
                    (Some(GenericArgument::Type(ty)), _) => ty.clone(),
                    (Some(arg), _) => {
                        return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone())))
                    }
                    (None, Some(default)) => substitute.default_of(default, Fold::fold_type)?,
                    (None, None) => return Err(missing()),
                };
                substitute.types.insert(tp.ident.unraw(), ty.clone());
                others.insert(tp.ident.unraw(), GenericArgument::Type(ty));
            }
            GenericParam::Const(cp) => {
                let value: Expr = match (other_args.next(), &cp.default) {
                    (Some(arg), _) => match const_argument(arg) {
                        Some(value) => value,
                        None => return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone()))),
                    },
                    (None, Some(default)) => substitute.default_of(default, Fold::fold_expr)?,
                    (None, None) => return Err(missing()),
                };
                let arg = GenericArgument::Const(normalize_const_arg(&value));
                others.insert(cp.ident.unraw(), arg);
                substitute.consts.insert(cp.ident.unraw(), value);
            }
        }
    }
    if let Some(arg) = lifetime_args.next().or_else(|| other_args.next()) {
        return Err(EncodeError::UnexpectedArgument(Box::new(arg.clone())));
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

/// `T::A..` for the parameter `name` with the argument `value`, `rest` the path after `T`:
/// `<value>::A` for one identifier, else `<value as Trait>::A` with `bound`, the one trait bound
/// of `T`, or `Err(<value>::A)` without one. `fold` writes `rest`, then `bound`.
fn projected<F: Fold>(
    fold: &mut F,
    name: &Ident,
    value: Type,
    rest: Path,
    bound: Option<Path>,
) -> std::result::Result<Type, Box<Type>> {
    let rest = fold.fold_path(rest);
    let span = name.span();
    // rustc resolves `<value>::A` only for a type parameter.
    if type_name(&value).is_none() {
        if let Some(bound) = bound {
            // The arguments of the bound may name the parameters.
            let bound = fold.fold_path(bound);
            return Ok(Type::Path(through_bound(value, bound, rest, span)));
        }
        return Err(Box::new(Type::Path(TypePath {
            qself: Some(qself_of(value, span)),
            path: rest,
        })));
    }
    Ok(Type::Path(TypePath {
        qself: Some(qself_of(value, span)),
        path: rest,
    }))
}

/// The name `T` and the rest of the path if `ty` is an associated type of it, `T::A..` or
/// `<T>::A..` (`T` one identifier, with no arguments). The rest has a leading `::`, as
/// `after_head()` writes it.
fn parameter_projection(ty: &TypePath) -> Option<(&Ident, Path)> {
    match &ty.qself {
        None => Some((projection_head(&ty.path)?, after_head(&ty.path))),
        Some(QSelf {
            ty: head,
            position: 0,
            as_token: None,
            ..
        }) => Some((
            type_name(head)?,
            Path {
                leading_colon: Some(Default::default()),
                segments: ty.path.segments.clone(),
            },
        )),
        _ => None,
    }
}

/// The one trait bound of `param`, inline or in a where predicate, without associated type
/// bindings (`Iterator` for `T: Iterator<Item = u8>`). `None` for none, several, or one with a
/// `for<'a>` binder, on the bound or on its where predicate; lifetime bounds and `?Sized` do not
/// count.
fn single_bound(param: &TypeParam, where_clause: Option<&WhereClause>) -> Option<Path> {
    let predicates = where_clause
        .into_iter()
        .flat_map(|where_clause| where_clause.predicates.iter())
        .filter_map(|predicate| match predicate {
            WherePredicate::Type(PredicateType {
                lifetimes,
                bounded_ty,
                bounds,
                ..
            }) if type_name(bounded_ty).map_or(false, |name| same_name(name, &param.ident)) => {
                Some(bounds.iter().map(move |bound| (bound, lifetimes.is_some())))
            }
            _ => None,
        })
        .flatten();
    let mut found = None;
    let inline = param.bounds.iter().map(|bound| (bound, false));
    for (bound, binder) in inline.chain(predicates) {
        match bound {
            TypeParamBound::Lifetime(_)
            | TypeParamBound::Trait(TraitBound {
                modifier: TraitBoundModifier::Maybe(_),
                ..
            }) => (),
            TypeParamBound::Trait(bound)
                if found.is_none() && !binder && bound.lifetimes.is_none() =>
            {
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

/// Braces everything but a literal or an identifier, or it could parse as a type.
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

/// The fold of [`complete_params()`]: it writes the default of a parameter with the
/// arguments of the parameters before it.
struct ParamFold {
    /// The argument of each type parameter completed so far, by its name without `r#`.
    types: HashMap<Ident, Type>,
    /// The one trait bound of each type parameter that has exactly one, without its associated type
    /// bindings (`Iterator` for `T: Iterator<Item = u8>`), by its name without `r#`.
    bounds: HashMap<Ident, Path>,
    /// The first associated type that could not be written.
    error: Option<EncodeError>,
    /// The value of each const parameter completed so far, by its name without `r#`.
    consts: HashMap<Ident, Expr>,
    /// The argument of each lifetime parameter completed so far, by its name.
    lifetimes: HashMap<String, Lifetime>,
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
                    Some((param.ident.unraw(), bound))
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

    /// The `default` of a parameter written by `fold` with the arguments so far, or the first
    /// associated type in it that could not be written.
    fn default_of<T: Clone>(
        &mut self,
        default: &T,
        fold: impl FnOnce(&mut Self, T) -> T,
    ) -> std::result::Result<T, EncodeError> {
        let value = fold(self, default.clone());
        match self.error.take() {
            Some(error) => Err(error),
            None => Ok(value),
        }
    }

    /// The argument that `ty` is the parameter of.
    fn type_argument(&self, ty: &Type) -> Option<Type> {
        match ty {
            Type::Path(TypePath { qself: None, path }) => {
                self.types.get(&path.get_ident()?.unraw()).cloned()
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
            }) => Some(braced_const(self.consts.get(&path.get_ident()?.unraw())?)),
            _ => None,
        }
    }

    /// The argument for a bare const parameter, which is parsed as a type argument (`N` in
    /// `Foo<N>`).
    fn const_argument_of(&self, arg: &GenericArgument) -> Option<GenericArgument> {
        match arg {
            GenericArgument::Type(Type::Path(TypePath { qself: None, path })) => {
                let value = self.consts.get(&path.get_ident()?.unraw())?;
                Some(GenericArgument::Const(normalize_const_arg(value)))
            }
            _ => None,
        }
    }

    fn lifetime(&self, lifetime: Lifetime) -> Lifetime {
        self.lifetimes
            .get(&lifetime_name(&lifetime))
            .cloned()
            .unwrap_or(lifetime)
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
                if let Some(value) = self.types.get(&name.unraw()).cloned() {
                    let bound = self.bounds.get(&name.unraw()).cloned();
                    return match projected(self, &name, value, rest, bound) {
                        Ok(projected) => projected,
                        // `ty` stays, and the error is kept.
                        Err(ambiguous) => {
                            let error = EncodeError::AmbiguousAssociatedType(ambiguous);
                            self.error.get_or_insert(error);
                            ty
                        }
                    };
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

/// `value` in braces, once: a block that holds it alone (`{ M }`) is taken off first, as rustc
/// rejects `{ { M } }` for a generic `M`.
fn braced_const(value: &Expr) -> Expr {
    let value = unwrap_block(value);
    parse_quote!({ #value })
}

/// Writes the arguments of a use site into `args`: each name becomes what `f` returns for it.
///
/// `f` gets a name and returns the argument for it, or `None` to keep the name (a generic that
/// the copy declares itself). It is called once for each name, in order, before anything is
/// replaced. A returned value is not substituted again. `f` is asked for:
///
/// - a type `T`, also in `T::A` and `<T>::A`;
/// - the first segment of a longer path in an expression (`T::LEN`, `T::f()`);
/// - a const `N`, in an expression (`[u8; N]`, `{ N }`) or as an argument;
/// - a lifetime `'a`, as `a` (never `'static` or `'_`);
/// - `Self` in a type, as the first segment in an expression (`Self::A`), and in the tokens of a
///   macro. `Self` alone in an expression is a value, and is not asked.
///
/// `f` is also asked for names that are not parameters: a type of one identifier (`u8`,
/// `String`) and the first segment of a longer path (`std` in `std::vec::Vec`, `crate`). Return
/// `None` for them. The names that this crate writes (the binder `'__tl_s0`) are not asked.
///
/// It also:
///
/// - writes `T::LEN` in an expression as `<value>::LEN` for any `Type` value (another kind of
///   value keeps the path);
/// - braces a const once (`[u8; { N }]` with `M` is `[u8; { M }]`);
/// - puts parentheses around a trait object with several bounds behind a reference or a pointer;
/// - keeps the elided output lifetime of a `fn(..)` or `Fn(..)` when a value adds lifetimes to
///   its inputs, with a `for<'__tl_s0>` binder: `fn(&U) -> &u8` with `&'static str` is
///   `for<'__tl_s0> fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8`. A lifetime that a path hides
///   cannot be named.
///
/// [`Substitute`] does the same for the items of an impl.
///
/// # Errors
///
/// On an error, `args` stay as they were.
///
/// - [`EncodeError::UnexpectedArgument`]: a value does not fit the place of its name. A whole
///   argument (`T` in `Vec<T>`) takes a `Type` or a `Const`. A part of a type (`&T`, `T::A`) and
///   `Self` in a macro take a `Type`. An expression takes a `Const`, or a `Type` that is one
///   identifier. A lifetime takes a `Lifetime`.
/// - [`EncodeError::AmbiguousAssociatedType`]: `T::A` in a type, and the value of `T` is not one
///   identifier. rustc would refuse `<value>::A` (E0223), and this function knows no bounds.
///   Write the trait in the copy (`<T as Iterator>::A`): `T` is replaced in it as usual.
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
    let answer = |name: &Ident, _: Kind| f(name);
    substitute_in(answer, args.into_iter().collect(), Mode::Args)
}

/// What [`substitute_in()`] writes.
#[derive(Clone, Copy)]
enum Mode<'b> {
    /// The arguments of [`substitute_args()`]: a name is asked once whatever its kind, and what
    /// does not fit is an error.
    Args,
    /// The items of an impl, for a [`Substitute`]: a lifetime and a type of one name are two
    /// questions, `T::A` is written through the one trait bound of `T`, and nothing fails.
    Impl(&'b HashMap<Ident, Path>),
}

/// The substitution of [`substitute_args()`], with the answers of `answer`, as `mode` says.
fn substitute_in<F: FnMut(&Ident, Kind) -> Option<GenericArgument>>(
    answer: F,
    mut args: Vec<&mut GenericArgument>,
    mode: Mode,
) -> std::result::Result<(), EncodeError> {
    let mut answers = Answers::new(answer, matches!(mode, Mode::Impl(_)));
    // The first pass asks for the names; what it writes is dropped.
    let mut asking = SiteFold {
        answers: &mut answers,
        names: FreshLifetimes::default(),
        naming: false,
        mode,
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
        answers: &mut answers,
        names,
        naming: true,
        mode,
        error: None,
    };
    let written: Vec<GenericArgument> = args
        .iter()
        .map(|arg| writing.fold_generic_argument((**arg).clone()))
        .collect();
    // A `Substitute` leaves what does not fit as it is: rustc reports it.
    if let Some(error) = writing.error.filter(|_| matches!(mode, Mode::Args)) {
        return Err(error);
    }
    for (arg, written) in args.iter_mut().zip(written) {
        **arg = written;
    }
    Ok(())
}

/// Writes the arguments of an impl into the items of the copy, as [`substitute_args()`] does.
///
/// [`Referrer::complete_args()`](crate::Referrer::complete_args) returns it. Visit an impl or a
/// trait with it: it writes the items, but not the header (generics, trait path, self type,
/// supertraits), which declares the generics itself. It also leaves an item nested in a body,
/// which has its own scope. A signature or a type can be visited alone.
///
/// It writes the types (also the arguments of the paths through the Repeater: `(T,)` becomes
/// `(u8,)`) and the lifetimes outside types (`&'a self`, `T: 'a`, `where Self: 'a`). It finds a
/// parameter by name and kind, so `'T` and `T` are two parameters. `Self` and other names stay.
///
/// `T::A`, for an argument `V` that is not one identifier, becomes `<V as Trait>::A` with the one
/// trait bound of `T` (inline or in the where clause). Without exactly one bound, it is `<V>::A`,
/// which rustc refuses (E0223). It never fails: what does not fit stays as written, for rustc to
/// report.
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
    /// For `impl Tr<'static, u8> for W` of `trait Tr<'a, K>`: `{ a: 'static }`. Each map is by
    /// the name without `r#`.
    lifetimes: Sorted<HashMap<Ident, GenericArgument>>,
    /// For the same impl: `{ K: u8 }`. The consts are here too.
    others: Sorted<HashMap<Ident, GenericArgument>>,
    /// For `trait Tr<T: Iterator<Item = u8>>`: `{ T: Iterator }`.
    bounds: Sorted<HashMap<Ident, Path>>,
}

impl Substitute {
    /// Substitutes the argument `arg` as a unit.
    fn write(&self, arg: &mut GenericArgument) {
        let answer = |name: &Ident, kind: Kind| {
            let found = match kind {
                Kind::Lifetime => &self.lifetimes,
                Kind::TypeOrConst => &self.others,
            };
            found.get(&name.unraw()).cloned()
        };
        // It leaves what does not fit as it is, and never fails.
        let _ = substitute_in(answer, vec![arg], Mode::Impl(&self.bounds));
    }

    /// Substitutes in `expr`, which starts a statement.
    fn visit_leading_mut(&mut self, expr: &mut Expr) {
        let had_block = leading_block(expr).is_some();
        self.visit_expr_mut(expr);
        parenthesize_new_leading_block(expr, had_block);
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

    // An item in a body has its own scope: the generics of the impl are not its (E0401). A
    // statement that starts with a block ends at it: a const written as one needs parentheses.
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Item(_) => (),
            Stmt::Expr(expr, _) => self.visit_leading_mut(expr),
            _ => syn::visit_mut::visit_stmt_mut(self, stmt),
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
    /// The answers, by kind and name without `r#`.
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
            name.unraw(),
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
    /// What it writes: the arguments of [`substitute_args()`], or the items for a [`Substitute`].
    mode: Mode<'f>,
    /// The first value that did not fit its place, or associated type that could not be written.
    error: Option<EncodeError>,
}

impl<F: FnMut(&Ident, Kind) -> Option<GenericArgument>> SiteFold<'_, F> {
    fn misfit(&mut self, value: GenericArgument) {
        self.error
            .get_or_insert(EncodeError::UnexpectedArgument(Box::new(value)));
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

    /// The type for `name`, the head of a path in an expression. An answer that is no type leaves
    /// the path and is no error (a module `a` may share its name with a lifetime), except for `Self`.
    fn head_type_for(&mut self, name: &Ident) -> Option<Type> {
        if name == "Self" {
            return self.type_for(name);
        }
        match self.answers.of(name, Kind::TypeOrConst)? {
            GenericArgument::Type(ty) => Some(ty),
            _ => None,
        }
    }

    /// The const for `name`, if there is an answer, written as the expression `{ value }`: a const,
    /// or a type that is one identifier (a bare const argument is parsed as a type).
    fn braced_const_for(&mut self, name: &Ident) -> Option<Expr> {
        let answer = self.answers.of(name, Kind::TypeOrConst)?;
        match const_argument(&answer) {
            Some(value) => Some(braced_const(&value)),
            None => {
                self.misfit(answer);
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

/// The first block on the leftmost path of `expr`, if `expr` goes on from it (`{ 3 }` in `{ 3 } * 2`
/// and in `{ 3 }.f()`).
fn leading_block(expr: &mut Expr) -> Option<&mut Expr> {
    let operand = match expr {
        Expr::Binary(ExprBinary { left: operand, .. })
        | Expr::Assign(ExprAssign { left: operand, .. })
        | Expr::Cast(ExprCast { expr: operand, .. })
        | Expr::Index(ExprIndex { expr: operand, .. })
        | Expr::Try(ExprTry { expr: operand, .. })
        | Expr::Call(ExprCall { func: operand, .. })
        | Expr::Field(ExprField { base: operand, .. })
        | Expr::Await(ExprAwait { base: operand, .. })
        | Expr::MethodCall(ExprMethodCall {
            receiver: operand, ..
        })
        | Expr::Range(ExprRange {
            start: Some(operand),
            ..
        }) => &mut **operand,
        _ => return None,
    };
    if matches!(operand, Expr::Block(_)) {
        Some(operand)
    } else {
        leading_block(operand)
    }
}

/// Writes the leading block of `expr` in parentheses (`({ 3 }) * 2`), unless it had one before the
/// substitution: a block that the substitution wrote.
fn parenthesize_new_leading_block(expr: &mut Expr, had_block: bool) {
    if had_block {
        return;
    }
    if let Some(block) = leading_block(expr) {
        let inner = std::mem::replace(block, Expr::Verbatim(TokenStream::new()));
        *block = Expr::Paren(ExprParen {
            attrs: Vec::new(),
            paren_token: Default::default(),
            expr: Box::new(inner),
        });
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
                    let bound = match self.mode {
                        Mode::Impl(bounds) => bounds.get(&name.unraw()).cloned(),
                        Mode::Args => None,
                    };
                    return match projected(self, &name, value, rest, bound) {
                        Ok(projected) => projected,
                        // A `Substitute` writes it, and leaves the failure to rustc.
                        Err(ambiguous) if matches!(self.mode, Mode::Impl(_)) => *ambiguous,
                        // The bounds of `T` are not known: `ty` stays, and the error is kept.
                        Err(ambiguous) => {
                            let error = EncodeError::AmbiguousAssociatedType(ambiguous);
                            self.error.get_or_insert(error);
                            ty
                        }
                    };
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
        // `T::A`: the head of a path is a type in an expression, as `<T>::A` has one.
        if let Expr::Path(ExprPath {
            qself: None, path, ..
        }) = &expr
        {
            if let Some(name) = projection_head(path) {
                if let Some(value) = self.head_type_for(name) {
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
    // A statement and an arm that start with a block end at it: a const written as one needs
    // parentheses.
    fn fold_stmt(&mut self, stmt: Stmt) -> Stmt {
        match stmt {
            Stmt::Expr(mut expr, semi) => {
                let had_block = leading_block(&mut expr).is_some();
                let mut expr = self.fold_expr(expr);
                parenthesize_new_leading_block(&mut expr, had_block);
                Stmt::Expr(expr, semi)
            }
            stmt => syn::fold::fold_stmt(self, stmt),
        }
    }
    fn fold_arm(&mut self, mut arm: Arm) -> Arm {
        let had_block = leading_block(&mut arm.body).is_some();
        let mut arm = syn::fold::fold_arm(self, arm);
        parenthesize_new_leading_block(&mut arm.body, had_block);
        arm
    }
    fn fold_lifetime(&mut self, lifetime: Lifetime) -> Lifetime {
        if is_named(&lifetime.ident, "static")
            || lifetime.ident == "_"
            || self.names.made(&lifetime)
        {
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

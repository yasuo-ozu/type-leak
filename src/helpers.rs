//! Helpers for the macros that use the [`Referrer`]: the pieces of its Repeater impls
//! ([`RepeaterImpl`]) and the arguments of the paths through the Repeater ([`UseSite`]).

use crate::context::{ContextId, SelfRequirement};
use crate::encode::{
    encode_generics_args_with_params, encode_generics_params_with, usize_array_length,
    without_encoded_bounds, EncodeError,
};
use crate::referrer::Referrer;
#[cfg(doc)]
use crate::repeater::repeater_definition;
use crate::util::ReplaceSelf;
use syn::punctuated::Punctuated;
use syn::*;

/// Says what a Repeater impl is written for: see [`Referrer::repeater_impl()`].
///
/// An entry that names `Self` is always implemented for what `Self` stands for. So the choice is
/// mainly about the entries that do not ([`Referrer::context_self()`] is `None`), and about how
/// the others are handled.
#[derive(Clone, Copy, Debug)]
pub enum ImplSelf<'a> {
    /// A type parameter with this name, which the impl declares (`impl<.., __S: ?Sized> .. for
    /// __S`).
    ///
    /// The impl is for any type if the entry does not name `Self`. If the
    /// [`context_self()`](Referrer::context_self) of the entry is [`SelfRequirement::Bound`], the
    /// parameter has the bound of the trait (`__S: ?Sized + Tr<T>`) and replaces `Self`, as
    /// [`Referrer::iter_with_self()`] does. If it is [`SelfRequirement::Type`], the impl is for
    /// that type, with `Self` as written, and no parameter is declared.
    Param(&'a Ident),
    /// This type, usually a marker of the user crate (`impl .. for crate::Marker`).
    ///
    /// Only for an entry that does not name `Self`, else [`EncodeError::NeedsSelf`]. The impl
    /// declares no parameter for it.
    Marker(&'a Type),
}

/// The pieces of the Repeater impl of an entry: [`Referrer::repeater_impl()`].
///
/// Write the impl with them, and with what is yours (the path of the Repeater, its other
/// arguments):
///
/// ```ignore
/// let (impl_generics, _, where_clause) = r.generics.split_for_impl();
/// quote! {
///     impl #impl_generics #repeater<#{&r.args}, #id, #ix> for #{&r.self_ty} #where_clause {
///         type Type = #{&r.ty};
///     }
/// }
/// ```
///
/// [`Referrer::repeater_impl()`] has an example that runs.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepeaterImpl {
    /// The generics that the impl declares, with its where clause.
    ///
    /// These are the generics of the entry without the lifetime bounds that
    /// [`args`](RepeaterImpl::args) encodes ([`Referrer::impl_generics()`]), with `Self` replaced
    /// where it is a parameter. For [`ImplSelf::Param`] they end with the parameter itself (`__S:
    /// ?Sized`, and the bound of the trait for [`SelfRequirement::Bound`]), unless the impl is for
    /// a type. A first where predicate that rustc would read as generics of the `where`
    /// (`<__S>::A: Clone`) is parenthesized.
    pub generics: Generics,
    /// The type that the impl is for.
    pub self_ty: Type,
    /// The first argument of the Repeater: the generics of the entry, encoded.
    ///
    /// These are the tokens that [`Referrer::entry_args()`] writes at a use site that passes every
    /// generic by its own name ([`UseSite::identity()`]).
    pub args: Type,
    /// The leaked type, which is the `Type` of the impl. `Self` is replaced where it is a
    /// parameter.
    pub ty: Type,
    /// The module of the entry, in which the impl is written: [`Referrer::context_module()`].
    pub module: Option<ContextId>,
}

/// Describes the place where the path through the Repeater of an entry is written:
/// [`Referrer::entry_args()`] uses it to write the arguments.
///
/// The default, [`UseSite::identity()`], is the scope of the analysis itself. Say how the generics
/// of the analyzed item are substituted (as in `impl MyTrait<u8> for ..`) with
/// [`UseSite::substituting()`], and what `Self` is with [`UseSite::with_self()`].
#[derive(Clone, Debug)]
pub struct UseSite {
    root_params: Punctuated<GenericParam, Token![,]>,
    completed_args: Vec<GenericArgument>,
    self_ty: Type,
}

impl Default for UseSite {
    fn default() -> Self {
        Self::identity()
    }
}

impl UseSite {
    /// The copy is in the scope of the analysis: every generic is passed by its own name.
    ///
    /// `Self` is `Self`, as in an impl.
    pub fn identity() -> Self {
        Self {
            root_params: Punctuated::new(),
            completed_args: Vec::new(),
            self_ty: parse_quote!(Self),
        }
    }

    /// The generics `root_params` of the analyzed item take the arguments `completed_args`.
    ///
    /// `root_params` are the generics of the item that the Leaker analyzed
    /// (`item_trait.generics.params`). `completed_args` are their arguments at this site, one for
    /// each: the result of [`Referrer::complete_args()`]. A generic of an entry that is named like
    /// a root parameter takes its argument. Any other generic (one of a method) is passed by its
    /// own name, so the copy must declare it. In the closure of `complete_args()`, which gets the
    /// arguments completed so far, `root_params` can still be all the parameters.
    pub fn substituting(
        root_params: &Punctuated<GenericParam, Token![,]>,
        completed_args: &[GenericArgument],
    ) -> Self {
        Self {
            root_params: root_params.clone(),
            completed_args: completed_args.to_vec(),
            ..Self::identity()
        }
    }

    /// Sets what `Self` is at this site, for `Self: 'a` of the entries that name it.
    ///
    /// It is `Self` by default. Give the type that the paths are written for if the site is not
    /// in the impl for it (`__S`, or the implementor written out).
    pub fn with_self(mut self, self_ty: Type) -> Self {
        self.self_ty = self_ty;
        self
    }
}

impl Referrer {
    /// Returns the first argument of the Repeater for the `ix`-th entry, as written at `site`.
    ///
    /// It is the encoding of the generics of the entry ([`Referrer::generics()`]) with the
    /// arguments the site gives them ([`UseSite`]), the lifetime bounds and `Self: 'a` included.
    /// The Repeater impl of the entry has the same encoding for the generics themselves
    /// ([`RepeaterImpl::args`]), so a macro that writes the paths of its copy with this and its
    /// impls with [`Referrer::repeater_impl()`] cannot disagree with itself. `const N: usize` is
    /// `[(); N]`: use [`Referrer::entry_args_with()`] for the other const types.
    ///
    /// It fails with [`EncodeError::MissingArgument`] if `site` gives too few arguments, with
    /// [`EncodeError::UnboundedTraitObject`] if an argument is a trait object without a lifetime
    /// bound where the entry needs one, and with [`EncodeError::UnsupportedConstType`] for a const
    /// parameter that is not `usize`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote! {
    ///     trait Tr<'a, T: 'a, const N: usize> { fn f(&self, x: &'a Wrap<T, N>) -> Rel; }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    /// let ix = referrer.iter().position(|ty| *ty == parse_quote!(&'a Wrap<T, N>)).unwrap();
    ///
    /// // In the scope of the trait, as in a default method: every generic by its own name.
    /// let args = referrer.entry_args(ix, &UseSite::identity()).unwrap();
    /// assert_eq!(args, parse_quote!((&'a (), T, [(); N], &'a T)));
    ///
    /// // In `impl Tr<'static, u8, 3> for Host`: the arguments of the trait.
    /// let trait_args: Vec<GenericArgument> =
    ///     vec![parse_quote!('static), parse_quote!(u8), parse_quote!(3)];
    /// let site = UseSite::substituting(&tr.generics.params, &trait_args);
    /// let args = referrer.entry_args(ix, &site).unwrap();
    /// assert_eq!(args, parse_quote!((&'static (), u8, [(); 3], &'static u8)));
    /// ```
    pub fn entry_args(&self, ix: usize, site: &UseSite) -> std::result::Result<Type, EncodeError> {
        self.entry_args_with(ix, site, usize_array_length)
    }

    /// Like [`Referrer::entry_args()`], with `const_fn` encoding the const arguments.
    ///
    /// `const_fn(value, param)` returns the element for the const parameter `param` with the
    /// argument `value`, as the site writes it. It is the closure of
    /// [`Referrer::repeater_impl_with()`] too, which calls it with the parameter itself (`N`) as
    /// `value`, so one closure serves both sides. For the slots of [`repeater_definition()`] it is
    /// `|value, param| slots.encode_const(value, &param.ty)`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote! { trait Tr<const B: bool> { fn f(&self) -> Wrap<B>; } };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// let trait_args: Vec<GenericArgument> = vec![parse_quote!(true)];
    /// let args = referrer
    ///     .entry_args_with(
    ///         0,
    ///         &UseSite::substituting(&tr.generics.params, &trait_args),
    ///         |value, param| slots.encode_const(value, &param.ty),
    ///     )
    ///     .unwrap();
    /// assert_eq!(args, parse_quote!((*const dyn crate::Repeater<(), 0, 0, true, Type = ()>,)));
    /// ```
    pub fn entry_args_with(
        &self,
        ix: usize,
        site: &UseSite,
        const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
    ) -> std::result::Result<Type, EncodeError> {
        let args = self.generic_args(ix, &site.root_params, &site.completed_args)?;
        encode_generics_args_with_params(
            &args,
            &self.generics(ix).params,
            Some((&site.self_ty, self.context_self_outlives(self.contexts[ix]))),
            const_fn,
        )
    }

    /// Describes the Repeater impl of the `ix`-th entry: its generics, trait arguments and type.
    ///
    /// `target` says what the impl is for ([`ImplSelf`]). The result is what
    /// [`Referrer::iter_with_self()`], [`without_encoded_bounds()`] and
    /// [`Referrer::impl_generics()`] give together, with the encoding of the generics and the
    /// declaration of the parameter for `Self`, which every macro that emits the impls writes by
    /// hand. `const N: usize` is `[(); N]`: use [`Referrer::repeater_impl_with()`] for the other
    /// const types. Write the impl in [`RepeaterImpl::module`] if it is `Some`.
    ///
    /// It fails with [`EncodeError::NeedsSelf`] for [`ImplSelf::Marker`] if the entry names
    /// `Self`, and with [`EncodeError::UnsupportedConstType`] for a const parameter that is not
    /// `usize`. [`Referrer::required()`] skips the optional entries, which need no impl.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote! {
    ///     trait Tr<'a, T: 'a> {
    ///         type Out;
    ///         fn f(&self, x: &'a Wrap<T>) -> Rel;
    ///         fn g(&self) -> Wrap2<Self::Out>;
    ///     }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    ///
    /// let self_param: Ident = parse_quote!(__S);
    /// let repeater: Path = parse_quote!(crate::Repeater);
    /// let impls: Vec<ItemImpl> = referrer
    ///     .required()
    ///     .map(|(ix, _)| {
    ///         let r = referrer.repeater_impl(ix, ImplSelf::Param(&self_param)).unwrap();
    ///         let (impl_generics, _, where_clause) = r.generics.split_for_impl();
    ///         let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
    ///         parse_quote! {
    ///             impl #impl_generics #repeater<#args, 7u64, #ix> for #self_ty #where_clause {
    ///                 type Type = #ty;
    ///             }
    ///         }
    ///     })
    ///     .collect();
    /// // `Rel` uses no generics, so any type can be the implementor.
    /// let rel: ItemImpl = parse_quote! {
    ///     impl<__S: ?::core::marker::Sized> crate::Repeater<(), 7u64, 1usize> for __S {
    ///         type Type = Rel;
    ///     }
    /// };
    /// // `Wrap2<Self::Out>` names `Self`: the implementor is bound by the trait, and replaces it.
    /// let wrap2: ItemImpl = parse_quote! {
    ///     impl<'a, T, __S: ?::core::marker::Sized + Tr<'a, T>>
    ///         crate::Repeater<(&'a (), T, &'a T), 7u64, 2usize> for __S
    ///     {
    ///         type Type = Wrap2<<__S>::Out>;
    ///     }
    /// };
    /// assert!(impls.contains(&rel));
    /// assert!(impls.contains(&wrap2));
    /// ```
    pub fn repeater_impl(
        &self,
        ix: usize,
        target: ImplSelf<'_>,
    ) -> std::result::Result<RepeaterImpl, EncodeError> {
        self.repeater_impl_with(ix, target, usize_array_length)
    }

    /// Like [`Referrer::repeater_impl()`], with `const_fn` encoding the const parameters.
    ///
    /// `const_fn(value, param)` is called with the parameter itself (`N`) as `value`, where
    /// [`Referrer::entry_args_with()`] calls it with the argument. For the slots of
    /// [`repeater_definition()`] it is `|value, param| slots.encode_const(value, &param.ty)`.
    pub fn repeater_impl_with(
        &self,
        ix: usize,
        target: ImplSelf<'_>,
        mut const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
    ) -> std::result::Result<RepeaterImpl, EncodeError> {
        use syn::fold::Fold;
        let context = self.contexts[ix];
        // The type the impl is for, `Self` as the impl writes it, the parameter to declare and its
        // bound.
        let (self_ty, self_in_impl, param, bound): (Type, Type, Option<&Ident>, Option<&Path>) =
            match (self.context_self(context), target) {
                (Some(SelfRequirement::Type(ty)), ImplSelf::Param(_)) => {
                    (ty.clone(), parse_quote!(Self), None, None)
                }
                (Some(_), ImplSelf::Marker(_)) => {
                    return Err(EncodeError::NeedsSelf(Box::new(
                        self.leak_types[ix].clone(),
                    )))
                }
                (Some(SelfRequirement::Bound(path)), ImplSelf::Param(ident)) => (
                    parse_quote!(#ident),
                    parse_quote!(#ident),
                    Some(ident),
                    Some(path),
                ),
                (None, ImplSelf::Param(ident)) => (
                    parse_quote!(#ident),
                    parse_quote!(#ident),
                    Some(ident),
                    None,
                ),
                (None, ImplSelf::Marker(marker)) => (marker.clone(), marker.clone(), None, None),
            };
        let generics = self.context_generics(context);
        let (ty, generics) = match self.context_self(context) {
            // The implementor is a parameter, where `Self` of the trait is not available.
            Some(SelfRequirement::Bound(_)) => {
                let mut folder = ReplaceSelf {
                    self_ty: &self_in_impl,
                };
                (
                    folder.fold_type(self.leak_types[ix].clone()),
                    folder.fold_generics(generics.clone()),
                )
            }
            _ => (self.leak_types[ix].clone(), generics.clone()),
        };
        let args = encode_generics_params_with(
            &generics.params,
            Some((&self_in_impl, self.context_self_outlives(context))),
            |param| {
                let ident = &param.ident;
                const_fn(&parse_quote!(#ident), param)
            },
        )?;
        let mut declared = without_encoded_bounds(&generics);
        if let Some(ident) = param {
            declared.params.push(match bound {
                Some(path) => parse_quote!(#ident: ?::core::marker::Sized + #path),
                None => parse_quote!(#ident: ?::core::marker::Sized),
            });
        }
        parenthesize_leading_qualified_path(&mut declared);
        Ok(RepeaterImpl {
            generics: declared,
            self_ty,
            args,
            ty,
            module: self.context_module(context),
        })
    }
}

/// rustc reads `where <__S>::A: Clone` as generics of the `where` (reserved for future use), so a
/// first predicate on `<__S>::A` is parenthesized.
fn parenthesize_leading_qualified_path(generics: &mut Generics) {
    let first = generics
        .where_clause
        .as_mut()
        .and_then(|clause| clause.predicates.first_mut());
    if let Some(WherePredicate::Type(PredicateType {
        lifetimes: None,
        bounded_ty,
        ..
    })) = first
    {
        if matches!(bounded_ty, Type::Path(TypePath { qself: Some(q), .. }) if q.as_token.is_none())
        {
            *bounded_ty = Type::Paren(TypeParen {
                paren_token: Default::default(),
                elem: Box::new(bounded_ty.clone()),
            });
        }
    }
}

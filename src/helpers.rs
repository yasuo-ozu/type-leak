//! Helpers for the macros that use the [`Referrer`]: the pieces of its Repeater impls
//! ([`RepeaterImpl`]) and the arguments of the paths through the Repeater ([`UseSite`]).

use crate::context::{ContextId, SelfRequirement};
use crate::encode::{
    encode_generics_args_with_params, encode_generics_params_with, without_encoded_bounds,
    EncodeError, SubstituteParams, SubstituteSite,
};
use crate::referrer::Referrer;
#[cfg(doc)]
use crate::repeater::repeater_definition;
use crate::util::{first_self_in_generics, ReplaceSelf};
use proc_macro2::Span;
use syn::fold::Fold;
use syn::punctuated::Punctuated;
use syn::*;

/// What a Repeater impl is written for: the `target` of [`Referrer::repeater_impl()`].
///
/// An entry that [needs `Self`](Referrer::needs_self) is always implemented for what `Self` stands
/// for, so the choice only matters for the others.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum ImplSelf<'a> {
    /// A type parameter that the impl declares (`impl<.., __S: ?Sized> .. for __S`).
    ///
    /// It replaces `Self` where the entry names it, and has the bound of the trait for
    /// [`SelfRequirement::Bound`]. For [`SelfRequirement::Type`] the impl is for that type, and
    /// declares none.
    Param(&'a Ident),
    /// This type, usually a marker of your crate. The impl declares no parameter.
    ///
    /// Fails with [`EncodeError::NeedsSelf`] if the entry needs `Self`.
    Marker(&'a Type),
    /// One of 17 primitive types, chosen by `id` and the entry; the entries that need `Self` get
    /// [`ImplSelf::Param`] with `param`.
    ///
    /// Cheaper than a blanket impl when there are many entries. The use site must write
    /// [`bucket_self(ix, id)`](Referrer::bucket_self) as the type of the path through the Repeater
    /// (what `Self` stands for if it is `None`), or E0277. `id` is the Repeater's `ID`, or any
    /// number that both macros know.
    Bucket {
        /// The number the type depends on.
        id: u64,
        /// The parameter for the entries that need `Self`.
        param: &'a Ident,
    },
}

/// The pieces of the Repeater impl of an entry: [`Referrer::repeater_impl()`].
///
/// Write the impl from them and your own parts (the path of the Repeater, its other arguments),
/// in [`module`](RepeaterImpl::module) if it is `Some`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepeaterImpl {
    /// The generics that the impl declares, with its where clause.
    ///
    /// The lifetimes come first, then the types (the parameter for `Self` last among them), then the
    /// consts: the order that rustc before 1.59 requires.
    pub generics: Generics,
    /// The type that the impl is for.
    pub self_ty: Type,
    /// The first argument of the Repeater: what [`Referrer::entry_args()`] writes for
    /// [`UseSite::identity()`], but with the self type of the impl for `Self` (`&'a __S`, not
    /// `&'a Self`, for `Self: 'a`).
    pub args: Type,
    /// The leaked type, which is the `Type` of the impl.
    pub ty: Type,
    /// The module in which to write the impl ([`Referrer::context_module()`]).
    pub module: Option<ContextId>,
}

/// Where the path through the Repeater of an entry is written: the `site` of
/// [`Referrer::entry_args()`].
///
/// The default, [`UseSite::identity()`], is the scope of the analysis. Use
/// [`UseSite::substituting()`] if the generics of the analyzed item get arguments
/// (`impl Tr<u8> for ..`), and [`UseSite::with_self()`] to say what `Self` is.
#[derive(Clone, Debug)]
#[must_use]
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
    /// Every generic is passed by its own name, and `Self` is `Self`: the scope of the analysis.
    pub fn identity() -> Self {
        Self {
            root_params: Punctuated::new(),
            completed_args: Vec::new(),
            self_ty: parse_quote!(Self),
        }
    }

    /// The generics `root_params` of the analyzed item take the arguments `completed_args`, one for
    /// each ([`Referrer::complete_args()`]).
    ///
    /// A generic of an entry that no root parameter names (a method's) is passed by its own name,
    /// so the copy must declare it. Nothing is checked here: [`Referrer::entry_args()`] fails if the
    /// arguments do not fit the parameters by position and kind.
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

    /// Sets what `Self` is at this site, for the `Self: 'a` of the entries that name it.
    ///
    /// Give the type the paths are written for if the site is not in the impl for it. The default
    /// is `Self`.
    pub fn with_self(mut self, self_ty: Type) -> Self {
        self.self_ty = self_ty;
        self
    }

    /// Writes `ty` as it is at this site: the generics of the analyzed item replaced by their
    /// arguments, and `Self` by [`with_self()`](UseSite::with_self).
    ///
    /// For the parts of a copy that are not entries, which [`Referrer::visitor()`] leaves as
    /// written (`t: T` in `impl Tr<u8> for ..`). A trait object with several bounds that a reference
    /// or pointer points to gets the parentheses it needs (`&'a (dyn Debug + 'a)`). A generic with
    /// no argument stays as it is, and `Self` in a macro is replaced, other tokens of it are not.
    ///
    /// A function pointer or `Fn(..)` whose output takes an elided lifetime from the one lifetime of
    /// its inputs keeps it where an argument adds lifetimes to them: the elided lifetime gets a
    /// `for<'__tl_s0>` binder (`fn(&U) -> &u8` with `&'static str` is `for<'__tl_s0>
    /// fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8`). A lifetime that a path hides cannot be named.
    #[must_use]
    pub fn substitute(&self, ty: &Type) -> Type {
        let is_self = matches!(&self.self_ty, Type::Path(TypePath { qself: None, path }) if path.is_ident("Self"));
        SubstituteSite::new(
            SubstituteParams::of(&self.root_params, &self.completed_args),
            (!is_self).then(|| &self.self_ty),
            ty,
        )
        .fold_type(ty.clone())
    }
}

impl Referrer {
    /// Returns the first argument of the Repeater for the `ix`-th entry, as written at `site`.
    ///
    /// It is [`RepeaterImpl::args`] with the arguments the site gives the generics.
    /// `const_fn(value, param)` encodes the const parameter `param` with the argument `value`: pass
    /// [`array_const`](crate::array_const), or for the slots of [`repeater_definition()`]
    /// [`RepeaterSlots::const_fn()`](crate::RepeaterSlots::const_fn).
    ///
    /// Fails with [`EncodeError::MissingArgument`] if `site` gives too few arguments, with
    /// [`EncodeError::UnboundedTraitObject`] if an argument is a trait object without the lifetime
    /// bound the entry needs, and with the error of `const_fn`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote! {
    ///     trait Tr<'a, T: 'a, const N: usize> { fn f(&self, x: &'a Wrap<T, N>) -> Rel; }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true);
    /// let referrer = leaker.result().finish().unwrap();
    /// let ix = referrer.iter().position(|ty| *ty == parse_quote!(&'a Wrap<T, N>)).unwrap();
    ///
    /// // In the scope of the trait, as in a default method: every generic by its own name.
    /// let args = referrer.entry_args(ix, &UseSite::identity(), array_const).unwrap();
    /// assert_eq!(args, parse_quote!((&'a (), T, [(); N], &'a T)));
    ///
    /// // In `impl Tr<'static, u8, 3> for Host`: the arguments of the trait.
    /// let trait_args: Vec<GenericArgument> =
    ///     vec![parse_quote!('static), parse_quote!(u8), parse_quote!(3)];
    /// let site = UseSite::substituting(&tr.generics.params, &trait_args);
    /// let args = referrer.entry_args(ix, &site, array_const).unwrap();
    /// assert_eq!(args, parse_quote!((&'static (), u8, [(); 3], &'static u8)));
    /// ```
    pub fn entry_args(
        &self,
        ix: usize,
        site: &UseSite,
        const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
    ) -> std::result::Result<Type, EncodeError> {
        let context = self.contexts[ix];
        let args = self.context_generic_args(context, &site.root_params, &site.completed_args)?;
        encode_generics_args_with_params(
            &args,
            &self.context_generics(context).params,
            Some((&site.self_ty, self.context_self_outlives(context))),
            const_fn,
        )
    }

    /// Returns `true` if the `ix`-th entry needs what `Self` stands for.
    ///
    /// That is so if it names `Self`, or its generics do (`trait Tr<T: AsRef<Self>, U = Rel<T>>`),
    /// even if it is only in `Self: 'a` (`trait Tr<'a> where Self: 'a`).
    /// Then [`ImplSelf::Marker`] fails and [`Referrer::bucket_self()`] is `None`: the impl is for
    /// [`ImplSelf::Param`], and the use site writes what `Self` stands for (the host or the type).
    pub fn needs_self(&self, ix: usize) -> bool {
        let context = self.contexts[ix];
        self.context_self(context).is_some()
            || first_self_in_generics(self.context_generics(context)).is_some()
    }

    /// Returns the primitive type that the `ix`-th entry's impl is for under [`ImplSelf::Bucket`]
    /// with `id`.
    ///
    /// It is `None` if the entry [needs `Self`](Referrer::needs_self): write what `Self` stands for
    /// there.
    #[must_use]
    pub fn bucket_self(&self, ix: usize, id: u64) -> Option<Type> {
        (!self.needs_self(ix)).then(|| bucket_type(id, ix))
    }

    /// Describes the Repeater impl of the `ix`-th entry: its generics, trait arguments and type.
    ///
    /// `target` says what the impl is for ([`ImplSelf`]); `const_fn` is as in
    /// [`Referrer::entry_args()`], but gets the parameter itself (`N`) as `value`. Write the impl
    /// in [`RepeaterImpl::module`] if it is `Some`, and one for every entry ([`Referrer::iter()`]):
    /// optional entries need one if some Referrer does not trust them
    /// ([`LeakerResult::allow()`](crate::LeakerResult::allow)).
    ///
    /// Fails with [`EncodeError::NeedsSelf`] for [`ImplSelf::Marker`] if the entry
    /// [needs `Self`](Referrer::needs_self), and with the error of `const_fn`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote! {
    ///     trait Tr<T> { type Out; fn f(&self) -> Wrap<T, Self::Out>; }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true);
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// let param: Ident = parse_quote!(__S);
    /// let r = referrer.repeater_impl(0, ImplSelf::Param(&param), slots.const_fn()).unwrap();
    /// let (impl_generics, _, where_clause) = r.generics.split_for_impl();
    /// let (self_ty, ty) = (&r.self_ty, &r.ty);
    /// let path = slots.trait_path(&r.args, 7, 0);
    /// let imp: ItemImpl = parse_quote! {
    ///     impl #impl_generics #path for #self_ty #where_clause {
    ///         type Type = #ty;
    ///     }
    /// };
    /// // `Self::Out` names `Self`: the implementor is bound by the trait, and replaces it.
    /// let expected: ItemImpl = parse_quote! {
    ///     impl<T, __S: ?::core::marker::Sized + Tr<T>>
    ///         crate::Repeater<(T,), 7u64, 0usize> for __S
    ///     {
    ///         type Type = Wrap<T, <__S>::Out>;
    ///     }
    /// };
    /// assert_eq!(imp, expected);
    /// ```
    pub fn repeater_impl(
        &self,
        ix: usize,
        target: ImplSelf<'_>,
        mut const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
    ) -> std::result::Result<RepeaterImpl, EncodeError> {
        let context = self.contexts[ix];
        let bucket;
        // An entry that needs `Self` has none for a marker, even if `context_self()` is `None`: the
        // `Self` of its generics would be the marker.
        if matches!(target, ImplSelf::Marker(_)) && self.needs_self(ix) {
            return Err(EncodeError::NeedsSelf(Box::new(
                self.leak_types[ix].clone(),
            )));
        }
        let target = match target {
            ImplSelf::Bucket { id, param } => match self.bucket_self(ix, id) {
                Some(ty) => {
                    bucket = ty;
                    ImplSelf::Marker(&bucket)
                }
                None => ImplSelf::Param(param),
            },
            target => target,
        };
        // The type the impl is for, `Self` as the impl writes it, the parameter to declare and its
        // bound.
        let (self_ty, self_in_impl, param, bound): (Type, Type, Option<&Ident>, Option<&Path>) =
            match (self.context_self(context), target) {
                (Some(SelfRequirement::Type(ty)), ImplSelf::Param(_)) => {
                    (ty.clone(), parse_quote!(Self), None, None)
                }
                (Some(_), ImplSelf::Marker(_)) => unreachable!("rejected above"),
                (Some(SelfRequirement::Bound(path)), ImplSelf::Param(ident)) => (
                    parse_quote!(#ident),
                    parse_quote!(#ident),
                    Some(ident),
                    Some(path),
                ),
                // Any type that the bounds hold for: the parameter has no bound of its own.
                (Some(SelfRequirement::Any), ImplSelf::Param(ident)) => (
                    parse_quote!(#ident),
                    parse_quote!(#ident),
                    Some(ident),
                    None,
                ),
                (None, ImplSelf::Param(ident)) => (
                    parse_quote!(#ident),
                    parse_quote!(#ident),
                    Some(ident),
                    None,
                ),
                (None, ImplSelf::Marker(marker)) => (marker.clone(), marker.clone(), None, None),
                (_, ImplSelf::Bucket { .. }) => unreachable!("resolved above"),
            };
        let generics = self.context_generics(context);
        let (ty, generics) = match self.context_self(context) {
            // The implementor is a parameter, where `Self` of the trait is not available.
            Some(SelfRequirement::Bound(_) | SelfRequirement::Any) => {
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
            // The `Self` of a struct, an enum or a union is a `Sized` type, and the bounds may need
            // it (`(Self, u8)`, `Vec<Self>`), so the parameter of `SelfRequirement::Any` is
            // `Sized` where they do. Else it is `?Sized`: rustc before 1.70 cannot prove `S<T>:
            // Sized` for the copy of a struct whose last field is the path through this impl, if
            // the impl needs `__S: Sized` (E0275: the type needs the impl, which needs the type).
            let sized = matches!(self.context_self(context), Some(SelfRequirement::Any))
                && needs_sized(&declared, &ty, ident);
            let declaration = match (bound, sized) {
                (Some(path), _) => parse_quote!(#ident: ?::core::marker::Sized + #path),
                (None, true) => parse_quote!(#ident),
                (None, false) => parse_quote!(#ident: ?::core::marker::Sized),
            };
            // Before the first const parameter: rustc before 1.59 wants the types before the consts.
            let at = declared
                .params
                .iter()
                .position(|param| matches!(param, GenericParam::Const(_)))
                .unwrap_or(declared.params.len());
            declared.params.insert(at, declaration);
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

/// The primitive type of the entry `ix` for `id`: 17 of them, chosen by a hash so that ids and
/// indices that are close spread. `ImplSelf::Bucket` of the Leaker's macro and the paths of the
/// Referrer's must agree on it, so it is part of [`Referrer::FORMAT_VERSION`].
fn bucket_type(id: u64, ix: usize) -> Type {
    const BUCKETS: [&str; 17] = [
        "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
        "i128", "isize", "f32", "f64", "str",
    ];
    // The finalizer of splitmix64.
    let mut z = id
        .wrapping_add((ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let name = Ident::new(
        BUCKETS[(z % BUCKETS.len() as u64) as usize],
        Span::call_site(),
    );
    parse_quote!(::core::primitive::#name)
}

/// Returns `true` if the impl may need `param: Sized`, and `false` if it is well-formed without.
///
/// `declared` are the generics of the impl without `param`, in which `param` replaced `Self`, and
/// `ty` is the type of the impl. A type that names `Self` (`Rel2<Self, T>`, with
/// [`nested_self_entries()`](crate::Options::nested_self_entries)) needs it as
/// the bounds do: `Rel2<__S, T>` is well-formed for `__S: Sized` only, unless the declaration of
/// `Rel2` relaxes it, which the Leaker cannot see. They need no `Sized` if every use of `param` is
///
/// - a type argument of its own, directly, of a path that accepts an unsized one
///   ([`accepts_unsized_argument()`]: `AsRef<__S>`, `Box<__S>`),
/// - behind a reference or a raw pointer (`&'a __S`, `*const __S`), or
/// - the type of a predicate that does not say `Sized` (`__S: 'a`).
///
/// Any other use may need it: a tuple element (`(__S, u8)`), the element of an array or a slice,
/// the argument of a type or a trait of the user (`Vec<__S>`, `Tr<__S>`), an associated type
/// (`Iterator<Item = __S>`, `<__S>::A`), the argument of an `Fn` bound, a macro. So this errs on
/// the side of `Sized`.
fn needs_sized(declared: &Generics, ty: &Type, param: &Ident) -> bool {
    use syn::visit::Visit;
    struct Uses<'a> {
        param: &'a Ident,
        needs_sized: bool,
    }
    impl Uses<'_> {
        fn is_param(&self, ty: &Type) -> bool {
            match ty {
                Type::Path(TypePath { qself: None, path }) => path.is_ident(self.param),
                Type::Paren(TypeParen { elem, .. }) => self.is_param(elem),
                Type::Group(TypeGroup { elem, .. }) => self.is_param(elem),
                _ => false,
            }
        }
    }
    impl<'ast> Visit<'ast> for Uses<'_> {
        // The uses that need `Sized` are those that are not skipped below.
        fn visit_ident(&mut self, ident: &'ast Ident) {
            if ident == self.param {
                self.needs_sized = true;
            }
        }
        // Tokens that are not parsed: any `Self` is out of reach.
        fn visit_macro(&mut self, _: &'ast Macro) {
            self.needs_sized = true;
        }
        fn visit_type_reference(&mut self, node: &'ast TypeReference) {
            if !self.is_param(&node.elem) {
                syn::visit::visit_type_reference(self, node);
            }
        }
        fn visit_type_ptr(&mut self, node: &'ast TypePtr) {
            if !self.is_param(&node.elem) {
                syn::visit::visit_type_ptr(self, node);
            }
        }
        fn visit_path(&mut self, path: &'ast Path) {
            if !accepts_unsized_argument(path) {
                return syn::visit::visit_path(self, path);
            }
            for segment in &path.segments {
                match &segment.arguments {
                    PathArguments::AngleBracketed(arguments) => {
                        for argument in &arguments.args {
                            match argument {
                                GenericArgument::Type(ty) if self.is_param(ty) => (),
                                argument => self.visit_generic_argument(argument),
                            }
                        }
                    }
                    arguments => self.visit_path_arguments(arguments),
                }
            }
        }
        fn visit_predicate_type(&mut self, node: &'ast PredicateType) {
            if !self.is_param(&node.bounded_ty) {
                return syn::visit::visit_predicate_type(self, node);
            }
            for bound in &node.bounds {
                match bound {
                    // `__S: Sized`
                    TypeParamBound::Trait(TraitBound {
                        modifier: TraitBoundModifier::None,
                        path,
                        ..
                    }) if path.segments.last().map_or(false, |s| s.ident == "Sized") => {
                        self.needs_sized = true
                    }
                    bound => self.visit_type_param_bound(bound),
                }
            }
        }
    }
    let mut uses = Uses {
        param,
        needs_sized: false,
    };
    uses.visit_generics(declared);
    uses.visit_type(ty);
    uses.needs_sized
}

/// Returns `true` if `path` is a trait or a type of `core`, `std` or `alloc` whose type parameter
/// is `?Sized`: `AsRef<__S>` is well-formed with `__S: ?Sized`.
///
/// It is written with an absolute path (`::core::convert::AsRef`), or with the name of the prelude
/// (`AsRef`, `Box`), which is taken for what it names there: the Leaker accepts it only if the
/// prelude is allowed. A module that declares an item of the same name is not told apart (the impl
/// would then not compile in the Leaker's crate).
fn accepts_unsized_argument(path: &Path) -> bool {
    // (the crates, the module, the items): `::core::convert::AsRef`, `::std::convert::AsRef`..
    const ABSOLUTE: &[(&[&str], &str, &[&str])] = &[
        (&["core", "std"], "convert", &["AsRef", "AsMut"]),
        (
            &["core", "std", "alloc"],
            "borrow",
            &["Borrow", "BorrowMut"],
        ),
        (&["core", "std"], "cmp", &["PartialEq", "PartialOrd"]),
        (&["core", "std"], "marker", &["PhantomData"]),
        (&["std", "alloc"], "boxed", &["Box"]),
        (&["std", "alloc"], "rc", &["Rc"]),
        (&["std", "alloc"], "sync", &["Arc"]),
    ];
    const PRELUDE: &[&str] = &["AsRef", "AsMut", "PartialEq", "PartialOrd", "Box"];
    let names: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    let names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    match (path.leading_colon.is_some(), &names[..]) {
        (false, [name]) => PRELUDE.contains(name),
        (true, [root, module, item]) => ABSOLUTE
            .iter()
            .any(|(roots, m, items)| roots.contains(root) && m == module && items.contains(item)),
        _ => false,
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

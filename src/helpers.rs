//! Helpers for the macros that use the [`Referrer`]: the pieces of its Repeater impls
//! ([`RepeaterImpl`]) and the arguments of the paths through the Repeater ([`UseSite`]).

use crate::context::ModuleScope;
#[cfg(doc)]
use crate::context::SelfRequirement;
use crate::encode::{check_root_args, param_as_arg, EncodeError, SubstituteParams, SubstituteSite};
#[cfg(doc)]
use crate::referrer::Referrer;
use crate::util::generic_param_name;
use syn::fold::Fold;
use syn::punctuated::Punctuated;
use syn::*;

/// What a Repeater impl is written for: the `target` of [`Referrer::repeater_impl()`].
///
/// An entry that names `Self`, or whose context's generics do, is always implemented for what
/// `Self` stands for, so the choice only matters for the others.
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
    /// (what `Self` stands for if it is `None`), or E0277. `id` is the `id` of
    /// [`repeater_key()`](crate::repeater_key), or any number that both macros know.
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
    /// The module in which to write the impl, if the entry is in one.
    ///
    /// It is the [`ModuleScope::Itself`] of that module, which tells where the module is. It is
    /// `None` if the entry is in no module.
    ///
    /// ```ignore
    /// // In crate a, analyzed with `intern_mod()`: the entry `Rel`.
    /// pub mod outer {
    ///     pub mod inner {
    ///         pub struct S { a: Rel }
    ///     }
    /// }
    ///
    /// // `module` of the impl of `Rel`, which is in `inner` (context 1):
    /// Some(Itself { id: 1, path: Some([inner]), hint: None, parent: Some(0) })
    /// ```
    pub module: Option<ModuleScope>,
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
    pub(crate) self_ty: Type,
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
    /// so the copy must declare it. Nothing is checked here: [`UseSite::generic_args()`] and
    /// [`Referrer::entry_args()`] fail if the arguments do not fit the parameters by position and
    /// kind.
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

    /// Returns the arguments for `generics`, as written at this site.
    ///
    /// `generics` are those of an entry (`Context::generics` in [`Referrer::iter()`]). The site's
    /// arguments are those of its root parameters ([`UseSite::substituting()`]), by position. A
    /// generic named like one of the parameters takes its argument, and any other is passed by its
    /// own name, so the copy must declare it. `Self` is not an argument.
    ///
    /// Fails with [`EncodeError::UnexpectedArgument`] if there are more arguments than parameters,
    /// or one is not of the kind of its parameter (a lifetime, a type, a const: a binding such as
    /// `Item = u8` is none), and with [`EncodeError::MissingArgument`] if the generics need an
    /// argument that is not there. The arguments may stop early, as `before` does, which
    /// [`Referrer::complete_args()`] gives `type_fn`. For the trait argument of the Repeater use
    /// [`Referrer::entry_args()`], which also encodes lifetime bounds, `Self: 'a` and consts.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait Tr<T, U> { fn f<V>(&self) -> Rel<T, V>; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// // The entry `Rel<T, V>` has the generics `<T, V>`. In `impl Tr<u8, u16> for X`, `T` is
    /// // `u8`, and the method's own `V` keeps its name.
    /// let (_, context) = referrer.iter().next().unwrap();
    /// let params = &item.generics.params;
    /// let completed: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(u16)];
    /// let site = UseSite::substituting(params, &completed);
    /// let args = site.generic_args(&context.generics).unwrap();
    /// let expected: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(V)];
    /// assert_eq!(args, expected);
    ///
    /// // An argument of the wrong kind is an error.
    /// let wrong: Vec<GenericArgument> = vec![parse_quote!('a), parse_quote!(u16)];
    /// let site = UseSite::substituting(params, &wrong);
    /// assert!(site.generic_args(&context.generics).is_err());
    /// ```
    pub fn generic_args(
        &self,
        generics: &Generics,
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        check_root_args(&self.root_params, &self.completed_args)?;
        generics
            .params
            .iter()
            .map(|param| {
                let name = generic_param_name(param);
                match self
                    .root_params
                    .iter()
                    .position(|root| generic_param_name(root) == name)
                {
                    Some(pos) => self
                        .completed_args
                        .get(pos)
                        .cloned()
                        .ok_or_else(|| EncodeError::MissingArgument(Box::new(param.clone()))),
                    None => Ok(param_as_arg(param)),
                }
            })
            .collect()
    }
}

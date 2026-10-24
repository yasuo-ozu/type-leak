//! The definition of the Repeater trait, and encoding generics into its const slots.

use crate::encode::{
    const_param_primitive, encode_generics_args_with_params, encode_generics_params_with,
    normalize_const_arg, EncodeError,
};
use proc_macro2::Span;
use syn::punctuated::Punctuated;
use syn::*;

/// Const parameter types with a slot, in slot order. `usize` uses `[(); N]`, and `u64` uses `ID`.
const REPEATER_SLOTS: &[(&str, &str, &str)] = &[
    ("bool", "BOOL", "false"),
    ("char", "CHAR", "'\\0'"),
    ("u8", "U8", "0"),
    ("u16", "U16", "0"),
    ("u32", "U32", "0"),
    ("u128", "U128", "0"),
    ("i8", "I8", "0"),
    ("i16", "I16", "0"),
    ("i32", "I32", "0"),
    ("i64", "I64", "0"),
    ("i128", "I128", "0"),
    ("isize", "ISIZE", "0"),
];

/// Returns the definition of the Repeater trait, the one trait that both crates see. It supports
/// const generics of any primitive type.
///
/// Define it once, at the root of the crate that writes the Repeater impls (the crate that
/// exports the item). It must be a trait of that crate, because the impls are for any `Self`,
/// which the orphan rule allows only for a local trait. So it is not in the crate of the
/// proc macros (which exports only macros), and the crate that copies the item names it through
/// the crate that defines it. Emit it from a macro, e.g. a function-like proc macro
/// `define_repeater!()` whose body is `quote!(#definition)`.
///
/// `T` is the encoding of the generics of an entry, `ID` the Leaker (the trait that was analyzed)
/// and `NTH` the entry. Besides them, the trait has a defaulted const slot per primitive type:
/// `BOOL`, `CHAR`, `U8`, `U16`, `U32`, `U128`, `I8`, `I16`, `I32`, `I64`, `I128` and `ISIZE`.
/// [`RepeaterSlots`] encodes const values into them, and `Repeater<T, ID, NTH>` still works.
///
/// A const parameter is treated like a type parameter, and the Repeater needs a type that carries
/// its value. `const N: usize` is encoded as `[(); N]` (see
/// [`encode_generics_params_to_ty()`](crate::encode_generics_params_to_ty)), a `u64` value uses the
/// `ID` slot, and any other primitive type uses its slot. Without the slots, the Repeater works if
/// every const parameter is `usize` or `u64`. In that case you can write the trait by hand instead:
/// `pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> { type Type: ?Sized; }`.
/// The generated trait needs Rust 1.59 and the 2018 edition, while `type-leak` itself supports
/// 1.56.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let def = repeater_definition(&parse_quote!(pub), &parse_quote!(Repeater));
/// // `T`, `ID`, `NTH` and a slot for each primitive const type.
/// assert_eq!(def.generics.params.len(), 3 + 12);
/// assert_eq!(def.ident, "Repeater");
/// // The trait is an `ItemTrait`: write it with `ToTokens`.
/// # use template_quote::ToTokens;
/// assert!(def.to_token_stream().to_string().starts_with("pub trait Repeater <"));
/// ```
pub fn repeater_definition(vis: &Visibility, ident: &Ident) -> ItemTrait {
    let slots = REPEATER_SLOTS
        .iter()
        .map(|(ty, name, default)| -> GenericParam {
            let ty = Ident::new(ty, Span::call_site());
            let name = Ident::new(name, Span::call_site());
            let default: Expr = parse_str(default).unwrap();
            parse_quote!(const #name: ::core::primitive::#ty = #default)
        });
    parse_quote! {
        #vis trait #ident<
            T: ?::core::marker::Sized,
            const ID: ::core::primitive::u64,
            const NTH: ::core::primitive::usize,
            #(#slots),*
        > {
            type Type: ?::core::marker::Sized;
        }
    }
}

/// Encodes the generics of an entry into the first argument of the Repeater, using the const
/// slots of the Repeater from [`repeater_definition()`].
///
/// It encodes like `encode_generics_params_to_ty()`, lifetime bounds included, except for const
/// parameters. `const N: usize` is `[(); N]`, `const X: u64` uses `ID`, and any other primitive is
/// a trait object of the Repeater with its slot filled. The impl declares the generics without the
/// encoded bounds: `Referrer::impl_generics()`.
///
/// Only a parameter whose type is a primitive type can be encoded: a const parameter of an alias is
/// an [`EncodeError::UnsupportedConstType`], whatever the alias is. To use the slots with
/// [`Referrer::repeater_impl_with()`](crate::Referrer::repeater_impl_with) and
/// [`Referrer::entry_args_with()`](crate::Referrer::entry_args_with), pass
/// `|value, param| slots.encode_const(value, &param.ty)`.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
/// let generics: Generics = parse_quote!(<const B: bool, const C: char = 'z'>);
/// let ty = slots.encode_generics_params_to_ty(&generics.params).unwrap();
/// assert_eq!(ty, parse_quote!((
///     *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
///     *const dyn crate::Repeater<(), 0, 0, false, C, Type = ()>
/// )));
/// ```
///
/// [`Referrer::repeater_impl()`](crate::Referrer::repeater_impl) and
/// [`Referrer::entry_args()`](crate::Referrer::entry_args) follow the encoders, so a macro that
/// uses them cannot disagree with itself. The example does the same by hand. One Repeater impl per
/// entry declares the generics of its context
/// ([`Referrer::impl_generics()`](crate::Referrer::impl_generics)) and encodes the generics that
/// the entry uses in its trait arguments. A copy encodes the arguments that its site gives to the
/// same generics: those of the trait itself come from
/// [`Referrer::generic_args()`](crate::Referrer::generic_args), and a generic that the copy
/// declares itself, such as one of a method, is passed by its own name.
///
/// ```
/// use syn::visit_mut::VisitMut;
/// use syn::{parse_quote, GenericArgument, ItemImpl, ItemTrait, Signature};
/// use type_leak::{Leaker, RepeaterSlots};
///
/// let item: ItemTrait = parse_quote! {
///     pub trait MyLeaker<T> {
///         fn f(&self, t: T) -> MyWrap<T>;
///         fn g(&self) -> MyType;
///     }
/// };
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_trait(&item, false);
/// let referrer = leaker.result().finish().unwrap();
/// assert_eq!(referrer.iter().count(), 2);
/// assert_eq!(referrer.context_generics(top_level).params.len(), 1);
///
/// // One Repeater impl per entry. It declares `impl_generics()` and encodes the generics the entry
/// // uses in its trait arguments.
/// let slots = RepeaterSlots::new(parse_quote!(path::Repeater));
/// let id = 42u64;
/// let impls: Vec<ItemImpl> = referrer
///     .iter()
///     .enumerate()
///     .map(|(ix, ty)| {
///         let context = referrer.context(ix);
///         let used = referrer.context_generics(context);
///         let encoded = slots.encode_generics_params_to_ty(&used.params).unwrap();
///         let declared = referrer.impl_generics(context);
///         let params = declared.params.iter();
///         parse_quote! {
///             impl<#(#params,)* __S: ?Sized> path::Repeater<#encoded, #id, #ix> for __S {
///                 type Type = #ty;
///             }
///         }
///     })
///     .collect();
/// let expected: ItemImpl = parse_quote! {
///     impl<T, __S: ?Sized> path::Repeater<(T,), 42u64, 0usize> for __S {
///         type Type = MyWrap<T>;
///     }
/// };
/// assert_eq!(impls[0], expected);
///
/// // The copy of `f` in `impl MyLeaker<u8> for ..`. Each leaked type becomes a path through the
/// // Repeater, with its generics as arguments.
/// let trait_args: Vec<GenericArgument> = vec![parse_quote!(u8)];
/// let mut sig: Signature = parse_quote!(fn f(&self, t: T) -> MyWrap<T>);
/// let mut visitor = referrer.clone().visitor(top_level, |_ty, ix| {
///     let args = referrer.generic_args(ix, &item.generics.params, &trait_args).unwrap();
///     let used = &referrer.generics(ix).params;
///     let encoded = slots.encode_generics_args_to_ty(&args, used).unwrap();
///     parse_quote!(<Self as path::Repeater<#encoded, #id, #ix>>::Type)
/// });
/// visitor.visit_signature_mut(&mut sig);
/// let expected: Signature = parse_quote!(
///     fn f(&self, t: T) -> <Self as path::Repeater<(u8,), 42u64, 0usize>>::Type
/// );
/// assert_eq!(sig, expected);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RepeaterSlots {
    /// The path of the Repeater trait, which [`repeater_definition()`] defines.
    pub repeater: Path,
}

impl RepeaterSlots {
    /// Creates the encoder for the Repeater at the path `repeater`, the trait that
    /// [`repeater_definition()`] defines in your crate.
    pub fn new(repeater: Path) -> Self {
        Self { repeater }
    }

    /// Encodes the const `value` of type `ty`. Fails with [`EncodeError::UnsupportedConstType`] if
    /// `ty` is not a primitive type.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// // The `bool` value goes into the slot of the Repeater for `bool`.
    /// let ty = slots.encode_const(&parse_quote!(true), &parse_quote!(bool)).unwrap();
    /// assert_eq!(ty, parse_quote!(*const dyn crate::Repeater<(), 0, 0, true, Type = ()>));
    /// // `usize` needs no slot: it is an array length.
    /// let ty = slots.encode_const(&parse_quote!(3), &parse_quote!(usize)).unwrap();
    /// assert_eq!(ty, parse_quote!([(); 3]));
    /// ```
    pub fn encode_const(&self, value: &Expr, ty: &Type) -> std::result::Result<Type, EncodeError> {
        let prim = const_param_primitive(ty)
            .ok_or_else(|| EncodeError::UnsupportedConstType(Box::new(ty.clone())))?;
        let value = normalize_const_arg(value);
        let repeater = &self.repeater;
        if prim == "usize" {
            return Ok(parse_quote!([(); #value]));
        }
        if prim == "u64" {
            return Ok(parse_quote!(*const dyn #repeater<(), #value, 0, Type = ()>));
        }
        let position = REPEATER_SLOTS
            .iter()
            .position(|(ty, _, _)| *ty == prim)
            .unwrap();
        let defaults = REPEATER_SLOTS[..position]
            .iter()
            .map(|(_, _, default)| -> Expr { parse_str(default).unwrap() });
        Ok(parse_quote!(*const dyn #repeater<(), 0, 0, #(#defaults,)* #value, Type = ()>))
    }

    /// Like `encode_generics_params_to_ty_with()`, with const parameters encoded by the slots.
    ///
    /// `Self: 'a` is not encoded: use [`RepeaterSlots::encode_generics_params_to_ty_with_self()`].
    /// Fails with [`EncodeError::UnsupportedConstType`] if the type of a const parameter is not a
    /// primitive type.
    pub fn encode_generics_params_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
    ) -> std::result::Result<Type, EncodeError> {
        self.encode_params(iter, None)
    }

    /// Like [`RepeaterSlots::encode_generics_params_to_ty()`], but also encodes `Self: 'a`.
    ///
    /// Pass `self_outlives` and `self_ty` as for `encode_generics_params_to_ty_with_self()`. Fails
    /// with [`EncodeError::UnboundedTraitObject`] if `self_ty` is a trait object without a
    /// lifetime bound, and as the function above does for a const parameter.
    pub fn encode_generics_params_to_ty_with_self<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
        self_ty: &Type,
        self_outlives: &[Lifetime],
    ) -> std::result::Result<Type, EncodeError> {
        self.encode_params(iter, Some((self_ty, self_outlives)))
    }

    fn encode_params<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
        self_outlives: Option<(&Type, &[Lifetime])>,
    ) -> std::result::Result<Type, EncodeError> {
        encode_generics_params_with(iter, self_outlives, |ConstParam { ident, ty, .. }| {
            self.encode_const(&parse_quote!(#ident), ty)
        })
    }

    /// Like `encode_generics_args_to_ty_with()`, with const arguments encoded by the slots.
    ///
    /// `params` are the generics of the Leaker that match `iter`. They give the type of each const
    /// argument. Every lifetime parameter needs an argument.
    /// Omitted type and const arguments take the defaults, with the preceding parameters
    /// substituted. Defaults are used as written, so they must not depend on the Leaker's context
    /// (`= MY_CONST` would not resolve in the Referrer). Give such arguments explicitly.
    /// `Self: 'a` is not encoded: see [`RepeaterSlots::encode_generics_args_to_ty_with_self()`].
    /// Fails with [`EncodeError`] as that function does, and with
    /// [`EncodeError::UnsupportedConstType`] if the type of a const parameter is not a primitive
    /// type.
    pub fn encode_generics_args_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
    ) -> std::result::Result<Type, EncodeError> {
        encode_generics_args_with_params(iter, params, None, |value, cp| {
            self.encode_const(value, &cp.ty)
        })
    }

    /// Like [`RepeaterSlots::encode_generics_args_to_ty()`], but also encodes `Self: 'a`.
    ///
    /// Pass `self_outlives` and `self_ty` as for `encode_generics_args_to_ty_with_self()`. The
    /// errors are those of both functions.
    pub fn encode_generics_args_to_ty_with_self<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
        self_ty: &Type,
        self_outlives: &[Lifetime],
    ) -> std::result::Result<Type, EncodeError> {
        encode_generics_args_with_params(
            iter,
            params,
            Some((self_ty, self_outlives)),
            |value, cp| self.encode_const(value, &cp.ty),
        )
    }
}

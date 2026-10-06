//! The definition of the Repeater trait, and encoding generics into its const slots.

use crate::encode::{
    const_param_primitive, encode_generics_args_with_params, encode_generics_params_with,
    normalize_const_arg, EncodeError,
};
use proc_macro2::Span;
use syn::punctuated::Punctuated;
use syn::*;

/// Types of const parameters which have a slot in the Repeater, in the order of the slots. `usize`
/// is encoded as `[(); N]`, and `u64` uses the Repeater's `ID` parameter.
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

/// Definition of a Repeater which supports const generics of any primitive type.
///
/// Encoding a const generic parameter into a type requires a type with a const parameter of the
/// same type, and `core` has no stable type with a const parameter of a type other than `usize`.
/// So the Repeater itself provides one: besides the leaked type's key `T`, the `ID` and the
/// index `NTH`, it takes a defaulted const "slot" parameter per primitive type. A const value is
/// then encoded as a pointer to a trait object of the Repeater with that slot filled (see
/// [`RepeaterSlots`]):
///
/// ```ignore
/// pub trait Repeater<
///     T: ?Sized, const ID: u64, const NTH: usize,
///     const BOOL: bool = false, const CHAR: char = '\0',
///     const U8: u8 = 0, const U16: u16 = 0, const U32: u32 = 0, const U128: u128 = 0,
///     const I8: i8 = 0, const I16: i16 = 0, const I32: i32 = 0, const I64: i64 = 0,
///     const I128: i128 = 0, const ISIZE: isize = 0,
/// > {
///     type Type: ?Sized;
/// }
/// ```
///
/// The slots have defaults, so a Repeater used as `Repeater<T, ID, NTH>` works as before. A
/// Repeater without the slots also works with [`RepeaterSlots`] as long as all const parameters are
/// `usize` or `u64`. The user crate defines it once, e.g. by emitting this definition from a
/// macro.
///
/// The generated definition requires Rust 1.59 (defaults of const parameters) and the 2018 edition
/// (paths starting with `::core`).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let def = repeater_definition(&parse_quote!(pub), &parse_quote!(Repeater));
/// assert_eq!(def.generics.params.len(), 3 + 12);
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

/// Encodes generics into types using the const slots of the Repeater defined by
/// [`repeater_definition()`].
///
/// Like [`encode_generics_params_to_ty()`](crate::encode_generics_params_to_ty()), generics are
/// encoded into a tuple type: a lifetime `'a` as `&'a ()`, a type `T` as `T` and a `const N: usize`
/// as `[(); N]`. A const parameter of another primitive type is encoded as a trait object of the
/// Repeater with the corresponding slot filled, e.g. `*const dyn path::Repeater<(), 0, 0, B, Type =
/// ()>` for `const B: bool`, and `const X: u64` uses the `ID` parameter. The lifetime bounds of the
/// parameters follow them: `T: 'a` as `&'a T`, and `'b: 'a` as `&'a &'b ()` (see
/// [`encode_generics_params_to_ty_with()`](crate::encode_generics_params_to_ty_with())). The
/// Repeater's `impl` declares the generics without these bounds, which its header implies: see
/// [`Referrer::impl_generics()`](crate::Referrer::impl_generics()).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::quote;
/// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
///
/// // In the Leaker
/// let generics: Generics = parse_quote!(<T, const N: usize, const B: bool, const C: char = 'z'>);
/// let ty = slots.encode_generics_params_to_ty(&generics.params).unwrap();
/// let expected: Type = parse_quote!((
///     T,
///     [(); N],
///     *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
///     *const dyn crate::Repeater<(), 0, 0, false, C, Type = ()>
/// ));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
///
/// // In the Referrer (for `MyTrait<u8, 3, true>`; the defaulted `C` is omitted)
/// let args: AngleBracketedGenericArguments = parse_quote!(<u8, 3, true>);
/// let ty = slots
///     .encode_generics_args_to_ty(&args.args, &generics.params)
///     .unwrap();
/// let expected: Type = parse_quote!((
///     u8,
///     [(); 3],
///     *const dyn crate::Repeater<(), 0, 0, true, Type = ()>,
///     *const dyn crate::Repeater<(), 0, 0, false, 'z', Type = ()>
/// ));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
///
/// // A lifetime bound is encoded after the parameters, with the arguments substituted.
/// let generics: Generics = parse_quote!(<'a, T: 'a>);
/// let ty = slots.encode_generics_params_to_ty(&generics.params).unwrap();
/// let expected: Type = parse_quote!((&'a (), T, &'a T));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
/// let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8>);
/// let ty = slots
///     .encode_generics_args_to_ty(&args.args, &generics.params)
///     .unwrap();
/// let expected: Type = parse_quote!((&'static (), u8, &'static u8));
/// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RepeaterSlots {
    /// Path to the Repeater.
    pub repeater: Path,
}

impl RepeaterSlots {
    /// Specifies the path to the Repeater defined by [`repeater_definition()`].
    pub fn new(repeater: Path) -> Self {
        Self { repeater }
    }

    /// Encodes a const value `value` of type `ty` into a type.
    ///
    /// Returns [`EncodeError::UnsupportedConstType`] if `ty` is not a primitive type.
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

    /// Encodes [`GenericParam`]s into a type, like
    /// [`encode_generics_params_to_ty()`](crate::encode_generics_params_to_ty()), including the
    /// lifetime bounds of the parameters, which follow the parameters (`T: 'a` as `&'a T`, and
    /// `'b: 'a` as `&'a &'b ()`, see
    /// [`encode_generics_params_to_ty_with()`](crate::encode_generics_params_to_ty_with())). The
    /// `impl` declares the generics without these bounds (see
    /// [`Referrer::impl_generics()`](crate::Referrer::impl_generics())).
    ///
    /// `Self: 'a` is not encoded: use [`RepeaterSlots::encode_generics_params_to_ty_with_self()`]
    /// for the generics of a context with
    /// [`Referrer::context_self_outlives()`](crate::Referrer::context_self_outlives()).
    ///
    /// Returns [`EncodeError::UnsupportedConstType`] if the type of a const parameter is not a
    /// primitive type.
    pub fn encode_generics_params_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
    ) -> std::result::Result<Type, EncodeError> {
        self.encode_params(iter, None)
    }

    /// Like [`RepeaterSlots::encode_generics_params_to_ty()`], with `Self: 'a` for each lifetime of
    /// `self_outlives` (the lifetimes of
    /// [`Referrer::context_self_outlives()`](crate::Referrer::context_self_outlives()) of the
    /// context whose generics `iter` are) encoded after the lifetime bounds as `&'a Self`, where
    /// `Self` is `self_ty`: the type the implementation of the Repeater is for, `__S` for an
    /// implementation for any type of the trait (`impl<'a, __S: ?Sized + MyLeaker> Repeater<(&'a
    /// (), &'a __S), ..> for __S`, see
    /// [`Referrer::iter_with_self()`](crate::Referrer::iter_with_self())), or `Self` for an
    /// implementation for a type. The header implies `__S: 'a` then, which the implementation must
    /// not declare, as nothing proves it where the type is copied but this encoding (see
    /// [`Referrer::generics()`](crate::Referrer::generics())).
    ///
    /// A lifetime which is not a parameter of `iter` is skipped, like a lifetime bound by one.
    /// Returns [`EncodeError::UnboundedTraitObject`] if `self_ty` is a trait object without a
    /// lifetime bound (its lifetime would default to the lifetime of the reference), and
    /// [`EncodeError::UnsupportedConstType`] if the type of a const parameter is not a primitive
    /// type.
    ///
    /// This is [`encode_generics_params_to_ty_with_self()`](crate::encode_generics_params_to_ty_with_self())
    /// with the const parameters encoded by the slots of the Repeater: both are implemented by the
    /// same function, so they encode the same, except for the const parameters.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// let generics: Generics = parse_quote!(<'a>);
    /// let ty = slots
    ///     .encode_generics_params_to_ty_with_self(&generics.params, &parse_quote!(__S), &[parse_quote!('a)])
    ///     .unwrap();
    /// let expected: Type = parse_quote!((&'a (), &'a __S));
    /// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
    /// ```
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

    /// Encodes [`GenericArgument`]s into a type. `params` are the [`GenericParam`]s of the Leaker
    /// corresponding to `iter`, which are used to find the types of const arguments.
    ///
    /// The result is the same type as [`RepeaterSlots::encode_generics_params_to_ty()`] with
    /// `params` substituted by `iter`:
    ///
    /// - Arguments are matched with parameters by position (lifetime arguments with lifetime
    ///   parameters, and the others with type and const parameters). Every lifetime parameter
    ///   needs an argument (lifetimes cannot be elided in the trait of an `impl`).
    /// - A bare identifier corresponding to a const parameter (e.g. `N` in `MyTrait<N>`, parsed as
    ///   a type argument) is treated as a const argument.
    /// - Omitted type and const arguments take the parameters' defaults, in which the preceding
    ///   parameters are substituted by their arguments. The defaults are used as they are written,
    ///   so they should not depend on the Leaker's type context (e.g. `= MY_CONST` or
    ///   `= MyType<N>` would not resolve in the Referrer's context); otherwise the arguments
    ///   should be given explicitly.
    /// - Associated item bindings and constraints (e.g. `Item = u8`) are skipped.
    /// - The lifetime bounds of `params` are encoded after the parameters, with the arguments
    ///   substituted: `T: 'a` with the arguments `'static` and `u8` is `&'static u8`, and `'b: 'a`
    ///   is `&'a &'b ()`.
    ///
    /// - `Self: 'a` is not encoded (see [`RepeaterSlots::encode_generics_args_to_ty_with_self()`]).
    ///
    /// Returns [`EncodeError`] if the type of a const parameter is not a primitive type, if there
    /// are too many arguments, if an argument without a default is missing, or if the argument of
    /// a type with a lifetime bound is a trait object without a lifetime bound (its lifetime bound
    /// defaults to that of the reference `&'a dyn Tr` it is encoded in, not to that of the other
    /// elements: write it, e.g. `dyn Tr + 'static`, see [`EncodeError::UnboundedTraitObject`]).
    pub fn encode_generics_args_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
    ) -> std::result::Result<Type, EncodeError> {
        encode_generics_args_with_params(iter, params, None, |value, cp| {
            self.encode_const(value, &cp.ty)
        })
    }

    /// Like [`RepeaterSlots::encode_generics_args_to_ty()`], with `Self: 'a` for each lifetime of
    /// `self_outlives` (the lifetimes of
    /// [`Referrer::context_self_outlives()`](crate::Referrer::context_self_outlives()), which are
    /// parameters of `params`) encoded after the lifetime bounds as `&'b Self`, where `'b` is the
    /// argument of `'a` and `Self` is `self_ty`: the type whose Repeater is used, i.e. `Self` in
    /// the implementation of the Leaker's trait. It is the same type as the encoding of
    /// [`RepeaterSlots::encode_generics_params_to_ty_with_self()`] (see there).
    ///
    /// This is [`encode_generics_args_to_ty_with_self()`](crate::encode_generics_args_to_ty_with_self())
    /// with the const arguments encoded by the slots of the Repeater: both are implemented by the
    /// same function, so they encode the same, except for the const arguments.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// let generics: Generics = parse_quote!(<'a>);
    /// // For `impl<'q> MyLeaker<'q> for ..`, where the Leaker's `'a` is `'q`
    /// let args: AngleBracketedGenericArguments = parse_quote!(<'q>);
    /// let ty = slots
    ///     .encode_generics_args_to_ty_with_self(
    ///         &args.args,
    ///         &generics.params,
    ///         &parse_quote!(Self),
    ///         &[parse_quote!('a)],
    ///     )
    ///     .unwrap();
    /// let expected: Type = parse_quote!((&'q (), &'q Self));
    /// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
    /// ```
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

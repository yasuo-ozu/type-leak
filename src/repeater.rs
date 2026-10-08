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

/// Returns the definition of a Repeater that supports const generics of any primitive type.
///
/// Besides `T`, `ID` and `NTH`, it has a defaulted const slot per primitive type: `BOOL`, `CHAR`,
/// `U8`, `U16`, `U32`, `U128`, `I8`, `I16`, `I32`, `I64`, `I128` and `ISIZE`.
/// [`RepeaterSlots`] encodes const values into them, and `Repeater<T, ID, NTH>` still works.
/// Without the slots, it works if every const parameter is `usize` or `u64`.
/// Define it once in the user crate, e.g. from a macro. It needs Rust 1.59 and the 2018 edition.
/// See [Const generics](crate#const-generics).
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

/// Encodes generics into types, using the const slots of the Repeater from
/// [`repeater_definition()`].
///
/// It encodes like `encode_generics_params_to_ty()`, lifetime bounds included, except for const
/// parameters. `const N: usize` is `[(); N]`, `const X: u64` uses `ID`, and any other primitive is
/// a trait object of the Repeater with its slot filled. The impl declares the generics without the
/// encoded bounds: `Referrer::impl_generics()`.
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RepeaterSlots {
    pub repeater: Path,
}

impl RepeaterSlots {
    /// Uses the Repeater at `repeater`, defined by [`repeater_definition()`].
    pub fn new(repeater: Path) -> Self {
        Self { repeater }
    }

    /// Encodes the const `value` of type `ty`. Fails with [`EncodeError::UnsupportedConstType`] if
    /// `ty` is not a primitive type.
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

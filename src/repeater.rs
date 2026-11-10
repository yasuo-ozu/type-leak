//! The definition of the Repeater trait, and encoding generics into its const slots.

use crate::encode::{const_param_primitive, normalize_const_arg, EncodeError};
use proc_macro2::Span;
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
/// its value. `const N: usize` is encoded as `[(); N]` ([`array_const()`](crate::array_const)), a
/// `u64` value uses the `ID` slot, and any other primitive type uses its slot. Without the slots,
/// the Repeater works if every const parameter is `usize` or `u64`. In that case you can write the
/// trait by hand instead:
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

/// Encodes const values into the const slots of the Repeater from [`repeater_definition()`].
///
/// Pass `|value, param| slots.encode_const(value, &param.ty)` as the `const_fn` of
/// [`Referrer::repeater_impl()`](crate::Referrer::repeater_impl) and
/// [`Referrer::entry_args()`](crate::Referrer::entry_args). `const N: usize` is `[(); N]`,
/// `const X: u64` uses `ID`, and any other primitive is a trait object of the Repeater with its
/// slot filled.
///
/// Only a parameter whose type is a primitive type can be encoded: a const parameter of an alias is
/// an [`EncodeError::UnsupportedConstType`], whatever the alias is.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let tr: ItemTrait = parse_quote! {
///     trait Tr<const B: bool, const C: char = 'z'> { fn f(&self) -> Wrap<B, C>; }
/// };
/// let mut leaker = Leaker::new();
/// leaker.intern_trait(&tr, true);
/// let referrer = leaker.result().finish().unwrap();
/// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
///
/// // The impl encodes the parameters themselves...
/// let param: Ident = parse_quote!(__S);
/// let r = referrer
///     .repeater_impl(0, ImplSelf::Param(&param), |value, param| {
///         slots.encode_const(value, &param.ty)
///     })
///     .unwrap();
/// assert_eq!(r.args, parse_quote!((
///     *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
///     *const dyn crate::Repeater<(), 0, 0, false, C, Type = ()>
/// )));
///
/// // ...and a copy in `impl Tr<true, 'x'> for ..` the arguments.
/// let trait_args: Vec<GenericArgument> = vec![parse_quote!(true), parse_quote!('x')];
/// let args = referrer
///     .entry_args(
///         0,
///         &UseSite::substituting(&tr.generics.params, &trait_args),
///         |value, param| slots.encode_const(value, &param.ty),
///     )
///     .unwrap();
/// assert_eq!(args, parse_quote!((
///     *const dyn crate::Repeater<(), 0, 0, true, Type = ()>,
///     *const dyn crate::Repeater<(), 0, 0, false, 'x', Type = ()>
/// )));
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
}

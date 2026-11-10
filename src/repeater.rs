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

/// Returns the definition of the Repeater trait, the one trait that both crates see.
///
/// Define it once, at the root of the crate that writes the Repeater impls (the one that exports
/// the item), not in the crate of the proc macros: the impls are for any `Self`, which only a
/// local trait allows. The copying crate names it through the crate that defines it.
///
/// `T` is the encoding of the generics of an entry, `ID` the Leaker and `NTH` the entry. A
/// defaulted const slot per primitive type (`BOOL`, `CHAR`, `U8`, .., `ISIZE`) carries the other
/// const generics ([`RepeaterSlots`]), and `Repeater<T, ID, NTH>` still works. `const N: usize` is
/// `[(); N]` ([`array_const()`](crate::array_const)) and a `u64` uses `ID`, so if every const
/// parameter is one of these you can write the trait by hand instead:
/// `pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> { type Type: ?Sized; }`. The
/// generated trait needs Rust 1.59 and the 2018 edition, while `type-leak` supports 1.56.
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
#[must_use]
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

/// Encodes const values into the const slots of the Repeater from [`repeater_definition()`], and
/// writes the paths through it.
///
/// Pass [`slots.const_fn()`](RepeaterSlots::const_fn) as the `const_fn` of
/// [`Referrer::repeater_impl()`](crate::Referrer::repeater_impl) and
/// [`Referrer::entry_args()`](crate::Referrer::entry_args). `const N: usize` is `[(); N]`,
/// `const X: u64` uses `ID`, and any other primitive is a trait object of the Repeater with its
/// slot filled. A const parameter of an alias type is an [`EncodeError::UnsupportedConstType`].
/// [`trait_path()`](RepeaterSlots::trait_path) and [`projection()`](RepeaterSlots::projection)
/// write the Repeater with its arguments, for the impl and for the use.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RepeaterSlots {
    repeater: Path,
}

impl RepeaterSlots {
    /// Creates the encoder for the Repeater at the path `repeater`.
    pub fn new(repeater: Path) -> Self {
        Self { repeater }
    }

    /// Returns the path of the Repeater trait, which [`repeater_definition()`] defines.
    pub fn repeater(&self) -> &Path {
        &self.repeater
    }

    /// Returns the path of the Repeater with the arguments of the entry `ix` of the Leaker `id`:
    /// `Repeater<args, IDu64, IXusize>`.
    ///
    /// `args` is the first argument: [`RepeaterImpl::args`](crate::RepeaterImpl::args) for the
    /// impl, [`Referrer::entry_args()`](crate::Referrer::entry_args) for a use. Write the impl for
    /// this path and the projection ([`RepeaterSlots::projection()`]) with it, so that a change of
    /// the parameters of the Repeater does not reach your macro.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// let path = slots.trait_path(&parse_quote!((&'a (), T)), 7, 0);
    /// assert_eq!(path, parse_quote!(crate::Repeater<(&'a (), T), 7u64, 0usize>));
    /// ```
    #[must_use]
    pub fn trait_path(&self, args: &Type, id: u64, ix: usize) -> Path {
        let repeater = &self.repeater;
        parse_quote!(#repeater<#args, #id, #ix>)
    }

    /// Returns `<self_ty as Repeater<args, IDu64, IXusize>>::Type`: the type that stands for the
    /// entry `ix` of the Leaker `id` at a use ([`RepeaterSlots::trait_path()`]).
    ///
    /// `self_ty` is what the path is written for: `Self` in the copy, or the host.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// let ty = slots.projection(&parse_quote!(Self), &parse_quote!((T,)), 7, 1);
    /// assert_eq!(ty, parse_quote!(<Self as crate::Repeater<(T,), 7u64, 1usize>>::Type));
    /// ```
    #[must_use]
    pub fn projection(&self, self_ty: &Type, args: &Type, id: u64, ix: usize) -> Type {
        let path = self.trait_path(args, id, ix);
        parse_quote!(<#self_ty as #path>::Type)
    }

    /// Returns `|value, param| self.encode_const(value, &param.ty)`: the `const_fn` of
    /// [`Referrer::entry_args()`](crate::Referrer::entry_args) and
    /// [`Referrer::repeater_impl()`](crate::Referrer::repeater_impl).
    pub fn const_fn(
        &self,
    ) -> impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError> + '_ {
        move |value, param| self.encode_const(value, &param.ty)
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

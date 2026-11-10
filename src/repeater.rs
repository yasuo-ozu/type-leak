//! The definition of the Repeater trait, encoding generics into its const slots, and the pieces of
//! its impls ([`RepeaterImpl`], [`ImplSelf`]).

use crate::context::ModuleScope;
#[cfg(doc)]
use crate::context::SelfRequirement;
use crate::encode::{const_param_primitive, normalize_const_arg, EncodeError};
use crate::referrer::EntryId;
#[cfg(doc)]
use crate::referrer::Referrer;
#[cfg(doc)]
use crate::referrer::UseSite;
use proc_macro2::{Span, TokenStream};
use syn::*;
use template_quote::ToTokens;

/// Const parameter types with a slot, in slot order. `usize` uses `[(); N]`, and `u128` uses `KEY`.
const REPEATER_SLOTS: &[(&str, &str, &str)] = &[
    ("bool", "BOOL", "false"),
    ("char", "CHAR", "'\\0'"),
    ("u8", "U8", "0"),
    ("u16", "U16", "0"),
    ("u32", "U32", "0"),
    ("u64", "U64", "0"),
    ("i8", "I8", "0"),
    ("i16", "I16", "0"),
    ("i32", "I32", "0"),
    ("i64", "I64", "0"),
    ("i128", "I128", "0"),
    ("isize", "ISIZE", "0"),
];

/// The `KEY` argument of the Repeater for the entry `nth` of the Leaker `id`: `id` in the high 64
/// bits and the [index](EntryId::index) of `nth` in the low ones, so no two entries share a key.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// // The `EntryId` 3 (a macro gets one from `Referrer::iter()`).
/// let nth: EntryId = parse_quote!(3);
/// assert_eq!(repeater_key(7, nth), (7u128 << 64) | 3);
/// ```
#[must_use]
pub fn repeater_key(id: u64, nth: EntryId) -> u128 {
    (u128::from(id) << 64) | nth.index() as u128
}

/// Returns the definition of the Repeater trait, the one trait that both crates see.
///
/// Define it once, at the root of the crate that writes the Repeater impls (the one that exports
/// the item), not in the crate of the proc macros: the impls are for any `Self`, which only a
/// local trait allows. The copying crate names it through the crate that defines it.
///
/// `T` is the encoding of the generics of an entry and `KEY` is `repeater_key(ID, NTH)` for the
/// Leaker `ID` and its entry `NTH` ([`repeater_key()`]). A defaulted const slot per other
/// primitive type (`BOOL`, `CHAR`, `U8`, .., `ISIZE`) carries the other const generics
/// ([`encode_slot_const()`]), and `Repeater<T, KEY>` still works. `const N: usize` is `[(); N]`
/// ([`array_const()`](crate::array_const)) and a `u128` uses `KEY`, so if every const parameter is
/// one of these you can write the trait by hand instead:
/// `pub trait Repeater<T: ?Sized, const KEY: u128> { type Type: ?Sized; }`. The generated trait
/// needs Rust 1.59 and the 2018 edition, while `type-leak` supports 1.56.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let def = repeater_definition(&parse_quote!(pub), &parse_quote!(Repeater));
/// // `T`, `KEY` and a slot for each primitive const type but `usize` and `u128`.
/// assert_eq!(def.generics.params.len(), 2 + 12);
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
            const KEY: ::core::primitive::u128,
            #(#slots),*
        > {
            type Type: ?::core::marker::Sized;
        }
    }
}

/// Returns the path of the Repeater with the arguments of the entry `ix` of the Leaker `id`:
/// `Repeater<args, KEY>` with `KEY` from [`repeater_key(id, ix)`](repeater_key).
///
/// `repeater` is the path of the Repeater trait, which [`repeater_definition()`] defines. `args`
/// is the first argument, as
/// [`encode_generics_args_with_params()`](crate::encode_generics_args_with_params) writes it for
/// a use. Write the projection ([`repeater_projection()`]) with this path, so that a change
/// of the parameters of the Repeater does not reach your macro. `ToTokens` of
/// [`RepeaterImpl`](crate::RepeaterImpl) writes the impl with the same path, `KEY` included.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let repeater: Path = parse_quote!(crate::Repeater);
/// let ix: EntryId = parse_quote!(0);
/// let path = repeater_trait_path(&repeater, &parse_quote!((&'a (), T)), 7, ix);
/// let key = repeater_key(7, ix);
/// assert_eq!(path, parse_quote!(crate::Repeater<(&'a (), T), #key>));
/// ```
#[must_use]
pub fn repeater_trait_path(repeater: &Path, args: &Type, id: u64, ix: EntryId) -> Path {
    let key = repeater_key(id, ix);
    parse_quote!(#repeater<#args, #key>)
}

/// Returns `<self_ty as Repeater<args, KEY>>::Type`: the type that stands for the entry `ix` of
/// the Leaker `id` at a use ([`repeater_trait_path()`]).
///
/// `self_ty` is what the path is written for: `Self` in the copy, or the host.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let repeater: Path = parse_quote!(crate::Repeater);
/// let ix: EntryId = parse_quote!(1);
/// let ty = repeater_projection(&repeater, &parse_quote!(Self), &parse_quote!((T,)), 7, ix);
/// let key = repeater_key(7, ix);
/// assert_eq!(ty, parse_quote!(<Self as crate::Repeater<(T,), #key>>::Type));
/// ```
#[must_use]
pub fn repeater_projection(
    repeater: &Path,
    self_ty: &Type,
    args: &Type,
    id: u64,
    ix: EntryId,
) -> Type {
    let path = repeater_trait_path(repeater, args, id, ix);
    parse_quote!(<#self_ty as #path>::Type)
}

/// Returns `|value, param| encode_slot_const(repeater, value, &param.ty)`: the `const_fn` of
/// [`encode_generics_args_with_params()`](crate::encode_generics_args_with_params) and
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls).
///
/// See [`encode_slot_const()`] for what it writes.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let repeater: Path = parse_quote!(crate::Repeater);
/// let mut const_fn = slot_const(&repeater);
/// // `usize` is an array length, and `bool` fills its slot of the Repeater.
/// let param: ConstParam = parse_quote!(const N: usize);
/// assert_eq!(const_fn(&parse_quote!(3), &param).unwrap(), parse_quote!([(); 3]));
/// let param: ConstParam = parse_quote!(const B: bool);
/// assert_eq!(
///     const_fn(&parse_quote!(true), &param).unwrap(),
///     parse_quote!(*const dyn crate::Repeater<(), 0, true, Type = ()>)
/// );
/// ```
pub fn slot_const(
    repeater: &Path,
) -> impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError> + '_ {
    move |value, param| encode_slot_const(repeater, value, &param.ty)
}

/// Encodes the const `value` of type `ty` for the Repeater at the path `repeater`. Fails with
/// [`EncodeError::UnsupportedConstType`] if `ty` is not a primitive type.
///
/// It is the encoding of the const slots of the Repeater from [`repeater_definition()`]:
/// `const N: usize` is `[(); N]`, `const X: u128` uses `KEY`, and any other primitive is a trait
/// object of the Repeater with its slot filled. A const parameter of an alias type is an
/// [`EncodeError::UnsupportedConstType`]. Pass [`slot_const()`] as the `const_fn` of
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls) and
/// [`encode_generics_args_with_params()`](crate::encode_generics_args_with_params).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let repeater: Path = parse_quote!(crate::Repeater);
/// // The `bool` value goes into the slot of the Repeater for `bool`.
/// let ty = encode_slot_const(&repeater, &parse_quote!(true), &parse_quote!(bool)).unwrap();
/// assert_eq!(ty, parse_quote!(*const dyn crate::Repeater<(), 0, true, Type = ()>));
/// // `u64` has a slot, and `u128` is the `KEY` itself.
/// let ty = encode_slot_const(&repeater, &parse_quote!(4), &parse_quote!(u64)).unwrap();
/// assert_eq!(
///     ty,
///     parse_quote!(*const dyn crate::Repeater<(), 0, false, '\0', 0, 0, 0, 4, Type = ()>)
/// );
/// let ty = encode_slot_const(&repeater, &parse_quote!(5), &parse_quote!(u128)).unwrap();
/// assert_eq!(ty, parse_quote!(*const dyn crate::Repeater<(), 5, Type = ()>));
/// // `usize` needs no slot: it is an array length.
/// let ty = encode_slot_const(&repeater, &parse_quote!(3), &parse_quote!(usize)).unwrap();
/// assert_eq!(ty, parse_quote!([(); 3]));
/// ```
pub fn encode_slot_const(
    repeater: &Path,
    value: &Expr,
    ty: &Type,
) -> std::result::Result<Type, EncodeError> {
    let prim = const_param_primitive(ty)
        .ok_or_else(|| EncodeError::UnsupportedConstType(Box::new(ty.clone())))?;
    let value = normalize_const_arg(value);
    if prim == "usize" {
        return Ok(parse_quote!([(); #value]));
    }
    if prim == "u128" {
        return Ok(parse_quote!(*const dyn #repeater<(), #value, Type = ()>));
    }
    let position = REPEATER_SLOTS
        .iter()
        .position(|(ty, _, _)| *ty == prim)
        .unwrap();
    let defaults = REPEATER_SLOTS[..position]
        .iter()
        .map(|(_, _, default)| -> Expr { parse_str(default).unwrap() });
    Ok(parse_quote!(*const dyn #repeater<(), 0, #(#defaults,)* #value, Type = ()>))
}

/// What a Repeater impl is written for: the `target` of
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls).
///
/// An entry that names `Self`, or whose context's generics do, is always implemented for what
/// `Self` stands for, so the choice only matters for the others. [`ImplSelf::Marker`] has no such
/// impl: the call fails if there is one.
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
    /// The call of [`repeater_impls()`](crate::LeakerResult::repeater_impls) fails with
    /// [`EncodeError::NeedsSelf`] if any entry needs `Self`.
    Marker(&'a Type),
    /// One of 17 primitive types, chosen by the `id` of
    /// [`repeater_impls()`](crate::LeakerResult::repeater_impls) and the entry; the entries that
    /// need `Self` get [`ImplSelf::Param`] with `param`.
    ///
    /// Cheaper than a blanket impl when there are many entries. The use site must write
    /// [`bucket_self(ix, id)`](Referrer::bucket_self) as the type of the path through the Repeater
    /// (what `Self` stands for if it is `None`), or E0277, with the same `id`.
    Bucket {
        /// The parameter for the entries that need `Self`.
        param: &'a Ident,
    },
}

/// The pieces of the Repeater impl of an entry, which
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls) yields.
///
/// [`ToTokens`] writes the whole impl, `impl #generics #path for #self_ty { type Type = #ty; }`,
/// with the `KEY` as the last argument of [`path`](RepeaterImpl::path). The `KEY` is
/// `repeater_key(id, entry)`, where `id` is the one given to `repeater_impls()`, so the caller
/// adds nothing. It does not write [`entry`](RepeaterImpl::entry) or
/// [`module`](RepeaterImpl::module): place the impl in the module, if it is `Some`.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::{quote, ToTokens};
/// let item: ItemStruct = parse_quote!(struct S<T> { a: Wrap<T> });
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&item, false);
/// let result = leaker.result();
///
/// let repeater: Path = parse_quote!(crate::Repeater);
/// let param: Ident = parse_quote!(__S);
/// let r = result
///     .repeater_impls(&repeater, 7, ImplSelf::Param(&param), slot_const(&repeater))
///     .unwrap()
///     .next()
///     .unwrap();
///
/// // `ToTokens` writes the `KEY` of the entry 0 of the id 7.
/// let key = repeater_key(7, r.entry);
/// let expected: ItemImpl = parse_quote! {
///     impl<T, __S: ?::core::marker::Sized> crate::Repeater<(T,), #key> for __S {
///         type Type = Wrap<T>;
///     }
/// };
/// assert_eq!(parse2::<ItemImpl>(quote!(#r)).unwrap(), expected);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepeaterImpl {
    /// The entry that the impl is for.
    ///
    /// It is the id that the `KEY` ([`repeater_key()`]) and the copy's side
    /// ([`Referrer::bucket_self()`], and the closures of [`Referrer::visitor()`]) use for this
    /// entry.
    ///
    /// ```ignore
    /// // In crate a: `Rel` is the entry 0.
    /// pub struct S { a: Rel }
    ///
    /// // `entry` of the impl of `Rel`:
    /// EntryId(0)
    /// ```
    pub entry: EntryId,
    /// The generics that the impl declares, with its where clause.
    ///
    /// The lifetimes come first, then the types (the parameter for `Self` last among them), then the
    /// consts: the order that rustc before 1.59 requires.
    pub generics: Generics,
    /// The type that the impl is for.
    pub self_ty: Type,
    /// The path of the Repeater with [`args`](RepeaterImpl::args), but not the `KEY`.
    ///
    /// [`ToTokens`] writes the `KEY` (`repeater_key(id, entry)`, see [`repeater_key()`]) as its
    /// second argument, so the caller adds nothing. This path has none: it is for a reader who
    /// wants the trait.
    ///
    /// ```ignore
    /// // In crate a, with the Repeater at `crate::Repeater`: the entry `Wrap<T>`.
    /// pub struct S<T> { a: Wrap<T> }
    ///
    /// // `path` of its impl. The `KEY` is not in it.
    /// crate::Repeater<(T,)>
    /// ```
    pub path: Path,
    /// The first argument of the Repeater: what
    /// [`encode_generics_args_with_params()`](crate::encode_generics_args_with_params) writes for
    /// the arguments of [`UseSite::identity()`], but with the self type of the impl for `Self`
    /// (`&'a __S`, not `&'a Self`, for `Self: 'a`).
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
    /// The id of the Leaker that was given to `repeater_impls()`: the high bits of the `KEY`.
    pub(crate) id: u64,
}

/// Writes `impl #generics #path_with_key for #self_ty #where_clause { type Type = #ty; }`.
///
/// `path_with_key` is [`path`](RepeaterImpl::path) with `repeater_key(id, entry)`
/// ([`repeater_key()`]) added as its last argument; `id` is the one given to
/// [`repeater_impls()`](crate::LeakerResult::repeater_impls). The impl is not wrapped in
/// [`module`](RepeaterImpl::module).
impl ToTokens for RepeaterImpl {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let imp: ItemImpl = self.clone().into();
        imp.to_tokens(tokens);
    }
}

impl RepeaterImpl {
    /// `path` with `repeater_key(id, entry)` as its last argument.
    fn path_with_key(&self) -> Path {
        let key = repeater_key(self.id, self.entry);
        let mut path = self.path.clone();
        if let Some(PathArguments::AngleBracketed(args)) = path
            .segments
            .last_mut()
            .map(|segment| &mut segment.arguments)
        {
            args.args.push(parse_quote!(#key));
        }
        path
    }
}

/// The impl as an [`ItemImpl`], with the `KEY` and not in [`module`](RepeaterImpl::module).
/// [`ToTokens`] writes this.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S<T> { a: Wrap<T> }), true);
/// let result = leaker.result();
/// let repeater: Path = parse_quote!(crate::Repeater);
/// let param: Ident = parse_quote!(__S);
/// let target = ImplSelf::Param(&param);
/// let r = result.repeater_impls(&repeater, 7, target, array_const).unwrap().next().unwrap();
/// let key = repeater_key(7, r.entry);
///
/// let imp = ItemImpl::from(r);
/// let expected: ItemImpl = parse_quote! {
///     impl<T, __S: ?::core::marker::Sized> crate::Repeater<(T,), #key> for __S {
///         type Type = Wrap<T>;
///     }
/// };
/// assert_eq!(imp, expected);
/// ```
impl From<RepeaterImpl> for ItemImpl {
    fn from(r: RepeaterImpl) -> Self {
        let path = r.path_with_key();
        let ty = r.ty;
        ItemImpl {
            attrs: Vec::new(),
            defaultness: None,
            unsafety: None,
            impl_token: Default::default(),
            generics: r.generics,
            trait_: Some((None, path, Default::default())),
            self_ty: Box::new(r.self_ty),
            brace_token: Default::default(),
            items: vec![parse_quote!(type Type = #ty;)],
        }
    }
}

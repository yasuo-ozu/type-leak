//! The definition of the Repeater trait, encoding generics into its const slots, and the pieces of
//! its impls ([`RepeaterImpl`], [`ImplSelf`]).

#[cfg(doc)]
use crate::context::SelfRequirement;
use crate::context::{Context, ModuleScope};
use crate::encode::{const_param_primitive, encode_entry_args, EncodeError};
#[cfg(doc)]
use crate::referrer::Referrer;
use crate::referrer::{bucket_type, EntryId};
use crate::util::tidy;
use proc_macro2::{Span, TokenStream};
use syn::*;
use template_quote::{quote, ToTokens};

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

/// How the macros of both crates write the Repeater of one Leaker: its path, the Leaker's id and
/// what the impls are for.
///
/// The Leaker's macro gives it to
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls) to write the impls, and
/// the Referrer's macro uses [`Projection::of()`] and [`Referrer::complete_args()`] to write the
/// types that stand for the entries, so both must make the same one. The path must stay without
/// arguments: the methods panic otherwise.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_trait(&parse_quote!(trait Tr { fn f(&self) -> Rel; }));
/// let result = leaker.result();
/// let projection = Projection::new(
///     parse_quote!(crate::Repeater),
///     7,
///     ImplSelf::Param(parse_quote!(__S)),
/// );
///
/// // Crate A writes one impl for each entry...
/// let impls = result.repeater_impls(&projection).unwrap();
/// // ...and crate B the type that stands for it.
/// let referrer = result.finish().unwrap();
/// let (entry, _, context) = referrer.iter().next().unwrap();
/// let written = projection.of(&parse_quote!(Self), context, entry).unwrap();
///
/// let key = projection.key(entry);
/// assert_eq!(written, parse_quote!(<Self as crate::Repeater<(), #key>>::Type));
/// assert_eq!(ItemImpl::from(impls[0].clone()), parse_quote! {
///     impl<__S: ?::core::marker::Sized> crate::Repeater<(), #key> for __S { type Type = Rel; }
/// });
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Projection {
    /// The path of the Repeater trait that [`repeater_definition()`] writes, without arguments.
    ///
    /// ```ignore
    /// // In crate a, at its root:
    /// pub trait Repeater<T: ?Sized, const KEY: u128, ..> { type Type: ?Sized; }
    ///
    /// // `path`, as the copy's crate names it:
    /// crate::Repeater
    /// ```
    pub path: Path,
    /// The id of the Leaker: the high 64 bits of the `KEY`.
    ///
    /// ```ignore
    /// // Two Leakers of one crate have two ids, so their entries have different keys:
    /// 7
    /// ```
    pub id: u64,
    /// What the impls of the Leaker's crate are for. [`Projection::of()`] writes each entry for the
    /// self type of its impl, so the copy's side passes `Self` or its host and nothing else.
    ///
    /// ```ignore
    /// // The impls of the entries are for any `Self`:
    /// impl<__S: ?Sized> crate::Repeater<(), KEY> for __S { type Type = Rel; }
    ///
    /// // `impl_self`:
    /// ImplSelf::Param(__S)
    /// ```
    pub impl_self: ImplSelf,
}

impl Projection {
    /// Returns the projection through the Repeater at `path` for the Leaker `id`, with `impl_self`
    /// for its impls.
    ///
    /// # Panics
    ///
    /// If `path` has arguments (`crate::Repeater<u8>`): it names the trait alone, or with an empty
    /// `<>`.
    #[must_use]
    pub fn new(path: Path, id: u64, impl_self: ImplSelf) -> Self {
        check_repeater_path(&path);
        Projection {
            path,
            id,
            impl_self,
        }
    }

    /// Returns the `KEY` argument of the Repeater for `entry`: [`id`](Projection::id) in the high 64
    /// bits and the [index](EntryId::index) of `entry` in the low ones, so no two entries share a
    /// key.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let projection = Projection::new(
    ///     parse_quote!(crate::Repeater),
    ///     7,
    ///     ImplSelf::Param(parse_quote!(__S)),
    /// );
    /// // `3` is an `EntryId`, as a macro gets one from `Referrer::iter()`.
    /// assert_eq!(projection.key(parse_quote!(3)), (7u128 << 64) | 3);
    /// ```
    #[must_use]
    pub fn key(&self, entry: EntryId) -> u128 {
        key_of(self.id, entry)
    }

    /// Returns `<S as Repeater<ARGS, KEY>>::Type`: the type that stands for `entry` in the copy.
    ///
    /// `self_ty` is what `Self` is in the copy: `Self` inside an impl, or the host type. `S` is
    /// `self_ty`, except where [`impl_self`](Projection::impl_self) says that the impl of the entry
    /// is for another type: the bucket type of [`ImplSelf::Bucket`] and the type of
    /// [`ImplSelf::Marker`], for an entry that does not need `Self`. `context` is the [`Context`]
    /// of the entry ([`Referrer::iter()`] gives it; [`Referrer::visitor()`] calls this method with
    /// it for each entry). `KEY` is [`key(entry)`](Projection::key), and
    /// `ARGS` is made from the generics of the entry, each by its own name, with the encoding that
    /// the Leaker's [`RepeaterImpl::path`] has: a lifetime `'a` is `&'a ()`, a type is itself (`*const
    /// T` if it is `?Sized`), and a bound `T: 'a` or `Self: 'a` adds `&'a T` (with `S` for `Self`),
    /// and `'b: 'a` adds `&'a &'b ()`. A const is encoded into the const slots of the Repeater
    /// ([`repeater_definition()`]): `const N: usize` is `[(); N]`, a `u128` uses `KEY`, and any other
    /// primitive type is a trait object of the Repeater with its slot filled. The copy of a use site
    /// gets the arguments of the site by [`Substitute`](crate::Substitute) afterwards.
    ///
    /// Fails with [`EncodeError::UnsupportedConstType`] for a const whose type is not a primitive
    /// type (an alias of one is not: the slot cannot be told), with
    /// [`EncodeError::UnboundedTraitObject`] if `S` is a trait object without a lifetime bound where
    /// `Self: 'a` needs one, and with [`EncodeError::NeedsSelf`] for [`ImplSelf::Marker`] and an
    /// entry that needs `Self` (the entry's leaked type is not known here: the error has the type
    /// that would be written for `self_ty`).
    ///
    /// # Panics
    ///
    /// If [`path`](Projection::path) has arguments.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&parse_quote!(trait Tr<T, const B: bool> { fn f(&self) -> Wrap<T, B>; }));
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// // The entry `Wrap<T, B>`, written with its own generics: `B` is in the slot of `bool`.
    /// let (entry, _, context) = referrer.iter().next().unwrap();
    /// let projection = Projection::new(
    ///     parse_quote!(crate::Repeater),
    ///     42,
    ///     ImplSelf::Param(parse_quote!(__S)),
    /// );
    /// let ty = projection.of(&parse_quote!(Self), context, entry).unwrap();
    /// let key = projection.key(entry);
    /// assert_eq!(ty, parse_quote! {
    ///     <Self as crate::Repeater<(T, *const dyn crate::Repeater<(), 0, B, Type = ()>), #key>>::Type
    /// });
    ///
    /// // The entry does not need `Self`: for a marker, the impl is for the marker.
    /// let marker = ImplSelf::Marker(parse_quote!(crate::Marker));
    /// let projection = Projection::new(parse_quote!(crate::Repeater), 42, marker);
    /// let ty = projection.of(&parse_quote!(Self), context, entry).unwrap();
    /// assert_eq!(ty, parse_quote! {
    ///     <crate::Marker as crate::Repeater<(T, *const dyn crate::Repeater<(), 0, B, Type = ()>), #key>>::Type
    /// });
    /// ```
    pub fn of(
        &self,
        self_ty: &Type,
        context: &Context,
        entry: EntryId,
    ) -> std::result::Result<Type, EncodeError> {
        check_repeater_path(&self.path);
        // The impl of an entry that needs `Self` is for what `Self` is in the copy, whatever the
        // Leaker chose; the others for a bucket or the marker.
        let needs_self = context.needs_self();
        let bucket;
        let impl_for = match &self.impl_self {
            ImplSelf::Bucket { .. } if !needs_self => {
                bucket = bucket_type(self.id, entry.index());
                &bucket
            }
            ImplSelf::Marker(marker) if !needs_self => marker,
            _ => self_ty,
        };
        let args = encode_entry_args(
            &context.generics.params,
            (impl_for, &context.self_outlives),
            &self.path,
        )?;
        let path = repeater_trait_path(&self.path, &args, self.id, entry);
        let written: Type = parse_quote!(<#impl_for as #path>::Type);
        match &self.impl_self {
            ImplSelf::Marker(_) if needs_self => Err(EncodeError::NeedsSelf(Box::new(written))),
            _ => Ok(written),
        }
    }
}

// The `KEY` argument of the Repeater for the entry `entry` of the Leaker `id` (`Projection::key()`).
pub(crate) fn key_of(id: u64, entry: EntryId) -> u128 {
    (u128::from(id) << 64) | entry.index() as u128
}

/// Returns the definition of the Repeater trait, the one trait that both crates see.
///
/// Define it once, at the root of the crate that writes the Repeater impls (the one that exports
/// the item), not in the crate of the proc macros: the impls are for any `Self`, which only a
/// local trait allows. The copying crate names it through the crate that defines it.
///
/// `T` is the encoding of the generics of an entry and `KEY` is [`Projection::key()`] for the
/// Leaker's id and the entry. A defaulted const slot per other
/// primitive type (`BOOL`, `CHAR`, `U8`, .., `ISIZE`) carries the other const generics
/// ([`Projection::of()`]), and `Repeater<T, KEY>` still works. `const N: usize` is `[(); N]` and a
/// `u128` uses `KEY`, so if every const parameter is one of these you can write the trait by hand
/// instead: `pub trait Repeater<T: ?Sized, const KEY: u128> { type Type: ?Sized; }`. The generated
/// trait needs Rust 1.59 and the 2018 edition, while `type-leak` supports 1.56.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let def = repeater_definition(&parse_quote!(pub), &parse_quote!(Repeater));
/// assert_eq!(def.ident, "Repeater");
/// // `T`, `KEY` and a slot for each primitive const type but `usize` and `u128`.
/// assert_eq!(def.generics.params.len(), 2 + 12);
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

// Panics unless `repeater` names the trait without arguments: `Repeater`, or `Repeater<>`.
pub(crate) fn check_repeater_path(repeater: &Path) {
    let bare = repeater
        .segments
        .last()
        .map_or(false, |segment| match &segment.arguments {
            PathArguments::None => true,
            PathArguments::AngleBracketed(arguments) => arguments.args.is_empty(),
            PathArguments::Parenthesized(_) => false,
        });
    if !bare {
        panic!(
            "the path of the Repeater names the trait without arguments: `{}`",
            tidy(&repeater.to_token_stream())
        );
    }
}

// The path of the Repeater with `args` as the arguments of its last segment (`Repeater<args>`).
// Panics as `check_repeater_path()` does.
pub(crate) fn repeater_path(repeater: &Path, args: impl ToTokens) -> Path {
    check_repeater_path(repeater);
    let mut path = repeater.clone();
    if let Some(last) = path.segments.last_mut() {
        last.arguments = PathArguments::None;
    }
    parse_quote!(#path<#args>)
}

// The path of the Repeater with the arguments of the entry `entry` of the Leaker `id`:
// `Repeater<args, KEY>` with `KEY` from `key_of(id, entry)`. `args` is the first argument, as
// `encode_entry_args()` writes it.
pub(crate) fn repeater_trait_path(repeater: &Path, args: &Type, id: u64, entry: EntryId) -> Path {
    let key = key_of(id, entry);
    repeater_path(repeater, quote!(#args, #key))
}

/// Encodes the const parameter `value` of type `ty` into the const slots of the Repeater at the
/// path `repeater`: `const N: usize` is `[(); N]`, a `u128` uses `KEY`, and any other primitive type
/// is a trait object of the Repeater with its slot filled. Fails with
/// [`EncodeError::UnsupportedConstType`] if `ty` is not a primitive type (an alias of one is not:
/// the slot cannot be told).
pub(crate) fn encode_slot_const(
    repeater: &Path,
    value: &Ident,
    ty: &Type,
) -> std::result::Result<Type, EncodeError> {
    let prim = const_param_primitive(ty)
        .ok_or_else(|| EncodeError::UnsupportedConstType(Box::new(ty.clone())))?;
    if prim == "usize" {
        return Ok(parse_quote!([(); #value]));
    }
    if prim == "u128" {
        let path = repeater_path(repeater, quote!((), #value, Type = ()));
        return Ok(parse_quote!(*const dyn #path));
    }
    let position = REPEATER_SLOTS
        .iter()
        .position(|(ty, _, _)| *ty == prim)
        .unwrap();
    let defaults = REPEATER_SLOTS[..position]
        .iter()
        .map(|(_, _, default)| -> Expr { parse_str(default).unwrap() });
    let path = repeater_path(repeater, quote!((), 0, #(#defaults,)* #value, Type = ()));
    Ok(parse_quote!(*const dyn #path))
}

/// What the Repeater impls are written for: the [`impl_self`](Projection::impl_self) of the
/// [`Projection`].
///
/// An entry that names `Self`, or whose context's generics do, is always implemented for what
/// `Self` stands for, so the choice only matters for the others. [`ImplSelf::Marker`] has no such
/// impl: [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls) fails if there is
/// one.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImplSelf {
    /// A type parameter that the impl declares (`impl<.., __S: ?Sized> .. for __S`).
    ///
    /// It replaces `Self` where the entry names it, and has the bound of the trait for
    /// [`SelfRequirement::Bound`]. For [`SelfRequirement::Type`] the impl is for that type, and
    /// declares none. It has the name given, unless the entry's type, the generics of its context
    /// or the bound use it: then the first of `__S0`, `__S1`, .. (for `__S`) that they do not.
    /// [`RepeaterImpl::self_ty`] is the name used.
    Param(Ident),
    /// This type, usually a marker of your crate. The impl declares no parameter.
    ///
    /// [`repeater_impls()`](crate::LeakerResult::repeater_impls) fails with
    /// [`EncodeError::NeedsSelf`] if any entry needs `Self`, and [`Projection::of()`] for that entry.
    Marker(Type),
    /// One of 17 primitive types, chosen by the [`id`](Projection::id) and the entry; the entries
    /// that need `Self` get [`ImplSelf::Param`] with `param`.
    ///
    /// Cheaper than a blanket impl when there are many entries. [`Projection::of()`] writes the
    /// same type for such an entry, so the copy and the impl agree.
    Bucket {
        /// The parameter for the entries that need `Self`, named as for [`ImplSelf::Param`].
        param: Ident,
    },
}

/// The pieces of the Repeater impl of an entry, which
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls) returns.
///
/// [`ToTokens`] writes the whole impl, `impl #generics #path for #self_ty { type Type = #ty; }`,
/// with the `KEY` as the last argument of [`path`](RepeaterImpl::path). The `KEY` is
/// [`Projection::key(entry)`](Projection::key) of the [`Projection`] given to `repeater_impls()`,
/// so the caller adds nothing. It does not write [`entry`](RepeaterImpl::entry) or
/// [`module`](RepeaterImpl::module): place the impl in the module, if it is `Some`.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(Rel);));
/// let result = leaker.result();
///
/// let projection = Projection::new(
///     parse_quote!(crate::Repeater),
///     7,
///     ImplSelf::Param(parse_quote!(__S)),
/// );
/// let r = result
///     .repeater_impls(&projection)
///     .unwrap()
///     .remove(0);
///
/// // `ToTokens` writes the `KEY` of the entry 0 of the id 7.
/// let key = projection.key(r.entry);
/// let imp: ItemImpl = parse_quote!(#r);
/// assert_eq!(imp, parse_quote! {
///     impl<__S: ?::core::marker::Sized> crate::Repeater<(), #key> for __S { type Type = Rel; }
/// });
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepeaterImpl {
    /// The entry that the impl is for.
    ///
    /// It is the id that the `KEY` ([`Projection::key()`]) and the copy's side
    /// ([`Projection::of()`], and [`Referrer::visitor()`] through it) use for this entry.
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
    /// The path of the Repeater with the argument of the entry, but not the `KEY`.
    ///
    /// The argument is what [`Projection::of()`] writes for the entry, but with the self type of
    /// the impl for `Self` (`&'a __S`, not `&'a Self`, for `Self: 'a`). [`ToTokens`] appends the
    /// `KEY` ([`Projection::key()`]) to the arguments of the last segment, and nothing if it has
    /// none, so the caller adds nothing.
    ///
    /// ```ignore
    /// // In crate a, with the Repeater at `crate::Repeater`: the entry `Wrap<T>`.
    /// pub struct S<T> { a: Wrap<T> }
    ///
    /// // `path` of its impl. The `KEY` is not in it.
    /// crate::Repeater<(T,)>
    /// ```
    pub path: Path,
    /// The leaked type, which is the `Type` of the impl.
    pub ty: Type,
    /// The module in which to write the impl, if the entry is in one.
    ///
    /// It is the [`ModuleScope::Itself`] of that module, which tells where the module is. It is
    /// `None` if the entry is in no module.
    ///
    /// ```ignore
    /// // In crate a, analyzed with `intern_mod()`: the entry `Rel`.
    /// pub mod outer { pub mod inner { pub struct S { a: Rel } } }
    ///
    /// // `module` of the impl of `Rel`, which is in `inner` (context 1):
    /// Some(Itself { id: 1, path: Some([inner]), hint: None, parent: Some(0) })
    /// ```
    pub module: Option<ModuleScope>,
    /// The [`id`](Projection::id) of the [`Projection`] given to `repeater_impls()`: the high bits
    /// of the `KEY`.
    pub(crate) id: u64,
}

/// Writes `impl #generics #path_with_key for #self_ty #where_clause { type Type = #ty; }`.
///
/// `path_with_key` is [`path`](RepeaterImpl::path) with the `KEY` ([`Projection::key()`]) added as
/// its last argument, for the [`Projection`] given to
/// [`repeater_impls()`](crate::LeakerResult::repeater_impls). The impl is not wrapped in
/// [`module`](RepeaterImpl::module).
impl ToTokens for RepeaterImpl {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let imp: ItemImpl = self.clone().into();
        imp.to_tokens(tokens);
    }
}

impl RepeaterImpl {
    /// `path` with the `KEY` of the entry as its last argument.
    fn path_with_key(&self) -> Path {
        let key = key_of(self.id, self.entry);
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
/// leaker.intern_struct(&parse_quote!(struct S(Rel);));
/// let result = leaker.result();
/// let projection = Projection::new(
///     parse_quote!(crate::Repeater),
///     7,
///     ImplSelf::Marker(parse_quote!(Marker)),
/// );
/// let r = result
///     .repeater_impls(&projection)
///     .unwrap()
///     .remove(0);
/// let key = projection.key(r.entry);
/// let expected: ItemImpl =
///     parse_quote!(impl crate::Repeater<(), #key> for Marker { type Type = Rel; });
/// assert_eq!(ItemImpl::from(r), expected);
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

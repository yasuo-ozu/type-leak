//! The Repeater trait ([`repeater_definition()`], its const slots) and both ends of a path through
//! it: the [`Projection`] that a copy writes for an entry, and the entry's impl ([`RepeaterImpl`],
//! [`ImplSelf`]).

use crate::context::{Context, SelfRequirement};
use crate::encode::{encode_entry_args, without_encoded_bounds, EncodeError};
use crate::referrer::EntryId;
#[cfg(doc)]
use crate::referrer::Referrer;
use crate::util::{
    ident_name, is_named, names_in, parenthesize_leading_qualified_path, same_name, tidy,
    unwrap_type_group, ReplaceSelf,
};
use proc_macro2::{Span, TokenStream};
use std::borrow::Cow;
use syn::fold::Fold;
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

/// A primitive that can type a const parameter, so it is context-free.
pub(crate) fn is_const_param_ty(ty: &Type) -> bool {
    const_param_primitive(ty).is_some()
}

/// The primitive type that `ty` names if a const parameter can have it on stable Rust: one with a
/// slot, `usize` or `u128`.
fn const_param_primitive(ty: &Type) -> Option<&'static str> {
    match unwrap_type_group(ty) {
        Type::Path(TypePath { qself: None, path }) => {
            if path.segments.iter().any(|seg| !seg.arguments.is_none()) {
                return None;
            }
            let segs: Vec<_> = path
                .segments
                .iter()
                .map(|seg| ident_name(&seg.ident))
                .collect();
            let name = match (path.leading_colon.is_some(), segs.as_slice()) {
                (false, [s]) => s,
                (_, [c, p, s]) if (c == "core" || c == "std") && p == "primitive" => s,
                _ => return None,
            };
            let slots = REPEATER_SLOTS.iter().map(|(prim, _, _)| *prim);
            ["usize", "u128"]
                .iter()
                .copied()
                .chain(slots)
                .find(|prim| *prim == name.as_str())
        }
        _ => None,
    }
}

/// How both crates name the Repeater of one Leaker: its path, the Leaker's id, and the type of the
/// impls.
///
/// The macro of crate A gives it to
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls). The macro of crate B
/// gives it to [`Referrer::visitor()`] and [`Referrer::complete_args()`]. Both must use the same
/// `id` and `impl_self`, and a `path` that names the same trait. The `path` must have no
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
    /// The path of the Repeater trait ([`repeater_definition()`]), without arguments.
    ///
    /// ```ignore
    /// // In crate a, at its root:
    /// pub trait Repeater<T: ?Sized, const KEY: u128, ..> { type Type: ?Sized; }
    ///
    /// // `path` in the macro of crate a, and in the macro of crate b:
    /// crate::Repeater
    /// ::a::Repeater
    /// ```
    pub path: Path,
    /// The id of the Leaker: the high 64 bits of each `KEY`.
    ///
    /// ```ignore
    /// // Give each Leaker of crate a its own id, so that no two entries have the same `KEY`:
    /// 7
    /// ```
    pub id: u64,
    /// The type that the Repeater impls are for ([`ImplSelf`]).
    ///
    /// ```ignore
    /// // The impls are for any type:
    /// impl<__S: ?Sized> crate::Repeater<(), KEY> for __S { type Type = Rel; }
    ///
    /// // `impl_self`:
    /// ImplSelf::Param(__S)
    /// ```
    pub impl_self: ImplSelf,
    // Set by `Projection::unused()`: it writes no entry, so a Referrer with entries refuses it.
    unused: bool,
}

impl Projection {
    /// Returns a projection with these fields.
    ///
    /// # Panics
    ///
    /// If the last segment of `path` has arguments (`crate::Repeater<u8>`). An empty `<>` is
    /// accepted.
    #[must_use]
    pub fn new(path: Path, id: u64, impl_self: ImplSelf) -> Self {
        check_repeater_path(&path);
        Projection {
            path,
            id,
            impl_self,
            unused: false,
        }
    }

    /// Returns a projection for a Referrer with no entries.
    ///
    /// Use it when crate A writes no Repeater impls, for example to call
    /// [`Referrer::complete_args()`] for an item with no entries. The fields are placeholders: the
    /// path `Repeater`, the id `0` and `ImplSelf::Marker(())`.
    ///
    /// With this projection, [`Referrer::visitor()`], [`Referrer::complete_args()`] and
    /// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls) panic if the
    /// Referrer has an entry. [`of()`](Projection::of) and [`key()`](Projection::key) always
    /// panic.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// // `u8` is a primitive type, so `Tr` has no entries.
    /// let item: ItemTrait = parse_quote!(trait Tr<K, D = u8> { fn get(&self, key: K) -> D; });
    /// let mut leaker = Leaker::with_rule(Rules::new().primitive());
    /// let top_level = leaker.intern_trait(&item);
    /// let result = leaker.result();
    /// let referrer = result.clone().finish().unwrap();
    /// assert!(referrer.is_empty());
    /// let projection = Projection::unused();
    /// assert!(result.repeater_impls(&projection).unwrap().is_empty());
    ///
    /// // `impl Tr<char> for W`: the default `D` is `u8`.
    /// let given: Vec<GenericArgument> = vec![parse_quote!(char)];
    /// let mut substitute = referrer.complete_args(top_level, &given, &projection).unwrap();
    /// let mut sig: Signature = parse_quote!(fn get(&self, key: K) -> D);
    /// substitute.visit_signature_mut(&mut sig);
    /// assert_eq!(sig, parse_quote!(fn get(&self, key: char) -> u8));
    ///
    /// // The visitor leaves a copy as it is.
    /// let mut imp: ItemImpl = parse_quote! {
    ///     impl Tr<char> for W { fn get(&self, key: char) -> u8 { todo!() } }
    /// };
    /// let before = imp.clone();
    /// referrer.visitor(top_level, &projection).visit_item_impl_mut(&mut imp);
    /// assert_eq!(imp, before);
    /// ```
    #[must_use]
    pub fn unused() -> Self {
        Projection {
            path: parse_quote!(Repeater),
            id: 0,
            impl_self: ImplSelf::Marker(parse_quote!(())),
            unused: true,
        }
    }

    // Panics if this is `Projection::unused()` and `has_entries`: the Referrer needs the
    // projection of crate A. The methods of the Referrer call it before anything else.
    pub(crate) fn check_entries(&self, has_entries: bool) {
        if self.unused && has_entries {
            panic!(
                "the Referrer has entries: pass the Projection of crate A, not Projection::unused()"
            );
        }
    }

    // Panics if this is `Projection::unused()`: it has no Repeater to write an entry with.
    fn check_used(&self) {
        if self.unused {
            panic!(
                "Projection::unused() has no Repeater to write an entry with: pass the Projection \
                 of crate A"
            );
        }
    }

    /// Returns the `KEY` argument of the Repeater for `entry`.
    ///
    /// It is [`id`](Projection::id) in the high 64 bits and the [index](EntryId::index) of `entry`
    /// in the low 64 bits.
    ///
    /// # Panics
    ///
    /// If this is [`Projection::unused()`].
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
        self.check_used();
        key_of(self.id, entry)
    }

    /// Returns the type that stands for `entry` in the copy: `<S as Repeater<ARGS, KEY>>::Type`.
    ///
    /// - `self_ty` is what `Self` means in the copy: `Self` in an impl, or a host type.
    /// - `context` is the [`Context`] of the entry ([`Referrer::iter()`] gives it).
    /// - `S` is the type that the impl of the entry is for. It is `self_ty`, except for an entry
    ///   that does not need `Self`: then it is the bucket type of [`ImplSelf::Bucket`], or the type
    ///   of [`ImplSelf::Marker`].
    /// - `KEY` is [`key(entry)`](Projection::key).
    /// - `ARGS` is a tuple of the generics of the context, each by its own name: `'a` is `&'a ()`,
    ///   and `T` is `T` (`*const T` if `T: ?Sized`). A const uses the const slots of the Repeater
    ///   ([`repeater_definition()`]). Each bound `T: 'a`, `'b: 'a` or `Self: 'a` adds `&'a T`,
    ///   `&'a &'b ()` or `&'a S`.
    ///
    /// [`Referrer::visitor()`] calls it for each entry. At a use site,
    /// [`Substitute`](crate::Substitute) then writes the arguments of the site into `ARGS`.
    ///
    /// # Errors
    ///
    /// - [`EncodeError::UnsupportedConstType`]: the type of a const parameter is not a primitive
    ///   type. A type alias is refused too: its const slot cannot be known.
    /// - [`EncodeError::UnboundedTraitObject`]: `S` is a trait object with no lifetime bound
    ///   (`dyn Tr`), and the entry needs `Self: 'a`.
    /// - [`EncodeError::NeedsSelf`]: `impl_self` is [`ImplSelf::Marker`], and the entry needs
    ///   `Self`. The error holds the type written for `self_ty`.
    ///
    /// # Panics
    ///
    /// If [`path`](Projection::path) has arguments, or this is [`Projection::unused()`].
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&parse_quote! {
    ///     trait Tr<T, const B: bool> { fn f(&self) -> Wrap<T, B>; }
    /// });
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
    /// let args: Type = parse_quote!((T, *const dyn crate::Repeater<(), 0, B, Type = ()>));
    /// assert_eq!(ty, parse_quote!(<Self as crate::Repeater<#args, #key>>::Type));
    ///
    /// // The entry does not need `Self`: for a marker, the impl is for the marker.
    /// let marker = ImplSelf::Marker(parse_quote!(crate::Marker));
    /// let projection = Projection::new(parse_quote!(crate::Repeater), 42, marker);
    /// let ty = projection.of(&parse_quote!(Self), context, entry).unwrap();
    /// assert_eq!(ty, parse_quote!(<crate::Marker as crate::Repeater<#args, #key>>::Type));
    /// ```
    pub fn of(
        &self,
        self_ty: &Type,
        context: &Context,
        entry: EntryId,
    ) -> std::result::Result<Type, EncodeError> {
        self.check_used();
        check_repeater_path(&self.path);
        // The impl of an entry that needs `Self` is for what `Self` is in the copy, whatever the
        // Leaker chose; the others for a bucket or the marker.
        let needs_self = context.needs_self();
        let target = self.impl_self.for_entry(self.id, entry, needs_self);
        let impl_for = match &*target {
            ImplSelf::Marker(marker) if !needs_self => marker,
            _ => self_ty,
        };
        let args = encode_entry_args(
            &context.generics.params,
            impl_for,
            &context.self_outlives,
            &self.path,
        )?;
        let key = key_of(self.id, entry);
        let path = repeater_path(&self.path, quote!(#args, #key));
        let written: Type = parse_quote!(<#impl_for as #path>::Type);
        match &self.impl_self {
            ImplSelf::Marker(_) if needs_self => Err(EncodeError::NeedsSelf(Box::new(written))),
            _ => Ok(written),
        }
    }
}

/// The `KEY` of the Repeater for the entry `entry` of the Leaker `id` (`Projection::key()`).
fn key_of(id: u64, entry: EntryId) -> u128 {
    (u128::from(id) << 64) | entry.index() as u128
}

/// The primitive type of the entry `entry` for `id`: 17 of them, chosen by a hash so that ids and
/// indices that are close spread. `ImplSelf::Bucket` of the Leaker's macro and the paths of the
/// Referrer's must agree on it, so it is part of [`Referrer::FORMAT_VERSION`].
fn bucket_type(id: u64, entry: usize) -> Type {
    const BUCKETS: [&str; 17] = [
        "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
        "i128", "isize", "f32", "f64", "str",
    ];
    // The finalizer of splitmix64.
    let mut z = id
        .wrapping_add((entry as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
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

/// Returns the definition of the Repeater trait, named `ident`, with the visibility `vis`.
///
/// Define it once, at the root of crate A (the crate that defines the item and has the impls).
/// Do not define it in the proc-macro crate: the impls are for any `Self`, which Rust allows only
/// for a local trait. Crate B names it through crate A.
///
/// - `T` holds the generics of an entry ([`Projection::of()`]).
/// - `KEY` is [`Projection::key()`] of the entry.
/// - The other parameters are const slots with defaults, one for each primitive type except
///   `usize` and `u128` (`BOOL`, `CHAR`, `U8`, .., `ISIZE`). They carry the const generics of
///   their types. A `usize` is `[(); N]` in `T`, and a `u128` uses the `KEY` slot.
///
/// If every const generic is a `usize` or a `u128`, the short form is enough:
/// `pub trait Repeater<T: ?Sized, const KEY: u128> { type Type: ?Sized; }`.
///
/// The generated trait needs Rust 1.59 and the 2018 edition. `type-leak` itself supports 1.56.
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

/// Panics unless `repeater` names the trait without arguments: `Repeater`, or `Repeater<>`.
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

/// The path of the Repeater with `args` as the arguments of its last segment (`Repeater<args>`).
/// Panics as `check_repeater_path()` does.
fn repeater_path(repeater: &Path, args: impl ToTokens) -> Path {
    check_repeater_path(repeater);
    let mut path = repeater.clone();
    if let Some(last) = path.segments.last_mut() {
        last.arguments = PathArguments::None;
    }
    parse_quote!(#path<#args>)
}

/// Encodes the const parameter `value` of type `ty` for the Repeater at `repeater`: `[(); N]` for a
/// `usize`, `KEY` for a `u128`, else a trait object of the Repeater with the slot of `ty` filled.
/// Fails with [`EncodeError::UnsupportedConstType`] for a type that is no primitive, an alias too.
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
    // `const_param_primitive()` takes the other names from `REPEATER_SLOTS`.
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

/// The type that the Repeater impls are for ([`Projection::impl_self`]).
///
/// The impl of an entry that needs `Self` (the entry or its context names `Self`) is always for
/// what `Self` means. So the choice matters only for the other entries.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImplSelf {
    /// A type parameter that each impl declares.
    ///
    /// `ImplSelf::Param(__S)` gives `impl<__S: ?Sized> Repeater<(), KEY> for __S { .. }` for the
    /// entry `Rel`. The parameter takes the place of `Self`. For [`SelfRequirement::Bound`] it has
    /// the trait as its bound. For [`SelfRequirement::Type`] the impl is for that type, with no
    /// parameter. If the impl already uses the name, the parameter is the first free one of
    /// `__S0`, `__S1`, .. ([`RepeaterImpl::self_ty`] has the name).
    Param(Ident),
    /// A type of your crate, usually a marker. The impls declare no parameter.
    ///
    /// `ImplSelf::Marker(crate::Marker)` gives `impl Repeater<(), KEY> for crate::Marker { .. }`
    /// for the entry `Rel`. If an entry needs `Self`,
    /// [`repeater_impls()`](crate::LeakerResult::repeater_impls) fails with
    /// [`EncodeError::NeedsSelf`], and so does [`Projection::of()`] for that entry.
    Marker(Type),
    /// One of 17 primitive types, chosen from the [`id`](Projection::id) and the entry.
    ///
    /// `ImplSelf::Bucket { param: __S }` gives, for example,
    /// `impl Repeater<(), KEY> for ::core::primitive::bool { .. }` for the entry `Rel`. An entry
    /// that needs `Self` gets [`ImplSelf::Param`] with `param`. With many entries, rustc checks
    /// these impls faster than impls for a parameter. [`Projection::of()`] writes the same
    /// primitive type, so the copy and the impl agree.
    Bucket {
        /// The parameter for the entries that need `Self`, as in [`ImplSelf::Param`].
        param: Ident,
    },
}

impl ImplSelf {
    /// What the impl of the entry `entry` of the Leaker `id` is for: [`ImplSelf::Bucket`] is the
    /// bucket type as a marker, or [`ImplSelf::Param`] with `param` for an entry that `needs_self`.
    fn for_entry(&self, id: u64, entry: EntryId, needs_self: bool) -> Cow<'_, Self> {
        match self {
            ImplSelf::Bucket { param } if needs_self => Cow::Owned(ImplSelf::Param(param.clone())),
            ImplSelf::Bucket { .. } => Cow::Owned(ImplSelf::Marker(bucket_type(id, entry.index()))),
            target => Cow::Borrowed(target),
        }
    }
}

/// The Repeater impl of one entry, in parts.
///
/// [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls) returns them.
/// [`ToTokens`] and `ItemImpl::from()` write the whole impl:
/// `impl #generics #path for #self_ty #where { type Type = #ty; }`. They add the `KEY`
/// ([`Projection::key()`] of [`entry`](RepeaterImpl::entry)) as the last argument of
/// [`path`](RepeaterImpl::path).
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
    /// ```ignore
    /// // In crate a: `Rel` is the entry 0.
    /// pub struct S { a: Rel }
    ///
    /// // `entry` of the impl of `Rel`:
    /// EntryId(0)
    /// ```
    pub entry: EntryId,
    /// The generics of the impl, with its where clause.
    ///
    /// Lifetimes come first, then types (the parameter for `Self` last), then consts, as rustc
    /// before 1.59 requires.
    pub generics: Generics,
    /// The type that the impl is for.
    pub self_ty: Type,
    /// The path of the Repeater with the first argument (`ARGS` of [`Projection::of()`]), without
    /// the `KEY`.
    ///
    /// In the argument, `Self` is the self type of the impl (`&'a __S` for `Self: 'a`).
    ///
    /// ```ignore
    /// // In crate a, with the Repeater at `crate::Repeater`: the entry `Wrap<T>`.
    /// pub struct S<T> { a: Wrap<T> }
    ///
    /// // `path` of its impl:
    /// crate::Repeater<(T,)>
    /// ```
    pub path: Path,
    /// The leaked type: the `Type` of the impl.
    pub ty: Type,
    /// The [`id`](Projection::id) of the [`Projection`] given to `repeater_impls()`: the high bits
    /// of the `KEY`.
    pub(crate) id: u64,
}

/// Writes the whole impl, with the `KEY` (see [`RepeaterImpl`]).
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

/// Builds the whole impl, with the `KEY` (see [`RepeaterImpl`]).
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

impl Projection {
    /// The Repeater impl of `entry`, whose type is `entry_ty` and whose context is `context`:
    /// `Referrer::repeater_impls()` makes one for each entry, after `check_marker()`.
    pub(crate) fn impl_of(
        &self,
        entry: EntryId,
        entry_ty: &Type,
        context: &Context,
    ) -> std::result::Result<RepeaterImpl, EncodeError> {
        // A bucket is the type that `Projection::of()` writes for an entry that does not need
        // `Self`.
        let target = self
            .impl_self
            .for_entry(self.id, entry, context.needs_self());
        // The type the impl is for, `Self` as the impl writes it, and the parameter to declare with
        // its bound.
        let (self_ty, self_in_impl, param): (Type, Type, Option<(Ident, Option<&Path>)>) =
            match (context.self_ty.as_ref(), &*target) {
                // The impl of a struct, an enum or a union declares no parameter.
                (Some(SelfRequirement::Type(ty)), ImplSelf::Param(_)) => {
                    (ty.clone(), parse_quote!(Self), None)
                }
                (Some(_), ImplSelf::Marker(_)) => unreachable!("rejected by check_marker()"),
                (None, ImplSelf::Marker(marker)) => (marker.clone(), marker.clone(), None),
                (requirement, ImplSelf::Param(ident)) => {
                    let ident = self_param(entry_ty, context, ident, &self.path);
                    // Any type that the bounds hold for (`SelfRequirement::Any`), or any type at
                    // all: the parameter has no bound of its own.
                    let bound = match requirement {
                        Some(SelfRequirement::Bound(path)) => Some(path),
                        _ => None,
                    };
                    (
                        parse_quote!(#ident),
                        parse_quote!(#ident),
                        Some((ident, bound)),
                    )
                }
                (_, ImplSelf::Bucket { .. }) => unreachable!("resolved above"),
            };
        // The implementor is a parameter, where `Self` of the trait is not available: the type and
        // the generics of the context have `self_in_impl` for it before the bounds are encoded.
        let (ty, generics) = match context.self_ty.as_ref() {
            Some(SelfRequirement::Bound(_) | SelfRequirement::Any) => {
                let mut replace = ReplaceSelf {
                    self_ty: &self_in_impl,
                };
                let ty = replace.fold_type(entry_ty.clone());
                (ty, replace.fold_generics(context.generics.clone()))
            }
            _ => (entry_ty.clone(), context.generics.clone()),
        };
        let args = encode_entry_args(
            &generics.params,
            &self_in_impl,
            &context.self_outlives,
            &self.path,
        )?;
        let mut declared = without_encoded_bounds(&generics);
        if let Some((ident, bound)) = param {
            // The `Self` of a struct, an enum or a union is a `Sized` type, and the bounds may need
            // it (`(Self, u8)`, `Vec<Self>`), so the parameter of `SelfRequirement::Any` is
            // `Sized` where they do. Else it is `?Sized`: rustc before 1.70 cannot prove `S<T>:
            // Sized` for the copy of a struct whose last field is the path through this impl, if
            // the impl needs `__S: Sized` (E0275: the type needs the impl, which needs the type).
            let sized = matches!(context.self_ty, Some(SelfRequirement::Any))
                && needs_sized(&declared, &ty, &ident);
            let declaration = match (bound, sized) {
                (Some(path), _) => parse_quote!(#ident: ?::core::marker::Sized + #path),
                (None, true) => parse_quote!(#ident),
                (None, false) => parse_quote!(#ident: ?::core::marker::Sized),
            };
            // Before the first const: rustc before 1.59 wants the types before the consts.
            let at = declared
                .params
                .iter()
                .position(|param| matches!(param, GenericParam::Const(_)))
                .unwrap_or(declared.params.len());
            declared.params.insert(at, declaration);
            // The generics of an entry with none have no angle brackets, which `Generics` compares.
            declared.lt_token.get_or_insert_with(Default::default);
            declared.gt_token.get_or_insert_with(Default::default);
        }
        // rustc reads a first predicate on `<__S>::A` as generics of the `where`.
        if let Some(clause) = &mut declared.where_clause {
            parenthesize_leading_qualified_path(clause);
        }
        Ok(RepeaterImpl {
            entry,
            generics: declared,
            self_ty,
            path: repeater_path(&self.path, &args),
            ty,
            id: self.id,
        })
    }
}

/// The parameter that stands for `Self` in the impl of an entry of the type `ty` and the context
/// `context`: `ident`, unless the type, the generics of the context, its bound or the start of the
/// path `repeater` use that name (E0403, or taken for that generic), else the first of `ident0`,
/// `ident1`, .. that they do not.
fn self_param(ty: &Type, context: &Context, ident: &Ident, repeater: &Path) -> Ident {
    let generics = &context.generics;
    let where_clause = &generics.where_clause;
    let bound = match &context.self_ty {
        Some(SelfRequirement::Bound(path)) => Some(path),
        _ => None,
    };
    // Every identifier counts, those of the tokens of a macro too. A lifetime has its own
    // names, as `'__S`.
    let mut used = names_in(quote!(#ty #generics #where_clause #bound));
    // The parameter hides a module that the path of the Repeater starts with. A path that starts
    // with `::` names a crate, which a parameter does not hide.
    if repeater.leading_colon.is_none() {
        used.extend(
            repeater
                .segments
                .first()
                .map(|first| ident_name(&first.ident)),
        );
    }
    let base = ident_name(ident);
    if !used.contains(&base) {
        return ident.clone();
    }
    (0..)
        .map(|n| format!("{}{}", base, n))
        .find(|name| !used.contains(name))
        .map(|name| Ident::new(&name, ident.span()))
        .unwrap()
}

/// Returns `true` if the impl may need `param: Sized`, and `false` if it is well-formed without:
/// it errs on the side of `Sized`. `declared` are the generics of the impl without `param`, which
/// replaced `Self` in them, and `ty` is the type of the impl.
fn needs_sized(declared: &Generics, ty: &Type, param: &Ident) -> bool {
    use syn::visit::Visit;
    struct Uses<'a> {
        param: &'a Ident,
        needs_sized: bool,
    }
    impl Uses<'_> {
        fn is_param(&self, ty: &Type) -> bool {
            matches!(unwrap_type_group(ty), Type::Path(TypePath { qself: None, path })
                if path.get_ident().map_or(false, |ident| same_name(ident, self.param)))
        }
    }
    impl<'ast> Visit<'ast> for Uses<'_> {
        // A use that the methods below do not skip may need `Sized`: `(__S, u8)`, `[__S]`,
        // `Rel2<__S, T>` (unless `Rel2` relaxes it, which is not seen), `<__S>::A`.
        fn visit_ident(&mut self, ident: &'ast Ident) {
            if same_name(ident, self.param) {
                self.needs_sized = true;
            }
        }
        // Tokens that are not parsed: any `Self` is out of reach.
        fn visit_macro(&mut self, _: &'ast Macro) {
            self.needs_sized = true;
        }
        // `&'a __S` and `*const __S` need no `Sized`.
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
        // Nor does a direct argument of a path that accepts an unsized one (`AsRef<__S>`).
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
        // Nor does the bounded type of a predicate that does not say `Sized` (`__S: 'a`).
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
                    }) if path
                        .segments
                        .last()
                        .map_or(false, |s| is_named(&s.ident, "Sized")) =>
                    {
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
/// is `?Sized` (`AsRef<__S>` is well-formed with `__S: ?Sized`), absolute or a prelude name.
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
    // Taken for what it names in the prelude, which the Leaker accepts only if the prelude is
    // allowed; an item of the same name in the module is not told apart.
    const PRELUDE: &[&str] = &["AsRef", "AsMut", "PartialEq", "PartialOrd", "Box"];
    let names: Vec<String> = path.segments.iter().map(|s| ident_name(&s.ident)).collect();
    let names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    match (path.leading_colon.is_some(), &names[..]) {
        (false, [name]) => PRELUDE.contains(name),
        (true, [root, module, item]) => ABSOLUTE
            .iter()
            .any(|(roots, m, items)| roots.contains(root) && m == module && items.contains(item)),
        _ => false,
    }
}

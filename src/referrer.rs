//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::analysis::{Anchoring, PartItem, SignatureFacts, SignatureMark};
use crate::context::{Context, ContextId, Enclosing, Gat, Scope, SelfRequirement};
use crate::elided::{after_fold, before_fold, FnSig};
use crate::encode::{complete_params, encode_self_outlives, EncodeError, Substitute};
use crate::error::Cause;
use crate::graph::TypeKey;
#[cfg(doc)]
use crate::repeater::repeater_definition;
use crate::repeater::{check_repeater_path, ImplSelf, Projection, RepeaterImpl};
use crate::util::{
    bound_cause, declared_names, generic_param_name, has_lifetime, ident_name, mentions_idents,
    parenthesize_bounded, parenthesize_leading_qualified_path, path_type, type_mentions_self,
    ReplaceSelf, Sorted,
};
use proc_macro2::{Literal, Punct, Spacing, Span, TokenStream};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use syn::ext::IdentExt;
use syn::fold::Fold;
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};

/// The leaked types of an analyzed item (its *entries*, each with an [`EntryId`]).
///
/// [`Leaker::result()`](crate::Leaker::result) returns it, in a
/// [`LeakerResult`](crate::LeakerResult). Crate A, which defines the item, emits the Repeater impl
/// of each entry ([`repeater_impls()`](crate::LeakerResult::repeater_impls)). Crate B, which copies
/// the item, replaces each entry by `<Self as Repeater<ARGS, KEY>>::Type` with
/// [`visitor()`](Referrer::visitor) ([`Projection::of()`](crate::Projection::of) writes it: `ARGS`
/// from the generics of the entry, `KEY` from [`Projection::key()`](crate::Projection::key)).
///
/// A's macro passes the `Referrer` and the top-level [`ContextId`] to B's as tokens ([`ToTokens`],
/// [`Parse`]); both crates must use the same version of `type-leak`. Emit an impl for every entry.
/// A type of another crate (`::foo::X`) is an entry like the others, unless the Leaker's rule allows
/// its path ([`Rules::external_crates()`](crate::Rules::external_crates)): it is no entry then, and
/// the copy writes it as it is. `examples/two-macros` is a complete pair.
///
/// An [`EntryId`] that does not come from this Referrer ([`iter()`](Referrer::iter) gives the
/// valid ones) may be out of range: the methods that take one then panic.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use quote::ToTokens;
///
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S<T>(Wrap<T>);));
/// let referrer = leaker.result().finish().unwrap();
///
/// // Crate A writes the Referrer into the input of the macro of crate B, which reads it back.
/// let tokens = referrer.to_token_stream();
/// assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
/// ```
#[derive(Clone, Debug)]
pub struct Referrer {
    /// The type of each entry and the id of its context (a key of `contexts`), by index.
    entries: Vec<(Type, ContextId)>,
    /// The indices of the entries that
    /// [`LeakerResult::ignore()`](crate::LeakerResult::ignore) removed. They keep their place, so
    /// the others keep their indices. They are no entries: the copy writes their types as written.
    removed: BTreeSet<usize>,
    /// Every context, by id.
    contexts: BTreeMap<ContextId, Context>,
    /// The indices of the entries, keyed by tokens as in the Leaker, so a type in an invisible
    /// group is found too.
    by_tokens: Sorted<HashMap<TypeKey, Vec<usize>>>,
    /// The types and bounds copied as written ([`PartItem`]). A part that is not whole, with no
    /// anchor and no mark, is there for an error and inert until `LeakerResult::ignore()`.
    /// Serialized after the format tag.
    parts: Vec<PartItem>,
    /// The indices in `parts`, by the part.
    parts_by_cause: Sorted<HashMap<Cause, Vec<usize>>>,
    /// The indices in `parts` of the parts with an anchor (`within`), by the anchor, as a
    /// `Cause::Type`.
    parts_by_anchor: Sorted<HashMap<Cause, Vec<usize>>>,
}

/// The maps are made from the entries and the parts (`Referrer::new()`), so they are not compared.
impl PartialEq for Referrer {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            entries,
            removed,
            contexts,
            parts,
            ..
        } = self;
        *entries == other.entries
            && *removed == other.removed
            && *contexts == other.contexts
            && *parts == other.parts
    }
}

impl Eq for Referrer {}

/// The index of an entry of a [`Referrer`], counted from 0 in the order of the entries.
///
/// [`Referrer::iter()`] gives it, and so does [`RepeaterImpl::entry`](crate::RepeaterImpl::entry)
/// of the impls that [`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls)
/// returns. The methods of [`Projection`] ([`key()`](crate::Projection::key),
/// [`of()`](crate::Projection::of)) take it. It cannot be made from a number, except by `Parse`.
///
/// `ToTokens` writes it as a `usize` literal (`3usize`), so that `P<#ix>` is `P<3usize>`. `Parse`
/// reads an integer with no suffix or with `usize`.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use quote::ToTokens;
///
/// // A macro gets an `EntryId` from `Referrer::iter()`; `Parse` makes one too.
/// let ix: EntryId = parse_quote!(1);
/// assert_eq!(ix.index(), 1);
/// assert_eq!(ix.to_token_stream().to_string(), "1usize");
/// assert_eq!(parse_str::<EntryId>("1usize").unwrap(), ix);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntryId(usize);

impl EntryId {
    fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the number of the entry, counted from 0 in the order of the entries.
    pub fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for EntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl ToTokens for EntryId {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::usize_suffixed(self.0).to_tokens(tokens)
    }
}

impl Parse for EntryId {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let lit: LitInt = input.parse()?;
        if !matches!(lit.suffix(), "" | "usize") {
            return Err(Error::new(
                lit.span(),
                "an entry id is an integer, with no suffix or `usize`",
            ));
        }
        Ok(Self(lit.base10_parse()?))
    }
}

/// How the visitor looks up the default of a type parameter, while it visits the generics of an
/// item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Defaults {
    /// Not in such generics: a type is looked up as anywhere.
    Plain,
    /// Without `Self` in the item's own scope (the scopes around keep theirs), as the Leaker
    /// analyzes a default. A nested struct, enum or union also drops its bounds that name its
    /// `Self`: its copy cannot write `Self` in a default (E0735).
    WithoutSelf,
    /// The generics of a trait nested in the visited syntax: the Leaker analyzed a default without
    /// `Self`, or with the trait's `Self` if it names it (`Rhs = Self`).
    OwnTrait,
    /// Visiting a default of `OwnTrait`: the entry analyzed without `Self` first, else the one of
    /// the trait's `Self`.
    OwnTraitDefault,
}

/// How [`Referrer::find()`] compares the context of a type with those of the entries.
///
/// The context of the type, computed from the generics in scope and `self_ty`, must be the entry's.
#[derive(Clone, Copy)]
struct Matching<'a> {
    /// What `Self` means here.
    self_ty: Option<&'a SelfRequirement>,
    /// `Self` means something else here, so no entry that names it matches.
    no_self: bool,
    /// `self_ty` is the `Self` of a struct, an enum or a union nested in the visited syntax, so a
    /// type that does not name it but has a bound that does is for any type
    /// ([`SelfRequirement::Any`]).
    nested_type: bool,
    /// The type is the default of a parameter of a nested trait, whose entry is the one analyzed
    /// without `Self` if there is one (see [`Defaults::OwnTrait`]).
    without_self_first: bool,
}

/// Where the visited syntax is in the item, for the parts: among its predicates, in an argument
/// of a function, in a part.
#[derive(Clone, Copy, Debug, Default)]
struct PartPlace {
    /// In a bound or a where predicate of the item, or in what is in them.
    in_predicate: bool,
    /// In an argument of a function, whose `impl Trait` bounds are predicates.
    in_fn_arg: bool,
    /// In a part: it is left as written, so no entry is looked up.
    protected: bool,
}

/// The generics in scope of a signature that a copy writes, for the scan of its inputs for the
/// occurrences of a lifetime
/// ([`unconstrained_in_copy()`](crate::analysis::unconstrained_in_copy)).
struct SignatureScopes<'a>(&'a [Scope]);

impl Anchoring for SignatureScopes<'_> {
    fn is_projection(&self, path: &Path) -> bool {
        path.segments.len() > 1
            && path.leading_colon.is_none()
            && matches!(path.segments.first(), Some(seg)
                if seg.ident == "Self"
                    || self.0.iter().any(|scope| scope.declares(&ident_name(&seg.ident))))
    }

    /// The copy replaces every entry, so no type is left to the Referrer.
    fn is_optional_type(&self, _: &Type) -> bool {
        false
    }
}

/// The type `()`, which stands for a type that is replaced where only what is left of the types
/// around matters.
fn unit_type() -> Type {
    Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: Punctuated::new(),
    })
}

/// How `Referrer::find()` and `Referrer::part_of()` look a type up.
#[derive(Clone, Copy)]
struct Lookup<'a> {
    /// How the context of the type is compared with the entries'.
    matching: Matching<'a>,
    /// The entries that a visited item hides (see `Visitor::shadow_items()`).
    hidden: &'a HashSet<usize>,
    /// Where the visited syntax is among the predicates of the item.
    place: PartPlace,
    /// The signature that the visited syntax is in, for the parts that a signature made.
    signature: Option<&'a SignatureFacts>,
}

impl Referrer {
    /// Adds the context 0 if there is none: the serialized form needs a group
    /// (`Leaker::new().result()` writes `"tag" ()`).
    pub(crate) fn new(
        entries: impl IntoIterator<Item = (ContextId, Type)>,
        removed: BTreeSet<usize>,
        mut contexts: BTreeMap<ContextId, Context>,
        parts: Vec<PartItem>,
    ) -> Self {
        if contexts.is_empty() {
            contexts.insert(ContextId::new(0), Context::none());
        }
        let mut referrer = Self {
            entries: Vec::new(),
            removed,
            contexts,
            by_tokens: Sorted::default(),
            parts,
            parts_by_cause: Sorted::default(),
            parts_by_anchor: Sorted::default(),
        };
        for (ix, item) in referrer.parts.iter().enumerate() {
            let PartItem { part, within, .. } = item;
            referrer
                .parts_by_cause
                .entry(part.clone())
                .or_default()
                .push(ix);
            if let Some((anchor, _)) = within {
                let anchor = Cause::Type(anchor.clone());
                referrer.parts_by_anchor.entry(anchor).or_default().push(ix);
            }
        }
        for (ix, (context, ty)) in entries.into_iter().enumerate() {
            referrer
                .by_tokens
                .entry(TypeKey::new(&ty))
                .or_default()
                .push(ix);
            referrer.entries.push((ty, context));
        }
        referrer
    }

    /// Returns `true` if there are no entries. An entry that
    /// [`LeakerResult::ignore()`](crate::LeakerResult::ignore) removed is none.
    pub fn is_empty(&self) -> bool {
        self.entries.len() == self.removed.len()
    }

    /// Returns `true` if `context` is one of the contexts of this Referrer.
    /// [`visitor()`](Referrer::visitor) and [`complete_args()`](Referrer::complete_args) panic for
    /// any other.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_struct(&parse_quote!(struct S<T>(Rel<T>);));
    /// let referrer = leaker.result().finish().unwrap();
    /// assert!(referrer.has_context(context));
    /// ```
    pub fn has_context(&self, context: ContextId) -> bool {
        self.contexts.contains_key(&context)
    }

    /// The number of entries, with the removed ones: the indices are below it.
    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// The number of parts, which the indices of the parts that an error is about are below.
    pub(crate) fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// The indices of the entries, without the removed ones, in order.
    fn live(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.entries.len()).filter(move |ix| !self.removed.contains(ix))
    }

    /// The context `id`. Panics if there is none, as the methods that take a [`ContextId`] say.
    fn context(&self, id: ContextId) -> &Context {
        self.contexts
            .get(&id)
            .unwrap_or_else(|| panic!("the context {} is not one of this Referrer", id))
    }

    /// The type and the context of the entry `ix`.
    fn entry(&self, ix: usize) -> (&Type, &Context) {
        let (ty, context) = &self.entries[ix];
        (ty, self.context(*context))
    }

    /// Makes the part `ix` whole: nothing is looked up in it
    /// ([`LeakerResult::ignore()`](crate::LeakerResult::ignore)).
    pub(crate) fn make_whole(&mut self, ix: usize) {
        self.parts[ix].whole = true;
    }

    /// Removes the entry `ix`, and keeps the indices of the others.
    pub(crate) fn remove(&mut self, ix: usize) {
        assert!(ix < self.entries.len(), "no entry {} to remove", ix);
        self.removed.insert(ix);
    }

    /// Returns the [`EntryId`], the type and the [`Context`] of each entry, in the order of the
    /// entries. An entry that [`LeakerResult::ignore()`](crate::LeakerResult::ignore) removed is
    /// left out, and the others keep their [`EntryId`]s.
    ///
    /// Write an impl for each
    /// ([`LeakerResult::repeater_impls()`](crate::LeakerResult::repeater_impls)). The id is what
    /// the other methods take. Entries that need the same share a context.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S<T>(Wrap<T>);));
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// let (ix, ty, context) = referrer.iter().next().unwrap();
    /// assert_eq!(ix.index(), 0);
    /// assert_eq!(*ty, parse_quote!(Wrap<T>));
    /// assert_eq!(context.generics, parse_quote!(<T>)); // the entry uses `T`
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (EntryId, &Type, &Context)> {
        self.live().map(move |ix| {
            let (ty, context) = self.entry(ix);
            (EntryId::new(ix), ty, context)
        })
    }

    /// Returns `true` if some entry names `Self`, or its context does.
    ///
    /// `Self::Item` and `<Self as ::p::Tr>::X` count, `::p::Self` does not. A top-level context
    /// with no entry is not looked at, nor is a removed entry.
    pub fn has_self(&self) -> bool {
        self.live().any(|ix| self.needs_self(ix))
    }

    /// Returns `true` if the entry `ix` needs what `Self` stands for, as its context says
    /// ([`Context::needs_self()`]).
    fn needs_self(&self, ix: usize) -> bool {
        self.entry(ix).1.needs_self()
    }

    /// Creates a [`Visitor`] that writes each leaked type of a copy as its path through the Repeater.
    ///
    /// `context` is the [`ContextId`] that the `intern_*` or `enter_*` method of the Leaker returned
    /// for the item, and `projection` is the one that crate A gave to
    /// [`repeater_impls()`](crate::LeakerResult::repeater_impls). The visitor writes the entry `ix`
    /// as `projection.of(Self, context, ix)` ([`Projection::of()`]). A type is an entry if its
    /// context, computed from the generics in scope, is that of an entry, so visit the whole item,
    /// not a type cut out of it:
    ///
    /// - A generic that the copy declares again (`T` in `impl<T: Clone> MyTrait<T> for ..`) must be
    ///   declared as the entry has it, bounds included, unless you call
    ///   [`Visitor::with_outer_generics()`].
    /// - A struct, enum, union, trait or impl nested in the copy has a `Self` of its own, as in the
    ///   Leaker, so nest the same items. The entries that name it need
    ///   [`Visitor::with_nested_self()`].
    /// - A `mod` in the copy makes its items nested items, where `Self` has no meaning, and the
    ///   names that it declares hide the entries that name them, as the items of a block do.
    ///
    /// A copy at a use site takes two steps: visit it, then write the arguments of the impl into its
    /// items with the [`Substitute`] that [`complete_args()`](Referrer::complete_args) returns. If a
    /// projection names lifetimes in the inputs of a function pointer or `Fn(..)` whose output takes
    /// its elided lifetime from the one lifetime of its inputs, the visitor names that lifetime with
    /// a fresh binder (`for<'__tl_e0> fn(&'__tl_e0 ..) -> &'__tl_e0 u8`).
    ///
    /// If `Projection::of()` fails for an entry, the visitor leaves its type as written, for rustc to
    /// report. With the same `projection` as crate A it cannot, except for a type that the closure of
    /// [`Visitor::with_nested_self()`] returns (`dyn Inner<'a>` without `+ 'a`, for an entry of a
    /// nested item).
    ///
    /// # Panics
    ///
    /// If `context` is not a context of this Referrer, or the path of `projection` has arguments, or
    /// `projection` is [`Projection::unused()`] and this Referrer has an entry.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// let item: ItemTrait = parse_quote!(trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; });
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_trait(&item);
    /// let referrer = leaker.result().finish().unwrap();
    /// let projection = Projection::new(
    ///     parse_quote!(Repeater),
    ///     7,
    ///     ImplSelf::Param(parse_quote!(__S)),
    /// );
    ///
    /// // `impl Tr<u8> for W`: write the entries, then the arguments of the impl.
    /// let mut imp: ItemImpl = parse_quote! {
    ///     impl Tr<u8> for W { fn f(&self, t: T) -> Wrap<T> { todo!() } }
    /// };
    /// referrer.visitor(top_level, &projection).visit_item_impl_mut(&mut imp);
    /// let given: Vec<GenericArgument> = vec![parse_quote!(u8)];
    /// let mut substitute = referrer.complete_args(top_level, &given, &projection).unwrap();
    /// substitute.visit_item_impl_mut(&mut imp);
    ///
    /// let key = projection.key(parse_quote!(0));
    /// assert_eq!(imp, parse_quote! {
    ///     impl Tr<u8> for W {
    ///         fn f(&self, t: u8) -> <Self as Repeater<(u8,), #key>>::Type { todo!() }
    ///     }
    /// });
    /// ```
    pub fn visitor<'r>(&'r self, context: ContextId, projection: &'r Projection) -> Visitor<'r> {
        projection.check_entries(!self.is_empty());
        check_repeater_path(&projection.path);
        let found = self.context(context);
        Visitor {
            referrer: self,
            projection,
            failed: Rc::default(),
            nested_self: None,
            state: VisitorState {
                shadows: Shadows::default(),
                self_req: found.self_ty.clone(),
                scopes: vec![Scope::of_context(found)],
                outer_generics: false,
                written_self: None,
                item_depth: 0,
                own_self: OwnSelf::Context,
                defaults: Defaults::Plain,
                place: PartPlace::default(),
                signature: None,
            },
        }
    }

    /// Completes the arguments of an impl of the item at `context`, and returns the [`Substitute`]
    /// that writes them into the items of the copy.
    ///
    /// `context` is the one that `intern_*` or `enter_*` of the Leaker returned for the item whose
    /// generics the impl instantiates (a trait, usually; a trait nested in the analyzed syntax has
    /// none: enter its scope with [`View::enter_trait()`](crate::View::enter_trait)),
    /// and `given` are the arguments of the impl (`impl Tr<'static, u8> for W`). Lifetimes and types or consts are matched with the
    /// parameters by kind, in the order of the declaration, and a binding (`Item = u8`) is skipped.
    /// A parameter without an argument takes its default, with the arguments before it written in.
    ///
    /// A leaked type in a default is written through the Repeater first, by
    /// [`Projection::of()`] with `Self` (`<Self as Repeater<(u8,), KEY>>::Type`), so its self type is
    /// the one that the impl of the entry is for. A default that names `Self` is left as written.
    /// `T::A` in a default is `<arg>::A` for an argument that is one identifier, else `<arg as
    /// Trait>::A` with the one trait bound of `T`, inline or in the where clause.
    ///
    /// `projection` is the one that crate A gave to
    /// [`repeater_impls()`](crate::LeakerResult::repeater_impls).
    ///
    /// Fails with [`EncodeError::MissingArgument`] for a parameter with no argument of its kind and
    /// no default, [`EncodeError::UnexpectedArgument`] for an argument left over or of the wrong
    /// kind, [`EncodeError::AmbiguousAssociatedType`] for a `T::A` in a default that needs a trait
    /// bound that `T` does not have exactly once, and with the error of [`Projection::of()`].
    ///
    /// # Panics
    ///
    /// If `context` is not a context of this Referrer, or has no generics as declared (only the
    /// top-level context of a scope has them), or if the path of `projection` has arguments, or
    /// `projection` is [`Projection::unused()`] and this Referrer has an entry.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// let item: ItemTrait = parse_quote!(trait Tr<K, D = Wrap<K>> { fn get(&self, key: K) -> D; });
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_trait(&item);
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// // `impl Tr<u8> for Host`: `D` is `Wrap<u8>`, which the copy writes through the Repeater.
    /// let projection = Projection::new(
    ///     parse_quote!(Repeater),
    ///     7,
    ///     ImplSelf::Param(parse_quote!(__S)),
    /// );
    /// let given: Vec<GenericArgument> = vec![parse_quote!(u8)];
    /// let mut substitute = referrer
    ///     .complete_args(top_level, &given, &projection)
    ///     .unwrap();
    /// let mut sig: Signature = parse_quote!(fn get(&self, key: K) -> D);
    /// substitute.visit_signature_mut(&mut sig);
    /// let (entry, _, _) = referrer.iter().next().unwrap();
    /// let key = projection.key(entry);
    /// assert_eq!(
    ///     sig,
    ///     parse_quote!(fn get(&self, key: u8) -> <Self as Repeater<(u8,), #key>>::Type)
    /// );
    ///
    /// // `K` has no default.
    /// let err = referrer.complete_args(top_level, &[], &projection).unwrap_err();
    /// assert!(matches!(err, EncodeError::MissingArgument(_)));
    /// ```
    pub fn complete_args<'a>(
        &self,
        context: ContextId,
        given: impl IntoIterator<Item = &'a GenericArgument>,
        projection: &Projection,
    ) -> std::result::Result<Substitute, EncodeError> {
        projection.check_entries(!self.is_empty());
        check_repeater_path(&projection.path);
        let found = self.context(context);
        let declared = found.declared.clone().unwrap_or_else(|| {
            panic!(
                "the context {} has no generics as declared: pass the ContextId that an intern_* \
                 or enter_* method returned for the item",
                context
            )
        });
        // The generics are visited as the generics of a trait are: the leaked types in the
        // defaults and the bounds are written through the Repeater, by the names of the parameters
        // that the Substitute replaces by their arguments.
        let mut shell: ItemTrait = parse_quote!(
            trait __TypeLeakDeclared {}
        );
        shell.generics = declared;
        // Inside the copy `Self` is what an impl of the entry needs, whatever the Leaker chose.
        let mut visitor = self.visitor(context, projection);
        visitor.visit_item_trait_mut(&mut shell);
        match visitor.failed.take() {
            Some(error) => Err(error),
            None => complete_params(&shell.generics, given),
        }
    }
}

// How a type is looked up: the entries and the parts that it may be.
impl Referrer {
    /// Returns the entry for `ty` that matches: the first of the least cost.
    fn find(
        &self,
        at: Lookup,
        ty: &Type,
        scopes: &[Scope],
        enclosing: &Enclosing,
    ) -> Option<usize> {
        let Lookup {
            matching, hidden, ..
        } = at;
        let no_self = matching.no_self;
        let mut best: Option<(usize, usize)> = None;
        for &ix in self.by_tokens.get(&TypeKey::new(ty))? {
            if hidden.contains(&ix) {
                continue;
            }
            let (entry_ty, entry) = self.entry(ix);
            if no_self && (entry.self_ty.is_some() || type_mentions_self(entry_ty)) {
                continue;
            }
            if let Some(cost) = Self::cost(matching, ty, scopes, enclosing, entry) {
                if best.map_or(true, |(better, _)| cost < better) {
                    best = Some((cost, ix));
                }
            }
        }
        Some(best?.1)
    }

    /// Returns `true` if a part is left as written in a signature where the method lifetimes
    /// `unconstrained` are unconstrained ([`SignatureFacts::covers()`], before it knows the copy).
    fn marks_unconstrained(&self, unconstrained: &[Lifetime]) -> bool {
        !unconstrained.is_empty()
            && self.parts.iter().any(|item| match &item.mark {
                Some(SignatureMark::Unconstrained(lifetimes)) => lifetimes
                    .iter()
                    .all(|lifetime| has_lifetime(unconstrained, lifetime)),
                _ => false,
            })
    }

    /// Returns `true` if `ty` has the context `context` at `at`, in `scopes`, as an entry of that
    /// context would be found ([`Referrer::find()`]). `Self` that means something else here
    /// (`no_self`) does not matter: a part is protected, not replaced.
    fn fits(
        &self,
        context: ContextId,
        ty: &Type,
        at: Lookup,
        scopes: &[Scope],
        enclosing: &Enclosing,
    ) -> bool {
        Self::cost(at.matching, ty, scopes, enclosing, self.context(context)).is_some()
    }

    /// Returns `true` if a part with the `context` that the Leaker recorded (`None`: one that
    /// cannot be written) is the one that `ty` has at `at`, in `scopes`.
    fn part_fits(
        &self,
        context: Option<ContextId>,
        ty: &Type,
        at: Lookup,
        scopes: &[Scope],
        enclosing: &Enclosing,
    ) -> bool {
        match context {
            Some(context) => self.fits(context, ty, at, scopes, enclosing),
            // The context of `ty` cannot be written here (see `Context::of()`).
            None => {
                let Matching {
                    self_ty,
                    nested_type,
                    ..
                } = at.matching;
                Context::of(ty, scopes, self_ty, nested_type, enclosing).is_err()
            }
        }
    }

    /// Whether `ty` is copied whole ([`PartItem::whole`]), if it is a part: one with its tokens, in
    /// a predicate as `at` says, with its context in `scopes` (or none), and with an `active`
    /// anchor ([`Referrer::anchors()`]), or without one a mark or `whole`.
    fn part_of(
        &self,
        cause: &Cause,
        ty: &Type,
        at: Lookup,
        scopes: &[Scope],
        enclosing: &Enclosing,
        active: &[usize],
    ) -> Option<bool> {
        let items = self.parts_by_cause.get(cause)?;
        let mut found = None;
        for &ix in items {
            let PartItem {
                context,
                in_predicate,
                within,
                mark,
                whole,
                ..
            } = &self.parts[ix];
            // A head-only part is inert until `LeakerResult::ignore()` (see `PartItem`).
            if !*whole && within.is_none() && mark.is_none() {
                continue;
            }
            let matches = *in_predicate == at.place.in_predicate
                && (within.is_none() || active.contains(&ix))
                && match (mark, at.signature) {
                    (None, _) => true,
                    // Made by a signature: in the arguments of one that has the same fault.
                    (Some(mark), Some(facts)) => {
                        at.place.in_fn_arg
                            && facts.covers(mark, context.map(|context| self.context(context)))
                    }
                    (Some(_), None) => false,
                }
                && self.part_fits(*context, ty, at, scopes, enclosing);
            if matches {
                found = Some(found.unwrap_or(false) || *whole);
            }
        }
        found
    }

    /// Returns `true` if a part that a signature made has the same tokens as `ty` and the context
    /// that `ty` has in `scopes`, whatever signature it was made for: the Leaker leaked `ty` whole
    /// there, and leaks it here ([`SignatureFacts::inputs_probe()`]).
    fn is_part_of_signature(
        &self,
        ty: &Type,
        at: Lookup,
        scopes: &[Scope],
        enclosing: &Enclosing,
    ) -> bool {
        let items = self.parts_by_cause.get(&Cause::Type(ty.clone()));
        items.map_or(false, |items| {
            items.iter().any(|&ix| {
                let PartItem {
                    context,
                    in_predicate,
                    mark,
                    ..
                } = &self.parts[ix];
                mark.is_some()
                    && *in_predicate == at.place.in_predicate
                    && self.part_fits(*context, ty, at, scopes, enclosing)
            })
        })
    }

    /// Pushes onto `active` the parts whose anchor `ty` is, as an entry would be found: they are
    /// left as written inside `ty`, and only there.
    fn anchors(
        &self,
        ty: &Type,
        at: Lookup,
        scopes: &[Scope],
        enclosing: &Enclosing,
        active: &mut Vec<usize>,
    ) {
        if self.parts_by_anchor.is_empty() {
            return;
        }
        let items = match self.parts_by_anchor.get(&Cause::Type(ty.clone())) {
            Some(items) => items,
            None => return,
        };
        for &ix in items {
            if let Some((_, anchor)) = &self.parts[ix].within {
                if self.fits(*anchor, ty, at, scopes, enclosing) {
                    active.push(ix);
                }
            }
        }
    }

    /// Returns the entries that items declared together with the `names` hide.
    fn hidden_by(&self, names: &[Ident]) -> Vec<usize> {
        let mut hidden = Vec::new();
        for (key, entries) in self.by_tokens.iter() {
            if !mentions_idents(&key.ty, names) {
                continue;
            }
            for &ix in entries {
                let (_, context) = self.entry(ix);
                // A generic of the entry's context is the name in the type wherever the entry is
                // found, and it hides the item of the same name there (the Leaker does not
                // take it for the item either: `starts_with_generic()`).
                let items = names.iter().filter(|name| {
                    let name = ident_name(name);
                    !context
                        .generics
                        .params
                        .iter()
                        .any(|param| generic_param_name(param) == name)
                });
                if mentions_idents(&key.ty, &items.cloned().collect::<Vec<_>>()) {
                    hidden.push(ix);
                }
            }
        }
        hidden
    }

    /// Returns the cost of `entry` for `ty`, or `None` if it does not match. It is `0` if the
    /// context of `ty` in `scopes` is `entry`'s.
    fn cost(
        matching: Matching,
        ty: &Type,
        scopes: &[Scope],
        enclosing: &Enclosing,
        entry: &Context,
    ) -> Option<usize> {
        let Matching {
            self_ty,
            nested_type,
            without_self_first,
            ..
        } = matching;
        if without_self_first {
            let mut scopes = scopes.to_vec();
            if let Some(last) = scopes.last_mut() {
                last.set_self(None);
            }
            let without_self = Context::of(ty, &scopes, None, false, enclosing);
            if without_self.map_or(false, |context| context == *entry) {
                return Some(0);
            }
        }
        // A bound that has no name here (see `Context::of()`) is in no entry. An entry for any type
        // has the `Self` of a nested struct, enum or union as the implementor, also where its type
        // names it (`Reason::NestedSelfInBounds`).
        let any = nested_type && matches!(entry.self_ty, Some(SelfRequirement::Any));
        let context = Context::of_as(ty, scopes, self_ty, nested_type, any, enclosing).ok()?;
        let after = usize::from(without_self_first);
        if context == *entry {
            Some(after)
        } else if context.is_entry_with_extra_self_predicates(entry) {
            Some(after + 1)
        } else {
            None
        }
    }
}

// The Repeater impls of the entries.
impl Referrer {
    /// The Repeater impls of the entries, in order (`LeakerResult::repeater_impls()`). The path is
    /// checked first, then `check_marker()`, so `NeedsSelf` goes before an error of a const.
    pub(crate) fn repeater_impls(
        &self,
        projection: &Projection,
    ) -> std::result::Result<Vec<RepeaterImpl>, EncodeError> {
        projection.check_entries(!self.is_empty());
        check_repeater_path(&projection.path);
        self.check_marker(&projection.impl_self)?;
        self.live()
            .map(|ix| {
                let (ty, context) = self.entry(ix);
                projection.impl_of(EntryId::new(ix), ty, context)
            })
            .collect()
    }

    /// `NeedsSelf` for a marker, before `repeater_impls()` builds any impl. `Projection::impl_of()`
    /// then fails only with `UnsupportedConstType`: its `Self` is a parameter, `Self` or a bucket
    /// type, and a marker meets no `self_outlives`, which only a context with a self type has.
    fn check_marker(&self, target: &ImplSelf) -> std::result::Result<(), EncodeError> {
        if !matches!(target, ImplSelf::Marker(_)) {
            return Ok(());
        }
        // An entry that needs `Self` has none for a marker, even if `self_ty` is `None`: the
        // `Self` of its generics would be the marker.
        match self.live().find(|&ix| self.needs_self(ix)) {
            Some(ix) => Err(EncodeError::NeedsSelf(Box::new(self.entry(ix).0.clone()))),
            None => Ok(()),
        }
    }
}

/// A nested struct, enum, union, trait or impl, which declares its own `Self`.
#[derive(Debug)]
struct Nested {
    /// A struct, an enum or a union (`SelfScope::nested` of the Leaker), not a trait or an impl.
    adt: bool,
    /// What `Self` is in it, as the Leaker has it.
    requirement: SelfRequirement,
    /// The name of the item; an impl has none.
    name: Option<Ident>,
    /// What the closure of `Visitor::with_nested_self()` answered for `requirement`, once asked.
    answer: RefCell<Option<Option<Type>>>,
}

impl Nested {
    fn of_item(ident: &Ident, generics: &Generics) -> Self {
        Self {
            adt: true,
            requirement: SelfRequirement::of_item(ident, generics),
            name: Some(ident.clone()),
            answer: RefCell::default(),
        }
    }

    fn of_trait(ident: &Ident, generics: &Generics) -> Self {
        Self {
            adt: false,
            requirement: SelfRequirement::of_trait(ident, generics),
            name: Some(ident.clone()),
            answer: RefCell::default(),
        }
    }

    fn of_impl(self_ty: &Type) -> Self {
        Self {
            adt: false,
            requirement: SelfRequirement::Type(self_ty.clone()),
            name: None,
            answer: RefCell::default(),
        }
    }

    /// The self type to write for the entries of this item that depend on its own `Self`: what
    /// `nested_self` answers for the requirement (asked once), or `None` without it.
    fn self_ty<N: FnMut(&SelfRequirement) -> Option<Type>>(
        &self,
        nested_self: Option<&RefCell<N>>,
    ) -> Option<Type> {
        let mut answer = self.answer.borrow_mut();
        answer
            .get_or_insert_with(|| nested_self.and_then(|f| (*f.borrow_mut())(&self.requirement)))
            .clone()
    }
}

/// How the folder writes an entry: `projection.of(self_ty, ..)`, with `Self` unless a host stands
/// for it, or in a nested item what `nested_self` answers for its own `Self` (`Type`, `Bound`).
/// Without an answer, or on a failure (the first is kept in `failed`), it is left as written.
struct EntryProjector<'a, N> {
    projection: &'a Projection,
    self_ty: Option<&'a Type>,
    failed: &'a RefCell<Option<EncodeError>>,
    /// The nested item that declares the `Self` in force, if the visit is in one.
    nested: Option<&'a Nested>,
    nested_self: Option<&'a RefCell<N>>,
}

/// Folds a type, replacing the leaked types in it (see `Visitor::visit_type_mut()`).
struct EntryFolder<'a, 'b, N> {
    referrer: &'a Referrer,
    at: Lookup<'b>,
    scopes: &'a mut Vec<Scope>,
    projector: EntryProjector<'a, N>,
    enclosing: Enclosing,
    /// The parts whose anchor is a type around the one being folded.
    active: Vec<usize>,
}

impl<'a, 'b, N: FnMut(&SelfRequirement) -> Option<Type>> EntryFolder<'a, 'b, N> {
    /// Replaces `Self` by the host, if there is one ([`Visitor::with_self()`]).
    fn replace_self(&self) -> Option<ReplaceSelf<'a>> {
        self.projector
            .self_ty
            .map(|self_ty| ReplaceSelf { self_ty })
    }

    /// Folds the signature `i` with `fold`, with its binder in scope, and names its elided lifetime
    /// where the copy would break it: before the binder is in scope for the host that replaces
    /// `Self` ([`before_fold()`]), after the fold for the entries ([`after_fold()`]).
    fn fold_sig<S: FnSig>(&mut self, mut i: S, fold: impl FnOnce(&mut Self, S) -> S) -> S {
        let elision = before_fold(&mut i, self.projector.self_ty, &self.scopes[..]);
        let depth = self.scopes.len();
        self.scopes.extend(i.binder().map(Scope::binder));
        let mut i = fold(self, i);
        self.scopes.truncate(depth);
        after_fold(&mut i, elision, &self.scopes[..]);
        i
    }

    /// The entry that stands for `ty`, unless `ty` is a part, which is left as written, or
    /// is inside a whole part: nothing in it is looked up.
    fn entry_for(&mut self, ty: &Type, enclosing: &Enclosing) -> Option<usize> {
        if self.at.place.protected {
            return None;
        }
        if !self.referrer.parts_by_cause.is_empty() {
            if let Some(whole) = self.referrer.part_of(
                &Cause::Type(ty.clone()),
                ty,
                self.at,
                self.scopes,
                enclosing,
                &self.active,
            ) {
                self.at.place.protected = whole;
                return None;
            }
        }
        match self.referrer.find(self.at, ty, self.scopes, enclosing) {
            // A removed entry is written as written, and nothing in it is looked up: it is no
            // entry, and neither are its components, which are in the type as written.
            Some(ix) if self.referrer.removed.contains(&ix) => {
                self.at.place.protected = true;
                None
            }
            found => found,
        }
    }

    /// Returns `true` in the probe of a signature ([`SignatureFacts::inputs_probe()`]) if `ty`,
    /// which is no entry, is a part that a signature made: it is a type leaked whole, replaced.
    fn leaked_in_probe(&self, ty: &Type, enclosing: &Enclosing) -> bool {
        !self.at.place.protected
            && self.at.signature.map_or(false, SignatureFacts::probing)
            && self
                .referrer
                .is_part_of_signature(ty, self.at, self.scopes, enclosing)
    }

    /// The path through the Repeater for the entry `ix`, or `None` if it is left as written: it
    /// fails, which `failed` keeps, or it is for the own type of a nested item that has no self
    /// type to write. The probe of a signature writes `()`, which holds no lifetime.
    fn write(&self, ix: usize) -> Option<Type> {
        let (_, context) = self.referrer.entry(ix);
        let EntryProjector {
            projection,
            self_ty,
            failed,
            nested,
            nested_self,
        } = &self.projector;
        let own;
        let self_ty = match (nested, &context.self_ty) {
            // The impl of the entry is for the Leaker's item, which the copy's `Self` is not.
            (Some(nested), Some(SelfRequirement::Type(_) | SelfRequirement::Bound(_))) => {
                own = nested.self_ty(*nested_self)?;
                &own
            }
            _ => match self_ty {
                Some(self_ty) => self_ty,
                None => {
                    own = parse_quote!(Self);
                    &own
                }
            },
        };
        if self.at.signature.map_or(false, SignatureFacts::probing) {
            return Some(unit_type());
        }
        match projection.of(self_ty, context, EntryId::new(ix)) {
            Ok(written) => Some(written),
            Err(error) => {
                failed.borrow_mut().get_or_insert(error);
                None
            }
        }
    }
}

impl<'a, 'b, N: FnMut(&SelfRequirement) -> Option<Type>> Fold for EntryFolder<'a, 'b, N> {
    fn fold_type(&mut self, ty: Type) -> Type {
        let enclosing = self.enclosing.clone();
        let outer = self.at.place;
        let active = self.active.len();
        let found = self.entry_for(&ty, &enclosing);
        let leaked = found.is_none() && self.leaked_in_probe(&ty, &enclosing);
        if found.is_none() && !leaked {
            self.enclosing = enclosing.components_of(&ty, self.scopes.len());
            // A type that owns the cause of a conflict is the anchor of the part in it.
            if !self.at.place.protected {
                self.referrer
                    .anchors(&ty, self.at, self.scopes, &enclosing, &mut self.active);
            }
            // The bounds of an `impl Trait` argument are predicates of the item.
            if self.at.place.in_fn_arg && matches!(ty, Type::ImplTrait(_)) {
                self.at.place.in_predicate = true;
            }
        }
        let folded = match (found, self.replace_self()) {
            (None, _) if leaked => unit_type(),
            // A failure leaves the entry as written (with the host for `Self`).
            (Some(ix), replace) => self.write(ix).unwrap_or_else(|| match replace {
                Some(mut replace) => replace.fold_type(ty),
                None => ty,
            }),
            (None, None) => syn::fold::fold_type(self, ty),
            (None, Some(replace)) => replace.fold_type_with(self, ty),
        };
        self.at.place = outer;
        self.active.truncate(active);
        self.enclosing = enclosing;
        folded
    }
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        // A type in an expression is not a component of the types around it.
        let enclosing = std::mem::take(&mut self.enclosing);
        let folded = match self.replace_self() {
            Some(replace) => replace.fold_expr_with(self, expr),
            None => syn::fold::fold_expr(self, expr),
        };
        self.enclosing = enclosing;
        folded
    }
    fn fold_macro(&mut self, i: Macro) -> Macro {
        match self.replace_self() {
            Some(mut replace) => replace.fold_macro(i),
            None => syn::fold::fold_macro(self, i),
        }
    }
    fn fold_type_bare_fn(&mut self, i: TypeBareFn) -> TypeBareFn {
        self.fold_sig(i, syn::fold::fold_type_bare_fn)
    }
    fn fold_trait_bound(&mut self, i: TraitBound) -> TraitBound {
        self.fold_sig(i, syn::fold::fold_trait_bound)
    }
}

/// Replaces the leaked types in a copy of the analyzed item with their paths through the Repeater.
/// Get it with [`Referrer::visitor()`].
///
/// It implements `syn::visit_mut::VisitMut`. Visit a whole type or item, not a type cut out of
/// another ([Outlives bounds](crate::Leaker#outlives-bounds)). A type is replaced only if an entry
/// has the context that it has there, from the generics in scope. The parts that the Leaker copied
/// as written ([recovery](crate::Leaker#errors-and-recovery)) are left as they are. Items and
/// imports declared in a block, a module or a `File` hide the leaked types that name them there,
/// but the name of a nested struct, enum, union or trait hides none inside the item itself (what
/// the item declares in it hides again). Values and bodies are visited too: a type there is an
/// entry only with [`Leaker::check_values`](crate::Leaker::check_values) or
/// [`Leaker::check_bodies`](crate::Leaker::check_bodies).
///
/// `N` is the closure of [`Visitor::with_nested_self()`]; there is none by default.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use syn::visit_mut::VisitMut;
/// let mut item: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> Rel; });
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_trait(&item);
/// let referrer = leaker.result().finish().unwrap();
/// let projection = Projection::new(
///     parse_quote!(Repeater),
///     7,
///     ImplSelf::Param(parse_quote!(__S)),
/// );
///
/// // Rewrite the whole trait, with the `ContextId` that the Leaker returned for it.
/// referrer.visitor(top_level, &projection).visit_item_trait_mut(&mut item);
/// let key = projection.key(parse_quote!(0));
/// assert_eq!(item, parse_quote! {
///     trait Tr { fn f(&self) -> <Self as Repeater<(), #key>>::Type; }
/// });
/// ```
#[must_use]
pub struct Visitor<'r, N = fn(&SelfRequirement) -> Option<Type>> {
    referrer: &'r Referrer,
    projection: &'r Projection,
    /// The first error of `Projection::of()`, with the visitors made from this one. The entry is
    /// left as written; `complete_args()` reads it.
    failed: Rc<RefCell<Option<EncodeError>>>,
    /// The closure of [`Visitor::with_nested_self()`], if it was called.
    nested_self: Option<Rc<RefCell<N>>>,
    state: VisitorState,
}

/// Where a [`Visitor`] is in the visited syntax; a nested visitor starts from a copy.
#[derive(Clone, Debug)]
struct VisitorState {
    /// What the items declared in the visited syntax hide.
    shadows: Shadows,
    /// What `Self` means here: in the context, or in the nested item that declares it. Each scope
    /// in `scopes` keeps the one in force where it was declared.
    self_req: Option<SelfRequirement>,
    /// The generics in scope, innermost last. The first are those of the context.
    scopes: Vec<Scope>,
    /// See [`Visitor::with_outer_generics()`].
    outer_generics: bool,
    /// What replaces `Self` in the copy ([`Visitor::with_self()`]).
    written_self: Option<Type>,
    /// How many items the visit is in. The outermost item visited is the scope itself (`0`).
    item_depth: usize,
    /// Whose `Self` is in force here.
    own_self: OwnSelf,
    /// While the generics of a scope that this visit declares are visited: how the defaults of
    /// their parameters are looked up (see `Visitor::visit_type_param_mut()`).
    defaults: Defaults,
    /// Where the visited syntax is among the predicates of the item.
    place: PartPlace,
    /// The signature being visited (not its body), for the parts that a signature made.
    signature: Option<Rc<SignatureFacts>>,
}

/// Whose `Self` is in force where a [`Visitor`] is.
#[derive(Clone, Debug)]
enum OwnSelf {
    /// That of the context: the visit is in no nested item that declares its own.
    Context,
    /// That of a nested struct, enum, union, trait or impl. A function, type alias, trait alias,
    /// const or static keeps the `Self` around, as in the Leaker.
    Nested(Rc<Nested>),
    /// None: in a `mod` inside a nested item.
    InMod,
}

/// The entries that the items declared in the visited syntax hide (`Visitor::shadow_items()`).
#[derive(Clone, Debug, Default)]
struct Shadows {
    /// The entries hidden.
    hidden: Rc<Sorted<HashSet<usize>>>,
    /// The names that each container around declares, outermost first: `hidden` is what they hide.
    names: Rc<Vec<Vec<Ident>>>,
}

impl Shadows {
    /// Adds the `names` that the items of a container declare together, and hides the entries
    /// that they hide (`Referrer::hidden_by()`).
    fn push(&mut self, referrer: &Referrer, names: Vec<Ident>) {
        if !names.is_empty() {
            let hidden = referrer.hidden_by(&names);
            if !hidden.is_empty() {
                Rc::make_mut(&mut self.hidden).extend(hidden);
            }
        }
        Rc::make_mut(&mut self.names).push(names);
    }

    /// Drops the innermost container's declaration of `name`, for the visit of the nested item
    /// `name`: its own name hides none of its entries.
    fn unshadow(&mut self, referrer: &Referrer, name: &Ident) {
        let own = ident_name(name);
        let declares = |names: &Vec<Ident>| names.iter().any(|n| ident_name(n) == own);
        if !self.names.last().map_or(false, declares) {
            return;
        }
        let shadows = Rc::make_mut(&mut self.names);
        if let Some(names) = shadows.last_mut() {
            names.retain(|n| ident_name(n) != own);
        }
        let mut hidden = HashSet::new();
        for names in shadows.iter().filter(|names| !names.is_empty()) {
            hidden.extend(referrer.hidden_by(names));
        }
        self.hidden = Rc::new(Sorted(hidden));
    }
}

impl<'r, N> Clone for Visitor<'r, N> {
    fn clone(&self) -> Self {
        Self {
            referrer: self.referrer,
            projection: self.projection,
            failed: self.failed.clone(),
            nested_self: self.nested_self.clone(),
            state: self.state.clone(),
        }
    }
}

impl<'r, N> std::fmt::Debug for Visitor<'r, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Visitor")
            .field("referrer", &self.referrer)
            .field("projection", &self.projection)
            .field("state", &self.state)
            .finish()
    }
}

impl<'r, N: FnMut(&SelfRequirement) -> Option<Type>> Visitor<'r, N> {
    /// Treats the generics of the outermost item visited as those of the context.
    ///
    /// Use it if the copy declares the generics of the scope again, with other bounds
    /// (`impl<T: Clone> MyTrait<T> for ..`): a type that names them is replaced if it was leaked,
    /// whatever bounds they have. Deeper generics stay declared, so they must match the entry's.
    pub fn with_outer_generics(mut self) -> Self {
        self.state.outer_generics = true;
        self
    }

    /// Writes `self_ty` for `Self` in the copy, and as the self type of each path through the
    /// Repeater (`<self_ty as Repeater<..>>::Type`).
    ///
    /// `Self` is replaced in the other types too, parts copied as written included, but not in
    /// receivers (`&self`) or in expressions. A nested struct, enum, union, trait or impl has a
    /// `Self` of its own: the visitor leaves it (see
    /// [`with_nested_self()`](Visitor::with_nested_self)).
    ///
    /// A `self_ty` with lifetimes (`::p::S<'a>`) would give `fn(&Self) -> &u8` more input lifetimes
    /// (E0106), so the visitor names the elided one with a fresh binder
    /// (`for<'__tl_e0> fn(&'__tl_e0 ::p::S<'a>) -> &'__tl_e0 u8`). Where it cannot, the type is
    /// leaked whole ([Elided lifetimes](crate::Leaker#elided-lifetimes)).
    ///
    /// Fails with [`EncodeError::UnboundedTraitObject`] if `self_ty` is a trait object without a
    /// lifetime bound (`dyn Tr`) and an entry of the visited item's `Self` has `Self: 'a` in its
    /// context: write `dyn Tr + 'static`. The entries of nested items are written for the closure of
    /// [`with_nested_self()`](Visitor::with_nested_self) instead.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote! {
    ///     pub trait Tr<'a> where Self: 'a { fn f(&self) -> Rel<'a, Self>; }
    /// };
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_trait(&item);
    /// let referrer = leaker.result().finish().unwrap();
    /// let projection = Projection::new(
    ///     parse_quote!(Repeater),
    ///     7,
    ///     ImplSelf::Param(parse_quote!(__S)),
    /// );
    ///
    /// // `dyn Shape` has no lifetime bound, which `Self: 'a` needs.
    /// let visitor = referrer.visitor(top_level, &projection);
    /// let error = visitor.with_self(parse_quote!(dyn Shape)).unwrap_err();
    /// assert!(matches!(error, EncodeError::UnboundedTraitObject(_)));
    ///
    /// let visitor = referrer.visitor(top_level, &projection);
    /// assert!(visitor.with_self(parse_quote!(dyn Shape + 'static)).is_ok());
    /// ```
    pub fn with_self(mut self, self_ty: Type) -> std::result::Result<Self, EncodeError> {
        for ix in self.referrer.live() {
            let (_, context) = self.referrer.entry(ix);
            // The host is written only for the `Self` of the visited context: an entry of another
            // `Self` (another item of the Leaker, a nested item) is never found with it.
            if context.self_ty == self.state.self_req && !context.self_outlives.is_empty() {
                let params: Vec<&GenericParam> = context.generics.params.iter().collect();
                encode_self_outlives(&params, &self_ty, &context.self_outlives)?;
            }
        }
        self.state.written_self = Some(self_ty);
        Ok(self)
    }

    /// Writes the entries for the own type of a nested item for the type that `self_ty` returns.
    ///
    /// A struct, enum, union, trait or impl nested in the visited syntax has a `Self` of its own.
    /// The Repeater's impl of an entry that names it ([`SelfRequirement::Type`],
    /// [`SelfRequirement::Bound`]) is for the Leaker's item, which the copy's `Self` is not. For
    /// each nested item that has such an entry, the visitor calls `self_ty` once, with the
    /// requirement of the item, and writes each of its entries as `projection.of(&type, ..)` for
    /// the `Some(type)` that it gets. For `None`, and without this method, the entries are left as
    /// written.
    ///
    /// An entry for any type ([`SelfRequirement::Any`]) needs no closure: it is always written
    /// with `Self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit::Visit;
    /// # use syn::visit_mut::VisitMut;
    /// // A struct nested in a scope: it has a `Self` of its own.
    /// let nested: Item = parse_quote!(struct N<T>(Wrap<Self>, Rel<T>););
    /// let mut leaker = Leaker::new();
    /// let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| view.visit_item(&nested));
    /// let referrer = leaker.result().finish().unwrap();
    /// let projection = Projection::new(
    ///     parse_quote!(Repeater),
    ///     7,
    ///     ImplSelf::Param(parse_quote!(__S)),
    /// );
    ///
    /// // The copy nests it in a module. The impl of `Wrap<Self>` is for the Leaker's `N<T>`, which
    /// // is `::leaker::N<T>` here.
    /// let mut item: ItemMod = parse_quote!(mod m { struct N<T>(Wrap<Self>, Rel<T>); });
    /// referrer
    ///     .visitor(scope, &projection)
    ///     .with_nested_self(|requirement| match requirement {
    ///         SelfRequirement::Type(ty) => Some(parse_quote!(::leaker::#ty)),
    ///         _ => None,
    ///     })
    ///     .visit_item_mod_mut(&mut item);
    /// let (wrap, rel) = (projection.key(parse_quote!(0)), projection.key(parse_quote!(1)));
    /// assert_eq!(item, parse_quote! {
    ///     mod m {
    ///         struct N<T>(
    ///             <::leaker::N<T> as Repeater<(T,), #wrap>>::Type,
    ///             <Self as Repeater<(T,), #rel>>::Type
    ///         );
    ///     }
    /// });
    /// ```
    pub fn with_nested_self<M: FnMut(&SelfRequirement) -> Option<Type>>(
        self,
        self_ty: M,
    ) -> Visitor<'r, M> {
        Visitor {
            referrer: self.referrer,
            projection: self.projection,
            failed: self.failed,
            nested_self: Some(Rc::new(RefCell::new(self_ty))),
            state: self.state,
        }
    }

    /// Hides the entries that the `items` declared together hide (`Referrer::hidden_by()`): their
    /// names are what the Leaker takes for local in the values of a block ([`declared_names()`]).
    fn shadow_items<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>) {
        let names = items.into_iter().flat_map(declared_names).collect();
        self.state.shadows.push(self.referrer, names);
    }

    /// Returns `true` if `cause`, a bound whose trait path conflicts, is a part: it is no type.
    /// `path` stands for it as a type, to give it its context.
    fn is_part_bound(&self, cause: impl FnOnce() -> Cause, path: &Path) -> bool {
        if self.referrer.parts_by_cause.is_empty() || self.state.place.protected {
            return false;
        }
        self.referrer
            .part_of(
                &cause(),
                &path_type(path),
                self.lookup(),
                &self.state.scopes,
                &Enclosing::default(),
                &[],
            )
            .is_some()
    }

    /// Returns how a type is looked up here.
    fn lookup(&self) -> Lookup<'_> {
        // In a nested item, the entries that name its own `Self` are found: one for any type is
        // written with `Self`, the others for the answer of `with_nested_self()` or as written. In
        // a `mod` inside it `Self` means nothing: no entry that names it is found.
        Lookup {
            matching: Matching {
                self_ty: self.state.self_req.as_ref(),
                no_self: matches!(self.state.own_self, OwnSelf::InMod),
                nested_type: self.nested_type(),
                without_self_first: self.state.defaults == Defaults::OwnTraitDefault,
            },
            hidden: &self.state.shadows.hidden,
            place: self.state.place,
            signature: self.state.signature.as_deref(),
        }
    }

    /// Returns `true` in a nested struct, enum or union.
    fn nested_type(&self) -> bool {
        matches!(&self.state.own_self, OwnSelf::Nested(nested) if nested.adt)
    }

    /// Returns what replaces `Self` here ([`Visitor::with_self()`]), unless a nested item declares
    /// its own `Self`.
    fn replaced_self(&self) -> Option<&Type> {
        self.state
            .written_self
            .as_ref()
            .filter(|_| matches!(self.state.own_self, OwnSelf::Context))
    }

    /// Runs `visit`, which visits a part that is left as written: no entry is looked up in it.
    fn protected(&mut self, visit: impl FnOnce(&mut Self)) {
        let outer = std::mem::replace(&mut self.state.place.protected, true);
        visit(self);
        self.state.place.protected = outer;
    }

    fn visit_attrs(&mut self, attrs: &mut [Attribute]) {
        for attr in attrs {
            self.visit_attribute_mut(attr);
        }
    }

    /// `generics` are those of the context, in any order and with the bounds anywhere.
    fn is_context_generics(&self, generics: &Generics) -> bool {
        let canonical = Context::new(generics.clone(), None);
        let canonical = Scope::new(&canonical.generics, None);
        matches!(self.state.scopes.first(), Some(first) if first.same_generics(&canonical))
    }

    /// Visits the attributes and returns a nested visitor with `generics` visited and in scope.
    /// `item`: an item, not an associated one. `declares`: what `Self` is in a struct, an enum, a
    /// union, a trait or an impl, as the Leaker has it; other items keep the `Self` around.
    fn enter(
        &mut self,
        attrs: &mut [Attribute],
        generics: &mut Generics,
        declares: Option<Nested>,
        item: bool,
    ) -> Self {
        self.visit_attrs(attrs);
        let mut visitor = self.clone();
        visitor.state.item_depth += 1;
        // Only an item nested in the visited syntax that declares `Self` has one of its own: the
        // Leaker keeps the `Self` around for a function, a type alias or a trait alias. The layer
        // of the item is the Leaker's: entries that name its `Self` exist
        // (`AnalyzeVisitor::with_item()`).
        let mut own_trait = false;
        if self.state.item_depth > 0 {
            if let Some(nested) = declares {
                if let Some(name) = &nested.name {
                    visitor.state.shadows.unshadow(visitor.referrer, name);
                }
                own_trait = matches!(nested.requirement, SelfRequirement::Bound(_));
                visitor.state.self_req = Some(nested.requirement.clone());
                visitor.state.own_self = OwnSelf::Nested(Rc::new(nested));
            }
        }
        // Declare before visiting: every parameter is in scope of each other's bounds. The
        // outermost item is the scope itself if it declares the generics of the context as they
        // are, or if `with_outer_generics()` says so: they are not declared again.
        let outermost = item && self.state.item_depth == 0;
        let declared =
            !(outermost && (self.state.outer_generics || self.is_context_generics(generics)));
        if declared {
            let declared_for = visitor.state.self_req.clone();
            visitor
                .state
                .scopes
                .push(Scope::new(generics, declared_for));
        }
        // How the defaults of the generics are looked up: see `Defaults`.
        visitor.state.defaults = if own_trait {
            Defaults::OwnTrait
        } else {
            Defaults::WithoutSelf
        };
        visitor.visit_generics_mut(generics);
        visitor.state.defaults = Defaults::Plain;
        visitor
    }

    /// Returns the inputs of `sig` as the copy writes them, but with each type leaked whole (an
    /// entry, or a part that a signature made: [`SignatureFacts::inputs_probe()`]) replaced by
    /// `()`, which holds no lifetime. No projection is written.
    fn inputs_without_entries(&self, sig: &Signature) -> Vec<FnArg> {
        let mut probe = self.clone();
        probe.state.signature = Some(Rc::new(SignatureFacts::inputs_probe()));
        sig.inputs
            .iter()
            .cloned()
            .map(|mut input| {
                probe.visit_fn_arg_mut(&mut input);
                input
            })
            .collect()
    }

    /// Visits the signature and returns the visitor for its body.
    fn enter_signature(
        &mut self,
        attrs: &mut [Attribute],
        sig: &mut Signature,
        item: bool,
    ) -> Self {
        let mut visitor = self.enter(attrs, &mut sig.generics, None, item);
        let mut facts = SignatureFacts::of(sig);
        // A part with a mark is left as written where the copy leaves the lifetime unconstrained,
        // not where an input keeps it: that is found out for the signatures that such a part may be
        // in.
        if self.referrer.marks_unconstrained(facts.unconstrained()) {
            let copy = visitor.inputs_without_entries(sig);
            facts = facts.in_copy(&SignatureScopes(&visitor.state.scopes), sig, &copy);
        }
        for input in sig.inputs.iter_mut() {
            visitor.state.signature = Some(Rc::new(facts.for_input(input)));
            visitor.visit_fn_arg_mut(input);
        }
        visitor.state.signature = Some(Rc::new(facts));
        visitor.visit_return_type_mut(&mut sig.output);
        visitor.state.signature = None;
        visitor
    }
}

impl<'r, N: FnMut(&SelfRequirement) -> Option<Type>> VisitMut for Visitor<'r, N> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        // The lookup reads the rest of the visitor, so the scopes are lent from outside it.
        let mut scopes = std::mem::take(&mut self.state.scopes);
        let nested = match &self.state.own_self {
            OwnSelf::Nested(nested) => Some(&**nested),
            _ => None,
        };
        *i = EntryFolder {
            referrer: self.referrer,
            at: self.lookup(),
            scopes: &mut scopes,
            projector: EntryProjector {
                projection: self.projection,
                // In a nested item there is no host: `Self` is the item's own.
                self_ty: self.replaced_self(),
                failed: &self.failed,
                nested,
                nested_self: self.nested_self.as_deref(),
            },
            enclosing: Enclosing::default(),
            active: Vec::new(),
        }
        .fold_type(i.clone());
        self.state.scopes = scopes;
    }
    fn visit_fn_arg_mut(&mut self, i: &mut FnArg) {
        // An `impl Trait` in an argument is a generic of the function, whose bounds are predicates
        // of the item (the Leaker's `in_fn_arg`).
        let outer = std::mem::replace(&mut self.state.place.in_fn_arg, true);
        syn::visit_mut::visit_fn_arg_mut(self, i);
        self.state.place.in_fn_arg = outer;
    }
    fn visit_type_param_bound_mut(&mut self, i: &mut TypeParamBound) {
        // A bound that is not in a type is a predicate of the item (the Leaker's `env_depth`). A
        // type is visited whole by `visit_type_mut()`, so a bound visited here is not in one.
        let outer = std::mem::replace(&mut self.state.place.in_predicate, true);
        syn::visit_mut::visit_type_param_bound_mut(self, i);
        self.state.place.in_predicate = outer;
    }
    fn visit_receiver_mut(&mut self, i: &mut Receiver) {
        if self.replaced_self().is_none() {
            return syn::visit_mut::visit_receiver_mut(self, i);
        }
        // The receiver type stays: rustc does not accept an associated type path there.
        self.visit_attrs(&mut i.attrs);
    }
    fn visit_predicate_type_mut(&mut self, i: &mut PredicateType) {
        let depth = self.state.scopes.len();
        self.state
            .scopes
            .extend(i.lifetimes.as_ref().map(Scope::binder));
        let outer = std::mem::replace(&mut self.state.place.in_predicate, true);
        syn::visit_mut::visit_predicate_type_mut(self, i);
        self.state.place.in_predicate = outer;
        self.state.scopes.truncate(depth);
        // What replaced the bounded type (`Self`, an entry) may need parentheses there.
        parenthesize_bounded(&mut i.bounded_ty);
    }
    // What replaced the bounded type of the first predicate (`Self::A`) may need parentheses.
    fn visit_where_clause_mut(&mut self, i: &mut WhereClause) {
        syn::visit_mut::visit_where_clause_mut(self, i);
        parenthesize_leading_qualified_path(i);
    }
    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        // A bound of the generics is not in a type, which the same rewriting handles. A bound whose
        // trait path the Leaker could not leak is copied as written, with the types in it.
        let part = self.is_part_bound(|| bound_cause(i), &i.path);
        let elision = before_fold(i, self.replaced_self(), &self.state.scopes[..]);
        let depth = self.state.scopes.len();
        self.state
            .scopes
            .extend(i.lifetimes.as_ref().map(Scope::binder));
        if part {
            self.protected(|visitor| syn::visit_mut::visit_trait_bound_mut(visitor, i));
        } else {
            syn::visit_mut::visit_trait_bound_mut(self, i);
        }
        self.state.scopes.truncate(depth);
        after_fold(i, elision, &self.state.scopes[..]);
    }
    fn visit_type_param_mut(&mut self, i: &mut TypeParam) {
        let defaults = self.state.defaults;
        if matches!(defaults, Defaults::Plain | Defaults::OwnTraitDefault) {
            return syn::visit_mut::visit_type_param_mut(self, i);
        }
        // The default is looked up apart from the bounds, which keep `self.defaults`.
        let default = i.default.take();
        syn::visit_mut::visit_type_param_mut(self, i);
        i.default = default;
        let default = match &mut i.default {
            Some(default) => default,
            None => return,
        };
        if defaults == Defaults::OwnTrait {
            // An entry of either kind is found, the one without `Self` first (it is what an impl
            // of the trait can use, for `complete_args()`): see `Defaults::OwnTrait`.
            self.state.defaults = Defaults::OwnTraitDefault;
            self.visit_type_mut(default);
            self.state.defaults = Defaults::OwnTrait;
        } else {
            // The default is looked up without `Self`: see `Defaults::WithoutSelf`.
            let nested_type = self.nested_type();
            let self_req = self.state.self_req.take();
            let own = self.state.scopes.last().cloned();
            if let Some(scope) = self.state.scopes.last_mut() {
                scope.set_self(None);
                if nested_type {
                    scope.without_self();
                }
            }
            self.visit_type_mut(default);
            if let (Some(scope), Some(own)) = (self.state.scopes.last_mut(), own) {
                *scope = own;
            }
            self.state.self_req = self_req;
        }
    }
    fn visit_const_param_mut(&mut self, i: &mut ConstParam) {
        // The type of a const parameter cannot be an associated type path, so it stays.
        self.visit_attrs(&mut i.attrs);
        if let Some(expr) = &mut i.default {
            self.visit_expr_mut(expr);
        }
    }
    fn visit_item_struct_mut(&mut self, i: &mut ItemStruct) {
        let declares = Nested::of_item(&i.ident, &i.generics);
        self.enter(&mut i.attrs, &mut i.generics, Some(declares), true)
            .visit_fields_mut(&mut i.fields);
    }
    fn visit_item_enum_mut(&mut self, i: &mut ItemEnum) {
        let declares = Nested::of_item(&i.ident, &i.generics);
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, Some(declares), true);
        for variant in i.variants.iter_mut() {
            visitor.visit_variant_mut(variant);
        }
    }
    fn visit_item_trait_mut(&mut self, i: &mut ItemTrait) {
        let declares = Nested::of_trait(&i.ident, &i.generics);
        let gats = Gat::of_items(&i.items);
        let depth = self.state.scopes.len();
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, Some(declares), true);
        // The scope of the trait knows what its generic associated types require, as the Leaker's
        // does. If the trait does not declare its generics (they are those of the context), the
        // scope of the context has them.
        if visitor.state.scopes.len() > depth {
            if let Some(scope) = visitor.state.scopes.last_mut() {
                scope.set_gats(gats);
            }
        }
        for supertrait in i.supertraits.iter_mut() {
            visitor.visit_type_param_bound_mut(supertrait);
        }
        for item in i.items.iter_mut() {
            visitor.visit_trait_item_mut(item);
        }
    }
    fn visit_item_trait_alias_mut(&mut self, i: &mut ItemTraitAlias) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, None, true);
        for bound in i.bounds.iter_mut() {
            visitor.visit_type_param_bound_mut(bound);
        }
    }
    fn visit_item_union_mut(&mut self, i: &mut ItemUnion) {
        let declares = Nested::of_item(&i.ident, &i.generics);
        self.enter(&mut i.attrs, &mut i.generics, Some(declares), true)
            .visit_fields_named_mut(&mut i.fields);
    }
    fn visit_item_type_mut(&mut self, i: &mut ItemType) {
        self.enter(&mut i.attrs, &mut i.generics, None, true)
            .visit_type_mut(&mut i.ty);
    }
    fn visit_signature_mut(&mut self, i: &mut Signature) {
        // The signature is visited on the way; the visitor inside it is not needed.
        let _ = self.enter_signature(&mut [], i, true);
    }
    fn visit_item_impl_mut(&mut self, i: &mut ItemImpl) {
        let declares = Nested::of_impl(&i.self_ty);
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, Some(declares), true);
        if let Some((_, path, _)) = &mut i.trait_ {
            visitor.visit_path_mut(path);
        }
        visitor.visit_type_mut(&mut i.self_ty);
        for item in i.items.iter_mut() {
            visitor.visit_impl_item_mut(item);
        }
    }
    fn visit_impl_item_fn_mut(&mut self, i: &mut ImplItemFn) {
        let mut visitor = self.enter_signature(&mut i.attrs, &mut i.sig, false);
        visitor.visit_block_mut(&mut i.block);
    }
    fn visit_impl_item_type_mut(&mut self, i: &mut ImplItemType) {
        self.enter(&mut i.attrs, &mut i.generics, None, false)
            .visit_type_mut(&mut i.ty);
    }
    fn visit_item_mod_mut(&mut self, i: &mut ItemMod) {
        self.visit_attrs(&mut i.attrs);
        if let Some((_, items)) = &mut i.content {
            let mut visitor = self.clone();
            // A module has no generics: it only makes its items nested. `Self` has no meaning in
            // it.
            visitor.state.item_depth += 1;
            visitor.state.self_req = None;
            if let OwnSelf::Nested(_) = visitor.state.own_self {
                visitor.state.own_self = OwnSelf::InMod;
            }
            visitor.shadow_items(items.iter());
            for item in items.iter_mut() {
                visitor.visit_item_mut(item);
            }
        }
    }
    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        let mut visitor = self.enter_signature(&mut i.attrs, &mut i.sig, true);
        visitor.visit_block_mut(i.block.as_mut());
    }
    fn visit_trait_item_fn_mut(&mut self, i: &mut TraitItemFn) {
        let mut visitor = self.enter_signature(&mut i.attrs, &mut i.sig, false);
        if let Some(block) = &mut i.default {
            visitor.visit_block_mut(block);
        }
    }
    fn visit_trait_item_type_mut(&mut self, i: &mut TraitItemType) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, None, false);
        for bound in i.bounds.iter_mut() {
            visitor.visit_type_param_bound_mut(bound);
        }
        if let Some((_, ty)) = &mut i.default {
            visitor.visit_type_mut(ty);
        }
    }
    fn visit_trait_item_const_mut(&mut self, i: &mut TraitItemConst) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, None, false);
        visitor.visit_type_mut(&mut i.ty);
        if let Some((_, expr)) = &mut i.default {
            visitor.visit_expr_mut(expr);
        }
    }
    fn visit_file_mut(&mut self, i: &mut File) {
        // The items of a file are declared together, as those of a module.
        let mut visitor = self.clone();
        visitor.shadow_items(i.items.iter());
        syn::visit_mut::visit_file_mut(&mut visitor, i)
    }
    fn visit_block_mut(&mut self, i: &mut Block) {
        let mut visitor = self.clone();
        visitor.shadow_items(i.stmts.iter().filter_map(|stmt| match stmt {
            Stmt::Item(item) => Some(item),
            _ => None,
        }));
        for stmt in i.stmts.iter_mut() {
            visitor.visit_stmt_mut(stmt);
        }
    }
}

// The serialized form of a Referrer, as its `ToTokens` writes it and its `Parse` reads it:
//
//     referrer    = tag [parts] [count] group ..
//     tag         = "type-leak-format-N"        (N is `FORMAT_VERSION`)
//     parts       = [ part , .. ]               (only if there is a part)
//     part        = KIND ( CAUSE ) context [whole] [sig ( mark )] [predicate] [in ( Type ) context]
//     mark        = elided | 'x , ..
//     context     = id | unwritable             (`unwritable`: no context can be written)
//     count       = the number of groups        (only if it is not 1)
//     group       = [# id] [declaration] ( [-] Type , .. )
//     declaration = [< generics > where .. ; | declared < params > where .. ;]
//                   [Self : path ; | Self = Type ; | Self ;]
//                   [Self : 'a + 'b ;]
//                   [type G < 'b > where Self : 'b ; ..]
//
// - The groups hold the entries in the order of their indices: a group is a run of the entries of
//   one context. A context is declared by its first group. Its other groups (the entries of a
//   context need not be together) have no declaration: `[# id] ( Type , .. )`.
// - `# id` is written only if `id` is not the one after that of the group before (the first group
//   is expected to have `0`).
// - A context with no entry has one group, `()`, before the first group of a larger id.
// - `< generics > where .. ;` is the generics of the context in their canonical form (sorted, the
//   bounds of a parameter in its own, no defaults). The top-level context of a scope writes
//   `declared < params > where .. ;` instead: the generics of the item as written (their order,
//   kinds, defaults, bounds and where clause), with `<` `>` when there are no parameters. The
//   parser computes the canonical form from it, and refuses a declaration that has both. The
//   generics as written are part of the identity of the context.
// - `- Type` is an entry that `LeakerResult::ignore()` removed: it has its index, no impl, and the
//   copy writes its type as written.
// - The `KIND ( CAUSE )` of a part is that of an `ErrorItem`. A bracket cannot start a group, so
//   `parts` needs no marker. The count lets other tokens follow the Referrer.
// - A `LeakerResult` writes the errors after the Referrer: `@ [ { error } .. ]`, each error
//   `KIND ( CAUSE ) Reason "message" notes [help] [value] [default] [paths [..]] [id N] [part N]
//   [entry N] [inner [N, ..]]`, where `notes` is `[ ( tokens ) "note" , .. ]`, even if empty.
//   `part N` and `entry N` are an index of `parts` and of all the entries (the removed ones
//   count): what `LeakerResult::ignore()` changes for the error. An error that another error holds
//   inside its part has an `id`, which the `inner` of the other lists. The parser refuses an index
//   that is not there.

const FORMAT_TAG_PREFIX: &str = "type-leak-format-";

impl Referrer {
    /// The version of the serialized form, written as the tag `"type-leak-format-N"`.
    ///
    /// Parsing rejects another tag, or none.
    // Bump by hand when an old serialized Referrer would behave differently: the syntax, what a
    // context holds, how enclosing lifetimes pass on, which paths are associated type paths, or how
    // an entry is found. Docs and error texts need no bump.
    //
    // 1: first tagged, contexts carry the bounds enclosing types imply.
    // 2: modules are contexts (`mod 3 ;`, `mod { .. } ;`), no `Self = # 0 ;`.
    // 3: the top-level context of a scope keeps what `Self` means and the module the scope is in,
    //    for `Referrer::visitor()`.
    // 4: it keeps `Self` also when an outer bound was rewritten for it (`T: Tr<Self>` as
    //    `T: Tr<S<T>>`). The tokens look the same, but the data of 3 would not find those entries
    //    the same way.
    // 5: the context of `&'a Self::A` in a trait has the where predicate `Self::A: 'a`, which the
    //    Referrer computes for a copy too, so the entries of 4 are not found.
    // 6: so has that of `&'a T::A` or `&'a <T as Tr>::A` with `T` a generic (`T::A: 'a`, and no
    //    `T: 'a`).
    // 7: so have those of `&'a <Self as Tr<..>>::A` and `&'a Self::A<'b>` (the predicate, and no
    //    `'b: 'a` or `T: 'a`), and of a `Self::A` that a where clause equates (what it equals is
    //    mentioned); no projection of `Self` records `Self: 'a` any more.
    // 8: the top-level context of a trait scope carries its generic associated types with outlives
    //    (`type G<'b> where Self: 'b ;`), and the context of an entry that names one has what its
    //    where clause requires (`Self: 'x`), as the Referrer computes it for a copy.
    // 9: the Referrer computes the context of a type from the first segment of a relative path
    //    only, so `inner::U` next to a generic `U` is found.
    // 10: the Referrer finds an entry of no module that names a generic of its context also where
    //    the copy declares an item of the same name (`crate::base::B1<X>` next to `fn f<X>` in a
    //    module that declares `X`).
    // 11: the context of an entry that does not name the `Self` of a struct, an enum or a union
    //    nested in the analyzed syntax, but has a bound that does, is `Self ;` (any type), and the
    //    Referrer finds it only in the item that declares that `Self`.
    // 12: an absolute type that names the `Self` of such an item is no optional entry any more (it
    //    is copied as written, so its `Self` is the copy's own), and a strict Referrer replaces
    //    less.
    // 13: the default of a parameter of a trait nested in the analyzed syntax is looked up with the
    //    trait's `Self` too (`U = Rel<T>` with `T: AsRef<Self>`), and that of the item itself
    //    without `Self`, as the Leaker analyzes it.
    // 14: the context of an entry in the default of a parameter of a struct, an enum or a union
    //    nested in the analyzed syntax has no bound of the generics that names the item's `Self`
    //    (`U = Wrap<T>` with `T: Clone + AsRef<Self>` is `T: Clone`, any type), which the Referrer
    //    computes for a copy, so the entries of 13 are not found the same way; and an entry that
    //    is for any type may name `Self` in its type (`NestedSelfInBounds`), which the Referrer
    //    finds too.
    // 15: the context of an entry leaves out a bound of a generic that names the generic in the
    //    arguments of its path (`T: Fn(Alias<T>)`), if another bound of it fixes an associated
    //    type (`T: Iterator<Item = u8>`), which the Referrer computes for a copy, so the entries of
    //    14 are not found the same way.
    // 16:
    //    - The Referrer carries the types that the Leaker copies as written, the parts (see the
    //      grammar above), and the visitor does not look each up as an entry; in a `whole` one
    //      nothing is looked up. A part that is whole, has an anchor or a mark is needed by the
    //      copy; a type copied as written whose head alone conflicts is a part too if an error is
    //      about it, but the visitor does not look at it until `LeakerResult::ignore()` makes it
    //      whole (an entry of its tokens and context would have the same conflict, and the types
    //      in it are replaced as any entry is). A part that a signature made has a mark:
    //      `sig ('x)`, the method lifetimes of the type that the output names and no bound does
    //      (E0195), where the copy leaves one unconstrained: only types leaked whole hold it, and
    //      no input keeps it as written; `sig (elided)`, the signature relies on elision (E0106).
    //      It is left only in the arguments of a signature that has the same fault, which the
    //      Referrer computes at each.
    //    - The context of an entry that depends on `Self` only through a bound or an outlives
    //      predicate (`Self: 'x`) keeps whole the bounds of the generics that the description of
    //      `Self` names (`Tr<T>`) and leaves out those of the others (15 kept all), which the
    //      Referrer computes for a copy.
    //    - The context of an entry has the bounds that a scope inside the item (a method, a generic
    //      associated type) writes on a generic of the item (`fn m(&self, a: Cl<T>) where T:
    //      Clone` gives `<T: Clone>`), which the Referrer computes for a copy.
    //    - A `LeakerResult` writes its errors after the Referrer as `@ [ {error} .. ]`, in one
    //      bracket group, so that a brace group after it is left to the caller.
    //    - An entry carries no list of the crates it names (15 wrote `@[a, b]` before its type)
    //      and no mark of its kind (it was `[?] Type`, `?` for an optional one), so it is `Type`:
    //      the Referrer does not tell a kind, and every entry is replaced.
    //    - The groups follow the entries in the order of their indices, which the parser keeps: a
    //      context whose entries are not together (the required ones come first) is declared by
    //      its first group and written again as `[# id] ( .. )`, and the number before the groups
    //      is the number of groups.
    //    - An error writes each of its paths with its kind (`paths [type (Rel), trait (Tr), ..]`;
    //      a path of unknown kind is `unknown (p)`).
    //    - An error writes what it is about in the Referrer, for `LeakerResult::ignore()`: the part
    //      (`part N`) or the entry (`entry N`), the errors inside the part (`inner [N, ..]`, each
    //      written with its `id N`).
    //    - An entry that `LeakerResult::ignore()` removed is written `- Type` in its group: it
    //      keeps its index, has no impl, and the copy writes its type as written.
    //    - The top-level context of a scope carries the generics of the item as written
    //      (`declared < .. > ;`, see the grammar above), for `Referrer::complete_args()`, in place
    //      of its canonical generics, which the parser computes from them. They are part of the
    //      identity of the context, so that two traits with the same canonical generics and
    //      other declarations (`<T, U = u8>` and `<T, U = u16>`) are two contexts: the top-level
    //      context of a scope is no more the context of an entry that has its generics, and the
    //      ids of the contexts that come after it are one more.
    //    - Modules are no contexts any more: a declaration has no `mod id ;` and no
    //      `mod { path = [..], hint = "..", parent = id } ;` (2 brought them), and a context is no
    //      module's, so the entries of 15 that a module keys are not found.
    //    - Not serialized: the Repeater has one `KEY: u128` (`Projection::key()`) instead of
    //      `ID: u64, NTH: usize`, and a `U64` slot instead of the `U128` one (a `u128` const goes
    //      in `KEY`), so the impls and projections of 15 do not match.
    //
    // Not serialized, but both macros must agree on it: the primitive type that `ImplSelf::Bucket`
    // and `Projection::of()` choose from the id and the index of an entry, the layout of
    // `Projection::key()`, and the slots of `repeater_definition()` and `encode_slot_const()`
    // (which const type goes where) must not change without a bump.
    pub const FORMAT_VERSION: u32 = 16;
}

/// Parses a [`Referrer`] written by its [`ToTokens`] implementation.
///
/// Entries keep their indices, and the tokens after the Referrer are not consumed. Fails if the
/// format tag ([`Referrer::FORMAT_VERSION`]) is missing or differs, or if the contexts are
/// inconsistent (a context used but not declared).
impl Parse for Referrer {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        read_tag(input)?;
        let (parts, part_contexts) = read_parts(input)?;
        let n_groups = if input.peek(LitInt) {
            input.parse::<LitInt>()?.base10_parse::<usize>()?
        } else {
            1
        };
        let mut reader = Reader::default();
        for _ in 0..n_groups {
            reader.group(input)?;
        }
        reader.check(&part_contexts)?;
        Ok(Self::new(
            reader.entries,
            reader.removed,
            reader.contexts,
            parts,
        ))
    }
}

/// Returns `true` if the next token is the identifier `word`.
fn peek_word(input: parse::ParseStream, word: &str) -> bool {
    input
        .cursor()
        .ident()
        .map_or(false, |(ident, _)| ident == word)
}

/// Reads the format tag, and fails if it is not the one of this version.
fn read_tag(input: parse::ParseStream) -> Result<()> {
    let span = input.span();
    let tag = input.parse::<Option<LitStr>>()?;
    let value = tag.as_ref().map(LitStr::value);
    let expected = format!("{}{}", FORMAT_TAG_PREFIX, Referrer::FORMAT_VERSION);
    if value.as_deref() != Some(expected.as_str()) {
        let found = match &value {
            None => "no format tag".to_string(),
            Some(value) => match value.strip_prefix(FORMAT_TAG_PREFIX) {
                Some(number) if !number.is_empty() => format!("format {}", number),
                _ => format!("an unrecognized tag {:?}", value),
            },
        };
        return Err(Error::new(
            tag.map_or(span, |tag| tag.span()),
            format!(
                "this data was serialized by an incompatible type-leak ({}, expected {}): \
                 the macros which emit and parse it must use the same version of type-leak",
                found,
                Referrer::FORMAT_VERSION
            ),
        ));
    }
    Ok(())
}

/// The contexts that the tokens name, each with its span: they are checked when all the contexts
/// are known.
type ContextRefs = Vec<(Span, ContextId)>;

/// Reads the parts copied as written, if any, and the contexts they name.
fn read_parts(input: parse::ParseStream) -> Result<(Vec<PartItem>, ContextRefs)> {
    let mut parts = Vec::new();
    let mut part_contexts = Vec::new();
    if input.peek(token::Bracket) {
        let list: parse::ParseBuffer<'_>;
        bracketed!(list in input);
        while !list.is_empty() {
            let span = list.span();
            let item: PartItem = list.parse()?;
            part_contexts.extend(item.context.map(|context| (span, context)));
            part_contexts.extend(item.within.iter().map(|(_, anchor)| (span, *anchor)));
            parts.push(item);
            if !list.is_empty() {
                list.parse::<Token![,]>()?;
            }
        }
    }
    Ok((parts, part_contexts))
}

/// Reads what a group has before its entries when it declares `context`.
fn read_declaration(input: parse::ParseStream, context: ContextId) -> Result<Context> {
    let has_generics = input.peek(Token![<]) || input.peek(Token![where]);
    let mut generics = Generics::default();
    if has_generics {
        generics = input.parse()?;
        generics.where_clause = input.parse()?;
        input.parse::<Token![;]>()?;
    }
    // The generics as written, in the top-level context of a scope. They give the generics of
    // the context, so a declaration has one of the two forms.
    let declared_span = input.span();
    let mut declared = None;
    if peek_word(input, "declared") {
        if has_generics {
            return Err(Error::new(
                declared_span,
                format!(
                    "the declaration of the context {} has both `< generics >` and \
                     `declared < .. >`: `declared` gives the generics, so write only one",
                    context
                ),
            ));
        }
        input.parse::<Ident>()?;
        if !input.peek(Token![<]) {
            return Err(input.error("expected `<` after `declared`"));
        }
        let mut written: Generics = input.parse()?;
        written.where_clause = input.parse()?;
        input.parse::<Token![;]>()?;
        declared = Some(written);
    }
    // `Self: 'a + 'b ;` is told from `Self: path ;` by the lifetime.
    let mut self_ty = None;
    let mut self_outlives = Vec::new();
    while input.peek(Token![Self]) {
        input.parse::<Token![Self]>()?;
        if input.peek(Token![;]) {
            self_ty = Some(SelfRequirement::Any);
        } else if input.parse::<Option<Token![:]>>()?.is_none() {
            input.parse::<Token![=]>()?;
            self_ty = Some(SelfRequirement::Type(input.parse()?));
        } else if input.peek(Lifetime) {
            self_outlives = Punctuated::<Lifetime, Token![+]>::parse_separated_nonempty(input)?
                .into_iter()
                .collect();
        } else {
            self_ty = Some(SelfRequirement::Bound(input.parse()?));
        }
        input.parse::<Token![;]>()?;
    }
    // The associated types of the trait of the scope, with the outlives they require.
    let mut gats = Vec::new();
    let gats_span = input.span();
    while input.peek(Token![type]) {
        gats.push(Gat::parse(input)?);
    }
    if !gats.is_empty() && !matches!(self_ty, Some(SelfRequirement::Bound(_))) {
        return Err(Error::new(
            gats_span,
            "the associated types of a trait need `Self: Trait ;` before them",
        ));
    }
    // As the Leaker makes it: `Context::new()` gives the canonical generics.
    let found = match declared {
        Some(declared) => Context::new(declared.clone(), self_ty).with_declared(declared),
        None => Context::new(generics, self_ty),
    };
    Ok(found.with_self_outlives(self_outlives).with_gats(gats))
}

/// What the parser has read of the groups.
#[derive(Default)]
struct Reader {
    /// The entries read, each with the id of its context.
    entries: Vec<(ContextId, Type)>,
    /// The indices of the entries that were written as removed (`- Type`).
    removed: BTreeSet<usize>,
    /// The contexts read.
    contexts: BTreeMap<ContextId, Context>,
    /// The id of a group that writes no `#`.
    next: usize,
}

impl Reader {
    /// Reads a group: `[# id] [declaration] ( Type , .. )`.
    fn group(&mut self, input: parse::ParseStream) -> Result<()> {
        let span = input.span();
        let context = if input.parse::<Option<Token![#]>>()?.is_some() {
            input.parse()?
        } else {
            ContextId::new(self.next)
        };
        self.next = context
            .index()
            .checked_add(1)
            .ok_or_else(|| Error::new(span, "the id of the context is too large"))?;
        if self.contexts.contains_key(&context) {
            // Another group of a declared context: the entries only.
            if !input.peek(token::Paren) {
                return Err(Error::new(
                    input.span(),
                    format!(
                        "the context {} is declared already: expected its entries `( .. )`",
                        context
                    ),
                ));
            }
            return self.read_entries(input, context);
        }
        let declared = read_declaration(input, context)?;
        self.contexts.insert(context, declared);
        self.read_entries(input, context)
    }

    /// Reads the entries `( Type , .. )` of a group of `context`.
    fn read_entries(&mut self, input: parse::ParseStream, context: ContextId) -> Result<()> {
        let content: parse::ParseBuffer<'_>;
        parenthesized!(content in input);
        // `Type` or `- Type` (removed), separated by commas.
        while !content.is_empty() {
            if content.parse::<Option<Token![-]>>()?.is_some() {
                self.removed.insert(self.entries.len());
            }
            self.entries.push((context, content.parse::<Type>()?));
            if !content.is_empty() {
                content.parse::<Token![,]>()?;
            }
        }
        Ok(())
    }

    /// Checks what refers to a context, now that all the contexts are known: the contexts of the
    /// parts.
    fn check(&self, part_contexts: &[(Span, ContextId)]) -> Result<()> {
        for (span, context) in part_contexts {
            if !self.contexts.contains_key(context) {
                return Err(Error::new(
                    *span,
                    format!(
                        "the context {} of a part is not one of the contexts",
                        context
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// Reads a part written by its [`ToTokens`] implementation.
impl Parse for PartItem {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let kind = input.call(Ident::parse_any)?;
        let cause;
        parenthesized!(cause in input);
        let part = Cause::parse(&kind, &cause)?;
        let context = if input.peek(Ident) {
            let word: Ident = input.parse()?;
            if word != "unwritable" {
                return Err(Error::new(
                    word.span(),
                    "expected a context or `unwritable`",
                ));
            }
            None
        } else {
            Some(input.parse()?)
        };
        let mut mark = None;
        let (mut whole, mut in_predicate) = (false, false);
        // `whole`, `sig (..)` and `predicate`, in this order: what the words so far allow is next.
        let mut stage = 0;
        while stage < 3 && input.peek(Ident) {
            let word: Ident = input.parse()?;
            match (word.to_string().as_str(), stage) {
                ("whole", 0) => {
                    whole = true;
                    stage = 1;
                }
                ("sig", 0 | 1) => {
                    let content;
                    parenthesized!(content in input);
                    mark = Some(content.parse()?);
                    stage = 2;
                }
                ("predicate", 0..=2) => {
                    in_predicate = true;
                    stage = 3;
                }
                _ => {
                    let expected = match stage {
                        0 => "expected `whole`, `sig`, `predicate`, `in` or `,`",
                        1 => "expected `sig`, `predicate`, `in` or `,`",
                        _ => "expected `predicate`, `in` or `,`",
                    };
                    return Err(Error::new(word.span(), expected));
                }
            }
        }
        // The anchor: the type that owns the cause of the conflict, in its context.
        let mut within = None;
        if input.parse::<Option<Token![in]>>()?.is_some() {
            let anchor;
            parenthesized!(anchor in input);
            within = Some((anchor.parse()?, input.parse()?));
        }
        Ok(PartItem {
            part,
            whole,
            context,
            in_predicate,
            within,
            mark,
        })
    }
}

/// Writes the part as tokens, to be read back by its [`Parse`] implementation.
impl ToTokens for PartItem {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let PartItem {
            part,
            whole,
            context,
            in_predicate,
            within,
            mark,
        } = self;
        let kind = Ident::new(part.kind(), Span::call_site());
        let context = match context {
            Some(context) => quote!(#context),
            None => quote!(unwritable),
        };
        tokens.extend(quote! {
            #kind (#part) #context
            #(if *whole) { whole }
            #(if let Some(mark) = mark) { sig (#mark) }
            #(if *in_predicate) { predicate }
            #(if let Some((anchor, anchor_context)) = within) {
                in (#anchor) #anchor_context
            }
        });
    }
}

/// Reads `elided`, or lifetimes separated by commas, with none twice.
impl Parse for SignatureMark {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        if input.peek(Ident) {
            let word: Ident = input.parse()?;
            if word != "elided" || !input.is_empty() {
                return Err(Error::new(word.span(), "expected `elided` or lifetimes"));
            }
            return Ok(Self::Elided);
        }
        let mut lifetimes: Vec<Lifetime> = Vec::new();
        loop {
            let lifetime: Lifetime = input.parse()?;
            if has_lifetime(&lifetimes, &lifetime) {
                return Err(Error::new(lifetime.span(), "the lifetime is written twice"));
            }
            lifetimes.push(lifetime);
            if input.is_empty() {
                return Ok(Self::Unconstrained(lifetimes));
            }
            input.parse::<Token![,]>()?;
        }
    }
}

impl ToTokens for SignatureMark {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Elided => tokens.extend(quote!(elided)),
            Self::Unconstrained(lifetimes) => {
                tokens.extend(quote!(#(for lifetime in lifetimes), { #lifetime }))
            }
        }
    }
}

/// Writes the [`Referrer`] as tokens, to be read back by its [`Parse`] implementation.
///
/// The tokens start with the format tag ([`Referrer::FORMAT_VERSION`]). Treat the rest as opaque.
impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        // The grammar is above, at `FORMAT_TAG_PREFIX`.
        let groups = self.groups();
        let tag = format!("{}{}", FORMAT_TAG_PREFIX, Self::FORMAT_VERSION);
        let n_groups = groups.len();
        tokens.extend(quote! {
            #tag
            #(if !self.parts.is_empty()) { [#(for item in &self.parts), { #item }] }
            // The group count lets other tokens follow the Referrer.
            #(if n_groups > 1) { #{Literal::usize_unsuffixed(n_groups)} }
        });
        let mut next = 0;
        let mut declared = HashSet::new();
        for (context, run) in groups {
            tokens.extend(quote! {
                // `#` cannot be written in `quote!`, where it interpolates.
                #(if context.index() != next) { #{Punct::new('#', Spacing::Alone)} #context }
            });
            if declared.insert(context) {
                self.write_declaration(context, tokens);
            }
            tokens.extend(quote! {
                (#(for ix in run), {
                    #(if self.removed.contains(&ix)) { - }
                    #{&self.entries[ix].0}
                })
            });
            next = context.index() + 1;
        }
    }
}

impl Referrer {
    /// Returns the groups to write, each with its context and the range of the entries it holds:
    /// the runs of the entries of one context, and, for each context with no entry, a group with
    /// the empty range, before the first group of a larger id.
    fn groups(&self) -> Vec<(ContextId, std::ops::Range<usize>)> {
        let mut runs: Vec<(ContextId, std::ops::Range<usize>)> = Vec::new();
        for (ix, &(_, context)) in self.entries.iter().enumerate() {
            match runs.last_mut() {
                Some((last, run)) if *last == context => run.end = ix + 1,
                _ => runs.push((context, ix..ix + 1)),
            }
        }
        let used: HashSet<ContextId> = self.entries.iter().map(|(_, context)| *context).collect();
        let mut empty = self
            .contexts
            .keys()
            .copied()
            .filter(|context| !used.contains(context))
            .peekable();
        let mut groups = Vec::new();
        for (context, run) in runs {
            while let Some(skipped) = empty.next_if(|skipped| *skipped < context) {
                groups.push((skipped, 0..0));
            }
            groups.push((context, run));
        }
        groups.extend(empty.map(|context| (context, 0..0)));
        groups
    }

    /// Writes what a group has before its entries when it declares `context`.
    fn write_declaration(&self, context: ContextId, tokens: &mut TokenStream) {
        let found = self.context(context);
        let (generics, outlives) = (&found.generics, &found.self_outlives);
        match &found.declared {
            // The parser computes the generics of the context from them.
            Some(declared) => {
                let (params, where_clause) = (&declared.params, &declared.where_clause);
                tokens.extend(quote!(declared < #params > #where_clause ;));
            }
            None => tokens.extend(quote! {
                #(if !generics.params.is_empty() || generics.where_clause.is_some()) {
                    #generics #{&generics.where_clause} ;
                }
            }),
        }
        match &found.self_ty {
            None => (),
            Some(SelfRequirement::Bound(path)) => tokens.extend(quote!(Self : #path ;)),
            Some(SelfRequirement::Type(ty)) => tokens.extend(quote!(Self = #ty ;)),
            Some(SelfRequirement::Any) => tokens.extend(quote!(Self ;)),
        }
        if !outlives.is_empty() {
            tokens.extend(quote!(Self : #(for lifetime in outlives)+ { #lifetime } ;));
        }
        for gat in &found.gats {
            tokens.extend(quote!(#gat));
        }
    }
}

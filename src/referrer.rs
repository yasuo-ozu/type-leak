//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::analysis::{PartItem, SignatureFacts, SignatureMark};
use crate::context::{Context, ContextId, Enclosing, Gat, ModuleScope, Scope, SelfRequirement};
use crate::elided::{
    name_elided_in_bare_fn, name_elided_in_bound, Elision, Names, SelfWriter, WrittenNames,
};
use crate::encode::{
    complete_generics_args, encode_generics_args_with_params, encode_generics_params_with,
    without_encoded_bounds, EncodeError,
};
use crate::error::Cause;
use crate::graph::TypeKey;
use crate::helpers::{ImplSelf, RepeaterImpl, UseSite};
#[cfg(doc)]
use crate::repeater::repeater_definition;
use crate::util::{
    bound_cause, first_self_in_generics, generic_param_name, ident_name, mentions_names,
    parenthesize_bounded, type_mentions_self, ReplaceSelf,
};
use proc_macro2::{Literal, Punct, Spacing, Span, TokenStream};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use syn::ext::IdentExt;
use syn::fold::Fold;
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};

/// The leaked types of an analyzed item (its *entries*, each with an index `ix`).
///
/// [`Leaker::result()`](crate::Leaker::result) returns it, in a
/// [`LeakerResult`](crate::LeakerResult). Crate A, which defines the item, emits the Repeater impl
/// of each entry ([`repeater_impl()`](Referrer::repeater_impl)). Crate B, which copies the item,
/// replaces each entry by `<Self as Repeater<ARGS, KEY>>::Type` with
/// [`visitor()`](Referrer::visitor) (`ARGS` from [`entry_args()`](Referrer::entry_args), `KEY`
/// from [`repeater_key()`](crate::repeater_key)).
///
/// A's macro passes the `Referrer` and the top-level [`ContextId`] to B's as tokens ([`ToTokens`],
/// [`Parse`]); both crates must use the same version of `type-leak`. Emit an impl for every entry.
/// A type of another crate (`::foo::X`) is an entry like the others, unless the Leaker's rule allows
/// its path ([`Rules::external_crates()`](crate::Rules::external_crates)): it is no entry then, and
/// the copy writes it as it is. `examples/two-macros` is a complete pair.
///
/// A method that takes an entry index `ix` panics if it is out of range ([`iter()`](Referrer::iter)
/// has the valid ones).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use syn::parse::Parser;
/// use quote::{quote, ToTokens};
///
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_struct(&parse_quote!(struct S<T> { a: Wrap<T> }), true);
/// let referrer = leaker.result().finish().unwrap();
///
/// // Crate A writes the Referrer and the `ContextId` into the input of the macro of crate B...
/// let tokens = quote!(#referrer, #top_level);
/// // ...which reads them back.
/// let (read, read_top_level) = (|input: parse::ParseStream| -> Result<(Referrer, ContextId)> {
///     let referrer = input.parse()?;
///     input.parse::<Token![,]>()?;
///     Ok((referrer, input.parse()?))
/// })
/// .parse2(tokens)
/// .unwrap();
/// assert_eq!(read, referrer);
/// assert_eq!(read_top_level, top_level);
/// ```
#[derive(Clone, Debug)]
pub struct Referrer {
    /// The type of each entry and the id of its context (a key of `contexts`), by index.
    entries: Vec<(Type, ContextId)>,
    /// Every context, by id. The context of a module has its path, hint and parent
    /// (`ModuleScope::Itself`).
    contexts: BTreeMap<ContextId, Context>,
    /// The indices of the entries, keyed by tokens as in the Leaker, so a type in an invisible
    /// group is found too.
    by_tokens: HashMap<TypeKey, Vec<usize>>,
    /// The types and bounds copied as written that a position tells from an entry of the same
    /// tokens and context: a part is whole (nothing in it is looked up), or has an anchor or a
    /// signature mark. Serialized after the format tag.
    parts: Vec<PartItem>,
    /// The indices in `parts`, by the part.
    parts_by_cause: HashMap<Cause, Vec<usize>>,
    /// The indices in `parts` of the parts with an anchor (`within`), by the anchor, as a
    /// `Cause::Type`.
    parts_by_anchor: HashMap<Cause, Vec<usize>>,
}

/// The path, hint and parent of a module are no part of the identity of its `Context`, but they are
/// part of the Referrer. The maps are made from the entries and the parts (`Referrer::new()`), so
/// they are not compared.
impl PartialEq for Referrer {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            entries,
            contexts,
            parts,
            ..
        } = self;
        *entries == other.entries
            && *contexts == other.contexts
            && contexts
                .values()
                .zip(other.contexts.values())
                .all(|(a, b)| a.module == b.module)
            && *parts == other.parts
    }
}

impl Eq for Referrer {}

/// Where a visitor is, among the modules of the Leaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModulePos {
    /// In no module: the context has none, so only entries with no module match. A `mod` item is
    /// a module found with no module around it.
    Outside,
    In(ContextId),
    /// In a module no entry depends on: only entries with no module match.
    Elsewhere,
}

impl ModulePos {
    /// Returns the module of the scope of `context`: the context itself if it is a module, else the
    /// module it was entered in.
    fn of(context: &Context) -> Self {
        context.module_id().map_or(Self::Outside, Self::In)
    }

    /// Returns how well an entry with the context `entry` fits here: `Some(0)` if it is of the
    /// module, `Some(1)` if it has no module, `None` if it is of another module.
    fn rank(self, entry: &Context) -> Option<usize> {
        match (self, entry.in_module()) {
            (Self::In(module), Some(entry)) if entry == module => Some(0),
            (_, Some(_)) => None,
            (_, None) => Some(1),
        }
    }
}

/// Returns the generics and what `Self` means in the scope of `context`.
fn scope_of(context: &Context) -> Scope {
    Scope::new(context.generics(), context.self_ty().cloned()).with_gats(context.gats().to_vec())
}

/// How the visitor looks up the default of a type parameter, while it visits the generics of an
/// item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Defaults {
    /// Not in such generics: a type is looked up as anywhere.
    Plain,
    /// With no `Self`. That is how the Leaker analyzes the default of the generics of a scope that
    /// it enters (`enter_*`, `intern_*`: alone, with the parameters before it), whatever `Self`
    /// the scope declares, and a default cannot name `Self` in a struct, an enum or a union. So the
    /// bounds of the scopes around that name `Self` are read for the `Self` they were written for,
    /// also where the nested scope has the same one, and the scope itself is declared without
    /// `Self` for its default. A struct, an enum or a union nested in the visited syntax has none
    /// of its own bounds that name its `Self` there: its copy cannot write `Self` in the default
    /// (E0735), so the entries in it do not depend on what the copy's own type proves
    /// (`T: AsRef<Self>`).
    WithoutSelf,
    /// The generics of a trait nested in the visited syntax, while they are visited. The Leaker
    /// has analyzed the default of such a trait without `Self`, as it does for a scope that it
    /// enters ([`Leaker::enter_trait()`](crate::Leaker::enter_trait)), unless the default names
    /// `Self` (`Rhs = Self`): then with the `Self` of the trait
    /// ([`View::visit_item()`](crate::View::visit_item), the items of
    /// [`Leaker::intern_mod()`](crate::Leaker::intern_mod)).
    OwnTrait,
    /// Visiting a default of `OwnTrait`: an entry analyzed without `Self` is preferred, else the
    /// one of the trait's `Self` (written by `Visitor::with_nested_self()`).
    OwnTraitDefault,
}

/// How [`Referrer::find()`] compares the context of a type with those of the entries.
#[derive(Clone, Copy)]
enum Matching<'a> {
    /// The context of the type, computed from the generics in scope and `self_ty`, is the entry's.
    Scope {
        self_ty: Option<&'a SelfRequirement>,
        /// `Self` means something else here, so no entry that names it matches.
        no_self: bool,
        /// `self_ty` is the `Self` of a struct, an enum or a union nested in the visited syntax,
        /// so a type that does not name it but has a bound that does is for any type
        /// ([`SelfRequirement::Any`]).
        nested_type: bool,
        /// The type is the default of a parameter of a nested trait, whose entry is the one
        /// analyzed without `Self` if there is one (see [`Defaults::OwnTrait`]).
        alone: bool,
    },
    /// The scope is that of the parameters before a defaulted one: the entry must use only them.
    Default,
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

/// How `Referrer::find()` and `Referrer::part_of()` look a type up.
#[derive(Clone, Copy)]
struct Lookup<'a> {
    matching: Matching<'a>,
    module: ModulePos,
    /// The entries that a visited item hides (see `Visitor::shadow_items()`).
    hidden: Option<&'a HashSet<usize>>,
    place: PartPlace,
    /// The signature that the visited syntax is in, for the parts that a signature made.
    signature: Option<&'a SignatureFacts>,
}

impl Referrer {
    /// Adds the context 0 if there is none: the serialized form needs a group
    /// (`Leaker::new().result()` writes `"tag" ()`).
    pub(crate) fn new(
        entries: impl IntoIterator<Item = (ContextId, Type)>,
        mut contexts: BTreeMap<ContextId, Context>,
        parts: Vec<PartItem>,
    ) -> Self {
        if contexts.is_empty() {
            contexts.insert(ContextId::new(0), Context::none());
        }
        let mut referrer = Self {
            entries: Vec::new(),
            contexts,
            by_tokens: HashMap::new(),
            parts,
            parts_by_cause: HashMap::new(),
            parts_by_anchor: HashMap::new(),
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

    /// Returns the entry for `ty` that matches: the first of the best rank, then of the least cost.
    /// An entry of `module` comes before one with no module, and an entry of another module never
    /// matches (see [`Referrer::visitor()`]).
    fn find(
        &self,
        at: Lookup,
        ty: &Type,
        scopes: &[Scope],
        enclosing: &Enclosing,
    ) -> Option<usize> {
        let Lookup {
            matching,
            module,
            hidden,
            ..
        } = at;
        let no_self = matches!(matching, Matching::Scope { no_self: true, .. });
        let mut best: Option<((usize, usize), usize)> = None;
        for &ix in self.by_tokens.get(&TypeKey::new(ty))? {
            if hidden.map_or(false, |hidden| hidden.contains(&ix)) {
                continue;
            }
            let (entry_ty, context) = &self.entries[ix];
            let entry = &self.contexts[context];
            if no_self && (entry.self_ty().is_some() || type_mentions_self(entry_ty)) {
                continue;
            }
            let rank = match module.rank(entry) {
                Some(rank) => rank,
                None => continue,
            };
            if let Some(cost) = Self::cost(matching, ty, scopes, enclosing, entry) {
                let key = (rank, cost);
                if best.map_or(true, |(better, _)| key < better) {
                    best = Some((key, ix));
                }
            }
        }
        Some(best?.1)
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
        let found = &self.contexts[&context];
        at.module.rank(found).is_some()
            && Self::cost(at.matching, ty, scopes, enclosing, found).is_some()
    }

    /// Returns `true` if the context of `ty` cannot be written at `at`, in `scopes` (see
    /// [`Context::of()`]): where the Leaker recorded a part with no context.
    fn unwritable(at: Lookup, ty: &Type, scopes: &[Scope], enclosing: &Enclosing) -> bool {
        match at.matching {
            Matching::Scope {
                self_ty,
                nested_type,
                ..
            } => Context::of(ty, scopes, self_ty, nested_type, enclosing).is_err(),
            Matching::Default => false,
        }
    }

    /// Returns what the part is that `ty` is, if it is one: whether it is copied whole (see
    /// [`PartItem::whole`]). A part has the same tokens, is in a predicate as `at` says, and has the
    /// context that `ty` has in `scopes` (none, if `ty` has none there). A part with an anchor counts
    /// only if its anchor is one of the `active` ones ([`Referrer::anchors()`]).
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
            let matches = *in_predicate == at.place.in_predicate
                && (within.is_none() || active.contains(&ix))
                && match (mark, at.signature) {
                    (None, _) => true,
                    // Made by a signature: in the arguments of one that has the same fault.
                    (Some(mark), Some(facts)) => {
                        at.place.in_fn_arg
                            && facts.covers(mark, context.map(|context| &self.contexts[&context]))
                    }
                    (Some(_), None) => false,
                }
                && match context {
                    Some(context) => self.fits(*context, ty, at, scopes, enclosing),
                    None => Self::unwritable(at, ty, scopes, enclosing),
                };
            if matches {
                found = Some(found.unwrap_or(false) || *whole);
            }
        }
        found
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

    /// Returns where a visited `mod ident` is, inside `module`: the module nested in it with that
    /// name (see [`Referrer::visitor()`]). A module that the Leaker did not see, or one nested in
    /// such a module, is a module no entry depends on.
    fn module_in(&self, module: ModulePos, ident: &Ident) -> ModulePos {
        let parent = match module {
            ModulePos::Outside => None,
            ModulePos::In(parent) => Some(parent),
            ModulePos::Elsewhere => return ModulePos::Elsewhere,
        };
        self.contexts
            .values()
            .find_map(|context| match &context.module {
                Some(ModuleScope::Itself {
                    id,
                    path: Some(path),
                    parent: found,
                    ..
                }) if *found == parent && path.last() == Some(ident) => Some(ModulePos::In(*id)),
                _ => None,
            })
            .unwrap_or(ModulePos::Elsewhere)
    }

    /// Returns the entries that items declared together with the `names` hide.
    ///
    /// For a `module` (or a `File`), only entries with no module are hidden: the others are keyed
    /// by their module.
    fn hidden_by(&self, names: &[Ident], module: bool) -> Vec<usize> {
        let mut hidden = Vec::new();
        for (key, entries) in &self.by_tokens {
            if !mentions_names(&key.ty, names.to_vec(), Vec::new()) {
                continue;
            }
            for &ix in entries {
                let context = &self.contexts[&self.entries[ix].1];
                if module && context.in_module().is_some() {
                    continue;
                }
                // A generic of the entry's context is the name in the type wherever the entry is
                // found, and it hides the item of the same name there (the Leaker does not
                // take it for the item either: `starts_with_generic()`).
                let items = names.iter().filter(|name| {
                    let name = ident_name(name);
                    !context
                        .generics()
                        .params
                        .iter()
                        .any(|param| generic_param_name(param) == name)
                });
                if mentions_names(&key.ty, items.cloned().collect(), Vec::new()) {
                    hidden.push(ix);
                }
            }
        }
        hidden
    }

    /// Returns the cost of `entry` for `ty`, or `None` if it does not match. It is `0` if the
    /// context of `ty` in `scopes` is `entry`'s. The entries of a default have a choice: the
    /// fewer parameters the entry uses, the less its cost.
    fn cost(
        matching: Matching,
        ty: &Type,
        scopes: &[Scope],
        enclosing: &Enclosing,
        entry: &Context,
    ) -> Option<usize> {
        match matching {
            Matching::Scope {
                self_ty,
                nested_type,
                alone,
                ..
            } => {
                let entry = entry.without_module();
                if alone {
                    let mut scopes = scopes.to_vec();
                    if let Some(last) = scopes.last_mut() {
                        last.set_self(None);
                    }
                    let alone = Context::of(ty, &scopes, None, false, enclosing);
                    if alone.map_or(false, |context| context == entry) {
                        return Some(0);
                    }
                }
                // A bound that has no name here (see `Context::of()`) is in no entry. An entry for
                // any type has the `Self` of a nested struct, enum or union as the implementor,
                // also where its type names it (`nested_self_entries`).
                let any = nested_type && matches!(entry.self_ty(), Some(SelfRequirement::Any));
                let context = if any {
                    Context::of_any(ty, scopes, self_ty, nested_type, enclosing)
                } else {
                    Context::of(ty, scopes, self_ty, nested_type, enclosing)
                }
                .ok()?;
                let after = usize::from(alone);
                if context == entry {
                    Some(after)
                } else if context.is_entry_with_extra_self_predicates(&entry) {
                    Some(after + 1)
                } else {
                    None
                }
            }
            Matching::Default => {
                if entry.self_ty().is_some() {
                    return None;
                }
                let root = param_names(scopes[0].params());
                let context = Context::of(ty, &scopes[..1], None, false, enclosing).ok()?;
                let used = param_names(&context.generics().params);
                let entry_params = param_names(&entry.generics().params);
                (used.is_subset(&entry_params) && entry_params.is_subset(&root))
                    .then(|| entry_params.len())
            }
        }
    }

    /// Returns `true` if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the type and the [`Context`] of each entry, by index.
    ///
    /// Write an impl for each ([`Referrer::repeater_impl()`]). Use `enumerate()` for the index
    /// that the other methods take. Entries that need the same share a context.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: Wrap<T>, b: Rel });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// let wrap: Type = parse_quote!(Wrap<T>);
    /// for (ix, (ty, context)) in referrer.iter().enumerate() {
    ///     // `Wrap<T>` uses the generic `T`, and `Rel` uses none.
    ///     let uses_t = !context.generics.params.is_empty();
    ///     assert_eq!(uses_t, *ty == wrap);
    ///     // The impl of this entry, and its path through the Repeater, take `ix`.
    ///     assert!(referrer.entry_args(ix, &UseSite::identity(), array_const).is_ok());
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&Type, &Context)> {
        self.entries
            .iter()
            .map(move |(ty, context)| (ty, &self.contexts[context]))
    }

    /// Returns `true` if some entry names `Self`, or its context's generics do.
    ///
    /// `Self::Item` and `<Self as ::p::Tr>::X` count, `::p::Self` does not. A top-level context
    /// with no entry is not looked at.
    pub fn has_self(&self) -> bool {
        (0..self.entries.len()).any(|ix| self.needs_self(ix))
    }

    /// Fills in the arguments that a copy omits because the trait gives them defaults.
    ///
    /// `context` is where the trait is written: the top-level [`ContextId`] that the Leaker
    /// returned for it, or for a trait of [`Leaker::intern_mod()`](crate::Leaker::intern_mod) the
    /// context of its module ([`ModuleScope::Itself`] tells which). `params` are the parameters
    /// of the trait; `args` match them by position, lifetimes first (a bare identifier for a const
    /// parameter is a const argument, and bindings such as `Item = u8` are skipped). Fails with
    /// [`EncodeError`] if there are too many arguments, or one without a default is missing.
    ///
    /// A default may name leaked types: `type_fn(leaked, ix, before)` replaces each one, where
    /// `before` are the complete arguments before the defaulted parameter. Return a path through
    /// the Repeater ([`Referrer::entry_args()`]); it is not substituted again, so write a type that
    /// you keep as written with [`UseSite::substitute()`](crate::UseSite::substitute), on a
    /// [`UseSite::substituting()`](crate::UseSite::substituting) of `params` and `before`. The
    /// entries of the module of `context` are looked up, and those of no module: a default that
    /// names an item of the module (`U = Rel2<T>` with `Rel2` imported there) is one. A const
    /// default is used as written, so one that names an item of the Leaker's scope (`= MY_CONST`)
    /// does not resolve in the Referrer: give such an argument explicitly. A default is written as
    /// [`UseSite::substitute()`](crate::UseSite::substitute) writes a type. The elided lifetime of
    /// a signature whose inputs `type_fn` gives lifetimes is named, `'__tl_s0` as at a use site.
    /// Panics if `context` is not a context of this Referrer.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait Tr<T, U = Rel<T>> { fn f(&self, t: T, u: U); });
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_trait(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
    /// let slots = RepeaterSlots::new(parse_quote!(::a::Repeater));
    ///
    /// // A copy writes `impl Tr<u8> for X`: `U` takes its default `Rel<T>`, with `T = u8`.
    /// let params = &item.generics.params;
    /// let given: Vec<GenericArgument> = vec![parse_quote!(u8)];
    /// let args = referrer
    ///     .complete_args(context, params, &given, |_leaked, ix, before| {
    ///         let site = UseSite::substituting(params, before);
    ///         let args = referrer.entry_args(ix, &site, array_const).unwrap();
    ///         slots.projection(&parse_quote!(::a::Marker), &args, 7, ix)
    ///     })
    ///     .unwrap();
    /// let key = repeater_key(7, 0);
    /// let expected: Vec<GenericArgument> = vec![
    ///     parse_quote!(u8),
    ///     parse_quote!(<::a::Marker as ::a::Repeater<(u8,), #key>>::Type),
    /// ];
    /// assert_eq!(args, expected);
    ///
    /// // `T` has no default, so it cannot be left out.
    /// let none = std::iter::empty();
    /// assert!(referrer.complete_args(context, params, none, |ty, _, _| ty).is_err());
    /// ```
    pub fn complete_args<'a>(
        &self,
        context: ContextId,
        params: &Punctuated<GenericParam, Token![,]>,
        args: impl IntoIterator<Item = &'a GenericArgument>,
        mut type_fn: impl FnMut(Type, usize, &[GenericArgument]) -> Type,
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        let module = ModulePos::of(&self.contexts[&context]);
        complete_generics_args(
            args,
            params,
            |index, _param, default, substitute, completed| {
                // The parameters before the defaulted one are the Leaker's generics for its default.
                let root = Generics {
                    params: params.iter().take(index).cloned().collect(),
                    ..Default::default()
                };
                let (ty, leaked) = self.expand_default(module, &root, default);
                let ty = substitute.fold_type(ty);
                Ok(Resolve::new(leaked, completed, &mut type_fn, params, &ty).fold_type(ty))
            },
        )
    }

    /// Returns the first argument of the Repeater for the `ix`-th entry, as written at `site`.
    ///
    /// It is [`RepeaterImpl::args`] with the arguments the site gives the generics.
    /// `const_fn(value, param)` encodes the const parameter `param` with the argument `value`: pass
    /// [`array_const`](crate::array_const), or for the slots of [`repeater_definition()`]
    /// [`RepeaterSlots::const_fn()`](crate::RepeaterSlots::const_fn).
    ///
    /// Fails with [`EncodeError::MissingArgument`] if `site` gives too few arguments, with
    /// [`EncodeError::UnboundedTraitObject`] if an argument is a trait object without the lifetime
    /// bound the entry needs, and with the error of `const_fn`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote! {
    ///     trait Tr<'a, T: 'a, const N: usize> { fn f(&self, x: &'a Wrap<T, N>) -> Rel; }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true);
    /// let referrer = leaker.result().finish().unwrap();
    /// let ix = referrer.iter().position(|(ty, _)| *ty == parse_quote!(&'a Wrap<T, N>)).unwrap();
    ///
    /// // In the scope of the trait, as in a default method: every generic by its own name.
    /// let args = referrer.entry_args(ix, &UseSite::identity(), array_const).unwrap();
    /// assert_eq!(args, parse_quote!((&'a (), T, [(); N], &'a T)));
    ///
    /// // In `impl Tr<'static, u8, 3> for Host`: the arguments of the trait.
    /// let trait_args: Vec<GenericArgument> =
    ///     vec![parse_quote!('static), parse_quote!(u8), parse_quote!(3)];
    /// let site = UseSite::substituting(&tr.generics.params, &trait_args);
    /// let args = referrer.entry_args(ix, &site, array_const).unwrap();
    /// assert_eq!(args, parse_quote!((&'static (), u8, [(); 3], &'static u8)));
    /// ```
    pub fn entry_args(
        &self,
        ix: usize,
        site: &UseSite,
        const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
    ) -> std::result::Result<Type, EncodeError> {
        let context = &self.contexts[&self.entries[ix].1];
        let args = site.generic_args(&context.generics)?;
        encode_generics_args_with_params(
            &args,
            &context.generics.params,
            Some((&site.self_ty, &context.self_outlives)),
            const_fn,
        )
    }

    /// Returns `true` if the `ix`-th entry needs what `Self` stands for.
    ///
    /// That is so if it names `Self`, or its generics do (`trait Tr<T: AsRef<Self>, U = Rel<T>>`),
    /// even if it is only in `Self: 'a` (`trait Tr<'a> where Self: 'a`).
    /// Then [`ImplSelf::Marker`] fails and [`Referrer::bucket_self()`] is `None`: the impl is for
    /// [`ImplSelf::Param`], and the use site writes what `Self` stands for (the host or the type).
    fn needs_self(&self, ix: usize) -> bool {
        let context = &self.contexts[&self.entries[ix].1];
        context.self_ty.is_some() || first_self_in_generics(&context.generics).is_some()
    }

    /// Returns the primitive type that the `ix`-th entry's impl is for under [`ImplSelf::Bucket`]
    /// with `id`.
    ///
    /// It is `None` if the entry names `Self`, or its context's generics do: write what `Self` stands
    /// for there.
    #[must_use]
    pub fn bucket_self(&self, ix: usize, id: u64) -> Option<Type> {
        (!self.needs_self(ix)).then(|| bucket_type(id, ix))
    }

    /// Describes the Repeater impl of the `ix`-th entry: its generics, trait arguments and type.
    ///
    /// `target` says what the impl is for ([`ImplSelf`]); `const_fn` is as in
    /// [`Referrer::entry_args()`], but gets the parameter itself (`N`) as `value`. Write the impl
    /// in [`RepeaterImpl::module`] if it is `Some`, and one for every entry ([`Referrer::iter()`]).
    ///
    /// Fails with [`EncodeError::NeedsSelf`] for [`ImplSelf::Marker`] if the entry names `Self`, or
    /// its context's generics do, and with the error of `const_fn`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote! {
    ///     trait Tr<T> { type Out; fn f(&self) -> Wrap<T, Self::Out>; }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true);
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// let param: Ident = parse_quote!(__S);
    /// let r = referrer.repeater_impl(0, ImplSelf::Param(&param), slots.const_fn()).unwrap();
    /// let (impl_generics, _, where_clause) = r.generics.split_for_impl();
    /// let (self_ty, ty) = (&r.self_ty, &r.ty);
    /// let path = slots.trait_path(&r.args, 7, 0);
    /// let key = repeater_key(7, 0);
    /// let imp: ItemImpl = parse_quote! {
    ///     impl #impl_generics #path for #self_ty #where_clause {
    ///         type Type = #ty;
    ///     }
    /// };
    /// // `Self::Out` names `Self`: the implementor is bound by the trait, and replaces it.
    /// let expected: ItemImpl = parse_quote! {
    ///     impl<T, __S: ?::core::marker::Sized + Tr<T>>
    ///         crate::Repeater<(T,), #key> for __S
    ///     {
    ///         type Type = Wrap<T, <__S>::Out>;
    ///     }
    /// };
    /// assert_eq!(imp, expected);
    /// ```
    pub fn repeater_impl(
        &self,
        ix: usize,
        target: ImplSelf<'_>,
        mut const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
    ) -> std::result::Result<RepeaterImpl, EncodeError> {
        let context = &self.contexts[&self.entries[ix].1];
        let bucket;
        // An entry that needs `Self` has none for a marker, even if `self_ty` is `None`: the
        // `Self` of its generics would be the marker.
        if matches!(target, ImplSelf::Marker(_)) && self.needs_self(ix) {
            return Err(EncodeError::NeedsSelf(Box::new(self.entries[ix].0.clone())));
        }
        let target = match target {
            ImplSelf::Bucket { id, param } => match self.bucket_self(ix, id) {
                Some(ty) => {
                    bucket = ty;
                    ImplSelf::Marker(&bucket)
                }
                None => ImplSelf::Param(param),
            },
            target => target,
        };
        // The type the impl is for, `Self` as the impl writes it, the parameter to declare and its
        // bound.
        let (self_ty, self_in_impl, param, bound): (Type, Type, Option<&Ident>, Option<&Path>) =
            match (context.self_ty.as_ref(), target) {
                (Some(SelfRequirement::Type(ty)), ImplSelf::Param(_)) => {
                    (ty.clone(), parse_quote!(Self), None, None)
                }
                (Some(_), ImplSelf::Marker(_)) => unreachable!("rejected above"),
                (Some(SelfRequirement::Bound(path)), ImplSelf::Param(ident)) => (
                    parse_quote!(#ident),
                    parse_quote!(#ident),
                    Some(ident),
                    Some(path),
                ),
                // Any type that the bounds hold for: the parameter has no bound of its own.
                (Some(SelfRequirement::Any), ImplSelf::Param(ident)) => (
                    parse_quote!(#ident),
                    parse_quote!(#ident),
                    Some(ident),
                    None,
                ),
                (None, ImplSelf::Param(ident)) => (
                    parse_quote!(#ident),
                    parse_quote!(#ident),
                    Some(ident),
                    None,
                ),
                (None, ImplSelf::Marker(marker)) => (marker.clone(), marker.clone(), None, None),
                (_, ImplSelf::Bucket { .. }) => unreachable!("resolved above"),
            };
        let generics = &context.generics;
        let (ty, generics) = match context.self_ty.as_ref() {
            // The implementor is a parameter, where `Self` of the trait is not available.
            Some(SelfRequirement::Bound(_) | SelfRequirement::Any) => {
                let mut folder = ReplaceSelf {
                    self_ty: &self_in_impl,
                };
                (
                    folder.fold_type(self.entries[ix].0.clone()),
                    folder.fold_generics(generics.clone()),
                )
            }
            _ => (self.entries[ix].0.clone(), generics.clone()),
        };
        let args = encode_generics_params_with(
            &generics.params,
            Some((&self_in_impl, &context.self_outlives)),
            |param| {
                let ident = &param.ident;
                const_fn(&parse_quote!(#ident), param)
            },
        )?;
        let mut declared = without_encoded_bounds(&generics);
        if let Some(ident) = param {
            // The `Self` of a struct, an enum or a union is a `Sized` type, and the bounds may need
            // it (`(Self, u8)`, `Vec<Self>`), so the parameter of `SelfRequirement::Any` is
            // `Sized` where they do. Else it is `?Sized`: rustc before 1.70 cannot prove `S<T>:
            // Sized` for the copy of a struct whose last field is the path through this impl, if
            // the impl needs `__S: Sized` (E0275: the type needs the impl, which needs the type).
            let sized = matches!(context.self_ty, Some(SelfRequirement::Any))
                && needs_sized(&declared, &ty, ident);
            let declaration = match (bound, sized) {
                (Some(path), _) => parse_quote!(#ident: ?::core::marker::Sized + #path),
                (None, true) => parse_quote!(#ident),
                (None, false) => parse_quote!(#ident: ?::core::marker::Sized),
            };
            // Before the first const parameter: rustc before 1.59 wants the types before the consts.
            let at = declared
                .params
                .iter()
                .position(|param| matches!(param, GenericParam::Const(_)))
                .unwrap_or(declared.params.len());
            declared.params.insert(at, declaration);
        }
        parenthesize_leading_qualified_path(&mut declared);
        Ok(RepeaterImpl {
            generics: declared,
            self_ty,
            args,
            ty,
            module: context
                .in_module()
                .and_then(|module| self.contexts[&module].module.clone()),
        })
    }

    /// Creates a [`Visitor`] that rewrites the leaked types in a copy of the analyzed item.
    ///
    /// `context` is the scope of the copy: the top-level [`ContextId`] that the `intern_*` or
    /// `enter_*` method of the Leaker returned for the item. Pass it along with the Referrer (it is
    /// written and parsed as an integer). `type_fn(leaked, ix)` returns the replacement, usually
    /// `<Self as Repeater<ARGS, KEY>>::Type` with `ARGS` from [`Referrer::entry_args()`]. It is
    /// called for every entry: return `leaked` to keep one as written. Only the leaked types are
    /// replaced: write the rest of the copy yourself (e.g. `u8`
    /// for `T` in a copy for `impl Tr<u8>`, which
    /// [`UseSite::substitute()`](crate::UseSite::substitute) does for a type). A type is replaced if
    /// its context, computed from the generics in scope, is that of an entry, so visit the whole
    /// item, not a type cut out of it:
    ///
    /// - A generic that the copy declares again (`T` in `impl<T: Clone> MyTrait<T> for ..`) must be
    ///   declared as the entry has it, bounds included, unless you call
    ///   [`Visitor::with_outermost_generics_as_leakers()`].
    /// - A struct, enum, union, trait or impl nested in the copy has a `Self` of its own, as in the
    ///   Leaker, so nest the same items. The entries that depend on it need
    ///   [`Visitor::with_nested_self()`].
    /// - In a module copy, an entry with a module matches only there
    ///   ([`Context::module`]), one with none everywhere. A `mod` item is a module
    ///   nested in that of `context`, found by name. For a module, visit its items: a `mod` that
    ///   no module of the context is named like is also taken as the module itself, but not one
    ///   that is.
    ///
    /// If `type_fn` writes lifetimes into the inputs of a function pointer or `Fn(..)` whose output
    /// takes its elided lifetime from the one lifetime of its inputs (the arguments of a site that
    /// holds `&'static str`), the visitor names that lifetime with a fresh binder (`for<'__tl_e0>
    /// fn(&'__tl_e0 ..) -> &'__tl_e0 u8`). A lifetime that a path hides is not named, nor is an
    /// elided lifetime that `type_fn` wrote itself.
    ///
    /// The Visitor borrows the Referrer, so `type_fn` may use it ([`Referrer::entry_args()`]), and
    /// several Visitors may share one Referrer.
    ///
    /// Panics if `context` is not a context of this Referrer.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; });
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_trait(&tr, true);
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// // The copy of `f` in `impl Tr<u8> for ..`.
    /// let trait_args: Vec<GenericArgument> = vec![parse_quote!(u8)];
    /// let site = UseSite::substituting(&tr.generics.params, &trait_args);
    /// let mut sig: Signature = parse_quote!(fn f(&self, t: T) -> Wrap<T>);
    /// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    /// referrer
    ///     .visitor(top_level, |_ty, ix| {
    ///         let args = referrer.entry_args(ix, &site, slots.const_fn()).unwrap();
    ///         slots.projection(&parse_quote!(Self), &args, 7, ix)
    ///     })
    ///     .visit_signature_mut(&mut sig);
    /// let key = repeater_key(7, 0);
    /// let expected: Signature = parse_quote!(
    ///     fn f(&self, t: T) -> <Self as crate::Repeater<(u8,), #key>>::Type
    /// );
    /// assert_eq!(sig, expected);
    /// ```
    pub fn visitor<'r, F: FnMut(Type, usize) -> Type>(
        &'r self,
        context: ContextId,
        type_fn: F,
    ) -> Visitor<'r, F> {
        let found = &self.contexts[&context];
        let in_module = found.is_module_scope();
        Visitor {
            referrer: self,
            hidden: Rc::default(),
            type_fn: Rc::new(RefCell::new(type_fn)),
            nested_fn: None,
            self_requirement: found.self_ty().cloned(),
            scopes: vec![scope_of(found)],
            outermost_as_leakers: false,
            self_ty: None,
            // The items of a module are nested items.
            item_depth: usize::from(in_module),
            own_self: false,
            nested: None,
            mod_path: Vec::new(),
            defaults: Defaults::Plain,
            module: ModulePos::of(found),
            module_scope: in_module,
            place: PartPlace::default(),
            signature: None,
        }
    }

    /// Replaces the leaked types in one type with what `type_fn(leaked, index)` returns.
    ///
    /// `ty` is inside the scope `context`, as for [`Referrer::visitor()`]. Pass the whole type, not
    /// one cut out of a larger type ([Outlives bounds](crate::Leaker#outlives-bounds)). Only the
    /// generics of `context` are known: a type that names a generic of a method is not found, and a
    /// type that the Leaker copied as written for the signature it is in is replaced. Use
    /// [`Referrer::visitor()`] on the signature then. Panics if `context` is not a context of this
    /// Referrer.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: ::core::option::Option<Wrap<T>> });
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_struct(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
    ///
    /// let ty: Type = parse_quote!(::core::option::Option<Wrap<T>>);
    /// let slots = RepeaterSlots::new(parse_quote!(Repeater));
    /// let ty = referrer.expand(context, ty, |_, ix| {
    ///     slots.projection(&parse_quote!(Marker), &parse_quote!((T,)), 7, ix)
    /// });
    /// let key = repeater_key(7, 0);
    /// let expected: Type =
    ///     parse_quote!(::core::option::Option<<Marker as Repeater<(T,), #key>>::Type>);
    /// assert_eq!(ty, expected);
    /// ```
    #[must_use]
    pub fn expand(
        &self,
        context: ContextId,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        // A type cut out of an item is not known to be in a predicate.
        let mut ty = ty;
        self.visitor(context, type_fn).visit_type_mut(&mut ty);
        ty
    }

    fn expand_with(
        &self,
        at: Lookup,
        scopes: &mut Vec<Scope>,
        self_ty: Option<&Type>,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        Folder {
            referrer: self,
            at,
            scopes,
            replace_self: self_ty.map(|self_ty| ReplaceSelf { self_ty }),
            enclosing: Enclosing::default(),
            active: Vec::new(),
            type_fn,
        }
        .fold_type(ty)
    }
}

/// The primitive type of the entry `ix` for `id`: 17 of them, chosen by a hash so that ids and
/// indices that are close spread. `ImplSelf::Bucket` of the Leaker's macro and the paths of the
/// Referrer's must agree on it, so it is part of [`Referrer::FORMAT_VERSION`].
fn bucket_type(id: u64, ix: usize) -> Type {
    const BUCKETS: [&str; 17] = [
        "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
        "i128", "isize", "f32", "f64", "str",
    ];
    // The finalizer of splitmix64.
    let mut z = id
        .wrapping_add((ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
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

/// Returns `true` if the impl may need `param: Sized`, and `false` if it is well-formed without.
///
/// `declared` are the generics of the impl without `param`, in which `param` replaced `Self`, and
/// `ty` is the type of the impl. A type that names `Self` (`Rel2<Self, T>`, with
/// [`nested_self_entries()`](crate::Options::nested_self_entries)) needs it as
/// the bounds do: `Rel2<__S, T>` is well-formed for `__S: Sized` only, unless the declaration of
/// `Rel2` relaxes it, which the Leaker cannot see. They need no `Sized` if every use of `param` is
///
/// - a type argument of its own, directly, of a path that accepts an unsized one
///   ([`accepts_unsized_argument()`]: `AsRef<__S>`, `Box<__S>`),
/// - behind a reference or a raw pointer (`&'a __S`, `*const __S`), or
/// - the type of a predicate that does not say `Sized` (`__S: 'a`).
///
/// Any other use may need it: a tuple element (`(__S, u8)`), the element of an array or a slice,
/// the argument of a type or a trait of the user (`Vec<__S>`, `Tr<__S>`), an associated type
/// (`Iterator<Item = __S>`, `<__S>::A`), the argument of an `Fn` bound, a macro. So this errs on
/// the side of `Sized`.
fn needs_sized(declared: &Generics, ty: &Type, param: &Ident) -> bool {
    use syn::visit::Visit;
    struct Uses<'a> {
        param: &'a Ident,
        needs_sized: bool,
    }
    impl Uses<'_> {
        fn is_param(&self, ty: &Type) -> bool {
            match ty {
                Type::Path(TypePath { qself: None, path }) => path.is_ident(self.param),
                Type::Paren(TypeParen { elem, .. }) => self.is_param(elem),
                Type::Group(TypeGroup { elem, .. }) => self.is_param(elem),
                _ => false,
            }
        }
    }
    impl<'ast> Visit<'ast> for Uses<'_> {
        // The uses that need `Sized` are those that are not skipped below.
        fn visit_ident(&mut self, ident: &'ast Ident) {
            if ident == self.param {
                self.needs_sized = true;
            }
        }
        // Tokens that are not parsed: any `Self` is out of reach.
        fn visit_macro(&mut self, _: &'ast Macro) {
            self.needs_sized = true;
        }
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
                    }) if path.segments.last().map_or(false, |s| s.ident == "Sized") => {
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
/// is `?Sized`: `AsRef<__S>` is well-formed with `__S: ?Sized`.
///
/// It is written with an absolute path (`::core::convert::AsRef`), or with the name of the prelude
/// (`AsRef`, `Box`), which is taken for what it names there: the Leaker accepts it only if the
/// prelude is allowed. A module that declares an item of the same name is not told apart (the impl
/// would then not compile in the Leaker's crate).
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
    const PRELUDE: &[&str] = &["AsRef", "AsMut", "PartialEq", "PartialOrd", "Box"];
    let names: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    let names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    match (path.leading_colon.is_some(), &names[..]) {
        (false, [name]) => PRELUDE.contains(name),
        (true, [root, module, item]) => ABSOLUTE
            .iter()
            .any(|(roots, m, items)| roots.contains(root) && m == module && items.contains(item)),
        _ => false,
    }
}

/// rustc reads `where <__S>::A: Clone` as generics of the `where` (reserved for future use), so a
/// first predicate on `<__S>::A` is parenthesized.
fn parenthesize_leading_qualified_path(generics: &mut Generics) {
    let first = generics
        .where_clause
        .as_mut()
        .and_then(|clause| clause.predicates.first_mut());
    if let Some(WherePredicate::Type(PredicateType {
        lifetimes: None,
        bounded_ty,
        ..
    })) = first
    {
        if matches!(bounded_ty, Type::Path(TypePath { qself: Some(q), .. }) if q.as_token.is_none())
        {
            *bounded_ty = Type::Paren(TypeParen {
                paren_token: Default::default(),
                elem: Box::new(bounded_ty.clone()),
            });
        }
    }
}

/// Folds a type, replacing the leaked types in it (see `Referrer::expand_with()`).
struct Folder<'a, 'b, F> {
    referrer: &'a Referrer,
    at: Lookup<'b>,
    scopes: &'a mut Vec<Scope>,
    replace_self: Option<ReplaceSelf<'a>>,
    enclosing: Enclosing,
    /// The parts whose anchor is a type around the one being folded.
    active: Vec<usize>,
    type_fn: F,
}

impl<'a, 'b, F: FnMut(Type, usize) -> Type> Folder<'a, 'b, F> {
    /// The entry that stands for `ty`, unless `ty` is a part, which is left as written, or
    /// is inside a whole part: nothing in it is looked up.
    fn lookup(&mut self, ty: &Type, enclosing: &Enclosing) -> Option<usize> {
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
        self.referrer.find(self.at, ty, self.scopes, enclosing)
    }
}

impl<'a, 'b, F: FnMut(Type, usize) -> Type> Fold for Folder<'a, 'b, F> {
    fn fold_type(&mut self, ty: Type) -> Type {
        let enclosing = self.enclosing.clone();
        let outer = self.at.place;
        let active = self.active.len();
        let found = self.lookup(&ty, &enclosing);
        if found.is_none() {
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
        let folded = match (found, self.replace_self) {
            (Some(ix), None) => (self.type_fn)(ty, ix),
            (Some(ix), Some(mut replace)) => (self.type_fn)(replace.fold_type(ty), ix),
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
        let folded = match self.replace_self {
            Some(replace) => replace.fold_expr_with(self, expr),
            None => syn::fold::fold_expr(self, expr),
        };
        self.enclosing = enclosing;
        folded
    }
    fn fold_macro(&mut self, i: Macro) -> Macro {
        match self.replace_self {
            Some(mut replace) => replace.fold_macro(i),
            None => syn::fold::fold_macro(self, i),
        }
    }
    fn fold_type_bare_fn(&mut self, mut i: TypeBareFn) -> TypeBareFn {
        // Before the binder is in scope: the one it gets names the lifetime that elision
        // gives, which `Self` written as a type with lifetimes would break.
        if let Some(replace) = self.replace_self {
            let mut writer = SelfWriter {
                self_ty: replace.self_ty,
                scopes: &self.scopes[..],
            };
            name_elided_in_bare_fn(&mut writer, &mut i);
        }
        // What the original relies on, found before an entry is written: the path that
        // replaces it may name lifetimes in the inputs (the arguments of the site).
        let elision = Elision::of_bare_fn(&i);
        let depth = self.scopes.len();
        self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
        let mut i = syn::fold::fold_type_bare_fn(self, i);
        self.scopes.truncate(depth);
        if let Some(elision) = elision {
            let mut names = WrittenNames::new(&self.scopes[..], &i);
            elision.name_written_bare_fn(&mut names, &mut i);
        }
        i
    }
    fn fold_trait_bound(&mut self, mut i: TraitBound) -> TraitBound {
        if let Some(replace) = self.replace_self {
            let mut writer = SelfWriter {
                self_ty: replace.self_ty,
                scopes: &self.scopes[..],
            };
            name_elided_in_bound(&mut writer, &mut i);
        }
        let elision = Elision::of_bound(&i);
        let depth = self.scopes.len();
        self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
        let mut i = syn::fold::fold_trait_bound(self, i);
        self.scopes.truncate(depth);
        if let Some(elision) = elision {
            let mut names = WrittenNames::new(&self.scopes[..], &i);
            elision.name_written_bound(&mut names, &mut i);
        }
        i
    }
}

/// The path `path` read as a type, which stands for a node that is no type (a bound).
fn path_type(path: &Path) -> Type {
    Type::Path(TypePath {
        qself: None,
        path: path.clone(),
    })
}

fn param_names<'a>(params: impl IntoIterator<Item = &'a GenericParam>) -> HashSet<String> {
    params.into_iter().map(generic_param_name).collect()
}

// Prefix of the placeholder names in `Referrer::complete_args()`: `Referrer::expand_default()` writes
// them, and `Resolve` reads them.
const PLACEHOLDER: &str = "__TypeLeakPlaceholder";

impl Referrer {
    /// Replaces each leaked type in `default`, the default of a parameter whose predecessors are
    /// `root`, by a placeholder, which keeps the substitution from touching the replaced paths.
    ///
    /// Returns the type, and the leaked types with their indices, by the number of the placeholder.
    fn expand_default(
        &self,
        module: ModulePos,
        root: &Generics,
        default: &Type,
    ) -> (Type, Vec<(Type, usize)>) {
        let mut leaked = Vec::new();
        let at = Lookup {
            matching: Matching::Default,
            module,
            hidden: None,
            place: PartPlace::default(),
            signature: None,
        };
        let ty = self.expand_with(
            at,
            &mut vec![Scope::new(root, None)],
            None,
            default.clone(),
            |ty, ix| {
                leaked.push((ty, ix));
                let placeholder = Ident::new(
                    &format!("{}{}", PLACEHOLDER, leaked.len() - 1),
                    Span::call_site(),
                );
                parse_quote!(#placeholder)
            },
        );
        (ty, leaked)
    }
}

/// Puts back what `type_fn` writes for the leaked types of a default, in place of the placeholders.
struct Resolve<'a, F> {
    leaked: Vec<(Type, usize)>,
    completed: &'a [GenericArgument],
    type_fn: &'a mut F,
    /// The lifetimes that a signature binds where `type_fn` wrote lifetimes into its inputs.
    names: Names,
}

impl<'a, F> Resolve<'a, F> {
    /// `ty` is the default with the placeholders, `params` the parameters of the trait, and
    /// `completed` the arguments before the defaulted parameter.
    fn new(
        leaked: Vec<(Type, usize)>,
        completed: &'a [GenericArgument],
        type_fn: &'a mut F,
        params: &Punctuated<GenericParam, Token![,]>,
        ty: &Type,
    ) -> Self {
        // The site's generics are not known: a lifetime that the type, the arguments or the
        // parameters name is not a fresh name.
        let mut names = Names::default();
        names.reserve(ty);
        names.reserve(params);
        for arg in completed {
            names.reserve(arg);
        }
        Self {
            leaked,
            completed,
            type_fn,
            names,
        }
    }
}

impl<'a, F: FnMut(Type, usize, &[GenericArgument]) -> Type> Fold for Resolve<'a, F> {
    fn fold_type(&mut self, ty: Type) -> Type {
        let leaked = match &ty {
            Type::Path(TypePath { qself: None, path }) => path
                .get_ident()
                .and_then(|ident| {
                    ident
                        .to_string()
                        .strip_prefix(PLACEHOLDER)?
                        .parse::<usize>()
                        .ok()
                })
                .and_then(|number| self.leaked.get(number).cloned()),
            _ => None,
        };
        match leaked {
            Some((leaked, ix)) => (self.type_fn)(leaked, ix, self.completed),
            None => syn::fold::fold_type(self, ty),
        }
    }

    // The placeholder is no lifetime, so the original's elision is read before `type_fn` writes
    // the path that replaces it.
    fn fold_type_bare_fn(&mut self, i: TypeBareFn) -> TypeBareFn {
        let elision = Elision::of_bare_fn(&i);
        let mut i = syn::fold::fold_type_bare_fn(self, i);
        if let Some(elision) = elision {
            // The signatures inside have been named: those names are taken too.
            self.names.reserve(&i);
            elision.name_written_bare_fn(&mut self.names, &mut i);
        }
        i
    }

    fn fold_trait_bound(&mut self, i: TraitBound) -> TraitBound {
        let elision = Elision::of_bound(&i);
        let mut i = syn::fold::fold_trait_bound(self, i);
        if let Some(elision) = elision {
            self.names.reserve(&i);
            elision.name_written_bound(&mut self.names, &mut i);
        }
        i
    }
}

/// A struct, enum, union, trait or impl nested in the visited syntax, which has a `Self` of its
/// own.
///
/// [`Visitor::with_nested_self()`] gives it to the closure that writes the replacement of an entry
/// that depends on that `Self`.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use syn::visit_mut::VisitMut;
/// let item: ItemMod = parse_quote! {
///     mod m { mod inner { struct S<T> { a: Rel<Self>, b: T } } }
/// };
/// let mut leaker = Leaker::new();
/// let module = leaker.intern_mod(&item);
/// let referrer = leaker.result().finish().unwrap();
///
/// // A copy of the items of `m`. The entry `Rel<Self>` names the `Self` of `S`.
/// let mut file = File { shebang: None, attrs: vec![], items: item.content.unwrap().1 };
/// let mut seen = Vec::new();
/// referrer
///     .visitor(module, |_leaked, ix| parse_quote!(Entry<#ix>))
///     .with_nested_self(|nested, _leaked, ix| {
///         seen.push(nested.clone());
///         parse_quote!(NestedEntry<#ix>)
///     })
///     .visit_file_mut(&mut file);
///
/// assert_eq!(seen.len(), 1);
/// // The item has the name `S`, and is in `inner`, which is in the module of the context.
/// assert_eq!(seen[0].ident.as_ref().unwrap(), "S");
/// assert_eq!(seen[0].modules.len(), 1);
/// assert_eq!(seen[0].modules[0], "inner");
/// // And `Self` is `S<T>`, as the Leaker has it.
/// match &seen[0].requirement {
///     SelfRequirement::Type(ty) => assert_eq!(*ty, parse_quote!(S<T>)),
///     _ => unreachable!(),
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct NestedSelf {
    /// What `Self` is in the item.
    ///
    /// ```ignore
    /// struct S<T> { .. }  // SelfRequirement::Type(S<T>), also for an enum or a union
    /// trait Tr<T> { .. } // SelfRequirement::Bound(Tr<T>)
    /// impl<T> W<T> { .. } // SelfRequirement::Type(W<T>), the self type as written
    /// ```
    pub requirement: SelfRequirement,
    /// The name of the struct, enum, union or trait; `None` for an impl.
    pub ident: Option<Ident>,
    /// The `mod` items between the module of the context and the item, outermost first.
    ///
    /// ```ignore
    /// mod m { mod a { mod b { struct S; } } }
    /// // `S`, in the items of the context of `m`: ident = Some(S), modules = [a, b]
    /// ```
    pub modules: Vec<Ident>,
}

impl NestedSelf {
    /// Returns `true` for a struct, an enum or a union, and `false` for a trait, which has no type
    /// of its own, and for an impl, which has the self type it is written for. As in the Leaker
    /// (`AnalyzeVisitor::self_nested`).
    fn is_adt(&self) -> bool {
        self.ident.is_some() && matches!(self.requirement, SelfRequirement::Type(_))
    }
}

/// What a nested item declares, to give it its own `Self`.
struct Declares {
    requirement: SelfRequirement,
    ident: Option<Ident>,
}

impl Declares {
    fn of_item(ident: &Ident, generics: &Generics) -> Self {
        Self {
            requirement: SelfRequirement::of_item(ident, generics),
            ident: Some(ident.clone()),
        }
    }

    fn of_trait(ident: &Ident, generics: &Generics) -> Self {
        Self {
            requirement: SelfRequirement::of_trait(ident, generics),
            ident: Some(ident.clone()),
        }
    }

    fn of_impl(self_ty: &Type) -> Self {
        Self {
            requirement: SelfRequirement::Type(self_ty.clone()),
            ident: None,
        }
    }
}

/// Returns the names that `items` declare together: types, traits, constants, modules and imports.
fn declared_names<'a>(items: impl IntoIterator<Item = &'a Item>) -> Vec<Ident> {
    fn use_names(tree: &UseTree, names: &mut Vec<Ident>) {
        match tree {
            UseTree::Path(UsePath { tree, .. }) => use_names(tree, names),
            UseTree::Name(UseName { ident }) if ident != "self" => names.push(ident.clone()),
            UseTree::Rename(UseRename { rename, .. }) => names.push(rename.clone()),
            UseTree::Group(UseGroup { items, .. }) => {
                for tree in items {
                    use_names(tree, names)
                }
            }
            _ => (),
        }
    }
    let mut names = Vec::new();
    for item in items {
        match item {
            Item::Struct(ItemStruct { ident, .. })
            | Item::Enum(ItemEnum { ident, .. })
            | Item::Union(ItemUnion { ident, .. })
            | Item::Trait(ItemTrait { ident, .. })
            | Item::TraitAlias(ItemTraitAlias { ident, .. })
            | Item::Type(ItemType { ident, .. })
            | Item::Const(ItemConst { ident, .. })
            | Item::Static(ItemStatic { ident, .. })
            | Item::Mod(ItemMod { ident, .. }) => names.push(ident.clone()),
            Item::Use(ItemUse { tree, .. }) => use_names(tree, &mut names),
            Item::ExternCrate(ItemExternCrate { ident, rename, .. }) => {
                names.push(rename.as_ref().map_or(ident, |(_, rename)| rename).clone())
            }
            _ => (),
        }
    }
    names
}

/// Replaces the leaked types in a copy of the analyzed item with what you write for them (usually a
/// path through the Repeater). Get it with [`Referrer::visitor()`].
///
/// It implements `syn::visit_mut::VisitMut`. Visit a whole type or item, not a type cut out of
/// another ([Outlives bounds](crate::Leaker#outlives-bounds)). A type is replaced only if an entry
/// has the context that it has there, from the generics in scope. The parts that the Leaker copied
/// as written ([recovery](crate::Leaker#errors-and-recovery)) are left as they are. Items and
/// imports declared in a block hide the leaked types that name them there, and those declared in a
/// module or a `File` hide the entries of no module that name them. Values are visited too: a type
/// there is an entry only with [`Options::check_values()`](crate::Options::check_values).
///
/// `G` is the closure of [`Visitor::with_nested_self()`]; none by default.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use syn::visit_mut::VisitMut;
/// let item: ItemTrait = parse_quote!(trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; });
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_trait(&item, true);
/// let referrer = leaker.result().finish().unwrap();
///
/// // Rewrite the whole copy of the trait, with the `ContextId` that the Leaker returned for it.
/// let mut copy = item.clone();
/// let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
/// referrer
///     .visitor(top_level, |_leaked, ix| {
///         let args = referrer.entry_args(ix, &UseSite::identity(), slots.const_fn()).unwrap();
///         slots.projection(&parse_quote!(Self), &args, 7, ix)
///     })
///     .visit_item_trait_mut(&mut copy);
/// let key = repeater_key(7, 0);
/// let expected: ItemTrait = parse_quote! {
///     trait Tr<T> { fn f(&self, t: T) -> <Self as crate::Repeater<(T,), #key>>::Type; }
/// };
/// assert_eq!(copy, expected);
/// ```
#[derive(Debug)]
#[must_use]
pub struct Visitor<'r, F, G = fn(&NestedSelf, Type, usize) -> Type> {
    referrer: &'r Referrer,
    /// The entries that the items declared in the visited syntax hide (`Visitor::shadow_items()`).
    hidden: Rc<HashSet<usize>>,
    type_fn: Rc<RefCell<F>>,
    /// Writes the replacement of an entry that depends on the own `Self` of a nested item.
    nested_fn: Option<Rc<RefCell<G>>>,
    /// What `Self` means here: in the context, or in the nested item that declares it. Each scope
    /// in `scopes` keeps the one in force where it was declared.
    self_requirement: Option<SelfRequirement>,
    /// The generics in scope, innermost last. The first are those of the context.
    scopes: Vec<Scope>,
    /// See [`Visitor::with_outermost_generics_as_leakers()`].
    outermost_as_leakers: bool,
    /// What replaces `Self` in the copy ([`Visitor::with_self()`]).
    self_ty: Option<Type>,
    /// How many items the visit is in. The outermost item visited is the scope itself (`0`), and
    /// the items of a module context are nested ones (`1`).
    item_depth: usize,
    /// In a nested struct, enum, union, trait or impl, which declare their own `Self`. A function,
    /// type alias, trait alias, const or static keeps the `Self` around, as in the Leaker.
    own_self: bool,
    /// The nested item that `own_self` is about, for the closure of `nested_fn`.
    nested: Option<Rc<NestedSelf>>,
    /// The `mod` items entered since the module of the context.
    mod_path: Vec<Ident>,
    /// While the generics of a scope that this visit declares are visited: how the defaults of
    /// their parameters are looked up (see `Visitor::visit_type_param_mut()`).
    defaults: Defaults,
    module: ModulePos,
    /// The context is a module: the visited items are its own until the visitor enters one.
    module_scope: bool,
    /// Where the visited syntax is among the predicates of the item.
    place: PartPlace,
    /// The signature being visited (not its body), for the parts that a signature made.
    signature: Option<Rc<SignatureFacts>>,
}

impl<'r, F, G> Clone for Visitor<'r, F, G> {
    fn clone(&self) -> Self {
        Self {
            referrer: self.referrer,
            hidden: self.hidden.clone(),
            type_fn: self.type_fn.clone(),
            nested_fn: self.nested_fn.clone(),
            self_requirement: self.self_requirement.clone(),
            scopes: self.scopes.clone(),
            outermost_as_leakers: self.outermost_as_leakers,
            self_ty: self.self_ty.clone(),
            item_depth: self.item_depth,
            own_self: self.own_self,
            nested: self.nested.clone(),
            mod_path: self.mod_path.clone(),
            defaults: self.defaults,
            module: self.module,
            module_scope: self.module_scope,
            place: self.place,
            signature: self.signature.clone(),
        }
    }
}

impl<'r, F, G> Visitor<'r, F, G>
where
    F: FnMut(Type, usize) -> Type,
    G: FnMut(&NestedSelf, Type, usize) -> Type,
{
    /// Treats the generics of the outermost item visited as those of the context.
    ///
    /// Use it if the copy declares the generics of the scope again, with other bounds
    /// (`impl<T: Clone> MyTrait<T> for ..`): a type that names them is replaced if it was leaked,
    /// whatever bounds they have. Deeper generics stay declared, so they must match the entry's.
    pub fn with_outermost_generics_as_leakers(mut self) -> Self {
        self.outermost_as_leakers = true;
        self
    }

    /// Replaces `Self` by `self_ty` in the copy.
    ///
    /// Use it for a Repeater implemented for a marker type, with `allow_self = false`
    /// ([`Leaker::intern_struct()`](crate::Leaker::intern_struct)). Leaked types are still found by
    /// their original tokens, and `type_fn` gets them with `Self` replaced. `Self` is replaced in
    /// the other types too, parts copied as written included, but not in what `type_fn` returns, in
    /// receivers (`&self`) or in expressions, nor in a nested struct, enum, union, trait or impl
    /// (which has a `Self` of its own: see [`Visitor::with_nested_self()`]).
    ///
    /// The default of a parameter of the outermost item (`trait Tr<T: AsRef<Self>, U = Rel<T>>`) is
    /// looked up without `Self`, so a leaked type in it goes to `type_fn`; if a bound of an earlier
    /// parameter names `Self`, `type_fn` picks the host that has the bound in the copy. So does the
    /// default of a trait nested in the visited syntax, unless it names `Self`
    /// ([`with_nested_self()`](Visitor::with_nested_self)).
    ///
    /// A `self_ty` with lifetimes (`::p::S<'a>`) would give `fn(&Self) -> &u8` more input lifetimes
    /// (E0106), so the visitor names the elided one with a fresh binder
    /// (`for<'__tl_e0> fn(&'__tl_e0 ::p::S<'a>) -> &'__tl_e0 u8`). Where it cannot, the type is
    /// leaked whole, an error with `allow_self = false`
    /// ([Elided lifetimes](crate::Leaker#elided-lifetimes)).
    pub fn with_self(mut self, self_ty: Type) -> Self {
        self.self_ty = Some(self_ty);
        self
    }

    /// Writes the entries that depend on the own `Self` of a nested item with `nested_fn`.
    ///
    /// A struct, enum, union, trait or impl nested in the visited syntax (in a module, a `File` or
    /// a function body) has a `Self` of its own. For the entries that name it, or use a generic
    /// whose bound names it, the visitor calls `nested_fn(nested, leaked, ix)` instead of
    /// `type_fn`: `nested` is the item ([`NestedSelf`]), `leaked` the type of the entry, with its
    /// `Self`. The usual result is `<self_ty as Repeater<args, KEY>>::Type`
    /// ([`RepeaterSlots::projection()`](crate::RepeaterSlots::projection)), with `args` from
    /// [`Referrer::entry_args()`] at a [`UseSite`](crate::UseSite) that has
    /// [`with_self(self_ty)`](crate::UseSite::with_self). The `self_ty` to pass depends on the
    /// [`SelfRequirement`] of the entry's context (`Context::self_ty`).
    ///
    /// A trait whose default names `Self` (`trait Tr<T, U = Rel<Self>>`) is one case: the arguments
    /// of the Repeater include the defaulted parameter, which a declaration cannot name in its own
    /// default (E0128), so pass a type for it with
    /// [`UseSite::substituting()`](crate::UseSite::substituting). A default that does not name
    /// `Self` (`U = Rel<T>`) goes to `type_fn`.
    ///
    /// Without this method the copy keeps the types of these entries as written, which its crate
    /// cannot resolve. For a scope that is not a module (`enter_generics()`, ..), the nested items
    /// are in a function body of the copy.
    pub fn with_nested_self<N: FnMut(&NestedSelf, Type, usize) -> Type>(
        self,
        nested_fn: N,
    ) -> Visitor<'r, F, N> {
        Visitor {
            referrer: self.referrer,
            hidden: self.hidden,
            type_fn: self.type_fn,
            nested_fn: Some(Rc::new(RefCell::new(nested_fn))),
            self_requirement: self.self_requirement,
            scopes: self.scopes,
            outermost_as_leakers: self.outermost_as_leakers,
            self_ty: self.self_ty,
            item_depth: self.item_depth,
            own_self: self.own_self,
            nested: self.nested,
            mod_path: self.mod_path,
            defaults: self.defaults,
            module: self.module,
            module_scope: self.module_scope,
            place: self.place,
            signature: self.signature,
        }
    }

    /// Hides the entries that the `items` declared together hide (`Referrer::hidden_by()`).
    fn shadow_items<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>, module: bool) {
        let names = declared_names(items);
        if names.is_empty() {
            return;
        }
        let hidden = self.referrer.hidden_by(&names, module);
        if !hidden.is_empty() {
            Rc::make_mut(&mut self.hidden).extend(hidden);
        }
    }

    /// Returns `true` if `cause`, a bound whose trait path conflicts, is a part: it is no type.
    /// `path` stands for it as a type, to give it its context.
    fn is_part_bound(&self, cause: impl FnOnce() -> Cause, path: &Path) -> bool {
        if self.referrer.parts_by_cause.is_empty() || self.place.protected {
            return false;
        }
        self.referrer
            .part_of(
                &cause(),
                &path_type(path),
                self.lookup(),
                &self.scopes,
                &Enclosing::default(),
                &[],
            )
            .is_some()
    }

    /// Returns how a type is looked up here.
    fn lookup(&self) -> Lookup<'_> {
        // In a nested item, the entries that name its own `Self` are found, and written by
        // `nested_fn`; with none, `Self` hides the one around and nothing that names it is found.
        let nested = self.own_self && self.nested.is_some() && self.nested_fn.is_some();
        Lookup {
            matching: Matching::Scope {
                self_ty: self.self_requirement.as_ref(),
                no_self: self.own_self && !nested,
                nested_type: self.nested_type(),
                alone: self.defaults == Defaults::OwnTraitDefault,
            },
            module: self.module,
            hidden: Some(&self.hidden),
            place: self.place,
            signature: self.signature.as_deref(),
        }
    }

    /// Returns `true` in a nested struct, enum or union (see [`NestedSelf::is_adt()`]).
    fn nested_type(&self) -> bool {
        self.own_self && self.nested.as_ref().map_or(false, |nested| nested.is_adt())
    }

    /// Returns what replaces `Self` here ([`Visitor::with_self()`]), unless a nested item declares
    /// its own `Self`.
    fn replaced_self(&self) -> Option<&Type> {
        self.self_ty.as_ref().filter(|_| !self.own_self)
    }

    /// Runs `visit`, which visits a part that is left as written: no entry is looked up in it.
    fn protected(&mut self, visit: impl FnOnce(&mut Self)) {
        let outer = std::mem::replace(&mut self.place.protected, true);
        visit(self);
        self.place.protected = outer;
    }

    fn visit_attrs(&mut self, attrs: &mut [Attribute]) {
        for attr in attrs {
            self.visit_attribute_mut(attr);
        }
    }

    /// `generics` are those of the context, in any order and with the bounds anywhere.
    fn is_context_generics(&self, generics: &Generics) -> bool {
        let canonical = Context::new(generics.clone(), None);
        let canonical = Scope::new(canonical.generics(), None);
        matches!(self.scopes.first(), Some(first) if first.same_generics(&canonical))
    }

    /// Visits the attributes, then returns a nested visitor that has visited `generics` and has
    /// them in scope. `item` is for an item, not an associated item. `declares` is what `Self` is
    /// in an item that declares it (a struct, an enum, a union, a trait or an impl), as the Leaker
    /// has it. The other items keep the `Self` around.
    fn enter(
        &mut self,
        attrs: &mut [Attribute],
        generics: &mut Generics,
        declares: Option<Declares>,
        item: bool,
    ) -> Self {
        self.visit_attrs(attrs);
        let mut visitor = self.clone();
        visitor.item_depth += 1;
        // Only an item nested in the visited syntax that declares `Self` has one of its own: the
        // Leaker keeps the `Self` around for a function, a type alias or a trait alias. The layer
        // of the item is the Leaker's: its `Self` is allowed whatever `allow_self` the scope
        // around has (`AnalyzeVisitor::with_item_of()`), so entries that name it exist.
        let mut own_trait = false;
        if self.item_depth > 0 {
            if let Some(declared) = declares {
                own_trait = matches!(declared.requirement, SelfRequirement::Bound(_));
                visitor.own_self = true;
                visitor.nested = Some(Rc::new(NestedSelf {
                    requirement: declared.requirement.clone(),
                    ident: declared.ident,
                    modules: self.mod_path.clone(),
                }));
                visitor.self_requirement = Some(declared.requirement);
            }
        }
        // Declare before visiting: every parameter is in scope of each other's bounds. The
        // outermost item is the scope itself if it declares the generics of the context as they
        // are, or if `with_outermost_generics_as_leakers()` says so: they are not declared again.
        let outermost = item && self.item_depth == 0;
        let declared =
            !(outermost && (self.outermost_as_leakers || self.is_context_generics(generics)));
        if declared {
            let declared_for = visitor.self_requirement.clone();
            visitor.scopes.push(Scope::new(generics, declared_for));
        }
        // How the defaults of the generics are looked up: see `Defaults`.
        visitor.defaults = if own_trait {
            Defaults::OwnTrait
        } else {
            Defaults::WithoutSelf
        };
        visitor.visit_generics_mut(generics);
        visitor.defaults = Defaults::Plain;
        visitor
    }

    /// Visits the signature and returns the visitor for its body.
    fn enter_signature(
        &mut self,
        attrs: &mut [Attribute],
        sig: &mut Signature,
        item: bool,
    ) -> Self {
        let mut visitor = self.enter(attrs, &mut sig.generics, None, item);
        let facts = SignatureFacts::of(sig);
        for input in sig.inputs.iter_mut() {
            visitor.signature = Some(Rc::new(facts.for_input(input)));
            visitor.visit_fn_arg_mut(input);
        }
        visitor.signature = Some(Rc::new(facts));
        visitor.visit_return_type_mut(&mut sig.output);
        visitor.signature = None;
        visitor
    }
}

impl<'r, F, G> VisitMut for Visitor<'r, F, G>
where
    F: FnMut(Type, usize) -> Type,
    G: FnMut(&NestedSelf, Type, usize) -> Type,
{
    fn visit_type_mut(&mut self, i: &mut Type) {
        let nested = match (&self.nested, &self.nested_fn) {
            (Some(nested), Some(nested_fn)) if self.own_self => Some((nested, nested_fn)),
            _ => None,
        };
        let referrer = self.referrer;
        let mut type_fn = self.type_fn.borrow_mut();
        let replace = |ty: Type, ix: usize| -> Type {
            let entry = &referrer.contexts[&referrer.entries[ix].1];
            match nested {
                Some((nested, nested_fn)) if entry.self_ty().is_some() => {
                    let mut nested_fn = nested_fn.borrow_mut();
                    (*nested_fn)(nested, ty, ix)
                }
                _ => (*type_fn)(ty, ix),
            }
        };
        // The lookup reads the rest of the visitor, so the scopes are lent from outside it.
        let mut scopes = std::mem::take(&mut self.scopes);
        *i = referrer.expand_with(
            self.lookup(),
            &mut scopes,
            self.replaced_self(),
            i.clone(),
            replace,
        );
        self.scopes = scopes;
    }
    fn visit_fn_arg_mut(&mut self, i: &mut FnArg) {
        // An `impl Trait` in an argument is a generic of the function, whose bounds are predicates
        // of the item (the Leaker's `in_fn_arg`).
        let outer = std::mem::replace(&mut self.place.in_fn_arg, true);
        syn::visit_mut::visit_fn_arg_mut(self, i);
        self.place.in_fn_arg = outer;
    }
    fn visit_type_param_bound_mut(&mut self, i: &mut TypeParamBound) {
        // A bound that is not in a type is a predicate of the item (the Leaker's `env_depth`). A
        // type is visited whole by `visit_type_mut()`, so a bound visited here is not in one.
        let outer = std::mem::replace(&mut self.place.in_predicate, true);
        syn::visit_mut::visit_type_param_bound_mut(self, i);
        self.place.in_predicate = outer;
    }
    fn visit_receiver_mut(&mut self, i: &mut Receiver) {
        if self.replaced_self().is_none() {
            return syn::visit_mut::visit_receiver_mut(self, i);
        }
        // The receiver type stays: rustc does not accept an associated type path there.
        self.visit_attrs(&mut i.attrs);
    }
    fn visit_predicate_type_mut(&mut self, i: &mut PredicateType) {
        let depth = self.scopes.len();
        self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
        let outer = std::mem::replace(&mut self.place.in_predicate, true);
        syn::visit_mut::visit_predicate_type_mut(self, i);
        self.place.in_predicate = outer;
        self.scopes.truncate(depth);
        // What replaced the bounded type (`Self`, an entry) may need parentheses there.
        parenthesize_bounded(&mut i.bounded_ty);
    }
    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        // A bound of the generics is not in a type, which the same rewriting handles. A bound whose
        // trait path the Leaker could not leak is copied as written, with the types in it.
        let part = self.is_part_bound(|| bound_cause(i), &i.path);
        if let Some(self_ty) = self.replaced_self() {
            let mut writer = SelfWriter {
                self_ty,
                scopes: &self.scopes[..],
            };
            name_elided_in_bound(&mut writer, i);
        }
        let elision = Elision::of_bound(i);
        let depth = self.scopes.len();
        self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
        if part {
            self.protected(|visitor| syn::visit_mut::visit_trait_bound_mut(visitor, i));
        } else {
            syn::visit_mut::visit_trait_bound_mut(self, i);
        }
        self.scopes.truncate(depth);
        if let Some(elision) = elision {
            let mut names = WrittenNames::new(&self.scopes[..], &*i);
            elision.name_written_bound(&mut names, i);
        }
    }
    fn visit_type_param_mut(&mut self, i: &mut TypeParam) {
        let defaults = self.defaults;
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
            // An entry of either kind is found, the one without `Self` first (it is what
            // `complete_args()` uses): see `Defaults::OwnTrait`.
            self.defaults = Defaults::OwnTraitDefault;
            self.visit_type_mut(default);
            self.defaults = Defaults::OwnTrait;
        } else {
            // The default is looked up without `Self`: see `Defaults::WithoutSelf`.
            let nested_type = self.nested_type();
            let self_requirement = self.self_requirement.take();
            let own = self.scopes.last().cloned();
            if let Some(scope) = self.scopes.last_mut() {
                scope.replace_self(None);
                if nested_type {
                    scope.without_self();
                }
            }
            self.visit_type_mut(default);
            if let (Some(scope), Some(own)) = (self.scopes.last_mut(), own) {
                *scope = own;
            }
            self.self_requirement = self_requirement;
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
        let declares = Declares::of_item(&i.ident, &i.generics);
        self.enter(&mut i.attrs, &mut i.generics, Some(declares), true)
            .visit_fields_mut(&mut i.fields);
    }
    fn visit_item_enum_mut(&mut self, i: &mut ItemEnum) {
        let declares = Declares::of_item(&i.ident, &i.generics);
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, Some(declares), true);
        for variant in i.variants.iter_mut() {
            visitor.visit_variant_mut(variant);
        }
    }
    fn visit_item_trait_mut(&mut self, i: &mut ItemTrait) {
        let declares = Declares::of_trait(&i.ident, &i.generics);
        let gats = Gat::of_items(&i.items);
        let depth = self.scopes.len();
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, Some(declares), true);
        // The scope of the trait knows what its generic associated types require, as the Leaker's
        // does. If the trait does not declare its generics (they are those of the context), the
        // scope of the context has them.
        if visitor.scopes.len() > depth {
            if let Some(scope) = visitor.scopes.last_mut() {
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
        let declares = Declares::of_item(&i.ident, &i.generics);
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
        let declares = Declares::of_impl(&i.self_ty);
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
            let found = self.referrer.module_in(self.module, &i.ident);
            // At the top of the module of the context, a `mod` that none of its modules is named
            // like is the module itself: its items are those of the context, as in a `File`.
            let itself = self.module_scope && self.item_depth == 1 && found == ModulePos::Elsewhere;
            if itself {
                // A `mod` in it is nested in the module, as for its items.
                visitor.module_scope = false;
            } else {
                visitor.module = found;
                visitor.mod_path.push(i.ident.clone());
                // A module has no generics: it only makes its items nested. `Self` has no meaning
                // in it.
                visitor.item_depth += 1;
                visitor.self_requirement = None;
                visitor.nested = None;
            }
            visitor.shadow_items(items.iter(), true);
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
        visitor.shadow_items(i.items.iter(), true);
        syn::visit_mut::visit_file_mut(&mut visitor, i)
    }
    fn visit_block_mut(&mut self, i: &mut Block) {
        let mut visitor = self.clone();
        visitor.shadow_items(
            i.stmts.iter().filter_map(|stmt| match stmt {
                Stmt::Item(item) => Some(item),
                _ => None,
            }),
            false,
        );
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
//     group       = [# id] [declaration] ( Type , .. )
//     declaration = [< generics > where .. ;]
//                   [Self : path ; | Self = Type ; | Self ;]
//                   [Self : 'a + 'b ;]
//                   [type G < 'b > where Self : 'b ; ..]
//                   [mod id ; | mod { path = [a, ..], hint = "h", parent = id } ;]
//
// - The groups hold the entries in the order of their indices: a group is a run of the entries of
//   one context. A context is declared by its first group. Its other groups (the entries of a
//   context need not be together) have no declaration: `[# id] ( Type , .. )`.
// - `# id` is written only if `id` is not the one after that of the group before (the first group
//   is expected to have `0`).
// - A context with no entry has one group, `()`, before the first group of a larger id.
// - The `KIND ( CAUSE )` of a part is that of an `ErrorItem`. A bracket cannot start a group, so
//   `parts` needs no marker. The count lets other tokens follow the Referrer.

const FORMAT_TAG_PREFIX: &str = "type-leak-format-";

impl Referrer {
    /// The version of the serialized form, written as the tag `"type-leak-format-N"`.
    ///
    /// Parsing rejects another tag, or none.
    // Bump by hand when an old serialized Referrer would behave differently: the syntax, what a
    // context holds, how enclosing lifetimes pass on, which paths are associated type paths, or how
    // an entry is found. Docs, errors and encoders need no bump.
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
    //    trait's `Self` too, and written by `Visitor::with_nested_self()` (`U = Rel<T>` with
    //    `T: AsRef<Self>`), and that of the item itself without `Self`, as the Leaker analyzes it.
    // 14: the context of an entry in the default of a parameter of a struct, an enum or a union
    //    nested in the analyzed syntax has no bound of the generics that names the item's `Self`
    //    (`U = Wrap<T>` with `T: Clone + AsRef<Self>` is `T: Clone`, any type), which the Referrer
    //    computes for a copy, so the entries of 13 are not found the same way; and an entry that
    //    is for any type may name `Self` in its type (`nested_self_entries`), which the Referrer
    //    finds too.
    // 15: the context of an entry leaves out a bound of a generic that names the generic in the
    //    arguments of its path (`T: Fn(Alias<T>)`), if another bound of it fixes an associated
    //    type (`T: Iterator<Item = u8>`), which the Referrer computes for a copy, so the entries of
    //    14 are not found the same way.
    // 16:
    //    - The Referrer carries the types that the Leaker copies as written, the parts (see the
    //      grammar above), and the visitor does not look each up as an entry; in a `whole` one
    //      nothing is looked up. Only a part that is whole, has an anchor or a mark is carried: a
    //      type copied as written whose head alone conflicts is no part (an entry of its tokens
    //      and context would have the same conflict), and the types in it are replaced as any
    //      entry is. A part that a signature made has a mark (`sig ('x)`: the method lifetimes of
    //      the type that the output names and no bound does, E0195; `sig (elided)`: the signature
    //      relies on elision, E0106) and is left only in the arguments of a signature that has
    //      the same fault, which the Referrer computes at each.
    //    - The context of an entry that depends on `Self` only through a bound or an outlives
    //      predicate (`Self: 'x`) keeps whole the bounds of the generics that the description of
    //      `Self` names (`Tr<T>`) and leaves out those of the others (15 kept all), which the
    //      Referrer computes for a copy.
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
    //    - Not serialized: the Repeater has one `KEY: u128` (`repeater_key(ID, NTH)`) instead of
    //      `ID: u64, NTH: usize`, and a `U64` slot instead of the `U128` one (a `u128` const goes
    //      in `KEY`), so the impls and projections of 15 do not match.
    //
    // Not serialized, but both macros must agree on it: the primitive type that `ImplSelf::Bucket`
    // and `Referrer::bucket_self()` choose from the id and the index of an entry, the layout of
    // `repeater_key()`, and the slots of `repeater_definition()` and `RepeaterSlots` (which const
    // type goes where) must not change without a bump.
    pub const FORMAT_VERSION: u32 = 16;
}

/// Parses a [`Referrer`] written by its [`ToTokens`] implementation.
///
/// Entries keep their indices, and the tokens after the Referrer are not consumed. Fails if the
/// format tag ([`Referrer::FORMAT_VERSION`]) is missing or differs, or if the contexts are
/// inconsistent (a module or context used but not declared, a cycle of module parents).
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
        Ok(Self::new(reader.entries, reader.contexts, parts))
    }
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
                "this Referrer was serialized by an incompatible type-leak ({}, expected {}): \
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
type Mentions = Vec<(Span, ContextId)>;

/// Reads the parts copied as written, if any, and the contexts they name.
fn read_parts(input: parse::ParseStream) -> Result<(Vec<PartItem>, Mentions)> {
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

/// What the parser has read of the groups.
#[derive(Default)]
struct Reader {
    entries: Vec<(ContextId, Type)>,
    contexts: BTreeMap<ContextId, Context>,
    /// The contexts that are modules (`mod { .. }`).
    modules: HashSet<ContextId>,
    /// The modules that contexts refer to (`mod 3 ;`).
    module_refs: Mentions,
    /// The parent of each module, with its span, checked when all the contexts are known.
    parents: BTreeMap<ContextId, (Span, ContextId)>,
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
            return self.read_entries(input, context, span);
        }
        let declared = self.declaration(input, context)?;
        let entries_span = input.span();
        self.contexts.insert(context, declared);
        self.read_entries(input, context, entries_span)
    }

    /// Reads what a group has before its entries when it declares `context`.
    fn declaration(&mut self, input: parse::ParseStream, context: ContextId) -> Result<Context> {
        let generics_span = input.span();
        let has_generics = input.peek(Token![<]) || input.peek(Token![where]);
        let mut generics = Generics::default();
        if has_generics {
            generics = input.parse()?;
            generics.where_clause = input.parse()?;
            input.parse::<Token![;]>()?;
        }
        // `Self: 'a + 'b ;` is told from `Self: path ;` by the lifetime.
        let mut self_ty = None;
        let mut self_outlives = Vec::new();
        let self_span = input.span();
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
        let (mut module, mut module_context) = (None, None);
        if input.parse::<Option<Token![mod]>>()?.is_some() {
            if input.peek(token::Brace) {
                module_context = Some(parse_module_info(input, context, &mut self.parents)?);
            } else {
                let span = input.span();
                let referred: ContextId = input.parse()?;
                self.module_refs.push((span, referred));
                module = Some(referred);
            }
            input.parse::<Token![;]>()?;
        }
        match module_context {
            Some(module_context) => {
                if has_generics {
                    return Err(in_module(context, generics_span, "which has no generics"));
                }
                if self_ty.is_some() || !self_outlives.is_empty() {
                    return Err(in_module(context, self_span, "where `Self` has no meaning"));
                }
                self.modules.insert(context);
                Ok(module_context)
            }
            None => Ok(Context::new(generics, self_ty)
                .with_self_outlives(self_outlives)
                .with_gats(gats)
                .with_module(module)),
        }
    }

    /// Reads the entries `( Type , .. )` of a group of `context`. `span` is for the error of a
    /// module that has entries.
    fn read_entries(
        &mut self,
        input: parse::ParseStream,
        context: ContextId,
        span: Span,
    ) -> Result<()> {
        let content: parse::ParseBuffer<'_>;
        parenthesized!(content in input);
        let group = Punctuated::<Type, Token![,]>::parse_terminated(&content)?;
        if self.modules.contains(&context) && !group.is_empty() {
            return Err(in_module(context, span, "which has no leaked type"));
        }
        self.entries
            .extend(group.into_iter().map(|ty| (context, ty)));
        Ok(())
    }

    /// Checks what refers to a context, now that all the contexts are known: the module of a
    /// context, the contexts of the parts, and the parents of the modules.
    fn check(&self, part_contexts: &[(Span, ContextId)]) -> Result<()> {
        let unknown = |id: ContextId| {
            if self.contexts.contains_key(&id) {
                format!(
                    "the context {} is not a module (it is not declared as `mod {{ .. }}`)",
                    id
                )
            } else {
                format!("there is no module {}", id)
            }
        };
        for (span, referred) in &self.module_refs {
            if !self.modules.contains(referred) {
                return Err(Error::new(*span, unknown(*referred)));
            }
        }
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
        for (module, (span, parent)) in &self.parents {
            if parent == module {
                return Err(Error::new(
                    *span,
                    format!("the module {} is its own parent", module),
                ));
            }
            if !self.modules.contains(parent) {
                return Err(Error::new(
                    *span,
                    format!("the parent of the module {}: {}", module, unknown(*parent)),
                ));
            }
        }
        for start in self.parents.keys() {
            let chain = std::iter::successors(Some(*start), |at| {
                self.parents.get(at).map(|(_, parent)| *parent)
            });
            if chain
                .skip(1)
                .take(self.parents.len())
                .any(|at| at == *start)
            {
                return Err(Error::new(
                    self.parents[start].0,
                    format!(
                        "the module {} is nested in itself (its parents form a cycle)",
                        start
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// Returns the error of a group of the module `context` that has what a module has not.
fn in_module(context: ContextId, span: Span, what: &str) -> Error {
    Error::new(
        span,
        format!(
            "the context {} is a module (`mod {{ .. }}`), {}",
            context, what
        ),
    )
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
            if lifetimes.contains(&lifetime) {
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

/// Reads the path, the hint and the parent of the module `context`, and returns its context. Each
/// key may appear once. The span and id of the parent go to `parents`, to be checked when all the
/// contexts are known.
fn parse_module_info(
    input: parse::ParseStream,
    context: ContextId,
    parents: &mut BTreeMap<ContextId, (Span, ContextId)>,
) -> Result<Context> {
    let content: parse::ParseBuffer<'_>;
    braced!(content in input);
    let (mut module_path, mut hint, mut parent) = (None, None, None);
    let mut seen: Vec<String> = Vec::new();
    while !content.is_empty() {
        let key: Ident = content.parse()?;
        let name = key.to_string();
        if seen.contains(&name) {
            return Err(Error::new(
                key.span(),
                format!("`{}` is written twice in the information of a module", name),
            ));
        }
        content.parse::<Token![=]>()?;
        match name.as_str() {
            "path" => {
                let path: parse::ParseBuffer<'_>;
                bracketed!(path in content);
                module_path = Some(
                    Punctuated::<Ident, Token![,]>::parse_terminated(&path)?
                        .into_iter()
                        .collect(),
                );
            }
            "hint" => hint = Some(content.parse::<LitStr>()?.value()),
            "parent" => {
                let span = content.span();
                let id: ContextId = content.parse()?;
                parent = Some(id);
                parents.insert(context, (span, id));
            }
            _ => {
                return Err(Error::new(
                    key.span(),
                    "expected `path`, `hint` or `parent`",
                ))
            }
        }
        seen.push(name);
        content.parse::<Option<Token![,]>>()?;
    }
    Ok(Context::module_scope(
        context.index(),
        module_path,
        hint,
        parent,
    ))
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
            tokens.extend(quote!((#(for ix in run), { #{&self.entries[ix].0} })));
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
        let found = &self.contexts[&context];
        let (generics, outlives) = (&found.generics, &found.self_outlives);
        tokens.extend(quote! {
            #(if !generics.params.is_empty() || generics.where_clause.is_some()) {
                #generics #{&generics.where_clause} ;
            }
        });
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
        match &found.module {
            Some(ModuleScope::Itself {
                path, hint, parent, ..
            }) => {
                let fields = [
                    path.as_ref()
                        .map(|path| quote!(path = [#(for ident in path), { #ident }])),
                    hint.as_ref().map(|hint| quote!(hint = #hint)),
                    parent.map(|parent| quote!(parent = #parent)),
                ];
                tokens
                    .extend(quote!(mod { #(for field in fields.iter().flatten()), { #field } } ;));
            }
            Some(ModuleScope::In(module)) => tokens.extend(quote!(mod #module ;)),
            None => (),
        }
    }
}

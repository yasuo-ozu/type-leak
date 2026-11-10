//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::allow::{Allow, Place as AllowPlace};
use crate::analysis::{PartItem, SignatureFacts, SignatureMark};
use crate::context::{Context, ContextId, Enclosing, Gat, ModuleInfo, Scope, SelfRequirement};
use crate::crates::absolute_paths;
use crate::elided::{name_elided_in_bare_fn, name_elided_in_bound};
use crate::encode::{check_root_args, complete_generics_args, param_as_arg, EncodeError};
use crate::error::Cause;
use crate::graph::TypeKey;
use crate::util::{
    bound_cause, generic_param_name, ident_name, mentions_names, parenthesize_bounded,
    type_mentions_self, ReplaceSelf,
};
use proc_macro2::{Literal, Punct, Spacing, Span, TokenStream};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use syn::ext::IdentExt;
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
/// replaces each entry by `<Self as Repeater<ARGS, ID, IX>>::Type` with
/// [`visitor()`](Referrer::visitor) (`ARGS` from [`entry_args()`](Referrer::entry_args)).
///
/// A's macro passes the `Referrer` and the top-level [`ContextId`] to B's as tokens ([`ToTokens`],
/// [`Parse`]); both crates must use the same version of `type-leak`. Emit an impl for every entry,
/// not only the [`required()`](Referrer::required) ones: B replaces an optional entry unless it
/// trusts it ([`LeakerResult::allow()`](crate::LeakerResult::allow)), and the trust is not
/// serialized (`==` ignores it), so a parsed Referrer trusts none. `examples/two-macros` is a
/// complete pair.
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
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    pub(crate) leak_types: Vec<Type>,
    pub(crate) contexts: Vec<ContextId>,
    pub(crate) musts: Vec<bool>,
    pub(crate) external_crates: Vec<Vec<String>>,
    /// By entry: a rule of [`LeakerResult::allow()`](crate::LeakerResult::allow) allowed every
    /// absolute path of the entry, so the copy may keep an optional entry as written. Not
    /// serialized, so not compared: a parsed Referrer trusts none.
    pub(crate) trusted: Unserialized<Vec<bool>>,
    pub(crate) context_map: BTreeMap<ContextId, Context>,
    /// Each is also a context of `context_map`.
    pub(crate) modules: BTreeMap<ContextId, ModuleInfo>,
    /// Keyed by tokens as in the Leaker, so a type in an invisible group is found too.
    pub(crate) map: HashMap<TypeKey, Vec<usize>>,
    /// The types copied as written that a position tells from an entry of the same tokens and
    /// context: a part is whole (nothing in it is looked up), or has an anchor or a signature
    /// mark. Serialized after the format tag.
    pub(crate) parts: Vec<PartItem>,
    /// The indices in `parts`, by the part.
    pub(crate) part_map: HashMap<Cause, Vec<usize>>,
    /// The indices in `parts` of the parts with an anchor (`within`), by the anchor, as a
    /// `Cause::Type`.
    pub(crate) anchor_map: HashMap<Cause, Vec<usize>>,
}

/// A value that the serialization does not keep: all are equal, so the derived `==` ignores it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Unserialized<T>(pub(crate) T);

impl<T> PartialEq for Unserialized<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T> Eq for Unserialized<T> {}

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

/// How the visitor looks up the default of a type parameter, while it visits the generics of an
/// item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Defaults {
    /// Not in such generics: a type is looked up as anywhere.
    Plain,
    /// With no `Self`. That is how the Leaker analyzes the default of the generics of a scope that
    /// it enters (`enter_*`, `intern_*`: alone, with the parameters before it), whatever `Self`
    /// the scope declares, and a default cannot name `Self` in a struct, an enum or a union.
    WithoutSelf,
    /// The generics of a trait nested in the visited syntax, while they are visited. The Leaker
    /// has either analyzed the default of such a trait with the `Self` of the trait
    /// ([`View::visit_item()`](crate::View::visit_item), the items of
    /// [`Leaker::intern_mod()`](crate::Leaker::intern_mod)), or, if the macro entered the trait
    /// with [`Leaker::enter_trait()`](crate::Leaker::enter_trait), without it.
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

/// Where a type or a node is among the predicates of the item, for the parts.
#[derive(Clone, Copy, Debug, Default)]
struct Place {
    /// In a bound or a where predicate of the item, or in what is in them.
    in_predicate: bool,
    /// In an argument of a function, whose `impl Trait` bounds are predicates.
    in_fn_arg: bool,
    /// In a part: it is left as written, so no entry is looked up.
    protected: bool,
}

/// How [`Referrer::expand_with()`] and [`Referrer::is_part()`] look a type up.
#[derive(Clone, Copy)]
struct Lookup<'a> {
    matching: Matching<'a>,
    module: ModulePos,
    /// The entries that a visited item hides (see `Visitor::shadow_items()`).
    hidden: Option<&'a HashSet<usize>>,
    place: Place,
    /// The signature that the visited syntax is in, for the parts that a signature made.
    signature: Option<&'a SignatureFacts>,
}

impl Referrer {
    /// `context_map` always has at least one context.
    pub(crate) fn new(
        entries: impl IntoIterator<Item = (ContextId, Type, bool, Vec<String>)>,
        mut context_map: BTreeMap<ContextId, Context>,
        modules: BTreeMap<ContextId, ModuleInfo>,
        parts: Vec<PartItem>,
    ) -> Self {
        if context_map.is_empty() {
            context_map.insert(ContextId::new(0), Context::none());
        }
        let mut referrer = Self {
            leak_types: Vec::new(),
            contexts: Vec::new(),
            musts: Vec::new(),
            external_crates: Vec::new(),
            trusted: Unserialized(Vec::new()),
            context_map,
            modules,
            map: HashMap::new(),
            parts,
            part_map: HashMap::new(),
            anchor_map: HashMap::new(),
        };
        for (ix, item) in referrer.parts.iter().enumerate() {
            let PartItem { part, within, .. } = item;
            referrer.part_map.entry(part.clone()).or_default().push(ix);
            if let Some((anchor, _)) = within {
                let anchor = Cause::Type(anchor.clone());
                referrer.anchor_map.entry(anchor).or_default().push(ix);
            }
        }
        for (ix, (context, ty, must, crates)) in entries.into_iter().enumerate() {
            referrer.map.entry(TypeKey::new(&ty)).or_default().push(ix);
            referrer.leak_types.push(ty);
            referrer.contexts.push(context);
            referrer.musts.push(must);
            referrer.external_crates.push(crates);
            referrer.trusted.0.push(false);
        }
        referrer
    }

    fn entry_context(&self, ix: usize) -> &Context {
        &self.context_map[&self.contexts[ix]]
    }

    fn get_context(&self, context: ContextId) -> &Context {
        self.context_map.get(&context).unwrap_or_else(|| {
            panic!(
                "the context {} is not one of the contexts of this Referrer",
                context
            )
        })
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
        for &ix in self.map.get(&TypeKey::new(ty))? {
            if hidden.map_or(false, |hidden| hidden.contains(&ix)) {
                continue;
            }
            let entry = self.entry_context(ix);
            if no_self && (entry.self_ty().is_some() || type_mentions_self(&self.leak_types[ix])) {
                continue;
            }
            let rank = match (module, entry.module()) {
                (ModulePos::In(module), Some(entry)) if entry == module => 0,
                (_, Some(_)) => continue,
                (_, None) => 1,
            };
            if let Some(cost) = Self::matches(matching, ty, scopes, enclosing, entry) {
                let key = (rank, cost);
                if best.map_or(true, |(better, _)| key < better) {
                    best = Some((key, ix));
                }
            }
        }
        let ix = best?.1;
        Some(ix).filter(|&ix| self.musts[ix] || !self.trusted.0[ix])
    }

    /// Trusts each optional entry for which `rule` allows every absolute path that it depends on
    /// ([`absolute_paths()`]), asked bare, at no place. A trust stays.
    pub(crate) fn trust<A: Allow + ?Sized>(&mut self, rule: &A) {
        for ix in 0..self.leak_types.len() {
            if self.musts[ix] || self.trusted.0[ix] {
                continue;
            }
            let paths = absolute_paths(&self.leak_types[ix], self.entry_context(ix));
            self.trusted.0[ix] = paths
                .iter()
                .all(|cause| rule.allow(cause, AllowPlace::default()));
        }
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
        let found = self.get_context(context);
        let in_module = match (at.module, found.module()) {
            (ModulePos::In(module), Some(found)) => found == module,
            (_, Some(_)) => false,
            (_, None) => true,
        };
        in_module && Self::matches(at.matching, ty, scopes, enclosing, found).is_some()
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
        let items = self.part_map.get(cause)?;
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
                            && facts.covers(mark, context.map(|context| self.get_context(context)))
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
        if self.anchor_map.is_empty() {
            return;
        }
        let items = match self.anchor_map.get(&Cause::Type(ty.clone())) {
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
        self.modules
            .iter()
            .find(|(_, info)| {
                info.parent == parent
                    && matches!(info.path.as_ref().and_then(|path| path.last()), Some(last) if last == ident)
            })
            .map_or(ModulePos::Elsewhere, |(id, _)| ModulePos::In(*id))
    }

    /// Returns the cost of `entry` for `ty`, or `None` if it does not match. It is `0` if the
    /// context of `ty` in `scopes` is `entry`'s. The entries of a default have a choice: the
    /// fewer parameters the entry uses, the less its cost.
    fn matches(
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

    /// Returns the generics, what `Self` means and the module of the scope of `context`.
    ///
    /// The module is the context itself if it is a module, else the module it was entered in.
    fn place(&self, context: ContextId) -> (Scope, Option<SelfRequirement>, ModulePos) {
        let found = self.get_context(context);
        (
            Scope::new(found.generics(), found.self_ty().cloned()).with_gats(found.gats().to_vec()),
            found.self_ty().cloned(),
            self.module_pos(context),
        )
    }

    /// Returns the module of the scope of `context`: the context itself if it is a module, else the
    /// module it was entered in.
    fn module_pos(&self, context: ContextId) -> ModulePos {
        let found = self.get_context(context);
        if found.is_module_scope() {
            ModulePos::In(context)
        } else {
            found.module().map_or(ModulePos::Outside, ModulePos::In)
        }
    }

    /// Returns the sorted names of the external crates that the `ix`-th entry names, in its type,
    /// in the bounds of its context and in its `Self` requirement.
    ///
    /// A crate is the first segment of an absolute path. `core` is not listed, nor are `crate`,
    /// `self`, `super` and `Self` (no crate), and `r#` is stripped.
    pub fn external_crates_require(&self, ix: usize) -> &[String] {
        &self.external_crates[ix]
    }

    /// Returns `true` if the `ix`-th entry is required, and `false` if it is optional.
    ///
    /// An entry is optional if only its own path is an absolute path that names an external crate
    /// (`::foo::X`) and the Leaker's rule does not allow it, and no type that depends on the
    /// context is in it: `::foo::W<Rel>` is required, as `Rel` is. A Referrer replaces an optional
    /// entry through the Repeater, like a required entry, unless it trusts it: the rule given to
    /// [`LeakerResult::allow()`](crate::LeakerResult::allow) allows every absolute path of the
    /// entry (it is then copied as written). The trust is not serialized. Required entries have the
    /// smaller indices.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S { a: ::foo::X, b: Rel }), true);
    /// let referrer = leaker.result().finish().unwrap();
    /// assert!(referrer.must(0)); // `Rel`
    /// assert!(!referrer.must(1)); // `::foo::X`
    /// ```
    pub fn must(&self, ix: usize) -> bool {
        self.musts[ix]
    }

    /// Returns the required entries ([`Referrer::must()`]) with their indices.
    ///
    /// Every Referrer replaces them, so each needs a Repeater impl; an optional entry needs one
    /// only if some Referrer does not trust it.
    pub fn required(&self) -> impl Iterator<Item = (usize, &Type)> {
        self.leak_types
            .iter()
            .enumerate()
            .filter(move |(ix, _)| self.musts[*ix])
    }

    /// Returns `true` if there are no entries, optional ones included.
    pub fn is_empty(&self) -> bool {
        self.leak_types.is_empty()
    }

    /// Returns the types of the entries by index, optional ones included.
    ///
    /// Write an impl for each ([`Referrer::repeater_impl()`]); a plain impl for any type is right
    /// only where [`Referrer::is_plain()`] holds.
    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.leak_types.iter()
    }

    /// Returns `true` if the `ix`-th entry can be written by an impl with no generics.
    ///
    /// That is, `impl Repeater<(), ID, IX> for Marker` is right for any marker: the context has no
    /// generics, no where clause, no `Self` and no module.
    pub fn is_plain(&self, ix: usize) -> bool {
        let context = self.contexts[ix];
        let generics = self.context_generics(context);
        generics.params.is_empty()
            && generics
                .where_clause
                .as_ref()
                .map_or(true, |clause| clause.predicates.is_empty())
            && self.context_self(context).is_none()
            && self.context_module(context).is_none()
    }

    /// Returns the context of the `ix`-th entry.
    ///
    /// Entries with the same generics, `Self` and module share it.
    pub fn context(&self, ix: usize) -> ContextId {
        self.contexts[ix]
    }

    /// Returns the contexts in ascending order, also those with no entry (top-level ones, parts).
    ///
    /// The ids may have gaps. There is always at least one.
    pub fn contexts(&self) -> impl Iterator<Item = ContextId> + '_ {
        self.context_map.keys().copied()
    }

    /// Returns the generics that the entries of `context` use.
    ///
    /// The Repeater impl encodes them in its trait arguments ([`Referrer::repeater_impl()`]). A
    /// top-level context has only the bounds its scope writes: encode the generics of each entry,
    /// not of the top-level context (E0309, see [Outlives bounds](crate::Leaker#outlives-bounds)).
    /// Panics if `context` is not in [`contexts()`](Referrer::contexts).
    pub fn context_generics(&self, context: ContextId) -> &Generics {
        self.get_context(context).generics()
    }

    /// Returns what `Self` means in `context`, if anything.
    ///
    /// For the context of an entry it is `Some` only if the entry, or a bound of a generic it uses,
    /// names `Self`. The Repeater impl is then for the implementors of a trait
    /// ([`SelfRequirement::Bound`]), for the type ([`SelfRequirement::Type`]), or for any type that
    /// a bound holds for ([`SelfRequirement::Any`]); with `None` it is for any type. `None` does
    /// not say that the generics name no `Self`: the entry in the default of a parameter has none
    /// (`U = Rel2<T>` with `T: AsRef<Self>`), though its generics do ([`Referrer::needs_self()`]
    /// tells). For a top-level context it is what the scope declares if an entry depends on it,
    /// else `None`.
    /// Panics if `context` is not in [`contexts()`](Referrer::contexts).
    pub fn context_self(&self, context: ContextId) -> Option<&SelfRequirement> {
        self.get_context(context).self_ty()
    }

    /// Returns the sorted lifetimes `'a` for which the types of `context` need `Self: 'a`.
    ///
    /// [`Referrer::context_generics()`] does not list them, and the Repeater impl must not declare
    /// them: its trait arguments encode them (`&'a Self`). See
    /// [Outlives bounds](crate::Leaker#outlives-bounds). A `Self: 'a` that only the where clause of
    /// the trait writes is not in the copy of a method: for a lifetime that the copy declares,
    /// write `where Self: 'a` there again. Panics if `context` is not in
    /// [`contexts()`](Referrer::contexts).
    pub fn context_self_outlives(&self, context: ContextId) -> &[Lifetime] {
        self.get_context(context).self_outlives()
    }

    /// Returns the module where the types of `context` are written, as its `ContextId`.
    ///
    /// For the context of an entry it is `Some` only for an entry that means something only in its
    /// module (a relative path, a name the module declares, a type macro): emit its Repeater impl
    /// inside that module (for [`Leaker::intern_mod()`](crate::Leaker::intern_mod), follow
    /// [`Referrer::module_path()`]; the module of `enter_mod()` is the one you entered). Other
    /// entries are shared by all modules. For the top-level context of a scope entered in a module
    /// it is that module, and `None` for a module context. Panics if `context` is not in
    /// [`contexts()`](Referrer::contexts).
    pub fn context_module(&self, context: ContextId) -> Option<ContextId> {
        self.get_context(context).module()
    }

    /// Returns the path of `module` from its root: `[]` for the module of `intern_mod()`.
    ///
    /// The path of a nested module is relative to it. A module found with no module around it is a
    /// root: its path starts with its own name. It is `None` for a module of `enter_mod()` (see
    /// [`Referrer::module_hint()`]) and for an unknown id, where
    /// [`context_module()`](Referrer::context_module) panics.
    pub fn module_path(&self, module: ContextId) -> Option<&[Ident]> {
        self.modules.get(&module)?.path.as_deref()
    }

    /// Returns the hint given to [`Leaker::enter_mod()`](crate::Leaker::enter_mod) for `module`, if
    /// any. It is `None` for an unknown id too, where
    /// [`context_module()`](Referrer::context_module) panics.
    pub fn module_hint(&self, module: ContextId) -> Option<&str> {
        self.modules.get(&module)?.hint.as_deref()
    }

    /// Returns `true` if some entry [needs `Self`](Referrer::needs_self).
    ///
    /// `Self::Item` and `<Self as ::p::Tr>::X` count, `::p::Self` does not. A top-level context
    /// with no entry is not looked at.
    pub fn has_self(&self) -> bool {
        (0..self.leak_types.len()).any(|ix| self.needs_self(ix))
    }

    /// Fills in the arguments that a copy omits because the trait gives them defaults.
    ///
    /// `context` is where the trait is written: the top-level [`ContextId`] that the Leaker
    /// returned for it, or for a trait of [`Leaker::intern_mod()`](crate::Leaker::intern_mod) the
    /// context of its module ([`Referrer::module_path()`] tells which). `params` are the parameters
    /// of the trait; `args` match them by position, lifetimes first (a bare identifier for a const
    /// parameter is a const argument, and bindings such as `Item = u8` are skipped). Fails with
    /// [`EncodeError`] if there are too many arguments, or one without a default is missing.
    ///
    /// A default may name leaked types: `type_fn(leaked, index, args)` replaces each one, where
    /// `args` are the complete arguments before the defaulted parameter. Return a path through the
    /// Repeater ([`Referrer::entry_args()`]); it is not substituted again. The entries of the
    /// module of `context` are looked up, and those of no module: a default that names an item of
    /// the module (`U = Rel2<T>` with `Rel2` imported there) is one. A const default is used as
    /// written, so one that names an item of the Leaker's scope (`= MY_CONST`) does not resolve in
    /// the Referrer: give such an argument explicitly. Panics if `context` is not in
    /// [`contexts()`](Referrer::contexts).
    pub fn complete_args<'a>(
        &self,
        context: ContextId,
        params: &Punctuated<GenericParam, Token![,]>,
        args: impl IntoIterator<Item = &'a GenericArgument>,
        mut type_fn: impl FnMut(Type, usize, &[GenericArgument]) -> Type,
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        let module = self.module_pos(context);
        use syn::fold::Fold;
        complete_generics_args(
            args,
            params,
            |index, _param, default, substitute, completed| {
                // The parameters before the defaulted one are the Leaker's generics for its default.
                let root = Generics {
                    params: params.iter().take(index).cloned().collect(),
                    ..Default::default()
                };
                // Placeholders keep the substitution from touching the replaced paths.
                let mut leaked = Vec::new();
                let at = Lookup {
                    matching: Matching::Default,
                    module,
                    hidden: None,
                    place: Place::default(),
                    signature: None,
                };
                let ty = self.expand_with(
                    at,
                    &mut vec![Scope::new(&root, None)],
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
                let ty = substitute.fold_type(ty);
                Ok(Resolve {
                    leaked,
                    completed,
                    type_fn: &mut type_fn,
                }
                .fold_type(ty))
            },
        )
    }

    /// Returns the arguments for the generics of `context`, as written in the copy.
    ///
    /// `completed_args` are the arguments of the Leaker's parameters `root_params`
    /// ([`Referrer::complete_args()`]), by position. A generic named like one of them takes its
    /// argument, and any other is passed by its own name, so the copy must declare it. `Self` is not
    /// an argument.
    ///
    /// Fails with [`EncodeError::UnexpectedArgument`] if there are more arguments than parameters,
    /// or one is not of the kind of its parameter (a lifetime, a type, a const: a binding such as
    /// `Item = u8` is none), and with [`EncodeError::MissingArgument`] if the generics need an
    /// argument that is not there. `completed_args` may stop early, as the `prefix` that
    /// `complete_args()` gives `type_fn` does. For the trait argument of the Repeater use
    /// [`Referrer::entry_args()`], which also encodes lifetime bounds, `Self: 'a` and consts.
    /// Panics if `context` is not in [`contexts()`](Referrer::contexts).
    pub fn context_generic_args(
        &self,
        context: ContextId,
        root_params: &Punctuated<GenericParam, Token![,]>,
        completed_args: &[GenericArgument],
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        check_root_args(root_params, completed_args)?;
        self.context_generics(context)
            .params
            .iter()
            .map(|param| {
                let name = generic_param_name(param);
                match root_params
                    .iter()
                    .position(|root| generic_param_name(root) == name)
                {
                    Some(pos) => completed_args
                        .get(pos)
                        .cloned()
                        .ok_or_else(|| EncodeError::MissingArgument(Box::new(param.clone()))),
                    None => Ok(param_as_arg(param)),
                }
            })
            .collect()
    }

    /// Creates a [`Visitor`] that rewrites the leaked types in a copy of the analyzed item.
    ///
    /// `context` is the scope of the copy: the top-level [`ContextId`] that the `intern_*` or
    /// `enter_*` method of the Leaker returned for the item. Pass it along with the Referrer (it is
    /// written and parsed as an integer). `type_fn(leaked, ix)` returns the replacement, usually
    /// `<Self as Repeater<ARGS, ID, IX>>::Type` with `ARGS` from [`Referrer::entry_args()`]. Only
    /// the leaked types are replaced: write the rest of the copy yourself (e.g. `u8` for `T` in a
    /// copy for `impl Tr<u8>`, which [`UseSite::substitute()`](crate::UseSite::substitute) does for
    /// a type). A type is replaced if its context, computed from the generics in
    /// scope, is that of an entry, so visit the whole item, not a type cut out of it:
    ///
    /// - A generic that the copy declares again (`T` in `impl<T: Clone> MyTrait<T> for ..`) must be
    ///   declared as the entry has it, bounds included, unless you call
    ///   [`Visitor::with_outermost_generics_as_leakers()`].
    /// - A struct, enum, union, trait or impl nested in the copy has a `Self` of its own, as in the
    ///   Leaker, so nest the same items. The entries that depend on it need
    ///   [`Visitor::with_nested_self()`].
    /// - In a module copy, an entry with a module matches only there
    ///   ([`Referrer::context_module()`]), one with none everywhere. A `mod` item is a module
    ///   nested in that of `context`, found by name. For a module, visit its items: a `mod` that
    ///   no module of the context is named like is also taken as the module itself, but not one
    ///   that is.
    ///
    /// The Visitor borrows the Referrer, so `type_fn` may use it ([`Referrer::entry_args()`]), and
    /// several Visitors may share one Referrer.
    ///
    /// Panics if `context` is not in [`contexts()`](Referrer::contexts).
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
    /// let expected: Signature = parse_quote!(
    ///     fn f(&self, t: T) -> <Self as crate::Repeater<(u8,), 7u64, 0usize>>::Type
    /// );
    /// assert_eq!(sig, expected);
    /// ```
    pub fn visitor<'r, F: FnMut(Type, usize) -> Type>(
        &'r self,
        context: ContextId,
        type_fn: F,
    ) -> Visitor<'r, F> {
        let (scope, self_requirement, module) = self.place(context);
        let in_module = self.get_context(context).is_module_scope();
        Visitor {
            referrer: self,
            hidden: Rc::default(),
            type_fn: Rc::new(RefCell::new(type_fn)),
            nested_fn: None,
            self_requirement,
            scopes: vec![scope],
            outermost_as_leakers: false,
            self_ty: None,
            // The items of a module are nested items.
            item_depth: usize::from(in_module),
            own_self: false,
            nested: None,
            mod_path: Vec::new(),
            defaults: Defaults::Plain,
            module,
            module_scope: in_module,
            place: Place::default(),
            signature: None,
        }
    }

    /// Replaces the leaked types in one type with what `type_fn(leaked, index)` returns.
    ///
    /// `ty` is inside the scope `context`, as for [`Referrer::visitor()`]. Pass the whole type, not
    /// one cut out of a larger type ([Outlives bounds](crate::Leaker#outlives-bounds)). Only the
    /// generics of `context` are known: a type that names a generic of a method is not found, and a
    /// type that the Leaker copied as written for the signature it is in is replaced. Use
    /// [`Referrer::visitor()`] on the signature then. Panics if `context` is not in
    /// [`contexts()`](Referrer::contexts).
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
    /// let ty = referrer.expand(context, ty, |_, ix| {
    ///     parse_quote!(<Marker as Repeater<#ix>>::Type)
    /// });
    /// let expected: Type =
    ///     parse_quote!(::core::option::Option<<Marker as Repeater<0usize>>::Type>);
    /// assert_eq!(ty, expected);
    /// ```
    #[must_use]
    pub fn expand(
        &self,
        context: ContextId,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        let (scope, self_requirement, module) = self.place(context);
        let matching = Matching::Scope {
            self_ty: self_requirement.as_ref(),
            no_self: false,
            nested_type: false,
            alone: false,
        };
        // A type cut out of an item is not known to be in a predicate.
        let at = Lookup {
            matching,
            module,
            hidden: None,
            place: Place::default(),
            signature: None,
        };
        self.expand_with(at, &mut vec![scope], None, ty, type_fn)
    }

    fn expand_with(
        &self,
        at: Lookup,
        scopes: &mut Vec<Scope>,
        self_ty: Option<&Type>,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        use syn::fold::Fold;
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
                if !self.referrer.part_map.is_empty() {
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
                        self.referrer.anchors(
                            &ty,
                            self.at,
                            self.scopes,
                            &enclosing,
                            &mut self.active,
                        );
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
                    name_elided_in_bare_fn(replace.self_ty, &self.scopes[..], &mut i);
                }
                let depth = self.scopes.len();
                self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
                let i = syn::fold::fold_type_bare_fn(self, i);
                self.scopes.truncate(depth);
                i
            }
            fn fold_trait_bound(&mut self, mut i: TraitBound) -> TraitBound {
                if let Some(replace) = self.replace_self {
                    name_elided_in_bound(replace.self_ty, &self.scopes[..], &mut i);
                }
                let depth = self.scopes.len();
                self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
                let i = syn::fold::fold_trait_bound(self, i);
                self.scopes.truncate(depth);
                i
            }
        }
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

// Prefix of the placeholder names in `Referrer::complete_args()`.
const PLACEHOLDER: &str = "__TypeLeakPlaceholder";

struct Resolve<'a, F> {
    leaked: Vec<(Type, usize)>,
    completed: &'a [GenericArgument],
    type_fn: &'a mut F,
}

impl<'a, F: FnMut(Type, usize, &[GenericArgument]) -> Type> syn::fold::Fold for Resolve<'a, F> {
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
}

/// A struct, enum, union, trait or impl nested in the visited syntax, which has a `Self` of its
/// own.
///
/// [`Visitor::with_nested_self()`] gives it to the closure that writes the replacement of an entry
/// that depends on that `Self`. A type copied as written keeps its `Self`, the copy's own type in a
/// struct, enum or union.
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
    /// What `Self` is in the item: `S<T>` ([`SelfRequirement::Type`]) for a struct, enum or union,
    /// `Tr<T>` ([`SelfRequirement::Bound`]) for a trait, the self type as written for an impl.
    pub requirement: SelfRequirement,
    /// The name of the struct, enum, union or trait; `None` for an impl.
    pub ident: Option<Ident>,
    /// The `mod` items between the module of the context and the item, outermost first.
    pub modules: Vec<Ident>,
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
/// let expected: ItemTrait = parse_quote! {
///     trait Tr<T> { fn f(&self, t: T) -> <Self as crate::Repeater<(T,), 7u64, 0usize>>::Type; }
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
    outermost_as_leakers: bool,
    self_ty: Option<Type>,
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
    place: Place,
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
    /// parameter names `Self`, `type_fn` picks the host that has the bound in the copy.
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
    /// `Self`. The usual result is `<self_ty as Repeater<args, ID, ix>>::Type`
    /// ([`RepeaterSlots::projection()`](crate::RepeaterSlots::projection)), with `args` from
    /// [`Referrer::entry_args()`] at a [`UseSite`](crate::UseSite) that has
    /// [`with_self(self_ty)`](crate::UseSite::with_self). `self_ty` depends on what the impl of the
    /// entry is for ([`Referrer::context_self()`]):
    ///
    /// - [`SelfRequirement::Type`] (struct, enum, union, impl): the Leaker's own type (`S<T>` in
    ///   the Leaker's module), which the copy cannot name as `Self`: it declares an `S` of its own.
    ///   Write it from [`NestedSelf::ident`], [`NestedSelf::modules`] and the path of the module in
    ///   the Leaker's crate. An impl needs no such type: its `Self` is its self type as written.
    /// - [`SelfRequirement::Bound`] (trait): a host that implements it, also for the default of a
    ///   parameter (`trait Tr<T: AsRef<Self>, U = Rel<T>>`). The arguments of the Repeater include
    ///   the defaulted parameter, which a declaration cannot name in its own default (E0128): pass
    ///   a type for it with [`UseSite::substituting()`](crate::UseSite::substituting).
    /// - [`SelfRequirement::Any`] (a struct, enum or union whose generic has a bound that names
    ///   `Self`: `Rel<T>` with `T: AsRef<Self>`): any type that the bound holds for. Pass the
    ///   copy's own `Self` (what `UseSite::identity()` has), not the Leaker's type (E0277). With
    ///   [`nested_self_entries()`](crate::Options::nested_self_entries) the type of the entry
    ///   may name `Self` too. The impl is `?Sized` unless a bound or the type needs `Self: Sized`;
    ///   then, before rustc 1.70, a struct whose last field is the entry is E0275 (put the field
    ///   first, or add `where Self: Sized`).
    ///
    /// Without this method the copy keeps the types of these entries as written, which its crate
    /// cannot resolve. For a scope that is not a module (`enter_generics()`, ..), the nested items
    /// are in a function body of the copy.
    ///
    /// A type copied as written keeps its `Self`, in a nested struct, enum or union the copy's own
    /// type. So an absolute type that names it (`::p::Rel2<Self, T>`, `Box<Self>`) is no entry
    /// there, not even an optional one: replaced, it would hold the Leaker's type, and `Box<Self>`
    /// would not be recursive.
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

    // For a module, only entries with no module are hidden: the others are keyed by their module.
    fn shadow_items<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>, module: bool) {
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
        if names.is_empty() {
            return;
        }
        let referrer = self.referrer;
        let mut hidden = Vec::new();
        for (key, entries) in &referrer.map {
            if !mentions_names(&key.ty, names.clone(), Vec::new()) {
                continue;
            }
            for &ix in entries {
                let context = &referrer.context_map[&referrer.contexts[ix]];
                if module && context.module().is_some() {
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
        if !hidden.is_empty() {
            Rc::make_mut(&mut self.hidden).extend(hidden);
        }
    }

    /// Returns `true` if `cause`, a bound whose trait path conflicts, is a part: it is no type.
    /// `path` stands for it as a type, to give it its context.
    fn is_part_node(&self, cause: impl FnOnce() -> Cause, path: &Path) -> bool {
        if self.referrer.part_map.is_empty() || self.place.protected {
            return false;
        }
        let nested = self.own_self && self.nested.is_some() && self.nested_fn.is_some();
        let at = Lookup {
            matching: Matching::Scope {
                self_ty: self.self_requirement.as_ref(),
                no_self: self.own_self && !nested,
                nested_type: self.own_self
                    && matches!(&self.nested, Some(nested)
                        if nested.ident.is_some()
                            && matches!(nested.requirement, SelfRequirement::Type(_))),
                alone: self.defaults == Defaults::OwnTraitDefault,
            },
            module: self.module,
            hidden: Some(&self.hidden),
            place: self.place,
            signature: self.signature.as_deref(),
        };
        self.referrer
            .part_of(
                &cause(),
                &path_type(path),
                at,
                &self.scopes,
                &Enclosing::default(),
                &[],
            )
            .is_some()
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
        // The Leaker analyzes the defaults of a scope without `Self`, and those of a trait nested
        // in the visited syntax with its own (`U = Rel<T>` with `T: AsRef<Self>` is an entry of
        // the trait's `Self`, written as any entry of it), see `Defaults`.
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
        // In a nested item, the entries that name its own `Self` are found, and written by
        // `nested_fn`; with none, `Self` hides the one around and nothing that names it is found.
        let nested = match (&self.nested, &self.nested_fn) {
            (Some(nested), Some(nested_fn)) if self.own_self => Some((nested, nested_fn)),
            _ => None,
        };
        let self_ty = self.self_requirement.as_ref();
        let own_self = self.own_self;
        let no_self = own_self && nested.is_none();
        // As in the Leaker (`AnalyzeVisitor::self_nested`): the item is a struct, an enum or a
        // union, and not an impl, which has the self type it is written for.
        let nested_type = own_self
            && matches!(&self.nested, Some(nested)
                if nested.ident.is_some() && matches!(nested.requirement, SelfRequirement::Type(_)));
        let referrer = &self.referrer;
        let mut type_fn = self.type_fn.borrow_mut();
        let replace = |ty: Type, ix: usize| -> Type {
            match nested {
                Some((nested, nested_fn)) if referrer.entry_context(ix).self_ty().is_some() => {
                    let mut nested_fn = nested_fn.borrow_mut();
                    (*nested_fn)(nested, ty, ix)
                }
                _ => (*type_fn)(ty, ix),
            }
        };
        let at = Lookup {
            matching: Matching::Scope {
                self_ty,
                no_self,
                nested_type,
                alone: self.defaults == Defaults::OwnTraitDefault,
            },
            module: self.module,
            hidden: Some(&self.hidden),
            place: self.place,
            signature: self.signature.as_deref(),
        };
        *i = referrer.expand_with(
            at,
            &mut self.scopes,
            self.self_ty.as_ref().filter(|_| !own_self),
            i.clone(),
            replace,
        );
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
        if self.self_ty.is_none() || self.own_self {
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
        let part = self.is_part_node(|| bound_cause(i), &i.path);
        if let Some(self_ty) = self.self_ty.as_ref().filter(|_| !self.own_self) {
            name_elided_in_bound(self_ty, &self.scopes[..], i);
        }
        let depth = self.scopes.len();
        self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
        if part {
            self.protected(|visitor| syn::visit_mut::visit_trait_bound_mut(visitor, i));
        } else {
            syn::visit_mut::visit_trait_bound_mut(self, i);
        }
        self.scopes.truncate(depth);
    }
    fn visit_type_param_mut(&mut self, i: &mut TypeParam) {
        if matches!(self.defaults, Defaults::Plain | Defaults::OwnTraitDefault) {
            return syn::visit_mut::visit_type_param_mut(self, i);
        }
        if self.defaults == Defaults::OwnTrait {
            // The default of a trait nested in the visited syntax: the Leaker has analyzed it with
            // the `Self` of the trait (`View::visit_item()`), unless the macro entered the trait
            // itself (`Leaker::enter_trait()`), which analyzes it without. An entry of either is
            // found, the one without `Self` first (it is what `complete_args()` uses, and it does
            // not name the parameters after the default), and the other is written like any entry
            // of the trait's `Self`: `Visitor::with_nested_self()`, or left as written without it.
            let default = i.default.take();
            syn::visit_mut::visit_type_param_mut(self, i);
            i.default = default;
            if let Some(default) = &mut i.default {
                self.defaults = Defaults::OwnTraitDefault;
                self.visit_type_mut(default);
                self.defaults = Defaults::OwnTrait;
            }
            return;
        }
        // The Leaker analyzes the default of a parameter of a scope without `Self`: it
        // cannot depend on it, so the bounds of the scopes around that mention `Self` are
        // rewritten for the `Self` they were written for, also where the nested scope has the same
        // one. The scope itself is declared without `Self` for its default. A struct, an enum or
        // a union nested in the visited syntax has none of its own bounds that mention its `Self`
        // either: its copy cannot write `Self` in the default (E0735), so the entries in it do not
        // depend on what the copy's own type proves (`T: AsRef<Self>`).
        let default = i.default.take();
        syn::visit_mut::visit_type_param_mut(self, i);
        i.default = default;
        if let Some(default) = &mut i.default {
            let nested_type = self.own_self
                && matches!(&self.nested, Some(nested)
                    if nested.ident.is_some() && matches!(nested.requirement, SelfRequirement::Type(_)));
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

const FORMAT_TAG_PREFIX: &str = "type-leak-format-";

impl Referrer {
    /// The version of the serialized form, written as the tag `"type-leak-format-N"`.
    ///
    /// Parsing rejects another tag, or none.
    // Bump by hand when an old serialized Referrer would behave differently: the syntax, what a
    // context holds, how enclosing lifetimes pass on, which paths are associated type paths, or how
    // an entry is found. Docs, errors and encoders need no bump.
    // 1: first tagged, contexts carry the bounds enclosing types imply. 2: modules are contexts
    // (`mod 3 ;`, `mod { .. } ;`), no `Self = # 0 ;`. 3: the top-level context of a scope keeps
    // what `Self` means and the module the scope is in, for `Referrer::visitor()`. 4: it keeps
    // `Self` also when an outer bound was rewritten for it (`T: Tr<Self>` as `T: Tr<S<T>>`). The
    // tokens look the same, but the data of 3 would not find those entries the same way.
    // 5: the context of `&'a Self::A` in a trait has the where predicate `Self::A: 'a`, which the
    // Referrer computes for a copy too, so the entries of 4 are not found. 6: so has that of `&'a
    // T::A` or `&'a <T as Tr>::A` with `T` a generic (`T::A: 'a`, and no `T: 'a`). 7: so have
    // those of `&'a <Self as Tr<..>>::A` and `&'a Self::A<'b>` (the predicate, and no `'b: 'a` or
    // `T: 'a`), and of a `Self::A` that a where clause equates (what it equals is mentioned); no
    // projection of `Self` records `Self: 'a` any more. 8: the top-level context of a trait scope
    // carries its generic associated types with outlives (`type G<'b> where Self: 'b ;`), and the
    // context of an entry that names one has what its where clause requires (`Self: 'x`), as the
    // Referrer computes it for a copy. 9: the Referrer computes the context of a type from the
    // first segment of a relative path only, so `inner::U` next to a generic `U` is found. 10: the
    // Referrer finds an entry of no module that names a generic of its context also where the copy
    // declares an item of the same name (`crate::base::B1<X>` next to `fn f<X>` in a module that
    // declares `X`). 11: the context of an entry that does not name the `Self` of a struct, an
    // enum or a union nested in the analyzed syntax, but has a bound that does, is `Self ;` (any
    // type), and the Referrer finds it only in the item that declares that `Self`. 12: an absolute
    // type that names the `Self` of such an item is no optional entry any more (it is copied as
    // written, so its `Self` is the copy's own), and a strict Referrer replaces less. 13: the
    // default of a parameter of a trait nested in the analyzed syntax is looked up with the
    // trait's `Self` too, and written by `Visitor::with_nested_self()` (`U = Rel<T>` with `T:
    // AsRef<Self>`), and that of the item itself without `Self`, as the Leaker analyzes it. 14: the
    // context of an entry in the default of a parameter of a struct, an enum or a union nested in
    // the analyzed syntax has no bound of the generics that names the item's `Self` (`U =
    // Wrap<T>` with `T: Clone + AsRef<Self>` is `T: Clone`, any type), which the Referrer
    // computes for a copy, so the entries of 13 are not found the same way; and an entry that is
    // for any type may name `Self` in its type (`nested_self_entries`), which
    // the Referrer finds too. 15: the context of an entry leaves out a bound of a generic that
    // names the generic in the arguments of its path (`T: Fn(Alias<T>)`), if another bound of it
    // fixes an associated type (`T: Iterator<Item = u8>`), which the Referrer computes for a copy,
    // so the entries of 14 are not found the same way. 16: the Referrer carries the types that the
    // Leaker copies as written, after the tag (`[KIND (CAUSE) CONTEXT [whole] [sig (MARK)]
    // [predicate] [in (TYPE) CONTEXT], ..]`, `unwritable` for a context that cannot be written),
    // and the visitor does not look each up as an entry; in a `whole` one nothing is looked up.
    // Only a part that is whole, has an anchor or a mark is carried: a type copied as written whose
    // head alone conflicts is no part (an entry of its tokens and context would have the same
    // conflict), and the types in it are replaced as any entry is. A part that a signature made
    // has a mark (`sig ('x)`: the method lifetimes of the type that the output names and no bound
    // does, E0195; `sig (elided)`: the signature relies on elision, E0106) and is left only in the
    // arguments of a signature that has the same fault, which the Referrer computes at each. Also
    // in 16: the context of an entry that depends on `Self` only through a bound or an outlives
    // predicate (`Self: 'x`) keeps whole the bounds of the generics that the description of `Self`
    // names (`Tr<T>`) and leaves out those of the others (15 kept all), which the Referrer
    // computes for a copy. And a `LeakerResult` writes its errors after the Referrer as `@ [
    // {error} .. ]`, in one bracket group, so that a brace group after it is left to the caller.
    // An error writes each of its paths with its kind (`paths [type (Rel), trait (Tr), ..]`; a path
    // of unknown kind is `unknown (p)`).
    // Not serialized, but both macros must agree on it: the primitive type that `ImplSelf::Bucket`
    // and `Referrer::bucket_self()` choose from the id and the index of an entry must not change
    // without a bump.
    pub const FORMAT_VERSION: u32 = 16;
}

/// Parses a [`Referrer`] written by its [`ToTokens`] implementation.
///
/// Entries keep their indices, and the tokens after the Referrer are not consumed. Fails if the
/// format tag ([`Referrer::FORMAT_VERSION`]) is missing or differs, or if the contexts are
/// inconsistent (a module or context used but not declared, a cycle of module parents).
///
/// The Referrer trusts no optional entry ([`Referrer::must()`]), so it replaces all. A macro that
/// trusts crates parses a [`LeakerResult`](crate::LeakerResult) instead (the same tokens parse as
/// one) and calls [`allow()`](crate::LeakerResult::allow) with a rule that allows their paths.
impl Parse for Referrer {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let span = input.span();
        let tag = input.parse::<Option<LitStr>>()?;
        let value = tag.as_ref().map(LitStr::value);
        let expected = format!("{}{}", FORMAT_TAG_PREFIX, Self::FORMAT_VERSION);
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
                    Self::FORMAT_VERSION
                ),
            ));
        }
        // The parts copied as written, if any. Their contexts are checked when all the contexts are
        // known.
        let mut parts = Vec::new();
        let mut part_contexts: Vec<(Span, ContextId)> = Vec::new();
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
        let n_contexts = if input.peek(LitInt) {
            input.parse::<LitInt>()?.base10_parse::<usize>()?
        } else {
            1
        };
        let mut entries = Vec::new();
        let mut context_map = BTreeMap::new();
        let mut modules = BTreeMap::new();
        // Checked when all the contexts are known.
        let mut module_refs: Vec<(Span, ContextId)> = Vec::new();
        let mut parents = BTreeMap::new();
        let mut next = 0;
        for _ in 0..n_contexts {
            let span = input.span();
            let context = if input.parse::<Option<Token![#]>>()?.is_some() {
                let context: ContextId = input.parse()?;
                if context.index() < next {
                    return Err(Error::new(
                        span,
                        "the ids of the contexts must be in ascending order",
                    ));
                }
                context
            } else {
                ContextId::new(next)
            };
            next = context
                .index()
                .checked_add(1)
                .ok_or_else(|| Error::new(span, "the id of the context is too large"))?;
            let in_module = |span, what: &str| {
                let message = format!(
                    "the context {} is a module (`mod {{ .. }}`), {}",
                    context, what
                );
                Error::new(span, message)
            };
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
                    self_outlives =
                        Punctuated::<Lifetime, Token![+]>::parse_separated_nonempty(input)?
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
            let (mut module, mut module_info) = (None, None);
            if input.parse::<Option<Token![mod]>>()?.is_some() {
                if input.peek(token::Brace) {
                    module_info = Some(parse_module_info(input, context, &mut parents)?);
                } else {
                    let span = input.span();
                    let referred: ContextId = input.parse()?;
                    module_refs.push((span, referred));
                    module = Some(referred);
                }
                input.parse::<Token![;]>()?;
            }
            let entries_span = input.span();
            let parsed = match module_info {
                Some(info) => {
                    if has_generics {
                        return Err(in_module(generics_span, "which has no generics"));
                    }
                    if self_ty.is_some() || !self_outlives.is_empty() {
                        return Err(in_module(self_span, "where `Self` has no meaning"));
                    }
                    if !gats.is_empty() {
                        return Err(in_module(gats_span, "which has no trait"));
                    }
                    modules.insert(context, info);
                    Context::module_scope(context.index())
                }
                None => Context::new(generics, self_ty)
                    .with_self_outlives(self_outlives)
                    .with_gats(gats)
                    .with_module(module),
            };
            context_map.insert(context, parsed);
            let content: parse::ParseBuffer<'_>;
            parenthesized!(content in input);
            let group = Punctuated::<_, Token![,]>::parse_terminated_with(&content, |input| {
                let must = input.parse::<Option<Token![?]>>()?.is_none();
                let mut crates = Vec::new();
                if input.parse::<Option<Token![@]>>()?.is_some() {
                    let list: parse::ParseBuffer<'_>;
                    bracketed!(list in input);
                    crates = Punctuated::<Ident, Token![,]>::parse_terminated(&list)?
                        .into_iter()
                        .map(|ident| ident.unraw().to_string())
                        .collect();
                }
                Ok((input.parse::<Type>()?, must, crates))
            })?;
            if modules.contains_key(&context) && !group.is_empty() {
                return Err(in_module(entries_span, "which has no leaked type"));
            }
            entries.extend(
                group
                    .into_iter()
                    .map(|(ty, must, crates)| (context, ty, must, crates)),
            );
        }
        let unknown = |id: ContextId| {
            if context_map.contains_key(&id) {
                format!(
                    "the context {} is not a module (it is not declared as `mod {{ .. }}`)",
                    id
                )
            } else {
                format!("there is no module {}", id)
            }
        };
        for (span, referred) in module_refs {
            if !modules.contains_key(&referred) {
                return Err(Error::new(span, unknown(referred)));
            }
        }
        for (span, context) in part_contexts {
            if !context_map.contains_key(&context) {
                return Err(Error::new(
                    span,
                    format!(
                        "the context {} of a part is not one of the contexts",
                        context
                    ),
                ));
            }
        }
        for (module, (span, parent)) in &parents {
            if parent == module {
                return Err(Error::new(
                    *span,
                    format!("the module {} is its own parent", module),
                ));
            }
            if !modules.contains_key(parent) {
                return Err(Error::new(
                    *span,
                    format!("the parent of the module {}: {}", module, unknown(*parent)),
                ));
            }
        }
        for start in parents.keys() {
            let chain = std::iter::successors(Some(*start), |at| {
                parents.get(at).map(|(_, parent)| *parent)
            });
            if chain.skip(1).take(parents.len()).any(|at| at == *start) {
                return Err(Error::new(
                    parents[start].0,
                    format!(
                        "the module {} is nested in itself (its parents form a cycle)",
                        start
                    ),
                ));
            }
        }
        // The required entries come first (`Referrer::must()`).
        entries.sort_by_key(|(_, _, must, _)| !must);
        Ok(Self::new(entries, context_map, modules, parts))
    }
}

/// Reads a part written by its [`ToTokens`] implementation: `KIND ( CAUSE ) CONTEXT [whole] [sig (
/// MARK )] [predicate] [in ( TYPE ) CONTEXT]`, with the `KIND ( CAUSE )` of an
/// [`ErrorItem`](crate::ErrorItem), `unwritable` for a `CONTEXT` that cannot be written, and a
/// `MARK` that is `elided` or lifetimes (`'x , 'y`).
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

/// Each key may appear once. The span and id of the parent go to `parents`, to be checked when
/// all the contexts are known.
fn parse_module_info(
    input: parse::ParseStream,
    context: ContextId,
    parents: &mut BTreeMap<ContextId, (Span, ContextId)>,
) -> Result<ModuleInfo> {
    let content: parse::ParseBuffer<'_>;
    braced!(content in input);
    let mut info = ModuleInfo {
        path: None,
        hint: None,
        parent: None,
    };
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
                info.path = Some(
                    Punctuated::<Ident, Token![,]>::parse_terminated(&path)?
                        .into_iter()
                        .collect(),
                );
            }
            "hint" => info.hint = Some(content.parse::<LitStr>()?.value()),
            "parent" => {
                let span = content.span();
                let id: ContextId = content.parse()?;
                info.parent = Some(id);
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
    Ok(info)
}

/// `name` as an identifier, `r#name` if it is a keyword, so that it parses back as an identifier.
///
/// It is `None` for what is no identifier, or cannot be a raw one (`crate`, `self`, `super`, `Self`,
/// `_`): the Leaker lists none of them.
fn crate_ident(name: &str) -> Option<Ident> {
    if parse_str::<Ident>(name).is_ok() {
        Some(Ident::new(name, Span::call_site()))
    } else if parse_str::<Ident>(&format!("r#{}", name)).is_ok() {
        Some(Ident::new_raw(name, Span::call_site()))
    } else {
        None
    }
}

/// Writes the [`Referrer`] as tokens, to be read back by its [`Parse`] implementation.
///
/// The tokens start with the format tag ([`Referrer::FORMAT_VERSION`]). Treat the rest as opaque.
impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        // Grammar: tag [parts] [count] then one group per context in ascending id order (at least
        // one):
        // [# id] [<generics> where .. ;] [Self: path ; | Self = Type ; | Self ;] [Self: 'a + 'b ;]
        // [type G<'b> where Self: 'b ; ..]
        // [mod 3 ; | mod { path = [a], hint = "h", parent = 1 } ;] ( [?] [@[crate, ..]] Type, .. )
        // `# id` only after a gap in the ids. `()` if no entry. `?`: optional. `@[..]`: crates named.
        // parts: `[ part, .. ]`, only if there is one, as `PartItem` writes it: the `KIND ( CAUSE )`
        // of an `ErrorItem`, a context of the groups (`unwritable` if it cannot be written),
        // `sig ( 'x , .. )` or `sig ( elided )` for the signature that made it, `predicate` in a
        // bound, and `in ( Type ) context` for the type that owns the cause (the anchor). A bracket
        // cannot start a context group, so it needs no marker.
        let tag = format!("{}{}", FORMAT_TAG_PREFIX, Self::FORMAT_VERSION);
        let n_contexts = self.context_map.len();
        tokens.extend(quote! {
            #tag
            #(if !self.parts.is_empty()) { [#(for item in &self.parts), { #item }] }
            // The group count lets other tokens follow the Referrer.
            #(if n_contexts > 1) { #{Literal::usize_unsuffixed(n_contexts)} }
        });
        let mut next = 0;
        for context in self.contexts() {
            let generics = self.context_generics(context);
            let outlives = self.context_self_outlives(context);
            let entries = (0..self.leak_types.len()).filter(|&ix| self.contexts[ix] == context);
            tokens.extend(quote! {
                // `#` cannot be written in `quote!`, where it interpolates.
                #(if context.index() != next) { #{Punct::new('#', Spacing::Alone)} #context }
                #(if !generics.params.is_empty() || generics.where_clause.is_some()) {
                    #generics #{&generics.where_clause} ;
                }
            });
            match self.context_self(context) {
                None => (),
                Some(SelfRequirement::Bound(path)) => tokens.extend(quote!(Self : #path ;)),
                Some(SelfRequirement::Type(ty)) => tokens.extend(quote!(Self = #ty ;)),
                Some(SelfRequirement::Any) => tokens.extend(quote!(Self ;)),
            }
            if !outlives.is_empty() {
                tokens.extend(quote!(Self : #(for lifetime in outlives)+ { #lifetime } ;));
            }
            for gat in self.get_context(context).gats() {
                tokens.extend(quote!(#gat));
            }
            if let Some(info) = self.modules.get(&context) {
                let fields = [
                    info.path
                        .as_ref()
                        .map(|path| quote!(path = [#(for ident in path), { #ident }])),
                    info.hint.as_ref().map(|hint| quote!(hint = #hint)),
                    info.parent.map(|parent| quote!(parent = #parent)),
                ];
                tokens
                    .extend(quote!(mod { #(for field in fields.iter().flatten()), { #field } } ;));
            } else if let Some(module) = self.context_module(context) {
                tokens.extend(quote!(mod #module ;));
            }
            let crates = |ix: usize| -> Vec<Ident> {
                self.external_crates[ix]
                    .iter()
                    .filter_map(|name| crate_ident(name))
                    .collect()
            };
            tokens.extend(quote! {
                (
                    #(for ix in entries), {
                        #(if !self.musts[ix]) { ? }
                        #(if !crates(ix).is_empty()) {
                            @[#(for name in crates(ix)), { #name }]
                        }
                        #{&self.leak_types[ix]}
                    }
                )
            });
            next = context.index() + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_ident_is_total() {
        for name in ["p", "fn", "async", "type", "try"] {
            let ident = crate_ident(name).unwrap_or_else(|| panic!("no identifier for `{}`", name));
            // It parses back to the same name.
            let read = parse2::<Ident>(ident.to_token_stream()).unwrap();
            assert_eq!(read.unraw().to_string(), name);
        }
        for name in [
            "crate", "self", "super", "Self", "_", "", "a-b", "1a", "a b", "::p",
        ] {
            assert!(crate_ident(name).is_none(), "`{}`", name);
        }
    }

    #[test]
    fn a_name_that_cannot_be_written_is_left_out_of_the_tokens() {
        let mut leaker = crate::Leaker::new();
        leaker.intern_struct(
            &parse_quote!(
                struct S {
                    a: ::p::X,
                }
            ),
            true,
        );
        let mut referrer = leaker.result().finish().unwrap();
        assert_eq!(referrer.external_crates, [["p"]]);
        referrer.external_crates[0] = vec!["crate".into(), "p".into(), "Self".into()];
        let read: Referrer = parse2(referrer.to_token_stream()).unwrap();
        assert_eq!(read.external_crates, [["p"]]);
        referrer.external_crates[0] = vec!["self".into()];
        let read: Referrer = parse2(referrer.to_token_stream()).unwrap();
        assert_eq!(read.external_crates, [Vec::<String>::new()]);
    }
}

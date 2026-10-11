//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::context::{Context, ContextId, Enclosing, Gat, ModuleInfo, Scope, SelfRequirement};
use crate::elided::{name_elided_in_bare_fn, name_elided_in_bound};
use crate::encode::{complete_generics_args, param_as_arg, without_encoded_bounds, EncodeError};
use crate::graph::TypeKey;
use crate::util::{
    first_self_in_generics, generic_param_name, mentions_names, parenthesize_bounded,
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

/// The result of an analysis: the leaked types (the entries), ready to be emitted as Repeater
/// impls and to be replaced in a copy.
///
/// Get it from [`Leaker::finish()`](crate::Leaker::finish). In the crate that defines the item,
/// list the entries that need a Repeater impl ([`Referrer::required()`]) and describe each impl
/// ([`Referrer::repeater_impl()`]). A `Referrer` converts to and from tokens ([`ToTokens`] and
/// [`Parse`], see [`Referrer::FORMAT_VERSION`]), so a macro can pass it to the macro of the crate
/// that copies the item. There, [`Referrer::visitor()`] rewrites the copy.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use syn::parse::Parser;
/// use template_quote::{quote, ToTokens};
///
/// let mut leaker = Leaker::new();
/// let top_level = leaker
///     .intern_struct(&parse_quote!(struct S<T> { a: Wrap<T> }), true)
///     .unwrap();
/// let referrer = leaker.finish();
///
/// // Crate A writes the Referrer and the `ContextId` into the input of the macro of crate B...
/// let tokens = quote!(#referrer, #top_level);
/// // ...which reads them back. Both crates must use the same version of `type-leak`.
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
    /// Not serialized.
    pub(crate) allow_ambiguous_crate_name: bool,
    pub(crate) context_map: BTreeMap<ContextId, Context>,
    /// Each is also a context of `context_map`.
    pub(crate) modules: BTreeMap<ContextId, ModuleInfo>,
    /// Keyed by tokens as in the Leaker, so a type in an invisible group is found too.
    pub(crate) map: HashMap<TypeKey, Vec<usize>>,
}

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

/// How [`Referrer::find()`] compares the context of a type with those of the entries.
#[derive(Clone, Copy)]
enum Matching<'a> {
    /// The context of the type, computed from the generics in scope and `self_ty`, is the entry's.
    Scope {
        self_ty: Option<&'a SelfRequirement>,
        /// `Self` means something else here, so no entry that names it matches.
        no_self: bool,
    },
    /// The scope is that of the parameters before a defaulted one: the entry must use only them.
    Default,
}

impl Referrer {
    /// `context_map` always has at least one context.
    pub(crate) fn new(
        entries: impl IntoIterator<Item = (ContextId, Type, bool, Vec<String>)>,
        mut context_map: BTreeMap<ContextId, Context>,
        modules: BTreeMap<ContextId, ModuleInfo>,
    ) -> Self {
        if context_map.is_empty() {
            context_map.insert(ContextId::new(0), Context::none());
        }
        let mut referrer = Self {
            leak_types: Vec::new(),
            contexts: Vec::new(),
            musts: Vec::new(),
            external_crates: Vec::new(),
            allow_ambiguous_crate_name: true,
            context_map,
            modules,
            map: HashMap::new(),
        };
        for (ix, (context, ty, must, crates)) in entries.into_iter().enumerate() {
            referrer.map.entry(TypeKey::new(&ty)).or_default().push(ix);
            referrer.leak_types.push(ty);
            referrer.contexts.push(context);
            referrer.musts.push(must);
            referrer.external_crates.push(crates);
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
    /// matches ([Modules](crate::rules#modules)).
    fn find(
        &self,
        matching: Matching,
        module: ModulePos,
        ty: &Type,
        scopes: &[Scope],
        enclosing: &Enclosing,
    ) -> Option<usize> {
        let no_self = matches!(matching, Matching::Scope { no_self: true, .. });
        let mut best: Option<((usize, usize), usize)> = None;
        for &ix in self.map.get(&TypeKey::new(ty))? {
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
        Some(ix).filter(|&ix| !self.allow_ambiguous_crate_name || self.musts[ix])
    }

    /// Returns where a visited `mod ident` is, inside `module`: the module nested in it with that
    /// name ([Modules](crate::rules#modules)). A module that the Leaker did not see, or one nested in such
    /// a module, is a module no entry depends on.
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
            Matching::Scope { self_ty, .. } => {
                // A bound that has no name here (see `Context::of()`) is in no entry.
                let context = Context::of(ty, scopes, self_ty, enclosing).ok()?;
                let entry = entry.without_module();
                if context == entry {
                    Some(0)
                } else if context.is_entry_with_extra_self_predicates(&entry) {
                    Some(1)
                } else {
                    None
                }
            }
            Matching::Default => {
                if entry.self_ty().is_some() {
                    return None;
                }
                let root = param_names(scopes[0].params());
                let context = Context::of(ty, &scopes[..1], None, enclosing).ok()?;
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
        let module = if found.is_module_scope() {
            ModulePos::In(context)
        } else {
            found.module().map_or(ModulePos::Outside, ModulePos::In)
        };
        (
            Scope::new(found.generics(), found.self_ty().cloned()).with_gats(found.gats().to_vec()),
            found.self_ty().cloned(),
            module,
        )
    }

    /// Chooses whether the absolute paths of other crates (the optional entries) are left as
    /// written (`true`, the default) or replaced (`false`, a strict Referrer).
    ///
    /// Required entries are always replaced. Crates are matched by name only, so pass `false` if
    /// the Referrer's crate may have other dependencies than the Leaker's. The flag is not
    /// serialized, so the crate that emits the Repeater impls cannot know it: it writes the impls
    /// of all the entries ([`Referrer::iter()`]), the optional ones too, or a strict Referrer
    /// replaces a type that has no impl. See [External crates](crate::rules#external-crates).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_struct(&parse_quote!(struct S { a: ::foo::X }), true).unwrap();
    /// let referrer = leaker.finish();
    /// let ty: Type = parse_quote!(::foo::X);
    /// let replaced: Type = parse_quote!(Replaced);
    /// assert_eq!(referrer.expand(context, ty.clone(), |_, _| replaced.clone()), ty);
    /// let referrer = referrer.allow_ambiguous_crate_name(false);
    /// assert_eq!(referrer.expand(context, ty.clone(), |_, _| replaced.clone()), replaced);
    /// ```
    pub fn allow_ambiguous_crate_name(mut self, allow: bool) -> Self {
        self.allow_ambiguous_crate_name = allow;
        self
    }

    /// Returns the sorted, distinct names of the external crates that the `ix`-th entry names.
    ///
    /// A crate is named by the first segment of an absolute path in the type, its generics or its
    /// `Self`. `core` is not listed, and `r#` is stripped (`async` for `::r#async::X`).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .intern_struct(&parse_quote!(struct S { a: ::foo::X<::bar::Y>, b: Rel }), true)
    ///     .unwrap();
    /// let referrer = leaker.finish();
    /// // `Rel` is entry 0, `::foo::X<::bar::Y>` entry 1.
    /// assert!(referrer.external_crates_require(0).is_empty());
    /// assert_eq!(referrer.external_crates_require(1), ["bar", "foo"]);
    /// ```
    pub fn external_crates_require(&self, ix: usize) -> &[String] {
        &self.external_crates[ix]
    }

    /// Returns `true` if the `ix`-th entry is required, and `false` if it is optional.
    ///
    /// An entry is optional if only its own path is an absolute path that names an external crate.
    /// [`Referrer::allow_ambiguous_crate_name()`] then decides whether it is replaced. Other
    /// entries are required and always replaced, and need a Repeater impl. Required entries have
    /// the smaller indices. See [External crates](crate::rules#external-crates).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S { a: ::foo::X, b: Rel }), true).unwrap();
    /// let referrer = leaker.finish();
    /// assert!(referrer.must(0)); // `Rel`
    /// assert!(!referrer.must(1)); // `::foo::X`
    /// ```
    pub fn must(&self, ix: usize) -> bool {
        self.musts[ix]
    }

    /// Returns the required entries with their indices: those with [`Referrer::must()`], which
    /// come first.
    ///
    /// These are the entries that every Referrer replaces, so each needs a Repeater impl. The
    /// others are optional entries (absolute paths of other crates, left as written by a
    /// non-strict Referrer: see [`Referrer::allow_ambiguous_crate_name()`]), which
    /// [`Referrer::iter()`] and [`Referrer::is_empty()`] include. An optional entry needs an impl
    /// only if a strict Referrer replaces it: a crate that exports an item to crates it does not
    /// know writes the impls of all the entries ([`Referrer::iter()`]), not only of these.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .intern_struct(&parse_quote!(struct S { a: Rel, b: ::foo::X }), true)
    ///     .unwrap();
    /// let referrer = leaker.finish();
    /// // `::foo::X` is optional: another crate may name it differently.
    /// assert_eq!(referrer.iter().count(), 2);
    /// assert_eq!(referrer.required().count(), 1);
    /// let (ix, ty) = referrer.required().next().unwrap();
    /// assert_eq!((ix, ty), (0, &parse_quote!(Rel)));
    /// assert!(referrer.has_required());
    /// ```
    pub fn required(&self) -> impl Iterator<Item = (usize, &Type)> {
        self.leak_types
            .iter()
            .enumerate()
            .filter(move |(ix, _)| self.musts[*ix])
    }

    /// Returns `true` if some entry is required: [`Referrer::required()`] is not empty.
    ///
    /// A copy needs a Repeater impl only then. [`Referrer::is_empty()`] is `false` also for a
    /// Referrer that has optional entries only.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S { a: ::foo::X }), true).unwrap();
    /// let referrer = leaker.finish();
    /// assert!(!referrer.is_empty()); // `::foo::X` is an optional entry
    /// assert!(!referrer.has_required()); // and no Referrer needs an impl for it, unless strict
    /// ```
    pub fn has_required(&self) -> bool {
        self.musts.iter().any(|&must| must)
    }

    /// Returns `true` if there are no entries.
    ///
    /// Optional entries (absolute paths of other crates, left as written by a non-strict
    /// Referrer) count. Use [`Referrer::has_required()`] to know whether any Repeater impl is
    /// needed.
    pub fn is_empty(&self) -> bool {
        self.leak_types.is_empty()
    }

    /// Returns the types of the entries, by index.
    ///
    /// It includes the optional entries (absolute paths of other crates, left as written by a
    /// non-strict Referrer): use [`Referrer::required()`] for the entries that every Referrer
    /// replaces. A crate that exports an item writes the impls of all the entries, so that a
    /// strict Referrer in another crate finds them ([`Referrer::allow_ambiguous_crate_name()`]).
    ///
    /// A plain impl for any type is right only for an entry for which [`Referrer::is_plain()`]
    /// holds. Otherwise use [`Referrer::iter_with_generics()`], [`Referrer::iter_with_self()`] or
    /// [`Referrer::repeater_impl()`].
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S { a: Rel, b: ::foo::X }), true).unwrap();
    /// let referrer = leaker.finish();
    /// // One impl for each entry, the optional one too.
    /// let param: Ident = parse_quote!(__S);
    /// let impls: Vec<RepeaterImpl> = (0..referrer.iter().count())
    ///     .map(|ix| referrer.repeater_impl(ix, ImplSelf::Param(&param)).unwrap())
    ///     .collect();
    /// assert_eq!(impls.len(), 2);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.leak_types.iter()
    }

    /// Returns `true` if the `ix`-th entry can be written by an impl with no generics.
    ///
    /// That is so if its context has no generics and no where clause, does not name `Self` and
    /// has no module ([`Referrer::context_generics()`], [`Referrer::context_self()`] and
    /// [`Referrer::context_module()`]). Then `impl Repeater<(), ID, IX> for Marker { type Type = ..; }`
    /// is right for any marker, for any type, and in any module. It does not matter whether the
    /// entry is required.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .intern_trait(&parse_quote!(trait Tr<T> { fn f(&self, a: Rel, b: Wrap<T>) -> Self::A; }), true)
    ///     .unwrap();
    /// let referrer = leaker.finish();
    /// let plain: Vec<_> = referrer.iter().enumerate().filter(|(ix, _)| referrer.is_plain(*ix)).collect();
    /// // `Wrap<T>` uses a generic, and `Self::A` names `Self`.
    /// assert_eq!(plain.len(), 1);
    /// assert_eq!(plain[0].1, &parse_quote!(Rel));
    /// ```
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

    /// Returns the generics that the `ix`-th entry uses, as [`Referrer::context_generics()`] does.
    ///
    /// The Repeater impl encodes them in its trait arguments and declares
    /// [`Referrer::impl_generics()`]. See [Contexts](crate::rules#contexts).
    pub fn generics(&self, ix: usize) -> &Generics {
        self.context_generics(self.contexts[ix])
    }

    /// Returns the context of the `ix`-th entry.
    ///
    /// Entries with the same generics, `Self` and module share it. See [Contexts](crate::rules#contexts).
    pub fn context(&self, ix: usize) -> ContextId {
        self.contexts[ix]
    }

    /// Returns the contexts in ascending order, with the top-level ones that have no entry.
    ///
    /// The ids may have gaps. There is always at least one context. See
    /// [Contexts](crate::rules#contexts).
    pub fn contexts(&self) -> impl Iterator<Item = ContextId> + '_ {
        self.context_map.keys().copied()
    }

    /// Returns the generics that the entries of `context` use.
    ///
    /// The Repeater impl encodes them in its trait arguments and declares
    /// [`Referrer::impl_generics()`]. A top-level context has only the bounds the scope writes, so
    /// do not encode it for every entry (E0309): encode the generics of each entry.
    /// See [Outlives bounds](crate::rules#outlives-bounds).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_generics(&self, context: ContextId) -> &Generics {
        self.get_context(context).generics()
    }

    /// Returns the generics that a Repeater impl declares for the entries of `context`.
    ///
    /// [`Referrer::repeater_impl()`] describes the whole impl of an entry, with these.
    ///
    /// These are [`Referrer::context_generics()`] without the lifetime bounds that the trait
    /// arguments encode. Other bounds and where predicates stay. For other generics use
    /// [`without_encoded_bounds()`](crate::without_encoded_bounds).
    /// See [Outlives bounds](crate::rules#outlives-bounds).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote! {
    ///     trait Tr<'a, T: 'a + ::core::clone::Clone> { fn f(&self, x: &'a Wrap<T>); }
    /// };
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    /// let context = referrer.context(0);
    /// // The arguments of the Repeater encode `T: 'a` (as `&'a T`), so the impl must not declare it.
    /// // `Clone` stays.
    /// assert_eq!(referrer.context_generics(context).params, tr.generics.params);
    /// assert_eq!(referrer.impl_generics(context), parse_quote!(<'a, T: ::core::clone::Clone>));
    /// ```
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn impl_generics(&self, context: ContextId) -> Generics {
        without_encoded_bounds(self.context_generics(context))
    }

    /// Returns what `Self` means in `context`, if anything.
    ///
    /// For the context of an entry, it is `Some` only if the entry names `Self`. The Repeater impl
    /// is then for the implementors of the trait ([`SelfRequirement::Bound`]) or for the type
    /// ([`SelfRequirement::Type`]). If it is `None`, the impl is for any type.
    ///
    /// For a top-level context, it is what the scope declares, but only if an entry names `Self`,
    /// or has a bound of the scope's generics rewritten for it (a nested scope has a `Self` of its
    /// own): nothing else depends on it. It is `None` for a scope that declares nothing
    /// (`enter_generics()`, a module). So a top-level id tells the Referrer what `Self` is where
    /// it matters. See [Self rules](crate::rules#self-rules).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .intern_trait(&parse_quote!(trait Tr { fn f(&self) -> Wrap<Self>; fn g(&self) -> Rel; }), true)
    ///     .unwrap();
    /// let referrer = leaker.finish();
    /// // `Wrap<Self>` is written for the implementors of the trait. `Rel` is for any type.
    /// let wrap = referrer.iter().position(|ty| *ty == parse_quote!(Wrap<Self>)).unwrap();
    /// let rel = referrer.iter().position(|ty| *ty == parse_quote!(Rel)).unwrap();
    /// assert!(matches!(referrer.context_self(referrer.context(wrap)), Some(SelfRequirement::Bound(_))));
    /// assert!(referrer.context_self(referrer.context(rel)).is_none());
    /// ```
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_self(&self, context: ContextId) -> Option<&SelfRequirement> {
        self.get_context(context).self_ty()
    }

    /// Returns the lifetime parameters `'a` of the context for which its types need `Self: 'a`.
    ///
    /// They are sorted, and empty if the types do not name `Self`. [`Referrer::context_generics()`]
    /// does not list them. The Repeater impl must not declare them: encode them with the
    /// `*_with_self()` encoders. See [Outlives bounds](crate::rules#outlives-bounds).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_self_outlives(&self, context: ContextId) -> &[Lifetime] {
        self.get_context(context).self_outlives()
    }

    /// Returns the module where the types of `context` are written, as its `ContextId`.
    ///
    /// For an entry's context it is `Some` only for types with a relative path. Emit the Repeater
    /// impls of such types inside that module. For the top-level context of a scope entered in a
    /// module it is that module. It is `None` for a module context, which is the module itself.
    /// See [Modules](crate::rules#modules).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_module(&self, context: ContextId) -> Option<ContextId> {
        self.get_context(context).module()
    }

    /// Returns the path of `module` from its root: `[]` for the module of `intern_mod()`.
    ///
    /// A module found with no module around it is a root too, and its path starts with its own
    /// name. It is `None` for a module entered by `enter_mod()` and for an id that is not a module
    /// of this Referrer. See [Modules](crate::rules#modules).
    pub fn module_path(&self, module: ContextId) -> Option<&[Ident]> {
        self.modules.get(&module)?.path.as_deref()
    }

    /// Returns the hint given to [`Leaker::enter_mod()`](crate::Leaker::enter_mod) for `module`.
    ///
    /// It is `None` for other modules and for an id that is not a module of this Referrer.
    pub fn module_hint(&self, module: ContextId) -> Option<&str> {
        self.modules.get(&module)?.hint.as_deref()
    }

    /// Returns `true` if a leaked type, or the generics of an entry, names `Self`.
    ///
    /// `Self::Item` and `<Self as ::p::Tr>::X` count, `::p::Self` does not. A top-level context
    /// with no entry is not looked at. Then the Repeater impl is for what `Self` is
    /// ([`Referrer::context_self()`]), or use [`Referrer::iter_with_self()`] for a marker type.
    /// See [Self rules](crate::rules#self-rules).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&parse_quote!(trait Tr { fn f(&self) -> Wrap<Self::A>; }), true).unwrap();
    /// assert!(leaker.finish().has_self());
    ///
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&parse_quote!(trait Tr { fn f(&self) -> Wrap; }), true).unwrap();
    /// assert!(!leaker.finish().has_self());
    /// ```
    pub fn has_self(&self) -> bool {
        self.leak_types.iter().any(type_mentions_self)
            || self.contexts.iter().any(|context| {
                first_self_in_generics(self.context_map[context].generics()).is_some()
            })
    }

    /// Returns the entries with the generics they use, as [`Referrer::iter_with_generics()`] does,
    /// with `Self` replaced by `self_ty` (usually `__S`).
    ///
    /// [`Referrer::repeater_impl()`] does this and the rest of what a Repeater impl needs.
    ///
    /// Use it for an entry whose [`context_self()`](Referrer::context_self) is `Bound`, or for a
    /// marker type. `Self::A::B` becomes `<self_ty>::A::B`. Qualified paths and macro tokens are
    /// rewritten too. Only the returned copies change: `visitor()` still finds the types by
    /// their original tokens. See [Self rules](crate::rules#self-rules).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&parse_quote!(trait Tr { fn f(&self) -> Wrap<Self::A>; }), true).unwrap();
    /// let referrer = leaker.finish();
    /// // The impl is for a type parameter `__S`, where `Self` of the trait is not available.
    /// let implementor: Type = parse_quote!(__S);
    /// let (ty, _generics) = referrer.iter_with_self(&implementor).next().unwrap();
    /// assert_eq!(ty, parse_quote!(Wrap<<__S>::A>));
    /// ```
    pub fn iter_with_self<'a>(
        &'a self,
        self_ty: &'a Type,
    ) -> impl Iterator<Item = (Type, Generics)> + 'a {
        use syn::fold::Fold;
        self.iter_with_generics().map(move |(ty, generics)| {
            let mut folder = ReplaceSelf { self_ty };
            (
                folder.fold_type(ty.clone()),
                folder.fold_generics(generics.clone()),
            )
        })
    }

    /// Fills in the arguments that a copy omits because the trait gives them defaults.
    ///
    /// `params` are the parameters of the trait that the Leaker analyzed. The result has one
    /// argument for each parameter, matched as
    /// [`encode_generics_args_to_ty_with()`](crate::encode_generics_args_to_ty_with) does. Fails
    /// with [`EncodeError`] if there are too many arguments, or one without a default is missing.
    /// A default may name leaked types. `type_fn(leaked, index, args)` replaces each one, where
    /// `args` are the complete arguments before the defaulted parameter. Return a path through the
    /// Repeater, for example built with [`Referrer::generic_args()`]. It is not substituted again.
    /// A default is written in the trait's own scope, so only the entries of no module are looked
    /// up.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> U; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    /// // For `impl Tr<u8> for ..`, omitting `U`.
    /// let args: AngleBracketedGenericArguments = parse_quote!(<u8>);
    /// let completed = referrer
    ///     .complete_args(&args.args, &tr.generics.params, |_, ix, _| parse_quote!(Leaked<#ix>))
    ///     .unwrap();
    /// let expected: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(Leaked<0usize>)];
    /// assert_eq!(quote!(#(#completed),*).to_string(), quote!(#(#expected),*).to_string());
    /// ```
    pub fn complete_args<'a>(
        &self,
        args: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
        mut type_fn: impl FnMut(Type, usize, &[GenericArgument]) -> Type,
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
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
                let ty = self.expand_with(
                    Matching::Default,
                    ModulePos::Outside,
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

    /// Returns the arguments for the generics of the `ix`-th entry, as written in the copy.
    ///
    /// `completed_args` are the arguments of the Leaker's parameters `root_params`
    /// ([`Referrer::complete_args()`]). A generic named like one of them takes its argument. Any
    /// other generic is passed by its own name, so the copy must declare it. `Self` is not an
    /// argument. Fails with [`EncodeError::MissingArgument`] if `completed_args` are too few.
    ///
    /// To get the encoded trait argument of the Repeater, which also needs the lifetime bounds,
    /// `Self: 'a` and the consts, use [`Referrer::entry_args()`], which calls this.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T> { fn f<U>(&self, t: T, u: U) -> Wrap<T, U>; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    /// // `impl Tr<u8> for ..`: `T` is `u8`. `U` is a generic of the method, passed by its name.
    /// let trait_args: Vec<GenericArgument> = vec![parse_quote!(u8)];
    /// let args = referrer.generic_args(0, &tr.generics.params, &trait_args).unwrap();
    /// let expected: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(U)];
    /// assert_eq!(args, expected);
    /// ```
    pub fn generic_args(
        &self,
        ix: usize,
        root_params: &Punctuated<GenericParam, Token![,]>,
        completed_args: &[GenericArgument],
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        self.context_generic_args(self.contexts[ix], root_params, completed_args)
    }

    /// Like [`Referrer::generic_args()`], for the generics of any `context`.
    ///
    /// The arguments are matched with `root_params` by name, because the generics of a context are
    /// sorted. Use it with the top-level context for the arguments of the trait itself. Pass the
    /// `ContextId` along with the `Referrer` (written and parsed as an integer), because the
    /// Referrer cannot tell which context is the top-level one.
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_generic_args(
        &self,
        context: ContextId,
        root_params: &Punctuated<GenericParam, Token![,]>,
        completed_args: &[GenericArgument],
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
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

    /// Returns the types of the entries with the generics they use ([`Referrer::generics()`]).
    pub fn iter_with_generics(&self) -> impl Iterator<Item = (&Type, &Generics)> {
        self.leak_types
            .iter()
            .zip(&self.contexts)
            .map(move |(ty, context)| (ty, self.context_map[context].generics()))
    }

    /// Creates a [`Visitor`] that rewrites the leaked types in a copy of the analyzed item.
    ///
    /// `context` is the scope of the copy: the top-level [`ContextId`] that the `intern_*` or
    /// `enter_*` method of the Leaker returned for the item the copy is made from. Pass it along
    /// with the Referrer (it is written and parsed as an integer). The visitor takes from it the
    /// generics in scope, what `Self` means ([`Referrer::context_self()`]) and the module
    /// ([`Referrer::context_module()`], or `context` itself if it is a module). The visited syntax
    /// is inside that scope. See [Modules](crate::rules#modules).
    ///
    /// `type_fn` gets each leaked type and its index, and returns the replacement. That is usually
    /// the path `<Self as Repeater<ARGS, ID, IX>>::Type`, with `ARGS` from
    /// [`Referrer::entry_args()`].
    ///
    /// A type is replaced if its context, computed from the generics in scope, is that of an
    /// entry:
    ///
    /// - The generics of the scope and those the visited syntax declares are compared exactly, and
    ///   other names are not generics. So a generic that the visited syntax declares again, such as
    ///   the `T` of `impl<T: Clone> MyTrait<T> for ..`, must be declared as the entry has it,
    ///   unless you call [`Visitor::with_outermost_generics_as_leakers()`].
    /// - An entry which has, besides, bounds on `Self` alone (`where Self: Sized`) is found too, if
    ///   no entry has exactly the context: a copy where `Self` is a type parameter cannot write
    ///   them.
    /// - A struct, enum, union, trait or impl nested in the copy has a `Self` of its own, as in the
    ///   Leaker. A type there that uses a generic of `context` whose bound names `Self` has it
    ///   rewritten for the `Self` of `context` ([Self rules](crate::rules#self-rules)). So the copy must
    ///   nest the same items as the scopes of the Leaker. The entries that depend on the own `Self` of
    ///   such an item are found only if you call [`Visitor::with_nested_self()`], which writes them.
    ///
    /// Visit the whole item, not a type cut out of another.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; });
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    ///
    /// // The copy of `f` in `impl Tr<u8> for ..`.
    /// let trait_args: Vec<GenericArgument> = vec![parse_quote!(u8)];
    /// let site = UseSite::substituting(&tr.generics.params, &trait_args);
    /// let mut sig: Signature = parse_quote!(fn f(&self, t: T) -> Wrap<T>);
    /// referrer
    ///     .clone()
    ///     .visitor(top_level, |_ty, ix| {
    ///         let args = referrer.entry_args(ix, &site).unwrap();
    ///         parse_quote!(<Self as crate::Repeater<#args, 7u64, #ix>>::Type)
    ///     })
    ///     .visit_signature_mut(&mut sig);
    /// let expected: Signature = parse_quote!(
    ///     fn f(&self, t: T) -> <Self as crate::Repeater<(u8,), 7u64, 0usize>>::Type
    /// );
    /// assert_eq!(sig, expected);
    /// ```
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn visitor<F: FnMut(Type, usize) -> Type>(
        self,
        context: ContextId,
        type_fn: F,
    ) -> Visitor<F> {
        let (scope, self_requirement, module) = self.place(context);
        let in_module = self.get_context(context).is_module_scope();
        Visitor {
            referrer: self,
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
            defaults_without_self: false,
            module,
        }
    }

    /// Replaces the leaked types in one type with what `type_fn(leaked, index)` returns.
    ///
    /// `ty` is inside the scope `context`, as for [`Referrer::visitor()`]. Pass the whole type, not
    /// one cut out of a larger type: see [Outlives bounds](crate::rules#outlives-bounds). Only the
    /// generics of `context` are known, so a type that names a generic of a method is not found:
    /// use [`Referrer::visitor()`] on the signature.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: ::core::option::Option<Wrap<T>> });
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_struct(&item, true).unwrap();
    /// let referrer = leaker.finish();
    ///
    /// let ty: Type = parse_quote!(::core::option::Option<Wrap<T>>);
    /// let ty = referrer.expand(context, ty, |_, ix| parse_quote!(<Marker as Repeater<#ix>>::Type));
    /// let expected: Type = parse_quote!(::core::option::Option<<Marker as Repeater<0usize>>::Type>);
    /// assert_eq!(ty, expected);
    /// ```
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn expand(
        &self,
        context: ContextId,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        let (scope, self_requirement, module) = self.place(context);
        self.expand_with(
            Matching::Scope {
                self_ty: self_requirement.as_ref(),
                no_self: false,
            },
            module,
            &mut vec![scope],
            None,
            ty,
            type_fn,
        )
    }

    fn expand_with(
        &self,
        matching: Matching,
        module: ModulePos,
        scopes: &mut Vec<Scope>,
        self_ty: Option<&Type>,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        use syn::fold::Fold;
        struct Folder<'a, 'b, F> {
            referrer: &'a Referrer,
            matching: Matching<'b>,
            module: ModulePos,
            scopes: &'a mut Vec<Scope>,
            replace_self: Option<ReplaceSelf<'a>>,
            enclosing: Enclosing,
            type_fn: F,
        }
        impl<'a, 'b, F: FnMut(Type, usize) -> Type> Fold for Folder<'a, 'b, F> {
            fn fold_type(&mut self, ty: Type) -> Type {
                let enclosing = self.enclosing.clone();
                let found =
                    self.referrer
                        .find(self.matching, self.module, &ty, self.scopes, &enclosing);
                if found.is_none() {
                    self.enclosing = enclosing.components_of(&ty, self.scopes.len());
                }
                let folded = match (found, self.replace_self) {
                    (Some(ix), None) => (self.type_fn)(ty, ix),
                    (Some(ix), Some(mut replace)) => (self.type_fn)(replace.fold_type(ty), ix),
                    (None, None) => syn::fold::fold_type(self, ty),
                    (None, Some(replace)) => replace.fold_type_with(self, ty),
                };
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
            matching,
            module,
            scopes,
            replace_self: self_ty.map(|self_ty| ReplaceSelf { self_ty }),
            enclosing: Enclosing::default(),
            type_fn,
        }
        .fold_type(ty)
    }
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

/// A struct, enum, union, trait or impl nested in the visited syntax, which has a `Self` of its own.
///
/// [`Visitor::with_nested_self()`] gives it to the closure that writes the replacement of an entry
/// that depends on that `Self`. It is the item the visitor is in, and the same `Self` that the
/// Leaker gave the item when it analyzed it ([Self rules](crate::rules#self-rules)).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use syn::visit_mut::VisitMut;
/// let item: ItemMod = parse_quote! {
///     mod m { mod inner { struct S<T> { a: Rel<Self>, b: T } } }
/// };
/// let mut leaker = Leaker::new();
/// let module = leaker.intern_mod(&item).unwrap();
/// let referrer = leaker.finish();
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
    /// What `Self` is in the item, as [`Referrer::context_self()`] tells for the entries: the
    /// item itself with its own generics as arguments (`S<T>`, [`SelfRequirement::Type`]) for a
    /// struct, enum or union, the trait (`Tr<T>`, [`SelfRequirement::Bound`]) for a trait, and the
    /// self type as written (before the visitor replaces anything in it) for an impl.
    pub requirement: SelfRequirement,
    /// The name of the struct, enum, union or trait. It is `None` for an impl.
    pub ident: Option<Ident>,
    /// The `mod` items between the module of the context and the item, outermost first. They are
    /// those of the copy.
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

/// Replaces the leaked types in a copy of the analyzed item with what you write for them (usually
/// a path through the Repeater). Get it with [`Referrer::visitor()`].
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// use syn::visit_mut::VisitMut;
/// let item: ItemTrait = parse_quote!(trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; });
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_trait(&item, true).unwrap();
/// let referrer = leaker.finish();
///
/// // Rewrite the whole copy of the trait, with the `ContextId` that the Leaker returned for it.
/// let mut copy = item.clone();
/// referrer
///     .clone()
///     .visitor(top_level, |_leaked, ix| {
///         let args = referrer.entry_args(ix, &UseSite::identity()).unwrap();
///         parse_quote!(<Self as crate::Repeater<#args, 7u64, #ix>>::Type)
///     })
///     .visit_item_trait_mut(&mut copy);
/// let expected: ItemTrait = parse_quote! {
///     trait Tr<T> { fn f(&self, t: T) -> <Self as crate::Repeater<(T,), 7u64, 0usize>>::Type; }
/// };
/// assert_eq!(copy, expected);
/// ```
///
/// It implements `syn::visit_mut::VisitMut`. It tracks the generics in scope: those of the
/// context it was made for, then those the visited syntax declares, each with the `Self` in force
/// where it is declared. A type is replaced only if an entry has the context the type has there.
/// So `T: Clone` of a visited `impl<T: Clone>` does not match an entry with `T` alone, unless you
/// call [`Visitor::with_outermost_generics_as_leakers()`].
///
/// Visit a whole type or item. A type cut out of another is looked up with nothing around it
/// ([Outlives bounds](crate::rules#outlives-bounds)). The visited syntax is inside the module of the
/// context, so a `mod` item is a module nested in it, found by name, and the items of a module
/// are visited as items of it ([Modules](crate::rules#modules)). Items and imports declared in a block
/// hide the leaked types that name them there. Those declared in a module, or in a `File`, hide
/// the entries of no module that name them.
///
/// The visitor keeps the layers of the Leaker's scopes: the context first, then, for each struct,
/// enum, union, trait or impl nested in the visited syntax, a layer with the `Self` of that item
/// (it hides the one around). A nested item has no replacement for its own `Self` unless
/// [`Visitor::with_nested_self()`] says how to write it.
///
/// `G` is the closure of [`Visitor::with_nested_self()`]; none by default.
#[derive(Debug)]
pub struct Visitor<F, G = fn(&NestedSelf, Type, usize) -> Type> {
    referrer: Referrer,
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
    /// While the generics of a scope that this visit declares are visited: the defaults of their
    /// parameters have no `Self`, as in the Leaker (see `Visitor::visit_type_param_mut()`).
    defaults_without_self: bool,
    module: ModulePos,
}

impl<F, G> Clone for Visitor<F, G> {
    fn clone(&self) -> Self {
        Self {
            referrer: self.referrer.clone(),
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
            defaults_without_self: self.defaults_without_self,
            module: self.module,
        }
    }
}

impl<F, G> Visitor<F, G>
where
    F: FnMut(Type, usize) -> Type,
    G: FnMut(&NestedSelf, Type, usize) -> Type,
{
    /// Treats the generics of the outermost item visited as those of the context.
    ///
    /// Use it if the copy declares the generics of the scope again, with other bounds. A type that
    /// names them is replaced if it was leaked, whatever bounds they have in the copy.
    ///
    /// The outermost item is a struct, enum, union, trait, type alias, function, impl or trait
    /// alias, or the signature if only that is visited. Deeper generics stay declared. By default
    /// they are declared too, so they must match the entry's, except that an outermost item which
    /// declares exactly the generics of the context (the copy of the item itself) does not declare
    /// them again. Either way, `T::A<X>` is an associated type path, as in the Leaker.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// let tr: ItemTrait = parse_quote!(trait MyTrait<T> { fn f(&self) -> Wrap<T>; });
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    /// let replace = |_: Type, ix: usize| -> Type { parse_quote!(Leaked<#ix>) };
    ///
    /// // The copy declares `T` with another bound than the trait did.
    /// let item: ItemImpl = parse_quote!(impl<T: Clone> MyTrait<T> for W { fn f(&self) -> Wrap<T> { todo!() } });
    /// let mut copy = item.clone();
    /// referrer.clone().visitor(context, replace).visit_item_impl_mut(&mut copy);
    /// assert_eq!(copy, item); // `T: Clone` is not the generic of the trait: nothing is replaced.
    ///
    /// let mut copy = item.clone();
    /// referrer
    ///     .visitor(context, replace)
    ///     .with_outermost_generics_as_leakers()
    ///     .visit_item_impl_mut(&mut copy);
    /// let expected: ItemImpl = parse_quote!(impl<T: Clone> MyTrait<T> for W { fn f(&self) -> Leaked<0usize> { todo!() } });
    /// assert_eq!(copy, expected);
    /// ```
    pub fn with_outermost_generics_as_leakers(mut self) -> Self {
        self.outermost_as_leakers = true;
        self
    }

    /// Replaces `Self` by `self_ty` in the copy, as [`Referrer::iter_with_self()`] does.
    ///
    /// Use it for a Repeater implemented for a marker type ([Self rules](crate::rules#self-rules)).
    /// Leaked types are still found by their original tokens, and `type_fn` gets them with `Self`
    /// replaced. `Self` is replaced in the other types too, but not in what `type_fn` returns, nor
    /// in receivers (`&self`) or expressions. In a nested struct, enum, union, trait or impl, `Self`
    /// means something else: it is not replaced. A leaked type that names it is found only with
    /// [`Visitor::with_nested_self()`]. The outermost item, or the signature if only that is
    /// visited, is not nested.
    ///
    /// A `self_ty` with lifetimes (`::p::S<'a>`) gives more input lifetimes to a function pointer
    /// or `Fn(..)` that names `Self` in its inputs and has an elided lifetime in its output
    /// (E0106). The visitor names the lifetime that elision gives, with a fresh binder
    /// (`for<'__tl_e0> fn(&'__tl_e0 ::p::S<'a>) -> &'__tl_e0 u8` for `fn(&Self) -> &u8`), in a
    /// type, a bound and a where predicate. A lifetime that a path hides cannot be named: for the
    /// `Self` of a struct, an enum or a union, the Leaker leaks such a signature whole or rejects
    /// the item. See [Elided lifetimes](crate::rules#elided-lifetimes).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self, x: Rel<Self>) -> Self::Out; });
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_trait(&tr, true).unwrap();
    /// let mut sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>) -> Self::Out);
    /// leaker
    ///     .finish()
    ///     .visitor(context, |_, ix| parse_quote!(<Marker<__S> as Repeater<#ix>>::Type))
    ///     .with_self(parse_quote!(__S))
    ///     .visit_signature_mut(&mut sig);
    /// let expected: Signature =
    ///     parse_quote!(fn f(&self, x: <Marker<__S> as Repeater<0usize>>::Type) -> <__S>::Out);
    /// assert_eq!(sig, expected);
    /// ```
    pub fn with_self(mut self, self_ty: Type) -> Self {
        self.self_ty = Some(self_ty);
        self
    }

    /// Writes the entries that depend on the own `Self` of a nested item with `nested_fn`.
    ///
    /// A struct, enum, union, trait or impl nested in the visited syntax (an item of a module or of
    /// a `File`, or one in a function body) has a `Self` of its own. The Leaker records the entries
    /// that depend on it: those that name it (`Rel2<Self>` in `struct S<T> { c: Vec<Rel2<Self>> }`),
    /// and those that use a generic with a bound that names it (`Rel<T>` with `T: ::p::Tr<Self>`).
    /// The visitor keeps a layer for the item with that `Self`, as the Leaker has it, so it
    /// computes the same contexts and finds those entries ([Self
    /// rules](crate::rules#self-rules)).
    ///
    /// It calls `nested_fn(nested, leaked, ix)` for them instead of `type_fn`, because only the macro
    /// knows what to write for the `Self` of the item. The Repeater impl of such an entry is for the
    /// Leaker's own type (`impl<T> Repeater<(T,), ID, 3> for S<T>`, in the module of the Leaker's
    /// crate), which the copy cannot name as `Self`: it declares a `S` of its own. `nested` is the
    /// item ([`NestedSelf`]): from its name, its [`modules`](NestedSelf::modules) and the path of
    /// the module of the context in the Leaker's crate, the macro writes the type. The usual result
    /// is `<#self_ty as Repeater<#args, ID, #ix>>::Type`, with `args` from
    /// [`Referrer::entry_args()`] at a [`UseSite`](crate::UseSite) that has
    /// [`with_self(self_ty)`](crate::UseSite::with_self), so that a `Self: 'a` of the entry is
    /// encoded as the impl encodes it. For a nested trait, `Self` is a type that implements the
    /// Leaker's trait (a host in the copy). `leaked` is the type of the entry, as it is in the
    /// item, with its `Self`.
    ///
    /// Only the layer of the item the visitor is in counts: an entry of the `Self` of the context, or
    /// of the scope around that the nested item hides, is not found there. The entries that do not
    /// depend on a `Self`, and all the entries outside nested items, go to `type_fn`.
    ///
    /// Without this, the visitor does not find the entries that depend on the own `Self` of a
    /// nested item, whatever `type_fn` is. The copy keeps their types as written (`Rel2<Self>`), which
    /// its crate cannot resolve. The `Self` of an item that the visitor is not in a nested position
    /// of (the outermost item visited, or the signature if only that is visited) is the `Self` of
    /// the context, and `type_fn` writes it.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// // In the module `m` of the crate `leaker`.
    /// let item: ItemStruct = parse_quote!(struct S<T> { a: Rel<T>, b: Rel2<Self>, });
    /// let mut leaker = Leaker::new();
    /// let (module, ()) = leaker
    ///     .enter_mod("m", [item.ident.clone()], |view| {
    ///         view.visit_item(&Item::Struct(item.clone()))
    ///     })
    ///     .unwrap();
    /// let referrer = leaker.finish();
    ///
    /// // The copy is in another crate. `Self` of the Leaker's `S<T>` is `::leaker::m::S<T>` there.
    /// let mut file = File { shebang: None, attrs: vec![], items: vec![Item::Struct(item)] };
    /// referrer
    ///     .clone()
    ///     .visitor(module, |_, ix| parse_quote!(<() as Repeater<#ix>>::Type))
    ///     .with_nested_self(|nested, _leaked, ix| {
    ///         let ident = nested.ident.as_ref().unwrap();
    ///         parse_quote!(<::leaker::m::#ident<T> as Repeater<#ix>>::Type)
    ///     })
    ///     .visit_file_mut(&mut file);
    /// let expected: File = parse_quote! {
    ///     struct S<T> {
    ///         a: <() as Repeater<0usize>>::Type,
    ///         b: <::leaker::m::S<T> as Repeater<1usize>>::Type,
    ///     }
    /// };
    /// assert_eq!(file, expected);
    /// ```
    pub fn with_nested_self<N: FnMut(&NestedSelf, Type, usize) -> Type>(
        self,
        nested_fn: N,
    ) -> Visitor<F, N> {
        Visitor {
            referrer: self.referrer,
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
            defaults_without_self: self.defaults_without_self,
            module: self.module,
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
        let context_map = &self.referrer.context_map;
        let contexts = &self.referrer.contexts;
        self.referrer.map.retain(|key, entries| {
            if mentions_names(&key.ty, names.clone(), Vec::new()) {
                entries.retain(|&ix| module && context_map[&contexts[ix]].module().is_some());
            }
            !entries.is_empty()
        });
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
        if self.item_depth > 0 {
            if let Some(declared) = declares {
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
        visitor.defaults_without_self = declared;
        visitor.visit_generics_mut(generics);
        visitor.defaults_without_self = false;
        visitor
    }

    fn enter_signature(
        &mut self,
        attrs: &mut [Attribute],
        sig: &mut Signature,
        item: bool,
    ) -> Self {
        let mut visitor = self.enter(attrs, &mut sig.generics, None, item);
        for input in sig.inputs.iter_mut() {
            visitor.visit_fn_arg_mut(input);
        }
        visitor.visit_return_type_mut(&mut sig.output);
        visitor
    }
}

impl<F, G> VisitMut for Visitor<F, G>
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
        *i = referrer.expand_with(
            Matching::Scope { self_ty, no_self },
            self.module,
            &mut self.scopes,
            self.self_ty.as_ref().filter(|_| !own_self),
            i.clone(),
            replace,
        );
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
        syn::visit_mut::visit_predicate_type_mut(self, i);
        self.scopes.truncate(depth);
        // What replaced the bounded type (`Self`, an entry) may need parentheses there.
        parenthesize_bounded(&mut i.bounded_ty);
    }
    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        // A bound of the generics is not in a type, which the same rewriting handles.
        if let Some(self_ty) = self.self_ty.as_ref().filter(|_| !self.own_self) {
            name_elided_in_bound(self_ty, &self.scopes[..], i);
        }
        let depth = self.scopes.len();
        self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
        syn::visit_mut::visit_trait_bound_mut(self, i);
        self.scopes.truncate(depth);
    }
    fn visit_type_param_mut(&mut self, i: &mut TypeParam) {
        if !self.defaults_without_self {
            return syn::visit_mut::visit_type_param_mut(self, i);
        }
        // The Leaker analyzes the default of a parameter of a nested scope without `Self`: it
        // cannot depend on it, so the bounds of the scopes around that mention `Self` are
        // rewritten for the `Self` they were written for, also where the nested scope has the same
        // one. The scope itself is declared without `Self` for its default.
        let default = i.default.take();
        syn::visit_mut::visit_type_param_mut(self, i);
        i.default = default;
        if let Some(default) = &mut i.default {
            let self_requirement = self.self_requirement.take();
            let own = self.scopes.last_mut().map(|scope| scope.replace_self(None));
            self.visit_type_mut(default);
            if let (Some(scope), Some(own)) = (self.scopes.last_mut(), own) {
                scope.replace_self(own);
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
        self.enter_signature(&mut [], i, true);
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
            visitor.module = self.referrer.module_in(self.module, &i.ident);
            visitor.mod_path.push(i.ident.clone());
            // A module has no generics: it only makes its items nested. `Self` has no meaning in it.
            visitor.item_depth += 1;
            visitor.self_requirement = None;
            visitor.nested = None;
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
    /// Parsing rejects another tag, or none. Formats are not compatible, so both sides must use the
    /// same version of `type-leak`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// use template_quote::ToTokens;
    /// let referrer = Leaker::new().finish();
    /// let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    /// assert!(referrer.to_token_stream().to_string().starts_with(&tag));
    ///
    /// // A Referrer written by another version of `type-leak` is refused with an error that says so.
    /// let error = parse_str::<Referrer>("\"type-leak-format-0\"").unwrap_err();
    /// assert!(error.to_string().contains("incompatible type-leak"));
    /// ```
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
    // Referrer computes it for a copy.
    pub const FORMAT_VERSION: u32 = 8;
}

/// Parses a [`Referrer`] written by its [`ToTokens`] implementation.
///
/// Entries keep their indices, and the tokens after the Referrer are not consumed. Fails if the
/// format tag is missing or differs ([`Referrer::FORMAT_VERSION`]), if a module is used but not
/// declared, or if module parents form a cycle.
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
                if input.parse::<Option<Token![:]>>()?.is_none() {
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
        Ok(Self::new(entries, context_map, modules))
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

/// `r#name` if `name` is a keyword, so that it parses back as an identifier.
fn crate_ident(name: &str) -> Ident {
    if parse_str::<Ident>(name).is_ok() {
        Ident::new(name, Span::call_site())
    } else {
        Ident::new_raw(name, Span::call_site())
    }
}

/// Writes the [`Referrer`] as tokens, to be read back by its [`Parse`] implementation.
///
/// The tokens start with the format tag ([`Referrer::FORMAT_VERSION`]). Treat the rest as opaque.
impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        // Grammar: tag [count] then one group per context in ascending id order (at least one):
        // [# id] [<generics> where .. ;] [Self: path ; | Self = Type ;] [Self: 'a + 'b ;]
        // [type G<'b> where Self: 'b ; ..]
        // [mod 3 ; | mod { path = [a], hint = "h", parent = 1 } ;] ( [?] [@[crate, ..]] Type, .. )
        // `# id` only after a gap in the ids. `()` if no entry. `?`: optional. `@[..]`: crates named.
        let tag = format!("{}{}", FORMAT_TAG_PREFIX, Self::FORMAT_VERSION);
        let n_contexts = self.context_map.len();
        tokens.extend(quote! {
            #tag
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
            tokens.extend(quote! {
                (
                    #(for ix in entries), {
                        #(if !self.musts[ix]) { ? }
                        #(if !self.external_crates[ix].is_empty()) {
                            @[#(for name in &self.external_crates[ix]), { #{crate_ident(name)} }]
                        }
                        #{&self.leak_types[ix]}
                    }
                )
            });
            next = context.index() + 1;
        }
    }
}

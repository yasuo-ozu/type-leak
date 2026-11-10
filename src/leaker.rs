//! The [`Leaker`], which finds the types depending on the type context.

use crate::allow::Allow;
use crate::analysis::{drop_optional, names_self, Analysis, PartItem};
use crate::analyze::{declared_names, AnalyzeVisitor, Finished, ModuleNames, ScopeSpec, Scopes};
use crate::config::LeakerConfig;
use crate::context::{Context, ContextId, Contexts, Gat, SelfRequirement};
#[cfg(doc)]
use crate::error::Reason;
use crate::error::{Cause, ErrorItem};
use crate::graph::{is_whole_reference, NodeKey, TypeGraph};
use crate::referrer::Referrer;
use crate::util::{ident_name, AllowedPaths};
use crate::view::View;
use proc_macro2::{Span, TokenStream};
use safegraph::algo::max_flow::edmonds_karp_with_flows;
use safegraph::graph::Graph;
use safegraph::BTreeGraph;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Display;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::result::Result;
use syn::parse::{Parse, ParseStream};
use syn::visit::Visit;
use syn::*;
use template_quote::{quote, ToTokens};

/// Finds the types of an item that depend on where they are written, and collects them in a
/// [`Referrer`].
///
/// Analyze a whole item with an `intern_*` method, or parts of one with an `enter_*` method, then
/// call [`Leaker::result()`]. Keep the returned [`ContextId`] for [`Referrer::visitor()`]. See the
/// [quick start](crate#quick-start).
///
/// ```
/// # use {syn::*, type_leak::*};
/// let item: ItemTrait = parse_quote! {
///     trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self) -> Rel; }
/// };
/// let mut leaker = Leaker::new();
/// let top_level = leaker.intern_trait(&item, true);
/// let referrer = leaker.result().finish().unwrap();
/// // `Wrap<T>` and `Rel` depend on where they are written, so each needs a Repeater impl.
/// assert_eq!(referrer.required().count(), 2);
/// assert_eq!(referrer.context_generics(top_level).params.len(), 1);
/// ```
///
/// # Errors and recovery
///
/// An error does not stop the analysis: the part that cannot be leaked is copied as written, and
/// the error is kept (once, in the order found). A panic in a closure rolls its scope back and
/// records nothing. The `allow*()` methods of [`LeakerConfig`] and [`LeakerResult`] drop the
/// errors you accept.
///
/// # Limitations
///
/// The Leaker reads only the item you give it, not the definitions of the paths it names, and
/// rejects what would not compile in the copy; the `help` of an error says what to write instead.
///
/// - `MyType<'_>`, `impl Trait`, and a relative trait or value in a bound: write an absolute path
///   (`Self: Sized` needs [`allow_prelude()`](LeakerConfig::allow_prelude)).
/// - An absolute path of another crate is copied as written, which assumes that both crates
///   resolve it alike ([`LeakerResult::allow_ambiguous_crate_name()`]).
/// - A shadowed primitive name (`type u8 = i32;`): emit [`Leaker::assertions()`].
/// - Values and bodies are not analyzed unless [`LeakerConfig::check_values()`] is on.
///
/// ## Outlives bounds
///
/// An entry under `&'a` needs `T: 'a` in its Repeater impl (E0309), and the Leaker records only the
/// bounds you write: write them (`fn f<'a, T: 'a>(x: Rel<'a, T>)`) and give [`Referrer::visitor()`]
/// the whole item. For `Self: 'a`, see [`Referrer::context_self_outlives()`].
///
/// ## Binder lifetimes
///
/// If the output of a `for<'x> ..` signature names `'x`, an input must keep `'x` as written (E0581,
/// E0582), so the whole function pointer or trait object is leaked, which is an error in a bound.
/// Keep `'x` in an input that is not leaked: `for<'x> Fn(&'x u8, Rel<'x>) -> &'x u8`.
///
/// ## Elided lifetimes
///
/// An elided output lifetime comes from the only input lifetime, and a leaked type adds more
/// (E0106), so the whole function pointer or trait object is leaked: an error in a bound, in
/// `impl Fn(..)`, under an elided reference and in a method without a reference receiver. Write the
/// lifetimes (`fn g<'x>(x: &'x Rel<T>) -> &'x u8`, `Ref<'_, u8>`) or box the trait object.
///
/// ## Trait objects
///
/// A `dyn Trait` without a lifetime bound takes its bound from where it is written, which the
/// Leaker cannot see (E0478), so the reference or path around it is leaked instead. That is an
/// error under an elided lifetime (`&dyn Tr`): write `&'a dyn Tr`.
///
/// ## Method lifetimes
///
/// A method lifetime `'b` that only a leaked type mentions is late-bound in a trait but early-bound
/// in the copy (E0195), if the return type names `'b`: keep `'b` in an input that is not leaked
/// (`fn f<'b>(&'b self, x: Wrap<'b>) -> &'b u8`). This affects methods of traits and trait impls.
///
/// ## Bounds that name leaked types
///
/// A leaked type in a bound or where predicate fails (E0277, E0271, E0275, E0284) if its Repeater
/// impl needs a bound that the environment has only in rewritten form, or the bound names `Self`,
/// `Self::A` or a normalized associated type, or projects a generic bounded by a non-standard
/// trait. If another bound of `T` fixes an associated type (`T: Iterator<Item = u8>`), a bound that
/// names `T` (`T: Fn(Wrap<T>)`) is left out of the entries' context: E0277 if the type needs it.
///
/// ## Bare identifiers and consts
///
/// A bare identifier argument (`MyType<MY_CONST>`) may be a type or a const. The Leaker leaks the
/// type that holds it as a whole, right for both; where it cannot (an elided lifetime or reference
/// in the holder, a trait bound), it leaks the identifier alone, which fails for a const (E0573,
/// E0747). Write an absolute path (`MyType<{ ::my_crate::MY_CONST }>`) or an
/// [allowed](LeakerConfig::allow_path) name.
#[derive(Default)]
pub struct Leaker {
    /// What the configuration allows: the paths that are left as written instead of being leaked,
    /// and the errors that are not recorded.
    allow: Allow,
    /// The types that depend on the context (the required entries).
    types: TypeGraph,
    /// The types the Referrer may replace (the optional entries).
    optional_types: TypeGraph,
    /// The result of [`Leaker::reduce_roots()`]: required, then optional. `None` if outdated.
    reachable_types: Option<(Vec<NodeKey>, Vec<NodeKey>)>,
    contexts: Contexts,
    /// Numbered on their own, so that optional entries do not change the ids of required ones.
    optional_contexts: Contexts,
    /// The top-level contexts of the scopes (also of those with errors), which the [`Referrer`]
    /// keeps without an entry too.
    top_levels: BTreeSet<ContextId>,
    assumed_primitives: Vec<Ident>,
    /// Whether the values of an item are read ([`LeakerConfig::check_values()`]).
    check_values: bool,
    /// Whether an entry that names the `Self` of a nested struct, enum or union together with a
    /// bound that does is accepted ([`LeakerConfig::nested_self_entries()`]).
    nested_self_entries: bool,
    /// The parts copied as written.
    parts: Vec<PartItem>,
    /// The errors of the outermost calls, in the order found.
    errors: Vec<ErrorItem>,
}

impl Leaker {
    /// Creates a [`Leaker`] that allows nothing, so every relative path is leaked.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`Leaker`] with the rules of `config`.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allow: config.allow,
            check_values: config.check_values,
            nested_self_entries: config.nested_self_entries,
            ..Self::default()
        }
    }

    /// Returns a compile-time check that the primitive names copied as written (`u8` in `const N:
    /// u8`, or by [`LeakerConfig::allow_primitive()`]) are not shadowed. Emit it in the Leaker's
    /// crate. It is empty if none was copied.
    pub fn assertions(&self) -> TokenStream {
        if self.assumed_primitives.is_empty() {
            return TokenStream::new();
        }
        quote! {
            #[allow(dead_code, non_snake_case)]
            const _: () = {
                #(for ident in &self.assumed_primitives) {
                    fn #{Ident::new(
                        &format!("__type_leak_assert_{}_is_primitive", ident),
                        Span::call_site(),
                    )}(x: &#ident) -> &::core::primitive::#ident { x }
                }
            };
        }
    }

    /// Analyzes a struct and returns its top-level context. `Self` is the struct
    /// ([`SelfRequirement::Type`]).
    ///
    /// `allow_self = true` leaks a type that names `Self` as one entry for the struct (`Self::A`
    /// cannot be leaked whole, E0220: write `<Self as ::path::Trait>::A`). `false` splits it into
    /// parts, which is an error for `Rel<Self>`: use it if the Repeater impls are for a marker type
    /// and [`Visitor::with_self()`](crate::Visitor::with_self) replaces `Self` in the copies.
    pub fn intern_struct(&mut self, item: &ItemStruct, allow_self: bool) -> ContextId {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_struct(view, item)
        })
        .0
    }

    /// Analyzes an enum like [`intern_struct()`](Leaker::intern_struct).
    pub fn intern_enum(&mut self, item: &ItemEnum, allow_self: bool) -> ContextId {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_enum(view, item)
        })
        .0
    }

    /// Analyzes a union like [`intern_struct()`](Leaker::intern_struct).
    pub fn intern_union(&mut self, item: &ItemUnion, allow_self: bool) -> ContextId {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_union(view, item)
        })
        .0
    }

    /// Analyzes an impl and returns its top-level context. `Self` is the self type
    /// ([`SelfRequirement::Type`]), and `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct).
    ///
    /// An entry in the self type that depends on `Self` through a bound of the impl (`impl<T> S<T>
    /// where T: AsRef<Option<Self>>`) is an error, also in an impl that [`View::visit_item()`] or
    /// [`intern_mod()`](Leaker::intern_mod) visits: use an absolute self type or a bound without
    /// `Self`. The trait path must not depend on the context.
    pub fn intern_impl(&mut self, item: &ItemImpl, allow_self: bool) -> ContextId {
        // The scope checks the path of the trait, and the walk analyzes its arguments.
        let trait_path = item.trait_.as_ref().map(|(_, path, _)| path);
        self.enter_impl_of(
            &item.generics,
            &item.self_ty,
            trait_path,
            allow_self,
            |view| view.walk_item_impl(item),
        )
        .0
    }

    /// Analyzes a trait and returns its top-level context. `Self` is the implementor
    /// ([`SelfRequirement::Bound`]), and `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct).
    ///
    /// Unlike [`enter_trait()`](Leaker::enter_trait), it knows the generic associated types (GATs)
    /// of the trait: `Self::G<'x>` records the outlives bounds of the GAT, which the copy must
    /// prove, so in an impl for another type add `where T: 'x` or `where Self: 'x` (unless the
    /// receiver is `&'x self`) to the method. GATs of supertraits or of `T::G<'x>` are not seen.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemTrait = parse_quote!(trait Tr { type Out; fn g(&self) -> Wrap<Self::Out>; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
    /// let context = referrer.context(0);
    /// assert!(matches!(referrer.context_self(context), Some(SelfRequirement::Bound(_))));
    /// ```
    pub fn intern_trait(&mut self, item: &ItemTrait, allow_self: bool) -> ContextId {
        // The scope knows the generic associated types of the trait: what they require is part of
        // the context of an entry that names them.
        let gats = Gat::of_items(&item.items);
        self.enter_trait_of(&item.ident, &item.generics, gats, allow_self, |view| {
            syn::visit::visit_item_trait(view, item)
        })
        .0
    }

    /// Analyzes a module and returns its [`ContextId`].
    ///
    /// Every item in it is nested: its generics are local, and `Self` has no meaning. The names
    /// declared in the module are copied with it, not leaked. Two calls give two modules: rewrite
    /// the copy of each with its own id.
    pub fn intern_mod(&mut self, item: &ItemMod) -> ContextId {
        let names = match &item.content {
            Some((_, items)) => declared_names(items),
            None => ModuleNames::default(),
        };
        let spec = ScopeSpec::Module {
            names,
            path: Some(Vec::new()),
            hint: None,
        };
        self.enter_spec(spec, |view| syn::visit::visit_item_mod(view, item))
            .0
    }

    /// Analyzes part of a module: runs `f` on a [`View`] inside it, and returns the [`ContextId`]
    /// of the module and the result of `f`.
    ///
    /// `declared` lists the names the module declares, which are copied with it, not leaked. Give
    /// them first, because a relative path is decided when it is visited. A name counts if it is
    /// declared directly (unless a generic shadows it), named by `self::` or `super::` within the
    /// module, or imported with an absolute path (`use super::X` and `use crate::X` are leaked).
    /// `hint` is for debugging ([`Referrer::module_hint()`]).
    pub fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_mod(self, hint, declared, f)
    }

    /// Analyzes types that use `generics`: runs `f` on a [`View`] in which they are declared, and
    /// returns the top-level [`ContextId`] and the result of `f`. The generics themselves are not
    /// analyzed: visit their bounds in `f` with [`View::visit_type_param_bound()`] and
    /// [`View::visit_where_clause()`]. Naming `Self` is an error ([`Reason::SelfWithoutMeaning`]).
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_generics(self, generics, f)
    }

    /// Analyzes parts of a struct, an enum or a union: runs `f` on a [`View`] inside the type
    /// `ident` with `generics`. Like [`enter_generics()`](Leaker::enter_generics), but `Self` is
    /// the type ([`SelfRequirement::Type`]) and `generics` may name it; `allow_self` is as for
    /// [`intern_struct()`](Leaker::intern_struct).
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_type(self, ident, generics, allow_self, f)
    }

    /// Analyzes parts of an impl: runs `f` on a [`View`] inside an impl with `generics` for
    /// `self_ty`. Like [`enter_type()`](Leaker::enter_type), with `self_ty` as `Self`.
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_impl(self, generics, self_ty, allow_self, f)
    }

    /// Analyzes parts of a trait: runs `f` on a [`View`] inside the trait `ident` with `generics`.
    /// Like [`enter_type()`](Leaker::enter_type), but `Self` is the implementor
    /// ([`SelfRequirement::Bound`]), and the GATs of the trait are not known: see
    /// [`intern_trait()`](Leaker::intern_trait).
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_trait(self, ident, generics, allow_self, f)
    }

    /// Ends the analysis and returns the [`LeakerResult`]: the [`Referrer`] and the errors found.
    /// Required entries come first, then optional ones ([`Referrer::must()`]), each sorted by
    /// context (in order of first appearance) and tokens.
    pub fn result(mut self) -> LeakerResult {
        let errors = std::mem::take(&mut self.errors);
        LeakerResult {
            referrer: self.referrer(),
            errors,
        }
    }

    fn referrer(mut self) -> Referrer {
        self.reduce_roots();
        let (required, optional) = self.reachable_types.take().unwrap_or_default();
        let contexts = self.contexts;
        let context_of = |id: ContextId| contexts.get(id).clone();
        let mut context_map: BTreeMap<ContextId, Context> = self
            .top_levels
            .iter()
            .map(|&id| (id, context_of(id)))
            .collect();
        // The parts are looked up by their contexts too.
        for item in &self.parts {
            let anchor = item.within.iter().map(|(_, id)| *id);
            for id in item.context.into_iter().chain(anchor) {
                context_map.entry(id).or_insert_with(|| context_of(id));
            }
        }
        let mut entries = Vec::new();
        let keys = required.into_iter().map(|key| (key, true));
        for (key, must) in keys.chain(optional.into_iter().map(|key| (key, false))) {
            let context = context_map
                .entry(key.context)
                .or_insert_with(|| context_of(key.context));
            let crates = external_crates(&key.ty.ty, context);
            entries.push((key.context, key.ty.ty, must, crates));
        }
        // Keep the modules the contexts depend on, and their parents, for nested lookups.
        let mut modules = BTreeMap::new();
        let mut pending: Vec<ContextId> = Vec::new();
        for (id, context) in &context_map {
            if context.is_module_scope() {
                pending.push(*id);
            }
            pending.extend(context.module());
        }
        while let Some(module) = pending.pop() {
            if let Some(info) = contexts.module(module) {
                if modules.insert(module, info.clone()).is_none() {
                    context_map
                        .entry(module)
                        .or_insert_with(|| context_of(module));
                    pending.extend(info.parent);
                }
            }
        }
        Referrer::new(entries, context_map, modules, self.parts)
    }

    /// Chooses the fewest entries from which every leaked type can be built;
    /// [`Leaker::result()`] calls it. `&'a Rel`, a path with a lifetime or bare-identifier argument
    /// (`Wrap<SIZE>`), a type with a type in an expression (`[u8; size_of::<Big>()]`) and a trait
    /// object are entries as a whole.
    pub fn reduce_roots(&mut self) {
        if self.reachable_types.is_some() {
            return;
        }
        let required = reduce(&self.types, &HashSet::new());
        // The required entries are cut for free in the graph of the optional ones.
        let blocked: HashSet<NodeKey> = required
            .iter()
            .filter_map(|key| {
                let context = self
                    .optional_contexts
                    .find(self.contexts.get(key.context))?;
                Some(NodeKey {
                    context,
                    ty: key.ty.clone(),
                })
            })
            .collect();
        let mut optional: Vec<NodeKey> = reduce(&self.optional_types, &blocked)
            .into_iter()
            .map(|key| NodeKey {
                context: self
                    .contexts
                    .intern(self.optional_contexts.get(key.context).clone()),
                ty: key.ty,
            })
            .collect();
        optional.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
        self.reachable_types = Some((required, optional));
    }
}

/// What a [`Leaker`] found: the [`Referrer`] and the errors that remain.
///
/// The `allow*` methods drop the errors you accept, by the rules of the same methods of
/// [`LeakerConfig`], and never change the Referrer: you promise that the part compiles as written.
/// A path rule here only approximates the config's (more errors may remain, and a copy may fail to
/// build): give path rules to the config when you can. The [nested errors](ErrorItem::nested) of a
/// dropped error come up in its place, so allow until [`errors()`](LeakerResult::errors) is empty.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let item: ItemStruct = parse_quote!(struct S { a: Rel, b: ::p::W<'_, dyn RelTr> });
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&item, true);
/// let mut result = leaker.result();
/// assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));
/// // `Rel` is an entry as usual, and `dyn RelTr` is copied as written.
/// let referrer = result.clone().into_referrer();
/// assert_eq!(referrer.iter().count(), 1);
/// // Allowing the error leaves the Referrer as it is, and `finish()` returns it.
/// result.allow(Cause::Trait(parse_quote!(RelTr)));
/// assert_eq!(result.finish().unwrap(), referrer);
/// ```
#[derive(Clone, Debug)]
pub struct LeakerResult {
    referrer: Referrer,
    errors: Vec<ErrorItem>,
}

impl LeakerResult {
    /// Drops the errors that the rules in `rules` accept.
    fn drop_accepted(&mut self, rules: impl FnOnce(&mut Allow)) -> &mut Self {
        let mut allow = Allow::default();
        rules(&mut allow);
        allow.filter(&mut self.errors);
        self
    }

    /// Drops the remaining errors whose cause equals `cause` (the same kind and tokens, spans
    /// ignored).
    ///
    /// You promise that the part can be written as it is in the copy. The cause may be smaller
    /// than the part that is copied: in `[u8; { let c = |a: u8| a; 3 }]` the error is about `u8`,
    /// and the array is copied whole. Allowing a [`Reason::UnprovableBound`] does not make the copy
    /// compile (E0277).
    pub fn allow(&mut self, cause: Cause) -> &mut Self {
        self.drop_accepted(|allow| allow.cause(cause))
    }

    /// Drops the errors whose relative paths all start with `path`, such as `my_mod::Tr`. An
    /// absolute path is never an error, so `::serde` changes nothing here.
    pub fn allow_path(&mut self, path: Path) -> &mut Self {
        self.drop_accepted(|allow| allow.path(path))
    }

    /// Drops the errors whose relative paths are all primitive type names.
    pub fn allow_primitive(&mut self) -> &mut Self {
        self.drop_accepted(Allow::primitive)
    }

    /// Drops the errors whose relative paths are all names of the standard prelude (`Clone`).
    pub fn allow_prelude(&mut self) -> &mut Self {
        self.drop_accepted(Allow::prelude)
    }

    /// Drops the errors whose relative paths all start with `std`, `core` or `alloc`.
    pub fn allow_std_crates(&mut self) -> &mut Self {
        self.drop_accepted(Allow::std_crates)
    }

    /// Drops the errors whose relative paths all start with `crate`.
    pub fn allow_crate(&mut self) -> &mut Self {
        self.drop_accepted(Allow::crate_root)
    }

    /// Drops the errors found in a value (see [`ErrorItem::in_value`]).
    pub fn allow_values(&mut self) -> &mut Self {
        self.drop_accepted(Allow::values)
    }

    /// Drops the errors found in a default (see [`ErrorItem::in_default`]).
    pub fn allow_default(&mut self) -> &mut Self {
        self.drop_accepted(Allow::defaults)
    }

    /// Chooses whether optional entries (absolute paths of other crates) are left as written
    /// (`true`, the default) or replaced as a whole through the Repeater (`false`, a strict
    /// Referrer).
    ///
    /// Crates are matched by name only: use `false` if the Referrer's crate may have other
    /// dependencies than the Leaker's. The types inside an optional entry are replaced either way.
    /// The flag is not serialized: the Leaker's side writes the impls of all entries
    /// ([`Referrer::iter()`]), and the macro that receives the result chooses.
    pub fn allow_ambiguous_crate_name(&mut self, allow: bool) -> &mut Self {
        self.referrer.allow_ambiguous_crate_name = allow;
        self
    }

    /// Returns the errors that remain, in the order found.
    pub fn errors(&self) -> &[ErrorItem] {
        &self.errors
    }

    /// Returns the Referrer, whatever errors remain: a part that conflicts is copied as written.
    pub fn into_referrer(self) -> Referrer {
        self.referrer
    }

    /// Returns the Referrer if no error remains, else the errors that remain.
    pub fn finish(self) -> Result<Referrer, Vec<ErrorItem>> {
        if self.errors.is_empty() {
            Ok(self.referrer)
        } else {
            Err(self.errors)
        }
    }
}

/// Writes the [`Referrer`], then one brace group for each remaining [`ErrorItem`].
impl ToTokens for LeakerResult {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.referrer.to_tokens(tokens);
        for error in &self.errors {
            error.to_tokens(tokens);
        }
    }
}

/// Reads a [`Referrer`], then one [`ErrorItem`] for each brace group that follows. Tokens after the
/// last error are not consumed. Fails as the [`Referrer`] does (a missing or different format tag),
/// and on an error that does not parse. A serialized Referrer parses with no errors, unless a brace
/// group follows it.
impl Parse for LeakerResult {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let referrer = input.parse()?;
        let mut errors = Vec::new();
        while input.peek(token::Brace) {
            errors.push(input.parse()?);
        }
        Ok(LeakerResult { referrer, errors })
    }
}

impl Scopes for Leaker {
    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> (ContextId, R) {
        let root = spec.declared_generics();
        let mut visitor = AnalyzeVisitor::new(AllowedPaths::new(self.allow.paths()), &root)
            .reading_values(self.check_values)
            .with_nested_self_entries(self.nested_self_entries);
        let mark = self.contexts.len();
        // The visitor owns the table of contexts while it analyzes, so the swap is the only step
        // before `catch_unwind`: no allocation, no user code. Whatever happens next (a result, or a
        // panic of `f`), the table is swapped back before anything else, and an unfinished
        // analysis is rolled back to `mark`.
        visitor.swap_contexts(&mut self.contexts);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            visitor.start(&spec);
            let result = f(&mut View::new(&mut visitor));
            let (finished, errors) = visitor.finish(&root);
            (finished, errors, result)
        }));
        visitor.swap_contexts(&mut self.contexts);
        let (finished, errors, result) = match outcome {
            Ok(done) => done,
            Err(payload) => {
                self.contexts.truncate(mark);
                resume_unwind(payload)
            }
        };
        let Finished {
            mut effects,
            top_levels,
            module,
            not_optional,
        } = finished;
        drop_optional(&mut effects, &not_optional.into_iter().collect());
        let mut analysis = Analysis::default();
        analysis.apply(&mut self.contexts, &effects, None);
        // Optional entries never fail the call: a type that cannot be leaked is copied as written.
        analysis.apply_optional(&mut self.optional_contexts, &effects, None);
        for ident in analysis.primitives {
            if !self.assumed_primitives.contains(&ident) {
                self.assumed_primitives.push(ident);
            }
        }
        for part in analysis.parts {
            if !self.parts.contains(&part) {
                self.parts.push(part);
            }
        }
        if !(analysis.types.is_empty() && analysis.optional.is_empty()) {
            self.reachable_types = None;
        }
        self.types.merge(analysis.types);
        self.optional_types.merge(analysis.optional);
        // Shares the context of an entry that uses exactly these generics.
        // `Self` is part of the top-level context only if an entry names it, or has a bound
        // rewritten for it (see `Context::of()`).
        let scope_self = spec
            .declared_self()
            .filter(|declared| names_self(&effects, declared));
        let top_level = module.unwrap_or_else(|| {
            let context = Context::new(root, scope_self).with_gats(spec.declared_gats());
            self.contexts.intern(context)
        });
        self.top_levels.insert(top_level);
        self.top_levels.extend(top_levels);
        // Nothing is recorded if the closure panics: the outcome above resumes the unwind. The errors
        // that the configuration accepts are dropped here, as `LeakerResult` drops them later.
        let allow = &self.allow;
        let mut errors = errors;
        allow.filter(&mut errors);
        self.errors.extend(errors);
        (top_level, result)
    }
}

/// Min vertex cut from the roots to the must-intern types, by max flow (Edmonds-Karp, node-split).
/// Ties go to the cut closest to the sink, so inner types win.
fn reduce(types: &TypeGraph, blocked: &HashSet<NodeKey>) -> Vec<NodeKey> {
    let mut keys: Vec<&NodeKey> = types.keys().filter(|key| !blocked.contains(*key)).collect();
    keys.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
    let index: HashMap<&NodeKey, usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let n = keys.len();
    let children: Vec<Vec<usize>> = keys
        .iter()
        .map(|key| {
            let children = types.children(key).into_iter();
            children.filter_map(|c| index.get(&c).copied()).collect()
        })
        .collect();
    let flags = |set: &HashSet<NodeKey>| -> Vec<bool> {
        keys.iter().map(|key| set.contains(*key)).collect()
    };
    let (red, root) = (flags(&types.must_intern), flags(&types.roots));
    // A type that must be an entry cannot be forbidden to be one.
    let forbidden: Vec<bool> = flags(&types.forbidden)
        .into_iter()
        .enumerate()
        .map(|(i, forbidden)| forbidden && !red[i] && !needs_parent(keys[i]))
        .collect();
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, children) in children.iter().enumerate() {
        for &child in children {
            parents[child].push(i);
        }
    }
    let mut has_red = red.clone();
    spread(&mut has_red, &parents);

    // A root that must be interned is in every minimum cut, so it is cut without a flow.
    let cut: Vec<bool> = (0..n).map(|i| red[i] && root[i]).collect();
    let mut kept: Vec<NodeKey> = (0..n)
        .filter(|&i| cut[i])
        .map(|i| keys[i].clone())
        .collect();

    // The edges of the network. A parent that its components cannot replace goes to the sink.
    let mut next: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut to_sink = vec![false; n];
    for i in (0..n).filter(|&i| has_red[i] && !cut[i]) {
        to_sink[i] = red[i];
        let whole = needs_parent(keys[i]);
        for &child in children[i].iter().filter(|&&child| has_red[child]) {
            if whole {
                // The parent is interned as a whole, like a must-intern type.
                to_sink[i] = true;
            } else if !cut[child] {
                next[i].push(child);
            }
        }
    }
    let mut reached: Vec<bool> = (0..n).map(|i| root[i] && has_red[i] && !red[i]).collect();
    spread(&mut reached, &next);

    // Components are independent. Solving each alone keeps the quadratic max flow small.
    let mut component: Vec<usize> = (0..n).collect();
    for i in (0..n).filter(|&i| reached[i]) {
        for &child in &next[i] {
            let (a, b) = (find(&mut component, i), find(&mut component, child));
            component[a.max(b)] = a.min(b);
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut local = vec![0; n];
    for i in (0..n).filter(|&i| reached[i]) {
        let members = groups.entry(find(&mut component, i)).or_default();
        local[i] = members.len();
        members.push(i);
    }
    for members in groups.values() {
        // The nodes: `v_in` is `2 * v`, `v_out` is `2 * v + 1`.
        let m = members.len();
        let (source, sink) = (2 * m, 2 * m + 1);
        // Larger than all node capacities together, so never the bottleneck.
        let infinite = m + 1;
        let mut edges = Vec::new();
        for &i in members {
            let (v_in, v_out) = (2 * local[i], 2 * local[i] + 1);
            // The capacity of a type is what cutting it costs: one entry, or none if it cannot be
            // one.
            edges.push((v_in, v_out, if forbidden[i] { infinite } else { 1 }));
            for &child in &next[i] {
                edges.push((v_out, 2 * local[child], infinite));
            }
            if root[i] {
                edges.push((source, v_in, infinite));
            }
            if to_sink[i] {
                edges.push((v_out, sink, infinite));
            }
        }
        let reaches_sink = sink_side(2 * m + 2, &edges, source, sink);
        // The cut closest to the sink: `v_in` cannot reach it, `v_out` can.
        kept.extend(
            members
                .iter()
                .filter(|&&i| {
                    let v_in = 2 * local[i];
                    reaches_sink[v_in + 1] && !reaches_sink[v_in]
                })
                .map(|&i| keys[i].clone()),
        );
    }

    kept.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
    kept
}

/// Marks every node reachable from a marked one.
fn spread(marked: &mut [bool], edges: &[Vec<usize>]) {
    let mut stack: Vec<usize> = (0..marked.len()).filter(|&i| marked[i]).collect();
    while let Some(i) = stack.pop() {
        for &j in &edges[i] {
            if !marked[j] {
                marked[j] = true;
                stack.push(j);
            }
        }
    }
}

/// See [`is_whole_reference()`].
fn needs_parent(parent: &NodeKey) -> bool {
    is_whole_reference(&parent.ty.ty)
}

/// Runs max flow from `source` to `sink` and returns the nodes that can still reach the sink.
fn sink_side(
    nodes: usize,
    edges: &[(usize, usize, usize)],
    source: usize,
    sink: usize,
) -> Vec<bool> {
    let mut network = BTreeGraph::<usize, usize>::default();
    for node in 0..nodes {
        network.insert_node(node).unwrap();
    }
    for (eix, &(from, to, _)) in edges.iter().enumerate() {
        network.insert_edge(eix, [from, to]).unwrap();
    }
    let (_, flows) = edmonds_karp_with_flows(&network, source, sink, |eix| edges[*eix].2);

    // Residual edges: forward with spare capacity, backward to cancel flow.
    let mut into: Vec<Vec<usize>> = vec![Vec::new(); nodes];
    for (eix, &(from, to, capacity)) in edges.iter().enumerate() {
        let flow = flows.get(&eix).copied().unwrap_or(0);
        if flow < capacity {
            into[to].push(from);
        }
        if flow > 0 {
            into[from].push(to);
        }
    }
    let mut reaches_sink = vec![false; nodes];
    reaches_sink[sink] = true;
    spread(&mut reaches_sink, &into);
    reaches_sink
}

/// Union-find: returns the representative of the set of `i`.
fn find(component: &mut [usize], mut i: usize) -> usize {
    while component[i] != i {
        component[i] = component[component[i]];
        i = component[i];
    }
    i
}

/// Returns the sorted names of the external crates (not `core`) that `ty` and `context` name as
/// `::name`.
fn external_crates(ty: &Type, context: &Context) -> Vec<String> {
    use proc_macro2::{Spacing, TokenTree};
    #[derive(Default)]
    struct Finder(BTreeSet<String>);
    impl Finder {
        fn add(&mut self, ident: &Ident) {
            let name = ident_name(ident);
            if name != "core" {
                self.0.insert(name);
            }
        }
        /// Scans macro tokens for a leading `::` followed by a name.
        fn scan(&mut self, tokens: TokenStream) {
            let mut prev: Option<TokenTree> = None;
            let mut iter = tokens.into_iter().peekable();
            while let Some(tt) = iter.next() {
                match &tt {
                    TokenTree::Group(group) => self.scan(group.stream()),
                    TokenTree::Punct(p) if p.as_char() == ':' && p.spacing() == Spacing::Joint => {
                        let after_name = matches!(&prev, Some(TokenTree::Ident(_)))
                            || matches!(&prev, Some(TokenTree::Punct(p)) if p.as_char() == '>');
                        let second =
                            matches!(iter.peek(), Some(TokenTree::Punct(p)) if p.as_char() == ':');
                        if !after_name && second {
                            iter.next();
                            if let Some(TokenTree::Ident(ident)) = iter.peek() {
                                self.add(ident);
                            }
                        }
                    }
                    _ => (),
                }
                prev = Some(tt);
            }
        }
    }
    impl<'ast> Visit<'ast> for Finder {
        fn visit_path(&mut self, i: &'ast Path) {
            if let (Some(_), Some(seg)) = (&i.leading_colon, i.segments.first()) {
                self.add(&seg.ident);
            }
            syn::visit::visit_path(self, i)
        }
        fn visit_macro(&mut self, i: &'ast Macro) {
            self.scan(i.tokens.clone());
            syn::visit::visit_macro(self, i)
        }
    }
    let mut finder = Finder::default();
    finder.visit_type(ty);
    finder.visit_generics(context.generics());
    match context.self_ty() {
        Some(SelfRequirement::Bound(path)) => finder.visit_path(path),
        Some(SelfRequirement::Type(ty)) => finder.visit_type(ty),
        _ => (),
    }
    finder.0.into_iter().collect()
}

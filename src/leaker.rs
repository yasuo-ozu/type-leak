//! The [`Leaker`], which finds the types depending on the type context.

use crate::allow::{filter, Allow};
use crate::analysis::{drop_optional, names_self, Analysis, PartItem};
use crate::analyze::{module_names, AnalyzeVisitor, Finished, ModuleNames, ScopeSpec, Scopes};
#[cfg(doc)]
use crate::context::SelfRequirement;
use crate::context::{Context, ContextId, Contexts, Gat, ModuleScope};
use crate::encode::EncodeError;
#[cfg(doc)]
use crate::error::Reason;
use crate::error::{forget_dropped, ErrorItem};
use crate::graph::{is_whole_reference, sorted_entries, NodeKey, TypeGraph, TypeKey};
#[cfg(doc)]
use crate::referrer::EntryId;
use crate::referrer::Referrer;
use crate::repeater::{check_repeater_path, ImplSelf, RepeaterImpl};
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
use syn::*;
use template_quote::{quote, ToTokens};

/// Finds the types of an item that depend on where they are written, and collects them in a
/// [`Referrer`].
///
/// Analyze a whole item with an `intern_*` method, or parts of one with an `enter_*` method, then
/// call [`Leaker::result()`]. Keep the returned [`ContextId`] for [`Referrer::visitor()`]. See the
/// [quick start](crate#quick-start). `A` is its [`Allow`]: a helper takes a configured Leaker as
/// `fn g<A: Allow>(leaker: &mut Leaker<A>)`.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(Rel);));
/// let referrer = leaker.result().finish().unwrap();
/// // `Rel` depends on where it is written, so it needs a Repeater impl.
/// let (_, ty, _) = referrer.iter().next().unwrap();
/// assert_eq!(*ty, parse_quote!(Rel));
/// ```
///
/// # Errors and recovery
///
/// An error does not stop the analysis: the part that cannot be leaked is copied as written, and
/// the error is kept (once, in the order found, in one flat list). A type whose own path conflicts
/// is written as it is, with an error for the path, and the types in its arguments are analyzed on
/// their own, so they are entries or have errors after it (`MyType<'_, Rel>`: an error for
/// `MyType`, an entry `Rel`). Some parts are copied whole, with nothing in them leaked, and the
/// relative paths in them are paths of their error: a bound or a type in a predicate of the item
/// (a leaked type there may be unprovable), a bare identifier that may be a const, a reference, a
/// type that holds a trait object that takes its lifetime from it. A panic in a closure rolls its
/// scope back and records nothing. A panicking [`Allow`] is treated alike: the call is rolled back and
/// panics. [`LeakerResult::allow()`] drops the errors you accept, and [`LeakerResult::ignore()`]
/// accepts one and has the copy write the whole part that it is about as written.
///
/// # Limitations
///
/// The Leaker reads only the item you give it, not the definitions of the paths it names, and
/// rejects what would not compile in the copy; the `help` of an error says what to write instead.
///
/// - `MyType<'_>`, `impl Trait`, and a relative trait or value in a bound: write an absolute path
///   (`Self: Sized` needs [`Rules::prelude()`](crate::Rules::prelude)).
/// - A type of another crate (`::foo::X`) is an entry, and the copy replaces it through the
///   Repeater, unless the Leaker's rule allows its path
///   ([`Rules::external_crates()`](crate::Rules::external_crates)): then it is
///   no entry, and the copy writes it as it is (you promise that the crate is the same on both
///   sides). A type that has a type that depends on the context in it (`::foo::W<Rel>`) is always
///   an entry. Otherwise, one in a bound or a where predicate that names a generic
///   (`T: Into<::foo::W<T>>`, `::foo::W<T>: Clone`) is no entry, whatever the rule: replaced,
///   rustc could not prove the predicate for the Repeater's impl (E0277). The copy writes it as it
///   is, so its crate must resolve there.
/// - A shadowed primitive name (`type u8 = i32;`): emit [`Leaker::assertions()`].
/// - Values and bodies are not analyzed unless [`Leaker::check_values`] is on.
/// - A type that names the `Self` of a nested struct, enum or union and uses a generic whose bound
///   names it (`Rel2<Self, T>` with `T: AsRef<Self>`) is an entry for any type, with the error
///   [`Reason::NestedSelfInBounds`]: when you accept it with [`LeakerResult::allow()`], the
///   declaration of the type must have no trait bound on the parameter that receives `Self`
///   (E0277), and rustc before 1.70 fails with E0275 if the copy's field is the last field of a
///   struct. [`LeakerResult::ignore()`] removes the entry instead, so neither holds.
///
/// ## Outlives bounds
///
/// An entry under `&'a` needs `T: 'a` in its Repeater impl (E0309), and the Leaker records only the
/// bounds you write: write them (`fn f<'a, T: 'a>(x: Rel<'a, T>)`) and give [`Referrer::visitor()`]
/// the whole item. For `Self: 'a`, see `Context::self_outlives`.
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
/// An entry that names `Self` keeps every bound, and in a predicate it is an error if one names `T`
/// in a path that may be a type alias of a projection, which the Leaker cannot read (`T:
/// Fn(Wrap<T>)`, not `T: From<T>`), also where a predicate of a method equates the `Item` of `T`.
///
/// ## Bare identifiers and consts
///
/// A bare identifier argument (`MyType<MY_CONST>`) may be a type or a const. The Leaker leaks the
/// type that holds it as a whole, right for both; where it cannot (an elided lifetime or reference
/// in the holder, a trait bound), it leaks the identifier alone, which fails for a const (E0573,
/// E0747). Write an absolute path (`MyType<{ ::my_crate::MY_CONST }>`) or an [allowed](Allow)
/// name.
pub struct Leaker<A = ()> {
    /// Makes the Leaker read the values of an item too, if it is `true` (it is `false` by default):
    /// the values of consts and statics, the defaults and values of associated consts, and the
    /// bodies of functions and methods. Set it before the calls that it should affect.
    ///
    /// Enum discriminants, the defaults of const parameters and array lengths are read either way
    /// (the first two as a value is, but their errors are no [`ErrorItem::in_value`]).
    ///
    /// Without it a relative path in a value breaks the copy and the Leaker does not tell. The
    /// types in a type position of a value (a cast, a `let`, a closure parameter or output, the
    /// generic arguments of a path or method call, the self type of a qualified path) are leaked
    /// like any type of the item. Another relative path (a function, a constant, a struct literal,
    /// a variant, the trait of a qualified path) is rejected unless it is absolute or starts with
    /// `$crate`, is [allowed](Allow) (the types too, by [`Rules::values()`](crate::Rules::values)),
    /// or starts with a generic or `Self` in scope or a name that the value declares. A macro whose
    /// arguments are not expressions (`cfg!`, `stringify!` and `matches!` excepted), a
    /// `macro_rules!`, a glob import and syntax that `syn` does not parse are rejected.
    ///
    /// ```
    /// # use type_leak::*;
    /// let mut leaker = Leaker::from_config(Rules::new().prelude());
    /// leaker.check_values = true;
    /// ```
    pub check_values: bool,
    /// What the Leaker leaves as written instead of leaking it, and the errors it does not
    /// record.
    allow: A,
    /// The types that depend on the context, or that the Referrer may replace.
    types: TypeGraph,
    /// The contexts of the scopes and of the entries.
    contexts: Contexts,
    /// The top-level contexts of the scopes (also of those with errors), which the [`Referrer`]
    /// keeps without an entry too.
    top_levels: BTreeSet<ContextId>,
    /// The primitive names that the copy writes as they are.
    assumed_primitives: Vec<Ident>,
    /// The parts copied as written.
    parts: Vec<PartItem>,
    /// The errors of the outermost calls, in the order found.
    errors: Vec<ErrorItem>,
    /// The number of errors that the calls found, also those that a rule dropped: the ids of the
    /// errors of the next call count from it (`Link::id`).
    found: usize,
}

impl Leaker {
    /// Creates a [`Leaker`] that allows nothing, so every relative path is leaked.
    pub fn new() -> Self {
        Self::from_config(())
    }
}

impl Default for Leaker {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Allow> Leaker<A> {
    /// Creates a [`Leaker`] with the rule `allow`. See [`Leaker::check_values`].
    pub fn from_config(allow: A) -> Self {
        Self {
            check_values: false,
            allow,
            types: TypeGraph::default(),
            contexts: Contexts::default(),
            top_levels: BTreeSet::new(),
            assumed_primitives: Vec::new(),
            parts: Vec::new(),
            errors: Vec::new(),
            found: 0,
        }
    }

    /// Returns a compile-time check that the primitive names copied as written (`u8` in `const N:
    /// u8`, or by [`Rules::primitive()`](crate::Rules::primitive)) are not shadowed, also those in a part that an error
    /// makes copied as written. Emit it in the Leaker's crate. It is empty if none was copied.
    #[must_use]
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
    /// A type that names `Self` is leaked as one entry for the struct (`Self::A` cannot be leaked
    /// whole, E0220: write `<Self as ::path::Trait>::A`).
    pub fn intern_struct(&mut self, item: &ItemStruct) -> ContextId {
        self.enter_type(&item.ident, &item.generics, |view| {
            syn::visit::visit_item_struct(view, item)
        })
        .0
    }

    /// Analyzes an enum like [`intern_struct()`](Leaker::intern_struct).
    pub fn intern_enum(&mut self, item: &ItemEnum) -> ContextId {
        self.enter_type(&item.ident, &item.generics, |view| {
            syn::visit::visit_item_enum(view, item)
        })
        .0
    }

    /// Analyzes a union like [`intern_struct()`](Leaker::intern_struct).
    pub fn intern_union(&mut self, item: &ItemUnion) -> ContextId {
        self.enter_type(&item.ident, &item.generics, |view| {
            syn::visit::visit_item_union(view, item)
        })
        .0
    }

    /// Analyzes an impl and returns its top-level context. `Self` is the self type
    /// ([`SelfRequirement::Type`]).
    ///
    /// An entry in the self type that depends on `Self` through a bound of the impl (`impl<T> S<T>
    /// where T: AsRef<Option<Self>>`) is an error, also in an impl that [`View::visit_item()`] or
    /// [`intern_mod()`](Leaker::intern_mod) visits: use an absolute self type or a bound without
    /// `Self`. The trait path must not depend on the context.
    pub fn intern_impl(&mut self, item: &ItemImpl) -> ContextId {
        // The scope checks the path of the trait, and the walk analyzes its arguments.
        let trait_path = item.trait_.as_ref().map(|(_, path, _)| path);
        self.enter_impl(&item.generics, &item.self_ty, trait_path, |view| {
            view.walk_item_impl(item)
        })
        .0
    }

    /// Analyzes a trait and returns its top-level context. `Self` is the implementor
    /// ([`SelfRequirement::Bound`]).
    ///
    /// Unlike [`enter_trait()`](Leaker::enter_trait), it knows the generic associated types (GATs)
    /// of the trait: `Self::G<'x>` records the outlives bounds of the GAT, which the copy must
    /// prove, so in an impl for another type add `where T: 'x` or `where Self: 'x` (unless the
    /// receiver is `&'x self`) to the method. GATs of supertraits or of `T::G<'x>` are not seen.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&parse_quote!(trait Tr { fn f(&self) -> Wrap<Self>; }));
    /// let referrer = leaker.result().finish().unwrap();
    /// let (_, _, context) = referrer.iter().next().unwrap();
    /// assert_eq!(context.self_ty, Some(SelfRequirement::Bound(parse_quote!(Tr))));
    /// ```
    pub fn intern_trait(&mut self, item: &ItemTrait) -> ContextId {
        // The scope knows the generic associated types of the trait: what they require is part of
        // the context of an entry that names them.
        let gats = Gat::of_items(&item.items);
        self.enter_trait_of(&item.ident, &item.generics, gats, |view| {
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
            Some((_, items)) => module_names(items),
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
    /// `hint` is for debugging ([`ModuleScope::Itself`]).
    pub fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
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
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_generics(self, generics, f)
    }

    /// Analyzes parts of a struct, an enum or a union: runs `f` on a [`View`] inside the type
    /// `ident` with `generics`. Like [`enter_generics()`](Leaker::enter_generics), but `Self` is
    /// the type ([`SelfRequirement::Type`]) and `generics` may name it.
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_type(self, ident, generics, f)
    }

    /// Analyzes parts of an impl: runs `f` on a [`View`] inside an impl with `generics` for
    /// `self_ty`. Like [`enter_type()`](Leaker::enter_type), with `self_ty` as `Self`.
    ///
    /// `trait_` is the trait that the impl implements, and `None` is an inherent impl. The path is
    /// checked here (a relative path is an error), not its arguments: `f` visits them. The methods
    /// of an inherent impl are not compared with a trait's
    /// ([Method lifetimes](Leaker#method-lifetimes)), and the help for `Self::A` names the trait.
    /// A caller that resolves the trait by its name elsewhere passes the path and [allows it](Allow),
    /// or `None` if it visits no method.
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        trait_: Option<&Path>,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_impl(self, generics, self_ty, trait_, f)
    }

    /// Analyzes parts of a trait: runs `f` on a [`View`] inside the trait `ident` with `generics`.
    /// Like [`enter_type()`](Leaker::enter_type), but `Self` is the implementor
    /// ([`SelfRequirement::Bound`]), and the GATs of the trait are not known: see
    /// [`intern_trait()`](Leaker::intern_trait).
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_trait(self, ident, generics, f)
    }

    /// Ends the analysis and returns the [`LeakerResult`]: the [`Referrer`] and the errors found.
    /// The entries that have a type that depends on the context come first, then those that are
    /// only a type of another crate (`::foo::X`), each sorted by context (in order of first
    /// appearance) and tokens.
    pub fn result(mut self) -> LeakerResult {
        let mut errors = std::mem::take(&mut self.errors);
        LeakerResult {
            referrer: self.referrer(&mut errors),
            errors,
        }
    }

    fn referrer(mut self, errors: &mut [ErrorItem]) -> Referrer {
        // Not `reduce_roots()`: the graph of the kept entries would be built only to be read.
        let types = std::mem::take(&mut self.types);
        let (required, optional) = if types.is_reduced() {
            types.into_kept_entries()
        } else {
            let (required, optional) = reduce(&types);
            (sorted_entries(required), sorted_entries(optional))
        };
        self.resolve_links(errors, required.iter().chain(&optional));
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
        // The required entries come first, then the optional ones: that is the order of the indices.
        let mut entries = Vec::new();
        for key in required.into_iter().chain(optional) {
            context_map
                .entry(key.context)
                .or_insert_with(|| context_of(key.context));
            entries.push((key.context, key.ty.ty));
        }
        // Keep the modules the contexts depend on, and their parents, for nested lookups.
        let mut seen = HashSet::new();
        let mut pending: Vec<ContextId> = Vec::new();
        for context in context_map.values() {
            pending.extend(context.module_id());
        }
        while let Some(module) = pending.pop() {
            if seen.insert(module) {
                let context = context_map
                    .entry(module)
                    .or_insert_with(|| context_of(module));
                if let Some(ModuleScope::Itself { parent, .. }) = &context.module {
                    pending.extend(*parent);
                }
            }
        }
        Referrer::new(entries, BTreeSet::new(), context_map, self.parts)
    }

    /// Resolves what the analysis knows of what each error is about (`Link::raw`) to indices: the
    /// part, which is recorded if the Referrer has not got it (a part that only its head
    /// conflicts in), and the entry. The contexts that this adds come last.
    fn resolve_links<'k>(
        &mut self,
        errors: &mut [ErrorItem],
        entries: impl Iterator<Item = &'k NodeKey>,
    ) {
        let entries: Vec<&NodeKey> = entries.collect();
        for error in errors {
            let raw = std::mem::take(&mut error.link.raw);
            if let Some(part) = raw.part {
                let item = part.item(&mut self.contexts);
                let at = self.parts.iter().position(|known| *known == item);
                error.link.part = Some(at.unwrap_or_else(|| {
                    self.parts.push(item);
                    self.parts.len() - 1
                }));
            }
            if let Some((context, ty)) = raw.entry {
                let context = self.contexts.find(&context);
                let key = TypeKey::new(&ty);
                error.link.entry = context.and_then(|context| {
                    entries
                        .iter()
                        .position(|entry| entry.context == context && entry.ty == key)
                });
            }
        }
    }

    /// Chooses the fewest entries from which every leaked type can be built, and removes every
    /// other type from the Leaker; [`Leaker::result()`] calls it. `&'a Rel`, a path with a lifetime
    /// or bare-identifier argument (`Wrap<SIZE>`), a type with a type in an expression
    /// (`[u8; size_of::<Big>()]`) and a trait object are entries as a whole. One choice covers the
    /// types that depend on the context and the types of other crates (`::foo::X`): an entry with a
    /// type that depends on the context in it is of the first kind, and has a smaller index.
    ///
    /// Analyzing more afterwards adds to what is left, and the kept entries stay entries and keep
    /// their kind, so calling it again changes nothing. The price: the result can have more
    /// entries than one reduction at the end gives (`(Rel1, Rel2)` is kept, then `Rel1` alone is
    /// an entry too), and a later scope in which a kept entry would make the copy cyclic (where
    /// its parts would be the entries) still gets it as an entry, so its copy may not compile.
    pub fn reduce_roots(&mut self) {
        if self.types.is_reduced() {
            return;
        }
        // The graph is dropped before the new one is built.
        let types = std::mem::take(&mut self.types);
        let (required, optional) = reduce(&types);
        drop(types);
        self.types = TypeGraph::kept(required, optional);
    }
}

/// What a [`Leaker`] found: the [`Referrer`] and the errors that remain.
///
/// [`allow()`](LeakerResult::allow) takes and returns the result, so it chains. It drops the errors
/// that an [`Allow`] accepts (you promise that the part compiles as written). It leaves the
/// entries as they are: to remove entries, give the [`Allow`] to [`Leaker::from_config()`]. A rule
/// here only approximates the one of the Leaker (more errors may remain, and a copy may fail to
/// build): give rules to the Leaker when you can. [`ignore()`](LeakerResult::ignore) accepts one
/// error, and has the copy write its whole part as written, or removes its entry.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(::p::W<'_, dyn RelTr>);));
/// let result = leaker.result();
/// assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));
/// // Allowing the error copies the part as written, and `finish()` is `Ok`.
/// let result = result.allow(|cause: &Cause, _: Place| *cause == Cause::Trait(parse_quote!(RelTr)));
/// assert!(result.finish().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct LeakerResult {
    /// The Referrer of the analysis.
    referrer: Referrer,
    /// The errors that remain.
    errors: Vec<ErrorItem>,
}

impl LeakerResult {
    /// Drops the errors that `rule` accepts: the part is copied as written. An error is accepted
    /// by its [`paths`](ErrorItem::paths) if it has any, else by its cause. The other errors stay,
    /// in order.
    ///
    /// You promise that the part can be written as it is in the copy. The cause may be smaller
    /// than the part that is copied: in `[u8; { let c = |a: u8| a; 3 }]` the error is about `u8`,
    /// and the array is copied whole. Allowing a [`Reason::UnprovableBound`] does not make the copy
    /// compile (E0277). To allow a cause, compare it with `==` (the same kind and tokens, spans
    /// ignored). An absolute path is never an error, so a rule for `::serde` changes no error. It
    /// changes no entry either: to have the copy write a type of another crate as it is, give
    /// [`Rules::external_crates()`](crate::Rules::external_crates) to
    /// [`Leaker::from_config()`].
    ///
    /// See [`Allow`] for what the rule is asked. A closure needs `&Cause` and
    /// [`Place`](crate::Place) annotated when it is not written in the call
    /// (`|cause: &Cause, _: Place| ..`); [`Rules::or_if()`](crate::Rules::or_if) needs none. To
    /// have the copy write the whole part as written, use [`ignore()`](LeakerResult::ignore).
    pub fn allow(mut self, rule: impl Allow) -> Self {
        filter(&rule, &mut self.errors);
        self
    }

    /// Accepts `error`, one of [`errors()`](LeakerResult::errors): the copy writes the part that it
    /// is about as written, whole, and nothing is looked up in it. The error goes, and so do the
    /// errors revealed inside that part and the errors about the same part.
    ///
    /// [`allow()`](LeakerResult::allow) writes only the head that is wrong as written (`MyType` in
    /// `MyType<'_, Rel>`) and replaces the entries in the arguments; `ignore()` writes all of
    /// `MyType<'_, Rel>` as written, so `Rel` is not replaced there. An entry that is used at
    /// another place is still one there. You promise that the part compiles as written.
    ///
    /// The error of an entry ([`Reason::NestedSelfInBounds`]) removes the entry instead: it gets no
    /// impl ([`repeater_impls()`](LeakerResult::repeater_impls)), [`Referrer::iter()`] skips it,
    /// and the copy writes its type as written wherever it would have been replaced. The other
    /// entries keep their [`EntryId`]s, so it can be called before or after the impls are written.
    ///
    /// An error that is about no part (a `Self` without meaning, a path in a value) is only
    /// dropped. An error about an entry inside the part stays: the entry may be used elsewhere, so
    /// ignore it on its own. If `error` is not one of the errors, nothing changes, so calling it
    /// again is harmless. Call it before [`allow()`](LeakerResult::allow) for the same error.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// # use syn::visit_mut::VisitMut;
    /// let item: ItemStruct = parse_quote!(struct S<'a>(MyType<'_, Inner<'_>, Rel>, Rel, &'a u8););
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_struct(&item);
    /// let result = leaker.result();
    /// // `MyType` and, inside it, `Inner<'_>`.
    /// assert_eq!(result.errors().len(), 2);
    ///
    /// let error = result.errors()[0].clone();
    /// let result = result.ignore(&error);
    /// assert!(result.errors().is_empty());
    /// let referrer = result.finish().unwrap();
    ///
    /// let mut copy = item.clone();
    /// referrer
    ///     .visitor(top_level, |_, ix, _| parse_str(&format!("E{}", ix.index())).unwrap())
    ///     .visit_item_struct_mut(&mut copy);
    /// // `Rel` is replaced in the second field only.
    /// assert_eq!(copy, parse_quote!(struct S<'a>(MyType<'_, Inner<'_>, Rel>, E0, E1);));
    /// ```
    pub fn ignore(mut self, error: &ErrorItem) -> Self {
        let at = match self.errors.iter().position(|known| known == error) {
            Some(at) => at,
            None => return self,
        };
        let (part, entry) = (self.errors[at].link.part, self.errors[at].link.entry);
        // The errors about the same part, or entry, go together.
        let same = |known: &ErrorItem| {
            (part.is_some() && known.link.part == part)
                || (entry.is_some() && known.link.entry == entry)
        };
        let mut inside = HashSet::new();
        for (ix, known) in self.errors.iter().enumerate() {
            if ix == at || same(known) {
                if let Some(part) = known.link.part {
                    self.referrer.make_whole(part);
                }
                if let Some(entry) = known.link.entry {
                    self.referrer.remove(entry);
                }
                inside.extend(known.link.inner.iter().copied());
            }
        }
        // An error about an entry inside the part stays: the entry may be used elsewhere.
        let mut ix = 0;
        self.errors.retain(|known| {
            ix += 1;
            let revealed = known.link.entry.is_none()
                && known.link.id.map_or(false, |id| inside.contains(&id));
            !(ix - 1 == at || same(known) || revealed)
        });
        forget_dropped(&mut self.errors);
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

    /// Returns the Referrer as [`into_referrer()`](LeakerResult::into_referrer) does, and the
    /// errors that remain, in the order found.
    pub fn into_parts(self) -> (Referrer, Vec<ErrorItem>) {
        (self.referrer, self.errors)
    }

    /// Returns the Referrer if no error remains, else the errors that remain.
    pub fn finish(self) -> Result<Referrer, Vec<ErrorItem>> {
        if self.errors.is_empty() {
            Ok(self.referrer)
        } else {
            Err(self.errors)
        }
    }

    /// Describes the Repeater impl of each entry: its generics, trait path and type.
    ///
    /// It is the Leaker's side that writes the impls: write one for every entry, in
    /// [`RepeaterImpl::module`] if it is `Some`. The items come in the order of the entries
    /// ([`Referrer::iter()`], so without those that [`ignore()`](LeakerResult::ignore) removed),
    /// each with its [`EntryId`] in [`RepeaterImpl::entry`], which the
    /// Repeater's `KEY` ([`repeater_key()`](crate::repeater_key)) and [`Referrer::bucket_self()`]
    /// take.
    ///
    /// `repeater` is the path of the Repeater trait; it becomes [`RepeaterImpl::path`], with the
    /// first argument. `id` is the Leaker's id: the one that the copy's side passes to
    /// [`repeater_projection()`](crate::repeater_projection) and
    /// [`Referrer::bucket_self()`]. `ToTokens` of the item writes the impl with the `KEY`
    /// ([`repeater_key(id, entry)`](crate::repeater_key)), and [`ImplSelf::Bucket`] chooses the
    /// type from the same `id`. `target` says what every impl is for ([`ImplSelf`]); `const_fn` is
    /// as in [`repeater_projection()`](crate::repeater_projection): it gets the parameter itself
    /// (`N`) as `value`.
    ///
    /// The call fails, before any item, with the first of these:
    ///
    /// - [`EncodeError::NeedsSelf`], for [`ImplSelf::Marker`], if some entry names `Self`, or its
    ///   context's generics do. The error has the first such entry.
    /// - The first error of `const_fn`. It is called once for each const parameter of each entry,
    ///   in the order of the entries and of the parameters.
    ///
    /// The items do not fail. They are made one at a time as you iterate, and call `const_fn`
    /// again, so it must give the same answer for the same parameter every time: if it fails for a
    /// parameter that it accepted in the call, the iterator panics.
    ///
    /// # Panics
    ///
    /// If `repeater` has arguments (`crate::Repeater<u8>`), before any item: it names the trait
    /// alone, or with an empty `<>`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S<T>(Wrap<T>);));
    /// let result = leaker.result();
    ///
    /// let repeater: Path = parse_quote!(crate::Repeater);
    /// let r = result
    ///     .repeater_impls(&repeater, 7, ImplSelf::Param(&parse_quote!(__S)), array_const)
    ///     .unwrap()
    ///     .next()
    ///     .unwrap();
    /// assert_eq!(r.ty, parse_quote!(Wrap<T>));
    /// // `path` has no `KEY`, and `ToTokens` writes the one of the id 7.
    /// assert_eq!(r.path, parse_quote!(crate::Repeater<(T,)>));
    /// ```
    ///
    /// An entry that names `Self` has no impl for a marker, so the call fails:
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S(Wrap<Self>);));
    /// let result = leaker.result();
    ///
    /// let repeater: Path = parse_quote!(Repeater);
    /// let marker: Type = parse_quote!(Marker);
    /// let err = result.repeater_impls(&repeater, 7, ImplSelf::Marker(&marker), array_const).err();
    /// assert!(matches!(err, Some(EncodeError::NeedsSelf(_))));
    /// ```
    pub fn repeater_impls<'a>(
        &'a self,
        repeater: &'a Path,
        id: u64,
        target: ImplSelf<'a>,
        mut const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError> + 'a,
    ) -> std::result::Result<impl Iterator<Item = RepeaterImpl> + 'a, EncodeError> {
        check_repeater_path(repeater);
        // The entries that no `ignore()` removed.
        let live: Vec<usize> = self.referrer.live().collect();
        for &ix in &live {
            self.referrer.check_marker(ix, target)?;
        }
        for &ix in &live {
            self.referrer.check_consts(ix, target, &mut const_fn)?;
        }
        Ok(live.into_iter().map(move |ix| {
            self.referrer
                .repeater_impl(repeater, id, ix, target, &mut const_fn)
                .unwrap_or_else(|error| {
                    panic!(
                        "`const_fn` of `repeater_impls()` failed for the entry {} after it had \
                         not failed in the call: {}",
                        ix, error
                    )
                })
        }))
    }
}

/// Writes the [`Referrer`], then, if errors remain, `@ [ {error} {error} .. ]`: one brace group for
/// each [`ErrorItem`], in the order found, in one bracket group after an `@`.
impl ToTokens for LeakerResult {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.referrer.to_tokens(tokens);
        if !self.errors.is_empty() {
            tokens.extend(quote! {
                @ [#(for error in &self.errors) { #error }]
            });
        }
    }
}

/// Reads a [`Referrer`], then the errors of `@ [ {error} .. ]` if that follows. Anything else
/// after the Referrer is left to the caller: only an `@` followed by a bracket group is read, so a
/// brace group after the result is not an error. Fails as the [`Referrer`] does (a missing or
/// different format tag), and on an `@ [ .. ]` that holds something else than error groups. A
/// serialized Referrer parses with no errors.
impl Parse for LeakerResult {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let referrer: Referrer = input.parse()?;
        let mut errors = Vec::new();
        let mut spans = Vec::new();
        if input.peek(Token![@]) && input.peek2(token::Bracket) {
            input.parse::<Token![@]>()?;
            let list;
            bracketed!(list in input);
            while !list.is_empty() {
                if !list.peek(token::Brace) {
                    return Err(list.error("expected an error `{ .. }` in `@ [ .. ]`"));
                }
                spans.push(list.span());
                errors.push(list.parse()?);
            }
        }
        check_links(&referrer, &errors, &spans)?;
        Ok(LeakerResult { referrer, errors })
    }
}

/// Fails at the span of the first error that is about a part, an entry or an error that is not
/// there, or has the `id` of another.
fn check_links(referrer: &Referrer, errors: &[ErrorItem], spans: &[Span]) -> syn::Result<()> {
    let mut ids = HashSet::new();
    for (error, span) in errors.iter().zip(spans) {
        let link = &error.link;
        if let Some(part) = link.part.filter(|&part| part >= referrer.part_count()) {
            let message = format!("the part {} of an error is not one of the parts", part);
            return Err(syn::Error::new(*span, message));
        }
        if let Some(entry) = link.entry.filter(|&entry| entry >= referrer.entry_count()) {
            let message = format!("the entry {} of an error is not one of the entries", entry);
            return Err(syn::Error::new(*span, message));
        }
        if let Some(id) = link.id.filter(|&id| !ids.insert(id)) {
            let message = format!("the id {} is the id of another error", id);
            return Err(syn::Error::new(*span, message));
        }
    }
    for (error, span) in errors.iter().zip(spans) {
        if let Some(id) = error.link.inner.iter().find(|id| !ids.contains(id)) {
            let message = format!(
                "an error inside an error has the id {}: no error has it",
                id
            );
            return Err(syn::Error::new(*span, message));
        }
    }
    Ok(())
}

impl<A: Allow> Scopes for Leaker<A> {
    type Rule = A;

    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        let root = spec.declared_generics();
        let mut visitor = AnalyzeVisitor::new(&mut self.allow, &root, self.check_values);
        let mark = self.contexts.len();
        // The visitor owns the table of contexts while it analyzes, so the swap is the only step
        // before `catch_unwind`: no allocation, no user code. Whatever happens next (a result, or a
        // panic of `f`), the table is swapped back before anything else, and an unfinished
        // analysis is rolled back to `mark`.
        visitor.swap_contexts(&mut self.contexts);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            visitor.start(&spec);
            let result = f(&mut View::new(&mut visitor));
            let (finished, mut errors) = visitor.finish(&spec, &root);
            let found = errors.len();
            // The errors that the rule accepts are dropped here, as `LeakerResult` drops
            // them later. It is user code too: if it panics, nothing is recorded.
            filter(visitor.rule(), &mut errors);
            (finished, errors, found, result)
        }));
        visitor.swap_contexts(&mut self.contexts);
        let (finished, mut errors, found, result) = match outcome {
            Ok(done) => done,
            Err(payload) => {
                self.contexts.truncate(mark);
                resume_unwind(payload)
            }
        };
        let top_level = self.record(&spec, root, finished);
        // The ids of the errors count from the first error of this call, also those dropped.
        for error in &mut errors {
            error.link.shift(self.found);
        }
        self.found += found;
        self.errors.extend(errors);
        (top_level, result)
    }
}

impl<A: Allow> Leaker<A> {
    /// Records what the outermost scope `spec` with the generics `root` found, and returns its
    /// top-level context.
    fn record(&mut self, spec: &ScopeSpec, root: Generics, finished: Finished) -> ContextId {
        let Finished {
            mut effects,
            top_levels,
            module,
            not_optional,
        } = finished;
        drop_optional(&mut effects, &not_optional.into_iter().collect());
        let mut analysis = Analysis::default();
        analysis.apply(&mut self.contexts, &effects, None);
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
        self.types.merge(analysis.types);
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
        top_level
    }
}

/// Min vertex cut from the roots to the red types, by max flow (Edmonds-Karp, node-split). Ties go
/// to the cut closest to the sink, so inner types win. Returns the required entries and the
/// optional ones.
///
/// A type is red if it is in `required`, or in `optional` and may be an entry. The kind of an entry
/// is what it covers: required if it is or holds a type of `required`, else optional. A type is cut
/// only as the kind it is: a candidate of the kind, not in `no_entry` (an optional type is a
/// candidate of both). An optional type that may not be an entry is not red, and its parts are.
///
/// An entry that an earlier reduction kept is red and cut whatever else the graph says of it, as
/// the kind it was kept as (or required, if a type merged in later makes it hold a required type).
fn reduce(types: &TypeGraph) -> (Vec<NodeKey>, Vec<NodeKey>) {
    let mut keys: Vec<&NodeKey> = types.keys().collect();
    keys.sort_by_cached_key(|key| (key.context, key.ty.tokens()));
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
    let (is_required, is_optional, root, is_kept) = (
        flags(&types.required),
        flags(&types.optional),
        flags(&types.roots),
        flags(&types.kept),
    );
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, children) in children.iter().enumerate() {
        for &child in children {
            parents[child].push(i);
        }
    }
    // An entry that covers a required type is a required entry.
    let mut covers_required = is_required.clone();
    spread(&mut covers_required, &parents);
    // An optional type is an entry anyway, so it is a candidate of either kind.
    let may_cut: Vec<bool> = (0..n)
        .map(|i| {
            let candidate = match covers_required[i] {
                true => is_optional[i] || types.required_candidates.contains(keys[i]),
                false => types.optional_candidates.contains(keys[i]),
            };
            candidate && !types.no_entry.contains(keys[i])
        })
        .collect();
    // An optional type that cannot be an entry is no entry, and its parts are. A kept entry is one.
    let red: Vec<bool> = (0..n)
        .map(|i| is_required[i] || is_kept[i] || (is_optional[i] && may_cut[i]))
        .collect();
    // A type that must be an entry cannot be forbidden to be one.
    let forbidden: Vec<bool> = (0..n)
        .map(|i| !may_cut[i] && !red[i] && !needs_parent(keys[i]))
        .collect();
    let mut has_red = red.clone();
    spread(&mut has_red, &parents);

    // A red root is in every minimum cut, so it is cut without a flow. So is a
    // kept entry, which is a root of the graph it was kept in.
    let cut: Vec<bool> = (0..n).map(|i| (red[i] && root[i]) || is_kept[i]).collect();
    let mut kept: Vec<usize> = (0..n).filter(|&i| cut[i]).collect();

    // The edges of the network. A parent that its components cannot replace goes to the sink.
    let mut next: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut to_sink = vec![false; n];
    for i in (0..n).filter(|&i| has_red[i] && !cut[i]) {
        to_sink[i] = red[i];
        let whole = needs_parent(keys[i]);
        for &child in children[i].iter().filter(|&&child| has_red[child]) {
            if whole {
                // The parent is interned as a whole, like a required type.
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
        kept.extend(members.iter().copied().filter(|&i| {
            let v_in = 2 * local[i];
            reaches_sink[v_in + 1] && !reaches_sink[v_in]
        }));
    }

    let entries = |kind: bool| {
        let found = kept.iter().filter(|&&i| covers_required[i] == kind);
        found.map(|&i| keys[i].clone()).collect()
    };
    (entries(true), entries(false))
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

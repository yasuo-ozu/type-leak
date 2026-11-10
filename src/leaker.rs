//! The [`Leaker`], which finds the types depending on the type context.

use crate::analysis::{drop_optional, names_self, Analysis, PartItem};
use crate::analyze::{AnalyzeVisitor, Finished, ScopeSpec, Scopes};
#[cfg(doc)]
use crate::context::SelfRequirement;
use crate::context::{Context, ContextId, Contexts, Gat};
use crate::error::ErrorItem;
#[cfg(doc)]
use crate::error::Reason;
use crate::graph::{sorted_entries, NodeKey, TypeGraph, TypeKey};
#[cfg(doc)]
use crate::referrer::EntryId;
use crate::referrer::Referrer;
use crate::result::{filter, Allow, LeakerResult};
use crate::util::push_once;
use crate::view::View;
use proc_macro2::{Span, TokenStream};
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use syn::*;
use template_quote::quote;

/// Finds the types of an item that depend on where they are written, and collects them in a
/// [`Referrer`].
///
/// Analyze a whole item with an `intern_*` method, or parts of one with an `enter_*` method, then
/// call [`Leaker::result()`]. Keep the returned [`ContextId`] for [`Referrer::visitor()`]. See the
/// [quick start](crate#quick-start). `A` is its [`Allow`]: a helper takes a configured Leaker as
/// `fn g<A: Allow>(leaker: &mut Leaker<A>)`.
///
/// Analyze only items written in one module with one Leaker: its Repeater impls go to one place,
/// where every relative path it saw must resolve.
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
/// [`LeakerResult::finish()`] gives the [`Referrer`] only when no error remains.
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
///   is, so its crate must resolve there. A `::core` path is never an entry; `::std` and `::alloc`
///   are crates like any other (see [`Rules::std_crates()`](crate::Rules::std_crates)).
/// - A shadowed primitive name (`type u8 = i32;`): emit [`Leaker::assertions()`].
/// - Values and bodies are not analyzed unless [`Leaker::check_values`] is on.
/// - A `mod` item is not analyzed: [`View::visit_item()`] of one is an error
///   ([`Reason::Unreadable`]). A `mod` in an expression of a type (an array length) goes with the
///   type.
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
    /// Reads the values of an item too, if `true` (default `false`): the values of consts and
    /// statics, the defaults and values of associated consts, and the bodies of functions and
    /// methods. Set it before the calls that it should affect.
    ///
    /// A type in a type position of a value (a cast, a `let`, a closure parameter, the arguments of
    /// a path or method call) is leaked like any type of the item. Another relative path in a value
    /// is an error, unless it is [allowed](Allow) ([`Rules::values()`](crate::Rules::values)
    /// allows all), or starts with a generic, `Self` or a name that the value declares. A macro
    /// whose arguments are not expressions (`cfg!`, `stringify!` and `matches!` excepted), a
    /// `macro_rules!`, a glob import and syntax that `syn` does not parse are errors. Without the
    /// flag the copy breaks on such a path and the Leaker does not tell.
    ///
    /// Enum discriminants, the defaults of const parameters and array lengths are read either way.
    ///
    /// ```ignore
    /// // With `check_values`: `Rel` is an entry, and `helper` is an error.
    /// pub trait Tr { const LEN: usize = helper::<Rel>(); }
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
        Self::with_rule(())
    }
}

impl Default for Leaker {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Allow> Leaker<A> {
    /// Creates a [`Leaker`] that asks `rule` what the copy may write as it is. A closure must
    /// annotate its first parameter (`|cause: &Cause, _| ..`). See [`Leaker::check_values`].
    pub fn with_rule(rule: A) -> Self {
        Self {
            check_values: false,
            allow: rule,
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
    /// where T: AsRef<Option<Self>>`) is an error, also in an impl that [`View::visit_item()`]
    /// visits: use an absolute self type or a bound without `Self`. The trait path must not depend
    /// on the context.
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
            let (required, optional) = types.reduce();
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
        // The required entries come first, then the optional ones: the order of the indices.
        let mut entries = Vec::new();
        for key in required.into_iter().chain(optional) {
            context_map
                .entry(key.context)
                .or_insert_with(|| context_of(key.context));
            entries.push((key.context, key.ty.ty));
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
    /// other type from the Leaker. [`Leaker::result()`] makes the same choice without it: call this
    /// only to go on analyzing on the reduced Leaker. `&'a Rel`, a path with a lifetime
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
        let (required, optional) = types.reduce();
        drop(types);
        self.types = TypeGraph::of_kept(required, optional);
    }
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
            let (finished, mut errors) = visitor.finish(&spec);
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
            not_optional,
        } = finished;
        drop_optional(&mut effects, &not_optional.into_iter().collect());
        let mut analysis = Analysis::default();
        analysis.apply(&mut self.contexts, &effects, None);
        for ident in analysis.primitives {
            push_once(&mut self.assumed_primitives, ident);
        }
        for part in analysis.parts {
            push_once(&mut self.parts, part);
        }
        self.types.merge(analysis.types);
        // `Self` is part of the top-level context only if an entry names it, or has a bound
        // rewritten for it (see `Context::of()`).
        let scope_self = spec
            .declared_self()
            .filter(|declared| names_self(&effects, declared));
        let context = Context::new(root.clone(), scope_self)
            .with_gats(spec.declared_gats())
            .with_declared(root);
        let top_level = self.contexts.intern(context);
        self.top_levels.insert(top_level);
        self.top_levels.extend(top_levels);
        top_level
    }
}

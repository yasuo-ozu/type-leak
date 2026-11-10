//! The [`Leaker`], which finds the types depending on the type context.

use crate::analysis::{drop_optional, names_self, Analysis, PartItem};
use crate::analyze::{AnalyzeVisitor, Finished, ScopeSpec, Scopes};
#[cfg(doc)]
use crate::context::SelfRequirement;
use crate::context::{Context, ContextId, Contexts, Gat};
use crate::error::ErrorItem;
#[cfg(doc)]
use crate::error::Reason;
use crate::graph::{NodeKey, TypeGraph, TypeKey};
use crate::referrer::Referrer;
use crate::result::{filter, Allow, LeakerResult};
use crate::util::{ident_name, push_once, same_name};
use crate::view::View;
use proc_macro2::TokenStream;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use syn::*;
use template_quote::{quote, quote_spanned};

/// Finds the types of an item that depend on where they are written: the entries of a
/// [`Referrer`].
///
/// Give it a whole item with an `intern_*` method, or parts of one with an `enter_*` method. Each
/// returns a [`ContextId`]: keep it for [`Referrer::visitor()`]. Then call [`Leaker::result()`].
/// To *leak* a type is to make it an entry. See the [quick start](crate#quick-start).
///
/// `A` is the [`Allow`] rule. A helper that takes a configured Leaker is
/// `fn g<A: Allow>(leaker: &mut Leaker<A>)`.
///
/// Use one Leaker for the items of one module. Its Repeater impls go to one place, and every
/// relative path that it saw must resolve there.
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
/// An error does not stop the analysis. The Leaker keeps each error once, in the order found, in
/// one flat list. The copy writes the part with the error as written:
///
/// - For a type whose own path cannot be leaked, the part is that path. The types in its
///   arguments are analyzed on their own: `MyType<'_, Rel>` gives an error for `MyType`, then the
///   entry `Rel`.
/// - For a type in a bound or a where predicate, a type that holds a bare identifier that may be a
///   const, a reference with a named lifetime, or a type that holds a trait object which takes its
///   lifetime from it, the part is the whole type. Nothing in it is leaked, and the error lists all
///   its relative paths ([`ErrorItem::paths`]).
///
/// If a closure or the [`Allow`] rule panics, the scope records nothing, and the panic goes on.
/// Accept the errors with [`LeakerResult::allow()`] or [`LeakerResult::ignore()`].
///
/// # Limitations
///
/// The Leaker reads only the item that you give it, not the definitions of the paths in it. It
/// reports what would not compile in the copy, and the `help` of each error says what to write.
///
/// - `MyType<'_>`, `impl Trait`, and a relative trait or value in a bound cannot be leaked: write
///   an absolute path. `Self: Sized` needs [`Rules::prelude()`](crate::Rules::prelude).
/// - A type of another crate (`::foo::X`) is an entry, unless the Leaker's rule allows its path
///   ([`Rules::external_crates()`](crate::Rules::external_crates)): then the copy writes it as
///   written. `::std` and `::alloc` are crates like any other, and a `::core` path is never an
///   entry. Whatever the rule:
///   - A type that holds a type that depends on where it is written (`::foo::W<Rel>`) is an entry.
///   - One in a bound or a where predicate that names a generic (`T: Into<::foo::W<T>>`,
///     `::foo::W<T>: Clone`), or that a bound needs as written, is no entry: rustc could not prove
///     the bound for the Repeater impl (E0277). Its crate must resolve in crate B.
/// - A shadowed primitive name (`type u8 = i32;`): emit [`Leaker::assertions()`].
/// - Values and function bodies are read only with [`Leaker::check_values`] and
///   [`Leaker::check_bodies`].
/// - A `mod` item is an error ([`Reason::Unreadable`]): analyze each of its items with its own
///   call. A `mod` inside a type (an array length) is read with the type.
/// - A type that names the `Self` of a nested struct, enum or union, with a generic whose bound
///   names that `Self` (`Rel2<Self, T>` with `T: AsRef<Self>`): see [`Reason::NestedSelfInBounds`].
///
/// ## Outlives bounds
///
/// The Leaker adds the outlives bounds that the references around an entry imply (`T: 'a` for
/// `Rel<T>` in `&'a Rel<T>`), but not those that a path implies: write them
/// (`fn f<'a, T: 'a>(x: Rel<'a, T>)` for `struct Rel<'a, T>(&'a T)`), or the Repeater impl fails
/// (E0309). Give [`Referrer::visitor()`] the whole item, so that it sees the same bounds. For
/// `Self: 'a`, see [`Context::self_outlives`](crate::Context::self_outlives).
///
/// ## Binder lifetimes
///
/// If the output of a `for<'x> ..` signature names `'x`, an input must keep `'x` as written (E0581,
/// E0582). So the whole function pointer or trait object is leaked, and in a bound this is an
/// error. Keep `'x` in an input that is not leaked: `for<'x> Fn(&'x u8, Rel<'x>) -> &'x u8`.
///
/// ## Elided lifetimes
///
/// An elided output lifetime (`-> &u8`) comes from the one lifetime of the inputs. A leaked type
/// whose context has a lifetime (`Wrap<T>` with `T: 'a`) adds it to the inputs of the copy
/// (E0106), so the whole function pointer or trait object is leaked instead. This is an error in a
/// bound, in `impl Fn(..)`, in a trait object under an elided reference, and in a method without a
/// reference receiver. Name the lifetimes (`fn g<'x>(x: &'x u8, y: Wrap<T>) -> &'x u8`), write
/// the lifetime that a path hides (`Ref<'_, u8>`), or box the trait object.
///
/// ## Trait objects
///
/// A `dyn Trait` without a lifetime bound takes its bound from where it is written, which the
/// Leaker cannot see (E0478). So the reference or the path around it is leaked instead. Under an
/// elided lifetime (`&dyn Tr`) this is an error: write `&'a dyn Tr`.
///
/// ## Method lifetimes
///
/// If the return type of a method names a lifetime `'b` that only a leaked type in the inputs
/// names, `'b` is late-bound in the trait but early-bound in the copy (E0195). In the methods of
/// traits and trait impls this is an error ([`Reason::UnconstrainedLifetime`]). Keep `'b` in an
/// input that is not leaked: `fn f<'b>(&'b self, x: Wrap<'b>) -> &'b u8`.
///
/// ## Bounds that name leaked types
///
/// While rustc checks the bounds of an item, it does not replace a path through the Repeater by
/// its type. So a leaked type in a bound or a where predicate is an error
/// ([`Reason::UnprovableBound`]) if a bound in its context names a leaked type (E0277, E0275,
/// E0284), `Self` that the copy writes through the Repeater, `Self::A` (E0277, E0271), or an
/// associated type that rustc replaces by its type (`<u8 as Tr>::A`; `T::Item` that a predicate
/// fixes, or where a bound of `T` is a trait outside the standard library; E0275).
///
/// If another bound of `T` fixes an associated type (`T: Iterator<Item = u8>`), the contexts leave
/// out the bounds that name `T` in their arguments (`T: Fn(Wrap<T>)`): the Repeater impl fails
/// (E0277) if the leaked type needs one. The context of an entry that names `Self` keeps them,
/// and in a predicate such an entry is an error (`T: Fn(Wrap<T>)`, not `T: From<T>`): `Wrap` may
/// be a type alias of a projection, and the Leaker cannot tell an alias from a struct.
///
/// A where predicate on a leaked type (`Pair<T, T>: Clone`) stays in the contexts, but the copy
/// has it only through the Repeater. If no impl proves it without the predicate (a derived `Clone`
/// needs `T: Clone`), the copy fails (E0277). The Leaker does not report this.
///
/// ## Bare identifiers and consts
///
/// A bare identifier argument (`MyType<MY_CONST>`) may be a type or a const. The Leaker leaks the
/// type that holds it as a whole, which is right for both. In a trait bound, and where the holder
/// has an elided lifetime or `impl Trait`, it leaks the identifier alone, which fails for a const
/// (E0573, E0747). Where another cause keeps the holder from being leaked, the copy writes the
/// holder as written, with the error [`Reason::MaybeConst`]. Write an absolute path
/// (`MyType<{ ::my_crate::MY_CONST }>`) or an [allowed](Allow) name.
pub struct Leaker<A = ()> {
    /// Reads values too, if `true` (default `false`): the values of consts and statics, and the
    /// values and defaults of associated consts. Set it before the calls that it should affect.
    ///
    /// In a value, a type in a type position (a cast, a `let`, a closure parameter, the arguments
    /// of a path or a method call) is leaked like any type of the item. Any other relative path is
    /// an error, unless the rule allows it ([`Rules::values()`](crate::Rules::values) allows all)
    /// or it starts with a generic, `Self` or a name that the value declares. Also errors: a macro
    /// whose arguments are not expressions (except `cfg!`, `stringify!`, `matches!`), a macro in
    /// item position, a glob import, and syntax that `syn` does not parse. Without the flag, the
    /// Leaker does not report these, and the copy may not compile.
    ///
    /// Read whatever the flag: enum discriminants, the defaults of const parameters, the const
    /// arguments of bounds, and all that is inside a type (an array length, a const argument).
    ///
    /// ```ignore
    /// // With `check_values`: `Rel` is an entry, and `helper` is an error.
    /// pub trait Tr { const LEN: usize = helper::<Rel>(); }
    /// ```
    pub check_values: bool,
    /// Reads function bodies too, if `true` (default `false`): the default bodies of trait
    /// methods, the bodies of impl methods, and the bodies of the functions that a scope visits.
    /// It reads a body as [`check_values`](Leaker::check_values) reads a value. Set it before the
    /// calls that it should affect.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: ItemTrait = parse_quote! {
    ///     pub trait Tr { const N: usize = inner::N; fn f() -> usize { inner::N } }
    /// };
    /// let note_of = |values, bodies| {
    ///     let mut leaker = Leaker::new();
    ///     leaker.check_values = values;
    ///     leaker.check_bodies = bodies;
    ///     leaker.intern_trait(&item);
    ///     let errors = leaker.result().errors().to_vec();
    ///     assert_eq!(errors.len(), values as usize + bodies as usize);
    ///     errors.into_iter().map(|error| error.notes[0].1.clone()).collect::<Vec<_>>()
    /// };
    /// assert_eq!(note_of(false, false), Vec::<String>::new());
    /// // `check_values` reads the default of the const, and `check_bodies` the body of `f`.
    /// assert_eq!(
    ///     note_of(true, false),
    ///     ["it is in the default value of `N`, which the copy writes as it is"],
    /// );
    /// assert_eq!(
    ///     note_of(false, true),
    ///     ["it is in the default body of `f`, which the copy writes as it is"],
    /// );
    /// ```
    pub check_bodies: bool,
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
    /// Creates a [`Leaker`] whose rule allows nothing (`()`).
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
    /// Creates a [`Leaker`] that asks `rule` which paths the copy may write as written. A closure
    /// needs the type of its first parameter: see [`Allow`].
    pub fn with_rule(rule: A) -> Self {
        Self {
            check_values: false,
            check_bodies: false,
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

    /// Returns a compile-time check that the primitive names that the copy writes as written (`u8`
    /// in `const N: u8`, those that [`Rules::primitive()`](crate::Rules::primitive) allows, also in
    /// the part of an error) are not shadowed in crate A. Emit it in crate A. Each check has the
    /// span of its name, so rustc reports a shadowed name there. It is empty if there is none.
    #[must_use]
    pub fn assertions(&self) -> TokenStream {
        if self.assumed_primitives.is_empty() {
            return TokenStream::new();
        }
        let checks = self.assumed_primitives.iter().map(|ident| {
            let name = Ident::new(
                &format!("__type_leak_assert_{}_is_primitive", ident_name(ident)),
                ident.span(),
            );
            quote_spanned! {ident.span() =>
                fn #name(x: &#ident) -> &::core::primitive::#ident { x }
            }
        });
        quote! {
            #[allow(dead_code, non_snake_case)]
            const _: () = {
                #(#checks)*
            };
        }
    }

    /// Analyzes a struct, and returns its top-level context. `Self` is the struct
    /// ([`SelfRequirement::Type`]).
    ///
    /// A type that names `Self` is one entry for the struct. `Self::A` cannot be leaked (E0220):
    /// write `<Self as ::path::Trait>::A`.
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

    /// Analyzes an impl, and returns its top-level context. `Self` is the self type
    /// ([`SelfRequirement::Type`]).
    ///
    /// A relative trait path is an error. So is an entry in the self type whose context depends on
    /// `Self` through a bound of the impl (`impl<T> S<T> where T: AsRef<Option<Self>>`), also in an
    /// impl that [`View::visit_item()`] visits: write an absolute self type or a bound without
    /// `Self`.
    pub fn intern_impl(&mut self, item: &ItemImpl) -> ContextId {
        // The scope checks the path of the trait, and the walk analyzes its arguments.
        let trait_path = item.trait_.as_ref().map(|(_, path, _)| path);
        self.enter_impl(&item.generics, &item.self_ty, trait_path, |view| {
            view.walk_item_impl(item)
        })
        .0
    }

    /// Analyzes a trait, and returns its top-level context. `Self` is the implementor
    /// ([`SelfRequirement::Bound`]).
    ///
    /// Unlike [`enter_trait()`](Leaker::enter_trait), it knows the generic associated types (GATs)
    /// of the trait. `Self::G<'x>` adds the outlives bounds of the GAT to the context, and the copy
    /// must prove them: in an impl for another type, add `where T: 'x` or `where Self: 'x` to the
    /// method (not needed with the receiver `&'x self`). It does not see the GATs of supertraits or
    /// of `T::G<'x>`.
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
    /// returns the top-level context and the result of `f`. It does not analyze `generics`: visit
    /// their bounds in `f` with [`View::visit_type_param_bound()`] and
    /// [`View::visit_where_clause()`]. `Self` has no meaning here, so naming it is an error
    /// ([`Reason::SelfWithoutMeaning`]).
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_generics(self, generics, f)
    }

    /// Analyzes parts of the struct, enum or union `ident`, like
    /// [`enter_generics()`](Leaker::enter_generics). `Self` is the type
    /// ([`SelfRequirement::Type`]), and `generics` may name it.
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_type(self, ident, generics, f)
    }

    /// Analyzes parts of an impl with `generics` for `self_ty`, like
    /// [`enter_type()`](Leaker::enter_type). `Self` is `self_ty`.
    ///
    /// `trait_` is the trait of the impl, or `None` for an inherent impl, whose methods are not
    /// compared with a trait's ([Method lifetimes](Leaker#method-lifetimes)). This call checks the
    /// trait path (a relative path is an error), and `f` visits its arguments. The help for
    /// `Self::A` names the trait. If your macro resolves the trait by its name elsewhere, pass the
    /// path and [allow it](Allow), or pass `None` if `f` visits no method.
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        trait_: Option<&Path>,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_impl(self, generics, self_ty, trait_, f)
    }

    /// Analyzes parts of the trait `ident`, like [`enter_type()`](Leaker::enter_type). `Self` is
    /// the implementor ([`SelfRequirement::Bound`]). The GATs of the trait are not known: see
    /// [`intern_trait()`](Leaker::intern_trait).
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        Scopes::enter_trait(self, ident, generics, f)
    }

    /// Ends the analysis, and returns the [`Referrer`] with the errors. The entries that hold a
    /// type that depends on where it is written come first, then those that are only a type of
    /// another crate (`::foo::X`). Each group is sorted by context (in the order first met), then
    /// by tokens.
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
            types.reduce()
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

    /// Chooses the entries now, and removes every other type from the Leaker.
    /// [`Leaker::result()`] makes the same choice by itself: call this only to go on analyzing
    /// with the reduced Leaker.
    ///
    /// It chooses the fewest entries from which every leaked type can be built. These are leaked
    /// whole, not in parts: `&'a Rel`, a path with a lifetime or a bare identifier argument
    /// (`Wrap<SIZE>`), a type with a type in an expression (`[u8; size_of::<Big>()]`), and a
    /// trait object. One choice covers both kinds of entries (see [`Leaker::result()`]): an entry
    /// that holds a type that depends on where it is written is of the first kind.
    ///
    /// The chosen entries stay entries of their kind whatever you analyze afterwards, so a second
    /// call changes nothing. The cost: the result can have more entries than one choice at the end
    /// (`(Rel1, Rel2)` is chosen, then `Rel1` alone is an entry too). And a later scope in which a
    /// chosen entry makes the copy cyclic (where its parts would be the entries) still gets it, so
    /// that copy may not compile.
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
        let mut visitor =
            AnalyzeVisitor::new(&mut self.allow, &root, self.check_values, self.check_bodies);
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
        let top_level = self.record(&spec, finished);
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
    /// Records what the outermost scope `spec` found, and returns its top-level context.
    fn record(&mut self, spec: &ScopeSpec, finished: Finished) -> ContextId {
        let Finished {
            mut effects,
            top_levels,
            not_optional,
        } = finished;
        drop_optional(&mut effects, &not_optional.into_iter().collect());
        let mut analysis = Analysis::default();
        analysis.apply(&mut self.contexts, &effects, None);
        for ident in analysis.primitives {
            if !self
                .assumed_primitives
                .iter()
                .any(|known| same_name(known, &ident))
            {
                self.assumed_primitives.push(ident);
            }
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
        let context = spec.top_level(scope_self, spec.declared_gats());
        let top_level = self.contexts.intern(context);
        self.top_levels.insert(top_level);
        self.top_levels.extend(top_levels);
        top_level
    }
}

//! The [`Leaker`], which finds the types depending on the type context.

use crate::analyze::{Analysis, AnalyzeVisitor, CheckResult};
use crate::config::LeakerConfig;
use crate::context::{Context, ContextId, Contexts};
use crate::error::NotInternableError;
use crate::graph::{NodeKey, TypeGraph};
use crate::referrer::Referrer;
use crate::util::{generic_param_name, path_starts_with, token_names, unwrap_type_group};
use proc_macro2::{Span, TokenStream};
use safegraph::algo::max_flow::edmonds_karp_with_flows;
use safegraph::graph::Graph;
use safegraph::BTreeGraph;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use syn::punctuated::Punctuated;
use syn::*;
use template_quote::quote;

/// The entry point of this crate.
pub struct Leaker {
    /// Relative paths treated as context-free (not leaked), e.g. the names of the standard
    /// prelude (`Vec`, `Option`, ...) with [`LeakerConfig::allow_std_prelude()`], primitive types
    /// with [`LeakerConfig::allow_primitive()`], or `crate` with [`LeakerConfig::allow_crate()`].
    /// A relative path is context-free if it starts with the segments of one of them; its generic
    /// arguments are still analyzed (e.g. `Rel` in `Option<(Rel,)>`). They apply to relative paths
    /// only (see [`Leaker::allow_external_crates`] for paths with a leading `::`). Primitive names
    /// copied as they are are checked by [`Leaker::assertions()`].
    pub allowed_paths: Vec<Path>,
    /// Whether a type mentioning `Self` may be leaked as a whole (default `true`).
    ///
    /// With `true`, the Repeater impl of such a type is written for the implementor, as given by
    /// its Self requirement (see [`Referrer::context_self()`](crate::Referrer::context_self())).
    /// Set it to `false` when the Repeater impls cannot refer to `Self` (e.g. they are written
    /// for a marker type without rewriting `Self`): a type mentioning `Self` is then decomposed
    /// (only its other components are leaked), and an error is reported when it cannot be (e.g.
    /// `Rel<Self>`) or when the bounds of the generics it uses mention `Self`.
    pub self_ty_can_be_interned: bool,
    /// Whether a path with a leading `::` (e.g. `::foo::X`) is context-free (default `true`).
    ///
    /// Such a path resolves against the external crates of the crate compiling it, so it is not
    /// portable to the Referrer's crate if that crate does not depend on `foo` (or renamed it, or
    /// uses another version). With `false`, only a path starting with `::core` (always
    /// available) or `$crate` is context-free (not even one starting with an
    /// [allowed path](Leaker::allowed_paths), which applies to relative paths only), and the
    /// others are treated like relative paths: leaked in a type (see
    /// [`Referrer::external_crates_require()`](crate::Referrer::external_crates_require()) and
    /// [`Referrer::must()`](crate::Referrer::must())), and
    /// [`NotInternableError::NotPortable`] in a trait bound, a value or the type of a const
    /// parameter. The Referrer tells which crates it can refer to (see
    /// [`Referrer::with_available_external_crates()`](crate::Referrer::with_available_external_crates())).
    pub allow_external_crates: bool,
    /// The types found by [`Leaker::intern_with()`] and their composition.
    types: TypeGraph,
    /// The nodes of the types to be interned, computed from `types` by [`Leaker::reduce_roots()`]
    /// (`None` if not computed yet, or outdated).
    reachable_types: Option<Vec<NodeKey>>,
    /// The contexts of the types found (see [`Referrer::context()`](crate::Referrer::context())),
    /// and the generics of the calls of [`Leaker::intern_with()`].
    contexts: Contexts,
    /// The top-level contexts of the successful calls of [`Leaker::intern_with()`] (see there),
    /// which the [`Referrer`] keeps even without an entry.
    top_levels: BTreeSet<ContextId>,
    /// The number of the successful calls of [`Leaker::intern_with()`].
    calls: usize,
    /// Names of primitive types which were assumed to be the primitive types (not shadowed), as
    /// they are copied as they are (see [`Leaker::assertions()`]).
    assumed_primitives: Vec<Ident>,
}

impl Default for Leaker {
    fn default() -> Self {
        Self::new()
    }
}

impl Leaker {
    /// Initialize with [`ItemStruct`], and returns its top-level context (see
    /// [`Leaker::intern_with()`]).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let test_struct: ItemStruct = parse_quote!(
    ///     pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
    ///         field1: MyType1,
    ///         field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
    ///         field3: &'a (T1, T2),
    ///     }
    /// );
    /// let (leaker, context) = Leaker::from_struct(&test_struct).unwrap();
    /// ```
    pub fn from_struct(
        input: &ItemStruct,
    ) -> std::result::Result<(Self, ContextId), NotInternableError> {
        let mut leaker = Self::new();
        let context = leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_struct(input);
        })?;
        Ok((leaker, context))
    }

    /// Initialize with [`ItemEnum`], and returns its top-level context (see
    /// [`Leaker::intern_with()`]).
    pub fn from_enum(
        input: &ItemEnum,
    ) -> std::result::Result<(Self, ContextId), NotInternableError> {
        let mut leaker = Self::new();
        let context = leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_enum(input);
        })?;
        Ok((leaker, context))
    }

    /// Initialize with [`ItemUnion`], and returns its top-level context (see
    /// [`Leaker::intern_with()`]).
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let u: ItemUnion = parse_quote! {
    ///     pub union MyUnion<T: ::core::marker::Copy> {
    ///         a: T,
    ///         b: MyType,
    ///     }
    /// };
    /// let (leaker, _context) = Leaker::from_union(&u).unwrap();
    /// let referrer = leaker.finish();
    /// assert_eq!(referrer.iter().collect::<Vec<_>>(), [&parse_quote!(MyType)]);
    /// ```
    pub fn from_union(
        input: &ItemUnion,
    ) -> std::result::Result<(Self, ContextId), NotInternableError> {
        let mut leaker = Self::new();
        let context = leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_union(input);
        })?;
        Ok((leaker, context))
    }

    /// Build a [`Leaker`] from a given trait, and returns its top-level context (see
    /// [`Leaker::intern_with()`]).
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let s: ItemTrait = parse_quote!{
    ///     pub trait MyTrait<T, U> {
    ///         fn func(self, t: T) -> U;
    ///     }
    /// };
    /// let _ = Leaker::from_trait(&s);
    /// ```
    pub fn from_trait(
        input: &ItemTrait,
    ) -> std::result::Result<(Self, ContextId), NotInternableError> {
        let mut leaker = Self::new();
        let context = leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_trait(input);
        })?;
        Ok((leaker, context))
    }

    /// Initialize [`Leaker`] from a module, and returns its top-level context (see
    /// [`Leaker::intern_with()`]), which is that of no generics.
    ///
    /// The module has no generics, and every item in it (structs, enums, unions, traits, type
    /// aliases, functions, impls and nested modules) is a nested item, so its generics are local
    /// generics, which the types depending on them are interned with (see
    /// [`Referrer::generics()`]). The bodies of functions and the values of consts are not
    /// analyzed. A module without content (`mod m;`) has nothing to analyze.
    ///
    /// The items declared in the module are copied with it, so paths to them are not leaked,
    /// like Rust resolves them: a name declared directly in the current module (not shadowed by
    /// a generic parameter), `self::` or `super::` followed by a name declared in that module
    /// (without leaving the analyzed module), and the names imported by `use` with an absolute
    /// path (starting with `::`). A type mentioning them is decomposed instead of being interned
    /// as a whole (e.g. only `Rel` is interned for `S<Rel>`, where `S` is declared in the
    /// module). Other relative paths (including the names imported by `use super::X` or
    /// `use crate::X`) are leaked.
    ///
    /// A relative path resolves in the module where it is written, so a leaked type depending on
    /// one (e.g. `Rel` in `mod m` and in `mod m::inner`, where `use super::super::Rel;` may
    /// import another one), or whose context does (what `Self` means, or a bound of its
    /// generics, mentioning a path which is not absolute, `crate::` or `$crate`), or which is a
    /// type macro, is keyed by the path of that module, relative to `input` (`[]` for
    /// `input` itself, as opposed to a type which depends on no module), together with its generics (see [`Referrer::context_module()`]). The
    /// Repeater's implementation of the type must be written inside that module, e.g. an
    /// attribute macro on `mod m` can emit the implementations of the types of `m::inner` inside
    /// it. A type which does not depend on its module (e.g. one with only absolute paths) is
    /// shared by all the modules, in the module `[]`.
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let m: ItemMod = parse_quote!{
    ///     mod m {
    ///         pub struct S<T> { x: Rel<T> }
    ///         pub type A = Rel2;
    ///     }
    /// };
    /// let (leaker, context) = Leaker::from_module(&m).unwrap();
    /// let referrer = leaker.finish();
    /// assert_eq!(referrer.iter().count(), 2);
    /// // The module has no generics.
    /// assert!(referrer.context_generics(context).params.is_empty());
    /// ```
    pub fn from_module(
        input: &ItemMod,
    ) -> std::result::Result<(Self, ContextId), NotInternableError> {
        let mut leaker = Self::new();
        let context = leaker.intern_with(&Generics::default(), |visitor| {
            visitor.visit_item_mod(input);
        })?;
        Ok((leaker, context))
    }

    /// Initialize empty [`Leaker`] with no generics.
    pub fn new() -> Self {
        Self {
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
            allow_external_crates: true,
            types: TypeGraph::default(),
            reachable_types: None,
            contexts: Contexts::default(),
            top_levels: BTreeSet::new(),
            calls: 0,
            assumed_primitives: Vec::new(),
        }
    }

    /// Initialize [`Leaker`] from configuration.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allowed_paths: config.allowed_paths,
            self_ty_can_be_interned: config.self_ty_can_be_interned,
            allow_external_crates: config.allow_external_crates,
            types: TypeGraph::default(),
            reachable_types: None,
            contexts: Contexts::default(),
            top_levels: BTreeSet::new(),
            calls: 0,
            assumed_primitives: Vec::new(),
        }
    }

    fn assume_primitive(&mut self, ident: &Ident) {
        if !self.assumed_primitives.iter().any(|p| p == ident) {
            self.assumed_primitives.push(ident.clone());
        }
    }

    /// Items to emit in the Leaker's type context, which check the assumptions made by the
    /// Leaker at compile time.
    ///
    /// The names of primitive types (e.g. `u8` in `const N: u8`, or in a type when allowed by
    /// [`LeakerConfig::allow_primitive()`]) are copied as they are, assuming that they are not
    /// shadowed in the Leaker's type context (e.g. by `type u8 = i32;`), which cannot be known
    /// without name resolution. These items fail to compile (with "mismatched types") if a name
    /// is shadowed, instead of the copies silently referring to another type.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait MyTrait<const N: u8> {});
    /// let (leaker, _) = Leaker::from_trait(&item).unwrap();
    /// assert!(leaker
    ///     .assertions()
    ///     .to_string()
    ///     .contains("__type_leak_assert_u8_is_primitive"));
    /// ```
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

    /// Returns `true` if `path` does not depend on the type context, i.e. it is absolute, starts
    /// with `$crate`, or is relative and starts with one of [`Leaker::allowed_paths`]. Unless
    /// [`Leaker::allow_external_crates`], an absolute path is context-free only if it starts with
    /// `::core`.
    pub(crate) fn is_context_free_path(&self, path: &Path) -> bool {
        (path.leading_colon.is_some()
            && (self.allow_external_crates
                || matches!(path.segments.first(), Some(seg) if seg.ident == "core")))
            || matches!(
                path.segments.iter().next(),
                Some(PathSegment {
                    ident,
                    arguments: PathArguments::None,
                }) if ident == "$crate"
            )
            || (path.leading_colon.is_none()
                && self
                    .allowed_paths
                    .iter()
                    .any(|allowed| path_starts_with(path, allowed)))
    }

    /// Intern AST elements using a visitor callback.
    ///
    /// `generics` are the generics of the Leaker, which the Repeater's implementations declare
    /// (the ones a type uses, see [`Referrer::generics()`](crate::Referrer::generics())). A type
    /// found is keyed by its tokens and its context: the generics it uses (resolved through
    /// their bounds and where predicates, with those of `generics` as the outermost scope) and
    /// what `Self` means if it mentions it. So the same type found by several calls (whatever
    /// the other generics of the calls are) is shared if its contexts are equal, and it is a
    /// separate entry otherwise (see [`Referrer::context()`](crate::Referrer::context())).
    /// Nothing is added if an error is returned.
    ///
    /// Returns the top-level context of the call: the [`ContextId`] of `generics` (in the canonical
    /// form of a context, so without the defaults, and the bounds sorted and moved to the
    /// parameters), with no `Self` and no module. It identifies the generics of the call itself
    /// (e.g. of the trait), whereas the entries are in the contexts of the generics they actually
    /// use (see [`Referrer::context()`](crate::Referrer::context())), which may differ: `Rel` in
    /// `trait Tr<T> { fn f(&self) -> Rel; }` is in the context of no generics. The two are the
    /// same context if the generics (with no `Self` and no module) are exactly equal (e.g. the
    /// generics of `Wrap<T>` in `struct S<T> { a: Wrap<T> }` are those of `S`), and a call with
    /// the same generics returns the same id as another one. Ids are assigned in the order of
    /// the first appearance of the contexts (that of the call, after those of the types it
    /// found), and they are the ones the [`Referrer`] uses.
    ///
    /// The [`Referrer`] has the top-level context of every successful call, also if no entry is
    /// in it (see
    /// [`Referrer::contexts()`](crate::Referrer::contexts())): its generics are given by
    /// [`Referrer::context_generics()`](crate::Referrer::context_generics()). A failed call
    /// takes no id.
    ///
    /// `Self` is the implementor of the visited trait, or the visited struct, enum, union or the
    /// self type of the visited impl (see [`SelfRequirement`](crate::SelfRequirement)). Outside
    /// any item (e.g. if only a type is visited), it is the `n`-th call (counting the successful
    /// calls from 0).
    ///
    /// The default of a type parameter of `generics` is not analyzed along with the item, as it
    /// is not copied there: a default is written in the Leaker's type context, so it is rewritten
    /// on its own by [`Referrer::complete_args()`]. It is analyzed with the parameters before it
    /// as the generics of the Leaker (and the where predicates which do not mention the others),
    /// so the types in it use only the earlier parameters, and cannot depend on `Self`: it is an
    /// error if a type of the default which is leaked mentions `Self`. The defaults of const
    /// parameters are analyzed along with the item.
    pub fn intern_with<'ast, F>(
        &mut self,
        generics: &Generics,
        f: F,
    ) -> std::result::Result<ContextId, NotInternableError>
    where
        F: FnOnce(&mut dyn syn::visit::Visit<'ast>),
    {
        let mut visitor = AnalyzeVisitor::new(self, self.calls, generics);
        f(&mut visitor);
        let mut effects = visitor.finish();
        for (index, param) in generics.params.iter().enumerate() {
            if let GenericParam::Type(TypeParam {
                default: Some(default),
                ..
            }) = param
            {
                let root = without_defaults(generics, index);
                effects.extend(AnalyzeVisitor::analyze_default(self, default, &root));
            }
        }
        let contexts = self.contexts.len();
        let mut analysis = Analysis::default();
        analysis.apply(&mut self.contexts, effects, None);
        if !analysis.errors.is_empty() {
            self.contexts.truncate(contexts);
            return Err(if self.allow_external_crates {
                NotInternableError::RelativePath(analysis.errors)
            } else {
                NotInternableError::NotPortable(analysis.errors)
            });
        }
        for ident in &analysis.primitives {
            self.assume_primitive(ident);
        }
        self.calls += 1;
        if !analysis.types.is_empty() {
            self.types.merge(analysis.types);
            self.reachable_types = None;
        }
        // The generics of the call, in the same context as an entry which requires exactly them.
        let top_level = self.contexts.intern(Context::new(generics.clone(), None));
        self.top_levels.insert(top_level);
        Ok(top_level)
    }

    /// Checks whether the given `ty` must be interned, and whether it can be. See [`CheckResult`].
    pub fn check(&self, ty: &Type, generics: &Generics) -> CheckResult {
        AnalyzeVisitor::check(self, ty, generics)
    }

    /// Finish building the [`Leaker`] and convert it into [`Referrer`].
    ///
    /// The types are reduced by [`Leaker::reduce_roots()`] (called here if needed), and sorted by
    /// their context (the contexts are numbered in the order of their first appearance) and tokens, so the result (including the index of each type) is
    /// deterministic. The [`ContextId`]s are those returned by [`Leaker::intern_with()`], not
    /// renumbered: the Referrer has the top-level contexts of the calls and the contexts of the
    /// types kept, and no other.
    pub fn finish(mut self) -> Referrer {
        self.reduce_roots();
        let keys = self.reachable_types.take().unwrap_or_default();
        let types = self.types;
        let contexts = self.contexts;
        let mut context_map: BTreeMap<ContextId, Context> = self
            .top_levels
            .iter()
            .map(|context| (*context, contexts.get(*context).clone()))
            .collect();
        let mut entries = Vec::new();
        for key in keys {
            let context = key.context;
            let ctx = context_map
                .entry(context)
                .or_insert_with(|| contexts.get(context).clone());
            let must = types.hard.contains(&key);
            let crates = external_crates(&key.ty.ty, ctx);
            entries.push((context, key.ty.ty, must, crates));
        }
        Referrer::new(entries, context_map)
    }

    /// Reduce nodes to decrease the cost of the [`Leaker`]'s implementation.
    ///
    /// # Algorithm
    ///
    /// Consider the following [`Leaker`] input:
    ///
    /// ```ignore
    /// pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
    ///     field1: MyType1,
    ///     field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
    ///     field3: &'a (T1, T2),
    /// }
    /// ```
    ///
    /// When initialized with [`Leaker::from_struct()`], the [`Leaker`] analyzes the AST and
    /// constructs a DAG representing all internable types and their dependency relations:
    ///
    /// ![before_graph](https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/resources/before_graph.svg)
    ///
    /// **Red** nodes are those whose own path depends on the type context
    /// ([`CheckResult::is_own`] by [`Leaker::check()`]), meaning the type literally depends on the
    /// type context and must be interned.
    ///
    /// This algorithm chooses the smallest set of nodes to intern such that every root type
    /// (annotated with ★) can still be expressed using the interned nodes, i.e. such that every
    /// path from a root to a **red** node in the graph contains an interned node (a root, or a
    /// **red** node itself, is on its own path). Since a Repeater implementation is needed for
    /// each node interned, this minimizes the number of the implementations. Either of the
    /// following is chosen for each **red** node, whichever is cheaper in total:
    ///
    /// - Intern the **red** node itself (non-red nodes either do not need to be interned or can be
    ///   reconstructed from the interned ones).
    /// - Skip interning it and instead intern a common ancestor, when one type covers several
    ///   **red** nodes at once.
    ///
    /// Only the nodes of the graph are candidates, and a node is not interned instead of a
    /// reference with a lifetime containing it (`Rel` instead of `&'a Rel`), as the copy of the
    /// reference needs `Rel: 'a`, which cannot be proved of the type interned in its place: such
    /// a reference is interned as a whole, like a **red** node. A trait object is the exception:
    /// the reference to it needs nothing of its arguments, so `Rel` is interned instead of `&'a
    /// dyn ::path::Tr<Item = Rel>`. So is any type which cannot be
    /// copied around the types interned in place of its components, so they are not nodes of the
    /// graph (its causes are its own, see [`CheckResult::is_own`]): a path with a lifetime
    /// argument or a reference with a lifetime of the generics, and a trait object without a
    /// lifetime bound as an argument or the referent which has to be
    /// interned itself (`Ref<'a, dyn RelTrait>` or `&'a dyn RelTrait`, where the trait object
    /// defaults to `'a` but would be `'static` interned alone; such a trait object is never
    /// interned alone, but what is inside it is, e.g. `Rel` of `Ref<'a, dyn ::path::Tr<Item =
    /// Rel>>`), a
    /// path with a bare identifier written as an argument (`Wrap<SIZE>`, where syn parses a const
    /// as a type, so it may be one, and it cannot be interned alone) and a type with a type in an
    /// expression (`[u8; size_of::<Big>()]`, where the copy would have a projection in an
    /// anonymous constant). A bare identifier is told from a type by its position only, not by
    /// its name: the path is interned as a whole if it can be, and otherwise (e.g. `Wrap<'_,
    /// SIZE>`, with a lifetime which cannot be interned, or `S<SIZE>`, where `S` is declared in the
    /// module analyzed by [`Leaker::from_module()`]) the identifier is interned as a type like
    /// any other, which does not compile if it is a const. Another position (`Wrap<(SIZE,)>`,
    /// `Wrap<[SIZE; 3]>`, `Wrap<(SIZE)>`, `Wrap<Item = SIZE>`, a bound `Sup<SIZE>` of the Leaker)
    /// is a type, or has no type to be interned as a whole.
    ///
    /// For example, `Vec<(Rel,)>`, `Option<(Rel,)>`, `(Rel, u8)` and `Box<(Rel,)>` in fields are
    /// covered by interning `Rel` only, and `(A, B, C)` used twice by interning the tuple only
    /// (`Vec<Rel>`, with a bare `Rel`, is interned as a whole).
    ///
    /// This is a minimum vertex cut between the roots and the **red** nodes, solved exactly as a
    /// maximum flow (Edmonds-Karp): each node `v` is split into `v_in -> v_out` of capacity 1, each
    /// edge of the graph `u -> v` becomes `u_out -> v_in` of infinite capacity, and a super source
    /// is connected to the roots and the **red** nodes to a super sink. When several cuts have
    /// the minimum size, the one closest to the sink is taken (the nodes which can reach the sink
    /// in the residual network of the maximum flow form its sink side), which prefers the inner
    /// types (`Rel` to `Vec<(Rel,)>` when that is all there is), so that the result is
    /// deterministic.
    ///
    /// The search of the maximum flow is repeated for each unit of flow, over the whole network,
    /// so it takes quadratic time in the size of the item. So a **red** root, which is in every
    /// minimum cut, is interned without a flow, and the network is split into the weakly
    /// connected components of the graph (an item has many independent fields), each solved on
    /// its own with the same result.
    ///
    /// The final result looks like:
    ///
    /// ![after_graph](https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/resources/after_graph.svg)
    pub fn reduce_roots(&mut self) {
        if self.reachable_types.is_none() {
            self.reachable_types = Some(self.build_graph_and_reduce());
        }
    }

    /// Returns the types to be interned, sorted by their context and tokens.
    ///
    /// This is a minimum vertex cut between the roots and the **red** nodes (see
    /// [`Leaker::reduce_roots()`]), found by a maximum flow.
    fn build_graph_and_reduce(&self) -> Vec<NodeKey> {
        let types = &self.types;
        let mut keys: Vec<&NodeKey> = types.keys().collect();
        keys.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
        let index: HashMap<&NodeKey, usize> =
            keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let n = keys.len();
        let children: Vec<Vec<usize>> = keys
            .iter()
            .map(|key| types.children(key).iter().map(|c| index[c]).collect())
            .collect();
        let red: Vec<bool> = keys
            .iter()
            .map(|k| types.must_intern.contains(*k))
            .collect();
        let root: Vec<bool> = keys.iter().map(|k| types.roots.contains(*k)).collect();
        // The types which are red or contain a red type.
        let mut has_red = red.clone();
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, children) in children.iter().enumerate() {
            for &child in children {
                parents[child].push(i);
            }
        }
        let mut stack: Vec<usize> = (0..n).filter(|&i| red[i]).collect();
        while let Some(i) = stack.pop() {
            for &parent in &parents[i] {
                if !has_red[parent] {
                    has_red[parent] = true;
                    stack.push(parent);
                }
            }
        }

        // A red root is in every minimum cut, as the only node of the path `source -> v_in ->
        // v_out -> sink`, so it is in the one closest to the sink as well. It is cut without a
        // flow, and removed from the network (the paths through it are cut).
        let mut kept: Vec<NodeKey> = (0..n)
            .filter(|&i| red[i] && root[i])
            .map(|i| keys[i].clone())
            .collect();

        // The edges of the network between the other types (a type which has red components goes
        // to the sink if it is red, or if its components cannot stand for it), without those that
        // cannot be on a path from a root to the sink.
        let mut next: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut to_sink = vec![false; n];
        for i in 0..n {
            if !has_red[i] || (red[i] && root[i]) {
                continue;
            }
            to_sink[i] = red[i];
            let whole = self.needs_parent(keys[i]);
            for &child in &children[i] {
                if !has_red[child] {
                    continue;
                }
                if whole {
                    // The child cannot stand for the parent, which is to be interned as a whole
                    // like a red type, and not through the child.
                    to_sink[i] = true;
                } else if !(red[child] && root[child]) {
                    next[i].push(child);
                }
            }
        }
        let mut reached = vec![false; n];
        let mut stack: Vec<usize> = (0..n)
            .filter(|&i| root[i] && has_red[i] && !red[i])
            .collect();
        for &i in &stack {
            reached[i] = true;
        }
        while let Some(i) = stack.pop() {
            for &child in &next[i] {
                if !reached[child] {
                    reached[child] = true;
                    stack.push(child);
                }
            }
        }

        // The flows of weakly connected components are independent (they share only the source
        // and the sink), and so are their minimum cuts closest to the sink, so each component is
        // solved on its own: the maximum flow takes a search of the whole network per unit of
        // flow, so it is quadratic in the size of the network.
        let mut component: Vec<usize> = (0..n).collect();
        for i in (0..n).filter(|&i| reached[i]) {
            for &child in &next[i] {
                let (a, b) = (find(&mut component, i), find(&mut component, child));
                component[a.max(b)] = a.min(b);
            }
        }
        let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in (0..n).filter(|&i| reached[i]) {
            members.entry(find(&mut component, i)).or_default().push(i);
        }
        let mut local = vec![0u32; n];
        for members in members.values() {
            for (l, &i) in members.iter().enumerate() {
                local[i] = l as u32;
            }
            // The nodes of the network: `v_in` is `2 * v`, `v_out` is `2 * v + 1`.
            let m = members.len() as u32;
            let (source, sink) = (2 * m, 2 * m + 1);
            // Larger than the sum of all the capacities of the nodes, so it is never the bottleneck.
            let infinite = m + 1;
            let mut edges: Vec<(u32, u32, u32)> = Vec::new();
            for &i in members {
                let l = local[i];
                edges.push((2 * l, 2 * l + 1, 1));
                for &child in &next[i] {
                    edges.push((2 * l + 1, 2 * local[child], infinite));
                }
                if root[i] {
                    edges.push((source, 2 * l, infinite));
                }
                if to_sink[i] {
                    edges.push((2 * l + 1, sink, infinite));
                }
            }
            let reaches_sink = sink_side(2 * m + 2, &edges, source, sink);
            // The cut is the nodes whose edge `v_in -> v_out` goes from the side which cannot
            // reach the sink to the side which can. The set of the nodes which can reach the sink
            // in the residual network is the smallest sink side of all the minimum cuts, so this
            // cut is the minimum cut closest to the sink, i.e. the one made of the innermost types
            // (a component is preferred to the type containing it when either of them cuts the
            // same paths).
            kept.extend(
                members
                    .iter()
                    .filter(|&&i| {
                        let l = local[i] as usize;
                        reaches_sink[2 * l + 1] && !reaches_sink[2 * l]
                    })
                    .map(|&i| keys[i].clone()),
            );
        }

        // Sort for deterministic output.
        kept.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
        kept
    }
}

impl Leaker {
    /// Returns `true` if `child` is not interned instead of `parent`, because the copy of
    /// `parent` is well-formed only with a bound which the copy of `child` may not have: a
    /// reference with a lifetime requires its referent to outlive it (e.g. `Rel<T>: 'a` for
    /// `&'a Rel<T>`, or `Rel: 'static`), which holds for the type itself but cannot be proved of
    /// the type which is interned in its place. The context of the referent carries the bound of a
    /// generic lifetime now (it has the lifetimes around it, see `Enclosing`), so this is no longer
    /// needed for those: it is kept, so that a reference stays one entry as it did, and it is
    /// needed for `'static`, which no context carries.
    ///
    /// A trait object (also inside parentheses and groups) is not affected: the copy keeps it in
    /// the reference, so it keeps the bound it defaults to (`'a` for `&'a dyn Trait`), and its
    /// arguments, which must outlive the lifetime as well (`dyn Tr<X> + 'b: 'a` requires `X: 'a`),
    /// are the components, whose contexts carry the bound. If the trait object has to be interned
    /// itself, it cannot be alone, as its default would be `'static`: the analysis makes the
    /// reference the type to intern then, so it has no components (see `Inherits::Reference`).
    /// With `'static`, the default is the same alone.
    fn needs_parent(&self, parent: &NodeKey) -> bool {
        match unwrap_type_group(&parent.ty.ty) {
            Type::Reference(TypeReference {
                lifetime: Some(_),
                elem,
                ..
            }) => !matches!(unwrap_type_group(elem), Type::TraitObject(_)),
            _ => false,
        }
    }
}

/// Solves the maximum flow from `source` to `sink` in the network of `nodes` nodes and `edges`
/// (`(from, to, capacity)`), and returns the nodes which can reach the sink in the residual
/// network.
fn sink_side(nodes: u32, edges: &[(u32, u32, u32)], source: u32, sink: u32) -> Vec<bool> {
    // The edges are keyed by their number in `edges`, which tells their capacities.
    let mut network = BTreeGraph::<u32, u32>::default();
    for node in 0..nodes {
        network.insert_node(node).unwrap();
    }
    for (eix, (from, to, _)) in edges.iter().enumerate() {
        network.insert_edge(eix as u32, [*from, *to]).unwrap();
    }
    let (_, flows) = edmonds_karp_with_flows(&network, source, sink, |eix| edges[*eix as usize].2);

    // The nodes which can reach the sink in the residual network, found by a BFS backwards.
    // An edge `from -> to` of the network is also usable as `to -> from` if it carries flow
    // (cancelling it), and as `from -> to` if it has spare capacity.
    let mut into: Vec<Vec<u32>> = vec![Vec::new(); nodes as usize];
    for (eix, (from, to, capacity)) in edges.iter().enumerate() {
        let flow = flows.get(&(eix as u32)).copied().unwrap_or(0);
        if flow < *capacity {
            into[*to as usize].push(*from);
        }
        if flow > 0 {
            into[*from as usize].push(*to);
        }
    }
    let mut reaches_sink = vec![false; nodes as usize];
    reaches_sink[sink as usize] = true;
    let mut queue = vec![sink];
    while let Some(node) = queue.pop() {
        for &pred in &into[node as usize] {
            if !reaches_sink[pred as usize] {
                reaches_sink[pred as usize] = true;
                queue.push(pred);
            }
        }
    }
    reaches_sink
}

/// Returns the representative of the set of `i` in the disjoint sets `component` (each element
/// points to another of its set, or to itself if it is the representative).
fn find(component: &mut [usize], mut i: usize) -> usize {
    while component[i] != i {
        component[i] = component[component[i]];
        i = component[i];
    }
    i
}

/// Returns the first `n` parameters of `generics`, without their defaults, and the where
/// predicates which do not mention the other parameters (matched by name).
fn without_defaults(generics: &Generics, n: usize) -> Generics {
    let params: Punctuated<GenericParam, Token![,]> = generics
        .params
        .iter()
        .take(n)
        .cloned()
        .map(|mut param| {
            match &mut param {
                GenericParam::Type(tp) => {
                    tp.eq_token = None;
                    tp.default = None;
                }
                GenericParam::Const(cp) => {
                    cp.eq_token = None;
                    cp.default = None;
                }
                GenericParam::Lifetime(_) => (),
            }
            param
        })
        .collect();
    let later: HashSet<String> = generics
        .params
        .iter()
        .skip(n)
        .map(generic_param_name)
        .collect();
    let predicates: Punctuated<WherePredicate, Token![,]> = generics
        .where_clause
        .iter()
        .flat_map(|wc| wc.predicates.iter())
        .filter(|predicate| {
            let mut names = HashSet::new();
            token_names(quote!(#predicate), &mut names);
            names.is_disjoint(&later)
        })
        .cloned()
        .collect();
    let has_params = !params.is_empty();
    Generics {
        lt_token: if has_params {
            Some(Default::default())
        } else {
            None
        },
        params,
        gt_token: if has_params {
            Some(Default::default())
        } else {
            None
        },
        where_clause: if predicates.is_empty() {
            None
        } else {
            Some(WhereClause {
                where_token: Default::default(),
                predicates,
            })
        },
    }
}

/// Returns the sorted names of the external crates (other than `core`) named by the absolute
/// paths (`::name::..`) in `ty` and `context` (the generics and what `Self` means), without
/// `r#` (e.g. `async` for `::r#async::X`).
fn external_crates(ty: &Type, context: &Context) -> Vec<String> {
    use crate::context::SelfRequirement;
    use proc_macro2::TokenTree;
    use syn::ext::IdentExt;
    use syn::visit::Visit;
    #[derive(Default)]
    struct Finder(BTreeSet<String>);
    impl Finder {
        /// Scans the tokens of a macro for `::` followed by a name at the start of a path.
        fn scan(&mut self, tokens: TokenStream) {
            let mut prev: Option<TokenTree> = None;
            let mut iter = tokens.into_iter().peekable();
            while let Some(tt) = iter.next() {
                match &tt {
                    TokenTree::Group(group) => self.scan(group.stream()),
                    TokenTree::Punct(p)
                        if p.as_char() == ':' && p.spacing() == proc_macro2::Spacing::Joint =>
                    {
                        let starts_path = !matches!(&prev, Some(TokenTree::Ident(_)))
                            && !matches!(&prev, Some(TokenTree::Punct(p)) if p.as_char() == '>');
                        if starts_path {
                            if let Some(TokenTree::Punct(second)) = iter.peek() {
                                if second.as_char() == ':' {
                                    iter.next();
                                    if let Some(TokenTree::Ident(ident)) = iter.peek() {
                                        if ident.unraw() != "core" {
                                            self.0.insert(ident.unraw().to_string());
                                        }
                                    }
                                }
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
            if i.leading_colon.is_some() {
                if let Some(seg) = i.segments.first() {
                    if seg.ident.unraw() != "core" {
                        self.0.insert(seg.ident.unraw().to_string());
                    }
                }
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

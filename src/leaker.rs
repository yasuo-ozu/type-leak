//! The [`Leaker`], which finds the types depending on the type context.

use crate::analysis::{drop_optional, names_self, Analysis};
use crate::analyze::{
    declared_names, AnalyzeVisitor, Finished, ModuleNames, Res, ScopeSpec, Scopes,
};
use crate::config::LeakerConfig;
use crate::context::{Context, ContextId, Contexts, Gat, SelfRequirement};
#[cfg(doc)]
use crate::error::NotInternableError;
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
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// Finds the leaked types of an item and turns them into a [`Referrer`].
///
/// Analyze a whole item with `intern_*`, or a part of one in a scope with `enter_*`. Then call
/// [`Leaker::finish()`]. Each method returns the top-level [`ContextId`], and fails with a
/// [`NotInternableError`]. See the [walkthrough](crate#walkthrough).
#[derive(Default)]
pub struct Leaker {
    /// Paths that are not leaked, such as `::serde`.
    ///
    /// A path matches if it starts with one of these, and both have a leading `::` or neither. Its
    /// generic arguments are still analyzed. An allowed absolute path must name the same crate on
    /// both sides. See [External crates](crate#external-crates). [`LeakerConfig`] has helpers.
    pub allowed_paths: Vec<Path>,
    /// The types that depend on the context (the required entries).
    types: TypeGraph,
    /// The types the Referrer may replace (the optional entries).
    optional_types: TypeGraph,
    /// The result of [`Leaker::reduce_roots()`]: required, then optional. `None` if outdated.
    reachable_types: Option<(Vec<NodeKey>, Vec<NodeKey>)>,
    contexts: Contexts,
    /// Numbered on their own, so that optional entries do not change the ids of required ones.
    optional_contexts: Contexts,
    /// The top-level contexts of successful scopes, which the [`Referrer`] keeps without an entry
    /// too.
    top_levels: BTreeSet<ContextId>,
    assumed_primitives: Vec<Ident>,
}

impl Leaker {
    /// Creates an empty [`Leaker`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`Leaker`] from a configuration.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allowed_paths: config.allowed_paths,
            ..Self::default()
        }
    }

    /// Returns items that fail to compile ("mismatched types") if a primitive type name is
    /// shadowed.
    ///
    /// The Leaker copies primitive names as they are, such as `u8` in `const N: u8` or a name
    /// allowed by [`LeakerConfig::allow_primitive()`]. Emit the result in the Leaker's type
    /// context. A name is shadowed by, for example, `type u8 = i32;`.
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

    /// Analyzes the struct `item` and returns its top-level context.
    ///
    /// The generics of the item are those of the Leaker, and `Self` is the struct
    /// ([`SelfRequirement::Type`]). For `allow_self` see [Self rules](crate#self-rules). This is
    /// [`Leaker::enter_type()`] with the item visited.
    pub fn intern_struct(&mut self, item: &ItemStruct, allow_self: bool) -> Res<ContextId> {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_struct(view, item)
        })
        .map(|(context, ())| context)
    }

    /// Analyzes the enum `item` and returns its top-level context.
    ///
    /// It is [`Leaker::intern_struct()`] for an enum, with the same `Self` and `allow_self`.
    pub fn intern_enum(&mut self, item: &ItemEnum, allow_self: bool) -> Res<ContextId> {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_enum(view, item)
        })
        .map(|(context, ())| context)
    }

    /// Analyzes the union `item` and returns its top-level context.
    ///
    /// It is [`Leaker::intern_struct()`] for a union, with the same `Self` and `allow_self`.
    pub fn intern_union(&mut self, item: &ItemUnion, allow_self: bool) -> Res<ContextId> {
        self.enter_type(&item.ident, &item.generics, allow_self, |view| {
            syn::visit::visit_item_union(view, item)
        })
        .map(|(context, ())| context)
    }

    /// Analyzes the impl `item` and returns its top-level context.
    ///
    /// The generics of the item are those of the Leaker, and `Self` is the self type
    /// ([`SelfRequirement::Type`]). It analyzes the generics, the trait and its arguments, the self
    /// type, and the signatures, associated types and consts. It skips function bodies and const
    /// values. The trait path must not depend on the context. An entry in the self type cannot
    /// depend on `Self` through a bound of the generics ([Self rules](crate#self-rules)). For
    /// `allow_self` see
    /// [`Leaker::intern_struct()`]. This is [`Leaker::enter_impl()`] with the item visited.
    pub fn intern_impl(&mut self, item: &ItemImpl, allow_self: bool) -> Res<ContextId> {
        // The scope checks the path of the trait, and the walk analyzes its arguments.
        let trait_path = item.trait_.as_ref().map(|(_, path, _)| path);
        self.enter_impl_of(
            &item.generics,
            &item.self_ty,
            trait_path,
            allow_self,
            |view| view.walk_item_impl(item),
        )
        .map(|(context, ())| context)
    }

    /// Analyzes the trait `item` and returns its top-level context.
    ///
    /// The generics of the item are those of the Leaker, and `Self` is the implementor
    /// ([`SelfRequirement::Bound`]). An entry that names `Self` is written for the implementors.
    /// For `allow_self` see [`Leaker::intern_struct()`]. This is [`Leaker::enter_trait()`] with the
    /// item visited, and with its generic associated types known: a projection that names one that
    /// has outlives bounds (`type G<'b> where Self: 'b;`) requires them
    /// ([Outlives bounds](crate#outlives-bounds)).
    pub fn intern_trait(&mut self, item: &ItemTrait, allow_self: bool) -> Res<ContextId> {
        // The scope knows the generic associated types of the trait: what they require is part of
        // the context of an entry that names them.
        let gats = Gat::of_items(&item.items);
        self.enter_trait_of(&item.ident, &item.generics, gats, allow_self, |view| {
            syn::visit::visit_item_trait(view, item)
        })
        .map(|(context, ())| context)
    }

    /// Analyzes the module `item` and returns the [`ContextId`] of the module, which has no
    /// generics.
    ///
    /// Every item in it is a nested item, so its generics are local. Function bodies and const
    /// values are not analyzed, and a `mod m;` has nothing to analyze. Names declared in the module
    /// are copied with it, so they are not leaked. Nested modules are numbered as in
    /// [`Leaker::enter_mod()`], with their paths recorded. See [Modules](crate#modules).
    pub fn intern_mod(&mut self, item: &ItemMod) -> Res<ContextId> {
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
            .map(|(context, ())| context)
    }

    /// Enters a module and runs `f` on a [`View`] of it.
    ///
    /// Returns the [`ContextId`] of the module, which has no generics, and the result of `f`.
    /// `declared` lists the names declared in the module (give them first). `hint` is for debugging
    /// only. Ids come from a counter, so they are unique and equal calls give equal ids. `Self` has
    /// no meaning directly in `f`. Errors and rollback are as in [`Leaker::enter_generics()`]. See
    /// [Modules](crate#modules).
    pub fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_mod(self, hint, declared, f)
    }

    /// Enters generics and runs `f` on a [`View`] of them.
    ///
    /// Returns the top-level [`ContextId`] of `generics` and the result of `f`. The generics are
    /// not analyzed. `Self` is not declared here, so naming it in `f` or in the generics is an
    /// error ([`NotInternableError::SelfWithoutMeaning`]). An error in `f` rolls the scope back and
    /// returns its errors, and an outer scope can go on. A panic in `f` rolls the scope back too.
    /// See [Scopes](crate#scopes).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .enter_generics(&parse_quote!(<T>), |view| {
    ///         view.visit_type(&parse_quote!(Rel<T>));
    ///         // A relative trait cannot be leaked, so the inner scope fails and the outer one goes on.
    ///         let failed = view.enter_generics(&parse_quote!(<U: Relative>), |view| {
    ///             view.visit_type_param_bound(&parse_quote!(Relative));
    ///         });
    ///         assert!(failed.is_err());
    ///     })
    ///     .unwrap();
    /// assert_eq!(leaker.finish().iter().count(), 1);
    /// ```
    pub fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_generics(self, generics, f)
    }

    /// Enters the type definition `ident` with `generics` and runs `f` on a [`View`] of it.
    ///
    /// Returns the top-level context and the result of `f`. The fields are not analyzed: visit them
    /// in `f`. `Self` is `ident` with the parameters as arguments ([`SelfRequirement::Type`]). For
    /// `allow_self` see [`Leaker::intern_struct()`]. Context, errors and rollback are as in
    /// [`Leaker::enter_generics()`], but `generics` may name `Self`.
    pub fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_type(self, ident, generics, allow_self, f)
    }

    /// Enters an impl with `generics` for `self_ty` and runs `f` on a [`View`] of it.
    ///
    /// Returns the top-level context and the result of `f`. Visit the fragments of the impl in `f`.
    /// `Self` is `self_ty` ([`SelfRequirement::Type`]). For `allow_self` see
    /// [`Leaker::intern_struct()`]. Context, errors and rollback are as in
    /// [`Leaker::enter_generics()`]. See [Scopes](crate#scopes).
    pub fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_impl(self, generics, self_ty, allow_self, f)
    }

    /// Enters the trait `ident` with `generics` and runs `f` on a [`View`] of it.
    ///
    /// Returns the top-level context and the result of `f`. `Self` is the implementor, built from
    /// `ident` and the parameters ([`SelfRequirement::Bound`]). For `allow_self` see
    /// [`Leaker::intern_struct()`]. Context, errors and rollback are as in
    /// [`Leaker::enter_generics()`]. The items of the trait are not known here, so the generic
    /// associated types it declares have no outlives bounds for a projection that names them (use
    /// [`Leaker::intern_trait()`], or write the bound).
    pub fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        Scopes::enter_trait(self, ident, generics, allow_self, f)
    }

    /// Finishes the analysis and returns the [`Referrer`]. It calls [`Leaker::reduce_roots()`] if
    /// needed.
    ///
    /// The result is deterministic. Required entries come first, then optional ones
    /// ([`Referrer::must()`]), each sorted by context (in order of first appearance) and tokens. So
    /// optional entries never change the indices of required ones, and contexts used only by
    /// optional entries are numbered last, in the order of the entries. The Referrer keeps the ids
    /// the scopes returned, for the top-level contexts, the contexts of the entries, and their
    /// modules with parents.
    pub fn finish(mut self) -> Referrer {
        self.reduce_roots();
        let (required, optional) = self.reachable_types.take().unwrap_or_default();
        let contexts = self.contexts;
        let context_of = |id: ContextId| contexts.get(id).clone();
        let mut context_map: BTreeMap<ContextId, Context> = self
            .top_levels
            .iter()
            .map(|&id| (id, context_of(id)))
            .collect();
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
        Referrer::new(entries, context_map, modules)
    }

    /// Chooses the fewest entries from which every leaked type can be built.
    ///
    /// [`Leaker::finish()`] calls it, so you rarely need to. Contexts used only by optional entries
    /// are numbered, after the others, when this runs. See [Which types become
    /// entries](crate#which-types-become-entries).
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

impl Scopes for Leaker {
    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let root = spec.declared_generics();
        let mut visitor = AnalyzeVisitor::new(AllowedPaths::new(&self.allowed_paths), &root);
        let mark = self.contexts.len();
        // The visitor owns the table of contexts while it analyzes, so the swap is the only step
        // before `catch_unwind`: no allocation, no user code. Whatever happens next (a result, an
        // error, or a panic of `f`), the table is swapped back before anything else, and an
        // unfinished analysis is rolled back to `mark`.
        visitor.swap_contexts(&mut self.contexts);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            visitor.start(&spec);
            let result = f(&mut View::new(&mut visitor));
            visitor.finish(&root).map(|finished| (finished, result))
        }));
        visitor.swap_contexts(&mut self.contexts);
        let (finished, result) = match outcome {
            Ok(Ok(done)) => done,
            Ok(Err(error)) => {
                self.contexts.truncate(mark);
                return Err(error);
            }
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
        Ok((top_level, result))
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

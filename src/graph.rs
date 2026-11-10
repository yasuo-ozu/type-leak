//! The graph of the types found by the analysis, and its reduction to the entries (a minimum cut).

use crate::context::ContextId;
use crate::util::{is_whole_reference, tokens_text};
use safegraph::algo::max_flow::edmonds_karp_with_flows;
use safegraph::collection::IndexKey;
use safegraph::graph::Graph;
use safegraph::raw_graph::linked_adj_edge::{EdgeRepr, LinkedAdjEdgeGraph, NodeRepr};
use safegraph::BTreeGraph;
use std::collections::{BTreeMap, HashMap, HashSet};
use syn::*;
use template_quote::ToTokens;

/// A type compared by its tokens: `syn::Type` has no `Ord`, which safegraph needs, and its `==`
/// tells a type in an invisible group (`$t:ty` of a `macro_rules!`) from the same type without one.
#[derive(Clone, Debug)]
pub(crate) struct TypeKey {
    pub(crate) ty: Type,
}

impl TypeKey {
    pub(crate) fn new(ty: &Type) -> Self {
        Self { ty: ty.clone() }
    }

    /// The tokens as a string, without `r#` ([`tokens_text()`]): what the key compares, hashes and
    /// orders.
    pub(crate) fn tokens(&self) -> String {
        tokens_text(self)
    }

    /// The tokens as written, `r#` kept: the order of the entries. Two keys of a graph differ in it.
    fn written(&self) -> String {
        self.to_token_stream().to_string()
    }
}

impl ToTokens for TypeKey {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.ty.to_tokens(tokens)
    }
}

impl PartialEq for TypeKey {
    fn eq(&self, other: &Self) -> bool {
        self.tokens() == other.tokens()
    }
}

impl Eq for TypeKey {}

// `Ord` is required by safegraph (`IndexKey`), which this crate never calls it through.
impl PartialOrd for TypeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TypeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.tokens().cmp(&other.tokens())
    }
}

impl std::hash::Hash for TypeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tokens().hash(state)
    }
}

/// A leaked type in a context. The same tokens in two contexts are two nodes of the [`TypeGraph`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeKey {
    pub(crate) context: ContextId,
    pub(crate) ty: TypeKey,
}

// Required by safegraph (the keys of a graph are `Display`), and never called by this crate.
impl std::fmt::Display for NodeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.context, self.ty.tokens())
    }
}

// SAFETY: `Eq`, `Ord` and `Hash` derive from deterministic fields, and none of them panics.
unsafe impl IndexKey for NodeKey {}

type CompositionGraph =
    LinkedAdjEdgeGraph<HashMap<NodeKey, NodeRepr<u32>>, Vec<((), EdgeRepr<NodeKey, u32>)>>;

/// The types found by the analysis that may be entries, and their composition (the components of
/// a type that may be no entry hang on its parent). [`TypeGraph::of_kept()`] makes the graph of the
/// entries that a reduction chose, which the next analysis adds to.
#[derive(Default)]
pub(crate) struct TypeGraph {
    /// An edge goes from a type to each internable type it is composed of.
    graph: CompositionGraph,
    /// The types of the graph.
    nodes: HashSet<NodeKey>,
    /// The types that appear as roots, not as components of another type.
    roots: HashSet<NodeKey>,
    /// The required types: they depend on the context, so each must be an entry or inside one. The
    /// kept required entries are in it too.
    pub(crate) required: HashSet<NodeKey>,
    /// The types that the Referrer may replace (their own path names an external crate): the same.
    pub(crate) optional: HashSet<NodeKey>,
    /// The types that may be a required entry: a part of them is a cause.
    pub(crate) required_candidates: HashSet<NodeKey>,
    /// The types that may be an optional entry: they or a part of them is optional.
    pub(crate) optional_candidates: HashSet<NodeKey>,
    /// The types that cannot be entries, so their components are: the copy would be cyclic
    /// (`AnalyzeVisitor::env_cycle()`), or the type names a nested item's `Self`, for example. A
    /// type is forbidden as the kind it is, so one set holds both kinds.
    pub(crate) no_entry: HashSet<NodeKey>,
    /// The entries that a reduction chose ([`TypeGraph::of_kept()`]): each stays an entry whatever
    /// is merged in later. A kept required one is in `required` too; a kept optional one becomes
    /// required if a type merged in later makes it hold a required type.
    kept: HashSet<NodeKey>,
}

impl TypeGraph {
    /// Adds an occurrence of `ty` in `context`, a component of `parent` or a root, and returns its
    /// key.
    pub(crate) fn add(
        &mut self,
        context: ContextId,
        ty: &Type,
        parent: Option<&NodeKey>,
    ) -> NodeKey {
        let key = NodeKey {
            context,
            ty: TypeKey::new(ty),
        };
        if self.nodes.insert(key.clone()) {
            self.graph.insert_node(key.clone()).ok().unwrap();
        }
        match parent {
            Some(parent) => self.add_edge(parent.clone(), key.clone()),
            None => {
                self.roots.insert(key.clone());
            }
        }
        key
    }

    fn add_edge(&mut self, from: NodeKey, to: NodeKey) {
        if !self
            .graph
            .neighbor_indices_from(&from)
            .any(|child| child == to)
        {
            self.graph.push_edge((), [from, to]).unwrap();
        }
    }

    /// Returns the graph of only the entries `required` and `optional`, each a kept root with no
    /// components. A required one is also in `required`, which keeps its kind; an optional one is
    /// not in `optional`, which says what a type is.
    pub(crate) fn of_kept(required: Vec<NodeKey>, optional: Vec<NodeKey>) -> Self {
        let mut graph = Self::default();
        for (key, is_required) in required
            .into_iter()
            .map(|key| (key, true))
            .chain(optional.into_iter().map(|key| (key, false)))
        {
            graph.nodes.insert(key.clone());
            graph.graph.insert_node(key.clone()).ok().unwrap();
            graph.roots.insert(key.clone());
            if is_required {
                graph.required.insert(key.clone());
            }
            graph.kept.insert(key);
        }
        graph
    }

    /// Returns `true` if the graph holds nothing but kept entries ([`TypeGraph::of_kept()`]): no
    /// type has been merged in since, as a type that is added is a candidate of a kind. Reducing
    /// it gives the same entries.
    pub(crate) fn is_reduced(&self) -> bool {
        self.required_candidates.is_empty()
            && self.optional_candidates.is_empty()
            && self.nodes.len() == self.kept.len()
    }

    /// Returns the kept entries: the required ones, then the optional ones, each sorted by context
    /// and tokens.
    pub(crate) fn into_kept_entries(self) -> (Vec<NodeKey>, Vec<NodeKey>) {
        let Self { kept, required, .. } = self;
        let (required_entries, optional_entries): (Vec<_>, Vec<_>) =
            kept.into_iter().partition(|key| required.contains(key));
        (
            sorted_entries(required_entries),
            sorted_entries(optional_entries),
        )
    }

    /// Returns the types, in no particular order.
    fn keys(&self) -> impl Iterator<Item = &NodeKey> {
        self.nodes.iter()
    }

    fn children(&self, key: &NodeKey) -> Vec<NodeKey> {
        self.graph.neighbor_indices_from(key).collect()
    }

    pub(crate) fn merge(&mut self, other: TypeGraph) {
        for key in other.keys() {
            if self.nodes.insert(key.clone()) {
                self.graph.insert_node(key.clone()).ok().unwrap();
            }
        }
        // The edges come in no fixed order: `reduce()` sorts the types, and its cut is unique.
        for from in other.keys() {
            for to in other.children(from) {
                self.add_edge(from.clone(), to);
            }
        }
        self.roots.extend(other.roots);
        self.required.extend(other.required);
        self.optional.extend(other.optional);
        self.required_candidates.extend(other.required_candidates);
        self.optional_candidates.extend(other.optional_candidates);
        self.no_entry.extend(other.no_entry);
        self.kept.extend(other.kept);
    }

    /// Min vertex cut from the roots to the red types, by max flow (Edmonds-Karp, node-split). Ties
    /// go to the cut closest to the sink, so inner types win. Returns the required entries and the
    /// optional ones, each sorted ([`sorted_entries()`]).
    pub(crate) fn reduce(&self) -> (Vec<NodeKey>, Vec<NodeKey>) {
        let (keys, children, parents) = self.indexed();
        let n = keys.len();
        let flags = |set: &HashSet<NodeKey>| -> Vec<bool> {
            keys.iter().map(|key| set.contains(*key)).collect()
        };
        let (is_required, is_optional, is_root, is_kept) = (
            flags(&self.required),
            flags(&self.optional),
            flags(&self.roots),
            flags(&self.kept),
        );

        // An entry that covers a required type is a required entry.
        let mut covers_required = is_required.clone();
        spread(&mut covers_required, &parents);
        // A type is cut only as the kind it is: a candidate of that kind, not in `no_entry`. An
        // optional type is an entry anyway, so it is a candidate of either kind.
        let may_cut: Vec<bool> = (0..n)
            .map(|i| {
                let candidate = match covers_required[i] {
                    true => is_optional[i] || self.required_candidates.contains(keys[i]),
                    false => self.optional_candidates.contains(keys[i]),
                };
                candidate && !self.no_entry.contains(keys[i])
            })
            .collect();
        // A red type must be cut. An optional type that cannot be an entry is not red, and its
        // optional or required parts are. A kept entry is red whatever else the graph says of it.
        let red: Vec<bool> = (0..n)
            .map(|i| is_required[i] || is_kept[i] || (is_optional[i] && may_cut[i]))
            .collect();
        // A type that must be an entry cannot be forbidden to be one.
        let forbidden: Vec<bool> = (0..n)
            .map(|i| !may_cut[i] && !red[i] && !is_whole_reference(&keys[i].ty.ty))
            .collect();
        let mut has_red = red.clone();
        spread(&mut has_red, &parents);

        // A red root is in every minimum cut, so it is cut without a flow. So is a
        // kept entry, which is a root of the graph it was kept in.
        let cut: Vec<bool> = (0..n)
            .map(|i| (red[i] && is_root[i]) || is_kept[i])
            .collect();
        let mut chosen: Vec<usize> = (0..n).filter(|&i| cut[i]).collect();

        // The edges of the network. A parent that its components cannot replace goes to the sink.
        let mut next: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut to_sink = vec![false; n];
        for i in (0..n).filter(|&i| has_red[i] && !cut[i]) {
            to_sink[i] = red[i];
            let whole = is_whole_reference(&keys[i].ty.ty);
            for &child in children[i].iter().filter(|&&child| has_red[child]) {
                if whole {
                    // The parent is interned as a whole, like a required type.
                    to_sink[i] = true;
                } else if !cut[child] {
                    next[i].push(child);
                }
            }
        }
        let mut reached: Vec<bool> = (0..n)
            .map(|i| is_root[i] && has_red[i] && !red[i])
            .collect();
        spread(&mut reached, &next);
        let network = Network {
            next,
            is_root,
            to_sink,
            forbidden,
        };
        for members in network.components(&reached).values() {
            chosen.extend(network.cut_component(members));
        }

        let entries = |kind: bool| {
            let found = chosen.iter().filter(|&&i| covers_required[i] == kind);
            sorted_entries(found.map(|&i| keys[i].clone()).collect())
        };
        (entries(true), entries(false))
    }

    /// The types sorted by context and tokens, and, by index, the components and the parents of
    /// each.
    fn indexed(&self) -> (Vec<&NodeKey>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut keys: Vec<&NodeKey> = self.keys().collect();
        keys.sort_by_cached_key(|key| (key.context, key.ty.written()));
        let index: HashMap<&NodeKey, usize> =
            keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let children: Vec<Vec<usize>> = keys
            .iter()
            .map(|key| {
                let children = self.children(key).into_iter();
                children.filter_map(|c| index.get(&c).copied()).collect()
            })
            .collect();
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); keys.len()];
        for (i, children) in children.iter().enumerate() {
            for &child in children {
                parents[child].push(i);
            }
        }
        (keys, children, parents)
    }
}

/// Sorts entries by context and tokens: the order of the Referrer's entries.
fn sorted_entries(mut entries: Vec<NodeKey>) -> Vec<NodeKey> {
    entries.sort_by_cached_key(|key| (key.context, key.ty.written()));
    entries
}

/// The flow network of [`TypeGraph::reduce()`], by the indices of the types.
struct Network {
    /// The components that a type may be cut at instead of it.
    next: Vec<Vec<usize>>,
    /// The source has an edge to each root.
    is_root: Vec<bool>,
    /// Each red type, and each that must be cut whole, has an edge to the sink.
    to_sink: Vec<bool>,
    /// The types that cannot be cut, as they cannot be entries.
    forbidden: Vec<bool>,
}

impl Network {
    /// The types that `reached` marks, grouped by the components of the network, each in
    /// ascending order and keyed by its smallest index. Solving each alone keeps the quadratic
    /// max flow small.
    fn components(&self, reached: &[bool]) -> BTreeMap<usize, Vec<usize>> {
        let n = reached.len();
        let mut component: Vec<usize> = (0..n).collect();
        for i in (0..n).filter(|&i| reached[i]) {
            for &child in &self.next[i] {
                let (a, b) = (find(&mut component, i), find(&mut component, child));
                component[a.max(b)] = a.min(b);
            }
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in (0..n).filter(|&i| reached[i]) {
            groups.entry(find(&mut component, i)).or_default().push(i);
        }
        groups
    }

    /// The types of the component `members` that its minimum cut closest to the sink goes through,
    /// in the order of `members`.
    fn cut_component(&self, members: &[usize]) -> Vec<usize> {
        // The nodes: `v_in` is `2 * v`, `v_out` is `2 * v + 1`, for the type `members[v]`.
        let local: HashMap<usize, usize> =
            members.iter().enumerate().map(|(v, &i)| (i, v)).collect();
        let m = members.len();
        let (source, sink) = (2 * m, 2 * m + 1);
        // Larger than all node capacities together, so never the bottleneck.
        let infinite = m + 1;
        let mut edges = Vec::new();
        for (v, &i) in members.iter().enumerate() {
            let (v_in, v_out) = (2 * v, 2 * v + 1);
            // The capacity of a type is what cutting it costs: one entry, or none if it cannot be
            // one.
            edges.push((v_in, v_out, if self.forbidden[i] { infinite } else { 1 }));
            for child in &self.next[i] {
                edges.push((v_out, 2 * local[child], infinite));
            }
            if self.is_root[i] {
                edges.push((source, v_in, infinite));
            }
            if self.to_sink[i] {
                edges.push((v_out, sink, infinite));
            }
        }
        let reaches_sink = sink_side(2 * m + 2, &edges, source, sink);
        // The cut closest to the sink: `v_in` cannot reach it, `v_out` can.
        (0..m)
            .filter(|&v| reaches_sink[2 * v + 1] && !reaches_sink[2 * v])
            .map(|v| members[v])
            .collect()
    }
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

//! The graph of the types found by the analysis.

use crate::context::ContextId;
use crate::util::unwrap_type_group;
use safegraph::collection::IndexKey;
use safegraph::graph::Graph;
use safegraph::raw_graph::linked_adj_edge::{EdgeRepr, LinkedAdjEdgeGraph, NodeRepr};
use std::collections::{HashMap, HashSet};
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

    /// The tokens as a string: what the key compares, hashes and orders.
    pub(crate) fn tokens(&self) -> String {
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

/// The types found by the analysis, and their composition.
///
/// A type is a node if it may be a required entry or an optional one (or both). A type that may
/// be neither is not in it: its components hang on its parent.
///
/// [`TypeGraph::kept()`] makes the graph of the entries that a reduction chose, which the next
/// analysis adds to.
#[derive(Default)]
pub(crate) struct TypeGraph {
    /// An edge goes from a type to each internable type it is composed of.
    graph: CompositionGraph,
    nodes: HashSet<NodeKey>,
    /// The types that appear as roots, not as components of another type.
    pub(crate) roots: HashSet<NodeKey>,
    /// The types that depend on the context: each must be an entry or inside one. The kept
    /// required entries are in it too.
    pub(crate) must_intern: HashSet<NodeKey>,
    /// The types that the Referrer may replace (their own path names an external crate): the same.
    pub(crate) optional: HashSet<NodeKey>,
    /// The types that may be a required entry: a part of them is a cause.
    pub(crate) required_candidates: HashSet<NodeKey>,
    /// The types that may be an optional entry: they or a part of them is optional.
    pub(crate) optional_candidates: HashSet<NodeKey>,
    /// The types that cannot be entries because the copy would be cyclic (see
    /// `AnalyzeVisitor::env_cycle()`): their components are entries instead. A type is forbidden
    /// as the kind it is, required if it is or holds a `must_intern` type, else optional, so one
    /// set holds the flag of that kind.
    pub(crate) no_entry: HashSet<NodeKey>,
    /// The entries that a reduction chose and kept ([`TypeGraph::kept()`]). Each stays an entry
    /// whatever is merged in later, though what made it one is gone. A kept required entry is also
    /// in `must_intern`, so it stays required; a kept optional one becomes required if a type merged
    /// in later makes it hold a required type.
    pub(crate) kept: HashSet<NodeKey>,
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

    /// Returns the graph that holds only the entries `required` and `optional`, each a root with
    /// no components, and kept as the kind it is.
    ///
    /// `kept` marks them, as a kept entry is a choice, which the types that made it an entry no
    /// longer back. A required one is also `must_intern`, which keeps its kind; `optional` says
    /// what a type is, so an optional entry is not in it.
    pub(crate) fn kept(required: Vec<NodeKey>, optional: Vec<NodeKey>) -> Self {
        let mut graph = Self::default();
        for (key, must) in required
            .into_iter()
            .map(|key| (key, true))
            .chain(optional.into_iter().map(|key| (key, false)))
        {
            graph.nodes.insert(key.clone());
            graph.graph.insert_node(key.clone()).ok().unwrap();
            graph.roots.insert(key.clone());
            if must {
                graph.must_intern.insert(key.clone());
            }
            graph.kept.insert(key);
        }
        graph
    }

    /// Returns `true` if the graph holds nothing but kept entries ([`TypeGraph::kept()`]): no type
    /// has been merged in since, as a type that is added is a candidate of a kind. Reducing it
    /// gives the same entries.
    pub(crate) fn is_reduced(&self) -> bool {
        self.required_candidates.is_empty()
            && self.optional_candidates.is_empty()
            && self.nodes.len() == self.kept.len()
    }

    /// Returns the kept entries: the required ones, then the optional ones, each sorted by context
    /// and tokens.
    pub(crate) fn into_kept_entries(self) -> (Vec<NodeKey>, Vec<NodeKey>) {
        let Self {
            kept, must_intern, ..
        } = self;
        let (required, optional): (Vec<_>, Vec<_>) =
            kept.into_iter().partition(|key| must_intern.contains(key));
        (sorted_entries(required), sorted_entries(optional))
    }

    /// Returns the types, in no particular order.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &NodeKey> {
        self.nodes.iter()
    }

    pub(crate) fn children(&self, key: &NodeKey) -> Vec<NodeKey> {
        self.graph.neighbor_indices_from(key).collect()
    }

    pub(crate) fn merge(&mut self, other: TypeGraph) {
        for key in other.keys() {
            if self.nodes.insert(key.clone()) {
                self.graph.insert_node(key.clone()).ok().unwrap();
            }
        }
        for from in other.keys() {
            for to in other.children(from) {
                self.add_edge(from.clone(), to);
            }
        }
        self.roots.extend(other.roots);
        self.must_intern.extend(other.must_intern);
        self.optional.extend(other.optional);
        self.required_candidates.extend(other.required_candidates);
        self.optional_candidates.extend(other.optional_candidates);
        self.no_entry.extend(other.no_entry);
        self.kept.extend(other.kept);
    }
}

/// Sorts entries by context and tokens: the order of the Referrer's entries.
pub(crate) fn sorted_entries(mut entries: Vec<NodeKey>) -> Vec<NodeKey> {
    entries.sort_by_cached_key(|key| (key.context, key.ty.tokens()));
    entries
}

/// Returns `true` for a reference with a lifetime: its components cannot stand for it, as the copy
/// needs a bound (`Rel: 'static`) that its referent's copy may lack. Only `'static` needs this (see
/// `Enclosing`), but all lifetimes get it. A trait object referent is exempt (see
/// `Inherits::Reference`).
pub(crate) fn is_whole_reference(ty: &Type) -> bool {
    match unwrap_type_group(ty) {
        Type::Reference(TypeReference {
            lifetime: Some(_),
            elem,
            ..
        }) => !matches!(unwrap_type_group(elem), Type::TraitObject(_)),
        _ => false,
    }
}

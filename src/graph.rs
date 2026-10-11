//! The graph of the types found by the analysis.

use crate::context::ContextId;
use crate::util::unwrap_type_group;
use safegraph::collection::IndexKey;
use safegraph::graph::Graph;
use safegraph::raw_graph::linked_adj_edge::{EdgeRepr, LinkedAdjEdgeGraph, NodeRepr};
use std::collections::{HashMap, HashSet};
use syn::*;
use template_quote::ToTokens;

/// A type compared by its tokens: `syn::Type` has no `Ord` or `Display`, which safegraph needs.
#[derive(Clone, Debug)]
pub(crate) struct TypeKey {
    pub(crate) ty: Type,
}

impl TypeKey {
    pub(crate) fn new(ty: &Type) -> Self {
        Self { ty: ty.clone() }
    }

    fn tokens(&self) -> String {
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

impl std::fmt::Display for TypeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.tokens())
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
        write!(f, "{}:{}", self.context, self.ty)
    }
}

// SAFETY: `Eq`, `Ord` and `Hash` derive from deterministic fields, and none of them panics.
unsafe impl IndexKey for NodeKey {}

type CompositionGraph =
    LinkedAdjEdgeGraph<HashMap<NodeKey, NodeRepr<u32>>, Vec<((), EdgeRepr<NodeKey, u32>)>>;

/// The types found by the analysis, and their composition.
#[derive(Default)]
pub(crate) struct TypeGraph {
    /// An edge goes from a type to each internable type it is composed of.
    graph: CompositionGraph,
    nodes: HashSet<NodeKey>,
    /// The types that appear as roots, not as components of another type.
    pub(crate) roots: HashSet<NodeKey>,
    /// The types that must be interned: they depend on the context, or the Referrer may replace
    /// them.
    pub(crate) must_intern: HashSet<NodeKey>,
    /// The types that cannot be entries, because the copy would be cyclic (see
    /// `AnalyzeVisitor::env_cycle()`): their components are entries instead.
    pub(crate) forbidden: HashSet<NodeKey>,
}

impl TypeGraph {
    /// Adds an occurrence of `ty` in `context`, a component of `parent` or a root, and returns its
    /// key.
    pub(crate) fn add(
        &mut self,
        context: ContextId,
        ty: &Type,
        parent: Option<&NodeKey>,
        is_must_intern: bool,
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
        if is_must_intern {
            self.must_intern.insert(key.clone());
        }
        key
    }

    /// Makes `key` a type that cannot be an entry.
    pub(crate) fn forbid(&mut self, key: &NodeKey) {
        self.forbidden.insert(key.clone());
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

    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
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
        self.forbidden.extend(other.forbidden);
    }
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

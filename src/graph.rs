//! The graph of the types found by the analysis.

use crate::context::ContextId;
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

impl parse::Parse for TypeKey {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        Ok(Self { ty: input.parse()? })
    }
}

impl PartialEq for TypeKey {
    fn eq(&self, other: &Self) -> bool {
        self.tokens() == other.tokens()
    }
}

impl Eq for TypeKey {}

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
    }
}

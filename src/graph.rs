//! The graph of the types found by the analysis.

use crate::requirements::ContextId;
use safegraph::collection::IndexKey;
use safegraph::graph::Graph;
use safegraph::raw_graph::linked_adj_edge::{EdgeRepr, LinkedAdjEdgeGraph, NodeRepr};
use std::collections::{HashMap, HashSet};
use syn::*;
use template_quote::ToTokens;

/// A type compared, hashed and displayed by its tokens, since `syn::Type` has no `Ord` or
/// `Display` (which safegraph requires of [`IndexKey`]s).
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

/// A leaked type, as a node of [`TypeGraph`] (and the key of a [`Referrer`](crate::Referrer)'s
/// entry).
///
/// Equal keys are the same leaked type, which is shared by all its occurrences, whichever call of
/// [`Leaker::intern_with()`](crate::Leaker::intern_with()) found them. The same tokens with
/// different requirements, i.e. in different contexts (see
/// [`Referrer::context()`](crate::Referrer::context())), are separate entries.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeKey {
    /// The context of the requirements of the type (see `Contexts`).
    pub(crate) context: ContextId,
    pub(crate) ty: TypeKey,
}

impl std::fmt::Display for NodeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.context, self.ty)
    }
}

// SAFETY: `Eq`, `Ord` and `Hash` are derived from those of the fields (an id and the tokens,
// each of which is deterministic), so they are consistent with each other, and none of them (nor
// `Clone`) panics.
unsafe impl IndexKey for NodeKey {}

/// A graph whose nodes are keyed by [`NodeKey`] (in a `HashMap`), and whose edges are stored in a
/// `Vec` with no payload, so an edge holds its endpoints only once (as the endpoints).
type CompositionGraph =
    LinkedAdjEdgeGraph<HashMap<NodeKey, NodeRepr<u32>>, Vec<((), EdgeRepr<NodeKey, u32>)>>;

/// The types found by the analysis, and their composition.
#[derive(Default)]
pub(crate) struct TypeGraph {
    /// An edge goes from a type to each internable type it is composed of.
    graph: CompositionGraph,
    /// The nodes of `graph`.
    nodes: HashSet<NodeKey>,
    /// The types appearing as roots (not as components of another internable type).
    pub(crate) roots: HashSet<NodeKey>,
    /// The types which must be interned (they directly depend on the type context).
    pub(crate) must_intern: HashSet<NodeKey>,
    /// The types with a cause which is not an external crate's path (see
    /// [`Referrer::must()`](crate::Referrer::must())).
    pub(crate) hard: HashSet<NodeKey>,
}

impl TypeGraph {
    /// Adds an occurrence of `ty` in `context` (a component of `parent`, or a root), and returns
    /// its key.
    pub(crate) fn add(
        &mut self,
        context: ContextId,
        ty: &Type,
        parent: Option<&NodeKey>,
        is_must_intern: bool,
        is_hard: bool,
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
        if is_hard {
            self.hard.insert(key.clone());
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

    /// Returns `true` if no type has been added.
    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the types (in no particular order).
    pub(crate) fn keys(&self) -> impl Iterator<Item = &NodeKey> {
        self.nodes.iter()
    }

    /// Returns the components of `key`.
    pub(crate) fn children(&self, key: &NodeKey) -> Vec<NodeKey> {
        self.graph.neighbor_indices_from(key).collect()
    }

    /// Merges `other` into this graph.
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
        self.hard.extend(other.hard);
    }
}

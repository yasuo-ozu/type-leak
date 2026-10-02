//! The graph of the types found by the analysis.

use safegraph::collection::IndexKey;
use safegraph::graph::Graph;
use safegraph::raw_graph::linked_adj_edge::{EdgeRepr, LinkedAdjEdgeGraph, NodeRepr};
use std::collections::{HashMap, HashSet};
use syn::*;
use template_quote::ToTokens;

/// A type as a node of [`TypeGraph`].
///
/// safegraph requires node keys to be [`IndexKey`] (`Clone + Eq + Ord + Hash`) and
/// [`Display`](std::fmt::Display), which `syn::Type` is not (it has no `Ord` or `Display`), so a
/// type is compared, hashed and displayed by its tokens.
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

// SAFETY: `Eq`, `Ord` and `Hash` are those of the tokens (a `String`), so they are consistent with
// each other and deterministic, and none of them (nor `Clone`) panics.
unsafe impl IndexKey for TypeKey {}

/// A graph whose nodes are keyed by [`TypeKey`] (in a `HashMap`), and whose edges are stored in a
/// `Vec` with no payload, so an edge holds its endpoints only once (as the endpoints).
type CompositionGraph =
    LinkedAdjEdgeGraph<HashMap<TypeKey, NodeRepr<u32>>, Vec<((), EdgeRepr<TypeKey, u32>)>>;

/// The types found by the analysis, and their composition.
#[derive(Default)]
pub(crate) struct TypeGraph {
    /// An edge goes from a type to each internable type it is composed of.
    graph: CompositionGraph,
    /// The types appearing as roots (not as components of another internable type).
    pub(crate) roots: HashSet<TypeKey>,
    /// The types which must be interned (they directly depend on the type context).
    pub(crate) must_intern: HashSet<TypeKey>,
    /// The local generics (e.g. of trait methods or GATs) which each type depends on (empty if
    /// none, see [`Referrer::local_generics()`](crate::Referrer::local_generics())). A type must
    /// depend on the same local generics wherever it appears, since the Referrer finds leaked types
    /// by the type alone.
    pub(crate) local_generics: HashMap<TypeKey, Generics>,
}

impl TypeGraph {
    /// Adds an occurrence of `ty` (a component of `parent`, or a root), and returns its key.
    /// Returns `Err` if it conflicts with another occurrence of the same type, depending on
    /// other local generics.
    pub(crate) fn add(
        &mut self,
        ty: &Type,
        parent: Option<&TypeKey>,
        is_must_intern: bool,
        local_generics: Generics,
    ) -> std::result::Result<TypeKey, ()> {
        let key = TypeKey::new(ty);
        match self.local_generics.get(&key) {
            Some(existing) if existing != &local_generics => return Err(()),
            Some(_) => (),
            None => {
                self.graph.insert_node(key.clone()).ok().unwrap();
                self.local_generics.insert(key.clone(), local_generics);
            }
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
        Ok(key)
    }

    fn add_edge(&mut self, from: TypeKey, to: TypeKey) {
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
        self.graph.len_node() == 0
    }

    /// Returns the types (in no particular order).
    pub(crate) fn keys(&self) -> impl Iterator<Item = &TypeKey> {
        self.local_generics.keys()
    }

    /// Returns the components of `key`.
    pub(crate) fn children(&self, key: &TypeKey) -> Vec<TypeKey> {
        self.graph.neighbor_indices_from(key).collect()
    }

    /// Returns a type in `other` which depends on other local generics than in this graph (the
    /// smallest one by its tokens, for a deterministic error).
    pub(crate) fn find_conflict<'a>(&self, other: &'a TypeGraph) -> Option<&'a TypeKey> {
        other
            .local_generics
            .iter()
            .filter(|(key, generics)| {
                matches!(self.local_generics.get(*key), Some(existing) if existing != *generics)
            })
            .map(|(key, _)| key)
            .min()
    }

    /// Merges `other` into this graph (which must not conflict, see
    /// [`TypeGraph::find_conflict()`]).
    pub(crate) fn merge(&mut self, other: TypeGraph) {
        for key in other.keys() {
            if !self.graph.contains_node_index(key) {
                self.graph.insert_node(key.clone()).ok().unwrap();
            }
        }
        for from in other.keys() {
            for to in other.children(from) {
                self.add_edge(from.clone(), to);
            }
        }
        for (key, generics) in other.local_generics {
            self.local_generics.entry(key).or_insert(generics);
        }
        self.roots.extend(other.roots);
        self.must_intern.extend(other.must_intern);
    }
}

/// Returns `true` if `start` or any of its descendants satisfies `pred`.
pub(crate) fn reaches(
    start: &TypeKey,
    children: &HashMap<&TypeKey, Vec<TypeKey>>,
    pred: impl Fn(&TypeKey) -> bool,
) -> bool {
    pred(start) || descendants(start, children).iter().any(pred)
}

/// Returns the descendants of `start` (excluding `start` itself, unless it is in a cycle).
pub(crate) fn descendants(
    start: &TypeKey,
    children: &HashMap<&TypeKey, Vec<TypeKey>>,
) -> Vec<TypeKey> {
    let mut visited = HashSet::new();
    let mut stack: Vec<TypeKey> = children.get(start).cloned().unwrap_or_default();
    let mut result = Vec::new();
    while let Some(key) = stack.pop() {
        if visited.insert(key.clone()) {
            stack.extend(children.get(&key).into_iter().flatten().cloned());
            result.push(key);
        }
    }
    result
}

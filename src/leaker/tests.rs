//! Tests of the reduction on graphs written by hand: the entries that an earlier reduction kept
//! (`TypeGraph::kept()`) against what a later scope merges in.

use super::*;
use crate::graph::TypeKey;

fn key(context: usize, ty: &str) -> NodeKey {
    NodeKey {
        context: ContextId::new(context),
        ty: TypeKey::new(&parse_str(ty).unwrap()),
    }
}

/// Adds `ty` in the context 0 to `graph`, under `parent`.
fn add(graph: &mut TypeGraph, ty: &str, parent: Option<&NodeKey>) -> NodeKey {
    graph.add(ContextId::new(0), &parse_str(ty).unwrap(), parent)
}

/// A type that depends on the context, and may be an entry.
fn depends(graph: &mut TypeGraph, ty: &str, parent: Option<&NodeKey>) -> NodeKey {
    let key = add(graph, ty, parent);
    graph.must_intern.insert(key.clone());
    graph.required_candidates.insert(key.clone());
    key
}

/// A tuple of two relative types, which one cut covers.
fn tuple(graph: &mut TypeGraph, parent: Option<&NodeKey>) -> NodeKey {
    let tuple = add(graph, "(R1, R2)", parent);
    graph.required_candidates.insert(tuple.clone());
    depends(graph, "R1", Some(&tuple));
    depends(graph, "R2", Some(&tuple));
    tuple
}

/// A type of an external crate: it may be an optional entry.
fn external(graph: &mut TypeGraph, ty: &str, parent: Option<&NodeKey>) -> NodeKey {
    let key = add(graph, ty, parent);
    graph.optional.insert(key.clone());
    graph.optional_candidates.insert(key.clone());
    key
}

/// The entries that `graph` is reduced to: the required ones, then the optional ones.
fn entries(graph: &TypeGraph) -> (Vec<String>, Vec<String>) {
    let (required, optional) = reduce(graph);
    let (required, optional) = TypeGraph::kept(required, optional).into_kept_entries();
    let show = |keys: Vec<NodeKey>| keys.iter().map(|key| key.ty.tokens()).collect();
    (show(required), show(optional))
}

fn strings(strings: &[&str]) -> Vec<String> {
    strings.iter().map(|s| s.to_string()).collect()
}

/// The graph that an earlier reduction left: the tuple is a required entry, though `R1` and `R2`,
/// which made it so, are gone, and `::ext::X` is an optional one.
fn pruned() -> TypeGraph {
    let (required, optional) = (vec![key(0, "(R1, R2)")], vec![key(0, ":: ext :: X")]);
    TypeGraph::kept(required, optional)
}

/// A graph is reduced until a type is merged in, and again after the next reduction.
#[test]
fn a_graph_is_reduced_until_a_type_is_merged_in() {
    assert!(TypeGraph::default().is_reduced());
    let mut graph = pruned();
    assert!(graph.is_reduced());
    graph.merge(TypeGraph::default());
    assert!(graph.is_reduced());
    // Even the type of a kept entry: the flags of the later scope may change the entries.
    let mut later = TypeGraph::default();
    tuple(&mut later, None);
    graph.merge(later);
    assert!(!graph.is_reduced());
    let (required, optional) = reduce(&graph);
    assert!(TypeGraph::kept(required, optional).is_reduced());
}

#[test]
fn a_kept_entry_keeps_its_kind() {
    let (required, optional) = entries(&pruned());
    assert_eq!(required, ["(R1 , R2)"]);
    assert_eq!(optional, [":: ext :: X"]);
}

#[test]
fn reducing_a_reduced_graph_changes_nothing() {
    let mut graph = TypeGraph::default();
    tuple(&mut graph, None);
    external(&mut graph, "::ext::X", None);
    let w = external(&mut graph, "::ext::W<R3>", None);
    depends(&mut graph, "R3", Some(&w));
    let (required, optional) = reduce(&graph);
    let once = TypeGraph::kept(required, optional);
    assert!(once.is_reduced());
    let (required, optional) = reduce(&once);
    let twice = TypeGraph::kept(required, optional);
    assert_eq!(entries(&graph), entries(&once));
    assert_eq!(entries(&once), entries(&twice));
}

/// A later scope may forbid a type to be a required entry (its copy would be cyclic): a kept entry
/// stays one, as the copy of the earlier scope is built from it. The later scope may get a copy
/// that does not compile; that is the price of pruning.
#[test]
fn a_kept_required_entry_stays_when_a_later_scope_forbids_it() {
    let mut later = TypeGraph::default();
    let t = tuple(&mut later, None);
    later.no_entry.insert(t);
    // Alone, the later scope has the parts as the entries.
    assert_eq!(entries(&later).0, ["R1", "R2"]);
    let mut graph = pruned();
    graph.merge(later);
    assert_eq!(entries(&graph).0, ["(R1 , R2)"]);
}

/// The same for a kept optional entry: the later scope forbids it as the optional entry it is.
#[test]
fn a_kept_optional_entry_stays_when_a_later_scope_forbids_it() {
    let mut later = TypeGraph::default();
    let x = external(&mut later, "::ext::X", None);
    later.no_entry.insert(x);
    assert_eq!(entries(&later), (strings(&[]), strings(&[])));
    let mut graph = pruned();
    graph.merge(later);
    assert_eq!(entries(&graph).1, [":: ext :: X"]);
}

/// A later type that has the kept entry as a component does not replace it, whether or not the
/// later scope forbids the entry: the parent is built from it.
#[test]
fn a_kept_entry_stays_when_a_later_type_holds_it() {
    let mut later = TypeGraph::default();
    let parent = add(&mut later, "((R1, R2), R3)", None);
    later.required_candidates.insert(parent.clone());
    let t = tuple(&mut later, Some(&parent));
    later.no_entry.insert(t);
    depends(&mut later, "R3", Some(&parent));
    // Alone, the later scope has the parent as the one entry. With the kept tuple, which it does
    // not replace, it has `R3` and the tuple: more entries, as one reduction at the end has fewer.
    assert_eq!(entries(&later).0, ["((R1 , R2) , R3)"]);
    let mut graph = pruned();
    graph.merge(later);
    assert_eq!(
        entries(&graph),
        (strings(&["(R1 , R2)", "R3"]), strings(&[":: ext :: X"]))
    );
}

/// A kept optional entry that a later scope shows to hold a required type is a required entry.
#[test]
fn a_kept_optional_entry_becomes_required_when_it_holds_a_required_type() {
    let mut later = TypeGraph::default();
    let x = external(&mut later, "::ext::X", None);
    depends(&mut later, "R", Some(&x));
    later.required_candidates.insert(x);
    let mut graph = pruned();
    graph.merge(later);
    assert_eq!(
        entries(&graph),
        (strings(&["(R1 , R2)", ":: ext :: X"]), strings(&[]))
    );
}

#![doc = include_str!("./README.md")]

use petgraph::graph::{DefaultIx, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Graph;
use proc_macro2::Span;
use std::collections::{HashMap, HashSet};
use syn::parse::Parse;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};

pub use syn;

/// The entry point of this crate.
pub struct Leaker {
    /// Referred by [`Leaker::finish()`]
    generics: Generics,
    graph: Graph<Type, ()>,
    map: HashMap<Type, NodeIndex<DefaultIx>>,
    must_intern_nodes: HashSet<NodeIndex<DefaultIx>>,
    root_nodes: HashSet<NodeIndex<DefaultIx>>,
    /// Marker path. It defaults to the type name when the input is enum or struct. When the input
    /// is a trait, the marker path should refers a type called marker type.
    ///
    /// Referred by [`Leaker::finish()`].
    pub implementor_type_fn: Box<dyn Fn(PathArguments) -> Type>,
}

/// Error represents that the type is not internable.
#[derive(Debug, Clone)]
pub struct NotInternableError(pub Span);

impl std::fmt::Display for NotInternableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Not internable")
    }
}

impl std::error::Error for NotInternableError {}

/// Encode [`GenericParam`]s to a type.
pub fn encode_generics_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericArgument>) -> Type {
    Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: iter
            .into_iter()
            .map(|param| -> Type {
                match param {
                    GenericArgument::Lifetime(lifetime) => {
                        parse_quote!(& #lifetime ())
                    }
                    GenericArgument::Const(expr) => {
                        parse_quote!([(); #expr as usize])
                    }
                    GenericArgument::Type(ty) => ty.clone(),
                    _ => panic!(),
                }
            })
            .collect(),
    })
}

/// Represents result of [`Leaker::check()`], holding the cause in its tuple item.
#[derive(Clone, Debug)]
pub enum CheckResult {
    /// The type must be interned because it directly depends on the type context.
    MustIntern(Span),
    /// The type must be interned because one of the types which it constituted with must be
    /// interned.
    MustInternOrInherit(Span),
    /// The type must not be interned, because `type_leak` cannot intern this. (e.g. `impl Trait`s,
    /// `_`, trait bounds with relative trait path, ...)
    MustNotIntern(Span),
    /// Other.
    Neutral,
}

struct AnalyzeVisitor<'a> {
    leaker: &'a mut Leaker,
    generics: Generics,
    parent: Option<NodeIndex<DefaultIx>>,
    error: Option<Span>,
}

impl<'a, 'ast> Visit<'ast> for AnalyzeVisitor<'a> {
    fn visit_trait_item_fn(&mut self, i: &TraitItemFn) {
        let mut visitor = AnalyzeVisitor {
            leaker: &mut self.leaker,
            generics: self.generics.clone(),
            parent: self.parent.clone(),
            error: self.error.clone(),
        };
        for g in &i.sig.generics.params {
            visitor.generics.params.push(g.clone());
        }
        let mut i = i.clone();
        i.default = None;
        i.semi_token = Some(Default::default());
        syn::visit::visit_trait_item_fn(&mut visitor, &mut i);
        self.error = visitor.error;
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        let mut visitor = AnalyzeVisitor {
            leaker: &mut self.leaker,
            generics: self.generics.clone(),
            parent: self.parent.clone(),
            error: self.error.clone(),
        };
        for g in &i.generics.params {
            visitor.generics.params.push(g.clone());
        }
        syn::visit::visit_trait_item_type(&mut visitor, i);
        self.error = visitor.error;
    }

    fn visit_type(&mut self, i: &Type) {
        match self.leaker.check(&self.generics, i) {
            Err((_, s)) | Ok(CheckResult::MustNotIntern(s)) => {
                // Emit error and terminate searching
                self.error = Some(s);
            }
            o => {
                let child = self
                    .leaker
                    .map
                    .entry(i.clone())
                    .or_insert_with(|| self.leaker.graph.add_node(i.clone()))
                    .clone();
                if let Some(parent) = &self.parent {
                    self.leaker
                        .graph
                        .add_edge(parent.clone(), child.clone(), ());
                } else {
                    self.leaker.root_nodes.insert(child.clone());
                }
                if let Ok(CheckResult::MustIntern(_)) = o {
                    // Terminate searching
                    self.leaker.must_intern_nodes.insert(child.clone());
                } else {
                    // Perform recuesive call
                    let parent = self.parent;
                    self.parent = Some(child);
                    syn::visit::visit_type(self, i);
                    self.parent = parent;
                }
            }
        }
    }

    fn visit_trait_bound(&mut self, i: &TraitBound) {
        if i.path.leading_colon.is_none() {
            // Trait bounds with relative oath
            self.error = Some(i.span());
        } else {
            syn::visit::visit_trait_bound(self, i)
        }
    }

    fn visit_expr_path(&mut self, i: &ExprPath) {
        if i.path.leading_colon.is_none() {
            // Value or trait bounds with relative oath
            self.error = Some(i.span());
        } else {
            syn::visit::visit_expr_path(self, i)
        }
    }
}

impl Leaker {
    /// Initialize with [`ItemStruct`]
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
    /// let leaker = Leaker::from_struct(&test_struct).unwrap();
    /// ```
    pub fn from_struct(input: &ItemStruct) -> std::result::Result<Self, NotInternableError> {
        let name = input.ident.clone();
        let mut leaker = Leaker::with_generics_and_implementor(
            input.generics.clone(),
            Box::new(move |args: PathArguments| parse_quote!(#name #args)),
        );
        let mut visitor = AnalyzeVisitor {
            leaker: &mut leaker,
            parent: None,
            error: None,
            generics: input.generics.clone(),
        };
        visitor.visit_item_struct(input);
        if let Some(e) = visitor.error {
            return Err(NotInternableError(e));
        }
        Ok(leaker)
    }

    /// Initialize with [`ItemEnum`]
    pub fn from_enum(input: &ItemEnum) -> std::result::Result<Self, NotInternableError> {
        let name = input.ident.clone();
        let mut leaker = Self::with_generics_and_implementor(
            input.generics.clone(),
            Box::new(move |args: PathArguments| parse_quote!(#name #args)),
        );
        let mut visitor = AnalyzeVisitor {
            leaker: &mut leaker,
            parent: None,
            error: None,
            generics: input.generics.clone(),
        };
        visitor.visit_item_enum(input);
        if let Some(e) = visitor.error {
            return Err(NotInternableError(e));
        }
        Ok(leaker)
    }

    /// Build an [`Leaker`] with given trait.
    ///
    /// Unlike enum nor struct, it requires ~alternative path~, an absolute path of a struct
    /// which is declared the same crate with the leaker trait and also visible from
    /// [`Referrer`]s' context. That struct is used as an `impl` target of `Repeater`
    /// instead of the Leaker's path.
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let s: ItemTrait = parse_quote!{
    ///     pub trait MyTrait<T, U> {
    ///         fn func(self, t: T) -> U;
    ///     }
    /// };
    /// let alternate: ItemStruct = parse_quote!{
    ///     pub struct MyAlternate;
    /// };
    /// let _ = Leaker::from_trait(&s, Box::new(|_| parse_quote!(::path::to::Implementor)));
    /// ```
    pub fn from_trait(
        input: &ItemTrait,
        implementor_type_fn: Box<dyn Fn(PathArguments) -> Type>,
    ) -> std::result::Result<Self, NotInternableError> {
        let mut leaker =
            Self::with_generics_and_implementor(input.generics.clone(), implementor_type_fn);
        let mut visitor = AnalyzeVisitor {
            leaker: &mut leaker,
            parent: None,
            error: None,
            generics: input.generics.clone(),
        };
        visitor.visit_item_trait(input);
        if let Some(e) = visitor.error {
            return Err(NotInternableError(e));
        }
        Ok(leaker)
    }

    /// Initialize empty [`Leaker`] with given generics.
    ///
    /// Types, consts, lifetimes defined in the [`Generics`] is treated as "no needs to be interned" although
    /// they looks like relative path names.
    pub fn with_generics_and_implementor(
        generics: Generics,
        implementor_type_fn: Box<dyn Fn(PathArguments) -> Type>,
    ) -> Self {
        Self {
            generics,
            graph: Graph::new(),
            map: HashMap::new(),
            must_intern_nodes: HashSet::new(),
            root_nodes: HashSet::new(),
            implementor_type_fn,
        }
    }

    /// Intern the given type as a root node.
    pub fn intern(
        &mut self,
        generics: Generics,
        ty: &Type,
    ) -> std::result::Result<&mut Self, NotInternableError> {
        let mut visitor = AnalyzeVisitor {
            leaker: self,
            parent: None,
            error: None,
            generics,
        };
        visitor.visit_type(ty);
        if let Some(e) = visitor.error {
            Err(NotInternableError(e))
        } else {
            Ok(self)
        }
    }

    /// Check that the internableness of give `ty`. It returns `Err` in contradiction (the type
    /// must and must not be interned).
    ///
    ///
    /// See [`CheckResult`].
    pub fn check(
        &self,
        generics: &Generics,
        ty: &Type,
    ) -> std::result::Result<CheckResult, (Span, Span)> {
        use syn::visit::Visit;
        #[derive(Clone)]
        struct Visitor {
            generic_lifetimes: Vec<Lifetime>,
            generic_idents: Vec<Ident>,
            impossible: Option<Span>,
            must: Option<(isize, Span)>,
        }

        const _: () = {
            use syn::visit::Visit;
            impl<'a> Visit<'a> for Visitor {
                fn visit_type(&mut self, i: &Type) {
                    match i {
                        Type::BareFn(TypeBareFn {
                            lifetimes,
                            inputs,
                            output,
                            ..
                        }) => {
                            let mut visitor = self.clone();
                            visitor.must = None;
                            visitor.generic_lifetimes.extend(
                                lifetimes
                                    .as_ref()
                                    .map(|ls| {
                                        ls.lifetimes.iter().map(|gp| {
                                            if let GenericParam::Lifetime(lt) = gp {
                                                lt.lifetime.clone()
                                            } else {
                                                panic!()
                                            }
                                        })
                                    })
                                    .into_iter()
                                    .flatten(),
                            );
                            for input in inputs {
                                visitor.visit_type(&input.ty);
                            }
                            if let ReturnType::Type(_, output) = output {
                                visitor.visit_type(output.as_ref());
                            }
                            if self.impossible.is_none() {
                                self.impossible = visitor.impossible;
                            }
                            match (self.must.clone(), visitor.must.clone()) {
                                (Some((s_l, _)), Some((v_l, v_m))) if v_l + 1 < s_l => {
                                    self.must = Some((v_l + 1, v_m))
                                }
                                (None, Some((v_l, v_m))) => self.must = Some((v_l + 1, v_m)),
                                _ => (),
                            }
                            return;
                        }
                        Type::TraitObject(TypeTraitObject { bounds, .. }) => {
                            for bound in bounds {
                                match bound {
                                    TypeParamBound::Trait(TraitBound {
                                        lifetimes, path, ..
                                    }) => {
                                        let mut visitor = self.clone();
                                        visitor.must = None;
                                        visitor.generic_lifetimes.extend(
                                            lifetimes
                                                .as_ref()
                                                .map(|ls| {
                                                    ls.lifetimes.iter().map(|gp| {
                                                        if let GenericParam::Lifetime(lt) = gp {
                                                            lt.lifetime.clone()
                                                        } else {
                                                            panic!()
                                                        }
                                                    })
                                                })
                                                .into_iter()
                                                .flatten(),
                                        );
                                        visitor.visit_path(path);
                                        if self.impossible.is_none() {
                                            self.impossible = visitor.impossible;
                                        }
                                        match (self.must.clone(), visitor.must.clone()) {
                                            (Some((s_l, _)), Some((v_l, v_m))) if v_l + 1 < s_l => {
                                                self.must = Some((v_l + 1, v_m))
                                            }
                                            (None, Some((v_l, v_m))) => {
                                                self.must = Some((v_l + 1, v_m))
                                            }
                                            _ => (),
                                        }
                                        return;
                                    }
                                    TypeParamBound::Verbatim(_) => {
                                        self.impossible = Some(bound.span());
                                        return;
                                    }
                                    _ => (),
                                }
                            }
                        }
                        Type::ImplTrait(_)
                        | Type::Infer(_)
                        | Type::Macro(_)
                        | Type::Verbatim(_) => {
                            self.impossible = Some(i.span());
                        }
                        _ => (),
                    }
                    let mut visitor = self.clone();
                    visitor.must = None;
                    syn::visit::visit_type(&mut visitor, i);
                    if visitor.impossible.is_some() {
                        self.impossible = visitor.impossible;
                    }
                    match (self.must.clone(), visitor.must.clone()) {
                        (Some((s_l, _)), Some((v_l, v_m))) if v_l + 1 < s_l => {
                            self.must = Some((v_l + 1, v_m))
                        }
                        (None, Some((v_l, v_m))) => self.must = Some((v_l + 1, v_m)),
                        _ => (),
                    }
                }
                fn visit_qself(&mut self, i: &QSelf) {
                    if i.as_token.is_none() {
                        self.impossible = Some(i.span());
                    }
                    syn::visit::visit_qself(self, i)
                }
                fn visit_lifetime(&mut self, i: &Lifetime) {
                    if i.to_string() != "'static"
                        && !self.generic_lifetimes.iter().any(|lt| lt == i)
                    {
                        self.must = Some((-1, i.span()));
                    }
                    syn::visit::visit_lifetime(self, i)
                }
                fn visit_expr(&mut self, i: &Expr) {
                    match i {
                        Expr::Closure(_)
                        | Expr::Assign(_)
                        | Expr::Verbatim(_)
                        | Expr::Macro(_)
                        | Expr::Infer(_) => {
                            self.impossible = Some(i.span());
                        }
                        _ => (),
                    }
                    syn::visit::visit_expr(self, i)
                }
                fn visit_path(&mut self, i: &Path) {
                    if matches!(i.segments.iter().next(), Some(PathSegment { ident, arguments }) if ident == "Self" && arguments.is_none())
                    {
                        // do nothing
                    } else {
                        match (i.leading_colon, i.get_ident()) {
                            // i is a generic parameter
                            (None, Some(ident))
                                if self.generic_idents.contains(&ident) || ident == "Self" => {}
                            // relative path, not a generic parameter
                            (None, _) => {
                                self.must = Some((-1, i.span()));
                            }
                            // absolute path
                            (Some(_), _) => (),
                        }
                    }

                    syn::visit::visit_path(self, i)
                }
            }
        };
        let mut visitor = Visitor {
            generic_lifetimes: generics
                .params
                .iter()
                .filter_map(|gp| {
                    if let GenericParam::Lifetime(lt) = gp {
                        Some(lt.lifetime.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            generic_idents: generics
                .params
                .iter()
                .filter_map(|gp| match gp {
                    GenericParam::Type(TypeParam { ident, .. })
                    | GenericParam::Const(ConstParam { ident, .. }) => Some(ident.clone()),
                    _ => None,
                })
                .collect(),
            impossible: None,
            must: None,
        };
        visitor.visit_type(ty);
        match (visitor.must, visitor.impossible) {
            (None, None) => Ok(CheckResult::Neutral),
            (Some((0, span)), None) => Ok(CheckResult::MustIntern(span)),
            (Some((_, span)), None) => Ok(CheckResult::MustInternOrInherit(span)),
            (None, Some(span)) => Ok(CheckResult::MustNotIntern(span)),
            (Some((_, span0)), Some(span1)) => Err((span0, span1)),
        }
    }

    /// Finish building the [`Leaker`] and convert it into [`Referrer`].
    pub fn finish(
        self,
        mut repeater_path_fn: impl FnMut(usize) -> Path,
    ) -> (Vec<ItemImpl>, Referrer) {
        let (impl_generics, _, where_clause) = self.generics.split_for_impl();
        let id_map: HashMap<_, _> = self
            .root_nodes
            .iter()
            .enumerate()
            .map(|(n, idx)| {
                (
                    self.graph.node_weight(idx.clone()).expect("hello3").clone(),
                    n,
                )
            })
            .collect();
        let path_args = PathArguments::AngleBracketed(AngleBracketedGenericArguments {
            colon2_token: None,
            lt_token: Token![<](Span::call_site()),
            args: self
                .generics
                .params
                .iter()
                .map(|param| match param {
                    GenericParam::Lifetime(lifetime_param) => {
                        GenericArgument::Lifetime(lifetime_param.lifetime.clone())
                    }
                    GenericParam::Type(TypeParam { ident, .. }) => {
                        GenericArgument::Type(parse_quote!(#ident))
                    }
                    GenericParam::Const(ConstParam { ident, .. }) => {
                        GenericArgument::Const(parse_quote!(#ident))
                    }
                })
                .collect(),
            gt_token: Token![>](Span::call_site()),
        });
        (
            id_map
                .iter()
                .map(|(ty, n)| {
                    parse2(quote! {
                impl #impl_generics #{repeater_path_fn(*n)} for #{(*self.implementor_type_fn)(path_args.clone())} #where_clause {
                    type Type = #ty;
                }
            }).expect("hello4")
                })
                .collect(),
            Referrer { map: id_map },
        )
    }

    #[cfg_attr(doc, aquamarine::aquamarine)]
    /// Reduce nodes to decrease cost of {`Leaker`}'s implementation.
    ///
    /// # Algorithm
    ///
    /// To consult the algorithm, see the following [`Leaker`]'s input:
    ///
    /// ```ignore
    /// pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
    ///     field1: MyType1,
    ///     field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
    ///     field3: &'a (T1, T2),
    /// }
    /// ```
    ///
    /// [`Leaker`], when initialized with [`Leaker::from_struct()`], analyze the AST and
    /// construct a DAG which represents all (internable) types and the dependency relations like
    /// this:
    ///
    /// ```mermaid
    /// graph TD
    ///   1["★MyType1
    ///   (Node 1)"]
    ///   2["★(MyType2, MyType3#lt;MyType1#gt;, MyType4, MyType5)
    ///   (Node 2)"] -->3["MyType2
    ///   (Node 3)"]
    ///   2 -->4["MyType3#lt;MyType1#gt;
    ///   (Node 4)"]
    ///   2 -->0["★MyType4
    ///   (Node 0)"]
    ///   2 -->5["MyType5
    ///   (Node 5)"]
    ///   6["★&a (T1, T2)
    ///   (Node 6)"] -->7["(T1, T2)
    ///   (Node 7)"]
    ///   7 -->8["T1
    ///   (Node 8)"]
    ///   7 -->9["T2
    ///   (Node 9)"]
    ///   classDef redNode stroke:#ff0000;
    ///   class 0,1,3,4,5 redNode;
    /// ```
    ///
    /// The **red** node is flagged as [`CheckResult::MustIntern`] by [`Leaker::check()`] (which means
    /// the type literature depends on the type context, so it must be interned).
    ///
    /// This algorithm reduce the nodes, remaining that all root type (annotated with ★) can be
    /// expressed with existing **red** nodes.
    ///
    /// Here, there are some choice in its freedom:
    ///
    /// - Intern all **red** nodes and ignore others (because other nodes are not needed to be
    /// intern, or constructable with red nodes)
    /// - Not directly intern **red** nodes; intern common ancessors instead if it is affordable.
    ///
    /// So finally, it results like:
    ///
    /// ```mermaid
    /// graph TD
    ///   0["MyType4
    ///   (Node 0)"]
    ///   1["MyType1
    ///   (Node 1)"]
    ///   2["(MyType2, MyType3 #lt; MyType1 #gt;, MyType4, MyType5)
    ///   (Node 2)"]
    ///   classDef redNode stroke:#ff0000;
    ///   class 0,1,2 redNode;
    /// ```
    ///
    pub fn reduce_roots(&mut self) {
        self.reduce_unreachable_nodes();
        self.reduce_obvious_nodes();
        // TODO: unobvious root reduction with heaulistics
    }

    fn reduce_obvious_nodes(&mut self) {
        let mut must_intern_nodes: Vec<_> = self.must_intern_nodes.iter().cloned().collect();
        let mut root_nodes: Vec<_> = self.root_nodes.iter().cloned().collect();
        let nodes: Vec<_> = self
            .graph
            .node_indices()
            .filter_map(|n| {
                let mut it = self.graph.edges(n.clone());
                if let Some(edge) = it.next() {
                    if let None = it.next() {
                        return Some((edge.source(), edge.target(), edge.id()));
                    }
                }
                None
            })
            .collect();
        for (node1, node2, edge) in nodes {
            self.graph.remove_edge(edge);
            for (edge_in_source, edge_in) in self
                .graph
                .edges_directed(node1, petgraph::Direction::Incoming)
                .map(|er| (er.source(), er.id()))
                .collect::<Vec<_>>()
            {
                self.graph.update_edge(edge_in_source, node2, ());
                self.graph.remove_edge(edge_in);
            }
            must_intern_nodes.iter_mut().for_each(|n| {
                if n == &node1 {
                    *n = node2.clone()
                }
            });
            root_nodes.iter_mut().for_each(|n| {
                if n == &node1 {
                    *n = node2.clone()
                }
            });
            self.graph.remove_node(node1);
        }
        self.must_intern_nodes = must_intern_nodes.into_iter().collect();
        self.root_nodes = root_nodes.into_iter().collect();
    }

    fn reduce_unreachable_nodes(&mut self) {
        let reachable_forward = get_reachable_nodes(&self.graph, self.root_nodes.iter().cloned());
        self.graph.reverse();
        let reachable_backward =
            get_reachable_nodes(&self.graph, self.must_intern_nodes.iter().cloned());
        self.graph.reverse();
        let reachable = reachable_forward
            .intersection(&reachable_backward)
            .cloned()
            .collect::<HashSet<_>>();
        // self.graph = self.graph.filter_map(
        //     |ix, node| reachable.contains(&ix).then_some(node.clone()),
        //     |ix, _| {
        //         let (e1, e2) = self.graph.edge_endpoints(ix).unwrap();
        //         (reachable.contains(&e1) && reachable.contains(&e2)).then_some(())
        //     },
        // );
        let mut new_graph = Graph::new();
        let mut node_map = HashMap::new();
        for node in self.graph.node_indices() {
            if reachable.contains(&node) {
                let new_node = new_graph.add_node(self.graph[node].clone());
                node_map.insert(node, new_node);
            }
        }
        for edge in self.graph.edge_indices() {
            let (n1, n2) = self.graph.edge_endpoints(edge).unwrap();
            if let (Some(nn1), Some(nn2)) = (node_map.get(&n1), node_map.get(&n2)) {
                new_graph.add_edge(*nn1, *nn2, ());
            }
        }
        self.root_nodes = self
            .root_nodes
            .iter()
            .filter_map(|n| node_map.get(n).cloned())
            .collect();
        self.must_intern_nodes = self
            .must_intern_nodes
            .iter()
            .filter_map(|n| node_map.get(n).cloned())
            .collect();
        let _ = std::mem::replace(&mut self.graph, new_graph);
    }
}

fn get_reachable_nodes<N, E>(
    graph: &Graph<N, E>,
    roots: impl IntoIterator<Item = NodeIndex<DefaultIx>>,
) -> HashSet<NodeIndex<DefaultIx>> {
    let mut ret: HashSet<_> = roots.into_iter().collect();
    let mut frontier = ret.clone();
    loop {
        let mut buf: HashSet<_> = frontier
            .iter()
            .map(|node| graph.neighbors(node.clone()))
            .flatten()
            .collect();
        let mut count = 0;
        for node in &buf {
            if ret.insert(node.clone()) {
                count += 1;
            }
        }
        if count == 0 {
            break;
        }
        std::mem::swap(&mut frontier, &mut buf);
    }
    ret
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    map: HashMap<Type, usize>,
}

impl Referrer {
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.map.keys()
    }

    pub fn into_visitor<F: FnMut(usize) -> Path + Clone>(
        self,
        leaker_ty: Type,
        repeater_path_fn: F,
    ) -> Visitor<F> {
        Visitor(self, leaker_ty, repeater_path_fn)
    }

    pub fn expand(
        &self,
        ty: Type,
        leaker_ty: &Type,
        repeater_path_fn: impl FnMut(usize) -> Path,
    ) -> Type {
        use syn::fold::Fold;

        struct Folder<'a, F>(&'a Referrer, &'a Type, F);
        impl<'a, F: FnMut(usize) -> Path> Fold for Folder<'a, F> {
            fn fold_type(&mut self, ty: Type) -> Type {
                if let Some(idx) = self.0.map.get(&ty) {
                    parse2(quote! {
                        <#{&self.1} as #{&self.2(*idx)}>::Type
                    })
                    .unwrap()
                } else {
                    syn::fold::fold_type(self, ty)
                }
            }
        }
        let mut folder = Folder(self, leaker_ty, repeater_path_fn);
        folder.fold_type(ty)
    }
}

#[derive(Clone, Debug)]
pub struct Visitor<F>(Referrer, Type, F);

impl<F: FnMut(usize) -> Path + Clone> Visitor<F> {
    fn with_generics(&mut self, generics: &mut Generics) -> Self {
        let mut visitor = self.clone();
        for gp in generics.params.iter_mut() {
            if let GenericParam::Type(TypeParam { ident, .. }) = gp {
                visitor.0.map.remove(&parse_quote!(#ident));
            }
            visitor.visit_generic_param_mut(gp);
        }
        visitor
    }

    fn with_signature(&mut self, sig: &mut Signature) -> Self {
        let mut visitor = self.with_generics(&mut sig.generics);
        for input in sig.inputs.iter_mut() {
            visitor.visit_fn_arg_mut(input);
        }
        visitor.visit_return_type_mut(&mut sig.output);
        visitor
    }
}

impl<F: FnMut(usize) -> Path + Clone> VisitMut for Visitor<F> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        *i = self.0.expand(i.clone(), &self.1, &mut self.2);
    }
    fn visit_item_struct_mut(&mut self, i: &mut ItemStruct) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_fields_mut(&mut i.fields);
    }
    fn visit_item_enum_mut(&mut self, i: &mut ItemEnum) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        for variant in i.variants.iter_mut() {
            visitor.visit_variant_mut(variant);
        }
    }
    fn visit_item_trait_mut(&mut self, i: &mut ItemTrait) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        for supertrait in i.supertraits.iter_mut() {
            visitor.visit_type_param_bound_mut(supertrait);
        }
        for item in i.items.iter_mut() {
            visitor.visit_trait_item_mut(item);
        }
    }
    fn visit_item_union_mut(&mut self, i: &mut ItemUnion) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_fields_named_mut(&mut i.fields);
    }
    fn visit_item_type_mut(&mut self, i: &mut ItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
    }
    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature(&mut i.sig);
        visitor.visit_block_mut(i.block.as_mut());
    }
    fn visit_trait_item_fn_mut(&mut self, i: &mut TraitItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature(&mut i.sig);
        if let Some(block) = &mut i.default {
            visitor.visit_block_mut(block);
        }
    }
    fn visit_trait_item_type_mut(&mut self, i: &mut TraitItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        for bound in i.bounds.iter_mut() {
            visitor.visit_type_param_bound_mut(bound);
        }
        if let Some((_, ty)) = &mut i.default {
            visitor.visit_type_mut(ty);
        }
    }
    fn visit_trait_item_const_mut(&mut self, i: &mut TraitItemConst) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
        if let Some((_, expr)) = &mut i.default {
            visitor.visit_expr_mut(expr);
        }
    }
    fn visit_block_mut(&mut self, i: &mut Block) {
        let mut visitor = self.clone();
        for stmt in &i.stmts {
            match stmt {
                Stmt::Item(Item::Struct(ItemStruct { ident, .. }))
                | Stmt::Item(Item::Enum(ItemEnum { ident, .. }))
                | Stmt::Item(Item::Union(ItemUnion { ident, .. }))
                | Stmt::Item(Item::Trait(ItemTrait { ident, .. }))
                | Stmt::Item(Item::Type(ItemType { ident, .. })) => {
                    visitor.0.map.remove(&parse_quote!(#ident));
                }
                _ => (),
            }
        }
        for stmt in i.stmts.iter_mut() {
            visitor.visit_stmt_mut(stmt);
        }
    }
}

impl Parse for Referrer {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let mut map = HashMap::new();
        let map_content: syn::parse::ParseBuffer<'_>;
        parenthesized!(map_content in input);
        while !map_content.is_empty() {
            let key: Type = map_content.parse()?;
            map_content.parse::<Token![:]>()?;
            let value: LitInt = map_content.parse()?;
            map.insert(key, value.base10_parse()?);
            if map_content.is_empty() {
                break;
            }
            map_content.parse::<Token![,]>()?;
        }
        Ok(Self { map })
    }
}

impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        tokens.extend(quote! {
            (
                #(for (key, val) in &self.map), {
                    #key: #val
                }
            )
        })
    }
}

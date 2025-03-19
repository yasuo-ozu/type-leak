#![doc = include_str!("./README.md")]

use petgraph::graph::{DefaultIx, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Graph;
use proc_macro2::Span;
use std::collections::{HashMap, HashSet};
use syn::parse::Parse;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::*;
use template_quote::{quote, ToTokens};

pub use syn;

fn random() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

/// The entry point of this crate.
#[derive(Clone, Debug)]
pub struct Leaker {
    /// Referred by [`Leaker::check()`]
    pub generics: Generics,
    graph: Graph<Type, ()>,
    map: HashMap<Type, NodeIndex<DefaultIx>>,
    must_intern_nodes: HashSet<NodeIndex<DefaultIx>>,
    root_nodes: HashSet<NodeIndex<DefaultIx>>,
    /// Marker path. It defaults to the type name when the input is enum or struct. When the input
    /// is a trait, the marker path should refers a type called marker type.
    ///
    /// Referred by [`Leaker::finish()`].
    pub default_marker_path: Path,
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
    parent: Option<NodeIndex<DefaultIx>>,
    error: Option<Span>,
}

impl<'a, 'ast> Visit<'ast> for AnalyzeVisitor<'a> {
    fn visit_type(&mut self, i: &Type) {
        match self.leaker.check(i) {
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
    /// let leaker = Leaker::from_item_struct(&test_struct).unwrap();
    /// ```
    pub fn from_item_struct(input: &ItemStruct) -> std::result::Result<Self, NotInternableError> {
        let ident = &input.ident;
        let mut leaker = Leaker::with_generics_and_ty(input.generics.clone(), parse_quote!(#ident));
        let mut visitor = AnalyzeVisitor {
            leaker: &mut leaker,
            parent: None,
            error: None,
        };
        visitor.visit_item_struct(input);
        if let Some(e) = visitor.error {
            return Err(NotInternableError(e));
        }
        Ok(leaker)
    }

    /// Initialize with [`ItemEnum`]
    pub fn from_item_enum(input: &ItemEnum) -> std::result::Result<Self, NotInternableError> {
        let name = &input.ident;
        let mut leaker = Self::with_generics_and_ty(input.generics.clone(), parse_quote!(#name));
        let mut visitor = AnalyzeVisitor {
            leaker: &mut leaker,
            parent: None,
            error: None,
        };
        visitor.visit_item_enum(input);
        if let Some(e) = visitor.error {
            return Err(NotInternableError(e));
        }
        Ok(leaker)
    }

    /// Build an [`Leaker`] with given trait.
    ///
    /// Unlike enum nor struct, it requires ~marker type~, which is declared the same crate with
    /// the leaker trait and also visible from [`Refferrer`]s' context. Marker types are used as
    /// leakers, instead of actual trait definitions.
    pub fn from_trait(
        input: &ItemTrait,
        marker_name: Option<&Ident>,
    ) -> std::result::Result<(Self, ItemStruct), NotInternableError> {
        let marker_name = marker_name.cloned().unwrap_or_else(|| {
            Ident::new(
                &format!("__TypeLeak_Marker_{}", random()),
                Span::call_site(),
            )
        });
        let mut leaker =
            Self::with_generics_and_ty(input.generics.clone(), parse_quote!(#marker_name));
        let mut visitor = AnalyzeVisitor {
            leaker: &mut leaker,
            parent: None,
            error: None,
        };
        visitor.visit_item_trait(input);
        if let Some(e) = visitor.error {
            return Err(NotInternableError(e));
        }
        let (impl_generics, _, _) = input.generics.split_for_impl();
        let phantom_tys = input
            .generics
            .params
            .iter()
            .filter_map(|p| match &p {
                GenericParam::Lifetime(_) => Some(quote!(&#p ())),
                GenericParam::Type(_) => Some(quote!(#p)),
                GenericParam::Const(_) => None,
            })
            .collect::<Vec<_>>();
        Ok((
            leaker,
            parse2(quote! {
                #{&input.vis} struct #marker_name #impl_generics {
                    _phantom: ::core::marker::PhantomData<(#(#phantom_tys,)*)>,
                    _infallible: ::core::convert::Infallible,
                }
            })
            .unwrap(),
        ))
    }

    /// Initialize empty [`Leaker`] with given generics.
    ///
    /// Types, consts, lifetimes defined in the [`Generics`] is treated as "no needs to be interned" although
    /// they looks like relative path names.
    pub fn with_generics_and_ty(generics: Generics, default_marker_path: Path) -> Self {
        Self {
            generics,
            graph: Graph::new(),
            map: HashMap::new(),
            must_intern_nodes: HashSet::new(),
            root_nodes: HashSet::new(),
            default_marker_path,
        }
    }

    /// Intern the given type as a root node.
    pub fn intern(&mut self, ty: &Type) -> std::result::Result<&mut Self, NotInternableError> {
        let mut visitor = AnalyzeVisitor {
            leaker: self,
            parent: None,
            error: None,
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
    pub fn check(&self, ty: &Type) -> std::result::Result<CheckResult, (Span, Span)> {
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
                        | Expr::Verbatim(_)
                        | Expr::MethodCall(_)
                        | Expr::Macro(_)
                        | Expr::Infer(_) => {
                            self.impossible = Some(i.span());
                        }
                        _ => (),
                    }
                    syn::visit::visit_expr(self, i)
                }
                fn visit_path(&mut self, i: &Path) {
                    match (i.leading_colon, i.get_ident()) {
                        // i is a generic parameter
                        (None, Some(ident)) if self.generic_idents.contains(&ident) => {}
                        // relative path, not a generic parameter
                        (None, _) => {
                            self.must = Some((-1, i.span()));
                        }
                        // absolute path
                        (Some(_), _) => (),
                    }
                    syn::visit::visit_path(self, i)
                }
            }
        };
        let mut visitor = Visitor {
            generic_lifetimes: self
                .generics
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
            generic_idents: self
                .generics
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
        leaker_ty: Type,
        repeater_path: Path,
        marker_path: Option<&Path>,
    ) -> (Vec<ItemImpl>, Referrer) {
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();
        let id_map: HashMap<_, _> = self
            .root_nodes
            .iter()
            .enumerate()
            .map(|(n, idx)| (self.graph.node_weight(idx.clone()).unwrap().clone(), n))
            .collect();
        let marker_path = marker_path.unwrap_or(&self.default_marker_path);
        (
            id_map
                .iter()
                .map(|(ty, n)| {
                    parse2(quote! {
                impl #impl_generics #repeater_path<#n> for #marker_path #ty_generics #where_clause {
                    type Type = #ty;
                }
            }).unwrap()
                })
                .collect(),
            Referrer {
                map: id_map,
                leaker_ty,
                default_repeater_path: repeater_path,
            },
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
    /// [`Leaker`], when initialized with [`Leaker::from_item_struct()`], analyze the AST and
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
        // self.reduce_obvious_nodes();
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
            }
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
        dbg!(&reachable);
        self.graph = self.graph.filter_map(
            |ix, node| reachable.contains(&ix).then_some(node.clone()),
            |ix, _| {
                let (e1, e2) = self.graph.edge_endpoints(ix).unwrap();
                (reachable.contains(&e1) && reachable.contains(&e2)).then_some(())
            },
        );
        self.root_nodes = self.root_nodes.intersection(&reachable).cloned().collect();
        self.must_intern_nodes = self
            .must_intern_nodes
            .intersection(&reachable)
            .cloned()
            .collect();
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
        if !buf.iter().any(|node| ret.insert(node.clone())) {
            break;
        }
        std::mem::swap(&mut frontier, &mut buf);
    }
    ret
}

pub struct Referrer {
    map: HashMap<Type, usize>,
    leaker_ty: Type,
    default_repeater_path: Path,
}

impl Referrer {
    pub fn expand(&self, ty: Type, repeater_path: Option<Path>) -> Type {
        use syn::fold::Fold;

        struct Folder<'a>(&'a Referrer);
        impl<'a> Fold for Folder<'a> {
            fn fold_type(&mut self, ty: Type) -> Type {
                if let Some(idx) = self.0.map.get(&ty) {
                    parse2(quote! {
                        <#{&self.0.leaker_ty} as #{&self.0.default_repeater_path}<#idx>>::Type
                    })
                    .unwrap()
                } else {
                    syn::fold::fold_type(self, ty)
                }
            }
        }
        let mut folder = Folder(self);
        folder.fold_type(ty)
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
        let leaker_ty = input.parse()?;
        input.parse::<Token![,]>()?;
        let repeater_path = input.parse()?;
        let _ = input.parse::<Token![,]>();
        Ok(Self {
            map,
            leaker_ty,
            default_repeater_path: repeater_path,
        })
    }
}

impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        tokens.extend(quote! {
            ( #(for (key, val) in &self.map), {
                #key: #val
            })
            #{&self.leaker_ty},
            #{&self.default_repeater_path}
        })
    }
}

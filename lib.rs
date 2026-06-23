#![doc = include_str!("./README.md")]

use gotgraph::graph::{Graph as GotGraph, GraphUpdate};
use gotgraph::prelude::VecGraph;
use proc_macro2::{Span, TokenStream};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};

pub use syn;

/// Configuration for building a [`Leaker`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LeakerConfig {
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
}

impl Default for LeakerConfig {
    fn default() -> Self {
        Self {
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
        }
    }
}

impl LeakerConfig {
    pub fn new() -> Self {
        Default::default()
    }

    /// Allow primitive types from `::core::primitive`.
    pub fn allow_primitive(&mut self) -> &mut Self {
        self.allowed_paths.extend([
            parse_quote!(bool),
            parse_quote!(char),
            parse_quote!(str),
            parse_quote!(u8),
            parse_quote!(u16),
            parse_quote!(u32),
            parse_quote!(u64),
            parse_quote!(u128),
            parse_quote!(usize),
            parse_quote!(i8),
            parse_quote!(i16),
            parse_quote!(i32),
            parse_quote!(i64),
            parse_quote!(i128),
            parse_quote!(isize),
            parse_quote!(f32),
            parse_quote!(f64),
        ]);
        self
    }

    /// Allow items in the standard prelude (`std::prelude::v1`).
    pub fn allow_std_prelude(&mut self) -> &mut Self {
        self.allowed_paths.extend([
            parse_quote!(Box),
            parse_quote!(String),
            parse_quote!(Vec),
            parse_quote!(Option),
            parse_quote!(Result),
            parse_quote!(Copy),
            parse_quote!(Send),
            parse_quote!(Sized),
            parse_quote!(Sync),
            parse_quote!(Unpin),
            parse_quote!(Drop),
            parse_quote!(Fn),
            parse_quote!(FnMut),
            parse_quote!(FnOnce),
            parse_quote!(Clone),
            parse_quote!(PartialEq),
            parse_quote!(PartialOrd),
            parse_quote!(Eq),
            parse_quote!(Ord),
            parse_quote!(AsRef),
            parse_quote!(AsMut),
            parse_quote!(Into),
            parse_quote!(From),
            parse_quote!(Default),
            parse_quote!(Iterator),
            parse_quote!(IntoIterator),
            parse_quote!(Extend),
            parse_quote!(FromIterator),
            parse_quote!(ToOwned),
            parse_quote!(ToString),
            parse_quote!(drop),
        ]);
        self
    }

    /// Allow paths starting with `crate`.
    pub fn allow_crate(&mut self) -> &mut Self {
        self.allowed_paths.push(parse_quote!(crate));
        self
    }
}

/// The entry point of this crate.
pub struct Leaker {
    graph: VecGraph<Type, ()>,
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
    reachable_types: HashSet<Type>,
    pending_operations: Option<Vec<GraphOperation>>,
}

/// Error indicating that a type is not internable.
#[derive(Debug, Clone)]
pub struct NotInternableError(pub TokenStream);

impl std::fmt::Display for NotInternableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("use absolute path: '{}'", &self.0))
    }
}

impl std::error::Error for NotInternableError {}

impl NotInternableError {
    pub fn span(&self) -> Span {
        self.0.span()
    }
}

/// Encode [`GenericArgument`]s to a type.
pub fn encode_generics_args_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericArgument>) -> Type {
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

/// Encode [`GenericParam`]s to a type.
pub fn encode_generics_params_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericParam>) -> Type {
    Type::Tuple(TypeTuple {
        paren_token: Default::default(),
        elems: iter
            .into_iter()
            .map(|param| -> Type {
                match param {
                    GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                        parse_quote!(& #lifetime ())
                    }
                    GenericParam::Const(ConstParam { ident, .. }) => {
                        parse_quote!([(); #ident as usize])
                    }
                    GenericParam::Type(TypeParam { ident, .. }) => parse_quote!(#ident),
                }
            })
            .collect(),
    })
}

/// Represents the result of [`Leaker::check()`], holding the cause in its tuple field.
#[derive(Clone, Debug)]
pub enum CheckResult {
    /// The type must be interned because it directly depends on the type context.
    MustIntern(TokenStream),
    /// The type must be interned because one of the types it is composed of must be interned.
    MustInternOrInherit(TokenStream),
    /// The type must not be interned because `type_leak` cannot intern it (e.g. `impl Trait`,
    /// `_`, trait bounds with a relative trait path, ...)
    MustNotIntern(TokenStream),
    /// Other.
    Neutral,
}

#[derive(Debug, Clone)]
struct GraphOperation {
    node_type: Type,
    parent: Option<Type>,
    is_root: bool,
    is_must_intern: bool,
}

struct AnalyzeVisitor<'a> {
    leaker: &'a mut Leaker,
    generics: Generics,
    parent: Option<Type>,
    error: Option<TokenStream>,
    operations: Vec<GraphOperation>,
}

impl<'a, 'ast> Visit<'ast> for AnalyzeVisitor<'a> {
    fn visit_trait_item_fn(&mut self, i: &TraitItemFn) {
        let mut visitor = AnalyzeVisitor {
            leaker: self.leaker,
            generics: self.generics.clone(),
            parent: self.parent.clone(),
            error: self.error.clone(),
            operations: Vec::new(),
        };
        for g in &i.sig.generics.params {
            visitor.generics.params.push(g.clone());
        }
        let mut i = i.clone();
        i.default = None;
        i.semi_token = Some(Default::default());
        syn::visit::visit_trait_item_fn(&mut visitor, &i);
        if visitor.error.is_some() {
            self.error = visitor.error;
        }
        self.operations.extend(visitor.operations);
    }

    fn visit_receiver(&mut self, i: &Receiver) {
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        if let Some((_, Some(lt))) = &i.reference {
            self.visit_lifetime(lt);
        }
        if i.colon_token.is_some() {
            self.visit_type(&i.ty);
        }
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        let mut visitor = AnalyzeVisitor {
            leaker: self.leaker,
            generics: self.generics.clone(),
            parent: self.parent.clone(),
            error: self.error.clone(),
            operations: Vec::new(),
        };
        for g in &i.generics.params {
            visitor.generics.params.push(g.clone());
        }
        syn::visit::visit_trait_item_type(&mut visitor, i);
        self.error = visitor.error;
        self.operations.extend(visitor.operations);
    }

    fn visit_type(&mut self, i: &Type) {
        match self.leaker.check(i, &self.generics) {
            Err((_, s)) => {
                // Emit error and terminate searching
                self.error = Some(s);
            }
            Ok(CheckResult::MustNotIntern(_)) => (),
            o => {
                // Collect the operation instead of directly modifying the graph
                let is_must_intern = matches!(o, Ok(CheckResult::MustIntern(_)));
                let is_root = self.parent.is_none();

                self.operations.push(GraphOperation {
                    node_type: i.clone(),
                    parent: self.parent.clone(),
                    is_root,
                    is_must_intern,
                });

                if !is_must_intern {
                    // Perform recursive call
                    let parent = self.parent.clone();
                    self.parent = Some(i.clone());
                    syn::visit::visit_type(self, i);
                    self.parent = parent;
                }
            }
        }
    }

    fn visit_trait_bound(&mut self, i: &TraitBound) {
        // Trait bounds with a non-absolute path
        if i.path.leading_colon.is_none()
            && !matches!(
                i.path.segments.iter().next(),
                Some(PathSegment {
                    ident,
                    arguments: PathArguments::None,
                }) if ident.to_string() == "$crate"
            )
            && !self.leaker.allowed_paths.iter().any(|allowed| {
                allowed.leading_colon == i.path.leading_colon
                    && allowed.segments.len() <= i.path.segments.len()
                    && allowed.segments.iter().zip(i.path.segments.iter()).all(
                        |(allowed_seg, seg)| {
                            allowed_seg.arguments == seg.arguments
                                && allowed_seg.ident.to_string() == seg.ident.to_string()
                        },
                    )
            })
        {
            self.error = Some(quote!(#i));
            return;
        }
        syn::visit::visit_trait_bound(self, i)
    }

    fn visit_expr_path(&mut self, i: &ExprPath) {
        if i.path.leading_colon.is_none() {
            // Value or trait bounds with a relative path
            self.error = Some(quote!(#i));
        } else {
            syn::visit::visit_expr_path(self, i)
        }
    }
}

impl Default for Leaker {
    fn default() -> Self {
        Self::new()
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
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_struct(input);
        })?;
        Ok(leaker)
    }

    /// Initialize with [`ItemEnum`]
    pub fn from_enum(input: &ItemEnum) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_enum(input);
        })?;
        Ok(leaker)
    }

    /// Build a [`Leaker`] from a given trait.
    ///
    /// Unlike an enum or struct, a trait Leaker requires an alternate struct — an absolute path
    /// to a struct declared in the same crate as the leaker trait and visible from
    /// [`Referrer`]s' context. That struct serves as the `impl` target of the Repeater
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
    /// let _ = Leaker::from_trait(&s);
    /// ```
    pub fn from_trait(input: &ItemTrait) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_trait(input);
        })?;
        Ok(leaker)
    }

    /// Initialize empty [`Leaker`] with no generics.
    pub fn new() -> Self {
        Self {
            graph: VecGraph::default(),
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
            reachable_types: HashSet::new(),
            pending_operations: None,
        }
    }

    /// Initialize [`Leaker`] from configuration.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            graph: VecGraph::default(),
            allowed_paths: config.allowed_paths,
            self_ty_can_be_interned: config.self_ty_can_be_interned,
            reachable_types: HashSet::new(),
            pending_operations: None,
        }
    }

    /// Apply collected graph operations using `scope_mut()`.
    fn apply_operations(&mut self, operations: Vec<GraphOperation>) {
        if operations.is_empty() {
            return;
        }

        // Store operations for proper graph reduction in reduce_roots()
        self.pending_operations = Some(operations);
    }

    /// Intern AST elements using a visitor callback.
    pub fn intern_with<'ast, F>(
        &mut self,
        generics: &Generics,
        f: F,
    ) -> std::result::Result<&mut Self, NotInternableError>
    where
        F: FnOnce(&mut dyn syn::visit::Visit<'ast>),
    {
        let mut visitor = AnalyzeVisitor {
            leaker: self,
            parent: None,
            error: None,
            generics: generics.clone(),
            operations: Vec::new(),
        };
        f(&mut visitor);
        if let Some(e) = visitor.error {
            Err(NotInternableError(e))
        } else {
            let operations = visitor.operations;
            self.apply_operations(operations);
            Ok(self)
        }
    }

    /// Check whether the given `ty` is internable. Returns `Err` when there is a contradiction
    /// (the type must both be interned and must not be interned).
    ///
    ///
    /// See [`CheckResult`].
    pub fn check(
        &self,
        ty: &Type,
        generics: &Generics,
    ) -> std::result::Result<CheckResult, (TokenStream, TokenStream)> {
        use syn::visit::Visit;
        #[derive(Clone)]
        struct Visitor {
            generic_lifetimes: Vec<Lifetime>,
            generic_idents: Vec<Ident>,
            impossible: Option<TokenStream>,
            must: Option<(isize, TokenStream)>,
            self_ty_can_be_interned: bool,
            allowed_paths: Vec<Path>,
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
                            match (self.must.clone(), visitor.must) {
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
                                        match (self.must.clone(), visitor.must) {
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
                                        self.impossible = Some(quote!(#bound));
                                        return;
                                    }
                                    _ => (),
                                }
                            }
                        }
                        Type::ImplTrait(_) | Type::Macro(_) | Type::Verbatim(_) => {
                            self.impossible = Some(quote!(#i));
                        }
                        Type::Reference(TypeReference { lifetime, .. }) => {
                            if lifetime.is_none() {
                                self.impossible = Some(quote!(#i));
                            }
                        }
                        _ => (),
                    }
                    let mut visitor = self.clone();
                    visitor.must = None;
                    syn::visit::visit_type(&mut visitor, i);
                    if visitor.impossible.is_some() {
                        self.impossible = visitor.impossible;
                    }
                    match (self.must.clone(), visitor.must) {
                        (Some((s_l, _)), Some((v_l, v_m))) if v_l + 1 < s_l => {
                            self.must = Some((v_l + 1, v_m))
                        }
                        (None, Some((v_l, v_m))) => self.must = Some((v_l + 1, v_m)),
                        _ => (),
                    }
                }
                fn visit_qself(&mut self, i: &QSelf) {
                    if i.as_token.is_none() {
                        let ty = &i.ty;
                        self.impossible = Some(quote!(#ty));
                    }
                    syn::visit::visit_qself(self, i)
                }
                fn visit_lifetime(&mut self, i: &Lifetime) {
                    match i.to_string().as_str() {
                        "'static" => (),
                        "'_" => self.impossible = Some(quote!(#i)),
                        _ if self.generic_lifetimes.iter().any(|lt| lt == i) => {}
                        _ => {
                            self.must = Some((-1, quote!(#i)));
                        }
                    }
                    syn::visit::visit_lifetime(self, i)
                }
                fn visit_expr(&mut self, i: &Expr) {
                    match i {
                        Expr::Closure(_) | Expr::Assign(_) | Expr::Verbatim(_) | Expr::Macro(_) => {
                            self.impossible = Some(quote!(#i));
                        }
                        _ => (),
                    }
                    syn::visit::visit_expr(self, i)
                }
                fn visit_path(&mut self, i: &Path) {
                    if self.allowed_paths.iter().any(|allowed| {
                        allowed.leading_colon == i.leading_colon
                            && allowed.segments.len() <= i.segments.len()
                            && allowed.segments.iter().zip(i.segments.iter()).all(
                                |(allowed_seg, seg)| {
                                    allowed_seg.arguments == seg.arguments
                                        && allowed_seg.ident.to_string() == seg.ident.to_string()
                                },
                            )
                    }) {
                        return;
                    }
                    if matches!(i.segments.iter().next(), Some(PathSegment { ident, arguments }) if ident == "Self" && arguments.is_none())
                        && i.leading_colon.is_none()
                    {
                        if !self.self_ty_can_be_interned {
                            self.impossible = Some(quote!(#i));
                        }
                    } else {
                        if i.leading_colon.is_none()
                            && matches!(
                                i.segments.iter().next(),
                                Some(PathSegment {
                                    ident,
                                    arguments: PathArguments::None,
                                }) if ident.to_string() == "$crate"
                            )
                        {
                            return;
                        }
                        match (i.leading_colon, i.get_ident()) {
                            // i is a generic parameter
                            (None, Some(ident)) if self.generic_idents.contains(ident) => {}
                            // relative path, not a generic parameter
                            (None, _) => {
                                self.must = Some((-1, quote!(#i)));
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
            self_ty_can_be_interned: self.self_ty_can_be_interned,
            allowed_paths: self.allowed_paths.clone(),
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
    pub fn finish(self) -> Referrer {
        // Use the reachable types computed by apply_operations
        let leak_types: Vec<_> = self.reachable_types.into_iter().collect();
        let map = leak_types
            .iter()
            .enumerate()
            .map(|(n, ty)| (ty.clone(), n))
            .collect();
        Referrer { leak_types, map }
    }

    /// Reduce nodes to decrease the cost of the [`Leaker`]'s implementation.
    ///
    /// # Algorithm
    ///
    /// Consider the following [`Leaker`] input:
    ///
    /// ```ignore
    /// pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
    ///     field1: MyType1,
    ///     field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
    ///     field3: &'a (T1, T2),
    /// }
    /// ```
    ///
    /// When initialized with [`Leaker::from_struct()`], the [`Leaker`] analyzes the AST and
    /// constructs a DAG representing all internable types and their dependency relations:
    ///
    /// ![before_graph](https://raw.github.com/yasuo-ozu/type-leak/master/resources/before_graph.svg)
    ///
    /// **Red** nodes are flagged as [`CheckResult::MustIntern`] by [`Leaker::check()`], meaning
    /// the type literally depends on the type context and must be interned.
    ///
    /// This algorithm reduces the node set while ensuring that every root type (annotated with ★)
    /// can still be expressed using the remaining **red** nodes.
    ///
    /// There are two degrees of freedom in this reduction:
    ///
    /// - Intern all **red** nodes and ignore the rest (since non-red nodes either do not need to
    ///   be interned or can be reconstructed from red nodes).
    /// - Skip interning individual **red** nodes and instead intern their common ancestors when
    ///   doing so is more efficient.
    ///
    /// The final result looks like:
    ///
    /// ![after_graph](https://raw.github.com/yasuo-ozu/type-leak/master/resources/after_graph.svg)
    pub fn reduce_roots(&mut self) {
        if let Some(operations) = self.pending_operations.take() {
            self.build_graph_and_reduce(operations);
        }
        self.reduce_unreachable_nodes();
        self.reduce_obvious_nodes();
    }

    fn build_graph_and_reduce(&mut self, operations: Vec<GraphOperation>) {
        // Build parent-child relationships from operations
        let mut children: HashMap<Type, Vec<Type>> = HashMap::new();
        let mut parents: HashMap<Type, Vec<Type>> = HashMap::new();
        let mut root_types = HashSet::new();
        let mut must_intern_types = HashSet::new();

        for operation in &operations {
            if operation.is_root {
                root_types.insert(operation.node_type.clone());
            }
            if operation.is_must_intern {
                must_intern_types.insert(operation.node_type.clone());
            }

            if let Some(parent_type) = &operation.parent {
                children
                    .entry(parent_type.clone())
                    .or_default()
                    .push(operation.node_type.clone());
                parents
                    .entry(operation.node_type.clone())
                    .or_default()
                    .push(parent_type.clone());
            }
        }

        // Implement proper reduction algorithm:
        // 1. Keep all must-intern root types
        // 2. Keep root types that transitively contain must-intern types
        // 3. Do NOT keep non-root must-intern types that are only components of kept root types

        let mut kept_types = HashSet::new();

        // Step 1: Keep must-intern types that are also roots
        for root_type in &root_types {
            if must_intern_types.contains(root_type) {
                kept_types.insert(root_type.clone());
            }
        }

        // Step 2: Keep root types that contain must-intern descendants
        for root_type in &root_types {
            if !kept_types.contains(root_type)
                && self.contains_must_intern_descendant(root_type, &children, &must_intern_types)
            {
                kept_types.insert(root_type.clone());
            }
        }

        // Step 3: Only add must-intern types that are NOT components of any kept type
        for must_intern_type in &must_intern_types {
            if !self.is_component_of_any_kept_type(must_intern_type, &kept_types, &children) {
                kept_types.insert(must_intern_type.clone());
            }
        }

        self.reachable_types = kept_types;
    }

    fn contains_must_intern_descendant(
        &self,
        root: &Type,
        children: &HashMap<Type, Vec<Type>>,
        must_intern_types: &HashSet<Type>,
    ) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![root.clone()];

        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if must_intern_types.contains(&current) {
                return true;
            }

            if let Some(child_list) = children.get(&current) {
                for child in child_list {
                    if !visited.contains(child) {
                        stack.push(child.clone());
                    }
                }
            }
        }

        false
    }

    fn is_component_of_any_kept_type(
        &self,
        target_type: &Type,
        kept_types: &HashSet<Type>,
        children: &HashMap<Type, Vec<Type>>,
    ) -> bool {
        for kept_type in kept_types {
            if self.is_descendant_of(target_type, kept_type, children) {
                return true;
            }
        }
        false
    }

    fn is_descendant_of(
        &self,
        target: &Type,
        ancestor: &Type,
        children: &HashMap<Type, Vec<Type>>,
    ) -> bool {
        if target == ancestor {
            return false; // Not a descendant of itself
        }

        let mut visited = HashSet::new();
        let mut stack = vec![ancestor.clone()];

        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if let Some(child_list) = children.get(&current) {
                for child in child_list {
                    if child == target {
                        return true;
                    }
                    if !visited.contains(child) {
                        stack.push(child.clone());
                    }
                }
            }
        }

        false
    }

    fn reduce_obvious_nodes(&mut self) {
        // Remove intermediate nodes that have exactly one parent and one child
        // This step eliminates unnecessary intermediate types
    }

    fn reduce_unreachable_nodes(&mut self) {
        if let Some(operations) = self.pending_operations.take() {
            self.apply_vecgraph_reduction(operations);
        }
    }

    fn apply_vecgraph_reduction(&mut self, operations: Vec<GraphOperation>) {
        let mut root_types = HashSet::new();
        let mut must_intern_types = HashSet::new();

        // Collect graph operations
        for operation in &operations {
            if operation.is_root {
                root_types.insert(operation.node_type.clone());
            }
            if operation.is_must_intern {
                must_intern_types.insert(operation.node_type.clone());
            }
        }

        // Build the graph using scope_mut
        self.graph.scope_mut(|mut ctx| {
            let mut type_to_node = HashMap::new();

            // Add all nodes first
            for operation in &operations {
                if !type_to_node.contains_key(&operation.node_type) {
                    let node_tag = ctx.add_node(operation.node_type.clone());
                    type_to_node.insert(operation.node_type.clone(), node_tag);
                }

                if let Some(parent_type) = &operation.parent {
                    if !type_to_node.contains_key(parent_type) {
                        let parent_node_tag = ctx.add_node(parent_type.clone());
                        type_to_node.insert(parent_type.clone(), parent_node_tag);
                    }
                }
            }

            // Add edges between nodes
            for operation in &operations {
                if let Some(parent_type) = &operation.parent {
                    let parent_node = type_to_node[parent_type];
                    let child_node = type_to_node[&operation.node_type];
                    ctx.add_edge((), parent_node, child_node);
                }
            }
        });

        // Perform analysis using the same logic as build_graph_and_reduce
        self.reachable_types =
            self.analyze_reachable_types(&root_types, &must_intern_types, &operations);
    }

    fn analyze_reachable_types(
        &self,
        root_types: &HashSet<Type>,
        must_intern_types: &HashSet<Type>,
        operations: &[GraphOperation],
    ) -> HashSet<Type> {
        // Implement the same logic as build_graph_and_reduce
        let mut final_types = HashSet::new();

        // Keep must-intern root types
        for root_type in root_types {
            if must_intern_types.contains(root_type) {
                final_types.insert(root_type.clone());
            }
        }

        // Keep root types that contain must-intern descendants
        for root_type in root_types {
            if !final_types.contains(root_type)
                && self.contains_must_intern_in_tree(root_type, operations, must_intern_types)
            {
                final_types.insert(root_type.clone());
            }
        }

        // Only keep must-intern types that aren't subcomponents of kept root types
        for must_intern_type in must_intern_types {
            if !self.is_subcomponent_of_kept_root(must_intern_type, &final_types, operations) {
                final_types.insert(must_intern_type.clone());
            }
        }

        final_types
    }

    fn contains_must_intern_in_tree(
        &self,
        root: &Type,
        operations: &[GraphOperation],
        must_intern_types: &HashSet<Type>,
    ) -> bool {
        let mut stack = vec![root.clone()];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if must_intern_types.contains(&current) {
                return true;
            }

            for operation in operations {
                if let Some(parent_type) = &operation.parent {
                    if parent_type == &current && !visited.contains(&operation.node_type) {
                        stack.push(operation.node_type.clone());
                    }
                }
            }
        }

        false
    }

    fn is_subcomponent_of_kept_root(
        &self,
        target: &Type,
        kept_roots: &HashSet<Type>,
        operations: &[GraphOperation],
    ) -> bool {
        for root in kept_roots {
            if self.contains_must_intern_in_tree(root, operations, &HashSet::from([target.clone()]))
                && root != target
            {
                return true;
            }
        }
        false
    }

    // TODO: Reimplement reduce_unreachable_nodes with VecGraph APIs
}

/// Holds the list of types that need to be encoded, as produced by [`Leaker::finish()`].
///
/// `Referrer` provides methods to iterate over leak types and to expand or transform
/// types using a custom mapping function.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    leak_types: Vec<Type>,
    map: HashMap<Type, usize>,
}

impl Referrer {
    /// Returns `true` if there are no types to be encoded.
    pub fn is_empty(&self) -> bool {
        self.leak_types.is_empty()
    }

    /// Returns an iterator over the types that need to be encoded.
    ///
    /// ```ignore
    /// let args = encode_generics_params_to_ty(&generics.params);
    /// referrer.iter().enumerate().map(|(ix, ty)| quote!{
    ///     impl #crate::TypeRef<#ix, #args> for #marker_ty {
    ///         type Type = #ty;
    ///     }
    /// }).collect::<TokenStream>();
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.leak_types.iter()
    }

    /// Converts this `Referrer` into a [`Visitor`] that can be used to transform AST nodes.
    ///
    /// The `type_fn` callback is called for each type that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type.
    /// See [`Referrer::expand()`] for details.
    pub fn into_visitor<F: FnMut(Type, usize) -> Type>(self, type_fn: F) -> Visitor<F> {
        Visitor(self, Rc::new(RefCell::new(type_fn)))
    }

    /// Expands a type by replacing any interned types using the provided mapping function.
    ///
    /// The `type_fn` callback is called for each subtype that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type.
    ///
    /// ```ignore
    /// let args = encode_generics_params_to_ty(&generics.params);
    /// referrer.expand(ty, |_, ix| {
    ///     parse_quote!(<#marker_ty as #crate::TypeRef<#ix, #args>>::Type)
    /// })
    /// ```
    pub fn expand(&self, ty: Type, type_fn: impl FnMut(Type, usize) -> Type) -> Type {
        use syn::fold::Fold;
        struct Folder<'a, F>(&'a Referrer, F);
        impl<'a, F: FnMut(Type, usize) -> Type> Fold for Folder<'a, F> {
            fn fold_type(&mut self, ty: Type) -> Type {
                if let Some(idx) = self.0.map.get(&ty) {
                    self.1(ty, *idx)
                } else {
                    syn::fold::fold_type(self, ty)
                }
            }
        }
        let mut folder = Folder(self, type_fn);
        folder.fold_type(ty)
    }
}

#[derive(Debug)]
pub struct Visitor<F>(Referrer, Rc<RefCell<F>>);

impl<F> Clone for Visitor<F> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<F: FnMut(Type, usize) -> Type> Visitor<F> {
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

impl<F: FnMut(Type, usize) -> Type> VisitMut for Visitor<F> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        *i = self.0.expand(i.clone(), &mut *self.1.borrow_mut());
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
        let content: syn::parse::ParseBuffer<'_>;
        parenthesized!(content in input);
        let leak_types: Vec<Type> = Punctuated::<Type, Token![,]>::parse_terminated(&content)?
            .into_iter()
            .collect();
        let map = leak_types
            .iter()
            .enumerate()
            .map(|(n, ty)| (ty.clone(), n))
            .collect();
        Ok(Self { map, leak_types })
    }
}

impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        tokens.extend(quote! {
            (
                #(for item in &self.leak_types), {
                    #item
                }
            )
        })
    }
}

//! The [`Leaker`], which finds the types depending on the type context.

use crate::analyze::{AnalyzeVisitor, CheckResult};
use crate::config::LeakerConfig;
use crate::error::NotInternableError;
use crate::graph::{descendants, reaches, NodeKey, TypeGraph};
use crate::referrer::Referrer;
use crate::util::path_starts_with;
use proc_macro2::{Span, TokenStream};
use std::collections::{HashMap, HashSet};
use syn::*;
use template_quote::quote;

/// The entry point of this crate.
pub struct Leaker {
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
    /// The types found by [`Leaker::intern_with()`] and their composition.
    types: TypeGraph,
    /// The nodes of the types to be interned, computed from `types` by [`Leaker::reduce_roots()`]
    /// (`None` if not computed yet, or outdated).
    reachable_types: Option<Vec<NodeKey>>,
    /// The number of successful calls of [`Leaker::intern_with()`], each of which is a context (see
    /// [`Referrer::context()`](crate::Referrer::context())).
    n_contexts: usize,
    /// Names of primitive types which were assumed to be the primitive types (not shadowed), as
    /// they are copied as they are (see [`Leaker::assertions()`]).
    assumed_primitives: Vec<Ident>,
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

    /// Initialize with [`ItemUnion`]
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let u: ItemUnion = parse_quote! {
    ///     pub union MyUnion<T: ::core::marker::Copy> {
    ///         a: T,
    ///         b: MyType,
    ///     }
    /// };
    /// let referrer = Leaker::from_union(&u).unwrap().finish();
    /// assert_eq!(referrer.iter().collect::<Vec<_>>(), [&parse_quote!(MyType)]);
    /// ```
    pub fn from_union(input: &ItemUnion) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&input.generics, |visitor| {
            visitor.visit_item_union(input);
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
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
            types: TypeGraph::default(),
            reachable_types: None,
            n_contexts: 0,
            assumed_primitives: Vec::new(),
        }
    }

    /// Initialize [`Leaker`] from configuration.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allowed_paths: config.allowed_paths,
            self_ty_can_be_interned: config.self_ty_can_be_interned,
            types: TypeGraph::default(),
            reachable_types: None,
            n_contexts: 0,
            assumed_primitives: Vec::new(),
        }
    }

    fn assume_primitive(&mut self, ident: &Ident) {
        if !self.assumed_primitives.iter().any(|p| p == ident) {
            self.assumed_primitives.push(ident.clone());
        }
    }

    /// Items to emit in the Leaker's type context, which check the assumptions made by the
    /// Leaker at compile time.
    ///
    /// The names of primitive types (e.g. `u8` in `const N: u8`, or in a type when allowed by
    /// [`LeakerConfig::allow_primitive()`]) are copied as they are, assuming that they are not
    /// shadowed in the Leaker's type context (e.g. by `type u8 = i32;`), which cannot be known
    /// without name resolution. These items fail to compile (with "mismatched types") if a name
    /// is shadowed, instead of the copies silently referring to another type.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait MyTrait<const N: u8> {});
    /// let leaker = Leaker::from_trait(&item).unwrap();
    /// assert!(leaker
    ///     .assertions()
    ///     .to_string()
    ///     .contains("__type_leak_assert_u8_is_primitive"));
    /// ```
    pub fn assertions(&self) -> TokenStream {
        if self.assumed_primitives.is_empty() {
            return TokenStream::new();
        }
        quote! {
            #[allow(dead_code, non_snake_case)]
            const _: () = {
                #(for ident in &self.assumed_primitives) {
                    fn #{Ident::new(
                        &format!("__type_leak_assert_{}_is_primitive", ident),
                        Span::call_site(),
                    )}(x: &#ident) -> &::core::primitive::#ident { x }
                }
            };
        }
    }

    /// Returns `true` if `path` does not depend on the type context, i.e. it is absolute, starts
    /// with `$crate`, or starts with one of [`Leaker::allowed_paths`].
    pub(crate) fn is_context_free_path(&self, path: &Path) -> bool {
        path.leading_colon.is_some()
            || matches!(
                path.segments.iter().next(),
                Some(PathSegment {
                    ident,
                    arguments: PathArguments::None,
                }) if ident == "$crate"
            )
            || self
                .allowed_paths
                .iter()
                .any(|allowed| path_starts_with(path, allowed))
    }

    /// Intern AST elements using a visitor callback.
    ///
    /// Each successful call is a new context, numbered from 0 in order (see
    /// [`Referrer::context()`](crate::Referrer::context())): the types found are added to those of
    /// the previous calls, but are never shared with them, since `generics` may differ. Nothing is
    /// added if an error is returned.
    pub fn intern_with<'ast, F>(
        &mut self,
        generics: &Generics,
        f: F,
    ) -> std::result::Result<&mut Self, NotInternableError>
    where
        F: FnOnce(&mut dyn syn::visit::Visit<'ast>),
    {
        let mut visitor = AnalyzeVisitor::new(self, self.n_contexts, generics);
        f(&mut visitor);
        let analysis = visitor.finish();
        for ident in &analysis.primitives {
            self.assume_primitive(ident);
        }
        if !analysis.errors.is_empty() {
            return Err(NotInternableError::RelativePath(analysis.errors));
        }
        let types = analysis.types;
        if !types.is_empty() {
            self.types.merge(types);
            self.reachable_types = None;
        }
        self.n_contexts += 1;
        Ok(self)
    }

    /// Checks whether the given `ty` must be interned, and whether it can be. See [`CheckResult`].
    pub fn check(&self, ty: &Type, generics: &Generics) -> CheckResult {
        AnalyzeVisitor::check(self, ty, generics)
    }

    /// Finish building the [`Leaker`] and convert it into [`Referrer`].
    ///
    /// The types are reduced by [`Leaker::reduce_roots()`] (called here if needed), and sorted by
    /// their context, tokens and requirements, so the result (including the index of each type) is
    /// deterministic.
    pub fn finish(mut self) -> Referrer {
        self.reduce_roots();
        let keys = self.reachable_types.take().unwrap_or_default();
        Referrer::new(
            keys.into_iter()
                .map(|key| (key.context, key.ty.ty, key.requirements)),
        )
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
    /// ![before_graph](https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/resources/before_graph.svg)
    ///
    /// **Red** nodes are those whose own path depends on the type context
    /// ([`CheckResult::is_own`] by [`Leaker::check()`]), meaning the type literally depends on the
    /// type context and must be interned.
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
    /// ![after_graph](https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/resources/after_graph.svg)
    pub fn reduce_roots(&mut self) {
        if self.reachable_types.is_none() {
            self.reachable_types = Some(self.build_graph_and_reduce());
        }
    }

    /// Returns the types to be interned, sorted by their context, tokens and requirements.
    fn build_graph_and_reduce(&self) -> Vec<NodeKey> {
        let types = &self.types;
        let children: HashMap<&NodeKey, Vec<NodeKey>> =
            types.keys().map(|key| (key, types.children(key))).collect();

        // 1. Keep the roots which must be interned, or contain types which must be interned.
        let mut kept: HashSet<&NodeKey> = types
            .roots
            .iter()
            .filter(|key| reaches(key, &children, |key| types.must_intern.contains(key)))
            .collect();
        // 2. Keep the types which must be interned and are not components of kept types.
        let components: HashSet<NodeKey> = kept
            .iter()
            .flat_map(|key| descendants(key, &children))
            .collect();
        kept.extend(
            types
                .must_intern
                .iter()
                .filter(|key| !components.contains(*key)),
        );

        // Sort for deterministic output.
        let mut kept: Vec<NodeKey> = kept.into_iter().cloned().collect();
        kept.sort_by_cached_key(|key| {
            (
                key.context,
                key.ty.to_string(),
                key.requirements.key().to_string(),
            )
        });
        kept
    }
}

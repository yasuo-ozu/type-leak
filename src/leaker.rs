//! The [`Leaker`], which finds the types depending on the type context.

use crate::analyze::{Analysis, AnalyzeVisitor, CheckResult};
use crate::config::LeakerConfig;
use crate::error::NotInternableError;
use crate::graph::{descendants, reaches, NodeKey, TypeGraph};
use crate::referrer::Referrer;
use crate::requirements::{Contexts, Requirements};
use crate::util::{generic_param_name, path_starts_with, token_names};
use proc_macro2::{Span, TokenStream};
use std::collections::{BTreeSet, HashMap, HashSet};
use syn::punctuated::Punctuated;
use syn::*;
use template_quote::quote;

/// The entry point of this crate.
pub struct Leaker {
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
    /// Whether a path with a leading `::` (e.g. `::foo::X`) is context-free (default `true`).
    ///
    /// Such a path resolves against the external crates of the crate compiling it, so it is not
    /// portable to the Referrer's crate if that crate does not depend on `foo` (or renamed it, or
    /// uses another version). With `false`, only a path starting with `::core` (always
    /// available) or `$crate` is context-free (not even one starting with an
    /// [allowed path](Leaker::allowed_paths), which applies to relative paths only), and the
    /// others are treated like relative paths: leaked in a type (see
    /// [`Referrer::external_crates_require()`](crate::Referrer::external_crates_require()) and
    /// [`Referrer::must()`](crate::Referrer::must())), and
    /// [`NotInternableError::NotPortable`] in a trait bound, a value or the type of a const
    /// parameter. The Referrer tells which crates it can refer to (see
    /// [`Referrer::with_available_external_crates()`](crate::Referrer::with_available_external_crates())).
    pub allow_external_crates: bool,
    /// The types found by [`Leaker::intern_with()`] and their composition.
    types: TypeGraph,
    /// The nodes of the types to be interned, computed from `types` by [`Leaker::reduce_roots()`]
    /// (`None` if not computed yet, or outdated).
    reachable_types: Option<Vec<NodeKey>>,
    /// The requirements of the types found (see [`Referrer::context()`](crate::Referrer::context())).
    contexts: Contexts,
    /// The number of the successful calls of [`Leaker::intern_with()`].
    calls: usize,
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
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let s: ItemTrait = parse_quote!{
    ///     pub trait MyTrait<T, U> {
    ///         fn func(self, t: T) -> U;
    ///     }
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

    /// Initialize [`Leaker`] from a module.
    ///
    /// The module has no generics, and every item in it (structs, enums, unions, traits, type
    /// aliases, functions, impls and nested modules) is a nested item, so its generics are local
    /// generics, which the types depending on them are interned with (see
    /// [`Referrer::generics()`]). The bodies of functions and the values of consts are not
    /// analyzed. A module without content (`mod m;`) has nothing to analyze.
    ///
    /// The items declared in the module are copied with it, so paths to them are not leaked,
    /// like Rust resolves them: a name declared directly in the current module (not shadowed by
    /// a generic parameter), `self::` or `super::` followed by a name declared in that module
    /// (without leaving the analyzed module), and the names imported by `use` with an absolute
    /// path (starting with `::`). A type mentioning them is decomposed instead of being interned
    /// as a whole (e.g. only `Rel` is interned for `S<Rel>`, where `S` is declared in the
    /// module). Other relative paths (including the names imported by `use super::X` or `use
    /// crate::X`) are leaked.
    ///
    /// A relative path resolves in the module where it is written, so a leaked type depending on
    /// one (e.g. `Rel` in `mod m` and in `mod m::inner`, where `use super::super::Rel;` may
    /// import another one), or whose requirements do (what `Self` means, or a bound of its
    /// generics, mentioning a path which is not absolute, `crate::` or `$crate`), or which is a
    /// type macro, is keyed by the path of that module, relative to `input` (`[]` for
    /// `input` itself, as opposed to a type which depends on no module), together with its generics (see [`Referrer::context_module()`]). The
    /// Repeater's implementation of the type must be written inside that module, e.g. an
    /// attribute macro on `mod m` can emit the implementations of the types of `m::inner` inside
    /// it. A type which does not depend on its module (e.g. one with only absolute paths) is
    /// shared by all the modules, in the module `[]`.
    ///
    /// ```
    /// # use syn::*;
    /// # use type_leak::Leaker;
    /// let m: ItemMod = parse_quote!{
    ///     mod m {
    ///         pub struct S<T> { x: Rel<T> }
    ///         pub type A = Rel2;
    ///     }
    /// };
    /// let referrer = Leaker::from_module(&m).unwrap().finish();
    /// assert_eq!(referrer.iter().count(), 2);
    /// ```
    pub fn from_module(input: &ItemMod) -> std::result::Result<Self, NotInternableError> {
        let mut leaker = Self::new();
        leaker.intern_with(&Generics::default(), |visitor| {
            visitor.visit_item_mod(input);
        })?;
        Ok(leaker)
    }

    /// Initialize empty [`Leaker`] with no generics.
    pub fn new() -> Self {
        Self {
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
            allow_external_crates: true,
            types: TypeGraph::default(),
            reachable_types: None,
            contexts: Contexts::default(),
            calls: 0,
            assumed_primitives: Vec::new(),
        }
    }

    /// Initialize [`Leaker`] from configuration.
    pub fn from_config(config: LeakerConfig) -> Self {
        Self {
            allowed_paths: config.allowed_paths,
            self_ty_can_be_interned: config.self_ty_can_be_interned,
            allow_external_crates: config.allow_external_crates,
            types: TypeGraph::default(),
            reachable_types: None,
            contexts: Contexts::default(),
            calls: 0,
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
    /// with `$crate`, or is relative and starts with one of [`Leaker::allowed_paths`]. Unless
    /// [`Leaker::allow_external_crates`], an absolute path is context-free only if it starts with
    /// `::core`.
    pub(crate) fn is_context_free_path(&self, path: &Path) -> bool {
        (path.leading_colon.is_some()
            && (self.allow_external_crates
                || matches!(path.segments.first(), Some(seg) if seg.ident == "core")))
            || matches!(
                path.segments.iter().next(),
                Some(PathSegment {
                    ident,
                    arguments: PathArguments::None,
                }) if ident == "$crate"
            )
            || (path.leading_colon.is_none()
                && self
                    .allowed_paths
                    .iter()
                    .any(|allowed| path_starts_with(path, allowed)))
    }

    /// Intern AST elements using a visitor callback.
    ///
    /// `generics` are the generics of the Leaker, which the Repeater's implementations declare
    /// (the ones a type uses, see [`Referrer::generics()`](crate::Referrer::generics())). A type
    /// found is keyed by its tokens and its requirements: the generics it uses (resolved through
    /// their bounds and where predicates, with those of `generics` as the outermost scope) and
    /// what `Self` means if it mentions it. So the same type found by several calls (whatever
    /// the other generics of the calls are) is shared if its requirements are equal, and it is a
    /// separate entry otherwise (see [`Referrer::context()`](crate::Referrer::context())).
    /// Nothing is added if an error is returned.
    ///
    /// `Self` is the implementor of the visited trait, or the visited struct, enum, union or the
    /// self type of the visited impl (see [`SelfRequirement`](crate::SelfRequirement)). Outside
    /// any item (e.g. if only a type is visited), it is the `n`-th call (counting the successful
    /// calls from 0).
    ///
    /// The default of a type parameter of `generics` is not analyzed along with the item, as it
    /// is not copied there: a default is written in the Leaker's type context, so it is rewritten
    /// on its own by [`Referrer::complete_args()`]. It is analyzed with the parameters before it
    /// as the generics of the Leaker (and the where predicates which do not mention the others),
    /// so the types in it use only the earlier parameters, and cannot depend on `Self`: it is an
    /// error if a type of the default which is leaked mentions `Self`. The defaults of const
    /// parameters are analyzed along with the item.
    pub fn intern_with<'ast, F>(
        &mut self,
        generics: &Generics,
        f: F,
    ) -> std::result::Result<&mut Self, NotInternableError>
    where
        F: FnOnce(&mut dyn syn::visit::Visit<'ast>),
    {
        let mut visitor = AnalyzeVisitor::new(self, self.calls, generics);
        f(&mut visitor);
        let mut effects = visitor.finish();
        for (index, param) in generics.params.iter().enumerate() {
            if let GenericParam::Type(TypeParam {
                default: Some(default),
                ..
            }) = param
            {
                let root = without_defaults(generics, index);
                effects.extend(AnalyzeVisitor::analyze_default(self, default, &root));
            }
        }
        let contexts = self.contexts.len();
        let mut analysis = Analysis::default();
        analysis.apply(&mut self.contexts, effects, None);
        if !analysis.errors.is_empty() {
            self.contexts.truncate(contexts);
            return Err(if self.allow_external_crates {
                NotInternableError::RelativePath(analysis.errors)
            } else {
                NotInternableError::NotPortable(analysis.errors)
            });
        }
        for ident in &analysis.primitives {
            self.assume_primitive(ident);
        }
        self.calls += 1;
        if !analysis.types.is_empty() {
            self.types.merge(analysis.types);
            self.reachable_types = None;
        }
        Ok(self)
    }

    /// Checks whether the given `ty` must be interned, and whether it can be. See [`CheckResult`].
    pub fn check(&self, ty: &Type, generics: &Generics) -> CheckResult {
        AnalyzeVisitor::check(self, ty, generics)
    }

    /// Finish building the [`Leaker`] and convert it into [`Referrer`].
    ///
    /// The types are reduced by [`Leaker::reduce_roots()`] (called here if needed), and sorted by
    /// their context (the contexts are numbered in the order of the first appearance of their
    /// requirements, and renumbered without those of the types removed) and tokens, so the result
    /// (including the index of each type) is deterministic.
    pub fn finish(mut self) -> Referrer {
        self.reduce_roots();
        let keys = self.reachable_types.take().unwrap_or_default();
        let types = self.types;
        // Renumber the contexts of the types kept (the keys are sorted by context).
        let mut renumbered: Vec<usize> = Vec::new();
        let mut requirements: Vec<Requirements> = Vec::new();
        let mut entries = Vec::new();
        for key in keys {
            if renumbered.last() != Some(&key.context) {
                renumbered.push(key.context);
                requirements.push(self.contexts.get(key.context).clone());
            }
            let must = types.hard.contains(&key);
            let crates = external_crates(&key.ty.ty, &requirements[requirements.len() - 1]);
            entries.push((requirements.len() - 1, key.ty.ty, must, crates));
        }
        Referrer::new(entries, requirements)
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

    /// Returns the types to be interned, sorted by their context and tokens.
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
        kept.sort_by_cached_key(|key| (key.context, key.ty.to_string()));
        kept
    }
}

/// Returns the first `n` parameters of `generics`, without their defaults, and the where
/// predicates which do not mention the other parameters (matched by name).
fn without_defaults(generics: &Generics, n: usize) -> Generics {
    let params: Punctuated<GenericParam, Token![,]> = generics
        .params
        .iter()
        .take(n)
        .cloned()
        .map(|mut param| {
            match &mut param {
                GenericParam::Type(tp) => {
                    tp.eq_token = None;
                    tp.default = None;
                }
                GenericParam::Const(cp) => {
                    cp.eq_token = None;
                    cp.default = None;
                }
                GenericParam::Lifetime(_) => (),
            }
            param
        })
        .collect();
    let later: HashSet<String> = generics
        .params
        .iter()
        .skip(n)
        .map(generic_param_name)
        .collect();
    let predicates: Punctuated<WherePredicate, Token![,]> = generics
        .where_clause
        .iter()
        .flat_map(|wc| wc.predicates.iter())
        .filter(|predicate| {
            let mut names = HashSet::new();
            token_names(quote!(#predicate), &mut names);
            names.is_disjoint(&later)
        })
        .cloned()
        .collect();
    let has_params = !params.is_empty();
    Generics {
        lt_token: if has_params {
            Some(Default::default())
        } else {
            None
        },
        params,
        gt_token: if has_params {
            Some(Default::default())
        } else {
            None
        },
        where_clause: if predicates.is_empty() {
            None
        } else {
            Some(WhereClause {
                where_token: Default::default(),
                predicates,
            })
        },
    }
}

/// Returns the sorted names of the external crates (other than `core`) named by the absolute
/// paths (`::name::..`) in `ty` and `requirements` (the generics and what `Self` means), without
/// `r#` (e.g. `async` for `::r#async::X`).
fn external_crates(ty: &Type, requirements: &Requirements) -> Vec<String> {
    use crate::requirements::SelfRequirement;
    use proc_macro2::TokenTree;
    use syn::ext::IdentExt;
    use syn::visit::Visit;
    #[derive(Default)]
    struct Finder(BTreeSet<String>);
    impl Finder {
        /// Scans the tokens of a macro for `::` followed by a name at the start of a path.
        fn scan(&mut self, tokens: TokenStream) {
            let mut prev: Option<TokenTree> = None;
            let mut iter = tokens.into_iter().peekable();
            while let Some(tt) = iter.next() {
                match &tt {
                    TokenTree::Group(group) => self.scan(group.stream()),
                    TokenTree::Punct(p)
                        if p.as_char() == ':' && p.spacing() == proc_macro2::Spacing::Joint =>
                    {
                        let starts_path = !matches!(&prev, Some(TokenTree::Ident(_)))
                            && !matches!(&prev, Some(TokenTree::Punct(p)) if p.as_char() == '>');
                        if starts_path {
                            if let Some(TokenTree::Punct(second)) = iter.peek() {
                                if second.as_char() == ':' {
                                    iter.next();
                                    if let Some(TokenTree::Ident(ident)) = iter.peek() {
                                        if ident.unraw() != "core" {
                                            self.0.insert(ident.unraw().to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => (),
                }
                prev = Some(tt);
            }
        }
    }
    impl<'ast> Visit<'ast> for Finder {
        fn visit_path(&mut self, i: &'ast Path) {
            if i.leading_colon.is_some() {
                if let Some(seg) = i.segments.first() {
                    if seg.ident.unraw() != "core" {
                        self.0.insert(seg.ident.unraw().to_string());
                    }
                }
            }
            syn::visit::visit_path(self, i)
        }
        fn visit_macro(&mut self, i: &'ast Macro) {
            self.scan(i.tokens.clone());
            syn::visit::visit_macro(self, i)
        }
    }
    let mut finder = Finder::default();
    finder.visit_type(ty);
    finder.visit_generics(requirements.generics());
    match requirements.self_ty() {
        Some(SelfRequirement::Bound(path)) => finder.visit_path(path),
        Some(SelfRequirement::Type(ty)) => finder.visit_type(ty),
        _ => (),
    }
    finder.0.into_iter().collect()
}

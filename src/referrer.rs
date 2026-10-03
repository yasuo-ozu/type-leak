//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::encode::{complete_generics_args, EncodeError};
use crate::graph::TypeKey;
use crate::requirements::{Requirements, Scope, SelfRequirement};
use crate::util::{
    first_self_in_generics, generic_param_name, mentions_names, token_names, type_mentions_self,
};
use proc_macro2::TokenStream;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};

/// Holds the list of types that need to be encoded, as produced by
/// [`Leaker::finish()`](crate::Leaker::finish()).
///
/// `Referrer` provides methods to iterate over leak types and to expand or transform
/// types using a custom mapping function.
///
/// Each type is an entry, identified by its tokens and its requirements: the generics it uses
/// (see [`Referrer::generics()`]) and, if it mentions `Self`, what `Self` means (see
/// [`SelfRequirement`]). The requirements are shared by the entries of a context (see
/// [`Referrer::context()`]). The same tokens with different requirements (e.g. `Wrap<B>` where
/// `B` is a `bool` in one method and a `u8` in another) are separate entries, and the one a copy
/// refers to is found by resolving the generics in scope where it appears.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    pub(crate) leak_types: Vec<Type>,
    /// The context of each entry (see [`Referrer::context()`]).
    pub(crate) contexts: Vec<usize>,
    /// Whether each entry is required (see [`Referrer::must()`]).
    pub(crate) musts: Vec<bool>,
    /// The sorted names of the external crates each entry requires (see
    /// [`Referrer::external_crates_require()`]).
    pub(crate) external_crates: Vec<Vec<String>>,
    /// The external crates available to the Referrer's crate, if known (see
    /// [`Referrer::with_available_external_crates()`]). Not serialized.
    pub(crate) available_external_crates: Option<Vec<String>>,
    /// The requirements of each context, which are distinct (see [`Referrer::context_generics()`]).
    pub(crate) requirements: Vec<Requirements>,
    /// The candidates of a type, keyed by [`TypeKey`] (by tokens) as in the Leaker, so that e.g. a
    /// type wrapped in an invisible group is found as well.
    pub(crate) map: HashMap<TypeKey, Vec<usize>>,
}

/// How [`Referrer::find()`] matches the requirements of the entries with the generics in scope.
#[derive(Clone, Copy)]
enum Matching<'a> {
    /// The generics declared in the scopes are checked, and those which are not are assumed to be
    /// generics of the Leaker. If `self_ty` is given, an entry mentioning `Self` must be for it.
    Lenient {
        self_ty: Option<&'a SelfRequirement>,
    },
    /// The first scope is that of the Leaker, so the requirements are compared exactly. If
    /// `self_ty` is given, `Self` means it, and otherwise it is not checked.
    Exact {
        self_ty: Option<&'a SelfRequirement>,
    },
    /// The scope is that of the parameters before a defaulted one: the entry must use only them.
    Default,
}

impl Referrer {
    /// Builds a Referrer from its entries (the context, the type, whether it is required and the
    /// external crates it requires), in the order of their indices, and the requirements of the
    /// contexts, which are distinct. There is at least one context, and one for each context of
    /// the entries.
    pub(crate) fn new(
        entries: impl IntoIterator<Item = (usize, Type, bool, Vec<String>)>,
        mut requirements: Vec<Requirements>,
    ) -> Self {
        if requirements.is_empty() {
            requirements.push(Requirements::none());
        }
        let mut referrer = Self {
            leak_types: Vec::new(),
            contexts: Vec::new(),
            musts: Vec::new(),
            external_crates: Vec::new(),
            available_external_crates: None,
            requirements,
            map: HashMap::new(),
        };
        for (ix, (context, ty, must, crates)) in entries.into_iter().enumerate() {
            referrer.map.entry(TypeKey::new(&ty)).or_default().push(ix);
            referrer.leak_types.push(ty);
            referrer.contexts.push(context);
            referrer.musts.push(must);
            referrer.external_crates.push(crates);
        }
        referrer
    }

    /// Returns the index of the entry of the type `ty` whose requirements match the generics in
    /// `scopes` (see [`Matching`]). If several entries match, the one which assumes the fewest
    /// generics of the Leaker (the generics not declared in `scopes`) is returned, and then the
    /// first.
    ///
    /// `module` is the path of the module where the type is, if known. An entry which does not
    /// depend on a module (see [`Referrer::context_module()`]) matches anywhere, and an entry of a
    /// module matches only in it. If the module is not known, an entry of any module matches.
    /// Among the entries, those of the module are preferred (if it is unknown, those of no module,
    /// then the analyzed module `[]`).
    fn find(
        &self,
        matching: Matching,
        module: Option<&[Ident]>,
        ty: &Type,
        scopes: &[Scope],
    ) -> Option<usize> {
        let candidates = self.map.get(&TypeKey::new(ty))?;
        let mut best: Option<(usize, usize, usize)> = None;
        for &ix in candidates {
            let entry = &self.requirements[self.contexts[ix]];
            let rank = match (module, entry.module()) {
                (Some(module), Some(entry)) if entry == module => 0,
                (Some(_), Some(_)) => continue,
                (Some(_), None) => 1,
                (None, None) => 0,
                (None, Some(entry)) => 1 + usize::from(!entry.is_empty()),
            };
            if let Some(assumed) = Self::matches(matching, ty, scopes, entry) {
                if best.map_or(true, |(r, fewest, _)| (rank, assumed) < (r, fewest)) {
                    best = Some((rank, assumed, ix));
                }
            }
        }
        let ix = best?.2;
        // An optional entry whose crates are all available is left as written.
        if !self.musts[ix] {
            if let Some(available) = &self.available_external_crates {
                if self.external_crates[ix]
                    .iter()
                    .all(|name| available.contains(name))
                {
                    return None;
                }
            }
        }
        Some(ix)
    }

    /// Returns the number of the generics of the Leaker which `entry` assumes (see
    /// [`Referrer::find()`]) if it matches the type `ty` in `scopes`.
    fn matches(
        matching: Matching,
        ty: &Type,
        scopes: &[Scope],
        entry: &Requirements,
    ) -> Option<usize> {
        match matching {
            Matching::Exact { self_ty } => {
                let requirements =
                    Requirements::of(ty, scopes, self_ty.or_else(|| entry.self_ty()));
                if requirements == entry.without_module() {
                    Some(0)
                } else {
                    None
                }
            }
            Matching::Lenient { self_ty } => {
                if let (Some(wanted), Some(actual)) = (self_ty, entry.self_ty()) {
                    if wanted != actual {
                        return None;
                    }
                }
                let requirements = Requirements::of(ty, scopes, entry.self_ty());
                let declared: HashSet<String> = scopes
                    .iter()
                    .flat_map(|scope| scope.params())
                    .map(generic_param_name)
                    .collect();
                Self::matches_declared(&declared, &requirements, entry)
            }
            Matching::Default => {
                if entry.self_ty().is_some() {
                    return None;
                }
                let root: HashSet<String> = scopes[0].params().map(generic_param_name).collect();
                let requirements = Requirements::of(ty, &scopes[..1], None);
                let used = names_of_params(requirements.generics());
                let entry_params = names_of_params(entry.generics());
                if used.is_subset(&entry_params) && entry_params.is_subset(&root) {
                    Some(entry_params.len())
                } else {
                    None
                }
            }
        }
    }

    /// Compares the requirements of the type (`requirements`, as it depends on the generics
    /// `declared`) with `entry`: a generic declared must be required by both, with the same
    /// declaration (the bounds in the canonical form), and the others are assumed to be generics
    /// of the Leaker. Returns the number of them.
    fn matches_declared(
        declared: &HashSet<String>,
        requirements: &Requirements,
        entry: &Requirements,
    ) -> Option<usize> {
        let render = |param: &GenericParam| quote!(#param).to_string();
        let wanted: HashMap<String, String> = requirements
            .generics()
            .params
            .iter()
            .map(|param| (generic_param_name(param), render(param)))
            .collect();
        let mut actual = HashMap::new();
        let mut assumed = 0;
        for param in &entry.generics().params {
            let name = generic_param_name(param);
            if declared.contains(&name) {
                // Declared here, so it must be used by the type here in the same way.
                if wanted.get(&name) != Some(&render(param)) {
                    return None;
                }
            } else {
                assumed += 1;
            }
            actual.insert(name, ());
        }
        if wanted.keys().any(|name| !actual.contains_key(name)) {
            return None;
        }
        // The predicates which are not moved to a parameter.
        let predicates = |generics: &Generics| -> HashSet<String> {
            generics
                .where_clause
                .iter()
                .flat_map(|wc| wc.predicates.iter())
                .map(|predicate| quote!(#predicate).to_string())
                .collect()
        };
        let wanted_predicates = predicates(requirements.generics());
        let actual_predicates = predicates(entry.generics());
        if !wanted_predicates.is_subset(&actual_predicates) {
            return None;
        }
        // The predicates of the entry which mention only generics declared here are required here.
        let entry_names = names_of_params(entry.generics());
        for predicate in entry
            .generics()
            .where_clause
            .iter()
            .flat_map(|wc| wc.predicates.iter())
        {
            let mut names = HashSet::new();
            token_names(quote!(#predicate), &mut names);
            let names: Vec<&String> = names
                .iter()
                .filter(|name| entry_names.contains(*name))
                .collect();
            if !names.is_empty()
                && names.iter().all(|name| declared.contains(*name))
                && !wanted_predicates.contains(&quote!(#predicate).to_string())
            {
                return None;
            }
        }
        Some(assumed)
    }

    /// Sets the names of the external crates which the Referrer's crate can refer to with an
    /// absolute path (`::name::..`), e.g. its dependencies (default: unknown, `None`). This is
    /// where the crates are allowed: the Leaker has no such list, only whether it leaks the
    /// external crates at all
    /// ([`Leaker::allow_external_crates`](crate::Leaker::allow_external_crates)). This is not
    /// serialized.
    ///
    /// An entry which is not [required](Referrer::must()) (it is leaked only because a path names
    /// an external crate, see [`Leaker::allow_external_crates`](crate::Leaker::allow_external_crates))
    /// is not replaced by [`Referrer::into_visitor()`], [`Referrer::expand()`] etc. if all the
    /// crates it [requires](Referrer::external_crates_require()) are in `crates`: it is left as
    /// written, and the types inside it are looked up as for a type which is not leaked. If the
    /// available crates are unknown, or one of the crates is missing, it is replaced as usual. An
    /// entry which is required is always replaced.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.allow_external_crates = false;
    /// let item: ItemStruct = parse_quote!(struct S { a: ::foo::X });
    /// leaker.intern_with(&item.generics, |v| v.visit_item_struct(&item)).unwrap();
    /// let referrer = leaker.finish();
    /// assert!(!referrer.must(0));
    /// assert_eq!(referrer.external_crates_require(0), ["foo"]);
    /// let ty: Type = parse_quote!(::foo::X);
    /// let replaced: Type = parse_quote!(Replaced);
    /// // Replaced when `foo` is not known to be available.
    /// assert_eq!(referrer.expand(ty.clone(), |_, _| replaced.clone()), replaced);
    /// let referrer = referrer.with_available_external_crates(["foo"]);
    /// assert_eq!(referrer.expand(ty.clone(), |_, _| replaced.clone()), ty);
    /// ```
    pub fn with_available_external_crates(
        mut self,
        crates: impl IntoIterator<Item = impl ToString>,
    ) -> Self {
        self.available_external_crates = Some(crates.into_iter().map(|c| c.to_string()).collect());
        self
    }

    /// Returns the sorted, deduplicated names of the external crates (other than `core`) named by
    /// absolute paths (`::name::..`, by their first segment) in the `ix`-th type and its
    /// [generics](Referrer::generics()) and `Self` (see [`Referrer::context_self()`]). The type
    /// can be left as written if the Referrer's crate has all of them (see
    /// [`Referrer::with_available_external_crates()`]).
    pub fn external_crates_require(&self, ix: usize) -> &[String] {
        &self.external_crates[ix]
    }

    /// Returns `false` if the `ix`-th type is leaked only because of paths to external crates,
    /// which is possible only with
    /// [`Leaker::allow_external_crates`](crate::Leaker::allow_external_crates) unset. Such an
    /// entry is not replaced when the crates it requires are available (see
    /// [`Referrer::with_available_external_crates()`]). It is `true` if any cause of the entry is
    /// something else (a relative path, a relative lifetime, a type macro, ...), and then the entry
    /// is always replaced.
    pub fn must(&self, ix: usize) -> bool {
        self.musts[ix]
    }

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

    /// Returns the generics which the `ix`-th type uses, and which the Repeater's implementation
    /// of it must declare (and encode exactly, in its trait arguments), e.g. the generics of the
    /// Leaker and those of a trait method or a GAT (or of a nested item, or a `for<..>` binder).
    /// It is empty for most types. This is the same as
    /// [`Referrer::context_generics()`] of the context of the type.
    ///
    /// They are in a canonical form, which is the same for declarations with the same meaning: the
    /// parameters are sorted by name (lifetimes first), and the bounds are sorted and moved from
    /// the where clause to the parameter they bound, if any. Defaults are removed. A type uses a
    /// generic if it mentions it (e.g. `T` in `Wrap<T>`), or mentions another generic whose
    /// bounds or where predicates do (e.g. `T` in `U: Into<T>` if the type uses `U`), or, if it
    /// mentions `Self`, the generics in the description of `Self` (e.g. `T` and `U` in
    /// `Self: MyTrait<T, U>`, see [`Referrer::context_self()`]).
    ///
    /// A type such as `Wrap<B>` in `fn f<const B: bool>(&self) -> Wrap<B>` (where `Wrap` depends on
    /// the type context) cannot be decomposed, so it is interned together with `B`. The Repeater's
    /// implementation must then declare these generics and constrain them in its trait arguments,
    /// e.g. with [`encode_generics_params_to_ty()`](crate::encode_generics_params_to_ty()). The
    /// Referrer declares the same generics with the same names where it copies the signature, or
    /// is given arguments for those of the Leaker (see [`Referrer::generic_args()`]), so it can
    /// encode them in the same way. A type which does not use a generic of the Leaker is not
    /// affected by it: it is shared by all the calls of
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()), whatever their generics are.
    ///
    /// If the type mentions `Self` (see [`Referrer::context_self()`]), the Repeater's
    /// implementation is for the implementor or the type `Self` is. The bounds and where
    /// predicates of the generics are kept as they are, so they may mention `Self` too (e.g.
    /// `fn f<T: PartialEq<Self>>(&self)`), with the same meaning.
    ///
    /// ```ignore
    /// // In the Leaker's crate
    /// for (ix, (ty, generics)) in referrer.iter_with_generics().enumerate() {
    ///     let encoded = slots.encode_generics_params_to_ty(&generics.params)?;
    ///     let (params, predicates) = (&generics.params, generics.where_clause.iter());
    ///     quote! {
    ///         impl<#params, __S: ?Sized> Repeater<#encoded, #ID, #ix> for __S
    ///         #predicates
    ///         {
    ///             type Type = #ty;
    ///         }
    ///     }
    /// }
    ///
    /// // In the Referrer's crate (in the implementation of `MyTrait<#trait_args>`)
    /// let completed = referrer.complete_args(&trait_args, &trait_params, |..| ..)?;
    /// let args = referrer.generic_args(ix, &trait_params, &completed)?;
    /// let encoded = slots.encode_generics_args_to_ty(&args, &referrer.generics(ix).params)?;
    /// parse_quote!(<Self as Repeater<#encoded, #ID, #ix>>::Type)
    /// ```
    pub fn generics(&self, ix: usize) -> &Generics {
        self.context_generics(self.contexts[ix])
    }

    /// Returns the generics which the `ix`-th type uses.
    #[deprecated(
        since = "0.8.0",
        note = "the generics of a type include those of the Leaker it uses; use `generics()`"
    )]
    pub fn local_generics(&self, ix: usize) -> &Generics {
        self.generics(ix)
    }

    /// Returns the context of the `ix`-th type: the number of its requirements, i.e. of the
    /// generics it uses and what `Self` means in it (see [`Referrer::generics()`] and
    /// [`Referrer::context_self()`]). The entries with equal requirements are in the same
    /// context, whichever call of [`Leaker::intern_with()`](crate::Leaker::intern_with()) found
    /// them, and the contexts are numbered from 0 in the order of their first entry.
    pub fn context(&self, ix: usize) -> usize {
        self.contexts[ix]
    }

    /// Returns the number of contexts (at least 1). There is no context without entries unless
    /// there are no entries at all, when it is the context of the types which use no generics.
    pub fn n_contexts(&self) -> usize {
        self.requirements.len()
    }

    /// Returns the generics which the types of `context` use, in the form to be declared by the
    /// Repeater's implementation of an entry in `context`, which must encode exactly their
    /// parameters in its trait arguments. See [`Referrer::generics()`].
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T, U = Rel<T>> {
    ///     fn f<const B: bool>(&self, t: T) -> Wrap<B>;
    ///     fn g(&self) -> Rel2;
    /// });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    /// // `Wrap<B>` uses `B`, `Rel2` uses nothing and `Rel<T>` (the default of `U`) uses `T`, in
    /// // the order of their first appearance.
    /// assert_eq!(referrer.n_contexts(), 3);
    /// let used: Vec<usize> = (0..3).map(|c| referrer.context_generics(c).params.len()).collect();
    /// assert_eq!(used, [1, 0, 1]);
    /// ```
    pub fn context_generics(&self, context: usize) -> &Generics {
        self.requirements[context].generics()
    }

    /// Returns what `Self` means in the types of `context`, if they mention it.
    ///
    /// The Repeater's implementation of an entry in `context` is for the type described: `None`
    /// for any type (`impl<.., __S: ?Sized> Repeater<..> for __S`), [`SelfRequirement::Bound`] for
    /// the types implementing the trait (`impl<.., __S: ?Sized + Trait<..>> Repeater<..> for
    /// __S`), and [`SelfRequirement::Type`] for the type (`impl<..> Repeater<..> for Type`).
    /// [`SelfRequirement::Call`] is not tied to a type, so the implementation may be for any type
    /// like `None`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T> { fn f(&self) -> Rel<Self>; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    /// match referrer.context_self(0) {
    ///     Some(SelfRequirement::Bound(path)) => {
    ///         assert_eq!(quote!(#path).to_string(), "Tr < T >")
    ///     }
    ///     _ => unreachable!(),
    /// }
    /// ```
    pub fn context_self(&self, context: usize) -> Option<&SelfRequirement> {
        self.requirements[context].self_ty()
    }

    /// Returns the path of the module where the types of `context` are written, relative to the
    /// module analyzed by [`Leaker::from_module()`](crate::Leaker::from_module()) (empty for the
    /// module itself).
    ///
    /// It is `Some` only for a type which depends on its module, i.e. has a relative path (e.g.
    /// `Rel`, which `use super::super::Rel;` may import in a nested module, where it is another
    /// type than in the analyzed module), so the same tokens in different modules are different
    /// entries. The Repeater's implementations of the entries of the context must be written
    /// inside that module (relative to the analyzed module), where the relative paths resolve
    /// like in the original: e.g. an attribute macro on `mod m` can emit them inside `m::inner`.
    /// It is `None` for the other types (e.g. with only absolute paths), which are shared by all
    /// the modules, and if no module is analyzed.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let m: ItemMod = parse_quote!(mod m {
    ///     pub struct A { y: Rel }
    ///     pub mod inner { pub struct C { y: Rel } }
    /// });
    /// let referrer = Leaker::from_module(&m).unwrap().finish();
    /// assert_eq!(referrer.n_contexts(), 2);
    /// assert_eq!(referrer.context_module(0), Some(&[][..]));
    /// assert_eq!(
    ///     referrer.context_module(1),
    ///     Some(&[Ident::new("inner", proc_macro2::Span::call_site())][..])
    /// );
    /// ```
    pub fn context_module(&self, context: usize) -> Option<&[Ident]> {
        self.requirements[context].module()
    }

    /// Returns `true` if a leaked type, or the generics (the bounds and the where predicates) of
    /// an entry, refers to `Self` (e.g. `Self`, `Self::Item`, `Rel<Self>`, `&Self`, or `<Self as
    /// ::p::Tr>::X`, but not `::p::Self`).
    ///
    /// Then the Repeater must be implemented for the implementor of the Leaker trait, or the type
    /// `Self` is, where `Self` has the same meaning (see [`Referrer::context_self()`]), and not
    /// for a separate marker type (see [`Referrer::iter_with_self()`] otherwise). No leaked type
    /// mentions `Self` with
    /// [`Leaker::self_ty_can_be_interned`](crate::Leaker::self_ty_can_be_interned) unset (it is
    /// an error, also if the bounds of the generics of the type do).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> Rel<Self>; });
    /// assert!(Leaker::from_trait(&tr).unwrap().finish().has_self());
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> Rel; });
    /// assert!(!Leaker::from_trait(&tr).unwrap().finish().has_self());
    /// ```
    pub fn has_self(&self) -> bool {
        self.leak_types.iter().any(type_mentions_self)
            || self
                .requirements
                .iter()
                .any(|requirements| first_self_in_generics(requirements.generics()).is_some())
    }

    /// Like [`Referrer::iter_with_generics()`], with `Self` replaced by `self_ty` in the types and
    /// the generics (see [`Referrer::has_self()`]), for a Repeater implemented for another type
    /// than the implementor of the Leaker trait (e.g. a marker type generic over `__S`), where
    /// `Self` would refer to that type instead.
    ///
    /// `Self` alone becomes `self_ty`, `Self::A::B` becomes `<self_ty>::A::B`, and the self type
    /// of a qualified path (e.g. `<Self as ::p::Tr>::X`) is replaced as well. In the tokens of a
    /// macro, the identifier `Self` is replaced by the tokens of `self_ty`. Only the copies
    /// returned are rewritten: [`Referrer::into_visitor()`] and [`Referrer::expand()`] still find
    /// the types in the code of the Leaker's trait, which is written with `Self`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> Rel<Self::Item>; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    /// assert!(referrer.has_self());
    /// let marker: Type = parse_quote!(__S);
    /// for (ix, (ty, _generics)) in referrer.iter_with_self(&marker).enumerate() {
    ///     // In the Leaker's crate: `impl<__S> Repeater<.., #ix> for Marker<__S> { type Type = #ty; }`
    ///     # let _ = ix;
    ///     let expected: Type = parse_quote!(Rel<<__S>::Item>);
    ///     assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
    /// }
    /// ```
    pub fn iter_with_self<'a>(
        &'a self,
        self_ty: &'a Type,
    ) -> impl Iterator<Item = (Type, Generics)> + 'a {
        use syn::fold::Fold;
        self.iter_with_generics().map(move |(ty, generics)| {
            let mut folder = ReplaceSelf { self_ty };
            (
                folder.fold_type(ty.clone()),
                folder.fold_generics(generics.clone()),
            )
        })
    }

    /// Completes the generic arguments `args` of the trait of the Leaker (whose parameters are
    /// `params`): returns one argument for each parameter, in the order of `params`, so the
    /// encoders (such as
    /// [`RepeaterSlots::encode_generics_args_to_ty()`](crate::RepeaterSlots::encode_generics_args_to_ty()))
    /// and the substitution of the parameters need not fill the defaults.
    ///
    /// The arguments are matched with `params` like
    /// [`encode_generics_args_to_ty_with()`](crate::encode_generics_args_to_ty_with()) does:
    /// lifetime arguments come first, a bare identifier corresponding to a const parameter (e.g.
    /// `N` in `MyTrait<N>`) is treated as a const argument, associated item bindings and
    /// constraints are skipped, and [`EncodeError`] is returned if there are too many arguments,
    /// or an argument without a default is missing. An omitted const argument takes the default
    /// as it is written, with the preceding parameters substituted.
    ///
    /// An omitted type argument takes the default, which is written in the Leaker's type context
    /// (e.g. `U = RelType<T>`), where its relative types resolve, so they are replaced by
    /// `type_fn`, like [`Referrer::expand()`] does. They are leaked with the parameters before the
    /// defaulted one as the generics of the Leaker (see
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with())), so they use no later parameter:
    /// the entries are found by the names of `params[..k]`, where `k` is the position of the
    /// defaulted parameter. `type_fn` receives the leaked type, its index, and the complete
    /// arguments of the parameters before the defaulted one (`params[..k]`), and returns the
    /// projection to the leaked type, e.g. built with [`Referrer::generic_args()`] for `params[..k]`
    /// and these arguments. The result of `type_fn` is not substituted again, so a projection may
    /// contain the names of the parameters.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::punctuated::Punctuated;
    /// # use template_quote::quote;
    /// // In the Leaker's crate
    /// let tr: ItemTrait = parse_quote!(trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> U; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    ///
    /// // In the Referrer's crate (for `impl Tr<u8> for ..`, omitting `U`)
    /// let args: AngleBracketedGenericArguments = parse_quote!(<u8>);
    /// let completed = referrer
    ///     .complete_args(&args.args, &tr.generics.params, |_, ix, prefix| {
    ///         let before: Punctuated<GenericParam, Token![,]> =
    ///             tr.generics.params.iter().take(prefix.len()).cloned().collect();
    ///         let args = referrer.generic_args(ix, &before, prefix).unwrap();
    ///         let params = &referrer.generics(ix).params;
    ///         let encoded =
    ///             encode_generics_args_to_ty_with(&args, Some(params), |_, _| unreachable!())
    ///                 .unwrap();
    ///         parse_quote!(<Self as ::leaker::Repeater<#encoded, #ix>>::Type)
    ///     })
    ///     .unwrap();
    /// let expected: Vec<GenericArgument> =
    ///     vec![parse_quote!(u8), parse_quote!(<Self as ::leaker::Repeater<(u8,), 0usize>>::Type)];
    /// assert_eq!(quote!(#(#completed),*).to_string(), quote!(#(#expected),*).to_string());
    /// ```
    pub fn complete_args<'a>(
        &self,
        args: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
        mut type_fn: impl FnMut(Type, usize, &[GenericArgument]) -> Type,
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        use syn::fold::Fold;
        complete_generics_args(
            args,
            params,
            |index, _param, default, substitute, completed| {
                // The parameters before the defaulted one are the generics of the Leaker for its
                // default.
                let root = Generics {
                    params: params.iter().take(index).cloned().collect(),
                    ..Default::default()
                };
                // Replace the leaked types with placeholders, substitute the preceding parameters,
                // and then replace the placeholders, so that the projections are not substituted.
                let mut leaked = Vec::new();
                let ty = self.expand_with(
                    Matching::Default,
                    None,
                    &mut vec![Scope::new(&root)],
                    None,
                    default.clone(),
                    |ty, ix| {
                        leaked.push((ty, ix));
                        let placeholder = Ident::new(
                            &format!("{}{}", PLACEHOLDER, leaked.len() - 1),
                            proc_macro2::Span::call_site(),
                        );
                        parse_quote!(#placeholder)
                    },
                );
                let ty = substitute.fold_type(ty);
                Ok(Resolve {
                    leaked,
                    completed,
                    type_fn: &mut type_fn,
                }
                .fold_type(ty))
            },
        )
    }

    /// Returns the arguments of the generics of the `ix`-th type (see [`Referrer::generics()`])
    /// at the use site, one for each parameter, in the order of the parameters, so that they can
    /// be encoded with [`RepeaterSlots::encode_generics_args_to_ty()`] and the parameters of
    /// `generics(ix)`, to refer to the type through the Repeater.
    ///
    /// `root_params` are the parameters of the Leaker (e.g. of the trait), and `completed_args` the
    /// complete arguments for them (see [`Referrer::complete_args()`]). A generic of the type
    /// named like a parameter of `root_params` takes the argument at its position. Any other
    /// generic is passed by its own name (a type or const identifier, or a lifetime), which is in
    /// scope at the use site if it is declared there (e.g. by a method). `Self` is not an
    /// argument: the Repeater is used for the implementor (see [`Referrer::context_self()`]).
    ///
    /// [`EncodeError::MissingArgument`] is returned if there are fewer `completed_args` than
    /// needed.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T> {
    ///     fn f<const B: bool>(&self, t: T) -> Wrap<T, B>;
    /// });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    /// // For `impl Tr<u8> for ..`
    /// let completed: Vec<GenericArgument> = vec![parse_quote!(u8)];
    /// let args = referrer.generic_args(0, &tr.generics.params, &completed).unwrap();
    /// // `T` is the argument of the trait, and `B` is declared by the method.
    /// let expected: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(B)];
    /// assert_eq!(quote!(#(#args),*).to_string(), quote!(#(#expected),*).to_string());
    /// ```
    ///
    /// [`RepeaterSlots::encode_generics_args_to_ty()`]: crate::RepeaterSlots::encode_generics_args_to_ty
    pub fn generic_args(
        &self,
        ix: usize,
        root_params: &Punctuated<GenericParam, Token![,]>,
        completed_args: &[GenericArgument],
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        self.generics(ix)
            .params
            .iter()
            .map(|param| {
                let name = generic_param_name(param);
                let root = root_params
                    .iter()
                    .position(|root| generic_param_name(root) == name);
                match root {
                    Some(pos) => completed_args
                        .get(pos)
                        .cloned()
                        .ok_or_else(|| EncodeError::MissingArgument(Box::new(param.clone()))),
                    None => Ok(match param {
                        GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                            GenericArgument::Lifetime(lifetime.clone())
                        }
                        GenericParam::Type(TypeParam { ident, .. }) => {
                            GenericArgument::Type(parse_quote!(#ident))
                        }
                        GenericParam::Const(ConstParam { ident, .. }) => {
                            GenericArgument::Const(parse_quote!(#ident))
                        }
                    }),
                }
            })
            .collect()
    }

    /// Returns an iterator over the types that need to be encoded, together with the generics
    /// they use. See [`Referrer::generics()`].
    pub fn iter_with_generics(&self) -> impl Iterator<Item = (&Type, &Generics)> {
        self.leak_types
            .iter()
            .zip(&self.contexts)
            .map(move |(ty, context)| (ty, self.requirements[*context].generics()))
    }

    /// Returns an iterator over the types that need to be encoded, together with the generics
    /// they use.
    #[deprecated(
        since = "0.8.0",
        note = "the generics of a type include those of the Leaker it uses; use `iter_with_generics()`"
    )]
    pub fn iter_with_local_generics(&self) -> impl Iterator<Item = (&Type, &Generics)> {
        self.iter_with_generics()
    }

    /// Converts this `Referrer` into a [`Visitor`] that can be used to transform AST nodes.
    ///
    /// The `type_fn` callback is called for each type that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type.
    /// See [`Referrer::expand()`] for details.
    ///
    /// A type is replaced if an entry has the same tokens and its requirements match the generics
    /// declared where the type is (see [`Visitor`]): the generics of the entry which are declared
    /// there (by an item, a method or a binder being visited) must be declared in the same way
    /// there and be used by the type, and the generics used by the type must be those of the
    /// entry. The other generics of the entry are assumed to be the generics of the Leaker, as
    /// the visited syntax does not declare them (see
    /// [`Visitor::with_outermost_generics_as_leakers()`]). If several entries match, the one which
    /// assumes the fewest generics of the Leaker is chosen, and then the first. So if the Leaker
    /// is analyzed by several calls of
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()) which declare the same generics
    /// differently (e.g. `T` with different bounds), use [`Referrer::into_visitor_with_root()`]
    /// to find the entry of the call.
    ///
    /// If `self_requirement` is given, an entry which mentions `Self` matches only if its
    /// description of `Self` is the same (see [`Referrer::context_self()`], e.g.
    /// `SelfRequirement::Bound` of the trait with its parameters as arguments for the signatures
    /// of the Leaker's trait). Entries which do not mention `Self` are not affected. If it is
    /// `None`, `Self` is not checked.
    pub fn into_visitor<F: FnMut(Type, usize) -> Type>(
        self,
        self_requirement: Option<SelfRequirement>,
        type_fn: F,
    ) -> Visitor<F> {
        Visitor {
            referrer: self,
            type_fn: Rc::new(RefCell::new(type_fn)),
            exact: false,
            self_requirement,
            scopes: Vec::new(),
            outermost_as_leakers: false,
            self_ty: None,
            item_depth: 0,
            module: None,
        }
    }

    /// Like [`Referrer::into_visitor()`], but the types are looked up with the exact requirements:
    /// `root_generics` are the generics of the Leaker (as passed to
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with())), the outermost scope in which the
    /// requirements of a type are computed, like the Leaker does, so only the entry with the
    /// same requirements is found (also if its generics are not declared in the visited syntax).
    ///
    /// If `self_requirement` is given, it is what `Self` means (it is the same everywhere, also
    /// in the items nested in the visited syntax), so the types mentioning it are found only if
    /// their entry has the same description of `Self`. If it is `None`, `Self` is not checked: an
    /// entry mentioning `Self` is compared with the description in the entry.
    ///
    /// The module is always compared (see [`Visitor::in_module()`]): a visitor which has not
    /// entered a module is in the analyzed module `[]`, so a type depending on its module is
    /// found only through the entry of that module.
    ///
    /// Use [`Visitor::with_outermost_generics_as_leakers()`] when visiting an item declaring
    /// `root_generics`, so as to not declare them twice.
    pub fn into_visitor_with_root<F: FnMut(Type, usize) -> Type>(
        self,
        root_generics: &Generics,
        self_requirement: Option<SelfRequirement>,
        type_fn: F,
    ) -> Visitor<F> {
        let mut visitor = self.into_visitor(self_requirement, type_fn);
        visitor.exact = true;
        visitor.scopes.push(Scope::new(root_generics));
        visitor
    }

    /// Expands a type by replacing any interned types using the provided mapping function.
    ///
    /// The `type_fn` callback is called for each subtype that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type. The type is
    /// expanded with no generics in scope, so the generics of the entries are assumed to be those
    /// of the Leaker (see [`Referrer::into_visitor()`]).
    ///
    /// ```ignore
    /// referrer.expand(ty, |_, ix| {
    ///     let encoded = ...;
    ///     parse_quote!(<#marker_ty as #crate::TypeRef<#encoded, #ix>>::Type)
    /// })
    /// ```
    pub fn expand(&self, ty: Type, type_fn: impl FnMut(Type, usize) -> Type) -> Type {
        self.expand_with(
            Matching::Lenient { self_ty: None },
            None,
            &mut Vec::new(),
            None,
            ty,
            type_fn,
        )
    }

    /// Expands `ty` where the generics of `scopes` are in scope, matching the entries as
    /// `matching` says. `Self` is replaced by `self_ty`, if any, in the leaked types passed to
    /// `type_fn` and in the rest of `ty` (not in the types returned by `type_fn`).
    fn expand_with(
        &self,
        matching: Matching,
        module: Option<&[Ident]>,
        scopes: &mut Vec<Scope>,
        self_ty: Option<&Type>,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        use syn::fold::Fold;
        struct Folder<'a, 'b, F> {
            referrer: &'a Referrer,
            matching: Matching<'b>,
            module: Option<&'b [Ident]>,
            scopes: &'a mut Vec<Scope>,
            replace_self: Option<ReplaceSelf<'a>>,
            type_fn: F,
        }
        impl<'a, 'b, F> Folder<'a, 'b, F> {
            /// Pushes the scope of the lifetimes of a `for<..>` binder, as the Leaker does.
            fn push_binder(&mut self, lifetimes: Option<&BoundLifetimes>) -> bool {
                lifetimes
                    .map(|lifetimes| self.scopes.push(Scope::binder(lifetimes)))
                    .is_some()
            }
        }
        impl<'a, 'b, F: FnMut(Type, usize) -> Type> Fold for Folder<'a, 'b, F> {
            fn fold_type(&mut self, ty: Type) -> Type {
                match (
                    self.referrer
                        .find(self.matching, self.module, &ty, self.scopes),
                    self.replace_self,
                ) {
                    (Some(ix), None) => (self.type_fn)(ty, ix),
                    (Some(ix), Some(mut replace)) => (self.type_fn)(replace.fold_type(ty), ix),
                    (None, None) => syn::fold::fold_type(self, ty),
                    (None, Some(replace)) => replace.fold_type_with(self, ty),
                }
            }
            fn fold_expr(&mut self, expr: Expr) -> Expr {
                match self.replace_self {
                    Some(replace) => replace.fold_expr_with(self, expr),
                    None => syn::fold::fold_expr(self, expr),
                }
            }
            fn fold_macro(&mut self, i: Macro) -> Macro {
                match self.replace_self {
                    Some(replace) => replace.fold_macro_with(i),
                    None => syn::fold::fold_macro(self, i),
                }
            }
            fn fold_type_bare_fn(&mut self, i: TypeBareFn) -> TypeBareFn {
                let pushed = self.push_binder(i.lifetimes.as_ref());
                let i = syn::fold::fold_type_bare_fn(self, i);
                if pushed {
                    self.scopes.pop();
                }
                i
            }
            fn fold_trait_bound(&mut self, i: TraitBound) -> TraitBound {
                let pushed = self.push_binder(i.lifetimes.as_ref());
                let i = syn::fold::fold_trait_bound(self, i);
                if pushed {
                    self.scopes.pop();
                }
                i
            }
        }
        Folder {
            referrer: self,
            matching,
            module,
            scopes,
            replace_self: self_ty.map(|self_ty| ReplaceSelf { self_ty }),
            type_fn,
        }
        .fold_type(ty)
    }
}

/// Returns the names of the parameters of `generics`.
fn names_of_params(generics: &Generics) -> HashSet<String> {
    generics.params.iter().map(generic_param_name).collect()
}

/// Replaces `Self` with a type, see [`Referrer::iter_with_self()`].
#[derive(Clone, Copy)]
struct ReplaceSelf<'a> {
    self_ty: &'a Type,
}

impl<'a> ReplaceSelf<'a> {
    /// Replaces the identifiers `Self` in `tokens` with the tokens of the type.
    fn replace_tokens(&self, tokens: TokenStream) -> TokenStream {
        use proc_macro2::{Group, TokenTree};
        tokens
            .into_iter()
            .map(|tt| match tt {
                TokenTree::Ident(ident) if ident == "Self" => {
                    let self_ty = self.self_ty;
                    quote!(#self_ty)
                }
                TokenTree::Group(group) => {
                    let mut new =
                        Group::new(group.delimiter(), self.replace_tokens(group.stream()));
                    new.set_span(group.span());
                    TokenTree::Group(new).into()
                }
                tt => TokenStream::from(tt),
            })
            .collect()
    }

    /// Returns the path without `Self` if `path` is `Self::A::B` (with no leading colon and no
    /// arguments on `Self`), or `None` if it is `Self` alone or does not start with `Self`.
    fn strip_self(path: &Path) -> Option<Option<Path>> {
        let first = path.segments.first()?;
        if path.leading_colon.is_some() || first.ident != "Self" || !first.arguments.is_none() {
            return None;
        }
        if path.segments.len() == 1 {
            return Some(None);
        }
        // The leading colon is the `::` after the qualified self type (see `qself()`).
        Some(Some(Path {
            leading_colon: Some(Default::default()),
            segments: path.segments.iter().skip(1).cloned().collect(),
        }))
    }

    fn qself(&self) -> QSelf {
        QSelf {
            lt_token: Default::default(),
            ty: Box::new(self.self_ty.clone()),
            position: 0,
            as_token: None,
            gt_token: Default::default(),
        }
    }

    /// Folds `ty` with `folder` (which folds the components), replacing `Self`.
    fn fold_type_with(&self, folder: &mut impl syn::fold::Fold, ty: Type) -> Type {
        match ty {
            Type::Path(TypePath { qself: None, path }) => match Self::strip_self(&path) {
                Some(None) => self.self_ty.clone(),
                Some(Some(rest)) => Type::Path(TypePath {
                    qself: Some(self.qself()),
                    path: folder.fold_path(rest),
                }),
                None => syn::fold::fold_type(folder, Type::Path(TypePath { qself: None, path })),
            },
            Type::Verbatim(tokens) => Type::Verbatim(self.replace_tokens(tokens)),
            ty => syn::fold::fold_type(folder, ty),
        }
    }

    /// Folds `expr` with `folder`, replacing `Self::A`. `Self` alone is a value, which cannot be
    /// replaced with a type.
    fn fold_expr_with(&self, folder: &mut impl syn::fold::Fold, expr: Expr) -> Expr {
        match expr {
            Expr::Path(ExprPath {
                attrs,
                qself: None,
                path,
            }) => match Self::strip_self(&path) {
                Some(Some(rest)) => Expr::Path(ExprPath {
                    attrs,
                    qself: Some(self.qself()),
                    path: folder.fold_path(rest),
                }),
                _ => syn::fold::fold_expr(
                    folder,
                    Expr::Path(ExprPath {
                        attrs,
                        qself: None,
                        path,
                    }),
                ),
            },
            expr => syn::fold::fold_expr(folder, expr),
        }
    }

    fn fold_macro_with(&self, mut i: Macro) -> Macro {
        i.tokens = self.replace_tokens(i.tokens);
        i
    }
}

impl<'a> syn::fold::Fold for ReplaceSelf<'a> {
    fn fold_type(&mut self, ty: Type) -> Type {
        let this = *self;
        this.fold_type_with(self, ty)
    }

    fn fold_expr(&mut self, expr: Expr) -> Expr {
        let this = *self;
        this.fold_expr_with(self, expr)
    }

    fn fold_macro(&mut self, i: Macro) -> Macro {
        self.fold_macro_with(i)
    }
}

/// The prefix of the names of the placeholders of the leaked types in [`Referrer::complete_args()`].
const PLACEHOLDER: &str = "__TypeLeakPlaceholder";

/// Replaces the placeholders with the result of `type_fn`.
struct Resolve<'a, F> {
    leaked: Vec<(Type, usize)>,
    completed: &'a [GenericArgument],
    type_fn: &'a mut F,
}

impl<'a, F: FnMut(Type, usize, &[GenericArgument]) -> Type> syn::fold::Fold for Resolve<'a, F> {
    fn fold_type(&mut self, ty: Type) -> Type {
        if let Type::Path(TypePath { qself: None, path }) = &ty {
            let number = path.get_ident().and_then(|ident| {
                ident
                    .to_string()
                    .strip_prefix(PLACEHOLDER)?
                    .parse::<usize>()
                    .ok()
            });
            if let Some((leaked, ix)) = number.and_then(|number| self.leaked.get(number).cloned()) {
                return (self.type_fn)(leaked, ix, self.completed);
            }
        }
        syn::fold::fold_type(self, ty)
    }
}

/// Rewrites AST nodes with the leaked types of a [`Referrer`], see [`Referrer::into_visitor()`].
///
/// It keeps track of the generics declared at the visited site (those of items, methods, impls and
/// `for<..>` binders), including those of the outermost item visited by default: a type is
/// rewritten only if an entry has the requirements it has with them (see
/// [`Referrer::into_visitor()`] for how they are matched, and
/// [`Referrer::generics()`]). A type mentioning generics which are declared differently from the
/// entry's (e.g. `T: Clone` in the `impl<T: Clone>` of the visited item, where it was leaked
/// with `T` alone) is not rewritten, since it refers to them rather than to the Leaker's items.
/// See [`Visitor::with_outermost_generics_as_leakers()`] to treat the generics of the outermost
/// item as those of the Leaker instead.
#[derive(Debug)]
pub struct Visitor<F> {
    referrer: Referrer,
    type_fn: Rc<RefCell<F>>,
    /// The first of the `scopes` is that of the Leaker, and the requirements are compared exactly
    /// (see [`Referrer::into_visitor_with_root()`]).
    exact: bool,
    /// What `Self` means, if known (see [`Referrer::into_visitor()`]).
    self_requirement: Option<SelfRequirement>,
    /// The generics in scope, the innermost last.
    scopes: Vec<Scope>,
    /// See [`Visitor::with_outermost_generics_as_leakers()`].
    outermost_as_leakers: bool,
    /// See [`Visitor::with_self()`].
    self_ty: Option<Type>,
    /// The number of items enclosing the syntax being visited.
    item_depth: usize,
    /// The path of the module being visited, relative to the first module visited, if known (see
    /// [`Visitor::in_module()`]).
    module: Option<Vec<Ident>>,
}

impl<F> Clone for Visitor<F> {
    fn clone(&self) -> Self {
        Self {
            referrer: self.referrer.clone(),
            type_fn: self.type_fn.clone(),
            exact: self.exact,
            self_requirement: self.self_requirement.clone(),
            scopes: self.scopes.clone(),
            outermost_as_leakers: self.outermost_as_leakers,
            self_ty: self.self_ty.clone(),
            item_depth: self.item_depth,
            module: self.module.clone(),
        }
    }
}

impl<F: FnMut(Type, usize) -> Type> Visitor<F> {
    /// Treats the generics of the outermost item visited as those of the Leaker, like
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()) does for the item it visits: they
    /// are not declared in the visited syntax, so a type depending on them (e.g. `Wrap<T>` in
    /// `impl<T: Clone> MyTrait<T> for W { fn f(&self) -> Wrap<T>; }`, visited as a whole, where
    /// `T` is a generic of the Leaker's trait) is rewritten if it was leaked, whichever bounds
    /// `T` has there.
    ///
    /// The outermost item is a struct, enum, union, trait, type alias, function, impl or trait
    /// alias (or the signature, if only a signature is visited). The generics of the items,
    /// methods, associated types and consts and `for<..>` binders in it are still declared, and a
    /// module (which has no generics) only makes the items in it nested.
    ///
    /// By default, the generics of the outermost item are declared too, so such a type is
    /// rewritten only if they are declared in the same way as the entry's (e.g. `T` without bounds
    /// in the example).
    pub fn with_outermost_generics_as_leakers(mut self) -> Self {
        self.outermost_as_leakers = true;
        self
    }

    /// Sets the path of the module where the visited syntax is, relative to the module analyzed by
    /// [`Leaker::from_module()`](crate::Leaker::from_module()) (see
    /// [`Referrer::context_module()`]), e.g. `vec![inner]` for a copy of an item of `m::inner`
    /// when `m` was analyzed. The modules visited below it push their names.
    ///
    /// A type depending on its module is rewritten only to the entry of the module it is in (an
    /// entry which does not depend on a module matches anywhere). Without this, a visitor which
    /// has not entered a module (see [`Referrer::into_visitor()`]) does not know the module, and
    /// picks the entry of any module, preferring the entries without a module, then those of the
    /// analyzed module itself. The first module visited (a copy of the analyzed module) is its
    /// root, so the path is `[]` there.
    pub fn in_module(mut self, path: Vec<Ident>) -> Self {
        self.module = Some(path);
        self
    }

    /// Replaces `Self` by `self_ty` in the rewritten copies, for a Repeater implemented for
    /// another type than the implementor of the Leaker trait (see [`Referrer::has_self()`] and
    /// [`Referrer::iter_with_self()`], which replaces `Self` in the leaked types in the same
    /// way).
    ///
    /// The leaked types are still found by their original tokens (e.g. `Rel<Self>` in the copy
    /// matches the leaked `Rel<Self>`), and the type passed to `type_fn` is the leaked type with
    /// `Self` replaced, so it agrees with the types of the Repeater's implementations. `Self` in
    /// the rest of the types of the copy is replaced as well (e.g. `::p::W<Self>` or
    /// `Self::Out`, also in bounds, where clauses and signatures), but not in the types returned
    /// by `type_fn`. Receivers (e.g. `&self`) and expressions other than those in types are not
    /// rewritten.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self, x: Rel<Self>) -> Self::Out; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().finish();
    /// let mut sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>) -> Self::Out);
    /// referrer
    ///     .into_visitor(None, |_, ix| parse_quote!(<Marker<__S> as Repeater<#ix>>::Type))
    ///     .with_self(parse_quote!(__S))
    ///     .visit_signature_mut(&mut sig);
    /// let expected: Signature = parse_quote!(
    ///     fn f(&self, x: <Marker<__S> as Repeater<0usize>>::Type) -> <__S>::Out
    /// );
    /// assert_eq!(quote!(#sig).to_string(), quote!(#expected).to_string());
    /// ```
    pub fn with_self(mut self, self_ty: Type) -> Self {
        self.self_ty = Some(self_ty);
        self
    }

    /// Stops rewriting the leaked types which mention `idents` (items or imports declared in a
    /// block or a module), except the entries in the module `keep`, if given: their relative
    /// paths were resolved in that very module.
    fn shadow_idents(&mut self, idents: Vec<Ident>, keep: Option<&[Ident]>) {
        let requirements = &self.referrer.requirements;
        let contexts = &self.referrer.contexts;
        self.referrer.map.retain(|key, entries| {
            if mentions_names(&key.ty, idents.clone(), Vec::new()) {
                entries.retain(|&ix| {
                    let module = requirements[contexts[ix]].module();
                    module.is_some() && module == keep
                });
            }
            !entries.is_empty()
        });
    }

    /// Shadows the items declared in `items` (see [`Visitor::shadow_idents()`]).
    fn shadow_items<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>, keep: bool) {
        fn use_names(tree: &UseTree, names: &mut Vec<Ident>) {
            match tree {
                UseTree::Path(UsePath { tree, .. }) => use_names(tree, names),
                UseTree::Name(UseName { ident }) if ident != "self" => names.push(ident.clone()),
                UseTree::Rename(UseRename { rename, .. }) => names.push(rename.clone()),
                UseTree::Group(UseGroup { items, .. }) => {
                    for tree in items {
                        use_names(tree, names)
                    }
                }
                _ => (),
            }
        }
        let mut names = Vec::new();
        for item in items {
            match item {
                Item::Struct(ItemStruct { ident, .. })
                | Item::Enum(ItemEnum { ident, .. })
                | Item::Union(ItemUnion { ident, .. })
                | Item::Trait(ItemTrait { ident, .. })
                | Item::TraitAlias(ItemTraitAlias { ident, .. })
                | Item::Type(ItemType { ident, .. })
                | Item::Const(ItemConst { ident, .. })
                | Item::Static(ItemStatic { ident, .. })
                | Item::Mod(ItemMod { ident, .. }) => names.push(ident.clone()),
                Item::Use(ItemUse { tree, .. }) => use_names(tree, &mut names),
                _ => (),
            }
        }
        if !names.is_empty() {
            let module = self.module.clone();
            self.shadow_idents(names, module.as_deref().filter(|_| keep));
        }
    }

    /// Returns a visitor with the generics of an item (or of a method, an associated type or a
    /// const if not `item`) in scope, and visits them. Those of the outermost item are not in
    /// scope if [`Visitor::with_outermost_generics_as_leakers()`] is set.
    fn with_generics_as(&mut self, generics: &mut Generics, item: bool) -> Self {
        let mut visitor = self.clone();
        visitor.item_depth += 1;
        // All the parameters are in scope of each other's bounds.
        if !(item && self.outermost_as_leakers && self.item_depth == 0) {
            visitor.scopes.push(Scope::new(generics));
        }
        for gp in generics.params.iter_mut() {
            visitor.visit_generic_param_mut(gp);
        }
        if let Some(where_clause) = &mut generics.where_clause {
            visitor.visit_where_clause_mut(where_clause);
        }
        visitor
    }

    fn with_item_generics(&mut self, generics: &mut Generics) -> Self {
        self.with_generics_as(generics, true)
    }

    fn with_generics(&mut self, generics: &mut Generics) -> Self {
        self.with_generics_as(generics, false)
    }

    fn with_signature_as(&mut self, sig: &mut Signature, item: bool) -> Self {
        let mut visitor = self.with_generics_as(&mut sig.generics, item);
        for input in sig.inputs.iter_mut() {
            visitor.visit_fn_arg_mut(input);
        }
        visitor.visit_return_type_mut(&mut sig.output);
        visitor
    }
}

impl<F: FnMut(Type, usize) -> Type> VisitMut for Visitor<F> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        let self_ty = self.self_requirement.as_ref();
        let (matching, module) = if self.exact {
            // The module is always compared.
            let module = self.module.as_deref().unwrap_or(&[]);
            (Matching::Exact { self_ty }, Some(module))
        } else {
            (Matching::Lenient { self_ty }, self.module.as_deref())
        };
        *i = self.referrer.expand_with(
            matching,
            module,
            &mut self.scopes,
            self.self_ty.as_ref(),
            i.clone(),
            &mut *self.type_fn.borrow_mut(),
        );
    }
    fn visit_receiver_mut(&mut self, i: &mut Receiver) {
        if self.self_ty.is_none() {
            return syn::visit_mut::visit_receiver_mut(self, i);
        }
        // The type of a receiver (`self: Box<Self>`, or `Self` in `&self`) is not rewritten.
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
    }
    fn visit_predicate_type_mut(&mut self, i: &mut PredicateType) {
        let binder = i.lifetimes.as_ref().map(Scope::binder);
        let pushed = binder.is_some();
        self.scopes.extend(binder);
        syn::visit_mut::visit_predicate_type_mut(self, i);
        if pushed {
            self.scopes.pop();
        }
    }
    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        let binder = i.lifetimes.as_ref().map(Scope::binder);
        let pushed = binder.is_some();
        self.scopes.extend(binder);
        syn::visit_mut::visit_trait_bound_mut(self, i);
        if pushed {
            self.scopes.pop();
        }
    }
    fn visit_const_param_mut(&mut self, i: &mut ConstParam) {
        // The type of a const parameter cannot be a projection, so it is kept as is.
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        if let Some(expr) = &mut i.default {
            self.visit_expr_mut(expr);
        }
    }
    fn visit_item_struct_mut(&mut self, i: &mut ItemStruct) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        visitor.visit_fields_mut(&mut i.fields);
    }
    fn visit_item_enum_mut(&mut self, i: &mut ItemEnum) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        for variant in i.variants.iter_mut() {
            visitor.visit_variant_mut(variant);
        }
    }
    fn visit_item_trait_mut(&mut self, i: &mut ItemTrait) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
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
        let mut visitor = self.with_item_generics(&mut i.generics);
        visitor.visit_fields_named_mut(&mut i.fields);
    }
    fn visit_item_type_mut(&mut self, i: &mut ItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
    }
    fn visit_signature_mut(&mut self, i: &mut Signature) {
        self.with_signature_as(i, true);
    }
    fn visit_item_impl_mut(&mut self, i: &mut ItemImpl) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        if let Some((_, path, _)) = &mut i.trait_ {
            visitor.visit_path_mut(path);
        }
        visitor.visit_type_mut(&mut i.self_ty);
        for item in i.items.iter_mut() {
            visitor.visit_impl_item_mut(item);
        }
    }
    fn visit_impl_item_fn_mut(&mut self, i: &mut ImplItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature_as(&mut i.sig, false);
        visitor.visit_block_mut(&mut i.block);
    }
    fn visit_impl_item_type_mut(&mut self, i: &mut ImplItemType) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
        visitor.visit_type_mut(&mut i.ty);
    }
    fn visit_item_mod_mut(&mut self, i: &mut ItemMod) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        if let Some((_, items)) = &mut i.content {
            let mut visitor = self.clone();
            // The first module visited is the root of the module paths.
            match &mut visitor.module {
                Some(path) => path.push(i.ident.clone()),
                None => visitor.module = Some(Vec::new()),
            }
            // The items in the module are nested items.
            visitor.item_depth += 1;
            visitor.shadow_items(items.iter(), true);
            for item in items.iter_mut() {
                visitor.visit_item_mut(item);
            }
        }
    }
    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature_as(&mut i.sig, true);
        visitor.visit_block_mut(i.block.as_mut());
    }
    fn visit_trait_item_fn_mut(&mut self, i: &mut TraitItemFn) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_signature_as(&mut i.sig, false);
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
        visitor.shadow_items(
            i.stmts.iter().filter_map(|stmt| match stmt {
                Stmt::Item(item) => Some(item),
                _ => None,
            }),
            false,
        );
        for stmt in i.stmts.iter_mut() {
            visitor.visit_stmt_mut(stmt);
        }
    }
}

/// The version of this crate, emitted with a serialized [`Referrer`] to detect a Referrer passed
/// between incompatible versions of `type-leak`.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Parses a [`Referrer`] serialized with its [`ToTokens`] implementation: the version of
/// `type-leak` as a string literal, (the number of contexts if it is more than one,) and a group
/// for each context (see [`Referrer::context()`]): the generics it uses if they are not empty
/// (the generics and the where clause, followed by `;`, see [`Referrer::context_generics()`]),
/// what `Self` means if it mentions it (`Self: path<args> ;` for a [`SelfRequirement::Bound`],
/// `Self = Type ;` for a [`SelfRequirement::Type`] or `Self = # n ;` for a
/// [`SelfRequirement::Call`], see [`Referrer::context_self()`]), the module if the types
/// depend on one (`mod a::b ;`, or `mod self ;` for the analyzed module itself, see [`Referrer::context_module()`]), and the leaked types in
/// parentheses. A leaked type may be preceded by `?` if it is not [required](Referrer::must()),
/// and by `@[crate, ..]` if it [requires external crates](Referrer::external_crates_require()),
/// in this order (e.g. `? @[foo, bar] ::foo::X`); a plain type is required and requires no
/// crate. The tokens after it are not consumed, so a Referrer can be followed by other tokens. It
/// is an error if the version differs from that of this crate.
impl Parse for Referrer {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let version: LitStr = input.parse()?;
        if version.value() != VERSION {
            return Err(Error::new(
                version.span(),
                format!(
                    "the Referrer was emitted by type-leak {}, but is parsed by type-leak {}",
                    version.value(),
                    VERSION
                ),
            ));
        }
        let n_contexts = if input.peek(LitInt) {
            input.parse::<LitInt>()?.base10_parse::<usize>()?
        } else {
            1
        };
        let mut entries = Vec::new();
        let mut requirements = Vec::new();
        for context in 0..n_contexts {
            // The generics, if any, are followed by `;`.
            let mut generics = Generics::default();
            if input.peek(Token![<]) || input.peek(Token![where]) {
                generics = input.parse()?;
                generics.where_clause = input.parse()?;
                input.parse::<Token![;]>()?;
            }
            // What `Self` means, if it is mentioned, is followed by `;` too.
            let mut self_ty = None;
            if input.peek(Token![Self]) {
                input.parse::<Token![Self]>()?;
                if input.parse::<Option<Token![:]>>()?.is_some() {
                    self_ty = Some(SelfRequirement::Bound(input.parse()?));
                } else {
                    input.parse::<Token![=]>()?;
                    if input.parse::<Option<Token![#]>>()?.is_some() {
                        self_ty = Some(SelfRequirement::Call(
                            input.parse::<LitInt>()?.base10_parse()?,
                        ));
                    } else {
                        self_ty = Some(SelfRequirement::Type(input.parse()?));
                    }
                }
                input.parse::<Token![;]>()?;
            }
            // The module, if it is not the analyzed one, is followed by `;` too.
            let mut module = None;
            if input.parse::<Option<Token![mod]>>()?.is_some() {
                module = Some(if input.parse::<Option<Token![self]>>()?.is_some() {
                    Vec::new()
                } else {
                    Punctuated::<Ident, Token![::]>::parse_separated_nonempty(input)?
                        .into_iter()
                        .collect()
                });
                input.parse::<Token![;]>()?;
            }
            requirements.push(Requirements::new(generics, self_ty).with_module(module));
            let content: syn::parse::ParseBuffer<'_>;
            parenthesized!(content in input);
            // Each entry is `[?] [@[crate, ..]] Type`.
            type Entry = (Type, bool, Vec<String>);
            let group = Punctuated::<Entry, Token![,]>::parse_terminated_with(&content, |input| {
                let must = input.parse::<Option<Token![?]>>()?.is_none();
                let mut crates = Vec::new();
                if input.parse::<Option<Token![@]>>()?.is_some() {
                    let list: syn::parse::ParseBuffer<'_>;
                    bracketed!(list in input);
                    crates = Punctuated::<Ident, Token![,]>::parse_terminated(&list)?
                        .into_iter()
                        .map(|ident| ident.to_string())
                        .collect();
                }
                Ok((input.parse()?, must, crates))
            })?;
            entries.extend(
                group
                    .into_iter()
                    .map(|(ty, must, crates)| (context, ty, must, crates)),
            );
        }
        Ok(Self::new(entries, requirements))
    }
}

/// Serializes the [`Referrer`] (with the version of `type-leak`), to be parsed back with its
/// [`Parse`] implementation, e.g. in another macro invocation. There is a group for each context
/// (at least one), preceded by the number of the groups if it is more than one (e.g. `"0.8.0" 2
/// (MyType) <T> ; (Wrap<T>)`). A group is the leaked types in parentheses, preceded by the
/// generics which the types of the context use (see [`Referrer::context_generics()`]) and `;` if
/// they are not empty (e.g. `<T> where T: Clone ; (Wrap<T>)`), and by what `Self` means (see
/// [`Referrer::context_self()`]) and `;` if the types mention it (e.g. `Self: MyTrait<T> ;
/// (Rel<Self>)`, `Self = MyStruct<T> ; (Rel<Self>)` or `Self = # 0 ; (Rel<Self>)`), and by `mod
/// a::b ;` (or `mod self ;`) if the types depend on their module (see [`Referrer::context_module()`]). An entry
/// which is not required is preceded by `?`, and one which requires external crates by `@[foo,
/// bar]` (e.g. `(? @[foo] ::foo::X)`).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::{quote, ToTokens};
/// let item: ItemStruct = parse_quote!(struct S { a: MyType });
/// let referrer = Leaker::from_struct(&item).unwrap().finish();
/// let tokens = referrer.to_token_stream();
/// assert_eq!(
///     tokens.to_string(),
///     format!("\"{}\" (MyType)", env!("CARGO_PKG_VERSION"))
/// );
/// assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
/// ```
impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        use proc_macro2::{Literal, Punct, Spacing};
        tokens.extend(quote!(#VERSION));
        let n_contexts = self.n_contexts();
        if n_contexts > 1 {
            // The number of the groups, so that the Referrer can be followed by other tokens.
            tokens.extend(quote!(#{Literal::usize_unsuffixed(n_contexts)}));
        }
        for context in 0..n_contexts {
            let entries: Vec<_> = self
                .leak_types
                .iter()
                .zip(&self.contexts)
                .zip(self.musts.iter().zip(&self.external_crates))
                .filter(|((_, c), _)| **c == context)
                .map(|((item, _), (must, crates))| (item, *must, crates))
                .collect();
            let generics = self.context_generics(context);
            if !generics.params.is_empty() || generics.where_clause.is_some() {
                tokens.extend(quote!(#generics #{&generics.where_clause} ;));
            }
            match self.context_self(context) {
                None => (),
                Some(SelfRequirement::Bound(path)) => tokens.extend(quote!(Self : #path ;)),
                Some(SelfRequirement::Type(ty)) => tokens.extend(quote!(Self = #ty ;)),
                Some(SelfRequirement::Call(n)) => {
                    tokens.extend(quote!(Self =));
                    tokens.extend(Some(proc_macro2::TokenTree::Punct(Punct::new(
                        '#',
                        Spacing::Alone,
                    ))));
                    tokens.extend(quote!(#{Literal::usize_unsuffixed(*n)} ;));
                }
            }
            if let Some(module) = self.context_module(context) {
                tokens.extend(quote!(mod));
                if module.is_empty() {
                    tokens.extend(quote!(self));
                }
                for (i, ident) in module.iter().enumerate() {
                    if i > 0 {
                        tokens.extend(quote!(::));
                    }
                    tokens.extend(quote!(#ident));
                }
                tokens.extend(quote!(;));
            }
            tokens.extend(quote! {
                (
                    #(for (item, must, crates) in entries), {
                        #(if !must) { ? }
                        #(if !crates.is_empty()) {
                            @[#(for name in crates), { #{Ident::new(name, proc_macro2::Span::call_site())} }]
                        }
                        #item
                    }
                )
            });
        }
    }
}

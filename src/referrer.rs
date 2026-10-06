//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::context::{
    assumed_bounds_hold, Assumed, Context, ContextId, Enclosing, Scope, SelfRequirement,
};
use crate::encode::{complete_generics_args, without_encoded_bounds, EncodeError};
use crate::graph::TypeKey;
use crate::util::{
    first_self_in_generics, generic_param_name, ident_name, mentions_names, token_names,
    type_mentions_self,
};
use proc_macro2::TokenStream;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use syn::ext::IdentExt;
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
/// Each type is an entry, identified by its tokens and its context: the generics it uses
/// (see [`Referrer::generics()`]) and, if it mentions `Self`, what `Self` means (see
/// [`SelfRequirement`]). A context is shared by the entries which have the same ones, and is
/// identified by a [`ContextId`] (see [`Referrer::context()`]). The Referrer has the contexts of
/// the generics of the calls of [`Leaker::intern_with()`](crate::Leaker::intern_with()) too (the
/// top-level contexts), also if no entry is in them. The same tokens in different contexts (e.g.
/// `Wrap<B>` where `B` is a `bool` in one method and a `u8` in another) are separate entries, and
/// the one a copy refers to is found by resolving the generics in scope where it appears.
///
/// # Required and optional entries
///
/// An entry is *required* if the type depends on the type context (a relative path, `Self`, a
/// generic of the Leaker, ...): the copy has to refer to it by the Repeater. It is *optional*
/// ([`Referrer::must()`] is `false`) if its own path is only an absolute one which names an
/// external crate (`::serde::Wrap<Rel>`, see
/// [`Leaker::allowed_paths`](crate::Leaker::allowed_paths)): it means the same type wherever the
/// crate has the same name, which the Leaker cannot tell of the Referrer's crate. The Referrer
/// decides, with [`Referrer::allow_ambiguous_crate_name()`] (default `true`: the optional entries
/// are left as written, so the output is that of a Leaker which does not see them at all, and
/// `false`: they are replaced like the required ones). The crates are matched by name only (see
/// there). The required entries come first, so their indices do not depend on the optional ones.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    pub(crate) leak_types: Vec<Type>,
    /// The context of each entry (see [`Referrer::context()`]).
    pub(crate) contexts: Vec<ContextId>,
    /// Whether each entry is required (see [`Referrer::must()`]). The required entries come
    /// first.
    pub(crate) musts: Vec<bool>,
    /// The sorted names of the external crates each entry requires (see
    /// [`Referrer::external_crates_require()`]).
    pub(crate) external_crates: Vec<Vec<String>>,
    /// Whether the optional entries are left as written (see
    /// [`Referrer::allow_ambiguous_crate_name()`]). Not serialized.
    pub(crate) allow_ambiguous_crate_name: bool,
    /// The contexts by their ids, which are distinct (see [`Referrer::context_generics()`]): those
    /// of the entries and the top-level contexts of the calls of the Leaker (which may have no
    /// entry). The keys may have gaps (see [`Referrer::contexts()`]).
    pub(crate) context_map: BTreeMap<ContextId, Context>,
    /// The candidates of a type, keyed by [`TypeKey`] (by tokens) as in the Leaker, so that e.g. a
    /// type wrapped in an invisible group is found as well.
    pub(crate) map: HashMap<TypeKey, Vec<usize>>,
}

/// How [`Referrer::find()`] matches the contexts of the entries with the generics in scope.
#[derive(Clone, Copy)]
enum Matching<'a> {
    /// The generics declared in the scopes are checked, and those which are not are assumed to be
    /// generics of the Leaker. If `self_ty` is given, an entry mentioning `Self` must be for it.
    Lenient {
        self_ty: Option<&'a SelfRequirement>,
        /// `Self` means something else here (in an item nested in the visited syntax which
        /// defines its own), so no entry which mentions it matches.
        no_self: bool,
        /// The names of the type parameters of the Leaker which the visited syntax is known to
        /// have and does not declare (see `Visitor::with_outermost_generics_as_leakers()`), so
        /// the paths starting with them are projections, whichever generics the entry has.
        leakers: &'a [String],
    },
    /// The first scope is that of the Leaker, so the contexts are compared exactly. If
    /// `self_ty` is given, `Self` means it, and otherwise it is not checked.
    Exact {
        self_ty: Option<&'a SelfRequirement>,
        no_self: bool,
    },
    /// The scope is that of the parameters before a defaulted one: the entry must use only them.
    Default,
}

impl Matching<'_> {
    /// Returns `true` if the entries which mention `Self` do not match.
    fn no_self(&self) -> bool {
        match self {
            Matching::Lenient { no_self, .. } | Matching::Exact { no_self, .. } => *no_self,
            Matching::Default => false,
        }
    }
}

impl Referrer {
    /// Builds a Referrer from its entries (the context, the type, whether it is required and the
    /// external crates it requires), in the order of their indices, and the contexts, which are
    /// distinct. There is at least one context (the context 0 of the types
    /// which use no generics, if there are no contexts at all), and one for each context of the
    /// entries; the others have no entry.
    pub(crate) fn new(
        entries: impl IntoIterator<Item = (ContextId, Type, bool, Vec<String>)>,
        mut context_map: BTreeMap<ContextId, Context>,
    ) -> Self {
        if context_map.is_empty() {
            context_map.insert(ContextId::new(0), Context::none());
        }
        let mut referrer = Self {
            leak_types: Vec::new(),
            contexts: Vec::new(),
            musts: Vec::new(),
            external_crates: Vec::new(),
            allow_ambiguous_crate_name: true,
            context_map,
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

    /// Returns the index of the entry of the type `ty` whose context matches the generics in
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
        enclosing: &Enclosing,
    ) -> Option<usize> {
        let candidates = self.map.get(&TypeKey::new(ty))?;
        let mut best: Option<((usize, usize, usize), usize)> = None;
        for &ix in candidates {
            let entry = &self.context_map[&self.contexts[ix]];
            if matching.no_self()
                && (entry.self_ty().is_some() || type_mentions_self(&self.leak_types[ix]))
            {
                continue;
            }
            let rank = match (module, entry.module()) {
                (Some(module), Some(entry)) if entry == module => 0,
                (Some(_), Some(_)) => continue,
                (Some(_), None) => 1,
                (None, None) => 0,
                (None, Some(entry)) => 1 + usize::from(!entry.is_empty()),
            };
            if let Some((assumed, hypothesised)) =
                Self::matches(matching, ty, scopes, enclosing, entry)
            {
                // The fewest generics assumed, among them the fewest hypotheses.
                let key = (rank, assumed + hypothesised, hypothesised);
                if best.map_or(true, |(better, _)| key < better) {
                    best = Some((key, ix));
                }
            }
        }
        let ix = best?.1;
        // An optional entry is left as written, if the Referrer is to trust the names of the
        // crates in it (the types inside it are looked up as for a type which is not leaked).
        if self.allow_ambiguous_crate_name && !self.musts[ix] {
            return None;
        }
        Some(ix)
    }

    /// Returns the number of the generics of the Leaker which `entry` assumes (see
    /// [`Referrer::find()`]), and the number of the names which are assumed to be type parameters
    /// of the Leaker for the paths around the type (`T::A<X>`, see below), if it matches the type
    /// `ty` in `scopes`.
    fn matches(
        matching: Matching,
        ty: &Type,
        scopes: &[Scope],
        enclosing: &Enclosing,
        entry: &Context,
    ) -> Option<(usize, usize)> {
        match matching {
            Matching::Exact { self_ty, .. } => {
                let context = Context::of(
                    ty,
                    scopes,
                    self_ty.or_else(|| entry.self_ty()),
                    &Assumed::default(),
                    enclosing,
                );
                if context == entry.without_module() {
                    Some((0, 0))
                } else {
                    None
                }
            }
            Matching::Lenient {
                self_ty, leakers, ..
            } => {
                if let (Some(wanted), Some(actual)) = (self_ty, entry.self_ty()) {
                    if wanted != actual {
                        return None;
                    }
                }
                let declared: HashSet<String> = scopes
                    .iter()
                    .flat_map(|scope| scope.params())
                    .map(generic_param_name)
                    .collect();
                // The generics of the entry which are not declared here are generics of the
                // Leaker, and so are the lifetimes around the type which are not declared (the
                // visited syntax does not declare the generics of the item, see
                // `Visitor::with_outermost_generics_as_leakers()`). So the bounds the type implies
                // on the generics declared here are those of the entry, which the context has to
                // agree with, and the bounds it implies on the others must be among theirs.
                let mut assumed = Assumed::default();
                for param in &entry.generics().params {
                    match param {
                        GenericParam::Lifetime(param)
                            if !declared.contains(&param.lifetime.to_string()) =>
                        {
                            assumed.lifetimes.push(param.lifetime.clone())
                        }
                        GenericParam::Type(param)
                            if !declared.contains(&ident_name(&param.ident)) =>
                        {
                            assumed.types.push(ident_name(&param.ident))
                        }
                        _ => (),
                    }
                }
                for name in leakers {
                    if !declared.contains(name) && !assumed.types.contains(name) {
                        assumed.types.push(name.clone());
                    }
                }
                let attempt = |assumed: Assumed| -> Option<usize> {
                    let mut assumed = assumed;
                    for lifetime in enclosing.lifetimes(scopes, &assumed) {
                        if !declared.contains(&lifetime.to_string())
                            && !assumed.lifetimes.contains(&lifetime)
                        {
                            assumed.lifetimes.push(lifetime);
                        }
                    }
                    let context = Context::of(ty, scopes, entry.self_ty(), &assumed, enclosing);
                    if !self_outlives_agree(
                        context.self_outlives(),
                        entry.self_outlives(),
                        &declared,
                    ) || !assumed_bounds_hold(ty, scopes, &assumed, enclosing, entry)
                    {
                        return None;
                    }
                    Self::matches_declared(&declared, &context, entry)
                };
                if let Some(assumed) = attempt(assumed.clone()) {
                    return Some((assumed, 0));
                }
                // A path around the type starting with a name which nothing here declares
                // (`T::A<X>` where `T` is a generic of the Leaker which the entry does not
                // mention: `Rel2<U>` of `&'a T::A<Rel2<U>>`) is a projection, which does not
                // pass on the lifetimes around it, if the name is a type parameter, and an
                // ordinary path if it is not. The entry cannot tell, so the former is assumed
                // (as for the other generics of the Leaker) if the latter does not match. Such an
                // entry is preferred less than one which matches without it, as the copy of an
                // entry with fewer bounds than needed is rejected by rustc (E0309), and one with
                // more is only stricter (see `Referrer::find()`).
                let heads = enclosing.undeclared_heads(scopes, &assumed);
                if heads.is_empty() {
                    return None;
                }
                let count = heads.len();
                assumed.types.extend(heads);
                attempt(assumed).map(|assumed| (assumed, count))
            }
            Matching::Default => {
                if entry.self_ty().is_some() {
                    return None;
                }
                let root: HashSet<String> = scopes[0].params().map(generic_param_name).collect();
                let context = Context::of(ty, &scopes[..1], None, &Assumed::default(), enclosing);
                let used = names_of_params(context.generics());
                let entry_params = names_of_params(entry.generics());
                if used.is_subset(&entry_params) && entry_params.is_subset(&root) {
                    Some((entry_params.len(), 0))
                } else {
                    None
                }
            }
        }
    }

    /// Compares the context of the type (`context`, as it depends on the generics
    /// `declared`) with `entry`: a generic declared must be required by both, with the same
    /// declaration (the bounds in the canonical form), and the others are assumed to be generics
    /// of the Leaker. Returns the number of them.
    fn matches_declared(
        declared: &HashSet<String>,
        context: &Context,
        entry: &Context,
    ) -> Option<usize> {
        let render = |param: &GenericParam| quote!(#param).to_string();
        let wanted: HashMap<String, String> = context
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
        let wanted_predicates = predicates(context.generics());
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

    /// Sets whether the Referrer's crate is trusted to name the same crates as the Leaker's by
    /// the same names (default `true`). It decides what is done to the *optional* entries, the
    /// types which are leaked only because their own path names an external crate
    /// ([`Referrer::must()`] is `false`, see
    /// [`Leaker::allowed_paths`](crate::Leaker::allowed_paths)).
    ///
    /// - With `true`, an optional entry is **left as written**: [`Referrer::into_visitor()`],
    ///   [`Referrer::expand()`] and the others do not replace it, and look the types inside it up
    ///   as for a type which is not leaked (a required entry in it is replaced). This is the
    ///   output of a Leaker which does not see the external crates at all.
    /// - With `false`, an optional entry is **replaced** like a required one, so the copy refers
    ///   to the type by the Repeater, whose implementation is written in the Leaker's crate, which
    ///   resolves the path there (an absolute type which contains a relative one, `::foo::Wrap<Rel>`,
    ///   is replaced as a whole).
    ///
    /// A required entry is always replaced. The flag is a choice of the Referrer's side, so it is
    /// not serialized: a Referrer which is parsed (see [`Parse`]) has the default, and the macro
    /// parsing it sets it where it knows the Referrer's crate.
    ///
    /// **The crates are matched by name only.** A crate with the same name in the Referrer's crate
    /// but another version, another package (a dependency renamed to that name, or an unrelated
    /// crate called so) or another origin (a registry, a git repository, a path) is treated as the
    /// one of the Leaker with `true`: the optional entry is left as written, and the copy names an
    /// item of the other crate, which fails to compile or compiles with another type. Pass `false`
    /// if the Referrer's crate may have other dependencies than the Leaker's (e.g. the Leaker is
    /// a macro which user crates call), and `true` if both crates are built from the same
    /// dependency graph (e.g. a workspace sharing its dependencies, or a crate and the macro crate
    /// which it ships with), which is the usual case. The crates which are known to be the same
    /// can also be allowed by the Leaker, which then has no optional entry for them (see
    /// [`Leaker::allowed_paths`](crate::Leaker::allowed_paths)).
    ///
    /// It replaces the types only: an absolute path in a trait bound, a value or the type of a
    /// const parameter cannot be leaked, so it is copied as written whatever this is, and so is a
    /// type which cannot be leaked (`::foo::Wrap<'_>`).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S { a: ::foo::X });
    /// let (leaker, _) = Leaker::from_struct(&item).unwrap();
    /// let referrer = leaker.finish();
    /// assert!(!referrer.must(0));
    /// assert_eq!(referrer.external_crates_require(0), ["foo"]);
    /// let ty: Type = parse_quote!(::foo::X);
    /// let replaced: Type = parse_quote!(Replaced);
    /// // Left as written by default.
    /// assert_eq!(referrer.expand(ty.clone(), |_, _| replaced.clone()), ty);
    /// let referrer = referrer.allow_ambiguous_crate_name(false);
    /// assert_eq!(referrer.expand(ty.clone(), |_, _| replaced.clone()), replaced);
    /// ```
    pub fn allow_ambiguous_crate_name(mut self, allow: bool) -> Self {
        self.allow_ambiguous_crate_name = allow;
        self
    }

    /// Returns the sorted, deduplicated names of the external crates (other than `core`) named by
    /// absolute paths (`::name::..`, by their first segment) in the `ix`-th type and its
    /// [generics](Referrer::generics()) and `Self` (see [`Referrer::context_self()`]), without
    /// `r#` (e.g. `async` for `::r#async::X`). They are matched by name only (see
    /// [`Referrer::allow_ambiguous_crate_name()`]).
    pub fn external_crates_require(&self, ix: usize) -> &[String] {
        &self.external_crates[ix]
    }

    /// Returns `false` if the `ix`-th type is an *optional* entry: it is leaked only because its
    /// own path is an absolute one naming an external crate (see
    /// [`Leaker::allowed_paths`](crate::Leaker::allowed_paths)), which the Referrer's crate may
    /// not mean the same crate by. Such an entry is left as written, or replaced, as
    /// [`Referrer::allow_ambiguous_crate_name()`] says. It is `true` if the entry depends on the
    /// type context (a relative path, a relative lifetime, a type macro, ...), and then it is
    /// always replaced, also if it contains an absolute path (`Rel<::foo::X>`).
    ///
    /// The required entries come first: all the entries for which this is `true` have smaller
    /// indices than the others, so the indices of the required entries do not depend on the
    /// optional ones. The Repeater's implementation of an optional entry is needed only by a
    /// Referrer which replaces it, so a Leaker whose Referrers all leave them as written may
    /// skip it.
    pub fn must(&self, ix: usize) -> bool {
        self.musts[ix]
    }

    /// Returns `true` if there are no types to be encoded.
    pub fn is_empty(&self) -> bool {
        self.leak_types.is_empty()
    }

    /// Returns an iterator over the leaked types, by their index.
    ///
    /// The Repeater implementation of an entry depends on its context: the generics it uses
    /// ([`Referrer::generics()`]), its `Self` ([`Referrer::context_self()`]) and, for
    /// [`Leaker::from_module()`](crate::Leaker::from_module()), the module where it must be
    /// written ([`Referrer::context_module()`]). See [`Referrer::generics()`] for the complete
    /// pattern (and [`Referrer::iter_with_generics()`] / [`Referrer::iter_with_self()`]); the
    /// simple form below is valid only for entries without any of them:
    ///
    /// ```ignore
    /// for (ix, ty) in referrer.iter().enumerate() {
    ///     let c = referrer.context(ix);
    ///     assert!(referrer.context_generics(c).params.is_empty()
    ///         && referrer.context_self(c).is_none()
    ///         && referrer.context_module(c).is_none());
    ///     quote!(impl<__S: ?Sized> #repeater<(), #id, #ix> for __S { type Type = #ty; })
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.leak_types.iter()
    }

    /// Returns the generics which the `ix`-th type uses, and which the Repeater's implementation
    /// of it must declare (and encode exactly, in its trait arguments), e.g. the generics of the
    /// Leaker and those of a trait method or a GAT (or of a nested item, or a `for<..>` binder).
    /// It is empty for most types. This is the same as
    /// [`Referrer::context_generics()`] of the context of the type. The implementation does not
    /// declare their lifetime bounds, which are encoded instead (see below): it declares
    /// [`Referrer::impl_generics()`].
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
    /// # Outlives bounds
    ///
    /// The bounds of a generic include the outlives bounds which the type implies (the Leaker adds
    /// them as if they were written), because the type is well-formed only if they hold, and
    /// neither the implementation nor the copies of a signature may assume them otherwise (rustc
    /// rejects them with E0309). Only the bounds which are exact are recorded: those which hold
    /// whatever the definition of the types says, and those which are written.
    ///
    /// - A reference `&'a X` or `&'a mut X` requires `X: 'a`.
    /// - The bounds written in the Leaker (`T: 'a`, `'b: 'a`, also in the where clause) are kept.
    ///
    /// For each such `'a` which is a generic lifetime (not `'static`, `'_`, or the lifetime of a
    /// `for<..>` binder), `T: 'a` is added for each type parameter `T` mentioned in `X`, and `'b:
    /// 'a` for each other lifetime parameter `'b` mentioned in `X`. Everything in `X` is looked
    /// into, as rustc does, the arguments of a trait object included: `dyn Tr<T> + 'b` outlives
    /// `'a` only if `'b` and `T` do. The bounds written in the Leaker are merged with these.
    ///
    /// A path with a lifetime argument `R<'a, .., X, ..>` is **not** assumed to require `X: 'a`:
    /// it does if the definition of `R` says so (`struct R<'a, T>(&'a T)` has `T: 'a` inferred,
    /// RFC 2093), but the Leaker cannot see it, and `struct Unrelated<'a, T>(&'a (), T)` does not,
    /// where an assumed bound would make the copy stricter than the original (a caller whose `T`
    /// is shorter than `'a` would be rejected). So the lifetime arguments of a path add nothing
    /// to the generics, and where the definition of the path needs a bound, **write it** in the
    /// signature: `fn f<'a, T: 'a>(&self, x: Rel<'a, T>)`, which makes it a written bound like
    /// any other. Without it the implementation `impl<'a, T> Repeater<(&'a (), T), ..> for S {
    /// type Type = Rel<'a, T>; }` is rejected with E0309 ("the parameter type `T` may not live
    /// long enough"), as the header does not imply `T: 'a` (the same goes for `Self`, see below,
    /// and for the type arguments of a path in the field of a struct, see below).
    ///
    /// The lifetime bounds by a lifetime parameter of the generics are encoded in the trait
    /// arguments of the Repeater (`T: 'a` as `&'a T`, see
    /// [`encode_generics_params_to_ty_with()`](crate::encode_generics_params_to_ty_with())),
    /// which implies them wherever the encoding is written: in the implementation, and in the
    /// copy of a signature or a field. So the implementation **must not declare them**: declare
    /// [`Referrer::impl_generics()`] (or [`without_encoded_bounds()`](crate::without_encoded_bounds())
    /// for the generics of [`Referrer::iter_with_self()`]), which are these generics without the
    /// lifetime bounds that are encoded, and keeps the others (`T: 'static`). A declared bound is
    /// an obligation wherever the implementation is selected, which may not hold there (see
    /// [`Referrer::impl_generics()`]).
    ///
    /// `Self: 'a` is not among them, as `Self` is not a generic. The same types imply it
    /// (`&'a Self`, `&'a Self::Item`, with the same rule: a reference), and it is a part of the
    /// context of the type instead, which the encoding carries as well, as `&'a Self` (see
    /// [`Referrer::context_self_outlives()`]). Like the others it is not declared by the
    /// implementation, which is rejected with E0309 without it (`__S` may not live long enough),
    /// whereas a declared `__S: 'a` fails where nothing proves it, such as the signature of a free
    /// function. A path with a lifetime does not imply it (`Rel<'a, Self>`, `Rel<'a, Self::Item>`),
    /// so it is written: `where Self: 'a` (also with other bounds, `where Self: Sized + 'a`) is
    /// moved from the predicates to [`Referrer::context_self_outlives()`] (the other bounds of the
    /// predicate stay, `Self: Sized`), and the context is the same whether `Self: 'a` is written
    /// or implied by a reference. Only a bound by a lifetime parameter of the generics is moved:
    /// `Self: 'static`, a bound by a lifetime which is not one of them, a bound under a `for<..>`
    /// binder and the bound of another type (`Self::Item: 'a`) are where predicates like the
    /// others, which the implementation declares.
    ///
    /// A type leaked alone is a component of the references around it, and has the bounds that
    /// they imply on it: `Rel2<T>` of `&'a Vec<Rel2<T>>` is leaked with the generics `<'a, T:
    /// 'a>`, as if it was written `&'a Rel2<T>`, and so is the one of `&'a Rel2<T>` and `&'a dyn
    /// Iterator<Item = Rel2<T>>`. The copy `&'a <Self as Repeater<(&'a (), T, &'a T),
    /// ..>>::Type` is well-formed in the field of a struct, where the one without the bound
    /// (`Repeater<(T,), ..>`) is rejected (E0309): the original has `T: 'a` inferred by RFC 2093,
    /// which does not go through a projection. `'a` is among the generics even if the type does
    /// not mention it, and the same type under other lifetimes, or under none, is another entry.
    /// A path with a lifetime argument is not such a reference: `Rel2<T>` of `Ref<'a, Rel2<T>>`
    /// is leaked with the generics which are written, `<T>` (and `<'a, T: 'a>` if the signature
    /// writes `T: 'a`, as it has to for the field of a struct `struct S<'a, T: 'a>(Ref<'a,
    /// Rel2<T>>)`, whose copy `Ref<'a, <Self as Repeater<(T,), ..>>::Type>` is rejected with
    /// E0309 otherwise: the struct must write the same bound the signature does for the Referrer
    /// to find the entry).
    ///
    /// The types that pass on the lifetimes they must outlive are those of rustc: the referent of
    /// `&'a X` and `&'a mut X` adds `'a` (it is the only type which adds one: the lifetime
    /// arguments of a path `R<'a, X>` are not recorded, as above), and everything made of types
    /// passes on what it must outlive (the elements of tuples, arrays and slices, parentheses,
    /// groups and raw pointers, the type arguments of any path, the inputs and the output of a
    /// function pointer, and the arguments and the associated type bindings of a trait object:
    /// `&'a Vec<X>` requires `X: 'a`), but not a projection (`<X as Tr>::A`, which does not
    /// require `X: 'a`, and neither does a path starting with a type parameter or `Self`,
    /// `T::A<X>` and `Self::A<X>`: `&'a T::A<Rel2<U>>` is well-formed without `U: 'a`, so
    /// `Rel2<U>` has the generics `<U>`), an `impl Trait` and the length of an array. Whether a
    /// path is a projection of a type parameter is told by the generics in scope (the Leaker knows
    /// them; the Referrer takes the generics which the entry has and does not see declared to be
    /// the Leaker's, and, where nothing else matches, a name which is not declared for a type
    /// parameter of the Leaker too, see [`Referrer::into_visitor()`]). `'static`, `'_` and the
    /// lifetimes of a `for<..>` binder are not generics, so they are not recorded (`T: 'static`
    /// stays to be declared by the struct, which the original needs as well).
    ///
    /// [`Referrer::into_visitor()`], [`Referrer::expand()`] and the others compute the same from
    /// the types around the one they look up, so give them the whole type (or item): a type taken
    /// out of another is looked up as if nothing was around it.
    ///
    /// If the type mentions `Self` (see [`Referrer::context_self()`]), the Repeater's
    /// implementation is for the implementor or the type `Self` is. The bounds and where
    /// predicates of the generics are kept as they are, so they may mention `Self` too (e.g.
    /// `fn f<T: PartialEq<Self>>(&self)`), with the same meaning.
    ///
    /// The Repeater's implementation also depends on what `Self` means in the type
    /// ([`Referrer::context_self()`]) and, with [`Leaker::from_module()`](crate::Leaker::from_module()),
    /// on the module ([`Referrer::context_module()`]). For a type mentioning `Self` whose
    /// `context_self()` is [`SelfRequirement::Bound`], the implementation is for `__S`, where
    /// `Self` of the trait is not available, so its type and generics are those of
    /// [`Referrer::iter_with_self()`] with `__S` (e.g. `Wrap<Self::Item>` is `Wrap<<__S>::Item>`,
    /// and `Self: Sized` is `__S: Sized`):
    ///
    /// ```ignore
    /// // In the Leaker's crate
    /// let marker: Type = parse_quote!(__S);
    /// let replaced: Vec<_> = referrer.iter_with_self(&marker).collect();
    /// for (ix, (ty, generics)) in referrer.iter_with_generics().enumerate() {
    ///     let context = referrer.context(ix);
    ///     let (ty, generics) = match referrer.context_self(context) {
    ///         Some(SelfRequirement::Bound(_)) => (&replaced[ix].0, &replaced[ix].1),
    ///         _ => (ty, generics),
    ///     };
    ///     // `Self: 'a` is encoded as `&'a Self` (`&'a __S` here) too.
    ///     let self_in_impl: Type = match referrer.context_self(context) {
    ///         Some(SelfRequirement::Type(_)) => parse_quote!(Self),
    ///         _ => marker.clone(),
    ///     };
    ///     let encoded = slots.encode_generics_params_to_ty_with_self(
    ///         &generics.params,
    ///         &self_in_impl,
    ///         referrer.context_self_outlives(context),
    ///     )?;
    ///     // There may be no parameters: `impl<__S: ?Sized>`. The lifetime bounds which `encoded`
    ///     // contains (`T: 'a`, `Self: 'a`) are implied by the header, so they are not declared.
    ///     // (`referrer.impl_generics(context)` is the same for the generics of the context.)
    ///     let params = without_encoded_bounds(generics).params;
    ///     let predicates = generics.where_clause.iter().flat_map(|wc| &wc.predicates);
    ///     let header = match referrer.context_self(context) {
    ///         // `path` is the trait with its parameters, written in the Leaker's module.
    ///         Some(SelfRequirement::Bound(path)) => quote! {
    ///             impl<#(#params,)* __S: ?Sized + #path> Repeater<#encoded, #ID, #ix> for __S
    ///         },
    ///         Some(SelfRequirement::Type(self_ty)) => quote! {
    ///             impl<#(#params),*> Repeater<#encoded, #ID, #ix> for #self_ty
    ///         },
    ///         // `None`, or `Call`, which is not tied to a type.
    ///         _ => quote! {
    ///             impl<#(#params,)* __S: ?Sized> Repeater<#encoded, #ID, #ix> for __S
    ///         },
    ///     };
    ///     // If `referrer.context_module(context)` is `Some`, emit it inside that module.
    ///     quote! { #header where #(#predicates,)* { type Type = #ty; } }
    /// }
    ///
    /// // In the Referrer's crate (in the implementation of `MyTrait<#trait_args>`)
    /// let completed = referrer.complete_args(&trait_args, &trait_params, |..| ..)?;
    /// let args = referrer.generic_args(ix, &trait_params, &completed)?;
    /// let encoded = slots.encode_generics_args_to_ty_with_self(
    ///     &args,
    ///     &referrer.generics(ix).params,
    ///     &parse_quote!(Self),
    ///     referrer.context_self_outlives(referrer.context(ix)),
    /// )?;
    /// parse_quote!(<Self as Repeater<#encoded, #ID, #ix>>::Type)
    /// ```
    ///
    /// # Lifetimes of `for<..>` binders
    ///
    /// The lifetimes of a binder are generics of the types inside it (`Rel<'x>` in `for<'x>
    /// fn(Rel<'x>)` or in `F: for<'x> Fn(Rel<'x>)` is leaked with `<'x>`), unless the output of the
    /// binder mentions them (the return type of a function pointer, or of the parenthesized
    /// arguments of a trait (`Fn(A) -> R`), or an associated type binding (`Iterator<Item = R>`))
    /// and no input keeps them as written. The copy `for<'x> fn(<Self as Repeater<(&'x (),),
    /// ..>>::Type) -> &'x u8` would be rejected then (E0581, or E0582 in a bound or a trait
    /// object), because a lifetime which occurs only in a projection is not constrained by the
    /// inputs.
    ///
    /// An input keeps `'x` (an *anchor*) if it occurs in a position of the input which constrains
    /// it (a reference, a pointer, a slice, an array, a tuple, a path, a function pointer or a trait
    /// object, with its associated type bindings, its lifetime bound and the output of its
    /// `Fn(..) -> R`, but not a projection, the length of an array, `impl Trait` or a type macro)
    /// and no type from the input down to it is leaked: in
    /// `F: for<'x> Fn(&'x str, Parser<'x>) -> &'x str` the reference `&'x str` stays, so
    /// `Parser<'x>` is leaked alone, as in `for<'x> fn(&'x u8) -> Rel<'x>`. A type using `'x` is
    /// never leaked unless its own path depends on the type context (`Parser<'x>`,
    /// `Rel2<&'x str>`, which then holds the anchor, so it is not one), so the types around the
    /// anchor, such as a tuple, are never leaked. A trait object, a path or a reference to a trait
    /// object which is a cause itself (`dyn RelIter<Item = &'x u8>`, `&'x dyn RelTrait`) is leaked
    /// as a whole, so it keeps no anchor. So is the type of an external crate (an absolute path
    /// which is not `::core::..` and not
    /// [allowed](crate::Leaker::allowed_paths), `::other::Wrap<&'x u8>`) in the eyes of the Leaker,
    /// which cannot tell whether the Referrer replaces it (see
    /// [`Referrer::allow_ambiguous_crate_name()`]): nothing in it is an anchor, whichever the
    /// Referrer does. The occurrences in the paths of `::core`, and in those the Leaker allows
    /// (also the relative ones, such as the prelude's `Option`), stay where they are.
    ///
    /// Without an anchor, a type using such a lifetime is not leaked on its own: the function
    /// pointer or the trait object holding the binder is leaked as a whole instead (it declares the
    /// lifetime, so it is not among its generics), and it is an error
    /// ([`NotInternableError`](crate::NotInternableError)) if that cannot be (it has an elided
    /// lifetime, or is `impl Trait`), or if the binder is in a bound or a where predicate, which is
    /// not a type. A path whose definition is not seen is assumed to be a struct, an enum or a
    /// union: a type alias of a projection does not constrain `'x`.
    ///
    /// # Lifetimes of methods and trait objects
    ///
    /// A trait object without a lifetime bound which is the referent of a reference with a named
    /// lifetime (`&'b dyn RelTrait`) or an argument of a path with that lifetime (`Ref<'b, dyn
    /// RelTrait>`) gets its lifetime bound from the reference or the path (the lifetime of the
    /// reference, unless the trait has a lifetime bound of its own, as `trait RelTrait: 'static`),
    /// which only the definitions tell, so it is never leaked alone (with the lifetime written out,
    /// the Repeater's implementation is rejected for a trait with a bound of its own, E0478). If it
    /// has to be leaked itself (it names a relative trait), the whole reference or path is leaked,
    /// with `'b` among its generics, so rustc computes the default in the implementation, whether
    /// `'b` is a lifetime of the Leaker or a local one: `&'b dyn RelTrait` is the entry, and the
    /// copy is `<Self as Repeater<(&'b (),), ..>>::Type`. Nothing needs to be written for such a
    /// trait object (the workaround of writing the bound, `&'b (dyn RelTrait + 'static)`, is not
    /// needed).
    ///
    /// The copy has a lifetime `'b` of a method only in a projection, which constrains nothing.
    /// rustc rejects the implementation of a method whose lifetime parameters differ in whether
    /// they are early- or late-bound from those of the trait (E0195), and `'b` is late-bound in the
    /// trait but early-bound in the copy if it is in the return type and no bound mentions it. So
    /// the Leaker reports an error ([`NotInternableError`](crate::NotInternableError), which names
    /// the lifetime) for a reference or a path with a method's lifetime, whose trait object has to
    /// be leaked itself, if the return type mentions it, no bound or where predicate mentions it,
    /// some other argument mentions it, and no other argument keeps it as written in the copy:
    /// `fn f<'b>(&self, x: &'b dyn RelTrait) -> &'b u8` is rejected, and `fn f<'b>(&'b self, x: &'b
    /// dyn RelTrait) -> &'b u8` (the receiver, or a reference like `y: &'b u8`, also in a tuple, an
    /// array, a slice or a pointer) or `fn f<'b, 'c: 'b>(&self, x: &'b dyn RelTrait, y: &'c u8) ->
    /// &'b u8` (a bound makes it early-bound in both) are not. A lifetime which no argument
    /// mentions is not affected either (`fn f<'b>(&self) -> &'b dyn RelTrait`, `-> Ref<'b, dyn
    /// RelTrait>`): it is only in the return type, so it is early-bound in the original as well as
    /// in the copy, which has it only in a projection. A reference to a trait object without a
    /// lifetime bound keeps it only if the trait object is certainly not a cause itself (so that the
    /// reference is never leaked as a whole): `y: &'b dyn ::core::fmt::Debug`, also with what is
    /// leaked alone inside it (`dyn ::core::iter::Iterator<Item = Rel>`), does, and `y: &'b dyn
    /// RelTrait` does not, as it may be leaked as a whole itself (nor a bare identifier in it which
    /// may be a const, an external crate which the Leaker does not allow, `y: &'b dyn
    /// ::other::Tr`, as the Referrer may replace the reference to it, or a path of the analyzed
    /// module, where the Leaker is unsure), and neither does a type which may be leaked
    /// (`Vec<&'b u8>`): conservative. Keep it anchored, make the trait absolute so that it need not be leaked, or
    /// write the lifetime bound of the trait object (`&'b (dyn RelTrait + 'b)`, which leaks the
    /// trait object alone, and keeps the reference in the copy). A lifetime of the Leaker, and one
    /// of the outermost function (which is not compared with a declaration), are not affected. An
    /// elided lifetime is an error as well (the Repeater cannot name it).
    pub fn generics(&self, ix: usize) -> &Generics {
        self.context_generics(self.contexts[ix])
    }

    /// Returns the context of the `ix`-th type, as a [`ContextId`]: it identifies the generics the
    /// type uses and what `Self` means in it (see [`Referrer::generics()`] and
    /// [`Referrer::context_self()`]). The entries which have the same ones are in the same
    /// context, whichever call of [`Leaker::intern_with()`](crate::Leaker::intern_with()) found
    /// them.
    ///
    /// The ids are those assigned by the [`Leaker`](crate::Leaker) when the contexts first
    /// appeared (from 0, in that order), and returned by
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()): they are not renumbered by
    /// [`Leaker::finish()`](crate::Leaker::finish()), so the ids of the contexts of the types
    /// removed by [`Leaker::reduce_roots()`](crate::Leaker::reduce_roots()) are missing here (see
    /// [`Referrer::contexts()`]).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T> { fn f(&self, t: T) -> Rel<T>; });
    /// let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    /// let referrer = leaker.finish();
    /// // `Rel<T>` uses exactly the generics of the trait.
    /// assert_eq!(top_level, referrer.context(0));
    /// ```
    pub fn context(&self, ix: usize) -> ContextId {
        self.contexts[ix]
    }

    /// Returns the contexts in ascending order, the arguments for
    /// [`Referrer::context_generics()`], [`Referrer::context_self()`] and
    /// [`Referrer::context_module()`]: the contexts of the entries (see [`Referrer::context()`]),
    /// and the top-level contexts returned by
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()) and the other methods of the
    /// [`Leaker`](crate::Leaker) analyzing an item, which have no entry if none of the types
    /// leaked uses exactly the generics of the item. The other ids are missing, so there may be
    /// gaps: a context of which [`Leaker::reduce_roots()`](crate::Leaker::reduce_roots()) keeps
    /// no type is not here, unless it is a top-level context.
    ///
    /// There is at least one context: if there are no others, it is the context 0 of the types
    /// which use no generics.
    pub fn contexts(&self) -> impl Iterator<Item = ContextId> + '_ {
        self.context_map.keys().copied()
    }

    /// Returns the generics which the types of `context` use, in the canonical form, which the
    /// Repeater's implementation of an entry in `context` must encode exactly (the parameters and
    /// the lifetime bounds, and `Self: 'a`, see [`Referrer::context_self_outlives()`]) in its trait
    /// arguments. See [`Referrer::generics()`], including the
    /// outlives bounds that a reference implies, and those the references around a type imply (and
    /// the lifetimes of them, which are among the generics), which are among the bounds here, with
    /// those which are written. The implementation
    /// declares [`Referrer::impl_generics()`] instead: these generics without the lifetime bounds
    /// that are encoded, which it must not declare. For the top-level context
    /// of a call of [`Leaker::intern_with()`](crate::Leaker::intern_with()), they are the generics
    /// of the call (without the defaults), also if no type uses exactly them.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T, U = Rel<T>> {
    ///     fn f<const B: bool>(&self, t: T) -> Wrap<B>;
    ///     fn g(&self) -> Rel2;
    /// });
    /// let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    /// let referrer = leaker.finish();
    /// // `Wrap<B>` uses `B`, `Rel2` uses nothing and `Rel<T>` (the default of `U`) uses `T`, in
    /// // the order of their first appearance, and then there are the generics of the trait
    /// // `<T, U>`, which no type uses exactly.
    /// assert_eq!(referrer.contexts().count(), 4);
    /// let used: Vec<usize> = referrer
    ///     .contexts()
    ///     .map(|c| referrer.context_generics(c).params.len())
    ///     .collect();
    /// assert_eq!(used, [1, 0, 1, 2]);
    /// assert_eq!(referrer.context_generics(top_level).params.len(), 2);
    /// ```
    ///
    /// # In the Referrer's crate
    ///
    /// The Referrer encodes the arguments of a context with the same generics as the Leaker's
    /// ([`encode_generics_args_to_ty_with()`](crate::encode_generics_args_to_ty_with()) requires
    /// them): for an entry, they are [`Referrer::generics()`], with the arguments of
    /// [`Referrer::generic_args()`]. To encode the arguments of the trait (or the struct) which
    /// the Leaker analyzed, e.g. `u8` of `impl MyTrait<u8> for ..`, they are the generics of its
    /// top-level context. The Referrer cannot tell which context that is (a Referrer may have
    /// several top-level contexts, and one which is also the context of an entry is not marked,
    /// so it is not serialized), but the [`ContextId`] which the Leaker returns is the same in the
    /// Referrer (ids are not renumbered), and it is written and parsed as an integer, so the
    /// Leaker passes it along with the Referrer. The generics are in the canonical form, whose
    /// parameters are not in the order of the trait (they are sorted by kind and name, and have
    /// no defaults), so the arguments are completed with [`Referrer::complete_args()`] and put in
    /// their order with [`Referrer::context_generic_args()`]:
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::parse::{ParseStream, Parser};
    /// # use template_quote::quote;
    /// // In the Leaker's crate: its Repeater's implementations encode these generics
    /// // (`<'a, T: 'a, const N: usize>`, with the lifetime bound encoded as `&'a T`)
    /// let tr: ItemTrait = parse_quote!(trait Tr<'a, const N: usize, T: 'a, U = ::core::primitive::u8> {
    ///     fn f(&self, t: &'a T) -> Rel;
    /// });
    /// let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    /// let referrer = leaker.finish();
    /// let params = &referrer.context_generics(top_level).params;
    /// let leaked = encode_generics_params_to_ty(params);
    /// let expected: Type = parse_quote!((&'a (), T, U, [(); N], &'a T));
    /// assert_eq!(quote!(#leaked).to_string(), quote!(#expected).to_string());
    /// // The tokens emitted by the Leaker's macro for the Referrer's
    /// let tokens = quote!(#referrer, #top_level);
    ///
    /// // In the Referrer's crate
    /// let (referrer, top_level) = Parser::parse2(
    ///     |input: ParseStream| {
    ///         let referrer: Referrer = input.parse()?;
    ///         input.parse::<Token![,]>()?;
    ///         Ok((referrer, input.parse::<ContextId>()?))
    ///     },
    ///     tokens,
    /// )
    /// .unwrap();
    /// // For `impl<'q> Tr<'q, 3, u16> for ..`, omitting `U`
    /// let args: AngleBracketedGenericArguments = parse_quote!(<'q, 3, u16>);
    /// let completed = referrer
    ///     .complete_args(&args.args, &tr.generics.params, |_, _, _| unreachable!())
    ///     .unwrap();
    /// let args = referrer
    ///     .context_generic_args(top_level, &tr.generics.params, &completed)
    ///     .unwrap();
    /// let params = &referrer.context_generics(top_level).params;
    /// let ty = encode_generics_args_to_ty_with(&args, params, |expr, _| {
    ///     parse_quote!([(); #expr])
    /// })
    /// .unwrap();
    /// let expected: Type = parse_quote!((&'q (), u16, ::core::primitive::u8, [(); 3], &'q u16));
    /// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
    /// ```
    ///
    /// The generics of a top-level context are those of the call, with only the bounds written
    /// there: unlike [`Referrer::generics()`] of an entry, they do not have the bounds that the
    /// types of the entries imply (`T: 'a` of `&'a Rel<T>` in `trait Tr<'a, T> { fn f(&self, x:
    /// &'a Rel<T>); }`), which the Repeater's implementation of such an entry must encode. So a
    /// Repeater whose implementations encode the generics of the trait for every entry carries
    /// only the bounds the trait declares, and its implementation is rejected (E0309) for such an
    /// entry: encode the generics of each entry instead.
    ///
    /// # Panics
    ///
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_generics(&self, context: ContextId) -> &Generics {
        self.context_map[&context].generics()
    }

    /// Returns the generics of `context` (see [`Referrer::context_generics()`]) to declare as the
    /// generics of the Repeater's implementation of an entry in it: without the lifetime bounds
    /// which the trait arguments encode, and with everything else as it is.
    ///
    /// The encoders encode the outlives bounds of the generics (the written ones, and those which
    /// a reference implies) as references in the trait arguments (`T: 'a` as `&'a T`, `'b: 'a` as
    /// `&'a &'b ()`, see
    /// [`encode_generics_params_to_ty_with()`](crate::encode_generics_params_to_ty_with())), which
    /// imply them wherever the arguments are written. So the implementation must not declare them
    /// as well: a declared bound is an obligation wherever the implementation is selected, which
    /// rustc may not prove there (the bound implied by the argument `&'a T` is enough: `impl<'a,
    /// T, __S: ?Sized> Repeater<(&'a (), T, &'a T), ..> for __S` compiles wherever the copy of the
    /// signature does). A bound which the signature does not write and no reference implies is
    /// not here (`T: 'a` of `Rel<'a, T>` is not assumed, so it is written: `fn f<'a, T:
    /// 'a>(&self, x: Rel<'a, T>)`, see [`Referrer::generics()`]).
    ///
    /// The rule is that of the encoders: a bound by a lifetime parameter of the list (`'a`
    /// above) is removed, from a type parameter or a lifetime parameter. A bound `T: 'static`, a
    /// bound by a lifetime which is not a parameter, the trait bounds and the where predicates
    /// are kept. Use [`without_encoded_bounds()`](crate::without_encoded_bounds()) for generics
    /// which are not those of a context, such as the ones returned by
    /// [`Referrer::iter_with_self()`] (which the Repeater's implementation of an entry whose
    /// [`context_self()`](Referrer::context_self()) is [`SelfRequirement::Bound`] declares).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr<'a, T> { fn f(&self, x: &'a Rel<T>); });
    /// let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
    /// let context = referrer.context(0);
    ///
    /// // `&'a Rel<T>` requires `T: 'a`, which the trait arguments encode (so does a bound which
    /// // is written, `trait Tr<'a, T: 'a> { fn f(&self, x: Rel<'a, T>); }`, but a path with a
    /// // lifetime does not imply one):
    /// let generics = referrer.context_generics(context);
    /// let encoded = encode_generics_params_to_ty(&generics.params);
    /// assert_eq!(quote!(#generics).to_string(), "< 'a , T : 'a >");
    /// assert_eq!(quote!(#encoded).to_string(), quote!((&'a (), T, &'a T)).to_string());
    ///
    /// // so the implementation declares the generics without it:
    /// // `impl<'a, T, __S: ?Sized> Repeater<(&'a (), T, &'a T), ..> for __S`
    /// let declared = referrer.impl_generics(context);
    /// assert_eq!(quote!(#declared).to_string(), "< 'a , T >");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn impl_generics(&self, context: ContextId) -> Generics {
        without_encoded_bounds(self.context_generics(context))
    }

    /// Returns what `Self` means in the types of `context`, if they mention it (it is `None` for
    /// a top-level context, see [`Leaker::intern_with()`](crate::Leaker::intern_with())).
    ///
    /// The Repeater's implementation of an entry in `context` is for the type described: `None`
    /// for any type (`impl<.., __S: ?Sized> Repeater<..> for __S`), [`SelfRequirement::Bound`] for
    /// the types implementing the trait (`impl<.., __S: ?Sized + Trait<..>> Repeater<..> for
    /// __S`), and [`SelfRequirement::Type`] for the type (`impl<..> Repeater<..> for Type`).
    /// [`SelfRequirement::Call`] is not tied to a type, so the implementation may be for any type
    /// like `None`.
    ///
    /// In an implementation for `__S` (`None`, `Call` and `Bound`), `Self` of the leaked type
    /// would be `__S` itself, so for a `Bound` entry (e.g. `Wrap<Self::Item>`, or a bound
    /// `Self: Sized`) emit the type and the generics from [`Referrer::iter_with_self()`] with
    /// `__S` (`Wrap<<__S>::Item>`, `__S: Sized`). For `Type`, `Self` is the type of the `impl`, so
    /// the type is emitted as it is (see [`Referrer::generics()`] for the whole pattern).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T> { fn f(&self) -> Rel<Self>; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
    /// match referrer.context_self(referrer.context(0)) {
    ///     Some(SelfRequirement::Bound(path)) => {
    ///         assert_eq!(quote!(#path).to_string(), "Tr < T >")
    ///     }
    ///     _ => unreachable!(),
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_self(&self, context: ContextId) -> Option<&SelfRequirement> {
        self.context_map[&context].self_ty()
    }

    /// Returns the lifetimes `'a` such that the types of `context` imply `Self: 'a` (they are
    /// parameters of [`Referrer::context_generics()`] of `context`, sorted), e.g. `'a` for `&'a
    /// Rel<Self>` and `&'a Rel<Self::Item>`: the type is well-formed only if `Self: 'a`, like
    /// `&'a T` requires `T: 'a` (a path with a lifetime such as `Rel<'a, Self>` does not imply it,
    /// as its definition is not seen, see [`Referrer::generics()`]), or the item says so: a
    /// written `where Self: 'a` (also `where Self: Sized + 'a`, or in the where clause of the
    /// trait) is moved here from the where predicates of the generics, merged with the implied
    /// ones, so the context is the same whichever way it is known, and the implementation does not
    /// declare it either (only a bound by a lifetime parameter of the context is moved: not `Self:
    /// 'static`, a bound under a `for<..>` binder, nor `Self::Item: 'a`, which stay where
    /// predicates). It is empty if the types do not imply it and the item does not say so, and if
    /// they do not mention `Self` ([`Referrer::context_self()`] is `None`).
    ///
    /// The encoding of the generics carries `Self: 'a` like a lifetime bound (`&'a Self`, or `&'a
    /// __S` in the implementation for any type of the trait), so the Repeater's implementation
    /// does not declare it, whereas the generics do not mention it, as `Self` is not a generic: use
    /// [`RepeaterSlots::encode_generics_params_to_ty_with_self()`](crate::RepeaterSlots::encode_generics_params_to_ty_with_self())
    /// and
    /// [`RepeaterSlots::encode_generics_args_to_ty_with_self()`](crate::RepeaterSlots::encode_generics_args_to_ty_with_self())
    /// (or [`encode_generics_params_to_ty_with_self()`](crate::encode_generics_params_to_ty_with_self())
    /// and
    /// [`encode_generics_args_to_ty_with_self()`](crate::encode_generics_args_to_ty_with_self())
    /// with your own encoding of the const parameters) to encode it, with the type of the
    /// implementation, or `Self` where the copy is, as `self_ty`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f<'a>(&'a self) -> &'a Rel<Self>; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
    /// let context = referrer.context(0);
    /// let lifetime: Lifetime = parse_quote!('a);
    /// assert_eq!(referrer.context_self_outlives(context), [lifetime.clone()]);
    ///
    /// // A path with a lifetime implies nothing, so it is written, and it is the same then.
    /// let tr: ItemTrait = parse_quote!(trait Tr {
    ///     fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized + 'a;
    /// });
    /// let written = Leaker::from_trait(&tr).unwrap().0.finish();
    /// assert_eq!(written.context_self_outlives(written.context(0)), [lifetime]);
    /// // Only `Self: Sized` is left of the predicate.
    /// let generics = written.context_generics(written.context(0));
    /// assert_eq!(
    ///     quote!(#{&generics.where_clause}).to_string(),
    ///     "where Self : :: core :: marker :: Sized"
    /// );
    ///
    /// // In the Leaker's crate, the implementation is for the implementor `__S` of the trait.
    /// let slots = RepeaterSlots::new(parse_quote!(Repeater));
    /// let marker: Type = parse_quote!(__S);
    /// let encoded = slots
    ///     .encode_generics_params_to_ty_with_self(
    ///         &referrer.context_generics(context).params,
    ///         &marker,
    ///         referrer.context_self_outlives(context),
    ///     )
    ///     .unwrap();
    /// let expected: Type = parse_quote!((&'a (), &'a __S));
    /// assert_eq!(quote!(#encoded).to_string(), quote!(#expected).to_string());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_self_outlives(&self, context: ContextId) -> &[Lifetime] {
        self.context_map[&context].self_outlives()
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
    /// the modules, if no module is analyzed, and for a top-level context (see
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with())).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let m: ItemMod = parse_quote!(mod m {
    ///     pub struct A { y: Rel }
    ///     pub mod inner { pub struct C { y: Rel } }
    /// });
    /// let (leaker, top_level) = Leaker::from_module(&m).unwrap();
    /// let referrer = leaker.finish();
    /// let contexts: Vec<ContextId> = referrer.contexts().collect();
    /// // The module itself, `inner` and the top-level context of the module.
    /// assert_eq!(contexts.len(), 3);
    /// assert_eq!(referrer.context_module(contexts[0]), Some(&[][..]));
    /// assert_eq!(
    ///     referrer.context_module(contexts[1]),
    ///     Some(&[Ident::new("inner", proc_macro2::Span::call_site())][..])
    /// );
    /// assert_eq!(referrer.context_module(top_level), None);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_module(&self, context: ContextId) -> Option<&[Ident]> {
        self.context_map[&context].module()
    }

    /// Returns `true` if a leaked type, or the generics (the bounds and the where predicates) of
    /// an entry, refers to `Self` (e.g. `Self`, `Self::Item`, `Rel<Self>`, `&Self`, or `<Self as
    /// ::p::Tr>::X`, but not `::p::Self`). The generics of a top-level context which has no entry
    /// are not considered.
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
    /// assert!(Leaker::from_trait(&tr).unwrap().0.finish().has_self());
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self) -> Rel; });
    /// assert!(!Leaker::from_trait(&tr).unwrap().0.finish().has_self());
    /// ```
    pub fn has_self(&self) -> bool {
        self.leak_types.iter().any(type_mentions_self)
            || self.contexts.iter().any(|context| {
                first_self_in_generics(self.context_map[context].generics()).is_some()
            })
    }

    /// Like [`Referrer::iter_with_generics()`], with `Self` replaced by `self_ty` in the types and
    /// the generics (see [`Referrer::has_self()`]), for a Repeater implemented for another type
    /// than the implementor of the Leaker trait (e.g. a marker type generic over `__S`), where
    /// `Self` would refer to that type instead.
    ///
    /// This is how the Repeater's implementation of an entry whose
    /// [`context_self()`](Referrer::context_self()) is [`SelfRequirement::Bound`] is written, with
    /// `self_ty` being its `__S` (`impl<.., __S: ?Sized + Trait<..>> Repeater<..> for __S`): for
    /// the types and for the generics (e.g. `where Self: Sized` becomes `where __S: Sized`),
    /// whereas [`Referrer::iter_with_generics()`] suits the other entries.
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
    /// let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
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
    /// let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
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
    ///             encode_generics_args_to_ty_with(&args, params, |_, _| unreachable!()).unwrap();
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
    /// This is [`Referrer::context_generic_args()`] of the context of the type.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T> {
    ///     fn f<const B: bool>(&self, t: T) -> Wrap<T, B>;
    /// });
    /// let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
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
        self.context_generic_args(self.contexts[ix], root_params, completed_args)
    }

    /// Like [`Referrer::generic_args()`], for the generics of any `context` (see
    /// [`Referrer::context_generics()`]): the arguments of its parameters at the use site, one for
    /// each, in the order of [`Referrer::context_generics()`], which may differ from that of
    /// `root_params`, as the generics of a context are in the canonical form: the parameters are
    /// sorted by kind and name, so the arguments of the Leaker's trait cannot be matched with
    /// them by position, but by name, as this method does.
    ///
    /// This is how the Referrer encodes the arguments of the trait with the generics of the
    /// top-level context (see [`Referrer::context_generics()`], which has the context, and tells
    /// what to do with the omitted ones):
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// // In the Leaker's crate: the generics are `<'a, T: 'a, const N: usize>`, in this order.
    /// let tr: ItemTrait = parse_quote!(trait Tr<'a, const N: usize, T: 'a> {
    ///     fn f(&self) -> Rel;
    /// });
    /// let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    /// let referrer = leaker.finish();
    /// let params = &referrer.context_generics(top_level).params;
    /// let leaked = encode_generics_params_to_ty(params);
    /// let expected: Type = parse_quote!((&'a (), T, [(); N], &'a T));
    /// assert_eq!(quote!(#leaked).to_string(), quote!(#expected).to_string());
    ///
    /// // In the Referrer's crate, for `impl<'q> Tr<'q, 3, u8> for ..`: the arguments in the order
    /// // of the parameters of the trait are put in that of the generics of the context.
    /// let args: AngleBracketedGenericArguments = parse_quote!(<'q, 3, u8>);
    /// let completed = referrer
    ///     .complete_args(&args.args, &tr.generics.params, |_, _, _| unreachable!())
    ///     .unwrap();
    /// let args = referrer
    ///     .context_generic_args(top_level, &tr.generics.params, &completed)
    ///     .unwrap();
    /// let expected: Vec<GenericArgument> =
    ///     vec![parse_quote!('q), parse_quote!(u8), parse_quote!(3)];
    /// assert_eq!(quote!(#(#args),*).to_string(), quote!(#(#expected),*).to_string());
    /// let ty = encode_generics_args_to_ty_with(&args, params, |expr, _| parse_quote!([(); #expr]))
    ///     .unwrap();
    /// let expected: Type = parse_quote!((&'q (), u8, [(); 3], &'q u8));
    /// assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_generic_args(
        &self,
        context: ContextId,
        root_params: &Punctuated<GenericParam, Token![,]>,
        completed_args: &[GenericArgument],
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        self.context_generics(context)
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
            .map(move |(ty, context)| (ty, self.context_map[context].generics()))
    }

    /// Converts this `Referrer` into a [`Visitor`] that can be used to transform AST nodes.
    ///
    /// The `type_fn` callback is called for each type that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type.
    /// See [`Referrer::expand()`] for details.
    ///
    /// A type is replaced if an entry has the same tokens and its context matches the generics
    /// declared where the type is (see [`Visitor`]): the generics of the entry which are declared
    /// there (by an item, a method or a binder being visited) must be declared in the same way
    /// there and be used by the type, and the generics used by the type must be those of the
    /// entry. The other generics of the entry are assumed to be the generics of the Leaker, as
    /// the visited syntax does not declare them (see
    /// [`Visitor::with_outermost_generics_as_leakers()`]). If several entries match, the one which
    /// assumes the fewest generics of the Leaker is chosen, and then the first. The context of a
    /// type has the lifetimes which the references around it make it outlive (see
    /// [`Referrer::generics()`]), which the visitor keeps track of while it walks down the types
    /// it visits: the bounds which a type implies on the generics of the Leaker which are not
    /// declared there must be among those of the entry, so `Rel2<T>` of `&'a Vec<Rel2<T>>` is not
    /// replaced by the entry of `Rel2<T>` on its own, whichever generics are declared. So if the
    /// Leaker is analyzed by several calls of
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
            leaker_types: Vec::new(),
            self_requirement,
            scopes: Vec::new(),
            outermost_as_leakers: false,
            self_ty: None,
            item_depth: 0,
            own_self: false,
            module: None,
        }
    }

    /// Like [`Referrer::into_visitor()`], but the types are looked up with the exact context:
    /// `root_generics` are the generics of the Leaker (as passed to
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with())), the outermost scope in which the
    /// context of a type is computed, like the Leaker does, so only the entry with the same
    /// context is found (also if its generics are not declared in the visited syntax).
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
    /// of the Leaker (see [`Referrer::into_visitor()`]), and with nothing around it: the
    /// lifetimes that the types around a type make it outlive are part of its context (see
    /// [`Referrer::generics()`]), and are known only for the types inside `ty`. A type taken out of
    /// a larger one is found as the one on its own, so expand the larger type.
    ///
    /// ```ignore
    /// referrer.expand(ty, |_, ix| {
    ///     let encoded = ...; // the generics of the entry, see `Referrer::generic_args()`
    ///     parse_quote!(<Self as #repeater<#encoded, #id, #ix>>::Type)
    /// })
    /// ```
    pub fn expand(&self, ty: Type, type_fn: impl FnMut(Type, usize) -> Type) -> Type {
        self.expand_with(
            Matching::Lenient {
                self_ty: None,
                no_self: false,
                leakers: &[],
            },
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
            /// What the type being folded must outlive because of the types around it, as the
            /// Leaker sees it (see [`Enclosing`]); while its components are folded, what they
            /// must.
            enclosing: Enclosing,
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
                let enclosing = self.enclosing.clone();
                let found =
                    self.referrer
                        .find(self.matching, self.module, &ty, self.scopes, &enclosing);
                if found.is_none() {
                    self.enclosing = enclosing.components_of(&ty, self.scopes.len());
                }
                let folded = match (found, self.replace_self) {
                    (Some(ix), None) => (self.type_fn)(ty, ix),
                    (Some(ix), Some(mut replace)) => (self.type_fn)(replace.fold_type(ty), ix),
                    (None, None) => syn::fold::fold_type(self, ty),
                    (None, Some(replace)) => replace.fold_type_with(self, ty),
                };
                self.enclosing = enclosing;
                folded
            }
            fn fold_expr(&mut self, expr: Expr) -> Expr {
                // The types in an expression are not components of the types around it.
                let enclosing = std::mem::take(&mut self.enclosing);
                let folded = match self.replace_self {
                    Some(replace) => replace.fold_expr_with(self, expr),
                    None => syn::fold::fold_expr(self, expr),
                };
                self.enclosing = enclosing;
                folded
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
            enclosing: Enclosing::default(),
            type_fn,
        }
        .fold_type(ty)
    }
}

/// Returns `true` if the lifetimes `Self` outlives in the context of a type found in the copy
/// (`found`) agree with those of the entry (`entry`): the type implies the same, and the entry
/// may have more only for the lifetimes the copy does not declare, which are assumed to be those
/// of the Leaker (see [`Referrer::into_visitor()`]). The entry may have a `Self: 'a` which the
/// Leaker found written (`where Self: 'a`, see [`Referrer::context_self_outlives()`]), which the
/// copy shows only where it is written again, as in a method, and not in the where clause of the
/// trait (the items of which are not copied).
fn self_outlives_agree(found: &[Lifetime], entry: &[Lifetime], declared: &HashSet<String>) -> bool {
    found.iter().all(|lifetime| entry.contains(lifetime))
        && entry
            .iter()
            .all(|lifetime| found.contains(lifetime) || !declared.contains(&lifetime.to_string()))
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
/// rewritten only if an entry has the context it has with them (see
/// [`Referrer::into_visitor()`] for how they are matched, and
/// [`Referrer::generics()`]). A type mentioning generics which are declared differently from the
/// entry's (e.g. `T: Clone` in the `impl<T: Clone>` of the visited item, where it was leaked
/// with `T` alone) is not rewritten, since it refers to them rather than to the Leaker's items.
/// See [`Visitor::with_outermost_generics_as_leakers()`] to treat the generics of the outermost
/// item as those of the Leaker instead.
///
/// A type visited (`VisitMut::visit_type_mut()`) is looked up as a whole, with the lifetimes
/// which the types inside it get from the ones around them (see [`Referrer::generics()`]), but
/// with nothing around it: visit the type or the item which holds it, not a type taken out of
/// another one.
///
/// The items and imports declared in a block shadow the leaked types mentioning their names in
/// the block, and those declared in a module do so in that module only, not in its submodules
/// (where the types depending on a module are keyed by that module, see
/// [`Referrer::context_module()`]), as in Rust's scoping rules.
#[derive(Debug)]
pub struct Visitor<F> {
    referrer: Referrer,
    type_fn: Rc<RefCell<F>>,
    /// The first of the `scopes` is that of the Leaker, and the contexts are compared exactly
    /// (see [`Referrer::into_visitor_with_root()`]).
    exact: bool,
    /// The names of the type parameters of the outermost item which are not declared in `scopes`
    /// (see [`Visitor::with_outermost_generics_as_leakers()`]).
    leaker_types: Vec<String>,
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
    /// Whether the syntax being visited is in an item nested in the visited syntax which defines
    /// its own `Self` (a struct, enum, union, trait or impl), where `Self` is not that of the
    /// copy.
    own_self: bool,
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
            leaker_types: self.leaker_types.clone(),
            self_requirement: self.self_requirement.clone(),
            scopes: self.scopes.clone(),
            outermost_as_leakers: self.outermost_as_leakers,
            self_ty: self.self_ty.clone(),
            item_depth: self.item_depth,
            own_self: self.own_self,
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
    ///
    /// The visitor still knows the type parameters of the outermost item, so a path starting with
    /// one (`T::A<X>`) is a projection of it here, as it is in the Leaker (see
    /// [`Referrer::generics()`]).
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
    /// `Self` means something else in a struct, an enum, a union, a trait or an impl nested in the
    /// visited syntax (e.g. in a method body), so there it is not replaced, and no leaked type
    /// mentioning `Self` is found. The outermost item visited, and the signature if only that is
    /// visited, are not nested.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self, x: Rel<Self>) -> Self::Out; });
    /// let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
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

    /// Returns the type which replaces `Self` (see [`Visitor::with_self()`]), unless `Self` means
    /// something else here.
    fn replaced_self(&self) -> Option<&Type> {
        self.self_ty.as_ref().filter(|_| !self.own_self)
    }

    /// Stops rewriting the leaked types which mention `idents` (items or imports declared in a
    /// block or a module). The items of a module are not in scope in its submodules, so, if
    /// `module`, only the entries which depend on no module are shadowed: the others are keyed by
    /// the module whose scope resolves them, which no other module's declarations shadow (and
    /// its own are the ones their paths were resolved with).
    fn shadow_idents(&mut self, idents: Vec<Ident>, module: bool) {
        let context_map = &self.referrer.context_map;
        let contexts = &self.referrer.contexts;
        self.referrer.map.retain(|key, entries| {
            if mentions_names(&key.ty, idents.clone(), Vec::new()) {
                entries.retain(|&ix| module && context_map[&contexts[ix]].module().is_some());
            }
            !entries.is_empty()
        });
    }

    /// Shadows the items declared in `items`, which are those of a module if `module` (see
    /// [`Visitor::shadow_idents()`]).
    fn shadow_items<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>, module: bool) {
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
                Item::ExternCrate(ItemExternCrate { ident, rename, .. }) => {
                    names.push(rename.as_ref().map_or(ident, |(_, rename)| rename).clone())
                }
                _ => (),
            }
        }
        if !names.is_empty() {
            self.shadow_idents(names, module);
        }
    }

    /// Returns a visitor with the generics of an item (or of a method, an associated type or a
    /// const if not `item`) in scope, and visits them. Those of the outermost item are not in
    /// scope if [`Visitor::with_outermost_generics_as_leakers()`] is set.
    fn with_generics_as(&mut self, generics: &mut Generics, item: bool) -> Self {
        let mut visitor = self.clone();
        visitor.item_depth += 1;
        // An item nested in the visited syntax defines its own `Self` if it is a struct, an enum,
        // a union, a trait or an impl (for the others, which cannot use `Self`, it does not
        // matter).
        if item && self.item_depth > 0 {
            visitor.own_self = true;
        }
        // All the parameters are in scope of each other's bounds.
        if !(item && self.outermost_as_leakers && self.item_depth == 0) {
            visitor.scopes.push(Scope::new(generics));
        } else {
            visitor
                .leaker_types
                .extend(generics.params.iter().filter_map(|param| match param {
                    GenericParam::Type(param) => Some(ident_name(&param.ident)),
                    _ => None,
                }));
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
        let no_self = self.own_self;
        let (matching, module) = if self.exact {
            // The module is always compared.
            let module = self.module.as_deref().unwrap_or(&[]);
            (Matching::Exact { self_ty, no_self }, Some(module))
        } else {
            (
                Matching::Lenient {
                    self_ty,
                    no_self,
                    leakers: &self.leaker_types,
                },
                self.module.as_deref(),
            )
        };
        *i = self.referrer.expand_with(
            matching,
            module,
            &mut self.scopes,
            self.self_ty.as_ref().filter(|_| !no_self),
            i.clone(),
            &mut *self.type_fn.borrow_mut(),
        );
    }
    fn visit_receiver_mut(&mut self, i: &mut Receiver) {
        if self.replaced_self().is_none() {
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
    fn visit_item_trait_alias_mut(&mut self, i: &mut ItemTraitAlias) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_item_generics(&mut i.generics);
        for bound in i.bounds.iter_mut() {
            visitor.visit_type_param_bound_mut(bound);
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

/// The prefix of the format tag which a serialized [`Referrer`] starts with: a string literal of
/// this and [`Referrer::FORMAT_VERSION`].
const FORMAT_TAG_PREFIX: &str = "type-leak-format-";

impl Referrer {
    /// The version of the serialized form of a Referrer (see its [`ToTokens`] and [`Parse`]
    /// implementations), written in front of it as the string literal `"type-leak-format-N"` and
    /// checked when it is parsed: a Referrer of another format, or one without the tag (e.g. one
    /// serialized by a version which had none), is rejected with an error. There is no
    /// compatibility with the other formats.
    ///
    /// A Leaker and a Referrer come from different macro expansions, possibly from different
    /// builds of `type-leak`. What the Referrer holds is meaningful only to the build which made
    /// it: the contexts of the entries are compared with the ones computed where a type is copied
    /// (see [`Referrer::generics()`]), so the data of another build may silently match nothing.
    ///
    /// **Bump this number by hand whenever a change makes a Referrer serialized by the previous
    /// version behave differently when it is used by the new one**, i.e. whenever
    ///
    /// - the syntax of the serialization changes, or
    /// - the rules which compute or compare the contexts change: what a context contains (the
    ///   generics, the implied outlives bounds, `Self`, the module), how the types around a type
    ///   pass on what they must outlive, which paths are projections, how an entry is found for a
    ///   type (the matching of the contexts), and so on.
    ///
    /// It need not change for the changes which do not affect them (documentation, the errors of
    /// the Leaker, the encoders, ...). The formats so far:
    ///
    /// 1. The first tagged one: the contexts have the outlives bounds that the types around a type
    ///    imply, and a path starting with a type parameter or `Self` (`T::A<X>`) is a projection,
    ///    like the qualified one (`<T as Tr>::A<X>`), which does not pass those bounds on to the
    ///    types inside it.
    pub const FORMAT_VERSION: u32 = 1;

    /// Returns the error of a Referrer whose first token is `found`: a string literal which is
    /// not the tag of this format (`Some`), or something else (`None`).
    fn format_error(span: proc_macro2::Span, found: Option<&LitStr>) -> Error {
        let found = match found.map(|lit| lit.value()) {
            Some(value) => match value.strip_prefix(FORMAT_TAG_PREFIX) {
                Some(number) if !number.is_empty() => format!("format {}", number),
                _ => format!("an unrecognized tag {:?}", value),
            },
            None => "no format tag".to_string(),
        };
        Error::new(
            span,
            format!(
                "this Referrer was serialized by an incompatible type-leak ({}, expected {}): the \
                 macros which emit and parse it must use the same version of type-leak",
                found,
                Self::FORMAT_VERSION
            ),
        )
    }
}

/// Parses a [`Referrer`] serialized with its [`ToTokens`] implementation: the format tag (see
/// [`Referrer::FORMAT_VERSION`]) as a string literal, (the number of contexts if it is more than
/// one,) and a group for each context (see [`Referrer::context()`]), in ascending order of their
/// ids: the id as `#
/// n` if it is not the expected one (0 for the first group, and the id of the previous group plus
/// one for the others, so it is omitted if there are no gaps), the generics it uses if they are not
/// empty (the generics and the where clause, followed by `;`, see [`Referrer::context_generics()`]),
/// what `Self` means if it mentions it (`Self: path<args> ;` for a [`SelfRequirement::Bound`],
/// `Self = Type ;` for a [`SelfRequirement::Type`] or `Self = # n ;` for a
/// [`SelfRequirement::Call`], see [`Referrer::context_self()`]), the lifetimes which `Self` is
/// implied to outlive if there are any (`Self : 'a + 'b ;`, see
/// [`Referrer::context_self_outlives()`]), the module if the types
/// depend on one (`mod a::b ;`, or `mod self ;` for the analyzed module itself, see
/// [`Referrer::context_module()`]), and the leaked types in parentheses (none for a top-level
/// context which no type is in, see [`Leaker::intern_with()`](crate::Leaker::intern_with())). A leaked type may be preceded by `?` if it is not [required](Referrer::must()),
/// and by `@[crate, ..]` if it [requires external crates](Referrer::external_crates_require()),
/// in this order (e.g. `? @[foo, bar] ::foo::X`); a plain type is required and requires no
/// crate. The entries are numbered in the order they are written, but for the required ones,
/// which come first (as they do in the Leaker's output, see [`Referrer::must()`]), so the
/// indices are the same as those of the Referrer which was serialized. The tokens after it are
/// not consumed, so a Referrer can be followed by other tokens. It
/// is an error if the format tag is missing or is that of another format (see
/// [`Referrer::FORMAT_VERSION`]): old data is not accepted either.
impl Parse for Referrer {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let span = input.span();
        let tag = if input.peek(LitStr) {
            Some(input.parse::<LitStr>()?)
        } else {
            None
        };
        let expected = format!("{}{}", FORMAT_TAG_PREFIX, Self::FORMAT_VERSION);
        match &tag {
            Some(tag) if tag.value() == expected => (),
            Some(tag) => return Err(Self::format_error(tag.span(), Some(tag))),
            None => return Err(Self::format_error(span, None)),
        }
        let n_contexts = if input.peek(LitInt) {
            input.parse::<LitInt>()?.base10_parse::<usize>()?
        } else {
            1
        };
        let mut entries = Vec::new();
        let mut context_map = BTreeMap::new();
        // The id of a group without one: 0, and then the one after the previous group.
        let mut next = 0;
        for _ in 0..n_contexts {
            let span = input.span();
            let context = if input.parse::<Option<Token![#]>>()?.is_some() {
                let context: ContextId = input.parse()?;
                if context.index() < next {
                    return Err(Error::new(
                        span,
                        "the ids of the contexts must be in ascending order",
                    ));
                }
                context
            } else {
                ContextId::new(next)
            };
            next = context
                .index()
                .checked_add(1)
                .ok_or_else(|| Error::new(span, "the id of the context is too large"))?;
            // The generics, if any, are followed by `;`.
            let mut generics = Generics::default();
            if input.peek(Token![<]) || input.peek(Token![where]) {
                generics = input.parse()?;
                generics.where_clause = input.parse()?;
                input.parse::<Token![;]>()?;
            }
            // What `Self` means, if it is mentioned, is followed by `;` too, and so are the
            // lifetimes it outlives (`Self: 'a + 'b ;`, which a path cannot start with).
            let mut self_ty = None;
            let mut self_outlives = Vec::new();
            while input.peek(Token![Self]) {
                input.parse::<Token![Self]>()?;
                if input.parse::<Option<Token![:]>>()?.is_some() {
                    if input.peek(Lifetime) {
                        self_outlives =
                            Punctuated::<Lifetime, Token![+]>::parse_separated_nonempty(input)?
                                .into_iter()
                                .collect();
                    } else {
                        self_ty = Some(SelfRequirement::Bound(input.parse()?));
                    }
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
            context_map.insert(
                context,
                Context::new(generics, self_ty)
                    .with_self_outlives(self_outlives)
                    .with_module(module),
            );
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
                        .map(|ident| ident.unraw().to_string())
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
        // The required entries come first (see `Referrer::must()`), in the order of the groups.
        entries.sort_by_key(|(_, _, must, _)| !must);
        Ok(Self::new(entries, context_map))
    }
}

/// Returns the token `#`, which cannot be written in `quote!` (it interpolates).
fn pound() -> proc_macro2::TokenTree {
    proc_macro2::TokenTree::Punct(proc_macro2::Punct::new('#', proc_macro2::Spacing::Alone))
}

/// Returns the identifier of the external crate `name` (stored without `r#`), as `r#name` if it is
/// a keyword, so that it is parsed back as an identifier.
fn crate_ident(name: &str) -> Ident {
    if parse_str::<Ident>(name).is_ok() {
        Ident::new(name, proc_macro2::Span::call_site())
    } else {
        Ident::new_raw(name, proc_macro2::Span::call_site())
    }
}

/// Serializes the [`Referrer`] (with the tag of its format, see [`Referrer::FORMAT_VERSION`]), to be parsed back with its
/// [`Parse`] implementation, e.g. in another macro invocation. There is a group for each context
/// (at least one, also the top-level ones which have no entry: `()`), in ascending order of
/// their ids ([`Referrer::contexts()`]), preceded by the
/// number of the groups if it is more than one (e.g. `"type-leak-format-1" 2 (MyType) <T> ; (Wrap<T>)`). A
/// group is the leaked types in parentheses, preceded by the id of the context as `# 3`, if it
/// is not the expected one (0 for the first group, and the id of the previous group plus one for
/// the others, so the ids are written only where there are gaps: `"type-leak-format-1" 2 (MyType) # 2 <T> ;
/// (Wrap<T>)` has the contexts 0 and 2, and a Referrer of the context 1 alone is `"type-leak-format-1" # 1
/// (MyType)`), by the generics which the types of the context use (see
/// [`Referrer::context_generics()`]) and `;` if they are not empty (e.g. `<T> where T: Clone ;
/// (Wrap<T>)`), and by what `Self` means (see
/// [`Referrer::context_self()`]) and `;` if the types mention it (e.g. `Self: MyTrait<T> ;
/// (Rel<Self>)`, `Self = MyStruct<T> ; (Rel<Self>)` or `Self = # 0 ; (Rel<Self>)`), by the
/// lifetimes which `Self` is implied to outlive (see [`Referrer::context_self_outlives()`]) and `;`
/// if there are any (e.g. `<'a> ; Self: MyTrait ; Self: 'a ; (& 'a Rel<Self>)`), and by `mod
/// a::b ;` (or `mod self ;`) if the types depend on their module (see [`Referrer::context_module()`]). An entry
/// which is not required is preceded by `?`, and one which requires external crates by `@[foo,
/// bar]` (e.g. `(? @[foo] ::foo::X)`).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # use template_quote::{quote, ToTokens};
/// let item: ItemStruct = parse_quote!(struct S { a: MyType });
/// let referrer = Leaker::from_struct(&item).unwrap().0.finish();
/// let tokens = referrer.to_token_stream();
/// assert_eq!(
///     tokens.to_string(),
///     format!("\"type-leak-format-{}\" (MyType)", Referrer::FORMAT_VERSION)
/// );
/// assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
/// // A Referrer of another format (or without the tag) is rejected.
/// let other = quote!("type-leak-format-0" (MyType));
/// assert!(parse2::<Referrer>(other)
///     .unwrap_err()
///     .to_string()
///     .contains("incompatible type-leak (format 0, expected 1)"));
/// ```
impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        use proc_macro2::Literal;
        let tag = format!("{}{}", FORMAT_TAG_PREFIX, Self::FORMAT_VERSION);
        tokens.extend(quote!(#tag));
        let n_contexts = self.context_map.len();
        if n_contexts > 1 {
            // The number of the groups, so that the Referrer can be followed by other tokens.
            tokens.extend(quote!(#{Literal::usize_unsuffixed(n_contexts)}));
        }
        // The id of a group which has none written: 0, and then the one after the previous group.
        let mut next = 0;
        for context in self.contexts() {
            if context.index() != next {
                tokens.extend(Some(pound()));
                tokens.extend(quote!(#context));
            }
            next = context.index() + 1;
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
                    tokens.extend(Some(pound()));
                    tokens.extend(quote!(#{Literal::usize_unsuffixed(*n)} ;));
                }
            }
            let outlives = self.context_self_outlives(context);
            if !outlives.is_empty() {
                tokens.extend(quote!(Self :));
                for (i, lifetime) in outlives.iter().enumerate() {
                    if i > 0 {
                        tokens.extend(quote!(+));
                    }
                    tokens.extend(quote!(#lifetime));
                }
                tokens.extend(quote!(;));
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
                            @[#(for name in crates), { #{crate_ident(name)} }]
                        }
                        #item
                    }
                )
            });
        }
    }
}

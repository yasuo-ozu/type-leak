//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::context::{Context, ContextId, Enclosing, ModuleInfo, Scope, SelfRequirement};
use crate::encode::{complete_generics_args, param_as_arg, without_encoded_bounds, EncodeError};
use crate::graph::TypeKey;
use crate::util::{first_self_in_generics, generic_param_name, mentions_names, type_mentions_self};
use proc_macro2::{Literal, Punct, Spacing, Span, TokenStream};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use syn::ext::IdentExt;
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};

/// The entries of a [`Leaker`](crate::Leaker), made by [`Leaker::finish()`](crate::Leaker::finish).
///
/// It finds the leaked types in a copy and replaces them. It converts to and from tokens
/// ([`ToTokens`] and [`Parse`]), see [`Referrer::FORMAT_VERSION`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    pub(crate) leak_types: Vec<Type>,
    pub(crate) contexts: Vec<ContextId>,
    pub(crate) musts: Vec<bool>,
    pub(crate) external_crates: Vec<Vec<String>>,
    /// Not serialized.
    pub(crate) allow_ambiguous_crate_name: bool,
    pub(crate) context_map: BTreeMap<ContextId, Context>,
    /// Each is also a context of `context_map`.
    pub(crate) modules: BTreeMap<ContextId, ModuleInfo>,
    /// Keyed by tokens as in the Leaker, so a type in an invisible group is found too.
    pub(crate) map: HashMap<TypeKey, Vec<usize>>,
}

/// Where a visitor is, among the modules of the Leaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModulePos {
    /// In no module: the context has none, so only entries with no module match. A `mod` item is
    /// a module found with no module around it.
    Outside,
    In(ContextId),
    /// In a module no entry depends on: only entries with no module match.
    Elsewhere,
}

/// How [`Referrer::find()`] compares the context of a type with those of the entries.
#[derive(Clone, Copy)]
enum Matching<'a> {
    /// The context of the type, computed from the generics in scope and `self_ty`, is the entry's.
    Scope {
        self_ty: Option<&'a SelfRequirement>,
        /// `Self` means something else here, so no entry that names it matches.
        no_self: bool,
    },
    /// The scope is that of the parameters before a defaulted one: the entry must use only them.
    Default,
}

impl Referrer {
    /// `context_map` always has at least one context.
    pub(crate) fn new(
        entries: impl IntoIterator<Item = (ContextId, Type, bool, Vec<String>)>,
        mut context_map: BTreeMap<ContextId, Context>,
        modules: BTreeMap<ContextId, ModuleInfo>,
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
            modules,
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

    fn entry_context(&self, ix: usize) -> &Context {
        &self.context_map[&self.contexts[ix]]
    }

    fn get_context(&self, context: ContextId) -> &Context {
        self.context_map.get(&context).unwrap_or_else(|| {
            panic!(
                "the context {} is not one of the contexts of this Referrer",
                context
            )
        })
    }

    /// Returns the entry for `ty` that matches: the first of the best rank, then of the least cost.
    /// An entry of `module` comes before one with no module, and an entry of another module never
    /// matches ([Modules](crate#modules)).
    fn find(
        &self,
        matching: Matching,
        module: ModulePos,
        ty: &Type,
        scopes: &[Scope],
        enclosing: &Enclosing,
    ) -> Option<usize> {
        let no_self = matches!(matching, Matching::Scope { no_self: true, .. });
        let mut best: Option<((usize, usize), usize)> = None;
        for &ix in self.map.get(&TypeKey::new(ty))? {
            let entry = self.entry_context(ix);
            if no_self && (entry.self_ty().is_some() || type_mentions_self(&self.leak_types[ix])) {
                continue;
            }
            let rank = match (module, entry.module()) {
                (ModulePos::In(module), Some(entry)) if entry == module => 0,
                (_, Some(_)) => continue,
                (_, None) => 1,
            };
            if let Some(cost) = Self::matches(matching, ty, scopes, enclosing, entry) {
                let key = (rank, cost);
                if best.map_or(true, |(better, _)| key < better) {
                    best = Some((key, ix));
                }
            }
        }
        let ix = best?.1;
        Some(ix).filter(|&ix| !self.allow_ambiguous_crate_name || self.musts[ix])
    }

    /// Returns where a visited `mod ident` is, inside `module`: the module nested in it with that
    /// name ([Modules](crate#modules)). A module that the Leaker did not see, or one nested in such
    /// a module, is a module no entry depends on.
    fn module_in(&self, module: ModulePos, ident: &Ident) -> ModulePos {
        let parent = match module {
            ModulePos::Outside => None,
            ModulePos::In(parent) => Some(parent),
            ModulePos::Elsewhere => return ModulePos::Elsewhere,
        };
        self.modules
            .iter()
            .find(|(_, info)| {
                info.parent == parent
                    && matches!(info.path.as_ref().and_then(|path| path.last()), Some(last) if last == ident)
            })
            .map_or(ModulePos::Elsewhere, |(id, _)| ModulePos::In(*id))
    }

    /// Returns the cost of `entry` for `ty`, or `None` if it does not match. It is `0` if the
    /// context of `ty` in `scopes` is `entry`'s. The entries of a default have a choice: the
    /// fewer parameters the entry uses, the less its cost.
    fn matches(
        matching: Matching,
        ty: &Type,
        scopes: &[Scope],
        enclosing: &Enclosing,
        entry: &Context,
    ) -> Option<usize> {
        match matching {
            Matching::Scope { self_ty, .. } => {
                let context = Context::of(ty, scopes, self_ty, enclosing);
                let entry = entry.without_module();
                if context == entry {
                    Some(0)
                } else if context.is_entry_with_extra_self_predicates(&entry) {
                    Some(1)
                } else {
                    None
                }
            }
            Matching::Default => {
                if entry.self_ty().is_some() {
                    return None;
                }
                let root = param_names(scopes[0].params());
                let context = Context::of(ty, &scopes[..1], None, enclosing);
                let used = param_names(&context.generics().params);
                let entry_params = param_names(&entry.generics().params);
                (used.is_subset(&entry_params) && entry_params.is_subset(&root))
                    .then(|| entry_params.len())
            }
        }
    }

    /// Returns the generics, what `Self` means and the module of the scope of `context`.
    ///
    /// The module is the context itself if it is a module, else the module it was entered in.
    fn place(&self, context: ContextId) -> (Scope, Option<SelfRequirement>, ModulePos) {
        let found = self.get_context(context);
        let module = if found.is_module_scope() {
            ModulePos::In(context)
        } else {
            found.module().map_or(ModulePos::Outside, ModulePos::In)
        };
        (
            Scope::new(found.generics()),
            found.self_ty().cloned(),
            module,
        )
    }

    /// Chooses whether optional entries are left as written (`true`, the default) or replaced.
    ///
    /// Required entries are always replaced. Crates are matched by name only, so pass `false` if
    /// the Referrer's crate may have other dependencies than the Leaker's. The flag is not
    /// serialized. See [External crates](crate#external-crates).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_struct(&parse_quote!(struct S { a: ::foo::X }), true).unwrap();
    /// let referrer = leaker.finish();
    /// let ty: Type = parse_quote!(::foo::X);
    /// let replaced: Type = parse_quote!(Replaced);
    /// assert_eq!(referrer.expand(context, ty.clone(), |_, _| replaced.clone()), ty);
    /// let referrer = referrer.allow_ambiguous_crate_name(false);
    /// assert_eq!(referrer.expand(context, ty.clone(), |_, _| replaced.clone()), replaced);
    /// ```
    pub fn allow_ambiguous_crate_name(mut self, allow: bool) -> Self {
        self.allow_ambiguous_crate_name = allow;
        self
    }

    /// Returns the sorted, distinct names of the external crates that the `ix`-th entry names.
    ///
    /// A crate is named by the first segment of an absolute path in the type, its generics or its
    /// `Self`. `core` is not listed, and `r#` is stripped (`async` for `::r#async::X`).
    pub fn external_crates_require(&self, ix: usize) -> &[String] {
        &self.external_crates[ix]
    }

    /// Returns `false` if the `ix`-th entry is optional.
    ///
    /// An entry is optional if only its own path is an absolute path that names an external crate.
    /// [`Referrer::allow_ambiguous_crate_name()`] then decides whether it is replaced. Other
    /// entries are required and always replaced. Required entries have the smaller indices.
    /// See [External crates](crate#external-crates).
    pub fn must(&self, ix: usize) -> bool {
        self.musts[ix]
    }

    /// Returns `true` if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.leak_types.is_empty()
    }

    /// Returns the types of the entries, by index.
    ///
    /// A plain impl for any type is right only for an entry whose context has no generics, no `Self`
    /// and no module. Otherwise use [`Referrer::iter_with_generics()`] or
    /// [`Referrer::iter_with_self()`].
    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.leak_types.iter()
    }

    /// Returns the generics that the `ix`-th entry uses, as [`Referrer::context_generics()`] does.
    ///
    /// The Repeater impl encodes them in its trait arguments and declares
    /// [`Referrer::impl_generics()`]. See [Contexts](crate#contexts).
    pub fn generics(&self, ix: usize) -> &Generics {
        self.context_generics(self.contexts[ix])
    }

    /// Returns the context of the `ix`-th entry.
    ///
    /// Entries with the same generics, `Self` and module share it. See [Contexts](crate#contexts).
    pub fn context(&self, ix: usize) -> ContextId {
        self.contexts[ix]
    }

    /// Returns the contexts in ascending order, with the top-level ones that have no entry.
    ///
    /// The ids may have gaps. There is always at least one context. See
    /// [Contexts](crate#contexts).
    pub fn contexts(&self) -> impl Iterator<Item = ContextId> + '_ {
        self.context_map.keys().copied()
    }

    /// Returns the generics that the entries of `context` use.
    ///
    /// The Repeater impl encodes them in its trait arguments and declares
    /// [`Referrer::impl_generics()`]. A top-level context has only the bounds the scope writes, so
    /// do not encode it for every entry (E0309): encode the generics of each entry.
    /// See [Outlives bounds](crate#outlives-bounds).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_generics(&self, context: ContextId) -> &Generics {
        self.get_context(context).generics()
    }

    /// Returns the generics that a Repeater impl declares for the entries of `context`.
    ///
    /// These are [`Referrer::context_generics()`] without the lifetime bounds that the trait
    /// arguments encode. Other bounds and where predicates stay. For other generics use
    /// [`without_encoded_bounds()`](crate::without_encoded_bounds).
    /// See [Outlives bounds](crate#outlives-bounds).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn impl_generics(&self, context: ContextId) -> Generics {
        without_encoded_bounds(self.context_generics(context))
    }

    /// Returns what `Self` means in `context`.
    ///
    /// For an entry's context it is `Some` only if the types name `Self`. The Repeater impl is for
    /// any type (`None`), for the implementors of the trait ([`SelfRequirement::Bound`]) or for
    /// the type ([`SelfRequirement::Type`]). For the top-level context of a scope it is what the
    /// scope declares, but only if an entry of the scope names `Self`: nothing else depends on it.
    /// It is `None` for a scope that declares nothing (`enter_generics()`, a module). So a
    /// top-level id tells the Referrer what `Self` is where it matters. See
    /// [Self rules](crate#self-rules).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_self(&self, context: ContextId) -> Option<&SelfRequirement> {
        self.get_context(context).self_ty()
    }

    /// Returns the lifetime parameters `'a` of the context for which its types need `Self: 'a`.
    ///
    /// They are sorted, and empty if the types do not name `Self`. [`Referrer::context_generics()`]
    /// does not list them. The Repeater impl must not declare them: encode them with the
    /// `*_with_self()` encoders. See [Outlives bounds](crate#outlives-bounds).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_self_outlives(&self, context: ContextId) -> &[Lifetime] {
        self.get_context(context).self_outlives()
    }

    /// Returns the module where the types of `context` are written, as its `ContextId`.
    ///
    /// For an entry's context it is `Some` only for types with a relative path. Emit the Repeater
    /// impls of such types inside that module. For the top-level context of a scope entered in a
    /// module it is that module. It is `None` for a module context, which is the module itself.
    /// See [Modules](crate#modules).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn context_module(&self, context: ContextId) -> Option<ContextId> {
        self.get_context(context).module()
    }

    /// Returns the path of `module` from its root: `[]` for the module of `intern_mod()`.
    ///
    /// A module found with no module around it is a root too, and its path starts with its own
    /// name. It is `None` for a module entered by `enter_mod()` and for an id that is not a module
    /// of this Referrer. See [Modules](crate#modules).
    pub fn module_path(&self, module: ContextId) -> Option<&[Ident]> {
        self.modules.get(&module)?.path.as_deref()
    }

    /// Returns the hint given to [`Leaker::enter_mod()`](crate::Leaker::enter_mod) for `module`.
    ///
    /// It is `None` for other modules and for an id that is not a module of this Referrer.
    pub fn module_hint(&self, module: ContextId) -> Option<&str> {
        self.modules.get(&module)?.hint.as_deref()
    }

    /// Returns `true` if a leaked type, or the generics of an entry, names `Self`.
    ///
    /// `Self::Item` and `<Self as ::p::Tr>::X` count, `::p::Self` does not. A top-level context
    /// with no entry is not looked at. Then the Repeater impl is for what `Self` is
    /// ([`Referrer::context_self()`]), or use [`Referrer::iter_with_self()`] for a marker type.
    /// See [Self rules](crate#self-rules).
    pub fn has_self(&self) -> bool {
        self.leak_types.iter().any(type_mentions_self)
            || self.contexts.iter().any(|context| {
                first_self_in_generics(self.context_map[context].generics()).is_some()
            })
    }

    /// Like [`Referrer::iter_with_generics()`], with `Self` replaced by `self_ty`, usually `__S`.
    ///
    /// Use it for an entry whose [`context_self()`](Referrer::context_self) is `Bound`, or for a
    /// marker type. `Self::A::B` becomes `<self_ty>::A::B`. Qualified paths and macro tokens are
    /// rewritten too. Only the returned copies change: `visitor()` still finds the types by
    /// their original tokens. See [Self rules](crate#self-rules).
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

    /// Fills in the omitted arguments of the Leaker's trait, whose parameters are `params`.
    ///
    /// Returns one argument for each parameter, matched as
    /// [`encode_generics_args_to_ty_with()`](crate::encode_generics_args_to_ty_with) does. Fails
    /// with [`EncodeError`] if there are too many arguments, or one without a default is missing.
    /// A default may name leaked types. `type_fn(leaked, index, args)` replaces each one, where
    /// `args` are the complete arguments before the defaulted parameter. Return a path through the
    /// Repeater, for example built with [`Referrer::generic_args()`]. It is not substituted again.
    /// A default is written in the trait's own scope, so only the entries of no module are looked
    /// up.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use template_quote::quote;
    /// let tr: ItemTrait = parse_quote!(trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> U; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&tr, true).unwrap();
    /// let referrer = leaker.finish();
    /// // For `impl Tr<u8> for ..`, omitting `U`.
    /// let args: AngleBracketedGenericArguments = parse_quote!(<u8>);
    /// let completed = referrer
    ///     .complete_args(&args.args, &tr.generics.params, |_, ix, _| parse_quote!(Leaked<#ix>))
    ///     .unwrap();
    /// let expected: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(Leaked<0usize>)];
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
                // The parameters before the defaulted one are the Leaker's generics for its default.
                let root = Generics {
                    params: params.iter().take(index).cloned().collect(),
                    ..Default::default()
                };
                // Placeholders keep the substitution from touching the replaced paths.
                let mut leaked = Vec::new();
                let ty = self.expand_with(
                    Matching::Default,
                    ModulePos::Outside,
                    &mut vec![Scope::new(&root)],
                    None,
                    default.clone(),
                    |ty, ix| {
                        leaked.push((ty, ix));
                        let placeholder = Ident::new(
                            &format!("{}{}", PLACEHOLDER, leaked.len() - 1),
                            Span::call_site(),
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

    /// Returns the arguments for the generics of the `ix`-th entry, as written in the copy.
    ///
    /// `completed_args` are the arguments of the Leaker's parameters `root_params`
    /// ([`Referrer::complete_args()`]). A generic named like one of them takes its argument. Any
    /// other generic is passed by its own name, so the copy must declare it. `Self` is not an
    /// argument. Fails with [`EncodeError::MissingArgument`] if `completed_args` are too few.
    pub fn generic_args(
        &self,
        ix: usize,
        root_params: &Punctuated<GenericParam, Token![,]>,
        completed_args: &[GenericArgument],
    ) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
        self.context_generic_args(self.contexts[ix], root_params, completed_args)
    }

    /// Like [`Referrer::generic_args()`], for the generics of any `context`.
    ///
    /// The arguments are matched with `root_params` by name, because the generics of a context are
    /// sorted. Use it with the top-level context for the arguments of the trait itself. Pass the
    /// `ContextId` along with the `Referrer` (written and parsed as an integer), because the
    /// Referrer cannot tell which context is the top-level one.
    ///
    /// # Panics
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
                match root_params
                    .iter()
                    .position(|root| generic_param_name(root) == name)
                {
                    Some(pos) => completed_args
                        .get(pos)
                        .cloned()
                        .ok_or_else(|| EncodeError::MissingArgument(Box::new(param.clone()))),
                    None => Ok(param_as_arg(param)),
                }
            })
            .collect()
    }

    /// Returns the types of the entries with the generics they use ([`Referrer::generics()`]).
    pub fn iter_with_generics(&self) -> impl Iterator<Item = (&Type, &Generics)> {
        self.leak_types
            .iter()
            .zip(&self.contexts)
            .map(move |(ty, context)| (ty, self.context_map[context].generics()))
    }

    /// Converts this Referrer into a [`Visitor`] that rewrites the leaked types in a copy.
    ///
    /// `context` is the scope of the copy: the top-level [`ContextId`] that the `intern_*` or
    /// `enter_*` method of the Leaker returned for the item the copy is made from. Pass it along
    /// with the Referrer (it is written and parsed as an integer). The visitor takes from it the
    /// generics in scope, what `Self` means ([`Referrer::context_self()`]) and the module
    /// ([`Referrer::context_module()`], or `context` itself if it is a module). The visited syntax
    /// is inside that scope. See [Modules](crate#modules).
    ///
    /// `type_fn` gets each leaked type and its index, and returns the replacement. A type is
    /// replaced if its context, computed from the generics in scope, is that of an entry: the
    /// generics of the scope and those the visited syntax declares are compared exactly, and
    /// others are not generics. So a generic that the visited syntax declares again, such as the
    /// `T` of `impl<T: Clone> MyTrait<T> for ..`, must be declared as the entry has it, unless you
    /// call [`Visitor::with_outermost_generics_as_leakers()`]. An entry which has, besides, bounds
    /// on `Self` alone (`where Self: Sized`) is found too, if no entry has exactly the context:
    /// a copy where `Self` is a type parameter cannot write them. Visit the whole item, not a
    /// type cut out of another.
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn visitor<F: FnMut(Type, usize) -> Type>(
        self,
        context: ContextId,
        type_fn: F,
    ) -> Visitor<F> {
        let (scope, self_requirement, module) = self.place(context);
        let in_module = self.get_context(context).is_module_scope();
        Visitor {
            referrer: self,
            type_fn: Rc::new(RefCell::new(type_fn)),
            self_requirement,
            scopes: vec![scope],
            outermost_as_leakers: false,
            self_ty: None,
            // The items of a module are nested items, which declare their own `Self`.
            item_depth: usize::from(in_module),
            own_self: false,
            module,
        }
    }

    /// Replaces the leaked types in `ty` with what `type_fn(leaked, index)` returns.
    ///
    /// `ty` is inside the scope `context`, as for [`Referrer::visitor()`]. Pass the whole type, not
    /// one cut out of a larger type: see [Outlives bounds](crate#outlives-bounds).
    ///
    /// # Panics
    /// Panics if `context` is not one of the [`Referrer::contexts()`].
    pub fn expand(
        &self,
        context: ContextId,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        let (scope, self_requirement, module) = self.place(context);
        self.expand_with(
            Matching::Scope {
                self_ty: self_requirement.as_ref(),
                no_self: false,
            },
            module,
            &mut vec![scope],
            None,
            ty,
            type_fn,
        )
    }

    fn expand_with(
        &self,
        matching: Matching,
        module: ModulePos,
        scopes: &mut Vec<Scope>,
        self_ty: Option<&Type>,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        use syn::fold::Fold;
        struct Folder<'a, 'b, F> {
            referrer: &'a Referrer,
            matching: Matching<'b>,
            module: ModulePos,
            scopes: &'a mut Vec<Scope>,
            replace_self: Option<ReplaceSelf<'a>>,
            enclosing: Enclosing,
            type_fn: F,
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
                // A type in an expression is not a component of the types around it.
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
                    Some(mut replace) => replace.fold_macro(i),
                    None => syn::fold::fold_macro(self, i),
                }
            }
            fn fold_type_bare_fn(&mut self, i: TypeBareFn) -> TypeBareFn {
                let depth = self.scopes.len();
                self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
                let i = syn::fold::fold_type_bare_fn(self, i);
                self.scopes.truncate(depth);
                i
            }
            fn fold_trait_bound(&mut self, i: TraitBound) -> TraitBound {
                let depth = self.scopes.len();
                self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
                let i = syn::fold::fold_trait_bound(self, i);
                self.scopes.truncate(depth);
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

fn param_names<'a>(params: impl IntoIterator<Item = &'a GenericParam>) -> HashSet<String> {
    params.into_iter().map(generic_param_name).collect()
}

#[derive(Clone, Copy)]
struct ReplaceSelf<'a> {
    self_ty: &'a Type,
}

impl<'a> ReplaceSelf<'a> {
    fn replace_tokens(&self, tokens: TokenStream) -> TokenStream {
        use proc_macro2::{Group, TokenTree};
        tokens
            .into_iter()
            .map(|tt| match tt {
                TokenTree::Ident(ident) if ident == "Self" => self.self_ty.to_token_stream(),
                TokenTree::Group(group) => {
                    let mut new =
                        Group::new(group.delimiter(), self.replace_tokens(group.stream()));
                    new.set_span(group.span());
                    TokenTree::Group(new).into()
                }
                tt => tt.into(),
            })
            .collect()
    }

    // `Some(Some(rest))` for `Self::A::B`, `Some(None)` for `Self` alone.
    fn strip_self(path: &Path) -> Option<Option<Path>> {
        let first = path.segments.first()?;
        if path.leading_colon.is_some() || first.ident != "Self" || !first.arguments.is_none() {
            return None;
        }
        // The `::` after the qualified self type (see `qself()`).
        Some((path.segments.len() > 1).then(|| Path {
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

    fn fold_type_with(&self, folder: &mut impl syn::fold::Fold, ty: Type) -> Type {
        let stripped = match &ty {
            Type::Path(TypePath { qself: None, path }) => Self::strip_self(path),
            _ => None,
        };
        match (ty, stripped) {
            (_, Some(None)) => self.self_ty.clone(),
            (_, Some(Some(rest))) => Type::Path(TypePath {
                qself: Some(self.qself()),
                path: folder.fold_path(rest),
            }),
            (Type::Verbatim(tokens), _) => Type::Verbatim(self.replace_tokens(tokens)),
            (ty, _) => syn::fold::fold_type(folder, ty),
        }
    }

    /// `Self` alone in an expression is a value, so only `Self::A` is replaced.
    fn fold_expr_with(&self, folder: &mut impl syn::fold::Fold, expr: Expr) -> Expr {
        let stripped = match &expr {
            Expr::Path(ExprPath {
                qself: None, path, ..
            }) => Self::strip_self(path),
            _ => None,
        };
        match (expr, stripped) {
            (Expr::Path(path), Some(Some(rest))) => Expr::Path(ExprPath {
                qself: Some(self.qself()),
                path: folder.fold_path(rest),
                ..path
            }),
            (expr, _) => syn::fold::fold_expr(folder, expr),
        }
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
        Macro {
            tokens: self.replace_tokens(i.tokens),
            ..i
        }
    }
}

// Prefix of the placeholder names in `Referrer::complete_args()`.
const PLACEHOLDER: &str = "__TypeLeakPlaceholder";

struct Resolve<'a, F> {
    leaked: Vec<(Type, usize)>,
    completed: &'a [GenericArgument],
    type_fn: &'a mut F,
}

impl<'a, F: FnMut(Type, usize, &[GenericArgument]) -> Type> syn::fold::Fold for Resolve<'a, F> {
    fn fold_type(&mut self, ty: Type) -> Type {
        let leaked = match &ty {
            Type::Path(TypePath { qself: None, path }) => path
                .get_ident()
                .and_then(|ident| {
                    ident
                        .to_string()
                        .strip_prefix(PLACEHOLDER)?
                        .parse::<usize>()
                        .ok()
                })
                .and_then(|number| self.leaked.get(number).cloned()),
            _ => None,
        };
        match leaked {
            Some((leaked, ix)) => (self.type_fn)(leaked, ix, self.completed),
            None => syn::fold::fold_type(self, ty),
        }
    }
}

/// Rewrites a copy by replacing its leaked types. Get it with [`Referrer::visitor()`].
///
/// It implements `syn::visit_mut::VisitMut` and tracks the generics in scope: those of the
/// context it was made for, then those the visited syntax declares. A type is replaced only if an
/// entry has the context it has there. So `T: Clone` of a visited `impl<T: Clone>` does not match
/// an entry with `T` alone, unless you call [`Visitor::with_outermost_generics_as_leakers()`].
///
/// Visit a whole type or item. A type cut out of another is looked up with nothing around it
/// ([Outlives bounds](crate#outlives-bounds)). The visited syntax is inside the module of the
/// context, so a `mod` item is a module nested in it, found by name, and the items of a module
/// are visited as items of it ([Modules](crate#modules)). Items and imports declared in a block
/// hide the leaked types that name them there. Those declared in a module, or in a `File`, hide
/// the entries of no module that name them.
#[derive(Debug)]
pub struct Visitor<F> {
    referrer: Referrer,
    type_fn: Rc<RefCell<F>>,
    /// What `Self` means in the context.
    self_requirement: Option<SelfRequirement>,
    /// The generics in scope, innermost last. The first are those of the context.
    scopes: Vec<Scope>,
    outermost_as_leakers: bool,
    self_ty: Option<Type>,
    item_depth: usize,
    /// In a nested item that defines its own `Self`.
    own_self: bool,
    module: ModulePos,
}

impl<F> Clone for Visitor<F> {
    fn clone(&self) -> Self {
        Self {
            referrer: self.referrer.clone(),
            type_fn: self.type_fn.clone(),
            self_requirement: self.self_requirement.clone(),
            scopes: self.scopes.clone(),
            outermost_as_leakers: self.outermost_as_leakers,
            self_ty: self.self_ty.clone(),
            item_depth: self.item_depth,
            own_self: self.own_self,
            module: self.module,
        }
    }
}

impl<F: FnMut(Type, usize) -> Type> Visitor<F> {
    /// Treats the generics of the outermost item visited as those of the context.
    ///
    /// Use it if the copy declares the generics of the scope again, with other bounds. A type that
    /// names them is replaced if it was leaked, whatever bounds they have in the copy. Example:
    /// `Wrap<T>` in `impl<T: Clone> MyTrait<T> for W { fn f(&self) -> Wrap<T>; }`, for the context
    /// of `trait MyTrait<T>`. The outermost item is a struct, enum, union, trait, type alias,
    /// function, impl or trait alias, or the signature if only that is visited. Deeper generics
    /// stay declared. By default they are declared too, so they must match the entry's, except
    /// that an outermost item which declares exactly the generics of the context (the copy of the
    /// item itself) does not declare them again. Either way, `T::A<X>` is an associated type path,
    /// as in the Leaker.
    pub fn with_outermost_generics_as_leakers(mut self) -> Self {
        self.outermost_as_leakers = true;
        self
    }

    /// Replaces `Self` by `self_ty` in the copy, as [`Referrer::iter_with_self()`] does.
    ///
    /// Use it for a Repeater implemented for a marker type ([Self rules](crate#self-rules)).
    /// Leaked types are still found by their original tokens, and `type_fn` gets them with `Self`
    /// replaced. `Self` is replaced in the other types too, but not in what `type_fn` returns, nor
    /// in receivers (`&self`) or expressions. In a nested struct, enum, union, trait or impl, `Self`
    /// means something else: it is not replaced, and no leaked type that names it is found.
    /// The outermost item, or the signature if only that is visited, is not nested.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// let tr: ItemTrait = parse_quote!(trait Tr { fn f(&self, x: Rel<Self>) -> Self::Out; });
    /// let mut leaker = Leaker::new();
    /// let context = leaker.intern_trait(&tr, true).unwrap();
    /// let mut sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>) -> Self::Out);
    /// leaker
    ///     .finish()
    ///     .visitor(context, |_, ix| parse_quote!(<Marker<__S> as Repeater<#ix>>::Type))
    ///     .with_self(parse_quote!(__S))
    ///     .visit_signature_mut(&mut sig);
    /// let expected: Signature =
    ///     parse_quote!(fn f(&self, x: <Marker<__S> as Repeater<0usize>>::Type) -> <__S>::Out);
    /// assert_eq!(sig, expected);
    /// ```
    pub fn with_self(mut self, self_ty: Type) -> Self {
        self.self_ty = Some(self_ty);
        self
    }

    // For a module, only entries with no module are hidden: the others are keyed by their module.
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
        if names.is_empty() {
            return;
        }
        let context_map = &self.referrer.context_map;
        let contexts = &self.referrer.contexts;
        self.referrer.map.retain(|key, entries| {
            if mentions_names(&key.ty, names.clone(), Vec::new()) {
                entries.retain(|&ix| module && context_map[&contexts[ix]].module().is_some());
            }
            !entries.is_empty()
        });
    }

    fn visit_attrs(&mut self, attrs: &mut [Attribute]) {
        for attr in attrs {
            self.visit_attribute_mut(attr);
        }
    }

    /// `generics` are those of the context, in any order and with the bounds anywhere.
    fn is_context_generics(&self, generics: &Generics) -> bool {
        let canonical = Context::new(generics.clone(), None);
        self.scopes.first() == Some(&Scope::new(canonical.generics()))
    }

    /// Visits the attributes, then returns a nested visitor that has visited `generics` and has
    /// them in scope. `item` is for an item, which may define its own `Self`.
    fn enter(&mut self, attrs: &mut [Attribute], generics: &mut Generics, item: bool) -> Self {
        self.visit_attrs(attrs);
        let mut visitor = self.clone();
        visitor.item_depth += 1;
        // Only an item nested in the visited syntax defines its own `Self`.
        visitor.own_self |= item && self.item_depth > 0;
        // Declare before visiting: every parameter is in scope of each other's bounds. The
        // outermost item is the scope itself if it declares the generics of the context as they
        // are, or if `with_outermost_generics_as_leakers()` says so: they are not declared again.
        let outermost = item && self.item_depth == 0;
        if !(outermost && (self.outermost_as_leakers || self.is_context_generics(generics))) {
            visitor.scopes.push(Scope::new(generics));
        }
        visitor.visit_generics_mut(generics);
        visitor
    }

    fn enter_signature(
        &mut self,
        attrs: &mut [Attribute],
        sig: &mut Signature,
        item: bool,
    ) -> Self {
        let mut visitor = self.enter(attrs, &mut sig.generics, item);
        for input in sig.inputs.iter_mut() {
            visitor.visit_fn_arg_mut(input);
        }
        visitor.visit_return_type_mut(&mut sig.output);
        visitor
    }
}

impl<F: FnMut(Type, usize) -> Type> VisitMut for Visitor<F> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        let (self_ty, no_self) = (self.self_requirement.as_ref(), self.own_self);
        *i = self.referrer.expand_with(
            Matching::Scope { self_ty, no_self },
            self.module,
            &mut self.scopes,
            self.self_ty.as_ref().filter(|_| !no_self),
            i.clone(),
            &mut *self.type_fn.borrow_mut(),
        );
    }
    fn visit_receiver_mut(&mut self, i: &mut Receiver) {
        if self.self_ty.is_none() || self.own_self {
            return syn::visit_mut::visit_receiver_mut(self, i);
        }
        // The receiver type stays: rustc does not accept an associated type path there.
        self.visit_attrs(&mut i.attrs);
    }
    fn visit_predicate_type_mut(&mut self, i: &mut PredicateType) {
        let depth = self.scopes.len();
        self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
        syn::visit_mut::visit_predicate_type_mut(self, i);
        self.scopes.truncate(depth);
    }
    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        let depth = self.scopes.len();
        self.scopes.extend(i.lifetimes.as_ref().map(Scope::binder));
        syn::visit_mut::visit_trait_bound_mut(self, i);
        self.scopes.truncate(depth);
    }
    fn visit_const_param_mut(&mut self, i: &mut ConstParam) {
        // The type of a const parameter cannot be an associated type path, so it stays.
        self.visit_attrs(&mut i.attrs);
        if let Some(expr) = &mut i.default {
            self.visit_expr_mut(expr);
        }
    }
    fn visit_item_struct_mut(&mut self, i: &mut ItemStruct) {
        self.enter(&mut i.attrs, &mut i.generics, true)
            .visit_fields_mut(&mut i.fields);
    }
    fn visit_item_enum_mut(&mut self, i: &mut ItemEnum) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, true);
        for variant in i.variants.iter_mut() {
            visitor.visit_variant_mut(variant);
        }
    }
    fn visit_item_trait_mut(&mut self, i: &mut ItemTrait) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, true);
        for supertrait in i.supertraits.iter_mut() {
            visitor.visit_type_param_bound_mut(supertrait);
        }
        for item in i.items.iter_mut() {
            visitor.visit_trait_item_mut(item);
        }
    }
    fn visit_item_trait_alias_mut(&mut self, i: &mut ItemTraitAlias) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, true);
        for bound in i.bounds.iter_mut() {
            visitor.visit_type_param_bound_mut(bound);
        }
    }
    fn visit_item_union_mut(&mut self, i: &mut ItemUnion) {
        self.enter(&mut i.attrs, &mut i.generics, true)
            .visit_fields_named_mut(&mut i.fields);
    }
    fn visit_item_type_mut(&mut self, i: &mut ItemType) {
        self.enter(&mut i.attrs, &mut i.generics, true)
            .visit_type_mut(&mut i.ty);
    }
    fn visit_signature_mut(&mut self, i: &mut Signature) {
        self.enter_signature(&mut [], i, true);
    }
    fn visit_item_impl_mut(&mut self, i: &mut ItemImpl) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, true);
        if let Some((_, path, _)) = &mut i.trait_ {
            visitor.visit_path_mut(path);
        }
        visitor.visit_type_mut(&mut i.self_ty);
        for item in i.items.iter_mut() {
            visitor.visit_impl_item_mut(item);
        }
    }
    fn visit_impl_item_fn_mut(&mut self, i: &mut ImplItemFn) {
        let mut visitor = self.enter_signature(&mut i.attrs, &mut i.sig, false);
        visitor.visit_block_mut(&mut i.block);
    }
    fn visit_impl_item_type_mut(&mut self, i: &mut ImplItemType) {
        self.enter(&mut i.attrs, &mut i.generics, false)
            .visit_type_mut(&mut i.ty);
    }
    fn visit_item_mod_mut(&mut self, i: &mut ItemMod) {
        self.visit_attrs(&mut i.attrs);
        if let Some((_, items)) = &mut i.content {
            let mut visitor = self.clone();
            visitor.module = self.referrer.module_in(self.module, &i.ident);
            // A module has no generics: it only makes its items nested.
            visitor.item_depth += 1;
            visitor.shadow_items(items.iter(), true);
            for item in items.iter_mut() {
                visitor.visit_item_mut(item);
            }
        }
    }
    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        let mut visitor = self.enter_signature(&mut i.attrs, &mut i.sig, true);
        visitor.visit_block_mut(i.block.as_mut());
    }
    fn visit_trait_item_fn_mut(&mut self, i: &mut TraitItemFn) {
        let mut visitor = self.enter_signature(&mut i.attrs, &mut i.sig, false);
        if let Some(block) = &mut i.default {
            visitor.visit_block_mut(block);
        }
    }
    fn visit_trait_item_type_mut(&mut self, i: &mut TraitItemType) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, false);
        for bound in i.bounds.iter_mut() {
            visitor.visit_type_param_bound_mut(bound);
        }
        if let Some((_, ty)) = &mut i.default {
            visitor.visit_type_mut(ty);
        }
    }
    fn visit_trait_item_const_mut(&mut self, i: &mut TraitItemConst) {
        let mut visitor = self.enter(&mut i.attrs, &mut i.generics, false);
        visitor.visit_type_mut(&mut i.ty);
        if let Some((_, expr)) = &mut i.default {
            visitor.visit_expr_mut(expr);
        }
    }
    fn visit_file_mut(&mut self, i: &mut File) {
        // The items of a file are declared together, as those of a module.
        let mut visitor = self.clone();
        visitor.shadow_items(i.items.iter(), true);
        syn::visit_mut::visit_file_mut(&mut visitor, i)
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

const FORMAT_TAG_PREFIX: &str = "type-leak-format-";

impl Referrer {
    /// The version of the serialized form, written as the tag `"type-leak-format-N"`.
    ///
    /// Parsing rejects another tag, or none. Formats are not compatible, so both sides must use the
    /// same version of `type-leak`.
    // Bump by hand when an old serialized Referrer would behave differently: the syntax, what a
    // context holds, how enclosing lifetimes pass on, which paths are associated type paths, or how
    // an entry is found. Docs, errors and encoders need no bump.
    // 1: first tagged, contexts carry the bounds enclosing types imply. 2: modules are contexts
    // (`mod 3 ;`, `mod { .. } ;`), no `Self = # 0 ;`. 3: the top-level context of a scope keeps
    // what `Self` means and the module the scope is in, for `Referrer::visitor()`.
    pub const FORMAT_VERSION: u32 = 3;
}

/// Parses a [`Referrer`] written by its [`ToTokens`] implementation.
///
/// Entries keep their indices, and the tokens after the Referrer are not consumed. Fails if the
/// format tag is missing or differs ([`Referrer::FORMAT_VERSION`]), if a module is used but not
/// declared, or if module parents form a cycle.
impl Parse for Referrer {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let span = input.span();
        let tag = input.parse::<Option<LitStr>>()?;
        let value = tag.as_ref().map(LitStr::value);
        let expected = format!("{}{}", FORMAT_TAG_PREFIX, Self::FORMAT_VERSION);
        if value.as_deref() != Some(expected.as_str()) {
            let found = match &value {
                None => "no format tag".to_string(),
                Some(value) => match value.strip_prefix(FORMAT_TAG_PREFIX) {
                    Some(number) if !number.is_empty() => format!("format {}", number),
                    _ => format!("an unrecognized tag {:?}", value),
                },
            };
            return Err(Error::new(
                tag.map_or(span, |tag| tag.span()),
                format!(
                    "this Referrer was serialized by an incompatible type-leak ({}, expected {}): \
                     the macros which emit and parse it must use the same version of type-leak",
                    found,
                    Self::FORMAT_VERSION
                ),
            ));
        }
        let n_contexts = if input.peek(LitInt) {
            input.parse::<LitInt>()?.base10_parse::<usize>()?
        } else {
            1
        };
        let mut entries = Vec::new();
        let mut context_map = BTreeMap::new();
        let mut modules = BTreeMap::new();
        // Checked when all the contexts are known.
        let mut module_refs: Vec<(Span, ContextId)> = Vec::new();
        let mut parents = BTreeMap::new();
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
            let in_module = |span, what: &str| {
                let message = format!(
                    "the context {} is a module (`mod {{ .. }}`), {}",
                    context, what
                );
                Error::new(span, message)
            };
            let generics_span = input.span();
            let has_generics = input.peek(Token![<]) || input.peek(Token![where]);
            let mut generics = Generics::default();
            if has_generics {
                generics = input.parse()?;
                generics.where_clause = input.parse()?;
                input.parse::<Token![;]>()?;
            }
            // `Self: 'a + 'b ;` is told from `Self: path ;` by the lifetime.
            let mut self_ty = None;
            let mut self_outlives = Vec::new();
            let self_span = input.span();
            while input.peek(Token![Self]) {
                input.parse::<Token![Self]>()?;
                if input.parse::<Option<Token![:]>>()?.is_none() {
                    input.parse::<Token![=]>()?;
                    self_ty = Some(SelfRequirement::Type(input.parse()?));
                } else if input.peek(Lifetime) {
                    self_outlives =
                        Punctuated::<Lifetime, Token![+]>::parse_separated_nonempty(input)?
                            .into_iter()
                            .collect();
                } else {
                    self_ty = Some(SelfRequirement::Bound(input.parse()?));
                }
                input.parse::<Token![;]>()?;
            }
            let (mut module, mut module_info) = (None, None);
            if input.parse::<Option<Token![mod]>>()?.is_some() {
                if input.peek(token::Brace) {
                    module_info = Some(parse_module_info(input, context, &mut parents)?);
                } else {
                    let span = input.span();
                    let referred: ContextId = input.parse()?;
                    module_refs.push((span, referred));
                    module = Some(referred);
                }
                input.parse::<Token![;]>()?;
            }
            let entries_span = input.span();
            let parsed = match module_info {
                Some(info) => {
                    if has_generics {
                        return Err(in_module(generics_span, "which has no generics"));
                    }
                    if self_ty.is_some() || !self_outlives.is_empty() {
                        return Err(in_module(self_span, "where `Self` has no meaning"));
                    }
                    modules.insert(context, info);
                    Context::module_scope(context.index())
                }
                None => Context::new(generics, self_ty)
                    .with_self_outlives(self_outlives)
                    .with_module(module),
            };
            context_map.insert(context, parsed);
            let content: parse::ParseBuffer<'_>;
            parenthesized!(content in input);
            let group = Punctuated::<_, Token![,]>::parse_terminated_with(&content, |input| {
                let must = input.parse::<Option<Token![?]>>()?.is_none();
                let mut crates = Vec::new();
                if input.parse::<Option<Token![@]>>()?.is_some() {
                    let list: parse::ParseBuffer<'_>;
                    bracketed!(list in input);
                    crates = Punctuated::<Ident, Token![,]>::parse_terminated(&list)?
                        .into_iter()
                        .map(|ident| ident.unraw().to_string())
                        .collect();
                }
                Ok((input.parse::<Type>()?, must, crates))
            })?;
            if modules.contains_key(&context) && !group.is_empty() {
                return Err(in_module(entries_span, "which has no leaked type"));
            }
            entries.extend(
                group
                    .into_iter()
                    .map(|(ty, must, crates)| (context, ty, must, crates)),
            );
        }
        let unknown = |id: ContextId| {
            if context_map.contains_key(&id) {
                format!(
                    "the context {} is not a module (it is not declared as `mod {{ .. }}`)",
                    id
                )
            } else {
                format!("there is no module {}", id)
            }
        };
        for (span, referred) in module_refs {
            if !modules.contains_key(&referred) {
                return Err(Error::new(span, unknown(referred)));
            }
        }
        for (module, (span, parent)) in &parents {
            if parent == module {
                return Err(Error::new(
                    *span,
                    format!("the module {} is its own parent", module),
                ));
            }
            if !modules.contains_key(parent) {
                return Err(Error::new(
                    *span,
                    format!("the parent of the module {}: {}", module, unknown(*parent)),
                ));
            }
        }
        for start in parents.keys() {
            let chain = std::iter::successors(Some(*start), |at| {
                parents.get(at).map(|(_, parent)| *parent)
            });
            if chain.skip(1).take(parents.len()).any(|at| at == *start) {
                return Err(Error::new(
                    parents[start].0,
                    format!(
                        "the module {} is nested in itself (its parents form a cycle)",
                        start
                    ),
                ));
            }
        }
        // The required entries come first (`Referrer::must()`).
        entries.sort_by_key(|(_, _, must, _)| !must);
        Ok(Self::new(entries, context_map, modules))
    }
}

/// Each key may appear once. The span and id of the parent go to `parents`, to be checked when
/// all the contexts are known.
fn parse_module_info(
    input: parse::ParseStream,
    context: ContextId,
    parents: &mut BTreeMap<ContextId, (Span, ContextId)>,
) -> Result<ModuleInfo> {
    let content: parse::ParseBuffer<'_>;
    braced!(content in input);
    let mut info = ModuleInfo {
        path: None,
        hint: None,
        parent: None,
    };
    let mut seen: Vec<String> = Vec::new();
    while !content.is_empty() {
        let key: Ident = content.parse()?;
        let name = key.to_string();
        if seen.contains(&name) {
            return Err(Error::new(
                key.span(),
                format!("`{}` is written twice in the information of a module", name),
            ));
        }
        content.parse::<Token![=]>()?;
        match name.as_str() {
            "path" => {
                let path: parse::ParseBuffer<'_>;
                bracketed!(path in content);
                info.path = Some(
                    Punctuated::<Ident, Token![,]>::parse_terminated(&path)?
                        .into_iter()
                        .collect(),
                );
            }
            "hint" => info.hint = Some(content.parse::<LitStr>()?.value()),
            "parent" => {
                let span = content.span();
                let id: ContextId = content.parse()?;
                info.parent = Some(id);
                parents.insert(context, (span, id));
            }
            _ => {
                return Err(Error::new(
                    key.span(),
                    "expected `path`, `hint` or `parent`",
                ))
            }
        }
        seen.push(name);
        content.parse::<Option<Token![,]>>()?;
    }
    Ok(info)
}

/// `r#name` if `name` is a keyword, so that it parses back as an identifier.
fn crate_ident(name: &str) -> Ident {
    if parse_str::<Ident>(name).is_ok() {
        Ident::new(name, Span::call_site())
    } else {
        Ident::new_raw(name, Span::call_site())
    }
}

/// Writes the [`Referrer`] as tokens, to be read back by its [`Parse`] implementation.
///
/// The tokens start with the format tag ([`Referrer::FORMAT_VERSION`]). Treat the rest as opaque.
impl ToTokens for Referrer {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        // Grammar: tag [count] then one group per context in ascending id order (at least one):
        // [# id] [<generics> where .. ;] [Self: path ; | Self = Type ;] [Self: 'a + 'b ;]
        // [mod 3 ; | mod { path = [a], hint = "h", parent = 1 } ;] ( [?] [@[crate, ..]] Type, .. )
        // `# id` only after a gap in the ids. `()` if no entry. `?`: optional. `@[..]`: crates named.
        let tag = format!("{}{}", FORMAT_TAG_PREFIX, Self::FORMAT_VERSION);
        let n_contexts = self.context_map.len();
        tokens.extend(quote! {
            #tag
            // The group count lets other tokens follow the Referrer.
            #(if n_contexts > 1) { #{Literal::usize_unsuffixed(n_contexts)} }
        });
        let mut next = 0;
        for context in self.contexts() {
            let generics = self.context_generics(context);
            let outlives = self.context_self_outlives(context);
            let entries = (0..self.leak_types.len()).filter(|&ix| self.contexts[ix] == context);
            tokens.extend(quote! {
                // `#` cannot be written in `quote!`, where it interpolates.
                #(if context.index() != next) { #{Punct::new('#', Spacing::Alone)} #context }
                #(if !generics.params.is_empty() || generics.where_clause.is_some()) {
                    #generics #{&generics.where_clause} ;
                }
            });
            match self.context_self(context) {
                None => (),
                Some(SelfRequirement::Bound(path)) => tokens.extend(quote!(Self : #path ;)),
                Some(SelfRequirement::Type(ty)) => tokens.extend(quote!(Self = #ty ;)),
            }
            if !outlives.is_empty() {
                tokens.extend(quote!(Self : #(for lifetime in outlives)+ { #lifetime } ;));
            }
            if let Some(info) = self.modules.get(&context) {
                let fields = [
                    info.path
                        .as_ref()
                        .map(|path| quote!(path = [#(for ident in path), { #ident }])),
                    info.hint.as_ref().map(|hint| quote!(hint = #hint)),
                    info.parent.map(|parent| quote!(parent = #parent)),
                ];
                tokens
                    .extend(quote!(mod { #(for field in fields.iter().flatten()), { #field } } ;));
            } else if let Some(module) = self.context_module(context) {
                tokens.extend(quote!(mod #module ;));
            }
            tokens.extend(quote! {
                (
                    #(for ix in entries), {
                        #(if !self.musts[ix]) { ? }
                        #(if !self.external_crates[ix].is_empty()) {
                            @[#(for name in &self.external_crates[ix]), { #{crate_ident(name)} }]
                        }
                        #{&self.leak_types[ix]}
                    }
                )
            });
            next = context.index() + 1;
        }
    }
}

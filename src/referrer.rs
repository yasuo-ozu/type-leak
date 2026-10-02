//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::graph::TypeKey;
use crate::requirements::{Requirements, Scope};
use crate::util::mentions_names;
use proc_macro2::TokenStream;
use std::cell::RefCell;
use std::collections::HashMap;
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
/// Each type is an entry, identified by its context (see [`Referrer::context()`]), its tokens, and
/// the local generics it depends on (see [`Referrer::local_generics()`]): the same tokens with
/// different local generics (e.g. `Wrap<B>` where `B` is a `bool` in one method and a `u8` in
/// another) are separate entries, and the one a copy refers to is found by resolving the generics
/// in scope where it appears.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    pub(crate) leak_types: Vec<Type>,
    pub(crate) requirements: Vec<Requirements>,
    pub(crate) contexts: Vec<usize>,
    /// The candidates (and their indices) of a type in a context, keyed by [`TypeKey`] (by tokens)
    /// as in the Leaker, so that e.g. a type wrapped in an invisible group is found as well.
    pub(crate) map: HashMap<(usize, TypeKey), Vec<(Requirements, usize)>>,
}

impl Referrer {
    /// Builds a Referrer from its entries (the context, the type and its requirements), in the
    /// order of their indices.
    pub(crate) fn new(entries: impl IntoIterator<Item = (usize, Type, Requirements)>) -> Self {
        let mut referrer = Self {
            leak_types: Vec::new(),
            requirements: Vec::new(),
            contexts: Vec::new(),
            map: HashMap::new(),
        };
        for (ix, (context, ty, requirements)) in entries.into_iter().enumerate() {
            referrer
                .map
                .entry((context, TypeKey::new(&ty)))
                .or_default()
                .push((requirements.clone(), ix));
            referrer.leak_types.push(ty);
            referrer.requirements.push(requirements);
            referrer.contexts.push(context);
        }
        referrer
    }

    /// Returns the index of the type `ty` in `context`, if it is leaked with the requirements it
    /// has with the generics in `scopes`.
    fn find(&self, context: usize, ty: &Type, scopes: &[Scope]) -> Option<usize> {
        let candidates = self.map.get(&(context, TypeKey::new(ty)))?;
        let requirements = Requirements::of(ty, scopes);
        candidates
            .iter()
            .find(|(candidate, _)| candidate == &requirements)
            .map(|(_, ix)| *ix)
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

    /// Returns the local generics which the `ix`-th type depends on, i.e. generics of a trait
    /// method or a GAT (or of a nested item, or a `for<..>` binder) which are not generics of the
    /// Leaker. It is empty for most types.
    ///
    /// They are in a canonical form, which is the same for declarations with the same meaning: the
    /// parameters are sorted by name (lifetimes first), and the bounds are sorted and moved from
    /// the where clause to the parameter they bound, if any. Defaults are removed.
    ///
    /// A type such as `Wrap<B>` in `fn f<const B: bool>(&self) -> Wrap<B>` (where `Wrap` depends on
    /// the type context) cannot be decomposed, so it is interned together with `B`. The Repeater's
    /// implementation must then declare these generics and constrain them in its trait arguments,
    /// e.g. with [`encode_generics_params_to_ty()`](crate::encode_generics_params_to_ty()). The
    /// Referrer declares the same generics with the same names (as it copies the method's
    /// signature), so it can encode them in the same way. Note that lifetimes must be declared
    /// before the other generics when merging them with the Leaker's generics.
    ///
    /// The bounds and where predicates of the local generics are kept as they are, so they may
    /// mention `Self` (e.g. `fn f<T: PartialEq<Self>>(&self)`), which refers to the implementor of
    /// the Leaker trait. The Repeater's implementation should therefore be for the implementor
    /// (e.g. `impl<.., S: ?Sized> Repeater<..> for S where S: MyTrait<..>`), where `Self` has the
    /// same meaning; in an implementation for another type (e.g. a marker type), `Self` would
    /// refer to that type instead.
    ///
    ///
    /// ```ignore
    /// // In the Leaker (`MyTrait` is the Leaker trait with `generics`)
    /// for (ix, (ty, locals)) in referrer.iter_with_local_generics().enumerate() {
    ///     let args = encode_generics_params_to_ty(&generics.params);
    ///     let local_args = encode_generics_params_to_ty(&locals.params);
    ///     let local_where = locals.where_clause.iter().flat_map(|wc| wc.predicates.iter());
    ///     quote! {
    ///         impl<#generics_params, #locals_params, S: ?Sized>
    ///             Repeater<(#args, #local_args), #ID, #ix> for S
    ///         where
    ///             S: MyTrait<#generics_args>,
    ///             #(#local_where,)*
    ///         {
    ///             type Type = #ty;
    ///         }
    ///     }
    /// }
    ///
    /// // In the Referrer (in the implementation of `MyTrait<#trait_args>`)
    /// let args = encode_generics_args_to_ty_with(&trait_args, Some(&generics.params), |expr, _| {
    ///     parse_quote!([(); #expr])
    /// })?;
    /// let local_args = encode_generics_params_to_ty(&referrer.local_generics(ix).params);
    /// parse_quote!(<Self as Repeater<(#args, #local_args), #ID, #ix>>::Type)
    /// ```
    pub fn local_generics(&self, ix: usize) -> &Generics {
        self.requirements[ix].generics()
    }

    /// Returns the context of the `ix`-th type: the number of the
    /// [`Leaker::intern_with()`](crate::Leaker::intern_with()) call it was found by (counting the
    /// successful calls from 0), or its constructor ([`Leaker::from_struct()`](crate::Leaker::from_struct())
    /// etc.), which is the call 0. A type found in several contexts has an entry for each, since
    /// the generics of the Leaker may differ.
    pub fn context(&self, ix: usize) -> usize {
        self.contexts[ix]
    }

    /// Returns an iterator over the types that need to be encoded, together with their local
    /// generics. See [`Referrer::local_generics()`].
    pub fn iter_with_local_generics(&self) -> impl Iterator<Item = (&Type, &Generics)> {
        self.leak_types
            .iter()
            .zip(self.requirements.iter().map(Requirements::generics))
    }

    /// Converts this `Referrer` into a [`Visitor`] that can be used to transform AST nodes of the
    /// context 0.
    ///
    /// The `type_fn` callback is called for each type that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type.
    /// See [`Referrer::expand()`] for details.
    pub fn into_visitor<F: FnMut(Type, usize) -> Type>(self, type_fn: F) -> Visitor<F> {
        self.into_visitor_in(0, type_fn)
    }

    /// Like [`Referrer::into_visitor()`], for AST nodes of `context` (see
    /// [`Referrer::context()`]): only the types found in it are replaced.
    pub fn into_visitor_in<F: FnMut(Type, usize) -> Type>(
        self,
        context: usize,
        type_fn: F,
    ) -> Visitor<F> {
        Visitor {
            referrer: self,
            type_fn: Rc::new(RefCell::new(type_fn)),
            context,
            scopes: Vec::new(),
        }
    }

    /// Expands a type by replacing any interned types using the provided mapping function.
    ///
    /// The `type_fn` callback is called for each subtype that needs to be converted,
    /// receiving the original type and its index, and returning the replacement type. The type is
    /// expanded as in the context 0, with no generics in scope other than those of the Leaker.
    ///
    /// ```ignore
    /// let args = encode_generics_params_to_ty(&generics.params);
    /// referrer.expand(ty, |_, ix| {
    ///     parse_quote!(<#marker_ty as #crate::TypeRef<#ix, #args>>::Type)
    /// })
    /// ```
    pub fn expand(&self, ty: Type, type_fn: impl FnMut(Type, usize) -> Type) -> Type {
        self.expand_in(0, ty, type_fn)
    }

    /// Like [`Referrer::expand()`], for a type of `context` (see [`Referrer::context()`]).
    pub fn expand_in(
        &self,
        context: usize,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        self.expand_with(context, &mut Vec::new(), ty, type_fn)
    }

    /// Expands `ty` as in `context`, where the generics of `scopes` are in scope.
    fn expand_with(
        &self,
        context: usize,
        scopes: &mut Vec<Scope>,
        ty: Type,
        type_fn: impl FnMut(Type, usize) -> Type,
    ) -> Type {
        use syn::fold::Fold;
        struct Folder<'a, F> {
            referrer: &'a Referrer,
            context: usize,
            scopes: &'a mut Vec<Scope>,
            type_fn: F,
        }
        impl<'a, F> Folder<'a, F> {
            /// Pushes the scope of the lifetimes of a `for<..>` binder, as the Leaker does.
            fn push_binder(&mut self, lifetimes: Option<&BoundLifetimes>) -> bool {
                lifetimes
                    .map(|lifetimes| self.scopes.push(Scope::binder(lifetimes)))
                    .is_some()
            }
        }
        impl<'a, F: FnMut(Type, usize) -> Type> Fold for Folder<'a, F> {
            fn fold_type(&mut self, ty: Type) -> Type {
                match self.referrer.find(self.context, &ty, self.scopes) {
                    Some(ix) => (self.type_fn)(ty, ix),
                    None => syn::fold::fold_type(self, ty),
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
            context,
            scopes,
            type_fn,
        }
        .fold_type(ty)
    }
}

/// Rewrites AST nodes with the leaked types of a [`Referrer`], see [`Referrer::into_visitor()`].
///
/// It keeps track of the generics declared at the visited site (those of items, methods, impls and
/// `for<..>` binders), including those of the outermost item visited: a type is rewritten only if
/// it was leaked with the same local generics as it depends on here (see
/// [`Referrer::local_generics()`]). A type mentioning generics which are not leaked with it (e.g.
/// `T` in the `impl<T>` of the visited item) is not rewritten, since it refers to them rather than
/// to the Leaker's items.
#[derive(Debug)]
pub struct Visitor<F> {
    referrer: Referrer,
    type_fn: Rc<RefCell<F>>,
    context: usize,
    /// The generics in scope, the innermost last.
    scopes: Vec<Scope>,
}

impl<F> Clone for Visitor<F> {
    fn clone(&self) -> Self {
        Self {
            referrer: self.referrer.clone(),
            type_fn: self.type_fn.clone(),
            context: self.context,
            scopes: self.scopes.clone(),
        }
    }
}

impl<F: FnMut(Type, usize) -> Type> Visitor<F> {
    /// Stops rewriting the leaked types which mention `idents` (items or imports declared in a
    /// block or a module).
    fn shadow_idents(&mut self, idents: Vec<Ident>) {
        self.referrer
            .map
            .retain(|(_, key), _| !mentions_names(&key.ty, idents.clone(), Vec::new()));
    }

    /// Shadows the items declared in `items`.
    fn shadow_items<'a>(&mut self, items: impl IntoIterator<Item = &'a Item>) {
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
            self.shadow_idents(names);
        }
    }

    fn with_generics(&mut self, generics: &mut Generics) -> Self {
        let mut visitor = self.clone();
        // All the parameters are in scope of each other's bounds.
        visitor.scopes.push(Scope::new(generics));
        for gp in generics.params.iter_mut() {
            visitor.visit_generic_param_mut(gp);
        }
        if let Some(where_clause) = &mut generics.where_clause {
            visitor.visit_where_clause_mut(where_clause);
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
        *i = self.referrer.expand_with(
            self.context,
            &mut self.scopes,
            i.clone(),
            &mut *self.type_fn.borrow_mut(),
        );
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
    fn visit_signature_mut(&mut self, i: &mut Signature) {
        self.with_signature(i);
    }
    fn visit_item_impl_mut(&mut self, i: &mut ItemImpl) {
        for attr in i.attrs.iter_mut() {
            self.visit_attribute_mut(attr);
        }
        let mut visitor = self.with_generics(&mut i.generics);
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
        let mut visitor = self.with_signature(&mut i.sig);
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
            visitor.shadow_items(items.iter());
            for item in items.iter_mut() {
                visitor.visit_item_mut(item);
            }
        }
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
        visitor.shadow_items(i.stmts.iter().filter_map(|stmt| match stmt {
            Stmt::Item(item) => Some(item),
            _ => None,
        }));
        for stmt in i.stmts.iter_mut() {
            visitor.visit_stmt_mut(stmt);
        }
    }
}

/// The version of this crate, emitted with a serialized [`Referrer`] to detect a Referrer passed
/// between incompatible versions of `type-leak`.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Parses a [`Referrer`] serialized with its [`ToTokens`] implementation: the version of
/// `type-leak` as a string literal, (the number of contexts if it is more than one,) and the
/// leaked types in parentheses, one group for each context (see [`Referrer::context()`]). The
/// tokens after it are not consumed, so a Referrer can be followed by other tokens. It is an error
/// if the version differs from that of this crate.
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
        for context in 0..n_contexts {
            let content: syn::parse::ParseBuffer<'_>;
            parenthesized!(content in input);
            // Each entry is either `Type` or `{ <local generics> where ..; Type }`.
            let group = Punctuated::<(Generics, Type), Token![,]>::parse_terminated_with(
                &content,
                |input| {
                    if input.peek(token::Brace) {
                        let content: syn::parse::ParseBuffer<'_>;
                        braced!(content in input);
                        let mut generics: Generics = content.parse()?;
                        generics.where_clause = content.parse()?;
                        content.parse::<Token![;]>()?;
                        Ok((generics, content.parse()?))
                    } else {
                        Ok((Generics::default(), input.parse()?))
                    }
                },
            )?;
            entries.extend(
                group
                    .into_iter()
                    .map(|(generics, ty)| (context, ty, Requirements::new(generics))),
            );
        }
        Ok(Self::new(entries))
    }
}

/// Serializes the [`Referrer`] (with the version of `type-leak`), to be parsed back with its
/// [`Parse`] implementation, e.g. in another macro invocation. The types are in parentheses, one
/// group for each context up to the last one with a type (at least one group), preceded by the
/// number of the groups if it is more than one (e.g. `"0.8.0" 2 (MyType) (Wrap<T>)`).
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
        tokens.extend(quote!(#VERSION));
        let n_contexts = self.contexts.iter().max().map_or(1, |context| context + 1);
        if n_contexts > 1 {
            // The number of the groups, so that the Referrer can be followed by other tokens.
            tokens.extend(quote!(#{proc_macro2::Literal::usize_unsuffixed(n_contexts)}));
        }
        for context in 0..n_contexts {
            let entries: Vec<_> = self
                .leak_types
                .iter()
                .zip(&self.requirements)
                .zip(&self.contexts)
                .filter(|(_, c)| **c == context)
                .map(|(entry, _)| entry)
                .collect();
            tokens.extend(quote! {
                (
                    #(for (item, requirements) in entries), {
                        #(if requirements.is_empty()) {
                            #item
                        } #(else) {
                            { #{requirements.generics()} #{&requirements.generics().where_clause}; #item }
                        }
                    }
                )
            });
        }
    }
}

//! The [`Referrer`], which rewrites copies of the analyzed items.

use crate::graph::TypeKey;
use crate::util::{generic_param_name, mentions_generics, mentions_names};
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
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Referrer {
    pub(crate) leak_types: Vec<Type>,
    /// Keyed by [`TypeKey`] (by tokens) as in the Leaker, so that e.g. a type wrapped in an
    /// invisible group is found as well.
    pub(crate) map: HashMap<TypeKey, usize>,
    pub(crate) local_generics: Vec<Generics>,
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

    /// Returns the local generics which the `ix`-th type depends on, i.e. generics of a trait
    /// method or a GAT which are not generics of the Leaker. It is empty for most types.
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
        &self.local_generics[ix]
    }

    /// Returns an iterator over the types that need to be encoded, together with their local
    /// generics. See [`Referrer::local_generics()`].
    pub fn iter_with_local_generics(&self) -> impl Iterator<Item = (&Type, &Generics)> {
        self.leak_types.iter().zip(self.local_generics.iter())
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
                if let Some(idx) = self.0.map.get(&TypeKey::new(&ty)) {
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
    /// Stops rewriting the leaked types which mention the generic parameters declared here (they
    /// refer to these parameters rather than to the Leaker's items), unless the leaked type was
    /// interned together with local generics of the same name (see
    /// [`Referrer::local_generics()`]).
    fn shadow_generics(&mut self, generics: &Generics) {
        let referrer = &self.0;
        let shadowed: Vec<TypeKey> = referrer
            .map
            .iter()
            .filter(|(key, ix)| {
                let locals = &referrer.local_generics[**ix];
                let params = generics.params.iter().filter(|gp| {
                    !locals
                        .params
                        .iter()
                        .any(|local| generic_param_name(local) == generic_param_name(gp))
                });
                mentions_generics(&key.ty, params)
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in shadowed {
            self.0.map.remove(&key);
        }
    }

    /// Stops rewriting the leaked types which mention `idents` (items or imports declared in a
    /// block or a module).
    fn shadow_idents(&mut self, idents: Vec<Ident>) {
        let shadowed: Vec<TypeKey> = self
            .0
            .map
            .keys()
            .filter(|key| mentions_names(&key.ty, idents.clone(), Vec::new()))
            .cloned()
            .collect();
        for key in shadowed {
            self.0.map.remove(&key);
        }
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
        visitor.shadow_generics(generics);
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
        *i = self.0.expand(i.clone(), &mut *self.1.borrow_mut());
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
/// `type-leak` as a string literal, followed by the leaked types in parentheses. It is an error if
/// the version differs from that of this crate.
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
        let content: syn::parse::ParseBuffer<'_>;
        parenthesized!(content in input);
        // Each entry is either `Type` or `{ <local generics> where ..; Type }`.
        let entries =
            Punctuated::<(Generics, Type), Token![,]>::parse_terminated_with(&content, |input| {
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
            })?;
        let (local_generics, leak_types): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
        let map = leak_types
            .iter()
            .enumerate()
            .map(|(n, ty)| (TypeKey::new(ty), n))
            .collect();
        Ok(Self {
            map,
            leak_types,
            local_generics,
        })
    }
}

/// Serializes the [`Referrer`] (with the version of `type-leak`), to be parsed back with its
/// [`Parse`] implementation, e.g. in another macro invocation.
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
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        tokens.extend(quote! {
            #VERSION
            (
                #(for (item, generics) in self.leak_types.iter().zip(&self.local_generics)), {
                    #(if generics.params.is_empty()) {
                        #item
                    } #(else) {
                        { #generics #{&generics.where_clause}; #item }
                    }
                }
            )
        })
    }
}

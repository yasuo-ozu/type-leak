//! The `visit_*` methods of the [`View`]: its own, and the [`Visit`] implementation, which forwards
//! the methods the analysis overrides.

use super::View;
use syn::visit::Visit;
use syn::*;

/// Forwards the `visit_*` methods `$method` of the nodes `$ty` to the analysis.
macro_rules! forward {
    ($($method:ident: $ty:ty,)*) => {
        $(
            fn $method(&mut self, i: &'ast $ty) {
                Visit::$method(&mut *self.visitor, i)
            }
        )*
    };
}

impl View<'_> {
    /// Walks the parts of an impl of the scope: the walk of `syn::visit::visit_item_impl()`, with
    /// the self type checked.
    pub(crate) fn walk_item_impl(&mut self, item: &ItemImpl) {
        self.visitor.walk_item_impl(item)
    }

    /// Analyzes a nested item, such as a struct, a trait, an impl, a module or a function.
    ///
    /// It is analyzed like an item of a module
    /// ([`Leaker::intern_mod()`](crate::Leaker::intern_mod)). Its generics are local, and the
    /// generics around stay in scope. To analyze an item with its own generics as the root, use
    /// an `intern_*` method instead. A struct, enum, union, trait or impl declares its own `Self`,
    /// which may be interned whatever `allow_self` the scope around has (the `Self` of the scope
    /// around is hidden in the item, and is denied as usual). The copy's
    /// [`Visitor`](crate::Visitor) finds the entries that name it with
    /// [`Visitor::with_nested_self()`](crate::Visitor::with_nested_self). A function, type alias,
    /// trait alias, const or static has no `Self` of its own: it keeps the one around.
    ///
    /// The copy of a nested struct, enum or union declares a type of its own, so `Self` in a type
    /// that is copied as written is the copy's own type, and only a type that is leaked is written
    /// by the macro, by convention with the Leaker's type. An absolute type that names `Self`
    /// (`::path::Wrap<Self>`, `::std::boxed::Box<Self>`) is copied as written: it is no entry, not
    /// even an optional one, whether a Referrer replaces the optional entries or not. A relative
    /// type that names it (`Wrap<Self>`) is leaked, and the macro writes the Leaker's type. A type
    /// that uses a generic with a bound that names its `Self`, and does not name `Self` itself
    /// (`struct S<T: AsRef<Self>>(Rel<T>);`), is for any type that the bound holds for
    /// ([`SelfRequirement::Any`](crate::SelfRequirement)): the copy has the bound for its own
    /// type, and writes `<Self as Repeater<..>>::Type` for `Rel<T>`. A type that names `Self` too
    /// (`Rel2<Self, T>`) is an error: the Repeater's impl of the type is for the Leaker's `S<T>`,
    /// which has not the bound in the copy's crate. The headline is "`Rel2<Self, T>` has a generic
    /// whose bound mentions `Self`, which cannot be leaked". Write the type, or the bound, without
    /// `Self`, write the type with an absolute path (`::path::Rel2<Self, T>` is copied as written,
    /// and its `Self` is the copy's own), or analyze the item with an `intern_*` method, where the
    /// Visitor writes `Self` as the Leaker's type.
    ///
    /// The bounds of a generic are written for the `Self` of the scope that declares it. A type in
    /// a nested item that uses the generic gets them rewritten for the `Self` of that item, and its
    /// entry does not depend on the other one: `T: ::p::Tr<Self>` of `struct S<T: ::p::Tr<Self>>`
    /// is `T: ::p::Tr<S<T>>` in `impl W { fn f(x: Rel<T>); }`. That is an error if `Self` is the
    /// implementor of a trait (it has no name), or if the bound names an associated item of `Self`
    /// (`Self::X`, which is ambiguous for a type): the headline is "the bound of `T` mentions a
    /// `Self` which has no name here" ([`CheckResult::headline`](crate::CheckResult::headline) has
    /// it for [`View::check()`]). The scope that declares the bound records its `Self` in its
    /// top-level context, so [`Referrer::visitor()`](crate::Referrer::visitor) knows it, and the
    /// copy must nest the same scopes as the Leaker did.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .enter_generics(&parse_quote!(<T>), |view| {
    ///         // `T` is a generic of the scope, and `U` is local to the struct. Each type is an
    ///         // entry with the generics it uses.
    ///         view.visit_item(&parse_quote!(struct Inner<U> { a: Rel<U>, b: Wrap<T> }));
    ///     })
    ///     .unwrap();
    /// let referrer = leaker.finish();
    /// assert_eq!(referrer.iter().count(), 2);
    /// assert_eq!(referrer.generics(0).params.len(), 1);
    /// assert_eq!(referrer.generics(1).params.len(), 1);
    /// ```
    pub fn visit_item(&mut self, item: &Item) {
        Visit::visit_item(&mut *self.visitor, item)
    }

    /// Analyzes a type, with the generics, `Self` and module of the scope.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker
    ///     .enter_generics(&parse_quote!(<T>), |view| {
    ///         view.visit_type(&parse_quote!(::core::option::Option<Wrap<T>>));
    ///     })
    ///     .unwrap();
    /// // `Option<..>` is a path of the standard library, which is the same in every crate.
    /// assert_eq!(leaker.finish().iter().collect::<Vec<_>>(), [&parse_quote!(Wrap<T>)]);
    /// ```
    pub fn visit_type(&mut self, ty: &Type) {
        Visit::visit_type(&mut *self.visitor, ty)
    }

    /// Analyzes the arguments of a path whose name is resolved elsewhere (`<Rel>` of `Wrap<Rel>`),
    /// not the path itself.
    pub fn visit_path_arguments(&mut self, arguments: &PathArguments) {
        Visit::visit_path_arguments(&mut *self.visitor, arguments)
    }

    /// Analyzes a where clause, such as the one of the generics of the scope.
    pub fn visit_where_clause(&mut self, where_clause: &WhereClause) {
        Visit::visit_where_clause(&mut *self.visitor, where_clause)
    }

    /// Analyzes a bound, such as a bound of a generic parameter of the scope.
    ///
    /// The path of a trait bound must not depend on the context, nor start with `Self` where it has
    /// no meaning: that is an error.
    pub fn visit_type_param_bound(&mut self, bound: &TypeParamBound) {
        Visit::visit_type_param_bound(&mut *self.visitor, bound)
    }
}

// Forwards only the methods the analysis overrides, so walking the view walks the analysis. The
// default walk of the other methods reaches the forwarded ones through the view. The test below
// keeps this list in sync with the overrides of the analysis.
impl<'ast> Visit<'ast> for View<'_> {
    forward! {
        visit_angle_bracketed_generic_arguments: AngleBracketedGenericArguments,
        visit_const_param: ConstParam,
        visit_expr: Expr,
        visit_expr_path: ExprPath,
        visit_fn_arg: FnArg,
        visit_foreign_item_fn: ForeignItemFn,
        visit_impl_item_const: ImplItemConst,
        visit_impl_item_fn: ImplItemFn,
        visit_impl_item_type: ImplItemType,
        visit_item_const: ItemConst,
        visit_item_enum: ItemEnum,
        visit_item_fn: ItemFn,
        visit_item_impl: ItemImpl,
        visit_item_mod: ItemMod,
        visit_item_static: ItemStatic,
        visit_item_struct: ItemStruct,
        visit_item_trait: ItemTrait,
        visit_item_trait_alias: ItemTraitAlias,
        visit_item_type: ItemType,
        visit_item_union: ItemUnion,
        visit_lifetime: Lifetime,
        visit_parenthesized_generic_arguments: ParenthesizedGenericArguments,
        visit_path: Path,
        visit_predicate_type: PredicateType,
        visit_qself: QSelf,
        visit_receiver: Receiver,
        visit_signature: Signature,
        visit_trait_bound: TraitBound,
        visit_trait_item_const: TraitItemConst,
        visit_trait_item_fn: TraitItemFn,
        visit_trait_item_type: TraitItemType,
        visit_type: Type,
        visit_type_bare_fn: TypeBareFn,
        visit_type_param: TypeParam,
        visit_type_param_bound: TypeParamBound,
        visit_type_path: TypePath,
        visit_type_trait_object: TypeTraitObject,
    }
}

#[cfg(test)]
mod tests {
    use proc_macro2::{TokenStream, TokenTree};
    use std::collections::BTreeSet;
    use syn::*;

    /// The `Visit` methods of the type `ty` in `source`: written out, or named in the `forward!`
    /// list.
    fn visit_methods(source: &str, ty: &str) -> BTreeSet<String> {
        fn idents(tokens: TokenStream, out: &mut BTreeSet<String>) {
            for token in tokens {
                match token {
                    TokenTree::Ident(ident) if ident.to_string().starts_with("visit_") => {
                        out.insert(ident.to_string());
                    }
                    TokenTree::Group(group) => idents(group.stream(), out),
                    _ => (),
                }
            }
        }
        let file = parse_file(source).unwrap();
        let mut out = BTreeSet::new();
        for item in file.items {
            let item = match item {
                Item::Impl(item) => item,
                _ => continue,
            };
            let is_visit = matches!(&item.trait_, Some((_, path, _))
                if path.segments.last().map_or(false, |seg| seg.ident == "Visit"));
            let is_ty = matches!(&*item.self_ty, Type::Path(path)
                if path.path.segments.last().map_or(false, |seg| seg.ident == ty));
            if !is_visit || !is_ty {
                continue;
            }
            for item in item.items {
                match item {
                    ImplItem::Fn(method) => {
                        out.insert(method.sig.ident.to_string());
                    }
                    ImplItem::Macro(mac) => idents(mac.mac.tokens, &mut out),
                    _ => (),
                }
            }
        }
        out
    }

    /// The view must forward exactly the methods the analysis overrides: a missing one would bypass
    /// the analysis silently, and an extra one is a stale entry.
    #[test]
    fn the_view_forwards_exactly_the_methods_the_analysis_overrides() {
        let overridden = visit_methods(include_str!("../analyze.rs"), "AnalyzeVisitor");
        let forwarded = visit_methods(include_str!("forward.rs"), "View");
        assert!(!overridden.is_empty(), "the analysis was not found");
        assert!(
            !forwarded.is_empty(),
            "the forwarded methods were not found"
        );
        let missing: Vec<_> = overridden.difference(&forwarded).collect();
        let stale: Vec<_> = forwarded.difference(&overridden).collect();
        assert!(
            missing.is_empty(),
            "overridden by the analysis but not forwarded by the view: {:?}",
            missing
        );
        assert!(
            stale.is_empty(),
            "forwarded by the view but not overridden by the analysis: {:?}",
            stale
        );
    }
}

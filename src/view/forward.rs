//! The [`Visit`] implementation of the [`View`], which forwards the methods the analysis overrides.

use super::View;
use syn::visit::Visit;
use syn::*;

/// Forwards the `visit_*` methods `$method` of the nodes `$ty` to the analysis.
macro_rules! forward {
    ($($method:ident: $ty:ty,)*) => {
        $(
            fn $method(&mut self, i: &'ast $ty) {
                Visit::$method(&mut *self.host, i)
            }
        )*
    };
}

/// Forwards only the methods the analysis overrides, so walking the view walks the analysis.
///
/// The default walk of the other methods reaches the forwarded ones through the view. The test
/// below keeps this list in sync with the overrides of the analysis.
impl<'ast> Visit<'ast> for View<'_> {
    forward! {
        visit_angle_bracketed_generic_arguments: AngleBracketedGenericArguments,
        visit_const_param: ConstParam,
        visit_expr: Expr,
        visit_expr_path: ExprPath,
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
        visit_path: Path,
        visit_predicate_type: PredicateType,
        visit_qself: QSelf,
        visit_receiver: Receiver,
        visit_trait_bound: TraitBound,
        visit_trait_item_const: TraitItemConst,
        visit_trait_item_fn: TraitItemFn,
        visit_trait_item_type: TraitItemType,
        visit_type: Type,
        visit_type_bare_fn: TypeBareFn,
        visit_type_param: TypeParam,
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

//! Tests for rewriting items with `Referrer::into_visitor()`.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::Referrer;

/// Rewrites `item` with the Referrer `referrer` (in its serialized form), replacing the `ix`-th
/// leaked type with `P<ix>`.
fn rewrite(referrer: proc_macro2::TokenStream, mut item: Item) -> String {
    let referrer: Referrer = parse2(referrer).unwrap();
    let mut visitor = referrer.into_visitor(|_, ix| parse_quote!(P<#ix>));
    visitor.visit_item_mut(&mut item);
    quote!(#item).to_string()
}

fn expected(item: Item) -> String {
    quote!(#item).to_string()
}

#[test]
fn test_where_clauses_are_rewritten() {
    let item: Item = parse_quote! {
        trait Tr<T> where T: ::x::Conv<RelA> {
            fn f<U>(&self, u: U) where U: ::x::Conv<RelB>;
            type G<V> where V: ::x::Conv<RelC>;
        }
    };
    assert_eq!(
        rewrite(quote!((RelA, RelB, RelC)), item),
        expected(parse_quote! {
            trait Tr<T> where T: ::x::Conv<P<0usize>> {
                fn f<U>(&self, u: U) where U: ::x::Conv<P<1usize>>;
                type G<V> where V: ::x::Conv<P<2usize>>;
            }
        })
    );
}

#[test]
fn test_generics_shadow_leaked_types() {
    // Const params, impl generics and method generics shadow leaked types mentioning them.
    let item: Item = parse_quote! {
        trait Tr {
            fn a(&self) -> B;
            fn g<const B: bool>(&self) -> ::x::Wrap<B>;
            fn h<T>(&self, t: T) -> T;
        }
    };
    assert_eq!(
        rewrite(quote!((B, T)), item),
        expected(parse_quote! {
            trait Tr {
                fn a(&self) -> P<0usize>;
                fn g<const B: bool>(&self) -> ::x::Wrap<B>;
                fn h<T>(&self, t: T) -> T;
            }
        })
    );

    let item: Item = parse_quote! {
        impl<T> X<T> {
            fn f<U>(&self, t: T, u: Wrap<U>) -> T { t }
        }
    };
    assert_eq!(
        rewrite(quote!((T, Wrap<U>)), item),
        expected(parse_quote! {
            impl<T> X<T> {
                fn f<U>(&self, t: T, u: Wrap<U>) -> T { t }
            }
        })
    );

    // All the parameters are in scope of each other's bounds.
    let item: Item = parse_quote! {
        struct S<A: ::x::Tr<T>, T>(A, T);
    };
    assert_eq!(
        rewrite(quote!((T)), item),
        expected(parse_quote! {
            struct S<A: ::x::Tr<T>, T>(A, T);
        })
    );
}

#[test]
fn test_local_generics_keep_matching() {
    // A type leaked together with local generics still matches where they are declared.
    let item: Item = parse_quote! {
        trait Tr {
            fn f<const B: bool>(&self) -> Wrap<B>;
        }
    };
    assert_eq!(
        rewrite(quote!(({ <const B: bool>; Wrap<B> })), item),
        expected(parse_quote! {
            trait Tr {
                fn f<const B: bool>(&self) -> P<0usize>;
            }
        })
    );
}

#[test]
fn test_items_in_blocks_and_modules_shadow_leaked_types() {
    let item: Item = parse_quote! {
        fn g() {
            struct Wrap<T>(T);
            use crate::other::RelType;
            let x: Wrap<u8>;
            let y: RelType;
            let z: Other;
        }
    };
    assert_eq!(
        rewrite(quote!((Wrap<u8>, RelType, Other)), item),
        expected(parse_quote! {
            fn g() {
                struct Wrap<T>(T);
                use crate::other::RelType;
                let x: Wrap<u8>;
                let y: RelType;
                let z: P<2usize>;
            }
        })
    );

    let item: Item = parse_quote! {
        mod m {
            pub struct RelType;
            pub fn f(x: RelType, y: Other) {}
        }
    };
    assert_eq!(
        rewrite(quote!((RelType, Other)), item),
        expected(parse_quote! {
            mod m {
                pub struct RelType;
                pub fn f(x: RelType, y: P<1usize>) {}
            }
        })
    );
}

#[test]
fn test_grouped_types_are_found() {
    // A type passed through `macro_rules!` (`$t:ty`) is wrapped in an invisible group, which
    // `syn` distinguishes from the bare type although the tokens are the same.
    let grouped = Type::Group(TypeGroup {
        group_token: Default::default(),
        elem: Box::new(parse_quote!(RelA)),
    });
    let item: ItemStruct = parse_quote! {
        struct S { a: #grouped, b: RelA }
    };
    let mut leaker = type_leak::Leaker::new();
    leaker
        .intern_with(&item.generics, |v| {
            syn::visit::Visit::visit_item_struct(v, &item)
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(
        rewrite(quote!(#referrer), Item::Struct(item)),
        expected(parse_quote! {
            struct S { a: P<0usize>, b: P<0usize> }
        })
    );
}

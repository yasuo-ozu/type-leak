//! Tests for rewriting items with `Referrer::into_visitor()`.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::Referrer;

/// Rewrites `item` with the Referrer of the leaked types `types` (in its serialized form, without
/// the version), replacing the `ix`-th leaked type with `P<ix>`.
fn rewrite(types: proc_macro2::TokenStream, item: Item) -> String {
    let version = env!("CARGO_PKG_VERSION");
    rewrite_with(parse2(quote!(#version #types)).unwrap(), item)
}

/// Rewrites `item` with `referrer`, replacing the `ix`-th leaked type with `P<ix>`.
fn rewrite_with(referrer: Referrer, mut item: Item) -> String {
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
        rewrite_with(referrer, Item::Struct(item)),
        expected(parse_quote! {
            struct S { a: P<0usize>, b: P<0usize> }
        })
    );
}

#[test]
fn test_version_is_checked() {
    // A Referrer emitted by another version of type-leak is rejected.
    let err = parse2::<Referrer>(quote!("0.0.0"(RelA))).unwrap_err();
    assert_eq!(
        err.to_string(),
        format!(
            "the Referrer was emitted by type-leak 0.0.0, but is parsed by type-leak {}",
            env!("CARGO_PKG_VERSION")
        )
    );
    // Without the version, it is not a Referrer.
    assert!(parse2::<Referrer>(quote!((RelA))).is_err());
}

#[test]
fn test_contexts() {
    let mut leaker = type_leak::Leaker::new();
    let item0: ItemStruct = parse_quote! { struct S<T> { a: RelA, b: Wrap<T> } };
    let item1: ItemStruct = parse_quote! { struct S<U> { a: RelA, b: Wrap<U> } };
    leaker
        .intern_with(&item0.generics, |v| {
            syn::visit::Visit::visit_item_struct(v, &item0)
        })
        .unwrap();
    // A failed call is not a context.
    let bad: ItemStruct = parse_quote! { struct S { a: RelB<'_> } };
    assert!(leaker
        .intern_with(&bad.generics, |v| {
            syn::visit::Visit::visit_item_struct(v, &bad)
        })
        .is_err());
    leaker
        .intern_with(&item1.generics, |v| {
            syn::visit::Visit::visit_item_struct(v, &item1)
        })
        .unwrap();
    let referrer = leaker.finish();
    // The same type in different contexts is not shared.
    assert_eq!(
        referrer
            .iter()
            .map(|ty| quote!(#ty).to_string())
            .collect::<Vec<_>>(),
        ["RelA", "Wrap < T >", "RelA", "Wrap < U >"]
    );
    assert_eq!(
        (0..4).map(|ix| referrer.context(ix)).collect::<Vec<_>>(),
        [0, 0, 1, 1]
    );

    // The contexts survive a round trip through `ToTokens` and `Parse`.
    let tokens = quote!(#referrer);
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 2 (RelA , Wrap < T >) (RelA , Wrap < U >)",
            env!("CARGO_PKG_VERSION")
        )
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(parsed.context(3), 1);

    // A visitor rewrites the types of its context only.
    let item: Item = parse_quote! { struct X { a: RelA, b: Wrap<U> } };
    assert_eq!(
        rewrite_with(referrer.clone(), item.clone()),
        expected(parse_quote! { struct X { a: P<0usize>, b: Wrap<U> } })
    );
    let mut rewritten = item;
    referrer
        .clone()
        .into_visitor_in(1, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut rewritten);
    assert_eq!(
        quote!(#rewritten).to_string(),
        expected(parse_quote! { struct X { a: P<2usize>, b: P<3usize> } })
    );
    let expected_ty: Type = parse_quote!(::p::Q<P<2usize>>);
    let ty = referrer.expand_in(1, parse_quote!(::p::Q<RelA>), |_, ix| parse_quote!(P<#ix>));
    assert_eq!(quote!(#ty).to_string(), quote!(#expected_ty).to_string());

    // Contexts without types are not serialized, unless followed by one with types.
    let mut leaker = type_leak::Leaker::new();
    let empty: ItemStruct = parse_quote! { struct S; };
    leaker
        .intern_with(&empty.generics, |v| {
            syn::visit::Visit::visit_item_struct(v, &empty)
        })
        .unwrap();
    leaker
        .intern_with(&item0.generics, |v| {
            syn::visit::Visit::visit_item_struct(v, &item0)
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.context(0), 1);
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
}

#[test]
fn test_referrer_followed_by_tokens() {
    // A serialized Referrer can be followed by other tokens (e.g. in a macro invocation), also
    // with several contexts.
    struct Embedded(Referrer, Ident);
    impl syn::parse::Parse for Embedded {
        fn parse(input: syn::parse::ParseStream) -> Result<Self> {
            let referrer = input.parse()?;
            input.parse::<Token![,]>()?;
            let content;
            parenthesized!(content in input);
            Ok(Embedded(referrer, content.parse()?))
        }
    }
    let item_a: ItemStruct = parse_quote!(
        struct A {
            a: RelA,
        }
    );
    let item_b: ItemStruct = parse_quote!(
        struct B {
            b: RelB,
        }
    );
    for items in [vec![&item_a], vec![&item_a, &item_b]] {
        let mut leaker = type_leak::Leaker::new();
        for item in items {
            leaker
                .intern_with(&item.generics, |v| {
                    syn::visit::Visit::visit_item_struct(v, item)
                })
                .unwrap();
        }
        let referrer = leaker.finish();
        let parsed: Embedded = parse2(quote!(#referrer, (after))).unwrap();
        assert_eq!(parsed.0, referrer);
        assert_eq!(parsed.1, "after");
    }
}

#[test]
fn test_outermost_generics_as_leakers() {
    // The Leaker is made from the trait, whose generics are those of the Leaker.
    let tr: ItemTrait = parse_quote! {
        trait MyTrait<T> {
            fn f(&self) -> Wrap<T>;
            fn g<const B: bool>(&self) -> Wrap2<B>;
        }
    };
    let referrer = type_leak::Leaker::from_trait(&tr).unwrap().finish();
    let item: Item = parse_quote! {
        impl<T> MyTrait<T> for W {
            fn f(&self) -> Wrap<T> { todo!() }
            fn g<const B: bool>(&self) -> Wrap2<B> { todo!() }
        }
    };
    let rewrite = |outermost_as_leakers: bool| {
        let mut item = item.clone();
        let mut visitor = referrer.clone().into_visitor(|_, ix| parse_quote!(P<#ix>));
        if outermost_as_leakers {
            visitor = visitor.with_outermost_generics_as_leakers();
        }
        visitor.visit_item_mut(&mut item);
        quote!(#item).to_string()
    };
    // By default, `T` is a local generic of the impl, so `Wrap<T>` is not rewritten.
    assert_eq!(
        rewrite(false),
        expected(parse_quote! {
            impl<T> MyTrait<T> for W {
                fn f(&self) -> Wrap<T> { todo!() }
                fn g<const B: bool>(&self) -> P<1usize> { todo!() }
            }
        })
    );
    // The generics of the method are still local.
    assert_eq!(
        rewrite(true),
        expected(parse_quote! {
            impl<T> MyTrait<T> for W {
                fn f(&self) -> P<0usize> { todo!() }
                fn g<const B: bool>(&self) -> P<1usize> { todo!() }
            }
        })
    );
}

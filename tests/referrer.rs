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
    let mut visitor = referrer.into_visitor(None, |_, ix| parse_quote!(P<#ix>));
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
        rewrite(quote!(<const B: bool>; (Wrap<B>)), item),
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
    // A failed call adds nothing.
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
    // `RelA` uses no generics, so it is shared by the calls, but `Wrap<T>` and `Wrap<U>` are not.
    assert_eq!(
        referrer
            .iter()
            .map(|ty| quote!(#ty).to_string())
            .collect::<Vec<_>>(),
        ["RelA", "Wrap < T >", "Wrap < U >"]
    );
    assert_eq!(
        (0..3).map(|ix| referrer.context(ix)).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(referrer.n_contexts(), 3);

    // The contexts survive a round trip through `ToTokens` and `Parse`.
    let tokens = quote!(#referrer);
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 3 (RelA) < T > ; (Wrap < T >) < U > ; (Wrap < U >)",
            env!("CARGO_PKG_VERSION")
        )
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(parsed.context(2), 2);

    // The generics which are not declared in the visited syntax are assumed to be the Leaker's,
    // so both are rewritten.
    let item: Item = parse_quote! { struct X { a: RelA, b: Wrap<U>, c: Wrap<T> } };
    assert_eq!(
        rewrite_with(referrer.clone(), item.clone()),
        expected(parse_quote! { struct X { a: P<0usize>, b: P<2usize>, c: P<1usize> } })
    );
    // With the generics of the Leaker, only the type with the same requirements is.
    for (generics, expected_item) in [
        (
            &item0.generics,
            parse_quote! { struct X { a: P<0usize>, b: Wrap<U>, c: P<1usize> } },
        ),
        (
            &item1.generics,
            parse_quote! { struct X { a: P<0usize>, b: P<2usize>, c: Wrap<T> } },
        ),
    ] {
        let mut rewritten = item.clone();
        referrer
            .clone()
            .into_visitor_with_root(generics, None, |_, ix| parse_quote!(P<#ix>))
            .visit_item_mut(&mut rewritten);
        assert_eq!(quote!(#rewritten).to_string(), expected(expected_item));
    }
    let expected_ty: Type = parse_quote!(::p::Q<P<0usize>>);
    let ty = referrer.expand(parse_quote!(::p::Q<RelA>), |_, ix| parse_quote!(P<#ix>));
    assert_eq!(quote!(#ty).to_string(), quote!(#expected_ty).to_string());

    // The contexts are those of the types, not of the calls: a call without types adds none.
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
    assert_eq!(referrer.context(0), 0);
    assert_eq!(referrer.n_contexts(), 2);
    assert_eq!(
        quote!(#referrer).to_string(),
        format!(
            "\"{}\" 2 (RelA) < T > ; (Wrap < T >)",
            env!("CARGO_PKG_VERSION")
        )
    );
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
    // Without any type, there is a context for the types which use nothing.
    let referrer = type_leak::Leaker::new().finish();
    assert_eq!(referrer.n_contexts(), 1);
    assert_eq!(
        quote!(#referrer).to_string(),
        format!("\"{}\" ()", env!("CARGO_PKG_VERSION"))
    );
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
    let rewrite = |item: &Item, outermost_as_leakers: bool| {
        let mut item = item.clone();
        let mut visitor = referrer
            .clone()
            .into_visitor(None, |_, ix| parse_quote!(P<#ix>));
        if outermost_as_leakers {
            visitor = visitor.with_outermost_generics_as_leakers();
        }
        visitor.visit_item_mut(&mut item);
        quote!(#item).to_string()
    };
    // `T` is declared in the same way (without bounds) as the Leaker's, so `Wrap<T>` is rewritten
    // in both cases.
    let item: Item = parse_quote! {
        impl<T> MyTrait<T> for W {
            fn f(&self) -> Wrap<T> { todo!() }
            fn g<const B: bool>(&self) -> Wrap2<B> { todo!() }
        }
    };
    let rewritten = expected(parse_quote! {
        impl<T> MyTrait<T> for W {
            fn f(&self) -> P<0usize> { todo!() }
            fn g<const B: bool>(&self) -> P<1usize> { todo!() }
        }
    });
    assert_eq!(rewrite(&item, false), rewritten);
    assert_eq!(rewrite(&item, true), rewritten);
    // By default, `T: Clone` of the impl is a different `T` from the Leaker's, so `Wrap<T>` is not
    // rewritten.
    let item: Item = parse_quote! {
        impl<T: ::core::clone::Clone> MyTrait<T> for W {
            fn f(&self) -> Wrap<T> { todo!() }
            fn g<const B: bool>(&self) -> Wrap2<B> { todo!() }
        }
    };
    assert_eq!(
        rewrite(&item, false),
        expected(parse_quote! {
            impl<T: ::core::clone::Clone> MyTrait<T> for W {
                fn f(&self) -> Wrap<T> { todo!() }
                fn g<const B: bool>(&self) -> P<1usize> { todo!() }
            }
        })
    );
    // The generics of the method are still declared.
    assert_eq!(
        rewrite(&item, true),
        expected(parse_quote! {
            impl<T: ::core::clone::Clone> MyTrait<T> for W {
                fn f(&self) -> P<0usize> { todo!() }
                fn g<const B: bool>(&self) -> P<1usize> { todo!() }
            }
        })
    );
}

fn referrer_of(tr: ItemTrait) -> Referrer {
    type_leak::Leaker::from_trait(&tr).unwrap().finish()
}

#[test]
fn test_has_self() {
    let with_self = [
        "fn f(&self) -> Rel<Self>;",
        "fn f(&self) -> Rel<Self::Item>;",
        "fn f(&self) -> Rel<<Self as ::p::Tr>::X>;",
        "fn f(&self) -> Wrap<[u8; <Self as ::p::Tr>::N]>;",
        "fn f(&self) -> Rel<m!(Self)>;",
        // A local bound mentioning `Self`.
        "fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap<T>;",
    ];
    let without_self = [
        "fn f(&self) -> Rel;",
        "fn f(&self) -> Wrap<u8>;",
        "fn f(&self) -> Rel<::p::Self>;",
        "fn f(&self) -> Rel<SelfLike>;",
        "fn f<T: ::core::cmp::PartialEq>(&self) -> Wrap<T>;",
        "fn f(&self);",
    ];
    for (items, expected) in [(&with_self[..], true), (&without_self[..], false)] {
        for item in items {
            let tr: ItemTrait = parse_str(&format!("trait Tr {{ {} }}", item)).unwrap();
            let referrer = referrer_of(tr);
            assert_eq!(referrer.has_self(), expected, "{}", item);
            // It is computed from the entries, so it survives a round trip.
            let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
            assert_eq!(parsed.has_self(), expected, "{}", item);
        }
    }
    // `Self` cannot be interned if `self_ty_can_be_interned` is unset.
    let leak = |item: &str| {
        let tr: ItemTrait = parse_str(&format!("trait Tr {{ {} }}", item)).unwrap();
        let mut config = type_leak::LeakerConfig::new();
        config.self_ty_can_be_interned = false;
        let mut leaker = type_leak::Leaker::from_config(config);
        leaker.intern_with(&tr.generics, |v| {
            syn::visit::Visit::visit_item_trait(v, &tr)
        })?;
        Ok::<_, type_leak::NotInternableError>(leaker.finish())
    };
    assert!(leak("fn f(&self, x: Rel<Self>) -> Rel2;").is_err());
    let referrer = leak("fn f(&self, x: Rel) -> Rel2;").unwrap();
    assert!(!referrer.is_empty());
    assert!(!referrer.has_self());
    // The bounds of the generics which a type uses are copied by the Repeater's implementation,
    // so they cannot mention `Self` either (but those of a generic which is not used can).
    let err = leak("fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap<T>;")
        .err()
        .unwrap();
    assert_eq!(err.to_string(), "use absolute path: 'Wrap < T >'");
    assert_eq!(
        err.note().as_deref(),
        Some("the bounds of its generics mention `Self`, which cannot be leaked")
    );
    assert!(leak("fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Rel;").is_ok());
}

#[test]
fn test_iter_with_self() {
    let referrer = referrer_of(parse_quote! {
        trait Tr {
            fn a(&self) -> Rel<Self>;
            fn b(&self) -> Wrap<Self::Item>;
            fn c(&self) -> Rel3<<Self as ::p::Tr>::X>;
            fn d(&self) -> Rel4;
            fn e<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap2<T>;
            fn g(&self) -> Rel5<m!(Self, [u8; 1])>;
        }
    });
    let marker: Type = parse_quote!(__S);
    let items: Vec<(String, String)> = referrer
        .iter_with_self(&marker)
        .map(|(ty, generics)| {
            (
                quote!(#ty).to_string(),
                quote!(#generics #{&generics.where_clause}).to_string(),
            )
        })
        .collect();
    let find = |ty: Type| {
        let ty = quote!(#ty).to_string();
        items
            .iter()
            .find(|(t, _)| *t == ty)
            .unwrap_or_else(|| panic!("{} not in {:?}", ty, items))
            .clone()
    };
    let is_str = |ty: Type| quote!(#ty).to_string();
    // `Self` alone, `Self::A`, the self type of a qualified path, a macro, and unchanged types.
    find(parse_quote!(Rel<__S>));
    find(parse_quote!(Wrap<<__S>::Item>));
    find(parse_quote!(Rel3<<__S as ::p::Tr>::X>));
    find(parse_quote!(Rel4));
    find(parse_quote!(Rel5<m!(__S, [u8; 1])>));
    // A local bound is rewritten.
    let (ty, generics) = find(parse_quote!(Wrap2<T>));
    assert_eq!(ty, is_str(parse_quote!(Wrap2<T>)));
    assert_eq!(
        generics,
        quote!(<T: ::core::cmp::PartialEq<__S> >).to_string()
    );
    // The originals keep `Self`, so a copy of the Leaker's trait is still found.
    assert!(referrer
        .iter()
        .any(|ty| quote!(#ty).to_string() == "Rel < Self >"));
    let mut sig: Signature = parse_quote!(fn a(&self) -> Rel<Self>);
    referrer
        .clone()
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn a(&self) -> P<0usize>);
    assert_eq!(quote!(#sig).to_string(), quote!(#expected).to_string());
}

#[test]
fn test_visitor_with_self() {
    let referrer = referrer_of(parse_quote! {
        trait Tr { fn f(&self, x: Rel<Self>, y: ::p::W<Self>) -> Self::Out; }
    });
    let sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>, y: ::p::W<Self>) -> Self::Out);

    // Without `with_self`, `Self` is kept.
    let mut plain = sig.clone();
    referrer
        .clone()
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .visit_signature_mut(&mut plain);
    let expected: Signature = parse_quote!(fn f(&self, x: P<0usize>, y: ::p::W<Self>) -> Self::Out);
    assert_eq!(quote!(#plain).to_string(), quote!(#expected).to_string());

    // With it, the callback receives the type with `Self` replaced, the rest is replaced as well,
    // and the receiver is untouched.
    let mut rewritten = sig.clone();
    let mut received = Vec::new();
    referrer
        .clone()
        .into_visitor(None, |ty, ix| {
            received.push((quote!(#ty).to_string(), ix));
            // Not rewritten again, though it mentions `Self`.
            parse_quote!(<Marker<Self> as R<#ix>>::Type)
        })
        .with_self(parse_quote!(__S))
        .visit_signature_mut(&mut rewritten);
    assert_eq!(received, [(quote!(Rel<__S>).to_string(), 0)],);
    let expected: Signature = parse_quote!(
        fn f(&self, x: <Marker<Self> as R<0usize>>::Type, y: ::p::W<__S>) -> <__S>::Out
    );
    assert_eq!(
        quote!(#rewritten).to_string(),
        quote!(#expected).to_string()
    );

    // The receiver with an explicit type is kept as well.
    let mut sig: Signature = parse_quote!(fn f(self: Box<Self>) -> Rel<Self>);
    referrer
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .with_self(parse_quote!(__S))
        .visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn f(self: Box<Self>) -> P<0usize>);
    assert_eq!(quote!(#sig).to_string(), quote!(#expected).to_string());
}

#[test]
fn test_nested_items_define_their_own_self() {
    let referrer = referrer_of(parse_quote! {
        trait Tr { fn f(&self) -> Rel<Self>; fn g(&self) -> Other; }
    });
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> Rel<Self> {
                struct Inner;
                impl Inner {
                    fn new() -> Rel<Self> { loop {} }
                    fn other() -> Other { loop {} }
                }
                let _: Rel<Self>;
                loop {}
            }
        }
    };
    // `Self` means `Inner` in the nested impl, so `Rel<Self>` there is not the leaked type, and
    // it is not replaced by `with_self`. The rest is as before.
    let mut rewritten = item.clone();
    referrer
        .clone()
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .with_self(parse_quote!(__S))
        .visit_item_trait_mut(&mut rewritten);
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> P<0usize> {
                struct Inner;
                impl Inner {
                    fn new() -> Rel<Self> { loop {} }
                    fn other() -> P<1usize> { loop {} }
                }
                let _: P<0usize>;
                loop {}
            }
        }
    };
    assert_eq!(
        quote!(#rewritten).to_string(),
        quote!(#expected).to_string()
    );

    // `Self` is kept in the other nested items as well, also without `with_self`, and in an
    // item of a block of a function copied alone.
    let mut item: ItemFn = parse_quote! {
        fn f() {
            enum E { A(Rel<Self>) }
            trait T { fn t() -> Self::Out; fn u() -> Rel<Self>; }
            union U { a: Rel<Self> }
        }
    };
    let expected = item.clone();
    referrer
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .with_self(parse_quote!(__S))
        .visit_item_fn_mut(&mut item);
    assert_eq!(quote!(#item).to_string(), quote!(#expected).to_string());
}

#[test]
fn test_module_items_do_not_shadow_submodules() {
    // The items of `m` are not in scope in `m::inner`, where `Rel` is imported from the parent
    // and keyed by that module.
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct Rel;
            pub struct A { x: Rel }
            pub mod inner { use super::Rel; pub struct C { y: Rel } }
        }
    };
    let referrer = type_leak::Leaker::from_module(&m).unwrap().finish();
    // Only the `Rel` of `m::inner` is leaked.
    assert_eq!(referrer.n_contexts(), 1);
    let mut item = Item::Mod(m);
    referrer
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut item);
    // `Rel` of `m` is declared there, and the one of `m::inner` is rewritten.
    let expected: Item = parse_quote! {
        mod m {
            pub struct Rel;
            pub struct A { x: Rel }
            pub mod inner { use super::Rel; pub struct C { y: P<0usize> } }
        }
    };
    assert_eq!(quote!(#item).to_string(), quote!(#expected).to_string());
}

#[test]
fn test_extern_crates_and_trait_aliases_shadow_and_declare() {
    // `extern crate foo as Other;` declares `Other`, like a `use`.
    let item: Item = parse_quote! {
        fn g() {
            extern crate foo as Other;
            let a: Other;
            let b: Rel;
        }
    };
    assert_eq!(
        rewrite(quote!((Other, Rel)), item),
        expected(parse_quote! {
            fn g() {
                extern crate foo as Other;
                let a: Other;
                let b: P<1usize>;
            }
        })
    );
    // The generics of a trait alias are in scope in its bounds.
    let item: Item = parse_quote! {
        trait Alias<T> = ::x::Conv<T, Rel>;
    };
    assert_eq!(
        rewrite(quote!((Rel, T)), item),
        expected(parse_quote! {
            trait Alias<T> = ::x::Conv<T, P<0usize>>;
        })
    );
}

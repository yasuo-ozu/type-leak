//! Tests for rewriting items with `Referrer::visitor()`.

mod common;

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::{ContextId, Referrer};

/// The tag of the format of a serialized Referrer, which it starts with.
fn tag() -> String {
    format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
}

/// Rewrites `item` with the Referrer of the leaked types `types` (in its serialized form, without
/// the version), replacing the `ix`-th leaked type with `P<ix>`. The scope of the copy is the first
/// context, the only one in the callers.
fn rewrite(types: proc_macro2::TokenStream, item: Item) -> String {
    let version = tag();
    let referrer: Referrer = parse2(quote!(#version #types)).unwrap();
    let context = referrer.contexts().next().unwrap();
    rewrite_with(referrer, context, item)
}

/// Rewrites `item`, a copy in the scope `context`, with `referrer`, replacing the `ix`-th leaked
/// type with `P<ix>`.
fn rewrite_with(referrer: Referrer, context: ContextId, mut item: Item) -> String {
    let mut visitor = referrer.visitor(context, |_, ix| parse_quote!(P<#ix>));
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
    let context = leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(
        rewrite_with(referrer, context, Item::Struct(item)),
        expected(parse_quote! {
            struct S { a: P<0usize>, b: P<0usize> }
        })
    );
}

/// The error of parsing `tokens` as a Referrer.
fn parse_error(tokens: proc_macro2::TokenStream) -> String {
    parse2::<Referrer>(tokens).unwrap_err().to_string()
}

#[test]
fn format_version_round_trip() {
    let item: ItemStruct = parse_quote! { struct S<T> { a: RelA, b: Wrap<T> } };
    let mut leaker = type_leak::Leaker::new();
    leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let tokens = quote!(#referrer);
    // The tag is the first token, a string literal.
    let first = tokens.clone().into_iter().next().unwrap().to_string();
    assert_eq!(first, format!("\"{}\"", tag()));
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
}

#[test]
fn format_version_is_checked() {
    let expected = format!(
        "(format {}, expected {})",
        Referrer::FORMAT_VERSION + 1,
        Referrer::FORMAT_VERSION
    );
    // A tampered version is rejected, with a clear error.
    let tampered = format!("type-leak-format-{}", Referrer::FORMAT_VERSION + 1);
    let message = parse_error(quote!(#tampered (RelA)));
    assert!(
        message.contains("serialized by an incompatible type-leak"),
        "{}",
        message
    );
    assert!(message.contains(&expected), "{}", message);
    // The whole message: it ends with what to do.
    assert_eq!(
        message,
        format!(
            "this Referrer was serialized by an incompatible type-leak (format {}, expected {}): \
             the macros which emit and parse it must use the same version of type-leak",
            Referrer::FORMAT_VERSION + 1,
            Referrer::FORMAT_VERSION
        )
    );
    // So is one of the previous format (no compatibility).
    let old = format!(
        "type-leak-format-{}",
        Referrer::FORMAT_VERSION.wrapping_sub(1)
    );
    assert!(parse_error(quote!(#old (RelA))).contains("incompatible type-leak (format"));
    // A literal which is not a tag (what a version of type-leak which had no format tag wrote:
    // its own version) is reported as such.
    let message = parse_error(quote!("0.8.0"(RelA)));
    assert_eq!(
        message,
        format!(
            "this Referrer was serialized by an incompatible type-leak (an unrecognized tag \
             \"0.8.0\", expected {}): the macros which emit and parse it must use the same \
             version of type-leak",
            Referrer::FORMAT_VERSION
        )
    );
    let message = parse_error(quote!("type-leak-format-"(RelA)));
    assert!(
        message.contains("an unrecognized tag \"type-leak-format-\""),
        "{}",
        message
    );
    // The check comes first, whatever follows.
    let message = parse_error(quote!("0.0.0" 2 (RelA) (RelB)));
    assert!(message.contains("incompatible type-leak"), "{}", message);
}

#[test]
fn data_without_the_format_tag_is_rejected() {
    // The data of a version without the tag does not start with a string literal at all.
    let message = parse_error(quote!((RelA)));
    assert_eq!(
        message,
        format!(
            "this Referrer was serialized by an incompatible type-leak (no format tag, expected \
             {}): the macros which emit and parse it must use the same version of type-leak",
            Referrer::FORMAT_VERSION
        )
    );
    // Nothing at all is no tag either.
    assert_eq!(
        parse_error(quote!()),
        format!(
            "this Referrer was serialized by an incompatible type-leak (no format tag, expected \
             {}): the macros which emit and parse it must use the same version of type-leak",
            Referrer::FORMAT_VERSION
        )
    );
    assert!(parse_error(quote!(2(RelA)(RelB))).contains("no format tag"));
    // The tag is accepted when it is there.
    let tag = tag();
    assert!(parse2::<Referrer>(quote!(#tag (RelA))).is_ok());
}

#[test]
fn test_contexts() {
    let mut leaker = type_leak::Leaker::new();
    let item0: ItemStruct = parse_quote! { struct S<T> { a: RelA, b: Wrap<T> } };
    let item1: ItemStruct = parse_quote! { struct S<U> { a: RelA, b: Wrap<U> } };
    let context0 = leaker.intern_struct(&item0, true);
    let context1 = leaker.intern_struct(&item1, true);
    let referrer = leaker.result().finish().unwrap();
    // `RelA` uses no generics, so it is shared by the calls, but `Wrap<T>` and `Wrap<U>` are not.
    assert_eq!(
        referrer
            .iter()
            .map(|ty| quote!(#ty).to_string())
            .collect::<Vec<_>>(),
        ["RelA", "Wrap < T >", "Wrap < U >"]
    );
    assert_eq!(
        (0..3)
            .map(|ix| referrer.context(ix).index())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(referrer.contexts().count(), 3);

    // The contexts survive a round trip through `ToTokens` and `Parse`.
    let tokens = quote!(#referrer);
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 3 (RelA) < T > ; (Wrap < T >) < U > ; (Wrap < U >)",
            tag()
        )
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(parsed.context(2).index(), 2);

    // The generics of a scope are those of the Leaker, so only the type with the same requirements
    // is rewritten: `Wrap<U>` is not found in the scope of the first call, nor `Wrap<T>` in the
    // scope of the second.
    let item: Item = parse_quote! { struct X { a: RelA, b: Wrap<U>, c: Wrap<T> } };
    for (context, expected_item) in [
        (
            context0,
            parse_quote! { struct X { a: P<0usize>, b: Wrap<U>, c: P<1usize> } },
        ),
        (
            context1,
            parse_quote! { struct X { a: P<0usize>, b: P<2usize>, c: Wrap<T> } },
        ),
    ] {
        assert_eq!(
            rewrite_with(referrer.clone(), context, item.clone()),
            expected(expected_item)
        );
    }
    // A generic that no scope declares is not assumed to be the Leaker's: both are found only in a
    // scope which declares both (here, a third scope of the same Leaker).
    let mut both = type_leak::Leaker::new();
    both.intern_struct(&item0, true);
    both.intern_struct(&item1, true);
    let (context, ()) = both.enter_generics(&parse_quote!(<T, U>), |_| ());
    assert_eq!(
        rewrite_with(both.result().finish().unwrap(), context, item.clone()),
        expected(parse_quote! { struct X { a: P<0usize>, b: P<2usize>, c: P<1usize> } })
    );
    let expected_ty: Type = parse_quote!(::p::Q<P<0usize>>);
    let ty = referrer.expand(
        context0,
        parse_quote!(::p::Q<RelA>),
        |_, ix| parse_quote!(P<#ix>),
    );
    assert_eq!(quote!(#ty).to_string(), quote!(#expected_ty).to_string());

    // The contexts are those of the types, not of the calls: a call without types adds none.
    let mut leaker = type_leak::Leaker::new();
    let empty: ItemStruct = parse_quote! { struct S; };
    leaker.intern_struct(&empty, true);
    leaker.intern_struct(&item0, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.context(0).index(), 0);
    assert_eq!(referrer.contexts().count(), 2);
    assert_eq!(
        quote!(#referrer).to_string(),
        format!("\"{}\" 2 (RelA) < T > ; (Wrap < T >)", tag())
    );
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
    // Without any type, there is a context for the types which use nothing.
    let referrer = type_leak::Leaker::new().result().finish().unwrap();
    assert_eq!(referrer.contexts().count(), 1);
    assert_eq!(quote!(#referrer).to_string(), format!("\"{}\" ()", tag()));
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
            leaker.intern_struct(item, true);
        }
        let referrer = leaker.result().finish().unwrap();
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
    let mut leaker = type_leak::Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let rewrite = |item: &Item, outermost_as_leakers: bool| {
        let mut item = item.clone();
        let mut visitor = referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>));
        if outermost_as_leakers {
            visitor = visitor.with_outermost_generics_as_leakers();
        }
        visitor.visit_item_mut(&mut item);
        quote!(#item).to_string()
    };
    // `T` is declared in the same way (without bounds) as the scope's, so the impl declares the
    // scope itself and `Wrap<T>` is rewritten in both cases.
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
    // By default, `T: Clone` of the impl is a different `T` from the scope's, so `Wrap<T>` is not
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

/// The Referrer of the trait, and its top-level context: the scope of its copies.
fn referrer_of(tr: ItemTrait) -> (Referrer, ContextId) {
    let mut leaker = type_leak::Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    (leaker.result().finish().unwrap(), context)
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
            let (referrer, _) = referrer_of(tr);
            assert_eq!(referrer.has_self(), expected, "{}", item);
            // It is computed from the entries, so it survives a round trip.
            let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
            assert_eq!(parsed.has_self(), expected, "{}", item);
        }
    }
    // `Self` cannot be interned if `allow_self` is unset.
    let leak = |item: &str| {
        let mut leaker = type_leak::Leaker::new();
        common::intern_fns(
            &mut leaker,
            &Generics::default(),
            item.parse().unwrap(),
            false,
        );
        leaker.result().finish()
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
    assert_eq!(err.len(), 1);
    assert_eq!(
        err[0].message,
        "`Wrap < T >` has a generic whose bound mentions `Self`, which cannot be leaked"
    );
    assert_eq!(
        err[0].note(),
        Some("the bounds of its generics mention `Self`, which cannot be leaked")
    );
    assert!(leak("fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Rel;").is_ok());
}

#[test]
fn test_iter_with_self() {
    let (referrer, context) = referrer_of(parse_quote! {
        trait Tr {
            fn a(&self) -> Rel<Self>;
            fn b(&self) -> Wrap<Self::Item>;
            fn c(&self) -> Rel3<<Self as ::p::Tr>::X>;
            fn d(&self) -> Rel4;
            fn e<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap2<T>;
            fn g(&self) -> Rel5<m!(Self, [u8; 1])>;
            const N: usize;
            fn h(&self, a: Rel6<[u8; Self::N]>);
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
    // `Self` in an array length (an expression) is replaced too. Audit mutant `s_expr_assoc_off`.
    find(parse_quote!(Rel6<[u8; <__S>::N]>));
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
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn a(&self) -> P<0usize>);
    assert_eq!(quote!(#sig).to_string(), quote!(#expected).to_string());
}

#[test]
fn test_visitor_with_self() {
    let (referrer, context) = referrer_of(parse_quote! {
        trait Tr { fn f(&self, x: Rel<Self>, y: ::p::W<Self>) -> Self::Out; }
    });
    let sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>, y: ::p::W<Self>) -> Self::Out);

    // Without `with_self`, `Self` is kept.
    let mut plain = sig.clone();
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_signature_mut(&mut plain);
    let expected: Signature = parse_quote!(fn f(&self, x: P<0usize>, y: ::p::W<Self>) -> Self::Out);
    assert_eq!(quote!(#plain).to_string(), quote!(#expected).to_string());

    // With it, the callback receives the type with `Self` replaced, the rest is replaced as well,
    // and the receiver is untouched.
    let mut rewritten = sig.clone();
    let mut received = Vec::new();
    referrer
        .clone()
        .visitor(context, |ty, ix| {
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
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .with_self(parse_quote!(__S))
        .visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn f(self: Box<Self>) -> P<0usize>);
    assert_eq!(quote!(#sig).to_string(), quote!(#expected).to_string());
}

/// Rewrites `sig` as a copy in the scope of `tr`, with `Self` written as `self_ty`.
fn rewrite_with_self(tr: &ItemTrait, self_ty: Type, sig: Signature) -> String {
    let (referrer, context) = referrer_of(tr.clone());
    let mut sig = sig;
    referrer
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .with_self(self_ty)
        .visit_signature_mut(&mut sig);
    quote!(#sig).to_string()
}

#[test]
fn test_visitor_with_self_names_the_elided_lifetime() {
    let tr: ItemTrait = parse_quote! { trait Tr { fn f(&self); } };
    let host: Type = parse_quote!(Host<'h>);
    let check = |sig: Signature, expected: Signature| {
        assert_eq!(
            rewrite_with_self(&tr, host.clone(), sig),
            quote!(#expected).to_string()
        );
    };
    // `Self` written as a type with lifetimes adds to the inputs of a function pointer, of a trait
    // object or of the sugar of a bound, so the lifetime that elision gives is named.
    check(
        parse_quote!(fn f(&self, x: fn(&Self) -> &u8)),
        parse_quote!(fn f(&self, x: for<'__tl_e0> fn(&'__tl_e0 Host<'h>) -> &'__tl_e0 u8)),
    );
    check(
        parse_quote!(fn f(&self, x: Box<dyn Fn(&(u8, Self)) -> &u8 + 'static>)),
        parse_quote!(fn f(&self, x: Box<dyn for<'__tl_e0> Fn(&'__tl_e0 (u8, Host<'h>)) -> &'__tl_e0 u8 + 'static>)),
    );
    check(
        parse_quote!(fn f<F: Fn(&Self) -> &u8>(&self, f: F) where F: Fn(&Vec<Self>) -> Option<&u8>),
        parse_quote!(
            fn f<F: for<'__tl_e0> Fn(&'__tl_e0 Host<'h>) -> &'__tl_e0 u8>(&self, f: F)
            where F: for<'__tl_e0> Fn(&'__tl_e0 Vec<Host<'h>>) -> Option<&'__tl_e0 u8>
        ),
    );
    // A lifetime in a path counts, and `Self::A` is written as the qualified path.
    check(
        parse_quote!(fn f(&self, x: fn(::core::cell::Ref<'_, Self::A>) -> &u8)),
        parse_quote!(
            fn f(&self, x: for<'__tl_e0> fn(::core::cell::Ref<'__tl_e0, <Host<'h>>::A>) -> &'__tl_e0 u8)
        ),
    );
    // The lifetime that the inputs name is the output's: no binder.
    check(
        parse_quote!(fn f(&self, x: for<'x> fn(&'x Self) -> &u8)),
        parse_quote!(fn f(&self, x: for<'x> fn(&'x Host<'h>) -> &'x u8)),
    );
    // Signatures inside each other have a binder each, with a name that the outer one leaves.
    check(
        parse_quote!(fn f(&self, x: fn(&(Self, fn(&Self) -> &u8)) -> &u8)),
        parse_quote!(
            fn f(&self, x: for<'__tl_e0> fn(
                &'__tl_e0 (Host<'h>, for<'__tl_e1> fn(&'__tl_e1 Host<'h>) -> &'__tl_e1 u8)
            ) -> &'__tl_e0 u8)
        ),
    );
    // The elision is not at stake: the output does not rely on it, `Self` is in the output only or
    // in a signature of its own, the inputs name every lifetime, or the type has none.
    for sig in [
        parse_quote!(fn f(&self, x: fn(&Self) -> Self)),
        parse_quote!(fn f(&self, x: fn(&u8) -> &Self)),
        parse_quote!(fn f(&self, x: for<'e> fn(&'e Self) -> &'e u8)),
        parse_quote!(fn f(&self, x: fn(&fn(&Self) -> Self) -> &u8)),
        parse_quote!(fn f(&self, x: fn(Self, u8) -> u8)),
    ] {
        let sig: Signature = sig;
        let expected = quote!(#sig).to_string();
        let expected = expected.replace("Self", "Host < 'h >");
        assert_eq!(rewrite_with_self(&tr, host.clone(), sig), expected);
    }
    let sig: Signature = parse_quote!(fn f(&self, x: fn(&Self) -> &u8));
    assert_eq!(
        rewrite_with_self(&tr, parse_quote!(__S), sig.clone()),
        quote!(#sig).to_string().replace("Self", "__S")
    );
    // A lifetime that a path hides cannot be named: the signature is left as it is (the Leaker
    // leaks it whole, or rejects the item).
    let sig: Signature = parse_quote!(fn f(&self, x: fn(::core::slice::Iter<u8>, Self) -> &u8));
    assert_eq!(
        rewrite_with_self(&tr, host.clone(), sig.clone()),
        quote!(#sig).to_string().replace("Self", "Host < 'h >")
    );
}

/// `Visitor::with_self()` docs: it names the elided lifetime with a fresh
/// `for<'__tl_e0>` binder. A binder that the function pointer has already gets the fresh lifetime
/// added to it (audit mutant `en_existing_binder_off`), the fresh name avoids the names of that
/// binder (`en_fresh_binder_off`), and the `Fn(..)` sugar inside an input keeps its own elision
/// (`en_name_fnsugar_not_own`).
#[test]
fn test_visitor_with_self_binder_and_nested_elision() {
    let tr: ItemTrait = parse_quote! { trait Tr { fn f(&self); } };
    let host: Type = parse_quote!(Host<'h>);
    let check = |sig: Signature, expected: Signature| {
        assert_eq!(
            rewrite_with_self(&tr, host.clone(), sig),
            quote!(#expected).to_string()
        );
    };
    // The binder that the function pointer has: the elided lifetime joins its lifetimes.
    check(
        parse_quote!(fn f(&self, x: for<'y> fn(&Self) -> &u8)),
        parse_quote!(fn f(&self, x: for<'y, '__tl_e0> fn(&'__tl_e0 Host<'h>) -> &'__tl_e0 u8)),
    );
    // The same for the sugar of a bound.
    check(
        parse_quote!(fn f<F>(&self, f: F) where F: for<'y> Fn(&Self) -> Option<&u8>),
        parse_quote!(
            fn f<F>(&self, f: F)
            where F: for<'y, '__tl_e0> Fn(&'__tl_e0 Host<'h>) -> Option<&'__tl_e0 u8>
        ),
    );
    // The name is not one of the binder: the next one is taken.
    check(
        parse_quote!(fn f(&self, x: for<'__tl_e0> fn(&Self) -> &u8)),
        parse_quote!(
            fn f(&self, x: for<'__tl_e0, '__tl_e1> fn(&'__tl_e1 Host<'h>) -> &'__tl_e1 u8)
        ),
    );
    check(
        parse_quote!(fn f(&self, x: for<'__tl_e0, '__tl_e1> fn(&Self) -> &u8)),
        parse_quote!(
            fn f(&self, x: for<'__tl_e0, '__tl_e1, '__tl_e2> fn(&'__tl_e2 Host<'h>) -> &'__tl_e2 u8)
        ),
    );
    // The sugar inside an input has its own elision: its `&u8` is not named, the function
    // pointer around it is.
    check(
        parse_quote!(fn f(&self, x: fn(&(Self, Box<dyn Fn(&u8)>)) -> &u8)),
        parse_quote!(
            fn f(&self, x: for<'__tl_e0> fn(&'__tl_e0 (Host<'h>, Box<dyn Fn(&u8)>)) -> &'__tl_e0 u8)
        ),
    );
    check(
        parse_quote!(fn f(&self, x: fn(&(Self, fn(&u8) -> &u8)) -> &u8)),
        parse_quote!(
            fn f(&self, x: for<'__tl_e0> fn(&'__tl_e0 (Host<'h>, fn(&u8) -> &u8)) -> &'__tl_e0 u8)
        ),
    );
}

#[test]
fn test_nested_items_define_their_own_self() {
    let (referrer, context) = referrer_of(parse_quote! {
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
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
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
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
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
    let mut leaker = type_leak::Leaker::new();
    let context = leaker.intern_mod(&m);
    let referrer = leaker.result().finish().unwrap();
    // Only the `Rel` of `m::inner` is leaked (in its context, which is keyed by the module
    // `inner`, and the module analyzed is another, as the one which contains it).
    assert_eq!(referrer.contexts().count(), 3);
    // The copy is the items of `m`, as a file of them, visited in the context of `m`.
    let items = m.content.unwrap().1;
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items,
    };
    referrer
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_file_mut(&mut file);
    // `Rel` of `m` is declared there, and the one of `m::inner` is rewritten.
    let expected: File = parse_quote! {
        pub struct Rel;
        pub struct A { x: Rel }
        pub mod inner { use super::Rel; pub struct C { y: P<0usize> } }
    };
    assert_eq!(quote!(#file).to_string(), quote!(#expected).to_string());
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

#[test]
fn the_associated_types_of_a_trait_with_outlives_are_in_its_top_level_context() {
    // The Referrer computes the context of a signature alone, with no trait around it (what a
    // macro that copies the methods does), so the top-level context of the trait carries the
    // outlives relations of its generic associated types that an entry depends on.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T> {
            type G<'b, U: 'b>
            where
                T: 'b,
                Self: 'b,
                U: ::core::clone::Clone;
            type H;
            fn m<'x>(&'x self, a: Rel<&'x Self::G<'x, T>>);
        }
    );
    let mut leaker = type_leak::Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let text = quote!(#referrer).to_string();
    // Only the outlives are kept (`U: Clone` is for the method to write), and a type with none
    // (`H`) is not.
    assert!(
        text.ends_with("type G < 'b , U > where U : 'b , T : 'b , Self : 'b ; ()"),
        "{}",
        text
    );
    // The entry has what the where clause of `G` requires, with its parameters substituted.
    assert!(
        text.contains(
            "< 'x , T : 'x > where Self :: G < 'x , T > : 'x ; Self : Tr < T > ; Self : 'x ;"
        ),
        "{}",
        text
    );
    // It is read back, and finds the entry in a signature alone.
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
    let TraitItem::Fn(method) = &tr.items[2] else {
        panic!()
    };
    let mut sig = method.sig.clone();
    parsed
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_signature_mut(&mut sig);
    assert_eq!(
        quote!(#sig).to_string(),
        quote!(fn m<'x>(&'x self, a: P<0usize>)).to_string()
    );
    // Without it (the trait with no items, as `enter_trait()` has it) the entry has the predicate
    // only.
    use syn::visit::Visit;
    let mut leaker = type_leak::Leaker::new();
    leaker.enter_trait(&parse_quote!(Tr), &tr.generics, true, |view| {
        let TraitItem::Fn(method) = &tr.items[2] else {
            panic!()
        };
        view.visit_signature(&method.sig);
    });
    let referrer = leaker.result().finish().unwrap();
    let text = quote!(#referrer).to_string();
    assert!(!text.contains("Self : 'x"), "{}", text);
    assert!(!text.contains("type G"), "{}", text);
}

#[test]
fn malformed_associated_types_in_a_context_are_errors() {
    let tag = tag();
    let ok = |text: &str| {
        let tokens: proc_macro2::TokenStream = format!("\"{}\" {}", tag, text).parse().unwrap();
        parse2::<Referrer>(tokens)
    };
    assert!(ok("Self : Tr ; type G < 'b > where Self : 'b ; (Rel)").is_ok());
    assert!(
        ok("Self : Tr ; type G < 'b > where Self : 'b ; type H where Self : 'static ; (Rel)")
            .is_ok()
    );
    let need_trait = "the associated types of a trait need `Self: Trait ;` before them";
    let need_outlives = "an associated type needs an outlives relation in its where clause";
    for (text, message) in [
        // The associated types are the ones of a trait: `Self` is the implementor.
        ("type G < 'b > where Self : 'b ; (Rel)", need_trait),
        (
            "Self = S ; type G < 'b > where Self : 'b ; (Rel)",
            need_trait,
        ),
        // A type with no outlives is not kept.
        ("Self : Tr ; type G < 'b > ; (Rel)", need_outlives),
        (
            "Self : Tr ; type G < 'b > where Self : Clone ; (Rel)",
            need_outlives,
        ),
        // They come before the module, and a module has no trait. (The error "which has no trait"
        // of a module that has associated types is not reachable: a module that has `Self : Tr ;`
        // for them has a `Self` already, which a module does not have.)
        (
            "Self : Tr ; mod { } ; type G < 'b > where Self : 'b ; (Rel)",
            "the context 0 is a module (`mod { .. }`), where `Self` has no meaning",
        ),
        ("type G < 'b > where Self : 'b ; mod { } ; ()", need_trait),
    ] {
        let err = ok(text)
            .err()
            .unwrap_or_else(|| panic!("accepted: {}", text));
        assert_eq!(err.to_string(), message, "{}", text);
    }
}

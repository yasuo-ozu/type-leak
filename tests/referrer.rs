//! Tests for rewriting items with `Referrer::visitor()`.

mod common;

use common::s;

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::{
    Context, ContextId, EntryId, ImplSelf, Leaker, LeakerResult, Projection, Referrer,
    SelfRequirement,
};

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
    let context = common::context_ids(&referrer).into_iter().next().unwrap();
    rewrite_with(referrer, context, item)
}

/// Rewrites `item`, a copy in the scope `context`, with `referrer`, replacing the `ix`-th leaked
/// type with `P<ix>`.
fn rewrite_with(referrer: Referrer, context: ContextId, mut item: Item) -> String {
    let mut visitor = common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>));
    visitor.visit_item_mut(&mut item);
    quote!(#item).to_string()
}

fn expected(item: Item) -> String {
    quote!(#item).to_string()
}

#[test]
fn the_visitor_rewrites_or_keeps_what_the_copy_declares() {
    // Each case is the serialized leaked types, the copied item, and the item with the `ix`-th
    // type replaced by `P<ix>`.
    let cases: Vec<(&str, proc_macro2::TokenStream, Item, Item)> = vec![
        (
            "where clauses are rewritten",
            quote!((RelA, RelB, RelC)),
            parse_quote! {
                trait Tr<T> where T: ::x::Conv<RelA> {
                    fn f<U>(&self, u: U) where U: ::x::Conv<RelB>;
                    type G<V> where V: ::x::Conv<RelC>;
                }
            },
            parse_quote! {
                trait Tr<T> where T: ::x::Conv<P<0usize>> {
                    fn f<U>(&self, u: U) where U: ::x::Conv<P<1usize>>;
                    type G<V> where V: ::x::Conv<P<2usize>>;
                }
            },
        ),
        (
            // Const params, impl generics and method generics shadow leaked types mentioning them.
            "method generics shadow leaked types",
            quote!((B, T)),
            parse_quote! {
                trait Tr {
                    fn a(&self) -> B;
                    fn g<const B: bool>(&self) -> ::x::Wrap<B>;
                    fn h<T>(&self, t: T) -> T;
                }
            },
            parse_quote! {
                trait Tr {
                    fn a(&self) -> P<0usize>;
                    fn g<const B: bool>(&self) -> ::x::Wrap<B>;
                    fn h<T>(&self, t: T) -> T;
                }
            },
        ),
        (
            "impl generics shadow leaked types",
            quote!((T, Wrap<U>)),
            parse_quote! {
                impl<T> X<T> {
                    fn f<U>(&self, t: T, u: Wrap<U>) -> T { t }
                }
            },
            parse_quote! {
                impl<T> X<T> {
                    fn f<U>(&self, t: T, u: Wrap<U>) -> T { t }
                }
            },
        ),
        (
            // All the parameters are in scope of each other's bounds.
            "the parameters are in scope of each other's bounds",
            quote!((T)),
            parse_quote! {
                struct S<A: ::x::Tr<T>, T>(A, T);
            },
            parse_quote! {
                struct S<A: ::x::Tr<T>, T>(A, T);
            },
        ),
        (
            // A type leaked together with local generics still matches where they are declared.
            "a type leaked with local generics matches where they are declared",
            quote!(<const B: bool>; (Wrap<B>)),
            parse_quote! {
                trait Tr {
                    fn f<const B: bool>(&self) -> Wrap<B>;
                }
            },
            parse_quote! {
                trait Tr {
                    fn f<const B: bool>(&self) -> P<0usize>;
                }
            },
        ),
        (
            "items in a block shadow leaked types",
            quote!((Wrap<u8>, RelType, Other)),
            parse_quote! {
                fn g() {
                    struct Wrap<T>(T);
                    use crate::other::RelType;
                    let x: Wrap<u8>;
                    let y: RelType;
                    let z: Other;
                }
            },
            parse_quote! {
                fn g() {
                    struct Wrap<T>(T);
                    use crate::other::RelType;
                    let x: Wrap<u8>;
                    let y: RelType;
                    let z: P<2usize>;
                }
            },
        ),
        (
            "items in a module shadow leaked types",
            quote!((RelType, Other)),
            parse_quote! {
                mod m {
                    pub struct RelType;
                    pub fn f(x: RelType, y: Other) {}
                }
            },
            parse_quote! {
                mod m {
                    pub struct RelType;
                    pub fn f(x: RelType, y: P<1usize>) {}
                }
            },
        ),
        (
            // `extern crate foo as Other;` declares `Other`, like a `use`.
            "an extern crate declares its name",
            quote!((Other, Rel)),
            parse_quote! {
                fn g() {
                    extern crate foo as Other;
                    let a: Other;
                    let b: Rel;
                }
            },
            parse_quote! {
                fn g() {
                    extern crate foo as Other;
                    let a: Other;
                    let b: P<1usize>;
                }
            },
        ),
        (
            // The generics of a trait alias are in scope in its bounds.
            "the generics of a trait alias are in scope in its bounds",
            quote!((Rel, T)),
            parse_quote! {
                trait Alias<T> = ::x::Conv<T, Rel>;
            },
            parse_quote! {
                trait Alias<T> = ::x::Conv<T, P<0usize>>;
            },
        ),
    ];
    for (name, types, item, rewritten) in cases {
        assert_eq!(rewrite(types, item), expected(rewritten), "{}", name);
    }
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
    let context = leaker.intern_struct(&item);
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
    leaker.intern_struct(&item);
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
    // (The previous format is rejected in `tests/scopes.rs`.)
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
    let context0 = leaker.intern_struct(&item0);
    let context1 = leaker.intern_struct(&item1);
    let referrer = leaker.result().finish().unwrap();
    // `RelA` uses no generics, so it is shared by the calls, but `Wrap<T>` and `Wrap<U>` are not.
    // Each call also has its generics as declared (the ids 2 and 4), which no entry has.
    assert_eq!(
        common::types(&referrer)
            .map(|ty| quote!(#ty).to_string())
            .collect::<Vec<_>>(),
        ["RelA", "Wrap < T >", "Wrap < U >"]
    );
    assert_eq!(
        (0..3)
            .map(|ix| common::entry_id(&referrer, ix).index())
            .collect::<Vec<_>>(),
        [0, 1, 3]
    );
    assert_eq!([context0, context1].map(|context| context.index()), [2, 4]);
    assert_eq!(common::context_ids(&referrer).len(), 5);

    // The contexts survive a round trip through `ToTokens` and `Parse`.
    let tokens = quote!(#referrer);
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 5 (RelA) < T > ; (Wrap < T >) declared < T > ; () \
             < U > ; (Wrap < U >) declared < U > ; ()",
            tag()
        )
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(common::entry_id(&parsed, 2).index(), 3);

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
    both.intern_struct(&item0);
    both.intern_struct(&item1);
    let (context, ()) = both.enter_generics(&parse_quote!(<T, U>), |_| ());
    assert_eq!(
        rewrite_with(both.result().finish().unwrap(), context, item.clone()),
        expected(parse_quote! { struct X { a: P<0usize>, b: P<2usize>, c: P<1usize> } })
    );
    let expected_ty: Type = parse_quote!(::p::Q<P<0usize>>);
    let ty = common::expand(
        &referrer,
        context0,
        parse_quote!(::p::Q<RelA>),
        |_, ix, _| parse_quote!(P<#ix>),
    );
    assert_eq!(quote!(#ty).to_string(), quote!(#expected_ty).to_string());

    // A call without types adds only its generics as declared (none here), a context with no
    // entry.
    let mut leaker = type_leak::Leaker::new();
    let empty: ItemStruct = parse_quote! { struct S; };
    leaker.intern_struct(&empty);
    leaker.intern_struct(&item0);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0).index(), 1);
    assert_eq!(common::context_ids(&referrer).len(), 4);
    assert_eq!(
        quote!(#referrer).to_string(),
        format!(
            "\"{}\" 4 declared < > ; () (RelA) < T > ; (Wrap < T >) declared < T > ; ()",
            tag()
        )
    );
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
}

#[test]
fn test_referrer_followed_by_tokens() {
    // A serialized Referrer can be followed by other tokens (e.g. in a macro invocation), whatever
    // its contexts are, also by tokens starting with `#`, which starts a group with an id.
    struct Followed(Referrer, proc_macro2::TokenStream);
    impl syn::parse::Parse for Followed {
        fn parse(input: syn::parse::ParseStream) -> Result<Self> {
            Ok(Followed(input.parse()?, input.parse()?))
        }
    }
    fn leak(f: impl FnOnce(&mut Leaker)) -> Referrer {
        let mut leaker = Leaker::new();
        f(&mut leaker);
        leaker.result().finish().unwrap()
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
    let tr: ItemTrait = parse_quote!(
        trait X<T> {
            fn f(&self) -> Rel<Self>;
            fn g(&self) -> Wrap<T>;
        }
    );
    let with_self: ItemStruct = parse_quote!(
        struct S<T> {
            a: RelS<Self>,
        }
    );
    let plain: ItemStruct = parse_quote!(
        struct P {
            a: RelP,
        }
    );
    let gaps: ItemStruct = parse_quote! { struct S<T, U> { a: Rel0, b: (Rel1, T), c: Wrap<U> } };
    let shapes: Vec<(&str, Referrer)> = vec![
        (
            "one context",
            leak(|leaker| {
                leaker.intern_struct(&item_a);
            }),
        ),
        (
            "two contexts",
            leak(|leaker| {
                leaker.intern_struct(&item_a);
                leaker.intern_struct(&item_b);
            }),
        ),
        (
            "contexts with Self",
            leak(|leaker| {
                leaker.intern_trait(&tr);
                leaker.intern_struct(&with_self);
                leaker.intern_struct(&plain);
            }),
        ),
        (
            "gaps in the ids",
            leak(|leaker| {
                leaker.intern_struct(&gaps);
            }),
        ),
        (
            "one group with an id",
            parse_str(&format!("\"{}\" # 3 (A)", tag())).unwrap(),
        ),
    ];
    let pound_group: proc_macro2::TokenStream = "# 5 (X)".parse().unwrap();
    let rests = [
        quote!(, (after)),
        quote!(after),
        quote!(
            struct Z;
        ),
        quote!(
            #[derive(Debug)]
            struct Z;
        ),
        pound_group,
    ];
    for (name, referrer) in &shapes {
        let tokens = quote!(#referrer);
        assert_eq!(&parse2::<Referrer>(tokens.clone()).unwrap(), referrer);
        for rest in &rests {
            let followed: Followed = parse2(quote!(#tokens #rest))
                .unwrap_or_else(|err| panic!("{} followed by {}: {}", name, rest, err));
            assert_eq!(&followed.0, referrer, "{} followed by {}", name, rest);
            assert_eq!(
                followed.1.to_string(),
                rest.to_string(),
                "{} followed by {}",
                name,
                rest
            );
        }
    }
}

#[test]
fn test_with_outer_generics() {
    // The Leaker is made from the trait, whose generics are those of the Leaker.
    let tr: ItemTrait = parse_quote! {
        trait MyTrait<T> {
            fn f(&self) -> Wrap<T>;
            fn g<const B: bool>(&self) -> Wrap2<B>;
        }
    };
    let mut leaker = type_leak::Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let rewrite = |item: &Item, outer_generics: bool| {
        let mut item = item.clone();
        let mut visitor = common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>));
        if outer_generics {
            visitor = visitor.with_outer_generics();
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
    let (result, context) = result_of(tr);
    (result.into_referrer(), context)
}

/// The result of the analysis of the trait (it has no error), and its top-level context.
fn result_of(tr: ItemTrait) -> (LeakerResult, ContextId) {
    let mut leaker = type_leak::Leaker::new();
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    result.clone().finish().unwrap();
    (result, context)
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
    let leak = |item: &str| {
        let mut leaker = type_leak::Leaker::new();
        common::intern_fns(&mut leaker, &Generics::default(), item.parse().unwrap());
        leaker.result().finish()
    };
    let referrer = leak("fn f(&self, x: Rel) -> Rel2;").unwrap();
    assert!(!referrer.is_empty());
    assert!(!referrer.has_self());
    // The bounds of a generic which no type uses do not make an entry name `Self`.
    let referrer = leak("fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Rel;").unwrap();
    assert!(!referrer.is_empty());
    assert!(!referrer.has_self());
}

#[test]
fn test_repeater_impl_replaces_self() {
    let (result, context) = result_of(parse_quote! {
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
    let referrer = result.clone().into_referrer();
    let marker: Ident = parse_quote!(__S);
    let repeater: Path = parse_quote!(crate::Repeater);
    let items: Vec<(String, String)> = result
        .repeater_impls(&Projection::new(
            repeater.clone(),
            7,
            ImplSelf::Param(marker.clone()),
        ))
        .unwrap()
        .into_iter()
        .map(|r| {
            // The implementor, `__S`, is the last parameter of the impl.
            let mut generics = r.generics;
            let mut params: Vec<GenericParam> = generics.params.into_iter().collect();
            assert!(matches!(params.pop(), Some(GenericParam::Type(p)) if p.ident == marker));
            generics.params = params.into_iter().collect();
            let ty = r.ty;
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
    assert!(common::types(&referrer).any(|ty| quote!(#ty).to_string() == "Rel < Self >"));
    let mut sig: Signature = parse_quote!(fn a(&self) -> Rel<Self>);
    common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>))
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

    // Without `with_self`, `Self` is kept, but for the entries that name it.
    let mut plain = sig.clone();
    common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>))
        .visit_signature_mut(&mut plain);
    let expected: Signature = parse_quote!(fn f(&self, x: P<0usize>, y: P<1usize>) -> Self::Out);
    assert_eq!(quote!(#plain).to_string(), quote!(#expected).to_string());

    // With it, the callback receives the type with `Self` replaced, the rest is replaced as well,
    // and the receiver is untouched.
    let mut rewritten = sig.clone();
    let mut received = Vec::new();
    common::visitor(&referrer, context, |ty, ix, _| {
        received.push((quote!(#ty).to_string(), ix.index()));
        // Not rewritten again, though it mentions `Self`.
        parse_quote!(<Marker<Self> as R<#ix>>::Type)
    })
    .with_self(parse_quote!(__S))
    .visit_signature_mut(&mut rewritten);
    assert_eq!(
        received,
        [
            (quote!(Rel<__S>).to_string(), 0),
            (quote!(::p::W<__S>).to_string(), 1)
        ],
    );
    let expected: Signature = parse_quote!(
        fn f(&self, x: <Marker<Self> as R<0usize>>::Type, y: <Marker<Self> as R<1usize>>::Type) -> <__S>::Out
    );
    assert_eq!(
        quote!(#rewritten).to_string(),
        quote!(#expected).to_string()
    );

    // The receiver with an explicit type is kept as well.
    let mut sig: Signature = parse_quote!(fn f(self: Box<Self>) -> Rel<Self>);
    common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>))
        .with_self(parse_quote!(__S))
        .visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn f(self: Box<Self>) -> P<0usize>);
    assert_eq!(quote!(#sig).to_string(), quote!(#expected).to_string());
}

/// Rewrites `sig` as a copy in the scope of `tr`, with `Self` written as `self_ty`.
fn rewrite_with_self(tr: &ItemTrait, self_ty: Type, sig: Signature) -> String {
    let (referrer, context) = referrer_of(tr.clone());
    let mut sig = sig;
    common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>))
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

/// `with_self()` writes the host for `Self` in the tokens of a macro that the copy keeps as written
/// (the bound of a generic whose type names `Self` is an error of the Leaker, `UnprovableBound`,
/// and the Referrer of `into_referrer()` leaves it so): inside groups too, and nothing without it.
#[test]
fn with_self_writes_the_host_in_the_tokens_of_a_macro() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T: ::p::Tr<::p::m!(Self, [Self; 2], (u8, (Self,)))>> {
            fn f(&self, t: T);
        }
    );
    let mut leaker = Leaker::with_rule(type_leak::Rules::new().primitive().prelude());
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(
        result.errors()[0].reason,
        type_leak::Reason::UnprovableBound
    );
    let referrer = result.into_referrer();
    assert_eq!(referrer.iter().count(), 0);
    let written = |host: Option<Type>| -> ItemTrait {
        let mut copy = tr.clone();
        let visitor = common::visitor(&referrer, context, |ty, _, _| ty);
        match host {
            Some(host) => visitor.with_self(host).visit_item_trait_mut(&mut copy),
            None => {
                let mut visitor = visitor;
                visitor.visit_item_trait_mut(&mut copy)
            }
        }
        copy
    };
    assert_eq!(s(&written(None)), s(&tr));
    let hosts: [Type; 3] = [
        parse_quote!(__S),
        parse_quote!(::p::S<'z, u8>),
        parse_quote!(&'static Host),
    ];
    for host in hosts {
        let expected: ItemTrait = parse_quote!(
            trait Tr<T: ::p::Tr<::p::m!(#host, [#host; 2], (u8, (#host,)))>> {
                fn f(&self, t: T);
            }
        );
        assert_eq!(
            s(&written(Some(host.clone()))),
            s(&expected),
            "{}",
            s(&host)
        );
    }
}

/// A written bound whose inputs show the elided lifetime that its output takes (`Fn(&Rel<Self>) ->
/// &u8`) gets it named if the path through the Repeater that the visitor writes has a lifetime in
/// it (the host of `with_self()` is `Host<'h>`), in a trait object, an `impl Trait`, a bound of a
/// generic and a predicate; not if it has none (`Host`) or the lifetime is named already.
#[test]
fn test_visitor_names_the_elided_lifetime_of_a_bound_where_the_path_names_a_lifetime() {
    let tr: ItemTrait = parse_quote!(
        trait Tr {
            fn a(&self, f: Box<dyn Fn(&Rel<Self>) -> &u8>);
            fn b(&self, f: impl Fn(&Rel<Self>) -> &u8);
            fn c<F: Fn(&Rel<Self>) -> &u8>(&self, f: F)
            where
                F: Fn(&Rel<Self>, u8) -> &u8;
            fn d(&self, f: Box<dyn Fn(&Rel<Self>) -> &u8 + 'static>);
            fn e(&self, f: Box<dyn for<'x> Fn(&'x Rel<Self>) -> &'x u8>);
            fn f(&self, f: Box<dyn Fn(&Rel<Self>) -> u8>);
        }
    );
    let mut leaker = Leaker::with_rule(type_leak::Rules::new().primitive().prelude());
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    let projection = common::projection(parse_quote!(R), 0);
    let key = projection.key(common::entry_id_at(&referrer, 0));
    let copy = |host: Type| -> ItemTrait {
        let mut copy = tr.clone();
        referrer
            .visitor(context, &projection)
            .with_self(host)
            .unwrap()
            .visit_item_trait_mut(&mut copy);
        copy
    };

    // The host has a lifetime: the path names it in the inputs.
    let with: Type = parse_quote!(<Host<'h> as R<(), #key>>::Type);
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn a(&self, f: Box<dyn for<'__tl_e0> Fn(&'__tl_e0 #with) -> &'__tl_e0 u8>);
            fn b(&self, f: impl for<'__tl_e0> Fn(&'__tl_e0 #with) -> &'__tl_e0 u8);
            fn c<F: for<'__tl_e0> Fn(&'__tl_e0 #with) -> &'__tl_e0 u8>(&self, f: F)
            where F: for<'__tl_e0> Fn(&'__tl_e0 #with, u8) -> &'__tl_e0 u8;
            fn d(&self, f: Box<dyn for<'__tl_e0> Fn(&'__tl_e0 #with) -> &'__tl_e0 u8 + 'static>);
            fn e(&self, f: Box<dyn for<'x> Fn(&'x #with) -> &'x u8>);
            fn f(&self, f: Box<dyn Fn(&#with) -> u8>);
        }
    );
    assert_eq!(s(&copy(parse_quote!(Host<'h>))), s(&expected));

    // No lifetime is added: the bounds are as they were.
    let without: Type = parse_quote!(<Host as R<(), #key>>::Type);
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn a(&self, f: Box<dyn Fn(&#without) -> &u8>);
            fn b(&self, f: impl Fn(&#without) -> &u8);
            fn c<F: Fn(&#without) -> &u8>(&self, f: F) where F: Fn(&#without, u8) -> &u8;
            fn d(&self, f: Box<dyn Fn(&#without) -> &u8 + 'static>);
            fn e(&self, f: Box<dyn for<'x> Fn(&'x #without) -> &'x u8>);
            fn f(&self, f: Box<dyn Fn(&#without) -> u8>);
        }
    );
    assert_eq!(s(&copy(parse_quote!(Host))), s(&expected));
}

/// `Visitor::with_self()` fails for a trait object with no lifetime bound only where an entry has
/// `Self: 'a` in its context: that is the one path `Projection::of()` refuses for it.
#[test]
fn with_self_refuses_an_unbounded_trait_object_only_where_an_entry_needs_a_bound() {
    let projection = common::projection(parse_quote!(Repeater), 7);
    let bare: Type = parse_quote!(dyn Shape);
    let bounded: Type = parse_quote!(dyn Shape + 'static);
    // `Self: 'a` is in the context of the entry.
    let (referrer, context) = referrer_of(parse_quote! {
        trait Tr<'a> where Self: 'a { fn f(&self) -> Rel<'a, Self>; }
    });
    let error = |host: &Type| {
        referrer
            .visitor(context, &projection)
            .with_self(host.clone())
            .err()
    };
    assert!(matches!(
        error(&bare),
        Some(type_leak::EncodeError::UnboundedTraitObject(ty)) if *ty == bare
    ));
    assert!(error(&bounded).is_none());
    // No entry has it, and a bound of `'static` is no argument of the path either.
    for tr in [
        parse_quote!(
            trait Tr {
                fn f(&self) -> Rel<Self>;
            }
        ),
        parse_quote!(
            trait Tr
            where
                Self: 'static,
            {
                fn f(&self) -> Rel<Self>;
            }
        ),
    ] {
        let (referrer, context) = referrer_of(tr);
        assert!(referrer
            .visitor(context, &projection)
            .with_self(bare.clone())
            .is_ok());
    }
}

/// A projection that `Projection::of()` refuses for an entry leaves its type as written, for rustc to
/// report: the visitor has no error to give (`ImplSelf::Marker` has no impl for an entry that needs
/// `Self`, and the call that wrote the impls fails with `NeedsSelf` for it).
#[test]
fn a_projection_that_fails_leaves_the_type_as_written() {
    let (referrer, context) = referrer_of(parse_quote! {
        trait Tr { fn f(&self, a: Rel<Self>, b: Other) -> Rel2; }
    });
    let marker = Projection::new(
        parse_quote!(Repeater),
        7,
        ImplSelf::Marker(parse_quote!(Marker)),
    );
    let mut copy: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, a: Rel<Self>, b: Other) -> Rel2;
        }
    );
    referrer
        .visitor(context, &marker)
        .visit_item_trait_mut(&mut copy);
    // `Rel<Self>` needs `Self`: it is left. The others are for the marker.
    let (other, rel2) = (
        marker.key(common::entry_id_at(&referrer, 1)),
        marker.key(common::entry_id_at(&referrer, 2)),
    );
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn f(
                &self,
                a: Rel<Self>,
                b: <Marker as Repeater<(), #other>>::Type
            ) -> <Marker as Repeater<(), #rel2>>::Type;
        }
    );
    assert_eq!(s(&copy), s(&expected));
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
    common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>))
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
    // item of a block of a function copied alone: their own `Self` is not the one of the entry.
    let item: ItemFn = parse_quote! {
        fn f<T>() {
            struct S(Rel<Self>);
            enum E { A(Rel<Self>) }
            trait Other { fn t() -> Self::Out; fn u() -> Rel<Self>; }
            union U { a: Rel<Self> }
        }
    };
    for with_self in [true, false] {
        let mut copy = item.clone();
        let mut visitor = common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>));
        if with_self {
            visitor = visitor.with_self(parse_quote!(__S));
        }
        visitor.visit_item_fn_mut(&mut copy);
        assert_eq!(
            quote!(#copy).to_string(),
            quote!(#item).to_string(),
            "with_self: {}",
            with_self
        );
    }
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
    let context = leaker.intern_trait(&tr);
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
    common::visitor(&parsed, context, |_, ix, _| parse_quote!(P<#ix>))
        .visit_signature_mut(&mut sig);
    assert_eq!(
        quote!(#sig).to_string(),
        quote!(fn m<'x>(&'x self, a: P<0usize>)).to_string()
    );
    // Without it (the trait with no items, as `enter_trait()` has it) the entry has the predicate
    // only.
    use syn::visit::Visit;
    let mut leaker = type_leak::Leaker::new();
    leaker.enter_trait(&parse_quote!(Tr), &tr.generics, |view| {
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

/// What follows the tag is refused with a clear error: a context declared twice, the associated
/// types of a trait, `Self` as a call, and the module words of earlier versions of the format.
#[test]
fn malformed_serialized_referrers_are_errors() {
    let tag = tag();
    let parse = |text: &str| parse_str::<Referrer>(&format!("\"{}\" {}", tag, text));
    let integer = "expected integer literal";
    let declared = "the context 0 is declared already: expected its entries `( .. )`";
    let need_trait = "the associated types of a trait need `Self: Trait ;` before them";
    let need_outlives = "an associated type needs an outlives relation in its where clause";
    let max_id = format!("# {} (A)", usize::MAX);
    let cases: &[(&str, &str)] = &[
        // A context is no module any more: `mod` is no word of a declaration.
        ("mod 0 ; (Rel)", "expected parentheses"),
        // A context that is written again has its entries only.
        ("2 (A) # 0 < T > ; (B)", declared),
        ("2 (A) # 0 Self ; (B)", declared),
        // Not a number.
        ("# (A)", integer),
        ("# x (A)", integer),
        (
            "# 1u8 (A)",
            "a context id is an integer, with no suffix or `usize`",
        ),
        // A group is missing.
        ("2 # 3 (A)", "unexpected end of input, expected parentheses"),
        // The largest id cannot be followed by another one.
        (max_id.as_str(), "the id of the context is too large"),
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
    ];
    for (text, message) in cases {
        let err = parse(text)
            .err()
            .unwrap_or_else(|| panic!("accepted: {}", text));
        assert_eq!(err.to_string(), *message, "{}", text);
    }
    // `Self` has no meaning outside the scopes which declare it, so there is no `Self` of a call
    // (`Self = # 2 ;` was one, in the first format). `Self = ` is followed by a type, and `#`
    // starts none (syn lists the tokens that do).
    let err = parse("Self = # 2 ; (Rel < Self >)").unwrap_err();
    assert!(err.to_string().starts_with("expected one of: "), "{}", err);

    // The consistent ones are fine: the associated types of a trait.
    for text in [
        "Self : Tr ; type G < 'b > where Self : 'b ; (Rel)",
        "Self : Tr ; type G < 'b > where Self : 'b ; type H where Self : 'static ; (Rel)",
    ] {
        let parsed = parse(text);
        assert!(parsed.is_ok(), "{}: {:?}", text, parsed.err());
    }
}

/// A visitor borrows the Referrer: its closure may use the Referrer it was made from, two visitors
/// may share one Referrer, and the Referrer is there after them, with no clone.
#[test]
fn a_visitor_borrows_the_referrer_so_its_closure_may_use_it() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self, t: T) -> Wrap<T>;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let (mut with_args, mut plain) = (tr.clone(), tr.clone());
    let repeater: Path = parse_quote!(crate::Repeater);
    let mut first = common::visitor(&referrer, top_level, |_, ix, context| {
        // The Referrer is usable here, as the context is.
        assert!(common::bucket_self(&referrer, 7, ix).is_some());
        common::projection(repeater.clone(), 7)
            .of(&parse_quote!(Self), context, ix)
            .unwrap()
    });
    let mut second = common::visitor(&referrer, top_level, |_, ix, _| parse_quote!(P<#ix>));
    first.visit_item_trait_mut(&mut with_args);
    second.visit_item_trait_mut(&mut plain);
    let key = common::key(7, common::entry_id_at(&referrer, 0));
    let expected: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self, t: T) -> <Self as crate::Repeater<(T,), #key>>::Type;
        }
    );
    assert_eq!(with_args, expected);
    let expected: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self, t: T) -> P<0usize>;
        }
    );
    assert_eq!(plain, expected);
    assert_eq!(referrer.iter().count(), 1);
}

// ── the context of the visitor ───────────────────────────────────────────────────────────────
//
// `Referrer::visitor(context, ..)`, on a whole item or on one type (`common::expand()`): the
// context is the scope of the copy, and the visited syntax is inside it. The visitor takes from the
// context the generics in scope and what `Self` means.

/// Rewrites each leaked type to `P<index>`.
fn replace(_: Type, ix: EntryId, _: &Context) -> Type {
    let ix = proc_macro2::Literal::usize_unsuffixed(ix.index());
    parse_quote!(P<#ix>)
}

/// The names that the items of a file declare hide the entries that name them.
#[test]
fn the_names_that_a_file_declares_hide_the_entries_that_name_them() {
    let mut leaker = Leaker::new();
    let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel));
    });
    let referrer = leaker.result().finish().unwrap();
    let mut file: File = parse_quote!(
        struct Rel;
        struct S(Rel);
    );
    common::visitor(&referrer, scope, replace).visit_file_mut(&mut file);
    assert_eq!(
        s(&file),
        s(&parse_quote!(
            struct Rel;
            struct S(Rel);
        ) as &File)
    );
    // Without the declaration, the entry is found.
    let mut file: File = parse_quote!(
        struct S(Rel);
    );
    common::visitor(&referrer, scope, replace).visit_file_mut(&mut file);
    assert_eq!(
        s(&file),
        s(&parse_quote!(
            struct S(P<0>);
        ) as &File)
    );
}

// ── `Self` comes from the context ────────────────────────────────────────────────────────────

#[test]
fn self_is_taken_from_the_context() {
    // The same leaked type in two traits with the same generics: `Rel<Self>` is an entry of each,
    // for the implementors of its own trait. The context says which.
    let tr1: ItemTrait = parse_quote!(
        trait Tr1<T> {
            fn f(&self, t: T) -> Rel<Self>;
        }
    );
    let tr2: ItemTrait = parse_quote!(
        trait Tr2<T> {
            fn f(&self, t: T) -> Rel<Self>;
        }
    );
    // And a struct, where `Self` is the type.
    let st: ItemStruct = parse_quote!(
        struct St<T>(Rel<Self>, T);
    );
    let mut leaker = Leaker::new();
    let c1 = leaker.intern_trait(&tr1);
    let c2 = leaker.intern_trait(&tr2);
    let c3 = leaker.intern_struct(&st);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 3);
    // The scopes differ by what `Self` is, as an entry of each names it.
    assert!(c1 != c2 && c2 != c3 && c1 != c3);
    // Each call has one entry, in the order of the calls, in a context that is not the call's.
    let ix_of = |context: ContextId| {
        [c1, c2, c3]
            .iter()
            .position(|&top| top == context)
            .expect("the top-level context of a call")
    };
    for ix in 0..3 {
        assert!(![c1, c2, c3].contains(&common::entry_id(&referrer, ix)));
    }
    assert!(matches!(
        common::entry(&referrer, ix_of(c1)).self_ty.as_ref(),
        Some(SelfRequirement::Bound(_))
    ));
    assert!(matches!(
        common::entry(&referrer, ix_of(c3)).self_ty.as_ref(),
        Some(SelfRequirement::Type(_))
    ));
    let expected = |context: ContextId| {
        let ix = proc_macro2::Literal::usize_unsuffixed(ix_of(context));
        s(&parse_quote!(fn f(&self, t: T) -> P<#ix>) as &Signature)
    };
    for (context, name) in [(c1, "Tr1"), (c2, "Tr2")] {
        let mut sig: Signature = parse_quote!(fn f(&self, t: T) -> Rel<Self>);
        common::visitor(&referrer, context, replace).visit_signature_mut(&mut sig);
        assert_eq!(s(&sig), expected(context), "{}", name);
    }
    // The copy of the struct is rewritten with its own entry, whatever the order of the entries.
    let mut copy = st.clone();
    common::visitor(&referrer, c3, replace).visit_item_struct_mut(&mut copy);
    let ix = proc_macro2::Literal::usize_unsuffixed(ix_of(c3));
    let expected: ItemStruct = parse_quote!(struct St<T>(P<#ix>, T););
    assert_eq!(s(&copy), s(&expected));
    // `expand()` takes `Self` from the context too.
    for context in [c1, c2, c3] {
        let ty = common::expand(&referrer, context, parse_quote!(Rel<Self>), replace);
        let ix = proc_macro2::Literal::usize_unsuffixed(ix_of(context));
        assert_eq!(s(&ty), s(&parse_quote!(P<#ix>) as &Type));
    }
    // A scope that declares no `Self` finds no entry that names it.
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr1);
    let (plain, ()) =
        leaker.enter_generics(&parse_quote!(<T>), |view| view.visit_type(&parse_quote!(T)));
    let referrer = leaker.result().finish().unwrap();
    let ty = common::expand(&referrer, plain, parse_quote!(Rel<Self>), replace);
    assert_eq!(s(&ty), "Rel < Self >");
}

/// What `Self` of the entry that is the type `ty` (as `to_string()` has it) means.
fn self_of_entry(referrer: &Referrer, ty: &str) -> String {
    let ix = common::types(referrer)
        .position(|entry| s(entry) == ty)
        .unwrap_or_else(|| panic!("no entry {}", ty));
    match common::entry(referrer, ix).self_ty.as_ref() {
        None => "-".to_string(),
        Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
        Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
        Some(other) => format!("{:?}", other),
    }
}

#[test]
fn a_nested_function_or_type_alias_keeps_the_self_around() {
    // The Leaker has the `Self` of the scope in a function or a type alias nested in it: `Self` is
    // E0401 there, so the one around is its only meaning. Only a struct, an enum, a union, a trait
    // and an impl have a `Self` of their own, and the Referrer must agree.
    let mut leaker = Leaker::new();
    let (context, ()) = leaker.enter_impl(&Generics::default(), &parse_quote!(W), None, |v| {
        for item in [
            parse_quote!(
                fn f(x: Rel<Self>) {}
            ),
            parse_quote!(
                type A = Rel2<Self>;
            ),
            parse_quote!(
                struct V {
                    a: Rel3<Self>,
                }
            ),
        ] {
            let item: Item = item;
            v.visit_item(&item);
        }
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(self_of_entry(&referrer, "Rel < Self >"), "Self = W");
    assert_eq!(self_of_entry(&referrer, "Rel2 < Self >"), "Self = W");
    // The struct has its own `Self`, which the entry names.
    assert_eq!(self_of_entry(&referrer, "Rel3 < Self >"), "Self = V");
    let index = |ty: &str| {
        common::types(&referrer)
            .position(|entry| s(entry) == ty)
            .unwrap()
    };
    let (rel, rel2, rel3) = (
        index("Rel < Self >"),
        index("Rel2 < Self >"),
        index("Rel3 < Self >"),
    );
    let tokens = quote!(#referrer);
    let read_back: Referrer = syn::parse2(tokens).unwrap();
    assert_eq!(read_back, referrer);
    for referrer in [referrer.clone(), read_back] {
        // In a method of an impl for `W`, the nested function and type alias are found. The struct
        // nested in the impl has `Self` of its own: the same `Rel<Self>` there is not the entry
        // (`V2`). The entry of its own `Self` (`V`) is left as written, as there is no hook.
        let mut copy: ItemImpl = parse_quote!(
            impl Foo for W {
                fn m() {
                    fn f(x: Rel<Self>) {}
                    type A = Rel2<Self>;
                    struct V2 {
                        a: Rel<Self>,
                    }
                    struct V {
                        a: Rel3<Self>,
                    }
                }
            }
        );
        let copy_before_hook = copy.clone();
        common::visitor(&referrer, context, replace).visit_item_impl_mut(&mut copy);
        let (rel, rel2, rel3) = (
            proc_macro2::Literal::usize_unsuffixed(rel),
            proc_macro2::Literal::usize_unsuffixed(rel2),
            proc_macro2::Literal::usize_unsuffixed(rel3),
        );
        let expected: ItemImpl = parse_quote!(
            impl Foo for W {
                fn m() {
                    fn f(x: P<#rel>) {}
                    type A = P<#rel2>;
                    struct V2 {
                        a: Rel<Self>,
                    }
                    struct V {
                        a: Rel3<Self>,
                    }
                }
            }
        );
        assert_eq!(s(&copy), s(&expected));
        // With a hook it is written for the type that the hook returns.
        let mut hooked = copy_before_hook.clone();
        common::visitor(&referrer, context, replace)
            .with_nested_self(|_| Some(parse_quote!(Nested)))
            .visit_item_impl_mut(&mut hooked);
        let expected: ItemImpl = parse_quote!(
            impl Foo for W {
                fn m() {
                    fn f(x: P<#rel>) {}
                    type A = P<#rel2>;
                    struct V2 {
                        a: Rel<Self>,
                    }
                    struct V {
                        a: P<#rel3>,
                    }
                }
            }
        );
        assert_eq!(s(&hooked), s(&expected));
        // Not in a method: a function that is not one, which has them nested.
        let mut copy: ItemFn = parse_quote!(
            fn outer() {
                fn f(x: Rel<Self>) {}
                type A = Rel2<Self>;
                fn g() {
                    type B = Rel2<Self>;
                }
            }
        );
        common::visitor(&referrer, context, replace).visit_item_fn_mut(&mut copy);
        let expected: ItemFn = parse_quote!(
            fn outer() {
                fn f(x: P<#rel>) {}
                type A = P<#rel2>;
                fn g() {
                    type B = P<#rel2>;
                }
            }
        );
        assert_eq!(s(&copy), s(&expected));
    }
}

#[test]
fn a_nested_type_alias_in_a_trait_keeps_the_self_of_the_trait() {
    // The implementors of the trait are `Self` in a nested function or type alias too.
    let mut leaker = Leaker::new();
    let (context, ()) = leaker.enter_trait(&parse_quote!(Tr), &Generics::default(), |v| {
        for item in [
            parse_quote!(
                type A = Rel<Self>;
            ),
            parse_quote!(
                fn f(x: Rel2<Self>) {}
            ),
            // The trait alias has no `Self` of its own either.
            parse_quote!(
                trait Al = ::p::Bar<Rel3<Self>>;
            ),
        ] {
            let item: Item = item;
            v.visit_item(&item);
        }
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(self_of_entry(&referrer, "Rel < Self >"), "Self: Tr");
    let mut copy: ItemTrait = parse_quote!(
        trait Tr {
            fn m() {
                type A = Rel<Self>;
                fn f(x: Rel2<Self>) {}
                trait Al = ::p::Bar<Rel3<Self>>;
                trait Inner {
                    type B = Rel<Self>;
                }
            }
        }
    );
    let ix = |ty: &str| {
        proc_macro2::Literal::usize_unsuffixed(
            common::types(&referrer)
                .position(|entry| s(entry) == ty)
                .unwrap(),
        )
    };
    let (rel, rel2, rel3) = (ix("Rel < Self >"), ix("Rel2 < Self >"), ix("Rel3 < Self >"));
    common::visitor(&referrer, context, replace).visit_item_trait_mut(&mut copy);
    // The nested trait has its own `Self`.
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn m() {
                type A = P<#rel>;
                fn f(x: P<#rel2>) {}
                trait Al = ::p::Bar<P<#rel3>>;
                trait Inner {
                    type B = Rel<Self>;
                }
            }
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

// ── the generics come from the context ───────────────────────────────────────────────────────

#[test]
fn the_generics_are_taken_from_the_context() {
    // `Wrap<T>` is an entry of each trait, which declares `T` differently. The signature of the
    // copy does not declare `T`: the context does, so it is looked up with the right declaration.
    // (It took the exact lookup `into_visitor_with_root()` with the generics of the trait.)
    let tr1: ItemTrait = parse_quote!(
        trait Tr1<T: ::core::clone::Clone> {
            fn f(&self, x: Wrap<T>);
        }
    );
    let tr2: ItemTrait = parse_quote!(
        trait Tr2<T: ::core::marker::Copy> {
            fn f(&self, x: Wrap<T>);
        }
    );
    let mut leaker = Leaker::new();
    let c1 = leaker.intern_trait(&tr1);
    let c2 = leaker.intern_trait(&tr2);
    assert_ne!(c1, c2);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    // The entry of each call is the one with its own generics: `Wrap<T>` is in a context that is
    // not the call's (the generics as declared).
    let entry = |context: ContextId| {
        let ix = [c1, c2].iter().position(|&top| top == context).unwrap();
        assert!(![c1, c2].contains(&common::entry_id(&referrer, ix)));
        proc_macro2::Literal::usize_unsuffixed(ix)
    };
    // (Each context finds its own entry: see `matching_with_the_generics_of_each_scope` in
    // `tests/requirements.rs`.)
    // A generic that neither the context nor the visited syntax declares is not the Leaker's
    // (every undeclared generic was taken to be, when the visitor did not know the scope).
    let mut sig: Signature = parse_quote!(fn f(&self, x: Wrap<U>));
    common::visitor(&referrer, c1, replace).visit_signature_mut(&mut sig);
    assert_eq!(
        s(&sig),
        s(&parse_quote!(fn f(&self, x: Wrap<U>)) as &Signature)
    );
    // The copy may declare the generics again, as the entry has them: the item that declares
    // exactly those of the context is the scope itself.
    let mut item = tr2.clone();
    common::visitor(&referrer, c2, replace).visit_item_trait_mut(&mut item);
    let ix = entry(c2);
    let expected: ItemTrait = parse_quote!(
        trait Tr2<T: ::core::marker::Copy> { fn f(&self, x: P<#ix>); }
    );
    assert_eq!(s(&item), s(&expected));
    // (With other bounds, they are declared by the copy and are another `T`, unless the outermost
    // generics are the context's: see `test_with_outer_generics`.)
}

#[test]
fn a_bound_on_self_alone_need_not_be_written_by_the_copy() {
    // The entry of `&'a Rel2<Self>` has the `where Self: Sized` of its method. A copy where
    // `Self` is a type parameter (`with_self()`) cannot write it, so the entry is found without
    // it: the only difference is a bound that names no generic. An entry with exactly the
    // context of the copy is preferred.
    let only: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'a>(&self, x: &'a Rel2<Self>)
            where
                Self: ::core::marker::Sized;
        }
    );
    let both: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'a>(&self, x: &'a Rel2<Self>)
            where
                Self: ::core::marker::Sized;
            fn g<'a>(&self, x: &'a Rel2<Self>);
        }
    );
    let copy = |tr: &ItemTrait, item: ItemFn| {
        let mut leaker = Leaker::new();
        let context = leaker.intern_trait(tr);
        let referrer = leaker.result().finish().unwrap();
        let predicate = |ix: usize| common::entry(&referrer, ix).generics.where_clause.is_some();
        let mut item = item;
        common::visitor(&referrer, context, replace)
            .with_self(parse_quote!(S))
            .visit_item_fn_mut(&mut item);
        let text = s(&item);
        // The index of the entry which replaced the type, and whether it has the predicate.
        let ix = (0..referrer.iter().count()).find(|ix| text.contains(&format!("P < {} >", ix)));
        (ix.map(predicate), text)
    };
    let without: ItemFn = parse_quote!(
        fn free<'a, S: Tr + Sized>(x: &'a Rel2<Self>) {}
    );
    let with: ItemFn = parse_quote!(
        fn free<'a, S: Tr + Sized>(x: &'a Rel2<Self>)
        where
            Self: ::core::marker::Sized,
        {
        }
    );
    // Found with or without the predicate when it is the only entry.
    assert_eq!(copy(&only, without.clone()).0, Some(true));
    assert_eq!(copy(&only, with.clone()).0, Some(true));
    // Both entries: the exact one is the one found.
    assert_eq!(copy(&both, without.clone()).0, Some(false));
    assert_eq!(copy(&both, with.clone()).0, Some(true));
    // The entry is not found if the copy writes another bound on `Self`, or if the extra bound
    // names a generic.
    let other: ItemFn = parse_quote!(
        fn free<'a, S: Tr + Sized>(x: &'a Rel2<Self>)
        where
            Self: ::core::fmt::Debug,
        {
        }
    );
    assert_eq!(copy(&only, other).0, None);
    let named: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f<'a>(&self, x: &'a Rel2<Self>)
            where
                Self: ::core::convert::Into<T>;
        }
    );
    let item: ItemFn = parse_quote!(
        fn free<'a, S: Tr<T> + Sized>(x: &'a Rel2<Self>) {}
    );
    assert_eq!(copy(&named, item).0, None);
    let item: ItemFn = parse_quote!(
        fn free<'a, S: Tr<T> + Sized>(x: &'a Rel2<Self>)
        where
            Self: ::core::convert::Into<T>,
        {
        }
    );
    assert_eq!(copy(&named, item).0, Some(true));
}

#[test]
fn an_entry_context_is_a_scope_too() {
    // Any context of the Referrer is accepted: the context of an entry has the generics the entry
    // uses (and what `Self` is, if it names it).
    let tr: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self, t: T) -> (Wrap<T>, Rel);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    for ix in 0..referrer.iter().count() {
        let ty = common::types(&referrer).nth(ix).unwrap().clone();
        let rewritten = common::expand(&referrer, common::entry_id(&referrer, ix), ty, replace);
        let ix = proc_macro2::Literal::usize_unsuffixed(ix);
        assert_eq!(s(&rewritten), s(&parse_quote!(P<#ix>) as &Type));
    }
}

// ── the shapes a downstream macro visits ─────────────────────────────────────────────────────

/// A trait with supertraits, associated types with bounds and a default, and a method, all naming
/// leaked types.
fn trait_with_bounds() -> ItemTrait {
    parse_quote! {
        pub trait Tr<T>: ::p::Sup<Rel> + ::p::Sup2<Rel2<T>> {
            type A: ::p::Bound<Rel3>;
            type B<'x>: ::p::Bound<Rel4<T>> = Rel5;
            fn f(&self, x: Rel6) -> Rel7<T>;
        }
    }
}

/// The docs of `Referrer::visitor()`: a macro rewrites the trait it leaked (`visit_item_trait_mut`
/// visits the generics, the supertraits and the items), or goes through the parts itself (as
/// newer-type does: the bounds of the supertraits, the associated types, the signatures of the
/// methods), and gets the same copy. Audit mutants `D3` (the bounds of an associated type are not
/// visited) and `D4` (the supertraits of a visited `ItemTrait` are not).
#[test]
fn a_trait_is_rewritten_whole_and_part_by_part() {
    let tr = trait_with_bounds();
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    // The entries, by context and tokens: those without generics, then those with `<T>`.
    assert_eq!(
        common::types(&referrer).map(s).collect::<Vec<_>>(),
        [
            "Rel",
            "Rel3",
            "Rel5",
            "Rel6",
            "Rel2 < T >",
            "Rel4 < T >",
            "Rel7 < T >"
        ]
    );
    let expected: ItemTrait = parse_quote! {
        pub trait Tr<T>: ::p::Sup<P<0>> + ::p::Sup2<P<4>> {
            type A: ::p::Bound<P<1>>;
            type B<'x>: ::p::Bound<P<5>> = P<2>;
            fn f(&self, x: P<3>) -> P<6>;
        }
    };
    let mut whole = tr.clone();
    common::visitor(&referrer, context, replace).visit_item_trait_mut(&mut whole);
    assert_eq!(s(&whole), s(&expected));

    // Part by part: the supertraits, the associated types and the signatures, as a macro that
    // keeps the trait and replaces what it chooses to does.
    let mut parts = tr.clone();
    let mut visitor = common::visitor(&referrer, context, replace);
    for bound in parts.supertraits.iter_mut() {
        visitor.visit_type_param_bound_mut(bound);
    }
    for item in parts.items.iter_mut() {
        match item {
            TraitItem::Fn(item) => visitor.visit_signature_mut(&mut item.sig),
            TraitItem::Type(item) => visitor.visit_trait_item_type_mut(item),
            _ => unreachable!(),
        }
    }
    assert_eq!(s(&parts), s(&expected));
}

/// The docs of `Referrer::visitor()`: a signature visited by itself is a nested item: its generics
/// are its own, unless `with_outer_generics()` says they are the context's. So the
/// method `f<T: Clone>` has a `T` of its own, and `Wrap<T>` is not the entry of the trait's `T`,
/// while with the flag it is. Audit mutant `D5` (`visit_signature_mut` treated as no item).
#[test]
fn a_signature_visited_by_itself_is_a_nested_item() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> {
            fn f(&self) -> Wrap<T>;
            fn g<const B: bool>(&self) -> Wrap2<B>;
        }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    for (text, expected, as_leakers) in [
        // A `T` of the method with a bound is not the trait's `T`.
        (
            "fn f<T: ::core::clone::Clone>(&self) -> Wrap<T>",
            "fn f<T: ::core::clone::Clone>(&self) -> Wrap<T>",
            false,
        ),
        // Unless the visitor takes the generics of the outermost item as the context's.
        (
            "fn f<T: ::core::clone::Clone>(&self) -> Wrap<T>",
            "fn f<T: ::core::clone::Clone>(&self) -> P<0>",
            true,
        ),
        // The generics of a method that the trait does not have are declared.
        (
            "fn g<const B: bool>(&self) -> Wrap2<B>",
            "fn g<const B: bool>(&self) -> P<1>",
            false,
        ),
        // The entry is found where the signature declares `B` as the entry has it, and not where
        // `B` is undeclared or declared otherwise.
        ("fn g(&self) -> Wrap2<B>", "fn g(&self) -> Wrap2<B>", false),
        (
            "fn g<const B: u8>(&self) -> Wrap2<B>",
            "fn g<const B: u8>(&self) -> Wrap2<B>",
            false,
        ),
    ] {
        let mut sig: Signature = parse_str(text).unwrap();
        let expected: Signature = parse_str(expected).unwrap();
        let mut visitor = common::visitor(&referrer, context, replace);
        if as_leakers {
            visitor = visitor.with_outer_generics();
        }
        visitor.visit_signature_mut(&mut sig);
        assert_eq!(
            s(&sig),
            s(&expected),
            "{} (as leakers: {})",
            text,
            as_leakers
        );
    }
}

/// The groups follow the entries in the order of their indices: a context whose entries are not
/// together is written again, with its entries only, and the entries keep their indices.
#[test]
fn the_entries_of_a_context_may_be_in_several_groups() {
    let tag = tag();
    let text = format!("\"{}\" 4 (A) (B) # 0 (C) # 3 (D)", tag);
    let referrer: Referrer = parse_str(&text).unwrap();
    let types: Vec<String> = common::types(&referrer).map(s).collect();
    assert_eq!(types, ["A", "B", "C", "D"]);
    let contexts: Vec<usize> = (0..4)
        .map(|ix| common::entry_id(&referrer, ix).index())
        .collect();
    assert_eq!(contexts, [0, 1, 0, 3]);
    // Written back as it was read, and equal to what is read from it.
    let written = quote!(#referrer);
    assert_eq!(written.to_string(), text);
    assert_eq!(parse2::<Referrer>(written).unwrap(), referrer);
    // A group has no `#` if its id is the one after that of the group before: the third group is
    // for the context 1, which the first group declares.
    let text = format!("\"{}\" 3 # 1 (A) # 0 (B) (C)", tag);
    let referrer: Referrer = parse_str(&text).unwrap();
    let contexts: Vec<usize> = (0..3)
        .map(|ix| common::entry_id(&referrer, ix).index())
        .collect();
    assert_eq!(contexts, [1, 0, 1]);
    let written = quote!(#referrer);
    assert_eq!(written.to_string(), text);
    assert_eq!(parse2::<Referrer>(written).unwrap(), referrer);
    // A context with no entry is a group of its own, before the first group of a larger id.
    let referrer: Referrer = parse_str(&format!("\"{}\" 3 () (A) # 5 (B)", tag)).unwrap();
    assert_eq!(common::context_ids(&referrer).len(), 3);
    assert_eq!(parse2::<Referrer>(quote!(#referrer)).unwrap(), referrer);
}

/// The paths through the Repeater that the visitor wrote in `node`, with the entry of each, which
/// the test reads from its `KEY`.
fn written_paths(node: &impl template_quote::ToTokens) -> Vec<(EntryId, Type)> {
    struct Collect(Vec<(EntryId, Type)>);
    impl VisitMut for Collect {
        fn visit_type_mut(&mut self, ty: &mut Type) {
            match common::written_entry(ty) {
                Some(entry) => self.0.push((entry, ty.clone())),
                None => syn::visit_mut::visit_type_mut(self, ty),
            }
        }
    }
    let mut file: File = parse2(quote!(#node)).unwrap();
    let mut collect = Collect(Vec::new());
    collect.visit_file_mut(&mut file);
    collect.0
}

/// Checks that each path in `written` is what `Projection::of()` writes for the entry that its `KEY`
/// names, from the context of that entry, with `Self`, and that the entries are the `count` of the
/// Referrer, each with a context of its own.
fn assert_paths_of_entries(
    referrer: &Referrer,
    projection: &Projection,
    written: &[(EntryId, Type)],
    count: usize,
) {
    assert_eq!(written.len(), count);
    for (ix, ty) in written {
        let (_, _, context) = referrer.iter().find(|(id, _, _)| id == ix).unwrap();
        let expected = projection.of(&parse_quote!(Self), context, *ix).unwrap();
        assert_eq!(ty, &expected, "{}", ix);
    }
    let distinct: std::collections::BTreeSet<String> =
        written.iter().map(|(ix, _)| ix.to_string()).collect();
    assert_eq!(distinct.len(), count);
    let contexts: std::collections::BTreeSet<String> = written
        .iter()
        .map(|(ix, _)| {
            let (_, _, context) = referrer.iter().find(|(id, _, _)| id == ix).unwrap();
            format!("{:?}", context)
        })
        .collect();
    assert_eq!(contexts.len(), count);
}

/// The path that the visitor writes for an entry is made from the context of that entry (what
/// `iter()` gives for it), with a context of its own for each entry, in a trait and in nested
/// items.
#[test]
fn the_path_written_for_an_entry_is_made_from_the_context_of_the_entry() {
    let projection = common::projection(parse_quote!(Repeater), 7);
    // Three entries, with the contexts `<'a, T: 'a>`, `<U: Clone>` and none.
    let item: ItemTrait = parse_quote!(
        trait Tr<'a, T: 'a> {
            fn f(&self, x: &'a Rel<T>) -> Wrap<u8>;
            fn g<U: ::core::clone::Clone>(&self, u: Rel2<U>);
        }
    );
    let mut leaker = Leaker::with_rule(type_leak::Rules::new().primitive().prelude());
    let top = leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 3);
    let mut copy = item.clone();
    referrer
        .visitor(top, &projection)
        .visit_item_trait_mut(&mut copy);
    assert_paths_of_entries(&referrer, &projection, &written_paths(&copy), 3);

    // The entries of nested items: two have the `Self` of their item, and the third has not. The
    // hook returns `Self`, so the paths are the ones of `Self` in each.
    let module: ItemMod = parse_quote!(
        mod m {
            pub struct A<T: ::core::convert::AsRef<Self>>(pub Wrap<T>);
            pub struct B(pub Rel<Self>);
            pub struct C<U: ::core::clone::Clone>(pub Rel2<U>);
        }
    );
    let mut leaker = Leaker::with_rule(type_leak::Rules::new().primitive().prelude());
    let (top, ()) = leaker.enter_generics(&Generics::default(), |view| {
        for item in &module.content.as_ref().unwrap().1 {
            view.visit_item(item);
        }
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 3);
    let mut copy = module.clone();
    referrer
        .visitor(top, &projection)
        .with_nested_self(|_| Some(parse_quote!(Self)))
        .visit_item_mod_mut(&mut copy);
    assert_paths_of_entries(&referrer, &projection, &written_paths(&copy), 3);
}

// ── a `mod` in a copy ────────────────────────────────────────────────────────────────────────

/// `item` as it is printed.
fn shown(item: Item) -> String {
    quote!(#item).to_string()
}

/// Rewrites `item`, a copy in the scope `context`, replacing the `ix`-th entry with `P<ix>`.
fn rewrite_in_scope(referrer: &Referrer, context: ContextId, mut item: Item) -> String {
    common::visitor(referrer, context, |_, ix, _| parse_quote!(P<#ix>)).visit_item_mut(&mut item);
    quote!(#item).to_string()
}

/// `Visitor` docs: the items of a `mod` that a copy holds are nested items, so their generics are
/// local even if the visitor is told that the outermost generics are those of the context
/// (`with_outer_generics()`). Audit mutants `vis_item_depth_module`, `vis_mod_item_not_nested` and
/// `vis_mod_item_self_kept`.
#[test]
fn the_items_of_a_mod_in_a_copy_are_nested_items() {
    // The Leaker analyzes the struct nested in a scope; the copy holds it in a `mod`.
    let mut leaker = Leaker::new();
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&parse_quote!(
            pub struct S<T>(Wrap<T>);
        ));
    });
    let referrer = leaker.result().finish().unwrap();
    let mut item: Item = parse_quote!(
        mod m {
            pub struct S<T>(Wrap<T>);
        }
    );
    common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>))
        .with_outer_generics()
        .visit_item_mut(&mut item);
    assert_eq!(
        shown(item),
        shown(parse_quote!(
            mod m {
                pub struct S<T>(P<0usize>);
            }
        ))
    );
}

/// `Visitor` docs: an item declared in a block hides the entry that names it, and the entry is
/// replaced where nothing hides it.
#[test]
fn a_block_item_hides_the_entry_that_names_it() {
    let mut leaker = Leaker::new();
    let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&parse_quote!(
            pub fn f(a: Rel) {}
        ));
    });
    let referrer = leaker.result().finish().unwrap();
    let out = rewrite_in_scope(
        &referrer,
        scope,
        parse_quote!(
            pub fn f(a: Rel) {
                struct Rel;
                let b: Rel = Rel;
            }
        ),
    );
    // `a` is replaced, `b` is the local struct.
    assert_eq!(
        out,
        shown(parse_quote!(
            pub fn f(a: P<0usize>) {
                struct Rel;
                let b: Rel = Rel;
            }
        ))
    );
}

/// `Self` has no meaning in a `mod` item that a copy holds, even inside the scope of a trait that
/// declares it, so the entry that names the `Self` of the trait is not found in it. Audit mutant
/// `vis_mod_item_self_kept` (the `Self` of the scope around stays in a `mod` item).
#[test]
fn self_has_no_meaning_in_a_mod_item_of_a_scope_which_declares_it() {
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&parse_quote!(
        trait Tr {
            fn f(&self) -> Rel<Self>;
        }
    ));
    let referrer = leaker.result().finish().unwrap();
    // The trait itself declares `Self`, and the entry is found.
    let expected: Item = parse_quote!(
        trait Tr {
            fn f(&self) -> P<0usize>;
        }
    );
    assert_eq!(
        rewrite_in_scope(
            &referrer,
            context,
            parse_quote!(
                trait Tr {
                    fn f(&self) -> Rel<Self>;
                }
            )
        ),
        quote!(#expected).to_string()
    );
    // In a `mod` it is not the `Self` of the trait: nothing is replaced.
    for src in [
        "mod inner { type X = Rel<Self>; }",
        "mod inner { const C: Option<Rel<Self>> = None; }",
        "mod inner { fn g() -> Rel<Self> { loop {} } }",
        "mod inner { static S: Rel<Self> = loop {}; }",
    ] {
        let item: Item = parse_str(src).unwrap();
        assert_eq!(
            rewrite_in_scope(&referrer, context, item.clone()),
            quote!(#item).to_string(),
            "{}",
            src
        );
    }
}

/// A nested item's own name hides none of its entries: the container declares it (`mod w { trait
/// Tr .. }`), but inside the item the name is the item's own, so `R1<<Self as Tr>::A>` is replaced
/// as in a copy of the item by itself. A declaration inside the item hides again: the local `trait
/// Tr` of the body is another trait, so the type there that names it is left as written.
#[test]
fn a_nested_items_own_name_hides_none_of_its_entries() {
    // The body names the local `Tr` in `Option<..>` (the owner's shape) and in the type of the
    // signature, which is an entry: the Leaker reads no body, the Visitor visits it.
    let body_types: [Type; 2] = [
        parse_quote!(Option<<Self as Tr>::A>),
        parse_quote!(R1<<Self as Tr>::A>),
    ];
    for body in body_types {
        let tr: ItemTrait = parse_quote!(
            pub trait Tr {
                type A;
                fn f(&self) -> R1<<Self as Tr>::A> {
                    trait Tr {
                        type A;
                    }
                    let x: #body = None;
                    todo!()
                }
            }
        );
        let projection = Projection::new(
            parse_quote!(Repeater),
            7,
            ImplSelf::Param(parse_quote!(__S)),
        );
        let key = projection.key(parse_quote!(0));
        let expected = |output: Type| -> ItemTrait {
            parse_quote!(
                pub trait Tr {
                    type A;
                    fn f(&self) -> #output {
                        trait Tr {
                            type A;
                        }
                        let x: #body = None;
                        todo!()
                    }
                }
            )
        };

        // A nested copy: the trait nested in a scope, copied in `mod w`, whose own `Self` is the
        // host that the closure gives.
        let mut leaker = Leaker::new();
        let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Trait(tr.clone()))
        });
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(referrer.iter().count(), 1, "{}", s(&body));
        let mut copy: ItemMod = parse_quote!(mod w { #tr });
        referrer
            .visitor(scope, &projection)
            .with_nested_self(|requirement| match requirement {
                SelfRequirement::Bound(_) => Some(parse_quote!(Host)),
                _ => None,
            })
            .visit_item_mod_mut(&mut copy);
        let inner = expected(parse_quote!(<Host as Repeater<(), #key>>::Type));
        let expected_mod: ItemMod = parse_quote!(mod w { #inner });
        assert_eq!(s(&copy), s(&expected_mod), "nested, {}", s(&body));

        // A copy of the trait by itself: the same, with `Self`.
        let mut leaker = Leaker::new();
        let top = leaker.intern_trait(&tr);
        let referrer = leaker.result().finish().unwrap();
        let mut copy = tr.clone();
        referrer
            .visitor(top, &projection)
            .visit_item_trait_mut(&mut copy);
        let expected_top = expected(parse_quote!(<Self as Repeater<(), #key>>::Type));
        assert_eq!(s(&copy), s(&expected_top), "top level, {}", s(&body));
    }
}

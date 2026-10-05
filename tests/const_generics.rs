//! Tests for const generics and method-local generics in the Leaker.

mod common;

use common::finish_as_strings;
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::{Leaker, Referrer};

/// Leaks the types of `item` (in the context 0).
fn leak(item: &ItemTrait) -> Referrer {
    let mut leaker = Leaker::new();
    leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(item))
        .unwrap();
    leaker.finish()
}

/// Returns the leaked types with their local generics (with the where clause), by index.
fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter_with_generics()
        .map(|(ty, generics)| {
            let wc = &generics.where_clause;
            (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
        })
        .collect()
}

fn entry(ty: Type, generics: Generics) -> (String, String) {
    let wc = &generics.where_clause;
    (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
}

/// Rewrites `item` with `referrer`, replacing the `ix`-th leaked type with `P<ix>`.
fn rewrite(referrer: Referrer, item: ItemTrait) -> String {
    let mut item = Item::Trait(item);
    referrer
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut item);
    quote!(#item).to_string()
}

fn expected(item: ItemTrait) -> String {
    quote!(#item).to_string()
}

#[test]
fn test_const_generic_param_in_expr() {
    // The Leaker's own const generic parameter used as an array length is context-free.
    let item: ItemStruct = parse_quote! {
        struct S<const N: usize> { a: [RelType; N], b: [::core::primitive::u8; N] }
    };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    // The element is interned (the innermost minimum cut), and the array is rebuilt around it.
    let expected: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_const_generic_param_ty_not_interned() {
    // The type of a const parameter cannot be a projection, so it must not be interned.
    let item: ItemStruct = parse_quote! { struct S<const N: usize, const B: bool>; };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());

    let item: ItemTrait = parse_quote! {
        trait MyTrait { fn f<const M: ::core::primitive::u8>(&self); }
    };
    let (mut leaker, _) = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());

    // A type alias cannot be leaked as the type of a const parameter.
    assert!(Leaker::from_struct(&parse_quote! { struct S<const N: MyIdx>; }).is_err());
}

#[test]
fn test_method_local_generics_not_interned() {
    // Types mentioning method-local generics cannot be interned as a whole because the
    // Repeater's implementation cannot refer to them; their components are interned instead.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<const M: usize>(&self) -> [RelType1; M];
            fn g<'b>(&self, x: &'b RelType2);
            fn h<S>(&self, x: (S, RelType3));
            fn i<const M: usize>(&self) -> [::core::primitive::u8; M];
        }
    };
    let (mut leaker, _) = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([
            "RelType1".to_string(),
            "RelType2".to_string(),
            "RelType3".to_string()
        ])
    );
}

#[test]
fn test_local_generics_leaked() {
    // A relative type constructor applied to local generics cannot be decomposed, so it is
    // interned together with the local generics it depends on (including those used in their
    // bounds and where predicates), in a canonical form (bounds moved to the parameters).
    let item: ItemTrait = parse_quote! {
        trait MyTrait<T> {
            fn f<const B: bool>(&self) -> MyWrap<B>;
            fn g<'b, S: ::core::iter::Iterator<Item = &'b U>, U>(&self) -> MyVec<S::Item>
            where
                U: ::core::clone::Clone,
                T: ::core::marker::Copy;
            fn h<'c>(&self, x: MyHolder<'c>, y: [RelType; 2]);
        }
    };
    let (mut leaker, _) = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    let referrer = leaker.finish();
    let actual: BTreeSet<(String, String)> = referrer
        .iter_with_generics()
        .map(|(ty, generics)| {
            let wc = &generics.where_clause;
            (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
        })
        .collect();
    let expected = |ty: Type, generics: Generics| {
        let wc = &generics.where_clause;
        (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
    };
    let g_generics: Generics =
        parse_quote!(<'b, S: ::core::iter::Iterator<Item = &'b U>, U: ::core::clone::Clone>);
    assert_eq!(
        actual,
        BTreeSet::from([
            expected(parse_quote!(MyWrap<B>), parse_quote!(<const B: bool>)),
            expected(parse_quote!(MyVec<S::Item>), g_generics),
            expected(parse_quote!(MyHolder<'c>), parse_quote!(<'c>)),
            expected(parse_quote!(RelType), Generics::default()),
        ])
    );

    // The local generics survive a round trip through `ToTokens` and `Parse`.
    let tokens = quote!(#referrer);
    let parsed: type_leak::Referrer = parse_quote!(#tokens);
    assert_eq!(parsed, referrer);
}

#[test]
fn test_local_generics_keep_self() {
    // Bounds of local generics are kept as they are, including `Self`, which refers to the
    // implementor of the Leaker trait.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> MyWrap<T>;
        }
    };
    let (mut leaker, _) = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    let referrer = leaker.finish();
    let (ty, generics) = referrer.iter_with_generics().next().unwrap();
    assert_eq!(quote!(#ty).to_string(), quote!(MyWrap<T>).to_string());
    let expected: Generics = parse_quote!(<T: ::core::cmp::PartialEq<Self>>);
    assert_eq!(quote!(#generics).to_string(), quote!(#expected).to_string());
}

#[test]
fn test_same_type_with_different_local_generics() {
    // The same tokens depending on different local generics are separate entries, and each copy
    // is rewritten to its own.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g<const B: u8>(&self) -> Wrap<B>;
        }
    };
    let referrer = leak(&item);
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Wrap<B>), parse_quote!(<const B: bool>)),
            entry(parse_quote!(Wrap<B>), parse_quote!(<const B: u8>)),
        ]
    );
    assert_eq!(
        rewrite(referrer, item),
        expected(parse_quote! {
            trait MyTrait {
                fn f<const B: bool>(&self) -> P<0usize>;
                fn g<const B: u8>(&self) -> P<1usize>;
            }
        })
    );
}

#[test]
fn test_same_local_generics_share_an_entry() {
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g<const B: bool>(&self, x: Wrap<B>);
        }
    };
    let referrer = leak(&item);
    assert_eq!(
        entries(&referrer),
        [entry(parse_quote!(Wrap<B>), parse_quote!(<const B: bool>))]
    );
    assert_eq!(
        rewrite(referrer, item),
        expected(parse_quote! {
            trait MyTrait {
                fn f<const B: bool>(&self) -> P<0usize>;
                fn g<const B: bool>(&self, x: P<0usize>);
            }
        })
    );
}

#[test]
fn test_type_with_and_without_local_generics() {
    // Here `B` is a type in `g`, which the Leaker's item does not declare.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g(&self) -> Wrap<B>;
        }
    };
    let referrer = leak(&item);
    // The contexts are numbered in the order of their first appearance.
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Wrap<B>), parse_quote!(<const B: bool>)),
            entry(parse_quote!(Wrap<B>), Generics::default()),
        ]
    );
    assert_eq!(
        rewrite(referrer, item),
        expected(parse_quote! {
            trait MyTrait {
                fn f<const B: bool>(&self) -> P<0usize>;
                fn g(&self) -> P<1usize>;
            }
        })
    );
}

#[test]
fn test_equivalent_local_generics_share_an_entry() {
    // The order of the parameters and the bounds, and the place of the bounds, do not matter.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<T: ::core::fmt::Debug + ::core::clone::Clone, U>(&self) -> Wrap<T, U>;
            fn g<U, T>(&self) -> Wrap<T, U> where T: ::core::clone::Clone + ::core::fmt::Debug;
        }
    };
    let referrer = leak(&item);
    assert_eq!(
        entries(&referrer),
        [entry(
            parse_quote!(Wrap<T, U>),
            parse_quote!(<T: ::core::clone::Clone + ::core::fmt::Debug, U>)
        )]
    );
    assert_eq!(
        rewrite(referrer, item),
        expected(parse_quote! {
            trait MyTrait {
                fn f<T: ::core::fmt::Debug + ::core::clone::Clone, U>(&self) -> P<0usize>;
                fn g<U, T>(&self) -> P<0usize> where T: ::core::clone::Clone + ::core::fmt::Debug;
            }
        })
    );
}

#[test]
fn test_local_generics_include_those_in_bounds() {
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<T, U: ::core::cmp::PartialEq<T>>(&self) -> Wrap<U>;
        }
    };
    let referrer = leak(&item);
    assert_eq!(
        entries(&referrer),
        [entry(
            parse_quote!(Wrap<U>),
            parse_quote!(<T, U: ::core::cmp::PartialEq<T>>)
        )]
    );
    assert_eq!(
        rewrite(referrer, item),
        expected(parse_quote! {
            trait MyTrait {
                fn f<T, U: ::core::cmp::PartialEq<T>>(&self) -> P<0usize>;
            }
        })
    );
}

#[test]
fn test_binder_in_type() {
    // The lifetimes of a `for<..>` binder are local generics inside it, so `&'b Rel<L>` is
    // decomposed, and `Rel<L>` is leaked with `L` only.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<L>(&self, x: ::std::boxed::Box<dyn for<'b> ::p::F<&'b Rel<L>>>);
        }
    };
    let referrer = leak(&item);
    assert_eq!(
        entries(&referrer),
        [entry(parse_quote!(Rel<L>), parse_quote!(<L>))]
    );
    assert_eq!(
        rewrite(referrer, item),
        expected(parse_quote! {
            trait MyTrait {
                fn f<L>(&self, x: ::std::boxed::Box<dyn for<'b> ::p::F<&'b P<0usize>>>);
            }
        })
    );
}

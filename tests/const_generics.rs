//! Tests for const generics and method-local generics in the Leaker.

mod common;

use common::finish_as_strings;
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::Leaker;

#[test]
fn test_const_generic_param_in_expr() {
    // The Leaker's own const generic parameter used as an array length is context-free.
    let item: ItemStruct = parse_quote! {
        struct S<const N: usize> { a: [RelType; N], b: [::core::primitive::u8; N] }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!([RelType; N]);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_const_generic_param_ty_not_interned() {
    // The type of a const parameter cannot be a projection, so it must not be interned.
    let item: ItemStruct = parse_quote! { struct S<const N: usize, const B: bool>; };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());

    let item: ItemTrait = parse_quote! {
        trait MyTrait { fn f<const M: ::core::primitive::u8>(&self); }
    };
    let mut leaker = Leaker::from_trait(&item).unwrap();
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
    let mut leaker = Leaker::from_trait(&item).unwrap();
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
    // bounds and where predicates).
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
    let mut leaker = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    let referrer = leaker.finish();
    let actual: BTreeSet<(String, String)> = referrer
        .iter_with_local_generics()
        .map(|(ty, generics)| {
            let wc = &generics.where_clause;
            (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
        })
        .collect();
    let expected = |ty: Type, generics: Generics| {
        let wc = &generics.where_clause;
        (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
    };
    let mut g_generics: Generics = parse_quote!(<'b, S: ::core::iter::Iterator<Item = &'b U>, U>);
    g_generics.where_clause = Some(parse_quote!(where U: ::core::clone::Clone));
    assert_eq!(
        actual,
        BTreeSet::from([
            expected(parse_quote!(MyWrap<B>), parse_quote!(<const B: bool>)),
            expected(parse_quote!(MyVec<S::Item>), g_generics),
            expected(parse_quote!(MyHolder<'c>), parse_quote!(<'c>)),
            expected(parse_quote!([RelType; 2]), Generics::default()),
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
    let mut leaker = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    let referrer = leaker.finish();
    let (ty, generics) = referrer.iter_with_local_generics().next().unwrap();
    assert_eq!(quote!(#ty).to_string(), quote!(MyWrap<T>).to_string());
    let expected: Generics = parse_quote!(<T: ::core::cmp::PartialEq<Self>>);
    assert_eq!(quote!(#generics).to_string(), quote!(#expected).to_string());
}

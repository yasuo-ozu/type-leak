//! Tests for `Leaker::check()` and `Leaker::allowed_paths`.

use syn::*;
use type_leak::{CheckResult, Leaker};

/// Classifies a `CheckResult` (as the variants of the former `CheckResult` enum).
fn kind(result: &CheckResult) -> &'static str {
    match (result.must.is_empty(), result.impossible.is_empty()) {
        (true, true) => "Neutral",
        (false, true) if result.is_own => "MustIntern",
        (false, true) => "MustInternOrInherit",
        (true, false) => "MustNotIntern",
        (false, false) => "Conflict",
    }
}

#[test]
fn test_check() {
    let test_struct: ItemStruct = parse_quote! {
        struct MyStruct<'l1, const N1: usize, T1, T2, T3>;
    };
    let leaker = Leaker::new();
    let generics = &test_struct.generics;
    assert_eq!(kind(&leaker.check(&parse_quote!(T1), generics)), "Neutral");
    assert_eq!(
        kind(&leaker.check(&parse_quote!((T1, T2)), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!([T1; 123]), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!([T1; N]), generics)),
        "MustIntern"
    ); // constant
    assert_eq!(
        kind(&leaker.check(&parse_quote!([T1; ::path::to::N]), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!([(); N1]), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(some::path), generics)),
        "MustIntern"
    ); // relavice path
    assert_eq!(
        kind(&leaker.check(&parse_quote!(::some::path), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(
            &parse_quote!(<T1 as ::abs::path::MyTrait<T2>>::Ty),
            generics
        )),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(
            &parse_quote!(<MyType as ::abs::path::MyTrait<T2>>::Ty),
            generics
        )),
        "MustInternOrInherit"
    ); // relative path
    assert_eq!(
        kind(&leaker.check(
            &parse_quote!(<::path::to::MyType as MyTrait<T2>>::Ty),
            generics
        )),
        "MustIntern"
    ); // relative path
    assert_eq!(
        kind(&leaker.check(
            &parse_quote!(<::path::to::MyType as ::path::to::MyTrait<(MyType2,)>>::Ty),
            generics
        )),
        "MustInternOrInherit"
    ); // relative path in an argument
    assert_eq!(
        kind(&leaker.check(
            &parse_quote!(<::path::to::MyType as ::path::to::MyTrait<MyType2>>::Ty),
            generics
        )),
        "MustIntern"
    ); // a bare identifier as an argument may be a const, so the type is interned as a whole
    assert_eq!(kind(&leaker.check(&parse_quote!(()), generics)), "Neutral");
    assert_eq!(
        kind(&leaker.check(&parse_quote!(&'l1 ()), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(&'static ()), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(&'my_lifetime ()), generics)),
        "MustIntern"
    ); // lifetime
    assert_eq!(
        kind(&leaker.check(&parse_quote!(impl ::path::to::MyTrait), generics)),
        "MustNotIntern"
    ); // impl Trait
    assert_eq!(
        kind(&leaker.check(&parse_quote!(impl MyTrait), generics)),
        "Conflict"
    ); // impl Trait and relative path
    assert_eq!(
        kind(&leaker.check(&parse_quote!(dyn ::path::to::MyTrait), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(Self), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(&Self), generics)),
        "MustNotIntern"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(&'a Self), generics)),
        "MustIntern"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(::core::fmt::Formatter<'_>), generics)),
        "MustNotIntern"
    ); // relative path
}

#[test]
fn test_check_self_no_internable() {
    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<'l1, const N1: usize, T1, T2, T3>;
    );

    // Test with self_ty_can_be_interned = true (default behavior)
    let leaker_true = Leaker::new();
    let generics_true = &test_struct.generics;
    assert_eq!(
        kind(&leaker_true.check(&parse_quote!(Self), generics_true)),
        "Neutral"
    ); // Self should be allowed by default

    // Test with self_ty_can_be_interned = false
    let mut leaker_false = Leaker::new();
    // Set the flag to false to disallow Self types
    leaker_false.self_ty_can_be_interned = false;
    let generics_false = &test_struct.generics;
    assert_eq!(
        kind(&leaker_false.check(&parse_quote!(Self), generics_false)),
        "MustNotIntern"
    ); // Self should be rejected when flag is false

    // Test with complex Self usage
    assert_eq!(
        kind(&leaker_false.check(&parse_quote!(&Self), generics_false)),
        "MustNotIntern"
    ); // Reference to Self should also be rejected

    // Test that non-Self types are unaffected by the flag
    assert_eq!(
        kind(&leaker_false.check(&parse_quote!(T1), generics_false)),
        "Neutral"
    ); // Generic type should still work
}

#[test]
fn test_check_allows_dollar_crate() {
    let mut leaker = Leaker::new();
    leaker.allowed_paths.push(parse_quote!(crate));

    let item: ItemTrait = parse_quote!(
        pub trait Parse<Atom>: ::core::marker::Sized {
            type Error: crate::error::Error;
            fn parse(
                stream: impl crate::parse::into_parse_stream::IntoParseStream<Atom = Atom>,
            ) -> ::core::result::Result<Self, Self::Error>;
        }
    );
    leaker
        .intern_with(&item.generics, |visitor| {
            visitor.visit_item_trait(&item);
        })
        .unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_check_allowed_paths() {
    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<'l1, const N1: usize, T1>;
    );
    let mut leaker = Leaker::new();
    let generics = &test_struct.generics;

    leaker.allowed_paths.push(parse_quote!(some::path));

    assert_eq!(
        kind(&leaker.check(&parse_quote!(some::path), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(some::path::Type), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(other::path), generics)),
        "MustIntern"
    );

    leaker.allowed_paths.push(parse_quote!(::abs::path));
    assert_eq!(
        kind(&leaker.check(&parse_quote!(::abs::path::Type), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(abs::path::Type), generics)),
        "MustIntern"
    );
}

#[test]
fn test_check_allowed_paths_with_generic_args() {
    use type_leak::LeakerConfig;

    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<T1>;
    );
    let generics = &test_struct.generics;
    let mut config = LeakerConfig::new();
    config.allow_std_prelude().allow_primitive();
    let leaker = Leaker::from_config(config);

    // An allowed path with generic arguments is still allowed.
    assert_eq!(
        kind(&leaker.check(&parse_quote!(Option<u8>), generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&leaker.check(&parse_quote!(Vec<Option<T1>>), generics)),
        "Neutral"
    );
    // Its generic arguments are still checked.
    assert_eq!(
        kind(&leaker.check(&parse_quote!(Option<(RelType,)>), generics)),
        "MustInternOrInherit"
    );
    // A bare identifier as an argument may be a const, so the type is interned as a whole.
    assert_eq!(
        kind(&leaker.check(&parse_quote!(Option<RelType>), generics)),
        "MustIntern"
    );

    // Trait bounds with generic arguments on allowed paths are accepted, and relative types in
    // them are leaked.
    let mut config = LeakerConfig::new();
    config.allow_std_prelude().allow_primitive();
    let mut leaker = Leaker::from_config(config);
    let item: ItemTrait = parse_quote! {
        trait MyTrait: Into<u8> {
            fn f<I: Iterator<Item = u8>>(&self, i: I);
            fn g<I: Iterator<Item = RelType>>(&self, i: I);
        }
    };
    leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(&item))
        .unwrap();
    leaker.reduce_roots();
    let leaked: std::collections::BTreeSet<String> = leaker
        .finish()
        .iter()
        .map(|t| template_quote::quote!(#t).to_string())
        .collect();
    assert_eq!(leaked, vec!["RelType".to_string()].into_iter().collect());
}

#[test]
fn test_check_conflict() {
    let generics: Generics = parse_quote!(<T>);
    let leaker = Leaker::new();
    let strings = |tokens: Vec<proc_macro2::TokenStream>| -> Vec<String> {
        tokens.iter().map(|t| t.to_string()).collect()
    };
    let conflict = |ty: Type| {
        let c = leaker.check(&ty, &generics);
        (strings(c.must), strings(c.impossible), c.is_own)
    };
    let v = |s: &[&str]| -> Vec<String> { s.iter().map(|s| s.to_string()).collect() };

    // The type's own path depends on the type context: it cannot be decomposed.
    assert_eq!(
        conflict(parse_quote!(MyType<'_>)),
        (v(&["MyType < '_ >"]), v(&["'_"]), true)
    );
    assert_eq!(
        conflict(parse_quote!(<T as MyTrait<'_>>::Ty)),
        (v(&["MyTrait < '_ > :: Ty"]), v(&["'_"]), true)
    );
    // Only components do: it can be decomposed.
    assert_eq!(
        conflict(parse_quote!(::path::Wrap<MyType, '_>)),
        (v(&["MyType"]), v(&["'_"]), false)
    );
    assert_eq!(
        conflict(parse_quote!(<T as ::path::Trait<MyType, '_>>::Ty)),
        (v(&["MyType"]), v(&["'_"]), false)
    );
    // All the causes are collected, the outermost ones first.
    assert_eq!(
        conflict(parse_quote!(
            ::path::Wrap<'_, ::path::Wrap<(C,)>, A, B<D>, impl ::path::Tr>
        )),
        (
            v(&["A", "B < D >", "D", "C"]),
            v(&["'_", "impl :: path :: Tr"]),
            false
        )
    );
    // A bare identifier as an argument may be a const, so the type containing it is interned as a
    // whole (with the cause in its own path) if it can be, as `::path::Wrap<C>` here, but not the
    // outer one, which has impossible causes (`A` is a type of its own).
    assert_eq!(
        conflict(parse_quote!(::path::Wrap<'_, ::path::Wrap<C>, A>)),
        (v(&["C", "A"]), v(&["'_"]), false)
    );
    assert_eq!(
        conflict(parse_quote!(dyn A + B<'_>)),
        (v(&["A", "B < '_ >"]), v(&["'_"]), true)
    );
}

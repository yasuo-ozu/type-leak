//! Tests for `Leaker::check()` and `Leaker::allowed_paths`.

use syn::*;
use type_leak::Leaker;

#[test]
fn test_check() {
    use type_leak::CheckResult;
    let test_struct: ItemStruct = parse_quote! {
        struct MyStruct<'l1, const N1: usize, T1, T2, T3>;
    };
    let leaker = Leaker::new();
    let generics = &test_struct.generics;
    assert!(matches!(
        leaker.check(&parse_quote!(T1), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!((T1, T2)), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!([T1; 123]), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!([T1; N]), generics),
        Ok(CheckResult::MustIntern(_)) // constant
    ));
    assert!(matches!(
        leaker.check(&parse_quote!([T1; ::path::to::N]), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!([(); N1]), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(some::path), generics),
        Ok(CheckResult::MustIntern(_)) // relavice path
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(::some::path), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &parse_quote!(<T1 as ::abs::path::MyTrait<T2>>::Ty),
            generics
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &parse_quote!(<MyType as ::abs::path::MyTrait<T2>>::Ty),
            generics
        ),
        Ok(CheckResult::MustInternOrInherit(_)) // relative path
    ));
    assert!(matches!(
        leaker.check(
            &parse_quote!(<::path::to::MyType as MyTrait<T2>>::Ty),
            generics
        ),
        Ok(CheckResult::MustIntern(_)) // relative path
    ));
    assert!(matches!(
        leaker.check(
            &parse_quote!(<::path::to::MyType as ::path::to::MyTrait<MyType2>>::Ty),
            generics
        ),
        Ok(CheckResult::MustInternOrInherit(_)) // relative path
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(()), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(&'l1 ()), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(&'static ()), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(&'my_lifetime ()), generics),
        Ok(CheckResult::MustIntern(_)) // lifetime
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(impl ::path::to::MyTrait), generics),
        Ok(CheckResult::MustNotIntern(_)) // impl Trait
    ));
    assert!(leaker.check(&parse_quote!(impl MyTrait), generics).is_err()); // impl Trait and relative path
    assert!(matches!(
        leaker.check(&parse_quote!(dyn ::path::to::MyTrait), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(Self), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(&Self), generics),
        Ok(CheckResult::MustNotIntern(_))
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(&'a Self), generics),
        Ok(CheckResult::MustIntern(_))
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(::core::fmt::Formatter<'_>), generics),
        Ok(CheckResult::MustNotIntern(_)) // relative path
    ));
}

#[test]
fn test_check_self_no_internable() {
    use type_leak::{CheckResult, Leaker};

    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<'l1, const N1: usize, T1, T2, T3>;
    );

    // Test with self_ty_can_be_interned = true (default behavior)
    let leaker_true = Leaker::new();
    let generics_true = &test_struct.generics;
    assert!(matches!(
        leaker_true.check(&parse_quote!(Self), generics_true),
        Ok(CheckResult::Neutral) // Self should be allowed by default
    ));

    // Test with self_ty_can_be_interned = false
    let mut leaker_false = Leaker::new();
    // Set the flag to false to disallow Self types
    leaker_false.self_ty_can_be_interned = false;
    let generics_false = &test_struct.generics;
    assert!(matches!(
        leaker_false.check(&parse_quote!(Self), generics_false),
        Ok(CheckResult::MustNotIntern(_)) // Self should be rejected when flag is false
    ));

    // Test with complex Self usage
    assert!(matches!(
        leaker_false.check(&parse_quote!(&Self), generics_false),
        Ok(CheckResult::MustNotIntern(_)) // Reference to Self should also be rejected
    ));

    // Test that non-Self types are unaffected by the flag
    assert!(matches!(
        leaker_false.check(&parse_quote!(T1), generics_false),
        Ok(CheckResult::Neutral) // Generic type should still work
    ));
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
    use type_leak::CheckResult;

    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<'l1, const N1: usize, T1>;
    );
    let mut leaker = Leaker::new();
    let generics = &test_struct.generics;

    leaker.allowed_paths.push(parse_quote!(some::path));

    assert!(matches!(
        leaker.check(&parse_quote!(some::path), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(some::path::Type), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(other::path), generics),
        Ok(CheckResult::MustIntern(_))
    ));

    leaker.allowed_paths.push(parse_quote!(::abs::path));
    assert!(matches!(
        leaker.check(&parse_quote!(::abs::path::Type), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(abs::path::Type), generics),
        Ok(CheckResult::MustIntern(_))
    ));
}

#[test]
fn test_check_allowed_paths_with_generic_args() {
    use type_leak::{CheckResult, LeakerConfig};

    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<T1>;
    );
    let generics = &test_struct.generics;
    let mut config = LeakerConfig::new();
    config.allow_std_prelude().allow_primitive();
    let leaker = Leaker::from_config(config);

    // An allowed path with generic arguments is still allowed.
    assert!(matches!(
        leaker.check(&parse_quote!(Option<u8>), generics),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&parse_quote!(Vec<Option<T1>>), generics),
        Ok(CheckResult::Neutral)
    ));
    // Its generic arguments are still checked.
    assert!(matches!(
        leaker.check(&parse_quote!(Option<RelType>), generics),
        Ok(CheckResult::MustInternOrInherit(_))
    ));

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
    let leaked: Vec<String> = leaker
        .finish()
        .iter()
        .map(|t| template_quote::quote!(#t).to_string())
        .collect();
    assert_eq!(leaked, vec!["RelType".to_string()]);
}

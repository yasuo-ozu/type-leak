use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::Leaker;

/// Helper: convert a `Referrer`'s types to a `BTreeSet<String>` for order-independent comparison.
fn finish_as_strings(leaker: Leaker) -> BTreeSet<String> {
    leaker.finish().iter().map(|t| quote!(#t).to_string()).collect()
}

#[test]
fn test_leaker() {
    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
            field1: MyType1,
            field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
            field3: &'a (T1, T2),
        }
    );
    let mut leaker = Leaker::from_struct(&test_struct).unwrap();
    leaker.reduce_roots();
    let referrer = leaker.finish();
    assert_eq!(
        referrer
            .iter()
            .map(|i| quote!(#i).to_string())
            .collect::<BTreeSet<_>>(),
        [
            "(MyType2 , MyType3 < MyType1 > , MyType4 , MyType5)",
            "MyType1",
            "MyType4"
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>()
    );
    let mut f = |_, _| parse_quote!(::path::to::Implementor);
    assert_eq!(
        quote!(#{referrer.expand(parse_quote!(MyType1),  &mut f)}).to_string(),
        quote!(::path::to::Implementor).to_string()
    );
    assert_eq!(
        quote!(#{referrer.expand(parse_quote!((MyType2 , MyType3 < MyType1 > , MyType4 , MyType5)),  &mut f)}).to_string(),
        quote!(::path::to::Implementor).to_string()
    );
}

#[test]
fn test_leaker_construction() {
    assert!(Leaker::from_struct(&parse_quote! {
        struct MyStruct<T: path::to::MyTrait>();
    })
    .is_err());
    assert!(Leaker::from_struct(&parse_quote! {
        struct MyStruct<T: ::path::to::MyTrait>();
    })
    .is_ok());
    assert!(Leaker::from_trait(&parse_quote! {
        trait MyTrait: MyTrait2 {}
    },)
    .is_err());
    assert!(Leaker::from_trait(&parse_quote! {
        trait MyTrait: ::path::to::MyTrait2 {}
    },)
    .is_ok());
    assert!(Leaker::from_trait(&parse_quote! {
        trait MyTrait: ::path::to::MyTrait2<{<Self as ::path::to::MyTrait3>::N}> {}
    },)
    .is_ok());
    assert!(Leaker::from_trait(&parse_quote! {
        trait MyTrait: ::path::to::MyTrait2<{<Self as MyTrait3>::N}> {}
    },)
    .is_err());
}

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
    assert!(matches!(
        leaker.check(&parse_quote!(impl MyTrait), generics),
        Err(_) // impl Trait and relative path
    ));
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
        pub trait Parse: ::core::marker::Sized {
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
fn test_intern() {
    let tokens = quote! {
        pub trait Unparse<Atom> {
            fn unparse<S: ::core::iter::Iterator>(
                &self,
                sink: &mut S,
            ) -> ::core::result::Result<(), S::Item>;
        }
    };
    let item: ItemTrait = parse_quote!(#tokens);
    let mut leaker = Leaker::new();
    leaker.allowed_paths.push(parse_quote!(crate));

    leaker
        .intern_with(&item.generics, |visitor| {
            visitor.visit_item_trait(&item);
        })
        .unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

// ── reduce_roots() behavioral tests ──────────────────────────────────────────

#[test]
fn test_reduce_roots_empty_leaker() {
    // A fresh Leaker with no intern_with calls produces an empty Referrer.
    let mut leaker = Leaker::new();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_neutral_generics() {
    // All fields are the struct's own generic params → all neutral → empty.
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: T, b: U } };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_absolute_paths_only() {
    // Absolute-path field types are neutral → nothing interned.
    let item: ItemStruct = parse_quote! {
        struct S { a: ::std::string::String }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_single_must_intern() {
    // A single relative-path field is a must-intern root and must be kept.
    let item: ItemStruct = parse_quote! { struct S { a: RelType } };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_reduce_roots_multiple_must_intern() {
    // Two independent must-intern roots: both survive reduction.
    let item: ItemStruct = parse_quote! { struct S { a: TypeA, b: TypeB } };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let ta: Type = parse_quote!(TypeA);
    let tb: Type = parse_quote!(TypeB);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#ta).to_string(), quote!(#tb).to_string()])
    );
}

#[test]
fn test_reduce_roots_neutral_root_without_must_intern_dropped() {
    // A neutral root whose children are all neutral is dropped entirely.
    let item: ItemStruct = parse_quote! {
        struct S<T> { a: (T, ::std::string::String) }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_root_covers_descendant() {
    // A must-intern type nested inside a compound wrapper: the wrapper root is kept
    // but the inner type must NOT appear separately.
    let item: ItemStruct = parse_quote! {
        struct S { a: (RelType, ::std::string::String) }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!((RelType, ::std::string::String));
    let result = finish_as_strings(leaker);
    assert_eq!(result, BTreeSet::from([quote!(#expected).to_string()]));
    // The inner type must not appear as its own entry.
    let inner: Type = parse_quote!(RelType);
    assert!(!result.contains(&quote!(#inner).to_string()));
}

#[test]
fn test_reduce_roots_mixed_neutral_and_must_intern() {
    // Generic field (neutral root) alongside a relative-path field (must-intern root):
    // only the must-intern root survives.
    let item: ItemStruct = parse_quote! { struct S<T> { a: T, b: RelType } };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_reduce_roots_deep_nesting() {
    // A relative type two levels deep: only the outermost root is kept; neither the
    // intermediate Vec wrapper nor the inner type appears as a separate entry.
    let item: ItemStruct = parse_quote! {
        struct S { a: (::std::vec::Vec<RelInner>, ::std::string::String) }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!((::std::vec::Vec<RelInner>, ::std::string::String));
    let result = finish_as_strings(leaker);
    assert_eq!(result, BTreeSet::from([quote!(#expected).to_string()]));
    let inner: Type = parse_quote!(RelInner);
    let vec_ty: Type = parse_quote!(::std::vec::Vec<RelInner>);
    assert!(!result.contains(&quote!(#inner).to_string()));
    assert!(!result.contains(&quote!(#vec_ty).to_string()));
}

#[test]
fn test_reduce_roots_type_in_both_positions() {
    // A type appears as a direct must-intern root (field a) AND as a must-intern
    // descendant inside a compound root (field b). Both must be kept.
    let item: ItemStruct = parse_quote! {
        struct S {
            a: RelType,
            b: (RelType, ::std::string::String),
        }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let ta: Type = parse_quote!(RelType);
    let tb: Type = parse_quote!((RelType, ::std::string::String));
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#ta).to_string(), quote!(#tb).to_string()])
    );
}

#[test]
fn test_reduce_roots_allowed_path_is_neutral() {
    // A type under an explicitly allowed path prefix is neutral → empty.
    let item: ItemStruct = parse_quote! { struct S { a: some_mod::SomeType } };
    let mut leaker = Leaker::new();
    leaker.allowed_paths.push(parse_quote!(some_mod));
    leaker
        .intern_with(&item.generics, |v| v.visit_item_struct(&item))
        .unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_self_assoc_type_neutral() {
    // Self::AssocType in a trait method return position is neutral (concrete to each
    // implementor).
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            type Assoc;
            fn method() -> Self::Assoc;
        }
    };
    let mut leaker = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_generic_assoc_type_neutral() {
    // S::Item where S is a method-level generic is neutral: the caller provides S
    // directly so its associated types are also concrete.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn method<S: ::core::iter::Iterator>() -> S::Item;
        }
    };
    let mut leaker = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

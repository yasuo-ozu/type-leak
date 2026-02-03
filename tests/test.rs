use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::Leaker;

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
    assert_ne!(
        quote!(#{referrer.expand(parse_quote!(MyType1),  &mut f)}).to_string(),
        quote!(MyType1).to_string()
    );
    assert_ne!(
        quote!(#{referrer.expand(parse_quote!((MyType2 , MyType3 < MyType1 > , MyType4 , MyType5)),  &mut f)}).to_string(),
        quote!((MyType2 , MyType3 < MyType1 > , MyType4 , MyType5)).to_string()
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
    use syn::visit::Visit;

    let tokens = quote! {
        pub trait Unparse<Atom> {
            fn unparse<S: crate::parse::unparse::Emitter<Atom>>(
                &self,
                sink: &mut S,
            ) -> ::core::result::Result<(), S::Error>;
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
    assert!(leaker.finish().is_empty());
}

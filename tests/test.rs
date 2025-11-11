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
        .into_iter()
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
    let leaker = Leaker::from_struct(&test_struct).unwrap();
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(T1)
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!((T1, T2))
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!([T1; 123])
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!([T1; N])
        ),
        Ok(CheckResult::MustIntern(_)) // constant
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!([T1; ::path::to::N])
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!([(); N1])
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(some::path)
        ),
        Ok(CheckResult::MustIntern(_)) // relavice path
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(::some::path)
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(<T1 as ::abs::path::MyTrait<T2>>::Ty)
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(<MyType as ::abs::path::MyTrait<T2>>::Ty)
        ),
        Ok(CheckResult::MustInternOrInherit(_)) // relative path
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(<::path::to::MyType as MyTrait<T2>>::Ty)
        ),
        Ok(CheckResult::MustIntern(_)) // relative path
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(<::path::to::MyType as ::path::to::MyTrait<MyType2>>::Ty)
        ),
        Ok(CheckResult::MustInternOrInherit(_)) // relative path
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(())
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(&'l1 ())
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(&'static ())
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(&'my_lifetime ())
        ),
        Ok(CheckResult::MustIntern(_)) // lifetime
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(impl ::path::to::MyTrait)
        ),
        Ok(CheckResult::MustNotIntern(_)) // impl Trait
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(impl MyTrait)
        ),
        Err(_) // impl Trait and relative path
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(dyn ::path::to::MyTrait)
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(Self)
        ),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(&Self)
        ),
        Ok(CheckResult::MustNotIntern(_))
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(&'a Self)
        ),
        Ok(CheckResult::MustIntern(_))
    ));
    assert!(matches!(
        leaker.check(
            &test_struct.generics,
            &parse_quote!(::core::fmt::Formatter<'_>)
        ),
        Ok(CheckResult::MustNotIntern(_)) // relative path
    ));
}

#[test]
fn test_check_self_no_internable() {
    use type_leak::{Leaker, CheckResult};
    
    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<'l1, const N1: usize, T1, T2, T3>;
    );
    
    // Test with self_ty_can_be_interned = true (default behavior)
    let leaker_true = Leaker::with_generics(test_struct.generics.clone());
    assert!(matches!(
        leaker_true.check(
            &test_struct.generics,
            &parse_quote!(Self)
        ),
        Ok(CheckResult::Neutral) // Self should be allowed by default
    ));
    
    // Test with self_ty_can_be_interned = false
    let mut leaker_false = Leaker::with_generics(test_struct.generics.clone());
    // Set the flag to false to disallow Self types
    leaker_false.self_ty_can_be_interned = false;
    assert!(matches!(
        leaker_false.check(
            &test_struct.generics,
            &parse_quote!(Self)
        ),
        Ok(CheckResult::MustNotIntern(_)) // Self should be rejected when flag is false
    ));
    
    // Test with complex Self usage
    assert!(matches!(
        leaker_false.check(
            &test_struct.generics,
            &parse_quote!(&Self)
        ),
        Ok(CheckResult::MustNotIntern(_)) // Reference to Self should also be rejected
    ));
    
    // Test that non-Self types are unaffected by the flag
    assert!(matches!(
        leaker_false.check(
            &test_struct.generics,
            &parse_quote!(T1)
        ),
        Ok(CheckResult::Neutral) // Generic type should still work
    ));
}

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
    let (repeater_impls, referrer) = leaker.finish(|_| parse_quote!(LeakerType));
    let repeater_impl_tys = repeater_impls
        .iter()
        .map(|rimp| {
            rimp.items.iter().filter_map(|item| match item {
                ImplItem::Type(ImplItemType { ty, .. }) => Some(format!("{}", quote!(#ty))),
                _ => None,
            })
        })
        .flatten()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        repeater_impl_tys,
        [
            "(MyType2 , MyType3 < MyType1 > , MyType4 , MyType5)",
            "MyType1",
            "MyType4"
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>()
    );
    let mut f = |_| parse_quote!(::path::to::Implementor);
    assert_ne!(
        quote!(#{referrer.expand(parse_quote!(MyType1), &parse_quote!(T), &mut f)}).to_string(),
        quote!(MyType1).to_string()
    );
    assert_ne!(
        quote!(#{referrer.expand(parse_quote!((MyType2 , MyType3 < MyType1 > , MyType4 , MyType5)), &parse_quote!(T), &mut f)}).to_string(),
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
    assert!(Leaker::from_trait(
        &parse_quote! {
            trait MyTrait: MyTrait2 {}
        },
        Box::new(|_| parse_quote!(T)),
    )
    .is_err());
    assert!(Leaker::from_trait(
        &parse_quote! {
            trait MyTrait: ::path::to::MyTrait2 {}
        },
        Box::new(|_| parse_quote!(T)),
    )
    .is_ok());
    assert!(Leaker::from_trait(
        &parse_quote! {
            trait MyTrait: ::path::to::MyTrait2<{<Self as ::path::to::MyTrait3>::N}> {}
        },
        Box::new(|_| parse_quote!(T)),
    )
    .is_ok());
    assert!(Leaker::from_trait(
        &parse_quote! {
            trait MyTrait: ::path::to::MyTrait2<{<Self as MyTrait3>::N}> {}
        },
        Box::new(|_| parse_quote!(T)),
    )
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
        leaker.check(&test_struct.generics, &parse_quote!(T1)),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!((T1, T2))),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!([T1; 123])),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!([T1; N])),
        Ok(CheckResult::MustIntern(_)) // constant
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!([T1; ::path::to::N])),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!([(); N1])),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!(some::path)),
        Ok(CheckResult::MustIntern(_)) // relavice path
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!(::some::path)),
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
        leaker.check(&test_struct.generics, &parse_quote!(())),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!(&'l1 ())),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!(&'static ())),
        Ok(CheckResult::Neutral)
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!(&'my_lifetime ())),
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
        leaker.check(&test_struct.generics, &parse_quote!(impl MyTrait)),
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
        leaker.check(&test_struct.generics, &parse_quote!(Self)),
        Ok(CheckResult::Neutral) // relative path
    ));
    assert!(matches!(
        leaker.check(&test_struct.generics, &parse_quote!(&Self)),
        Ok(CheckResult::Neutral) // relative path
    ));
}

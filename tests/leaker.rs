//! Tests for building a `Leaker` and interning items.

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
    let (mut leaker, _) = Leaker::from_struct(&test_struct).unwrap();
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

// ── regressions ──────────────────────────────────────────────────────────────

fn leak_struct(item: &ItemStruct) -> Result<BTreeSet<String>> {
    let mut config = type_leak::LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let mut leaker = Leaker::from_config(config);
    leaker
        .intern_with(&item.generics, |v| v.visit_item_struct(item))
        .map_err(|e| Error::new(e.span(), e))?;
    Ok(leaker
        .finish()
        .iter()
        .map(|t| quote!(#t).to_string())
        .collect())
}

fn leak_trait(item: &ItemTrait) -> Result<BTreeSet<String>> {
    let mut config = type_leak::LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let mut leaker = Leaker::from_config(config);
    leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(item))
        .map_err(|e| Error::new(e.span(), e))?;
    Ok(leaker
        .finish()
        .iter()
        .map(|t| quote!(#t).to_string())
        .collect())
}

fn strings(tys: &[&str]) -> BTreeSet<String> {
    tys.iter()
        .map(|s| {
            let ty: Type = parse_str(s).unwrap();
            quote!(#ty).to_string()
        })
        .collect()
}

#[test]
fn test_hrtb_lifetimes_are_not_leaked() {
    // Lifetimes of `for<..>` binders are in scope inside them, not leaked.
    let item: ItemStruct = parse_quote! {
        struct S<T, F: for<'a> ::core::ops::Fn(&'a u8)>
        where
            for<'a> &'a T: ::core::iter::IntoIterator,
        {
            t: T,
            f: F,
            g: for<'a> fn(&'a u8),
        }
    };
    assert_eq!(leak_struct(&item).unwrap(), BTreeSet::<String>::new());

    // Relative types inside them are leaked without the lifetimes.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f<F: for<'a> ::core::ops::Fn(&'a MyType)>(&self, f: F);
        }
    };
    assert_eq!(leak_trait(&item).unwrap(), strings(&["MyType"]));
}

#[test]
fn test_finish_reduces_and_accumulates() {
    // `finish()` without `reduce_roots()`.
    let item: ItemStruct = parse_quote!(
        struct S {
            a: A,
        }
    );
    let (leaker, _) = Leaker::from_struct(&item).unwrap();
    assert_eq!(leaker.finish().iter().count(), 1);

    // Several `intern_with()` and `reduce_roots()` calls accumulate.
    let s: ItemStruct = parse_quote!(
        struct S {
            a: A,
        }
    );
    let u: ItemStruct = parse_quote!(
        struct U {
            b: B,
        }
    );
    let mut leaker = Leaker::new();
    leaker
        .intern_with(&s.generics, |v| v.visit_item_struct(&s))
        .unwrap();
    leaker.reduce_roots();
    leaker
        .intern_with(&u.generics, |v| v.visit_item_struct(&u))
        .unwrap();
    leaker.reduce_roots();
    let leaked: BTreeSet<String> = leaker
        .finish()
        .iter()
        .map(|t| quote!(#t).to_string())
        .collect();
    assert_eq!(leaked, strings(&["A", "B"]));
}

#[test]
fn test_finish_order_is_deterministic() {
    // The leaked types (and so their indices) are the same in every run, whatever the order.
    let item: ItemStruct = parse_quote! {
        struct S { e: E, c: C, a: A, d: D, b: B, a2: A }
    };
    let finish = || -> Vec<String> {
        Leaker::from_struct(&item)
            .unwrap()
            .0
            .finish()
            .iter()
            .map(|t| quote!(#t).to_string())
            .collect()
    };
    let first = finish();
    assert_eq!(
        first.iter().cloned().collect::<BTreeSet<_>>(),
        strings(&["A", "B", "C", "D", "E"])
    );
    for _ in 0..8 {
        assert_eq!(finish(), first);
    }
}

#[test]
fn test_uninternable_types_are_decomposed() {
    // An elided lifetime or `impl Trait` prevents interning the type as a whole, but its
    // components are leaked.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, x: &MyType1) -> &MyType2;
            fn g(&self) -> impl Iterator<Item = MyType3>;
            fn h(&self, x: Box<dyn Fn(&MyType4)>);
        }
    };
    assert_eq!(
        leak_trait(&item).unwrap(),
        strings(&["MyType1", "MyType2", "MyType3", "MyType4"])
    );

    // A relative path with an elided lifetime cannot be decomposed.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, x: MyType<'_>);
        }
    };
    assert_eq!(
        leak_trait(&item).unwrap_err().to_string(),
        "use absolute path: 'MyType < '_ >'"
    );
}

#[test]
fn test_all_dyn_bounds_are_checked() {
    // Both orders of the bounds are leaked as a whole (inside the `Box`, which is rebuilt).
    let item: ItemStruct = parse_quote! {
        struct S {
            a: Box<dyn MyTrait + Send>,
            b: Box<dyn Send + MyTrait>,
        }
    };
    assert_eq!(
        leak_struct(&item).unwrap(),
        strings(&["dyn MyTrait + Send", "dyn Send + MyTrait"])
    );
}

#[test]
fn test_type_macros_are_leaked_whole() {
    let item: ItemStruct = parse_quote! {
        struct S {
            a: my_ty!(),
            b: ::k::ty!(MyType),
        }
    };
    assert_eq!(
        leak_struct(&item).unwrap(),
        strings(&["my_ty!()", "::k::ty!(MyType)"])
    );
}

#[test]
fn test_trait_const_defaults_are_ignored() {
    // Like method bodies, defaults of associated consts are not analyzed.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            const X: usize = helper();
            fn f(&self) { helper(); }
        }
    };
    assert_eq!(leak_trait(&item).unwrap(), BTreeSet::<String>::new());
}

#[test]
fn test_assertions_for_assumed_primitives() {
    // Primitive names in const parameter types, and in types where they are allowed, are
    // checked once each; absolute paths and leaked (not allowed) primitives are not.
    let item: ItemTrait = parse_quote! {
        trait Tr<const N: u8> {
            fn f<const M: u8>(&self, x: u16) -> ::core::primitive::u32;
        }
    };
    let mut config = type_leak::LeakerConfig::new();
    config.allow_primitive();
    let mut leaker = Leaker::from_config(config);
    leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(&item))
        .unwrap();
    let assertions = leaker.assertions().to_string();
    assert!(assertions.contains("__type_leak_assert_u8_is_primitive"));
    assert!(assertions.contains("__type_leak_assert_u16_is_primitive"));
    assert!(!assertions.contains("__type_leak_assert_u32_is_primitive"));
    assert_eq!(
        assertions
            .matches("__type_leak_assert_u8_is_primitive")
            .count(),
        1
    );

    // Without `allow_primitive()`, `u16` in a type is leaked, so it is not assumed.
    let mut leaker = Leaker::new();
    leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(&item))
        .unwrap();
    let assertions = leaker.assertions().to_string();
    assert!(assertions.contains("__type_leak_assert_u8_is_primitive"));
    assert!(!assertions.contains("__type_leak_assert_u16_is_primitive"));

    // No primitives, no items.
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            t: T,
        }
    );
    assert!(Leaker::from_struct(&item)
        .unwrap()
        .0
        .assertions()
        .is_empty());
}

#[test]
fn test_leaker_from_union() {
    let item: ItemUnion = parse_quote! {
        pub union MyUnion<'a, T: ::core::marker::Copy> {
            a: ::core::mem::ManuallyDrop<MyType1>,
            b: &'a T,
            c: (MyType2, u8),
        }
    };
    let referrer = Leaker::from_union(&item).unwrap().0.finish();
    assert_eq!(
        referrer
            .iter()
            .map(|i| quote!(#i).to_string())
            .collect::<BTreeSet<_>>(),
        ["MyType1", "(MyType2 , u8)"]
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<_>>()
    );
    // Bounds must use absolute paths as for structs.
    assert!(Leaker::from_union(&parse_quote! {
        union MyUnion<T: Copy> { a: T }
    })
    .is_err());
}

#[test]
fn test_receivers() {
    // The type of a receiver (`&self` is `self: &Self`) is analyzed as any other type, where
    // `Self` alone is never leaked.
    let leak = |sig: proc_macro2::TokenStream,
                self_ty_can_be_interned: bool|
     -> std::result::Result<BTreeSet<String>, String> {
        let item: ItemTrait = parse_quote!(trait Tr<'a> { #sig });
        let mut leaker = Leaker::new();
        leaker.self_ty_can_be_interned = self_ty_can_be_interned;
        leaker
            .intern_with(&item.generics, |v| v.visit_item_trait(&item))
            .map_err(|e| e.to_string())?;
        Ok(leaker
            .finish()
            .iter()
            .map(|t| quote!(#t).to_string())
            .collect::<BTreeSet<_>>())
    };
    for self_ty_can_be_interned in [true, false] {
        for sig in [
            quote!(
                fn f(&self);
            ),
            quote!(
                fn f(&'a mut self);
            ),
            quote!(
                fn f(mut self);
            ),
            quote!(
                fn f(self: ::std::boxed::Box<Self>);
            ),
            quote!(
                fn f(self: &'a Self);
            ),
            quote!(
                fn f(&self) -> &Self;
            ),
        ] {
            assert_eq!(
                leak(sig, self_ty_can_be_interned),
                Ok(BTreeSet::new()),
                "{}",
                self_ty_can_be_interned
            );
        }
    }

    // A relative type applied to `Self` is leaked as a whole only if `Self` can be interned.
    assert_eq!(
        leak(
            quote!(
                fn f(self: Rel<Self>);
            ),
            true
        ),
        Ok(strings(&["Rel<Self>"]))
    );
    assert_eq!(
        leak(
            quote!(
                fn f(self: Rel<Self>);
            ),
            false
        ),
        Err("use absolute path: 'Rel < Self >'".to_string())
    );
    assert_eq!(
        leak(
            quote!(
                fn f(self: ::k::W<Rel, Self>);
            ),
            false
        ),
        Ok(strings(&["Rel"]))
    );
}

#[test]
fn test_grouped_const_param_type_is_asserted() {
    // A primitive type of a const parameter passed through `macro_rules!` (`$t:ty`, wrapped in an
    // invisible group) is copied as it is, so it is asserted not to be shadowed as a bare one is.
    let grouped = Type::Group(TypeGroup {
        group_token: Default::default(),
        elem: Box::new(parse_quote!(u8)),
    });
    for ty in [parse_quote!(u8), grouped] {
        let item: ItemTrait = parse_quote!(trait Tr<const N: #ty> {});
        let assertions = Leaker::from_trait(&item)
            .unwrap()
            .0
            .assertions()
            .to_string();
        assert!(
            assertions.contains("__type_leak_assert_u8_is_primitive"),
            "{}",
            assertions
        );
    }
    // An absolute path to a primitive type needs no assertion.
    let item: ItemTrait = parse_quote!(
        trait Tr<const N: ::core::primitive::u8> {}
    );
    assert!(Leaker::from_trait(&item).unwrap().0.assertions().is_empty());
}

#[test]
fn test_qualified_paths_are_decomposed() {
    // A qualified path with a context-free trait is decomposed like any other path, so only its
    // components which depend on the type context are leaked.
    let item: ItemTrait = parse_quote! {
        trait Tr<T> {
            fn f(&self, a: <T as ::p::Tr<RelA, '_>>::Ty, b: <(RelB, &'_ u8) as ::p::Tr>::Ty);
            fn g(&self, c: <T as ::p::Tr>::Ty<'_, RelC>);
        }
    };
    assert_eq!(
        leak_trait(&item).unwrap(),
        strings(&["RelA", "RelB", "RelC"])
    );
    // A relative trait cannot be decomposed.
    let item: ItemTrait = parse_quote! {
        trait Tr<T> {
            fn f(&self, a: <T as RelTrait<'_>>::Ty);
        }
    };
    assert_eq!(
        leak_trait(&item).unwrap_err().to_string(),
        "use absolute path: 'RelTrait < '_ > :: Ty'"
    );
}

#[test]
fn test_impl_trait_binders() {
    // The lifetimes of a `for<..>` binder in an `impl Trait` bound are in scope inside it, so they
    // are not leaked (nor reported as relative).
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, x: impl for<'b> ::p::F<&'b RelA>);
        }
    };
    assert_eq!(leak_trait(&item).unwrap(), strings(&["RelA"]));
}

#[test]
fn test_self_ty_cannot_be_interned() {
    let leak = |sig: proc_macro2::TokenStream,
                self_ty_can_be_interned: bool|
     -> std::result::Result<BTreeSet<String>, syn::Error> {
        let item: ItemTrait = parse_quote!(trait Tr { #sig });
        let mut leaker = Leaker::new();
        leaker.self_ty_can_be_interned = self_ty_can_be_interned;
        leaker
            .intern_with(&item.generics, |v| v.visit_item_trait(&item))
            .map_err(syn::Error::from)?;
        Ok(leaker
            .finish()
            .iter()
            .map(|t| quote!(#t).to_string())
            .collect::<BTreeSet<_>>())
    };
    let set = |types: &[&str]| types.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();

    // A relative type applied to `Self` cannot be leaked as a whole, so the path is an error.
    let err = leak(
        quote!(
            fn f(&self) -> Rel<Self>;
        ),
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.starts_with("use absolute path: 'Rel < Self >'"),
        "{}",
        err
    );
    assert!(
        err.contains("it cannot be leaked as a whole because of `Self`"),
        "{}",
        err
    );
    assert_eq!(
        leak(
            quote!(
                fn f(&self) -> Rel<Self>;
            ),
            true
        )
        .unwrap(),
        set(&["Rel < Self >"])
    );

    // The components are leaked instead if the type can be decomposed.
    for ty in [quote!(::k::W<Rel, Self>), quote!((Rel, Self))] {
        let sig = quote!(
            fn f(&self) -> #ty;
        );
        assert_eq!(leak(sig.clone(), false).unwrap(), set(&["Rel"]));
        // `Self` does not depend on the context by itself, so `Rel` is enough.
        assert_eq!(leak(sig, true).unwrap(), set(&["Rel"]), "{}", ty);
    }

    // `Self` alone is never leaked.
    for ty in [
        quote!(Self),
        quote!(Self::Item),
        quote!(&Self),
        quote!(::k::W<Self>),
    ] {
        assert_eq!(
            leak(
                quote!(
                    fn f(&self) -> #ty;
                ),
                false
            )
            .unwrap(),
            BTreeSet::new(),
            "{}",
            ty
        );
    }
}

#[test]
fn test_bodies_are_not_analyzed() {
    // The bodies of functions and the values of consts are not analyzed, while the signatures and
    // the types are.
    let item: ItemImpl = parse_quote! {
        impl ::k::Tr for ::k::W {
            const C: RelC = RelCVal::new();
            fn f(&self, a: RelArg) -> RelRet {
                let x: RelBody = RelBody::new();
                fn g() -> RelNested { RelNested::new() }
                x
            }
        }
    };
    let mut leaker = Leaker::new();
    leaker
        .intern_with(&Generics::default(), |v| v.visit_item_impl(&item))
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer
            .iter()
            .map(|t| quote!(#t).to_string())
            .collect::<BTreeSet<_>>(),
        ["RelC", "RelArg", "RelRet"]
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<_>>()
    );

    // Also those of the nested items.
    let item: ItemMod = parse_quote! {
        mod m {
            const C: RelC = RelCVal::new();
            static S: RelS = RelSVal::new();
            fn f() -> RelRet { let x: RelBody = RelBody::new(); x }
        }
    };
    let mut leaker = Leaker::new();
    leaker
        .intern_with(&Generics::default(), |v| v.visit_item_mod(&item))
        .unwrap();
    assert_eq!(
        leaker
            .finish()
            .iter()
            .map(|t| quote!(#t).to_string())
            .collect::<BTreeSet<_>>(),
        ["RelC", "RelS", "RelRet"]
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<_>>()
    );
}

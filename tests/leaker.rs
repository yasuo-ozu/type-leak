//! Tests for building a `Leaker` and interning items.

mod common;

use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::{Leaker, LeakerConfig};

#[test]
fn test_leaker() {
    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
            field1: MyType1,
            field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
            field3: &'a (T1, T2),
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_struct(&test_struct, true).unwrap();
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
    // `expand()` replaces a leaked type by what the callback returns for its index, and nothing
    // else (the entries are sorted by tokens: the tuple, `MyType1`, `MyType4`).
    let mut f = |_, ix: usize| -> Type {
        let ix = syn::Index::from(ix);
        parse_quote!(::path::to::Implementor<#ix>)
    };
    assert_eq!(
        quote!(#{referrer.expand(context, parse_quote!(MyType1),  &mut f)}).to_string(),
        quote!(::path::to::Implementor<1>).to_string()
    );
    assert_eq!(
        quote!(#{referrer.expand(context, parse_quote!((MyType2 , MyType3 < MyType1 > , MyType4 , MyType5)),  &mut f)}).to_string(),
        quote!(::path::to::Implementor<0>).to_string()
    );
    assert_eq!(
        quote!(#{referrer.expand(context, parse_quote!(MyType4),  &mut f)}).to_string(),
        quote!(::path::to::Implementor<2>).to_string()
    );
    // A type that is not leaked is left as it is.
    assert_eq!(
        quote!(#{referrer.expand(context, parse_quote!(Other),  &mut f)}).to_string(),
        quote!(Other).to_string()
    );
}

#[test]
fn test_leaker_construction() {
    assert!(Leaker::new()
        .intern_struct(
            &parse_quote! {
                struct MyStruct<T: path::to::MyTrait>();
            },
            true
        )
        .is_err());
    assert!(Leaker::new()
        .intern_struct(
            &parse_quote! {
                struct MyStruct<T: ::path::to::MyTrait>();
            },
            true
        )
        .is_ok());
    assert!(Leaker::new()
        .intern_trait(
            &parse_quote! {
                trait MyTrait: MyTrait2 {}
            },
            true
        )
        .is_err());
    assert!(Leaker::new()
        .intern_trait(
            &parse_quote! {
                trait MyTrait: ::path::to::MyTrait2 {}
            },
            true
        )
        .is_ok());
    assert!(Leaker::new()
        .intern_trait(
            &parse_quote! {
                trait MyTrait: ::path::to::MyTrait2<{<Self as ::path::to::MyTrait3>::N}> {}
            },
            true
        )
        .is_ok());
    assert!(Leaker::new()
        .intern_trait(
            &parse_quote! {
                trait MyTrait: ::path::to::MyTrait2<{<Self as MyTrait3>::N}> {}
            },
            true
        )
        .is_err());
}

/// The number of entries of `struct S(<ty>);` with the Leaker `leaker` (`ty` is a path written
/// the relative way, which is leaked unless it is allowed).
fn entries_of_a_field(leaker: Leaker, ty: &str) -> usize {
    let item: ItemStruct = parse_str(&format!("struct S({});", ty)).unwrap();
    let mut leaker = leaker;
    leaker.intern_struct(&item, true).unwrap();
    leaker.finish().iter().count()
}

/// The docs of `LeakerConfig`: `allow_primitive()` allows the primitive types, and
/// `allow_std_prelude()` the names of the standard prelude: a relative path with such a name is not
/// leaked, while `Leaker::new()` allows no path and leaks it. Audit mutants `cfg_primitive_f64`,
/// `cfg_primitive_f32`, `cfg_primitive_i128`, `cfg_prelude_drop`, `cfg_prelude_fromiter`,
/// `cfg_prelude_toowned` and `cfg_prelude_ord`.
#[test]
fn test_config_allows_the_documented_names() {
    let primitives = [
        "bool", "char", "str", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32",
        "i64", "i128", "isize", "f32", "f64",
    ];
    let prelude = [
        "Box",
        "String",
        "Vec",
        "Option",
        "Result",
        "Copy",
        "Send",
        "Sized",
        "Sync",
        "Unpin",
        "Drop",
        "Fn",
        "FnMut",
        "FnOnce",
        "Clone",
        "PartialEq",
        "PartialOrd",
        "Eq",
        "Ord",
        "AsRef",
        "AsMut",
        "Into",
        "From",
        "Default",
        "Iterator",
        "IntoIterator",
        "Extend",
        "FromIterator",
        "ToOwned",
        "ToString",
    ];
    let mut only_primitives = LeakerConfig::new();
    only_primitives.allow_primitive();
    let mut only_prelude = LeakerConfig::new();
    only_prelude.allow_std_prelude();
    for name in primitives.iter().chain(prelude.iter()) {
        // Nothing is allowed by default.
        assert_eq!(entries_of_a_field(Leaker::new(), name), 1, "{}", name);
        // Each list allows its names, and not the other list's.
        let (own, other) = if primitives.contains(name) {
            (&only_primitives, &only_prelude)
        } else {
            (&only_prelude, &only_primitives)
        };
        assert_eq!(
            entries_of_a_field(Leaker::from_config(own.clone()), name),
            0,
            "{}",
            name
        );
        assert_eq!(
            entries_of_a_field(Leaker::from_config(other.clone()), name),
            1,
            "{}",
            name
        );
    }
    // The function `drop` is a name of the prelude too (here in an array length).
    let item: ItemStruct = parse_quote!(
        struct S([::core::primitive::u8; drop]);
    );
    let entries = |config: LeakerConfig| {
        let mut leaker = Leaker::from_config(config);
        leaker.intern_struct(&item, true).unwrap();
        leaker.finish().iter().count()
    };
    assert_eq!(entries(LeakerConfig::new()), 1);
    assert_eq!(entries(only_primitives.clone()), 1);
    assert_eq!(entries(only_prelude.clone()), 0);
    // `allow_crate()` allows the paths that start with `crate`, and the lists add up.
    let item: ItemStruct = parse_quote!(
        struct S(crate::A, u8, Vec<crate::B>);
    );
    let mut all = LeakerConfig::new();
    all.allow_primitive().allow_std_prelude().allow_crate();
    let mut leaker = Leaker::from_config(all);
    leaker.intern_struct(&item, true).unwrap();
    assert!(leaker.finish().is_empty());
    let mut leaker = Leaker::from_config(only_prelude);
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer
            .iter()
            .map(|ty| quote!(#ty).to_string())
            .collect::<Vec<_>>(),
        ["crate :: A", "crate :: B", "u8"]
    );
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

    leaker.intern_trait(&item, true).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

// ── regressions ──────────────────────────────────────────────────────────────

fn leak_struct(item: &ItemStruct) -> Result<BTreeSet<String>> {
    let mut config = type_leak::LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let mut leaker = Leaker::from_config(config);
    leaker
        .intern_struct(item, true)
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
        .intern_trait(item, true)
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
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    assert_eq!(leaker.finish().iter().count(), 1);

    // Several `intern_struct()` and `reduce_roots()` calls accumulate.
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
    leaker.intern_struct(&s, true).unwrap();
    leaker.reduce_roots();
    leaker.intern_struct(&u, true).unwrap();
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
        struct S { e: E, c: C, a: A, d: D, b: B, a2: A, t: (T2, A, T3) }
    };
    let finish = || -> Vec<String> {
        {
            let mut leaker = Leaker::new();
            leaker.intern_struct(&item, true).unwrap();
            leaker.finish()
        }
        .iter()
        .map(|t| quote!(#t).to_string())
        .collect()
    };
    let first = finish();
    // The entries are sorted by context, and then by tokens, whatever the order they appear in
    // (the run-to-run comparison below cannot tell a sorted list from one in the order of a hash
    // map that happens to repeat). The tuple is kept whole (one entry instead of `T2` and `T3`),
    // next to `A`, which it contains: they are one component of the graph, and the entries of a
    // component are not in the order of the others without the sort. Audit mutant
    // `sort_kept_off`.
    assert_eq!(first, ["(T2 , A , T3)", "A", "B", "C", "D", "E"]);
    for _ in 0..8 {
        assert_eq!(finish(), first);
    }
    // Two contexts: `Y<T>` has the generics `<T>` (the context 1), and comes after the types with
    // none (the context 0), although its tokens are between theirs. The optional entries (of an
    // external crate) come after the required ones, sorted the same way by the contexts of the
    // result: `::ext::W<T>` was the first optional entry, but its context is the second one of
    // the Referrer, as the required `Y<T>` made it. Audit mutant `sort_optional_off`.
    let item: ItemStruct = parse_quote! {
        struct S<T> {
            z: Z,
            y: Y<T>,
            a: A,
            w: ::ext::W<T>,
            v: ::ext::V,
            o: ::ext::O,
            n: ::ext::N,
            x: X<T>,
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    let listed: Vec<(String, bool, usize)> = (0..referrer.iter().count())
        .map(|ix| {
            (
                quote!(#{referrer.iter().nth(ix).unwrap()}).to_string(),
                referrer.must(ix),
                referrer.context(ix).index(),
            )
        })
        .collect();
    assert_eq!(
        listed,
        [
            ("A".to_string(), true, 0),
            ("Z".to_string(), true, 0),
            ("X < T >".to_string(), true, 1),
            ("Y < T >".to_string(), true, 1),
            (":: ext :: N".to_string(), false, 0),
            (":: ext :: O".to_string(), false, 0),
            (":: ext :: V".to_string(), false, 0),
            (":: ext :: W < T >".to_string(), false, 1),
        ]
    );
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
    leaker.intern_trait(&item, true).unwrap();
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
    leaker.intern_trait(&item, true).unwrap();
    let assertions = leaker.assertions().to_string();
    assert!(assertions.contains("__type_leak_assert_u8_is_primitive"));
    assert!(!assertions.contains("__type_leak_assert_u16_is_primitive"));

    // No primitives, no items.
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            t: T,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    assert!(leaker.assertions().is_empty());
}

#[test]
fn test_leaker_intern_union() {
    let item: ItemUnion = parse_quote! {
        pub union MyUnion<'a, T: ::core::marker::Copy> {
            a: ::core::mem::ManuallyDrop<MyType1>,
            b: &'a T,
            c: (MyType2, u8),
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_union(&item, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer
            .iter()
            .map(|i| quote!(#i).to_string())
            .collect::<BTreeSet<_>>(),
        // `MyType1` is a bare identifier as an argument, which may be a const, so the path is
        // leaked as a whole.
        [
            ":: core :: mem :: ManuallyDrop < MyType1 >",
            "(MyType2 , u8)"
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<BTreeSet<_>>()
    );
    // Bounds must use absolute paths as for structs.
    assert!(Leaker::new()
        .intern_union(
            &parse_quote! {
                union MyUnion<T: Copy> { a: T }
            },
            true
        )
        .is_err());
}

#[test]
fn test_receivers() {
    // The type of a receiver (`&self` is `self: &Self`) is analyzed as any other type, where
    // `Self` alone is never leaked.
    let leak = |sig: proc_macro2::TokenStream,
                allow_self: bool|
     -> std::result::Result<BTreeSet<String>, String> {
        let mut leaker = Leaker::new();
        common::intern_fns(&mut leaker, &parse_quote!(<'a>), sig, allow_self)
            .map_err(|e| e.to_string())?;
        Ok(leaker
            .finish()
            .iter()
            .map(|t| quote!(#t).to_string())
            .collect::<BTreeSet<_>>())
    };
    for allow_self in [true, false] {
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
            assert_eq!(leak(sig, allow_self), Ok(BTreeSet::new()), "{}", allow_self);
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
    // `Rel` may be a const, and the path cannot be leaked as a whole because of `Self`: it is not
    // leaked as a type (see `tests/bare_const.rs`).
    assert_eq!(
        leak(
            quote!(
                fn f(self: ::k::W<Rel, Self>);
            ),
            false
        ),
        Err(
            "`Rel` may be a const, and the type that holds it cannot be leaked as a whole"
                .to_string()
        )
    );
    assert_eq!(
        leak(
            quote!(
                fn f(self: ::k::W<(Rel,), Self>);
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
        let mut leaker = Leaker::new();
        leaker.intern_trait(&item, true).unwrap();
        let assertions = leaker.assertions().to_string();
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
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true).unwrap();
    assert!(leaker.assertions().is_empty());
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
                allow_self: bool|
     -> std::result::Result<BTreeSet<String>, syn::Error> {
        let mut leaker = Leaker::new();
        common::intern_fns(&mut leaker, &Generics::default(), sig, allow_self)
            .map_err(syn::Error::from)?;
        // The required entries (the optional ones are those of external crates, below).
        let referrer = leaker.finish();
        Ok(referrer
            .iter()
            .enumerate()
            .filter(|(ix, _)| referrer.must(*ix))
            .map(|(_, t)| quote!(#t).to_string())
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
    for ty in [quote!(::k::W<(Rel,), Self>), quote!((Rel, Self))] {
        let sig = quote!(
            fn f(&self) -> #ty;
        );
        assert_eq!(leak(sig.clone(), false).unwrap(), set(&["Rel"]));
        // `Self` does not depend on the context by itself, so `Rel` is enough.
        assert_eq!(leak(sig, true).unwrap(), set(&["Rel"]), "{}", ty);
    }

    // `Rel` as an argument may be a const, so the path is interned as a whole if it can be (with
    // `Self` allowed), and otherwise it is an error: `Rel` is not leaked as a type (see
    // `tests/bare_const.rs`).
    let sig = quote!(
        fn f(&self) -> ::k::W<Rel, Self>;
    );
    let err = leak(sig.clone(), false).unwrap_err().to_string();
    assert!(
        err.starts_with(
            "`Rel` may be a const, and the type that holds it cannot be leaked as a whole"
        ),
        "{}",
        err
    );
    assert_eq!(leak(sig, true).unwrap(), set(&[":: k :: W < Rel , Self >"]));

    // The type of an external crate with `Self` in it is an optional entry (which the Referrer
    // may replace) only if `Self` can be leaked (old: not an entry, as the crate was trusted).
    let optional = |allow_self: bool| {
        let mut leaker = Leaker::new();
        common::intern_fns(
            &mut leaker,
            &Generics::default(),
            quote!(
                fn f(&self) -> ::k::W<(Rel,), Self>;
            ),
            allow_self,
        )
        .unwrap();
        let referrer = leaker.finish();
        referrer
            .iter()
            .enumerate()
            .filter(|(ix, _)| !referrer.must(*ix))
            .map(|(_, t)| quote!(#t).to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(optional(true), [":: k :: W < (Rel ,) , Self >"]);
    assert!(optional(false).is_empty());

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
    leaker.intern_impl(&item, true).unwrap();
    let referrer = leaker.finish();
    // `::k::W`, the type of an external crate, is an optional entry, after the required ones.
    assert_eq!(
        referrer
            .iter()
            .enumerate()
            .map(|(ix, t)| (quote!(#t).to_string(), referrer.must(ix)))
            .collect::<Vec<_>>(),
        [
            ("RelArg", true),
            ("RelC", true),
            ("RelRet", true),
            (":: k :: W", false)
        ]
        .iter()
        .map(|(s, must)| (s.to_string(), *must))
        .collect::<Vec<_>>()
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
    leaker.intern_mod(&item).unwrap();
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

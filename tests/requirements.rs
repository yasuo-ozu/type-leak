//! Tests for the requirements of the leaked types: the generics they use and what `Self` means,
//! which decide the entries (and contexts) of the `Referrer`.

use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::{EncodeError, Leaker, LeakerConfig, Referrer, SelfRequirement, Visitor};

/// The tag of the format of a serialized Referrer, which it starts with.
fn tag() -> String {
    format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

/// Leaks the types of the traits (one call each).
fn leak_traits(traits: &[ItemTrait]) -> Referrer {
    let mut leaker = Leaker::new();
    for item in traits {
        leaker
            .intern_with(&item.generics, |v| v.visit_item_trait(item))
            .unwrap();
    }
    leaker.finish()
}

fn leak_trait(item: ItemTrait) -> Referrer {
    leak_traits(&[item])
}

/// Returns the leaked types with the generics (and the where clause) they use.
fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter_with_generics()
        .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
        .collect()
}

fn generics_of(generics: Generics) -> String {
    s(&quote!(#generics #{&generics.where_clause}))
}

fn entry(ty: Type, generics: Generics) -> (String, String) {
    (s(&ty), generics_of(generics))
}

/// What `Self` means in each entry, as a string.
fn selfs(referrer: &Referrer) -> Vec<String> {
    (0..referrer.iter().count())
        .map(|ix| match referrer.context_self(referrer.context(ix)) {
            None => "-".to_string(),
            Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
            Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
            Some(SelfRequirement::Call(n)) => format!("Self = #{}", n),
            Some(_) => unreachable!(),
        })
        .collect()
}

#[test]
fn types_are_shared_by_calls() {
    // `Rel` is the same type in every call, so it is one entry.
    let a: ItemStruct = parse_quote!(
        struct A {
            a: Rel,
            b: RelA,
        }
    );
    let b: ItemStruct = parse_quote!(
        struct B<T> {
            a: Rel,
            b: RelB<u8>,
        }
    );
    let mut leaker = Leaker::new();
    let tops: Vec<_> = [&a, &b]
        .iter()
        .map(|item| {
            leaker
                .intern_with(&item.generics, |v| v.visit_item_struct(item))
                .unwrap()
        })
        .collect();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel", "RelA", "RelB < u8 >"]
    );
    // They use no generics, whichever the generics of the calls are: the generics of `A` (none)
    // are the context of the types, and those of `B` are another context without an entry.
    assert_eq!(referrer.generics(0).params.len(), 0);
    assert_eq!(referrer.contexts().count(), 2);
    assert_eq!(tops[0], referrer.context(0));
    assert_eq!(referrer.context_generics(tops[1]).params.len(), 1);
}

#[test]
fn equal_requirements_are_shared_and_different_ones_are_not() {
    let call = |generics: &str| -> ItemStruct {
        parse_str(&format!("struct S{} {{ a: Wrap<T> }}", generics)).unwrap()
    };
    let leak = |items: &[ItemStruct]| {
        let mut leaker = Leaker::new();
        for item in items {
            leaker
                .intern_with(&item.generics, |v| v.visit_item_struct(item))
                .unwrap();
        }
        leaker.finish()
    };
    // The same `<T>` (here with other generics unused by the type).
    let referrer = leak(&[call("<T>"), call("<T, U>"), call("<U, T>")]);
    assert_eq!(
        entries(&referrer),
        [entry(parse_quote!(Wrap<T>), parse_quote!(<T>))]
    );
    // The same up to the order and the place of the bounds.
    let referrer = leak(&[
        call("<T: ::core::fmt::Debug + ::core::clone::Clone>"),
        call("<T> where T: ::core::clone::Clone + ::core::fmt::Debug"),
    ]);
    assert_eq!(referrer.iter().count(), 1);
    // `<T>` and `<const T: usize>` are different.
    let referrer = leak(&[call("<T>"), call("<const T: usize>")]);
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Wrap<T>), parse_quote!(<T>)),
            entry(parse_quote!(Wrap<T>), parse_quote!(<const T: usize>)),
        ]
    );
    assert_eq!(referrer.contexts().count(), 2);
    assert_eq!(
        (referrer.context(0).index(), referrer.context(1).index()),
        (0, 1)
    );
    // The bounds are different.
    let referrer = leak(&[
        call("<T: ::core::clone::Clone>"),
        call("<T: ::core::marker::Copy>"),
    ]);
    assert_eq!(referrer.iter().count(), 2);
}

#[test]
fn only_the_generics_used_are_required() {
    let referrer = leak_trait(parse_quote! {
        trait X<T, U> {
            fn f(&self) -> Wrap<T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [entry(parse_quote!(Wrap<T>), parse_quote!(<T>))]
    );
    assert!(!referrer.has_self());

    // The generics of the bounds and the where predicates mentioning them are used.
    let referrer = leak_trait(parse_quote! {
        trait X<T, U, V> where U: ::core::convert::Into<T>, V: ::core::clone::Clone {
            fn f(&self) -> Wrap<T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [entry(
            parse_quote!(Wrap<T>),
            parse_quote!(<T, U: ::core::convert::Into<T>>)
        )]
    );

    // The generics of the Leaker and of a method are used together.
    let referrer = leak_trait(parse_quote! {
        trait X<'a, T, U> {
            fn f<'b, const B: bool>(&self) -> Wrap<'a, 'b, T, B>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [entry(
            parse_quote!(Wrap<'a, 'b, T, B>),
            parse_quote!(<'a, 'b, T, const B: bool>)
        )]
    );

    // A type which uses a generic of a method only is separate from the same one using also that of
    // the Leaker.
    let referrer = leak_trait(parse_quote! {
        trait X<T> {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g(&self) -> Wrap<T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Wrap<B>), parse_quote!(<const B: bool>)),
            entry(parse_quote!(Wrap<T>), parse_quote!(<T>)),
        ]
    );
}

#[test]
fn self_of_a_trait() {
    // `Self` is the implementor of the trait, and the parameters of the trait are used.
    let referrer = leak_trait(parse_quote! {
        trait X<'a, T, const N: usize, U> {
            fn f(&self) -> Rel<Self>;
            fn g(&self) -> Rel2;
            fn h(&self) -> Wrap<Self::Item, T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [
            entry(
                parse_quote!(Rel<Self>),
                parse_quote!(<'a, T, U, const N: usize>)
            ),
            entry(
                parse_quote!(Wrap<Self::Item, T>),
                parse_quote!(<'a, T, U, const N: usize>)
            ),
            entry(parse_quote!(Rel2), Generics::default()),
        ]
    );
    assert_eq!(
        selfs(&referrer),
        [
            "Self: X < 'a , T , N , U >",
            "Self: X < 'a , T , N , U >",
            "-"
        ]
    );
    // The two types are in one context, and the generics of the trait (without `Self`) are
    // another one.
    assert_eq!(referrer.contexts().count(), 3);
    assert_eq!(referrer.context(0), referrer.context(1));
    assert!(referrer.has_self());

    // A trait without parameters.
    let referrer = leak_trait(parse_quote!(
        trait X {
            fn f(&self) -> Rel<Self>;
        }
    ));
    assert_eq!(selfs(&referrer), ["Self: X"]);
}

#[test]
fn self_in_different_traits_is_different() {
    let referrer = leak_traits(&[
        parse_quote!(
            trait X<T> {
                fn f(&self) -> Rel<Self>;
                fn g(&self) -> Rel2;
            }
        ),
        parse_quote!(
            trait Y<T> {
                fn f(&self) -> Rel<Self>;
                fn g(&self) -> Rel2;
            }
        ),
    ]);
    // `Rel2` is shared, but `Rel<Self>` is not (`Self` is bound by another trait).
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel < Self >", "Rel2", "Rel < Self >"]
    );
    assert_eq!(selfs(&referrer), ["Self: X < T >", "-", "Self: Y < T >"]);
    // The generics of the traits are the same (`<T>`), so the calls share their top-level context.
    assert_eq!(referrer.contexts().count(), 4);

    // The same trait twice is shared.
    let tr: ItemTrait = parse_quote!(
        trait X<T> {
            fn f(&self) -> Rel<Self>;
        }
    );
    let referrer = leak_traits(&[tr.clone(), tr]);
    assert_eq!(referrer.iter().count(), 1);

    // `Self` is used by a type only if it mentions it, and it needs a requirement if the bound of
    // a generic it uses mentions it.
    let referrer = leak_trait(parse_quote! {
        trait X {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap<T>;
            fn g<T: ::core::cmp::PartialEq>(&self) -> Wrap<T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [
            entry(
                parse_quote!(Wrap<T>),
                parse_quote!(<T: ::core::cmp::PartialEq<Self>>)
            ),
            entry(
                parse_quote!(Wrap<T>),
                parse_quote!(<T: ::core::cmp::PartialEq>)
            ),
        ]
    );
    assert_eq!(selfs(&referrer), ["Self: X", "-"]);
}

#[test]
fn self_of_a_struct_enum_union_and_impl() {
    let mut leaker = Leaker::new();
    let st: ItemStruct = parse_quote!(
        struct S<T> {
            a: Rel<Self>,
            b: Wrap<T>,
        }
    );
    leaker
        .intern_with(&st.generics, |v| v.visit_item_struct(&st))
        .unwrap();
    let en: ItemEnum = parse_quote!(
        enum E<'a, T> {
            A(Rel<Self>, &'a T),
        }
    );
    leaker
        .intern_with(&en.generics, |v| v.visit_item_enum(&en))
        .unwrap();
    let un: ItemUnion = parse_quote!(union U { a: Rel<Self> });
    leaker
        .intern_with(&un.generics, |v| v.visit_item_union(&un))
        .unwrap();
    let im: ItemImpl = parse_quote! {
        impl<T> ::p::Tr for W<T> {
            fn f(&self) -> Rel<Self> { todo!() }
        }
    };
    leaker
        .intern_with(&im.generics, |v| v.visit_item_impl(&im))
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Rel<Self>), parse_quote!(<T>)),
            // The self type of the impl is leaked, and shared with the `Wrap<T>` using `T` only.
            entry(parse_quote!(W<T>), parse_quote!(<T>)),
            entry(parse_quote!(Wrap<T>), parse_quote!(<T>)),
            entry(parse_quote!(Rel<Self>), parse_quote!(<'a, T>)),
            entry(parse_quote!(Rel<Self>), Generics::default()),
            entry(parse_quote!(Rel<Self>), parse_quote!(<T>)),
        ]
    );
    assert_eq!(
        selfs(&referrer),
        [
            "Self = S < T >",
            "-",
            "-",
            "Self = E < 'a , T >",
            "Self = U",
            "Self = W < T >"
        ]
    );
}

#[test]
fn self_of_a_module_is_that_of_its_items() {
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct S<T> { a: Rel<Self, T> }
            pub trait Tr { fn f(&self) -> Rel2<Self>; }
        }
    };
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
    assert_eq!(selfs(&referrer), ["Self = S < T >", "Self: Tr"]);
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Rel<Self, T>), parse_quote!(<T>)),
            entry(parse_quote!(Rel2<Self>), Generics::default()),
        ]
    );
}

#[test]
fn self_without_an_item_is_the_call() {
    let ty: Type = parse_quote!(Rel<Self>);
    let other: Type = parse_quote!(Rel2);
    let mut leaker = Leaker::new();
    // The numbers are those of the successful calls.
    leaker
        .intern_with(&Generics::default(), |v| v.visit_type(&ty))
        .unwrap();
    leaker
        .intern_with(&Generics::default(), |v| v.visit_type(&other))
        .unwrap();
    let failing: Type = parse_quote!(Rel3<'_>);
    assert!(leaker
        .intern_with(&Generics::default(), |v| v.visit_type(&failing))
        .is_err());
    leaker
        .intern_with(&Generics::default(), |v| v.visit_type(&ty))
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel < Self >", "Rel2", "Rel < Self >"]
    );
    assert_eq!(selfs(&referrer), ["Self = #0", "-", "Self = #2"]);
}

#[test]
fn self_in_the_bounds_of_the_generics() {
    let tr: ItemTrait = parse_quote! {
        trait X {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap<T>;
        }
    };
    // It is leaked with the meaning of `Self`, if `Self` can be leaked.
    let referrer = leak_trait(tr.clone());
    assert_eq!(selfs(&referrer), ["Self: X"]);
    assert!(referrer.has_self());

    // Otherwise it cannot be, as the Repeater's implementation copies the bounds.
    let mut config = LeakerConfig::new();
    config.self_ty_can_be_interned = false;
    let mut leaker = Leaker::from_config(config.clone());
    let err = leaker
        .intern_with(&tr.generics, |v| v.visit_item_trait(&tr))
        .err()
        .unwrap();
    assert!(matches!(
        err,
        type_leak::NotInternableError::RelativePath(_)
    ));
    assert_eq!(
        err.note().as_deref(),
        Some("the bounds of its generics mention `Self`, which cannot be leaked")
    );
    assert!(err
        .to_string()
        .lines()
        .next()
        .unwrap()
        .contains("Wrap < T >"));
    // Nothing is added by a failed call.
    assert!(leaker.finish().is_empty());

    // A type which is not interned as a whole (it has no cause itself) is decomposed.
    let tr: ItemTrait = parse_quote! {
        trait X {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> (Rel, ::core::option::Option<T>);
        }
    };
    let mut leaker = Leaker::from_config(config.clone());
    leaker
        .intern_with(&tr.generics, |v| v.visit_item_trait(&tr))
        .unwrap();
    assert_eq!(leaker.finish().iter().map(s).collect::<Vec<_>>(), ["Rel"]);
}

#[test]
fn generic_args_of_the_root_and_local_generics() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T, U> {
            fn f<'b, const B: bool, V>(&self) -> Wrap<'a, 'b, T, B, V>;
        }
    };
    let referrer = leak_trait(tr.clone());
    assert_eq!(referrer.iter().count(), 1);
    // `'a` and `T` are the arguments of the trait, by position, and the others are passed by
    // their names.
    let completed: Vec<GenericArgument> =
        vec![parse_quote!('x), parse_quote!(u8), parse_quote!(u16)];
    let args = referrer
        .generic_args(0, &tr.generics.params, &completed)
        .unwrap();
    assert_eq!(
        args.iter().map(s).collect::<Vec<_>>(),
        ["'x", "'b", "u8", "V", "B"]
    );
    // A const argument of the trait is passed as it is.
    let tr: ItemTrait = parse_quote! {
        trait Tr<const N: usize, U> {
            fn f<V>(&self) -> Wrap<N, V>;
        }
    };
    let referrer = leak_trait(tr.clone());
    let completed: Vec<GenericArgument> = vec![parse_quote!({ 1 + 2 }), parse_quote!(u16)];
    let args = referrer
        .generic_args(0, &tr.generics.params, &completed)
        .unwrap();
    assert_eq!(args.iter().map(s).collect::<Vec<_>>(), ["V", "{ 1 + 2 }"]);
    // Too few arguments.
    assert!(matches!(
        referrer.generic_args(0, &tr.generics.params, &[]),
        Err(EncodeError::MissingArgument(_))
    ));
    // Without the parameters of the trait, everything is passed by name.
    let args = referrer.generic_args(0, &Punctuated::new(), &[]).unwrap();
    assert_eq!(args.iter().map(s).collect::<Vec<_>>(), ["V", "N"]);
}

fn rewrite<F: FnMut(Type, usize) -> Type>(mut visitor: Visitor<F>, sig: &str) -> String {
    let mut sig: Signature = parse_str(sig).unwrap();
    visitor.visit_signature_mut(&mut sig);
    s(&sig)
}

fn placeholder(_: Type, ix: usize) -> Type {
    parse_quote!(P<#ix>)
}

fn rewritten(sig: &str) -> String {
    s(&parse_str::<Signature>(sig).unwrap())
}

#[test]
fn lenient_matching() {
    let tr: ItemTrait = parse_quote! {
        trait X<T: ::core::clone::Clone, U> {
            fn f<const B: bool>(&self, a: Wrap<T>, b: Wrap<B>, c: Rel, d: Wrap<U>);
        }
    };
    let referrer = leak_trait(tr);
    // The generics which are not declared in the visited syntax are assumed to be those of the
    // Leaker, whatever their declaration is.
    assert_eq!(
        rewrite(
            referrer.clone().into_visitor(None, placeholder),
            "fn f<const B: bool>(&self, a: Wrap<T>, b: Wrap<B>, c: Rel, d: Wrap<U>)"
        ),
        rewritten(
            "fn f<const B: bool>(&self, a: P<0usize>, b: P<1usize>, c: P<2usize>, d: P<3usize>)"
        )
    );
    // The generics declared in the visited syntax must be declared in the same way, and the same
    // generics must be used.
    for (generics, a) in [
        ("<const B: bool, T: ::core::clone::Clone>", "P<0usize>"),
        ("<const B: bool, T>", "Wrap<T>"),
        ("<const B: bool, T: ::core::marker::Copy>", "Wrap<T>"),
        ("<const B: u8, T: ::core::clone::Clone>", "P<0usize>"),
    ] {
        let sig = format!("fn f{}(&self, a: Wrap<T>)", generics);
        let expected = format!("fn f{}(&self, a: {})", generics, a);
        assert_eq!(
            rewrite(referrer.clone().into_visitor(None, placeholder), &sig),
            rewritten(&expected),
            "{}",
            generics
        );
    }
    // `Wrap<B>` is for a `bool`.
    assert_eq!(
        rewrite(
            referrer.clone().into_visitor(None, placeholder),
            "fn f<const B: bool>(&self, b: Wrap<B>)"
        ),
        rewritten("fn f<const B: bool>(&self, b: P<1usize>)")
    );
    assert_eq!(
        rewrite(
            referrer.clone().into_visitor(None, placeholder),
            "fn f<const B: u8>(&self, b: Wrap<B>)"
        ),
        rewritten("fn f<const B: u8>(&self, b: Wrap<B>)")
    );
}

#[test]
fn matching_prefers_the_entry_assuming_the_fewest_generics() {
    // `B` is a `bool` in `f`, and `B` is not declared in `g` (it is a type there).
    let tr: ItemTrait = parse_quote! {
        trait X {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g(&self) -> Wrap<B>;
        }
    };
    let referrer = leak_trait(tr);
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(
        rewrite(
            referrer.clone().into_visitor(None, placeholder),
            "fn g(&self) -> Wrap<B>"
        ),
        rewritten("fn g(&self) -> P<1usize>")
    );
    assert_eq!(
        rewrite(
            referrer.into_visitor(None, placeholder),
            "fn f<const B: bool>(&self) -> Wrap<B>"
        ),
        rewritten("fn f<const B: bool>(&self) -> P<0usize>")
    );
}

#[test]
fn matching_with_the_root_generics_of_the_call() {
    // Two calls declare `T` differently, so the entries of `Wrap<T>` are ambiguous without the
    // generics of the call.
    let a: ItemTrait = parse_quote!(
        trait A<T: ::core::clone::Clone> {
            fn f(&self) -> Wrap<T>;
        }
    );
    let b: ItemTrait = parse_quote!(
        trait B<T: ::core::marker::Copy> {
            fn f(&self) -> Wrap<T>;
        }
    );
    let referrer = leak_traits(&[a.clone(), b.clone()]);
    assert_eq!(referrer.iter().count(), 2);
    let sig = "fn f(&self) -> Wrap<T>";
    // Leniently, the first of the entries is used.
    assert_eq!(
        rewrite(referrer.clone().into_visitor(None, placeholder), sig),
        rewritten("fn f(&self) -> P<0usize>")
    );
    // With the generics of the call, the entry is exact.
    for (tr, ix) in [(&a, 0usize), (&b, 1)] {
        assert_eq!(
            rewrite(
                referrer
                    .clone()
                    .into_visitor_with_root(&tr.generics, None, placeholder),
                sig
            ),
            rewritten(&format!("fn f(&self) -> P<{}usize>", ix))
        );
    }
    // No entry has these requirements.
    let other: Generics = parse_quote!(<T>);
    assert_eq!(
        rewrite(
            referrer
                .clone()
                .into_visitor_with_root(&other, None, placeholder),
            sig
        ),
        rewritten(sig)
    );
    // A type which does not use the generics is found whatever they are.
    let tr: ItemTrait = parse_quote!(
        trait C<T: ::core::clone::Clone> {
            fn f(&self) -> Rel;
        }
    );
    let referrer = leak_trait(tr);
    assert_eq!(
        rewrite(
            referrer.into_visitor_with_root(&other, None, placeholder),
            "fn f(&self) -> Rel"
        ),
        rewritten("fn f(&self) -> P<0usize>")
    );
}

#[test]
fn matching_self() {
    let x: ItemTrait = parse_quote!(
        trait X<T> {
            fn f(&self, a: Rel<Self>, b: Rel2, c: Wrap<T>);
        }
    );
    let y: ItemTrait = parse_quote!(
        trait Y<T> {
            fn f(&self, a: Rel<Self>, b: Rel2, c: Wrap<T>);
        }
    );
    let referrer = leak_traits(&[x.clone(), y.clone()]);
    // `Rel<Self>` of X, `Rel2`, `Wrap<T>` and `Rel<Self>` of Y.
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel < Self >", "Rel2", "Wrap < T >", "Rel < Self >"]
    );
    let sig = "fn f(&self, a: Rel<Self>, b: Rel2, c: Wrap<T>)";
    let bound = |name: &str| SelfRequirement::Bound(parse_str(&format!("{}<T>", name)).unwrap());
    // The entries which do not mention `Self` are not affected by it.
    assert_eq!(
        rewrite(
            referrer.clone().into_visitor(Some(bound("X")), placeholder),
            sig
        ),
        rewritten("fn f(&self, a: P<0usize>, b: P<1usize>, c: P<2usize>)")
    );
    assert_eq!(
        rewrite(
            referrer.clone().into_visitor(Some(bound("Y")), placeholder),
            sig
        ),
        rewritten("fn f(&self, a: P<3usize>, b: P<1usize>, c: P<2usize>)")
    );
    assert_eq!(
        rewrite(
            referrer.clone().into_visitor(Some(bound("Z")), placeholder),
            sig
        ),
        rewritten("fn f(&self, a: Rel<Self>, b: P<1usize>, c: P<2usize>)")
    );
    // Without it, `Self` is not checked, so the first entry is used.
    assert_eq!(
        rewrite(referrer.clone().into_visitor(None, placeholder), sig),
        rewritten("fn f(&self, a: P<0usize>, b: P<1usize>, c: P<2usize>)")
    );
    // With the root, it is exact: `Self` is the one given or, if it is not given, the entry's.
    assert_eq!(
        rewrite(
            referrer
                .clone()
                .into_visitor_with_root(&y.generics, Some(bound("Y")), placeholder),
            sig
        ),
        rewritten("fn f(&self, a: P<3usize>, b: P<1usize>, c: P<2usize>)")
    );
    assert_eq!(
        rewrite(
            referrer
                .clone()
                .into_visitor_with_root(&y.generics, Some(bound("Z")), placeholder),
            sig
        ),
        rewritten("fn f(&self, a: Rel<Self>, b: P<1usize>, c: P<2usize>)")
    );
    assert_eq!(
        rewrite(
            referrer.into_visitor_with_root(&y.generics, None, placeholder),
            sig
        ),
        rewritten("fn f(&self, a: P<0usize>, b: P<1usize>, c: P<2usize>)")
    );
}

#[test]
fn serialization_with_self() {
    let mut leaker = Leaker::new();
    let tr: ItemTrait = parse_quote!(
        trait X<T> {
            fn f(&self) -> Rel<Self>;
            fn g(&self) -> Wrap<T>;
        }
    );
    leaker
        .intern_with(&tr.generics, |v| v.visit_item_trait(&tr))
        .unwrap();
    let st: ItemStruct = parse_quote!(
        struct S<T> {
            a: RelS<Self>,
        }
    );
    leaker
        .intern_with(&st.generics, |v| v.visit_item_struct(&st))
        .unwrap();
    let ty: Type = parse_quote!(RelC<Self>);
    leaker
        .intern_with(&Generics::default(), |v| v.visit_type(&ty))
        .unwrap();
    let plain: ItemStruct = parse_quote!(
        struct P {
            a: RelP,
        }
    );
    leaker
        .intern_with(&plain.generics, |v| v.visit_item_struct(&plain))
        .unwrap();
    let referrer = leaker.finish();
    let tokens = quote!(#referrer);
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 5 \
             < T > ; Self : X < T > ; (Rel < Self >) \
             < T > ; (Wrap < T >) \
             < T > ; Self = S < T > ; (RelS < Self >) \
             Self = # 2 ; (RelC < Self >) \
             (RelP)",
            tag()
        )
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(selfs(&parsed), selfs(&referrer));
    assert_eq!(entries(&parsed), entries(&referrer));
    // A Referrer can be followed by other tokens.
    struct Followed(Referrer, Ident);
    impl parse::Parse for Followed {
        fn parse(input: parse::ParseStream) -> Result<Self> {
            Ok(Followed(input.parse()?, input.parse()?))
        }
    }
    let followed: Followed = parse2(quote!(#referrer after)).unwrap();
    assert_eq!(followed.0, referrer);
    assert_eq!(followed.1, "after");
    // Only a self part (no generics).
    let version = tag();
    let parsed: Referrer = parse2(quote!(#version Self : X ; (Rel < Self >))).unwrap();
    assert_eq!(selfs(&parsed), ["Self: X"]);
}

#[test]
fn node_numbers_follow_the_first_appearance() {
    // `Rel2` appears first, and then `Wrap<T>`, so the context of `Rel2` is 0 although the types
    // are sorted by tokens.
    let referrer = leak_trait(parse_quote! {
        trait X<T> {
            fn f(&self) -> Rel2;
            fn g(&self) -> Wrap<T>;
            fn h(&self) -> Rel1;
        }
    });
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel1", "Rel2", "Wrap < T >"]
    );
    assert_eq!(
        (0..3)
            .map(|ix| referrer.context(ix).index())
            .collect::<Vec<_>>(),
        [0, 0, 1]
    );
}

//! The lifetimes which the types around a type make it outlive are carried by the type when it is
//! leaked alone (see `Referrer::generics()`): `Rel2<T>` of `::core::cell::Ref<'a, Rel2<T>>` (where
//! `Ref<'a, T: 'a>`) is leaked with `<'a, T: 'a>`, as `Ref<'a, X>` is well-formed only if
//! `X: 'a`. rustc infers it for the original field of a struct (RFC 2093), but not through the
//! projection which the copy has instead of `Rel2<T>`.
//!
//! The Leaker and the Referrer compute it from the same rules (which components of a type carry
//! what the type outlives, see the table below), so that the Referrer finds the entry, in the
//! exact and in the lenient matching, and whether the generics of the Leaker are declared where it
//! looks the type up or not.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{Leaker, LeakerConfig, Referrer, SelfRequirement};

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    Leaker::from_config(config)
}

fn referrer_of(item: &ItemTrait) -> Referrer {
    let mut leaker = leaker();
    leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(item))
        .unwrap();
    leaker.finish()
}

/// The generics (and the where clause) of each leaked type, by its tokens.
fn contexts_of(referrer: &Referrer, ty: &str) -> Vec<String> {
    referrer
        .iter_with_generics()
        .filter(|(leaked, _)| s(leaked) == ty)
        .map(|(_, generics)| s(&quote!(#generics #{&generics.where_clause})))
        .collect()
}

/// The contexts of `Rel2<T>` in the method `f` of a trait whose parameter is `x: construct`
/// (`{X}` in it is `Rel2<T>`), with the generics `generics`.
fn rel2_in(generics: &str, construct: &str) -> Vec<String> {
    let construct = construct.replace("{X}", "Rel2<T>");
    let tr: ItemTrait = parse_str(&format!(
        "trait Tr {{ fn f{}(&self, x: {}); }}",
        generics, construct
    ))
    .unwrap();
    contexts_of(&referrer_of(&tr), "Rel2 < T >")
}

/// The components which carry what a type must outlive down, and those which do not, each
/// verified with rustc (the field of a struct with the type has the bound inferred, and its copy
/// with a projection in place of `Rel2<T>` is rejected with E0309 without it): `&'a X` requires
/// `X: 'a`, and so does what is made of types, but a projection. See `Enclosing` in the source.
#[test]
fn the_requirement_goes_down_through_the_components_it_goes_through_in_rustc() {
    const G: &str = "<'a, 'b, T>";
    let bounded = "< 'a , T : 'a >";
    for construct in [
        // A reference, and a raw pointer, a slice, an array, a tuple and parentheses in it.
        "&'a {X}",
        "&'a mut {X}",
        "&'a *const {X}",
        "&'a *mut {X}",
        "&'a [{X}]",
        "&'a [{X}; 3]",
        "&'a ({X}, u8)",
        "&'a (u8, {X})",
        "&'a ({X})",
        // The type arguments of a path with a lifetime argument (assumed), and of a path without.
        "::core::cell::Ref<'a, {X}>",
        "::core::cell::Ref<'a, ::std::vec::Vec<{X}>>",
        "&'a ::std::option::Option<::std::vec::Vec<{X}>>",
        "&'a ::core::cell::Cell<{X}>",
        "&'a mut ::std::boxed::Box<{X}>",
        // The inputs and the output of a function pointer, with a binder or not.
        "&'a fn({X})",
        "&'a fn() -> {X}",
        "&'a for<'x> fn(&'x {X}) -> &'x u8",
        "&'a fn() -> ({X}, u8)",
        // The arguments of a trait object: `dyn Tr<X> + 'b: 'a` requires `X: 'a` as well as
        // `'b: 'a`, whatever the lifetime bound is (also `'static`).
        "::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = {X}>>",
        "&'a dyn ::core::iter::Iterator<Item = {X}>",
        "&'a (dyn ::core::convert::AsRef<{X}> + 'static)",
        "&'a dyn for<'x> ::core::ops::Fn(&'x {X})",
        "&'a ::std::boxed::Box<dyn ::core::ops::Fn({X}) -> u8>",
        "&'a ::std::boxed::Box<dyn ::core::ops::Fn() -> {X}>",
        "&'a ::std::vec::Vec<::std::boxed::Box<dyn ::core::iter::Iterator<Item = {X}>>>",
    ] {
        assert_eq!(rel2_in(G, construct), [bounded], "{}", construct);
    }
    // Not through a projection: `<X as Tr>::A` outlives `'a` if its components do, but it does
    // not require `X: 'a` (it does not for the original struct either).
    for construct in [
        "&'a <{X} as ::core::ops::Deref>::Target",
        "&'a <u8 as ::core::convert::Into<{X}>>::Output",
        "&'a ::std::vec::Vec<<{X} as ::core::ops::Deref>::Target>",
    ] {
        assert_eq!(rel2_in(G, construct), ["< T >"], "{}", construct);
    }
    // Nor through the length of an array (an expression, whose types are not components).
    assert_eq!(
        contexts_of(
            &referrer_of(&parse_quote! {
                trait Tr { fn f<'a, T>(&self, x: &'a [u8; ::core::mem::size_of::<Rel2<T>>()]); }
            }),
            "Rel2 < T >"
        ),
        Vec::<String>::new(),
        "the type in an array length is not leaked alone"
    );
}

#[test]
fn every_lifetime_around_the_type_bounds_it() {
    // Both lifetimes of `&'a &'b X`, in the canonical order.
    assert_eq!(
        rel2_in("<'a, 'b, T>", "&'a &'b {X}"),
        ["< 'a , 'b , T : 'a + 'b >"]
    );
    assert_eq!(
        rel2_in("<'a, 'b, T>", "&'b &'a {X}"),
        ["< 'a , 'b , T : 'a + 'b >"]
    );
    // A path with its own lifetime argument (assumed) and the reference around it.
    assert_eq!(
        rel2_in(
            "<'a, 'b, T>",
            "&'a ::core::cell::Ref<'b, ::std::vec::Vec<{X}>>"
        ),
        ["< 'a , 'b , T : 'a + 'b >"]
    );
    // Nested: the inner lifetime bounds the type too, as does what is outside of the reference.
    assert_eq!(
        rel2_in(
            "<'a, 'b, T>",
            "::core::cell::Ref<'a, &'b ::core::cell::Ref<'b, {X}>>"
        ),
        ["< 'a , 'b , T : 'a + 'b >"]
    );
    // The lifetimes in the type are bounded as well (`'c: 'a`).
    let tr: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, 'c>(&self, x: &'a Rel2<&'c u8>, y: ::core::cell::Ref<'a, Rel2<&'c u8>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < & 'c u8 >"),
        ["< 'a , 'c : 'a >"]
    );
    // A type which mentions no generic has nothing to bound: its context has no generics.
    let tr: ItemTrait = parse_quote! {
        trait Tr { fn f<'a>(&self, x: &'a Rel, y: ::core::cell::Ref<'a, Rel>); }
    };
    assert_eq!(contexts_of(&referrer_of(&tr), "Rel"), [""]);
    // The bounds which are written are merged with them, and so are those from the inside of the
    // type (the bound is not duplicated).
    assert_eq!(
        rel2_in("<'a, T: 'a + ::core::clone::Clone>", "&'a {X}"),
        ["< 'a , T : 'a + :: core :: clone :: Clone >"]
    );
    let tr: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: &'a Rel<'a, T>, y: ::core::cell::Ref<'a, Rel<'a, T>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel < 'a , T >"),
        ["< 'a , T : 'a >"]
    );
    // The generics of the trait.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> { fn f(&self, x: ::core::cell::Ref<'a, Rel2<T>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < T >"),
        ["< 'a , T : 'a >"]
    );
    // A lifetime of the trait and a type of the method, and the other way round.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a> { fn f<T>(&self, x: ::core::cell::Ref<'a, Rel2<T>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < T >"),
        ["< 'a , T : 'a >"]
    );
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> { fn f<'a>(&self, x: ::core::cell::Ref<'a, Rel2<T>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < T >"),
        ["< 'a , T : 'a >"]
    );
}

#[test]
fn a_const_parameter_is_not_bounded() {
    // Nothing to bound by the lifetime: it is not among the generics of the type (it would only
    // be a generic nothing refers to).
    let tr: ItemTrait = parse_quote! {
        trait Tr<const N: usize> {
            fn f<'a>(&self, x: &'a Rel2<N>, y: ::core::cell::Ref<'a, Rel2<N>>);
        }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < N >"),
        ["< const N : usize >"]
    );
    // With a type parameter next to it, only the type parameter is.
    let tr: ItemTrait = parse_quote! {
        trait Tr<const N: usize> {
            fn f<'a, T>(&self, y: ::core::cell::Ref<'a, Rel2<N, T>>);
        }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < N , T >"),
        ["< 'a , T : 'a , const N : usize >"]
    );
}

#[test]
fn what_is_not_a_generic_lifetime_bounds_nothing() {
    // `'static`, `'_` (which is not leaked alone).
    assert_eq!(
        rel2_in("<T: 'static>", "::core::cell::Ref<'static, {X}>"),
        ["< T : 'static >"]
    );
    assert_eq!(rel2_in("<T>", "&'static {X}"), ["< T >"]);
    // The lifetime of a `for<..>` binder: the copy cannot declare it, and rustc does not need it.
    assert_eq!(rel2_in("<T>", "for<'x> fn(&'x {X}) -> usize"), ["< T >"]);
    assert_eq!(
        rel2_in(
            "<T>",
            "::std::boxed::Box<dyn for<'x> ::core::ops::Fn(&'x {X}) -> u8>"
        ),
        ["< T >"]
    );
    // Not across it: what the types around the binder require stays (`'a` is not bound by it).
    assert_eq!(
        rel2_in("<'a, T>", "&'a for<'x> fn(&'x {X}) -> usize"),
        ["< 'a , T : 'a >"]
    );
    // A binder which shadows nothing here, with the lifetime of the trait declared as a binder.
    assert_eq!(
        rel2_in("<'a, T>", "for<'a> fn(&'a {X}) -> usize"),
        ["< T >"]
    );
    // A lifetime which no scope declares is not a generic.
    assert_eq!(rel2_in("<T>", "&'_ {X}"), ["< T >"]);
    // With nothing around it.
    assert_eq!(rel2_in("<T>", "{X}"), ["< T >"]);
    assert_eq!(
        rel2_in("<'a, T>", "(&'a u8, {X})"),
        ["< T >"],
        "the lifetime of a neighbour is not around it"
    );
}

#[test]
fn the_same_type_under_different_lifetimes_is_another_entry() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> {
            fn f(
                &self,
                x: ::core::cell::Ref<'a, Rel2<T>>,
                y: Rel2<T>,
                z: &'a Rel2<T>,
            );
        }
    };
    let referrer = referrer_of(&tr);
    let mut contexts = contexts_of(&referrer, "Rel2 < T >");
    contexts.sort();
    // The reference is leaked as a whole (it has a lifetime of the trait).
    assert_eq!(contexts, ["< 'a , T : 'a >", "< T >"]);
    assert_eq!(
        contexts_of(&referrer, "& 'a Rel2 < T >"),
        ["< 'a , T : 'a >"]
    );
}

#[test]
fn self_is_bounded_too() {
    // `Self: 'a` is part of the context of `Rel2<Self>`, like `T: 'a` of `Rel2<T>`.
    let tr: ItemTrait = parse_quote! {
        trait Tr {
            fn f<'a>(&self, x: ::core::cell::Ref<'a, Rel2<Self>>, y: Rel2<Self>)
            where Self: ::core::marker::Sized;
        }
    };
    let referrer = referrer_of(&tr);
    let found: Vec<(String, Vec<String>)> = referrer
        .iter()
        .enumerate()
        .map(|(ix, _)| {
            let context = referrer.context(ix);
            assert!(matches!(
                referrer.context_self(context),
                Some(SelfRequirement::Bound(_))
            ));
            (
                s(&referrer.context_generics(context).params),
                referrer
                    .context_self_outlives(context)
                    .iter()
                    .map(|l| l.to_string())
                    .collect(),
            )
        })
        .collect();
    assert_eq!(found.len(), 2);
    // The lifetime is a generic of the context, so that `&'a Self` can be encoded, and the entry
    // with nothing around the type has neither.
    assert!(
        found.contains(&("'a".to_string(), vec!["'a".to_string()])),
        "{:?}",
        found
    );
    assert!(found.contains(&(String::new(), vec![])), "{:?}", found);
}

/// How a Referrer looks the types up.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// The generics of the item visited are declared (and so are those of the trait if the item
    /// is a copy of a signature, which is not the case here), the others assumed.
    Lenient,
    /// The generics of the outermost item visited are not declared.
    Leakers,
    /// The exact contexts, with the generics of the trait as the root.
    Exact,
}

const MODES: [Mode; 3] = [Mode::Lenient, Mode::Leakers, Mode::Exact];

/// Rewrites `item`, replacing each leaked type by `P<ix>`.
fn rewrite(referrer: &Referrer, root: &Generics, mode: Mode, item: &Item) -> String {
    let mut item = item.clone();
    let replace = |_: Type, ix: usize| -> Type {
        let ix = proc_macro2::Literal::usize_unsuffixed(ix);
        parse_quote!(P<#ix>)
    };
    match mode {
        Mode::Lenient => referrer
            .clone()
            .into_visitor(None, replace)
            .visit_item_mut(&mut item),
        Mode::Leakers => referrer
            .clone()
            .into_visitor(None, replace)
            .with_outermost_generics_as_leakers()
            .visit_item_mut(&mut item),
        Mode::Exact => referrer
            .clone()
            .into_visitor_with_root(root, None, replace)
            .visit_item_mut(&mut item),
    }
    s(&item)
}

/// The index of the entry of the type `ty` with the generics `generics`.
fn entry(referrer: &Referrer, ty: &str, generics: &str) -> usize {
    referrer
        .iter_with_generics()
        .position(|(leaked, g)| s(leaked) == ty && s(&quote!(#g #{&g.where_clause})) == generics)
        .unwrap_or_else(|| panic!("no entry {} {}", ty, generics))
}

#[test]
fn the_referrer_finds_the_entry_of_a_type_under_the_lifetimes_around_it() {
    let tr: ItemTrait = parse_quote! {
        trait Tr {
            fn f<'a, T>(&self, x: ::core::cell::Ref<'a, Rel2<T>>, y: Rel2<T>);
        }
    };
    let referrer = referrer_of(&tr);
    let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
    let alone = entry(&referrer, "Rel2 < T >", "< T >");
    assert_ne!(around, alone);
    let cases: Vec<(Item, usize)> = vec![
        // A struct field: what the copy needs, in each of the lookups.
        (
            parse_quote!(
                struct F<'a, T>(::core::cell::Ref<'a, Rel2<T>>);
            ),
            around,
        ),
        (
            parse_quote!(
                struct G<'a, T>(&'a Rel2<T>);
            ),
            around,
        ),
        (
            parse_quote!(
                struct H<'a, T>(::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>);
            ),
            around,
        ),
        (
            parse_quote!(
                struct D<'a, T>(::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Rel2<T>>>);
            ),
            around,
        ),
        // Nothing around it, whichever lifetimes are declared.
        (
            parse_quote!(
                struct V<T>(Rel2<T>);
            ),
            alone,
        ),
        (
            parse_quote!(
                struct W<'a, T>(Rel2<T>, ::core::marker::PhantomData<&'a ()>);
            ),
            alone,
        ),
        // A projection does not carry them.
        (
            parse_quote!(
                struct Q<'a, T>(&'a <Rel2<T> as ::core::ops::Deref>::Target);
            ),
            alone,
        ),
        // A binder lifetime does not.
        (
            parse_quote!(
                struct B<T>(for<'x> fn(&'x Rel2<T>) -> usize);
            ),
            alone,
        ),
    ];
    for (item, ix) in cases {
        let item: Item = item;
        for mode in MODES {
            let copy = rewrite(&referrer, &tr.generics, mode, &item);
            assert!(!copy.contains("Rel2"), "{:?}: {}", mode, copy);
            assert!(
                copy.contains(&format!("P < {} >", ix)),
                "{:?}: {} has not P<{}>",
                mode,
                copy,
                ix
            );
            let other = if ix == around { alone } else { around };
            assert!(
                !copy.contains(&format!("P < {} >", other)),
                "{:?}: {}",
                mode,
                copy
            );
        }
    }
}

#[test]
fn the_referrer_does_not_mistake_a_type_for_another_lifetime() {
    let tr: ItemTrait = parse_quote! {
        trait Tr {
            fn f<'a, T>(&self, x: ::core::cell::Ref<'a, Rel2<T>>);
        }
    };
    let referrer = referrer_of(&tr);
    let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
    // The lifetime is named differently: the generics declared here are not those of the entry.
    let item: Item = parse_quote!(
        struct F<'b, T>(::core::cell::Ref<'b, Rel2<T>>);
    );
    for mode in [Mode::Lenient, Mode::Exact] {
        let copy = rewrite(&referrer, &tr.generics, mode, &item);
        assert!(copy.contains("Rel2"), "{:?}: {}", mode, copy);
    }
    // The same bounds, with other generics around: not replaced either in the exact lookup.
    let item: Item = parse_quote!(
        struct F<'a, 'b: 'a, T>(::core::cell::Ref<'a, Rel2<T>>);
    );
    let copy = rewrite(&referrer, &tr.generics, Mode::Exact, &item);
    assert!(copy.contains(&format!("P < {} >", around)), "{}", copy);
}

#[test]
fn the_referrer_tells_the_entries_apart_with_the_generics_of_the_leaker_undeclared() {
    // The generics of the trait are not declared where the signature is copied, so the Referrer
    // assumes them: the entry with the fewest assumed generics would be the one without the
    // lifetime, were it not for the bounds which the type implies where it is found.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> {
            fn f(
                &self,
                x: ::core::cell::Ref<'a, Rel2<T>>,
                y: Rel2<T>,
                z: ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>,
            );
        }
    };
    let referrer = referrer_of(&tr);
    let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
    let alone = entry(&referrer, "Rel2 < T >", "< T >");
    // `expand()` declares nothing.
    let expand = |ty: Type| -> String {
        s(&referrer.expand(ty, |_, ix| {
            let ix = proc_macro2::Literal::usize_unsuffixed(ix);
            parse_quote!(P<#ix>)
        }))
    };
    assert_eq!(
        expand(parse_quote!(::core::cell::Ref<'a, Rel2<T>>)),
        format!(":: core :: cell :: Ref < 'a , P < {} > >", around)
    );
    assert_eq!(expand(parse_quote!(Rel2<T>)), format!("P < {} >", alone));
    assert_eq!(
        expand(parse_quote!(
            ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>
        )),
        format!(
            ":: core :: cell :: Ref < 'a , :: std :: vec :: Vec < P < {} > > >",
            around
        )
    );
    // The signatures, with the generics of the trait undeclared (the default of the visitor
    // `Referrer::into_visitor()`, for a signature, declares only those of the method).
    let sig: Signature = parse_quote!(
        fn f(
            &self,
            x: ::core::cell::Ref<'a, Rel2<T>>,
            y: Rel2<T>,
            z: ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>,
        )
    );
    let replace = |_: Type, ix: usize| -> Type {
        let ix = proc_macro2::Literal::usize_unsuffixed(ix);
        parse_quote!(P<#ix>)
    };
    for mut visitor in [
        referrer.clone().into_visitor(None, replace),
        referrer
            .clone()
            .into_visitor(None, replace)
            .with_outermost_generics_as_leakers(),
        // The generics of the trait are the root, those of the method are declared.
        referrer
            .clone()
            .into_visitor_with_root(&tr.generics, None, replace),
    ] {
        let mut copy = sig.clone();
        visitor.visit_signature_mut(&mut copy);
        let copy = s(&copy);
        assert!(!copy.contains("Rel2"), "{}", copy);
        assert!(copy.contains(&format!("P < {} >", alone)), "{}", copy);
        assert!(
            copy.contains(&format!("Ref < 'a , P < {} > >", around)),
            "{}",
            copy
        );
        assert!(
            copy.contains(&format!(
                "Ref < 'a , :: std :: vec :: Vec < P < {} > > >",
                around
            )),
            "{}",
            copy
        );
    }
}

#[test]
fn the_referrer_finds_the_entry_with_a_lifetime_of_the_method() {
    // The lifetime is declared by the method, and the type by the trait (assumed), the way round
    // of the lenient lookup of the generics which the entry has and the type here does not
    // mention.
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> {
            fn f<'a>(&self, x: ::core::cell::Ref<'a, Rel2<T>>, y: Rel2<T>);
        }
    };
    let referrer = referrer_of(&tr);
    let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
    let alone = entry(&referrer, "Rel2 < T >", "< T >");
    let sig: Signature = parse_quote!(
        fn f<'a>(&self, x: ::core::cell::Ref<'a, Rel2<T>>, y: Rel2<T>)
    );
    let replace = |_: Type, ix: usize| -> Type {
        let ix = proc_macro2::Literal::usize_unsuffixed(ix);
        parse_quote!(P<#ix>)
    };
    for mut visitor in [
        referrer.clone().into_visitor(None, replace),
        referrer
            .clone()
            .into_visitor(None, replace)
            .with_outermost_generics_as_leakers(),
        // The generics of the trait are the root, those of the method are declared.
        referrer
            .clone()
            .into_visitor_with_root(&tr.generics, None, replace),
    ] {
        let mut copy = sig.clone();
        visitor.visit_signature_mut(&mut copy);
        let copy = s(&copy);
        assert!(!copy.contains("Rel2"), "{}", copy);
        assert!(
            copy.contains(&format!("Ref < 'a , P < {} > >", around)),
            "{}",
            copy
        );
        assert!(copy.contains(&format!("y : P < {} >", alone)), "{}", copy);
    }
}

#[test]
fn the_entries_of_the_types_around_survive_serialization() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> {
            fn f<'b, U>(
                &self,
                x: ::core::cell::Ref<'b, Rel2<U>>,
                y: Rel2<U>,
                z: &'a ::std::vec::Vec<Rel2<T>>,
            );
        }
    };
    let referrer = referrer_of(&tr);
    let tokens = referrer.to_token_stream();
    let parsed: Referrer = parse2(tokens.clone()).unwrap();
    assert_eq!(parsed, referrer);
    // The lifetime of the type around is a generic of the context, written in it.
    assert!(tokens.to_string().contains("'b , U : 'b"), "{}", tokens);
    // And the entries are found in the parsed one.
    let item: Item = parse_quote!(
        struct F<'b, U>(::core::cell::Ref<'b, Rel2<U>>);
    );
    let replace = |_: Type, ix: usize| -> Type {
        let ix = proc_macro2::Literal::usize_unsuffixed(ix);
        parse_quote!(P<#ix>)
    };
    let mut copy = item.clone();
    parsed.into_visitor(None, replace).visit_item_mut(&mut copy);
    assert!(!s(&copy).contains("Rel2"), "{}", s(&copy));
}

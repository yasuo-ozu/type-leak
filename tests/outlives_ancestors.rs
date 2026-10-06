//! The lifetimes which the references around a type make it outlive are carried by the type when it
//! is leaked alone (see `Referrer::generics()`): `Rel2<T>` of `&'a Vec<Rel2<T>>` is leaked with
//! `<'a, T: 'a>`, as `&'a X` is well-formed only if `X: 'a` (the language rule), which holds for
//! the original but not through the projection which the copy has instead of `Rel2<T>`.
//!
//! Only a reference adds a lifetime: the lifetime argument of a path (`Ref<'a, Rel2<T>>`) does
//! not, as it is well-formed without `T: 'a` unless the definition of the path says so, which is
//! not visible (`struct Unrelated<'a, T>(&'a (), T)`), so the bound is not assumed: `Rel2<T>` of
//! `Ref<'a, Rel2<T>>` is leaked with `<T>`, and the bound is written where it is needed (`T: 'a`
//! in the signature, which makes it `<'a, T: 'a>` as written). A lifetime which is active (that of
//! a reference around) is passed down through the components, whatever they are (the type
//! arguments of a path, the elements of a tuple, the inputs and the output of a function pointer,
//! the arguments of a trait object, ..), as rustc does.
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
/// `X: 'a`, and so does what is made of types, but a projection. A path with a lifetime argument
/// passes down what is required of it (`&'a Ref<'b, X>`), but does not add its own lifetime. See
/// `Enclosing` in the source.
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
        // The type arguments of a path with a lifetime argument (the one of the reference is the
        // active lifetime), and of a path without.
        "&'a ::core::cell::Ref<'a, {X}>",
        "&'a ::core::cell::Ref<'static, ::std::vec::Vec<{X}>>",
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
        "&'a ::core::cell::Ref<'b, dyn ::core::iter::Iterator<Item = {X}>>",
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
    // A path with its own lifetime argument under the reference passes the lifetime of the
    // reference down, and adds none (`'b` is not among the generics, nor bounds the type).
    assert_eq!(
        rel2_in(
            "<'a, 'b, T>",
            "&'a ::core::cell::Ref<'b, ::std::vec::Vec<{X}>>"
        ),
        ["< 'a , T : 'a >"]
    );
    // Nested: the inner reference's lifetime bounds the type too, as does what is outside of it,
    // and the lifetime arguments of the paths between them do not.
    assert_eq!(
        rel2_in(
            "<'a, 'b, T>",
            "&'a ::core::cell::Ref<'a, &'b ::core::cell::Ref<'b, {X}>>"
        ),
        ["< 'a , 'b , T : 'a + 'b >"]
    );
    // The lifetimes in the type are bounded as well (`'c: 'a`), by the reference only.
    let tr: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, 'c>(&self, x: &'a Rel2<&'c u8>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < & 'c u8 >"),
        ["< 'a , 'c : 'a >"]
    );
    let tr: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, 'c>(&self, y: ::core::cell::Ref<'a, Rel2<&'c u8>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < & 'c u8 >"),
        ["< 'c >"],
        "a path's own lifetime is not related to the lifetimes in its argument"
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
    // `Rel<'a, T>` is leaked as written inside a reference, as it is under a path.
    let tr: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: &'a Rel<'a, T>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel < 'a , T >"),
        ["< 'a , T : 'a >"]
    );
    let tr: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, T>(&self, y: ::core::cell::Ref<'a, Rel<'a, T>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel < 'a , T >"),
        ["< 'a , T >"]
    );
    // The generics of the trait: the reference is leaked as a whole if its lifetime is a generic
    // of the trait, except when it is to a trait object.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> { fn f(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < T >"),
        ["< 'a , T : 'a >"]
    );
    // A lifetime of the trait and a type of the method, and the other way round.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a> { fn f<T>(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < T >"),
        ["< 'a , T : 'a >"]
    );
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> { fn f<'a>(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>); }
    };
    assert_eq!(
        contexts_of(&referrer_of(&tr), "Rel2 < T >"),
        ["< 'a , T : 'a >"]
    );
}

/// A path with a lifetime argument does not make the types inside it outlive the lifetime: it is
/// well-formed without `T: 'a` unless its definition says so (it is not visible, and `struct
/// Unrelated<'a, T>(&'a (), T)` does not), so no bound is assumed, neither for the path itself
/// nor for the types inside it. A written bound is a bound like any other.
#[test]
fn a_path_with_a_lifetime_argument_adds_no_bound() {
    // The type inside the path.
    assert_eq!(rel2_in("<'a, T>", "::core::cell::Ref<'a, {X}>"), ["< T >"]);
    assert_eq!(
        rel2_in("<'a, T>", "::core::cell::Ref<'a, ::std::vec::Vec<{X}>>"),
        ["< T >"]
    );
    assert_eq!(
        rel2_in(
            "<'a, T>",
            "::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = {X}>>"
        ),
        ["< T >"]
    );
    // Also with several lifetime arguments, and in the arguments of a path inside it.
    assert_eq!(
        rel2_in("<'a, 'b, T>", "::p::Two<'a, 'b, ::p::Wrap<{X}>>"),
        ["< T >"]
    );
    // A written bound stays as it is written.
    assert_eq!(
        rel2_in("<'a, T: 'a>", "::core::cell::Ref<'a, {X}>"),
        ["< 'a , T : 'a >"]
    );
    // The path itself, which is leaked as a whole when it is relative: the generics it uses, with
    // no bound between them, whatever the lifetime and the types are.
    for (generics, construct, leaked, expected) in [
        ("<'a, T>", "Rel<'a, T>", "Rel < 'a , T >", "< 'a , T >"),
        (
            "<'a, 'b, T>",
            "Rel<'a, 'b, (T, u8)>",
            "Rel < 'a , 'b , (T , u8) >",
            "< 'a , 'b , T >",
        ),
        (
            "<'a, 'b, T>",
            "Rel<'a, Rel<'b, T>>",
            "Rel < 'a , Rel < 'b , T > >",
            "< 'a , 'b , T >",
        ),
        (
            "<'a, 'b>",
            "Rel<'a, &'b u8>",
            "Rel < 'a , & 'b u8 >",
            "< 'a , 'b >",
        ),
        // The lifetimes of the arguments are bounded by a reference only (the path is a
        // component of it, leaked alone since the lifetime of the reference is a local one).
        (
            "<'a, 'b, T>",
            "&'a Rel<'b, T>",
            "Rel < 'b , T >",
            "< 'a , 'b : 'a , T : 'a >",
        ),
        // The bounds which are written are the generics' own.
        (
            "<'a, T: 'a>",
            "Rel<'a, T>",
            "Rel < 'a , T >",
            "< 'a , T : 'a >",
        ),
    ] {
        let tr: ItemTrait = parse_str(&format!(
            "trait Tr {{ fn f{}(&self, x: {}); }}",
            generics, construct
        ))
        .unwrap();
        assert_eq!(
            contexts_of(&referrer_of(&tr), leaked),
            [expected],
            "{}",
            construct
        );
    }
    // The lifetime of a reference around is passed down through the path, and the path's own is
    // not added: the type inside is leaked with the first only.
    assert_eq!(
        rel2_in("<'a, 'b, T>", "&'a ::core::cell::Ref<'b, {X}>"),
        ["< 'a , T : 'a >"]
    );
    assert_eq!(
        rel2_in("<'a, 'b, T>", "&'a Vec<::core::cell::Ref<'b, {X}>>"),
        ["< 'a , T : 'a >"]
    );
    // Neither does `Self`: `Self: 'a` comes from a reference or is written.
    let referrer = referrer_of(&parse_quote! {
        trait Tr {
            fn f<'a>(&self, x: ::core::cell::Ref<'a, Rel2<Self>>, y: Rel<'a, Self>)
            where Self: Sized;
        }
    });
    for ix in 0..referrer.iter().count() {
        assert!(
            referrer
                .context_self_outlives(referrer.context(ix))
                .is_empty(),
            "{}",
            s(referrer.iter().nth(ix).unwrap())
        );
    }
    assert_eq!(
        contexts_of(&referrer, "Rel < 'a , Self >"),
        ["< 'a > where Self : Sized"]
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
            fn f<'a, T>(&self, y: &'a dyn ::core::iter::Iterator<Item = Rel2<N, T>>);
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
                x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
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
    // A path with the lifetime is not another entry: it is the one of the type with nothing
    // around it.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> {
            fn f(&self, x: ::core::cell::Ref<'a, Rel2<T>>, y: Rel2<T>);
        }
    };
    assert_eq!(contexts_of(&referrer_of(&tr), "Rel2 < T >"), ["< T >"]);
}

#[test]
fn self_is_bounded_too() {
    // `Self: 'a` is part of the context of `Rel2<Self>` under a reference, like `T: 'a` of
    // `Rel2<T>`.
    let tr: ItemTrait = parse_quote! {
        trait Tr {
            fn f<'a>(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<Self>>, y: Rel2<Self>)
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
            fn f<'a, T>(
                &self,
                x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                y: Rel2<T>,
            );
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
                struct G<'a, T>(&'a Rel2<T>);
            ),
            around,
        ),
        (
            parse_quote!(
                struct H<'a, T>(&'a ::std::vec::Vec<Rel2<T>>);
            ),
            around,
        ),
        (
            parse_quote!(
                struct D<'a, T>(&'a dyn ::core::iter::Iterator<Item = Rel2<T>>);
            ),
            around,
        ),
        // The lifetime of the reference is passed down through the path with a lifetime, and its
        // own is not.
        (
            parse_quote!(
                struct K<'a, 'b, T>(&'a ::core::cell::Ref<'b, Rel2<T>>);
            ),
            around,
        ),
        // A path with a lifetime does not carry it: the type is the one with nothing around it,
        // whichever lifetimes are declared.
        (
            parse_quote!(
                struct F<'a, T>(::core::cell::Ref<'a, Rel2<T>>);
            ),
            alone,
        ),
        (
            parse_quote!(
                struct E<'a, T>(::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>);
            ),
            alone,
        ),
        (
            parse_quote!(
                struct O<'a, T>(::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Rel2<T>>>);
            ),
            alone,
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
    // The bound is written in the struct: the same generics as the entry of the reference's
    // component, so it is found where the generics of the struct are seen declared (not with the
    // outermost generics as the Leaker's, where `T: 'a` is not seen and the entry with the fewest
    // generics assumed is the one with nothing around).
    let item: Item = parse_quote!(
        struct F<'a, T: 'a>(::core::cell::Ref<'a, Rel2<T>>);
    );
    for mode in [Mode::Lenient, Mode::Exact] {
        let copy = rewrite(&referrer, &tr.generics, mode, &item);
        assert!(!copy.contains("Rel2"), "{:?}: {}", mode, copy);
        assert!(
            copy.contains(&format!("P < {} >", around)),
            "{:?}: {}",
            mode,
            copy
        );
    }
}

#[test]
fn the_referrer_does_not_mistake_a_type_for_another_lifetime() {
    let tr: ItemTrait = parse_quote! {
        trait Tr {
            fn f<'a, T>(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>);
        }
    };
    let referrer = referrer_of(&tr);
    let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
    // The lifetime is named differently: the generics declared here are not those of the entry.
    let item: Item = parse_quote!(
        struct F<'b, T>(&'b dyn ::core::iter::Iterator<Item = Rel2<T>>);
    );
    for mode in [Mode::Lenient, Mode::Exact] {
        let copy = rewrite(&referrer, &tr.generics, mode, &item);
        assert!(copy.contains("Rel2"), "{:?}: {}", mode, copy);
    }
    // The same bounds, with other generics around: not replaced either in the exact lookup.
    let item: Item = parse_quote!(
        struct F<'a, 'b: 'a, T>(&'a dyn ::core::iter::Iterator<Item = Rel2<T>>);
    );
    let copy = rewrite(&referrer, &tr.generics, Mode::Exact, &item);
    assert!(copy.contains(&format!("P < {} >", around)), "{}", copy);
    // The lifetime of a path is not the lifetime of the entry either: it is no lifetime around
    // the type, so the entry of the type alone is the one which is found, not this one.
    let item: Item = parse_quote!(
        struct F<'a, T>(::core::cell::Ref<'a, Rel2<T>>);
    );
    for mode in [Mode::Lenient, Mode::Exact] {
        let copy = rewrite(&referrer, &tr.generics, mode, &item);
        assert!(copy.contains("Rel2"), "{:?}: {}", mode, copy);
    }
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
                x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                y: Rel2<T>,
                z: &'a dyn ::core::iter::Iterator<Item = ::std::vec::Vec<Rel2<T>>>,
                w: ::core::cell::Ref<'a, Rel2<T>>,
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
        expand(parse_quote!(&'a dyn ::core::iter::Iterator<Item = Rel2<T>>)),
        format!(
            "& 'a dyn :: core :: iter :: Iterator < Item = P < {} > >",
            around
        )
    );
    assert_eq!(expand(parse_quote!(Rel2<T>)), format!("P < {} >", alone));
    assert_eq!(
        expand(parse_quote!(
            &'a dyn ::core::iter::Iterator<Item = ::std::vec::Vec<Rel2<T>>>
        )),
        format!(
            "& 'a dyn :: core :: iter :: Iterator < Item = :: std :: vec :: Vec < P < {} > > >",
            around
        )
    );
    // A path with the lifetime carries nothing, so it is the entry of the type alone.
    assert_eq!(
        expand(parse_quote!(::core::cell::Ref<'a, Rel2<T>>)),
        format!(":: core :: cell :: Ref < 'a , P < {} > >", alone)
    );
    // The signatures, with the generics of the trait undeclared (the default of the visitor
    // `Referrer::into_visitor()`, for a signature, declares only those of the method).
    let sig: Signature = parse_quote!(
        fn f(
            &self,
            x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
            y: Rel2<T>,
            z: &'a dyn ::core::iter::Iterator<Item = ::std::vec::Vec<Rel2<T>>>,
            w: ::core::cell::Ref<'a, Rel2<T>>,
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
        assert!(copy.contains(&format!("y : P < {} >", alone)), "{}", copy);
        assert!(
            copy.contains(&format!(
                "x : & 'a dyn :: core :: iter :: Iterator < Item = P < {} > >",
                around
            )),
            "{}",
            copy
        );
        assert!(
            copy.contains(&format!(
                "Item = :: std :: vec :: Vec < P < {} > > >",
                around
            )),
            "{}",
            copy
        );
        assert!(
            copy.contains(&format!(
                "w : :: core :: cell :: Ref < 'a , P < {} > >",
                alone
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
            fn f<'a>(
                &self,
                x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                y: Rel2<T>,
                w: ::core::cell::Ref<'a, Rel2<T>>,
            );
        }
    };
    let referrer = referrer_of(&tr);
    let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
    let alone = entry(&referrer, "Rel2 < T >", "< T >");
    let sig: Signature = parse_quote!(
        fn f<'a>(
            &self,
            x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
            y: Rel2<T>,
            w: ::core::cell::Ref<'a, Rel2<T>>,
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
        assert!(
            copy.contains(&format!(
                "x : & 'a dyn :: core :: iter :: Iterator < Item = P < {} > >",
                around
            )),
            "{}",
            copy
        );
        assert!(copy.contains(&format!("y : P < {} >", alone)), "{}", copy);
        assert!(
            copy.contains(&format!(
                "w : :: core :: cell :: Ref < 'a , P < {} > >",
                alone
            )),
            "{}",
            copy
        );
    }
}

#[test]
fn the_entries_of_the_types_around_survive_serialization() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> {
            fn f<'b, U>(
                &self,
                x: &'b dyn ::core::iter::Iterator<Item = Rel2<U>>,
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
        struct F<'b, U>(&'b dyn ::core::iter::Iterator<Item = Rel2<U>>);
    );
    let replace = |_: Type, ix: usize| -> Type {
        let ix = proc_macro2::Literal::usize_unsuffixed(ix);
        parse_quote!(P<#ix>)
    };
    let mut copy = item.clone();
    parsed.into_visitor(None, replace).visit_item_mut(&mut copy);
    assert!(!s(&copy).contains("Rel2"), "{}", s(&copy));
}

/// A path starting with a type parameter or `Self` (`T::A<X>`) is a projection, like the one with
/// a qualified self (`<T as Tr>::A<X>`): rustc does not require `X: 'a` of `&'a T::A<X>` (the
/// original field of a struct is well-formed without `U: 'a`, whereas the one which the copy has
/// for `Rel2<U>` would be stricter with it), so the type inside does not carry what the types
/// around it outlive (a reference's lifetime: a path's own lifetime is never carried). The
/// Referrer, which may not see `T` declared, finds the entry.
mod projection_paths {
    use super::*;

    /// The contexts of `Rel2<U>` in a method whose parameter is `x: construct` (`{X}` in it is
    /// `Rel2<U>`), with `T: ::p::Gat`.
    fn rel2_under(construct: &str) -> Vec<String> {
        let construct = construct.replace("{X}", "Rel2<U>");
        let tr: ItemTrait = parse_str(&format!(
            "trait Tr {{ fn f<'a, 'b, T: ::p::Gat, U>(&self, x: {}); }}",
            construct
        ))
        .unwrap();
        contexts_of(&referrer_of(&tr), "Rel2 < U >")
    }

    #[test]
    fn the_type_inside_a_projection_path_has_no_bound_from_the_types_around() {
        for construct in [
            "&'a T::A<{X}>",
            "&'a mut T::A<{X}>",
            "&'a (u8, T::A<{X}>)",
            "&'a ::std::vec::Vec<T::A<{X}>>",
            "&'a ::core::cell::Ref<'b, T::A<{X}>>",
            // More segments, and arguments elsewhere in the path.
            "&'a T::A::B<{X}>",
            "&'a T::A<u8, {X}>",
            "&'a T::A<(u8, {X})>",
            // A lifetime argument of the projection does not pass its bound on either (nor does
            // the one of any path, see `a_path_with_a_lifetime_argument_adds_no_bound()`).
            "T::A<'b, {X}>",
            "&'a T::A<'b, {X}>",
            // A path with a lifetime has none to pass on, projection or not.
            "::core::cell::Ref<'a, T::A<{X}>>",
            "::core::cell::Ref<'a, ::std::vec::Vec<T::A<{X}>>>",
        ] {
            assert_eq!(rel2_under(construct), ["< U >"], "{}", construct);
        }
        // The qualified path is the same, with `T` as a parameter or not.
        for construct in [
            "&'a <T as ::p::Gat>::A<{X}>",
            "&'a <::p::X as ::p::Gat>::A<{X}>",
            "::core::cell::Ref<'a, <T as ::p::Gat>::A<{X}>>",
        ] {
            assert_eq!(rel2_under(construct), ["< U >"], "{}", construct);
        }
        // What is around the projection is not what is inside it: the type of the argument of
        // another type, and what is in a reference inside the projection, are bounded as always.
        for construct in [
            "&'a (T::A<u8>, {X})",
            "&'a ::std::vec::Vec<{X}>",
            "&'a ::core::cell::Ref<'b, (T::A<u8>, {X})>",
        ] {
            assert_eq!(rel2_under(construct), ["< 'a , U : 'a >"], "{}", construct);
        }
        // A path with a lifetime and no reference around it bounds nothing.
        for construct in [
            "::core::cell::Ref<'a, (T::A<u8>, {X})>",
            "::core::cell::Ref<'a, ::std::vec::Vec<{X}>>",
        ] {
            assert_eq!(rel2_under(construct), ["< U >"], "{}", construct);
        }
        assert_eq!(
            rel2_under("T::A<&'a {X}>"),
            ["< 'a , U : 'a >"],
            "the reference requires it itself"
        );
        // The projection as a whole is still a component: `Ref<'a, T::A<X>>` requires
        // `T::A<X>: 'a`, which an implementation which has the type as a whole needs the bounds
        // for (`T: 'a` and `U: 'a`, E0309 otherwise), so it is not the same as no bound at all.
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, T: ::p::Gat, U>(&self, x: ::p::Wrap<'a, T::A<U>>); }
        };
        // An absolute path is not a required entry. It is an optional one (old: no entry), which
        // the Referrer may replace: it has the generics it uses, with no bound which its
        // definition may imply (`T: 'a` and `U: 'a` are not assumed: they are written when they
        // are needed).
        assert_eq!(
            contexts_of(&referrer_of(&tr), ":: p :: Wrap < 'a , T :: A < U > >"),
            ["< 'a , T : :: p :: Gat , U >"],
            "an absolute path is an optional entry"
        );
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, T: ::p::Gat + 'a, U: 'a>(&self, x: ::p::Wrap<'a, T::A<U>>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), ":: p :: Wrap < 'a , T :: A < U > >"),
            ["< 'a , T : 'a + :: p :: Gat , U : 'a >"],
            "the bounds which are written"
        );
    }

    #[test]
    fn a_path_which_does_not_start_with_a_type_parameter_is_not_a_projection() {
        // `m` is not a generic here, so it is an ordinary path (a module, say): `A<X>` is its
        // argument, which passes on what is required of it by the reference around it.
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, U>(&self, x: &'a ::p::m::A<Rel2<U>>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < U >"),
            ["< 'a , U : 'a >"]
        );
        // And nothing is passed on by a path with a lifetime and no reference around it.
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, U>(&self, x: ::core::cell::Ref<'a, ::p::m::A<Rel2<U>>>); }
        };
        assert_eq!(contexts_of(&referrer_of(&tr), "Rel2 < U >"), ["< U >"]);
        // A const parameter is not a type parameter either.
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, const T: usize, U>(&self, x: &'a T::A<Rel2<U>>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < U >"),
            ["< 'a , U : 'a >"]
        );
        // A type parameter of an inner scope which shadows the name does not change it.
        let tr: ItemTrait = parse_quote! {
            trait Tr<T> {
                fn f<'a, U>(&self, x: &'a T::A<Rel2<U>>);
            }
        };
        assert_eq!(contexts_of(&referrer_of(&tr), "Rel2 < U >"), ["< U >"]);
    }

    #[test]
    fn self_paths_are_projections() {
        // `Self::A<X>` is an associated type of the trait, whatever `Self` is.
        let tr: ItemTrait = parse_quote! {
            trait Tr {
                type B<X>;
                fn f<'a, U>(&self, x: &'a Self::B<Rel2<U>>);
                fn g<'a, U>(&self, x: &'a Self::B<'a, Rel2<U>>);
                fn h<'a, U>(&self, x: ::core::cell::Ref<'a, Self::B<Rel2<U>>>);
            }
        };
        let referrer = referrer_of(&tr);
        assert_eq!(contexts_of(&referrer, "Rel2 < U >"), ["< U >"]);
        // `Self` is not a generic: `Self::A` mentions `Self`, but the type inside does not.
        let rel2 = (0..referrer.iter().count())
            .find(|&ix| s(referrer.iter().nth(ix).unwrap()) == "Rel2 < U >")
            .unwrap();
        assert!(referrer.context_self(referrer.context(rel2)).is_none());
        assert!(referrer
            .context_self_outlives(referrer.context(rel2))
            .is_empty());
    }

    /// The trait of the Leaker: `x` has the type inside a projection path, and `y` the same type
    /// under the lifetime of a reference without it.
    fn both() -> (ItemTrait, Referrer) {
        let tr: ItemTrait = parse_quote! {
            trait Tr {
                fn f<'a, T: ::p::Gat, U>(
                    &self,
                    x: &'a T::A<Rel2<U>>,
                    y: &'a ::std::vec::Vec<Rel2<U>>,
                );
            }
        };
        let referrer = referrer_of(&tr);
        (tr, referrer)
    }

    #[test]
    fn the_referrer_finds_the_entry_in_the_exact_and_in_the_lenient_matching() {
        let (tr, referrer) = both();
        let projected = entry(&referrer, "Rel2 < U >", "< U >");
        let around = entry(&referrer, "Rel2 < U >", "< 'a , U : 'a >");
        assert_ne!(projected, around);
        let cases: Vec<(Item, usize)> = vec![
            // A struct field, which is what the copy needs.
            (
                parse_quote!(
                    struct F<'a, T: ::p::Gat, U>(::core::cell::Ref<'a, T::A<Rel2<U>>>);
                ),
                projected,
            ),
            (
                parse_quote!(
                    struct G<'a, T: ::p::Gat, U>(&'a T::A<Rel2<U>>);
                ),
                projected,
            ),
            (
                parse_quote!(
                    struct H<'a, T: ::p::Gat, U>(
                        ::core::cell::Ref<'a, ::std::vec::Vec<T::A<Rel2<U>>>>,
                    );
                ),
                projected,
            ),
            // The same type under the lifetime of a reference without a projection is the other
            // entry, and under the lifetime of a path it is the one with nothing around it.
            (
                parse_quote!(
                    struct V<'a, U>(&'a ::std::vec::Vec<Rel2<U>>);
                ),
                around,
            ),
            (
                parse_quote!(
                    struct W<'a, U>(::core::cell::Ref<'a, ::std::vec::Vec<Rel2<U>>>);
                ),
                projected,
            ),
            // A qualified projection (which is one whatever its self type is).
            (
                parse_quote!(
                    struct Q<'a, U>(::core::cell::Ref<'a, <U as ::p::Gat>::A<Rel2<U>>>);
                ),
                projected,
            ),
        ];
        for (item, ix) in cases {
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
                let other = if ix == around { projected } else { around };
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
    fn the_referrer_finds_the_entry_with_the_generics_of_the_method_declared() {
        // The generics of the method are declared where its signature is copied, those of the
        // trait are not (`Lenient`), or are the root (`Exact`).
        let tr: ItemTrait = parse_quote! {
            trait Tr<T: ::p::Gat> {
                fn f<'a, U>(&self, x: ::core::cell::Ref<'a, T::A<Rel2<U>>>);
            }
        };
        let referrer = referrer_of(&tr);
        assert_eq!(contexts_of(&referrer, "Rel2 < U >"), ["< U >"]);
        let sig: Signature = parse_quote!(
            fn f<'a, U>(&self, x: ::core::cell::Ref<'a, T::A<Rel2<U>>>)
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
            referrer
                .clone()
                .into_visitor_with_root(&tr.generics, None, replace),
        ] {
            let mut copy = sig.clone();
            visitor.visit_signature_mut(&mut copy);
            assert!(!s(&copy).contains("Rel2"), "{}", s(&copy));
        }
    }

    #[test]
    fn the_referrer_finds_the_entry_with_nothing_declared() {
        // `T` is not declared where the type is looked up, and the entry (`Rel2<U>`) does not
        // mention it, so the Referrer cannot know it is a type parameter of the Leaker: it assumes
        // it where the type is not found otherwise.
        let tr: ItemTrait = parse_quote! {
            trait Tr {
                fn f<'a, T: ::p::Gat, U>(&self, x: ::core::cell::Ref<'a, T::A<Rel2<U>>>);
            }
        };
        let referrer = referrer_of(&tr);
        let projected = entry(&referrer, "Rel2 < U >", "< U >");
        let expand = |ty: Type| -> String {
            s(&referrer.expand(ty, |_, ix| {
                let ix = proc_macro2::Literal::usize_unsuffixed(ix);
                parse_quote!(P<#ix>)
            }))
        };
        for ty in [
            parse_quote!(::core::cell::Ref<'a, T::A<Rel2<U>>>),
            parse_quote!(&'a T::A<Rel2<U>>),
            parse_quote!(&'a ::std::vec::Vec<T::A<Rel2<U>>>),
            parse_quote!(&'a T::A::B<(u8, Rel2<U>)>),
            parse_quote!(::core::cell::Ref<'a, ::std::vec::Vec<T::A<Rel2<U>>>>),
        ] {
            let ty: Type = ty;
            assert!(!expand(ty.clone()).contains("Rel2"), "{}", s(&ty));
            assert!(
                expand(ty.clone()).contains(&format!("P < {} >", projected)),
                "{}",
                s(&ty)
            );
        }
        // Another name is no better than `T`: the type could be one of the Leaker's.
        assert!(!expand(parse_quote!(::core::cell::Ref<'a, X::A<Rel2<U>>>)).contains("Rel2"));
        // Without any projection, the type under the lifetime of a reference is no entry of this
        // Referrer, whereas under a path it is the one with nothing around it.
        assert!(expand(parse_quote!(&'a ::std::vec::Vec<Rel2<U>>)).contains("Rel2"));
        assert!(!expand(parse_quote!(::core::cell::Ref<'a, Rel2<U>>)).contains("Rel2"));
    }

    #[test]
    fn an_entry_with_the_bound_wins_where_the_name_may_be_anything() {
        // Both types are in the Leaker: with nothing declared, `X::A<Rel2<U>>` may be a path
        // which is no projection, so the entry which is stricter is the one found (the copy of
        // the other would be rejected by rustc if `X` was no type parameter), unless `X` is known
        // to be a type parameter: declared, or a generic of the outermost item.
        let (tr, referrer) = both();
        let projected = entry(&referrer, "Rel2 < U >", "< U >");
        let around = entry(&referrer, "Rel2 < U >", "< 'a , U : 'a >");
        let ty: Type = parse_quote!(&'a T::A<Rel2<U>>);
        let replace = |_: Type, ix: usize| -> Type {
            let ix = proc_macro2::Literal::usize_unsuffixed(ix);
            parse_quote!(P<#ix>)
        };
        assert!(s(&referrer.expand(ty, replace)).contains(&format!("P < {} >", around)));
        // With a path's lifetime there is no bound to be stricter by: the entry of the projection.
        let ty: Type = parse_quote!(::core::cell::Ref<'a, T::A<Rel2<U>>>);
        assert!(s(&referrer.expand(ty, replace)).contains(&format!("P < {} >", projected)));
        // The generics of the item are known to the visitor, declared or not.
        let item: Item = parse_quote!(
            struct F<'a, T: ::p::Gat, U>(::core::cell::Ref<'a, T::A<Rel2<U>>>);
        );
        for mode in MODES {
            let copy = rewrite(&referrer, &tr.generics, mode, &item);
            assert!(
                copy.contains(&format!("P < {} >", projected)),
                "{:?}: {}",
                mode,
                copy
            );
        }
    }

    #[test]
    fn the_entries_survive_serialization() {
        let (_, referrer) = both();
        let tokens = referrer.to_token_stream();
        let parsed: Referrer = parse2(tokens).unwrap();
        assert_eq!(parsed, referrer);
        let item: Item = parse_quote!(
            struct F<'a, T: ::p::Gat, U>(::core::cell::Ref<'a, T::A<Rel2<U>>>);
        );
        let replace = |_: Type, ix: usize| -> Type {
            let ix = proc_macro2::Literal::usize_unsuffixed(ix);
            parse_quote!(P<#ix>)
        };
        let mut copy = item.clone();
        parsed.into_visitor(None, replace).visit_item_mut(&mut copy);
        assert!(!s(&copy).contains("Rel2"), "{}", s(&copy));
    }
}

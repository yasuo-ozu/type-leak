//! Tests for building a `Leaker` and interning items.
//!
//! The methods of the `Leaker` which analyze a whole item (`intern_struct()`, `intern_enum()`,
//! `intern_union()`, `intern_impl()`, `intern_trait()` and `intern_mod()`) come first. Their
//! expected values are those of the constructors of the Leaker which analyzed an item and which
//! they replace (`Leaker::from_*()`), taken from the previous version of the crate: the item is
//! analyzed with its own generics as those of the Leaker, which is what the top-level context is
//! that of. Then come the configuration, the other ways to build a Leaker, and the regressions.

mod common;

use common::{messages, s, starts_with, to_syn_error};
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::{
    Allow, Cause, Context, ContextId, EntryId, Leaker, Place, Referrer, Rules, SelfRequirement,
    View,
};

/// The entries as `type | generics | self`, in the order of the Referrer.
fn entries(referrer: &Referrer) -> Vec<String> {
    (0..referrer.iter().count())
        .map(|ix| {
            let ty = common::types(referrer).nth(ix).unwrap();
            let context = common::entry(referrer, ix);
            let generics = &context.generics;
            let self_ty = match context.self_ty.as_ref() {
                None => "-".to_string(),
                Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
                Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
                Some(other) => format!("{:?}", other),
            };
            format!("{} | {} | {}", s(ty), s(generics), self_ty)
        })
        .collect()
}

fn ids(referrer: &Referrer) -> Vec<usize> {
    (0..referrer.iter().count())
        .map(|ix| common::entry_id(referrer, ix).index())
        .collect()
}

fn id(n: usize) -> ContextId {
    parse_str(&n.to_string()).unwrap()
}

/// Whether the item is analyzed without an error.
fn is_accepted(item: Item) -> bool {
    let mut leaker = Leaker::new();
    match &item {
        Item::Struct(item) => leaker.intern_struct(item),
        Item::Trait(item) => leaker.intern_trait(item),
        Item::Union(item) => leaker.intern_union(item),
        other => panic!("{}", s(other)),
    };
    leaker.result().finish().is_ok()
}

/// A bound and a supertrait must use an absolute path, in a struct, a trait and a union.
#[test]
fn test_leaker_construction() {
    let cases: [(Item, bool); 7] = [
        (
            parse_quote! { struct MyStruct<T: path::to::MyTrait>(); },
            false,
        ),
        (
            parse_quote! { struct MyStruct<T: ::path::to::MyTrait>(); },
            true,
        ),
        (parse_quote! { trait MyTrait: MyTrait2 {} }, false),
        (
            parse_quote! { trait MyTrait: ::path::to::MyTrait2 {} },
            true,
        ),
        (
            parse_quote! { trait MyTrait: ::path::to::MyTrait2<{<Self as ::path::to::MyTrait3>::N}> {} },
            true,
        ),
        (
            parse_quote! { trait MyTrait: ::path::to::MyTrait2<{<Self as MyTrait3>::N}> {} },
            false,
        ),
        (parse_quote! { union MyUnion<T: Copy> { a: T } }, false),
    ];
    for (item, accepted) in cases {
        assert_eq!(is_accepted(item.clone()), accepted, "{}", s(&item));
    }
}

/// `Leaker::default()` is `Leaker::new()`: it allows nothing and does not read values, so it gives
/// the same entries and errors as one with the rule `()`.
#[test]
fn leaker_default_is_leaker_new() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T> {
            const N: usize = rel();
            fn f(&self, a: Rel<T>, b: Option<u8>) -> Vec<Rel2>;
        }
    );
    let analyze = |mut leaker: Leaker| {
        assert!(!leaker.check_values);
        leaker.intern_trait(&item);
        leaker.result()
    };
    let (default, new) = (analyze(Leaker::default()), analyze(Leaker::new()));
    assert!(default.errors().is_empty());
    let (default, new) = (default.into_referrer(), new.into_referrer());
    assert_eq!(default, new);
    let mut entries = entries(&default);
    entries.sort();
    assert_eq!(
        entries,
        [
            "Option < u8 > |  | -",
            "Rel < T > | < T > | -",
            "Vec < Rel2 > |  | -",
            "usize |  | -"
        ]
    );
    // The same with the rule `()` named: the type of the default is the one of `Leaker::new()`.
    let mut named: Leaker<()> = Leaker::with_rule(());
    named.intern_trait(&item);
    assert_eq!(named.result().into_referrer(), new);
}

#[test]
fn intern_struct() {
    let item: ItemStruct = parse_quote!(
        pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
            field1: MyType1,
            field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
            field3: &'a (T1, T2),
            field4: MyType6<'a, T1, T2>,
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    // `MyType6` uses all the generics: it is the entry of the top-level context.
    assert_eq!(
        entries(&referrer),
        [
            "(MyType2 , MyType3 < MyType1 > , MyType4 , MyType5) |  | -",
            "MyType1 |  | -",
            "MyType4 |  | -",
            "MyType6 < 'a , T1 , T2 > | < 'a , T1 , T2 : :: path :: to :: MyType1 < MyType4 > > | -",
        ]
    );
    assert_eq!(ids(&referrer), [0, 0, 0, 1]);
    assert_eq!(top_level.index(), 1);
    assert_eq!(common::context_ids(&referrer).len(), 2);
    assert!(!referrer.has_self());
    // `Self` alone is never leaked.
    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote!(
        pub struct MyStruct<T1> {
            a: Rel<T1>,
            b: Self,
        }
    ));
    assert!(leaker.result().finish().is_ok());
    // `reduce_roots()` changes nothing here.
    let mut leaker = Leaker::new();
    let context = leaker.intern_struct(&item);
    leaker.reduce_roots();
    let reduced = leaker.result().finish().unwrap();
    assert_eq!(entries(&reduced), entries(&referrer));
    // `expand()` replaces a leaked type by what the callback returns for its index, and nothing
    // else (the entries are sorted by tokens: the tuple, `MyType1`, `MyType4`).
    let mut f = |_, ix: EntryId, _: &Context| -> Type {
        let ix = syn::Index::from(ix.index());
        parse_quote!(::path::to::Implementor<#ix>)
    };
    assert_eq!(
        quote!(#{common::expand(&reduced, context, parse_quote!(MyType1),  &mut f)}).to_string(),
        quote!(::path::to::Implementor<1>).to_string()
    );
    assert_eq!(
        quote!(#{common::expand(&reduced, context, parse_quote!((MyType2 , MyType3 < MyType1 > , MyType4 , MyType5)),  &mut f)}).to_string(),
        quote!(::path::to::Implementor<0>).to_string()
    );
    assert_eq!(
        quote!(#{common::expand(&reduced, context, parse_quote!(MyType4),  &mut f)}).to_string(),
        quote!(::path::to::Implementor<2>).to_string()
    );
    // A type that is not leaked is left as it is.
    assert_eq!(
        quote!(#{common::expand(&reduced, context, parse_quote!(Other),  &mut f)}).to_string(),
        quote!(Other).to_string()
    );
}

#[test]
fn intern_enum() {
    let item: ItemEnum = parse_quote!(
        pub enum MyEnum<T, U: ::core::marker::Copy> {
            A(Rel, T),
            B { x: Wrap<U>, y: Self2 },
            C(Rel2<Self>),
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_enum(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "Rel |  | -",
            "Self2 |  | -",
            "Wrap < U > | < U : :: core :: marker :: Copy > | -",
            "Rel2 < Self > | < T , U : :: core :: marker :: Copy > | Self = MyEnum < T , U >",
        ]
    );
    assert_eq!(ids(&referrer), [0, 0, 1, 2]);
    // The top-level context has the generics of the enum and, as `Rel2<Self>` names `Self`, what
    // `Self` is: it is the context of that entry.
    assert_eq!(top_level.index(), 2);
    assert_eq!(top_level, common::entry_id(&referrer, 3));
    assert!(referrer.has_self());
}

#[test]
fn intern_union() {
    let item: ItemUnion = parse_quote!(
        pub union MyUnion<T: ::core::marker::Copy> {
            a: T,
            b: MyType,
            c: Wrap<T>,
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_union(&item);
    let referrer = leaker.result().finish().unwrap();
    // `Wrap<T>` uses the generics of the union: it is the entry of the top-level context.
    assert_eq!(
        entries(&referrer),
        [
            "MyType |  | -",
            "Wrap < T > | < T : :: core :: marker :: Copy > | -"
        ]
    );
    assert_eq!(ids(&referrer), [0, 1]);
    assert_eq!(top_level.index(), 1);
    // A union with a lifetime, a reference, a tuple and the type of an external crate.
    let item: ItemUnion = parse_quote! {
        pub union MyUnion<'a, T: ::core::marker::Copy> {
            a: ::core::mem::ManuallyDrop<MyType1>,
            b: &'a T,
            c: (MyType2, u8),
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_union(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        common::types(&referrer)
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
}

#[test]
fn intern_trait() {
    let item: ItemTrait = parse_quote!(
        pub trait MyTrait<T, U = Rel<T>> {
            type Out: ::core::fmt::Debug;
            fn f(&self, t: T) -> Rel2<Self>;
            fn g<V: ::core::marker::Copy>(&self, v: V) -> Wrap<V, U>;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "Rel2 < Self > | < T , U > | Self: MyTrait < T , U >",
            "Wrap < V , U > | < U , V : :: core :: marker :: Copy > | -",
            "Rel < T > | < T > | -",
        ]
    );
    assert_eq!(ids(&referrer), [0, 1, 2]);
    // The top-level context has the generics of the trait and, as `Rel2<Self>` names `Self`, what
    // `Self` is: it is the context of that entry.
    assert_eq!(top_level.index(), 0);
    assert_eq!(top_level, common::entry_id(&referrer, 0));
}

#[test]
fn intern_impl() {
    let item: ItemImpl = parse_quote!(
        impl<T> ::path::MyTrait<Rel<T>> for MyType<T> {
            type Out = Rel2<T>;
            fn f(&self, x: Rel3) -> Self::Out {
                todo!()
            }
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_impl(&item);
    let referrer = leaker.result().finish().unwrap();
    // The self type, the trait with its arguments and the signatures are analyzed, and the bodies
    // are not.
    assert_eq!(
        entries(&referrer),
        [
            "MyType < T > | < T > | -",
            "Rel < T > | < T > | -",
            "Rel2 < T > | < T > | -",
            "Rel3 |  | -",
        ]
    );
    assert_eq!(ids(&referrer), [0, 0, 0, 1]);
    // The generics of the impl are those of the Leaker, which entries use exactly.
    assert_eq!(top_level.index(), 0);
    assert_eq!(top_level, common::entry_id(&referrer, 0));

    // `Self` is the self type, which is leaked as a whole with it.
    let item: ItemImpl = parse_quote!(
        impl<T> ::path::MyTrait for ::path::W<T> {
            fn f(&self) -> Rel<Self> {
                todo!()
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "Rel < Self > | < T > | Self = :: path :: W < T >",
            // An optional entry: the type of an external crate.
            ":: path :: W < T > | < T > | -",
        ]
    );
    // The trait must not depend on the type context, as a bound.
    let item: ItemImpl = parse_quote!(
        impl Relative for ::path::W {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1);
    assert_eq!(err[0].message, "use absolute path: 'Relative'");
}

#[test]
fn intern_mod() {
    let item: ItemMod = parse_quote!(
        mod m {
            pub struct S<T> {
                x: Rel<T>,
            }
            pub type A = Rel2;
            pub mod inner {
                pub type B = Rel3;
                use super::super::Rel4;
                pub struct C(Rel4);
            }
        }
    );
    let mut leaker = Leaker::new();
    let module = leaker.intern_mod(&item);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "Rel < T > | < T > | -",
            "Rel2 |  | -",
            "Rel3 |  | -",
            "Rel4 |  | -",
        ]
    );
    // The module of each: the analyzed one for the first two (its path is `[]`), and `inner`.
    let paths: Vec<Vec<String>> = (0..4)
        .map(|ix| {
            common::entry_module_path(&result, ix)
                .unwrap()
                .iter()
                .map(|i| i.to_string())
                .collect()
        })
        .collect();
    assert_eq!(paths, [vec![], vec![], vec!["inner"], vec!["inner"]]);
    // The top-level context is the module: the entry of the module is in it.
    assert_eq!(common::in_module(common::entry(&referrer, 0)), Some(module));
    // A module without a content has nothing to analyze.
    let mut leaker = Leaker::new();
    leaker.intern_mod(&parse_quote!(
        mod m;
    ));
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.is_empty());
}

#[test]
fn with_rule_and_the_items_accumulate() {
    // A Leaker with a configuration analyzes the items like one without it, and the contexts of
    // the items are numbered in the order they appear.
    let config = Rules::new().primitive().prelude();
    let mut leaker = Leaker::with_rule(config);
    let first: ItemStruct = parse_quote!(
        struct A<T> {
            x: Vec<(Rel, u8)>,
            y: T,
        }
    );
    let second: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self) -> Option<Wrap<T>>;
        }
    );
    let third: ItemEnum = parse_quote!(
        enum E {
            A(Rel),
            B(Rel2),
        }
    );
    let a = leaker.intern_struct(&first);
    let b = leaker.intern_trait(&second);
    let c = leaker.intern_enum(&third);
    // The next generics take the next id.
    let d = leaker.intern_struct(&parse_quote!(
        struct D<V> {
            x: Wrap<V>,
        }
    ));
    assert_eq!(d, id(2));
    let referrer = leaker.result().finish().unwrap();
    // `Rel` is leaked once, in the context of no generics, which is that of `E`.
    assert_eq!(
        entries(&referrer),
        [
            "Rel |  | -",
            "Rel2 |  | -",
            "Wrap < T > | < T > | -",
            "Wrap < V > | < V > | -"
        ]
    );
    assert_eq!(a, id(1));
    // The same generics are the same context, whichever item they are the generics of.
    assert_eq!(b, a);
    assert_eq!(c, id(0));
    // `Wrap<T>` is the entry of the generics `<T>` of both, and `Rel` of those of `E`.
    assert_eq!(common::entry_id(&referrer, 2), a);
    assert_eq!(s(&common::entry(&referrer, 2).generics), "< T >");
    assert_eq!(common::entry_id(&referrer, 0), c);
    assert!(common::entry(&referrer, 0).generics.params.is_empty());
    // `with_rule()` is what `new()` is with the allowed paths.
    let config = Rules::new().or_if(starts_with(parse_quote!(Vec)));
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_struct(&first);
    // `Vec` is allowed, and `u8` is not (it is a relative path, as it is not a configured one).
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        ["(Rel , u8) |  | -"]
    );
}

/// `Rules::prelude()` allows the function `drop` in an array length, `Rules::primitive()` does
/// not, and the lists of `Rules` add up with `crate_paths()`. The names of each list are checked
/// one by one in the module `names` of `tests/allow_rules.rs`.
#[test]
fn test_config_allows_drop_in_an_array_length_and_the_lists_add_up() {
    let only_primitives = || Rules::new().primitive();
    let only_prelude = || Rules::new().prelude();
    let item: ItemStruct = parse_quote!(
        struct S([::core::primitive::u8; drop]);
    );
    fn entry_count<A: Allow>(config: A, item: &ItemStruct) -> usize {
        let mut leaker = Leaker::with_rule(config);
        leaker.intern_struct(item);
        leaker.result().finish().unwrap().iter().count()
    }
    assert_eq!(entry_count(Rules::new(), &item), 1);
    assert_eq!(entry_count(only_primitives(), &item), 1);
    assert_eq!(entry_count(only_prelude(), &item), 0);
    // `Rules::crate_paths()` allows the paths that start with `crate`, and the lists add up.
    let item: ItemStruct = parse_quote!(
        struct S(crate::A, u8, Vec<crate::B>);
    );
    let all = Rules::new().primitive().prelude().crate_paths();
    let mut leaker = Leaker::with_rule(all);
    leaker.intern_struct(&item);
    assert!(leaker.result().finish().unwrap().is_empty());
    let mut leaker = Leaker::with_rule(only_prelude());
    leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        common::types(&referrer)
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
    let config = Rules::new().crate_paths();
    let mut leaker = Leaker::with_rule(config);

    leaker.intern_trait(&item);
    leaker.reduce_roots();
    assert!(leaker.result().finish().unwrap().is_empty());
}

// ── regressions ──────────────────────────────────────────────────────────────

fn leak_struct(item: &ItemStruct) -> Result<BTreeSet<String>> {
    let config = type_leak::Rules::new().primitive().prelude();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_struct(item);
    let referrer = leaker
        .result()
        .finish()
        .map_err(|e| Error::new(e[0].span(), messages(&e)))?;
    Ok(common::types(&referrer)
        .map(|t| quote!(#t).to_string())
        .collect())
}

fn leak_trait(item: &ItemTrait) -> Result<BTreeSet<String>> {
    let config = type_leak::Rules::new().primitive().prelude();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(item);
    let referrer = leaker
        .result()
        .finish()
        .map_err(|e| Error::new(e[0].span(), messages(&e)))?;
    Ok(common::types(&referrer)
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
    // `result()` without `reduce_roots()`.
    let item: ItemStruct = parse_quote!(
        struct S {
            a: A,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);

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
    leaker.intern_struct(&s);
    leaker.reduce_roots();
    leaker.intern_struct(&u);
    leaker.reduce_roots();
    let leaked: BTreeSet<String> = leaker
        .result()
        .finish()
        .unwrap()
        .iter()
        .map(|(_, t, _)| quote!(#t).to_string())
        .collect();
    assert_eq!(leaked, strings(&["A", "B"]));
}

// ── the entries that reduce_roots() keeps ────────────────────────────────────
//
// `reduce_roots()` prunes the Leaker to the entries it chose, and what is analyzed afterwards is
// added to them: a kept entry stays one, of the kind it was kept as. The tests of `tests/
// reduce_roots.rs` show the pruning for the usual items; these show it against a later scope that
// changes the verdict on a kept entry.

/// The types of the entries, in the order of the Referrer.
fn listed(referrer: &Referrer) -> Vec<String> {
    common::types(referrer).map(s).collect()
}

/// The entries after two scopes in `generics`, with `reduce_roots()` between them if `reduce`.
fn two_scopes(
    generics: &Generics,
    first: impl Fn(&mut View<'_, '_, ()>),
    second: impl Fn(&mut View<'_, '_, ()>),
    reduce: bool,
) -> Vec<String> {
    let mut leaker = Leaker::new();
    leaker.enter_generics(generics, |view| first(view));
    if reduce {
        leaker.reduce_roots();
    }
    leaker.enter_generics(generics, |view| second(view));
    listed(&leaker.result().finish().unwrap())
}

/// A scope that adds no entry leaves the entries as `reduce_roots()` left them, and so does one
/// that names a kept entry again (the pair): a reduction after either changes nothing.
#[test]
fn a_scope_that_adds_no_entry_after_reduce_roots_changes_nothing() {
    let first: ItemStruct = parse_quote! { struct S1 { a: (R1, R2), b: ::ext::X } };
    let plain: ItemStruct = parse_quote! { struct S2 { a: u8 } };
    let again: ItemStruct = parse_quote! { struct S3 { a: (R1, R2) } };
    let run = |later: &[&ItemStruct], reduce_after: bool| -> Referrer {
        let mut leaker = Leaker::with_rule(Rules::new().primitive());
        leaker.intern_struct(&first);
        leaker.reduce_roots();
        for item in later {
            leaker.intern_struct(item);
        }
        if reduce_after {
            leaker.reduce_roots();
        }
        leaker.result().finish().unwrap()
    };
    let kept = run(&[], false);
    assert_eq!(listed(&kept), ["(R1 , R2)", ":: ext :: X"]);
    for (later, reduce_after) in [
        (vec![&plain], false),
        (vec![&plain], true),
        (vec![&again], false),
        (vec![&again], true),
        (vec![&plain, &again, &plain], true),
    ] {
        let case = format!(
            "{} later items, reduced after: {}",
            later.len(),
            reduce_after
        );
        assert_eq!(run(&later, reduce_after), kept, "{}", case);
    }
}

// The generics of the next tests: the bound of `T` names `::ext::X`, a type of another crate, so
// in a predicate a type that uses `T` cannot be an optional entry: the Referrer would write `X` of
// the bound as a path through the Repeater, where the impl of the entry needs it as written
// (`tests/bound_cycles.rs`). In a field it can. The pair is one entry in a field (cheaper than
// `A<U>` and `B<T>`), and the same type in a predicate, in the same context, cannot be one.

fn pair_generics() -> Generics {
    parse_quote! { <T: ::core::ops::Fn(::ext::X), U> }
}

fn pair() -> Type {
    parse_quote! { (::ext::A<U>, ::ext::B<T>) }
}

fn pair_in_a_predicate() -> WhereClause {
    parse_quote! { where (::ext::A<U>, ::ext::B<T>): ::core::marker::Sized }
}

/// A later scope may forbid an optional entry that an earlier one kept: it stays an entry, as the
/// copy of the earlier scope is built from it. The later scope may get a copy that does not
/// compile; that is the price of pruning.
#[test]
fn a_kept_optional_entry_stays_when_a_later_scope_forbids_it() {
    let (generics, pair, predicate) = (pair_generics(), pair(), pair_in_a_predicate());
    let in_field = |view: &mut View<'_, '_, ()>| view.visit_type(&pair);
    let in_predicate = |view: &mut View<'_, '_, ()>| view.visit_where_clause(&predicate);
    let nothing = |_: &mut View<'_, '_, ()>| {};
    // Each scope alone: the pair in a field is one entry; in a predicate its parts are the entries,
    // and `B<T>`, which the bound of `T` forbids, is written as it is.
    assert_eq!(
        two_scopes(&generics, nothing, in_field, false),
        ["(:: ext :: A < U > , :: ext :: B < T >)"]
    );
    assert_eq!(
        two_scopes(&generics, nothing, in_predicate, false),
        [":: ext :: A < U >"]
    );
    // One reduction at the end has the parts of the pair for both scopes.
    assert_eq!(
        two_scopes(&generics, in_field, in_predicate, false),
        [":: ext :: A < U >", ":: ext :: B < T >"]
    );
    // After `reduce_roots()` the pair is kept as the entry that it was.
    assert_eq!(
        two_scopes(&generics, in_field, in_predicate, true),
        ["(:: ext :: A < U > , :: ext :: B < T >)"]
    );
}

/// A later type that has the kept pair as a component does not replace it, whether or not the later
/// scope forbids the pair: the parent is built from it.
#[test]
fn a_kept_entry_stays_when_a_later_type_holds_it() {
    let (generics, pair, predicate) = (pair_generics(), pair(), pair_in_a_predicate());
    let parent: Type = parse_quote! { ((::ext::A<U>, ::ext::B<T>), ::ext::C<U>) };
    let in_field = |view: &mut View<'_, '_, ()>| view.visit_type(&pair);
    let parent_alone = |view: &mut View<'_, '_, ()>| view.visit_type(&parent);
    let with_predicate = |view: &mut View<'_, '_, ()>| {
        view.visit_where_clause(&predicate);
        view.visit_type(&parent);
    };
    let nothing = |_: &mut View<'_, '_, ()>| {};
    // Alone, the parent is the one entry. After `reduce_roots()` the pair is kept, and the parent,
    // which the pair and `C<U>` build, is not one: two entries.
    assert_eq!(
        two_scopes(&generics, nothing, parent_alone, false),
        ["((:: ext :: A < U > , :: ext :: B < T >) , :: ext :: C < U >)"]
    );
    assert_eq!(
        two_scopes(&generics, in_field, parent_alone, true),
        [
            "(:: ext :: A < U > , :: ext :: B < T >)",
            ":: ext :: C < U >"
        ]
    );
    // The same where the later scope forbids the pair (in a predicate, as well as a part of the
    // parent): alone, its parts are the entries.
    assert_eq!(
        two_scopes(&generics, nothing, with_predicate, false),
        [":: ext :: A < U >", ":: ext :: C < U >"]
    );
    assert_eq!(
        two_scopes(&generics, in_field, with_predicate, true),
        [
            "(:: ext :: A < U > , :: ext :: B < T >)",
            ":: ext :: C < U >"
        ]
    );
}

/// A kept optional entry that a later scope shows to hold a required type is a required entry.
///
/// The rule allows `Rel` in a default only, so `::ext::X<(Rel,)>` in a default holds no required
/// type and is an optional entry, and in a field it holds `Rel`, which is required. The required
/// entries come before the optional ones in the Referrer.
#[test]
fn a_kept_optional_entry_becomes_required_when_it_holds_a_required_type() {
    let rule =
        |cause: &Cause, place: Place| place.in_default && *cause == Cause::Type(parse_quote!(Rel));
    let in_default: ItemStruct = parse_quote! { struct S1<T = ::ext::X<(Rel,)>>(T); };
    let in_field: ItemStruct = parse_quote! { struct S2 { a: ::ext::X<(Rel,)> } };
    let other: ItemStruct = parse_quote! { struct S3 { b: Rel2 } };
    let run = |items: &[&ItemStruct], reduce: bool| -> Vec<String> {
        let mut leaker = Leaker::with_rule(rule);
        for (ix, item) in items.iter().enumerate() {
            leaker.intern_struct(item);
            if reduce && ix == 0 {
                leaker.reduce_roots();
            }
        }
        listed(&leaker.result().finish().unwrap())
    };
    // In a default, `X<(Rel,)>` is an optional entry: it comes after `Rel2`, kept or not.
    for reduce in [false, true] {
        assert_eq!(
            run(&[&in_default, &other], reduce),
            ["Rel2", ":: ext :: X < (Rel ,) >"]
        );
    }
    // In a field it is a required entry, and comes first (the entries are sorted by their tokens).
    assert_eq!(
        run(&[&in_field, &other], false),
        [":: ext :: X < (Rel ,) >", "Rel2"]
    );
    // Kept as an optional entry, and then shown to hold `Rel`: it is a required one, as it is for
    // one reduction at the end.
    for reduce in [false, true] {
        assert_eq!(
            run(&[&in_default, &in_field, &other], reduce),
            [":: ext :: X < (Rel ,) >", "Rel2"],
            "reduced in between: {}",
            reduce
        );
    }
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
            leaker.intern_struct(&item);
            leaker.result().finish().unwrap()
        }
        .iter()
        .map(|(_, t, _)| quote!(#t).to_string())
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
    // none (the context 0), although its tokens are between theirs. The types of an external crate
    // come after the others, sorted the same way by the contexts of the result: `::ext::W<T>` was
    // the first of them, but its context is the second one of the Referrer, as `Y<T>` made it.
    // Audit mutant `sort_optional_off`.
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
    leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    let listed: Vec<(String, usize)> = (0..referrer.iter().count())
        .map(|ix| {
            (
                quote!(#{common::types(&referrer).nth(ix).unwrap()}).to_string(),
                common::entry_id(&referrer, ix).index(),
            )
        })
        .collect();
    assert_eq!(
        listed,
        [
            ("A".to_string(), 0),
            ("Z".to_string(), 0),
            ("X < T >".to_string(), 1),
            ("Y < T >".to_string(), 1),
            (":: ext :: N".to_string(), 0),
            (":: ext :: O".to_string(), 0),
            (":: ext :: V".to_string(), 0),
            (":: ext :: W < T >".to_string(), 1),
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
        "use absolute path: 'MyType<'_>'"
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
fn test_assertions_for_assumed_primitives() {
    // Primitive names in const parameter types, and in types where they are allowed, are
    // checked once each; absolute paths and leaked (not allowed) primitives are not.
    let item: ItemTrait = parse_quote! {
        trait Tr<const N: u8> {
            fn f<const M: u8>(&self, x: u16) -> ::core::primitive::u32;
        }
    };
    let config = type_leak::Rules::new().primitive();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(&item);
    let assertions = leaker.assertions().to_string();
    assert!(leaker.result().errors().is_empty());
    assert!(assertions.contains("__type_leak_assert_u8_is_primitive"));
    assert!(assertions.contains("__type_leak_assert_u16_is_primitive"));
    assert!(!assertions.contains("__type_leak_assert_u32_is_primitive"));
    assert_eq!(
        assertions
            .matches("__type_leak_assert_u8_is_primitive")
            .count(),
        1
    );

    // Without `Rules::primitive()`, `u16` in a type is leaked, so it is not assumed.
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let assertions = leaker.assertions().to_string();
    assert!(leaker.result().errors().is_empty());
    assert!(assertions.contains("__type_leak_assert_u8_is_primitive"));
    assert!(!assertions.contains("__type_leak_assert_u16_is_primitive"));

    // No primitives, no items.
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            t: T,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    assert!(leaker.assertions().is_empty());
    assert!(leaker.result().errors().is_empty());
}

#[test]
fn test_receivers() {
    // The type of a receiver (`&self` is `self: &Self`) is analyzed as any other type, where
    // `Self` alone is never leaked.
    let leak = |sig: proc_macro2::TokenStream| -> std::result::Result<BTreeSet<String>, String> {
        let mut leaker = Leaker::new();
        common::intern_fns(&mut leaker, &parse_quote!(<'a>), sig);
        let referrer = leaker.result().finish().map_err(|e| messages(&e))?;
        Ok(common::types(&referrer)
            .map(|t| quote!(#t).to_string())
            .collect::<BTreeSet<_>>())
    };
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
        assert_eq!(leak(sig.clone()), Ok(BTreeSet::new()), "{}", sig);
    }

    // A relative type applied to `Self` is leaked as a whole.
    assert_eq!(
        leak(quote!(
            fn f(self: Rel<Self>);
        )),
        Ok(strings(&["Rel<Self>"]))
    );
    // `Rel` may be a const, so the path is leaked as a whole, with `Self` in it, and `Rel` is not
    // leaked as a type (see `tests/bare_const.rs`).
    assert_eq!(
        leak(quote!(
            fn f(self: ::k::W<Rel, Self>);
        )),
        Ok(strings(&["::k::W<Rel, Self>"]))
    );
    // A receiver has no optional entry: the path of another crate is written as it is, and `Rel`
    // is leaked.
    assert_eq!(
        leak(quote!(
            fn f(self: ::k::W<(Rel,), Self>);
        )),
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
        leaker.intern_trait(&item);
        let assertions = leaker.assertions().to_string();
        assert!(leaker.result().errors().is_empty());
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
    leaker.intern_trait(&item);
    assert!(leaker.assertions().is_empty());
    assert!(leaker.result().errors().is_empty());
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
        "use absolute path: 'RelTrait<'_>::Ty'"
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
fn test_a_type_with_self_is_leaked_as_a_whole() {
    let leak =
        |sig: proc_macro2::TokenStream| -> std::result::Result<BTreeSet<String>, syn::Error> {
            let mut leaker = Leaker::new();
            common::intern_fns(&mut leaker, &Generics::default(), sig);
            let referrer = leaker.result().finish().map_err(|e| to_syn_error(&e))?;
            Ok(common::types(&referrer)
                .map(|t| quote!(#t).to_string())
                .collect::<BTreeSet<_>>())
        };
    let set = |types: &[&str]| types.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();

    // A relative type applied to `Self` is leaked as a whole.
    assert_eq!(
        leak(quote!(
            fn f(&self) -> Rel<Self>;
        ))
        .unwrap(),
        set(&["Rel < Self >"])
    );

    // `Self` does not depend on the context by itself, so `Rel` is enough.
    let sig = quote!(
        fn f(&self) -> (Rel, Self);
    );
    assert_eq!(leak(sig).unwrap(), set(&["Rel"]));
    // The type of an external crate with `Self` in it is one required entry with `Rel` in it (it
    // was the entry `Rel` and an optional one).
    let sig = quote!(
        fn f(&self) -> ::k::W<(Rel,), Self>;
    );
    assert_eq!(leak(sig).unwrap(), set(&[":: k :: W < (Rel ,) , Self >"]));

    // `Rel` as an argument may be a const, so the path is interned as a whole: `Rel` is not leaked
    // as a type (see `tests/bare_const.rs`).
    let sig = quote!(
        fn f(&self) -> ::k::W<Rel, Self>;
    );
    assert_eq!(leak(sig).unwrap(), set(&[":: k :: W < Rel , Self >"]));

    // The type of an external crate with `Self` in it is an entry, which the Referrer replaces.
    for (ty, entry) in [
        (quote!(::k::W<(), Self>), ":: k :: W < () , Self >"),
        (quote!(::k::W<Self>), ":: k :: W < Self >"),
    ] {
        let sig = quote!(
            fn f(&self) -> #ty;
        );
        assert_eq!(leak(sig).unwrap(), set(&[entry]), "{}", ty);
    }

    // `Self` alone is never leaked.
    for ty in [quote!(Self), quote!(Self::Item), quote!(&Self)] {
        assert_eq!(
            leak(quote!(
                fn f(&self) -> #ty;
            ))
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
    leaker.intern_impl(&item);
    let referrer = leaker.result().finish().unwrap();
    // `::k::W`, the type of an external crate, is an entry, after the others.
    assert_eq!(
        common::types(&referrer)
            .map(|t| quote!(#t).to_string())
            .collect::<Vec<_>>(),
        ["RelArg", "RelC", "RelRet", ":: k :: W"]
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
    leaker.intern_mod(&item);
    assert_eq!(
        leaker
            .result()
            .finish()
            .unwrap()
            .iter()
            .map(|(_, t, _)| quote!(#t).to_string())
            .collect::<BTreeSet<_>>(),
        ["RelC", "RelS", "RelRet"]
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<_>>()
    );
}

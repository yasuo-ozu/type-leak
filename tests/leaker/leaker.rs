//! Tests for building a `Leaker` and interning items.
//!
//! The methods of the `Leaker` which analyze a whole item (`intern_struct()`, `intern_enum()`,
//! `intern_union()`, `intern_impl()` and `intern_trait()`) come first. Their
//! expected values are those of the constructors of the Leaker which analyzed an item and which
//! they replace (`Leaker::from_*()`), taken from the previous version of the crate: the item is
//! analyzed with its own generics as those of the Leaker, which is what the top-level context has,
//! as declared. Then come the configuration and the rules, the other ways to build a Leaker, the
//! scopes (`enter_*()`), what `result()` gives, and the regressions (among them those of the
//! audit: the requirements of the leaked types, the causes found in trait objects, impls, macros
//! and qualified paths, and the assumptions about primitive types).

use crate::common;

use common::{
    check_in, e, entries_with_generics, id, messages, s, starts_with, to_syn_error, Recovered,
};
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::{
    Allow, Cause, Context, EntryId, ImplSelf, Leaker, Place, Projection, Reason, Referrer, Rules,
    SelfRequirement, View,
};

/// The entries as `type | generics | self`, in the order of the Referrer.
fn entries(referrer: &Referrer) -> Vec<String> {
    (0..referrer.iter().count())
        .map(|ix| {
            let ty = common::types(referrer).nth(ix).unwrap();
            let context = common::entry(referrer, ix);
            let generics = &context.generics;
            format!(
                "{} | {} | {}",
                s(ty),
                s(generics),
                common::self_text(context)
            )
        })
        .collect()
}

fn ids(referrer: &Referrer) -> Vec<usize> {
    (0..referrer.iter().count())
        .map(|ix| common::entry_id(referrer, ix).index())
        .collect()
}

/// Whether the item is analyzed without an error.
fn is_accepted(item: Item) -> bool {
    let mut leaker = Leaker::new();
    common::intern_item(&mut leaker, &item);
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
        assert!(!leaker.check_values && !leaker.check_bodies);
        leaker.intern_trait(&item);
        leaker.result()
    };
    let (default, new) = (analyze(Leaker::default()), analyze(Leaker::new()));
    assert!(default.errors().is_empty());
    let (default, new) = (default.finish().unwrap(), new.finish().unwrap());
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
    assert_eq!(named.result().finish().unwrap(), new);
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
    // `MyType6` uses all the generics, but its context is not the top-level one.
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
    assert_eq!(top_level.index(), 2);
    assert_eq!(common::context_ids(&referrer).len(), 3);
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
    // `expand()` replaces a leaked type by what the callback returns for its index, and nothing
    // else (the entries are sorted by tokens: the tuple, `MyType1`, `MyType4`).
    let mut f = |_, ix: EntryId, _: &Context| -> Type {
        let ix = syn::Index::from(ix.index());
        parse_quote!(::path::to::Implementor<#ix>)
    };
    let rows: [(Type, Type); 4] = [
        (
            parse_quote!(MyType1),
            parse_quote!(::path::to::Implementor<1>),
        ),
        (
            parse_quote!((MyType2, MyType3<MyType1>, MyType4, MyType5)),
            parse_quote!(::path::to::Implementor<0>),
        ),
        (
            parse_quote!(MyType4),
            parse_quote!(::path::to::Implementor<2>),
        ),
        // A type that is not leaked is left as it is.
        (parse_quote!(Other), parse_quote!(Other)),
    ];
    for (ty, expected) in rows {
        assert_eq!(common::expand(&referrer, top_level, ty, &mut f), expected);
    }
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
    // The top-level context has the generics of the enum as declared and, as `Rel2<Self>` names
    // `Self`, what `Self` is: it is a context of its own, after that of the entry.
    assert_eq!(top_level.index(), 3);
    assert_ne!(top_level, common::entry_id(&referrer, 3));
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
    // `Wrap<T>` uses the generics of the union, but its context is not the top-level one.
    assert_eq!(
        entries(&referrer),
        [
            "MyType |  | -",
            "Wrap < T > | < T : :: core :: marker :: Copy > | -"
        ]
    );
    assert_eq!(ids(&referrer), [0, 1]);
    assert_eq!(top_level.index(), 2);
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
    // The top-level context has the generics of the trait as declared and, as `Rel2<Self>` names
    // `Self`, what `Self` is: it is a context of its own, after that of the entry.
    assert_eq!(top_level.index(), 3);
    assert_ne!(top_level, common::entry_id(&referrer, 0));
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
    // The generics of the impl as declared are the top-level context, which the entries that use
    // them exactly do not have: it is a context of its own, after theirs.
    assert_eq!(top_level.index(), 2);
    assert_ne!(top_level, common::entry_id(&referrer, 0));

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

/// `intern_impl()` leaks the trait arguments and the signature, not the body.
#[test]
fn intern_impl_skips_the_body() {
    let item: ItemImpl = parse_quote! {
        impl<T> ::path::MyTrait<Rel<T>> for ::path::Host<T> {
            fn f(&self) -> Rel2<Self> { Skipped::new() } // the body is skipped
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    // `Rel<T>` and `Rel2<Self>`, and the self type, which names another crate.
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 3);
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
    assert_eq!(d, id(5));
    let referrer = leaker.result().finish().unwrap();
    // `Rel` is leaked once, in the context of no generics.
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
    // The same generics as declared are the same context, whichever item they are the generics of.
    assert_eq!(b, a);
    assert_eq!(c, id(3));
    // `Wrap<T>` is the entry of the generics `<T>` of both, and `Rel` of the no generics of `E`,
    // each in a context of its own: neither is the top-level context of an item.
    assert_eq!(common::entry_id(&referrer, 2), id(2));
    assert_eq!(s(&common::entry(&referrer, 2).generics), "< T >");
    assert_eq!(common::entry_id(&referrer, 0), id(0));
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
/// one by one in the module `names` of `tests/allow/allow_rules.rs`.
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
        common::type_texts(&referrer),
        ["crate :: A", "crate :: B", "u8"]
    );
}

/// `?Sized` is left as written, and `Self: Sized` is a relative trait.
#[test]
fn allow_prelude_is_needed_for_sized_but_not_for_relaxed_sized() {
    let relaxed: ItemTrait = parse_quote! { trait Tr<T: ?Sized> { fn f(&self, t: &T); } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&relaxed);
    assert!(leaker.result().finish().is_ok());
    let required: ItemTrait = parse_quote! { trait Tr { fn f(&self) where Self: Sized; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&required);
    assert_eq!(
        leaker.result().errors()[0].message,
        "use absolute path: 'Sized'"
    );
}

/// `Rules::prelude()` leaves `Clone` and `Vec` as written, so `Vec<Rel>` is the only entry.
#[test]
fn allow_prelude_leaves_the_names_of_the_prelude_as_written() {
    let item: ItemTrait = parse_quote! { trait Tr: Clone { fn f(&self) -> Vec<Rel>; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    assert!(leaker.result().finish().is_err());

    let config = Rules::new().prelude();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(&item);
    // `Clone` and `Vec` are left as written. The only entry is `Vec<Rel>`: a bare `Rel` may be
    // a const, so the type that holds it is leaked whole.
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
}

/// `LeakerResult::allow()` drops the error, and the part is copied as written either way.
#[test]
fn allow_a_cause_copies_that_part_as_written() {
    let item: ItemImpl = parse_quote! {
        impl ::p::S {
            fn f(_: ::p::W<'_, dyn RelTr>) {}  // an elided lifetime, and `RelTr` is relative
            fn g(_: ::p::X<dyn RelTr>) {}      // fine: `dyn RelTr` is leaked
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let mut result = leaker.result();
    // The error is about the trait, which is smaller than the type `W<'_, dyn RelTr>` it is in.
    assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));

    // `f` is copied as written either way, and only `g` leaks `dyn RelTr`: the entries are that
    // type in `g` and the self type.
    result = result.allow(|c: &Cause, _: Place| *c == Cause::Trait(parse_quote!(RelTr)));
    assert_eq!(result.finish().unwrap().iter().count(), 2);
}

/// `Rel2<Self, T>` is an entry for any `Self` with the bound, with the error `NestedSelfInBounds`
/// that `allow()` drops.
#[test]
fn a_nested_type_that_names_self_with_a_self_bound_is_an_entry_with_an_error() {
    let item: Item = parse_quote! {
        pub struct S<T: ::core::convert::AsRef<Self>> { pub b: Rel2<Self, T> }
    };
    let mut leaker = Leaker::with_rule(Rules::new());
    leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let result = leaker.result();
    // The risk is an error, with no paths, and the type is an entry all the same.
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::NestedSelfInBounds);
    assert!(result.errors()[0].paths.is_empty());
    assert!(result.clone().finish().is_err());
    assert_eq!(result.clone().recovered().iter().count(), 1);

    // A rule that allows the cause of the error accepts the risk.
    let cause = result.errors()[0].cause.clone();
    let accepted = cause.clone();
    let result = result.allow(move |c: &Cause, _: Place| *c == accepted);
    let referrer = result.clone().finish().unwrap();
    // The entry is for any type that the bound holds for, with `Self` as its implementor.
    let (_, _, context) = referrer.iter().next().unwrap();
    assert!(matches!(context.self_ty, Some(SelfRequirement::Any)));
    let repeater: Path = parse_quote!(Repeater);
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&Projection::new(
            repeater.clone(),
            7,
            ImplSelf::Param(param.clone()),
        ))
        .unwrap()
        .remove(0);
    // `r.path` has no `KEY`, and `ToTokens` writes the one of the id 7.
    let key = common::key(7, r.entry);
    let imp: ItemImpl = parse_quote!(#r);
    let expected: ItemImpl = parse_quote! {
        impl<T: ::core::convert::AsRef<__S>, __S> Repeater<(T,), #key> for __S {
            type Type = Rel2<__S, T>;
        }
    };
    assert_eq!(imp, expected);

    // The rule of the Leaker drops the error as well, at the end of the call: the entry is the same.
    let mut leaker = Leaker::with_rule(move |c: &Cause, _: Place| *c == cause);
    leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let result = leaker.result();
    assert!(result.errors().is_empty());
    assert_eq!(result.finish().unwrap(), referrer);
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

// ── the scopes leak what their closure visits ────────────────────────────────

/// `enter_generics()` records the error of an inner scope, and the outer one goes on.
#[test]
fn enter_generics_goes_on_after_an_error_of_an_inner_scope() {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| {
        view.visit_type(&parse_quote!(Rel<T>));
        // A relative trait cannot be leaked: an error of the inner scope, the outer one goes on.
        view.enter_generics(&parse_quote!(<U: Relative>), |view| {
            view.visit_type_param_bound(&parse_quote!(Relative));
        });
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.recovered().iter().count(), 1);
}

/// `enter_type()` leaks only what the closure visits.
#[test]
fn enter_type_leaks_only_the_visited_types() {
    let item: ItemStruct = parse_quote! { struct S<T> { a: Wrap<T>, b: Skipped } };
    let mut leaker = Leaker::new();
    leaker.enter_type(&item.ident, &item.generics, |view| {
        view.visit_type(&item.fields.iter().next().unwrap().ty); // only the first field
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Wrap<T>`
}

/// `enter_impl()` leaks only what the closure visits.
#[test]
fn enter_impl_leaks_only_the_visited_trait_arguments() {
    let item: ItemImpl = parse_quote! { impl<T> ::path::MyTrait<Rel<T>> for MyType<T, Rel2> {} };
    let (_, trait_path, _) = item.trait_.as_ref().unwrap();
    let mut leaker = Leaker::new();
    leaker.enter_impl(&item.generics, &item.self_ty, Some(trait_path), |view| {
        for segment in &trait_path.segments {
            view.visit_path_arguments(&segment.arguments);
        }
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Rel<T>`, not `Rel2`
}

/// `enter_trait()` leaks only what the closure visits.
#[test]
fn enter_trait_leaks_only_the_visited_signature() {
    use syn::visit::Visit;
    let item: ItemTrait = parse_quote! {
        trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self); }
    };
    let TraitItem::Fn(method) = &item.items[0] else {
        unreachable!()
    };
    let mut leaker = Leaker::new();
    leaker.enter_trait(&item.ident, &item.generics, |view| {
        view.visit_signature(&method.sig)
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Wrap<T>`
}

// ── regressions ──────────────────────────────────────────────────────────────

/// The entries of `item`, with the primitive types and the prelude allowed, or its first error.
fn leak<I: Clone + Into<Item>>(item: &I) -> Result<BTreeSet<String>> {
    let config = type_leak::Rules::new().primitive().prelude();
    let mut leaker = Leaker::with_rule(config);
    common::intern_item(&mut leaker, &item.clone().into());
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
    assert_eq!(leak(&item).unwrap(), BTreeSet::<String>::new());

    // Relative types inside them are leaked without the lifetimes.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f<F: for<'a> ::core::ops::Fn(&'a MyType)>(&self, f: F);
        }
    };
    assert_eq!(leak(&item).unwrap(), strings(&["MyType"]));
}

fn config() -> Rules {
    Rules::new()
}

/// The Referrer of `item`, or the text of its errors.
fn leak_trait_with<A: Allow>(config: A, item: &ItemTrait) -> std::result::Result<Referrer, String> {
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(item);
    leaker
        .result()
        .finish()
        .map_err(|e| to_syn_error(&e).to_string())
}

fn leak_trait(item: ItemTrait) -> Referrer {
    leak_trait_with(config(), &item).unwrap()
}

/// The compile-time checks that the analysis of `item` asks for (`Leaker::assertions()`); it has
/// no error.
fn assertions_of<A: Allow>(config: A, item: &ItemTrait) -> String {
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(item);
    let assertions = leaker.assertions().to_string();
    leaker.result().finish().unwrap();
    assertions
}

#[test]
fn self_predicates_are_requirements() {
    // `where Self: Sized` of a method.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f(&self) -> Wrap<Self> where Self: ::core::marker::Sized;
        }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e(
            "Wrap < Self >",
            "where Self : :: core :: marker :: Sized"
        )]
    );
    assert!(matches!(
        common::entry(&referrer, 0).self_ty.as_ref(),
        Some(SelfRequirement::Bound(_))
    ));

    // The root predicates and those bounding `Self` with a lifetime too: `Self: 'a` is not a
    // predicate of the generics (it is in `context_self_outlives()`, encoded and never declared).
    let referrer = leak_trait(parse_quote! {
        trait Tr<T> where Self: ::core::marker::Send, T: ::core::marker::Sync {
            fn f<'a>(&'a self) -> Wrap<'a, Self> where Self: 'a;
        }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e(
            "Wrap < 'a , Self >",
            "< 'a , T : :: core :: marker :: Sync > where Self : :: core :: marker :: Send"
        )]
    );
    let lifetime: Lifetime = parse_quote!('a);
    assert_eq!(
        common::entry(&referrer, 0).self_outlives.as_slice(),
        [lifetime]
    );

    // A type which does not mention `Self` does not get them.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f(&self) -> Wrap where Self: ::core::marker::Sized;
        }
    });
    assert_eq!(entries_with_generics(&referrer), [e("Wrap", "")]);
}

#[test]
fn dyn_lifetime_bounds_are_visited() {
    // A generic lifetime is used by the type.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f<'a>(&self) -> ::std::boxed::Box<dyn RelTr + 'a>;
        }
    });
    // The `Box` is an entry as a whole, with the lifetime among its generics.
    assert_eq!(
        entries_with_generics(&referrer),
        [e(":: std :: boxed :: Box < dyn RelTr + 'a >", "< 'a >")]
    );
    // If the Leaker allows the `Box`, only the trait object is leaked.
    let referrer = leak_trait_with(
        config().or_if(starts_with(parse_quote!(::std::boxed::Box))),
        &parse_quote! {
            trait Tr {
                fn f<'a>(&self) -> ::std::boxed::Box<dyn RelTr + 'a>;
            }
        },
    )
    .unwrap();
    assert_eq!(
        entries_with_generics(&referrer),
        [e("dyn RelTr + 'a", "< 'a >")]
    );
    let mut leaker = Leaker::new();
    let generics = Generics::default();
    // `'_` is impossible.
    let ty: Type = parse_quote!(::p::Wrap<dyn ::p::Tr + '_>);
    let result = check_in(&mut leaker, &ty, &generics);
    assert_eq!(result.impossible.iter().map(s).collect::<Vec<_>>(), ["'_"]);
    // A lifetime which is not a generic is a cause.
    let ty: Type = parse_quote!(::p::Wrap<dyn ::p::Tr + 'b>);
    let result = check_in(&mut leaker, &ty, &generics);
    assert_eq!(result.must.iter().map(s).collect::<Vec<_>>(), ["'b"]);
    assert!(result.impossible.is_empty());
    // A generic one is not.
    let generics: Generics = parse_quote!(<'b>);
    let result = check_in(&mut leaker, &ty, &generics);
    assert!(result.must.is_empty());
}

#[test]
fn reference_to_dyn_defaults_to_its_lifetime() {
    // With an elided lifetime, the trait object cannot be leaked alone.
    let err = leak_trait_with(
        config(),
        &parse_quote! { trait Tr { fn f(&self) -> &dyn RelTr; } },
    )
    .err()
    .unwrap();
    assert!(err.contains("use absolute path: 'RelTr'"), "{}", err);
    assert!(
        err.contains(
            "= note: the lifetime of the trait object defaults to that of the enclosing \
             reference or path\n"
        ),
        "{}",
        err
    );
    for ty in ["&mut dyn RelTr", "&(dyn RelTr)", "&'_ dyn RelTr"] {
        let ty: Type = syn::parse_str(ty).unwrap();
        let item: ItemTrait = parse_quote! { trait Tr { fn f(&self) -> #ty; } };
        assert!(leak_trait_with(config(), &item).is_err(), "{}", s(&ty));
    }
    // Its components can be leaked.
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f(&self) -> &dyn ::p::Tr<Rel>; }
    });
    assert_eq!(entries_with_generics(&referrer), [e("Rel", "")]);
    // A named lifetime is inherited by the trait object (or its trait has a bound of its own, which
    // rustc knows and the Leaker does not), so the reference is interned whole.
    let referrer = leak_trait(parse_quote! {
        trait Tr<'a> { fn f(&self) -> &'a dyn RelTr; }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e("& 'a dyn RelTr", "< 'a >")]
    );
    // So is the reference of a lifetime of the method, which the receiver keeps constrained in the
    // copy.
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f<'a>(&'a self) -> &'a dyn RelTr; }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e("& 'a dyn RelTr", "< 'a >")]
    );
    // Unless the return type mentions it and nothing else keeps it as written: the copy would have
    // it only in a projection, which rustc rejects (E0195).
    let err = leak_trait_with(
        config(),
        &parse_quote! { trait Tr { fn f<'a>(&self, x: &'a dyn RelTr) -> &'a dyn RelTr; } },
    )
    .err()
    .unwrap();
    assert!(
        err.contains(
            "`RelTr` leaves the lifetime `'a` of the method unconstrained in the copy (E0195)"
        ),
        "{}",
        err
    );
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f<'a>(&'a self) -> &'a (dyn RelTr + 'a); }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e("dyn RelTr + 'a", "< 'a >")]
    );
    // No argument mentions the lifetime (it is in the return type only: early-bound in the
    // original and in the copy alike, which has it only in a projection, so rustc accepts the
    // implementation): the reference or the path is leaked.
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f<'a>(&self) -> &'a dyn RelTr; }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e("& 'a dyn RelTr", "< 'a >")]
    );
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f<'a>(&self) -> ::core::cell::Ref<'a, dyn RelTr>; }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e(":: core :: cell :: Ref < 'a , dyn RelTr >", "< 'a >")]
    );
    // An argument which mentions it without keeping it as written does not exempt it.
    let err = leak_trait_with(
        config(),
        &parse_quote! { trait Tr { fn f<'a>(&self, x: ::std::vec::Vec<&'a u8>) -> &'a dyn RelTr; } },
    )
    .err()
    .unwrap();
    assert!(err.contains("E0195"), "{}", err);
    // An absolute trait object of `::core` (which is not leaked as a whole) keeps it as written,
    // whatever is inside it, which is leaked alone: the other reference is leaked.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f<'a>(&self, x: &'a dyn RelTr, y: &'a dyn ::core::fmt::Debug) -> &'a ();
            fn g<'a>(&self, x: &'a dyn RelTr, y: &'a dyn ::core::iter::Iterator<Item = Rel>) -> &'a ();
        }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e("& 'a dyn RelTr", "< 'a >"), e("Rel", "")]
    );
    // One which may be a cause itself does not (conservative): a relative trait, a bare
    // identifier which may be a const, and an external crate which is not allowed: the Referrer
    // may replace the reference to it, which then keeps nothing as written.
    for y in [
        "&'a dyn RelTr2",
        "&'a dyn ::core::convert::AsRef<N>",
        "&'a dyn ::p::Tr",
        "&'a dyn ::p::Tr<Item = Rel>",
    ] {
        let y: Type = syn::parse_str(y).unwrap();
        let err = leak_trait_with(
            config(),
            &parse_quote! { trait Tr { fn f<'a>(&self, x: &'a dyn RelTr, y: #y) -> &'a u8; } },
        )
        .err()
        .unwrap();
        assert!(err.contains("E0195"), "{}: {}", s(&y), err);
    }
    // Unless it is allowed, as `::core` is.
    let allowed = config().or_if(starts_with(parse_quote!(::p)));
    let referrer = leak_trait_with(
        allowed,
        &parse_quote! { trait Tr { fn f<'a>(&self, x: &'a dyn RelTr, y: &'a dyn ::p::Tr) -> &'a (); } },
    )
    .unwrap();
    assert_eq!(
        entries_with_generics(&referrer),
        [e("& 'a dyn RelTr", "< 'a >")]
    );
    // The default of a `Box` is `'static`, so the trait object is interned alone, if the `Box` is
    // allowed. If it is not, the `Box` is interned as a whole.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::std::boxed::Box<dyn RelTr>; }
    };
    let allowed = config().or_if(starts_with(parse_quote!(::std::boxed::Box)));
    let referrer = leak_trait_with(allowed, &item).unwrap();
    assert_eq!(entries_with_generics(&referrer), [e("dyn RelTr", "")]);
    let referrer = leak_trait(item);
    assert_eq!(
        entries_with_generics(&referrer),
        [e(":: std :: boxed :: Box < dyn RelTr >", "")]
    );
    // `check()` agrees.
    let result = check_in(
        &mut Leaker::new(),
        &parse_quote!(&dyn RelTr),
        &Generics::default(),
    );
    assert_eq!(
        result.impossible.iter().map(s).collect::<Vec<_>>(),
        ["& dyn RelTr", "dyn RelTr"]
    );
}

#[test]
fn impl_trait_path_is_checked_like_a_bound() {
    let leak = |items: &[ItemImpl]| {
        let mut leaker = Leaker::with_rule(config());
        for item in items {
            leaker.intern_impl(item);
        }
        leaker
            .result()
            .finish()
            .map_err(|e| to_syn_error(&e).to_string())
    };
    let err = leak(&[parse_quote! { impl Tr for () {} }]).err().unwrap();
    assert!(err.contains("use absolute path: 'Tr'"), "{}", err);
    let err = leak(&[parse_quote! { impl Clone for () {} }])
        .err()
        .unwrap();
    assert!(
        err.contains("use `::core::clone::Clone` instead of `Clone`"),
        "{}",
        err
    );
    // An absolute path is fine, and its arguments are analyzed.
    let referrer = leak(&[
        parse_quote! { impl ::p::Tr<Rel> for () {} },
        parse_quote! { impl ::p::Tr<Rel2> for () {} },
    ])
    .unwrap();
    assert_eq!(
        entries_with_generics(&referrer),
        [e("Rel", ""), e("Rel2", "")]
    );
}

#[test]
fn type_macros() {
    // `Self` in the tokens is leaked with the type.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::p::Wrap<::p::m!(Self)>; }
    };
    assert!(leak_trait_with(config(), &item).is_ok());
    let item: ItemStruct = parse_quote! {
        struct S { f: ::p::Wrap<::p::m!(Self)> }
    };
    let mut leaker = Leaker::with_rule(config());
    leaker.intern_struct(&item);
    assert!(leaker.result().finish().is_ok());
}

#[test]
fn primitives_are_assumed_when_the_type_is_copied() {
    let config = || config().primitive();
    let assertions = assertions_of(config(), &parse_quote! { trait Tr { fn f(&self) -> &u8; } });
    assert!(assertions.contains("__type_leak_assert_u8_is_primitive"));
    let assertions = assertions_of(
        config(),
        &parse_quote! { trait Tr { fn f(&self) -> impl ::p::Into<u16>; } },
    );
    assert!(assertions.contains("__type_leak_assert_u16_is_primitive"));
    // The analysis recovers from an error, so what it found is recorded: `u8` is copied.
    let mut leaker = Leaker::with_rule(config());
    let item: ItemTrait = parse_quote! { trait Tr: Clone { fn f(&self) -> u8; } };
    leaker.intern_trait(&item);
    assert!(leaker
        .assertions()
        .to_string()
        .contains("__type_leak_assert_u8_is_primitive"));
    assert!(leaker.result().finish().is_err());
    // But not on a panic.
    let mut leaker = Leaker::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = leaker.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Vec<u8>));
            panic!("a closure panics");
        });
    }));
    assert!(result.is_err());
    assert!(leaker.assertions().is_empty());
}

#[test]
fn raw_identifiers_are_compared_by_name() {
    let referrer = leak_trait(parse_quote! {
        trait Tr<r#T> { fn f(&self) -> Rel<T>; }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e("Rel < T >", "< r#T >")]
    );
    let referrer = leak_trait(parse_quote! {
        trait Tr<T> { fn f(&self) -> Rel<r#T>; }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e("Rel < r#T >", "< T >")]
    );
    let item: ItemStruct = parse_quote! {
        pub struct S<r#T>(Rel<T>);
    };
    let mut leaker = Leaker::with_rule(config());
    leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries_with_generics(&referrer),
        [e("Rel < T >", "< r#T >")]
    );
}

#[test]
fn qualified_self_through_generics() {
    let referrer = leak_trait(parse_quote! {
        trait Tr<T> { fn f(&self) -> Wrap<<T>::X>; }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e("Wrap < < T > :: X >", "< T >")]
    );
    // `check()` in the trait agrees, where `Self` is the implementor.
    let generics: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let (_, results) = leaker.enter_trait(&parse_quote!(Tr), &generics, |view| {
        let mut check = |ty: Type| view.check(&ty);
        [
            check(parse_quote!(<T>::X)),
            check(parse_quote!(<Self>::X)),
            check(parse_quote!(<Rel>::X)),
            check(parse_quote!(<U>::X)),
        ]
    });
    // A check records no error.
    assert!(leaker.result().errors().is_empty());
    let [through_param, through_self, relative, undeclared] = results;
    assert!(through_param.must.is_empty() && through_param.impossible.is_empty());
    assert!(through_self.must.is_empty() && through_self.impossible.is_empty());
    // Otherwise it is impossible.
    assert_eq!(
        relative.impossible.iter().map(s).collect::<Vec<_>>(),
        ["Rel"]
    );
    assert_eq!(
        undeclared.impossible.iter().map(s).collect::<Vec<_>>(),
        ["U"]
    );
}

#[test]
fn parameters_are_ordered_by_kind() {
    let referrer = leak_trait(parse_quote! {
        trait Tr<'b, 'a, const N: usize, U, T> { fn f(&self) -> Wrap<'b, 'a, N, U, T>; }
    });
    assert_eq!(
        entries_with_generics(&referrer),
        [e(
            "Wrap < 'b , 'a , N , U , T >",
            "< 'a , 'b , T , U , const N : usize >"
        )]
    );
}

#[test]
fn the_note_of_an_impl_trait_in_an_argument() {
    let err = leak_trait_with(
        config(),
        &parse_quote! { trait Tr { fn f(&self) -> Wrap<impl Into<u8>>; } },
    )
    .err()
    .unwrap();
    assert!(
        err.contains("use absolute path: 'Wrap<impl Into<u8>>'"),
        "{}",
        err
    );
    assert!(
        err.contains("it cannot be leaked as a whole because of `impl Into<u8>`"),
        "{}",
        err
    );
}

// ── the entries that reduce_roots() keeps ────────────────────────────────────
//
// `reduce_roots()` prunes the Leaker to the entries it chose, and what is analyzed afterwards is
// added to them: a kept entry stays one, of the kind it was kept as. The tests of `tests/
// reduce_roots.rs` show the pruning for the usual items; these show it against a later scope that
// changes the verdict on a kept entry.

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
    common::type_texts(&leaker.result().finish().unwrap())
}

// The generics of the next tests: the bound of `T` names `::ext::X`, a type of another crate, so
// in a predicate a type that uses `T` cannot be an optional entry: the Referrer would write `X` of
// the bound as a path through the Repeater, where the impl of the entry needs it as written
// (`tests/leaker/bound_cycles.rs`). In a field it can. The pair is one entry in a field (cheaper
// than `A<U>` and `B<T>`), and the same type in a predicate, in the same context, cannot be one.

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
        common::type_texts(&leaker.result().finish().unwrap())
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

/// `result()` puts the entries that depend on the context first, then the types of other crates.
#[test]
fn result_lists_the_types_of_other_crates_last() {
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::foo::X; fn g(&self) -> Rel; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    // `Rel` depends on the context. `::foo::X` names another crate: it comes after.
    let found: Vec<&Type> = referrer.iter().map(|(_, ty, _)| ty).collect();
    assert_eq!(found, [&parse_quote!(Rel), &parse_quote!(::foo::X)]);
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
        leak(&item).unwrap(),
        strings(&["MyType1", "MyType2", "MyType3", "MyType4"])
    );

    // A relative path with an elided lifetime cannot be decomposed.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, x: MyType<'_>);
        }
    };
    assert_eq!(
        leak(&item).unwrap_err().to_string(),
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
        leak(&item).unwrap(),
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
        leak(&item).unwrap(),
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
    // leaked as a type (see `tests/leaker/bare_const.rs`).
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
    assert_eq!(leak(&item).unwrap(), strings(&["RelA", "RelB", "RelC"]));
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
    assert_eq!(leak(&item).unwrap(), strings(&["RelA"]));
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
    // as a type (see `tests/leaker/bare_const.rs`).
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
        common::type_texts(&referrer),
        ["RelArg", "RelC", "RelRet", ":: k :: W"]
    );

    // Also those of the nested items.
    let items: [Item; 3] = [
        parse_quote! { const C: RelC = RelCVal::new(); },
        parse_quote! { static S: RelS = RelSVal::new(); },
        parse_quote! { fn f() -> RelRet { let x: RelBody = RelBody::new(); x } },
    ];
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        for item in &items {
            view.visit_item(item);
        }
    });
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

/// The items with `len` as the length of an array in a bound, a self type, a binder or a field: a
/// probe of the bounds or of the binder walks it too.
fn items_with_length(len: &Expr) -> Vec<(&'static str, Item)> {
    vec![
        (
            "struct, a bound",
            parse_quote!(
                pub struct S<T>(T)
                where
                    T: From<[u8; #len]> + Into<Rel<T>>;
            ),
        ),
        (
            "impl, a bound",
            parse_quote!(
                impl<T: From<[u8; #len]> + Into<Rel<T>>> S<T> {}
            ),
        ),
        (
            "impl, the self type",
            parse_quote!(
                impl<T: Into<Rel<Self>>> S<[u8; #len], T> {}
            ),
        ),
        (
            "method",
            parse_quote!(
                impl S {
                    fn f<T: From<[u8; #len]> + Into<Rel<T>>>(t: T) {}
                }
            ),
        ),
        (
            "trait method",
            parse_quote!(
                trait Tr {
                    fn f<T>(t: T)
                    where
                        T: From<[u8; #len]> + Into<Rel<T>>;
                }
            ),
        ),
        (
            "binder",
            parse_quote!(
                pub struct S(for<'a> fn(&'a [u8; #len]) -> &'a u8, Rel);
            ),
        ),
        ("field", parse_quote!(pub struct S([u8; #len], Rel);)),
    ]
}

/// A `mod` in an expression of a type (an array length) goes with the type: the analysis is that
/// of the same item without it, the serialized result included (no context is made for the
/// module), whether the item is analyzed by itself or nested in a scope, and nothing panics.
#[test]
fn a_mod_in_an_expression_of_a_type_goes_with_the_type() {
    let with_mod: Expr = parse_quote!({
        mod m {}
        1
    });
    let without: Expr = parse_quote!({ 1 });
    // The serialized result (the Referrer and the errors), with the module's tokens left out.
    let analyze = |item: &Item, nested: bool| -> String {
        let mut leaker = Leaker::new();
        match (nested, item) {
            (true, _) => {
                leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
            }
            (false, Item::Struct(item)) => {
                leaker.intern_struct(item);
            }
            (false, Item::Trait(item)) => {
                leaker.intern_trait(item);
            }
            (false, Item::Impl(item)) => {
                leaker.intern_impl(item);
            }
            _ => unreachable!("not an item of the table"),
        }
        let result = leaker.result();
        quote!(#result).to_string().replace("mod m { } ", "")
    };
    for ((name, with), (_, without)) in items_with_length(&with_mod)
        .into_iter()
        .zip(items_with_length(&without))
    {
        for nested in [false, true] {
            assert_eq!(
                analyze(&with, nested),
                analyze(&without, nested),
                "{} (nested: {})",
                name,
                nested
            );
        }
    }
}

/// The items of a `mod` in an expression of a type are nested items: a relative path in them is a
/// cause of the type, which is leaked whole with the module.
#[test]
fn the_items_of_a_mod_in_an_expression_of_a_type_are_nested_items() {
    let ty: Type = parse_quote!(
        [u8; {
            mod m {
                pub struct N(pub Rel);
            }
            1
        }]
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote!(pub struct S(#ty, Rel2);));
    let referrer = leaker.result().finish().unwrap();
    let types: Vec<String> = common::type_texts(&referrer);
    assert_eq!(types, ["Rel2".to_string(), s(&ty)]);
}

/// A `mod` item is not analyzed: `View::visit_item()` of one, or a walk of the View that meets
/// one, gives one `Unreadable` error whose cause is `mod name`, with no paths, and nothing in it
/// is leaked. `allow()` of the cause accepts it.
#[test]
fn a_mod_item_is_one_unreadable_error() {
    let module: ItemMod = parse_quote!(
        mod m {
            pub struct S(Rel);
        }
    );
    let check = |result: type_leak::LeakerResult, case: &str| {
        let errors = result.errors();
        assert_eq!(errors.len(), 1, "{}: {}", case, messages(errors));
        let error = &errors[0];
        assert_eq!(error.reason, type_leak::Reason::Unreadable, "{}", case);
        assert_eq!(error.cause, Cause::Unreadable(quote!(mod m)), "{}", case);
        assert!(error.paths.is_empty(), "{}", case);
        assert_eq!(
            messages(errors),
            "cannot analyze the module `m`",
            "{}",
            case
        );
        let cause = error.cause.clone();
        let accepted = result.allow(move |c: &Cause, _: Place| *c == cause);
        let referrer = accepted
            .finish()
            .unwrap_or_else(|errors| panic!("{}: allow() left {}", case, messages(&errors)));
        assert!(referrer.is_empty(), "{}", case);
    };
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Mod(module.clone()))
    });
    check(leaker.result(), "View::visit_item()");
    let file: File = parse_quote!(#module);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        syn::visit::visit_file(view, &file)
    });
    check(leaker.result(), "a walk of the View");
}

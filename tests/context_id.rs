//! Tests for `ContextId`: the ids of the contexts are assigned by the Leaker when the requirements
//! first appear, the top-level context of a call (the id of its generics) is returned by
//! `Leaker::enter_generics()` and its relatives, and the ids are not renumbered by
//! `Leaker::result()`, so the Referrer may have gaps.

mod common;

use common::{panic_message, s, AcceptCause};
use proc_macro2::{TokenStream, TokenTree};
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{
    array_const, Cause, ContextId, ImplSelf, Leaker, LeakerResult, ModuleScope, Referrer,
    SelfRequirement,
};

/// The tag of the format of a serialized Referrer, which it starts with.
fn version() -> String {
    format!("type-leak-format-{}", Referrer::FORMAT_VERSION)
}

fn id(n: usize) -> ContextId {
    parse_str(&n.to_string()).unwrap()
}

fn ids(referrer: &Referrer) -> Vec<usize> {
    common::context_ids(referrer)
        .into_iter()
        .map(ContextId::index)
        .collect()
}

fn leaked(referrer: &Referrer) -> Vec<String> {
    common::types(referrer).map(s).collect()
}

/// The Referrer of `struct S<T, U> { a: Rel0, b: (Rel1, T), c: Wrap<U> }`, together with the
/// top-level context returned by the Leaker. `Rel0` and `Rel1` use no generics (the context 0),
/// the tuple uses `T` (the context 1) and `Wrap<U>` uses `U` (the context 2), and the generics of
/// the struct `<T, U>` are the context 3. Only `Rel1` is interned for the tuple, so no type of
/// the context 1 is kept, and it is not top-level either.
fn gap_referrer() -> (Referrer, ContextId) {
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: Rel0, b: (Rel1, T), c: Wrap<U> } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    (leaker.result().finish().unwrap(), top_level)
}

#[test]
fn context_id_tokens_parse_and_display() {
    let three: ContextId = parse_quote!(3);
    assert_eq!(three.index(), 3);
    assert_eq!(three.to_string(), "3");
    assert_eq!(format!("{:>3}", three), "  3");

    // It is written as an unsuffixed integer literal.
    let tokens = three.to_token_stream();
    assert_eq!(tokens.to_string(), "3");
    let tokens: Vec<TokenTree> = tokens.into_iter().collect();
    assert!(matches!(&tokens[..], [TokenTree::Literal(_)]));

    // It is parsed from an integer, with the suffix `usize` or without any.
    assert_eq!(parse_str::<ContextId>("3usize").unwrap(), three);
    assert_eq!(parse_str::<ContextId>("0").unwrap().index(), 0);
    let message = |text: &str| parse_str::<ContextId>(text).unwrap_err().to_string();
    assert_eq!(
        message("3u8"),
        "a context id is an integer, with no suffix or `usize`"
    );
    assert_eq!(
        message("3i64"),
        "a context id is an integer, with no suffix or `usize`"
    );
    assert!(parse_str::<ContextId>("-3").is_err());
    assert_eq!(message("three"), "expected integer literal");
    assert_eq!(message("3 4"), "unexpected token");
    assert_eq!(
        message(""),
        "unexpected end of input, expected integer literal"
    );
    // It can be followed by other tokens.
    struct Followed(ContextId, TokenStream);
    impl parse::Parse for Followed {
        fn parse(input: parse::ParseStream) -> Result<Self> {
            Ok(Self(input.parse()?, input.parse()?))
        }
    }
    let followed: Followed = parse2(quote!(#three struct Z;)).unwrap();
    assert_eq!(
        (followed.0, followed.1.to_string()),
        (three, "struct Z ;".to_string())
    );

    // The round trip, and the order of the numbers.
    for n in [0, 1, 7, 1000] {
        assert_eq!(parse2::<ContextId>(id(n).to_token_stream()).unwrap(), id(n));
    }
    assert!(id(1) < id(2));
    let set: BTreeSet<ContextId> = [id(2), id(0), id(1), id(2)].iter().copied().collect();
    assert_eq!(
        set.into_iter().map(ContextId::index).collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn the_scopes_return_the_top_level_context() {
    // The generics of the trait `<T>` are not those `Rel` uses (none), so the top-level context
    // has no entry, but it is in the Referrer, with those generics.
    let tr: ItemTrait = parse_quote! { trait Tr<T> { fn f(&self) -> Rel; } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    assert_eq!(top_level, id(1));
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(leaked(&referrer), ["Rel"]);
    assert_eq!(common::entry_id(&referrer, 0), id(0));
    assert_eq!(ids(&referrer), [0, 1]);
    assert_eq!(s(&common::entry(&referrer, 0).generics), "");
    // It is written as an empty group, and kept by the round trip.
    let tokens = referrer.to_token_stream();
    assert_eq!(
        tokens.to_string(),
        format!("\"{}\" 2 (Rel) < T > ; ()", version())
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(ids(&parsed), [0, 1]);

    // The generics are in the canonical form of the requirements (the bounds are moved to the
    // parameters, the parameters are sorted, and the defaults are removed), and the `Self` of a
    // predicate is not a requirement of any type.
    let tr: ItemTrait = parse_quote! {
        trait Tr<U: ::core::fmt::Debug, T = u8> where T: ::core::clone::Clone, Self: ::core::marker::Sized {
            fn f(&self) -> Rel<T, U>;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let generics = &common::entry(&referrer, 0).generics;
    let where_clause = &generics.where_clause;
    assert_eq!(
        s(&quote!(#generics #where_clause)),
        "< T : :: core :: clone :: Clone , U : :: core :: fmt :: Debug >"
    );
    assert_eq!(common::entry(&referrer, 0).self_ty.as_ref(), None);
    assert!(!referrer.has_self());
}

#[test]
fn top_level_context_is_shared_with_the_entries() {
    // The generics of `Wrap<T>` are those of the struct.
    let item: ItemStruct = parse_quote! { struct S<T> { a: Wrap<T> } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    assert_eq!(top_level, id(0));
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0), top_level);
    assert_eq!(ids(&referrer), [0]);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("\"{}\" < T > ; (Wrap < T >)", version())
    );

    // With the bounds, whichever way they are written, and an entry of no generics aside.
    let item: ItemStruct =
        parse_quote! { struct S<T> where T: ::core::clone::Clone { a: Rel, b: Wrap<T> } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(leaked(&referrer), ["Rel", "Wrap < T >"]);
    assert_eq!(top_level, id(1));
    assert_eq!(common::entry_id(&referrer, 1), top_level);
    assert_eq!(ids(&referrer), [0, 1]);
}

#[test]
fn calls_with_the_same_generics_return_the_same_id() {
    let mut leaker = Leaker::new();
    let mut call = |item: ItemStruct| leaker.intern_struct(&item);
    let first = call(parse_quote! { struct A<T> { x: Rel } });
    // `Rel` (the context 0) is shared, and the generics `<T>` are the context 1.
    assert_eq!(first, id(1));
    assert_eq!(call(parse_quote! { struct B<T> { y: RelB } }), first);
    // The defaults are not part of the generics, nor are the order and the way bounds are written.
    assert_eq!(
        call(parse_quote! { struct C<T = ::core::primitive::u8> { y: Rel } }),
        first
    );
    assert_eq!(call(parse_quote! { struct D<U> { y: Rel } }), id(2));
    let bounded = call(parse_quote! { struct E<T: ::core::clone::Clone, U> { y: Rel } });
    assert_eq!(bounded, id(3));
    assert_eq!(
        call(parse_quote! { struct F<U, T> where T: ::core::clone::Clone { y: Rel } }),
        bounded
    );
    // The generics of a call without any are the context of the types which use none.
    assert_eq!(call(parse_quote! { struct G { y: Rel } }), id(0));
    assert_eq!(call(parse_quote! { struct H { } }), id(0));
    // The names matter.
    assert_ne!(call(parse_quote! { struct I<T, V> { y: Rel } }), bounded);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(leaked(&referrer), ["Rel", "RelB"]);
    assert_eq!(ids(&referrer), [0, 1, 2, 3, 4]);
}

/// The calls of `a_failed_call_takes_the_ids_it_would_if_its_causes_were_accepted()`: the ids they
/// return and the result.
fn failed_calls() -> (Vec<ContextId>, LeakerResult) {
    let mut leaker = Leaker::new();
    let a: ItemStruct = parse_quote! { struct A { x: RelA } };
    let mut ids = vec![leaker.intern_struct(&a)];
    // `Wrap<U>` of the failed call takes the id 1 (and the generics of the call as well).
    let bad: ItemStruct = parse_quote! { struct B<U> { y: Wrap<U>, z: RelB<'_> } };
    ids.push(leaker.intern_struct(&bad));
    // A scope which completes inside a failed call has taken its id by the time the call fails
    // (the ids are assigned when a nested scope is left), and keeps it.
    let (context, ()) = leaker.enter_generics(&parse_quote!(<U>), |view| {
        view.enter_generics(&parse_quote!(<W>), |view| {
            view.visit_type(&parse_quote!(Wrap<W>))
        });
        view.visit_type(&parse_quote!(RelB<'_>));
    });
    ids.push(context);
    let c: ItemStruct = parse_quote! { struct C<V> { y: Wrap<V> } };
    ids.push(leaker.intern_struct(&c));
    (ids, leaker.result())
}

#[test]
fn a_failed_call_takes_the_ids_it_would_if_its_causes_were_accepted() {
    let (ids, result) = failed_calls();
    // Both failed calls report their error, in the order of the calls.
    let cause = Cause::Type(parse_quote!(RelB<'_>));
    assert_eq!(result.errors().len(), 2);
    assert!(result.errors().iter().all(|error| error.cause == cause));
    let referrer = result.clone().into_referrer();
    // Accepting the causes drops the errors, and neither the ids nor the Referrer change.
    let accepted = result.clone().accept_cause(&cause);
    assert!(accepted.errors().is_empty());
    let accepted_referrer = accepted.finish().unwrap();
    assert_eq!(referrer, accepted_referrer);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        accepted_referrer.to_token_stream().to_string()
    );
    // `A` takes the id 0, `B<U>` the id 1, the generics `<U>` of the scope are the same context,
    // and `<V>` of the call after them takes the id 3 (the generics `<W>` of the scope inside it
    // took the id 2, which stays).
    assert_eq!(ids, [0, 1, 1, 3].iter().map(|&n| id(n)).collect::<Vec<_>>());
    assert_eq!(
        leaked(&referrer),
        ["RelA", "Wrap < U >", "Wrap < W >", "Wrap < V >"]
    );
    assert_eq!(common::entry_id(&referrer, 3), ids[3]);
    assert_eq!(s(&common::entry(&referrer, 3).generics), "< V >");
}

#[test]
fn the_intern_methods_return_the_top_level_context() {
    fn check(leaker: Leaker, top_level: ContextId, expected: &[&str]) {
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(leaked(&referrer), expected);
        assert!(common::context_ids(&referrer)
            .into_iter()
            .any(|context| context == top_level));
        let used: BTreeSet<ContextId> = (0..expected.len())
            .map(|ix| common::entry_id(&referrer, ix))
            .collect();
        assert!(used.len() <= common::context_ids(&referrer).len());
    }
    let item: ItemStruct = parse_quote! { struct S<T> { a: Rel, b: Wrap<T> } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemEnum = parse_quote! { enum E<T> { A(Rel), B(Wrap<T>) } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_enum(&item);
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemUnion = parse_quote! { union U<T: ::core::marker::Copy> { a: Rel, b: Wrap<T> } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_union(&item);
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemTrait =
        parse_quote! { trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self) -> Rel; } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&item);
    assert_eq!(top_level, id(0));
    check(leaker, top_level, &["Wrap < T >", "Rel"]);

    let item: ItemMod = parse_quote! {
        mod m { pub struct S<T> { a: Wrap<T> } pub mod inner { pub struct C { a: Rel } } }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_mod(&item);
    // The module is numbered when it is entered, and `inner` when it is found, before the types.
    assert_eq!(top_level, id(0));
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(leaked(&referrer), ["Wrap < T >", "Rel"]);
    assert_eq!(ids(&referrer), [0, 1, 2, 3]);
    let mut again = Leaker::new();
    assert_eq!(again.intern_mod(&item), top_level);
    check(again, top_level, &["Wrap < T >", "Rel"]);

    // A call with an error returns its id as well, and the error is in the result.
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&parse_quote! { struct S { a: RelB<'_> } });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert!(common::context_ids(&result.into_referrer()).contains(&top_level));
}

#[test]
fn ids_of_dropped_contexts_are_gaps() {
    let (referrer, top_level) = gap_referrer();
    // The context 1 is that of the tuple, which is recorded, but not kept by `reduce_roots()`:
    // `Rel1` is interned instead, and the tuple is rebuilt around it. The generics of the struct
    // are the context 3, which has no entry.
    assert_eq!(top_level, id(3));
    assert_eq!(leaked(&referrer), ["Rel0", "Rel1", "Wrap < U >"]);
    assert_eq!(ids(&referrer), [0, 2, 3]);
    assert_eq!(
        (0..3)
            .map(|ix| common::entry_id(&referrer, ix).index())
            .collect::<Vec<_>>(),
        [0, 0, 2]
    );
    // The ids are those of the Leaker, not renumbered.
    assert_eq!(s(&common::entry(&referrer, 2).generics), "< U >");
    for (_, _, context) in referrer.iter() {
        assert_eq!(context.self_ty.as_ref(), None);
        assert_eq!(common::in_module(context), None);
    }
    assert_eq!(s(&common::entry(&referrer, 0).generics), "");

    // No type is left, but the generics of the call are in the Referrer.
    let item: ItemStruct = parse_quote! { struct S<T> { a: T } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    assert_eq!(top_level, id(0));
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.is_empty());
    assert_eq!(ids(&referrer), [0]);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("\"{}\" < T > ; ()", version())
    );
}

// The lookups of a context that document `# Panics` panic with a message that names the id: the
// one of a dropped context (`gap_referrer()` has no context 1), and one beyond the last.

#[test]
fn the_lookups_of_a_context_it_does_not_have_panic() {
    fn leak(f: impl FnOnce(&mut Leaker)) -> Referrer {
        let mut leaker = Leaker::new();
        f(&mut leaker);
        leaker.result().finish().unwrap()
    }
    let shapes: Vec<(&str, Referrer)> = vec![
        ("gaps", gap_referrer().0),
        // The context 0 is dropped: the ids start at 1.
        (
            "the first id dropped",
            leak(|leaker| {
                leaker.intern_struct(&parse_quote! { struct S<T, U> { a: (Rel, T) } });
            }),
        ),
        (
            "one context",
            leak(|leaker| {
                leaker.intern_struct(&parse_quote! { struct S(Rel); });
            }),
        ),
        (
            "modules",
            leak(|leaker| {
                leaker.intern_mod(&parse_quote! { mod m { pub struct A { y: Rel } } });
            }),
        ),
    ];
    type Lookup = fn(&Referrer, ContextId);
    let lookups: [(&str, Lookup); 2] = [
        ("visitor", |referrer, context| {
            let _ = referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>));
        }),
        ("expand", |referrer, context| {
            let _ = common::expand(
                referrer,
                context,
                parse_quote!(Rel),
                |_, ix, _| parse_quote!(P<#ix>),
            );
        }),
    ];
    for (shape, referrer) in &shapes {
        // Every id below the largest that the Referrer does not have, the first one beyond it,
        // and one far away.
        let last = ids(referrer).into_iter().max().unwrap();
        let mut unknown: Vec<usize> = (0..last)
            .filter(|n| {
                !common::context_ids(referrer)
                    .into_iter()
                    .any(|known| known == id(*n))
            })
            .collect();
        unknown.extend([last + 1, 99]);
        for n in unknown {
            for (lookup, call) in &lookups {
                assert_eq!(
                    panic_message(|| call(referrer, id(n))),
                    format!("the context {} is not one of this Referrer", n),
                    "{} of {} with the context {}",
                    lookup,
                    shape,
                    n
                );
            }
        }
    }
    // The first two have the ids that the tests are about.
    assert_eq!(ids(&shapes[0].1), [0, 2, 3]);
    assert_eq!(ids(&shapes[1].1), [1, 2]);
}

#[test]
fn the_contexts_that_exist_do_not_panic() {
    // The control of the tests above: the same calls on a context of the Referrer.
    let (referrer, _) = gap_referrer();
    for context in common::context_ids(&referrer) {
        let _ = referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>));
        let _ = common::expand(
            &referrer,
            context,
            parse_quote!(Rel0),
            |_, ix, _| parse_quote!(P<#ix>),
        );
    }
    for (_, _, context) in referrer.iter() {
        assert_eq!(context.self_ty.as_ref(), None);
        assert!(context.self_outlives.as_slice().is_empty());
        assert_eq!(common::in_module(context), None);
    }
    // `Wrap<U>` uses `<U>`: in `Tr<A, B>` that is `B`.
    let params: Generics = parse_quote!(<T, U>);
    let args: Vec<GenericArgument> = vec![parse_quote!(A), parse_quote!(B)];
    let args = common::generic_args(
        &common::entry(&referrer, 2).generics,
        common::answers(&params.params, &args),
    )
    .unwrap();
    assert_eq!(args.iter().map(s).collect::<Vec<_>>(), ["B"]);
}

#[test]
fn lookups_work_with_gaps() {
    let (referrer, top_level) = gap_referrer();
    let rewritten = common::expand(
        &referrer,
        top_level,
        parse_quote!((Rel1, T)),
        |_, ix, _| parse_quote!(P<#ix>),
    );
    let expected: Type = parse_quote!((P<1usize>, T));
    assert_eq!(s(&rewritten), s(&expected));
    let rewritten = common::expand(
        &referrer,
        top_level,
        parse_quote!(Wrap<U>),
        |_, ix, _| parse_quote!(P<#ix>),
    );
    let expected: Type = parse_quote!(P<2usize>);
    assert_eq!(s(&rewritten), s(&expected));
    assert_eq!(
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .map(|(ty, generics)| (s(ty), s(generics)))
            .collect::<Vec<_>>(),
        [
            ("Rel0".to_string(), "".to_string()),
            ("Rel1".to_string(), "".to_string()),
            ("Wrap < U >".to_string(), "< U >".to_string()),
        ]
    );

    // The item itself is rewritten, where `U` is declared as in the entry.
    let mut item: ItemStruct =
        parse_quote! { struct S<T, U> { a: Rel0, b: (Rel1, T), c: Wrap<U> } };
    referrer
        .visitor(top_level, |_, ix, _| parse_quote!(P<#ix>))
        .visit_item_struct_mut(&mut item);
    let expected: ItemStruct =
        parse_quote! { struct S<T, U> { a: P<0usize>, b: (P<1usize>, T), c: P<2usize> } };
    assert_eq!(s(&item), s(&expected));

    // An item declared in a block shadows the entries mentioning it, wherever their context is.
    let mut item: ItemFn = parse_quote! {
        fn f<U>() { struct Wrap<X>(X); let _a: Wrap<U>; let _b: Rel0; }
    };
    referrer
        .visitor(top_level, |_, ix, _| parse_quote!(P<#ix>))
        .visit_item_fn_mut(&mut item);
    let expected: ItemFn = parse_quote! {
        fn f<U>() { struct Wrap<X>(X); let _a: Wrap<U>; let _b: P<0usize>; }
    };
    assert_eq!(s(&item), s(&expected));
}

#[test]
fn serialization_of_the_simplest_referrer_is_unchanged() {
    // One context, with the id 0: no number of groups, no id. A struct without generics: the
    // context of the types of no generics is its own.
    let item: ItemStruct = parse_quote! { struct S { a: A, b: B } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    assert_eq!(top_level, id(0));
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0), top_level);
    let tokens = referrer.to_token_stream();
    assert_eq!(tokens.to_string(), format!("\"{}\" (A , B)", version()));
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);

    // Contexts with consecutive ids from 0 carry no id either (the last one is that of the
    // generics of the struct, which has no entry).
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: A, b: Wrap<T>, c: Wrap<U> } };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    let tokens = referrer.to_token_stream();
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 4 (A) < T > ; (Wrap < T >) < U > ; (Wrap < U >) < T , U > ; ()",
            version()
        )
    );
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);

    // Neither has a Referrer without any call: there is the context of the types which use no
    // generics.
    let referrer = Leaker::new().result().finish().unwrap();
    assert_eq!(ids(&referrer), [0]);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("\"{}\" ()", version())
    );
}

#[test]
fn serialization_with_gaps() {
    // An id is written where it is not the expected one: after the previous id. The last group
    // is the generics of the struct, which have no entry.
    let (referrer, _) = gap_referrer();
    let tokens = referrer.to_token_stream();
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 3 (Rel0 , Rel1) # 2 < U > ; (Wrap < U >) < T , U > ; ()",
            version()
        )
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(ids(&parsed), [0, 2, 3]);
    assert_eq!(common::entry_id(&parsed, 2), id(2));
    assert_eq!(s(&common::entry(&parsed, 2).generics), "< U >");

    // The first group may not have the id 0 (the first id is not 0 if the context 0 is dropped),
    // and the ids after a gap are expected to follow it.
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: (Rel, T) } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    assert_eq!(top_level, id(2));
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(leaked(&referrer), ["Rel"]);
    assert_eq!(ids(&referrer), [1, 2]);
    assert_eq!(common::entry_id(&referrer, 0), id(1));
    let tokens = referrer.to_token_stream();
    assert_eq!(
        tokens.to_string(),
        format!("\"{}\" 2 # 1 (Rel) < T , U > ; ()", version())
    );
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
    // The context of the tuple is that of the generics of the struct, which is kept without an
    // entry in it.
    let item: ItemStruct = parse_quote! { struct S<T> { a: (Rel, T) } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(top_level, id(0));
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("\"{}\" 2 < T > ; () (Rel)", version())
    );

    let text = format!("\"{}\" 4 (A) # 2 (B) (C) # 7 (D)", version());
    let referrer: Referrer = parse_str(&text).unwrap();
    assert_eq!(ids(&referrer), [0, 2, 3, 7]);
    assert_eq!(
        (0..4)
            .map(|ix| common::entry_id(&referrer, ix).index())
            .collect::<Vec<_>>(),
        [0, 2, 3, 7]
    );
    assert_eq!(referrer.to_token_stream().to_string(), text);
    assert_eq!(
        parse2::<Referrer>(referrer.to_token_stream()).unwrap(),
        referrer
    );

    // The ids are written the same however they are written (the integer may have a suffix).
    let explicit = format!("\"{}\" 2 # 0usize (A) # 1 (B)", version());
    let implicit = format!("\"{}\" 2 (A) (B)", version());
    let parsed: Referrer = parse_str(&explicit).unwrap();
    assert_eq!(parsed, parse_str::<Referrer>(&implicit).unwrap());
    assert_eq!(parsed.to_token_stream().to_string(), implicit);

    // (It can be followed by other tokens, see `tests/referrer.rs`.)
    let one: Referrer = parse_str(&format!("\"{}\" # 3 (A)", version())).unwrap();
    assert_eq!(ids(&one), [3]);
}

#[test]
fn serialization_with_gaps_keeps_the_requirements() {
    // The generics, `Self` and the module of a group are those of its id, whichever it is.
    let text = format!(
        "\"{}\" 6 mod {{ path = [] }} ; () mod 0 ; (Rel) # 4 < T > ; Self : Tr < T > ; \
         (Wrap < T , Self >) # 6 Self = W ; (Rel2 < Self >) \
         # 8 mod {{ path = [a , b] , parent = 0 }} ; () mod 8 ; (Rel3)",
        version()
    );
    let result: LeakerResult = parse_str(&text).unwrap();
    let referrer = result.clone().into_referrer();
    assert_eq!(ids(&referrer), [0, 1, 4, 6, 8, 9]);
    assert_eq!(referrer.to_token_stream().to_string(), text);
    assert_eq!(
        parse2::<Referrer>(referrer.to_token_stream()).unwrap(),
        referrer
    );
    let [c0, c1, c4, c8, c9] = [id(0), id(1), id(4), id(8), id(9)];
    assert_eq!(common::entry_id(&referrer, 0), c1);
    assert_eq!(common::entry_id(&referrer, 1), c4);
    assert_eq!(common::entry_id(&referrer, 3), c9);
    // The entries 0 to 3 are in the contexts 1, 4, 6 and 9.
    assert_eq!(common::in_module(common::entry(&referrer, 0)), Some(c0));
    assert_eq!(common::entry_module_path(&result, 0), Some(vec![]));
    assert_eq!(common::in_module(common::entry(&referrer, 1)), None);
    assert_eq!(s(&common::entry(&referrer, 1).generics), "< T >");
    assert_eq!(s(&common::entry(&referrer, 2).generics), "");
    assert!(matches!(
        common::entry(&referrer, 1).self_ty.as_ref(),
        Some(SelfRequirement::Bound(_))
    ));
    assert!(matches!(
        common::entry(&referrer, 2).self_ty.as_ref(),
        Some(SelfRequirement::Type(_))
    ));
    assert_eq!(common::entry(&referrer, 3).self_ty.as_ref(), None);
    assert_eq!(common::in_module(common::entry(&referrer, 3)), Some(c8));
    assert_eq!(
        common::entry_module_path(&result, 3),
        Some(vec![parse_quote!(a), parse_quote!(b)])
    );
    // The module of an entry is a module's own context, which has the id of the module.
    assert!(matches!(
        common::entry_module(&result, 3),
        Some(ModuleScope::Itself { id, parent: Some(parent), .. }) if id == c8 && parent == c0
    ));

    // The requirements survive the removal of a context (here, one of a trait).
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U> { fn f(&self) -> Rel<Self>; fn g(&self, t: T) -> (RelB, T); }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    // `Rel<Self>` uses `Self`, `T` and `U` (the context 0), the tuple uses `T` (the context 1),
    // and `RelB` nothing (2). The top-level context has the generics of the trait and, as an
    // entry names `Self`, what `Self` is: it is the context 0.
    assert_eq!(top_level, id(0));
    assert_eq!(ids(&referrer), [0, 2]);
    let (with_self, without) = (
        referrer
            .iter()
            .position(|(_, ty, _)| s(ty) == "Rel < Self >"),
        referrer.iter().position(|(_, ty, _)| s(ty) == "RelB"),
    );
    let (with_self, without) = (with_self.unwrap(), without.unwrap());
    assert_eq!(common::entry_id(&referrer, with_self), top_level);
    assert_eq!(common::entry_id(&referrer, without), id(2));
    assert!(matches!(
        common::entry(&referrer, with_self).self_ty.as_ref(),
        Some(SelfRequirement::Bound(_))
    ));
    assert_eq!(common::entry(&referrer, without).self_ty.as_ref(), None);
    assert_eq!(
        s(&common::entry(&referrer, with_self).generics),
        "< T , U >"
    );
    let parsed: Referrer = parse2(referrer.to_token_stream()).unwrap();
    assert_eq!(parsed, referrer);
    for ((_, _, context), (_, _, read)) in referrer.iter().zip(parsed.iter()) {
        assert_eq!(read.self_ty.as_ref(), context.self_ty.as_ref());
    }
}

#[test]
fn the_top_level_generics_do_not_have_the_bounds_the_entries_imply() {
    // `&'a Rel<T>` implies `T: 'a`, which the trait does not declare: the generics of its entry
    // have the bound, and those of the top-level context do not, so a Repeater encoding the
    // generics of the trait for every entry does not carry it (the implementation of such an
    // entry is rejected with E0309), whereas the generics of the entry do. (A path with a
    // lifetime argument, `Rel<'a, T>`, implies none: it is not assumed to.)
    // `All` is an entry of the generics of the trait: its context is the top-level one.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T, const N: usize> {
            fn f(&self, x: &'a Rel<T>) -> Wrap<N>;
            fn g(&self) -> All<'a, T, N>;
        }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let ix = common::types(&referrer)
        .position(|ty| s(ty) == "& 'a Rel < T >")
        .unwrap();
    let all = common::types(&referrer)
        .position(|ty| s(ty) == "All < 'a , T , N >")
        .unwrap();
    assert_ne!(common::entry_id(&referrer, ix), top_level);
    assert_eq!(common::entry_id(&referrer, all), top_level);
    assert_eq!(s(&common::entry(&referrer, ix).generics), "< 'a , T : 'a >");
    assert_eq!(
        s(&common::entry(&referrer, all).generics),
        "< 'a , T , const N : usize >"
    );
    let marker: Type = parse_quote!(crate::Marker);
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Marker(&marker),
            array_const,
        )
        .unwrap()
        .remove(ix);
    assert_eq!(s(&r.generics), "< 'a , T >");
}

/// What `common::context_ids()` and `common::entry_id()` read from the serialized form agrees with
/// the Leaker and with the Referrer: the ids are listed once and in order, those the Leaker
/// returned are among them, `visitor()` takes each listed id and no other up to past the last, and
/// the id of an entry is listed. `interned`: the contexts come from a Leaker, so two entries have
/// the same id exactly when their contexts are equal.
fn check_serialized_ids(shape: &str, referrer: &Referrer, returned: &[ContextId], interned: bool) {
    let ids = common::context_ids(referrer);
    assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "{}", shape);
    for context in returned {
        assert!(
            ids.contains(context),
            "{}: {} is not listed",
            shape,
            context
        );
    }
    let last = ids.last().unwrap().index();
    for n in 0..=last + 2 {
        let readable = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = referrer.visitor(id(n), |_, ix, _| parse_quote!(P<#ix>));
        }))
        .is_ok();
        assert_eq!(ids.contains(&id(n)), readable, "{}: the id {}", shape, n);
    }
    let read = common::serialized_ids(referrer);
    assert_eq!(read.entries.len(), referrer.iter().count(), "{}", shape);
    for ix in 0..referrer.iter().count() {
        let entry = common::entry_id(referrer, ix);
        assert!(ids.contains(&entry), "{}: the entry {}", shape, ix);
    }
    if interned {
        let count = referrer.iter().count();
        for first in 0..count {
            for second in first + 1..count {
                assert_eq!(
                    common::entry_id(referrer, first) == common::entry_id(referrer, second),
                    common::entry(referrer, first) == common::entry(referrer, second),
                    "{}: the entries {} and {}",
                    shape,
                    first,
                    second
                );
            }
        }
    }
}

#[test]
fn the_ids_read_from_the_serialized_form_agree_with_the_referrer() {
    let from_text =
        |text: &str| -> Referrer { parse_str(&format!("\"{}\" {}", version(), text)).unwrap() };
    let entries_of = |referrer: &Referrer| -> Vec<usize> {
        (0..referrer.iter().count())
            .map(|ix| common::entry_id(referrer, ix).index())
            .collect()
    };

    // Gaps: the context 1 is dropped, and an id is written where it is not the one after the
    // group before.
    let (gaps, top_level) = gap_referrer();
    check_serialized_ids("gaps", &gaps, &[top_level], true);
    assert_eq!(ids(&gaps), [0, 2, 3]);
    assert_eq!(entries_of(&gaps), [0, 0, 2]);
    // The first id is not 0.
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&parse_quote! { struct S<T, U> { a: (Rel, T) } });
    let first_dropped = leaker.result().finish().unwrap();
    check_serialized_ids("first dropped", &first_dropped, &[top_level], true);
    assert_eq!(ids(&first_dropped), [1, 2]);
    assert_eq!(entries_of(&first_dropped), [1]);
    // The ids of a Referrer are not limited by their number.
    let far = from_text("2 # 70 (A) # 1000 (B)");
    assert_eq!(ids(&far), [70, 1000]);
    assert_eq!(entries_of(&far), [70, 1000]);

    // Interleaved groups: the required entries come first, then the optional ones, so the context
    // 0 is written again (`# 0 (..)`) after the contexts 1 and 2. The last group is the top-level
    // context, which has no entry.
    let item: ItemStruct = parse_quote! {
        struct S<T, U> { a: ::ext::X, b: Wrap<T>, c: ::ext::Z, d: Rel, e: Wrap2<U> }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item);
    let interleaved = leaker.result().finish().unwrap();
    assert_eq!(
        interleaved.to_token_stream().to_string(),
        format!(
            "\"{}\" 5 (Rel) < T > ; (Wrap < T >) < U > ; (Wrap2 < U >) \
             # 0 (:: ext :: X , :: ext :: Z) # 3 < T , U > ; ()",
            version()
        )
    );
    check_serialized_ids("interleaved", &interleaved, &[top_level], true);
    assert_eq!(top_level, id(3));
    assert_eq!(ids(&interleaved), [0, 1, 2, 3]);
    assert_eq!(entries_of(&interleaved), [0, 1, 2, 0, 0]);
    // The same written by hand. A group has no `#` where its id follows that of the group before,
    // also if the context is declared already (the last group of the second).
    let text = from_text("4 (A) (B) # 0 (C) # 3 (D)");
    check_serialized_ids("text", &text, &[], false);
    assert_eq!(ids(&text), [0, 1, 3]);
    assert_eq!(entries_of(&text), [0, 1, 0, 3]);
    let text = from_text("4 (A) (B) # 0 (C) (D)");
    assert_eq!(ids(&text), [0, 1]);
    assert_eq!(entries_of(&text), [0, 1, 0, 1]);
    // Contexts with no entry, between the others and at the end.
    let text = from_text("4 (A) () # 2 (B) ()");
    assert_eq!(ids(&text), [0, 1, 2, 3]);
    assert_eq!(entries_of(&text), [0, 2]);

    // Modules: the own context of a module has no entry and declares its path and parent in braces
    // (a hint too), and the contexts of the entries in it say `mod 1 ;`.
    let module: ItemMod = parse_quote! {
        mod m {
            pub struct A { y: Rel }
            pub mod inner { pub struct C { y: Rel, z: Wrap<u8> } }
        }
    };
    let mut leaker = Leaker::new();
    let root = leaker.intern_mod(&module);
    let (hinted, ()) = leaker.enter_mod("h", [], |view| view.visit_type(&parse_quote!(Rel2)));
    let result = leaker.result();
    let modules = result.clone().finish().unwrap();
    let text = modules.to_token_stream().to_string();
    assert!(text.contains("mod { path = [] }"), "{}", text);
    assert!(text.contains("mod { hint = \"h\" }"), "{}", text);
    assert!(text.contains("parent = "), "{}", text);
    check_serialized_ids("modules", &modules, &[root, hinted], true);
    // Each of the three modules has an entry: the module of an entry is the module's own context,
    // and is listed.
    let listed = common::context_ids(&modules);
    let mut own: BTreeSet<ContextId> = BTreeSet::new();
    for ix in 0..modules.iter().count() {
        match common::entry_module(&result, ix) {
            Some(ModuleScope::Itself { id, .. }) => {
                assert_eq!(common::in_module(common::entry(&modules, ix)), Some(id));
                assert!(listed.contains(&id), "the entry {}", ix);
                own.insert(id);
            }
            other => panic!("the module of the entry {}: {:?}", ix, other),
        }
    }
    assert_eq!(own.len(), 3, "{:?}", own);
    assert!(own.contains(&root) && own.contains(&hinted));

    // The items of a declaration end at the `;` of the top level, whatever their groups hold: the
    // parentheses of `Fn(u8)`, the brackets of `[T; 3]`, `Self`, its outlives, and a GAT.
    let items: Vec<(&str, Item)> = vec![
        (
            "fn bound",
            parse_quote! { trait Tr<T: ::core::ops::Fn(u8) -> u8> { fn f(&self, t: T) -> Rel<T>; } },
        ),
        (
            "where, Self and GAT",
            parse_quote! {
                trait Tr<'a, T> where Self: 'a, [T; 3]: ::core::marker::Sized {
                    type G<'b> where Self: 'b;
                    fn f<'x>(&'x self, t: T) -> Rel<&'x Self, T, Self::G<'x>>;
                }
            },
        ),
        (
            "Self is a type",
            parse_quote! { struct S<T> { a: Rel<Self, T>, b: Wrap<(T, u8)> } },
        ),
        (
            "Self is any type",
            parse_quote! { mod m { struct N<T: ::core::convert::AsRef<Self>>(Rel<T>); } },
        ),
    ];
    for (shape, item) in items {
        let mut leaker = Leaker::new();
        let top_level = match &item {
            Item::Trait(item) => leaker.intern_trait(item),
            Item::Struct(item) => leaker.intern_struct(item),
            Item::Mod(item) => leaker.intern_mod(item),
            _ => unreachable!(),
        };
        let referrer = leaker.result().finish().unwrap();
        assert!(referrer.iter().count() > 0, "{}", shape);
        let text = referrer.to_token_stream().to_string();
        match shape {
            "fn bound" => assert!(text.contains("Fn (u8)"), "{}", text),
            "where, Self and GAT" => assert!(text.contains("type G"), "{}", text),
            "Self is a type" => assert!(text.contains("Self = S < T >"), "{}", text),
            _ => assert!(text.contains("Self ;"), "{}", text),
        }
        check_serialized_ids(shape, &referrer, &[top_level], true);
    }

    // The parts (`[..]` after the tag), no group count, and no context but the first.
    let parts = from_text("[type (R) 0 in (& R) 0] (Rel)");
    check_serialized_ids("parts", &parts, &[], false);
    assert_eq!(ids(&parts), [0]);
    assert_eq!(entries_of(&parts), [0]);
    let empty = Leaker::new().result().into_referrer();
    check_serialized_ids("empty", &empty, &[], false);
    assert_eq!(ids(&empty), [0]);
}

/// The reader fails for a shape it does not know, and for tokens after the groups.
#[test]
fn the_reader_of_the_serialized_form_does_not_guess() {
    let read = |text: &str| common::read_serialized_ids(parse_str(text).unwrap());
    let tag = version();
    // What it reads.
    let ok = read(&format!("\"{}\" 3 (A) < T > ; (B) # 0 (C)", tag)).unwrap();
    assert_eq!(ok.declared, [id(0), id(1)]);
    assert_eq!(ok.entries, [id(0), id(1), id(0)]);
    // What it does not: another tag, none, an item out of its place or twice, no `;`, no entries,
    // the entries of an undeclared context after a declared one, and trailing tokens.
    for text in [
        "\"type-leak-format-1\" (A)".to_string(),
        "(A)".to_string(),
        format!("\"{}\" Self ; < T > ; (A)", tag),
        format!("\"{}\" < T > ; < U > ; (A)", tag),
        format!("\"{}\" mod 1 ; Self ; (A)", tag),
        format!("\"{}\" < T > (A)", tag),
        format!("\"{}\" < T > ;", tag),
        format!("\"{}\" 2 (A) # 0 < T > ; (B)", tag),
        format!("\"{}\" (A) (B)", tag),
        format!("\"{}\" 2 (A) # x (B)", tag),
        format!("\"{}\" (A, , B)", tag),
        format!("\"{}\" ({{}})", tag),
    ] {
        assert!(read(&text).is_err(), "{}", text);
    }
}

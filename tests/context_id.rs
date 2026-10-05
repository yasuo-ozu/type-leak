//! Tests for `ContextId`: the ids of the contexts are assigned by the Leaker when the requirements
//! first appear, the top-level context of a call (the id of its generics) is returned by
//! `Leaker::intern_with()` and its relatives, and the ids are not renumbered by
//! `Leaker::finish()`, so the Referrer may have gaps.

use proc_macro2::{TokenStream, TokenTree};
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{ContextId, Leaker, Referrer, SelfRequirement};

fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn id(n: usize) -> ContextId {
    parse_str(&n.to_string()).unwrap()
}

fn ids(referrer: &Referrer) -> Vec<usize> {
    referrer.contexts().map(ContextId::index).collect()
}

fn s(x: &impl ToTokens) -> String {
    quote!(#x).to_string()
}

fn leaked(referrer: &Referrer) -> Vec<String> {
    referrer.iter().map(s).collect()
}

/// The Referrer of `struct S<T, U> { a: Rel0, b: (Rel1, T), c: Wrap<U> }`, together with the
/// top-level context returned by the Leaker. `Rel0` and `Rel1` use no generics (the context 0),
/// the tuple uses `T` (the context 1) and `Wrap<U>` uses `U` (the context 2), and the generics of
/// the struct `<T, U>` are the context 3. Only `Rel1` is interned for the tuple, so no type of
/// the context 1 is kept, and it is not top-level either.
fn gap_referrer() -> (Referrer, ContextId) {
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: Rel0, b: (Rel1, T), c: Wrap<U> } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    (leaker.finish(), top_level)
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
    assert!(parse_str::<ContextId>("3u8").is_err());
    assert!(parse_str::<ContextId>("-3").is_err());
    assert!(parse_str::<ContextId>("three").is_err());
    assert!(parse_str::<ContextId>("3 4").is_err());
    assert!(parse_str::<ContextId>("").is_err());
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
fn intern_with_returns_the_top_level_context() {
    // The generics of the trait `<T>` are not those `Rel` uses (none), so the top-level context
    // has no entry, but it is in the Referrer, with those generics.
    let tr: ItemTrait = parse_quote! { trait Tr<T> { fn f(&self) -> Rel; } };
    let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    assert_eq!(top_level, id(1));
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["Rel"]);
    assert_eq!(referrer.context(0), id(0));
    assert_eq!(ids(&referrer), [0, 1]);
    assert_eq!(s(referrer.context_generics(top_level)), "< T >");
    assert_eq!(referrer.context_self(top_level), None);
    assert_eq!(referrer.context_module(top_level), None);
    assert_eq!(s(referrer.context_generics(id(0))), "");
    // It is written as an empty group, and kept by the round trip.
    let tokens = referrer.to_token_stream();
    assert_eq!(
        tokens.to_string(),
        format!("\"{}\" 2 (Rel) < T > ; ()", version())
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(s(parsed.context_generics(top_level)), "< T >");
    assert_eq!(parsed.context_self(top_level), None);
    assert_eq!(ids(&parsed), [0, 1]);

    // The generics are in the canonical form of the requirements (the bounds are moved to the
    // parameters, the parameters are sorted, and the defaults are removed), and the `Self` of a
    // predicate is not a requirement of any type.
    let tr: ItemTrait = parse_quote! {
        trait Tr<U: ::core::fmt::Debug, T = u8> where T: ::core::clone::Clone, Self: ::core::marker::Sized {
            fn f(&self) -> Rel;
        }
    };
    let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    let referrer = leaker.finish();
    let generics = referrer.context_generics(top_level);
    let where_clause = &generics.where_clause;
    assert_eq!(
        s(&quote!(#generics #where_clause)),
        "< T : :: core :: clone :: Clone , U : :: core :: fmt :: Debug > \
         where Self : :: core :: marker :: Sized"
    );
    assert_eq!(referrer.context_self(top_level), None);
    assert!(!referrer.has_self());
}

#[test]
fn top_level_context_is_shared_with_the_entries() {
    // The generics of `Wrap<T>` are those of the struct.
    let item: ItemStruct = parse_quote! { struct S<T> { a: Wrap<T> } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    assert_eq!(top_level, id(0));
    let referrer = leaker.finish();
    assert_eq!(referrer.context(0), top_level);
    assert_eq!(ids(&referrer), [0]);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("\"{}\" < T > ; (Wrap < T >)", version())
    );

    // With the bounds, whichever way they are written, and an entry of no generics aside.
    let item: ItemStruct =
        parse_quote! { struct S<T> where T: ::core::clone::Clone { a: Rel, b: Wrap<T> } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["Rel", "Wrap < T >"]);
    assert_eq!(top_level, id(1));
    assert_eq!(referrer.context(1), top_level);
    assert_eq!(ids(&referrer), [0, 1]);

    // A struct without generics: the context of the types of no generics is its own, and the
    // Referrer is the simplest one.
    let item: ItemStruct = parse_quote! { struct S { a: A, b: B } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    assert_eq!(top_level, id(0));
    let referrer = leaker.finish();
    assert_eq!(referrer.context(0), top_level);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("\"{}\" (A , B)", version())
    );
}

#[test]
fn calls_with_the_same_generics_return_the_same_id() {
    let mut leaker = Leaker::new();
    let mut call = |item: ItemStruct| {
        leaker
            .intern_with(&item.generics, |v| v.visit_item_struct(&item))
            .unwrap()
    };
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
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["Rel", "RelB"]);
    assert_eq!(ids(&referrer), [0, 1, 2, 3, 4]);
}

#[test]
fn a_failed_call_does_not_take_ids() {
    let mut leaker = Leaker::new();
    let a: ItemStruct = parse_quote! { struct A { x: RelA } };
    assert_eq!(
        leaker
            .intern_with(&a.generics, |v| v.visit_item_struct(&a))
            .unwrap(),
        id(0)
    );
    // `Wrap<U>` of a failed call would take the id 1 (and the generics of the call as well).
    let bad: ItemStruct = parse_quote! { struct B<U> { y: Wrap<U>, z: RelB<'_> } };
    assert!(leaker
        .intern_with(&bad.generics, |v| v.visit_item_struct(&bad))
        .is_err());
    let c: ItemStruct = parse_quote! { struct C<V> { y: Wrap<V> } };
    let top_level = leaker
        .intern_with(&c.generics, |v| v.visit_item_struct(&c))
        .unwrap();
    assert_eq!(top_level, id(1));
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["RelA", "Wrap < V >"]);
    assert_eq!(referrer.context(1), top_level);
    assert_eq!(ids(&referrer), [0, 1]);
    assert_eq!(s(referrer.context_generics(top_level)), "< V >");
}

#[test]
fn from_functions_return_the_top_level_context() {
    fn check(leaker: Leaker, top_level: ContextId, expected: &[&str]) {
        let referrer = leaker.finish();
        assert_eq!(leaked(&referrer), expected);
        assert!(referrer.contexts().any(|context| context == top_level));
        let used: BTreeSet<ContextId> =
            (0..expected.len()).map(|ix| referrer.context(ix)).collect();
        assert!(used.len() <= referrer.contexts().count());
    }
    let item: ItemStruct = parse_quote! { struct S<T> { a: Rel, b: Wrap<T> } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemEnum = parse_quote! { enum E<T> { A(Rel), B(Wrap<T>) } };
    let (leaker, top_level) = Leaker::from_enum(&item).unwrap();
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemUnion = parse_quote! { union U<T: ::core::marker::Copy> { a: Rel, b: Wrap<T> } };
    let (leaker, top_level) = Leaker::from_union(&item).unwrap();
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemTrait =
        parse_quote! { trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self) -> Rel; } };
    let (leaker, top_level) = Leaker::from_trait(&item).unwrap();
    assert_eq!(top_level, id(0));
    check(leaker, top_level, &["Wrap < T >", "Rel"]);

    // The module has no generics, and no module is a requirement of its top-level context.
    let item: ItemMod = parse_quote! {
        mod m { pub struct S<T> { a: Wrap<T> } pub mod inner { pub struct C { a: Rel } } }
    };
    let (leaker, top_level) = Leaker::from_module(&item).unwrap();
    assert_eq!(top_level, id(2));
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["Wrap < T >", "Rel"]);
    assert_eq!(ids(&referrer), [0, 1, 2]);
    assert_eq!(s(referrer.context_generics(top_level)), "");
    assert_eq!(referrer.context_module(top_level), None);
    check(
        Leaker::from_module(&item).unwrap().0,
        top_level,
        &["Wrap < T >", "Rel"],
    );

    // An error returns no id.
    assert!(Leaker::from_struct(&parse_quote! { struct S { a: RelB<'_> } }).is_err());
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
            .map(|ix| referrer.context(ix).index())
            .collect::<Vec<_>>(),
        [0, 0, 2]
    );
    // The ids are those of the Leaker, not renumbered.
    assert_eq!(s(referrer.generics(2)), "< U >");
    for context in referrer.contexts() {
        assert_eq!(referrer.context_self(context), None);
        assert_eq!(referrer.context_module(context), None);
    }
    assert_eq!(s(referrer.context_generics(id(0))), "");
    assert_eq!(s(referrer.context_generics(id(2))), "< U >");
    assert_eq!(s(referrer.context_generics(top_level)), "< T , U >");

    // The first id is not 0 if the context 0 is dropped.
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: (Rel, T) } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    assert_eq!(top_level, id(2));
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["Rel"]);
    assert_eq!(ids(&referrer), [1, 2]);
    assert_eq!(referrer.context(0), id(1));

    // No type is left, but the generics of the call are in the Referrer.
    let item: ItemStruct = parse_quote! { struct S<T> { a: T } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    assert_eq!(top_level, id(0));
    let referrer = leaker.finish();
    assert!(referrer.is_empty());
    assert_eq!(ids(&referrer), [0]);
    assert_eq!(s(referrer.context_generics(top_level)), "< T >");
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("\"{}\" < T > ; ()", version())
    );

    // Without any call, there is the context of the types which use no generics.
    let referrer = Leaker::new().finish();
    assert_eq!(ids(&referrer), [0]);
    assert_eq!(s(referrer.context_generics(id(0))), "");
}

#[test]
#[should_panic]
fn context_of_a_dropped_id_panics() {
    let (referrer, _) = gap_referrer();
    let _ = referrer.context_generics(id(1));
}

#[test]
fn lookups_work_with_gaps() {
    let (referrer, _) = gap_referrer();
    let rewritten = referrer.expand(parse_quote!((Rel1, T)), |_, ix| parse_quote!(P<#ix>));
    let expected: Type = parse_quote!((P<1usize>, T));
    assert_eq!(s(&rewritten), s(&expected));
    let rewritten = referrer.expand(parse_quote!(Wrap<U>), |_, ix| parse_quote!(P<#ix>));
    let expected: Type = parse_quote!(P<2usize>);
    assert_eq!(s(&rewritten), s(&expected));
    assert_eq!(
        referrer
            .iter_with_generics()
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
        .clone()
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .visit_item_struct_mut(&mut item);
    let expected: ItemStruct =
        parse_quote! { struct S<T, U> { a: P<0usize>, b: (P<1usize>, T), c: P<2usize> } };
    assert_eq!(s(&item), s(&expected));

    // An item declared in a block shadows the entries mentioning it, wherever their context is.
    let mut item: ItemFn = parse_quote! {
        fn f<U>() { struct Wrap<X>(X); let _a: Wrap<U>; let _b: Rel0; }
    };
    referrer
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .visit_item_fn_mut(&mut item);
    let expected: ItemFn = parse_quote! {
        fn f<U>() { struct Wrap<X>(X); let _a: Wrap<U>; let _b: P<0usize>; }
    };
    assert_eq!(s(&item), s(&expected));
}

#[test]
fn serialization_of_the_simplest_referrer_is_unchanged() {
    // One context, with the id 0: no number of groups, no id.
    let item: ItemStruct = parse_quote! { struct S { a: A, b: B } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    assert_eq!(top_level, id(0));
    let referrer = leaker.finish();
    let tokens = referrer.to_token_stream();
    assert_eq!(tokens.to_string(), format!("\"{}\" (A , B)", version()));
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);

    // Contexts with consecutive ids from 0 carry no id either (the last one is that of the
    // generics of the struct, which has no entry).
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: A, b: Wrap<T>, c: Wrap<U> } };
    let referrer = Leaker::from_struct(&item).unwrap().0.finish();
    let tokens = referrer.to_token_stream();
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 4 (A) < T > ; (Wrap < T >) < U > ; (Wrap < U >) < T , U > ; ()",
            version()
        )
    );
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);

    // Neither has a Referrer without any context.
    let referrer = Leaker::new().finish();
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("\"{}\" ()", version())
    );
}

#[test]
fn serialization_with_gaps() {
    // An id is written where it is not the expected one: after the previous id. The last group
    // is the generics of the struct, which have no entry.
    let (referrer, top_level) = gap_referrer();
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
    assert_eq!(parsed.context(2), id(2));
    assert_eq!(s(parsed.context_generics(id(2))), "< U >");
    assert_eq!(s(parsed.context_generics(top_level)), "< T , U >");

    // The first group may not have the id 0, and the ids after a gap are expected to follow it.
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: (Rel, T) } };
    let referrer = Leaker::from_struct(&item).unwrap().0.finish();
    let tokens = referrer.to_token_stream();
    assert_eq!(
        tokens.to_string(),
        format!("\"{}\" 2 # 1 (Rel) < T , U > ; ()", version())
    );
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
    // The context of the tuple is that of the generics of the struct, which is kept without an
    // entry in it.
    let item: ItemStruct = parse_quote! { struct S<T> { a: (Rel, T) } };
    let (leaker, top_level) = Leaker::from_struct(&item).unwrap();
    let referrer = leaker.finish();
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
            .map(|ix| referrer.context(ix).index())
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

    // A Referrer with gaps can be followed by other tokens, also one starting with `#`.
    struct Followed(Referrer, TokenStream);
    impl parse::Parse for Followed {
        fn parse(input: parse::ParseStream) -> Result<Self> {
            Ok(Self(input.parse()?, input.parse()?))
        }
    }
    let pound_group: TokenStream = "# 5 (X)".parse().unwrap();
    for rest in [
        quote!(
            struct Z;
        ),
        quote!(
            #[derive(Debug)]
            struct Z;
        ),
        pound_group,
    ] {
        let (referrer, _) = gap_referrer();
        let tokens = referrer.to_token_stream();
        let followed: Followed = parse2(quote!(#tokens #rest)).unwrap();
        assert_eq!(followed.0, referrer);
        assert_eq!(followed.1.to_string(), rest.to_string());
    }
    let one: Referrer = parse_str(&format!("\"{}\" # 3 (A)", version())).unwrap();
    assert_eq!(ids(&one), [3]);
    let tokens = one.to_token_stream();
    let followed: Followed = parse2(quote!(#tokens #[derive(Debug)] struct Z;)).unwrap();
    assert_eq!(followed.0, one);
}

#[test]
fn serialization_with_gaps_keeps_the_requirements() {
    // The generics, `Self` and the module of a group are those of its id, whichever it is.
    let text = format!(
        "\"{}\" 4 mod self ; (Rel) # 4 < T > ; Self : Tr < T > ; (Wrap < T , Self >) \
         # 6 Self = # 0 ; (Rel2 < Self >) # 9 mod a :: b ; (Rel3)",
        version()
    );
    let referrer: Referrer = parse_str(&text).unwrap();
    assert_eq!(ids(&referrer), [0, 4, 6, 9]);
    assert_eq!(referrer.to_token_stream().to_string(), text);
    assert_eq!(
        parse2::<Referrer>(referrer.to_token_stream()).unwrap(),
        referrer
    );
    let [c0, c4, c6, c9] = [id(0), id(4), id(6), id(9)];
    assert_eq!(referrer.context(1), c4);
    assert_eq!(referrer.context(3), c9);
    assert_eq!(referrer.context_module(c0), Some(&[][..]));
    assert_eq!(referrer.context_module(c4), None);
    assert_eq!(s(referrer.context_generics(c4)), "< T >");
    assert_eq!(s(referrer.context_generics(c6)), "");
    assert!(matches!(
        referrer.context_self(c4),
        Some(SelfRequirement::Bound(_))
    ));
    assert!(matches!(
        referrer.context_self(c6),
        Some(SelfRequirement::Call(0))
    ));
    assert_eq!(referrer.context_self(c9), None);
    assert_eq!(
        referrer.context_module(c9),
        Some(&[parse_quote!(a), parse_quote!(b)][..])
    );

    // The requirements survive the removal of a context (here, one of a trait).
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U> { fn f(&self) -> Rel<Self>; fn g(&self, t: T) -> (RelB, T); }
    };
    let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    let referrer = leaker.finish();
    // `Rel<Self>` uses `Self`, `T` and `U` (the context 0), the tuple uses `T` (the context 1),
    // `RelB` nothing (2), and the generics of the trait are the context 3 (without `Self`).
    assert_eq!(top_level, id(3));
    assert_eq!(ids(&referrer), [0, 2, 3]);
    assert!(matches!(
        referrer.context_self(id(0)),
        Some(SelfRequirement::Bound(_))
    ));
    assert_eq!(referrer.context_self(id(2)), None);
    assert_eq!(referrer.context_self(top_level), None);
    assert_eq!(s(referrer.context_generics(top_level)), "< T , U >");
    let parsed: Referrer = parse2(referrer.to_token_stream()).unwrap();
    assert_eq!(parsed, referrer);
    for context in referrer.contexts() {
        assert_eq!(parsed.context_self(context), referrer.context_self(context));
    }
}

#[test]
fn ids_must_ascend() {
    let version = version();
    for text in [
        // The same id twice.
        format!("\"{}\" 2 (A) # 0 (B)", version),
        format!("\"{}\" 2 # 3 (A) # 3 (B)", version),
        // A smaller id.
        format!("\"{}\" 2 # 3 (A) # 1 (B)", version),
        format!("\"{}\" 2 # 3 (A) # 2 (B)", version),
        // Not a number.
        format!("\"{}\" # (A)", version),
        format!("\"{}\" # x (A)", version),
        format!("\"{}\" # 1u8 (A)", version),
        // A group is missing.
        format!("\"{}\" 2 # 3 (A)", version),
    ] {
        assert!(parse_str::<Referrer>(&text).is_err(), "{}", text);
    }
    let message = parse_str::<Referrer>(&format!("\"{}\" 2 # 3 (A) # 1 (B)", version))
        .unwrap_err()
        .to_string();
    assert!(message.contains("ascending"), "{}", message);
    // The largest id cannot be followed by another one.
    let text = format!("\"{}\" # {} (A)", version, usize::MAX);
    assert!(parse_str::<Referrer>(&text).is_err());
}

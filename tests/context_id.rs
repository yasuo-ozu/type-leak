//! Tests for `ContextId`: the ids of the contexts are assigned by the Leaker when the requirements
//! first appear, the top-level context of a call (the id of its generics) is returned by
//! `Leaker::enter_generics()` and its relatives, and the ids are not renumbered by
//! `Leaker::finish()`, so the Referrer may have gaps.

use proc_macro2::{TokenStream, TokenTree};
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{ContextId, Leaker, Referrer, SelfRequirement};

/// The tag of the format of a serialized Referrer, which it starts with.
fn version() -> String {
    format!("type-leak-format-{}", Referrer::FORMAT_VERSION)
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
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true).unwrap();
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
    let top_level = leaker.intern_trait(&tr, true).unwrap();
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
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true).unwrap();
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
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true).unwrap();
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
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["Rel", "Wrap < T >"]);
    assert_eq!(top_level, id(1));
    assert_eq!(referrer.context(1), top_level);
    assert_eq!(ids(&referrer), [0, 1]);

    // A struct without generics: the context of the types of no generics is its own, and the
    // Referrer is the simplest one.
    let item: ItemStruct = parse_quote! { struct S { a: A, b: B } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true).unwrap();
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
    let mut call = |item: ItemStruct| leaker.intern_struct(&item, true).unwrap();
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
    assert_eq!(leaker.intern_struct(&a, true).unwrap(), id(0));
    // `Wrap<U>` of a failed call would take the id 1 (and the generics of the call as well).
    let bad: ItemStruct = parse_quote! { struct B<U> { y: Wrap<U>, z: RelB<'_> } };
    assert!(leaker.intern_struct(&bad, true).is_err());
    let c: ItemStruct = parse_quote! { struct C<V> { y: Wrap<V> } };
    let top_level = leaker.intern_struct(&c, true).unwrap();
    assert_eq!(top_level, id(1));
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["RelA", "Wrap < V >"]);
    assert_eq!(referrer.context(1), top_level);
    assert_eq!(ids(&referrer), [0, 1]);
    assert_eq!(s(referrer.context_generics(top_level)), "< V >");
}

#[test]
fn the_intern_methods_return_the_top_level_context() {
    fn check(leaker: Leaker, top_level: ContextId, expected: &[&str]) {
        let referrer = leaker.finish();
        assert_eq!(leaked(&referrer), expected);
        assert!(referrer.contexts().any(|context| context == top_level));
        let used: BTreeSet<ContextId> =
            (0..expected.len()).map(|ix| referrer.context(ix)).collect();
        assert!(used.len() <= referrer.contexts().count());
    }
    let item: ItemStruct = parse_quote! { struct S<T> { a: Rel, b: Wrap<T> } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true).unwrap();
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemEnum = parse_quote! { enum E<T> { A(Rel), B(Wrap<T>) } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_enum(&item, true).unwrap();
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemUnion = parse_quote! { union U<T: ::core::marker::Copy> { a: Rel, b: Wrap<T> } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_union(&item, true).unwrap();
    assert_eq!(top_level, id(1));
    check(leaker, top_level, &["Rel", "Wrap < T >"]);

    let item: ItemTrait =
        parse_quote! { trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self) -> Rel; } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&item, true).unwrap();
    assert_eq!(top_level, id(0));
    check(leaker, top_level, &["Wrap < T >", "Rel"]);

    // The module has no generics, and no module is a requirement of its top-level context.
    let item: ItemMod = parse_quote! {
        mod m { pub struct S<T> { a: Wrap<T> } pub mod inner { pub struct C { a: Rel } } }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_mod(&item).unwrap();
    // The module is numbered when it is entered, and `inner` when it is found, before the types.
    assert_eq!(top_level, id(0));
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["Wrap < T >", "Rel"]);
    assert_eq!(ids(&referrer), [0, 1, 2, 3]);
    assert_eq!(s(referrer.context_generics(top_level)), "");
    assert_eq!(referrer.context_module(top_level), None);
    let mut again = Leaker::new();
    assert_eq!(again.intern_mod(&item).unwrap(), top_level);
    check(again, top_level, &["Wrap < T >", "Rel"]);

    // An error returns no id.
    assert!(Leaker::new()
        .intern_struct(&parse_quote! { struct S { a: RelB<'_> } }, true)
        .is_err());
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
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true).unwrap();
    assert_eq!(top_level, id(2));
    let referrer = leaker.finish();
    assert_eq!(leaked(&referrer), ["Rel"]);
    assert_eq!(ids(&referrer), [1, 2]);
    assert_eq!(referrer.context(0), id(1));

    // No type is left, but the generics of the call are in the Referrer.
    let item: ItemStruct = parse_quote! { struct S<T> { a: T } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true).unwrap();
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
#[should_panic(expected = "the context 1 is not one of the contexts of this Referrer")]
fn context_of_a_dropped_id_panics() {
    let (referrer, _) = gap_referrer();
    let _ = referrer.context_generics(id(1));
}

// The lookups of a context that document `# Panics` panic with the same message, which names the
// id: the one of a dropped context (`gap_referrer()` has no context 1), and one beyond the last.

#[test]
#[should_panic(expected = "the context 1 is not one of the contexts of this Referrer")]
fn context_self_of_a_dropped_id_panics() {
    let (referrer, _) = gap_referrer();
    let _ = referrer.context_self(id(1));
}

#[test]
#[should_panic(expected = "the context 99 is not one of the contexts of this Referrer")]
fn context_self_outlives_of_an_unknown_id_panics() {
    let (referrer, _) = gap_referrer();
    let _ = referrer.context_self_outlives(id(99));
}

#[test]
#[should_panic(expected = "the context 1 is not one of the contexts of this Referrer")]
fn context_module_of_a_dropped_id_panics() {
    let (referrer, _) = gap_referrer();
    let _ = referrer.context_module(id(1));
}

#[test]
#[should_panic(expected = "the context 99 is not one of the contexts of this Referrer")]
fn impl_generics_of_an_unknown_id_panics() {
    let (referrer, _) = gap_referrer();
    let _ = referrer.impl_generics(id(99));
}

#[test]
#[should_panic(expected = "the context 1 is not one of the contexts of this Referrer")]
fn context_generic_args_of_a_dropped_id_panics() {
    let (referrer, _) = gap_referrer();
    let params: Generics = parse_quote!(<T, U>);
    let _ = referrer.context_generic_args(id(1), &params.params, &[]);
}

#[test]
fn the_contexts_that_exist_do_not_panic() {
    // The control of the tests above: the same calls on a context of the Referrer.
    let (referrer, top_level) = gap_referrer();
    for context in referrer.contexts() {
        assert_eq!(referrer.context_self(context), None);
        assert!(referrer.context_self_outlives(context).is_empty());
        assert_eq!(referrer.context_module(context), None);
    }
    assert_eq!(s(&referrer.impl_generics(top_level)), "< T , U >");
    let params: Generics = parse_quote!(<T, U>);
    let args: Vec<GenericArgument> = vec![parse_quote!(A), parse_quote!(B)];
    let args = referrer
        .context_generic_args(top_level, &params.params, &args)
        .unwrap();
    assert_eq!(args.iter().map(s).collect::<Vec<_>>(), ["A", "B"]);
}

#[test]
fn lookups_work_with_gaps() {
    let (referrer, top_level) = gap_referrer();
    let rewritten = referrer.expand(
        top_level,
        parse_quote!((Rel1, T)),
        |_, ix| parse_quote!(P<#ix>),
    );
    let expected: Type = parse_quote!((P<1usize>, T));
    assert_eq!(s(&rewritten), s(&expected));
    let rewritten = referrer.expand(
        top_level,
        parse_quote!(Wrap<U>),
        |_, ix| parse_quote!(P<#ix>),
    );
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
        .visitor(top_level, |_, ix| parse_quote!(P<#ix>))
        .visit_item_struct_mut(&mut item);
    let expected: ItemStruct =
        parse_quote! { struct S<T, U> { a: P<0usize>, b: (P<1usize>, T), c: P<2usize> } };
    assert_eq!(s(&item), s(&expected));

    // An item declared in a block shadows the entries mentioning it, wherever their context is.
    let mut item: ItemFn = parse_quote! {
        fn f<U>() { struct Wrap<X>(X); let _a: Wrap<U>; let _b: Rel0; }
    };
    referrer
        .visitor(top_level, |_, ix| parse_quote!(P<#ix>))
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
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true).unwrap();
    assert_eq!(top_level, id(0));
    let referrer = leaker.finish();
    let tokens = referrer.to_token_stream();
    assert_eq!(tokens.to_string(), format!("\"{}\" (A , B)", version()));
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);

    // Contexts with consecutive ids from 0 carry no id either (the last one is that of the
    // generics of the struct, which has no entry).
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: A, b: Wrap<T>, c: Wrap<U> } };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
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
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
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
    let top_level = leaker.intern_struct(&item, true).unwrap();
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
        "\"{}\" 6 mod {{ path = [] }} ; () mod 0 ; (Rel) # 4 < T > ; Self : Tr < T > ; \
         (Wrap < T , Self >) # 6 Self = W ; (Rel2 < Self >) \
         # 8 mod {{ path = [a , b] , parent = 0 }} ; () mod 8 ; (Rel3)",
        version()
    );
    let referrer: Referrer = parse_str(&text).unwrap();
    assert_eq!(ids(&referrer), [0, 1, 4, 6, 8, 9]);
    assert_eq!(referrer.to_token_stream().to_string(), text);
    assert_eq!(
        parse2::<Referrer>(referrer.to_token_stream()).unwrap(),
        referrer
    );
    let [c0, c1, c4, c6, c8, c9] = [id(0), id(1), id(4), id(6), id(8), id(9)];
    assert_eq!(referrer.context(0), c1);
    assert_eq!(referrer.context(1), c4);
    assert_eq!(referrer.context(3), c9);
    assert_eq!(referrer.context_module(c1), Some(c0));
    assert_eq!(referrer.module_path(c0), Some(&[][..]));
    assert_eq!(referrer.context_module(c4), None);
    assert_eq!(s(referrer.context_generics(c4)), "< T >");
    assert_eq!(s(referrer.context_generics(c6)), "");
    assert!(matches!(
        referrer.context_self(c4),
        Some(SelfRequirement::Bound(_))
    ));
    assert!(matches!(
        referrer.context_self(c6),
        Some(SelfRequirement::Type(_))
    ));
    assert_eq!(referrer.context_self(c9), None);
    assert_eq!(referrer.context_module(c9), Some(c8));
    assert_eq!(
        referrer.module_path(c8),
        Some(&[parse_quote!(a), parse_quote!(b)][..])
    );
    // A module has no module of its own.
    assert_eq!(referrer.context_module(c8), None);

    // The requirements survive the removal of a context (here, one of a trait).
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U> { fn f(&self) -> Rel<Self>; fn g(&self, t: T) -> (RelB, T); }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true).unwrap();
    let referrer = leaker.finish();
    // `Rel<Self>` uses `Self`, `T` and `U` (the context 0), the tuple uses `T` (the context 1),
    // and `RelB` nothing (2). The top-level context has the generics of the trait and, as an
    // entry names `Self`, what `Self` is: it is the context 0.
    assert_eq!(top_level, id(0));
    assert_eq!(ids(&referrer), [0, 2]);
    assert!(matches!(
        referrer.context_self(id(0)),
        Some(SelfRequirement::Bound(_))
    ));
    assert_eq!(referrer.context_self(id(2)), None);
    assert_eq!(
        referrer.context_self(top_level),
        referrer.context_self(id(0))
    );
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
    let ascending = "the ids of the contexts must be in ascending order";
    let integer = "expected integer literal";
    for (text, message) in [
        // The same id twice.
        (format!("\"{}\" 2 (A) # 0 (B)", version), ascending),
        (format!("\"{}\" 2 # 3 (A) # 3 (B)", version), ascending),
        // A smaller id.
        (format!("\"{}\" 2 # 3 (A) # 1 (B)", version), ascending),
        (format!("\"{}\" 2 # 3 (A) # 2 (B)", version), ascending),
        // Not a number.
        (format!("\"{}\" # (A)", version), integer),
        (format!("\"{}\" # x (A)", version), integer),
        (
            format!("\"{}\" # 1u8 (A)", version),
            "a context id is an integer, with no suffix or `usize`",
        ),
        // A group is missing.
        (
            format!("\"{}\" 2 # 3 (A)", version),
            "unexpected end of input, expected parentheses",
        ),
        // The largest id cannot be followed by another one.
        (
            format!("\"{}\" # {} (A)", version, usize::MAX),
            "the id of the context is too large",
        ),
    ] {
        let err = parse_str::<Referrer>(&text).unwrap_err();
        assert_eq!(err.to_string(), message, "{}", text);
    }
}

/// Parses `Referrer, ContextId`, as the Referrer's macro receives them from the Leaker's.
fn parse_referrer_and_top_level(tokens: TokenStream) -> (Referrer, ContextId) {
    use parse::Parser;
    (|input: parse::ParseStream| -> Result<(Referrer, ContextId)> {
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        let top_level = input.parse()?;
        input.parse::<parse::Nothing>()?;
        Ok((referrer, top_level))
    })
    .parse2(tokens)
    .unwrap()
}

#[test]
fn the_top_level_context_reaches_the_referrer_side_through_tokens() {
    use type_leak::{encode_generics_args_to_ty_with, encode_generics_params_to_ty_with};
    let usize_const =
        |ConstParam { ident, .. }: &ConstParam| -> Type { parse_quote!([(); #ident]) };

    // The Referrer cannot tell the top-level context (it is not serialized, as it may also be
    // the context of an entry), but the id returned by the Leaker is the same in the Referrer
    // parsed from tokens, and it is written and parsed as an integer.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, const N: usize, T: 'a, U = ::core::primitive::u8> {
            fn f(&self, x: Rel<'a, T>) -> Wrap<N>;
        }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true).unwrap();
    let referrer = leaker.finish();
    // The generics are in the canonical form: sorted by kind and name, without defaults.
    assert_eq!(
        s(referrer.context_generics(top_level)),
        "< 'a , T : 'a , U , const N : usize >"
    );
    let leaker_params = &referrer.context_generics(top_level).params;
    let on_leaker_side = encode_generics_params_to_ty_with(leaker_params, usize_const);
    let expected: Type = parse_quote!((&'a (), T, U, [(); N], &'a T));
    assert_eq!(s(&on_leaker_side), s(&expected));

    let tokens = quote!(#referrer, #top_level);
    let (parsed, parsed_top_level) = parse_referrer_and_top_level(tokens);
    assert_eq!(parsed, referrer);
    assert_eq!(parsed_top_level, top_level);
    let params = &parsed.context_generics(parsed_top_level).params;
    assert_eq!(s(params), s(leaker_params));

    // The arguments of the trait are completed (the generics of a context have no defaults), and
    // put in the order of the generics (that of the trait differs): the encoding is that of the
    // generics with the arguments substituted. Matching them by position would be wrong.
    let args: AngleBracketedGenericArguments = parse_quote!(<'q, 3, u16>);
    let err =
        encode_generics_args_to_ty_with(&args.args, params, |_, _| unreachable!()).unwrap_err();
    assert!(
        matches!(err, type_leak::EncodeError::UnexpectedArgument(_)),
        "{}",
        err
    );
    let completed = parsed
        .complete_args(&args.args, &tr.generics.params, |_, _, _| unreachable!())
        .unwrap();
    let args = parsed
        .context_generic_args(parsed_top_level, &tr.generics.params, &completed)
        .unwrap();
    assert_eq!(
        args.iter().map(s).collect::<Vec<_>>(),
        ["'q", "u16", ":: core :: primitive :: u8", "3"]
    );
    let on_referrer_side =
        encode_generics_args_to_ty_with(&args, params, |expr, _| parse_quote!([(); #expr]))
            .unwrap();
    let expected: Type = parse_quote!((&'q (), u16, ::core::primitive::u8, [(); 3], &'q u16));
    assert_eq!(s(&on_referrer_side), s(&expected));

    // It is `generic_args()` of an entry for the context of the entry.
    for ix in 0..parsed.iter().count() {
        assert_eq!(
            parsed
                .generic_args(ix, &tr.generics.params, &completed)
                .unwrap()
                .iter()
                .map(s)
                .collect::<Vec<_>>(),
            parsed
                .context_generic_args(parsed.context(ix), &tr.generics.params, &completed)
                .unwrap()
                .iter()
                .map(s)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn the_top_level_generics_do_not_have_the_bounds_the_entries_imply() {
    // `&'a Rel<T>` implies `T: 'a`, which the trait does not declare: the generics of its entry
    // have the bound, and those of the top-level context do not, so a Repeater encoding the
    // generics of the trait for every entry does not carry it (the implementation of such an
    // entry is rejected with E0309), whereas the generics of the entry do. (A path with a
    // lifetime argument, `Rel<'a, T>`, implies none: it is not assumed to.)
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T, const N: usize> { fn f(&self, x: &'a Rel<T>) -> Wrap<N>; }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true).unwrap();
    let referrer = leaker.finish();
    let ix = referrer
        .iter()
        .position(|ty| s(ty) == "& 'a Rel < T >")
        .unwrap();
    assert_ne!(referrer.context(ix), top_level);
    assert_eq!(s(referrer.generics(ix)), "< 'a , T : 'a >");
    assert_eq!(
        s(referrer.context_generics(top_level)),
        "< 'a , T , const N : usize >"
    );
    assert_eq!(
        s(&referrer.impl_generics(referrer.context(ix))),
        "< 'a , T >"
    );
}

//! Tests of the behavior of the `Referrer` API that its docs used to show as examples.

mod common;

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// `iter()` has a type of another crate too, and `repeater_impls()` writes an impl for each entry.
#[test]
fn iter_has_the_types_of_other_crates_and_each_entry_gets_a_repeater_impl() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote!(
        struct S {
            a: Rel,
            b: ::foo::X,
        }
    ));
    let result = leaker.result();
    result.clone().finish().unwrap();
    // One impl for each entry, the one of another crate too.
    let param: Ident = parse_quote!(__S);
    let impls = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap();
    assert_eq!(impls.len(), 2);
}

/// `context_self()` is `Bound` for an entry that names `Self` in a trait, else `None`.
#[test]
fn context_self_is_bound_for_self_in_a_trait_and_none_otherwise() {
    let mut leaker = Leaker::new();
    leaker.intern_trait(&parse_quote!(
        trait Tr {
            fn f(&self) -> Wrap<Self>;
            fn g(&self) -> Rel;
        }
    ));
    let referrer = leaker.result().finish().unwrap();
    // `Wrap<Self>` is written for the implementors of the trait. `Rel` is for any type.
    for (_, ty, context) in referrer.iter() {
        if *ty == parse_quote!(Wrap<Self>) {
            assert!(matches!(context.self_ty, Some(SelfRequirement::Bound(_))));
        } else {
            assert!(*ty == parse_quote!(Rel) && context.self_ty.is_none());
        }
    }
}

/// `has_self()` is true for `Self::A` in a leaked type and false without it.
#[test]
fn has_self_tells_whether_an_entry_names_self() {
    let mut leaker = Leaker::new();
    leaker.intern_trait(&parse_quote!(
        trait Tr {
            fn f(&self) -> Wrap<Self::A>;
        }
    ));
    assert!(leaker.result().finish().unwrap().has_self());

    let mut leaker = Leaker::new();
    leaker.intern_trait(&parse_quote!(
        trait Tr {
            fn f(&self) -> Wrap;
        }
    ));
    assert!(!leaker.result().finish().unwrap().has_self());
}

/// `has_self()` is true if some entry names `Self` or its context's generics do, also if its
/// context has `Self` only in `Self: 'a`, which is not a leaked type nor a generic of the context.
#[test]
fn has_self_is_true_if_some_entry_needs_self() {
    let mut leaker = Leaker::new();
    leaker.intern_trait(&parse_quote!(
        trait Tr<'a>
        where
            Self: 'a,
        {
            fn f(&'a self) -> &'a Rel;
        }
    ));
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer
        .bucket_self(7, common::entry_id_at(&referrer, 0))
        .is_none());
    assert!(referrer.has_self());
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert!(parsed.has_self());

    // No entry needs `Self`.
    let mut leaker = Leaker::new();
    leaker.intern_trait(&parse_quote!(
        trait Tr<T> {
            fn f(&self, t: Rel<T>) -> Rel2;
        }
    ));
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer
        .iter()
        .all(|(ix, _, _)| referrer.bucket_self(7, ix).is_some()));
    assert!(!referrer.has_self());
}

/// `complete_params()` adds the omitted defaulted argument, written by the visitor in the default.
#[test]
fn complete_params_fills_the_defaulted_parameter() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<T>> {
            fn f(&self, t: T) -> U;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    // For `impl Tr<u8> for ..`, omitting `U`.
    let args: AngleBracketedGenericArguments = parse_quote!(<u8>);
    let completed = common::completed_args(
        &referrer,
        top_level,
        &tr,
        &args.args,
        |_, ix, _| parse_quote!(Leaked<#ix>),
    )
    .unwrap();
    let expected: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(Leaked<0usize>)];
    assert_eq!(
        quote!(#(#completed),*).to_string(),
        quote!(#(#expected),*).to_string()
    );
}

/// A copy that declares `T: Clone` again is rewritten only with this option.
#[test]
fn with_outer_generics_replaces_despite_other_bounds() {
    let tr: ItemTrait = parse_quote!(
        trait MyTrait<T> {
            fn f(&self) -> Wrap<T>;
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let replace = |_: Type, ix: EntryId, _: &Context| -> Type { parse_quote!(Leaked<#ix>) };

    // The copy declares `T` with another bound than the trait did.
    let item: ItemImpl = parse_quote!(
        impl<T: Clone> MyTrait<T> for W {
            fn f(&self) -> Wrap<T> {
                todo!()
            }
        }
    );
    let mut copy = item.clone();
    referrer
        .visitor(context, replace)
        .visit_item_impl_mut(&mut copy);
    assert_eq!(copy, item); // `T: Clone` is not the generic of the trait: nothing is replaced.

    let mut copy = item.clone();
    referrer
        .visitor(context, replace)
        .with_outer_generics()
        .visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote!(
        impl<T: Clone> MyTrait<T> for W {
            fn f(&self) -> Leaked<0usize> {
                todo!()
            }
        }
    );
    assert_eq!(copy, expected);
}

/// `with_self()` writes `__S` for `Self` in types, but not in the receiver.
#[test]
fn with_self_replaces_self_with_the_marker_host() {
    let tr: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: Rel<Self>) -> Self::Out;
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let mut sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>) -> Self::Out);
    let repeater: Path = parse_quote!(Repeater);
    let referrer = leaker.result().finish().unwrap();
    referrer
        .visitor(context, |_, ix, context| {
            repeater_projection(
                &repeater,
                &parse_quote!(Marker<__S>),
                context,
                7,
                ix,
                array_const,
            )
            .unwrap()
        })
        .with_self(parse_quote!(__S))
        .visit_signature_mut(&mut sig);
    let key = repeater_key(7, common::entry_id_at(&referrer, 0));
    let expected: Signature =
        parse_quote!(fn f(&self, x: <Marker<__S> as Repeater<(), #key>>::Type) -> <__S>::Out);
    assert_eq!(sig, expected);
}

/// `with_nested_self()` writes the entry that names the nested `Self`; `type_fn` the others.
#[test]
fn nested_self_writes_the_leakers_type_for_a_relative_self_entry() {
    // In the module `m` of the crate `leaker`.
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            a: Rel<T>,
            b: Rel2<Self>,
        }
    );
    let mut leaker = Leaker::new();
    let (module, ()) = leaker.enter_mod("m", [item.ident.clone()], |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let referrer = leaker.result().finish().unwrap();

    // The copy is in another crate. `Self` of the Leaker's `S<T>` is `::leaker::m::S<T>` there.
    let mut file = File {
        shebang: None,
        attrs: vec![],
        items: vec![Item::Struct(item)],
    };
    let repeater: Path = parse_quote!(Repeater);
    let unit: Type = parse_quote!(());
    referrer
        .visitor(module, |_, ix, context| {
            repeater_projection(&repeater, &unit, context, 7, ix, array_const).unwrap()
        })
        .with_nested_self(|nested, _leaked, ix, context| {
            let ident = nested.ident.as_ref().unwrap();
            let host = parse_quote!(::leaker::m::#ident<T>);
            repeater_projection(&repeater, &host, context, 7, ix, array_const).unwrap()
        })
        .visit_file_mut(&mut file);
    let (ix0, ix1) = (
        common::entry_id_at(&referrer, 0),
        common::entry_id_at(&referrer, 1),
    );
    let (key0, key1) = (repeater_key(7, ix0), repeater_key(7, ix1));
    let expected: File = parse_quote! {
        struct S<T> {
            a: <() as Repeater<(T,), #key0>>::Type,
            b: <::leaker::m::S<T> as Repeater<(T,), #key1>>::Type,
        }
    };
    assert_eq!(file, expected);
}

/// An entry for any type (`T: AsRef<Self>`) is implemented for `__S`; the copy passes its `Self`.
#[test]
fn nested_self_passes_the_copys_own_self_for_a_bound_naming_self() {
    let item: ItemStruct = parse_quote! {
        struct S<T: ::core::convert::AsRef<Self>> { a: Rel<T>, }
    };
    let mut leaker = Leaker::new();
    let (module, ()) = leaker.enter_mod("m", [item.ident.clone()], |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let (_, _, context) = referrer.iter().next().unwrap();
    assert!(matches!(context.self_ty, Some(SelfRequirement::Any)));

    // The Leaker's crate: the impl is for any type `__S` that `T: AsRef<__S>` holds for.
    let repeater: Path = parse_quote!(Repeater);
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&repeater, 7, ImplSelf::Param(&param), slot_const(&repeater))
        .unwrap()
        .remove(0);
    // `r.path` has no `KEY`, and `ToTokens` writes the one of the id 7.
    let key = repeater_key(7, r.entry);
    let imp: ItemImpl = parse_quote!(#r);
    let expected: ItemImpl = parse_quote! {
        impl<T: ::core::convert::AsRef<__S>, __S: ?::core::marker::Sized>
            Repeater<(T,), #key> for __S
        {
            type Type = Rel<T>;
        }
    };
    assert_eq!(imp, expected);

    // The Referrer's crate: the copy has the bound for its own type, `Self`.
    let mut file = File {
        shebang: None,
        attrs: vec![],
        items: vec![Item::Struct(item)],
    };
    referrer
        .visitor(module, |_, ix, context| {
            repeater_projection(&repeater, &parse_quote!(()), context, 7, ix, array_const).unwrap()
        })
        .with_nested_self(|_nested, _leaked, ix, context| match &context.self_ty {
            Some(SelfRequirement::Any) => repeater_projection(
                &repeater,
                &parse_quote!(Self),
                context,
                7,
                ix,
                slot_const(&repeater),
            )
            .unwrap(),
            _ => unimplemented!("the Leaker's type, from `nested`"),
        })
        .visit_file_mut(&mut file);
    let expected: File = parse_quote! {
        struct S<T: ::core::convert::AsRef<Self>> {
            a: <Self as Repeater<(T,), #key>>::Type,
        }
    };
    assert_eq!(file, expected);
}

/// An absolute type that names `Self` stays as written, also if the macro replaces the optional
/// entries.
#[test]
fn nested_self_keeps_an_absolute_self_type_as_written_when_optional_entries_are_replaced() {
    let item: ItemStruct = parse_quote! {
        struct Node<T> {
            next: ::std::boxed::Box<Self>, // the copy's `Node<T>`: no entry
            other: ::p::Wrap<T>,           // an optional entry: it does not name `Self`
            leaked: Rel<Self>,             // an entry: the macro writes the type
        }
    };
    let mut leaker = Leaker::new();
    let (module, ()) = leaker.enter_mod("m", [item.ident.clone()], |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2); // `Rel<Self>`, then `::p::Wrap<T>` (optional)

    let mut file = File {
        shebang: None,
        attrs: vec![],
        items: vec![Item::Struct(item)],
    };
    referrer
        .visitor(module, |_leaked, ix, _| parse_quote!(Entry<#ix>))
        .with_nested_self(|_nested, _leaked, ix, _| parse_quote!(LeakersType<#ix>))
        .visit_file_mut(&mut file);
    let expected: File = parse_quote! {
        struct Node<T> {
            next: ::std::boxed::Box<Self>,
            other: Entry<1usize>,
            leaked: LeakersType<0usize>,
        }
    };
    assert_eq!(file, expected);
}

/// A macro has a type of another crate copied as written by giving the Leaker
/// `Rules::external_crates()`: it is no entry, so `type_fn` and `nested_fn` are not asked.
#[test]
fn a_leaker_that_allows_the_crates_makes_no_entry_of_their_types() {
    let item: ItemStruct = parse_quote! {
        struct Node<T> {
            next: ::std::boxed::Box<Self>, // the copy's `Node<T>`: no entry
            other: ::p::Wrap<T>,           // no entry: the Leaker allows the crate
            more: ::q::Wrap<T>,            // no entry: the Leaker allows the crate
            leaked: Rel<Self>,             // an entry: the macro writes the type
        }
    };
    let mut leaker = Leaker::with_rule(Rules::new().external_crates());
    let (module, ()) = leaker.enter_mod("m", [item.ident.clone()], |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);

    let mut file = File {
        shebang: None,
        attrs: vec![],
        items: vec![Item::Struct(item)],
    };
    referrer
        .visitor(module, |_leaked, ix, _| parse_quote!(Entry<#ix>))
        .with_nested_self(|_nested, _leaked, ix, _| parse_quote!(LeakersType<#ix>))
        .visit_file_mut(&mut file);
    let expected: File = parse_quote! {
        struct Node<T> {
            next: ::std::boxed::Box<Self>,
            other: ::p::Wrap<T>,
            more: ::q::Wrap<T>,
            leaked: LeakersType<0usize>,
        }
    };
    assert_eq!(file, expected);
}

/// The serialized form starts with the `FORMAT_VERSION` tag, and another format is refused.
#[test]
fn format_version_is_the_serialized_tag_and_another_tag_is_refused() {
    let referrer = Leaker::new().result().finish().unwrap();
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    assert!(referrer.to_token_stream().to_string().starts_with(&tag));

    // A Referrer written by another version of `type-leak` is refused with an error that says so.
    let error = parse_str::<Referrer>("\"type-leak-format-0\"").unwrap_err();
    assert!(error.to_string().contains("incompatible type-leak"));
}

/// `visitor()` writes the entries, and `Substitute::with_params()` completes the arguments of an
/// impl and writes them into the copy.
#[test]
fn visitor_then_substitute_with_params_give_the_copy_at_a_use_site() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self, t: T) -> Wrap<T>;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();

    // 1. The copy of `f` in `impl Tr<u8> for ..`, with the entry written.
    let mut sig: Signature = parse_quote!(fn f(&self, t: T) -> Wrap<T>);
    let repeater: Path = parse_quote!(crate::Repeater);
    let self_ty: Type = parse_quote!(Self);
    referrer
        .visitor(top_level, |_ty, ix, context| {
            repeater_projection(&repeater, &self_ty, context, 7, ix, slot_const(&repeater)).unwrap()
        })
        .visit_signature_mut(&mut sig);
    let (ix, _, _) = referrer.iter().next().unwrap();
    let key = repeater_key(7, ix);
    let expected: Signature = parse_quote!(
        fn f(&self, t: T) -> <Self as crate::Repeater<(T,), #key>>::Type
    );
    assert_eq!(sig, expected);

    // 2. and 3. The arguments of the impl, written into the copy: `T` is `u8`.
    let given: Vec<GenericArgument> = vec![parse_quote!(u8)];
    let mut site = Substitute::with_params(&tr.generics.params, &given).unwrap();
    site.visit_signature_mut(&mut sig);
    site.finish().unwrap();
    let expected: Signature = parse_quote!(
        fn f(&self, t: u8) -> <Self as crate::Repeater<(u8,), #key>>::Type
    );
    assert_eq!(sig, expected);
}

/// The same for a trait with a lifetime parameter: `Substitute` writes `'a` where it is not in a
/// type (the receiver, a bound), and the items of the impl only, not its header.
#[test]
fn visitor_then_substitute_with_params_give_the_items_of_an_impl_with_a_lifetime() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<'a, T: 'a> {
            fn keep(&'a self, t: T) -> Wrap<&'a T>
            where
                Self: 'a;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();

    // The copy of the items in `impl<'x> Tr<'x, u8> for Host`.
    let mut imp: ItemImpl = parse_quote!(
        impl<'x> Tr<'x, u8> for Host {
            fn keep(&'a self, t: T) -> Wrap<&'a T>
            where
                Self: 'a,
            {
                todo!()
            }
        }
    );
    let repeater: Path = parse_quote!(crate::Repeater);
    let self_ty: Type = parse_quote!(Self);
    referrer
        .visitor(top_level, |_ty, ix, context| {
            repeater_projection(&repeater, &self_ty, context, 7, ix, slot_const(&repeater)).unwrap()
        })
        .visit_item_impl_mut(&mut imp);
    let given: Vec<GenericArgument> = vec![parse_quote!('x), parse_quote!(u8)];
    let mut site = Substitute::with_params(&tr.generics.params, &given).unwrap();
    for item in &mut imp.items {
        site.visit_impl_item_mut(item);
    }
    site.finish().unwrap();
    let (ix, _, _) = referrer.iter().next().unwrap();
    let key = repeater_key(7, ix);
    let expected: ItemImpl = parse_quote!(
        impl<'x> Tr<'x, u8> for Host {
            fn keep(&'x self, t: u8) -> <Self as crate::Repeater<(&'x (), u8, &'x u8), #key>>::Type
            where
                Self: 'x,
            {
                todo!()
            }
        }
    );
    assert_eq!(
        imp.to_token_stream().to_string(),
        expected.to_token_stream().to_string()
    );
}

/// A `Referrer` and the top-level `ContextId` are written one after the other and read back.
#[test]
fn referrer_and_context_id_are_read_back_from_tokens() {
    use syn::parse::Parser;
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&parse_quote!(
        struct S<T> {
            a: Wrap<T>,
        }
    ));
    let referrer = leaker.result().finish().unwrap();

    // Crate A writes both into the input of the macro of crate B...
    let tokens = quote!(#referrer, #top_level);
    // ...which reads them back.
    let (read, read_top_level) = (|input: parse::ParseStream| -> Result<(Referrer, ContextId)> {
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        Ok((referrer, input.parse()?))
    })
    .parse2(tokens)
    .unwrap();
    assert_eq!(read, referrer);
    assert_eq!(read_top_level, top_level);
}

/// An `EntryId` is written as a `usize` literal and read with or without the suffix.
#[test]
fn entry_id_is_written_with_the_usize_suffix_and_read_with_or_without_it() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote!(
        struct S {
            a: Rel,
            b: Rel2,
        }
    ));
    let referrer = leaker.result().finish().unwrap();

    let ids: Vec<EntryId> = referrer.iter().map(|(ix, _, _)| ix).collect();
    assert_eq!(ids.len(), 2);
    let ix = ids[1];
    assert_eq!(ix.index(), 1);
    assert_eq!(ix.to_string(), "1");
    assert_eq!(ix.to_token_stream().to_string(), "1usize");

    // With or without the suffix.
    assert_eq!(parse_str::<EntryId>("1").unwrap(), ix);
    assert_eq!(parse_str::<EntryId>("1usize").unwrap(), ix);
    assert!(parse_str::<EntryId>("1u8").is_err());
}

/// What is not a non-negative integer that fits a `usize` is a parse error, with a message that
/// says why.
#[test]
fn entry_id_parse_errors() {
    let too_large = format!("{}0", usize::MAX);
    let cases: [(&str, &str); 8] = [
        ("", "unexpected end of input, expected integer literal"),
        ("x", "expected integer literal"),
        ("'a", "expected integer literal"),
        ("1.0", "expected integer literal"),
        ("-1", "invalid digit found in string"),
        (&too_large, "number too large to fit in target type"),
        (
            "1u8",
            "an entry id is an integer, with no suffix or `usize`",
        ),
        ("1 2", "unexpected token"),
    ];
    for (text, message) in cases {
        let err = parse_str::<EntryId>(text).unwrap_err();
        assert_eq!(err.to_string(), message, "{:?}", text);
    }
    // The largest one that fits is read.
    let largest = usize::MAX.to_string();
    assert_eq!(
        parse_str::<EntryId>(&largest).unwrap().index(),
        usize::MAX,
        "{}",
        largest
    );
}

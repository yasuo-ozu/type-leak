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
    let repeater: Path = parse_quote!(crate::Repeater);
    let impls = result
        .repeater_impls(&Projection::new(
            repeater.clone(),
            7,
            ImplSelf::Param(param.clone()),
        ))
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
    assert!(common::bucket_self(&referrer, 7, common::entry_id_at(&referrer, 0)).is_none());
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
        .all(|(ix, _, _)| common::bucket_self(&referrer, 7, ix).is_some()));
    assert!(!referrer.has_self());
}

/// `complete_args()` adds the omitted defaulted argument, with the leaked type of the default
/// written through the Repeater.
#[test]
fn complete_args_fills_the_defaulted_parameter() {
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
    let completed = common::completed_args(&referrer, top_level, &tr.generics, &args.args).unwrap();
    let key = common::key(0, common::entry_id_at(&referrer, 0));
    let expected: Vec<GenericArgument> = vec![
        parse_quote!(u8),
        parse_quote!(<Self as Repeater<(u8,), #key>>::Type),
    ];
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
    let projection = common::projection(parse_quote!(Repeater), 7);

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
        .visitor(context, &projection)
        .visit_item_impl_mut(&mut copy);
    assert_eq!(copy, item); // `T: Clone` is not the generic of the trait: nothing is replaced.

    let mut copy = item.clone();
    referrer
        .visitor(context, &projection)
        .with_outer_generics()
        .visit_item_impl_mut(&mut copy);
    let key = projection.key(common::entry_id_at(&referrer, 0));
    let expected: ItemImpl = parse_quote!(
        impl<T: Clone> MyTrait<T> for W {
            fn f(&self) -> <Self as Repeater<(T,), #key>>::Type {
                todo!()
            }
        }
    );
    assert_eq!(copy, expected);
}

/// `with_self()` writes `__S` for `Self` in types, but not in the receiver, and as the self type of
/// the paths through the Repeater.
#[test]
fn with_self_replaces_self_with_the_host() {
    let tr: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: Rel<Self>) -> Self::Out;
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let mut sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>) -> Self::Out);
    let projection = common::projection(parse_quote!(Repeater), 7);
    let referrer = leaker.result().finish().unwrap();
    referrer
        .visitor(context, &projection)
        .with_self(parse_quote!(__S))
        .unwrap()
        .visit_signature_mut(&mut sig);
    let key = common::key(7, common::entry_id_at(&referrer, 0));
    let expected: Signature =
        parse_quote!(fn f(&self, x: <__S as Repeater<(), #key>>::Type) -> <__S>::Out);
    assert_eq!(sig, expected);
}

/// `with_nested_self()` writes the entry that names the own `Self` of a nested item with the type
/// that its closure returns; the others are written with `Self`.
#[test]
fn nested_self_writes_the_leakers_type_for_a_relative_self_entry() {
    // A struct nested in a scope of the crate `leaker`.
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            a: Rel<T>,
            b: Rel2<Self>,
        }
    );
    let mut leaker = Leaker::new();
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let referrer = leaker.result().finish().unwrap();
    let projection = common::projection(parse_quote!(Repeater), 7);
    let (key0, key1) = (
        common::key(7, common::entry_id_at(&referrer, 0)),
        common::key(7, common::entry_id_at(&referrer, 1)),
    );
    let copy = |hook: &mut dyn FnMut(&SelfRequirement) -> Option<Type>| {
        let mut module: ItemMod = parse_quote!(mod w { #item });
        referrer
            .visitor(context, &projection)
            .with_nested_self(|requirement| hook(requirement))
            .visit_item_mod_mut(&mut module);
        module
    };

    // The copy is in another crate. `Self` of the Leaker's `S<T>` is `::leaker::m::S<T>` there. The
    // closure gets the requirement of `S`, once.
    let mut asked = Vec::new();
    let module = copy(&mut |requirement| {
        asked.push(requirement.clone());
        match requirement {
            SelfRequirement::Type(ty) => Some(parse_quote!(::leaker::m::#ty)),
            _ => None,
        }
    });
    assert_eq!(asked, [SelfRequirement::Type(parse_quote!(S<T>))]);
    let expected: ItemMod = parse_quote! {
        mod w {
            struct S<T> {
                a: <Self as Repeater<(T,), #key0>>::Type,
                b: <::leaker::m::S<T> as Repeater<(T,), #key1>>::Type,
            }
        }
    };
    assert_eq!(module, expected);

    // With `None` the entry is left as written.
    let module = copy(&mut |_| None);
    let expected: ItemMod = parse_quote! {
        mod w {
            struct S<T> {
                a: <Self as Repeater<(T,), #key0>>::Type,
                b: Rel2<Self>,
            }
        }
    };
    assert_eq!(module, expected);
}

/// Without `with_nested_self()` the entry for the own `Self` of a nested item (`Type`, `Bound`) is
/// left as written, and the entry for any type (`Any`) is written with `Self`: it needs no hook.
#[test]
fn nested_self_entries_are_left_as_written_without_the_hook_but_for_any_type() {
    let module: ItemMod = parse_quote!(
        mod m {
            struct N<T>(Wrap<Self>, Rel<T>);
            trait Inner {
                fn get(&self) -> Wrap<Self>;
            }
            struct M<T: ::core::convert::AsRef<Self>>(Rel2<Self, T>);
        }
    );
    let mut leaker = Leaker::new();
    let (top, ()) = leaker.enter_generics(&Generics::default(), |view| {
        for item in &module.content.as_ref().unwrap().1 {
            view.visit_item(item);
        }
    });
    let result = leaker.result();
    // `Rel2<Self, T>` is the risk of an entry for any type: the macro takes it.
    let referrer = result
        .allow(|cause: &Cause, _: Place| matches!(cause, Cause::Type(_)))
        .finish()
        .unwrap();
    let projection = common::projection(parse_quote!(Repeater), 7);
    let key = |text: &str| {
        let (ix, _, _) = referrer
            .iter()
            .find(|(_, ty, _)| common::s(ty) == text)
            .unwrap();
        common::key(7, ix)
    };
    let (rel, rel2) = (key("Rel < T >"), key("Rel2 < Self , T >"));
    let mut copy = module.clone();
    referrer
        .visitor(top, &projection)
        .visit_item_mod_mut(&mut copy);
    let expected: ItemMod = parse_quote!(
        mod m {
            struct N<T>(Wrap<Self>, <Self as Repeater<(T,), #rel>>::Type);
            trait Inner {
                fn get(&self) -> Wrap<Self>;
            }
            struct M<T: ::core::convert::AsRef<Self>>(<Self as Repeater<(T,), #rel2>>::Type);
        }
    );
    assert_eq!(copy, expected);

    // With the hook the `Type` and the `Bound` entries are written for the type it returns, and
    // the entry for any type is still written with `Self`; the hook is asked once for each item.
    let mut asked = Vec::new();
    let mut copy = module.clone();
    referrer
        .visitor(top, &projection)
        .with_nested_self(|requirement| {
            asked.push(format!("{:?}", requirement));
            Some(parse_quote!(Leaker))
        })
        .visit_item_mod_mut(&mut copy);
    // `Wrap<Self>` of `N` and of `Inner` are two entries with the same type and two requirements.
    assert_eq!(asked.len(), 2, "{:?}", asked);
    let text = common::s(&copy);
    assert_eq!(text.matches("< Leaker as Repeater").count(), 2, "{}", text);
    assert!(text.contains(&format!(
        "< Self as Repeater < (T ,) , {}u128 > > :: Type",
        rel
    )));
    assert!(text.contains(&format!(
        "< Self as Repeater < (T ,) , {}u128 > > :: Type",
        rel2
    )));
}

/// An entry for any type (`T: AsRef<Self>`) is implemented for `__S`; the copy passes its `Self`.
#[test]
fn nested_self_passes_the_copys_own_self_for_a_bound_naming_self() {
    let item: ItemStruct = parse_quote! {
        struct S<T: ::core::convert::AsRef<Self>> { a: Rel<T>, }
    };
    let mut leaker = Leaker::new();
    let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| {
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
        impl<T: ::core::convert::AsRef<__S>, __S: ?::core::marker::Sized>
            Repeater<(T,), #key> for __S
        {
            type Type = Rel<T>;
        }
    };
    assert_eq!(imp, expected);

    // The Referrer's crate: the copy has the bound for its own type, `Self`.
    let mut module: ItemMod = parse_quote!(mod w { #item });
    referrer
        .visitor(scope, &common::projection(repeater.clone(), 7))
        .visit_item_mod_mut(&mut module);
    let expected: ItemMod = parse_quote! {
        mod w {
            struct S<T: ::core::convert::AsRef<Self>> {
                a: <Self as Repeater<(T,), #key>>::Type,
            }
        }
    };
    assert_eq!(module, expected);
}

/// An absolute type that names `Self` stays as written, also if the optional entries are replaced.
#[test]
fn nested_self_keeps_an_absolute_self_type_as_written_when_optional_entries_are_replaced() {
    let item: ItemStruct = parse_quote! {
        struct Node<T> {
            next: ::std::boxed::Box<Self>, // the copy's `Node<T>`: no entry
            other: ::p::Wrap<T>,           // an optional entry: it does not name `Self`
            leaked: Rel<Self>,             // an entry
        }
    };
    let mut leaker = Leaker::new();
    let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2); // `Rel<Self>`, then `::p::Wrap<T>` (optional)

    let mut module: ItemMod = parse_quote!(mod w { #item });
    let projection = common::projection(parse_quote!(Repeater), 7);
    referrer
        .visitor(scope, &projection)
        .with_nested_self(|requirement| match requirement {
            SelfRequirement::Type(ty) => Some(parse_quote!(::leaker::m::#ty)),
            _ => None,
        })
        .visit_item_mod_mut(&mut module);
    let (key0, key1) = (
        common::key(7, common::entry_id_at(&referrer, 0)),
        common::key(7, common::entry_id_at(&referrer, 1)),
    );
    let expected: ItemMod = parse_quote! {
        mod w {
            struct Node<T> {
                next: ::std::boxed::Box<Self>,
                other: <Self as Repeater<(T,), #key1>>::Type,
                leaked: <::leaker::m::Node<T> as Repeater<(T,), #key0>>::Type,
            }
        }
    };
    assert_eq!(module, expected);
}

/// A macro has a type of another crate copied as written by giving the Leaker
/// `Rules::external_crates()`: it is no entry, so the visitor does not write it.
#[test]
fn a_leaker_that_allows_the_crates_makes_no_entry_of_their_types() {
    let item: ItemStruct = parse_quote! {
        struct Node<T> {
            next: ::std::boxed::Box<Self>, // the copy's `Node<T>`: no entry
            other: ::p::Wrap<T>,           // no entry: the Leaker allows the crate
            more: ::q::Wrap<T>,            // no entry: the Leaker allows the crate
            leaked: Rel<Self>,             // an entry
        }
    };
    let mut leaker = Leaker::with_rule(Rules::new().external_crates());
    let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);

    let mut module: ItemMod = parse_quote!(mod w { #item });
    let projection = common::projection(parse_quote!(Repeater), 7);
    referrer
        .visitor(scope, &projection)
        .with_nested_self(|requirement| match requirement {
            SelfRequirement::Type(ty) => Some(parse_quote!(::leaker::m::#ty)),
            _ => None,
        })
        .visit_item_mod_mut(&mut module);
    let key = common::key(7, common::entry_id_at(&referrer, 0));
    let expected: ItemMod = parse_quote! {
        mod w {
            struct Node<T> {
                next: ::std::boxed::Box<Self>,
                other: ::p::Wrap<T>,
                more: ::q::Wrap<T>,
                leaked: <::leaker::m::Node<T> as Repeater<(T,), #key>>::Type,
            }
        }
    };
    assert_eq!(module, expected);
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

/// `visitor()` writes the entries, and the `Substitute` of `complete_args()` writes the arguments
/// of an impl into the copy.
#[test]
fn visitor_then_complete_args_give_the_copy_at_a_use_site() {
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
    let projection = common::projection(repeater.clone(), 7);
    referrer
        .visitor(top_level, &projection)
        .visit_signature_mut(&mut sig);
    let (ix, _, _) = referrer.iter().next().unwrap();
    let key = common::key(7, ix);
    let expected: Signature = parse_quote!(
        fn f(&self, t: T) -> <Self as crate::Repeater<(T,), #key>>::Type
    );
    assert_eq!(sig, expected);

    // 2. and 3. The arguments of the impl, written into the copy: `T` is `u8`.
    let given: Vec<GenericArgument> = vec![parse_quote!(u8)];
    let mut substitute = referrer
        .complete_args(
            top_level,
            &given,
            &Projection::new(repeater.clone(), 7, ImplSelf::Param(parse_quote!(__S))),
        )
        .unwrap();
    substitute.visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(
        fn f(&self, t: u8) -> <Self as crate::Repeater<(u8,), #key>>::Type
    );
    assert_eq!(sig, expected);
}

/// The same for a trait with a lifetime parameter: `Substitute` writes `'a` where it is not in a
/// type (the receiver, a bound), and the items of the impl only, not its header.
#[test]
fn visitor_then_complete_args_give_the_items_of_an_impl_with_a_lifetime() {
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
    let projection = common::projection(repeater.clone(), 7);
    referrer
        .visitor(top_level, &projection)
        .visit_item_impl_mut(&mut imp);
    let given: Vec<GenericArgument> = vec![parse_quote!('x), parse_quote!(u8)];
    let mut substitute = referrer
        .complete_args(
            top_level,
            &given,
            &Projection::new(repeater.clone(), 7, ImplSelf::Param(parse_quote!(__S))),
        )
        .unwrap();
    substitute.visit_item_impl_mut(&mut imp);
    let (ix, _, _) = referrer.iter().next().unwrap();
    let key = common::key(7, ix);
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

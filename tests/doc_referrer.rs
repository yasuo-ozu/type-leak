//! Tests of the behavior of the `Referrer` API that its docs used to show as examples.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// `iter()` has a type of another crate too, and `repeater_impl()` writes an impl for each entry.
#[test]
fn iter_has_the_types_of_other_crates_and_each_entry_gets_a_repeater_impl() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(
        &parse_quote!(
            struct S {
                a: Rel,
                b: ::foo::X,
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    // One impl for each entry, the one of another crate too.
    let param: Ident = parse_quote!(__S);
    let impls: Vec<RepeaterImpl> = (0..referrer.iter().count())
        .map(|ix| {
            referrer
                .repeater_impl(ix, ImplSelf::Param(&param), array_const)
                .unwrap()
        })
        .collect();
    assert_eq!(impls.len(), 2);
}

/// `context_self()` is `Bound` for an entry that names `Self` in a trait, else `None`.
#[test]
fn context_self_is_bound_for_self_in_a_trait_and_none_otherwise() {
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr {
                fn f(&self) -> Wrap<Self>;
                fn g(&self) -> Rel;
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    // `Wrap<Self>` is written for the implementors of the trait. `Rel` is for any type.
    for (ty, context) in referrer.iter() {
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
    leaker.intern_trait(
        &parse_quote!(
            trait Tr {
                fn f(&self) -> Wrap<Self::A>;
            }
        ),
        true,
    );
    assert!(leaker.result().finish().unwrap().has_self());

    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr {
                fn f(&self) -> Wrap;
            }
        ),
        true,
    );
    assert!(!leaker.result().finish().unwrap().has_self());
}

/// `has_self()` is true if some entry names `Self` or its context's generics do, also if its
/// context has `Self` only in `Self: 'a`, which is not a leaked type nor a generic of the context.
#[test]
fn has_self_is_true_if_some_entry_needs_self() {
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr<'a>
            where
                Self: 'a,
            {
                fn f(&'a self) -> &'a Rel;
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.bucket_self(0, 7).is_none());
    assert!(referrer.has_self());
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert!(parsed.has_self());

    // No entry needs `Self`.
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr<T> {
                fn f(&self, t: Rel<T>) -> Rel2;
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    assert!((0..referrer.iter().count()).all(|ix| referrer.bucket_self(ix, 7).is_some()));
    assert!(!referrer.has_self());
}

/// `complete_args()` adds the omitted defaulted argument, written by `type_fn`.
#[test]
fn complete_args_fills_the_defaulted_parameter() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<T>> {
            fn f(&self, t: T) -> U;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    // For `impl Tr<u8> for ..`, omitting `U`.
    let args: AngleBracketedGenericArguments = parse_quote!(<u8>);
    let completed = referrer
        .complete_args(
            top_level,
            &tr.generics.params,
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
fn outermost_generics_as_leakers_replaces_despite_other_bounds() {
    let tr: ItemTrait = parse_quote!(
        trait MyTrait<T> {
            fn f(&self) -> Wrap<T>;
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let replace = |_: Type, ix: usize| -> Type { parse_quote!(Leaked<#ix>) };

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
        .with_outermost_generics_as_leakers()
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
    let context = leaker.intern_trait(&tr, true);
    let mut sig: Signature = parse_quote!(fn f(&self, x: Rel<Self>) -> Self::Out);
    let slots = RepeaterSlots::new(parse_quote!(Repeater));
    leaker
        .result()
        .finish()
        .unwrap()
        .visitor(context, |_, ix| {
            slots.projection(&parse_quote!(Marker<__S>), &parse_quote!(()), 7, ix)
        })
        .with_self(parse_quote!(__S))
        .visit_signature_mut(&mut sig);
    let key = repeater_key(7, 0);
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
    let slots = RepeaterSlots::new(parse_quote!(Repeater));
    let unit: Type = parse_quote!(());
    referrer
        .visitor(module, |_, ix| slots.projection(&unit, &unit, 7, ix))
        .with_nested_self(|nested, _leaked, ix| {
            let ident = nested.ident.as_ref().unwrap();
            let host = parse_quote!(::leaker::m::#ident<T>);
            slots.projection(&host, &unit, 7, ix)
        })
        .visit_file_mut(&mut file);
    let (key0, key1) = (repeater_key(7, 0), repeater_key(7, 1));
    let expected: File = parse_quote! {
        struct S<T> {
            a: <() as Repeater<(), #key0>>::Type,
            b: <::leaker::m::S<T> as Repeater<(), #key1>>::Type,
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
    let referrer = leaker.result().finish().unwrap();
    let (_, context) = referrer.iter().next().unwrap();
    assert!(matches!(context.self_ty, Some(SelfRequirement::Any)));

    // The Leaker's crate: the impl is for any type `__S` that `T: AsRef<__S>` holds for.
    let slots = RepeaterSlots::new(parse_quote!(Repeater));
    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(0, ImplSelf::Param(&param), slots.const_fn())
        .unwrap();
    let (generics, ty) = (&r.generics, &r.ty);
    let repeater = slots.trait_path(&r.args, 7, 0);
    let imp: ItemImpl = parse_quote! {
        impl #generics #repeater for __S { type Type = #ty; }
    };
    let key = repeater_key(7, 0);
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
        .visitor(module, |_, ix| {
            slots.projection(&parse_quote!(()), &parse_quote!(()), 7, ix)
        })
        .with_nested_self(|_nested, _leaked, ix| {
            let args = referrer
                .entry_args(ix, &UseSite::identity(), slots.const_fn())
                .unwrap();
            match &referrer.iter().nth(ix).unwrap().1.self_ty {
                Some(SelfRequirement::Any) => slots.projection(&parse_quote!(Self), &args, 7, ix),
                _ => unimplemented!("the Leaker's type, from `nested`"),
            }
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
        .visitor(module, |_leaked, ix| parse_quote!(Entry<#ix>))
        .with_nested_self(|_nested, _leaked, ix| parse_quote!(LeakersType<#ix>))
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
    let mut leaker = Leaker::from_config(Rules::new().external_crates(), Options::new());
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
        .visitor(module, |_leaked, ix| parse_quote!(Entry<#ix>))
        .with_nested_self(|_nested, _leaked, ix| parse_quote!(LeakersType<#ix>))
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

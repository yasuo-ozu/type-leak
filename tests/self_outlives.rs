//! `Self: 'a`, which a leaked type implies (`&'a Rel<Self>`, `&'a (Self, T)`: a reference requires
//! its referent to outlive it) or the item writes (`where Self: 'a`). An associated type of `Self`
//! (`&'a Self::Item`, `&'a <Self as Tr>::Item`) implies `Self::Item: 'a` instead, which is a where
//! predicate of the context and no `Self: 'a`: the original gets no more from the well-formedness
//! of its signature, and a copy that asks for `Self: 'a` is stricter (F34). `Self` is
//! not a generic parameter, so it is not among the bounds of the generics of the context, but its
//! own part of it (`Referrer::context_self_outlives()`), which the encoders encode as `&'a Self`
//! (`&'a __S` in the Repeater's implementation for the implementor of the trait), after the
//! lifetime bounds. The implementation does not declare it, as nothing proves it at a use but the
//! encoding.
//!
//! A path with a lifetime argument (`Rel<'a, Self>`, where `struct Rel<'a, T>(&'a T)` has `T: 'a`
//! inferred) implies nothing: its definition is not seen, so the bound is not assumed, and it is
//! only there if it is written.

mod common;

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{
    without_encoded_bounds, ContextId, EncodeError, Leaker, Referrer, RepeaterSlots,
    SelfRequirement,
};

/// The tag of the format of a serialized Referrer, which it starts with.
fn tag() -> String {
    format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn of_trait(item: ItemTrait) -> Referrer {
    {
        let mut leaker = Leaker::new();
        leaker.intern_trait(&item, true).unwrap();
        leaker.finish()
    }
}

fn lifetimes(list: &[&str]) -> Vec<Lifetime> {
    list.iter().map(|l| parse_str(l).unwrap()).collect()
}

/// The lifetimes which `Self` outlives in each context, with the types in it.
fn outlives(referrer: &Referrer) -> Vec<(Vec<String>, Vec<String>)> {
    referrer
        .contexts()
        .map(|context| {
            let types = referrer
                .iter()
                .enumerate()
                .filter(|(ix, _)| referrer.context(*ix) == context)
                .map(|(_, ty)| s(ty))
                .collect();
            let outlives = referrer
                .context_self_outlives(context)
                .iter()
                .map(|l| l.to_string())
                .collect();
            (types, outlives)
        })
        .collect()
}

fn slots() -> RepeaterSlots {
    RepeaterSlots::new(parse_quote!(crate::Repeater))
}

#[test]
fn a_type_implying_it_records_it_in_the_context() {
    // `&'a Rel<Self>` needs `Self: 'a`, with a lifetime of the method or of the trait.
    for item in [
        parse_quote! { trait Tr { fn f<'a>(&self, x: &'a Rel<Self>); } },
        // `Self` apart from its associated type.
        parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: &'a Rel<(Self, Self::Item)>); } },
        parse_quote! { trait Tr<'a> { fn f(&self, x: &'a Rel<Self>); } },
        parse_quote! { trait Tr { fn f<'a>(&self, x: &'a Rel<(u8, Vec<Self>)>); } },
        // The arguments of a trait object must outlive too (`dyn Tr<Self> + 'b: 'a`).
        parse_quote! { trait Tr { fn f<'a>(&self, x: &'a Rel<Box<dyn Other<Self>>>); } },
        // A reference is interned as a whole if its lifetime is a generic of the trait.
        parse_quote! { trait Tr<'a> { fn f(&self, x: &'a mut Rel<Self>); } },
    ] {
        let item: ItemTrait = item;
        let referrer = of_trait(item.clone());
        let found = outlives(&referrer);
        assert_eq!(
            found.iter().filter(|(_, o)| !o.is_empty()).count(),
            1,
            "{}: {:?}",
            s(&item),
            found
        );
        let (types, o) = found.iter().find(|(_, o)| !o.is_empty()).unwrap();
        assert_eq!(o, &["'a"], "{}", s(&item));
        assert_eq!(types.len(), 1, "{}", s(&item));
        // `Self` has a meaning in the context.
        let context = referrer.context(0);
        assert!(
            matches!(
                referrer.context_self(context),
                Some(SelfRequirement::Bound(_))
            ),
            "{}",
            s(&item)
        );
        assert_eq!(referrer.context_self_outlives(context), lifetimes(&["'a"]));
    }
    // Every lifetime, sorted, each once (the reference of a generic lifetime is the type, with the
    // lifetime of the reference inside it too).
    let referrer = of_trait(parse_quote! {
        trait Tr<'b, 'a> { fn f(&self, x: &'a &'b Rel<Self>); }
    });
    assert_eq!(
        outlives(&referrer)[0],
        (
            vec!["& 'a & 'b Rel < Self >".to_string()],
            vec!["'a".to_string(), "'b".to_string()]
        )
    );
}

#[test]
fn an_associated_type_of_self_implies_its_own_outlives_and_not_self() {
    // `&'a Rel<Self::Item>`, `&'a Rel<<Self as Tr>::Item>`, `&'a Rel<<Self as Tr<T>>::Item>` and
    // `&'a Rel<Self::G<'b>>` need `<path>: 'a`, which the original has from its signature. The
    // context has it as a where predicate, and `Self: 'a` is not recorded (F34).
    for (item, predicate) in [
        (
            parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: &'a Rel<Self::Item>); } },
            "Self :: Item : 'a",
        ),
        (
            parse_quote! {
                trait Tr { type Item; fn f<'a>(&self, x: &'a Rel<<Self as Tr>::Item>); }
            },
            "< Self as Tr > :: Item : 'a",
        ),
        (
            parse_quote! {
                trait Tr<T> { type Item; fn f<'a>(&self, x: &'a Rel<<Self as Tr<T>>::Item>); }
            },
            "< Self as Tr < T > > :: Item : 'a",
        ),
        (
            parse_quote! {
                trait Tr<'b> { type G<'g>; fn f<'a>(&self, x: &'a Rel<Self::G<'b>>); }
            },
            "Self :: G < 'b > : 'a",
        ),
        (
            parse_quote! {
                trait Tr<'a> { type Item; fn f(&self, x: &'a Rel<Self::Item>); }
            },
            "Self :: Item : 'a",
        ),
    ] {
        let item: ItemTrait = item;
        let referrer = of_trait(item.clone());
        for (types, outlives) in outlives(&referrer) {
            assert!(
                outlives.is_empty(),
                "{}: {:?} has {:?}",
                s(&item),
                types,
                outlives
            );
        }
        let predicates: Vec<String> = referrer
            .context_generics(referrer.context(0))
            .where_clause
            .iter()
            .flat_map(|w| &w.predicates)
            .map(s)
            .collect();
        assert_eq!(predicates, [predicate], "{}", s(&item));
    }
}

#[test]
fn a_type_which_does_not_imply_it_does_not() {
    for item in [
        // No lifetime.
        parse_quote! { trait Tr { fn f(&self, x: Rel<Self>); } },
        // The lifetime of a path (`struct Rel<'a, T>(&'a T)` has `T: 'a` inferred, but its
        // definition is not seen, so nothing is assumed): not a reference.
        parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, Self>); } },
        parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: Rel<'a, Self::Item>); } },
        parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: Rel<'a, <Self as Tr>::Item>); } },
        parse_quote! { trait Tr<'a> { fn f(&self, x: Rel<'a, Self>); } },
        parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, (u8, Vec<Self>)>); } },
        parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, Box<dyn Other<Self>>>); } },
        // Nor does a path with a lifetime make the types in its arguments outlive it for the
        // types inside them.
        parse_quote! { trait Tr<'a> { fn f(&self, x: ::core::cell::Ref<'a, Rel2<Self>>); } },
        parse_quote! { trait Tr<'a> { fn f(&self, x: ::core::cell::Ref<'a, Vec<Rel2<Self>>>); } },
        parse_quote! { trait Tr<'a> { fn f(&self, x: Rel<'a, Rel<'a, Self>>); } },
        // No `Self`.
        parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>); } },
        // Not a lifetime parameter.
        parse_quote! { trait Tr { fn f(&self, x: Rel<'static, Self>); } },
        parse_quote! { trait Tr { fn f(&self, x: for<'x> fn(Rel<'x, Self>)); } },
        parse_quote! { trait Tr { fn f(&self, x: for<'x> fn(&'x Rel<Self>)); } },
        // A projection does not require its qualified self to outlive.
        parse_quote! {
            trait Tr { fn f<'a>(&self, x: &'a <Rel2<Self> as ::core::ops::Deref>::Target); }
        },
        // `Self` is a lifetime argument's neighbour only: not a referent.
        parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a>, y: Rel2<Self>); } },
        // A bound of a generic (the generic is not a referent of the lifetime).
        parse_quote! { trait Tr { fn f<'a, T: ::core::convert::Into<Self>>(&self, x: Rel2<T>); } },
    ] {
        let item: ItemTrait = item;
        let referrer = of_trait(item.clone());
        for (types, outlives) in outlives(&referrer) {
            assert!(
                outlives.is_empty(),
                "{}: {:?} has {:?}",
                s(&item),
                types,
                outlives
            );
        }
    }
}

#[test]
fn the_context_of_a_struct_has_it_too() {
    // `SelfRequirement::Type`: `Self` is the struct, with its parameters.
    let item: ItemStruct = parse_quote! { struct MyStruct<'a, T> { f: &'a Rel<Self> } };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    let context = referrer.context(0);
    match referrer.context_self(context) {
        Some(SelfRequirement::Type(ty)) => assert_eq!(s(ty), "MyStruct < 'a , T >"),
        other => panic!("{:?}", other),
    }
    assert_eq!(referrer.context_self_outlives(context), lifetimes(&["'a"]));
    // The encoding in the implementation `impl<'a, T> Repeater<(&'a (), T, &'a Self), ..> for
    // MyStruct<'a, T>`; and `T: 'a` follows from it (`MyStruct<'a, T>: 'a`).
    let generics = referrer.context_generics(context);
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(
            &generics.params,
            &parse_quote!(Self),
            referrer.context_self_outlives(context),
        )
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a Self));
    assert_eq!(s(&encoded), s(&expected));
    // A concrete type in place of `Self` is as good.
    let self_ty = match referrer.context_self(context) {
        Some(SelfRequirement::Type(ty)) => ty,
        _ => unreachable!(),
    };
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(
            &generics.params,
            self_ty,
            referrer.context_self_outlives(context),
        )
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a MyStruct<'a, T>));
    assert_eq!(s(&encoded), s(&expected));
}

#[test]
fn it_is_part_of_the_identity_of_the_context() {
    // The same generics and `Self`, but only one implies `Self: 'a`: separate contexts. A path
    // with the lifetime implies nothing, so it shares the context of a type without one.
    let referrer = of_trait(parse_quote! {
        trait Tr<'a> {
            fn f(&self, x: &'a Rel<Self>, y: Rel2<Self>, z: &'a Rel3, w: Rel4<'a, Self>);
        }
    });
    let found = outlives(&referrer);
    assert!(found.contains(&(
        vec!["& 'a Rel < Self >".to_string()],
        vec!["'a".to_string()]
    )));
    assert!(found.contains(&(
        vec![
            "Rel2 < Self >".to_string(),
            "Rel4 < 'a , Self >".to_string()
        ],
        vec![]
    )));
    // Not for a type which does not mention `Self`: no `Self` in its context, so no bound.
    assert!(found.contains(&(vec!["& 'a Rel3".to_string()], vec![])));
    // The ids of the contexts are distinct.
    let contexts: Vec<ContextId> = referrer.contexts().collect();
    assert_eq!(contexts.len(), 3);
}

#[test]
fn the_encoders_encode_it_after_the_lifetime_bounds() {
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: &'a Rel<(Self, T)>); }
    });
    let context = referrer.context(0);
    let generics = referrer.context_generics(context);
    assert_eq!(s(generics), "< 'a , T : 'a >");
    let outlives = referrer.context_self_outlives(context);
    // The Repeater's implementation for the implementor `__S`: the parameters, the bounds and then
    // `&'a __S`.
    let marker: Type = parse_quote!(__S);
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &marker, outlives)
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a T, &'a __S));
    assert_eq!(s(&encoded), s(&expected));
    // Without it, the encoding is that of the generics alone.
    let encoded = slots()
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a T));
    assert_eq!(s(&encoded), s(&expected));
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &marker, &[])
        .unwrap();
    assert_eq!(s(&encoded), s(&expected));
    // At the use site, with the arguments substituted, and `Self` being the implementor.
    let args: Vec<GenericArgument> = vec![parse_quote!('q), parse_quote!(u8)];
    let self_ty: Type = parse_quote!(Self);
    let encoded = slots()
        .encode_generics_args_to_ty_with_self(&args, &generics.params, &self_ty, outlives)
        .unwrap();
    let expected: Type = parse_quote!((&'q (), u8, &'q u8, &'q Self));
    assert_eq!(s(&encoded), s(&expected));
    let args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
    let encoded = slots()
        .encode_generics_args_to_ty_with_self(&args, &generics.params, &self_ty, outlives)
        .unwrap();
    let expected: Type = parse_quote!((&'static (), u8, &'static u8, &'static Self));
    assert_eq!(s(&encoded), s(&expected));
    // A lifetime which is not a parameter is skipped, like a bound by one.
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &marker, &lifetimes(&["'z"]))
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a T));
    assert_eq!(s(&encoded), s(&expected));
}

#[test]
fn a_trait_object_without_a_lifetime_bound_cannot_be_encoded() {
    let generics: Generics = parse_quote!(<'a>);
    let outlives = lifetimes(&["'a"]);
    // `&'a dyn Tr` would default to `&'a (dyn Tr + 'a)`.
    let unbounded: Type = parse_quote!(dyn Tr);
    assert!(matches!(
        slots().encode_generics_params_to_ty_with_self(&generics.params, &unbounded, &outlives),
        Err(EncodeError::UnboundedTraitObject(_))
    ));
    // Nothing to encode: no error.
    assert!(slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &unbounded, &[])
        .is_ok());
    let bounded: Type = parse_quote!(dyn Tr + 'static);
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &bounded, &outlives)
        .unwrap();
    let expected: Type = parse_quote!((&'a (), &'a (dyn Tr + 'static)));
    assert_eq!(s(&encoded), s(&expected));
}

#[test]
fn the_implementation_does_not_declare_it() {
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: &'a Rel<(Self, T)>) where Self: ::core::marker::Sized; }
    });
    let context = referrer.context(0);
    // The generics of the context do not mention it (`Self` is not a generic, and the predicate
    // `Self: Sized` is the one written), and neither do those the implementation declares.
    let with_where = |g: &Generics| s(&quote!(#g #{&g.where_clause}));
    assert_eq!(
        with_where(referrer.context_generics(context)),
        "< 'a , T : 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(
        with_where(&referrer.impl_generics(context)),
        "< 'a , T > where Self : :: core :: marker :: Sized"
    );
    // For the generics of `iter_with_self()` as well.
    let marker: Type = parse_quote!(__S);
    let (_, replaced) = referrer.iter_with_self(&marker).next().unwrap();
    assert_eq!(
        with_where(&without_encoded_bounds(&replaced)),
        "< 'a , T > where __S : :: core :: marker :: Sized"
    );
    // A bound `Self: 'a` which is written is the same as one which is implied: it is moved to
    // `context_self_outlives()`, and the other bounds of the predicate are kept (see
    // `a_written_bound_is_the_same_as_an_implied_one`).
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized + 'a; }
    });
    let context = referrer.context(0);
    assert_eq!(
        with_where(referrer.context_generics(context)),
        "< 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(
        with_where(&referrer.impl_generics(context)),
        "< 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(referrer.context_self_outlives(context), lifetimes(&["'a"]));
}

#[test]
fn it_survives_serialization() {
    let referrer = of_trait(parse_quote! {
        trait Tr<'b> {
            fn f<'a>(&self, x: &'a &'b Rel<Self>, y: Rel2<Self>, z: &'a Rel<Self>);
        }
    });
    let tokens = referrer.to_token_stream();
    let text = tokens.to_string();
    assert!(text.contains("Self : 'a + 'b ;"), "{}", text);
    let parsed: Referrer = parse2(tokens.clone()).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(outlives(&parsed), outlives(&referrer));
    // Serializing again gives the same tokens.
    assert_eq!(parsed.to_token_stream().to_string(), text);
    // The order of the clauses does not matter, and the module follows them.
    let version = tag();
    let tokens: proc_macro2::TokenStream = format!(
        "\"{}\" 2 mod {{ path = [] }} ; () < 'a > ; Self : 'a ; Self : Tr ; mod 0 ; (Rel < 'a , Self >)",
        version
    )
    .parse()
    .unwrap();
    let parsed: Referrer = parse2(tokens).unwrap();
    let context = parsed.context(0);
    assert_eq!(parsed.context_self_outlives(context), lifetimes(&["'a"]));
    assert!(matches!(
        parsed.context_self(context),
        Some(SelfRequirement::Bound(_))
    ));
    let module = parsed.context_module(context).unwrap();
    assert_eq!(module.index(), 0);
    assert_eq!(parsed.module_path(module), Some(&[][..]));
    // A Referrer serialized before the clause existed has none.
    let tokens: proc_macro2::TokenStream = format!("\"{}\" Self : Tr ; (Rel < Self >)", version)
        .parse()
        .unwrap();
    let parsed: Referrer = parse2(tokens).unwrap();
    assert!(parsed.context_self_outlives(parsed.context(0)).is_empty());
}

#[test]
fn the_referrer_finds_the_entries_with_the_lifetimes_declared_where_the_copy_is() {
    // `Self: 'a` is implied by the references (the first trait), and not by the paths with a
    // lifetime (the second), whose entries have none.
    for tr in [
        parse_quote! {
            trait Tr<'t> {
                fn me<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized;
                fn field(&self, x: &'t Rel<Self>) where Self: ::core::marker::Sized;
                fn other<'a>(&self, x: &'a Rel<Self>, y: Rel2<Self>) where Self: ::core::marker::Sized;
            }
        },
        parse_quote! {
            trait Tr<'t> {
                fn me<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized;
                fn field(&self, x: Rel<'t, Self>) where Self: ::core::marker::Sized;
                fn other<'a>(&self, x: Rel<'a, Self>, y: Rel2<Self>) where Self: ::core::marker::Sized;
            }
        },
    ] {
        let tr: ItemTrait = tr;
        let mut leaker = Leaker::new();
        let top_level = leaker.intern_trait(&tr, true).unwrap();
        let referrer = leaker.finish();
        // The copy of each method is rewritten, with the lifetimes declared there or not (`'t` is
        // a generic of the trait, which the signature does not declare).
        let mut copy = tr.clone();
        referrer
            .clone()
            .visitor(top_level, |_, ix| parse_quote!(P<#ix>))
            .visit_item_trait_mut(&mut copy);
        let text = s(&copy);
        assert_eq!(text.matches("P < ").count(), 4, "{}", text);
        assert!(!text.contains("Rel <"), "{}", text);
        // Taking the generics of the trait as those of the context agrees.
        let mut copy = tr.clone();
        referrer
            .clone()
            .visitor(top_level, |_, ix| parse_quote!(P<#ix>))
            .with_outermost_generics_as_leakers()
            .visit_item_trait_mut(&mut copy);
        let text = s(&copy);
        assert_eq!(text.matches("P < ").count(), 4, "{}", text);
    }
}

/// The generics (with the where clause) of the context of the entry `ix`.
fn declared(referrer: &Referrer, ix: usize) -> String {
    let generics = referrer.generics(ix);
    let where_clause = &generics.where_clause;
    s(&quote!(#generics #where_clause))
}

#[test]
fn a_written_bound_is_the_same_as_an_implied_one() {
    // The same entry of the same trait, with `Self: 'a` written in the method or implied by the
    // reference: one context (the same id, generics, `Self` and lifetimes), and so one entry.
    let written: ItemTrait = parse_quote! {
        trait Tr { fn f<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized + 'a; }
    };
    let implied: ItemTrait = parse_quote! {
        trait Tr { fn f<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized; }
    };
    let mut leaker = Leaker::new();
    let first = leaker.intern_trait(&written, true).unwrap();
    let second = leaker.intern_trait(&implied, true).unwrap();
    assert_eq!(first, second);
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 1);
    let context = referrer.context(0);
    assert_eq!(referrer.context_self_outlives(context), lifetimes(&["'a"]));
    assert_eq!(
        declared(&referrer, 0),
        "< 'a > where Self : :: core :: marker :: Sized"
    );
    // Alone, each gives the same Referrer.
    let by_written = of_trait(written.clone());
    let by_implied = of_trait(implied.clone());
    assert_eq!(by_written, by_implied);
    assert_eq!(
        s(&by_written.to_token_stream()),
        s(&by_implied.to_token_stream())
    );
    // A path with the lifetime implies nothing, so it has the bound only if it is written: the
    // context of the type is then the same as that of the one the reference implies it for.
    let written_path: ItemTrait = parse_quote! {
        trait Tr { fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized + 'a; }
    };
    let plain_path: ItemTrait = parse_quote! {
        trait Tr { fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized; }
    };
    let by_written_path = of_trait(written_path.clone());
    let by_plain_path = of_trait(plain_path.clone());
    assert_eq!(
        by_written_path.context_self_outlives(by_written_path.context(0)),
        lifetimes(&["'a"])
    );
    assert!(by_plain_path
        .context_self_outlives(by_plain_path.context(0))
        .is_empty());
    assert_ne!(by_written_path, by_plain_path);
    assert_eq!(
        declared(&by_written_path, 0),
        "< 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(
        declared(&by_plain_path, 0),
        "< 'a > where Self : :: core :: marker :: Sized"
    );
    let mut leaker = Leaker::new();
    for item in [&written_path, &implied] {
        leaker.intern_trait(item, true).unwrap();
    }
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(referrer.context(0), referrer.context(1));

    // The predicate may be written in the where clause of the trait, or in several predicates.
    for item in [
        parse_quote! {
            trait Tr where Self: ::core::marker::Sized {
                fn f<'a>(&'a self) -> Rel<'a, Self> where Self: 'a;
            }
        },
        parse_quote! {
            trait Tr {
                fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized, Self: 'a;
            }
        },
        parse_quote! {
            trait Tr<'a> where Self: 'a {
                fn f(&self) -> Rel<'a, Self> where Self: ::core::marker::Sized;
            }
        },
    ] {
        let item: ItemTrait = item;
        let referrer = of_trait(item.clone());
        assert_eq!(referrer.iter().count(), 1, "{}", s(&item));
        let context = referrer.context(0);
        assert_eq!(
            referrer.context_self_outlives(context),
            lifetimes(&["'a"]),
            "{}",
            s(&item)
        );
        let generics = s(referrer.context_generics(context));
        assert!(
            !generics.contains("Self : 'a"),
            "{}: {}",
            s(&item),
            generics
        );
        assert!(
            !declared(&referrer, 0).contains("Self : 'a"),
            "{}",
            s(&item)
        );
        assert!(
            declared(&referrer, 0).contains("Self : :: core :: marker :: Sized"),
            "{}: {}",
            s(&item),
            declared(&referrer, 0)
        );
    }
    // Every lifetime, merged with those the type implies (`'b`, by the reference).
    let referrer = of_trait(parse_quote! {
        trait Tr<'b> {
            fn f<'a, 'c>(&self, x: &'b Rel<Self>) where Self: ::core::marker::Sized + 'c + 'a;
        }
    });
    let context = referrer.context(0);
    assert_eq!(
        referrer.context_self_outlives(context),
        lifetimes(&["'a", "'b", "'c"])
    );
    assert_eq!(
        declared(&referrer, 0),
        "< 'a , 'b , 'c > where Self : :: core :: marker :: Sized"
    );
    // A path with a lifetime implies none: only those which are written.
    let referrer = of_trait(parse_quote! {
        trait Tr<'b> {
            fn f<'a, 'c>(&self, x: Rel<'a, 'b, Self>) where Self: ::core::marker::Sized + 'c + 'a;
        }
    });
    let context = referrer.context(0);
    assert_eq!(
        referrer.context_self_outlives(context),
        lifetimes(&["'a", "'c"])
    );
    assert_eq!(
        declared(&referrer, 0),
        "< 'a , 'b , 'c > where Self : :: core :: marker :: Sized"
    );
    // Written alone, with no bound which the type implies (`Rel<'a, T>` has no `Self`, and a path
    // with a lifetime implies no bound on `T` either).
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>) where Self: 'a; }
    });
    let context = referrer.context(0);
    assert_eq!(referrer.context_self_outlives(context), lifetimes(&["'a"]));
    assert_eq!(declared(&referrer, 0), "< 'a , T >");
}

#[test]
fn a_bound_which_is_not_a_lifetime_parameter_stays_a_predicate() {
    // A bound by `'static`, by a lifetime which is not a parameter, a bound under a `for<..>`
    // binder, and a bound of another type (`Self::Item: 'a`, which `Self: 'a` implies but is not).
    // The reference implies `Self: 'a`.
    for (item, expected) in [
        (
            parse_quote! {
                trait Tr { fn f<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized + 'static; }
            },
            "< 'a > where Self : 'static + :: core :: marker :: Sized",
        ),
        (
            parse_quote! {
                trait Tr { fn f<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized, Self: 'static; }
            },
            "< 'a > where Self : 'static , Self : :: core :: marker :: Sized",
        ),
        (
            parse_quote! {
                trait Tr {
                    fn f<'a>(&self, x: &'a Rel<Self>)
                    where Self: ::core::marker::Sized, for<'x> Self: 'x;
                }
            },
            "< 'a > where Self : :: core :: marker :: Sized , for < 'x > Self : 'x",
        ),
        (
            parse_quote! {
                trait Tr {
                    type Item;
                    fn f<'a>(&self, x: &'a Rel<Self>)
                    where Self: ::core::marker::Sized, Self::Item: 'a;
                }
            },
            "< 'a > where Self : :: core :: marker :: Sized , Self :: Item : 'a",
        ),
        // The ones which are parameters are moved, the others stay in the predicate (the same for
        // a path with a lifetime, which implies nothing: the bound is the written one).
        (
            parse_quote! {
                trait Tr {
                    fn f<'a>(&self, x: &'a Rel<Self>)
                    where Self: ::core::marker::Sized + 'static + 'a;
                }
            },
            "< 'a > where Self : 'static + :: core :: marker :: Sized",
        ),
        (
            parse_quote! {
                trait Tr {
                    fn f<'a>(&self, x: Rel<'a, Self>)
                    where Self: ::core::marker::Sized + 'static + 'a;
                }
            },
            "< 'a > where Self : 'static + :: core :: marker :: Sized",
        ),
    ] {
        let item: ItemTrait = item;
        let referrer = of_trait(item.clone());
        assert_eq!(declared(&referrer, 0), expected, "{}", s(&item));
        // `Self: 'a` is implied by the reference, or written.
        assert_eq!(
            referrer.context_self_outlives(referrer.context(0)),
            lifetimes(&["'a"]),
            "{}",
            s(&item)
        );
    }
    // The path with a lifetime does not imply it, so what is not written is not there: the
    // bounds above stay predicates, and `Self: 'a` is not among the lifetimes.
    for (item, expected) in [
        (
            parse_quote! {
                trait Tr { fn f<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized + 'static; }
            },
            "< 'a > where Self : 'static + :: core :: marker :: Sized",
        ),
        (
            parse_quote! {
                trait Tr {
                    type Item;
                    fn f<'a>(&self, x: Rel<'a, Self>)
                    where Self: ::core::marker::Sized, Self::Item: 'a;
                }
            },
            "< 'a > where Self : :: core :: marker :: Sized , Self :: Item : 'a",
        ),
    ] {
        let item: ItemTrait = item;
        let referrer = of_trait(item.clone());
        assert_eq!(declared(&referrer, 0), expected, "{}", s(&item));
        assert!(
            referrer
                .context_self_outlives(referrer.context(0))
                .is_empty(),
            "{}",
            s(&item)
        );
    }
    // Written alone, a bound by a lifetime which no type mentions: no `Self`, so no bound.
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a>(&self, x: Rel2<Self>) where Self: 'a; }
    });
    assert_eq!(declared(&referrer, 0), "< 'a >");
    assert_eq!(
        referrer.context_self_outlives(referrer.context(0)),
        lifetimes(&["'a"])
    );
}

/// `Referrer::context_self_outlives()` docs: "only a bound by a lifetime parameter moves
/// here; ... a bound under `for<..>` ... stay where predicates". The bound is
/// by the parameter `'a` here, but under a binder, so it stays a predicate. Audit mutant
/// `o_written_self_forall_moves` (a written `Self: 'a` under a binder moves).
#[test]
fn a_bound_by_a_lifetime_parameter_under_a_binder_stays_a_predicate() {
    // Written in the where clause of the trait, and in the one of the method; the path with a
    // lifetime implies nothing, so `Self: 'a` is only there if it is written.
    for item in [
        parse_quote! {
            trait Tr<'a> where for<'x> Self: 'a { fn f(&self, x: Rel<'a, Self>); }
        },
        parse_quote! {
            trait Tr<'a> { fn f(&self, x: Rel<'a, Self>) where for<'x> Self: 'a; }
        },
    ] {
        let item: ItemTrait = item;
        let referrer = of_trait(item.clone());
        assert_eq!(
            declared(&referrer, 0),
            "< 'a > where for < 'x > Self : 'a",
            "{}",
            s(&item)
        );
        assert!(
            referrer
                .context_self_outlives(referrer.context(0))
                .is_empty(),
            "{}",
            s(&item)
        );
    }
    // The control: without the binder the bound moves.
    let referrer = of_trait(parse_quote! {
        trait Tr<'a> where Self: 'a { fn f(&self, x: Rel<'a, Self>); }
    });
    assert_eq!(declared(&referrer, 0), "< 'a >");
    assert_eq!(
        referrer.context_self_outlives(referrer.context(0)),
        lifetimes(&["'a"])
    );
    // The reference implies `Self: 'a` whatever else is written, and the written predicate under
    // the binder stays too.
    let referrer = of_trait(parse_quote! {
        trait Tr<'a> where for<'x> Self: 'a { fn f(&self, x: &'a Rel<Self>); }
    });
    assert_eq!(declared(&referrer, 0), "< 'a > where for < 'x > Self : 'a");
    assert_eq!(
        referrer.context_self_outlives(referrer.context(0)),
        lifetimes(&["'a"])
    );
}

#[test]
fn a_written_bound_is_encoded_and_never_declared() {
    let referrer = of_trait(parse_quote! {
        trait Tr {
            fn f<'a, T: 'a>(&self, x: Rel<'a, (Self, T)>) where Self: ::core::marker::Sized + 'a;
        }
    });
    let context = referrer.context(0);
    let outlives = referrer.context_self_outlives(context);
    assert_eq!(outlives, lifetimes(&["'a"]));
    // Neither the generics of the context nor those the implementation declares have it, and
    // `Sized` of the same predicate is kept.
    let with_where = |g: &Generics| s(&quote!(#g #{&g.where_clause}));
    assert_eq!(
        with_where(referrer.context_generics(context)),
        "< 'a , T : 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(
        with_where(&referrer.impl_generics(context)),
        "< 'a , T > where Self : :: core :: marker :: Sized"
    );
    let marker: Type = parse_quote!(__S);
    let (_, replaced) = referrer.iter_with_self(&marker).next().unwrap();
    assert_eq!(
        with_where(&without_encoded_bounds(&replaced)),
        "< 'a , T > where __S : :: core :: marker :: Sized"
    );
    // It is encoded as `&'a __S`, as the implied one is.
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(
            &referrer.context_generics(context).params,
            &marker,
            outlives,
        )
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a T, &'a __S));
    assert_eq!(s(&encoded), s(&expected));
}

/// Rewrites a copy of `tr` in the scope `top_level` with each leaked type replaced by `P<index>`,
/// as the lookup `how` finds them (`"declared"`: the generics of the trait are declared again by
/// the copy, `"leakers"`: they are those of the context), and counts those it found.
fn found(tr: &ItemTrait, referrer: &Referrer, top_level: ContextId, how: &str) -> usize {
    let mut copy = tr.clone();
    let visitor = |ix| -> Type { parse_quote!(P<#ix>) };
    match how {
        "declared" => referrer
            .clone()
            .visitor(top_level, |_, ix| visitor(ix))
            .visit_item_trait_mut(&mut copy),
        "leakers" => referrer
            .clone()
            .visitor(top_level, |_, ix| visitor(ix))
            .with_outermost_generics_as_leakers()
            .visit_item_trait_mut(&mut copy),
        other => panic!("{}", other),
    }
    s(&copy).matches("P <").count()
}

#[test]
fn the_referrer_finds_an_entry_with_a_written_bound() {
    // The copy has the bound written (in the method, or in the where clause of the trait, where
    // the `impl` does not repeat it), or it is implied by a reference (or there is none, for a
    // path with a lifetime, which implies nothing): the entry is found by both lookups, with the
    // generics of the trait declared again by the copy or taken from the context.
    for tr in [
        parse_quote! {
            trait Tr {
                fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized + 'a;
                fn g<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized, Self: 'a;
            }
        },
        parse_quote! {
            trait Tr {
                fn f<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized;
                fn g<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized;
            }
        },
        // The same with the bound written as well.
        parse_quote! {
            trait Tr {
                fn f<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized + 'a;
                fn g<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized, Self: 'a;
            }
        },
        // Nothing is written or implied: the type of a path is the entry without the bound.
        parse_quote! {
            trait Tr {
                fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized;
                fn g<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized;
            }
        },
        // Written, and not implied by the type: it makes the context depend on `Self`.
        parse_quote! {
            trait Tr {
                fn f<'a, T>(&self, x: Rel<'a, T>) where Self: 'a;
                fn g<'a, T>(&self, x: Rel<'a, T>) where Self: 'a;
            }
        },
        parse_quote! {
            trait Tr<'a> where Self: ::core::marker::Sized + 'a {
                fn f(&'a self) -> Rel<'a, Self>;
                fn g(&self, x: Rel<'a, Self>);
            }
        },
        parse_quote! {
            trait Tr<'a> where Self: ::core::marker::Sized {
                fn f(&'a self) -> &'a Rel<Self>;
                fn g(&self, x: &'a Rel<Self>);
            }
        },
    ] {
        let tr: ItemTrait = tr;
        let mut leaker = Leaker::new();
        let top_level = leaker.intern_trait(&tr, true).unwrap();
        let referrer = leaker.finish();
        assert_eq!(referrer.iter().count(), 1, "{}", s(&tr));
        for how in ["declared", "leakers"] {
            assert_eq!(
                found(&tr, &referrer, top_level, how),
                2,
                "{}: {}",
                how,
                s(&tr)
            );
        }
        // `expand()` takes the generics and `Self` of the context. A lifetime of a method is not
        // among them, so the entry that has it is not found there: visit the method instead.
        let ty = referrer.iter().next().unwrap().clone();
        let expanded = referrer.expand(top_level, ty.clone(), |_, ix| parse_quote!(P<#ix>));
        if tr.generics.params.is_empty() {
            assert_eq!(s(&expanded), s(&ty), "{}", s(&tr));
        } else {
            assert_eq!(s(&expanded), "P < 0usize >", "{}", s(&tr));
        }
        // And the same after the serialization.
        let parsed: Referrer = parse2(referrer.to_token_stream()).unwrap();
        assert_eq!(parsed, referrer);
        for how in ["declared", "leakers"] {
            assert_eq!(
                found(&tr, &parsed, top_level, how),
                2,
                "{}: {}",
                how,
                s(&tr)
            );
        }
    }
    // A bound by another lifetime of the method, which the type does not mention, is part of the
    // context too (the predicate bounding `Self` is swept in, as it was): the copy is found where
    // it is written again, and an entry without it is not found there.
    let other: ItemTrait = parse_quote! {
        trait Tr { fn f<'z, 'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized, Self: 'z; }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&other, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.context_self_outlives(referrer.context(0)),
        lifetimes(&["'a", "'z"])
    );
    for how in ["declared", "leakers"] {
        assert_eq!(found(&other, &referrer, top_level, how), 1, "{}", how);
    }
    let implied: ItemTrait = parse_quote! {
        trait Tr { fn f<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized; }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&implied, true).unwrap();
    let referrer = leaker.finish();
    for how in ["declared", "leakers"] {
        assert_eq!(found(&other, &referrer, top_level, how), 0, "{}", how);
    }
    // With a path, which implies `Self: 'a` for neither, the written `'z` alone is the context.
    let path: ItemTrait = parse_quote! {
        trait Tr { fn f<'z, 'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized, Self: 'z; }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&path, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.context_self_outlives(referrer.context(0)),
        lifetimes(&["'z"])
    );
    for how in ["declared", "leakers"] {
        assert_eq!(found(&path, &referrer, top_level, how), 1, "{}", how);
    }
}

#[test]
fn a_written_bound_still_needs_self_to_be_leakable() {
    // The bound is not among the predicates of the generics any more, but the implementation
    // needs `Self` to encode it (`&'a __S`), so it cannot be leaked where `Self` cannot (it is
    // the same as when the predicate was one of the bounds of the generics which mention `Self`).
    let leak = |items: proc_macro2::TokenStream| {
        let mut leaker = Leaker::new();
        common::intern_fns(&mut leaker, &Generics::default(), items, false)
    };
    let err = leak(quote! { fn f<'a, T>(&self, x: Rel<'a, T>) where Self: 'a; })
        .err()
        .unwrap();
    assert_eq!(
        err.note().as_deref(),
        Some("the bounds of its generics mention `Self`, which cannot be leaked")
    );
    // Without it, or with a bound which the type does not mention, it is fine (nothing in the
    // context mentions `Self`).
    for items in [
        quote! { fn f<'a, T>(&self, x: Rel<'a, T>); },
        quote! { fn f<'a, T>(&self, x: Rel<'a, T>) where Self: 'static; },
    ] {
        let result = leak(items.clone());
        assert!(result.is_ok(), "{}", items);
    }
}

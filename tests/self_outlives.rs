//! `Self: 'a`, which a leaked type implies (`Rel<'a, Self>` and `Rel<'a, Self::Item>` where
//! `struct Rel<'a, T>(&'a T)` has `T: 'a` inferred, `&'a Wrap<Self>`): `Self` is not a generic
//! parameter, so it is not among the bounds of the generics of the context, but its own part of it
//! (`Referrer::context_self_outlives()`), which the encoders encode as `&'a Self` (`&'a __S` in
//! the Repeater's implementation for the implementor of the trait), after the lifetime bounds.
//! The implementation does not declare it, as nothing proves it at a use but the encoding.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{
    without_encoded_bounds, ContextId, EncodeError, Leaker, Referrer, RepeaterSlots,
    SelfRequirement,
};

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn of_trait(item: ItemTrait) -> Referrer {
    Leaker::from_trait(&item).unwrap().0.finish()
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
    // `Rel<'a, Self>` and `Rel<'a, Self::Item>` need `Self: 'a`, with a lifetime of the method or
    // of the trait.
    for item in [
        parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, Self>); } },
        parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: Rel<'a, Self::Item>); } },
        parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: Rel<'a, <Self as Tr>::Item>); } },
        parse_quote! { trait Tr<'a> { fn f(&self, x: Rel<'a, Self>); } },
        parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, (u8, Vec<Self>)>); } },
        // The arguments of a trait object must outlive too (`dyn Tr<Self> + 'b: 'a`).
        parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, Box<dyn Other<Self>>>); } },
        // A reference is interned as a whole if its lifetime is a generic of the trait.
        parse_quote! { trait Tr<'a> { fn f(&self, x: &'a Rel<Self>); } },
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
    // Every lifetime, sorted, each once.
    let referrer = of_trait(parse_quote! {
        trait Tr<'b> { fn f<'a>(&self, x: Rel<'a, 'b, Rel<'a, Self>>); }
    });
    assert_eq!(
        outlives(&referrer)[0],
        (
            vec!["Rel < 'a , 'b , Rel < 'a , Self > >".to_string()],
            vec!["'a".to_string(), "'b".to_string()]
        )
    );
}

#[test]
fn a_type_which_does_not_imply_it_does_not() {
    for item in [
        // No lifetime.
        parse_quote! { trait Tr { fn f(&self, x: Rel<Self>); } },
        // No `Self`.
        parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>); } },
        // Not a lifetime parameter.
        parse_quote! { trait Tr { fn f(&self, x: Rel<'static, Self>); } },
        parse_quote! { trait Tr { fn f(&self, x: for<'x> fn(Rel<'x, Self>)); } },
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
    let item: ItemStruct = parse_quote! { struct MyStruct<'a, T> { f: Rel<'a, Self> } };
    let (leaker, _) = Leaker::from_struct(&item).unwrap();
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
    // The same generics and `Self`, but only one implies `Self: 'a`: separate contexts.
    let referrer = of_trait(parse_quote! {
        trait Tr<'a> {
            fn f(&self, x: Rel<'a, Self>, y: Rel2<Self>, z: &'a Rel3);
        }
    });
    let found = outlives(&referrer);
    assert!(found.contains(&(
        vec!["Rel < 'a , Self >".to_string()],
        vec!["'a".to_string()]
    )));
    assert!(found.contains(&(vec!["Rel2 < Self >".to_string()], vec![])));
    // Not for a type which does not mention `Self`: no `Self` in its context, so no bound.
    assert!(found.contains(&(vec!["& 'a Rel3".to_string()], vec![])));
    // The ids of the contexts are distinct.
    let contexts: Vec<ContextId> = referrer.contexts().collect();
    assert!(contexts.len() >= 3);
}

#[test]
fn the_encoders_encode_it_after_the_lifetime_bounds() {
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: Rel<'a, (Self, T)>); }
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
        trait Tr { fn f<'a, T>(&self, x: Rel<'a, (Self, T)>) where Self: ::core::marker::Sized; }
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
    // A bound which is written is a where predicate like another: it is kept.
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized + 'a; }
    });
    let context = referrer.context(0);
    assert_eq!(
        with_where(&referrer.impl_generics(context)),
        "< 'a > where Self : 'a + :: core :: marker :: Sized"
    );
    assert_eq!(referrer.context_self_outlives(context), lifetimes(&["'a"]));
}

#[test]
fn it_survives_serialization() {
    let referrer = of_trait(parse_quote! {
        trait Tr<'b> {
            fn f<'a>(&self, x: Rel<'a, 'b, Rel<'a, Self>>, y: Rel2<Self>, z: Rel<'a, Self>);
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
    let version = env!("CARGO_PKG_VERSION");
    let tokens: proc_macro2::TokenStream = format!(
        "\"{}\" < 'a > ; Self : 'a ; Self : Tr ; mod self ; (Rel < 'a , Self >)",
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
    assert_eq!(parsed.context_module(context).map(|m| m.len()), Some(0));
    // A Referrer serialized before the clause existed has none.
    let tokens: proc_macro2::TokenStream = format!("\"{}\" Self : Tr ; (Rel < Self >)", version)
        .parse()
        .unwrap();
    let parsed: Referrer = parse2(tokens).unwrap();
    assert!(parsed.context_self_outlives(parsed.context(0)).is_empty());
}

#[test]
fn the_referrer_finds_the_entries_with_the_lifetimes_declared_where_the_copy_is() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'t> {
            fn me<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized;
            fn field(&self, x: Rel<'t, Self>) where Self: ::core::marker::Sized;
            fn other<'a>(&self, x: Rel<'a, Self>, y: Rel2<Self>) where Self: ::core::marker::Sized;
        }
    };
    let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    let referrer = leaker.finish();
    let self_requirement = match referrer.context_self(top_level) {
        Some(requirement) => requirement.clone(),
        None => SelfRequirement::Bound(parse_quote!(Tr<'t>)),
    };
    // The copy of each method is rewritten, with the lifetimes declared there or not (`'t` is a
    // generic of the trait, which the signature does not declare).
    let mut copy = tr.clone();
    referrer
        .clone()
        .into_visitor(Some(self_requirement.clone()), |_, ix| parse_quote!(P<#ix>))
        .visit_item_trait_mut(&mut copy);
    let text = s(&copy);
    assert_eq!(text.matches("P < ").count(), 4, "{}", text);
    assert!(!text.contains("Rel <"), "{}", text);
    // The exact lookup agrees.
    let mut copy = tr.clone();
    referrer
        .clone()
        .into_visitor_with_root(
            &tr.generics,
            Some(self_requirement),
            |_, ix| parse_quote!(P<#ix>),
        )
        .with_outermost_generics_as_leakers()
        .visit_item_trait_mut(&mut copy);
    let text = s(&copy);
    assert_eq!(text.matches("P < ").count(), 4, "{}", text);
}

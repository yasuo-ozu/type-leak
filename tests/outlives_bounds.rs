//! The outlives bounds which a leaked type implies are added to the bounds of the generics it
//! uses (see `Referrer::generics()`), so that the Repeater's implementation and the copies of the
//! type have them: `T: 'a` for `&'a X` and for the type arguments `X` of a path with a lifetime
//! argument (assumed), where `X` mentions `T`, and `'b: 'a` where it mentions `'b`.

use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{
    encode_generics_params_to_ty, without_encoded_bounds, ContextId, Leaker, Referrer,
    RepeaterSlots,
};

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

/// The leaked types of a trait with the generics each uses (and the where clause).
fn leaked(item: ItemTrait) -> Vec<(String, String)> {
    let (leaker, _) = Leaker::from_trait(&item).unwrap();
    let referrer = leaker.finish();
    entries(&referrer)
}

fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter_with_generics()
        .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
        .collect()
}

fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

#[test]
fn a_path_with_a_lifetime_argument_assumes_the_bounds() {
    // `struct Rel<'a, T>(&'a T)` has `T: 'a` inferred, which the signature implies.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>); } }),
        [e("Rel < 'a , T >", "< 'a , T : 'a >")]
    );
    // The generics of the trait, and a mix of both.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } }),
        [e("Rel < 'a , T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a> { fn f<T>(&self, x: Rel<'a, T>); } }),
        [e("Rel < 'a , T >", "< 'a , T : 'a >")]
    );
    // Each type argument is bounded by each lifetime argument, and every type parameter in it.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, 'b, T, U> { fn f(&self, x: Rel<'a, 'b, (T, Vec<U>)>); }
        }),
        [e(
            "Rel < 'a , 'b , (T , Vec < U >) >",
            "< 'a , 'b , T : 'a + 'b , U : 'a + 'b >"
        )]
    );
    // Not for a lifetime argument which is not a lifetime parameter (`'static`).
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<T>(&self, x: Rel<'static, T>); } }),
        [e("Rel < 'static , T >", "< T >")]
    );
    // Not for a const parameter, which has no lifetime bound, nor a type that is not a parameter.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<const N: usize> { fn f<'a>(&self, x: Rel<'a, N, u8>); }
        }),
        [e("Rel < 'a , N , u8 >", "< 'a , const N : usize >")]
    );
    // The lifetime arguments of the path are not related to each other.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, 'b>(&self, x: Rel<'a, 'b>); } }),
        [e("Rel < 'a , 'b >", "< 'a , 'b >")]
    );
    // Nested: the inner path bounds by its own lifetime too.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, 'b, T>(&self, x: Rel<'a, Rel<'b, T>>); } }),
        [e(
            "Rel < 'a , Rel < 'b , T > >",
            "< 'a , 'b : 'a , T : 'a + 'b >"
        )]
    );
}

#[test]
fn a_reference_requires_its_referent_to_outlive_it() {
    // The reference is interned as a whole (its referent cannot stand for it) if its lifetime is
    // a generic of the Leaker.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T> { fn f(&self, x: &'a Vec<T>); } }),
        [e("& 'a Vec < T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T> { fn f(&self, x: &'a mut Vec<T>); } }),
        [e("& 'a mut Vec < T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T, U> { fn f(&self, x: &'a (Rel, T, [U; 2])); } }),
        [e("& 'a (Rel , T , [U ; 2])", "< 'a , T : 'a , U : 'a >")]
    );
    // A referent which mentions no type parameter has nothing to bound, but its lifetimes are.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, 'b> { fn f(&self, x: &'a &'b Rel); } }),
        [e("& 'a & 'b Rel", "< 'a , 'b : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a> { fn f(&self, x: &'a Rel); } }),
        [e("& 'a Rel", "< 'a >")]
    );
    // A component of a reference with a local lifetime is leaked alone, with the bound which the
    // reference implies on it (see `tests/outlives_ancestors.rs`).
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, T>(&self, x: &'a Vec<T>); } }),
        [e("Vec < T >", "< 'a , T : 'a >")]
    );
}

#[test]
fn lifetimes_in_the_arguments_are_bounded_too() {
    // `R<'a, &'b T>`: `'b: 'a` and `T: 'a` (by the path), and `T: 'b` (by the reference).
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, 'b, T>(&self, x: Rel<'a, &'b T>); } }),
        [e("Rel < 'a , & 'b T >", "< 'a , 'b : 'a , T : 'a + 'b >")]
    );
    // The lifetimes in an argument which are `'static`, elided or not generic are not.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, &'static T>); } }),
        [e("Rel < 'a , & 'static T >", "< 'a , T : 'a >")]
    );
    // The bounds written are merged with these, without duplicates.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<'a, 'b: 'a, T: 'a + ::core::clone::Clone>(&self, x: Rel<'a, &'b T>); }
        }),
        [e(
            "Rel < 'a , & 'b T >",
            "< 'a , 'b : 'a , T : 'a + 'b + :: core :: clone :: Clone >"
        )]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> where T: 'a { fn f(&self, x: Rel<'a, T>); }
        }),
        [e("Rel < 'a , T >", "< 'a , T : 'a >")]
    );
}

#[test]
fn a_trait_object_requires_its_arguments_to_outlive_too() {
    // `dyn Tr<T> + 'b` outlives `'a` only if `'b` and `T` do (rustc checks each argument, also
    // with the lifetime bound `'static`, whatever `T: 'b` says: `struct S<'a, T>(Ref<'a, dyn
    // Iterator<Item = Rel2<T>>>)` has `T: 'a` inferred, which the copy of the field needs too).
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, dyn ::p::Tr<T>>); } }),
        [e("Rel < 'a , dyn :: p :: Tr < T > >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, 'b, T> { fn f(&self, x: Rel<'a, dyn ::p::Tr<T> + 'b>); }
        }),
        [e(
            "Rel < 'a , dyn :: p :: Tr < T > + 'b >",
            "< 'a , 'b : 'a , T : 'a >"
        )]
    );
    // Also for a reference.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: &'a (dyn RelTr<T> + 'a)); }
        }),
        [e("& 'a (dyn RelTr < T > + 'a)", "< 'a , T : 'a >")]
    );
    // A trait with a lifetime argument in a bound is not a path to assume a bound for.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: Rel<Box<dyn ::p::Tr<'a, T>>>); }
        }),
        [e("Rel < Box < dyn :: p :: Tr < 'a , T > > >", "< 'a , T >")]
    );
}

#[test]
fn binder_lifetimes_are_not_generics() {
    // `'x` is declared by the binder, which the Repeater's implementation cannot refer to: `T:
    // 'x` is not a bound of `T`.
    assert_eq!(
        leaked(parse_quote! { trait Tr<T> { fn f(&self, x: for<'x> fn(Rel<'x, T>)); } }),
        [e("Rel < 'x , T >", "< 'x , T >")]
    );
    // Nor is `'x: 'a` for the argument of a path with a generic lifetime.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: Rel<'a, for<'x> fn(&'x T) -> &'x ()>); }
        }),
        [e(
            "Rel < 'a , for < 'x > fn (& 'x T) -> & 'x () >",
            "< 'a , T : 'a >"
        )]
    );
    // Nor in a `dyn for<'x> Tr<'x, ..>`, whose arguments are looked into like any other (so `T`
    // is bounded by `'a`), and in which a path with the lifetime is not assumed to bound by it.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> {
                fn f(&self, x: Rel<'a, Box<dyn for<'x> ::p::Tr<'x, Rel<'x, T>>>>);
            }
        }),
        [e(
            "Rel < 'a , Box < dyn for < 'x > :: p :: Tr < 'x , Rel < 'x , T > > > >",
            "< 'a , T : 'a >"
        )]
    );
}

#[test]
fn bounds_are_part_of_the_context() {
    // The context of the type differs from the top-level context of the trait (which has the
    // generics as they are written), although it uses all of them.
    let tr: ItemTrait = parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } };
    let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    let referrer = leaker.finish();
    assert_ne!(referrer.context(0), top_level);
    assert_eq!(s(referrer.context_generics(top_level)), "< 'a , T >");
    assert_eq!(
        s(referrer.context_generics(referrer.context(0))),
        "< 'a , T : 'a >"
    );
    // A struct with the bound written is the same context as the type.
    let st: ItemStruct = parse_quote! { struct S<'a, T: 'a>(Rel<'a, T>); };
    let (leaker, top_level) = Leaker::from_struct(&st).unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.context(0), top_level);
    // Two declarations of the same meaning give the same context (the bound is not duplicated).
    let a: ItemTrait = parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } };
    let b: ItemTrait = parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>) where T: 'a; } };
    let contexts = |item: &ItemTrait| -> Vec<ContextId> {
        let referrer = Leaker::from_trait(item).unwrap().0.finish();
        vec![referrer.context(0)]
    };
    assert_eq!(contexts(&a), contexts(&b));
    assert_eq!(leaked(a), leaked(b));
}

#[test]
fn the_referrer_finds_the_entries_with_the_generics_it_sees_declared() {
    // The Referrer sees only the generics declared by the signature, the others being those of the
    // Leaker: the bounds the type implies on the declared generics must be the Leaker's.
    for tr in [
        // `T` is declared by the method and `'a` is not.
        parse_quote! { trait Tr<'a> { fn f<T>(&self, x: Rel<'a, T>); } },
        // `'b` is declared by the method and `T` and `'a` are not.
        parse_quote! { trait Tr<'a, T> { fn f<'b>(&self, x: Rel<'b, (&'a T, u8)>); } },
        // `T` is bounded by the lifetime written at the method.
        parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } },
        // Nothing is declared.
        parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } },
    ] {
        let tr: ItemTrait = tr;
        let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
        let rewrite = |visitor: &mut dyn syn::visit_mut::VisitMut| {
            let mut copy = tr.clone();
            visitor.visit_item_trait_mut(&mut copy);
            s(&copy)
        };
        let replace = |_: Type, ix: usize| -> Type { parse_quote!(P<#ix>) };
        // The generics of the trait are not declared in the copy, as if it were a signature.
        let copy = rewrite(
            &mut referrer
                .clone()
                .into_visitor(None, replace)
                .with_outermost_generics_as_leakers(),
        );
        assert!(!copy.contains("Rel"), "{}", copy);
        // They are declared.
        let copy = rewrite(&mut referrer.clone().into_visitor(None, replace));
        assert!(!copy.contains("Rel"), "{}", copy);
        // The exact context, with the generics of the Leaker as the root.
        let copy = rewrite(
            &mut referrer
                .clone()
                .into_visitor_with_root(&tr.generics, None, replace)
                .with_outermost_generics_as_leakers(),
        );
        assert!(!copy.contains("Rel"), "{}", copy);
    }
    // A signature which declares them differently from the entry (`T` without the bound `'a`
    // which the entry has) is not rewritten, as it refers to other generics.
    let tr: ItemTrait = parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } };
    let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
    let mut sig: Signature = parse_quote!(fn f<'a, T: ::core::clone::Clone>(x: Rel<'a, T>));
    syn::visit_mut::VisitMut::visit_signature_mut(
        &mut referrer.into_visitor(None, |_, ix| parse_quote!(P<#ix>)),
        &mut sig,
    );
    assert!(s(&sig).contains("Rel"), "{}", s(&sig));
}

#[test]
fn the_bounds_survive_serialization() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> {
            fn f<'b, U>(&self, x: Rel<'b, (&'a T, U)>) -> &'a Rel2<T>;
        }
    };
    let referrer = Leaker::from_trait(&tr).unwrap().0.finish();
    let tokens = referrer.to_token_stream();
    assert!(tokens.to_string().contains("'a : 'b"), "{}", tokens);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(entries(&parsed), entries(&referrer));
}

/// The generics of the first entry of a trait with the Repeater's implementation declaring them
/// (`impl_generics()`), and those the trait arguments encode (`context_generics()`).
fn impl_and_context(item: ItemTrait) -> (String, String) {
    let referrer = Leaker::from_trait(&item).unwrap().0.finish();
    let context = referrer.context(0);
    let with_where = |g: &Generics| s(&quote!(#g #{&g.where_clause}));
    (
        with_where(&referrer.impl_generics(context)),
        with_where(referrer.context_generics(context)),
    )
}

fn pair(impl_generics: &str, context_generics: &str) -> (String, String) {
    (impl_generics.to_string(), context_generics.to_string())
}

#[test]
fn impl_generics_drop_the_encoded_bounds() {
    // `Rel<'a, T>` gives the context generics `<'a, T: 'a>`, which the trait arguments encode
    // (`&'a T`), so the implementation declares `<'a, T>`.
    assert_eq!(
        impl_and_context(parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } }),
        pair("< 'a , T >", "< 'a , T : 'a >")
    );
    // `'b: 'a` is encoded (`&'a &'b ()`) as well.
    assert_eq!(
        impl_and_context(parse_quote! {
            trait Tr { fn f<'a, 'b, T>(&self, x: Rel<'a, Rel<'b, T>>); }
        }),
        pair("< 'a , 'b , T >", "< 'a , 'b : 'a , T : 'a + 'b >")
    );
    // A bound written by the Leaker is not special.
    assert_eq!(
        impl_and_context(parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } }),
        pair("< 'a , T >", "< 'a , T : 'a >")
    );
    // `'static` is not a lifetime parameter, so it is kept, as are the trait bounds, in the order
    // of the canonical form, and the other bounds of the same parameter are kept.
    let (declared, context) = impl_and_context(parse_quote! {
        trait Tr { fn f<'a, T: 'static + ::core::clone::Clone + 'a>(&self, x: Rel<'a, T>); }
    });
    assert_eq!(context.matches("'a").count(), 2, "{}", context);
    assert!(
        context.contains("'static") && context.contains("Clone"),
        "{}",
        context
    );
    assert!(
        declared.contains("'static") && declared.contains("Clone"),
        "{}",
        declared
    );
    assert_eq!(declared.matches("'a").count(), 1, "{}", declared);
    assert_eq!(
        impl_and_context(
            parse_quote! { trait Tr { fn f<T: 'static>(&self, x: Rel<'static, T>); } }
        ),
        pair("< T : 'static >", "< T : 'static >")
    );
    // A where predicate on a type which is not a parameter is kept, as is one the bounds of the
    // parameters do not take.
    assert_eq!(
        impl_and_context(parse_quote! {
            trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>) where ::std::vec::Vec<T>: ::core::clone::Clone; }
        }),
        pair(
            "< 'a , T > where :: std :: vec :: Vec < T > : :: core :: clone :: Clone",
            "< 'a , T : 'a > where :: std :: vec :: Vec < T > : :: core :: clone :: Clone"
        )
    );
    // Nothing to drop: the generics as they are.
    assert_eq!(
        impl_and_context(
            parse_quote! { trait Tr<T: ::core::clone::Clone> { fn f(&self, x: Rel<T>); } }
        ),
        pair(
            "< T : :: core :: clone :: Clone >",
            "< T : :: core :: clone :: Clone >"
        )
    );
    // The generics of the top-level context are the call's (no bound is implied there).
    let tr: ItemTrait = parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } };
    let (leaker, top_level) = Leaker::from_trait(&tr).unwrap();
    let referrer = leaker.finish();
    assert_eq!(s(&referrer.impl_generics(top_level)), "< 'a , T >");
    // The context generics are not changed.
    assert_eq!(
        s(referrer.context_generics(referrer.context(0))),
        "< 'a , T : 'a >"
    );
}

#[test]
fn without_encoded_bounds_removes_what_the_encoders_encode() {
    let mut with_where: Generics = parse_quote!(<'a, T>);
    with_where.where_clause = Some(parse_quote!(where T: 'a, Vec<T>: Clone));
    let cases: Vec<(Generics, &str)> = vec![
        (
            parse_quote!(<'a, 'b: 'a, T: 'a + Clone, U: 'static, V: 'c, W: ?Sized + 'b, const N: usize>),
            "< 'a , 'b , T : Clone , U : 'static , V : 'c , W : ? Sized , const N : usize >",
        ),
        // Only the lifetime parameters of the list count.
        (parse_quote!(<'b: 'b>), "< 'b >"),
        (parse_quote!(<T: 'a>), "< T : 'a >"),
        // The where clause is the encoders' blind spot, so it is kept.
        (with_where, "< 'a , T > where T : 'a , Vec < T > : Clone"),
    ];
    for (generics, expected) in cases {
        let declared = without_encoded_bounds(&generics);
        assert_eq!(s(&quote!(#declared #{&declared.where_clause})), expected);
    }
    // The encoders encode exactly the bounds which are removed: the elements after the
    // parameters are as many as the bounds dropped.
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let count = |generics: &Generics| -> usize {
        generics
            .params
            .iter()
            .map(|param| match param {
                GenericParam::Lifetime(param) => param.bounds.len(),
                GenericParam::Type(param) => param.bounds.len(),
                GenericParam::Const(_) => 0,
            })
            .sum()
    };
    for generics in [
        parse_quote!(<'a, 'b: 'a, T: 'a + 'b + Clone, U: 'static, V: 'c, const N: usize>),
        parse_quote!(<'a, 'b: 'a + 'c, T: 'a>),
        parse_quote!(<'a, T: Clone + 'a, U: 'a + 'static>),
        parse_quote!(<'a, T>),
        parse_quote!(<>),
    ] {
        let generics: Generics = generics;
        let declared = without_encoded_bounds(&generics);
        let elements = |ty: Type| match ty {
            Type::Tuple(tuple) => tuple.elems.len(),
            _ => unreachable!(),
        };
        let extra = |generics: &Generics| -> usize {
            elements(encode_generics_params_to_ty(&generics.params)) - generics.params.len()
        };
        assert_eq!(
            extra(&generics),
            count(&generics) - count(&declared),
            "{}",
            s(&generics)
        );
        // Nothing is left to encode once they are removed.
        assert_eq!(extra(&declared), 0, "{}", s(&declared));
        let slotted = slots
            .encode_generics_params_to_ty(&generics.params)
            .unwrap();
        assert_eq!(
            elements(slotted) - generics.params.len(),
            count(&generics) - count(&declared)
        );
    }
}

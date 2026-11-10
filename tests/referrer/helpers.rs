//! Tests for the helpers for macro authors: the description of the Repeater impl of an entry
//! (`LeakerResult::repeater_impls()`) and the path of an entry through the Repeater
//! (`Projection::of()`).
//!
//! The pieces of the impl and the arguments of an entry are compared with the entry, its context
//! and `Projection::of()`, over all the entries of a set of items (the last tests of the
//! section on the pieces of the impls).

use crate::common;

use common::{messages, repeater, s, t, with_where};

use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// The Repeater impl of the `ix`-th entry for the Leaker 7 and the Repeater `crate::Repeater`.
fn impl_of(result: &LeakerResult, target: ImplSelf, ix: usize) -> RepeaterImpl {
    result
        .repeater_impls(&Projection::new(repeater(), 7, target))
        .unwrap()
        .remove(ix)
}

/// Analyzes a whole item the way a downstream macro does. A `mod` stands for the items it holds,
/// nested in a scope of no generics: each has its own `Self`.
fn analyze(item: &Item) -> (Referrer, ContextId) {
    let (result, context) = analyze_result(item);
    (result.finish().unwrap(), context)
}

/// The same, with the result that writes the impls.
fn analyze_result(item: &Item) -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::with_rule(common::std_config());
    let context = common::intern_item(&mut leaker, item);
    let result = leaker.result();
    assert!(
        result.errors().is_empty(),
        "{}: {}",
        s(item),
        messages(result.errors())
    );
    (result, context)
}

/// Items with lifetimes and their bounds, consts, `?Sized`, defaults, the three meanings of `Self`
/// (none, the implementor of a trait, a type), `Self: 'a`, `Self::A` under a reference, items
/// nested in a scope and an optional entry.
fn samples() -> Vec<Item> {
    vec![
        parse_quote! {
            trait Tr<'a, T: 'a, const N: usize> {
                fn f(&self, x: &'a Wrap<T, N>) -> Rel;
                fn g<U: ?Sized>(&self, u: &U) -> Rel2<U>;
            }
        },
        parse_quote! {
            trait Tr<'a> { fn f(&self, x: &'a Rel<Self>); }
        },
        parse_quote! {
            trait Tr {
                type A;
                fn f(&self) -> Wrap<Self::A>;
                fn g<'a>(&self, x: &'a Rel<Self::A>);
            }
        },
        parse_quote! {
            trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> Rel<T>; }
        },
        parse_quote! {
            trait Tr<const B: bool> { fn f(&self) -> Wrap<B>; }
        },
        parse_quote! {
            trait Tr<T: ::p::Bound<Self>>: ::core::fmt::Debug {
                fn f<U: ::core::cmp::PartialEq<Self>>(&self) -> Wrap2<U>;
                fn g(&self) -> Rel4<m!(Self, [u8; 1])>;
                fn h(&self, t: Rel5<T>);
            }
        },
        parse_quote! {
            struct S<'a, T: 'a + Clone> { f: &'a Rel<Self>, g: Wrap<T>, h: ::foo::X }
        },
        parse_quote! {
            enum E<T> { A(Wrap<Self>), B(Rel<T>) }
        },
        parse_quote! {
            union U<T: Copy> { a: ::core::mem::ManuallyDrop<Wrap<T>> }
        },
        parse_quote! {
            impl<T: Clone> ::p::Tr for W<T> { fn f(&self) -> Wrap<Self> { todo!() } }
        },
        parse_quote! {
            mod m {
                pub struct S<T> { x: Rel<T> }
                pub trait Tr {
                    fn f(&self) -> Rel2;
                    fn g<const B: bool>(&self) -> Wrap<B>;
                }
                pub type A = Rel3;
            }
        },
    ]
}

// ── entries ──────────────────────────────────────────────────────────────────────────────────

/// The impl of an entry for a marker declares nothing: no generics or where clause, and `()` as its
/// arguments.
fn assert_declares_nothing(r: &RepeaterImpl) {
    assert!(r.generics.params.is_empty());
    assert!(r
        .generics
        .where_clause
        .as_ref()
        .map_or(true, |clause| clause.predicates.is_empty()));
    assert_eq!(s(&common::impl_args(r)), "()");
}

/// The sumtype shape: `Leaker::new()` (no allowed path) and `intern_trait(..)` on a trait
/// whose only entries are the absolute types of external crates. `iter()` counts them, and an impl
/// of the Repeater for a marker declares nothing (no generics, no `Self`), with the
/// entry as the type.
#[test]
fn a_trait_with_only_external_types_has_entries_that_declare_nothing_for_a_marker() {
    let item: ItemTrait = parse_quote! {
        pub trait Reader {
            fn read(&mut self, buf: &mut [::core::primitive::u8])
                -> ::std::io::Result<::core::primitive::usize>;
            fn cause(&self) -> ::core::option::Option<&(dyn ::std::error::Error + 'static)>;
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    assert_eq!(
        common::type_texts(&referrer),
        [
            ":: std :: io :: Result < :: core :: primitive :: usize >",
            "dyn :: std :: error :: Error + 'static"
        ]
    );
    let marker: Type = parse_quote!(__Marker);
    for (ix, ty) in common::types(&referrer).enumerate() {
        assert!(common::entry(&referrer, ix).self_ty.is_none());
        assert!(common::entry(&referrer, ix).generics.params.is_empty());
        // The impl for the marker, which is what the macro emits (`type Type = #ty`).
        let r = impl_of(&result, ImplSelf::Marker(marker.clone()), ix);
        assert_declares_nothing(&r);
        assert_eq!(s(&r.self_ty), "__Marker");
        assert_eq!(s(&r.ty), s(ty));
    }
}

// ── repeater_impls ───────────────────────────────────────────────────────────────────────────

#[test]
fn repeater_impl_for_an_entry_that_does_not_name_self() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<'a, T: 'a + Clone, U> where U: Into<T> {
            fn f(&self, x: &'a Wrap<T, U>) -> Rel;
        }
    });
    let referrer = result.clone().finish().unwrap();
    let param: Ident = parse_quote!(__S);
    let wrap = common::types(&referrer)
        .position(|ty| s(ty) == "& 'a Wrap < T , U >")
        .unwrap();
    // Any type implements it: the parameter is declared last, and `T: 'a` is encoded, not
    // declared.
    let r = impl_of(&result, ImplSelf::Param(param.clone()), wrap);
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(
            <'a, T: Clone, U: Into<T>, __S: ?::core::marker::Sized>
        ))
    );
    assert_eq!(s(&r.self_ty), "__S");
    assert_eq!(
        s(&common::impl_args(&r)),
        t(parse_quote!((&'a (), T, U, &'a T, &'a U)))
    );
    assert_eq!(s(&r.ty), "& 'a Wrap < T , U >");
    // For a marker, nothing else is declared.
    let marker: Type = parse_quote!(crate::Marker);
    let r = impl_of(&result, ImplSelf::Marker(marker.clone()), wrap);
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<'a, T: Clone, U: Into<T>>))
    );
    assert_eq!(s(&r.self_ty), "crate :: Marker");
    assert_eq!(
        s(&common::impl_args(&r)),
        t(parse_quote!((&'a (), T, U, &'a T, &'a U)))
    );
    // An entry of a context with no generics: nothing is declared.
    let rel = common::types(&referrer)
        .position(|ty| s(ty) == "Rel")
        .unwrap();
    let r = impl_of(&result, ImplSelf::Marker(marker.clone()), rel);
    assert_declares_nothing(&r);
    assert_eq!(s(&r.ty), "Rel");
}

#[test]
fn repeater_impl_for_the_implementor_of_a_trait() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<'a, T> {
            type A;
            fn f(&self, x: &'a Rel<Self::A>) -> Wrap<T, Self>;
        }
    });
    let referrer = result.clone().finish().unwrap();
    let param: Ident = parse_quote!(__Impl);
    let wrap = common::types(&referrer)
        .position(|ty| s(ty) == "Wrap < T , Self >")
        .unwrap();
    // The parameter has the bound of the trait with its arguments, and replaces `Self`.
    let r = impl_of(&result, ImplSelf::Param(param.clone()), wrap);
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<'a, T, __Impl: ?::core::marker::Sized + Tr<'a, T>>))
    );
    assert_eq!(s(&r.self_ty), "__Impl");
    assert_eq!(s(&r.ty), "Wrap < T , __Impl >");
    // There is no `Self` to replace in a marker's impl: the call fails, for the first of the entries
    // that name `Self` (`Wrap<T, Self>` and `&'a Rel<Self::A>`).
    let rel = common::types(&referrer)
        .position(|ty| s(ty) == "& 'a Rel < Self :: A >")
        .unwrap();
    let marker: Type = parse_quote!(crate::Marker);
    match result.repeater_impls(&Projection::new(
        repeater(),
        7,
        ImplSelf::Marker(marker.clone()),
    )) {
        Err(EncodeError::NeedsSelf(ty)) => {
            let first = common::types(&referrer).nth(wrap.min(rel)).unwrap();
            assert_eq!(s(&*ty), s(first));
        }
        Err(other) => panic!("{:?}", other),
        Ok(_) => panic!("no error"),
    }
    // `&'a Rel<Self::A>` is under a reference: the predicate of `Self::A` is in its where clause,
    // and as the first predicate it is parenthesized, or rustc takes `where <__Impl>` for generics.
    let r = impl_of(&result, ImplSelf::Param(param.clone()), rel);
    let clause = r.generics.where_clause.as_ref().unwrap();
    assert_eq!(
        s(&clause.predicates.first().unwrap()),
        "(< __Impl > :: A) : 'a"
    );
    assert_eq!(s(&r.ty), "& 'a Rel < < __Impl > :: A >");
}

/// `repeater_impls()` for every entry of a trait: `Rel` uses no generics, so any type implements
/// it; `Wrap2<Self::Out>` names `Self`, so the implementor is bound by the trait and replaces it.
#[test]
fn repeater_impl_for_each_entry_of_a_trait() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a> {
            type Out;
            fn f(&self, x: &'a Wrap<T>) -> Rel;
            fn g(&self) -> Wrap2<Self::Out>;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();

    let self_param: Ident = parse_quote!(__S);
    let repeater = common::repeater();
    let impls: Vec<ItemImpl> = result
        .repeater_impls(&Projection::new(
            repeater.clone(),
            7,
            ImplSelf::Param(self_param.clone()),
        ))
        .unwrap()
        .into_iter()
        .map(|r| parse_quote!(#r))
        .collect();
    let (rel_ix, wrap2_ix) = (
        common::entry_id_at(&referrer, 1),
        common::entry_id_at(&referrer, 2),
    );
    let (key1, key2) = (common::key(7, rel_ix), common::key(7, wrap2_ix));
    let rel: ItemImpl = parse_quote! {
        impl<__S: ?::core::marker::Sized> crate::Repeater<(), #key1> for __S {
            type Type = Rel;
        }
    };
    let wrap2: ItemImpl = parse_quote! {
        impl<'a, T, __S: ?::core::marker::Sized + Tr<'a, T>>
            crate::Repeater<(&'a (), T, &'a T), #key2> for __S
        {
            type Type = Wrap2<<__S>::Out>;
        }
    };
    assert!(impls.contains(&rel));
    assert!(impls.contains(&wrap2));
}

#[test]
fn repeater_impl_for_a_type_is_for_the_type() {
    let (result, _) = analyze_result(&parse_quote! {
        struct S<'a, T: 'a> { f: &'a Rel<Self>, g: Wrap<T> }
    });
    let referrer = result.clone().finish().unwrap();
    let param: Ident = parse_quote!(__S);
    let rel = common::types(&referrer)
        .position(|ty| s(ty) == "& 'a Rel < Self >")
        .unwrap();
    // The impl is for the struct, with `Self` as written and as the argument of `Self: 'a`.
    let r = impl_of(&result, ImplSelf::Param(param.clone()), rel);
    assert_eq!(s(&r.self_ty), "S < 'a , T >");
    assert_eq!(with_where(&r.generics), with_where(&parse_quote!(<'a, T>)));
    assert_eq!(
        s(&common::impl_args(&r)),
        t(parse_quote!((&'a (), T, &'a T, &'a Self)))
    );
    assert_eq!(s(&r.ty), "& 'a Rel < Self >");
    // What does not name `Self` is for any type.
    let wrap = common::types(&referrer)
        .position(|ty| s(ty) == "Wrap < T >")
        .unwrap();
    let r = impl_of(&result, ImplSelf::Param(param.clone()), wrap);
    assert_eq!(s(&r.self_ty), "__S");
}

/// The call fails with the first const that has no slot, in the order of the entries, and makes the
/// impls if there is none.
#[test]
fn repeater_impls_fails_with_the_first_const_that_has_no_slot() {
    let unsupported = |generics: &str| {
        let (result, _) = analyze_result(&parse_str(&format!(
            "trait Tr{} {{ fn f(&self) -> Wrap<N>; fn g(&self) -> Wrap2<M>; fn h(&self) -> Wrap3<K>; }}",
            generics
        ))
        .unwrap());
        result.repeater_impls(&common::projection(repeater(), 7))
    };
    // `M` and `K` have no slot: the error is that of `M`, the parameter of the first entry that fails.
    match unsupported("<const N: usize, const M: ::my_crate::AliasM, const K: ::my_crate::AliasK>")
    {
        Err(EncodeError::UnsupportedConstType(ty)) => assert_eq!(s(&*ty), ":: my_crate :: AliasM"),
        other => panic!("{:?}", other.map(|impls| impls.len())),
    }
    match unsupported("<const N: ::my_crate::AliasN, const M: ::my_crate::AliasM, const K: usize>")
    {
        Err(EncodeError::UnsupportedConstType(ty)) => assert_eq!(s(&*ty), ":: my_crate :: AliasN"),
        other => panic!("{:?}", other.map(|impls| impls.len())),
    }
    // No error: the impls are made.
    let impls = unsupported("<const N: usize, const M: usize, const K: bool>").unwrap();
    assert_eq!(impls.len(), 3);
    assert_eq!(s(&impls[0].ty), "Wrap < N >");
    assert_eq!(s(&impls[1].ty), "Wrap2 < M >");
    assert_eq!(s(&impls[2].ty), "Wrap3 < K >");
}

/// The error for a marker comes first: the call fails for `Self` before it looks at the consts.
#[test]
fn repeater_impls_checks_a_marker_before_the_const_parameters() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<const N: ::my_crate::Alias> {
            fn f(&self) -> Wrap<N>;
            fn g(&self) -> Wrap2<Self>;
        }
    });
    let marker: Type = parse_quote!(crate::Marker);
    let at = |impl_self| Projection::new(repeater(), 3, impl_self);
    let error = result
        .repeater_impls(&at(ImplSelf::Marker(marker)))
        .err()
        .unwrap();
    assert!(matches!(&error, EncodeError::NeedsSelf(ty) if s(&**ty) == "Wrap2 < Self >"));
    // A parameter or a bucket does not fail for `Self`, only for the const.
    let param: Ident = parse_quote!(__S);
    for impl_self in [ImplSelf::Param(param.clone()), ImplSelf::Bucket { param }] {
        let error = result.repeater_impls(&at(impl_self)).err().unwrap();
        assert!(matches!(error, EncodeError::UnsupportedConstType(_)));
    }
}

/// Asserts that `generics` are declared in the order that every rustc accepts: the lifetimes, then
/// the types, then the consts (before 1.59 a type after a const is an error).
fn assert_declared_in_order(generics: &Generics, what: &str) {
    let rank = |param: &GenericParam| match param {
        GenericParam::Lifetime(_) => 0,
        GenericParam::Type(_) => 1,
        GenericParam::Const(_) => 2,
    };
    let ranks: Vec<_> = generics.params.iter().map(rank).collect();
    assert!(
        ranks.windows(2).all(|w| w[0] <= w[1]),
        "{}: {}",
        what,
        s(generics)
    );
}

/// Every entry of every sample, in every mode, is declared in the order of every rustc (F53 in
/// `tests/findings/findings.rs`: the parameter for the implementor came after the const
/// parameters).
#[test]
fn the_generics_of_every_impl_are_declared_in_the_order_of_every_rustc() {
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    let mut with_consts = 0;
    for item in bucket_samples() {
        let (result, _) = analyze_result(&item);
        let repeater = repeater();
        for target in [
            ImplSelf::Param(param.clone()),
            ImplSelf::Marker(marker.clone()),
            ImplSelf::Bucket {
                param: param.clone(),
            },
        ] {
            // The call for a marker fails if an entry names `Self`: only the others are compared.
            if let Ok(impls) =
                result.repeater_impls(&Projection::new(repeater.clone(), 5, target.clone()))
            {
                for r in impls {
                    let what = format!("{}: {} {:?}", s(&item), r.entry, target);
                    assert_declared_in_order(&r.generics, &what);
                    with_consts += r
                        .generics
                        .params
                        .iter()
                        .any(|param| matches!(param, GenericParam::Const(_)))
                        as usize;
                }
            }
        }
    }
    assert!(with_consts > 5, "{}", with_consts);
}

/// `ToTokens` (and `ItemImpl::from(r)`, which it writes) is the impl written by hand from the
/// pieces, with the path `Repeater<args, KEY>`, token for token, for every target.
#[test]
fn the_tokens_of_a_repeater_impl_are_the_impl_written_by_hand() {
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    let mut compared = 0;
    for item in samples().into_iter().chain(bucket_samples()) {
        let (result, _) = analyze_result(&item);
        let repeater = repeater();
        for target in [
            ImplSelf::Param(param.clone()),
            ImplSelf::Marker(marker.clone()),
            ImplSelf::Bucket {
                param: param.clone(),
            },
        ] {
            if let Ok(impls) =
                result.repeater_impls(&Projection::new(repeater.clone(), 5, target.clone()))
            {
                for r in impls {
                    let what = format!("{}: {} {:?}", s(&item), r.entry, target);
                    let (impl_generics, _, where_clause) = r.generics.split_for_impl();
                    let (args, key) = (common::impl_args(&r), common::key(5, r.entry));
                    let path: Path = parse_quote!(#repeater<#args, #key>);
                    let (self_ty, ty) = (&r.self_ty, &r.ty);
                    let by_hand = quote! {
                        impl #impl_generics #path for #self_ty #where_clause {
                            type Type = #ty;
                        }
                    };
                    assert_eq!(
                        r.to_token_stream().to_string(),
                        by_hand.to_string(),
                        "{}",
                        what
                    );
                    compared += 1;
                }
            }
        }
    }
    assert!(compared > 50, "{}", compared);
}

// ── The arguments of the Repeater ────────────────────────────────────────────────────────────

#[test]
fn the_encoder_substitutes_the_generics_of_the_trait_and_not_those_of_a_method() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a> {
            fn f<'b, U: 'b>(&self, x: &'a Wrap<T>, y: &'b Wrap2<T, U>);
        }
    };
    let (referrer, _) = analyze(&Item::Trait(tr.clone()));
    let wrap2 = common::types(&referrer)
        .position(|ty| s(ty) == "Wrap2 < T , U >")
        .unwrap();
    let args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(Vec<u8>)];
    let site = common::answers(&tr.generics.params, &args);
    // `'b` and `U` are the method's, passed by their own names.
    assert_eq!(
        s(
            &common::repeater_args_at(&repeater(), common::entry(&referrer, wrap2), site, None)
                .unwrap()
        ),
        t(parse_quote!((
            &'static (),
            &'b (),
            Vec<u8>,
            U,
            &'static Vec<u8>,
            &'b Vec<u8>,
            &'b U
        )))
    );
    // Too few arguments: a name that has no answer stays.
    let site = common::answers(&tr.generics.params, &args[..1]);
    assert_eq!(
        s(
            &common::repeater_args_at(&repeater(), common::entry(&referrer, wrap2), site, None)
                .unwrap()
        ),
        t(parse_quote!((
            &'static (),
            &'b (),
            T,
            U,
            &'static T,
            &'b T,
            &'b U
        )))
    );
}

// ── The pieces of the impls, against the entry and its projection ────────────────────────────

/// The lifetime bounds written on the parameters of `generics`: on a lifetime or on a type.
fn lifetime_bounds(generics: &Generics) -> usize {
    generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(param) => param.bounds.len(),
            GenericParam::Type(param) => param
                .bounds
                .iter()
                .filter(|bound| matches!(bound, TypeParamBound::Lifetime(_)))
                .count(),
            GenericParam::Const(_) => 0,
        })
        .sum()
}

/// The elements of a tuple type.
fn elements(ty: &Type) -> usize {
    match ty {
        Type::Tuple(tuple) => tuple.elems.len(),
        other => panic!("not a tuple: {}", s(other)),
    }
}

/// The generics with their where clause, as the impl declares them.
fn declared(generics: &Generics) -> String {
    with_where(generics).trim().to_string()
}

/// The first argument of the Repeater that `Projection::of()` writes for an entry, with
/// `self_ty` for `Self` (the entry's own generics, by their names).
fn projected(context: &Context, entry: EntryId, self_ty: &Type) -> Type {
    let repeater = repeater();
    let projection = common::projection(repeater.clone(), 7)
        .of(self_ty, context, entry)
        .unwrap();
    common::projection_args(&projection)
}

/// An impl of an entry that needs `Self: 'a`, implied by the reference or written, declares the
/// generics of the context and the parameter for the implementor, but not the bound: the argument
/// of the Repeater encodes it as `&'a __S`. A predicate of the same where clause stays, with the
/// parameter for `Self`. The context is checked by `a_written_bound_is_the_same_as_an_implied_one`
/// in `tests/output/outlives_bounds.rs`.
#[test]
fn the_impl_of_an_entry_that_needs_self_outlives_encodes_the_bound_and_never_declares_it() {
    let items: Vec<ItemTrait> = vec![
        parse_quote! {
            trait Tr {
                fn f<'a, T>(&self, x: &'a Rel<(Self, T)>) where Self: ::core::marker::Sized;
            }
        },
        parse_quote! {
            trait Tr {
                fn f<'a, T: 'a>(&self, x: Rel<'a, (Self, T)>)
                where Self: ::core::marker::Sized + 'a;
            }
        },
    ];
    for item in items {
        let (result, _) = analyze_result(&Item::Trait(item.clone()));
        let referrer = result.clone().finish().unwrap();
        let what = s(&item);
        let context = common::entry(&referrer, 0);
        // The context has the bound of `T` and the `Sized`, and `Self: 'a` apart.
        assert_eq!(
            declared(&context.generics),
            "< 'a , T : 'a > where Self : :: core :: marker :: Sized",
            "{}",
            what
        );
        assert_eq!(
            context.self_outlives.iter().map(s).collect::<Vec<_>>(),
            ["'a"],
            "{}",
            what
        );
        // The impl: neither `T: 'a` nor `Self: 'a` is declared, and `Self` is the parameter.
        let param: Ident = parse_quote!(__S);
        let r = impl_of(&result, ImplSelf::Param(param.clone()), 0);
        assert_eq!(
            declared(&r.generics),
            "< 'a , T , __S : ? :: core :: marker :: Sized + Tr > where __S : :: core :: marker :: Sized",
            "{}",
            what
        );
        assert_eq!(
            s(&common::impl_args(&r)),
            t(parse_quote!((&'a (), T, &'a T, &'a __S))),
            "{}",
            what
        );
    }
}

/// The impl declares the generics of the entry's context without the bounds that its argument of the
/// Repeater encodes: a bound by a lifetime parameter, on a type or on a lifetime. A bound by
/// `'static` or by a lifetime that is no parameter, a trait bound, a const, and a predicate that is
/// not on a parameter stay. What is dropped is exactly what the argument adds after the parameters,
/// and nothing is left to encode in what is declared.
#[test]
fn the_impl_declares_the_generics_but_the_bounds_that_the_argument_encodes() {
    let rows: Vec<(ItemTrait, &str)> = vec![
        (
            parse_quote! {
                trait Tr<
                    'a, 'b: 'a, T: 'a + ::core::clone::Clone, U: 'static, V: 'c, W: ?Sized + 'b,
                    const N: usize
                > {
                    fn f(&self) -> All<'a, 'b, T, U, V, W, N>;
                }
            },
            "< 'a , 'b , T : :: core :: clone :: Clone , U : 'static , V : 'c , W : ? Sized , const N : usize >",
        ),
        // Only the lifetime parameters of the generics count.
        (
            parse_quote!(trait Tr<'b: 'b> { fn f(&self) -> All<'b>; }),
            "< 'b >",
        ),
        (
            parse_quote!(trait Tr<T: 'a> { fn f(&self) -> All<T>; }),
            "< T : 'a >",
        ),
        (
            parse_quote!(trait Tr<'a, 'b: 'a + 'c, T: 'a> { fn f(&self) -> All<'a, 'b, T>; }),
            "< 'a , 'b : 'c , T >",
        ),
        (
            parse_quote! {
                trait Tr<'a, T: ::core::clone::Clone + 'a, U: 'a + 'static> {
                    fn f(&self) -> All<'a, T, U>;
                }
            },
            "< 'a , T : :: core :: clone :: Clone , U : 'static >",
        ),
        (parse_quote!(trait Tr<'a, T> { fn f(&self) -> All<'a, T>; }), "< 'a , T >"),
        (parse_quote!(trait Tr { fn f(&self) -> All; }), ""),
        // A where predicate on a parameter is moved to it, so it is encoded as a bound is; one
        // on another type is not, and is declared.
        (
            parse_quote! {
                trait Tr<'a, T> where T: 'a, ::std::vec::Vec<T>: ::core::clone::Clone {
                    fn f(&self) -> All<'a, T>;
                }
            },
            "< 'a , T > where :: std :: vec :: Vec < T > : :: core :: clone :: Clone",
        ),
        (
            parse_quote! {
                trait Tr<'a, T> where ::std::vec::Vec<T>: 'a { fn f(&self) -> All<'a, T>; }
            },
            "< 'a , T > where :: std :: vec :: Vec < T > : 'a",
        ),
    ];
    let marker: Type = parse_quote!(crate::Marker);
    for (item, expected) in rows {
        let what = s(&item);
        let (result, _) = analyze_result(&Item::Trait(item));
        let referrer = result.clone().finish().unwrap();
        let (entry, _, context) = referrer.iter().next().unwrap();
        let r = impl_of(&result, ImplSelf::Marker(marker.clone()), 0);
        assert_eq!(declared(&r.generics), expected, "{}", what);
        // The argument has an element for each parameter, then one for each bound dropped.
        let extra = elements(&common::impl_args(&r)) - context.generics.params.len();
        let dropped = lifetime_bounds(&context.generics) - lifetime_bounds(&r.generics);
        assert_eq!(extra, dropped, "{}", what);
        // Nothing is left to encode in what is declared: a context that has it as its generics
        // encodes the parameters only.
        let mut context = context.clone();
        context.generics = r.generics.clone();
        let args = projected(&context, entry, &marker);
        assert_eq!(elements(&args), r.generics.params.len(), "{}", what);
    }
}

/// The top-level context that the Leaker returns, and the Referrer, reach the Referrer's side as
/// tokens. The arguments of the trait written at a site are completed for the generics as written
/// (the order and the defaults are the trait's), and then go by name to the generics of each context,
/// which are in the canonical form (sorted by kind and name, no defaults): matching them by
/// position would be wrong.
#[test]
fn the_top_level_context_reaches_the_referrer_side_through_tokens() {
    use syn::parse::Parser;

    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, const N: usize, T: 'a, U = ::core::primitive::u8> {
            fn f(&self, x: Rel<'a, T>) -> Wrap<N>;
            fn g(&self) -> All<'a, N, T, U>;
        }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let tokens = quote!(#referrer, #top_level);
    let (parsed, top_level) = (|input: syn::parse::ParseStream| -> Result<(Referrer, ContextId)> {
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        let top_level = input.parse()?;
        input.parse::<syn::parse::Nothing>()?;
        Ok((referrer, top_level))
    })
    .parse2(tokens)
    .unwrap();

    // The arguments of the trait, as written at a site: the default for `U` is omitted.
    let args: AngleBracketedGenericArguments = parse_quote!(<'q, 3, u16>);
    let completed = common::completed_args(&parsed, top_level, &tr.generics, &args.args).unwrap();
    let rows: [(&str, &str, &[&str]); 3] = [
        ("Rel < 'a , T >", "< 'a , T : 'a >", &["'q", "u16"]),
        ("Wrap < N >", "< const N : usize >", &["3"]),
        (
            "All < 'a , N , T , U >",
            "< 'a , T : 'a , U , const N : usize >",
            &["'q", "u16", ":: core :: primitive :: u8", "3"],
        ),
    ];
    assert_eq!(parsed.iter().count(), rows.len());
    for (entry, generics, expected) in rows {
        let (_, _, context) = parsed.iter().find(|(_, ty, _)| s(*ty) == entry).unwrap();
        assert_eq!(s(&context.generics), generics, "{}", entry);
        let site = common::answers(&tr.generics.params, &completed);
        let args = common::generic_args(&context.generics, site).unwrap();
        assert_eq!(
            args.iter().map(s).collect::<Vec<_>>(),
            expected,
            "{}",
            entry
        );
    }
}

/// `Self` replaced by the parameter in the printed tokens of a type: `Self::A` is `<__S>::A`.
fn replace_self(text: &str) -> String {
    text.replace("Self ::", "< __S > ::").replace("Self", "__S")
}

/// The pieces of the impl of every entry of every sample, for a parameter that stands for the
/// implementor: the entry, the argument of the Repeater (the one that
/// `Projection::of()` writes with the same `Self`), the type (`Self` is the parameter where
/// the implementor is one), the type it is for, and the generics it declares: the generics of the
/// context with the parameter for the implementor before the consts, no bound that the argument
/// encodes, and no `Self` left where it is a parameter
#[test]
fn the_pieces_of_every_repeater_impl_agree_with_the_entry_and_its_projection() {
    let self_param: Ident = parse_quote!(__S);
    let mut checked = 0;
    for item in samples() {
        let (result, _) = analyze_result(&item);
        let referrer = result.clone().finish().unwrap();
        let repeater = repeater();
        let impls = result
            .repeater_impls(&Projection::new(
                repeater.clone(),
                7,
                ImplSelf::Param(self_param.clone()),
            ))
            .unwrap_or_else(|e| panic!("{}: {}", s(&item), e));
        for (n, r) in impls.into_iter().enumerate() {
            let (entry, ty, context) = referrer.iter().nth(n).unwrap();
            let what = format!("{}: {}", s(&item), s(ty));
            assert_eq!(r.entry, entry, "{}", what);

            // `Self` as the impl writes it, and the type it is for.
            let is_type = matches!(context.self_ty, Some(SelfRequirement::Type(_)));
            let (self_ty, self_in_impl): (String, Type) = match context.self_ty.as_ref() {
                Some(SelfRequirement::Type(self_ty)) => (s(self_ty), parse_quote!(Self)),
                _ => ("__S".to_string(), parse_quote!(__S)),
            };
            assert_eq!(s(&r.self_ty), self_ty, "{}", what);

            // The path has the argument of the projection and no key.
            let args = projected(context, entry, &self_in_impl);
            let path: Path = parse_quote!(#repeater<#args>);
            assert_eq!(s(&r.path), s(&path), "{}", what);

            // The leaked type, with the parameter for `Self` where `Self` is the implementor.
            let leaked = s(ty);
            let leaked = match context.self_ty {
                Some(SelfRequirement::Type(_)) | None => leaked,
                _ => replace_self(&leaked),
            };
            assert_eq!(s(&r.ty), leaked, "{}", what);

            // The generics: those of the context and the parameter for the implementor.
            let declares_param = r.generics.params.iter().any(
                |param| matches!(param, GenericParam::Type(param) if param.ident == self_param),
            );
            assert_eq!(declares_param, !is_type, "{}", what);
            assert_eq!(
                r.generics.params.len(),
                context.generics.params.len() + declares_param as usize,
                "{}",
                what
            );
            if !is_type && context.self_ty.is_some() {
                assert!(
                    !with_where(&r.generics).contains("Self"),
                    "{}: {}",
                    what,
                    with_where(&r.generics)
                );
            }
            // No bound is left that the argument encodes, and the argument has an element for each
            // bound dropped and for each `Self: 'a` of a lifetime of the generics.
            let lifetimes: Vec<&Lifetime> = r
                .generics
                .params
                .iter()
                .filter_map(|param| match param {
                    GenericParam::Lifetime(param) => Some(&param.lifetime),
                    _ => None,
                })
                .collect();
            let has_lifetime_bound = r.generics.params.iter().any(|param| match param {
                GenericParam::Lifetime(param) => {
                    param.bounds.iter().any(|bound| lifetimes.contains(&bound))
                }
                GenericParam::Type(param) => param.bounds.iter().any(|bound| {
                    matches!(bound, TypeParamBound::Lifetime(lifetime) if lifetimes.contains(&lifetime))
                }),
                GenericParam::Const(_) => false,
            });
            assert!(!has_lifetime_bound, "{}: {}", what, s(&r.generics));
            let self_elements = context
                .self_outlives
                .iter()
                .filter(|lifetime| lifetimes.contains(lifetime))
                .count();
            assert_eq!(
                elements(&common::impl_args(&r)) - context.generics.params.len(),
                lifetime_bounds(&context.generics) - lifetime_bounds(&r.generics) + self_elements,
                "{}",
                what
            );
            checked += 1;
        }
    }
    assert!(checked > 20, "{}", checked);
}

/// The arguments of the Repeater for an entry at a site are the projection's, with the names that
/// the site answers replaced: where it answers none (the scope of the analysis) they are the impl's
/// own arguments, and where it answers the generics of the item, and `Self` is the host, they are
/// the arguments for it. Both are written here, for some entries.
#[test]
fn the_arguments_at_a_site_are_the_projection_with_the_names_of_the_site_replaced() {
    // The scope of the analysis: a lifetime is `&'a ()`, a type is itself or, if it is `?Sized`,
    // `*const T`, a const is `[(); N]` or a slot of the Repeater, `T: 'a` is `&'a T`, and `Self:
    // 'a` is `&'a` and the host.
    let own_host: Type = parse_quote!(__S);
    let scope_rows: Vec<(Item, &str, &str)> = vec![
        (
            parse_quote!(
                trait Tr<'a, T: 'a, const N: usize> {
                    fn f(&self, x: &'a Wrap<T, N>);
                }
            ),
            "& 'a Wrap < T , N >",
            "(&'a (), T, [(); N], &'a T)",
        ),
        (
            parse_quote!(
                struct S<T: ?Sized, const N: u8>(Wrap<T, N>);
            ),
            "Wrap < T , N >",
            "(*const T, *const dyn crate::Repeater<(), 0, false, '\\0', N, Type = ()>)",
        ),
        (
            parse_quote!(
                trait Tr<const B: bool> {
                    fn f(&self) -> Wrap<B>;
                }
            ),
            "Wrap < B >",
            "(*const dyn crate::Repeater<(), 0, B, Type = ()>,)",
        ),
        (
            parse_quote!(
                struct S<'a, T: 'a + Clone> {
                    f: &'a Rel<Self>,
                }
            ),
            "& 'a Rel < Self >",
            "(&'a (), T, &'a T, &'a __S)",
        ),
        // The generics of a method are those of the entry's context, by their own names.
        (
            parse_quote!(
                trait Tr {
                    fn f<'a, T: 'a + ?Sized, const N: usize>(&self, x: &'a Rel<T, [u8; N]>);
                }
            ),
            "Rel < T , [u8 ; N] >",
            "(&'a (), *const T, [(); N], &'a T)",
        ),
    ];
    for (item, entry, expected) in scope_rows {
        let (referrer, _) = analyze(&item);
        let ix = common::types(&referrer)
            .position(|ty| s(ty) == entry)
            .unwrap();
        let got = common::repeater_args_at(
            &repeater(),
            common::entry(&referrer, ix),
            common::identity,
            Some(&own_host),
        )
        .unwrap();
        let expected: Type = parse_str(expected).unwrap();
        assert_eq!(s(&got), s(&expected), "{}: {}", s(&item), entry);
    }

    // At a site: the generics of the item take the arguments it gives, those of a method are passed
    // by their own names, and `Self: 'a` names the host.
    let host: Type = parse_quote!(Host<u8>);
    let items = samples();
    let rows: Vec<(usize, &str, &str, &str)> = vec![
        (
            0,
            "& 'a Wrap < T , N >",
            "<'static, u8, 3>",
            "(&'static (), u8, [(); { 3 }], &'static u8)",
        ),
        (0, "Rel2 < U >", "<'static, u8, 3>", "(*const U,)"),
        (
            1,
            "& 'a Rel < Self >",
            "<'static>",
            "(&'static (), &'static Host<u8>)",
        ),
        (
            6,
            "& 'a Rel < Self >",
            "<'static, u8>",
            "(&'static (), u8, &'static u8, &'static Host<u8>)",
        ),
        // The bound `T: 'a` of the struct is in the context of `Wrap<T>`, with its lifetime.
        (
            6,
            "Wrap < T >",
            "<'static, u8>",
            "(&'static (), u8, &'static u8)",
        ),
    ];
    for (which, entry, args, expected) in rows {
        let item = &items[which];
        let generics = match item {
            Item::Trait(item) => &item.generics,
            Item::Struct(item) => &item.generics,
            other => panic!("{}", s(other)),
        };
        let (referrer, top_level) = analyze(item);
        let ix = common::types(&referrer)
            .position(|ty| s(ty) == entry)
            .unwrap();
        let args: AngleBracketedGenericArguments = parse_str(args).unwrap();
        let completed = common::completed_args(&referrer, top_level, generics, &args.args).unwrap();
        let got = common::repeater_args_at(
            &repeater(),
            common::entry(&referrer, ix),
            common::answers(&generics.params, &completed),
            Some(&host),
        )
        .unwrap();
        let expected: Type = parse_str(expected).unwrap();
        assert_eq!(s(&got), s(&expected), "{}: {}", s(item), entry);
    }
}

// ── bucket_self and ImplSelf::Bucket ─────────────────────────────────────────────────────────

/// The samples, and the shapes that bucketing tells apart: a trait whose generics name `Self` and
/// whose default does not, a struct whose bound names its own `Self` (`SelfRequirement::Any`), an
/// impl and nested items with a trait.
fn bucket_samples() -> Vec<Item> {
    let mut items = samples();
    items.extend([
        parse_quote! {
            trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; fn g(&self) -> Wrap<Self>; }
        },
        parse_quote! {
            struct S<T: AsRef<Self>> { a: Wrap<T>, b: Rel, c: ::p::W<Self> }
        },
        parse_quote! {
            impl<T: Clone> ::p::Tr<Rel2<T>> for W<T> { fn f(&self) -> Rel3 { todo!() } }
        },
        parse_quote! {
            mod m {
                pub trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self) -> Wrap<Self::A>; }
                pub struct S<'a, T: 'a> { x: &'a Rel<T> }
            }
        },
    ]);
    items
}

/// The last segment of the primitive type of a bucket: `f32`.
fn bucket_name(ty: &Type) -> String {
    let text = t(ty.clone());
    let name = text.strip_prefix(":: core :: primitive :: ");
    name.unwrap_or_else(|| panic!("not a primitive: {}", text))
        .to_string()
}

fn names_self(generics: &Generics) -> bool {
    fn has_self(tokens: proc_macro2::TokenStream) -> bool {
        tokens.into_iter().any(|tree| match tree {
            proc_macro2::TokenTree::Ident(ident) => ident == "Self",
            proc_macro2::TokenTree::Group(group) => has_self(group.stream()),
            _ => false,
        })
    }
    let where_clause = &generics.where_clause;
    has_self(quote!(#generics #where_clause))
}

/// `Projection::of()` and `repeater_impls(ImplSelf::Bucket)` agree for every entry (`common::bucket_self()`
/// reads the bucket from the first): the
/// impl of an entry that has a bucket is the one for it as for a marker, any other is the one for a
/// parameter, an entry has a bucket exactly if neither its type nor its generics name `Self`, and
/// the call for `ImplSelf::Marker` fails exactly if some entry has none (for the first of them).
///
/// The impl for a marker is made only where no entry names `Self`: where one does, the impl of an
/// entry that has a bucket is compared with the impl for the parameter, without the parameter and
/// for the bucket.
#[test]
fn bucket_self_agrees_with_the_impl_of_every_entry() {
    let param: Ident = parse_quote!(__S);
    let (mut bucketed, mut other, mut generic_self) = (0, 0, 0);
    let (mut marked_ok, mut marked_err) = (0, 0);
    for item in bucket_samples() {
        let (result, _) = analyze_result(&item);
        let referrer = result.clone().finish().unwrap();
        for id in [0, 1, 7, 12345, u64::MAX] {
            let what = format!("{}: ({})", s(&item), id);
            // The call for a marker fails exactly if some entry has no bucket.
            let repeater = repeater();
            let marker: Type = parse_quote!(crate::Marker);
            let first_without = referrer
                .iter()
                .find(|(entry, _, _)| common::bucket_self(&referrer, id, *entry).is_none());
            let as_marker = result.repeater_impls(&Projection::new(
                repeater.clone(),
                id,
                ImplSelf::Marker(marker.clone()),
            ));
            match (&as_marker, first_without) {
                (Err(EncodeError::NeedsSelf(ty)), Some((_, first, _))) => {
                    assert_eq!(s(&**ty), s(first), "{}", what);
                    marked_err += 1;
                }
                (Ok(_), None) => marked_ok += 1,
                _ => panic!(
                    "{}: the call for a marker, and the entries without a bucket",
                    what
                ),
            }
            for ix in 0..referrer.iter().count() {
                let what = format!("{}: {} ({})", s(&item), ix, id);
                let entry = common::entry_id_at(&referrer, ix);
                let bucket = common::bucket_self(&referrer, id, entry);
                let r = result
                    .repeater_impls(&Projection::new(
                        repeater.clone(),
                        id,
                        ImplSelf::Bucket {
                            param: param.clone(),
                        },
                    ))
                    .unwrap_or_else(|e| panic!("{}: {}", what, e))
                    .remove(ix);
                let own_self = common::entry(&referrer, ix).self_ty.is_some();
                let generics_self = names_self(&common::entry(&referrer, ix).generics);
                assert_eq!(bucket.is_some(), !own_self && !generics_self, "{}", what);
                let for_param = result
                    .repeater_impls(&Projection::new(
                        repeater.clone(),
                        id,
                        ImplSelf::Param(param.clone()),
                    ))
                    .unwrap()
                    .remove(ix);
                match bucket {
                    Some(ty) => {
                        bucketed += 1;
                        bucket_name(&ty);
                        assert_eq!(r.self_ty, ty, "{}", what);
                        // The impl for a marker, where there is one.
                        if let Ok(mut impls) = result.repeater_impls(&Projection::new(
                            repeater.clone(),
                            id,
                            ImplSelf::Marker(ty.clone()),
                        )) {
                            assert_eq!(r, impls.remove(ix), "{}", what);
                        }
                        // The impl for the parameter, for the type and without the parameter.
                        let mut expected = for_param;
                        expected.self_ty = ty.clone();
                        expected.generics.params = expected
                            .generics
                            .params
                            .iter()
                            .filter(|p| !matches!(p, GenericParam::Type(t) if t.ident == param))
                            .cloned()
                            .collect();
                        // No parameter left, no angle brackets (F75).
                        if expected.generics.params.is_empty() {
                            expected.generics.lt_token = None;
                            expected.generics.gt_token = None;
                        }
                        assert_eq!(r, expected, "{}", what);
                        // No `__S`.
                        assert!(!with_where(&r.generics).contains("__S"), "{}", what);
                        // It does not depend on how often it is asked.
                        assert_eq!(
                            common::bucket_self(&referrer, id, entry),
                            Some(ty),
                            "{}",
                            what
                        );
                    }
                    None => {
                        other += 1;
                        generic_self += (!own_self) as usize;
                        assert_eq!(r, for_param, "{}", what);
                    }
                }
            }
        }
    }
    assert!(bucketed > 50 && other > 50, "{} {}", bucketed, other);
    // Both outcomes of the call for a marker were met.
    assert!(
        marked_ok > 5 && marked_err > 5,
        "{} {}",
        marked_ok,
        marked_err
    );
    // The entries of a default whose generics name `Self` while it does not: they were looked for.
    assert!(generic_self > 0);
}

/// `Projection::of()` writes the self type of the impl of every entry, for every target: `Self` as
/// given where the impl is for it (`Param`, and `Bucket` for an entry that needs `Self`), the
/// marker for `Marker`, and the bucket type for the other entries of `Bucket`.
/// With the self type of the impl given for `Self`, it is the trait path of the impl, `KEY` and
/// arguments included.
#[test]
fn projection_of_writes_the_self_type_of_the_impl_of_every_entry() {
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    let host: Type = parse_quote!(Host);
    // The entries written for the host, for the marker, for a bucket, refused, and compared with
    // an impl.
    let mut counts = [0; 5];
    for item in bucket_samples() {
        let (result, _) = analyze_result(&item);
        let referrer = result.clone().finish().unwrap();
        for id in [0, 7, 12345] {
            for target in [
                ImplSelf::Param(param.clone()),
                ImplSelf::Marker(marker.clone()),
                ImplSelf::Bucket {
                    param: param.clone(),
                },
            ] {
                let projection = Projection::new(repeater(), id, target.clone());
                // The call for a marker fails if an entry needs `Self`.
                let impls = result.repeater_impls(&projection).ok();
                for (entry, _, context) in referrer.iter() {
                    let what = format!("{}: {} ({}) {:?}", s(&item), entry, id, target);
                    let bucket = common::bucket_self(&referrer, id, entry);
                    let written = projection.of(&host, context, entry);
                    let (expected, count) = match (&target, bucket) {
                        (ImplSelf::Marker(_), None) => {
                            let error = written.err().unwrap_or_else(|| panic!("{}", what));
                            assert!(matches!(error, EncodeError::NeedsSelf(_)), "{}", what);
                            counts[3] += 1;
                            continue;
                        }
                        (ImplSelf::Marker(marker), Some(_)) => (marker.clone(), 1),
                        (ImplSelf::Bucket { .. }, Some(bucket)) => (bucket, 2),
                        _ => (host.clone(), 0),
                    };
                    let written = written.unwrap_or_else(|e| panic!("{}: {}", what, e));
                    assert_eq!(common::projection_self(&written), expected, "{}", what);
                    counts[count] += 1;

                    // The impl is for that type, or for `Self` as the impl writes it: with that in
                    // place of `Self`, the path is the one of the impl.
                    if let Some(impls) = &impls {
                        let r = impls.iter().find(|r| r.entry == entry).unwrap();
                        let self_in_impl: Type = match context.self_ty {
                            Some(SelfRequirement::Type(_)) => parse_quote!(Self),
                            _ => r.self_ty.clone(),
                        };
                        let in_impl = projection.of(&self_in_impl, context, entry).unwrap();
                        let path = ItemImpl::from(r.clone()).trait_.unwrap().1;
                        assert_eq!(
                            in_impl,
                            parse_quote!(<#self_in_impl as #path>::Type),
                            "{}",
                            what
                        );
                        if count != 0 {
                            assert_eq!(r.self_ty, expected, "{}", what);
                        }
                        counts[4] += 1;
                    }
                }
            }
        }
    }
    assert!(counts.iter().all(|count| *count > 30), "{:?}", counts);
}

/// `Projection::of()` with a marker refuses an entry that needs `Self` (the type of the entry, or
/// the generics of its context, name it), as `repeater_impls()` does, and writes the marker for the
/// others. The error has the type that would be written for the host.
#[test]
fn projection_of_with_a_marker_refuses_an_entry_that_needs_self() {
    let items: [(ItemTrait, &str); 3] = [
        // The type names it.
        (
            parse_quote!(
                trait Tr<T> {
                    fn f(&self) -> Wrap<T>;
                    fn g(&self) -> Wrap2<Self>;
                }
            ),
            "Wrap2 < Self >",
        ),
        // Only the generics of the context do.
        (
            parse_quote!(
                trait Tr<T: AsRef<Self>, U = Rel<T>> {
                    fn f(&self, t: T) -> Rel<T>;
                }
            ),
            "Rel < T >",
        ),
        // Only `Self: 'a` does.
        (
            parse_quote!(
                trait Tr<'a>
                where
                    Self: 'a,
                {
                    fn f(&self) -> Rel<'a>;
                    fn g(&self) -> Wrap<'a, u8>;
                }
            ),
            "Rel < 'a >",
        ),
    ];
    let marker: Type = parse_quote!(crate::Marker);
    let host: Type = parse_quote!(Host);
    let projection = Projection::new(repeater(), 7, ImplSelf::Marker(marker.clone()));
    let mut refused = 0;
    for (item, first) in items {
        let what = s(&item);
        let (result, _) = analyze_result(&Item::Trait(item));
        let referrer = result.clone().finish().unwrap();
        // `repeater_impls()` fails for the first entry that needs `Self`.
        match result.repeater_impls(&projection) {
            Err(EncodeError::NeedsSelf(ty)) => assert_eq!(s(&*ty), first, "{}", what),
            other => panic!("{}: {:?}", what, other.map(|impls| impls.len())),
        }
        for (entry, ty, context) in referrer.iter() {
            let written = projection.of(&host, context, entry);
            if common::bucket_self(&referrer, 7, entry).is_some() {
                let written = written.unwrap_or_else(|e| panic!("{}: {}", what, e));
                assert_eq!(common::projection_self(&written), marker, "{}", what);
                continue;
            }
            // `of()` does not know the entry's type: the error has the path written for the host.
            let key = projection.key(entry);
            let args = common::projection_args(
                &common::projection(repeater(), 7)
                    .of(&host, context, entry)
                    .unwrap(),
            );
            match written {
                Err(EncodeError::NeedsSelf(error)) => {
                    assert_eq!(
                        *error,
                        parse_quote!(<Host as crate::Repeater<#args, #key>>::Type),
                        "{}: {}",
                        what,
                        s(ty)
                    );
                    refused += 1;
                }
                other => panic!("{}: {}: {:?}", what, s(ty), other),
            }
        }
    }
    assert!(refused >= 5, "{}", refused);
}

/// The entries of one item do not all share a bucket, and the same entry of other ids does not
/// either: the type is `splitmix64(id + ix * 0x9E3779B97F4A7C15)` modulo 17 of the list.
#[test]
fn bucket_types_follow_the_documented_hash() {
    const BUCKETS: [&str; 17] = [
        "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
        "i128", "isize", "f32", "f64", "str",
    ];
    fn splitmix64(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    let (referrer, _) = analyze(&parse_quote! {
        struct S { a: Rel, b: Rel2, c: Rel3, d: Rel4 }
    });
    for id in [0u64, 1, 7, 255, 1 << 40, u64::MAX] {
        for ix in 0..referrer.iter().count() {
            let h = splitmix64(id.wrapping_add((ix as u64).wrapping_mul(0x9E3779B97F4A7C15)));
            let expected = BUCKETS[(h % 17) as usize];
            assert_eq!(
                bucket_name(
                    &common::bucket_self(&referrer, id, common::entry_id_at(&referrer, ix))
                        .unwrap()
                ),
                expected,
                "{} {}",
                id,
                ix
            );
        }
    }
    // The mapping is part of the format: these values must not change without a bump of
    // `Referrer::FORMAT_VERSION`.
    for (id, ix, expected) in [
        (0, 0, "i128"),
        (1, 0, "i32"),
        (7, 0, "bool"),
        (7, 1, "usize"),
        (0, 1, "i128"),
        (0, 2, "i16"),
        (u64::MAX, 0, "f32"),
        (u64::MAX, 3, "f64"),
    ] {
        assert_eq!(
            bucket_name(
                &common::bucket_self(&referrer, id, common::entry_id_at(&referrer, ix)).unwrap()
            ),
            expected,
            "{} {}",
            id,
            ix
        );
    }
}

/// How many entries each of the 17 buckets gets: every one is used, and none gets more than twice
/// its share.
fn assert_spread(names: impl Iterator<Item = String>, what: &str) {
    let mut counts = std::collections::BTreeMap::new();
    let mut total = 0;
    for name in names {
        *counts.entry(name).or_insert(0usize) += 1;
        total += 1;
    }
    assert_eq!(counts.len(), 17, "{}: {:?}", what, counts);
    let share = total as f64 / 17.0;
    for (name, count) in &counts {
        assert!(
            (*count as f64) <= 2.0 * share,
            "{}: {} has {} of {}: {:?}",
            what,
            name,
            count,
            total,
            counts
        );
    }
}

/// 15 entries a bucket on average, for ids and for the entries of one item.
#[test]
fn bucket_types_spread_over_ids_and_entries() {
    let (referrer, _) = analyze(&parse_quote! { struct S { a: Rel } });
    let first = common::entry_id_at(&referrer, 0);
    let bucket = |id: u64| bucket_name(&common::bucket_self(&referrer, id, first).unwrap());
    assert_spread((0..255u64).map(bucket), "ids 0..255");
    assert_spread((0..255u64).map(|k| k << 8).map(bucket), "k << 8");
    assert_spread((0..255u64).map(|k| k << 16).map(bucket), "k << 16");
    assert_spread((0..255u64).map(|k| k << 32).map(bucket), "k << 32");
    assert_spread((0..255u64).map(|k| k * 17).map(bucket), "k * 17");
    // The entries of one item, with one id.
    let fields: Vec<_> = (0..255)
        .map(|n| {
            let (name, ty) = (
                common::ident(&format!("f{}", n)),
                common::ident(&format!("R{}", n)),
            );
            quote!(#name: #ty,)
        })
        .collect();
    let (referrer, _) = analyze(&parse_quote! { struct Big { #(#fields)* } });
    assert_eq!(referrer.iter().count(), 255);
    for id in [0u64, 7, u64::MAX] {
        assert_spread(
            referrer
                .iter()
                .map(|(ix, _, _)| bucket_name(&common::bucket_self(&referrer, id, ix).unwrap())),
            "the entries of one item",
        );
    }
}

/// `ImplSelf` owns its data, and is `Clone`, `Debug` and `Eq`.
#[test]
fn impl_self_is_clone_debug_and_eq() {
    let param: Ident = parse_quote!(__S);
    let target = ImplSelf::Bucket {
        param: param.clone(),
    };
    let copy = target.clone();
    assert_eq!(format!("{:?}", target), format!("{:?}", copy));
    assert!(format!("{:?}", target).contains("Bucket"));
    assert_eq!(target, copy);
    assert_ne!(target, ImplSelf::Param(param));
    fn is_owned<T: Clone + Eq + 'static>() {}
    is_owned::<ImplSelf>();
}

// ── An entry whose generics name Self ────────────────────────────────────────────────────────

/// The entry of `trait Tr<T: AsRef<Self>, U = Rel<T>>`: the default `Rel<T>` does not name `Self`,
/// but the generics of its context do (like the entry `Wrap<Self>`). With a concrete type that
/// `Self` would be that type, so there is no bucket and `Marker` is refused: `impl<T: AsRef<Self>>
/// Repeater<..> for Marker` bounds `T: AsRef<Marker>`, where the copy has `T: AsRef<Host>`, so an
/// impl of the copy that instantiates the default (`impl Tr<String> for Host`) fails with E0277
/// (`String: AsRef<Marker>`). The impl is for a parameter, which is the implementor of the trait in
/// the copy. (`bucket_self_agrees_with_the_impl_of_every_entry` checks the bucket and the targets.)
#[test]
fn the_entry_of_a_default_whose_generics_name_self_is_for_a_parameter() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; fn g(&self) -> Wrap<Self>; }
    });
    let referrer = result.clone().finish().unwrap();
    let rel = common::types(&referrer)
        .position(|ty| s(ty) == "Rel < T >")
        .unwrap();
    // A bucket falls back to a parameter, for which `Self` of the generics is the parameter.
    let param: Ident = parse_quote!(__S);
    let r = impl_of(&result, ImplSelf::Bucket { param }, rel);
    assert_eq!(s(&r.self_ty), "__S");
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<T: AsRef<Self>, __S: ?::core::marker::Sized>))
    );
}

/// An entry whose impl replaces `Self` by the parameter for the implementor has its const
/// parameters encoded as any other. (The default of a const parameter, which names `Self` here, is
/// not in the context, so the replacement changes no const parameter.)
#[test]
fn the_const_parameters_of_an_impl_that_replaces_self_are_encoded_as_any_other() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<const N: usize = { Self::LEN }> {
            const LEN: usize;
            fn f(&self) -> Wrap<N, Self>;
        }
    });
    let impls = result
        .repeater_impls(&common::projection(repeater(), 7))
        .unwrap();
    assert_eq!(impls.len(), 1);
    assert_eq!(
        s(&common::impl_args(&impls[0])),
        t(parse_quote!(([(); N],)))
    );
}

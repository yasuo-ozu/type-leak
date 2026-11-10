//! Tests for the defaults of type parameters, analyzed with the parameters before them, and for
//! `complete_params()` on the parameters of the visited trait.

mod common;

use syn::*;
use template_quote::quote;
use type_leak::{ContextId, Leaker, Referrer};

fn referrer_of(tr: &ItemTrait) -> Referrer {
    let mut leaker = Leaker::new();
    leaker.intern_trait(tr);
    leaker.result().finish().unwrap()
}

/// The context of the trait that `referrer_of()` analyzed: the top-level one, numbered after the
/// contexts of the entries.
fn top_level(referrer: &Referrer) -> ContextId {
    common::context_ids(referrer).last().copied().unwrap()
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn parse_generics(tokens: proc_macro2::TokenStream) -> Generics {
    parse2(tokens).unwrap()
}

fn args(args: AngleBracketedGenericArguments) -> Vec<GenericArgument> {
    args.args.into_iter().collect()
}

/// A call of `type_fn`: the leaked type and its index.
type Call = (String, usize);

/// Completes `given` with projections `<Self as R<ix>>::Type` written by the visitor in the
/// defaults of the parameters of `tr`, and returns the calls of `type_fn`.
fn complete_calls(
    referrer: &Referrer,
    tr: &ItemTrait,
    given: &[GenericArgument],
) -> Result<(Vec<String>, Vec<Call>)> {
    let mut calls = Vec::new();
    let completed =
        common::completed_args(referrer, top_level(referrer), tr, given, |ty, ix, _| {
            calls.push((s(&ty), ix.index()));
            parse_quote!(<Self as R<#ix>>::Type)
        })
        .map_err(Error::from)?;
    Ok((completed.iter().map(s).collect(), calls))
}

fn complete(referrer: &Referrer, tr: &ItemTrait, given: &[GenericArgument]) -> Result<Vec<String>> {
    complete_calls(referrer, tr, given).map(|(completed, _)| completed)
}

#[test]
fn test_default_requirements() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: ::core::clone::Clone, const N: usize = 3, U = Rel<T>, V = (RelB, U)>
        where
            T: ::p::Conv<RelW>,
            U: ::p::Late,
        {
            fn f(&self) -> RelC;
            fn g(&self) -> All<'a, T, U, V, N>;
        }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    // The contexts are the requirements of the types, numbered in the order of their first
    // appearance: nothing (`RelB`, `RelW` and `RelC`), the generics of the trait, which `All`
    // uses exactly (it is the top-level context), and `Rel<T>` (the default of `U`, which uses the
    // parameters before it). The default of `V`, `(RelB, U)`, is not interned, as `RelB` is
    // enough (it is rebuilt around it), so its context (`<.., U: ::p::Late>`) is not in the
    // Referrer. The default of `N` is a const.
    assert_eq!(common::context_ids(&referrer).len(), 3);
    assert_eq!(top_level.index(), 1);
    let leaked: Vec<_> = common::types(&referrer)
        .enumerate()
        .map(|(ix, ty)| (common::entry_id(&referrer, ix).index(), s(ty)))
        .collect();
    assert_eq!(
        leaked,
        [
            (0, "RelB".to_string()),
            (0, "RelC".to_string()),
            (0, "RelW".to_string()),
            (1, "All < 'a , T , U , V , N >".to_string()),
            (2, "Rel < T >".to_string()),
        ]
    );
    // The generics of the context of the entry `ix`.
    let generics = |ix: usize| {
        let generics = &common::entry(&referrer, ix).generics;
        s(&quote!(#generics #{&generics.where_clause}))
    };
    assert_eq!(generics(0), "");
    // The bounds of `T` (also in the where clause) and the predicates mentioning it are used.
    assert_eq!(
        generics(4),
        s(&parse_generics(
            quote!(<T: ::core::clone::Clone + ::p::Conv<RelW>>)
        ))
    );
    // The defaults are removed, and the bounds are moved to the parameters.
    assert_eq!(common::entry_id(&referrer, 3), top_level);
    assert_eq!(
        generics(3),
        s(&parse_generics(quote!(
            <'a, T: ::core::clone::Clone + ::p::Conv<RelW>, U: ::p::Late, V, const N: usize>
        )))
    );
    assert!(referrer
        .iter()
        .all(|(_, _, context)| context.self_ty.is_none()));

    // The round trip keeps the requirements.
    let tokens = quote!(#referrer);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    for ((_, _, read), (_, _, context)) in parsed.iter().zip(referrer.iter()) {
        assert_eq!(s(&read.generics), s(&context.generics));
        assert_eq!(
            s(&read.generics.where_clause),
            s(&context.generics.where_clause)
        );
    }
}

/// `complete_params()`: a default is analyzed with the parameters before it, and cannot
/// name a later one. A where predicate that names the later parameter `U` (`T: Conv<U>`) is not
/// in the context of the default `Rel<T>`, which has `T` alone. Audit mutant
/// `default_root_all_params` (the default is analyzed with all the parameters as its root).
#[test]
fn test_default_is_analyzed_with_the_parameters_before_it() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U = Rel<T>> where T: ::p::Conv<U> { fn f(&self); }
    };
    let referrer = referrer_of(&tr);
    assert_eq!(
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
            .collect::<Vec<_>>(),
        [("Rel < T >".to_string(), "< T >".to_string())]
    );
    // The predicate is on `U`, a parameter the default of `V` has, and names `V` itself, which it
    // has not: it is left out of the context of the default.
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U = Rel<T>, V = Rel2<U>> where U: ::p::Conv<V> { fn f(&self); }
    };
    let referrer = referrer_of(&tr);
    assert_eq!(
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
            .collect::<Vec<_>>(),
        [
            ("Rel < T >".to_string(), "< T >".to_string()),
            ("Rel2 < U >".to_string(), "< U >".to_string()),
        ]
    );
}

#[test]
fn test_default_is_shared_with_the_items() {
    // `Rel<T>` in the default of `U` and in a method is the same type with the same requirements.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<T>> {
            fn f(&self, t: T) -> Rel<T>;
            fn g(&self, t: T, u: U) -> Both<T, U>;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    let at = |text: &str| {
        common::types(&referrer)
            .position(|ty| s(ty) == text)
            .unwrap()
    };
    let (rel, both) = (at("Rel < T >"), at("Both < T , U >"));
    // `Rel<T>` uses `T` only, whereas the generics of the trait are `<T, U>`, which `Both` uses.
    assert_eq!(common::context_ids(&referrer).len(), 2);
    assert_ne!(common::entry_id(&referrer, rel), top_level);
    assert_eq!(common::entry_id(&referrer, both), top_level);
    assert_eq!(common::entry(&referrer, both).generics.params.len(), 2);
    assert_eq!(common::entry(&referrer, rel).generics.params.len(), 1);
}

#[test]
fn test_default_cannot_depend_on_self() {
    // A leaked type of a default mentioning `Self` is an error.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<Self>> {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1);
    assert_eq!(
        err[0].message,
        "`Rel<Self>` is in the default of a parameter, and names `Self`"
    );
    assert_eq!(
        err[0].notes.first().map(|(_, n)| n.as_str()),
        Some("a default of a parameter cannot depend on `Self`")
    );
    // Nothing is leaked in `::p::W<Self>` or `Self`, which are copied, and in `(Rel, Self)` only
    // `Rel` is.
    for (default, leaked) in [
        (quote!(::p::W<Self>), 0),
        (quote!(Self), 0),
        (quote!((Rel, Self)), 1),
    ] {
        let tr: ItemTrait = parse_quote!(trait Tr<T, U = #default> {});
        let referrer = referrer_of(&tr);
        assert_eq!(referrer.iter().count(), leaked, "{}", s(&default));
        assert!(!referrer.has_self());
    }
}

#[test]
fn test_defaults_of_several_calls_are_shared() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = ::std::vec::Vec<T>> {
            fn f(&self) -> Rel;
        }
    );
    let plain: ItemStruct = parse_quote!(
        struct S<T> {
            a: RelA,
        }
    );
    let mut leaker = Leaker::new();
    let top_plain = leaker.intern_struct(&plain);
    let top_tr = leaker.intern_trait(&tr);
    // A failed call adds no entry (its error is the only one).
    let bad: ItemTrait = parse_quote!(
        trait Tr<T, U = ::std::vec::Vec<T>> {
            fn f(&self) -> Rel<'_>;
        }
    );
    leaker.intern_trait(&bad);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let referrer = result.into_referrer();
    // The types use no generics, so they are in one context, whichever call found them, and the
    // generics of the calls (without the defaults) are two other contexts. The default of `U`,
    // the type of an external crate, is an optional entry (after the required ones), which uses
    // `T`: it is in the context of the generics of `S<T>` (old: it was no entry, and that context
    // had none).
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(common::context_ids(&referrer).len(), 3);
    assert_eq!(common::entry(&referrer, 0).generics.params.len(), 0);
    assert_eq!(common::entry(&referrer, 1).generics.params.len(), 0);
    assert_eq!(
        s(common::types(&referrer).nth(2).unwrap()),
        ":: std :: vec :: Vec < T >"
    );
    assert_eq!(common::entry_id(&referrer, 2), top_plain);
    assert_eq!(common::entry(&referrer, 2).generics.params.len(), 1);
    assert_ne!(top_tr, top_plain);
    // A macro that writes the default as it is, at the site (`complete_params()` replaces the
    // parameters before it), as in any call.
    let completed = common::completed_args(
        &referrer,
        top_tr,
        &tr,
        &args(parse_quote!(<u8>)),
        |ty, ix, _| {
            assert_eq!(ix.index(), 2);
            ty
        },
    )
    .unwrap();
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        ["u8", ":: std :: vec :: Vec < u8 >"]
    );
    // A macro that replaces the default by the entry.
    let mut replaced = Vec::new();
    let completed = common::completed_args(
        &referrer,
        top_tr,
        &tr,
        &args(parse_quote!(<u8>)),
        |ty, ix, _| {
            replaced.push((s(&ty), ix.index()));
            parse_quote!(P)
        },
    )
    .unwrap();
    assert_eq!(completed.iter().map(s).collect::<Vec<_>>(), ["u8", "P"]);
    assert_eq!(replaced, [(":: std :: vec :: Vec < T >".to_string(), 2)]);
}

#[test]
fn test_default_const_is_kept() {
    // A relative path in the default of a const parameter is reported, as before.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize = MY_N> {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let result = leaker.result();
    assert!(result.finish().is_err());
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize = 3, const M: usize = { 1 + 2 }> {}
    );
    let referrer = referrer_of(&tr);
    let completed = complete(&referrer, &tr, &args(parse_quote!(<u8>))).unwrap();
    assert_eq!(completed, ["u8", "3", "{ 1 + 2 }"]);
}

#[test]
fn test_complete_params() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T, const N: usize = 3, U = Rel<T, N>, V = (RelB, U, T)> {
            fn f(&self) -> RelC;
        }
    };
    let referrer = referrer_of(&tr);

    // Omitted arguments take the defaults, rewritten with the entries which use the preceding
    // parameters. `type_fn` receives the leaked type and its index, once for each entry in the
    // defaults of the parameters, whatever the arguments are.
    let (completed, calls) = complete_calls(&referrer, &tr, &args(parse_quote!(<'x, u8>))).unwrap();
    let projection = |ix: usize| format!("< Self as R < {}usize > > :: Type", ix);
    assert_eq!(
        completed,
        [
            "'x".to_string(),
            "u8".to_string(),
            "3".to_string(),
            projection(2),
            format!("({} , {} , u8)", projection(0), projection(2)),
        ]
    );
    assert_eq!(
        calls,
        [
            ("Rel < T , N >".to_string(), 2),
            // The tuple is rebuilt around `RelB`, which is interned.
            ("RelB".to_string(), 0),
        ]
    );

    // Explicit arguments are kept as they are (a bare `N` is a const argument).
    let completed = complete(&referrer, &tr, &args(parse_quote!(<'x, u8, 5, i32, i64>))).unwrap();
    assert_eq!(completed, ["'x", "u8", "5", "i32", "i64"]);

    // Only `V` is omitted.
    let completed = complete(&referrer, &tr, &args(parse_quote!(<'x, u8, 5, i32>))).unwrap();
    assert_eq!(completed[..4], ["'x", "u8", "5", "i32"]);
    assert_eq!(completed[4], format!("({} , i32 , u8)", projection(0)));

    // A parameter named like a leaked type is not substituted inside a projection: here `T`
    // is the argument (a type named `U`), and `U` is the omitted argument.
    let completed = complete(&referrer, &tr, &args(parse_quote!(<'x, U>))).unwrap();
    assert_eq!(completed[1], "U");
    assert_eq!(completed[3], projection(2));
    assert_eq!(
        completed[4],
        format!("({} , {} , U)", projection(0), projection(2))
    );
}

#[test]
fn test_complete_params_const_bare_ident_and_errors() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize, U = Rel<T>> {
            fn f(&self);
        }
    );
    let referrer = referrer_of(&tr);
    // A bare identifier for a const parameter is a const argument.
    let completed = complete(&referrer, &tr, &args(parse_quote!(<u8, M, u16>))).unwrap();
    assert_eq!(completed, ["u8", "M", "u16"]);
    // Missing and unexpected arguments.
    let err = complete(&referrer, &tr, &args(parse_quote!(<u8>))).unwrap_err();
    assert!(err.to_string().contains("missing generic argument"));
    let err = complete(&referrer, &tr, &args(parse_quote!(<u8, 1, u16, u32>))).unwrap_err();
    assert!(err.to_string().contains("unexpected generic argument"));
    let err = complete(&referrer, &tr, &args(parse_quote!(<'a, u8, 1>))).unwrap_err();
    assert!(err.to_string().contains("unexpected generic argument"));
}

#[test]
fn test_complete_params_substitutes_the_copied_part_only() {
    // A leaked type inside a type which is copied (it cannot be leaked as a whole), twice.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = ::p::W<Rel<T>, &'_ T, Rel<T>>> {}
    );
    let referrer = referrer_of(&tr);
    let completed = common::completed_args(
        &referrer,
        top_level(&referrer),
        &tr,
        &args(parse_quote!(<Rel<u8>>)),
        |ty, ix, _| {
            assert_eq!(ix.index(), 0);
            assert_eq!(s(&ty), "Rel < T >");
            parse_quote!(P<T>)
        },
    )
    .unwrap();
    // `T` is substituted in the copied part and in the projections alike.
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        [
            "Rel < u8 >",
            ":: p :: W < P < Rel < u8 > > , & '_ Rel < u8 > , P < Rel < u8 > > >"
        ]
    );
    // What the visitor writes for the entry may be any type that names `T`, and it is substituted
    // with the rest of the default.
    let completed = common::completed_args(
        &referrer,
        top_level(&referrer),
        &tr,
        &args(parse_quote!(<u8>)),
        |_, _, _| parse_quote!(<T as Proj>::Type),
    )
    .unwrap();
    assert_eq!(
        s(&completed[1]),
        ":: p :: W < < u8 as Proj > :: Type , & '_ u8 , < u8 as Proj > :: Type >"
    );
}

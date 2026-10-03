//! Tests for the defaults of type parameters, analyzed with the parameters before them, and for
//! `Referrer::complete_args()`.

use syn::*;
use template_quote::quote;
use type_leak::{Leaker, Referrer};

fn referrer(tr: &ItemTrait) -> Referrer {
    Leaker::from_trait(tr).unwrap().finish()
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

/// A call of `type_fn`: the leaked type, its index and the arguments of the preceding parameters.
type Call = (String, usize, Vec<String>);

/// Completes `given` with projections `<Self as R<ix>>::Type`, and returns the calls of `type_fn`.
fn complete_calls(
    referrer: &Referrer,
    params: &punctuated::Punctuated<GenericParam, Token![,]>,
    given: &[GenericArgument],
) -> Result<(Vec<String>, Vec<Call>)> {
    let mut calls = Vec::new();
    let completed = referrer
        .complete_args(given, params, |ty, ix, prefix| {
            calls.push((s(&ty), ix, prefix.iter().map(s).collect()));
            parse_quote!(<Self as R<#ix>>::Type)
        })
        .map_err(Error::from)?;
    Ok((completed.iter().map(s).collect(), calls))
}

fn complete(
    referrer: &Referrer,
    params: &punctuated::Punctuated<GenericParam, Token![,]>,
    given: &[GenericArgument],
) -> Result<Vec<String>> {
    complete_calls(referrer, params, given).map(|(completed, _)| completed)
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
        }
    };
    let referrer = referrer(&tr);
    // The contexts are the requirements of the types, numbered in the order of their first
    // appearance: nothing (`RelB`, `RelW` and `RelC`) and `Rel<T>` (the default of `U`, which uses
    // the parameters before it). The default of `V`, `(RelB, U)`, is not interned, as `RelB` is
    // enough (it is rebuilt around it). The default of `N` is a const.
    assert_eq!(referrer.n_contexts(), 2);
    let leaked: Vec<_> = referrer
        .iter()
        .enumerate()
        .map(|(ix, ty)| (referrer.context(ix), s(ty)))
        .collect();
    assert_eq!(
        leaked,
        [
            (0, "RelB".to_string()),
            (0, "RelC".to_string()),
            (0, "RelW".to_string()),
            (1, "Rel < T >".to_string()),
        ]
    );
    let generics = |c: usize| {
        let generics = referrer.context_generics(c);
        s(&quote!(#generics #{&generics.where_clause}))
    };
    assert_eq!(generics(0), "");
    // The bounds of `T` (also in the where clause) and the predicates mentioning it are used.
    assert_eq!(
        generics(1),
        s(&parse_generics(
            quote!(<T: ::core::clone::Clone + ::p::Conv<RelW>>)
        ))
    );
    assert!((0..2).all(|c| referrer.context_self(c).is_none()));

    // The round trip keeps the requirements.
    let tokens = quote!(#referrer);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    for context in 0..2 {
        assert_eq!(
            s(parsed.context_generics(context)),
            s(referrer.context_generics(context))
        );
        assert_eq!(
            s(&parsed.context_generics(context).where_clause),
            s(&referrer.context_generics(context).where_clause)
        );
    }
}

#[test]
fn test_default_is_shared_with_the_items() {
    // `Rel<T>` in the default of `U` and in a method is the same type with the same requirements.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<T>> {
            fn f(&self, t: T) -> Rel<T>;
        }
    );
    let referrer = referrer(&tr);
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(referrer.n_contexts(), 1);
}

#[test]
fn test_default_cannot_depend_on_self() {
    // A leaked type of a default mentioning `Self` is an error.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<Self>> {}
    );
    let err = Leaker::from_trait(&tr).err().unwrap();
    assert_eq!(err.to_string(), "use absolute path: 'Self'");
    assert_eq!(
        err.note().as_deref(),
        Some("a default of a parameter cannot depend on `Self`")
    );
    let mut leaker = Leaker::from_config({
        let mut config = type_leak::LeakerConfig::new();
        config.allow_external_crates(false);
        config
    });
    let err = leaker
        .intern_with(&tr.generics, |v| {
            syn::visit::Visit::visit_item_trait(v, &tr)
        })
        .err()
        .unwrap();
    assert!(matches!(err, type_leak::NotInternableError::NotPortable(_)));

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

fn referrer_of(tr: &ItemTrait) -> Referrer {
    Leaker::from_trait(tr).unwrap().finish()
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
    leaker
        .intern_with(&plain.generics, |v| {
            syn::visit::Visit::visit_item_struct(v, &plain)
        })
        .unwrap();
    leaker
        .intern_with(&tr.generics, |v| {
            syn::visit::Visit::visit_item_trait(v, &tr)
        })
        .unwrap();
    // A failed call adds nothing.
    let bad: ItemTrait = parse_quote!(
        trait Tr<T, U = ::std::vec::Vec<T>> {
            fn f(&self) -> Rel<'_>;
        }
    );
    assert!(leaker
        .intern_with(&bad.generics, |v| syn::visit::Visit::visit_item_trait(
            v, &bad
        ))
        .is_err());
    let referrer = leaker.finish();
    // The types use no generics, so they are in one context, whichever call found them.
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(referrer.n_contexts(), 1);
    assert_eq!(referrer.generics(0).params.len(), 0);
    // The default is copied (nothing in it is leaked), as in any call.
    let completed = referrer
        .complete_args(
            &args(parse_quote!(<u8>)),
            &tr.generics.params,
            |_, _, _| unreachable!(),
        )
        .unwrap();
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        ["u8", ":: std :: vec :: Vec < u8 >"]
    );
}

#[test]
fn test_default_const_is_kept() {
    // A relative path in the default of a const parameter is reported, as before.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize = MY_N> {}
    );
    assert!(Leaker::from_trait(&tr).is_err());
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize = 3, const M: usize = { 1 + 2 }> {}
    );
    let referrer = referrer(&tr);
    let completed = complete(&referrer, &tr.generics.params, &args(parse_quote!(<u8>))).unwrap();
    assert_eq!(completed, ["u8", "3", "{ 1 + 2 }"]);
}

#[test]
fn test_complete_args() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T, const N: usize = 3, U = Rel<T, N>, V = (RelB, U, T)> {
            fn f(&self) -> RelC;
        }
    };
    let referrer = referrer(&tr);
    let params = &tr.generics.params;

    // Omitted arguments take the defaults, rewritten with the entries which use the preceding
    // parameters. `type_fn` receives the leaked type, its index and the complete arguments of the
    // preceding parameters.
    let (completed, calls) =
        complete_calls(&referrer, params, &args(parse_quote!(<'x, u8>))).unwrap();
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
    let args_of = |args: &[&str]| args.iter().map(|a| a.to_string()).collect::<Vec<_>>();
    assert_eq!(
        calls,
        [
            ("Rel < T , N >".to_string(), 2, args_of(&["'x", "u8", "3"])),
            (
                // The tuple is rebuilt around `RelB`, which is interned.
                "RelB".to_string(),
                0,
                args_of(&["'x", "u8", "3", &projection(2)])
            ),
        ]
    );

    // Explicit arguments are kept as they are (a bare `N` is a const argument).
    let completed = complete(
        &referrer,
        params,
        &args(parse_quote!(<'x, u8, 5, i32, i64>)),
    )
    .unwrap();
    assert_eq!(completed, ["'x", "u8", "5", "i32", "i64"]);

    // Only `V` is omitted.
    let completed = complete(&referrer, params, &args(parse_quote!(<'x, u8, 5, i32>))).unwrap();
    assert_eq!(completed[..4], ["'x", "u8", "5", "i32"]);
    assert_eq!(completed[4], format!("({} , i32 , u8)", projection(0)));

    // A parameter named like a leaked type is not substituted inside a projection: here `T`
    // is the argument (a type named `U`), and `U` is the omitted argument.
    let completed = complete(&referrer, params, &args(parse_quote!(<'x, U>))).unwrap();
    assert_eq!(completed[1], "U");
    assert_eq!(completed[3], projection(2));
    assert_eq!(
        completed[4],
        format!("({} , {} , U)", projection(0), projection(2))
    );
}

#[test]
fn test_complete_args_const_bare_ident_and_errors() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize, U = Rel<T>> {
            fn f(&self);
        }
    );
    let referrer = referrer(&tr);
    let params = &tr.generics.params;
    // A bare identifier for a const parameter is a const argument.
    let completed = complete(&referrer, params, &args(parse_quote!(<u8, M, u16>))).unwrap();
    assert_eq!(completed, ["u8", "M", "u16"]);
    // Missing and unexpected arguments.
    let err = complete(&referrer, params, &args(parse_quote!(<u8>))).unwrap_err();
    assert!(err.to_string().contains("missing generic argument"));
    let err = complete(&referrer, params, &args(parse_quote!(<u8, 1, u16, u32>))).unwrap_err();
    assert!(err.to_string().contains("unexpected generic argument"));
    let err = complete(&referrer, params, &args(parse_quote!(<'a, u8, 1>))).unwrap_err();
    assert!(err.to_string().contains("unexpected generic argument"));
}

#[test]
fn test_complete_args_nested_leaked_types() {
    // A leaked type inside a type which is copied (it cannot be leaked as a whole), twice.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = ::p::W<Rel<T>, &'_ T, Rel<T>>> {}
    );
    let referrer = referrer(&tr);
    let completed = referrer
        .complete_args(
            &args(parse_quote!(<Rel<u8>>)),
            &tr.generics.params,
            |ty, ix, prefix| {
                assert_eq!(ix, 0);
                assert_eq!(s(&ty), "Rel < T >");
                assert_eq!(prefix.len(), 1);
                parse_quote!(P<#(#prefix),*>)
            },
        )
        .unwrap();
    // `T` is substituted in the copied part, but not in the projections.
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        [
            "Rel < u8 >",
            ":: p :: W < P < Rel < u8 > > , & '_ Rel < u8 > , P < Rel < u8 > > >"
        ]
    );
}

#[test]
fn test_complete_args_no_substitution_in_projections() {
    // The result of `type_fn` mentions the parameter `T`, which is not substituted, unlike the
    // copied part of the default.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = ::p::W<Rel<T>, &'_ T>> {}
    );
    let referrer = referrer(&tr);
    let completed = referrer
        .complete_args(&args(parse_quote!(<u8>)), &tr.generics.params, |_, _, _| {
            parse_quote!(<T as Proj>::Type)
        })
        .unwrap();
    assert_eq!(
        s(&completed[1]),
        ":: p :: W < < T as Proj > :: Type , & '_ u8 >"
    );
}

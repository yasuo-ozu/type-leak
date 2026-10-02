//! Tests for the defaults of type parameters, analyzed in contexts of their own, and for
//! `Referrer::complete_args()`.

use syn::*;
use template_quote::quote;
use type_leak::{EncodeError, Leaker, Referrer};

fn referrer(tr: &ItemTrait) -> Referrer {
    Leaker::from_trait(tr).unwrap().finish()
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn args(args: AngleBracketedGenericArguments) -> Vec<GenericArgument> {
    args.args.into_iter().collect()
}

/// A call of `type_fn`: the leaked type, its index and the arguments of its context.
type Call = (String, usize, Vec<String>);

/// Completes `given` with projections `<Self as R<ix>>::Type`, and returns the calls of `type_fn`.
fn complete_calls(
    referrer: &Referrer,
    params: &punctuated::Punctuated<GenericParam, Token![,]>,
    given: &[GenericArgument],
) -> Result<(Vec<String>, Vec<Call>)> {
    let mut calls = Vec::new();
    let completed = referrer
        .complete_args(0, given, params, |ty, ix, context_args| {
            calls.push((s(&ty), ix, context_args.iter().map(s).collect()));
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
fn test_default_contexts() {
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
    // The trait, the default of `U` and the default of `V`. The default of `N` is a const.
    assert_eq!(referrer.n_contexts(), 3);
    let generics = referrer.context_generics(0);
    assert_eq!(
        s(generics),
        s(
            &parse2::<Generics>(quote!(<'a, T: ::core::clone::Clone, const N: usize, U, V>))
                .unwrap()
        )
    );
    assert_eq!(generics.where_clause.as_ref().unwrap().predicates.len(), 2);
    // The root generics of a default are the parameters before it, and the predicates which do not
    // mention the others.
    let generics = referrer.context_generics(1);
    assert_eq!(
        s(generics),
        s(&parse2::<Generics>(quote!(<'a, T: ::core::clone::Clone, const N: usize>)).unwrap())
    );
    assert_eq!(
        s(&generics.where_clause),
        s(&Some(
            parse2::<WhereClause>(quote!(where T: ::p::Conv<RelW>)).unwrap()
        ))
    );
    let generics = referrer.context_generics(2);
    assert_eq!(
        s(generics),
        s(&parse2::<Generics>(quote!(<'a, T: ::core::clone::Clone, const N: usize, U>)).unwrap())
    );
    assert_eq!(generics.where_clause.as_ref().unwrap().predicates.len(), 2);

    // The defaults are not leaked in the context 0.
    let leaked: Vec<_> = referrer
        .iter()
        .enumerate()
        .map(|(ix, ty)| (referrer.context(ix), s(ty)))
        .collect();
    assert_eq!(
        leaked,
        [
            (0, "RelC".to_string()),
            (0, "RelW".to_string()),
            (1, "Rel < T >".to_string()),
            (2, "(RelB , U)".to_string())
        ]
    );

    // The round trip keeps the root generics.
    let tokens = quote!(#referrer);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    for context in 0..3 {
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
fn test_default_contexts_are_allocated_per_call() {
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
    // A failed call allocates nothing.
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
    // A context for the struct, the trait, and its default (which leaks nothing).
    assert_eq!(referrer.n_contexts(), 3);
    assert_eq!(referrer.context_generics(0).params.len(), 1);
    assert_eq!(referrer.context_generics(1).params.len(), 2);
    assert_eq!(referrer.context_generics(2).params.len(), 1);
    assert_eq!(referrer.context(1), 1);
    // The main context is given to `complete_args()`.
    let completed = referrer
        .complete_args(
            1,
            &args(parse_quote!(<u8>)),
            &tr.generics.params,
            |_, _, _| unreachable!(),
        )
        .unwrap();
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        ["u8", ":: std :: vec :: Vec < u8 >"]
    );
    // There is no context for the default in the context 2.
    assert!(matches!(
        referrer.complete_args(
            2,
            &args(parse_quote!(<u8>)),
            &tr.generics.params,
            |_, _, _| { unreachable!() }
        ),
        Err(EncodeError::MissingArgument(_))
    ));
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
    assert_eq!(referrer.n_contexts(), 1);
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

    // Omitted arguments take the defaults rewritten in the context of the default. `type_fn`
    // receives the leaked type, its index and the complete arguments of the preceding parameters.
    let (completed, calls) =
        complete_calls(&referrer, params, &args(parse_quote!(<'x, u8>))).unwrap();
    let projection = |ix: usize| format!("< Self as R < {}usize > > :: Type", ix);
    assert_eq!(
        completed,
        [
            "'x".to_string(),
            "u8".to_string(),
            "3".to_string(),
            projection(1),
            projection(2),
        ]
    );
    let args_of = |args: &[&str]| args.iter().map(|a| a.to_string()).collect::<Vec<_>>();
    assert_eq!(
        calls,
        [
            ("Rel < T , N >".to_string(), 1, args_of(&["'x", "u8", "3"])),
            (
                // A root is interned as a whole, together with its components.
                "(RelB , U , T)".to_string(),
                2,
                args_of(&["'x", "u8", "3", &projection(1)])
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
    assert_eq!(completed[4], "< Self as R < 2usize > > :: Type");

    // A parameter named like a leaked type is not substituted inside a projection: here `T`
    // is the argument (a type named `U`), and `U` is the omitted argument.
    let completed = complete(&referrer, params, &args(parse_quote!(<'x, U>))).unwrap();
    assert_eq!(completed[1], "U");
    assert_eq!(completed[3], "< Self as R < 1usize > > :: Type");
    assert_eq!(completed[4], "< Self as R < 2usize > > :: Type");
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
            0,
            &args(parse_quote!(<Rel<u8>>)),
            &tr.generics.params,
            |ty, ix, context_args| {
                assert_eq!(ix, 0);
                assert_eq!(s(&ty), "Rel < T >");
                assert_eq!(context_args.len(), 1);
                parse_quote!(P<#(#context_args),*>)
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
        .complete_args(
            0,
            &args(parse_quote!(<u8>)),
            &tr.generics.params,
            |_, _, _| parse_quote!(<T as Proj>::Type),
        )
        .unwrap();
    assert_eq!(
        s(&completed[1]),
        ":: p :: W < < T as Proj > :: Type , & '_ u8 >"
    );
}

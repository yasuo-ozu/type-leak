//! Tests for `View::check()` and the paths that `Rules::or_if()` allows.

mod common;

use common::{check_in, starts_with};
use syn::*;
use template_quote::ToTokens;
use type_leak::{
    to_syn_error, Cause, CheckResult, ContextId, Leaker, Options, Reason, Referrer, Rules, View,
};

/// Classifies a `CheckResult` (as the variants of the former `CheckResult` enum).
fn kind(result: &CheckResult) -> &'static str {
    match (result.must.is_empty(), result.impossible.is_empty()) {
        (true, true) => "Neutral",
        (false, true) if result.is_own => "MustIntern",
        (false, true) => "MustInternOrInherit",
        (true, false) => "MustNotIntern",
        (false, false) => "Conflict",
    }
}

#[test]
fn test_check() {
    let test_struct: ItemStruct = parse_quote! {
        struct MyStruct<'l1, const N1: usize, T1, T2, T3>;
    };
    let mut leaker = Leaker::new();
    let generics = &test_struct.generics;
    // `Self` is the struct, and may be leaked.
    let mut check = |ty: Type| {
        let ident = &test_struct.ident;
        leaker
            .enter_type(ident, generics, true, |view| view.check(&ty))
            .1
    };
    assert_eq!(kind(&check(parse_quote!(T1))), "Neutral");
    assert_eq!(kind(&check(parse_quote!((T1, T2)))), "Neutral");
    assert_eq!(kind(&check(parse_quote!([T1; 123]))), "Neutral");
    assert_eq!(kind(&check(parse_quote!([T1; N]))), "MustIntern"); // constant
    assert_eq!(kind(&check(parse_quote!([T1; ::path::to::N]))), "Neutral");
    assert_eq!(kind(&check(parse_quote!([(); N1]))), "Neutral");
    assert_eq!(kind(&check(parse_quote!(some::path))), "MustIntern"); // relavice path
    assert_eq!(kind(&check(parse_quote!(::some::path))), "Neutral");
    assert_eq!(
        kind(&check(parse_quote!(<T1 as ::abs::path::MyTrait<T2>>::Ty))),
        "Neutral"
    );
    assert_eq!(
        kind(&check(parse_quote!(
            <MyType as ::abs::path::MyTrait<T2>>::Ty
        ))),
        "MustInternOrInherit"
    ); // relative path
    assert_eq!(
        kind(&check(parse_quote!(
            <::path::to::MyType as MyTrait<T2>>::Ty
        ))),
        "MustIntern"
    ); // relative path
    assert_eq!(
        kind(&check(parse_quote!(
            <::path::to::MyType as ::path::to::MyTrait<(MyType2,)>>::Ty
        ))),
        "MustInternOrInherit"
    ); // relative path in an argument
    assert_eq!(
        kind(&check(parse_quote!(
            <::path::to::MyType as ::path::to::MyTrait<MyType2>>::Ty
        ))),
        "MustIntern"
    ); // a bare identifier as an argument may be a const, so the type is interned as a whole
    assert_eq!(kind(&check(parse_quote!(()))), "Neutral");
    assert_eq!(kind(&check(parse_quote!(&'l1 ()))), "Neutral");
    assert_eq!(kind(&check(parse_quote!(&'static ()))), "Neutral");
    assert_eq!(kind(&check(parse_quote!(&'my_lifetime ()))), "MustIntern"); // lifetime
    assert_eq!(
        kind(&check(parse_quote!(impl ::path::to::MyTrait))),
        "MustNotIntern"
    ); // impl Trait
    assert_eq!(kind(&check(parse_quote!(impl MyTrait))), "Conflict"); // impl Trait and relative path
    assert_eq!(
        kind(&check(parse_quote!(dyn ::path::to::MyTrait))),
        "Neutral"
    );
    assert_eq!(kind(&check(parse_quote!(Self))), "Neutral");
    assert_eq!(kind(&check(parse_quote!(&Self))), "MustNotIntern");
    assert_eq!(kind(&check(parse_quote!(&'a Self))), "MustIntern");
    assert_eq!(
        kind(&check(parse_quote!(::core::fmt::Formatter<'_>))),
        "MustNotIntern"
    ); // relative path
       // A check records nothing: the scopes found no error.
    assert!(leaker.result().errors().is_empty());
}

/// `Rules::crate_paths()`: a path that starts with `crate` is leaked, unless the allowed
/// paths have `crate`. A real `$crate` token (which a `macro_rules!`
/// expansion gives to a proc macro) cannot be made outside one, so the `$crate` paths are pinned
/// by the fixture crate (`DollarCrate` in `tests/crates/leaker`, with `check_leak!`, and
/// `cross_crate::dollar_crate_paths`).
#[test]
fn test_check_allows_crate_paths_only_if_allowed() {
    let config = Rules::new().crate_paths();
    let mut leaker = Leaker::from_config(config, Options::new());

    let item: ItemTrait = parse_quote!(
        pub trait Parse<Atom>: ::core::marker::Sized {
            type Error: crate::error::Error;
            fn parse(
                stream: impl crate::parse::into_parse_stream::IntoParseStream<Atom = Atom>,
            ) -> ::core::result::Result<Self, Self::Error>;
        }
    );
    leaker.intern_trait(&item, true);
    leaker.reduce_roots();
    assert!(leaker.result().finish().unwrap().is_empty());

    // Without it the same paths are leaked (here in bounds, which cannot be leaked): the allowed
    // `crate` is what lets them through.
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let err = result.errors();
    let message = to_syn_error(err).to_string();
    assert!(
        message.contains("use absolute path: 'crate::error::Error'"),
        "{}",
        message
    );
}

#[test]
fn test_check_allowed_paths() {
    let generics: Generics = parse_quote!(<'l1, const N1: usize, T1>);
    let config = || Rules::new().or_if(starts_with(parse_quote!(some::path)));
    let mut leaker = Leaker::from_config(config(), Options::new());

    assert_eq!(
        kind(&check_in(&mut leaker, &parse_quote!(some::path), &generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&check_in(
            &mut leaker,
            &parse_quote!(some::path::Type),
            &generics
        )),
        "Neutral"
    );
    assert_eq!(
        kind(&check_in(
            &mut leaker,
            &parse_quote!(other::path),
            &generics
        )),
        "MustIntern"
    );

    let config = config().or_if(starts_with(parse_quote!(::abs::path)));
    let mut leaker = Leaker::from_config(config, Options::new());
    assert_eq!(
        kind(&check_in(
            &mut leaker,
            &parse_quote!(::abs::path::Type),
            &generics
        )),
        "Neutral"
    );
    assert_eq!(
        kind(&check_in(
            &mut leaker,
            &parse_quote!(abs::path::Type),
            &generics
        )),
        "MustIntern"
    );
}

#[test]
fn test_check_allowed_paths_with_generic_args() {
    let generics: Generics = parse_quote!(<T1>);
    let config = Rules::new().prelude().primitive();
    let mut leaker = Leaker::from_config(config, Options::new());

    // An allowed path with generic arguments is still allowed.
    assert_eq!(
        kind(&check_in(&mut leaker, &parse_quote!(Option<u8>), &generics)),
        "Neutral"
    );
    assert_eq!(
        kind(&check_in(
            &mut leaker,
            &parse_quote!(Vec<Option<T1>>),
            &generics
        )),
        "Neutral"
    );
    // Its generic arguments are still checked.
    assert_eq!(
        kind(&check_in(
            &mut leaker,
            &parse_quote!(Option<(RelType,)>),
            &generics
        )),
        "MustInternOrInherit"
    );
    // A bare identifier as an argument may be a const, so the type is interned as a whole.
    assert_eq!(
        kind(&check_in(
            &mut leaker,
            &parse_quote!(Option<RelType>),
            &generics
        )),
        "MustIntern"
    );

    // Trait bounds with generic arguments on allowed paths are accepted, and relative types in
    // them are leaked.
    let config = Rules::new().prelude().primitive();
    let mut leaker = Leaker::from_config(config, Options::new());
    let item: ItemTrait = parse_quote! {
        trait MyTrait: Into<u8> {
            fn f<I: Iterator<Item = u8>>(&self, i: I);
            fn g<I: Iterator<Item = RelType>>(&self, i: I);
        }
    };
    leaker.intern_trait(&item, true);
    leaker.reduce_roots();
    let leaked: std::collections::BTreeSet<String> = leaker
        .result()
        .finish()
        .unwrap()
        .iter()
        .map(|(_, t, _)| template_quote::quote!(#t).to_string())
        .collect();
    assert_eq!(leaked, vec!["RelType".to_string()].into_iter().collect());
}

/// The kind of each cause and its tokens, as text.
fn kinds(causes: &[Cause]) -> Vec<(&'static str, String)> {
    causes
        .iter()
        .map(|cause| {
            let kind = match cause {
                Cause::Type(_) => "type",
                Cause::Trait(_) => "trait",
                Cause::Expr(_) => "expr",
                Cause::Macro(_) => "macro",
                Cause::Lifetime(_) => "lifetime",
                Cause::Unreadable(_) => "unreadable",
                other => panic!("{:?}", other),
            };
            (kind, cause.to_token_stream().to_string())
        })
        .collect()
}

/// The causes of `must` and `impossible` are the syntax that they are: a path in a type is a type,
/// the path of a bound a trait, a path in an expression an expression, the path of a macro call a
/// macro, and a lifetime a lifetime. The first of `must` is the cause of the error. A type whose own
/// path depends on the type context cannot be decomposed (`is_own`), and one whose components do
/// can; all the causes are collected, the outermost ones first.
#[test]
fn the_causes_of_a_check_have_the_kind_of_their_syntax() {
    let generics: Generics = parse_quote!(<T>);
    let check = |ty: &Type| check_in(&mut Leaker::new(), ty, &generics);
    // The type, `must` and `impossible` as (kind, text), and `is_own`.
    type Row = (
        Type,
        Vec<(&'static str, &'static str)>,
        Vec<(&'static str, &'static str)>,
        bool,
    );
    let rows: Vec<Row> = vec![
        (
            parse_quote!(MyType<'_>),
            vec![("type", "MyType < '_ >")],
            vec![("lifetime", "'_")],
            true,
        ),
        // A path in a qualified type is no node of the syntax.
        (
            parse_quote!(<T as RelTrait<'_>>::Ty),
            vec![("unreadable", "RelTrait < '_ > :: Ty")],
            vec![("lifetime", "'_")],
            true,
        ),
        // Only components depend on the context: the type can be decomposed.
        (
            parse_quote!(::path::Wrap<MyType, '_>),
            vec![("type", "MyType")],
            vec![("lifetime", "'_")],
            false,
        ),
        (
            parse_quote!(<T as ::path::Trait<MyType, '_>>::Ty),
            vec![("type", "MyType")],
            vec![("lifetime", "'_")],
            false,
        ),
        (
            parse_quote!(::path::Wrap<'_, ::path::Wrap<(C,)>, A, B<D>, impl ::path::Tr>),
            vec![
                ("type", "A"),
                ("type", "B < D >"),
                ("type", "D"),
                ("type", "C"),
            ],
            vec![("lifetime", "'_"), ("type", "impl :: path :: Tr")],
            false,
        ),
        // A bare identifier as an argument may be a const, so the type containing it is interned as
        // a whole (with the cause in its own path) if it can be, as `::path::Wrap<C>` here, but not
        // the outer one, which has impossible causes (`A` is a type of its own).
        (
            parse_quote!(::path::Wrap<'_, ::path::Wrap<C>, A>),
            vec![("type", "C"), ("type", "A")],
            vec![("lifetime", "'_")],
            false,
        ),
        // The trait of a bound is a trait.
        (
            parse_quote!(dyn A + B<'_>),
            vec![("trait", "A"), ("trait", "B < '_ >")],
            vec![("lifetime", "'_")],
            true,
        ),
        // The trait of a `dyn` type is a trait, and the type that cannot be leaked is a type (under
        // the elided lifetime, that is a lifetime).
        (
            parse_quote!(::p::W<'_, dyn RelTr>),
            vec![("trait", "RelTr")],
            vec![("lifetime", "'_"), ("type", "dyn RelTr")],
            false,
        ),
        // The closure stops the array from being leaked, and the paths in it are causes of their
        // own: `u8` a type, `a` an expression.
        (
            parse_quote!(
                [u8; {
                    let c = |a: u8| a;
                    3
                }]
            ),
            vec![("type", "u8"), ("expr", "a")],
            vec![("expr", "| a : u8 | a")],
            true,
        ),
        // A macro call is a macro in `must` (its path, written with the `!`) and an expression, with
        // its arguments, in `impossible`.
        (
            parse_quote!([u8; two!()]),
            vec![("macro", "two !")],
            vec![("expr", "two ! ()")],
            true,
        ),
    ];
    for (ty, must, impossible, is_own) in rows {
        let result = check(&ty);
        let own = |causes: Vec<(&str, &str)>| -> Vec<(String, String)> {
            causes
                .into_iter()
                .map(|(kind, text)| (kind.to_string(), text.to_string()))
                .collect()
        };
        let got = |causes: &[Cause]| -> Vec<(String, String)> {
            kinds(causes)
                .into_iter()
                .map(|(kind, text)| (kind.to_string(), text))
                .collect()
        };
        let what = ty.to_token_stream().to_string();
        assert_eq!(got(&result.must), own(must), "must of {}", what);
        assert_eq!(
            got(&result.impossible),
            own(impossible),
            "impossible of {}",
            what
        );
        assert_eq!(result.is_own, is_own, "is_own of {}", what);
    }
    let result = check(&parse_quote!(
        [u8; {
            let c = |a: u8| a;
            3
        }]
    ));
    assert!(matches!(&result.must[1], Cause::Expr(Expr::Path(_))));
    assert!(matches!(
        &result.impossible[0],
        Cause::Expr(Expr::Closure(_))
    ));
}

fn strings(causes: &[Cause]) -> Vec<String> {
    causes
        .iter()
        .map(|cause| cause.to_token_stream().to_string())
        .collect()
}

fn check_ty(view: &mut View<'_, '_>, ty: Type) -> CheckResult {
    view.check(&ty)
}

#[test]
fn check_in_a_module_does_not_report_the_names_it_declares() {
    let mut leaker = Leaker::new();
    let local: Type = parse_quote!(Local);
    // Outside a module it is a cause: the path means something else where the copy is.
    let outside = check_in(&mut leaker, &local, &Generics::default());
    assert_eq!(kind(&outside), "MustIntern");
    assert_eq!(strings(&outside.must), ["Local"]);
    // In the module that declares it, it is not: the copy has the module's items too.
    let (_, results) = leaker.enter_mod("m", [parse_quote!(Local)], |view| {
        [
            check_ty(view, parse_quote!(Local)),
            check_ty(view, parse_quote!(self::Local)),
            check_ty(view, parse_quote!(::p::W<(Local,)>)),
            check_ty(view, parse_quote!(Other)),
            check_ty(view, parse_quote!(super::Local)),
        ]
    });
    let kinds: Vec<_> = results.iter().map(kind).collect();
    assert_eq!(
        kinds,
        ["Neutral", "Neutral", "Neutral", "MustIntern", "MustIntern"]
    );
    assert_eq!(strings(&results[3].must), ["Other"]);
    assert_eq!(strings(&results[4].must), ["super :: Local"]);
    // A module declares only its own names: not the ones of the module around.
    let (_, inner) = leaker.enter_mod("m", [parse_quote!(Local)], |view| {
        view.enter_mod("n", [parse_quote!(Deeper)], |view| {
            [
                check_ty(view, parse_quote!(Deeper)),
                check_ty(view, parse_quote!(Local)),
            ]
        })
        .1
    });
    assert_eq!(kind(&inner[0]), "Neutral");
    assert_eq!(kind(&inner[1]), "MustIntern");
}

#[test]
fn check_self_in_a_struct_scope() {
    let item: ItemStruct = parse_quote!(
        struct S<T>(T);
    );
    let mut leaker = Leaker::new();
    let mut check_all = |allow_self: bool| {
        leaker
            .enter_type(&item.ident, &item.generics, allow_self, |view| {
                [
                    check_ty(view, parse_quote!(Self)),
                    check_ty(view, parse_quote!(Rel<Self>)),
                    check_ty(view, parse_quote!(::p::W<(Rel, Self)>)),
                    check_ty(view, parse_quote!(T)),
                ]
            })
            .1
    };
    // `Self` can be leaked: it is not a cause, and `Rel` is the cause of `Rel<Self>`.
    let [own, in_rel, in_path, plain] = check_all(true);
    assert_eq!(kind(&own), "Neutral");
    assert!(own.note.is_none());
    assert_eq!(kind(&in_rel), "MustIntern");
    assert_eq!(strings(&in_rel.must), ["Rel < Self >"]);
    assert!(in_rel.impossible.is_empty());
    assert_eq!(kind(&in_path), "MustInternOrInherit");
    assert_eq!(kind(&plain), "Neutral");
    // With `allow_self` unset it cannot, so it is an impossible cause, and the type that holds it
    // is a conflict if it is a cause itself.
    let [own, in_rel, in_path, plain] = check_all(false);
    assert_eq!(kind(&own), "MustNotIntern");
    assert_eq!(strings(&own.impossible), ["Self"]);
    assert_eq!(kind(&in_rel), "Conflict");
    assert_eq!(strings(&in_rel.must), ["Rel < Self >"]);
    assert_eq!(strings(&in_rel.impossible), ["Self"]);
    assert!(in_rel.is_own);
    assert_eq!(kind(&in_path), "Conflict");
    assert!(!in_path.is_own);
    assert_eq!(kind(&plain), "Neutral");
    // A check records nothing: the scopes found no error.
    assert!(leaker.result().errors().is_empty());
}

#[test]
fn check_self_has_no_meaning_in_generics_and_modules() {
    let generics: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let checks = |view: &mut View<'_, '_>| {
        [
            check_ty(view, parse_quote!(Self)),
            check_ty(view, parse_quote!(Self::Item)),
            check_ty(view, parse_quote!(Rel<Self>)),
            check_ty(view, parse_quote!(::p::W<(Self,)>)),
            check_ty(view, parse_quote!(())),
        ]
    };
    let (_, in_generics) = leaker.enter_generics(&generics, checks);
    let (_, in_module) = leaker.enter_mod("m", [], checks);
    for results in [in_generics, in_module] {
        let [own, assoc, in_rel, in_path, plain] = results;
        // It is an impossible cause, as written, and the note tells why.
        assert_eq!(kind(&own), "MustNotIntern");
        assert_eq!(strings(&own.impossible), ["Self"]);
        let note = own.note.expect("a note");
        assert!(note.starts_with("`Self` has no meaning here"), "{}", note);
        // And the help tells what to write.
        let help = own.help.expect("a help");
        assert_eq!(help, "write the type instead of `Self`");
        assert_eq!(strings(&assoc.impossible), ["Self :: Item"]);
        assert!(assoc.note.is_some());
        // A type that holds it is a conflict if it is a cause itself, and its parts can be leaked
        // otherwise.
        assert_eq!(kind(&in_rel), "Conflict");
        assert_eq!(in_rel.must.len(), 1);
        assert_eq!(strings(&in_rel.impossible), ["Self"]);
        assert!(in_rel.is_own);
        assert_eq!(kind(&in_path), "MustNotIntern");
        assert!(!in_path.is_own && in_path.note.is_some());
        // The types without it are not affected.
        assert_eq!(kind(&plain), "Neutral");
        assert!(plain.note.is_none());
    }
    // `Self` is that of the item in a scope nested in one, and in a module nested in none.
    let item: ItemStruct = parse_quote!(
        struct S<T>(T);
    );
    let (_, results) = leaker.enter_type(&item.ident, &item.generics, true, |view| {
        view.enter_generics(&parse_quote!(<'a>), checks).1
    });
    assert_eq!(kind(&results[0]), "Neutral");
    assert_eq!(kind(&results[4]), "Neutral");
    let (_, results) = leaker.enter_type(&item.ident, &item.generics, true, |view| {
        view.enter_mod("m", [], checks).1
    });
    assert_eq!(kind(&results[0]), "MustNotIntern");
}

#[test]
fn check_is_not_the_error_of_the_scope() {
    let mut leaker = Leaker::new();
    // `check()` finds `Self` without leaving the error a visit leaves.
    let (_, found) = leaker.enter_generics(&Generics::default(), |view| {
        view.check(&parse_quote!(Self)).impossible.len()
    });
    assert_eq!(found, 1);
    leaker.enter_generics(&Generics::default(), |view| {
        view.check(&parse_quote!(Self));
        view.visit_type(&parse_quote!(Self));
        view.check(&parse_quote!(Self));
    });
    // Neither does the conflict of a type leave one.
    let (_, found) = leaker.enter_generics(&Generics::default(), |view| {
        view.check(&parse_quote!(MyType<'_>)).impossible.len()
    });
    assert_eq!(found, 1);
    // The errors of the Leaker are that of the visit only: no check left one.
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::SelfWithoutMeaning);
}

#[test]
fn check_tells_an_optional_entry_of_an_external_crate() {
    let mut leaker = Leaker::new();
    let generics = Generics::default();
    // The path of an external crate is optional: the Referrer may replace it. It is not a cause.
    let result = check_in(&mut leaker, &parse_quote!(::foo::X), &generics);
    assert_eq!(kind(&result), "Neutral");
    assert!(result.optional);
    // Its parts can be causes.
    let result = check_in(&mut leaker, &parse_quote!(::foo::W<(Rel,)>), &generics);
    assert_eq!(strings(&result.must), ["Rel"]);
    assert!(!result.is_own && result.optional);
    // Not if it has a cause of its own, or a cause that cannot be leaked (it is copied then).
    let result = check_in(&mut leaker, &parse_quote!(::foo::W<Rel>), &generics);
    assert!(result.is_own && !result.optional);
    let result = check_in(&mut leaker, &parse_quote!(::foo::W<'_>), &generics);
    assert_eq!(kind(&result), "MustNotIntern");
    assert!(!result.optional);
    // Not if it is the part of one, or its path is not of an external crate.
    for ty in [
        parse_quote!((::foo::X,)),
        parse_quote!(Rel),
        parse_quote!(&'static ()),
        parse_quote!(::core::option::Option<u8>),
    ] {
        assert!(!check_in(&mut leaker, &ty, &generics).optional, "{:?}", ty);
    }
    // Not if the Leaker allows the crate.
    let config = Rules::new().or_if(starts_with(parse_quote!(::foo)));
    let mut leaker = Leaker::from_config(config, Options::new());
    let result = check_in(&mut leaker, &parse_quote!(::foo::X), &generics);
    assert_eq!(kind(&result), "Neutral");
    assert!(!result.optional);
    // The names of the module are not copied by the Referrer, so a type using one is not leaked
    // alone, and its path is not an optional entry either.
    let mut leaker = Leaker::new();
    let (_, results) = leaker.enter_mod("m", [parse_quote!(Local)], |view| {
        [
            check_ty(view, parse_quote!(::foo::W<(Local,)>)),
            check_ty(view, parse_quote!(::foo::W<(Other,)>)),
        ]
    });
    assert!(results[0].must.is_empty() && !results[0].optional);
    assert_eq!(strings(&results[1].must), ["Other"]);
    assert!(results[1].optional);
    let result = check_in(
        &mut leaker,
        &parse_quote!(::foo::W<(Local,)>),
        &Generics::default(),
    );
    assert_eq!(strings(&result.must), ["Local"]);
    assert!(result.optional);
}

#[test]
fn check_sees_the_generics_of_the_scopes_around() {
    let mut leaker = Leaker::new();
    let (_, (outer, inner)) = leaker.enter_generics(&parse_quote!(<T>), |view| {
        let outer = [
            check_ty(view, parse_quote!(Wrap<T>)),
            check_ty(view, parse_quote!((T, U))),
            check_ty(view, parse_quote!(&'a ())),
        ];
        let (_, inner) = view.enter_generics(&parse_quote!(<'a, U>), |view| {
            [
                check_ty(view, parse_quote!((T, U))),
                check_ty(view, parse_quote!(&'a ())),
                check_ty(view, parse_quote!(::p::W<'a, T, U>)),
                check_ty(view, parse_quote!((T, V))),
            ]
        });
        // Back in the scope around, which does not know what the nested one declared.
        assert_eq!(kind(&check_ty(view, parse_quote!(U))), "MustIntern");
        (outer, inner)
    });
    // `T` is a generic of the scope, and `U` and `'a` are not yet.
    assert_eq!(kind(&outer[0]), "MustIntern");
    assert_eq!(strings(&outer[0].must), ["Wrap < T >"]);
    assert_eq!(strings(&outer[1].must), ["U"]);
    assert_eq!(strings(&outer[2].must), ["'a"]);
    // The nested scope has both.
    assert_eq!(kind(&inner[0]), "Neutral");
    assert_eq!(kind(&inner[1]), "Neutral");
    assert_eq!(kind(&inner[2]), "Neutral");
    assert_eq!(strings(&inner[3].must), ["V"]);
    // A check records nothing: the scopes found no error.
    assert!(leaker.result().errors().is_empty());
}

#[test]
fn check_reports_the_conflict_a_scope_reports() {
    // The bounds of the generics mention `Self`, which `allow_self` forbids to leak: the type
    // using `T` cannot be leaked, which only the scope knows.
    let item: ItemStruct = parse_quote!(
        struct S<T: ::p::Bound<Self>>(T);
    );
    let ty: Type = parse_quote!(Rel<T>);
    let mut leaker = Leaker::new();
    let mut check = |allow_self| {
        leaker
            .enter_type(&item.ident, &item.generics, allow_self, |view| {
                view.check(&ty)
            })
            .1
    };
    let allowed = check(true);
    assert_eq!(kind(&allowed), "MustIntern");
    assert!(allowed.note.is_none());
    let forbidden = check(false);
    assert_eq!(kind(&forbidden), "Conflict");
    assert_eq!(strings(&forbidden.must), ["Rel < T >"]);
    assert!(forbidden.is_own);
    let note = forbidden.note.clone().expect("a note");
    assert!(note.contains("mention `Self`"), "{}", note);
    let help = forbidden.help.clone().expect("a help");
    assert_eq!(
        help,
        "write the bound of the generics of `Rel<T>` without `Self`"
    );
    // It is what the scope reports for the same type.
    leaker.enter_type(&item.ident, &item.generics, false, |view| {
        view.visit_type(&ty)
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1);
    let item = &err[0];
    assert_eq!(item.reason, Reason::DeniedSelfInBounds);
    // The error is reported at the first of `must`, with the causes of `impossible` as notes, after
    // the note of the conflict, and the help of the conflict.
    assert_eq!(
        item.cause.to_token_stream().to_string(),
        forbidden.must[0].to_token_stream().to_string()
    );
    assert_eq!(
        item.notes[0].0.to_string(),
        forbidden.must[0].to_token_stream().to_string()
    );
    assert_eq!(Some(&item.notes[0].1), forbidden.note.as_ref());
    assert_eq!(
        item.notes[1..]
            .iter()
            .map(|(at, _)| at.to_string())
            .collect::<Vec<_>>(),
        strings(&forbidden.impossible)
    );
    assert_eq!(item.help_text(), forbidden.help.as_deref());
}

/// Types that a visit would record in some way: entries, optional entries, conflicts, errors for
/// `Self`, the probe of a binder, and a module found in an array length; and `&'static u8`, which it
/// would not.
fn recording_types() -> Vec<Type> {
    vec![
        parse_quote!(Rel),
        parse_quote!(Rel<'_>),
        parse_quote!(Rel<T>),
        parse_quote!(::foo::W<(Rel2<T>,)>),
        parse_quote!(MyType<'_>),
        parse_quote!(::p::W<'_, dyn RelTr>),
        parse_quote!(::p::X<dyn RelTr>),
        parse_quote!(impl MyTrait),
        parse_quote!(Self),
        parse_quote!(Rel<Self::Item>),
        parse_quote!(&'a Rel<T>),
        parse_quote!(&'static u8),
        parse_quote!(for<'x> fn(Rel<'x>, &u8) -> &'x u8),
        parse_quote!(
            ::foo::W<
                [u8; {
                    mod inner {
                        pub const N: usize = 1;
                    }
                    inner::N
                }],
            >
        ),
        parse_quote!(Local),
    ]
}

/// Runs the same analysis of several scopes, and calls `check()` in each of them with the types of
/// `recording_types()` if `checked`. Returns the ids of the scopes and the Referrer.
fn analyze(checked: bool) -> (Vec<ContextId>, Referrer) {
    let types = recording_types();
    let checks = |view: &mut View<'_, '_>| {
        if checked {
            for ty in &types {
                view.check(ty);
            }
        }
    };
    let item: ItemStruct = parse_quote!(
        struct S<T>(T);
    );
    let mut leaker = Leaker::new();
    let mut ids = Vec::new();
    let (id, ()) = leaker.enter_generics(&item.generics, |view| {
        checks(view);
        view.visit_type(&parse_quote!(Rel<T>));
        checks(view);
        view.visit_type(&parse_quote!(::foo::W<(Rel2<T>,)>));
        let (id, ()) = view.enter_mod("m", [parse_quote!(Local)], |view| {
            checks(view);
            view.visit_type(&parse_quote!(Rel3));
            checks(view);
        });
        ids.push(id);
        let (id, ()) = view.enter_type(&item.ident, &item.generics, true, |view| {
            checks(view);
            view.visit_type(&parse_quote!(Rel4<Self>));
        });
        ids.push(id);
        checks(view);
    });
    ids.push(id);
    // The scopes after it get the ids that the ones before left free.
    let (id, ()) = leaker.enter_mod("n", [], |view| {
        checks(view);
        view.visit_type(&parse_quote!(Rel5));
    });
    ids.push(id);
    let (id, ()) = leaker.enter_generics(&parse_quote!(<U>), |view| {
        view.visit_type(&parse_quote!(Rel6<U>))
    });
    ids.push(id);
    (ids, leaker.result().finish().unwrap())
}

#[test]
fn check_records_nothing() {
    let (plain_ids, plain) = analyze(false);
    let (checked_ids, checked) = analyze(true);
    assert_eq!(checked_ids, plain_ids);
    assert_eq!(checked, plain);
    assert!(!plain.is_empty());
}

#[test]
fn a_check_alone_leaves_an_empty_leaker() {
    let mut leaker = Leaker::new();
    let types = recording_types();
    let (id, ()) = leaker.enter_generics(&parse_quote!(<T>), |view| {
        for ty in &types {
            view.check(ty);
        }
    });
    let (other_id, ()) = Leaker::new().enter_generics(&parse_quote!(<T>), |_| ());
    assert_eq!(id, other_id);
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.is_empty());
}
